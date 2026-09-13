# ReSTIR Path Tracing (ReSTIR PT): Technical Analysis, Historical Post-Mortem & Feasibility Assessment

## 1. Executive Summary

Pathways previously explored two implementations of reservoir-based spatio-temporal resampling:
1. **ReSTIR DI (Direct Illumination, v1.13.0)**: Resampled emissive light candidates across space and time using weighted reservoir sampling.
2. **ReSTIR GI (Global Illumination, v1.14.0)**: Extended resampling to indirect one-bounce diffuse hits using split-buffer coalescing.

Both attempts exhibited persistent instability, including **temporal boiling**, **light leaks across occlusion boundaries**, and **correlation bias during camera motion**. In commit `a0e53d4`, ReSTIR was demoted to an opt-in CLI flag, and Pathways defaulted back to pure reference Wavefront path tracing.

This document provides a technical post-mortem of why previous ReSTIR attempts struggled, explains the theoretical hurdles of true **ReSTIR Path Tracing (ReSTIR PT)** under the Generalized Resampled Importance Sampling (GRIS) framework, evaluates the memory and compute overhead on AMD RDNA 4 (`gfx1201`), and delivers a definitive feasibility verdict.

---

## 2. Technical Post-Mortem: Why Prior ReSTIR Implementations Failed in Pathways

### 2.1 The Dimensionality Gap: 1D Light Resampling vs. Multi-Bounce Path Resampling
Traditional **ReSTIR DI** operates on a low-dimensional domain: selecting a light index $i \in [0, N-1]$ and a UV coordinate on an emissive triangle. If a candidate sample is occluded, a single shadow ray determines visibility.

In contrast, **ReSTIR GI and ReSTIR PT** attempt to resample high-dimensional geometric paths:
$$\bar{x} = (x_0, x_1, x_2, \dots, x_k)$$
A path is not an isolated point; it is a chain of vertices embedded in high-dimensional scene geometry. Transferring path $\bar{y}$ from neighbor pixel $r$ to pixel $q$ alters the entire integration measure and visibility topology.

```
       Pixel r Path                            Pixel q Candidate
   x0 (Camera)                             x0 (Camera)
      │                                       │
      ▼                                       ▼
   y1 (Primary Hit)                        x1 (Primary Hit)
      │                                       :
      │  Shift Mapping                        : Reconnection
      │  (y1 -> x1)                           : Visibility Test
      ▼                                       ▼
   y2 (Secondary Hit) ─────────────────────► y2 (Secondary Hit)
      │                                       │
      ▼                                       ▼
   y3 (Light / Sky)                        y3 (Light / Sky)
```

### 2.2 Shift Mapping Failures & Geometric Occlusion Stalls
When reusing path $\bar{y}$ at pixel $q$, the connection between $x_1$ and $y_2$ must be validated:
1. **Unchecked Reconnection (Zero-Ray Fallback)**:
   - In Pathways' earlier ReSTIR GI implementation, testing visibility between $x_1$ and $y_2$ for every candidate reservoir required an additional ray query per spatial tap.
   - To stay within the real-time budget, spatial gathering skipped explicit ray queries, relying instead on bilateral depth/normal thresholds ($\Delta d < 0.10, \vec{n}_q \cdot \vec{n}_r > 0.90$).
   - **Failure Mode**: Bilateral thresholds cannot detect intervening occluders between $x_1$ and $y_2$. In complex scenes (e.g. `Living Room`, `Classroom`), light from unshadowed secondary hits bled through walls, table legs, and door frames, producing bright halo artifacts and glowing corners.
2. **Checked Reconnection (Explicit Visibility Rays)**:
   - Testing visibility with $k=3$ spatial taps requires 3 extra hardware ray queries per pixel.
   - For a 4K frame ($3840 \times 2160 = 8.29\text{M}$ pixels), this introduces **24.8 million additional ray traversal queries**, increasing frame latency by 6–10 ms and defeating the purpose of real-time variance reduction.

### 2.3 Jacobian Determinant Instabilities & Firefly Explosion
Transferring a path between pixels alters solid angles. Evaluating the shift mapping requires multiplying by the **Jacobian determinant** of the transformation:
$$w_{r \to q} = \frac{\hat{p}_q(T_{r \to q}(\bar{y}))}{\hat{p}_r(\bar{y})} \cdot \left| \det \mathbf{J}_{T_{r \to q}}(\bar{y}) \right|$$
For the hybrid reconnection shift:
$$\left| \det \mathbf{J}_{\text{recon}} \right| = \frac{G(x_1, y_2)}{G(y_1, y_2)} = \frac{\cos\theta_{x_1} \cos\theta_{y_2 \to x_1} \, \|y_1 - y_2\|^2}{\cos\theta_{y_1} \cos\theta_{y_2 \to y_1} \, \|x_1 - y_2\|^2}$$

**Failure Mode**: When $x_1$ is close to $y_2$ (e.g. geometric creases, contact corners, near geometry), the denominator $\|x_1 - y_2\|^2 \to 0$, causing the Jacobian to blow up to infinity. This produced extreme fireflies, energy inflation, and blinding flashes that corrupted frame accumulation. Clamping the Jacobian eliminated fireflies but introduced significant energy loss and darkened indirect lighting.

### 2.4 High-Roughness vs. Specular Incompatibility
Reconnection shifts are only valid when the material at $x_1$ has a broad diffuse lobe that can evaluate scattering toward $y_2$.
- For **dielectrics, conductors, and anisotropic metals** (which Pathways now supports in Tier 2), the phase function is a delta or near-delta distribution.
- The probability that the specular lobe at $x_1$ reflects light directly toward $y_2$ is essentially zero.
- Reconnecting to $y_2$ produces black results.
- Overcoming this requires **Manifold Exploration** (using Newton-Raphson solvers to find a valid specular reflection path), which causes severe SIMD thread divergence and register spilling on GPU compute architectures.

### 2.5 Temporal Correlation & Boiling
Without **Pairwise MIS (P-MIS)** or defensive resampling:
- Reservoirs accumulate sample weight over successive frames ($M \to M_{\text{max}}$).
- High-weight historical samples dominate a local pixel neighborhood.
- When camera motion causes disocclusion or dynamic geometry moves, these over-weighted reservoirs cannot be cleanly replaced in a single frame.
- The result is objectionable **temporal boiling**, swimming noise, and ghosting trails behind moving objects.

---

## 3. Hardware & Memory Overhead on AMD RDNA 4 (`gfx1201`)

Implementing a mathematically complete ReSTIR PT engine (Lin et al., 2022/2023) on Pathways' target architecture introduces severe hardware penalties:

| Resource | Pure Wavefront Path Tracing (Current) | ReSTIR PT (GRIS + P-MIS) | Impact on RDNA 4 |
| :--- | :--- | :--- | :--- |
| **Reservoir Storage** | 0 bytes | 48–64 bytes per pixel (ping-pong: 96–128 bytes) | **1.06 GB VRAM** at 4K resolution dedicated purely to reservoir state |
| **VRAM Bandwidth** | Streaming ray queue writes | Random-access spatial/temporal reads & writes | Slashes L2 cache hit rate; causes memory bus saturation |
| **Register Pressure (VGPRs)** | 38 VGPRs (Max Occupancy: 100%) | 80–128+ VGPRs in reuse passes | Halves active SIMD waves per Dual Compute Unit (WGP) |
| **Ray Traversal Budget** | 1 bounce shade + 1 shadow + 1 intersect | + 2 to 4 visibility rays per pixel for reconnection | Adds 16M–33M ray queries/frame at 4K |
| **Pipeline Divergence** | Coherent (sorted by material archetype) | High (divergent reconnection topologies) | Degrades Wave32 execution efficiency |

---

## 4. Expected Outcome & Performance Projection

If full ReSTIR PT were implemented in Pathways:
- **Visual Quality**: Reduced variance on static diffuse-to-diffuse indirect lighting in scenes like Cornell Box.
- **Artifacts**: Persistent failure modes on specular/metallic geometry (Damaged Helmet visor, conductors), light leaks on complex thin geometry, and ghosting during dynamic camera motion.
- **Latency Impact**:
  - Current 4K Pure Wavefront Latency: **5.3 ms – 8.9 ms** (112–186 FPS).
  - Projected ReSTIR PT Latency: **12.0 ms – 18.5 ms** (54–83 FPS).
  - Fails the project's sub-8ms 4K real-time target.

---

## 5. Architectural Verdict & Strategic Recommendation

> [!CAUTION]
> **Verdict: DO NOT PURSUE ReSTIR PT.**
> Having twice attempted ReSTIR and observed its mathematical brittleness, high memory bandwidth footprint, and register bloat, investing further engineering effort into ReSTIR PT is **not justified**.

### Strategic Alternatives for Pathways:
1. **Neural Radiance Caching (NRC)**: Eliminates secondary bounce rays altogether by querying a Wave32 WMMA-accelerated neural representation, providing noise-free indirect illumination with zero light leaks and bounded execution time.
2. **BMFR & Temporal Radiance Accumulation**: Pathways incorporates high-throughput BMFR regression (`--bmfr`) and motion-vector guided temporal accumulation (`--temporal-accum`), resolving residual Monte Carlo noise at sub-millisecond overhead without altering path integration measures.
3. **Pure Wavefront Scaling**: The dual AMD Radeon AI PRO R9700 setup delivers 18+ GigaRays/s. Direct Monte Carlo path tracing remains the most physically robust, artifact-free, and driver-reliable foundation.
