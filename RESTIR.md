# ReSTIR: Reservoir-Based Spatiotemporal Importance Resampling & Vulkan 1.4 DGC Architecture

## 1. Executive Summary & Problem Formulation

In real-time Monte Carlo path tracing at **1 Sample Per Pixel (SPP)**, light transport simulation is constrained by variance:
- **Direct Illumination (ReSTIR DI):** In scenes with numerous complex light sources (hundreds of analytical lights, spot lights, or emissive triangles), picking a single random light uniformly per pixel yields extreme variance and noise. Precomputed CDFs (such as light BVHs or octrees) are expensive to update for dynamic scenes and do not account for local surface BRDFs or visibility.
- **Indirect Path Tracing (ReSTIR PT / GI):** Indirect bounces scatter into high-dimensional path space. Screen-space heuristic denoisers (such as SVGF or A-SVGF) attempt to blur this noise, but they introduce severe temporal lag during camera motion, blur away high-frequency contact shadows, and cause perceptible visual "boiling".

**ReSTIR (Reservoir-based Spatiotemporal Importance Resampling)** fundamentally transforms real-time rendering: instead of tracing additional rays, it **reuses existing samples across space and time** through streaming weighted reservoir sampling. By evaluating hundreds of virtual candidates per pixel using inexpensive unshadowed target functions, ReSTIR achieves the sample efficiency of 32–128 SPP brute-force path tracing while tracing only **a single shadow ray per pixel**.

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│                              ReSTIR Conceptual Flow                               │
└───────────────────────────────────────────────────────────────────────────────────┘

  Frame (t-1) Reservoirs         Initial Candidates (Frame t)         Neighbor Reservoirs
           │                                    │                               │
           ▼                                    ▼                               ▼
    ┌──────────────┐                     ┌──────────────┐                ┌──────────────┐
    │  Temporal    │ ──── Resample ────> │ Stream Sample│ <── Resample ─ │   Spatial    │
    │  Reprojection│                     │  Reservoir   │                │   Exchange   │
    └──────────────┘                     └──────┬───────┘                └──────────────┘
                                                │
                                                ▼
                                  ┌───────────────────────────┐
                                  │ ONLY 1 FINAL SHADOW RAY   │
                                  │   (Hardware Ray Query)    │
                                  └───────────────────────────┘
```

---

## 2. Mathematical Foundation & Algorithmic Principles

### 2.1 Streaming Resampled Importance Sampling (RIS) & Chao's Algorithm

Traditional Importance Sampling selects a sample $y$ from a proposal distribution $q(y)$ to estimate an integral $I = \int f(x) \, dx$. Resampled Importance Sampling (RIS) draws $M$ candidate samples $y_1, y_2, \dots, y_M$ from $q(y)$, and then selects a single sample $y$ with probability proportional to a target function $\hat{p}(y)$ (an unshadowed approximation of $f(y)$):

$$
P(y = y_i) = \frac{w_i}{\sum_{j=1}^M w_j}, \quad \text{where } w_i = \frac{\hat{p}(y_i)}{q(y_i)}
$$

To avoid storing all $M$ candidates in memory, ReSTIR uses **Chao's streaming algorithm** (Weighted Reservoir Sampling) to maintain a single fixed-size **Reservoir** struct per pixel in GPU memory:

```cpp
struct ReservoirDI {
    uint32_t sampleID;   // Index of the selected light or triangle
    vec2     sampleUV;   // Surface parametric coordinates on the light source
    float    wSum;       // Sum of all candidate weights seen so far
    float    M;          // Total effective count of candidates considered
    float    W;          // Unbiased Monte Carlo evaluation weight
};
```

For each incoming candidate $(y_i, q_i)$ with target weight $w_i = \hat{p}(y_i) / q(y_i)$:
1. Accumulate weight: $w_{\text{sum}} \gets w_{\text{sum}} + w_i$
2. Update count: $M \gets M + 1$
3. Accept candidate: With probability $P = \frac{w_i}{w_{\text{sum}}}$, set $y \gets y_i$
4. Compute final unbiased weight:
   $$
   W = \frac{1}{\hat{p}(y)} \cdot \left( \frac{w_{\text{sum}}}{M} \right)
   $$

### 2.2 Unshadowed Target Function $\hat{p}(y)$
Crucially, evaluating candidate weights does **not** trace any shadow rays. The target function $\hat{p}(y)$ is an unshadowed radiance estimate:

$$
\hat{p}(y) = \left\| f_r(\mathbf{x}, \mathbf{\omega}_o, \mathbf{\omega}_i) \cdot L_e(y) \cdot \frac{\cos\theta_{\mathbf{x}} \cos\theta_{y}}{\|\mathbf{x} - y\|^2} \right\|
$$

where $f_r$ is the surface BRDF (Lambertian diffuse + Cook-Torrance specular), $L_e(y)$ is the light emission, and the geometric coupling term accounts for distance and orientation.

### 2.3 Temporal Resampling & History Clamping
Pixels in the current frame $t$ reproject their world-space surface coordinate $\mathbf{x}$ to screen-space coordinates $\mathbf{u}_{t-1}$ in frame $t-1$ using camera velocity vectors:
- If the reprojected surface is consistent (depth difference $< 5\%$, normal angle $< 25^\circ$), the historical reservoir $R_{\text{prev}}$ is retrieved.
- To prevent stale temporal feedback loops and ensure agility under dynamic lighting, the historical candidate count is clamped:
  $$
  M_{\text{prev}}' = \min(R_{\text{prev}}.M, M_{\text{max}}), \quad \text{where } M_{\text{max}} \in [20, 30]
  $$
- The historical reservoir is resampled as a single combined candidate with weight:
  $$
  w_{\text{prev}} = \hat{p}(\mathbf{x}, R_{\text{prev}}.y) \cdot R_{\text{prev}}.W \cdot M_{\text{prev}}'
  $$
- The pixel's reservoir is updated using standard streaming combination.

### 2.4 Spatial Resampling & Edge-Stopping Bilateral Filtering
Surfaces exhibit strong spatial coherence. In a spatial pass, each pixel selects $K$ neighbors within a screen-space radius $r \in [4, 16]$ pixels:
- **Edge-Stopping Bilateral Weights:**
  Candidates from neighbor $\mathbf{x}_n$ are rejected if:
  $$
  |\mathbf{n} \cdot \mathbf{n}_n| < 0.8 \quad \text{or} \quad \frac{|z - z_n|}{z} > 0.05
  $$
- **Combining Reservoirs:**
  Each valid neighbor reservoir $R_n$ is treated as a candidate sample with weight:
  $$
  w_n = \hat{p}(\mathbf{x}, R_n.y) \cdot R_n.W \cdot R_n.M
  $$
- By gathering 3–4 spatial neighbors over 1 or 2 iterations, the effective candidate pool reaches **100+ high-quality light samples per pixel**.

### 2.5 Deferred Occlusion Verification (The 1-Ray Win)
All candidate filtering, temporal reprojection, and spatial sharing occur without evaluating visibility. Only the **single winning candidate sample** per pixel is tested with a hardware ray query (`isShadowOccluded`):
- If unoccluded, the final direct irradiance contribution added to the pixel is:
  $$
  L_d = f_r(\mathbf{x}, \mathbf{\omega}_o, \mathbf{\omega}_i) \cdot L_e(y) \cdot \frac{\cos\theta_{\mathbf{x}} \cos\theta_{y}}{\|\mathbf{x} - y\|^2} \cdot W
  $$
- If occluded, $L_d = 0$.

---

## 3. ReSTIR DI vs. ReSTIR PT / GI

| Attribute | ReSTIR DI (Direct Illumination) | ReSTIR PT / GI (Indirect Paths) |
| :--- | :--- | :--- |
| **Resampled Domain** | 1D/2D Light Indices & Surface UVs | Multi-Segment High-Dimensional Paths $\bar{x} = (x_0, x_1, \dots, x_k)$ |
| **Shift Mapping Required** | Trivial (direct line of sight to light) | Complex (Hybrid Reconnection, RNR, Manifold Exploration) |
| **Jacobian Determinants** | $|\det \mathbf{J}| = 1$ | $|\det \mathbf{J}| = \frac{G(x_1, y_k)}{G(y_1, y_k)}$ (coupling ratio) |
| **Target Function** | Unshadowed direct BRDF $\times$ light | Multi-bounce unshadowed path throughput |
| **Visibility Tests** | 1 shadow ray per pixel | 1 reconnection shadow ray per pixel |
| **Complexity & Overhead** | Low (~0.8 ms – 1.5 ms at 4K) | Moderate (~2.0 ms – 4.5 ms at 4K) |
| **Primary Benefit** | Clean direct lighting & soft shadows for 10,000+ lights | Clean multi-bounce diffuse GI & specular caustics |

---

## 4. Vulkan 1.4 & Device-Generated Commands (DGC) Acceleration

While naive ReSTIR drastically reduces total ray counts, a fixed screen-grid compute implementation introduces distinct GPU execution inefficiencies:
1. **Thread Divergence in Visibility Passes:** On a 4K viewport ($3840 \times 2160$), 30%–50% of screen pixels do not require a shadow ray (sky dome pixels, fully occluded surfaces, surfaces with zero diffuse/specular albedo).
2. **Fixed-Grid Compute Dispatch Waste:** Dispathing uniform $3840 \times 2160$ grids across all spatial passes executes redundant work over flat, static surfaces with saturated temporal histories.
3. **CPU-GPU Synchronization Latency:** Multi-pass ReSTIR pipelines (generate $\to$ temporal $\to$ spatial $\to$ shadow) require multi-stage command submission.

Vulkan 1.4 standardized **Device-Generated Commands (`VK_EXT_device_generated_commands`)**, enabling the GPU command processor to record and dispatch indirect work streams **entirely on-chip**:

```
┌───────────────────────────────────────────────────────────────────────────┐
│                      Vulkan 1.4 DGC ReSTIR Pipeline                       │
└───────────────────────────────────────────────────────────────────────────┘
                                      │
  [Compute Pass 1: Spatiotemporal Resampling]
  • Merges reservoirs across time & space
  • Evaluates unshadowed target functions
  • Filters edge-stopping normals & depth
                                      │
                                      ▼
  [Compute Pass 2: Work Compaction & DGC Token Generation]
  • Filters active, surviving reservoirs needing occlusion verification
  • Compacts ray indices into `ActiveRayQueue` via atomic counter
  • Writes `VkTraceRaysIndirectCommandKHR` and DGC Token Buffer ON-CHIP
                                      │
                                      ▼
  [GPU Command Engine: vkCmdExecuteGeneratedCommandsEXT]
  • Reads tokens directly from device VRAM
  • Dispatches `vkCmdTraceRaysIndirectKHR` or indirect compute dispatches
  • 100% SIMD lane occupancy (ZERO idle lanes / ZERO wave divergence)
                                      │
                                      ▼
  [Final Hardware Shadow Validation Pass]
  • Executes compacted shadow ray queries
  • Applies unbiased weight $W$ to irradiance output
```

### 4.1 Dense SIMD Compaction via Indirect Ray Tracing (`vkCmdTraceRaysIndirectKHR`)
Instead of launching an $8.3\text{M}$-thread ray query dispatch where half of the waves encounter early-outs:
1. Spatial resampling writes surviving active pixel coordinates into a compact linear queue (`ActiveShadowRayBuffer`).
2. An atomic counter computes `activeRayCount`.
3. A 64-thread classifier kernel writes an indirect dispatch payload:
   ```cpp
   VkTraceRaysIndirectCommandKHR {
       .width = activeRayCount,
       .height = 1,
       .depth = 1
   };
   ```
4. `vkCmdTraceRaysIndirectKHR` (or DGC indirect compute dispatch) launches **only** the active rays. Every single wave on AMD RDNA 4 (Wave32) executes with 100% active SIMD lanes, maximizing hardware Ray Accelerator utilization.

### 4.2 On-Chip Light Type Binning & Shader Binding Table (SBT) Switching
In complex scenes with mixed light types (directional sun, point/spot lights, emissive mesh geometry):
- Traditional shaders branch dynamically on `lightType`, triggering register bloat and execution divergence.
- Using DGC token sequences (`VK_INDIRECT_COMMANDS_TOKEN_TYPE_SHADER_GROUP_INDEX_EXT`), a GPU classification pass bins surviving reservoirs into specialized queues (e.g. `AnalyticQueue`, `MeshLightQueue`).
- DGC dispatches separate specialized indirect ray tracing passes with dedicated SBT offsets and push constants on the fly, eliminating monolithic uber-shader stalls.

### 4.3 Adaptive Spatial Sizing
- Static regions with full temporal convergence ($M \ge M_{\text{max}}$) require zero spatial passes.
- High-variance disoccluded regions require aggressive spatial filtering.
- A variance-analysis compute pass emits indirect compute dispatch tokens (`vkCmdDispatchIndirect`), sizing spatial filter workgroups only over areas of high residual variance.

---

## 5. Pathways Implementation Roadmap

### Phase 1: Toggleable ReSTIR DI Core (Baseline Integration)
- **Goal:** Implement clean, unbiased Direct Illumination reservoir resampling across all scene lights with a zero-cost toggle (`--restir-di` and ImGui HUD checkbox).
- **Architecture:**
  - Double-buffered ping-pong reservoir buffers in device-local VRAM (`2 x (width * height * sizeof(ReservoirDI))`).
  - Compute shader pass 1: Initial candidate generation (3–4 local light samples per pixel).
  - Compute shader pass 2: Temporal reprojection and history clamping.
  - Compute shader pass 3: Spatial resampling with bilateral normal/depth filtering.
  - Hardware shadow ray query in `raytrace.rgen` / `raytrace.rchit` evaluating the winning reservoir.
- **Verification:** Benchmarking frametimes and noise reduction across `CornellBox`, `Classroom`, `LivingRoom`, and `CoffeeMaker`.

### Phase 2: Vulkan 1.4 DGC Work Compaction & Indirect Trace
- **Goal:** Eliminate wave divergence in the final shadow validation pass.
- **Architecture:**
  - Compaction compute kernel using atomic wave-level prefix sums.
  - Generation of `VkTraceRaysIndirectCommandKHR` command tokens on-chip.
  - Execution via `vkCmdExecuteGeneratedCommandsEXT`.

### Phase 3: ReSTIR PT / GI (Multi-Bounce Path Resampling)
- **Goal:** Extend reservoir resampling to indirect secondary bounces.
- **Architecture:**
  - Hybrid Reconnection shift mapping for rough/diffuse hits.
  - Specular manifold exploration for caustic light paths.
  - Pairwise MIS (P-MIS) to prevent boiling and temporal correlation clumping.

---

## 6. Empirical Findings, Telemetry, and Post-Mortem Rationale for Removal

Following rigorous implementation and profiling across Cornell Box, Classroom, and Many-Lights scenes on dual AMD Radeon AI PRO R9700 GPUs (RDNA 4, `gfx1201`), **ReSTIR was found to be a severe net-negative architectural trade-off in real-time rendering**. 

As of September 2026, ReSTIR DI and ReSTIR GI have been **completely excised from the Pathways codebase**. This section documents the empirical telemetry, performance degradation, image degradation modes, and architectural analysis for archival reference.

### 6.1 Empirical Telemetry: Pure Wavefront vs. ReSTIR

All tests executed at **Native 4K (3840 × 2160), 1 SPP, 4 Bounces, FP16 HDR** on AMD RDNA 4 hardware:

#### A. Procedural Many-Lights Stress Test (64 Analytical Quad Lights)
```
Pure Native Wavefront Path Tracer:
  Rendered Frames:     1222
  Average Frame Time:  7.438 ms (134.4 FPS) [Min: 6.812 ms, Max: 9.140 ms]
  Primary RT GPU:      7.294 ms
  Ray Throughput:      4.46 GigaRays/sec
  Sub-8ms Budget:      ACHIEVED (120 Hz Target Met)
  Pipeline Stages:
    - Classify (Primary RayGen): 0.792 ms
    - Bounce 0: Shade: 1.906 ms | Shadow: 0.812 ms | Intersect: 0.941 ms
    - Bounce 1: Shade: 0.842 ms | Shadow: 0.315 ms | Intersect: 0.720 ms
    - ReSTIR GI Pass: 0.000 ms

ReSTIR Enabled (DI + GI):
  Rendered Frames:     1222
  Average Frame Time:  9.857 ms (101.5 FPS) [Min: 7.838 ms, Max: 14.128 ms]
  Primary RT GPU:      9.713 ms
  Ray Throughput:      3.37 GigaRays/sec
  Sub-8ms Budget:      EXCEEDED (Failed 120 Hz Target)
  Pipeline Stages:
    - Classify (Primary RayGen): 0.804 ms
    - Bounce 0: Shade: 3.263 ms (+71.2% slower) | Shadow: 0.825 ms | Intersect: 0.952 ms
    - Bounce 1: Shade: 0.910 ms | Shadow: 0.320 ms | Intersect: 0.731 ms
    - ReSTIR GI Pass: 1.445 ms
```
- **Net Cost of ReSTIR:** **+2.419 ms per frame (+32.5% slower)**.
- **Visual Variance Benefit:** **0% perceptible noise reduction**. Single-pixel 1 SPP Monte Carlo noise remained visibly unresolved.
- **Motion Quality:** Heavy smearing, ghost trails, and historical dragging under camera movement.

#### B. Architectural Interior Benchmark (Classroom Scene, Dynamic Camera Motion)
- **Pure Wavefront Path Tracer:** **7.64 ms (130.8 FPS) — sub-8ms budget achieved**. Crisp contact shadows under desks/chairs (deep shadow retention: 13.5%), sharp geometric silhouettes, zero ghosting.
- **ReSTIR Enabled:** **10.3 – 12.6 ms (79.3 FPS) — sub-8ms budget failed**. Frame rate collapsed by 39%. Camera movement caused trailing halos behind chairs, delayed shadow response, and severe temporal lag.

---

### 6.2 The Three Fundamental Failure Modes of ReSTIR

#### 1. Sampling vs. Filtering Confusion
ReSTIR is purely an *importance resampling* algorithm — it selects which light source or indirect ray candidate to evaluate. **It is NOT an image filter, and it cannot replace a denoiser**.
At 1 SPP, evaluating an optimal light candidate still yields stochastic Monte Carlo variance per pixel. To turn 1 SPP noisy radiance into clean real-time imagery requires a spatial/wavelet reconstructor (such as an edge-stopping À-Trous filter). ReSTIR's temporal reservoir accumulation attempted to smooth this variance over time, but at the cost of catastrophic temporal lag and ghosting during dynamic camera or object motion.

#### 2. Excessive VRAM Memory Bandwidth and Cache Thrashing
Maintaining ping-pong reservoir state at 4K ($3840 \times 2160 = 8,294,400$ pixels):
- ReSTIR DI Reservoirs (32 bytes $\times 2$ ping-pong): **530.8 MiB**
- ReSTIR GI Reservoirs (32 bytes $\times 2$ ping-pong): **530.8 MiB**
- Raw GI Sample Queue (32 bytes): **265.4 MiB**
- **Total VRAM Allocated to Reservoirs: ~1.32 GiB**.
Every single frame, the compute kernels streamed hundreds of megabytes through the L2/Infinity Cache to read and write reservoir metadata, competing directly with BVH traversal, triangle vertex fetching, and ray queues.

#### 3. Register Pressure & Compute Shader Spills
Evaluating 4-candidate Chao's WRS, spatial bilateral edge-stopping neighbor gathers, and unshadowed target functions inside the material shade shaders expanded register usage dramatically:
- In `wavefront_shade_diffuse.comp`, VGPR pressure increased from 38 to 64+ registers, cutting wave occupancy on RDNA 4 Compute Units in half.
- As observed in the telemetry, **Bounce 0 Shade execution jumped from 1.906 ms to 3.263 ms (+71.2% slower)** simply from running the ReSTIR DI candidate loop.

---

### 6.3 Conclusion and Architectural Commitment

Pathways is designed for maximum deterministic rendering speed, predictable frame pacing, and absolute physical correctness. The empirical evidence decisively demonstrated that:
1. ReSTIR introduces substantial memory and compute overhead (+32.5% frame time penalty).
2. It causes unacceptable motion artifacts (ghosting, smear, temporal disocclusion lag).
3. It does not eliminate the need for spatial reconstruction filters.

Consequently, ReSTIR has been permanently removed. Pathways relies exclusively on its high-throughput native Wavefront pipeline, dynamic SPP regulation, and lean spatial reconstruction (À-Trous / AMD FidelityFX denoiser) to deliver high-framerate 4K hardware path tracing.
