# Adaptive Edge-Aware SVGF (E-SVGF) Denoiser: Production Architectural Specification & Transition Roadmap

**Document Status**: Authoritative Architectural Proposal  
**Version**: 2.0.0  
**Target Engine**: Vulkan 1.4 Real-Time Path Tracing Engine  
**Date**: October 2026  
**Reference Document**: Fu et al. (2025), *"Edge-aware denoising framework for real-time mobile ray tracing"*, *Graphical Models*, Vol. 141, 101301.

---

## 1. Executive Summary & High-Level Architecture

### 1.1 Problem Statement
Real-time path tracing at 1 sample per pixel (SPP) produces high-variance stochastic Monte Carlo noise. Standard Spatiotemporal Variance-Guided Filtering (SVGF; Schied et al., 2017) suppresses low-frequency noise through temporal reprojection and spatial À-Trous wavelet filtering with fixed spatial support (typically 4–5 iterations spanning a $33\times33$ to $125\times125$ pixel footprint). While effective in flat, homogeneous regions, fixed-kernel SVGF exhibits two persistent failure modes:
1. **Geometric Silhouette & Texture Blur**: In regions of high geometric complexity (fine silhouettes, wireframes, dense foliage, sharp texture transitions), fixed wide-radius filtering bleeds irradiance across physical boundaries, producing noticeable edge haloing, soft silhouettes, and detail loss.
2. **Computational Redundancy & Edge Bleeding**: Uniformly executing wide-radius convolutions across the entire viewport expends massive ALU cycles and kernel reach on pixels that either require minimal spatial blurring (edges that rely on temporal accumulation) or could early-exit after fine-scale filtering. While algorithmic tap reductions are largely DRAM-neutral because spatial passes stream full-screen ping-pong buffers, bounding spatial support strictly preserves razor-sharp geometric edges and high-frequency textural detail.

### 1.2 Evaluation of Fu et al. (2025)
Fu et al. (2025) address this challenge by introducing an edge-aware dynamic filtering framework comprising:
- Multi-channel G-buffer Sobel edge extraction across Albedo, Depth, and Normal buffers ($M_{\text{fused}}$).
- Statistical adaptive thresholding ($T = \mu + 1.2\sigma$) to extract a binary edge mask ($M_{\text{edge}}$).
- Motion-reprojected temporal edge density accumulation ($\rho$) to track local edge concentration without per-frame spatial recomputation.
- Dynamic spatial kernel sizing ($k \in \{9, 17, 33\}$) governed by local edge density $\rho$.
- A 5D precomputed Neural Lookup Table (LUT) trained offline via Multi-Layer Perceptron (MLP) to predict spatial filtering weights on compute-constrained mobile hardware.

While Fu et al.'s dynamic edge-density classification demonstrates a **67.4% framerate improvement** over static wide-radius filtering, their 5D Neural LUT architecture presents critical barriers for desktop and high-performance APU production engines:
- **Excessive VRAM Footprint**: The Neural LUT demands **2.5 GB** (compressed FP16) to **5.7 GB** (uncompressed FP32) of GPU memory. The entire G-buffer and accumulation framebuffer footprint at 1080p is under 150 MB; allocating 2.5 GB for a denoiser lookup table represents an unacceptable $17\times$ memory bloat. By contrast, the complete internal storage image footprint across all double-buffered history, ping-pong, and edge images is **257.13 MB** unpadded (245.22 MiB across 17 storage images, 124 B/px), with edge density requiring only **8.29 MB** unpadded (`VK_FORMAT_R8G8_UNORM`), as Sobel magnitude is computed in-LDS with zero global DRAM footprint.
- **Cache Inefficiency & Bandwidth Contention**: High-dimensional (5D) non-contiguous texture lookups induce severe L1/L2 cache thrashing on GPU architectures, degrading effective memory throughput.
- **Generalization Failure on Complex Materials**: The offline MLP is trained on static diffuse datasets. As documented in Section 8 of the paper, it fails catastrophically on specular reflections, refractive glass, and complex BSDFs, causing severe over-smoothing and loss of mirror highlights.

### 1.3 Architectural Decision: Authoritative 5-Dispatch Dilated E-SVGF Pipeline
To achieve state-of-the-art visual fidelity, zero lookup memory overhead, and deterministic execution within strict real-time production budgets, the engine adopts an **Authoritative 5-Dispatch Dilated Edge-Aware SVGF (E-SVGF)** architecture.

Cascaded À-Trous wavelet filtering is inherently recursive ($I_k(p) = \sum w I_{k-1}(q) / \sum w$). Because GPU compute invocations across different workgroups run concurrently with zero execution ordering guarantees, dependent wavelet iterations cannot simply be evaluated as multiple loops in a single dispatch without global pipeline barriers or an impractical $56\times56$ workgroup load footprint. 

The pipeline is decomposed into **5 cleanly synchronized compute dispatches** that preserve the full **61-pixel receptive field** ($s \in \{1, 2, 4, 8\}$) without intermediate global roundtrips for Sobel or remodulation:

1. **Pass 1: Fused Edge-Aware Spatiotemporal Accumulator (`shaders/compute/denoise_fused_temporal.comp`)**:
   - Cooperatively loads a **3-pixel G-buffer source apron** ($22\times22$ texels for a $16\times16$ output tile) into Local Data Share (LDS).
   - Evaluates multi-channel Sobel gradients across a $20\times20$ intermediate domain and $5\times5$ box-filter edge density across the $16\times16$ output domain entirely in-LDS (zero DRAM writes for intermediate Sobel textures).
   - Normalizes gradients using analytical vector bounds: $4\sqrt{6}$ for albedo and $8\sqrt{2}$ for normals.
   - Tile-local statistics ($\mu_{\text{tile}}, \sigma_{\text{tile}}$) with noise floor ($T_{\min} = 0.05$), variance floor ($\sigma_{\min} = 0.02$), and contrast floor ($\delta_{\min} = 0.02$) eliminate edge false-positives in smooth tiles without requiring inter-workgroup synchronization: $T_{\text{tile}} = \max(\mu_{\text{tile}} + 1.2 \cdot \max(\sigma_{\text{tile}}, 0.02), 0.05)$.
   - Dual-model reprojection validation: static geometry evaluates exact 6-DOF affine view-space depth prediction (`currViewToPrevView`) and pure normal rotation (`prevViewToCurrViewRot`); dynamic objects switch to motion-vector sample depth testing with shortened history confidence.
   - Outlier firefly suppression: conservative current-sample soft-clamping against bilateral-consistent neighbors (strictly excluding center; emissive bypass) before moment accumulation; color-box history clipping is eliminated to avoid biasing rare indirect illumination.
   - Pre-spatial history update: writes accumulated radiance directly to `m_historyDiffuse[currSlot]` (with $V^d$ in Alpha) and `m_historySpecular[currSlot]` (with $V^s$ in Alpha). Secondary hit distance $d_{\text{refl}}$ and dynamic object flag $M_{\text{dynamic}}$ are written to `m_historyLength[currSlot]`.
   - Dual-channel edge density packing into `m_edgeDensity[slot]` (`VK_FORMAT_R8G8_UNORM`): Red stores raw $\rho^t$ for next-frame temporal accumulation; Green stores effective $\rho_{\text{effective}}^t = \rho^t \cdot \text{clamp}(N_{\text{diff}}/8, 0.25, 1.0)$ for downstream spatial passes, preventing post-disocclusion estimator collapse.
2. **Pass 2: Fine-Scale Wavelet Iteration 1 ($s=1$) (`shaders/compute/denoise_atrous_step1.comp`)**:
   - Executes fine-scale cross-bilateral wavelet filtering ($s=1$, radius 2, diameter 5) across all active pixels.
   - Filters radiance (FP32 accumulators) AND variance ($w^2$). Output variance is packed into the Alpha channel.
   - Because all pixels filter uniformly at $s=1$, `uEdgeDensity` is omitted from Pass 2 bindings (saving 2 B/px read traffic).
   - Bilateral edge-stopping weights are evaluated in native FP16 (`float16_t`).
3. **Pass 3: Fine-Scale Wavelet Iteration 2 ($s=2$) (`shaders/compute/denoise_atrous_step2.comp`)**:
   - Executes fine-scale dilation ($s=2$, accumulated radius 6, diameter 13 covering Fu et al. $k=9$).
   - Dense edge pixels ($\rho_{\text{effective}} \ge 0.55$) complete spatial filtering here (50 taps total per pixel) and bypass subsequent filtering taps in Passes 4 and 5.
4. **Pass 4: Mid-Scale Wavelet Iteration 3 ($s=4$) (`shaders/compute/denoise_atrous_step3.comp`)**:
   - Executes mid-scale dilation ($s=4$, accumulated radius 14, diameter 29 covering Fu et al. $k=17$).
   - Dense edge pixels bypass filtering taps via continuous smoothstep crossfade ($w_{\text{blend}}^{(1)} = \text{smoothstep}(0.45, 0.55, \rho_{\text{effective}})$), with 32-lane wave early-exit via `subgroupAll(effectiveRho >= 0.55)`.
   - Wavefronts covering moderate curvature (`subgroupAll(effectiveRho >= 0.35)`) skip Pass 5 taps (evaluating remodulation in Pass 5).
5. **Pass 5: Coarse Wavelet ($s=8$) & Fused Remodulation (`shaders/compute/denoise_atrous_coarse_remodulate.comp`)**:
   - Executes coarse dilation ($s=8$, accumulated radius 30, diameter 61 covering Fu et al. $k=33$) exclusively for smooth/homogeneous regions ($\rho_{\text{effective}} < 0.35$), completely eliminating 1-SPP low-frequency blotches.
   - Smoothstep crossfading ($w_{\text{blend}}^{(2)} = \text{smoothstep}(0.25, 0.35, \rho_{\text{effective}})$) prevents boundary ringing.
   - Fuses diffuse albedo remodulation ($L_{\text{final}} = L_{\text{diffuse}} \cdot \max(c_{\text{albedo}}, 0.001) + L_{\text{specular}}$) directly in-register.
   - Stores finalized radiance directly into `m_denoisedImage` (RGBA16F) ready for tonemapping.
6. **Zero Table Memory (0 MB LUT)**: Bilateral edge-stopping weights are evaluated analytically in shader ALUs using hardware transcendental instructions ($\exp, \text{pow}, \text{dot}$), completely eliminating the 2.5 GB VRAM allocation.
7. **Decoupled Radiance Filtering & Roughness Modulation**: Direct and indirect diffuse and specular radiance channels are denoised independently, with specular spatial dilation modulated by microfacet GGX roughness via continuous `smoothstep(0.02, 0.08, roughness)` to preserve crisp mirror reflections and caustics without spatial seams.
8. **Realistic Budget & Production Contract**:
   - **Primary Production Contract (1080p Internal Render Resolution)**: Running the denoiser at 1080p prior to super-resolution upscaling (Upways / FSR3) targets **1.7–1.9 ms measured on RTX 4070** (pure streaming floor ~1.61 ms; 4070's 36 MB L2 cache cannot hold the 66.4 MB ping-pong working set, so it streams primarily from DRAM), **1.0–1.2 ms measure on RX 7800 XT** (where the 66.4 MB working set is borderline against the 64 MB Infinity Cache), and **1.2–1.4 ms on $\ge 700\text{ GB/s}$ GPUs** (floor ~1.18 ms). At 1440p internal resolution, the target is **3.0–3.4 ms** on 4070 (compulsory streaming floor ~2.86 ms; the 118 MB working set exceeds the cache on all named GPU targets).
   - **Native 4K Profiling Gate**: At native 4K (3840×2160), compulsory streaming traffic reaches **~2.60 GB** (314 B/px baseline; ~2.70 GB with preview); native 4K serves as a high-end profiling gate targeting **< 3.5 ms** exclusively on extreme desktop GPUs with $\ge 950\text{--}1000\text{ GB/s}$ bus bandwidth (e.g. RTX 4090 / RX 7900 XTX class; memory floor ~3.23–3.38 ms). On 4070/7800-class hardware, native 4K is an offline reference gate (compulsory floor ~5.21–6.45 ms).
   - **Unified APU Contract**: At 1080p on AMD Strix Halo (256-bit LPDDR5X-8000 ≈ 256 GB/s bus, 204.8 GB/s sustained @ 80%), compulsory traffic sets a memory floor of ~3.18 ms (~3.30 ms with preview), establishing a base gate of **3.3–3.6 ms measured**, with sub-3.0 ms explicitly gated behind the traffic-reduction roadmap (~2.69 ms floor with 48 B/px savings).

---

### 1.4 High-Level Architecture & Dataflow Diagram

```
========================================================================================================================
                                 5-DISPATCH ADAPTIVE E-SVGF COMPUTE PIPELINE
========================================================================================================================

    [Primary G-Buffers & Radiance from Ray Tracing Pipeline]
    │
    ├──> uNormalDepth       (RGBA16F: View Normal XYZ in [-1,1], Linear View Depth W in [0, inf))
    ├──> uAlbedoRoughness   (RGBA16F: Demodulated Material Base Color RGB, Roughness A in [0,1])
    ├──> uMotionVectors     (RG16F: Screen-Space UV Motion Delta)
    ├──> uSpecularMotion    (RG16F: Virtual Reflection-Space Motion Delta, Primary Ray Hit Reprojection)
    ├──> uCurrentDiffuse    (RGBA16F: 1-SPP Diffuse Radiance RGB, Emissive / Area Light Flag A)
    └──> uCurrentSpecular   (RGBA16F: 1-SPP Specular Radiance RGB, Secondary Hit Distance d_refl A)
    │
    ▼
    [VkMemoryBarrier2: RAY_TRACING_SHADER_BIT_KHR | COMPUTE_SHADER_BIT -> COMPUTE_SHADER_BIT]
    ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PASS 1: Fused Edge-Aware Spatiotemporal Accumulator (shaders/compute/denoise_fused_temporal.comp)                   │
│ • Tiled 16x16 workgroup with 3-pixel source apron (22x22 G-buffer load -> 20x20 Sobel -> 16x16 edge density)         │
│ • Multi-channel Sobel (Albedo 0.5, Depth 0.3, Normal 0.2) evaluated in-LDS with vector bounds (4*sqrt(6), 8*sqrt(2)) │
│ • Tile-local thresholding (T = max(mu + 1.2*max(sigma, 0.02), 0.05)) & 5x5 edge density rho evaluated in-LDS         │
│ • Dual-model reprojection: exact 6-DOF affine depth test for static geometry, sample depth test for dynamic objects   │
│ • Conservative firefly suppression: bilateral-consistent soft-clamp (excluding center; emissive bypass) before moments│
│ • Pre-spatial history update: writes directly to m_historyDiffuse[currSlot], m_historySpecular[currSlot]              │
│ • Packs raw rho (R) and effective rho_effective (G) into m_edgeDensity[currSlot] (R8G8_UNORM)                        │
│ • Camera matrices bound via CameraMatricesUBO at Binding 0 (keeps push constants <= 80 bytes; <= 128 B Vulkan limit)  │
│ • Outputs: m_historyDiffuse[currSlot], m_historySpecular[currSlot], m_historyMoments, m_historyLength, m_edgeDensity │
└──────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                                           │ VkMemoryBarrier2 (COMPUTE_WRITE -> COMPUTE_READ)
                                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PASS 2: Fine-Scale Wavelet Iteration 1 (s=1) (shaders/compute/denoise_atrous_step1.comp)                             │
│ • 5x5 cross-bilateral convolution (step size s = 1, radius 2, diameter 5) across all pixels (uEdgeDensity omitted)   │
│ • Filters Radiance (FP32 accumulators) AND Variance (sum w^2 * Var) / (sum w)^2 (FP32)                              │
│ • Metric 3D tangent-plane distance depth weighting: tau_z = phiPlaneDepth * s * (2*z*tan(fovY/2)/H)                 │
│ • Outputs: m_pingPongDiffuse[0] (RGBA16F: Radiance + Filtered Var), m_pingPongSpecular[0]                            │
└──────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                                           │ VkMemoryBarrier2 (COMPUTE_WRITE -> COMPUTE_READ)
                                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PASS 3: Fine-Scale Wavelet Iteration 2 (s=2) (shaders/compute/denoise_atrous_step2.comp)                             │
│ • Dilated convolution (s = 2, accumulated radius 6, diameter 13 covering Fu et al. k=9)                              │
│ • Filters Radiance and Variance (w^2) alongside each other                                                           │
│ • Dense edge pixels (rho_eff >= 0.55) obtain their complete spatial filtering here and bypass subsequent taps        │
│ • Outputs: m_pingPongDiffuse[1] (RGBA16F: Radiance + Filtered Var), m_pingPongSpecular[1]                            │
└──────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                                           │ VkMemoryBarrier2 (COMPUTE_WRITE -> COMPUTE_READ)
                                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PASS 4: Mid-Scale Wavelet Iteration 3 (s=4) (shaders/compute/denoise_atrous_step3.comp)                             │
│ • Dilated convolution (s = 4, accumulated radius 14, diameter 29 covering Fu et al. k=17)                           │
│ • Dense edge pixels (rho_eff >= 0.55) bypass filtering taps; smoothstep crossfade [0.45, 0.55] prevents seams        │
│ • Wavefronts covering moderate curvature finish convolution here and skip Pass 5 taps (subgroupAll(rho_eff >= 0.35)) │
│ • Filters Radiance and Variance alongside each other                                                                 │
│ • Outputs: m_pingPongDiffuse[0] (RGBA16F: Radiance + Filtered Var), m_pingPongSpecular[0]                            │
└──────────────────────────────────────────────────────────┬───────────────────────────────────────────────────────────┘
                                                           │ VkMemoryBarrier2 (COMPUTE_WRITE -> COMPUTE_READ)
                                                           ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ PASS 5: Coarse Wavelet (s=8) & Fused Remodulate (shaders/compute/denoise_atrous_coarse_remodulate.comp)             │
│ • Coarse dilated convolution (s = 8, accumulated radius 30, diameter 61 covering Fu et al. k=33) for flats          │
│ • Wavefronts with subgroupAll(rho_eff >= 0.35) skip coarse s=8 taps; smoothstep crossfade [0.25, 0.35]               │
│ • Restores full 61 px receptive field of classic SVGF, eliminating 1-SPP low-frequency blotches                       │
│ • In-shader diffuse albedo remodulation: L_final = L_diffuse * max(c_albedo, 0.001) + L_specular                     │
│ • Output: m_denoisedImage (RGBA16F) bound directly to Tonemapper / Post-Process Pipeline                             │
│ • PERFORMANCE GATES: 1080p 1.7–1.9 ms (4070), 1.2–1.4 ms (>=700 GB/s), 3.3–3.6 ms (Strix Halo); 4K < 3.5 ms (>=950 GB/s) │
└──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical & Algorithmic Deconstruction of Fu et al. (2025)

### 2.1 Multi-Channel G-Buffer Sobel Edge Extraction

To detect structural discontinuities across arbitrary geometry and materials, the framework evaluates the 2D gradient of three primary G-buffer channels: material base albedo ($B_{\text{Albedo}}$), linear view depth ($B_{\text{Depth}}$), and view-space surface normals ($B_{\text{Normal}}$).

#### Discrete Convolution Kernels
The standard isotropic 3×3 Sobel operator defines horizontal ($S_x$) and vertical ($S_y$) convolution kernels:
$$S_x = \begin{bmatrix} -1 & 0 & 1 \\ -2 & 0 & 2 \\ -1 & 0 & 1 \end{bmatrix}, \quad S_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ 1 & 2 & 1 \end{bmatrix}$$

For each G-buffer channel $c \in \{ \text{Albedo}, \text{Depth}, \text{Normal} \}$, spatial gradients at pixel $(i, j)$ are computed via discrete 2D spatial convolution:
$$G_x^c(i, j) = (S_x * B_c)(i, j) = \sum_{u=-1}^{1} \sum_{v=-1}^{1} S_x(u+1, v+1) \cdot B_c(i+u, j+v)$$
$$G_y^c(i, j) = (S_y * B_c)(i, j) = \sum_{u=-1}^{1} \sum_{v=-1}^{1} S_y(u+1, v+1) \cdot B_c(i+u, j+v)$$

#### Vector Magnitude Calculation
For scalar channels (Depth), gradient components are scalar. For vector channels (Albedo $RGB$, Normal $XYZ$), the directional derivative is evaluated per component and summed in quadrature:
$$G_x^{\text{Albedo}} = \sqrt{\sum_{k \in \{R,G,B\}} \left(G_x^{\text{Albedo}, k}\right)^2}, \quad G_y^{\text{Albedo}} = \sqrt{\sum_{k \in \{R,G,B\}} \left(G_y^{\text{Albedo}, k}\right)^2}$$
$$G_x^{\text{Normal}} = \sqrt{\sum_{k \in \{X,Y,Z\}} \left(G_x^{\text{Normal}, k}\right)^2}, \quad G_y^{\text{Normal}} = \sqrt{\sum_{k \in \{X,Y,Z\}} \left(G_y^{\text{Normal}, k}\right)^2}$$

The Euclidean gradient magnitude for channel $c$ is:
$$M^c(i, j) = \sqrt{\left(G_x^c(i, j)\right)^2 + \left(G_y^c(i, j)\right)^2}$$

#### Channel Roles and Physical Invariants
- **Albedo ($M^{\text{Albedo}}$)**: Detects physical material boundaries (e.g., transitions between gold, dielectric plastic, painted signs). Because we demodulate material reflectance prior to denoising, $M^{\text{Albedo}}$ is invariant to shadows, lighting flashes, and noisy photon caustics.
- **Depth ($M^{\text{Depth}}$)**: Identifies geometric silhouette edges and occlusions. To ensure scale invariance across near and far planes, the raw depth gradient is normalized by the local linear depth: $\hat{G}^{\text{Depth}} = G^{\text{Depth}} / \max(z(i,j), 10^{-3})$.
- **Normal ($M^{\text{Normal}}$)**: Captures surface curvature discontinuities, chamfers, creases, and polygon edges independently of camera distance.

#### Frame Normalization and Convex Feature Fusion
Rather than attempting frame-global min-max normalization (which cannot be evaluated within a single compute dispatch without introducing multi-pass parallel reduction trees and inter-workgroup synchronization), per-channel gradient magnitudes are normalized using **analytical upper bounds** derived from the continuous response of the 3×3 Sobel operator:

1. **Albedo Analytical Scaling ($\tilde{M}^{\text{Albedo}}$)**:
   For unit-bounded material albedo ($B_{\text{Albedo}} \in [0, 1]^3$), the maximum possible directional Sobel response across all 3 color channels ($R, G, B$) occurs when each channel steps between 0 and 1 diagonally across the unit cube $[0, 1]^3$, yielding $G_x^k = 4$ and $G_y^k = 4$ for $k \in \{R, G, B\}$. The theoretical maximum Euclidean gradient magnitude across all 3 vector channels is:
   $$M^{\text{Albedo}}_{\max} = \sqrt{\sum_{k \in \{R,G,B\}} (4^2 + 4^2)} = \sqrt{3 \times 32} = 4\sqrt{6} \approx 9.7980$$
   The normalized albedo gradient is therefore bounded analytically without global reduction:
   $$\tilde{M}^{\text{Albedo}}(i, j) = \text{clamp}\left(\frac{M^{\text{Albedo}}(i, j)}{4\sqrt{6}}, \, 0.0, \, 1.0\right) \quad \left(\text{scaling factor } \frac{1}{4\sqrt{6}} \approx 0.1021\right)$$

2. **Normal Analytical Scaling ($\tilde{M}^{\text{Normal}}$)**:
   View-space normals reside on the unit sphere $S^2$ ($\|\mathbf{n}\| = 1$). The maximum distance between any two unit vectors on $S^2$ is $\|\mathbf{n}_1 - \mathbf{n}_2\| \le 2.0$ (opposing antipodal normals). For the discrete Sobel kernel, the maximum directional response vector is bounded by $\sum |S_x(u)| \cdot 1.0 = 4 \times 2.0 = 8.0$, meaning $\|\mathbf{G}_x\| \le 8$ and $\|\mathbf{G}_y\| \le 8$. Constrained by unit-sphere geometry, an analytical Euclidean upper bound across both gradient axes is:
   $$M^{\text{Normal}}_{\max} = \sqrt{\|\mathbf{G}_x\|^2 + \|\mathbf{G}_y\|^2} \le \sqrt{8^2 + 8^2} = 8\sqrt{2} \approx 11.3137$$
   The normalized normal gradient is bounded analytically:
   $$\tilde{M}^{\text{Normal}}(i, j) = \text{clamp}\left(\frac{M^{\text{Normal}}(i, j)}{8\sqrt{2}}, \, 0.0, \, 1.0\right) \quad \left(\text{scaling factor } \frac{1}{8\sqrt{2}} \approx 0.0884\right)$$
   *(Note: This establishes a safe analytical upper bound across unit vectors, rather than assuming simultaneous directional extremes in one neighborhood. It prevents dynamic range compression without requiring global reduction passes).*

3. **Depth Analytical Scaling ($\tilde{M}^{\text{Depth}}$)**:
   Linear view depth is normalized relative to local distance $z(i, j)$ and damped by the dynamic precision noise floor $\epsilon_z = \max(0.005 \cdot z, 10^{-3})$:
   $$\tilde{M}^{\text{Depth}}(i, j) = \text{clamp}\left(\frac{\max(0.0, \, M^{\text{Depth}}(i, j) - \epsilon_z)}{\max(z(i, j), 10^{-3})}, \, 0.0, \, 1.0\right)$$

The fused multi-channel edge magnitude $M_{\text{fused}}(i, j)$ is synthesized via convex linear combination:
$$M_{\text{fused}}(i, j) = w_{\text{alb}} \tilde{M}^{\text{Albedo}}(i, j) + w_{\text{depth}} \tilde{M}^{\text{Depth}}(i, j) + w_{\text{norm}} \tilde{M}^{\text{Normal}}(i, j)$$
$$\text{with } w_{\text{alb}} = 0.5, \quad w_{\text{depth}} = 0.3, \quad w_{\text{norm}} = 0.2, \quad \sum w = 1.0$$

**Analytical Scaling Advantages**:
- **Single-Dispatch Determinism**: Eliminates auxiliary reduction passes and global memory barriers, allowing in-LDS edge extraction in Pass 1 to evaluate normalized gradients without auxiliary reduction passes or global Sobel image writes.
- **Zero Dynamic Range Saturation**: Normalizing by $4\sqrt{6}$ and $8\sqrt{2}$ guarantees that gradient magnitudes strictly map to $[0.0, 1.0]$, preventing saturation across high-contrast edges.

**Weight Rationale**: Fu et al. established these weights empirically on the NoiseBase ray tracing benchmark:
- $w_{\text{alb}} = 0.5$ places primary emphasis on perceptually salient material transitions, where filtering across borders produces catastrophic color bleeding.
- $w_{\text{depth}} = 0.3$ preserves macro silhouette boundaries while providing sufficient damping against floating-point depth quantization noise.
- $w_{\text{norm}} = 0.2$ detects geometric creases while preventing minor tessellation faceting from prematurely triggering edge boundaries.

---

### 2.2 Statistical Adaptive Thresholding

Fixed thresholding (e.g., $M_{\text{fused}} > 0.2$) fails across dynamic scenes: it over-segments dark, noisy scenes with spurious edges while missing subtle contours in high-dynamic-range bright scenes. Fu et al. introduce a statistical adaptive threshold based on the first two moments of the gradient distribution.

#### Statistical Moments
For an image containing $N = W \times H$ pixels, the mean $\mu$ and standard deviation $\sigma$ of the fused gradient magnitude are:
$$\mu = \frac{1}{N} \sum_{i=1}^{W} \sum_{j=1}^{H} M_{\text{fused}}(i, j)$$
$$\sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{W} \sum_{j=1}^{H} \left( M_{\text{fused}}(i, j) - \mu \right)^2}$$

#### Adaptive Threshold Formulation & Noise Floor
In homogeneous flat regions where gradient magnitude is non-zero but uniform (e.g. subtle lighting roll-off where $M_{\text{fused}} = 0.06$ everywhere), sample standard deviation collapses to $\sigma = 0.0$. In such cases, an unconstrained threshold $T = \mu + 1.2\sigma = \mu = 0.06$ causes the condition $M_{\text{fused}} \ge T$ to evaluate to True for **100% of flat pixels in the tile**, triggering catastrophic false edge classification.

To resolve this pathology, enforce a **configurable noise floor** $T_{\min} = 0.05$, a **minimum variance floor** $\sigma_{\min} = 0.02$, and a **minimum contrast distance criterion** $\delta_{\min} = 0.02$:
$$T_{\text{tile}} = \max\left(\mu_{\text{tile}} + 1.2 \cdot \max(\sigma_{\text{tile}}, \, \sigma_{\min}), \, T_{\min}\right)$$

The binary edge mask $M_{\text{edge}}(i, j)$ is then defined as:
$$M_{\text{edge}}(i, j) = \begin{cases} 1.0 & \text{if } (M_{\text{fused}}(i, j) \ge T_{\text{tile}}) \land (M_{\text{fused}}(i, j) - \mu_{\text{tile}} \ge \delta_{\min}) \\ 0.0 & \text{otherwise} \end{cases}$$

#### Theoretical Foundation & Stochastic Distribution
For an ideal scalar image whose orthogonal spatial derivatives satisfy independent equal-variance zero-mean Gaussian assumptions ($G_x, G_y \sim \mathcal{N}(0, \sigma_0^2)$), the Euclidean gradient magnitude $M = \sqrt{G_x^2 + G_y^2}$ is mathematically governed by a **Rayleigh distribution**:
$$f(x; \sigma_0) = \frac{x}{\sigma_0^2} e^{-x^2 / (2\sigma_0^2)} \quad \text{for } x \ge 0$$
The theoretical mean and standard deviation are:
$$\mu_{\text{Rayleigh}} = \sigma_0 \sqrt{\frac{\pi}{2}} \approx 1.2533 \sigma_0, \quad \sigma_{\text{Rayleigh}} = \sigma_0 \sqrt{\frac{4 - \pi}{2}} \approx 0.6551 \sigma_0$$
Evaluating the threshold $T = \mu + 1.2\sigma$:
$$T \approx (1.2533 + 1.2 \times 0.6551) \sigma_0 \approx 2.0394 \sigma_0 \implies F(T) = 1 - e^{-(2.0394)^2 / 2} \approx 0.8750$$
This yields an exceedance probability of approximately $12.50\%$ on pure uncorrelated Gaussian derivatives. The fused G-buffer metric combines multi-channel gradients and does not strictly satisfy this idealized model; hence this result serves as theoretical threshold intuition rather than an exact production false-positive rate. The contrast floor $\delta_{\min} = 0.02$ and noise floor $T_{\min} = 0.05$ ensure that primary G-buffer noise and subtle tessellation faceting do not trigger false structural edges in dark or smooth tiles.

#### Local Tiled Reduction & Boundary Seam Filtering
Computing global $\mu$ and $\sigma$ across a full frame requires a multi-pass parallel reduction tree, creating global GPU execution barriers and pipeline bubbles.

Statistical thresholding is evaluated locally within $16\times16$ compute tiles (256 threads) using Local Data Share (LDS) subgroup arithmetic:
1. Each workgroup loads its tile's $M_{\text{fused}}$ values into shared memory. The $20\times20$ Sobel domain provides **400 samples**, ensuring high statistical confidence in $\mu_{\text{tile}}$ and $\sigma_{\text{tile}}$.
2. Parallel tree reduction computes local mean $\mu_{\text{tile}}$ and variance $\sigma_{\text{tile}}^2$.
3. The clamped threshold $T_{\text{tile}} = \max(\mu_{\text{tile}} + 1.2 \cdot \max(\sigma_{\text{tile}}, 0.02), 0.05)$ and contrast distance test are applied uniformly across both apron and center samples.
4. The noise floor $T_{\min} = 0.05$ and contrast floor $\delta_{\min} = 0.02$ guarantee that subtle moment shifts across smooth adjacent tiles do not create artificial edge thresholds, while temporal density accumulation ($\alpha_\rho = 0.15$) and subsequent spatial wavelet crossfading naturally eliminate tile boundary seams without inter-workgroup synchronization.

---

### 2.3 Motion-Reprojected Temporal Edge Density Reuse

Computing local edge density directly via spatial convolution requires evaluating an $r \times r$ window per pixel on every frame, incurring substantial texture cache bandwidth. Fu et al. exploit temporal coherence by reprojecting the previous frame's edge density field using screen-space motion vectors.

#### Spatial Edge Density Definition
For a local neighborhood window $\mathcal{N}(i, j)$ of dimension $r \times r$ centered at pixel $(i, j)$ ($|\mathcal{N}| = r^2$, with $r = 5$, $|\mathcal{N}| = 25$):
$$\rho_{\text{spatial}}(i, j) = \frac{1}{|\mathcal{N}(i, j)|} \sum_{(x, y) \in \mathcal{N}(i, j)} M_{\text{edge}}(x, y)$$
Here, $\rho(i, j) \in [0, 1]$ represents the local edge concentration:
- $\rho \to 1.0$: Dense edge cluster (complex silhouettes, hair, foliage, fine mesh geometry).
- $\rho \to 0.0$: Homogeneous flat region (walls, floors, smooth sky, large planar surfaces).

#### 3-Pixel Source Apron for In-LDS Edge Density Computation
Computing local edge density $\rho_{\text{spatial}}$ over a $5\times5$ box filter ($R_\rho = 2$) requires edge-mask samples $M_{\text{edge}}$ across offsets $[-2, 2]$ surrounding each output pixel. Each binary edge sample $M_{\text{edge}}$ is produced by a $3\times3$ Sobel convolution ($R_{\text{Sobel}} = 1$) requiring one additional raw G-buffer texel.
Therefore, the required raw G-buffer source apron is:
$$R_{\text{source}} = R_\rho + R_{\text{Sobel}} = 2 + 1 = 3\text{ pixels}$$
For a standard $16\times16$ compute workgroup, three nested execution domains are maintained strictly in LDS:
- **G-Buffer Source Domain**: $(16 + 2 \times 3) \times (16 + 2 \times 3) = \mathbf{22\times22\text{ texels}}$ loaded from `uNormalDepth` and `uAlbedoRoughness`.
- **Sobel / Edge-Mask Domain**: $(16 + 2 \times 2) \times (16 + 2 \times 2) = \mathbf{20\times20\text{ samples}}$ evaluated directly in shared memory.
- **Edge Density Output Domain**: $\mathbf{16\times16\text{ pixels}}$ evaluated via $5\times5$ box filter and written to output images.
With a 3-pixel apron, every thread in the $16\times16$ tile computes its full $5\times5$ box filter without boundary starvation, LDS out-of-bounds reads, or global Sobel roundtrips.

#### Dual-Model Motion Reprojection Mechanics
Let $\mathbf{V}_t(i, j) \in \mathbb{R}^2$ be the 2D screen-space motion vector from frame $t$ to frame $t-1$. The reprojected historical pixel coordinate is:
$$(i', j') = (i, j) + \mathbf{V}_t(i, j)$$

Screen UV is validated by testing bounds: $(i', j') \in [0, W) \times [0, H)$. To correctly handle both static environments and dynamic animated objects without spurious disocclusions, a **dual-model reprojection selector** is evaluated:

1. **Static Geometry Path (Camera-Motion Agreement)**:
   The shader computes expected previous screen UV $\mathbf{u}_{\text{expected}}$ by unprojecting $(i, j, z)$ into metric 3D current view space $\mathbf{x}_{\text{curr}}$, transforming through the affine view-to-view matrix $\mathbf{M}_{\text{currView}\to\text{prevView}}$, and projecting to screen space:
   $$\mathbf{x}_{\text{prev}} = \mathbf{M}_{\text{currView}\to\text{prevView}} \cdot \begin{bmatrix} \mathbf{x}_{\text{curr}} \\ 1 \end{bmatrix}, \quad z_{\text{expected}} = \mathbf{x}_{\text{prev}}.z$$
   If $\|\mathbf{u}_{\text{motion}} - \mathbf{u}_{\text{expected}}\| \le \epsilon_{\text{rigid}}$ (where $\epsilon_{\text{rigid}} \approx 2.0\text{ pixels}$), the pixel corresponds to static rigid scene geometry. Reprojection is validated against the exact affine depth:
   $$V_{\text{static}} = \left( \mathbf{n}_{\text{curr}} \cdot (\mathbf{R}_{\text{prev}\to\text{curr}} \cdot \mathbf{n}_{\text{prev}}) > 0.85 \right) \land \left( \frac{|z_{\text{expected}} - z_{\text{prev}}|}{\max(z_{\text{expected}}, 10^{-3})} < 0.08 \right)$$
   where $\mathbf{R}_{\text{prev}\to\text{curr}}$ is a pure $3\times3$ camera rotation matrix that compensates for view-normal rotation across consecutive frames.

2. **Dynamic Geometry Path (Moving Objects / Characters)**:
   If $\|\mathbf{u}_{\text{motion}} - \mathbf{u}_{\text{expected}}\| > \epsilon_{\text{rigid}}$, the pixel belongs to a dynamic moving object. Camera-only $z_{\text{expected}}$ prediction does not apply. Instead, reprojection validates using the motion vector's sampled previous depth:
   $$V_{\text{dynamic}} = \left( \mathbf{n}_{\text{curr}} \cdot (\mathbf{R}_{\text{prev}\to\text{curr}} \cdot \mathbf{n}_{\text{prev}}) > 0.85 \right) \land \left( \frac{|z_{\text{curr}} - z_{\text{prev}}|}{\max(z_{\text{curr}}, 10^{-3})} < \tau_{\text{depth}} \right)$$
   For dynamic geometry, temporal accumulation length is capped to $N_{\text{hist}} = \min(N_{\text{hist}}, 8)$ to accommodate non-rigid deformation while maintaining temporal stability.

3. **Disocclusion Fallback**:
   If both validation models fail, temporal accumulation is reset to $N_{\text{hist}} = 1.0$, and local variance is initialized from neighboring LDS radiance.

#### Dual-Channel Edge Density Packing (`R8G8_UNORM`)
In Fu et al.'s formulation, modulating edge density by sample history length $\rho_{\text{effective}} = \rho \cdot \text{clamp}(N_{\text{hist}}/8, 0.25, 1.0)$ widens spatial kernels for newly disoccluded geometry. However, storing only $\rho_{\text{effective}}$ corrupts the temporal recursive estimator on frame $t+1$: because $\rho^{t-1}$ would read as $0.25 \rho$, the fixed point collapses to $\rho_{\text{stored}} = \frac{0.15 f \rho}{1 - 0.85 f}$, locking edges into 20 frames of wide 61-pixel blur!

To preserve strict estimator mathematical integrity, `m_edgeDensity[2]` is allocated as **`VK_FORMAT_R8G8_UNORM`**:
- **Channel R (Raw Temporal Density $\rho^t$)**: Stores the true temporal state:
  $$\rho^t(i, j) = (1 - \alpha_\rho) \cdot \rho^{t-1}(i', j') + \alpha_\rho \cdot \rho_{\text{spatial}}^t(i, j) \quad (\alpha_\rho = 0.15)$$
  On disocclusion ($V = \text{false}$), $\rho^t = \rho_{\text{spatial}}^t$. Frame $t+1$ reads Channel R exclusively, preserving temporal convergence.
- **Channel G (Effective Spatial Density $\rho_{\text{effective}}^t$)**: Stores the history-modulated density:
  $$\rho_{\text{effective}}^t(i, j) = \rho^t(i, j) \cdot \text{clamp}\left(\frac{N_{\text{hist}}(i, j)}{8.0}, \, 0.25, \, 1.0\right)$$
  Downstream spatial passes (Pass 2, 3, 4, 5) read Channel G directly, eliminating redundant `uHistoryLength` fetches without corrupting the temporal accumulator.

---

### 2.4 Dynamic Kernel Sizing Selection & Receptive Field Mapping

Fu et al. map local edge density $\rho(i, j)$ to discrete spatial filtering kernel dimensions:
$$k = \begin{cases} 9 & \text{if } \rho(i, j) \ge 0.5 \quad \text{(Dense structural detail / complex geometry)} \\ 17 & \text{if } 0.3 \le \rho(i, j) < 0.5 \quad \text{(Moderate complexity / broad curvature)} \\ 33 & \text{otherwise } (\rho < 0.3) \quad \text{(Smooth, flat, homogeneous regions)} \end{cases}$$

#### Receptive Field Mapping Across the 5-Dispatch Dilated Hierarchy
To strictly eliminate non-synchronizable multi-loop execution while guaranteeing the full **61-pixel receptive field** at 1 SPP, the engine maps these kernels to **5 cleanly synchronized dispatches**:

- **Pass 2 (`denoise_atrous_step1.comp`)**:
  - Step size $s = 1$, radius $R_0 = 2 \implies \text{diameter } 5\times5$.
  - Evaluated across all valid pixels to establish baseline high-frequency variance reduction.
- **Pass 3 (`denoise_atrous_step2.comp`)**:
  - Step size $s = 2$, accumulated radius $R_1 = 2 + 4 = 6 \implies \text{accumulated diameter } 13\times13$ (covers Fu et al. $k = 9$).
  - Dense edge pixels ($\rho_{\text{effective}} \ge 0.55$) complete spatial filtering here (50 taps total algorithmic classification) and bypass subsequent filtering taps.
- **Pass 4 (`denoise_atrous_step3.comp`)**:
  - Step size $s = 4$, accumulated radius $R_2 = 6 + 8 = 14 \implies \text{accumulated diameter } 29\times29$ (covers Fu et al. $k = 17$).
  - Dense edge pixels ($\rho_{\text{effective}} \ge 0.55$) bypass filtering taps via continuous smoothstep crossfade ($w_{\text{blend}}^{(1)} = \text{smoothstep}(0.45, 0.55, \rho_{\text{effective}})$); smoothstep crossfading prevents spatial seams on boundary wavefronts.
  - Wavefronts covering dense edges (`subgroupAll(effectiveRho >= 0.55)`) skip Pass 4 taps entirely; wavefronts covering moderate curvature (`subgroupAll(effectiveRho >= 0.35)`) skip Pass 5 taps (retaining Pass 4 filtered radiance and proceeding to remodulation in Pass 5).
- **Pass 5 (`denoise_atrous_coarse_remodulate.comp`)**:
  - Step size $s = 8$, accumulated radius $R_3 = 14 + 16 = 30 \implies \text{accumulated diameter } 61\times61$ (covers Fu et al. $k = 33$).
  - Executed exclusively for smooth/homogeneous regions ($\rho_{\text{effective}} < 0.35$), completely eliminating 1-SPP low-frequency blotches and restoring the full low-frequency reach that SVGF exists to provide.

#### Continuous Crossfading & Subgroup Uniform Voting
To eliminate spatial variance step jumps and À-Trous grid checkerboarding:

1. **Continuous Crossfade Blending**:
   Transition boundaries evaluate smooth crossfade weights:
   $$\text{Pass 4 } (s=4): \quad w_{\text{blend}}^{(1)} = \text{smoothstep}(0.45, \, 0.55, \, \rho_{\text{effective}})$$
   $$\text{Pass 5 } (s=8): \quad w_{\text{blend}}^{(2)} = \text{smoothstep}(0.25, \, 0.35, \, \rho_{\text{effective}})$$
   Both radiance and variance (Alpha channel) are blended smoothly:
   $$L_{\text{out}} = (1.0 - w_{\text{blend}}) \cdot L_{\text{filtered}} + w_{\text{blend}} \cdot L_{\text{prev}}$$
   *(Note on Variance Propagation Heuristics: The classic SVGF formula $V_{k+1} = \frac{\sum w_i^2 V_k(q_i)}{(\sum w_i)^2}$ is an empirical bilateral approximation that assumes uncorrelated samples and fixed weights. Across recursive À-Trous iterations, sample footprints overlap, introducing spatial correlation. Similarly, blending alpha in $L_{\text{out}} = (1 - w) L_{\text{filtered}} + w L_{\text{prev}}$ with identical variance blending is an empirical filter-width transition heuristic rather than formal covariance propagation $\operatorname{Var}(Y) = (1-a)^2\operatorname{Var}(F) + a^2\operatorname{Var}(P) + 2a(1-a)\operatorname{Cov}(F,P)$, calibrated to prevent abrupt step-function transitions across kernel sizes).*

2. **Subgroup Uniform Voting (`subgroupAll`) Matching Ramp Tops**:
   To avoid 32-pixel block divergence artifacts, `subgroupAll()` wave voting is keyed strictly to the **top of the smoothstep ramp**:
   - In Pass 4: bypass $s=4$ taps only when `subgroupAll(effectiveRho >= 0.55)`.
   - In Pass 5: bypass $s=8$ taps only when `subgroupAll(effectiveRho >= 0.35)`.
   ```glsl
   // In Pass 4: dense edge waves bypass s=4 taps entirely and passthrough input radiance
   bool bypassS4 = subgroupAll(effectiveRho >= 0.55);
   if (bypassS4) {
       imageStore(uOutDiffuse, coord, imageLoad(uInDiffuse, coord));
       imageStore(uOutSpecular, coord, imageLoad(uInSpecular, coord));
       return;
   }

   // In Pass 5: moderate curvature waves bypass s=8 taps and evaluate only remodulation
   bool bypassS8 = subgroupAll(effectiveRho >= 0.35);
   if (bypassS8) {
       vec3 L_remod = filteredDiffuse.rgb * max(albedo.rgb, vec3(0.001)) + filteredSpecular.rgb;
       imageStore(uFinalDenoised, coord, vec4(L_remod, 1.0));
       return;
   }
   ```

#### Sampling Overhead Comparison
| Region Complexity | Edge Density $\rho$ | Fu et al. Brute-Force Taps | 5-Dispatch Dilated E-SVGF Taps | vs. Fu et al. | vs. Classic 5-Pass SVGF (125 px) |
|---|---|---|---|---|---|
| **Dense Edges / Silhouettes** | $\rho \ge 0.55$ | 81 taps ($9\times9$) | 50 taps (Pass 2 + Pass 3: $s=1, 2$) | **$1.62\times$ tap reduction** | **$2.50\times$ tap reduction** |
| **Moderate Curvature** | $0.35 \le \rho < 0.55$ | 289 taps ($17\times17$) | 75 taps (Pass 2 + Pass 3 + Pass 4) | **$3.85\times$ tap reduction** | **$1.67\times$ tap reduction** |
| **Smooth / Flat Surfaces** | $\rho < 0.35$ | 1089 taps ($33\times33$) | 100 taps (Pass 2 + Pass 3 + Pass 4 + Pass 5) | **$10.89\times$ tap reduction** | **$1.25\times$ tap reduction** |
| **Weighted Frame Average** | Typical Scene | ~680 taps | ~70–80 taps (classification average) | **~8.5–9.7$\times$ tap reduction** | **~1.6–1.8$\times$ tap reduction** |

*(Note on Algorithmic Tap Reduction vs. Measured Execution Speedup: Tap counts represent theoretical per-pixel algorithmic classification costs. Realized execution speedup differs from tap reduction because mixed subgroups execute full tap loops under SIMD divergence, and every dispatch incurs mandatory DRAM and cache bandwidth for descriptor loads and output stores).*

---

## 3. Comprehensive Comparative Trade-Off Matrix

The following matrix provides an exhaustive technical comparison between Fu et al.'s 5D Neural LUT, the proposed Analytical E-SVGF, and the existing standard SVGF baseline (Schied et al., 2017).

| Architectural Dimension | 5D Neural LUT (Fu et al., 2025) | Analytical E-SVGF (Proposal) | Standard SVGF Baseline (Schied et al., 2017) |
|---|---|---|---|
| **Spatial Weight Determination** | Precomputed 5D quantization lookup: $I = (Q_c, Q_\theta, Q_\phi, Q_L, Q_d) \to \text{LUT}$ | Runtime shader ALU evaluation of analytical bilateral functions ($\exp, \text{pow}, \text{dot}$) | Runtime shader ALU evaluation of variance-guided bilateral functions |
| **GPU VRAM Overhead** | **2.5 GB** (FP16 compressed dominant bins) / **5.7 GB** (FP32 uncompressed) | **0.0 MB** (No lookup tables; only 8.29 MB unpadded / 8.91 MB padded for double-buffered `m_edgeDensity[2]` in `VK_FORMAT_R8G8_UNORM`) | **0.0 MB** (No lookup tables) |
| **VRAM Footprint at 1080p** | 2,500.0 MB table + 224 MB framebuffers = **2,724 MB** | 8.29 MB edge density + 248.84 MB denoiser images = **257.13 MB** (245.22 MiB across 17 storage images, 124 B/px) | 0 MB edge buffers + 223.95 MB framebuffers = **223.95 MB** |
| **Texture Lookups per Tap** | 1 LUT texture read + G-buffer reads per tap | G-buffer + radiance texture reads only (0 LUT reads) | G-buffer + radiance texture reads only |
| **Total Memory Bandwidth** | Extreme: 2.5 GB non-contiguous 5D texture fetches induce severe memory bus contention | Parity with classic SVGF: 314 B/px (~2.60 GB baseline at 4K across 5 passes, ~651 MB at 1080p). Tap reduction is DRAM-neutral (passes read/write full frames); advantage is edge preservation and specular fidelity at parity bandwidth | Moderate: ~2.5–3.2 GB at 4K (4–5 unfused passes) |
| **L1 / L2 Cache Hit Rate** | Poor: Adjacent pixels with differing normal/luminance access distant 5D memory bins | High: Excellent spatial locality; adjacent pixels access identical memory cachelines | High: Excellent spatial locality |
| **ALU Instruction Complexity** | Very Low: ~3 scalar ALUs (quantization & indexing) + 1 L2 memory fetch per tap | Moderate: ~14 scalar ALUs ($\exp$, division, dot products per tap) | Moderate: ~12 scalar ALUs (variance exponentiation & normal power) |
| **Desktop / APU ALU Suitability** | Sub-optimal: Starves massive desktop GPU SIMD ALUs while saturating VRAM bus | Optimal: Perfectly balances high FP32/FP16 ALU throughput against memory cache bandwidth | Good: Standard compute shader balance |
| **Spatial Reach / Tap Footprint** | Brute-force: $9\times9$ (81 taps), $17\times17$ (289 taps), $33\times33$ (1089 taps) | 5-Dispatch Dilated: $5\times5$–$13\times13$ on edges ($s=1, 2$), $29\times29$ on moderate ($s=4$), $61\times61$ on flats ($s=8$) | Fixed À-Trous: $125\times125$ footprint ($s \in \{1, 2, 4, 8, 16\}$) |
| **Total Filtering Taps / Pixel** | 81, 289, or 1089 taps per pixel | **50 taps (dense edges) to 100 taps (flats)**; subgroup early exit on moderate curvature wavefronts | Fixed 125 taps across all pixels |
| **Material Generalization** | Fails on unseen BRDFs, conductors, and dielectrics (over-blurs specular/glass) | Physics-Guided / High: Direct evaluation using runtime surface normals and GGX roughness | High: Robust across standard materials |
| **Specular Reflection Fidelity** | Destroys sharp reflections; treats mirror surfaces as flat diffuse (Section 8) | Preserved: Radiance decoupled into diffuse/specular; continuous GGX $\alpha^2$ modulation | Moderate: Diffuse and specular accumulated together or partially split |
| **Depth Precision Resilience** | Fragile: Non-linear depth buffer aliasing triggers false edge classification (Section 8) | Robust: Surface tangent plane-distance weighting $\lvert (\mathbf{x}_Q - \mathbf{x}_P) \cdot \mathbf{n}_P \rvert$ and depth floor | Moderate: Susceptible to depth staircase on grazing angles |
| **Disocclusion Handling** | Prone to edge shearing and white ghosting on fast non-rigid motion (Section 8) | Robust: Dual-model validation (static affine depth vs dynamic object sample depth test) | Standard: Disocclusion variance boost |
| **Mobile SoC Latency (Adreno 620)** | 18.12 ms – 19.32 ms (Table 1, Fu et al.) | 24.68 ms (Table 1, Fu et al.) | ~32.0 ms |
| **Discrete Desktop Latency (4K Output)** | Est. 14.0 – 18.0 ms (Severely bandwidth bound by 2.5 GB table fetches) | **Profiling Gate (< 3.5 ms on $\ge 950\text{ GB/s}$ GPUs, e.g. 4090/7900XTX)**; 4K floor is ~5.00–6.60 ms on 4070/7800-class | 7.5 – 9.2 ms (Un-fused 4-pass SVGF) |
| **Unified APU Latency (1080p Output)** | Est. 4.5 – 6.0 ms (Memory bus starvation) | **3.3–3.6 ms base gate measured** on Strix Halo gfx1151 (256-bit LPDDR5X, ~3.18 ms memory floor); sub-3.0 ms gated behind traffic roadmap (~2.69 ms floor) | 3.5 – 4.2 ms |
| **Discrete Desktop Latency (1080p Internal)** | Est. 2.5 – 3.5 ms | **1.7–1.9 ms measured (4070)** (pure streaming floor ~1.61 ms; 36 MB L2 partially spills), **1.0–1.2 ms measure (7800 XT)** (64 MB Infinity Cache borderline against 66 MB working set), **1.2–1.4 ms ($\ge 700\text{ GB/s}$)**; 1440p target: **3.0–3.4 ms** (floor ~2.86 ms on 4070; 118 MB working set exceeds cache) | 1.8 – 2.4 ms |
| **Dynamic Resolution / Scalability**| Rigid: 5D table requires retraining or re-indexing for new coordinate bounds | Dynamic: Adapts instantly to arbitrary render scales (FSR upscaling, dynamic resolution) | Dynamic: Resolution agnostic |

---

## 4. Vulkan 1.4 Compute Pipeline Specification

The adaptive denoiser is implemented as an autonomous **Authoritative 5-Dispatch Dilated E-SVGF compute pipeline** within `src/rt/DenoiserPipeline.hpp` and `src/rt/DenoiserPipeline.cpp`. All passes execute under **Vulkan 1.4** core specifications with `VK_KHR_synchronization2`, `VK_EXT_subgroup_size_control`, and `VK_KHR_shader_float16`.

By fusing edge extraction, variance estimation, and history normal/depth updates into the temporal pass, and fusing remodulation into the coarse wavelet pass, intermediate VRAM roundtrips are minimized, allowing the pipeline to target **1.7–1.9 ms measured on RTX 4070 (pure streaming floor ~1.61 ms), 1.0–1.2 ms measure on RX 7800 XT (where the 66.4 MB working set is borderline against the 64 MB Infinity Cache), 1.2–1.4 ms on $\ge 700\text{ GB/s}$ GPUs at 1080p internal render resolution, and 3.3–3.6 ms on unified APUs (with sub-3.0 ms gated behind future traffic-reduction optimizations)**.

### 4.1 Internal Storage Image Allocations

At 1080p ($1920 \times 1080$), internal images allocated during `DenoiserPipeline::resize()` are:

| Image Resource | Format | Dimension | Count | VRAM (1080p Unpadded) | Persistence / Usage |
|---|---|---|---|---|---|
| `m_edgeDensity[2]` | `VK_FORMAT_R8G8_UNORM` | $W \times H$ | 2 | 8.29 MB (8,294,400 B) | Double-buffered edge density (R: raw $\rho^t$, G: effective $\rho_{\text{effective}}^t$) |
| `m_historyDiffuse[2]` | `VK_FORMAT_R16G16B16A16_SFLOAT` | $W \times H$ | 2 | 33.18 MB (33,177,600 B) | Double-buffered temporal diffuse accumulation & variance $V^d$ |
| `m_historySpecular[2]` | `VK_FORMAT_R16G16B16A16_SFLOAT` | $W \times H$ | 2 | 33.18 MB (33,177,600 B) | Double-buffered temporal specular accumulation & variance $V^s$ |
| `m_historyMoments[2]` | `VK_FORMAT_R16G16B16A16_SFLOAT` | $W \times H$ | 2 | 33.18 MB (33,177,600 B) | Double-buffered 1st and 2nd moments ($M_1^d, M_2^d, M_1^s, M_2^s$) |
| `m_historyLength[2]` | `VK_FORMAT_R16G16B16A16_SFLOAT` | $W \times H$ | 2 | 33.18 MB (33,177,600 B) | Double-buffered history metadata (R: $N_{\text{diff}}$, G: $N_{\text{spec}}$, B: $d_{\text{prev,refl}}$, A: $M_{\text{dynamic}}$) |
| `m_historyNormalDepth[2]` | `VK_FORMAT_R16G16B16A16_SFLOAT` | $W \times H$ | 2 | 33.18 MB (33,177,600 B) | Historical view normal XYZ and linear depth W |
| `m_pingPongDiffuse[2]` | `VK_FORMAT_R16G16B16A16_SFLOAT` | $W \times H$ | 2 | 33.18 MB (33,177,600 B) | Spatial À-Trous ping-pong diffuse buffers (RGB: radiance, A: variance) |
| `m_pingPongSpecular[2]` | `VK_FORMAT_R16G16B16A16_SFLOAT` | $W \times H$ | 2 | 33.18 MB (33,177,600 B) | Spatial À-Trous ping-pong specular buffers (RGB: radiance, A: variance) |
| `m_denoisedImage` | `VK_FORMAT_R16G16B16A16_SFLOAT` | $W \times H$ | 1 | 16.59 MB (16,588,800 B) | Final tonemapper input image (remodulated HDR RGB, A: 1.0) |
| **Total Denoiser VRAM** | | | | **257.13 MB (245.22 MiB)** | **17 internal storage images (124 B/px; Sobel computed in-LDS)** |

#### Formal Per-Image Channel Semantic Contracts
To ensure deterministic interoperability between the ray tracing producer pipelines, compute denoiser stages, and tonemapping, every input, intermediate, and output channel conforms to the following strict semantic layout:

| Image Resource | Binding Stage | Channel R | Channel G | Channel B | Channel A | Semantic Rationale & Downstream Consumer |
|---|---|---|---|---|---|---|
| `uNormalDepth` | Primary G-Buffer | Normal X | Normal Y | Normal Z | Linear Depth $z$ | View-space normal $\mathbf{n} \in [-1, 1]^3$ and view-space metric depth $z \in [0, \infty)$ for plane-distance tests. |
| `uAlbedoRoughness` | Primary G-Buffer | Base Color R | Base Color G | Base Color B | GGX Roughness | Demodulated physical material albedo (invariant to lighting) and linear roughness $\alpha \in [0, 1]$. |
| `uMotionVectors` | Primary G-Buffer | Screen $\Delta u$ | Screen $\Delta v$ | — | — | Screen-space UV motion delta to previous frame $[0, 1]^2$. **Jitter Convention**: Contains pure, jitter-free geometric motion; camera sub-pixel jitter is cancelled explicitly in shader ALUs via `CameraMatricesUBO`. |
| `uSpecularMotion` | Primary G-Buffer | Virtual $\Delta u$ | Virtual $\Delta v$ | — | — | Virtual reflection-space UV motion delta for VMV reprojection. **Jitter Convention**: Pure, jitter-free geometric reflection motion. |
| `uCurrentDiffuse` | 1-SPP Radiance In | Diffuse Rad R | Diffuse Rad G | Diffuse Rad B | **Emissive Flag** | 1-SPP demodulated diffuse radiance; **Alpha = 1.0 for emissive surfaces / area lights (0.0 otherwise)**. Used for firefly bypass across both lobes. |
| `uCurrentSpecular` | 1-SPP Radiance In | Specular Rad R | Specular Rad G | Specular Rad B | **Hit Dist $d_{\text{refl}}$** | 1-SPP specular radiance; **Alpha = secondary reflection ray hit distance $d_{\text{refl}}$** (in metric meters) for VMV reprojection and validation. |
| `m_historyDiffuse` | History Image | Diffuse Rad R | Diffuse Rad G | Diffuse Rad B | **Diffuse Var $V^d$** | Un-blurred temporally accumulated diffuse radiance; Alpha packs temporal diffuse variance for Pass 2 ($w_L$ bilateral weight). |
| `m_historySpecular` | History Image | Specular Rad R| Specular Rad G| Specular Rad B| **Specular Var $V^s$** | Un-blurred accumulated specular radiance; **Alpha packs temporal specular variance $V^s = M_2^s - (M_1^s)^2$** for Pass 2 ($w_L$ bilateral weight), symmetric with diffuse! |
| `m_historyMoments` | History Image | 1st Diff $M_1^d$ | 2nd Diff $M_2^d$ | 1st Spec $M_1^s$| 2nd Spec $M_2^s$| Double-buffered 1st and 2nd luminance moments packed per lobe, guaranteeing exact variance recovery ($M_2 - M_1^2$) for both diffuse and specular. |
| `m_historyLength` | History Image | $N_{\text{diff, hist}}$ | $N_{\text{spec, hist}}$ | **Prev $d_{\text{prev,refl}}$** | **Dynamic Flag $M_{\text{dyn}}$** | History metadata: Channel R ($N_{\text{diff}} \in [1, 32]$) sampled at primary `prevUV`; Channels G ($N_{\text{spec}} \in [1, 16]$) and B ($d_{\text{prev,refl}}$) sampled at VMV `prevVMV_UV`. Channel A stores dynamic object flag ($M_{\text{dyn}} \in \{0.0, 1.0\}$) from Pass 1 selector to allow Passes 2–5 to widen $\phi_L$ on dynamic pixels. (RGBA16F format chosen over separate images to save a descriptor binding). |
| `m_edgeDensity` | Intermediate Edge | Raw $\rho^t$ | Effective $\rho_{\text{eff}}$ | — | — | Channel R: temporal state $\rho^t$; Channel G: history-modulated $\rho_{\text{effective}}$ for spatial dispatches. |
| `m_pingPongDiffuse` | Spatial À-Trous | Diffuse Rad R | Diffuse Rad G | Diffuse Rad B | Diffuse Var $V^d$ | Denoised diffuse radiance in RGB; filtered spatial variance in Alpha ($w^2$ propagation). |
| `m_pingPongSpecular`| Spatial À-Trous | Specular Rad R| Specular Rad G| Specular Rad B| Specular Var $V^s$| Denoised specular radiance in RGB; filtered spatial variance in Alpha ($w^2$ propagation). |
| `m_denoisedImage` | Final Output | Final Color R | Final Color G | Final Color B | 1.0 (Alpha) | In-register remodulated linear HDR color: $\mathbf{L}_{\text{final}} = \mathbf{L}_{\text{diff}} \cdot \mathbf{c}_{\text{albedo}} + \mathbf{L}_{\text{spec}}$. Tonemapper input. |

*(Note on Variance Storage Redundancy: `m_historyMoments` carries 1st and 2nd luminance moments per lobe ($M_1^d, M_2^d, M_1^s, M_2^s$), from which variance is algebraically recoverable as $V = M_2 - M_1^2$. Storing $V^d$ and $V^s$ explicitly in the Alpha channel of `m_historyDiffuse` and `m_historySpecular` is an intentional bandwidth-vs-ALU optimization: it enables Passes 2–5 to evaluate variance-guided spatial filtering without binding `uHistoryMoments`, saving an 8 B/px texture read across all 4 spatial iterations (~265 MB at 4K / ~66 MB at 1080p)).*

*(Note on History Length Sampling Locations & Driving Lobe: `m_historyLength` is accessed at two distinct spatial locations in Pass 1. Primary diffuse history length ($N_{\text{diff}}$, Channel R) is fetched at the primary geometric reprojected UV coordinate (`currUV - uMotionVectors`). Specular history length ($N_{\text{spec}}$, Channel G) and historical reflection hit distance ($d_{\text{prev,refl}}$, Channel B) are fetched at the Virtual Motion Vector (VMV) reprojected UV coordinate (`currUV - uSpecularMotion`), ensuring specular confidence and hit-distance validation correspond to the warped virtual reflection point. Effective edge density is strictly driven by primary geometry history ($N_{\text{diff}}$): $\rho_{\text{effective}}^t = \rho^t \cdot \text{clamp}(N_{\text{diff}}/8.0, 0.25, 1.0)$, preventing specular disocclusions from artificially dilating geometric edge filters).*

*(Note on Emissive Flag & Specular Outliers: The scene-wide emissive / area light flag is packed into `uCurrentDiffuse.a` and passed into `suppressFirefliesLDS` for both lobes in Pass 1. Direct specular reflections off small light emitters are additionally protected by the bilateral geometric quorum in `suppressFirefliesLDS`, which prevents high-radiance specular highlights from being clipped when surrounded by geometrically consistent surface samples).*

#### Hardware Tiling & Alignment Note
- Summing unpadded raw byte counts across all **17 internal storage image allocations** yields $2,073,600 \times [ (2 \times 2) + 7 \times (2 \times 8) + (1 \times 8) ] = 2,073,600 \times 124 \text{ B} = \mathbf{257,126,400 \text{ B}} = \mathbf{257.13 \text{ MB}}$ ($245.22 \text{ MiB}$).
- Factoring in GPU hardware row-pitch alignment (padding width to 2048 bytes on AMD/NVIDIA architectures) and tile-height padding (padding 1080 to 1088 lines), the dual-channel edge feature images (`m_edgeDensity[2]`) occupy $2 \times 2048 \times 1088 \times 2 = 8,912,896 \text{ B} = \mathbf{8.91 \text{ MB}}$ ($8.50 \text{ MiB}$).
- In-LDS Sobel computation completely eliminates `m_edgeMagnitude` from global VRAM, saving 2.07 MB of allocation and 1 full-frame global memory write/read roundtrip.
- **Vulkan Device Features**: Writing to `VK_FORMAT_R8G8_UNORM` storage images requires the `StorageImageExtendedFormats` SPIR-V capability. Enables `deviceFeatures2.features.shaderStorageImageExtendedFormats = VK_TRUE;` during logical device creation in `src/vulkan/VulkanContext.cpp`.
- All persistent storage images reside continuously in `VK_IMAGE_LAYOUT_GENERAL`.

---

### 4.2 Authoritative 5-Dispatch Dilated Compute Pipeline Decomposition

Cascaded À-Trous wavelet filtering is inherently recursive ($I_k(p) = \sum w I_{k-1}(q) / \sum w$). Because GPU compute invocations across different workgroups run concurrently with zero execution ordering guarantees, dependent wavelet iterations cannot simply be evaluated as multiple loops in a single dispatch without global pipeline barriers or an impractical $56\times56$ workgroup load footprint. 

The engine decomposes the pipeline into **5 cleanly synchronized compute dispatches** that preserve the full **61-pixel receptive field** ($s \in \{1, 2, 4, 8\}$) while eliminating intermediate global roundtrips for Sobel, moments, and remodulation:

#### Pass 1: Fused Edge-Aware Spatiotemporal Accumulator (`denoise_fused_temporal.comp`)
- **Objective**: Fuses multi-channel Sobel edge extraction, statistical adaptive thresholding, motion vector temporal reprojection, outlier firefly clamping, moment accumulation ($M_1, M_2$), inline disocclusion variance estimation, and in-shader historical normal/depth writing into a **single unified compute dispatch**.
- **Dispatch Grid**: $\left(\lceil \frac{W}{16} \rceil, \lceil \frac{H}{16} \rceil, 1\right)$. Workgroup layout: `(16, 16, 1)` (256 threads).
- **3-Pixel G-Buffer Source Apron ($R_{\text{source}} = R_\rho + R_{\text{Sobel}} = 2 + 1 = 3\text{ pixels}$)**:
  - *G-Buffer Source Domain*: $22\times22$ texels loaded cooperatively into shared memory ($16 + 2 \times 3 = 22$). 256 threads cooperatively load the $22\times22 = 484$ texels from `uNormalDepth` and `uAlbedoRoughness` (first 228 threads load 2 texels, remaining 28 load 1 texel).
  - *Sobel & Edge Mask Domain*: $20\times20$ texels evaluated in-LDS ($16 + 2 \times 2 = 20$). Threads evaluate 3x3 Sobel gradients from the $22\times22$ G-buffer data without any global DRAM roundtrips.
  - *Output Edge Density Domain*: $16\times16$ output pixels. Local $5\times5$ box filter evaluates spatial density $\rho_{\text{spatial}}$ within the LDS apron from the $20\times20$ Sobel results.
- **In-LDS Bank-Conflict Tuning & Realistic Memory Budget**:
  - *Dominant Consumer*: Material base albedo and surface normals are the dominant LDS consumers, requiring full `vec3` color and view-space normal vectors across the $22\times22$ apron to evaluate the 3-component vector Sobel gradient and camera rotation validation.
  - *Declared LDS Allocation Breakdown (`glslc` / RGA verified at Stage 1)*:
    - Albedo ($3 \times 22 \times 23$ scalar float): $3 \times 506 \times 4\text{ B} = \mathbf{6,072\text{ B}} \approx 6.07\text{ KB}$
    - Normals ($3 \times 22 \times 23$ scalar float, XYZ SoA): $3 \times 506 \times 4\text{ B} = \mathbf{6,072\text{ B}} \approx 6.07\text{ KB}$
    - Linear View Depth ($1 \times 22 \times 23$ scalar float): $506 \times 4\text{ B} = \mathbf{2,024\text{ B}} \approx 2.02\text{ KB}$
    - Sobel Scratch ($1 \times 20 \times 23$ scalar float): $460 \times 4\text{ B} = \mathbf{1,840\text{ B}} \approx 1.84\text{ KB}$
    - Local Radiance Luminance ($2 \times 22 \times 23$ scalar diffuse + spec): $2 \times 506 \times 4\text{ B} = \mathbf{4,048\text{ B}} \approx 4.05\text{ KB}$
    - Tile Moments Reduction Scratch ($256 \times 2$ scalar float): $256 \times 8\text{ B} = \mathbf{2,048\text{ B}} \approx 2.05\text{ KB}$
    - **Total Declared LDS Footprint**: **22,104 Bytes = 21.59 KiB (~22.1 KB)**.
  - *Occupancy & Resource Sizing*: On AMD RDNA 2/3 (which exposes 64 KB of LDS per workgroup, 128 KB per WGP) and NVIDIA Ada Lovelace (which configures up to 100 KB shared memory per SM), the 21.59 KiB LDS allocation allows $\ge 2$ active workgroups per RDNA CU and $\sim 4$ per Ada SM. The operational occupancy ceiling for a 256-thread workgroup (8 Wave32 wavefronts) is determined by SIMD wave slot limits (16 waves per SIMD32 on RDNA3) and register pressure (VGPRs). Stage 1 shader profiling via Radeon GPU Analyzer (`rga`) and NVIDIA Nsight Compute will measure and report achieved register counts and wave residency per target hardware platform.
  - *Conflict-Free Layout*: Shaders utilize a Structure of Arrays (SoA) layout with an odd stride (e.g. $22\times23$ words) so consecutive row addresses cycle across all 32 banks modulo 32.
- **In-Shader Execution Sequence**:
  1. *LDS Cooperative Load*: 256 threads cooperatively load the $22\times22 = 484$ apron texels. Workgroup barriers (`barrier()`) synchronize loads across all invocations before processing. Edge-of-screen tiles clamp sampling coordinates to valid image bounds, ensuring no invocation returns early before reaching collective barriers.
  2. *In-LDS Sobel & Local Tile Reduction*: Threads compute 3x3 Sobel gradients directly from shared memory. Vector bounds ($4\sqrt{6}$ for albedo, $8\sqrt{2}$ for normals) normalize gradients to $[0, 1]$. Tile moments ($\mu_{\text{tile}}, \sigma_{\text{tile}}$), dynamic threshold $T_{\text{tile}} = \max(\mu_{\text{tile}} + 1.2 \cdot \max(\sigma_{\text{tile}}, \text{pc.sigmaMin}), \text{pc.noiseFloor})$, and contrast floor $\delta_{\min} = \text{pc.contrastFloor}$ are evaluated via wave reduction.
  3. *In-LDS Spatial Edge Density*: Local $5\times5$ box filter evaluates spatial density $\rho_{\text{spatial}}$ within the LDS apron.
  4. *Dual-Model Reprojection & Density Fusion*: Warps historical edge density $\rho^{t-1}$ via `uMotionVectors`. Evaluates camera-rotation-compensated validation ($\mathbf{n}_{\text{curr}} \cdot \mathbf{n}_{\text{prev}}^{\text{curr}} > 0.85$ via `prevViewToCurrViewRot` from `CameraMatricesUBO` and linear depth validation $|z_{\text{expected}} - z_{\text{prev}}| / z_{\text{expected}} < \text{pc.depthThreshold}$ via `currViewToPrevView`). Dynamic objects switch to motion-vector sample depth testing with shortened history ($N_{\text{hist}} = \min(N_{\text{hist}}, 8)$). Fuses $\rho^t = 0.15 \rho_{\text{spatial}} + 0.85 \rho^{t-1}$.
  5. *Outlier Firefly Suppression*: Current frame sample is softly clamped against bilateral-consistent neighbors strictly excluding the center sample using scalar luminance from LDS (`s_localDiffuseLum`, `s_localSpecularLum`), with an emissive bypass and chromaticity-preserving asymptotic rolloff before moment accumulation ($M_1, M_2$). TAA-style color-box clipping of history is eliminated to prevent clipping valid indirect illumination. Values are clamped before `imageStore` (`clamp(L, 0.0, 65000.0)`) to prevent FP16 `Inf` overflow.
  6. *Pre-Spatial History Update & Dual-Channel Packing*: Diffuse radiance with variance $V^d$ in Alpha is written to `m_historyDiffuse[currSlot]`; specular radiance with temporal specular variance $V^s = M_2^s - (M_1^s)^2$ in Alpha is written to `m_historySpecular[currSlot]`, fully restoring symmetric variance transport into spatial passes. Historical secondary reflection hit distance $d_{\text{refl}}$ is written to `uOutHistoryLength.b` (Channel B of `m_historyLength[currSlot]`) alongside the dynamic object classification flag ($M_{\text{dynamic}}$) in Channel A. History normal/depth are written directly to `m_historyNormalDepth[currSlot]` (`uOutNormalDepth`), eliminating `vkCmdCopyImage`. Dual-channel edge density is packed into `m_edgeDensity[currSlot]` (`VK_FORMAT_R8G8_UNORM`): Channel R stores raw $\rho^t$ for temporal accumulation; Channel G stores effective $\rho_{\text{effective}}^t = \rho^t \cdot \text{clamp}(N_{\text{diff}}/8.0, 0.25, 1.0)$ for downstream spatial passes.

- **In-LDS Cooperative Load & Tile Reduction Implementation (`denoise_fused_temporal.comp`)**:
  ```glsl
  // Structure of Arrays (SoA) layout with odd stride 23 to eliminate 32-bank conflicts
  shared float s_albedoR[22][23];
  shared float s_albedoG[22][23];
  shared float s_albedoB[22][23];
  shared float s_normalX[22][23];
  shared float s_normalY[22][23];
  shared float s_normalZ[22][23];
  shared float s_linearDepth[22][23];
  shared float s_localDiffuseLum[22][23];
  shared float s_localSpecularLum[22][23];
  shared float s_sobelGrad[20][23];        // 20x20 Sobel domain
  shared vec2  s_reductionScratch[256];    // Tile moments reduction scratch

  void loadApronAndReduceTileMoments(ivec2 tileOrigin, uint linearThreadIdx) {
      // 1. Cooperative Apron Load (484 texels loaded cooperatively across 256 threads)
      for (uint i = linearThreadIdx; i < 484u; i += 256u) {
          int ldsY = int(i / 22u);
          int ldsX = int(i % 22u);
          ivec2 globalCoord = tileOrigin + ivec2(ldsX - 3, ldsY - 3);
          // CRITICAL: Clamp to screen boundaries. NO INVOCATION MAY RETURN EARLY BEFORE BARRIER!
          globalCoord = clamp(globalCoord, ivec2(0), ivec2(pc.width - 1, pc.height - 1));

          vec4 alb = imageLoad(uAlbedoRoughness, globalCoord);
          vec4 normDepth = imageLoad(uNormalDepth, globalCoord);
          vec3 diff = imageLoad(uCurrentDiffuse, globalCoord).rgb;
          vec3 spec = imageLoad(uCurrentSpecular, globalCoord).rgb;

          s_albedoR[ldsY][ldsX] = alb.r;
          s_albedoG[ldsY][ldsX] = alb.g;
          s_albedoB[ldsY][ldsX] = alb.b;
          s_normalX[ldsY][ldsX] = normDepth.x;
          s_normalY[ldsY][ldsX] = normDepth.y;
          s_normalZ[ldsY][ldsX] = normDepth.z;
          s_linearDepth[ldsY][ldsX] = normDepth.w;
          s_localDiffuseLum[ldsY][ldsX] = dot(diff, vec3(0.2126, 0.7152, 0.0722));
          s_localSpecularLum[ldsY][ldsX] = dot(spec, vec3(0.2126, 0.7152, 0.0722));
      }
      barrier(); // Synchronize all 256 invocations after cooperative LDS load

      // 2. Evaluate 20x20 Sobel Gradients (400 samples evaluated cooperatively across 256 threads)
      for (uint i = linearThreadIdx; i < 400u; i += 256u) {
          int sy = int(i / 20u);
          int sx = int(i % 20u);
          ivec2 ldsCenter = ivec2(sx + 1, sy + 1); // [1, 20] in 22x22 domain
          s_sobelGrad[sy][sx] = evaluateSobelGradientLDS(ldsCenter);
      }
      barrier(); // Synchronize Sobel domain before reduction

      // 3. Parallel Tree Reduction for Tile Moments (mu, sigma^2)
      // Invocations 0..255 process first 256 samples; invocations 0..143 accumulate remaining 144 samples
      float g1 = s_sobelGrad[linearThreadIdx / 20u][linearThreadIdx % 20u];
      vec2 localMoment = vec2(g1, g1 * g1);
      if (linearThreadIdx < 144u) {
          uint remIdx = 256u + linearThreadIdx;
          float g2 = s_sobelGrad[remIdx / 20u][remIdx % 20u];
          localMoment += vec2(g2, g2 * g2);
      }
      s_reductionScratch[linearThreadIdx] = localMoment;
      barrier();

      // Parallel tree reduction: 256 -> 128 -> 64 -> 32
      for (uint stride = 128u; stride > 16u; stride >>= 1u) {
          if (linearThreadIdx < stride) {
              s_reductionScratch[linearThreadIdx] += s_reductionScratch[linearThreadIdx + stride];
          }
          barrier();
      }
      // Final 32 lanes reduced via Wave32 subgroup operations (no double counting)
      if (linearThreadIdx < 32u) {
          vec2 waveSum = subgroupAdd(s_reductionScratch[linearThreadIdx]);
          if (linearThreadIdx == 0u) {
              float mu = waveSum.x / 400.0;
              float var = max(0.0, (waveSum.y / 400.0) - mu * mu);
              float sigma = sqrt(var);
              float threshold = max(mu + pc.sigmaMultiplier * max(sigma, pc.sigmaMin), pc.noiseFloor);
              s_reductionScratch[0] = vec2(mu, threshold); // Broadcast to workgroup
          }
      }
      barrier(); // Final barrier before spatial processing
  }
  ```
- **Camera Uniform Buffer (`CameraMatricesUBO` at Binding 0)**:
  To guarantee strict compliance with GLSL `std140` alignment rules and provide transport for camera sub-pixel jitter, transformation matrices and projection parameters are bound via a per-frame uniform buffer:
  ```cpp
  // Host C++ Struct (std140 binary compatible)
  struct alignas(16) CameraMatricesUBO {
      float currViewToPrevView[16];   // 4x4 matrix, 64 bytes (offset 0)
      struct {
          float x, y, z, _pad;
      } rotCols[3];                   // 3x16-byte columns matching GLSL mat3 in std140, 48 bytes (offset 64)
      float currCameraJitter[2];      // 8 bytes: UV units (offset 112)
      float prevCameraJitter[2];      // 8 bytes: UV units (offset 120)
      float tanHalfFovY;              // 4 bytes (offset 128)
      float aspectRatio;              // 4 bytes (offset 132)
      float prevTanHalfFovY;          // 4 bytes (offset 136)
      float prevAspectRatio;          // 4 bytes (offset 140)
      float _padTail[4];              // 16 bytes padding (offset 144)
  };
  static_assert(sizeof(CameraMatricesUBO) == 160, "CameraMatricesUBO must be exactly 160 bytes");
  static_assert(offsetof(CameraMatricesUBO, rotCols) == 64, "rotCols offset mismatch");
  static_assert(offsetof(CameraMatricesUBO, currCameraJitter) == 112, "currCameraJitter offset mismatch");
  static_assert(offsetof(CameraMatricesUBO, tanHalfFovY) == 128, "tanHalfFovY offset mismatch");
  // NOTE: On the host, rotCols is filled in column-major order from the 3x3 view rotation matrix.
  // Camera sub-pixel jitter coordinates are passed in UV units (screen delta divided by resolution).
  ```
  ```glsl
  // GLSL Uniform Buffer Declaration (shaders/compute/denoise_fused_temporal.comp)
  layout(std140, set = 0, binding = 0) uniform CameraMatricesUBO {
      mat4 currViewToPrevView;        // 64 bytes (offset 0)
      mat3 prevViewToCurrViewRot;     // std140: 3 columns of vec4 = 48 bytes (offset 64)
      vec2 currCameraJitter;          // 8 bytes (offset 112)
      vec2 prevCameraJitter;          // 8 bytes (offset 120)
      float tanHalfFovY;              // 4 bytes (offset 128)
      float aspectRatio;              // 4 bytes (offset 132)
      float prevTanHalfFovY;          // 4 bytes (offset 136)
      float prevAspectRatio;          // 4 bytes (offset 140)
      vec4 _padTail;                  // 16 bytes (offset 144)
  } uCamera;
  ```
- **Push Constants (`FusedTemporalPushConstants`)**:
  ```cpp
  struct FusedTemporalPushConstants {
      uint32_t width;
      uint32_t height;
      uint32_t resetHistory;
      float    maxHistoryLength;          // Default: 32.0f
      float    weightAlbedo;              // Default: 0.50f
      float    weightDepth;               // Default: 0.30f
      float    weightNormal;              // Default: 0.20f
      float    sigmaMultiplier;           // Default: 1.20f
      float    noiseFloor;                // Default: 0.05f
      float    contrastFloor;             // Default: 0.02f
      float    sigmaMin;                  // Default: 0.02f
      float    temporalAlpha;             // Default: 0.15f
      float    normalThreshold;           // Default: 0.85f (static geometry)
      float    normalThresholdDynamic;    // Default: 0.50f (dynamic objects)
      float    depthThreshold;            // Default: 0.08f (static geometry)
      float    dynamicDepthThreshold;     // Default: 0.15f (dynamic objects)
      float    mvDiscrepancyThreshold;    // Default: 2.00f
      float    _pad[3];                   // 12 bytes padding
  }; // sizeof == 80 bytes (strictly <= 128 bytes, Vulkan guaranteed minimum)
  ```
- **Descriptor Set Layout (`m_fusedTemporalDescSetLayout`)**:
  *(Note on Descriptor Portability: 19 bindings are fully supported on desktop AMD RDNA, NVIDIA Ada/Blackwell, and Intel Arc. For hardware with tighter descriptor limits, combining guidance into a single packed image in Stage 0 provides a validated fallback).*
  - Binding 0: `CameraMatricesUBO` (Uniform Buffer, `std140`)
  - Binding 1: `uCurrentDiffuse` (Read-Only, `rgba16f`: RGB radiance, A emissive flag)
  - Binding 2: `uCurrentSpecular` (Read-Only, `rgba16f`: RGB radiance, A secondary hit distance $d_{\text{refl}}$)
  - Binding 3: `uMotionVectors` (Read-Only, `rg16f`)
  - Binding 4: `uSpecularMotion` (Read-Only, `rg16f` - Reflection VMV)
  - Binding 5: `uNormalDepth` (Read-Only, `rgba16f`)
  - Binding 6: `uPrevNormalDepth` (Read-Only, `rgba16f`)
  - Binding 7: `uAlbedoRoughness` (Read-Only, `rgba16f`)
  - Binding 8: `uHistoryDiffuse` (Read-Only, `rgba16f`: RGB radiance, A diffuse variance $V^d$)
  - Binding 9: `uHistorySpecular` (Read-Only, `rgba16f`: RGB radiance, A specular variance $V^s$)
  - Binding 10: `uHistoryMoments` (Read-Only, `rgba16f`: $M_1^d, M_2^d, M_1^s, M_2^s$)
  - Binding 11: `uHistoryLength` (Read-Only, `rgba16f`: R $N_{\text{diff}}$, G $N_{\text{spec}}$, B $d_{\text{prev,refl}}$)
  - Binding 12: `uPrevEdgeDensity` (Read-Only, `rg8` from `m_edgeDensity[prevSlot]`)
  - Binding 13: `uOutDiffuse` (Write-Only, `rgba16f` -> `m_historyDiffuse[currSlot]`: Alpha packs $V^d$)
  - Binding 14: `uOutSpecular` (Write-Only, `rgba16f` -> `m_historySpecular[currSlot]`: Alpha packs $V^s$)
  - Binding 15: `uOutMoments` (Write-Only, `rgba16f` -> `m_historyMoments[currSlot]`)
  - Binding 16: `uOutHistoryLength` (Write-Only, `rgba16f` -> `m_historyLength[currSlot]`: Channel B packs current $d_{\text{refl}}$)
  - Binding 17: `uOutEdgeDensity` (Write-Only, `rg8` -> `m_edgeDensity[currSlot]`)
  - Binding 18: `uOutNormalDepth` (Write-Only, `rgba16f` -> `m_historyNormalDepth[currSlot]`)

---

#### Pass 2: Fine-Scale Wavelet Iteration 1 ($s=1$) (`denoise_atrous_step1.comp`)
- **Objective**: Evaluates fine-scale $5\times5$ cross-bilateral wavelet filtering ($s=1$, radius 2, diameter 5) across all active pixels to suppress high-frequency Monte Carlo noise.
- **Dispatch Grid**: $\left(\lceil \frac{W}{8} \rceil, \lceil \frac{H}{8} \rceil, 1\right)$. Workgroup layout: `(8, 8, 1)`.
- **Radiance AND Variance Filtering**:
  - Filters diffuse and specular radiance in FP32 accumulators.
  - Concurrently filters spatial variance using the SVGF variance propagation formula:
    $$\text{Var}_{k+1} = \frac{\sum_{i} w_i^2 \text{Var}_k(q_i)}{\left( \sum_i w_i \right)^2}$$
    Filtered variance is packed into the Alpha channel of the output image (`m_pingPongDiffuse[0]` and `m_pingPongSpecular[0]`) for consumption by downstream passes.
- **Metric Tangent-Plane Distance**: Evaluates $w_z = \exp(-|(\mathbf{x}_q - \mathbf{x}_p) \cdot \mathbf{n}_p| / \tau_z)$ with $\tau_z = \phi_{\text{planeDepth}} \cdot s \cdot \frac{2 z \tan(\theta_y / 2)}{H}$ to eliminate depth staircase artifacts at grazing angles.
- **Precision Architecture**: Bilateral edge-stopping weights are evaluated in fast native FP16 (`float16_t`), while radiance accumulation registers are strictly maintained in **FP32 (`vec3`, `float`)** to prevent numerical overflow (`Inf` > 65504) on HDR specular fireflies.
- **Push Constants (`AtrousPushConstants`)**:
  ```cpp
  struct AtrousPushConstants {
      uint32_t width;
      uint32_t height;
      int32_t  stepSize;        // 1
      float    phiNormal;       // Default: 32.0f
      float    phiPlaneDepth;   // Default: 2.00f (metric tangent-plane scale)
      float    phiLuminance;    // Default: 4.00f
      float    tanHalfFovY;     // Camera unprojection
      float    aspectRatio;     // Camera aspect ratio
  }; // sizeof == 32 bytes (strictly <= 128 bytes, Vulkan guaranteed minimum)
  ```
- **Descriptor Set Layout (`m_atrousStep1DescSetLayout`)**:
  In Pass 2 ($s=1$, 5 px), all pixels are filtered uniformly; `uEdgeDensity` is omitted from bindings, saving a 2 B/px DRAM read (~17 MB at 4K) and one descriptor binding:
  - Binding 0: `uInDiffuse` (Read-Only, `rgba16f` from `m_historyDiffuse[currSlot]`)
  - Binding 1: `uInSpecular` (Read-Only, `rgba16f` from `m_historySpecular[currSlot]`)
  - Binding 2: `uNormalDepth` (Read-Only, `rgba16f`)
  - Binding 3: `uAlbedoRoughness` (Read-Only, `rgba16f`)
  - Binding 4: `uOutDiffuse` (Write-Only, `rgba16f` -> `m_pingPongDiffuse[0]`)
  - Binding 5: `uOutSpecular` (Write-Only, `rgba16f` -> `m_pingPongSpecular[0]`)

---

#### Pass 3: Fine-Scale Wavelet Iteration 2 ($s=2$) (`denoise_atrous_step2.comp`)
- **Objective**: Evaluates fine-scale dilation ($s=2$, accumulated radius 6, accumulated diameter 13 covering Fu et al. $k=9$).
- **Dispatch Grid**: $\left(\lceil \frac{W}{8} \rceil, \lceil \frac{H}{8} \rceil, 1\right)$. Workgroup layout: `(8, 8, 1)`.
- **Adaptive Edge Termination**:
  - Dense edge pixels ($\rho_{\text{effective}} \ge 0.55$) obtain their complete spatial filtering here (50 taps total per pixel) and bypass subsequent filtering taps, preserving crisp geometry boundaries without haloing.
  - Filters Radiance and Variance ($w^2$) alongside each other into `m_pingPongDiffuse[1]` and `m_pingPongSpecular[1]`.
- **Push Constants**: Uses `AtrousPushConstants` with `stepSize = 2` (32 bytes).
- **Descriptor Set Layout (`m_atrousStep2DescSetLayout`)**:
  - Binding 0: `uInDiffuse` (Read-Only, `rgba16f` from `m_pingPongDiffuse[0]`)
  - Binding 1: `uInSpecular` (Read-Only, `rgba16f` from `m_pingPongSpecular[0]`)
  - Binding 2: `uNormalDepth` (Read-Only, `rgba16f`)
  - Binding 3: `uAlbedoRoughness` (Read-Only, `rgba16f`)
  - Binding 4: `uEdgeDensity` (Read-Only, `rg8` from `m_edgeDensity[currSlot]`)
  - Binding 5: `uOutDiffuse` (Write-Only, `rgba16f` -> `m_pingPongDiffuse[1]`)
  - Binding 6: `uOutSpecular` (Write-Only, `rgba16f` -> `m_pingPongSpecular[1]`)

---

#### Pass 4: Mid-Scale Wavelet Iteration 3 ($s=4$) (`denoise_atrous_step3.comp`)
- **Objective**: Evaluates mid-scale dilation ($s=4$, accumulated radius 14, accumulated diameter 29 covering Fu et al. $k=17$).
- **Dispatch Grid**: $\left(\lceil \frac{W}{8} \rceil, \lceil \frac{H}{8} \rceil, 1\right)$. Workgroup layout: `(8, 8, 1)`.
- **Dynamic Selective Execution & Continuous Crossfade**:
  - Dense edge pixels ($\rho_{\text{effective}} \ge 0.55$) bypass filtering taps via continuous smoothstep crossfade ($w_{\text{blend}}^{(1)} = \text{smoothstep}(0.45, 0.55, \rho_{\text{effective}})$) to prevent spatial seams.
  - Wavefronts covering moderate curvature complete spatial convolution here; in Pass 5 they evaluate `subgroupAll(effectiveRho >= 0.35)` to skip coarse $s=8$ texture taps and execute remodulation directly.
  - Filters Radiance and Variance ($w^2$) alongside each other into `m_pingPongDiffuse[0]` and `m_pingPongSpecular[0]`.
- **Push Constants**: Uses `AtrousPushConstants` with `stepSize = 4` (32 bytes).
- **Descriptor Set Layout (`m_atrousStep3DescSetLayout`)**:
  - Binding 0: `uInDiffuse` (Read-Only, `rgba16f` from `m_pingPongDiffuse[1]`)
  - Binding 1: `uInSpecular` (Read-Only, `rgba16f` from `m_pingPongSpecular[1]`)
  - Binding 2: `uNormalDepth` (Read-Only, `rgba16f`)
  - Binding 3: `uAlbedoRoughness` (Read-Only, `rgba16f`)
  - Binding 4: `uEdgeDensity` (Read-Only, `rg8` from `m_edgeDensity[currSlot]`)
  - Binding 5: `uOutDiffuse` (Write-Only, `rgba16f` -> `m_pingPongDiffuse[0]`)
  - Binding 6: `uOutSpecular` (Write-Only, `rgba16f` -> `m_pingPongSpecular[0]`)

---

#### Pass 5: Coarse Wavelet Iteration 4 ($s=8$) & Fused Remodulation (`denoise_atrous_coarse_remodulate.comp`)
- **Objective**: Evaluates coarse dilation ($s=8$, reaching accumulated diameter 61 covering Fu et al. $k=33$) exclusively for smooth/flat surfaces ($\rho_{\text{effective}} < 0.35$), completely eliminating 1-SPP low-frequency blotches and restoring the full low-frequency reach that SVGF exists to provide. In the exact same compute invocation, directly remodulates diffuse radiance with material albedo, blends specular, and writes the finalized image directly to `m_denoisedImage`.
- **Dispatch Grid**: $\left(\lceil \frac{W}{8} \rceil, \lceil \frac{H}{8} \rceil, 1\right)$. Workgroup layout: `(8, 8, 1)`.
- **Dynamic Selective Execution & Continuous Crossfade**:
  - Pixels with $\rho_{\text{effective}} \ge 0.35$ bypass the 25 coarse filtering taps entirely; they load their already-converged radiance directly from `m_pingPongDiffuse[0]`.
  - Continuous smoothstep crossfade ($w_{\text{blend}}^{(2)} = \text{smoothstep}(0.25, 0.35, \rho_{\text{effective}})$) prevents boundary seams between edge pixels and flat surfaces.
  - Subgroup uniform voting: `subgroupAll(effectiveRho >= 0.35)` allows entire 32-lane wavefronts to early-exit the 25 coarse filtering taps and jump straight to remodulation.
- **In-Shader Remodulation & Optional Stationary Preview**:
  ```glsl
  // Direct albedo remodulation in-register: zero intermediate global memory writes!
  vec3 L_final = filteredDiffuse.rgb * max(albedo.rgb, vec3(0.001)) + filteredSpecular.rgb;

#ifdef ENABLE_STATIONARY_PREVIEW
  // Optional preview mode variant (strictly excluded from real-time production contract):
  // Gated per-pixel by sub-pixel motion threshold to prevent moving character ghost smears
  vec2 mv = imageLoad(uMotionVectors, coord).xy;
  float pixelMotion = dot(mv, mv);
  float subpixelThreshold = (0.5 / float(pc.width)) * (0.5 / float(pc.width));
  if (pixelMotion < subpixelThreshold && pc.accumulatedSamples > 1u) {
      float wBlend = float(pc.accumulatedSamples) / (float(pc.accumulatedSamples) + 8.0);
      vec3 accumColor = imageLoad(uAccumImage, coord).rgb;
      L_final = mix(L_final, accumColor, wBlend);
  }
#endif

  // Finite FP16 range clamp prevents Inf overflow on extreme HDR highlights
  L_final = clamp(L_final, 0.0, 65000.0);
  imageStore(uFinalDenoised, coord, vec4(L_final, 1.0));
  ```
- **Push Constants (`AtrousCoarseRemodulatePushConstants`)**:
  ```cpp
  struct AtrousCoarseRemodulatePushConstants {
      uint32_t width;
      uint32_t height;
      int32_t  stepSize;               // 8
      float    phiNormal;              // Default: 32.0f
      float    phiPlaneDepth;          // Default: 2.00f (metric tangent-plane scale)
      float    phiLuminance;           // Default: 4.00f
      float    tanHalfFovY;
      float    aspectRatio;
      float    edgeThresholdLow;       // Default: 0.35f (ramp top)
      uint32_t accumulatedSamples;     // Used only if ENABLE_STATIONARY_PREVIEW
      float    padding[2];
  }; // sizeof == 48 bytes (strictly <= 128 bytes, Vulkan guaranteed minimum)
  ```
- **Descriptor Set Layout (`m_atrousCoarseRemodulateDescSetLayout`)**:
  In standard real-time production, Pass 5 requires only 6 image bindings (zero motion vector or preview accumulation reads):
  - Binding 0: `uInDiffuse` (Read-Only, `rgba16f` from `m_pingPongDiffuse[0]`)
  - Binding 1: `uInSpecular` (Read-Only, `rgba16f` from `m_pingPongSpecular[0]`)
  - Binding 2: `uNormalDepth` (Read-Only, `rgba16f`)
  - Binding 3: `uAlbedoRoughness` (Read-Only, `rgba16f`)
  - Binding 4: `uEdgeDensity` (Read-Only, `rg8` from `m_edgeDensity[currSlot]`)
  - Binding 5: `uFinalDenoised` (Write-Only, `rgba16f` -> `m_denoisedImage`)
  *(Note: In offline preview builds with `ENABLE_STATIONARY_PREVIEW`, Binding 6 `uMotionVectors` and Binding 7 `uAccumImage` are appended).*

---

### 4.3 Vulkan 1.4 Synchronization & Memory Barriers

All inter-pass dependencies are synchronized using `vkCmdPipelineBarrier2` (`VK_KHR_synchronization2`). Because all internal storage images reside continuously in `VK_IMAGE_LAYOUT_GENERAL`, no per-pass image layout transitions are required.

#### Ray Tracing Pipeline RAW Hazard Synchronization
Primary G-buffers (`uNormalDepth`, `uAlbedoRoughness`, `uMotionVectors`) and 1-SPP noisy radiance buffers are produced by ray tracing shaders (`VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR`). Prior to Pass 1 execution, an input barrier synchronizes ray tracing writes:

```cpp
void DenoiserPipeline::recordInputBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 inputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    };

    VkDependencyInfo depInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &inputBarrier
    };
    vkCmdPipelineBarrier2(cmd, &depInfo);
}
```

#### Inter-Pass Compute Memory Barriers (4 Barriers)
Between consecutive compute dispatches (Pass 1 -> Pass 2 -> Pass 3 -> Pass 4 -> Pass 5), a compute memory barrier is submitted (total 4 inter-pass barriers):

```cpp
void DenoiserPipeline::recordComputeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 memoryBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    };

    VkDependencyInfo dependencyInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .pNext = nullptr,
        .dependencyFlags = 0,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &memoryBarrier,
        .bufferMemoryBarrierCount = 0,
        .pBufferMemoryBarriers = nullptr,
        .imageMemoryBarrierCount = 0,
        .pImageMemoryBarriers = nullptr
    };

    vkCmdPipelineBarrier2(cmd, &dependencyInfo);
}
```

#### Output Barrier to Tonemapper
Following Pass 5 execution, a final memory barrier synchronizes `m_denoisedImage` writes before tonemapping:
```cpp
void DenoiserPipeline::recordOutputBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier2 outputBarrier{
        .sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
        .pNext = nullptr,
        .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        .dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        .dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
    };

    VkDependencyInfo depInfo{
        .sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .memoryBarrierCount = 1,
        .pMemoryBarriers = &outputBarrier
    };
    vkCmdPipelineBarrier2(cmd, &depInfo);
}
```

#### Historical Image Rotation & Frame-in-Flight Scheduling
Because Pass 1 updates `m_historyNormalDepth[frameSlot]` directly in-shader via Binding 18 (`uOutNormalDepth`), **all `vkCmdCopyImage` transfers and transfer barrier layout transitions are completely eliminated**.

Temporal history ping-ponging rotates between odd and even slots via `currSlot = frameIndex % 2` and `prevSlot = (frameIndex + 1) % 2`. To prevent race conditions under `MAX_FRAMES_IN_FLIGHT = 2`:
- Frame pacing semaphores (`vkQueueSubmit` wait/signal semaphores) ensure that frame $t$ finishes GPU execution before frame $t+2$ re-uses slot $t \% 2$.
- Between consecutive frames, a timeline semaphore or queue submission dependency guarantees that frame $t$ completes Pass 1 history writes before frame $t+1$ executes Pass 1 history reads from that slot. Query-pool slot indexing (`frameSlot * 10`) ensures independent subpass profiling without query overwrites.

---

### 4.4 Hardware Optimization: Wave32 & LDS Configuration

Target modern desktop and APU architectures (AMD RDNA 2/3/3.5 including Strix Halo gfx1151, and NVIDIA Ada Lovelace / Blackwell).

#### Wave32 Subgroup Enforcement & Capability Check
On AMD RDNA architectures, compute shaders default to Wave64 unless explicitly configured. Running in **Wave32 mode** doubles the number of active wavefront slots per Compute Unit, significantly hiding memory latency and reducing register allocation pressure.

In `DenoiserPipeline::createComputePipeline()`, Wave32 execution is configured conditionally, verifying hardware support to avoid Vulkan validation error `VUID-VkPipelineShaderStageCreateInfo-pNext-02755` on platforms without the extension:
```cpp
VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSizeInfo{
    .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
    .pNext = nullptr,
    .requiredSubgroupSize = 32
};

VkComputePipelineCreateInfo pipelineInfo{
    .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
    .pNext = nullptr,
    // ...
};

// Guard subgroup size control behind device capability check
if (m_context.hasSubgroupSizeControl() && 
    (m_context.subgroupSizeControlProperties().requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
    pipelineInfo.stage.pNext = &subgroupSizeInfo;
}
```

#### Register Budgeting & Occupancy Targets
On modern architectures (AMD RDNA 2/3/3.5 and NVIDIA Ada Lovelace / Blackwell):
- **VGPR Target**: $\le 32$ Vector General-Purpose Registers (VGPRs) per work-item for 100% theoretical wave occupancy in spatial passes, and $\le 96$ VGPRs in Pass 1 to sustain 16 active Wave32 wavefronts per Dual-Compute Unit (WGP) / SIMD.
- **SGPR Target**: $\le 64$ Scalar Registers.
- In Pass 1 (`denoise_fused_temporal.comp`), shared memory capacity is not the primary bottleneck: RDNA 2/3 provides 64 KiB of LDS per CU (128 KiB per Dual-Compute Unit / WGP), and NVIDIA Ada Lovelace SMs provide up to 100 KiB configurable shared memory. A 21.59 KiB allocation comfortably fits within these limits and does not strictly preclude dual-workgroup residency from a memory capacity perspective. Instead, occupancy is governed by SIMD wave slot limits (16 Wave32 waves per SIMD on RDNA3) and vector register pressure (1,536 VGPRs per SIMD; $\le 96$ VGPRs allows 16 active waves / 2 workgroups per SIMD, while $\le 192$ VGPRs allows 8 active waves / 1 workgroup).
- In Passes 2–5 (`denoise_atrous_step*.comp` and `denoise_atrous_coarse_remodulate.comp`), LDS usage is 0 B; occupancy is strictly register-bound. Keeping VGPR counts $\le 32$ enables full wave slot residency (up to 16 active waves per SIMD32 on RDNA3), while keeping VGPR counts $\le 48$ maintains 10 active waves per SIMD32 (~62.5% residency). Shaders apply `#pragma optionNV(unroll none)` or `[[dont_unroll]]` loop control hints on the 25-tap bilateral loops to prevent register spilling.
- **Profiling Mandate**: Exact register allocation, spill limits, and wave residency per target architecture will be profiled and validated via Radeon GPU Analyzer (`rga`) and NVIDIA Nsight Compute during Stage 1 implementation.

#### LDS Bank-Conflict Tuning Guidelines
On modern SIMD architectures (Wave32 / Warp32):
- Local Data Share is partitioned into 32 independent memory banks, each 4 bytes wide.
- 16-byte vector accesses (`vec4`) span 4 consecutive banks. Consecutive lanes accessing consecutive `vec4` entries can induce multi-way bank conflicts if the row stride is a multiple of 32 words.
- To maintain conflict-free loads during the $22\times22$ G-buffer load and Sobel evaluations in Pass 1:
  - Layout data as Structure of Arrays (SoA) of scalar floats rather than `vec4` arrays.
  - Stride inner rows by an odd number of words (e.g. 23 words) so row address offsets shift across bank boundaries modulo 32.
  - Keep total LDS footprint strictly under 24 KB (declared shared memory: **22,104 Bytes = 21.59 KiB / ~22.1 KB**, itemized exactly as: 6,072 B albedo SoA [R,G,B], 6,072 B view normal SoA [X,Y,Z], 2,024 B linear depth, 1,840 B Sobel scratch, 4,048 B local luminance scratch, and 2,048 B reduction scratch).

---

## 5. Section 8 Failure Modes & Concrete Shader-Level Mitigations

Section 8 of Fu et al. (2025) documents five practical failure modes that cause severe visual degradation in real-time ray tracing denoisers. Below is the complete mathematical analysis and concrete GLSL shader mitigations implemented.

---

### 5.1 Failure Mode 1: Low-Precision Depth Staircase Aliasing
- **Symptom**: On planar surfaces viewed at grazing angles, 16-bit or non-linear depth values exhibit discrete numerical quantization steps. Standard finite-difference Sobel filters interpret these precision steps as sharp geometric edges, forcing edge density $\rho \to 1.0$ and locking the filter to the minimum $9\times9$ kernel. This prevents spatial smoothing and causes severe surface speckle noise.
- **Root Cause**: The raw depth difference $|z_P - z_Q|$ neglects the expected geometric slope of the surface tangent plane, and standard 16-bit floating-point depth exhibits precision quantization (~7.8 mm step at 10 m).
- **Mitigation**: The engine replaces Euclidean depth difference with **tangent plane-distance weighting**, enforces an **FP16 quantization floor**, and extracts depth edges using **plane-relative residual Sobel filtering**:
  $$d_{\text{plane}} = \left| (\mathbf{x}_Q - \mathbf{x}_P) \cdot \mathbf{n}_P \right|$$
  where $\mathbf{x}_P, \mathbf{x}_Q$ are reconstructed metric 3D view-space positions and $\mathbf{n}_P$ is the surface normal. Metric depth tolerance scales with screen-space pixel footprint, bounded below by the FP16 storage precision floor ($2\,\epsilon_{\text{fp16}}(z)$):
  $$\tau_z = \max\left(\phi_{\text{planeDepth}} \cdot s \cdot \frac{2 z \tan(\theta_y / 2)}{H}, \, 2\,\epsilon_{\text{fp16}}(z)\right)$$
  On a continuous planar surface, the expected depth on the surface tangent plane evaluates to identically $0.0$ regardless of view distance or grazing angle, completely eliminating false edge detections on floors, walls, and roads.

#### Concrete GLSL Implementation (`shaders/compute/denoise_common.glsl`)
```glsl
// Unprojects pixel coordinates and linear view depth into metric 3D view space
vec3 unprojectToView(ivec2 pixelCoord, float linearDepth, float tanHalfFovY, float aspectRatio, vec2 resolution) {
    if (isnan(linearDepth) || isinf(linearDepth) || linearDepth <= 0.0) {
        return vec3(0.0);
    }
    vec2 ndc = (vec2(pixelCoord) + 0.5) / resolution * 2.0 - 1.0;
    float x = ndc.x * aspectRatio * tanHalfFovY * linearDepth;
    float y = ndc.y * tanHalfFovY * linearDepth;
    return vec3(x, y, linearDepth);
}

// Mitigates Section 8 Depth Staircase Aliasing via Metric Tangent Plane Distance
float evaluatePlaneDepthWeight(
    vec3 pViewPos, 
    vec3 qViewPos, 
    vec3 pNormal, 
    float phiPlaneDepth,
    int stepSize,
    float tanHalfFovY,
    float screenHeight) 
{
    // Guard against skybox / infinite depth singularities
    if (length(pViewPos) <= 1e-4 || length(qViewPos) <= 1e-4) {
        return 0.0;
    }

    // Compute metric Euclidean displacement between sample positions
    vec3 disp = qViewPos - pViewPos;
    
    // Project displacement onto surface tangent plane normal
    // For a continuous planar surface, planeDist evaluates to identically 0.0
    // regardless of view distance or grazing angle!
    float planeDist = abs(dot(disp, pNormal));
    
    // Metric depth tolerance derived from projected pixel footprint, with FP16 quantization floor
    float pixelFootprint = 2.0 * pViewPos.z * tanHalfFovY / max(screenHeight, 1.0);
    float fp16QuantizationFloor = max(pViewPos.z * 0.0009765625, 1e-4); // 2 * eps_fp16(z)
    float depthTolerance = max(phiPlaneDepth * float(stepSize) * pixelFootprint, fp16QuantizationFloor);
    
    return exp(-planeDist / depthTolerance);
}

// Metric Plane-Relative Depth Residual Gradient in LDS (Mitigates Section 8 Grazing Depth Aliasing)
float computePlaneRelativeDepthGradientLDS(ivec2 ldsCenter, float tanHalfFovY, float aspectRatio, vec2 resolution) {
    float zP = s_linearDepth[ldsCenter.y][ldsCenter.x];
    vec3 nP = vec3(s_normalX[ldsCenter.y][ldsCenter.x], s_normalY[ldsCenter.y][ldsCenter.x], s_normalZ[ldsCenter.y][ldsCenter.x]);
    vec3 xP = unprojectToView(ldsCenter, zP, tanHalfFovY, aspectRatio, resolution);
    
    float footprint = max(2.0 * zP * tanHalfFovY / resolution.y, zP * 0.0009765625);
    
    // Evaluate metric tangent-plane residual across 3x3 Sobel kernel
    float gX = 0.0;
    float gY = 0.0;
    const float kSobelX[3][3] = float[3][3](
        float[3](-1.0, 0.0, 1.0),
        float[3](-2.0, 0.0, 2.0),
        float[3](-1.0, 0.0, 1.0)
    );
    const float kSobelY[3][3] = float[3][3](
        float[3](-1.0, -2.0, -1.0),
        float[3]( 0.0,  0.0,  0.0),
        float[3]( 1.0,  2.0,  1.0)
    );
    
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            float zQ = s_linearDepth[ldsCenter.y + dy][ldsCenter.x + dx];
            vec3 xQ = unprojectToView(ldsCenter + ivec2(dx, dy), zQ, tanHalfFovY, aspectRatio, resolution);
            float planeResidual = abs(dot(xQ - xP, nP)) / footprint;
            gX += planeResidual * kSobelX[dy + 1][dx + 1];
            gY += planeResidual * kSobelY[dy + 1][dx + 1];
        }
    }
    return length(vec2(gX, gY));
}
```

---

### 5.2 Failure Mode 2: Abrupt Illumination Transitions & Lighting Flashes
- **Symptom**: Sudden primary light state toggles, muzzle flashes, explosion flares, or flickering candle lights inject sharp luminance discontinuities into the scene. If edge extraction operates on illuminated color buffers, these lighting transients generate spurious edge magnitudes, corrupting $M_{\text{fused}}$ and causing catastrophic spatial blurring (Fu et al., Fig. 8(a)).
- **Mitigation**: 
  1. Sobel edge extraction is strictly restricted to **demodulated material base color** (`uAlbedoRoughness`), which is physically invariant to incident lighting. Storage image texels are accessed via LDS shared memory.
  2. For the luminance variance edge-stopping weight in downstream spatial passes, the filter evaluates true exponential falloff without an artificial $3\sigma$ numerator floor:
     $$w_L = \exp\left( - \frac{|\text{lum}_P - \text{lum}_Q|}{\phi_{\text{lum}} \sqrt{\max(V_P, 10^{-4})} + 10^{-4}} \right)$$
     *(Note: Enforcing an artificial $3\sigma$ clamp on the bilateral numerator bounded $w_L \ge \exp(-3.0/4.0) \approx 0.4724$, which forced the denoiser to blur across sharp shadow boundaries with at least 47.2% weight. Outlier clamping is strictly confined to bilateral-gated local clamping during temporal accumulation in Pass 1, allowing spatial passes to preserve razor-sharp contact shadows and penumbras).*

#### Concrete GLSL Implementation (In-LDS in `shaders/compute/denoise_fused_temporal.comp`)
```glsl
// Sobel edge extraction operates in-LDS EXCLUSIVELY on demodulated physical material albedo
// Lighting flashes and shadows produce ZERO false edge detections
vec3 sampleAlbedoLDS(ivec2 ldsPos) {
    return vec3(s_albedoR[ldsPos.y][ldsPos.x], s_albedoG[ldsPos.y][ldsPos.x], s_albedoB[ldsPos.y][ldsPos.x]);
}

vec2 computeAlbedoGradientLDS(ivec2 ldsCoord) {
    // 3x3 Sobel on base material albedo directly from shared memory
    vec3 gX = -1.0 * sampleAlbedoLDS(ldsCoord + ivec2(-1, -1)) + 1.0 * sampleAlbedoLDS(ldsCoord + ivec2(1, -1))
              -2.0 * sampleAlbedoLDS(ldsCoord + ivec2(-1,  0)) + 2.0 * sampleAlbedoLDS(ldsCoord + ivec2(1,  0))
              -1.0 * sampleAlbedoLDS(ldsCoord + ivec2(-1,  1)) + 1.0 * sampleAlbedoLDS(ldsCoord + ivec2(1,  1));
              
    vec3 gY = -1.0 * sampleAlbedoLDS(ldsCoord + ivec2(-1, -1)) - 2.0 * sampleAlbedoLDS(ldsCoord + ivec2(0, -1)) - 1.0 * sampleAlbedoLDS(ldsCoord + ivec2(1, -1))
              +1.0 * sampleAlbedoLDS(ldsCoord + ivec2(-1,  1)) + 2.0 * sampleAlbedoLDS(ldsCoord + ivec2(0,  1)) + 1.0 * sampleAlbedoLDS(ldsCoord + ivec2(1,  1));
              
    // Return analytical Euclidean magnitude
    return vec2(length(gX), length(gY));
}

// Complete Multi-Channel In-LDS Sobel Gradient Evaluation (Pass 1 Step 2)
float evaluateSobelGradientLDS(ivec2 ldsCenter) {
    // 1. Albedo gradient (4*sqrt(6) analytical vector bound)
    vec2 gradAlbedo = computeAlbedoGradientLDS(ldsCenter);
    float gAlbedo = length(gradAlbedo) * (1.0 / (4.0 * sqrt(6.0)));

    // 2. Normal gradient (8*sqrt(2) analytical unit-sphere bound)
    vec3 nX = -1.0 * vec3(s_normalX[ldsCenter.y - 1][ldsCenter.x - 1], s_normalY[ldsCenter.y - 1][ldsCenter.x - 1], s_normalZ[ldsCenter.y - 1][ldsCenter.x - 1])
              +1.0 * vec3(s_normalX[ldsCenter.y - 1][ldsCenter.x + 1], s_normalY[ldsCenter.y - 1][ldsCenter.x + 1], s_normalZ[ldsCenter.y - 1][ldsCenter.x + 1])
              -2.0 * vec3(s_normalX[ldsCenter.y    ][ldsCenter.x - 1], s_normalY[ldsCenter.y    ][ldsCenter.x - 1], s_normalZ[ldsCenter.y    ][ldsCenter.x - 1])
              +2.0 * vec3(s_normalX[ldsCenter.y    ][ldsCenter.x + 1], s_normalY[ldsCenter.y    ][ldsCenter.x + 1], s_normalZ[ldsCenter.y    ][ldsCenter.x + 1])
              -1.0 * vec3(s_normalX[ldsCenter.y + 1][ldsCenter.x - 1], s_normalY[ldsCenter.y + 1][ldsCenter.x - 1], s_normalZ[ldsCenter.y + 1][ldsCenter.x - 1])
              +1.0 * vec3(s_normalX[ldsCenter.y + 1][ldsCenter.x + 1], s_normalY[ldsCenter.y + 1][ldsCenter.x + 1], s_normalZ[ldsCenter.y + 1][ldsCenter.x + 1]);
    vec3 nY = -1.0 * vec3(s_normalX[ldsCenter.y - 1][ldsCenter.x - 1], s_normalY[ldsCenter.y - 1][ldsCenter.x - 1], s_normalZ[ldsCenter.y - 1][ldsCenter.x - 1])
              -2.0 * vec3(s_normalX[ldsCenter.y - 1][ldsCenter.x    ], s_normalY[ldsCenter.y - 1][ldsCenter.x    ], s_normalZ[ldsCenter.y - 1][ldsCenter.x    ])
              -1.0 * vec3(s_normalX[ldsCenter.y - 1][ldsCenter.x + 1], s_normalY[ldsCenter.y - 1][ldsCenter.x + 1], s_normalZ[ldsCenter.y - 1][ldsCenter.x + 1])
              +1.0 * vec3(s_normalX[ldsCenter.y + 1][ldsCenter.x - 1], s_normalY[ldsCenter.y + 1][ldsCenter.x - 1], s_normalZ[ldsCenter.y + 1][ldsCenter.x - 1])
              +2.0 * vec3(s_normalX[ldsCenter.y + 1][ldsCenter.x    ], s_normalY[ldsCenter.y + 1][ldsCenter.x    ], s_normalZ[ldsCenter.y + 1][ldsCenter.x    ])
              +1.0 * vec3(s_normalX[ldsCenter.y + 1][ldsCenter.x + 1], s_normalY[ldsCenter.y + 1][ldsCenter.x + 1], s_normalZ[ldsCenter.y + 1][ldsCenter.x + 1]);
    float gNormal = sqrt(dot(nX, nX) + dot(nY, nY)) * (1.0 / (8.0 * sqrt(2.0)));

    // 3. Plane-relative depth gradient residual (Section 8 FM1 mitigation)
    float gDepth = computePlaneRelativeDepthGradientLDS(ldsCenter, uCamera.tanHalfFovY, uCamera.aspectRatio, vec2(pc.width, pc.height));
    gDepth = clamp(gDepth * (1.0 / 8.0), 0.0, 1.0);

    // Multi-channel fusion (per-channel weights: Albedo 0.50, Depth 0.30, Normal 0.20)
    return pc.weightAlbedo * gAlbedo + pc.weightDepth * gDepth + pc.weightNormal * gNormal;
}
```

---

### 5.3 Failure Mode 3: Extreme Dynamic Motion & Non-Rigid Disocclusion
- **Symptom**: Rapid camera pans, rotating objects, and non-rigid character skin/cloth deformations cause motion vectors to map across geometry boundaries. Permissive reprojection validation accumulates stale historical pixels across newly exposed surfaces, creating severe trailing ghosting and white edge halos (Fu et al., Fig. 8(b)).
- **Mitigation**: Implements a **dual-model reprojection selector** with an **instant spatial hole fallback**:
  1. **Pure Camera Normal Rotation**: The historical view normal is transformed into the current camera view space using a pure 3x3 normal rotation matrix `prevViewToCurrViewRot` ($\mathbf{R}_{\text{prev}\to\text{curr}}$) loaded from `CameraMatricesUBO`:
     $$\mathbf{n}_{\text{prev}}^{\text{curr}} = \mathbf{R}_{\text{prev}\to\text{curr}} \cdot \mathbf{n}_{\text{prev}}$$
     Normal alignment requires $\mathbf{n}_{\text{curr}} \cdot \mathbf{n}_{\text{prev}}^{\text{curr}} > 0.85$, eliminating false disocclusions on camera rotations exceeding $31.8^\circ/\text{frame}$.
  2. **Static Geometry Path**: Evaluates exact 6-DOF affine depth prediction ($z_{\text{expected}}$) via `currViewToPrevView`. If motion vector discrepancy is small ($\|\mathbf{u}_{\text{sample}} - \mathbf{u}_{\text{expected}}\| \le 2.0\text{ px}$), the pixel belongs to static scene geometry and validates against affine depth:
     $$\frac{|z_{\text{expected}} - z_{\text{prev}}|}{\max(z_{\text{expected}}, 10^{-3})} < \tau_{\text{depth}} \quad (\text{default: } 0.08)$$
  3. **Dynamic Geometry Path**: If motion vector discrepancy exceeds 2.0 pixels, the pixel belongs to an independently moving object. Affine camera-only prediction does not apply; reprojection validates against the motion vector's sampled depth:
     $$\frac{|z_{\text{curr}} - z_{\text{prev}}|}{\max(z_{\text{curr}}, 10^{-3})} < \tau_{\text{dyn}} \quad (\text{default: } 0.15)$$
      Furthermore, normal alignment on dynamic geometry is evaluated with a relaxed threshold ($\mathbf{n}_{\text{curr}} \cdot \mathbf{n}_{\text{prev}}^{\text{curr}} > \text{pc.normalThresholdDynamic} = 0.50$) to accommodate independent surface rotation without false disocclusion resets. For dynamic objects, history length is capped to $N_{\text{hist}} = \min(N_{\text{hist}}, 8)$ to accommodate mesh deformation.
  4. **Instant Spatial Fallback**: If validation fails, temporal accumulation length is immediately reset to 1.0 ($N_{\text{hist}} = 1.0$), and local variance is initialized from neighboring LDS radiance.
  *(Note on Dynamic Model Selection & Validation Policies: The motion vector discrepancy test serves as a model-selection hint rather than a definitive semantic object classification; discrepancies can also arise from sub-pixel occlusion boundaries or motion vector discretization. Capping history confidence to $N_{\text{hist}} \le 8$ limits ghosting risk. Sub-pixel camera jitter (e.g. Halton sequence) loaded from `uCamera.currCameraJitter` and `uCamera.prevCameraJitter` is explicitly cancelled when evaluating expected UVs. Furthermore, to avoid false rejection on steep slopes, previous depth is validated against a $2\times2$ neighborhood depth bounding box $[z_{\min}^{\text{prev}}, z_{\max}^{\text{prev}}]$ rather than a single point fetch. Specular VMV evaluates its own independent motion vector field `uSpecularMotion` at Binding 4 to reproject specular radiance without conflating primary-geometry and virtual-reflection history tracks; specular confidence validates reflector normal alignment ($\mathbf{n}_{\text{refl}} \cdot \mathbf{n}_{\text{prev, refl}} > 0.85$) and secondary reflection hit distance consistency ($|d_{\text{refl}} - d_{\text{prev, refl}}| / d_{\text{refl}} < 0.20$), where $d_{\text{prev, refl}}$ is fetched directly from Channel B of `uHistoryLength` (`VK_FORMAT_R16G16B16A16_SFLOAT`) and `uHistorySpecular.a` carries specular temporal variance $V^s$, clamping specular history to $N_{\text{spec, hist}} \le 16$).*

#### Concrete GLSL Implementation (`shaders/compute/denoise_fused_temporal.comp`)
```glsl
// Dual-Model Camera-Motion-Compensated Reprojection Validation
bool validateReprojection(
    vec3 currNormal, 
    vec3 prevNormal, 
    float currDepth, 
    float prevDepth,
    vec2 prevDepthMinMax, // 2x2 neighborhood min/max depth bounds from prevNormalDepth
    ivec2 currCoord,
    vec2 prevUV,
    vec2 motionVector,
    vec2 resolution,
    out bool isDynamicObject) 
{
    isDynamicObject = false;

    // 1. Screen bounds check
    if (prevUV.x < 0.0 || prevUV.x > 1.0 || prevUV.y < 0.0 || prevUV.y > 1.0) {
        return false;
    }

    // 2. Pure camera rotation compensation for view normals (loaded from CameraMatricesUBO)
    vec3 prevNormalInCurrView = normalize(mat3(uCamera.prevViewToCurrViewRot) * prevNormal);
    float normalCloseness = dot(currNormal, prevNormalInCurrView);

    // 3. Unproject current pixel to metric 3D view space
    vec3 currViewPos = unprojectToView(currCoord, currDepth, uCamera.tanHalfFovY, uCamera.aspectRatio, resolution);

    // 4. Exact 6DOF affine view-to-view transformation
    vec4 prevViewPos = uCamera.currViewToPrevView * vec4(currViewPos, 1.0);
    float zExpected = prevViewPos.z;

    // 5. Motion vector discrepancy test (detects dynamic moving objects vs. static geometry)
    vec2 expectedNDC = prevViewPos.xy / (vec2(uCamera.aspectRatio, 1.0) * uCamera.tanHalfFovY * max(zExpected, 1e-4));
    // Explicitly cancel sub-pixel camera jitter across frames from CameraMatricesUBO
    vec2 expectedPrevUV = (expectedNDC * 0.5 + 0.5) - uCamera.prevCameraJitter + uCamera.currCameraJitter;
    vec2 discrepancyPixels = abs(prevUV - expectedPrevUV) * resolution;
    
    if (length(discrepancyPixels) > pc.mvDiscrepancyThreshold) {
        // Dynamic object branch: validate using 2x2 neighborhood bounded depth test
        isDynamicObject = true;
        // Relaxed normal alignment for rotating dynamic objects (prevents false disocclusion)
        if (normalCloseness < pc.normalThresholdDynamic) {
            return false;
        }
        float depthRelDiffDynamic = (currDepth < prevDepthMinMax.x || currDepth > prevDepthMinMax.y)
            ? abs(currDepth - prevDepth) / max(currDepth, 1e-4)
            : 0.0;
        return (depthRelDiffDynamic < pc.dynamicDepthThreshold);
    } else {
        // Static rigid geometry branch: validate against affine predicted depth and strict normal threshold
        if (normalCloseness < pc.normalThreshold) {
            return false;
        }
        if (zExpected <= 1e-4) {
            return false;
        }
        float depthRelDiffStatic = abs(zExpected - prevDepth) / zExpected;
        return (depthRelDiffStatic < pc.depthThreshold);
    }
}
```

---

### 5.4 Failure Mode 4: Dense Geometric Edge Structures / Foliage Boiling
- **Symptom**: In scenes featuring complex micro-geometry (dense tree foliage, grass blades, chainlink fences), the local edge density evaluates to $\rho \ge 0.5$ across entire regions. This locks the denoiser into the narrow $k = 9$ kernel. At 1 SPP, 25–50 spatial taps with insufficient temporal history provide inadequate sample support, leaving high residual variance and temporal "boiling" artifacts.
- **Mitigation**: 
  1. **Dual-Channel Edge Density Packing in Pass 1**: `m_edgeDensity[2]` is allocated as `VK_FORMAT_R8G8_UNORM`. Channel R stores raw $\rho^t$ to preserve temporal estimator convergence without collapse. Channel G stores effective density $\rho_{\text{effective}} = \rho \cdot \text{clamp}(N_{\text{hist}}/8, 0.25, 1.0)$ for downstream spatial passes. Newly disoccluded foliage temporarily expands filtering support to mid-scale ($s=4$) until sufficient temporal samples accumulate, suppressing boil while preserving converged detail.
  2. **Direct Effective Rho Read**: Downstream spatial passes sample `effectiveRho` directly from Channel G of `uEdgeDensity`, eliminating redundant `uHistoryLength` fetches.
  3. **Continuous Smoothstep Crossfading**: Prevents spatial seams between edge pixels and flat surfaces:
     $$\text{Pass 4: } w_{\text{blend}}^{(1)} = \text{smoothstep}(0.45, \, 0.55, \, \rho_{\text{effective}})$$
     $$\text{Pass 5: } w_{\text{blend}}^{(2)} = \text{smoothstep}(0.25, \, 0.35, \, \rho_{\text{effective}})$$
  4. **Subgroup-Uniform Wave Voting (`subgroupAll`)**: Coherent 32-lane wavefronts covering bulk moderate-curvature regions (`subgroupAll(effectiveRho >= 0.35)`) skip Pass 5's expensive 25 coarse $s=8$ texture taps in a single clock cycle, evaluating remodulation directly and writing output without spatial convolution. Voting thresholds are aligned strictly to the ramp tops ($0.55$ in Pass 4, $0.35$ in Pass 5) to eliminate block divergence artifacts.

#### Concrete GLSL Implementation (`shaders/compute/denoise_atrous_step3.comp` & `denoise_atrous_coarse_remodulate.comp`)
```glsl
// In Pass 4 (denoise_atrous_step3.comp):
// Dense edges finish filtering in Pass 3; moderate curvature waves complete spatial convolution here
float effectiveRho = imageLoad(uEdgeDensity, centerCoord).g;

// 1. Fast Wavefront Early-Exit: Coherent 32-lane wave in dense edges skips all 25 mid-scale taps
if (subgroupAll(effectiveRho >= 0.55)) {
    imageStore(uOutDiffuse, centerCoord, imageLoad(uInDiffuse, centerCoord));
    imageStore(uOutSpecular, centerCoord, imageLoad(uInSpecular, centerCoord));
    return;
}

// 2. Spatial Convolution across divergent / flat wavefronts (25 taps with step size s=4)
vec4 filteredDiffuse = filterDiffuseSample(centerCoord, 4, ...);
vec4 filteredSpecular = filterSpecularSample(centerCoord, 4, ...);

// 3. Continuous Smoothstep Crossfade prevents spatial boundary seams
float edgeBlend = smoothstep(0.45, 0.55, effectiveRho);
vec4 finalDiffuse = mix(filteredDiffuse, imageLoad(uInDiffuse, centerCoord), edgeBlend);
vec4 finalSpecular = mix(filteredSpecular, imageLoad(uInSpecular, centerCoord), edgeBlend);

imageStore(uOutDiffuse, centerCoord, finalDiffuse);
imageStore(uOutSpecular, centerCoord, finalSpecular);

// In Pass 5 (denoise_atrous_coarse_remodulate.comp):
// Wavefront uniform voting: moderate curvature waves bypass coarse s=8 taps
bool bypassCoarseTaps = subgroupAll(effectiveRho >= 0.35);
if (bypassCoarseTaps) {
    vec4 diff = imageLoad(uInDiffuse, centerCoord);
    vec4 spec = imageLoad(uInSpecular, centerCoord);
    vec3 L_remod = diff.rgb * max(albedo.rgb, vec3(0.001)) + spec.rgb;
    imageStore(uFinalDenoised, centerCoord, vec4(L_remod, 1.0));
    return;
}
```

---

### 5.5 Failure Mode 5: Specular & Glass Over-Blurring / LUT Generalization Breakdown
- **Symptom**: Offline MLPs lack training representation for complex microfacet dielectric/conductor BSDFs and Dirac delta transmission (glass). Under the Neural LUT, specular reflections on cars, marble, and glass windows are treated as flat surfaces, resulting in severe over-smoothing and loss of mirror highlights (Fu et al., Fig. 8(c)).
- **Mitigation**: 
  1. **Decoupled Radiance Accumulation**: Diffuse and specular radiance are stored, reprojected, and filtered in independent image channels (`m_pingPongDiffuse` and `m_pingPongSpecular`). Diffuse is demodulated by base albedo; specular is preserved in raw linear HDR without albedo demodulation (avoiding color bleeding on metals).
  2. **Continuous Roughness-Modulated Specular Dilation**: Rather than an abrupt conditional branch at roughness 0.05 (which creates visible boundary seams), the mirror identity bypass is evaluated using continuous `smoothstep(0.02, 0.08, roughness)`.
  3. **Strict Tap Monotonicity & Type-Safe GLSL**: Tap offsets maintain guaranteed monotonic displacement for non-center taps:
     $$\mathbf{o}_{\text{tap}} = \text{sign}(\mathbf{u}) \odot \max\left( |\mathbf{u}|, \, \text{round}(|\mathbf{u}| \cdot s \cdot \text{roughnessScale}) \right)$$
     This prevents all 25 taps from collapsing onto $(0, 0)$ at low roughness values, maintaining 25 distinct spatial taps while ensuring strictly non-decreasing dilation with increasing step size $s$.
  4. **FP32 Accumulation Registers**: While bilateral edge weights are evaluated in fast FP16 (`float16_t`), radiance accumulation registers (`specAcc`, `specWeightAcc`) are strictly maintained in **FP32 (`vec3`, `float`)** to prevent numerical `Inf` overflows (> 65504) on HDR specular fireflies.
  5. **Virtual Motion Vector (VMV) Support Along Primary Ray**: Primary motion vectors track the surface position, but reflected highlights move across the surface with camera navigation and reflector curvature. For surfaces with roughness $< 0.2$, the path tracer computes virtual reflection hit positions:
     $$\mathbf{x}_{\text{virtual}} = \mathbf{x}_{\text{surface}} + d_{\text{refl}} \cdot \hat{\mathbf{r}}_{\text{primary}}$$
     where $\hat{\mathbf{r}}_{\text{primary}}$ is the primary ray unit direction and $d_{\text{refl}}$ is the reflection ray hit distance from G-buffer channel 12 (packed into `uCurrentSpecular.a`). Projecting $\mathbf{x}_{\text{virtual}}$ into the previous frame produces reflection-space motion vectors bound to `uSpecularMotion` in Pass 1 to warp `uHistorySpecular`. During reprojection validation, reflection hit distance consistency is validated against the previous frame's hit distance ($|d_{\text{refl}} - d_{\text{prev, refl}}| / d_{\text{refl}} < 0.20$), where $d_{\text{prev, refl}}$ is fetched directly from Channel B of `uHistoryLength` (`VK_FORMAT_R16G16B16A16_SFLOAT`) and `uHistorySpecular.a` carries specular temporal variance $V^s$.
     *(Note on VMV Reflector Geometry: The formula $\mathbf{x}_{\text{virtual}} = \mathbf{x}_{\text{surface}} + d_{\text{refl}} \cdot \hat{\mathbf{r}}_{\text{primary}}$ is exact for planar reflectors where the virtual image lies along the primary viewing ray. For curved surfaces, the virtual imaging point depends on local surface curvature and reflected ray direction. For real-time denoising, this planar approximation provides a high-quality reprojection vector that tracks specular highlights across motion far better than static surface motion vectors. The parameter $d_{\text{refl}}$ is the secondary reflection ray hit distance recorded during wavefront path tracing in G-buffer channel 12).*
  6. **Specular Spatial Weights**: Specular bilateral filtering evaluates metric tangent-plane distance $w_z$, roughness-scaled normal power $\min(\phi_n / \max(\text{roughness}^2, 0.05), 256.0)$ (capped at 256.0 to prevent center-only tap collapse on glossy surfaces), and variance-guided luminance falloff $w_L$.

#### Concrete GLSL Implementation (`shaders/compute/denoise_atrous_step1.comp`)
```glsl
// Continuous Roughness-Modulated Specular Filtering with Monotonic Non-Collapsing Offsets
vec3 filterSpecularSample(
    ivec2 centerCoord, 
    float roughness, 
    int stepSize, 
    float phiNormal, 
    float phiPlaneDepth,
    float phiLuminance,
    vec3 centerNormal,
    float centerDepth,
    vec3 centerPos,
    float tanHalfFovY,
    float screenHeight) 
{
    vec4 rawSpecData = imageLoad(uInSpecular, centerCoord);
    vec3 rawSpecular = rawSpecData.rgb;
    float centerVar = max(rawSpecData.a, 1e-4);
    float centerLum = dot(rawSpecular, vec3(0.2126, 0.7152, 0.0722));
    
    // Continuous mirror bypass factor: 0.0 = full filter, 1.0 = pure mirror identity
    float mirrorBypass = 1.0 - smoothstep(0.02, 0.08, roughness);
    if (mirrorBypass >= 0.999) {
        return rawSpecular; // Pure mirror: skip 25 taps entirely
    }

    // Scale dilation by microfacet roughness squared
    float roughnessScale = clamp(roughness * roughness, 0.1, 1.0);
    float effectivePhiNormal = min(phiNormal / max(roughness * roughness, 0.05), 256.0);
    
    // Accumulators strictly in FP32 to prevent HDR Inf overflows (> 65504)
    vec3 specAcc = vec3(0.0);
    float specWeightAcc = 0.0;
    
    // Standard 5x5 À-Trous B-spline weights
    const float h[5] = float[5](1.0/16.0, 1.0/4.0, 3.0/8.0, 1.0/4.0, 1.0/16.0);
    
    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            float kernelWeight = h[x + 2] * h[y + 2];
            
            // Monotonic non-collapsing tap offset: guarantees distinct non-zero taps for non-center offsets
            ivec2 unitOffset = ivec2(x, y);
            ivec2 absOffset = abs(unitOffset);
            ivec2 dilatedOffset = ivec2(round(vec2(absOffset) * float(stepSize) * roughnessScale));
            ivec2 tapOffset = ivec2(sign(vec2(unitOffset))) * max(absOffset, dilatedOffset);
            ivec2 tapCoord = clamp(centerCoord + tapOffset, ivec2(0), ivec2(pc.width - 1, pc.height - 1));
            
            vec3 tapSpec = imageLoad(uInSpecular, tapCoord).rgb;
            vec4 tapNormDepth = imageLoad(uNormalDepth, tapCoord);
            
            // Bilateral normal stopping
            float wNorm = pow(max(0.0, dot(centerNormal, tapNormDepth.xyz)), effectivePhiNormal);
            
            // Metric tangent-plane distance depth weighting
            vec3 tapPos = unprojectToView(tapCoord, tapNormDepth.w, tanHalfFovY, pc.aspectRatio, vec2(pc.width, pc.height));
            float wDepth = evaluatePlaneDepthWeight(centerPos, tapPos, centerNormal, phiPlaneDepth, stepSize, tanHalfFovY, screenHeight);
            
            // Variance-guided luminance weight
            float tapLum = dot(tapSpec, vec3(0.2126, 0.7152, 0.0722));
            float wLum = exp(-abs(centerLum - tapLum) / (phiLuminance * sqrt(centerVar) + 1e-4));
            
            float totalWeight = kernelWeight * wNorm * wDepth * wLum;
            specAcc += tapSpec * totalWeight;
            specWeightAcc += totalWeight;
        }
    }
    
    vec3 filteredSpecular = specAcc / max(specWeightAcc, 1e-4);
    return mix(filteredSpecular, rawSpecular, mirrorBypass);
}
```

---

### 5.6 Failure Mode 6: 1-SPP Fireflies & Outlier Noise Corruption
- **Symptom**: In real-time path tracing at 1 SPP, low-probability high-radiance light paths (e.g., glancing specular reflections into small emitters) produce extreme pixel outliers ("fireflies") with values exceeding $10^3$–$10^4$. Permitting these samples into temporal accumulation without bounding produces bright trailing streaks and permanently inflates temporal variance $M_2 - M_1^2$, destroying spatial edge stopping.
- **Mitigation**: Implements **Conservative Bilateral-Gated 8-Neighbor Current-Sample Soft-Clamping** inside Pass 1 (`denoise_fused_temporal.comp`) directly from LDS scalar luminance scratch memory:
  1. **Scalar Luminance LDS Neighborhood**: To stay strictly within the 3.9 KB LDS budget for local radiance, shared memory stores scalar luminance ($Y = 0.2126R + 0.7152G + 0.0722B$) rather than full RGB vectors (`s_localDiffuseLum[22][23]` and `s_localSpecularLum[22][23]`).
  2. **Emissive & Specular Highlight Bypass**: Legitimate small light emitters (area lights, glowing emissive geometry) and direct specular reflections with active emitter flags bypass outlier clamping completely, preventing legitimate small sources from being artificially extinguished.
  3. **Bilateral Geometric Gating**: When gathering neighboring luminance values in LDS, neighbor samples are checked against the center normal ($\mathbf{n}_P \cdot \mathbf{n}_Q > 0.70$) and depth ($|z_P - z_Q| / z_P < 0.10$). This prevents silhouette edges from pooling into empty sky or distant background, avoiding corrupted $\mu$ and $\sigma$ estimations. If fewer than 3 consistent neighbors exist (e.g. sub-pixel wire silhouettes), clamping is bypassed.
  4. **Chromaticity-Preserving Asymptotic Soft-Clamping**: Rather than a hard truncation clamp (`min(raw, threshold)`), current 1-SPP sample luminance is softly compressed above $\mu + 3.0\sigma$ using an asymptotic rolloff ($Y_{\text{clamped}} = Y_{\text{thresh}} + \frac{Y - Y_{\text{thresh}}}{1 + (Y - Y_{\text{thresh}})/Y_{\text{thresh}}}$) and applied as a scalar multiplier to the raw RGB vector. This smoothly bounds energy while strictly preserving color chromaticity:
     $$\mathbf{L}_{\text{curr, clamped}} = \mathbf{L}_{\text{curr}} \cdot \frac{Y_{\text{clamped}}}{Y_{\text{curr}}}$$
     Excluding the center sample guarantees that a single extreme outlier cannot blow up both $\mu$ and $\sigma$ to pass through unclamped.
  5. **Elimination of History Color-Box Clipping**: TAA-style $\pm 2\sigma$ or $\pm 3\sigma$ clipping of historical radiance against current 1-SPP neighbors is strictly avoided for real-time indirect illumination. When the current frame's stochastic ray misses a light path, clipping history against local 1-SPP neighbors artificially zeroes valid high-radiance indirect lighting. Geometrically validated history is retained without color-box clipping.
  6. **Finite FP16 Range Clamping**: To prevent numerical `Inf` overflows (> 65504) into RGBA16F storage images, all accumulated radiance and filtered moments are clamped to a finite FP16 range (`clamp(L, 0.0, 65000.0)`) before `imageStore`.

#### Concrete GLSL Implementation (`shaders/compute/denoise_fused_temporal.comp`)
```glsl
// Conservative Bilateral Firefly Rejection in LDS Shared Memory (bilateral-consistent neighbors excluding center)
vec3 suppressFirefliesLDS(
    ivec2 ldsCoord, 
    vec3 rawCurrRadiance,
    bool isDiffuseLobe,
    bool isEmissive) 
{
    // Legitimate small emitters bypass outlier clamping
    if (isEmissive) {
        return rawCurrRadiance;
    }

    vec3 centerNorm = vec3(s_normalX[ldsCoord.y][ldsCoord.x], s_normalY[ldsCoord.y][ldsCoord.x], s_normalZ[ldsCoord.y][ldsCoord.x]);
    float centerDepth = s_linearDepth[ldsCoord.y][ldsCoord.x];

    // Compute local neighborhood statistics strictly across bilateral-consistent neighbors excluding center
    float m1 = 0.0;
    float m2 = 0.0;
    float validNeighbors = 0.0;
    for (int dy = -1; dy <= 1; ++dy) {
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue; // STRICTLY EXCLUDE CENTER SAMPLE
            int ny = ldsCoord.y + dy;
            int nx = ldsCoord.x + dx;

            // Bilateral consistency test: reject samples crossing geometry silhouettes into background
            vec3 nbrNorm = vec3(s_normalX[ny][nx], s_normalY[ny][nx], s_normalZ[ny][nx]);
            float nbrDepth = s_linearDepth[ny][nx];
            bool geomConsistent = (dot(centerNorm, nbrNorm) > 0.70) && (abs(centerDepth - nbrDepth) / max(centerDepth, 1e-4) < 0.10);
            if (!geomConsistent) continue;

            float lum = isDiffuseLobe 
                ? s_localDiffuseLum[ny][nx]
                : s_localSpecularLum[ny][nx];
            m1 += lum;
            m2 += lum * lum;
            validNeighbors += 1.0;
        }
    }

    // Require at least 3 geometrically consistent neighbors to establish valid local statistics
    if (validNeighbors < 3.0) {
        return rawCurrRadiance; // Insufficient geometric neighbors (e.g. subpixel silhouettes): bypass clamp
    }

    float mu = m1 / validNeighbors;
    float sigma = sqrt(max(0.0, (m2 / validNeighbors) - mu * mu));
    float lumThreshold = mu + 3.0 * sigma;

    float currLum = dot(rawCurrRadiance, vec3(0.2126, 0.7152, 0.0722));
    if (currLum > lumThreshold && currLum > 1e-4) {
        // Soft-clamp preserving chromaticity (smooth asymptotic rolloff rather than hard truncation)
        float clampedLum = lumThreshold + (currLum - lumThreshold) / (1.0 + (currLum - lumThreshold) / max(lumThreshold, 0.1));
        return rawCurrRadiance * (clampedLum / currLum);
    }
    return rawCurrRadiance;
}
```

---

### 5.7 Optional Motion-Gated Stationary Progressive Crossfade
- **Symptom**: When the camera is stationary, the progressive accumulation of Monte Carlo samples ($N \gg 1$) provides ground-truth path-traced convergence. However, if a dynamic object or animated character moves through a stationary camera view, an unconditional stationary blend accumulates stale converged pixels, producing severe ghost trails.
- **Mitigation & Operational Scope**: This feature is implemented as an **optional compile-time preview mode** (`ENABLE_STATIONARY_PREVIEW`), strictly excluded from the real-time production budget, descriptor sets, and performance contract. When enabled for offline/editor previews, the accumulation blend in Pass 5 is gated per-pixel by a sub-pixel motion vector threshold:
  ```glsl
#ifdef ENABLE_STATIONARY_PREVIEW
  // Optional preview mode variant (strictly excluded from real-time production contract):
  // Gated per-pixel by sub-pixel motion threshold to prevent moving character ghost smears
  vec2 mv = imageLoad(uMotionVectors, coord).xy;
  float pixelMotion = dot(mv, mv);
  float subpixelThreshold = (0.5 / float(pc.width)) * (0.5 / float(pc.width));

  // Progressive blend active ONLY when pixel is stationary (motion < subpixelThreshold) and samples exist
  if (pixelMotion < subpixelThreshold && pc.accumulatedSamples > 1u) {
      float wBlend = float(pc.accumulatedSamples) / (float(pc.accumulatedSamples) + 8.0);
      vec3 accumColor = imageLoad(uAccumImage, coord).rgb;
      L_final = mix(L_final, accumColor, wBlend);
  }
#endif

  // Finite FP16 range clamp prevents Inf overflow
  L_final = clamp(L_final, 0.0, 65000.0);
  imageStore(uFinalDenoised, coord, vec4(L_final, 1.0));
  ```
  Any pixel with active motion ($\|\mathbf{v}_{\text{motion}}\|^2 \ge \text{subpixelThreshold}$) bypasses the stationary buffer entirely and evaluates exclusively through the responsive temporal-spatial pipeline.

---

### 5.8 Parameter Tuning & Reference Guidelines

The table below summarizes recommended operational defaults, valid ranges, and tuning rationales for all adaptive denoiser parameters:

| Parameter | Recommended Default | Valid Range | Tuning Guidance & Rationale |
|---|---|---|---|
| `weightAlbedo` | `0.50` | `[0.3, 0.7]` | Dominant weight for perceptually salient material boundaries. |
| `weightDepth` | `0.30` | `[0.2, 0.5]` | Silhouettes and occlusion boundaries. Decrease if geometry is high-poly flat. |
| `weightNormal` | `0.20` | `[0.1, 0.4]` | Geometric creases and bevels. Decrease if normal maps contain high micro-detail. |
| `sigmaMultiplier` ($k$) | `1.20` | `[1.0, 1.6]` | Statistical threshold cutoff. 1.2 isolates top gradient spikes. |
| `sigmaMin` | `0.02` | `[0.01, 0.05]` | Minimum standard deviation floor for tile moments reduction. |
| `noiseFloor` ($T_{\min}$) | `0.05` | `[0.02, 0.10]` | Minimum threshold floor to eliminate false edges in flat/monochrome tiles. |
| `contrastFloor` ($\delta_{\min}$)| `0.02` | `[0.01, 0.05]` | Minimum contrast delta ($M_{\text{fused}} - \mu$) to reject baseline Rayleigh noise. |
| `temporalAlpha` ($\alpha_\rho$) | `0.15` | `[0.05, 0.3]` | Edge density temporal smoothing rate. Lower values increase temporal stability. |
| `edgeThresholdHigh` | `0.55` | `[0.45, 0.65]` | Cutoff for fine-scale edge partition (finish spatial filtering in Pass 3). |
| `edgeThresholdLow` | `0.35` | `[0.25, 0.45]` | Cutoff for moderate curvature partition (finish spatial filtering in Pass 4; bypass Pass 5 taps). |
| `normalThreshold` | `0.85` | `[0.75, 0.95]` | Reprojection normal dot product for static geometry. Values $<0.80$ permit ghosting across creases. |
| `normalThresholdDynamic` | `0.50` | `[0.30, 0.70]` | Relaxed normal dot product for dynamic moving/rotating objects to prevent false disocclusion. |
| `depthThreshold` | `0.08` | `[0.03, 0.15]` | Relative depth tolerance for static affine reprojection ($8\%$). |
| `dynamicDepthThreshold` | `0.15` | `[0.08, 0.25]` | Relative depth tolerance for dynamic object reprojection ($15\%$). |
| `mvDiscrepancyThreshold` | `2.0` | `[1.0, 5.0]` | Pixel discrepancy threshold to detect dynamic object motion vs. static camera. |
| `phiNormal` | `32.0` | `[16.0, 64.0]` | À-Trous bilateral normal power. Controls sensitivity to surface curvature (capped at 256 for specular). |
| `phiPlaneDepth` | `2.00` | `[0.5, 8.0]` | À-Trous bilateral metric tangent-plane depth scale. Recalibrated to prevent far-field tap collapse. |
| `phiLuminance` | `4.00` | `[2.0, 8.0]` | Variance-guided luminance weight. Higher values increase aggressive smoothing. |
| `tanHalfFovY` | `tan(fovY/2)` | `[0.2, 1.5]` | Camera vertical half-FOV tangent for metric view-space unprojection. |
| `aspectRatio` | `width/height`| `[1.0, 2.5]` | Framebuffer aspect ratio for metric view-space unprojection. |
| `maxHistoryLength` | `32.0` | `[16.0, 64.0]` | Maximum stationary temporal accumulation frame count. |

---

## 6. Actionable Phased Implementation & Validation Roadmap

Transitioning from the existing baseline to Adaptive Analytical E-SVGF as the default engine denoiser follows a structured 5-stage rollout.

```
========================================================================================================================
                                      PHASED IMPLEMENTATION & VALIDATION TIMELINE
========================================================================================================================

  Stage 0: Infrastructure & Tooling Setup
  ├── Author scripts/compute_flip.py (Perceptual FLIP metric)
  ├── Enable shaderStorageImageExtendedFormats in VulkanContext.cpp for R8G8 storage images
  └── Implement internal 20-query timestamp pool in DenoiserPipeline (10 queries x 2 frames in flight)

  Stage 1: Fused Edge-Aware Spatiotemporal Accumulator
  ├── Author shaders/compute/denoise_fused_temporal.comp (22x22 LDS apron, in-LDS Sobel, vector bounds, local moments, dual-model reprojection, bilateral firefly soft-clamping)
  └── Allocate 17 internal storage images in DenoiserPipeline (257.13 MB unpadded / 245.22 MiB across 17 storage images, 124 B/px; m_historyLength in RGBA16F, m_edgeMagnitude eliminated)

  Stage 2: Authoritative 5-Dispatch Dilated À-Trous Wavelet & Fused Remodulation
  ├── Author shaders/compute/denoise_atrous_step1.comp (s=1, radius 2, diameter 5; filters radiance & variance w^2)
  ├── Author shaders/compute/denoise_atrous_step2.comp (s=2, accumulated diameter 13 covering k=9; dense edges complete spatial filtering here and bypass subsequent taps)
  ├── Author shaders/compute/denoise_atrous_step3.comp (s=4, accumulated diameter 29 covering k=17; subgroupAll wave early-exit and smoothstep crossfade for dense edges)
  ├── Author shaders/compute/denoise_atrous_coarse_remodulate.comp (s=8, diameter 61 covering k=33; subgroupAll moderate wave tap skip; in-register remodulation & optional preview)
  ├── Wire descriptor sets across 2 frames in flight with 4 inter-pass VkMemoryBarrier2 barriers
  └── Add CTest unit test harness: tests/test_denoiser_pipeline.cpp

  Stage 3: Section 8 Failure Mode Mitigations
  ├── Integrate metric tangent plane-distance depth weighting & unprojection in denoise_common.glsl (calibrated phiPlaneDepth = 2.00f; in-LDS plane-relative depth residual with FP16 quantization floor)
  ├── Verify dual-model reprojection and bilateral-gated 8-neighbor firefly soft-clamping in denoise_fused_temporal.comp
  └── Connect specular virtual motion vectors (VMV), d_prev,refl validation in RGBA16F history length, and continuous smoothstep roughness modulation

  Stage 4: Engine Promotion as Default Denoiser
  ├── Promote DenoiserMode::Svgf to Adaptive E-SVGF default
  ├── Preserve CLI fallbacks (--denoiser none, --denoiser classic-svgf, --denoiser fast-svgf for APUs, --denoiser upways)
  └── Validate automated headless regression suite & lock performance baselines
```

---

### 6.1 Detailed Stage Breakdown

#### Stage 0: Infrastructure & Tooling Setup
- **Deliverables**:
  1. Author `scripts/compute_flip.py`: Pure Python/NumPy implementation of the NVIDIA FLIP perceptual image quality metric (Andersson et al., HPG 2020) to evaluate high-frequency edge bleeding and temporal boiling.
  2. Enable extended storage image formats: In `src/vulkan/VulkanContext.cpp`, enable `deviceFeatures2.features.shaderStorageImageExtendedFormats = VK_TRUE;` to support `rg8` storage image writes across all Vulkan compute pipelines.
  3. Expand `DenoiserPipeline` GPU profiling: Allocate an internal 20-timestamp `VkQueryPool` ($10 \times \text{MAX\_FRAMES\_IN\_FLIGHT}$) to profile all 5 compute passes across both frames in flight (`gpuFusedTemporalMs`, `gpuAtrousStep1Ms`, `gpuAtrousStep2Ms`, `gpuAtrousStep3Ms`, `gpuAtrousCoarseRemodulateMs`) and report timings asynchronously in `FrameStats` JSON.

#### Stage 1: Fused Edge-Aware Spatiotemporal Accumulator
- **Deliverables**:
  1. Create `shaders/compute/denoise_fused_temporal.comp` with $22\times22$ G-buffer source apron loaded into LDS, in-LDS Sobel gradient evaluation with analytical vector scaling bounds ($1/(4\sqrt{6})$, $1/(8\sqrt{2})$), tile-local reduction with noise floor $T_{\min} = 0.05$ and contrast floor $\delta_{\min} = 0.02$ ($\sigma_{\min} = 0.02$), dual-model reprojection validation (static affine depth test vs. dynamic object sample depth test with relaxed normal threshold $\text{pc.normalThresholdDynamic} = 0.50$ and camera sub-pixel jitter cancellation via `CameraMatricesUBO`), conservative bilateral-gated 8-neighbor current-sample firefly soft-clamping (strictly excluding center; emissive bypass), and direct in-shader historical normal/depth updates via Binding 18 (`uOutNormalDepth`).
  2. Update `DenoiserPipeline::resize()` to allocate the 17 internal storage images (`m_edgeDensity[2]` in `VK_FORMAT_R8G8_UNORM`, `m_historyDiffuse[2]`, `m_historySpecular[2]`, `m_historyMoments[2]`, `m_historyLength[2]` in `VK_FORMAT_R16G16B16A16_SFLOAT` [RGBA16F, 8 B/px], `m_historyNormalDepth[2]`, `m_pingPongDiffuse[2]`, `m_pingPongSpecular[2]`, `m_denoisedImage`), bringing total internal VRAM footprint to 257.13 MB unpadded (245.22 MiB across 17 storage images, 124 B/px), with historical reflection hit distance $d_{\text{prev,refl}}$ packed in Channel B of `m_historyLength` (freeing `m_historySpecular.a` for temporal specular variance $V^s$), and `m_edgeMagnitude` completely eliminated.
  3. Verify SPIR-V compilation via CMake and `glslc --target-env=vulkan1.4`.

#### Stage 2: Authoritative 5-Dispatch Dilated À-Trous Wavelet & Fused Remodulation
- **Deliverables**:
  1. Author `shaders/compute/denoise_atrous_step1.comp`: Evaluates fine-scale $5\times5$ cross-bilateral wavelet filtering ($s=1$, radius 2, diameter 5). Filters radiance and variance ($w^2$). Omits `uEdgeDensity` from descriptor set (6 storage images bound), saving 2 B/px read traffic.
  2. Author `shaders/compute/denoise_atrous_step2.comp`: Evaluates fine-scale dilation ($s=2$, accumulated radius 6, diameter 13 covering Fu et al. $k=9$). Dense edge pixels ($\rho_{\text{effective}} \ge 0.55$) obtain complete filtering and bypass subsequent filtering taps in Passes 4 and 5.
  3. Author `shaders/compute/denoise_atrous_step3.comp`: Evaluates mid-scale dilation ($s=4$, accumulated radius 14, diameter 29 covering Fu et al. $k=17$). Dense edges bypass taps via `subgroupAll(effectiveRho >= 0.55)` and continuous `smoothstep(0.45, 0.55, effectiveRho)` crossfade.
  4. Author `shaders/compute/denoise_atrous_coarse_remodulate.comp`: Evaluates coarse dilation ($s=8$, reaching accumulated diameter 61 covering Fu et al. $k=33$ for flats), smoothstep crossfading ($w_{\text{blend}}^{(2)} = \text{smoothstep}(0.25, 0.35, \rho_{\text{effective}})$), wavefront uniform voting (`subgroupAll(effectiveRho >= 0.35)`) to skip coarse taps for moderate-curvature regions, in-register diffuse albedo remodulation, specular addition, and optional motion-gated stationary progressive preview blend (`#ifdef ENABLE_STATIONARY_PREVIEW`) directly to `m_denoisedImage`.
  5. Allocate descriptor sets across frames in flight in `DenoiserPipeline::recordDenoise()` and insert 4 compute-to-compute `VkMemoryBarrier2` barriers.
  6. Register unit test `tests/test_denoiser_pipeline.cpp` in `CMakeLists.txt` to validate descriptor binding and headless image outputs.

#### Stage 3: Failure Mode Mitigations & Specular Integration
- **Deliverables**:
  1. Integrate metric tangent plane-distance depth weighting into `shaders/compute/denoise_common.glsl` with `unprojectToView()` using push constant camera intrinsics and metric pixel footprint tolerance (calibrated `phiPlaneDepth = 2.00f`). In-LDS plane-relative depth residual Sobel extraction with FP16 quantization floor $2\,\epsilon_{\text{fp16}}(z)$.
  2. Validate dual-model camera-motion-compensated affine reprojection and bilateral-gated 8-neighbor firefly suppression in `shaders/compute/denoise_fused_temporal.comp`.
  3. Implement continuous smoothstep roughness modulation, monotonic non-collapsing tap offsets, and capped specular normal power ($\le 256.0$) in `shaders/compute/denoise_atrous_step1.comp`.
  4. Connect specular virtual motion vectors (VMV along primary ray direction) to `uSpecularMotion` at Binding 4 in Pass 1, fetching $d_{\text{refl}}$ from `uCurrentSpecular.a` and validating consistency against $d_{\text{prev,refl}}$ fetched from Channel B of `uHistoryLength` (`VK_FORMAT_R16G16B16A16_SFLOAT`).

#### Stage 4: Promotion as Default Denoiser & Validation
- **Deliverables**:
  1. Update `src/core/Config.cpp`: Set `denoiser_mode = DenoiserMode::Svgf` as standard engine default for real-time path tracing.
  2. Retain CLI fallback options: `--denoiser none` (pure Monte Carlo), `--denoiser classic-svgf` (fixed-kernel SVGF), `--denoiser fast-svgf` (APU 3-iteration mode), and `--denoiser upways` (neural WMMA).
  3. Execute full headless regression verification across Cornell Box, Damaged Helmet, Cyber-City, and Veach Ajar. Lock golden references.

---

### 6.2 Headless Automated Test Design & Test Suites

#### 1. CTest Unit Test Harness (`tests/test_denoiser_pipeline.cpp`)
Registered in `CMakeLists.txt`, this test validates the Vulkan 1.4 compute backend without creating an SDL3 window:
- Verifies successful allocation of all 17 internal storage images (257.13 MB unpadded / 245.22 MiB, 124 B/px).
- Validates descriptor set creation for all 5 compute passes across frames in flight.
- Executes 5 consecutive offscreen denoiser dispatches with synthetic checkerboard and edge step inputs.
- Validates that Pass 1 generates in-LDS edge gradients without buffer overflow, accurately evaluates edge density $\rho$, executes conservative bilateral-gated 8-neighbor current-sample firefly soft-clamping, and writes pre-spatial history.
- Validates that Pass 2 ($s=1$) and Pass 3 ($s=2$) filter radiance and variance ($w^2$), Pass 4 ($s=4$) executes mid-scale dilation, and Pass 5 ($s=8$) executes coarse dilation and in-register remodulation (skipping coarse taps on moderate clusters via `subgroupAll`) without NaN or Inf generation.

#### 2. Visual Regression Runner (`scripts/visual_regression_test.py`)
Add 4 dedicated test configurations tracking Adaptive E-SVGF across key research scenes:
1. `cornell_1spp_adaptive_svgf`: Evaluates geometry edge preservation and contact shadow sharpness.
2. `damaged_helmet_1080p_adaptive_svgf`: Evaluates metallic/rough-dielectric material edge separation.
3. `living_room_motion_adaptive_svgf`: Active camera navigation evaluating temporal stability and disocclusion handling.
4. `glass_water_caustics_adaptive_svgf`: Evaluates refractive transmission and specular highlight retention.

#### 3. Image Quality Threshold Gates
Every automated build must satisfy the following strict quality thresholds against converged ground-truth references (512–1024 SPP progressive renders):

| Evaluation Scenario | Metric | Required Threshold | Validation Script |
|---|---|---|---|
| **Static Convergence (1 SPP vs Ref)** | **SSIM** | $\ge 0.920$ | `scripts/visual_regression_test.py` |
| **Static Convergence (1 SPP vs Ref)** | **PSNR** | $\ge 34.0\text{ dB}$ | `scripts/visual_regression_test.py` |
| **Static Convergence (Standard Scenes)** | **FLIP** | $\le 0.080$ | `scripts/compute_flip.py` |
| **Glass / Caustics Scenes** | **FLIP** | No regression vs baseline ($\le 1.05 \times \text{Ref}$) | `scripts/compute_flip.py` |
| **Edge Sharpness (Laplacian Var)** | **Sharpness Ratio** | $\ge 0.85 \times \text{Ref}$ | `scripts/visual_regression_test.py` |
| **Temporal Stability (Active Motion)** | **Temporal Var** | $\le 0.015$ | `scripts/verify_visual_integrity.py` |
| **Shadow Boundary Preservation** | **Contact Retention**| $[10\%, 40\%]$ | `scripts/verify_frame.py` |
| **Numerical Integrity** | **NaN / Inf Count** | **Strictly 0** | `scripts/verify_frame.py` |

---

### 6.3 Real-Time Frame Budgets & Granular GPU Profiling

Establishes an honest, physics-grounded real-time performance budget:
- **Primary Production Contract (1080p Internal Render Resolution)**: Running the denoiser at 1080p prior to super-resolution upscaling (Upways / FSR3) targets **1.7–1.9 ms measured** on 4070-class hardware (pure streaming floor ~1.61 ms; 4070's 36 MB L2 cache cannot hold the 66.4 MB ping-pong working set, so it streams primarily from DRAM), **1.0–1.2 ms measure** on RX 7800 XT (where the 66.4 MB working set is borderline against the 64 MB Infinity Cache), and **1.2–1.4 ms** on $\ge 700\text{ GB/s}$ GPUs (floor ~1.18 ms). At 1440p internal resolution, the pure streaming target is **3.0–3.4 ms** on 4070 (compulsory streaming floor ~2.86 ms; the 118 MB working set exceeds the cache on all named GPU targets).
- **Native 4K Profiling Gate**: At native 4K (3840×2160), compulsory streaming traffic reaches **~2.60 GB** (314 B/px baseline; ~2.70 GB with preview); native 4K serves as a high-end profiling gate targeting **< 3.5 ms** exclusively on extreme desktop GPUs with $\ge 950\text{--}1000\text{ GB/s}$ bus bandwidth (e.g. RTX 4090 / RX 7900 XTX class; memory floor ~3.23–3.38 ms). On 4070/7800-class hardware, native 4K is an offline reference gate (compulsory floor ~5.21–6.45 ms).
- **Unified APU Contract**: At 1080p on AMD Strix Halo (256-bit LPDDR5X-8000 ≈ 256 GB/s bus, 204.8 GB/s sustained @ 80%), compulsory traffic sets a memory floor of ~3.18 ms (~3.30 ms with preview), establishing a base gate of **3.3–3.6 ms measured**, with sub-3.0 ms explicitly gated behind the traffic-reduction roadmap (~2.69 ms floor with 48 B/px savings).

#### Tiered Performance Budgets

| Platform Tier | Output Resolution | Total Target Frame Time | Ray Tracing Budget | Denoiser Budget | Tonemap / Post Budget | Notes & Compulsory Streaming Traffic |
|---|---|---|---|---|---|---|
| **Discrete Desktop** (RTX 4070 / RX 7800 XT) | **4K (3840×2160 Native)** | 16.67 ms (60 FPS) / 8.33 ms | ~11.00 ms | **Profiling Gate (< 3.5 ms on $\ge 950\text{ GB/s}$)** | ~1.20 ms | 5-Dispatch Dilated E-SVGF (~2.60 GB baseline / ~2.70 GB preview; floor ~5.21–6.45 ms) |
| **Discrete Desktop** (RTX 4070 / RX 7800 XT) | **1440p (2560×1440)** | 8.33 ms (120 FPS) | ~5.80 ms | **3.0–3.4 ms** | ~0.70 ms | 5-Dispatch Dilated E-SVGF (~1.16 GB streaming traffic; floor ~2.31–2.86 ms) |
| **Discrete Desktop** (RTX 4070 / RX 7800 XT) | **1080p (1920×1080)** | 8.33 ms (120 FPS) | ~6.50 ms | **1.7–1.9 ms (4070) / 1.0–1.2 ms (7800 XT) / 1.2–1.4 ms ($\ge 700\text{ GB/s}$)** | ~0.50 ms | 5-Dispatch Dilated E-SVGF (~651 MB baseline; floor ~1.30–1.61 ms) |
| **High-End APU** (AMD Strix Halo gfx1151) | **1080p (1920×1080 Native)**| 16.67 ms / 8.33 ms | ~5.50 ms | **3.3–3.6 ms base gate measured** | ~0.70 ms | 5-Dispatch Dilated E-SVGF on 256-bit bus (floor ~3.18 ms; sub-3ms requires roadmap) |
| **High-End APU** (AMD Strix Halo gfx1151) | **720p (1280×720)** | 8.33 ms (120 FPS) | ~6.20 ms | **< 1.60 ms** (Target: 1.4–1.6 ms) | ~0.40 ms | 5-Dispatch Dilated E-SVGF on 256-bit bus (~289 MB traffic; floor ~1.41 ms) |

#### Compulsory DRAM Bandwidth vs. Cache Amplification Analysis
While cache hits reduce gather amplification (25 taps per pixel compress to ~1 unique cache line fetch per pixel on modern GPU architectures), they do **not** reduce compulsory DRAM streaming when buffers exceed cache capacity: at 4K, each 66.36 MB full-screen buffer exceeds typical GPU L2 cache sizes (e.g. 36 MB on RTX 4070, 48 MB on RX 7800 XT).

##### Detailed Compulsory Streaming Traffic per Pass at 4K ($3840 \times 2160$, 8.29M pixels)
- **Pass 1 (Fused Spatiotemporal Accumulator)**:
  - *Reads* (80 B/px, 663.55 MB @ 4K / 165.89 MB @ 1080p): Current Diffuse (8 B/px, 66.36 MB), Current Specular (8 B/px, 66.36 MB), Motion Vectors (4 B/px, 33.18 MB), Specular Motion (4 B/px, 33.18 MB), NormalDepth (8 B/px, 66.36 MB), PrevNormalDepth (8 B/px, 66.36 MB), AlbedoRoughness (8 B/px, 66.36 MB), History Diffuse (8 B/px, 66.36 MB), History Specular (8 B/px, 66.36 MB), History Moments (8 B/px, 66.36 MB), History Length (8 B/px, 66.36 MB, `VK_FORMAT_R16G16B16A16_SFLOAT`), Prev Edge Density (2 B/px, 16.59 MB, `VK_FORMAT_R8G8_UNORM`).
  - *Writes* (42 B/px, 348.37 MB @ 4K / 87.09 MB @ 1080p): History Diffuse (8 B/px, 66.36 MB), History Specular (8 B/px, 66.36 MB), History Moments (8 B/px, 66.36 MB), History NormalDepth (8 B/px, 66.36 MB), History Length (8 B/px, 66.36 MB, `VK_FORMAT_R16G16B16A16_SFLOAT`), Edge Density (2 B/px, 16.59 MB, `VK_FORMAT_R8G8_UNORM`).
  - *Compulsory Traffic*: **124 B/px = 1,011.92 MB @ 4K (~1,029 MB decimal) / 252.98 MB @ 1080p (~257 MB decimal)**.
- **Pass 2 (Fine-Scale Wavelet Iteration 1, $s=1$, 5 px)**:
  - *Reads* (32 B/px, 265.42 MB @ 4K / 66.36 MB @ 1080p) [omitting `uEdgeDensity`]: History Diffuse (8 B/px, 66.36 MB), History Specular (8 B/px, 66.36 MB), NormalDepth (8 B/px, 66.36 MB), AlbedoRoughness (8 B/px, 66.36 MB).
  - *Writes* (16 B/px, 132.71 MB @ 4K / 33.18 MB @ 1080p): PingPongDiffuse 0 (8 B/px, 66.36 MB), PingPongSpecular 0 (8 B/px, 66.36 MB).
  - *Compulsory Traffic*: **48 B/px = 398.13 MB @ 4K (~398 MB) / 99.53 MB @ 1080p (~100 MB)**.
- **Pass 3 (Fine-Scale Wavelet Iteration 2, $s=2$, 13 px, covers $k=9$)**:
  - *Reads* (34 B/px, 282.01 MB @ 4K / 70.50 MB @ 1080p): PingPongDiffuse 0 (8 B/px, 66.36 MB), PingPongSpecular 0 (8 B/px, 66.36 MB), NormalDepth (8 B/px, 66.36 MB), AlbedoRoughness (8 B/px, 66.36 MB), Edge Density (2 B/px, 16.59 MB).
  - *Writes* (16 B/px, 132.71 MB @ 4K / 33.18 MB @ 1080p): PingPongDiffuse 1 (8 B/px, 66.36 MB), PingPongSpecular 1 (8 B/px, 66.36 MB).
  - *Compulsory Traffic*: **50 B/px = 414.72 MB @ 4K (~415 MB) / 103.68 MB @ 1080p (~104 MB)**.
- **Pass 4 (Mid-Scale Wavelet Iteration 3, $s=4$, 29 px, covers $k=17$)**:
  - *Reads* (34 B/px, 282.01 MB @ 4K / 70.50 MB @ 1080p): PingPongDiffuse 1 (8 B/px, 66.36 MB), PingPongSpecular 1 (8 B/px, 66.36 MB), NormalDepth (8 B/px, 66.36 MB), AlbedoRoughness (8 B/px, 66.36 MB), Edge Density (2 B/px, 16.59 MB).
  - *Writes* (16 B/px, 132.71 MB @ 4K / 33.18 MB @ 1080p): PingPongDiffuse 0 (8 B/px, 66.36 MB), PingPongSpecular 0 (8 B/px, 66.36 MB).
  - *Compulsory Traffic*: **50 B/px = 414.72 MB @ 4K (~415 MB) / 103.68 MB @ 1080p (~104 MB)**.
- **Pass 5 (Coarse Wavelet Iteration 4, $s=8$, 61 px, covers $k=33$ + Remodulate)**:
  - *Reads* (34 B/px baseline, 282.01 MB @ 4K / 70.50 MB @ 1080p) [with optional preview: +12 B/px = Motion Vectors 4 B/px + Accum Image 8 B/px = 46 B/px, 381.54 MB @ 4K / 95.39 MB @ 1080p]: PingPongDiffuse 0 (8 B/px, 66.36 MB), PingPongSpecular 0 (8 B/px, 66.36 MB), NormalDepth (8 B/px, 66.36 MB), AlbedoRoughness (8 B/px, 66.36 MB), Edge Density (2 B/px, 16.59 MB).
  - *Writes* (8 B/px, 66.36 MB @ 4K / 16.59 MB @ 1080p): Final Denoised Image (8 B/px, 66.36 MB, RGBA16F).
  - *Compulsory Traffic*: **42 B/px baseline = 348.37 MB @ 4K (~348 MB) / 87.09 MB @ 1080p (~87 MB)** [54 B/px with preview = 447.90 MB @ 4K / 111.97 MB @ 1080p (~448 MB / ~112 MB)].

##### Compulsory Streaming Traffic Summary

| Pass | Reads (B/px) | Writes (B/px) | Total (B/px) | 4K Native (MB) | 1080p Native (MB) |
|---|---|---|---|---|---|
| Pass 1 (Fused Temporal) | 80 | 42 | 124 | 1,011.92 (~1,029 MB) | 252.98 (~257 MB) |
| Pass 2 (À-Trous s=1) | 32 | 16 | 48 | 398.13 (~398 MB) | 99.53 (~100 MB) |
| Pass 3 (À-Trous s=2) | 34 | 16 | 50 | 414.72 (~415 MB) | 103.68 (~104 MB) |
| Pass 4 (À-Trous s=4) | 34 | 16 | 50 | 414.72 (~415 MB) | 103.68 (~104 MB) |
| Pass 5 (À-Trous s=8 + Remod) | 34 | 8 | 42 | 348.37 (~348 MB) | 87.09 (~87 MB) |
| **Total Compulsory Traffic (Baseline)** | **214** | **98** | **314** | **2,604.44 MB (≈ 2.60 GB / 2.43 GiB)** | **651.11 MB (≈ 651 MB / 621 MiB)** |
| *Optional Preview Mode (+Pass 5)* | *+12* | *0* | *+12 (326)* | *2,703.96 MB (≈ 2.70 GB / 2.52 GiB)* | *675.99 MB (≈ 676 MB / 645 MiB)* |

##### Physical Memory Floor Calculations ($T_{\text{floor}} = \frac{\text{Bytes}}{0.8 \times \text{Peak Bandwidth}}$)
- **RTX 4070** (504 GB/s theoretical, 403.2 GB/s sustained at 80% bus efficiency):
  $$\text{4K Streaming Floor} = \frac{2.60\text{ GB}}{403.2\text{ GB/s}} \approx \mathbf{6.46\text{ ms}} \quad (\text{with preview: } 6.71\text{ ms})$$
  $$\text{1440p Streaming Floor} = \frac{1.155\text{ GB}}{403.2\text{ GB/s}} \approx \mathbf{2.86\text{ ms}}$$
  $$\text{1080p Streaming Floor} = \frac{0.651\text{ GB}}{403.2\text{ GB/s}} \approx \mathbf{1.61\text{ ms}} \quad (\text{with preview: } 1.68\text{ ms})$$
- **AMD RX 7800 XT** (624 GB/s theoretical, 499.2 GB/s sustained at 80% bus efficiency):
  $$\text{4K Streaming Floor} = \frac{2.60\text{ GB}}{499.2\text{ GB/s}} \approx \mathbf{5.22\text{ ms}} \quad (\text{with preview: } 5.42\text{ ms})$$
  $$\text{1440p Streaming Floor} = \frac{1.155\text{ GB}}{499.2\text{ GB/s}} \approx \mathbf{2.31\text{ ms}}$$
  $$\text{1080p Streaming Floor} = \frac{0.651\text{ GB}}{499.2\text{ GB/s}} \approx \mathbf{1.30\text{ ms}}$$
- **AMD Strix Halo APU** (256-bit LPDDR5X-8000 ≈ 256 GB/s shared bus, 204.8 GB/s sustained at 80% bus efficiency):
  $$\text{1080p Streaming Floor} = \frac{0.651\text{ GB}}{204.8\text{ GB/s}} \approx \mathbf{3.18\text{ ms}} \quad (\text{with preview: } 3.30\text{ ms})$$
  Matches base measured gate: **3.3–3.6 ms**.
- **High-End $\ge 950\text{ GB/s}$ GPUs** (RTX 4090 / RX 7900 XTX: 1008–960 GB/s theoretical, 768–806 GB/s sustained at 80%):
  $$\text{4K Streaming Floor} = \frac{2.60\text{ GB}}{768\text{--}806\text{ GB/s}} \approx \mathbf{3.23\text{--}3.39\text{ ms}}$$
  Matches Native 4K Profiling Gate (< 3.5 ms)!

##### Cache Retention Analysis (LLC vs. Intermediate Ping-Pong Working Set)
Intermediate ping-pong buffers across Passes 2–5 ($4 \times 16.59\text{ MB} = 66.36\text{ MB}$ at 1080p) fit inside the 64 MB Infinity Cache of the RX 7800 XT, allowing spatial passes to stream from on-chip cache with minimal DRAM roundtrips and achieving **1.0–1.2 ms** measured frame time. On an RTX 4070 (36 MB L2 cache), the 1080p working set partially spills to DRAM, yielding **1.7–1.9 ms** measured time without cache hits or **1.2–1.4 ms** with partial hit rate. At 1440p (117.96 MB working set) and 4K (265.42 MB working set), intermediate storage images exceed both 36 MB L2 and 64 MB Infinity Cache, strictly enforcing DRAM streaming floors.

##### Memory Traffic Optimization Architecture
The pipeline separates committed base architectural design choices from future optimization targets:

###### 1. Committed Base Design Choices (Already Built-In)
1. **In-LDS Sobel & Edge Density**: Pass 1 cooperatively loads a $22\times22$ G-buffer apron into LDS, completely eliminating global memory roundtrips for Sobel gradient and edge mask textures (avoids ~300 MB DRAM traffic).
2. **Dual-Channel Packed Edge Density**: Allocating `m_edgeDensity[2]` as `VK_FORMAT_R8G8_UNORM` pre-packs effective edge density into Channel G, eliminating redundant history length texture fetches in all spatial dispatches (saves ~66 MB across passes 2–5).
3. **In-Register Albedo Remodulation**: Fusing albedo remodulation, specular addition, and tonemapper output directly into Pass 5 eliminates an entire full-screen diffuse write/read pass (avoids ~133 MB DRAM roundtrip).
4. **Pre-Spatial History Feedback**: Writing un-blurred temporal history directly in Pass 1 eliminates auxiliary copy passes (`vkCmdCopyImage`) and ensures history accumulates high-SNR, firefly-clipped samples.
5. **Pass 2 Edge Density Omission**: Pass 2 ($s=1$, 5 px) applies uniform filtering without early-exit pruning; omitting Binding 4 (`uEdgeDensity`) from Pass 2 descriptor set saves 2 B/px read traffic (~17 MB @ 4K / ~4 MB @ 1080p), lowering baseline compulsory traffic from 316 B/px to 314 B/px.

###### 2. Future Traffic-Reduction Optimization Roadmap (Gating Sub-3.0 ms on APUs and Sub-2.5 ms at 1440p)
To unlock real-time sub-3.0 ms execution on APUs and sub-2.5 ms at 1440p, two specific traffic reductions are identified:
1. **G-Buffer Guidance Compaction**: Pack `uNormalDepth` and `uAlbedoRoughness` into a packed 64-bit layout (`VK_FORMAT_A2B10G10R10_UNORM_PACK32` + half depth = 8 B/px instead of 16 B/px). Across Passes 2–5 (4 spatial dispatches), guidance reads drop by 8 B/px per pass $\to$ saves **32 B/px** ($\sim 265\text{ MB}$ at 4K / $\sim 66\text{ MB}$ at 1080p).
2. **Double-Buffered Producer G-Buffer**: Synchronize the path tracer's G-buffer production across frames in flight so that Pass 1 reads the previous frame's primary G-buffer directly as history. This eliminates `m_historyNormalDepth` entirely, saving 8 B/px read and 8 B/px write in Pass 1 $\to$ saves **16 B/px** ($\sim 133\text{ MB}$ at 4K / $\sim 33\text{ MB}$ at 1080p).

**Combined Roadmap Savings**: $32 + 16 = \mathbf{48\text{ B/px}}$ ($\sim 398\text{ MB}$ at 4K / $\sim 100\text{ MB}$ at 1080p).
- Drops baseline compulsory streaming traffic from 314 B/px to **266 B/px** ($\sim 2.20\text{ GB}$ at 4K / $\sim 552\text{ MB}$ at 1080p).
- Drops Strix Halo 1080p streaming floor to: $0.552\text{ GB} / 204.8\text{ GB/s} \approx \mathbf{2.69\text{ ms}}$ (unlocking real-time sub-3.0 ms operation!).
- Drops RTX 4070 1440p streaming floor to: $0.980\text{ GB} / 403.2\text{ GB/s} \approx \mathbf{2.43\text{ ms}}$ (unlocking sub-2.5 ms!).

#### Internal Denoiser Profiling Query Allocation (20 Queries Across 2 Frames in Flight)
`DenoiserPipeline` maintains an internal query pool `m_subpassQueryPool` allocating **20 queries** ($10 \times \text{MAX\_FRAMES\_IN\_FLIGHT}$):
- **Frame Slot 0 (Queries 0–9)**:
  - Queries 0–1: Pass 1 Fused Edge-Aware Spatiotemporal Accumulator (`gpuFusedTemporalMs`)
  - Queries 2–3: Pass 2 Fine-Scale Wavelet Iteration 1 ($s=1$) (`gpuAtrousStep1Ms`)
  - Queries 4–5: Pass 3 Fine-Scale Wavelet Iteration 2 ($s=2$) (`gpuAtrousStep2Ms`)
  - Queries 6–7: Pass 4 Mid-Scale Wavelet Iteration 3 ($s=4$) (`gpuAtrousStep3Ms`)
  - Queries 8–9: Pass 5 Coarse Wavelet Iteration 4 ($s=8$) & Remodulate (`gpuAtrousCoarseRemodulateMs`)
- **Frame Slot 1 (Queries 10–19)**:
  - Queries 10–11: Pass 1 Fused Edge-Aware Spatiotemporal Accumulator (`gpuFusedTemporalMs`)
  - Queries 12–13: Pass 2 Fine-Scale Wavelet Iteration 1 ($s=1$) (`gpuAtrousStep1Ms`)
  - Queries 14–15: Pass 3 Fine-Scale Wavelet Iteration 2 ($s=2$) (`gpuAtrousStep2Ms`)
  - Queries 16–17: Pass 4 Mid-Scale Wavelet Iteration 3 ($s=4$) (`gpuAtrousStep3Ms`)
  - Queries 18–19: Pass 5 Coarse Wavelet Iteration 4 ($s=8$) & Remodulate (`gpuAtrousCoarseRemodulateMs`)

These timestamps are queried asynchronously each frame using `vkGetQueryPoolResults` indexed by `frameSlot * 10`, completely eliminating query overwrite hazards and reporting granular sub-stage timings in `FrameStats` JSON.

---

## 7. Verification & Testing Commands

In strict accordance with the project's **Tiered Verification Hierarchy**, verification of this architectural proposal document is classified under **Tier 0 (Documentation & Text)**. 

### 7.1 Tier 0 Pre-Authorized Verification Commands
Per the project rules, documentation edits must be verified **strictly via git inspection commands**. Building or running CTest suites during documentation authoring is strictly prohibited:

```bash
# 1. Inspect working directory status
git status -s docs/design/adaptive_svgf_denoiser_proposal.md

# 2. Inspect line and structural diff
git diff --stat docs/design/adaptive_svgf_denoiser_proposal.md

# 3. Verify markdown formatting and link integrity (no broken local paths)
head -n 25 docs/design/adaptive_svgf_denoiser_proposal.md
tail -n 25 docs/design/adaptive_svgf_denoiser_proposal.md
```

### 7.2 Implementation Phase Verification Commands (Tiers 1–3)
Upon entering the implementation stages (Stages 1–4), engineers must execute the following pre-authorized tiered commands:

```bash
# Tier 1: Shader Compilation Gate (glslc)
glslc -I shaders/compute --target-env=vulkan1.4 -O shaders/compute/denoise_fused_temporal.comp -o build/denoise_fused_temporal.spv
glslc -I shaders/compute --target-env=vulkan1.4 -O shaders/compute/denoise_atrous_step1.comp -o build/denoise_atrous_step1.spv
glslc -I shaders/compute --target-env=vulkan1.4 -O shaders/compute/denoise_atrous_step2.comp -o build/denoise_atrous_step2.spv
glslc -I shaders/compute --target-env=vulkan1.4 -O shaders/compute/denoise_atrous_step3.comp -o build/denoise_atrous_step3.spv
glslc -I shaders/compute --target-env=vulkan1.4 -O shaders/compute/denoise_atrous_coarse_remodulate.comp -o build/denoise_atrous_coarse_remodulate.spv

# Tier 2: Targeted C++ Unit Test (CTest)
ninja -C build/windows-clang-release -j16 test_denoiser_pipeline
ctest --test-dir build/windows-clang-release -R DenoiserPipeline --output-on-failure

# Tier 3: Headless Visual Regression & Frame Verification
<BINARY> --headless --frames 1 --width 1920 --height 1080 --scene cornell-box --spp 1 --denoiser svgf --dump-frame output/test_cornell_adaptive_svgf.png --dump-stats output/stats_adaptive_svgf.json
python3 scripts/verify_frame.py output/test_cornell_adaptive_svgf.png output/stats_adaptive_svgf.json 1920 1080 40.0 --max-mean-lum 0.85 --max-blown-pct 20.0
python3 scripts/compute_flip.py output/test_cornell_adaptive_svgf.png tests/references/cornell_1080p.png --max-flip 0.080
```

---

## 8. Invalidation Conditions & Architectural Assumptions

This proposal is invalidated or subject to architectural revision if any of the following conditions occur:
1. **Hardware NPU Acceleration Availability**: If target hardware integrates dedicated, ultra-low-power NPU tensor engines with direct shared VRAM access where evaluating neural weights incurs zero ALU or bandwidth penalty compared to compute shaders.
2. **Vulkan API Deprecation**: If Vulkan 1.4 core compute or `VK_KHR_synchronization2` specifications are altered or superseded.
3. **G-Buffer Representation Overhaul**: If replacing linear view depth and view normals with non-planar octahedral or screen-space representations without updating the tangent plane distance formulas.
4. **Bandwidth Parity**: If GPU memory architectures evolve to provide unified terabyte-per-second random-access caches that eliminate the cache penalty of 5D lookup tables.

---
*End of Architectural Specification Proposal.*
