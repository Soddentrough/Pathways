# Neural Reconstruction & Spatiotemporal Super-Resolution in Pathways

## 1. Executive Summary & Problem Formulation

In real-time path tracing at low sample counts (1–2 Samples Per Pixel, or SPP), the primary visual artifact of Monte Carlo integration is **variance**. 
- **Offline Path Tracers**: Unbiased estimators permit isolated, extreme energy spikes ($\frac{f_r L_i}{\text{pdf}} \gg 1$) known as **fireflies**. Given thousands of samples ($4096+\text{ spp}$), these spikes converge to ground truth.
- **Real-Time Heuristic Denoisers (SVGF, BMFR, NRD)**: Bound by an 8–16 ms frame budget, real-time engines cannot afford unbiased convergence. To suppress fireflies, they clamp luminance outliers, blur radiance across spatial wavelet kernels (e.g. À-Trous), and accumulate frames over time using Exponential Moving Averages (EMA).

```
[1-SPP Raw Rays] ───► [Outlier Clamping] ───► [Spatial Wavelet Blur] ───► [Temporal EMA (30-60 frames)]
(Sharp Fireflies)       (Capped Energy)          (Blurred Discs)            (Morphing / Boiling Patches)
```

### The Fundamental Signal Trade-Off
Heuristic filtering does not eliminate variance; it **transforms high-frequency spatial/temporal noise into low-frequency spatio-temporal artifacts**:
1. Spatial filters blur single-pixel fireflies into diffuse discs.
2. Temporal accumulation stretches these discs across tens of frames.
3. Camera motion, disocclusion, and lighting changes force temporal history invalidation.
4. As history resets and spatial blur radiuses dynamically modulate with local variance ($\sigma^2$), the energy clusters crawl, pulse, and smear across the screen.

The human visual system perceives sharp, flickering per-frame dots as **fireflies**, but perceives morphing, low-frequency blurred energy clusters as **"boiling"** or **"swirling"**.

**The Solution in Pathways**: Rather than passing noisy output into a heuristic denoiser followed by an independent upscaler (which magnifies boiling artifacts), Pathways implements a **unified Neural Reconstruction and Super-Resolution** subsystem. A custom, lightweight recurrent convolutional network maps raw 1080p Monte Carlo samples, geometric buffers, and internal path tracer state directly to clean, stable 4K frames.

---

## 2. Architectural Independence & Open Vulkan 1.4 Standards

Pathways avoids proprietary, vendor-locked libraries (such as NVIDIA DLSS or closed Intel XeSS binaries) in favor of **100% open, vendor-neutral Vulkan 1.4 compute primitives**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Inference Pipeline Stack                          │
└─────────────────────────────────────────────────────────────────────────────┘
  PyTorch + ROCm 10 (Offline Training & Weight Export)
           │
           ▼ (Export Quantized Weights & Biases)
  Custom Vulkan 1.4 Compute Pipeline (`shaders/compute/neural_reconstruct.comp`)
           │
           ├── VK_KHR_cooperative_matrix (Wave32 WMMA / Matrix Cores)
           ├── VK_KHR_shader_float16_int8 (Native FP16 Operations)
           └── VK_KHR_shader_subgroup_extended_types (Subgroup Shuffle/Quad)
           │
           ▼
  AMD RDNA 4 (gfx1201) / Intel (XMX) / NVIDIA (Tensor Cores)
```

### 2.1 Hardware Acceleration via `VK_KHR_cooperative_matrix`
- **Cross-Vendor Portability**: Operates across AMD (WMMA on RDNA 3/4), NVIDIA (Tensor Cores), and Intel (XMX).
- **Subgroup Configuration**: Pathways targets AMD RDNA 4 (`gfx1201`, Wave32) using `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` (`requiredSubgroupSize = 32`), matching the hardware's native `v_wmma_f32_16x16x16_f16` matrix instructions.
- **Zero External Runtime Dependencies**: The neural network is executed directly as 3–5 dispatch passes of standard SPIR-V compute shaders. No third-party dynamic link libraries (`.so` / `.dll`), black boxes, or vendor telemetry are required.

---

## 3. Multi-Device Architecture: Scaling from 2 to $N$ Computational Devices

Pathways is designed for multi-GPU workstations (such as Dual AMD Radeon AI PRO R9700 GPUs) and multi-node compute clusters. The scaling problem is bifurcated into **offline training** and **real-time runtime execution**.

### 3.1 Multi-GPU Model Training (PyTorch + ROCm 10)
Real-time neural reconstructors are deliberately compact (**5M to 25M parameters**, ~20–100 MB). Consequently, training is not constrained by model memory, but by **data throughput**:

```
                       [Training Dataset / Sequence Generator]
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
   [GPU 0: Worker 1]                                 [GPU 1: Worker 2]
 ├── Forward Pass (Frames t..t+12)                 ├── Forward Pass (Frames t..t+12)
 ├── Temporal BPTT Loss Computation                ├── Temporal BPTT Loss Computation
 └── Local Gradients Calculation                   └── Local Gradients Calculation
         │                                                 │
         └─────────────► [RCCL Ring AllReduce] ◄───────────┘
                                  │
                      (Synchronized Weight Update)
```

1. **DistributedDataParallel (DDP) via RCCL**:
   - Each GPU maintains an identical copy of the network.
   - Synchronizing gradients for a 15M-parameter FP16 model requires an `AllReduce` of only ~30 MB. Over PCIe Gen 4 x16 (or xGMI), this completes in **under 1 millisecond**, resulting in near $1.0\times$ linear training speedup per device.
2. **Sequential Batching with Truncated BPTT**:
   - Training requires consecutive video sequences (8–16 frames) with moving cameras, dynamic animations, and varying lighting.
   - Each GPU runs Truncated Backpropagation Through Time (TBPTT) across distinct rendered sequence batches simultaneously.

---

### 3.2 Real-Time Multi-GPU Runtime Architecture: Parallel Upscaling Paradigms

#### Why Naive Alternate Frame Rendering (AFR) Fails
Under AFR (GPU 0 renders frame $t$, GPU 1 renders frame $t+1$), frame $t+1$ depends on the recurrent neural latent state $H_t$ and temporal history from GPU 0. This creates an inter-GPU synchronization dependency that stalls the GPU pipelines and adds $N$ frames of input latency. Furthermore, temporal history buffers are typically 20–50 MB; ping-ponging them across PCIe every frame creates massive bus contention.

#### The Post-Processing Serialization Bottleneck (Amdahl's Law)
In high-performance path tracing, ray tracing dispatch scales near linearly ($1.95\times$ on dual AMD Radeon AI PRO R9700s). For example, at 1 SPP 1080p, ray tracing takes only $\approx 2.10\text{ ms}$ on each card.
However, running a neural reconstructor or upscaler (e.g., FSR 4 or custom Wave32 WMMA autoencoder) from 1080p to 4K takes **$2.2 – 3.2\text{ ms}$**.

If post-processing runs **exclusively on GPU 0** while GPU 1 idles:
$$\text{Frame Time} = T_{\text{RT}} + T_{\text{Post}} = 2.10\text{ ms} + 2.80\text{ ms} = 4.90\text{ ms} \quad (\implies 204\text{ FPS})$$
Compared to single-GPU execution ($4.10\text{ ms} + 2.80\text{ ms} = 6.90\text{ ms} \implies 145\text{ FPS}$), the speedup is only **$1.41\times$**, despite doubling GPU compute resources.

To break through Amdahl's Law and restore multi-GPU scaling to **$\ge 1.92\times$**, the neural upscaling and reconstruction workloads must be **distributed concurrently across both GPUs**. Pathways develops two primary paradigms to achieve this:

---

#### Strategy A: Split-Viewport FSR on Merged 2 SPP (Guard-Band Apron Exchange)

In this architecture, both GPUs render the same frame simultaneously, divide the Monte Carlo sample paths, exchange viewport halves to construct a variance-reduced 2 SPP input, and execute upscaling concurrently on their respective screen regions.

```
                         [CPU Engine / Frame Orchestrator]
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
     [GPU 1: Secondary Device]                             [GPU 0: Primary Device]
 ├── Traces Ray Paths: 1 SPP (Seed B)                  ├── Traces Ray Paths: 1 SPP (Seed A)
 ├── G-Buffer & Motion Vectors (Full Frame)            ├── G-Buffer & Motion Vectors (Full Frame)
 ├── Exports Top Half (Y: 0..H/2) via DMA-BUF          ├── Exports Bottom Half (Y: H/2..H) via DMA-BUF
 │                      │                                  │
 │                      ├────────► [PCIe BAR Transfer] ───►│ (Half-Frame: ~8.3 MB, 0.03 ms)
 │◄─────────────────────┴───────── [PCIe BAR Transfer] ────┤
 │                                                         │
 ├── Accumulates Bottom Half: Seed A + Seed B (2 SPP)  ├── Accumulates Top Half: Seed A + Seed B (2 SPP)
 ├── Executes FSR on Bottom Half + Apron A             ├── Executes FSR on Top Half + Apron A
 │   (Output: 4K 10-Bit A2R10G10B10)                   │   (Output: 4K 10-Bit A2R10G10B10)
 ├── Blits Bottom Half via DMA-BUF ───────────────────►├── Composites UI & Swapchain Present
     (4K Bottom Half: ~16.58 MB, 0.05 ms)
```

1. **Sample-Parallel Ray Tracing**:
   - Both GPUs trace full-frame rays at 1 SPP using decorrelated PRNG seeds (`uboSec.frameIndex = m_frameIndex + 1000003u`).
2. **Half-Frame Cross-Exchange over PCIe Resizable BAR**:
   - Screen space is divided vertically: Top Half ($Y \in [0, H/2]$) and Bottom Half ($Y \in [H/2, H]$).
   - GPU 1 transfers its rendered Top Half (radiance, motion vectors, linear depth) to GPU 0 over DMA-BUF PCIe BAR (`VK_EXT_external_memory_dma_buf`).
   - Concurrently, GPU 0 transfers its rendered Bottom Half to GPU 1.
   - At 1080p input ($1920 \times 540$ at 8 bytes/pixel for FP16 radiance), each half-frame is only **$8.29\text{ MB}$**, completing over PCIe 4.0 x16 in **$0.026\text{ ms}$ ($26\ \mu\text{s}$)**.
3. **Local 2 SPP Variance-Halved Accumulation**:
   - GPU 0 adds GPU 1's Top Half into its local Top Half, yielding a converged **2 SPP Top Half**.
   - GPU 1 adds GPU 0's Bottom Half into its local Bottom Half, yielding a converged **2 SPP Bottom Half**.
   - Input variance is cut by 50% ($\sigma^2 \to \sigma^2 / 2$) before neural feature extraction begins.
4. **Parallel Inference with Guard-Band Apron ($A$ Texels)**:
   - Neural convolutions and temporal reprojection require valid neighbor samples. Clamping at $Y = H/2$ causes boundary distortion.
   - Pathways dispatches inference with an overlap apron of $A = 16 – 32$ texels:
     - GPU 0 reconstructs $Y \in [0, H/2 + A]$.
     - GPU 1 reconstructs $Y \in [H/2 - A, H]$.
   - The apron texels provide full spatial stencil context and allow motion-vector reprojection across the split boundary with zero seam artifacts.
5. **Display-Ready 10-Bit Packed Output (`VK_FORMAT_A2R10G10B10_UNORM_PACK32`)**:
   - Upscaling targets output directly to packed 10-bit HDR (4 bytes/pixel), matching 8-bit bandwidth while eliminating color banding in dark path-traced shadows.
   - GPU 1 blits its final upscaled 4K bottom half ($3840 \times 1080 \times 4\text{ bytes} \approx 16.58\text{ MB}$) directly into GPU 0's swapchain image over DMA-BUF BAR ($\approx 0.05\text{ ms}$).
   - GPU 0 handles UI overlay and swapchain presentation.
6. **Performance & Scaling**:
   - Post-processing time drops by $\approx 48\%$ ($2.80\text{ ms} \to 1.45\text{ ms}$).
   - Total dual-GPU frame time: $2.10\text{ ms} + 1.45\text{ ms} + 0.08\text{ ms (P2P)} = 3.63\text{ ms}$ (**275 FPS**).
   - Scaling efficiency: **$1.90\times – 1.94\times$**.

---

#### Strategy B: Dual-Stream Ensemble Denoising (Independent 1 SPP FSR)

Rather than splitting the screen spatially and exchanging intermediate radiance, Strategy B allows both GPUs to execute fully decoupled end-to-end pipelines, leveraging neural ensembling to suppress noise.

```
                         [CPU Engine / Frame Orchestrator]
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
     [GPU 1: Secondary Device]                             [GPU 0: Primary Device]
 ├── Traces Full Frame: 1 SPP (Seed B)                 ├── Traces Full Frame: 1 SPP (Seed A)
 ├── Local Temporal History (GPU 1 VRAM)               ├── Local Temporal History (GPU 0 VRAM)
 ├── Executes Full-Frame FSR on 1 SPP                  ├── Executes Full-Frame FSR on 1 SPP
 │   (Output: Full 4K 10-Bit A2R10G10B10)              │   (Output: Full 4K 10-Bit A2R10G10B10)
 │                      │                                  │
 └── Transfers Full 4K ─┴───────► [PCIe BAR Transfer] ────►├── Ensemble Merge Pass (ensemble_blend.comp)
     (33.17 MB, 0.10 ms)                                   │   (Weighted Average of Stream A & Stream B)
                                                           └── Composites UI & Swapchain Present
```

1. **Zero Pre-Inference Synchronization**:
   - Both GPUs trace a full-screen 1 SPP frame with decorrelated PRNG seeds.
   - Neither GPU transfers radiance or G-buffer data prior to upscaling; both pipelines proceed immediately to neural inference.
2. **Independent Full-Frame Neural Execution**:
   - GPU 0 executes FSR / neural reconstruction on Stream A (1 SPP).
   - GPU 1 executes FSR / neural reconstruction on Stream B (1 SPP).
   - Each GPU maintains its own private temporal history buffer locally in VRAM, eliminating history broadcast overhead.
   - Because each GPU processes the full viewport, **spatial split seams and boundary aprons are completely eliminated**.
3. **Post-Inference 10-Bit Transfer**:
   - GPU 1 exports its display-ready 4K 10-bit HDR output buffer (`A2R10G10B10`, 33.17 MB) and transfers it over PCIe Resizable BAR via DMA-BUF in **$\approx 0.105\text{ ms}$**.
4. **Ensemble Blending Pass (`ensemble_blend.comp`)**:
   - GPU 0 runs a high-speed compute pass combining the two upscaled streams:
     $$I_{\text{final}}(x, y) = w_A(x, y) \cdot I_A(x, y) + w_B(x, y) \cdot I_B(x, y)$$
   - *Statistical Variance Reduction*: Because the Monte Carlo noise in Stream A is statistically independent of Stream B ($\text{Cov}(S_A, S_B) = 0$), the residual errors in the neural reconstruction are uncorrelated. Blending the two neural inferences attenuates residual reconstruction artifacts and suppresses temporal boiling by a factor of $\approx 1/\sqrt{2}$ ($29.3\%$).
5. **Trade-Offs**:
   - Strategy B eliminates all pre-inference synchronization stalls and split seam logic.
   - However, each FSR instance operates on a noisier 1 SPP input compared to Strategy A's cleaner 2 SPP input, which can impact fine edge reconstruction in complex geometry.

---

#### Architectural Comparison: Strategy A vs. Strategy B

| Metric / Dimension | Strategy A: Split-Viewport (Merged 2 SPP) | Strategy B: Dual-Stream Ensemble (1 SPP) |
| :--- | :--- | :--- |
| **Input Sample Density** | **2 SPP** (Merged prior to inference) | **1 SPP** (Independent per GPU) |
| **Input Variance ($\sigma^2$)** | **$0.50 \times \sigma^2$** (Halved before neural model) | **$1.00 \times \sigma^2$** (Full raw Monte Carlo noise) |
| **Pre-Inference Transfer** | Half-frame FP16 + G-buffer ($\approx 8.3\text{ MB}$, $0.03\text{ ms}$) | **None** (Zero pre-inference synchronization) |
| **Inference Viewport** | Half-screen + Apron ($1920 \times 572 \to 3840 \times 1124$) | Full-screen ($1920 \times 1080 \to 3840 \times 2160$) |
| **Post-Inference Transfer** | 4K Bottom-half 10-bit ($\approx 16.58\text{ MB}$, $0.05\text{ ms}$) | 4K Full-frame 10-bit ($\approx 33.17\text{ MB}$, $0.10\text{ ms}$) |
| **Boundary Seam Vulnerability** | Mitigated via $A$-texel Guard Band Apron | **Zero** (Inherently seam-free full-frame rendering) |
| **Denoising Mechanism** | Analytical Monte Carlo variance reduction | Neural ensemble averaging ($\sim 29\%$ noise reduction) |
| **Temporal History** | Local to viewport half | Independent per-device temporal feedback |
| **Theoretical Dual-GPU Scaling** | **$1.90\times – 1.94\times$** | **$1.94\times – 1.97\times$** |

---

#### Strategy C: Disaggregated Functional Pipelining (Scales to $N$ GPUs)
When scaling to 4, 8, or more devices:
- **Primary Display GPU (GPU 0)**: Executes low-latency tasks: primary ray visibility, local G-buffer rasterization, Neural Reconstruction inference, tone mapping, UI, and swapchain presentation (locked to 120+ FPS).
- **Compute Worker Pool (GPUs 1 .. $N-1$)**: Compute heavy asynchronous indirect path bounces, ReSTIR spatio-temporal reservoir evaluations, or world-space radiance cache updates.

#### Zero-Copy Vulkan Inter-GPU Communications
Inter-device communication is handled via standard Linux DMA-BUF and timeline semaphores:
- **`VK_KHR_external_memory_fd` / `VK_EXT_external_memory_dma_buf`**: Exports GPU 1's `VkDeviceMemory` allocation as a file descriptor and imports it into GPU 0 over PCIe Resizable BAR without staging through host memory.
- **`VK_KHR_external_semaphore_fd`**: Synchronizes transfer completion and command buffer execution across `dev0` and `dev1` at the hardware scheduler level without CPU spinning.

---

### 3.3 Near-Term Precursor Testing: AMD FSR 3 Vulkan SDK Validation

While AMD has announced that FidelityFX Super Resolution 4 (FSR 4) will transition to a dedicated machine-learning / neural architecture, **FSR 4 is not yet available for Vulkan**. 

To avoid idle waiting and ensure all multi-GPU memory exchange pipelines, synchronization primitives, and seam mitigation algorithms are battle-tested in advance, Pathways utilizes **AMD FidelityFX Super Resolution 3 (FSR 3)** as the immediate validation vehicle.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 FSR 3 VULKAN PRECURSOR VALIDATION HARNESS                   │
└─────────────────────────────────────────────────────────────────────────────┘
  FidelityFX SDK 1.1 (Open-Source Vulkan 1.3/1.4 Track)
           │
           ├── Dual Independent VkDevice Contexts (dev0, dev1)
           ├── DMA-BUF PCIe BAR Inter-Device Buffer Exchange
           ├── Timeline Semaphore Cross-GPU Scheduling (VK_KHR_external_semaphore_fd)
           └── Packed 10-Bit HDR Swapchain Target (VK_FORMAT_A2R10G10B10_UNORM_PACK32)
           │
           ├── [Test 1: Split-Viewport Mode] ──► Apron Guard Band & Seam Continuity
           └── [Test 2: Dual-Stream Mode]    ──► Decoupled Pipeline & Ensemble Blend
           │
           ▼
  Seamless Drop-In Migration to FSR 4 Neural Tensor Inference (RDNA 4 WMMA)
```

#### Why FSR 3 is the Ideal Near-Term Stepping Stone
1. **Fully Open-Source Vulkan Implementation**: FSR 3 provides clean, production-grade Vulkan shaders and C++ host runtime code via the AMD FidelityFX SDK. It compiles and executes cleanly on Fedora Linux with Mesa RADV and RDNA 4 (`gfx1201`).
2. **Identical Buffer Requirements**: FSR 3 requires the exact same inputs as next-generation neural upscalers:
   - Low-resolution color (FP16 HDR)
   - Screen-space motion vectors ($16$-bit signed float)
   - Non-linear camera depth (24/32-bit float)
   - Reactive mask and transparency composition masks
3. **Validating Split-Viewport Apron Dynamics**: FSR 3 includes both spatial lanczos filtering and temporal history accumulation. Testing FSR 3 in Split-Viewport mode (Strategy A) will empirically determine the minimal apron width $A \in [16, 32]$ pixels required to prevent seam artifacts during rapid camera rotation and object disocclusion.
4. **Validating Dual-Device Context Architecture**: [`MultiGpuManager`](file:///home/naoki/Development/Pathways/src/mgpu/MultiGpuManager.cpp) must instantiate and manage two concurrent `FfxFsr3Context` structures across disparate `VkDevice` contexts without cross-device handle pollution.
5. **Drop-In Transition to FSR 4**: When AMD releases the FSR 4 Vulkan SDK, the host orchestration code, DMA-BUF buffer sharing, apron clipping, and 10-bit blit pipelines will already be fully debugged in Pathways. Upgrading will only require replacing the FSR 3 dispatch calls with FSR 4 neural tensor invocations.

---

---

## 4. Mastering High Dynamics: Motion, Animation, and Lighting Shifts

Real-time game engines feature high-speed camera pans, animated skinned characters, sudden disocclusions, and instantaneous lighting changes (e.g., muzzle flashes, explosions). 

Blindly accumulating past frames leads to **ghosting, motion smear, and light trails**. Discarding history too aggressively reverts to **boiling**. Pathways solves this via **learned temporal gating** and **physics-informed features**.

```
[Current Features X_t] ──────┬──────────────────────┬─────────────┐
                             ▼                      ▼             │
                  [Reset Gate r_t]       [Update Gate z_t]        │
                         │                      │                 │
                         ▼                      │                 │
[Warped Latent H_{t-1}] ─┴──► [Candidate State H̃_t]               │
                                       │                          │
                                       ▼                          ▼
                         [Gated Blend: (1 - z_t)·H̃_t + z_t·X_t] ──┴──► [Output H_t]
```

### 4.1 ConvGRU Bottleneck Architecture
At the network bottleneck, standard convolutions are replaced with a **Convolutional Gated Recurrent Unit (ConvGRU)**:
- **Reset Gate ($r_t \in [0, 1]$)**: Detects when historical memory has become invalid (e.g., sudden shadow transitions or disocclusions). When $r_t \to 0$, stale latent features are erased.
- **Update Gate ($z_t \in [0, 1]$)**: Evaluates per-pixel confidence. On static geometry, $z_t \to 0$ (relying on accumulated multi-frame history to eliminate boiling). On fast-moving edges or new light sources, $z_t \to 1$ (relying on current-frame features to eliminate ghosting).

### 4.2 Handling Specular Parallax (Non-Surface Motion)
Specular highlights and reflections on curved or flat surfaces do **not** move according to surface velocity $\mathbf{v}_{\text{surf}}$. Warping specular features with surface motion vectors causes reflections to drag and smear.
- Pathways calculates a dedicated **Specular Parallax Vector** ($\mathbf{v}_{\text{spec}}$) derived from the first-bounce hit distance $d$:
  $$\mathbf{x}_{\text{virtual}} = \mathbf{x}_{\text{hit}} + \mathbf{r}_{\text{refl}} \cdot d$$
- Specular radiance and diffuse irradiance are warped along their respective velocity fields before entering the recurrent cell.

### 4.3 Instantaneous Lighting Changes (Explosions, Flashes)
When a light turns on within a single frame, raw radiance spikes by orders of magnitude compared to $H_{t-1}$.
1. **Relative Log-Luminance Anomaly Metric**:
   $$\Delta E_t = \log_{10}(L_{\text{current}} + \epsilon) - \log_{10}(L_{\text{warped\_history}} + \epsilon)$$
   This delta is fed directly to the ConvGRU update gate. Positive energy spikes across multiple neighboring pixels force the gate to immediately accept current-frame data, preventing the flash from "fading in" over several frames.
2. **Log-Space / Tonemapped Activation Domain**:
   Radiance is compressed prior to neural inference:
   $$\tilde{L} = \frac{\ln(1 + \mu L)}{\ln(1 + \mu)}$$
   This prevents high-dynamic-range spikes ($>10{,}000\text{ nits}$) from destabilizing FP16 precision or blowing out latent states.

### 4.4 Training for High Dynamics
The network is trained on synthetic stress-test sequences:
- **Aggressive Trajectories**: 180° snap-turns, rapid zoom, and high-velocity linear acceleration.
- **Dynamic Occluders**: Animated procedural objects sweeping across the camera view to force disocclusion recovery.
- **Single-Frame Lighting Toggles**: Lights modulating on/off on single-frame boundaries.
- **Loss Formulation**:
  $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{spatial}} + \lambda_{\text{temp}} \mathcal{L}_{\text{temporal}} + \lambda_{\text{disc}} \mathcal{L}_{\text{disocclusion}}$$
  Where $\mathcal{L}_{\text{temporal}}$ enforces smoothness on continuous surfaces ($M_{\text{valid}}$), and $\mathcal{L}_{\text{disocclusion}}$ heavily penalizes ghosting on newly revealed geometry ($1 - M_{\text{valid}}$).

---

## 5. The Ground-Up Advantage: Full Engine & Neural Co-Design

Commercial upscalers are middleware plugins limited to consuming 2D screen buffers. Because Pathways controls both the path tracer and the neural model, we expose internal simulation states that middleware never sees.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                 Bespoke Engine Co-Design Architecture                       │
└─────────────────────────────────────────────────────────────────────────────┘

  Path Tracer Internals                      Simulation State
  ├── Monte Carlo PDF & Path Throughput      ├── Direct Light Delta Triggers
  ├── ReSTIR Reservoir Confidence (M, W)     ├── Bone / Object Acceleration
  └── Secondary Hit G1-Buffer                └── World-Space Surfel Cache
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                                      ▼
                        [Neural Reconstructor]
                                      │
             ┌────────────────────────┴────────────────────────┐
             ▼                                                 ▼
     [Final 4K Frame]                                [Uncertainty Map σ²]
             │                                                 │
    (Swapchain Presentation)                                   ▼
                                                    [Adaptive Ray Allocator]
                                                    (Dynamic Next-Frame SPP)
```

### 5.1 Exposing Monte Carlo Integrator Internals
1. **Path Throughput and Sample PDF ($w_i = \frac{f_r \cos\theta}{p}$)**:
   The network receives the exact Monte Carlo sample weight. It knows immediately whether a bright pixel is an unreliable low-probability outlier or a reliable sample, eliminating heuristic variance guesswork.
2. **ReSTIR Reservoir States ($M$ and $W$)**:
   - $M$ = Number of historical/spatial candidates aggregated in the reservoir.
   - $W$ = Final unbiased sample weight.
   $M$ acts as an **analytical confidence map**: if $M \ge 32$, the sample is mathematically converged; if $M = 1$ (e.g. at a newly cast shadow edge), the network knows it must actively reconstruct.
3. **Secondary Hit $G_1$-Buffer**:
   Pathways records the normal, depth, and material ID of where the **first indirect bounce landed**. The network uses this to detect disocclusion in reflections and refractions rather than relying solely on primary camera hits.

### 5.2 World-Space Latent Persistence (Decoupling from Screen Space)
Traditional screen-space temporal history is lost when the camera turns, forcing the scene to re-converge and boil. 
- Pathways maintains a **World-Space Surfel / Sparse Voxel Latent Cache**.
- Latent feature vectors are anchored to static 3D world geometry. When the camera pans away and returns, the network projects the world-space latents back into screen space as an initial prior, achieving **instant convergence with zero boiling on camera rotation**.

### 5.3 Direct Semantic Engine Signals
- **Simulation Light Event Triggers**: When a weapon fires or a light switches, the CPU/GPU simulation emits a direct event flag (`LightEvent { Position, Radius, Delta }`). The network's temporal reset gate is updated proactively, without waiting for pixel analysis.
- **Physical Acceleration Vectors ($\mathbf{a}_t$)**: Skeletal physics and camera acceleration are provided to the network, enabling predictive non-linear temporal warping.

### 5.4 The Closed Feedback Loop: Neural Path Guiding
Pathways turns the rendering pipeline into a bidirectional closed loop:
1. Alongside the reconstructed frame, the network outputs an **Uncertainty Heatmap** ($\sigma_{\text{pred}}^2$).
2. The ray tracer reads $\sigma_{\text{pred}}^2$ in the subsequent frame to dynamically steer its ray allocation:
   - Flat, resolved surfaces receive **0.25 SPP**.
   - Complex caustics, glossy reflections, and thin geometry receive **4.0 to 8.0 SPP**.

---

## 6. Implementation Blueprint & Phased Roadmap

| Phase | Objective | Deliverables |
| :--- | :--- | :--- |
| **Phase 1: Data Capture Harness** | Automated dataset generation pipeline within Pathways. | - Scripted camera flythroughs with high-speed snaps and light toggles.<br>- Concurrent capture of: 1080p 1-SPP radiance, G-buffer, ReSTIR reservoirs, and 4K 4096-SPP ground truth pairs. |
| **Phase 2: Offline PyTorch Training** | Model architecture design and multi-GPU training. | - ConvGRU temporal autoencoder with depthwise-separable convolutions.<br>- Loss formulation: spatial + temporal consistency + disocclusion penalty.<br>- DDP multi-GPU training with ROCm 10 on Dual Radeon AI PRO R9700. |
| **Phase 3: Vulkan Cooperative Matrix Inference** | Native real-time inference compute pipeline. | - `shaders/compute/neural_reconstruct.comp` utilizing `VK_KHR_cooperative_matrix` (Wave32 WMMA).<br>- Weight packing into storage buffers with FP16 precision.<br>- Performance target: $\le 2.5\text{ ms}$ at 1440p/4K on `gfx1201`. |
| **Phase 4: Dual-GPU & Engine Co-Design** | Full system integration into Pathways. | - Sample-split 2-spp accumulation across primary/secondary GPUs via `VK_KHR_external_memory_fd`.<br>- Integration of ReSTIR $M$-weights and secondary $G_1$-buffers into inference inputs.<br>- Bidirectional uncertainty feedback loop driving the dynamic SPP governor. |
