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

### 3.2 Real-Time Multi-GPU Runtime Architecture (Vulkan Engine)

#### Why Naive Alternate Frame Rendering (AFR) Fails
Under AFR (GPU 0 renders frame $t$, GPU 1 renders frame $t+1$), frame $t+1$ depends on the recurrent neural latent state $H_t$ from GPU 0. This creates an inter-GPU synchronization dependency that stalls the GPU pipelines and adds $N$ frames of input latency.

#### Strategy A: Monte Carlo Sample Splitting (Optimal for 2 GPUs)
Both GPUs render the **same frame** simultaneously, dividing the Monte Carlo sample paths:

```
                         [CPU Engine / Frame Orchestrator]
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
     [GPU 1: Secondary Ray Worker]                         [GPU 0: Primary Display GPU]
 ├── Traces Ray Paths (Seed B)                         ├── Rasterizes G-Buffer & Motion Vectors
 ├── Computes Direct & Indirect Radiance               ├── Traces Ray Paths (Seed A)
 └── Exports Radiance Buffer via DMA-BUF               ├── Accumulates Ray Samples (Seed A + Seed B)
                        │                                  ├── Executes Neural Reconstructor
                        └───────► [PCIe P2P Transfer] ────►├── Composites Post-Processing & UI
                                  (1080p FP16: ~16 MB,     └── Presents to Swapchain
                                   transfer: ~0.53 ms)
```

- **Sample Density**: GPU 0 merges its 1-spp output with GPU 1's 1-spp output to form a clean **2-spp input**. Input variance is reduced by $\approx 50\%$ before the neural network ever touches it.
- **Unidirectional Data Flow**: Only untextured radiance ($L_i$) transfers from GPU 1 to GPU 0. The neural reconstructor runs exclusively on GPU 0 right before presentation, keeping the recurrent temporal history ($H_{t-1}$) completely local to GPU 0's VRAM.

#### Strategy B: Disaggregated Functional Pipelining (Scales to $N$ GPUs)
When scaling to 4, 8, or more devices:
- **Primary Display GPU (GPU 0)**: Executes low-latency tasks: primary ray visibility, local G-buffer rasterization, Neural Reconstruction inference, tone mapping, UI, and swapchain presentation (locked to 120+ FPS).
- **Compute Worker Pool (GPUs 1 .. $N-1$)**: Compute heavy asynchronous indirect path bounces, ReSTIR spatio-temporal reservoir evaluations, or world-space radiance cache updates.

#### Zero-Copy Vulkan Inter-GPU Communications
Inter-device communication is handled via standard Linux DMA-BUF and timeline semaphores:
- **`VK_KHR_external_memory_fd`**: Exports GPU 1's `VkDeviceMemory` allocation as a file descriptor and imports it into GPU 0 without staging through host memory. On PCIe 4.0 x16 (31.5 GB/s), transferring a 1080p FP16 radiance buffer (16.6 MB) takes **$\approx 0.53\text{ ms}$**, hidden behind primary ray dispatch.
- **`VK_KHR_external_semaphore_fd`**: Synchronizes transfer completion at the hardware scheduler level without CPU intervention.

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
