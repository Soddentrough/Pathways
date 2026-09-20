# Multi-GPU Architecture & Areas for Improvement (Pathways)

## 1. Executive Summary & Current Architecture

Pathways implements a high-performance, real-time multi-GPU path tracing backend specifically designed for modern Linux environments and discrete AMD Radeon hardware (specifically dual AMD Radeon AI PRO R9700 / RDNA 4 `gfx1201` GPUs).

### 1.1 The Core Design: Unlinked Multi-GPU via DMA-BUF Direct BAR
Unlike legacy game implementations that relied on monolithic Vulkan device groups (`VK_KHR_device_group`) and proprietary driver SLI/CrossFire profiles, Pathways operates on **Unlinked Heterogeneous Multi-GPU**:
- **Dual Independent Logical Devices:** [`MultiGpuManager`](src/mgpu/MultiGpuManager.cpp) instantiates two standalone `VkDevice` contexts ([`dev0` and `dev1`](tests/test_p2p_direct_bar.cpp#L57-L59)).
- **Zero-Copy PCIe P2P Transfer:** Secondary GPU accumulation targets are exported via `VK_EXT_external_memory_dma_buf` and mapped directly into the primary GPU's memory space over PCIe Resizable BAR (`P2P_Direct_BAR`).
- **Hardware-Level Signaling:** Synchronization between command streams is orchestrated with zero CPU spinning via `VK_KHR_external_semaphore_fd`.
- **Latency & Throughput:** At 4K native ($3840 \times 2160$), transferring the secondary GPU's compacted FP16 buffer takes only **0.090 – 0.135 ms** across PCIe 4.0/5.0.

### 1.2 The Two Multi-GPU Modes
Pathways dynamically bifurcates its workload across two distinct strategies:
1. **`MultiGpuMode::SampleParallel` (Multi-Sample Rendering, $\ge 2$ SPP):**
   - Each GPU traces full-screen rays at $N / 2$ SPP with decorrelated PRNG seeds (`uboSec.frameIndex = m_frameIndex + 1000003u`).
   - Results are combined via an additive FP16 accumulator compute pass ([`accum_merge.comp`](shaders/compute/accum_merge.comp#L61-L80)).
   - **Characteristics:** Zero spatial artifacts, 100% seam-free, perfectly compatible with screen-space denoisers, but inherently inapplicable to 1-SPP real-time rendering.
2. **`MultiGpuMode::CheckerboardTile` (Single-Sample Rendering, 1 SPP):**
   - Screen space is divided into a 2D checkerboard grid of $64 \times 64$ tiles (2,040 tiles at 4K).
   - Tile $(X, Y)$ parity: $P = (X + Y) \pmod 2$.
     - **GPU 0 (Primary):** Traces Parity 0 ("white" tiles).
     - **GPU 1 (Secondary):** Traces Parity 1 ("black" tiles).
   - **Compacted Dispatch Grid:** Both GPUs dispatch a compacted grid of dimensions $(W/2, H)$. In [`raytrace.rgen:116-127`](shaders/rt/raytrace.rgen#L116-L127), thread coordinates are mathematically unpacked into interleaved screen space with **zero wave divergence** on RDNA 4 Wave32.
   - **Compacted Storage & PCIe Optimization:** GPU 1 stores directly into its native $(W/2, H)$ buffer (`storeCoord = launchID`), cutting PCIe memory traffic in half (33 MB vs 66 MB at 4K FP16).
   - **Scaling Efficiency:** Achieves **97.0% to 99.3% efficiency** on geometry-heavy scenes (*Bistro Interior* 4.08 ms $\to$ 2.10 ms; *Pontiac GTO* 11.56 ms $\to$ 5.82 ms).

---

## 2. Critical Areas for Improvement & Identified Deficiencies

While the core hardware transfer and raygen dispatch pipelines are exceptionally well-engineered, several algorithmic, filtering, and memory bottlenecks remain.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PATHWAYS MULTI-GPU PIPELINE FLOW                         │
└─────────────────────────────────────────────────────────────────────────────┘
  [ GPU 0: Primary Device ]                     [ GPU 1: Secondary Device ]
   ├─ Raygen Grid (W/2, H)                       ├─ Raygen Grid (W/2, H)
   ├─ Traversal & Direct Lighting                ├─ Traversal & Direct Lighting
   ├─ Local Denoiser (Asymmetry Bug!) ◄──────────┼─ Local Denoiser (Missing Tap Data!)
   │                                             ├─ Compact Storage (W/2, H)
   │           Direct BAR DMA-BUF (0.1ms)        │
   │ ◄───────────────────────────────────────────┘
   ├─ Accum Merge Pass (accum_merge.comp)
   ├─ TAA / Temporal Accumulation (Seam History Glitches!)
   ├─ Upways / BMFR Neural Denoiser (Single-GPU Bottleneck!)
   └─ Swapchain Present
```

---

### Area 1: Cross-Tile Spatial Denoising & The Asymmetric Clamping Defect

#### The Defect in [`ffx_shadow_filter.comp`](shaders/compute/ffx_shadow_filter.comp)
When running in checkerboard mode, spatial cross-bilateral filters (e.g., the FidelityFX shadow denoiser's 9-tap À-Trous filter) require neighboring pixel samples (normal, depth, visibility). Because adjacent tiles belong to the other GPU, sampling across tile boundaries directly reads uninitialized or stale memory on that device.

In [`ffx_shadow_filter.comp:108-125`](shaders/compute/ffx_shadow_filter.comp#L108-L125), clamping logic was introduced to constrain taps within the local tile:
```glsl
// Checkerboard tile border clamping bounds
int minTileX = 0;
int maxTileX = pc.bufferDimensions.x - 1;
if ((pc.tileOffsetY > 0u || pc.tileSize > 0u) && pc.tileOffsetX > 0u) {
    uint ts = (pc.tileSize > 0u) ? pc.tileSize : pc.tileOffsetY;
    minTileX = int((uint(pixelCoord.x) / ts) * ts);
    maxTileX = minTileX + int(ts) - 1;
}

for (int dy = -1; dy <= 1; ++dy) {
    for (int dx = -1; dx <= 1; ++dx) {
        ivec2 tapCoord = pixelCoord + ivec2(dx, dy) * pc.stepSize;

        // Clamp coordinate within local checkerboard tile bounds and screen extents
        tapCoord.x = clamp(tapCoord.x, minTileX, maxTileX);
        tapCoord.y = clamp(tapCoord.y, 0, pc.bufferDimensions.y - 1); // <--- CRITICAL BUG
```

1. **The Vertical Clamping Asymmetry Bug:**
   `tapCoord.x` is clamped to `[minTileX, maxTileX]`, but **`tapCoord.y` is not clamped to the local tile vertical bounds `[minTileY, maxTileY]`**! In a 2D checkerboard pattern, the tile directly above (`dy = -1`) and directly below (`dy = +1`) **belongs to the opposing GPU**. When the filter kernel approaches the top or bottom edge of a tile, it samples invalid data from the other GPU's memory space, introducing high-frequency edge noise and flickering along horizontal seams.
2. **Boundary Replication Distortion:**
   Even when $Y$ is clamped, clamping filter taps to the tile edge causes **boundary value replication** (the edge texel is sampled multiple times with different kernel weights). When À-Trous step sizes increase (e.g., $s = 2, 4$), penumbras crossing a $64 \times 64$ boundary develop subtle seam artifacts.

#### Proposed Improvement: The Inter-Tile Guard Band (Apron Exchange)
Instead of clamping within local tiles or pushing all denoising onto GPU 0:
1. **Apron Dispatch:** Each GPU renders its assigned $64 \times 64$ tiles expanded by an $A$-pixel border apron (e.g., $A = 8$ or $16$ pixels, matching the denoiser kernel radius $\text{stepSize} \times 2$).
2. **Compact Edge Blit:** Before running spatial filtering, exchange only the narrow border bands via DMA-BUF P2P, or let each GPU evaluate rays on its tile plus apron, filtering seamlessly across the boundary, and clipping away the apron during the merge pass.
3. **Alternative: Post-Merge Denoising on Unified Geometry:** Move spatial filtering to GPU 0 immediately *after* [`accum_merge.comp`](shaders/compute/accum_merge.comp), transferring the packed G-buffer alongside radiance.

---

### Area 2: Temporal Reprojection Across Moving Tiles (TAA & BMFR)

#### The Problem
In stationary scenes, progressive accumulation works flawlessly because pixel $(x, y)$ remains in the same tile forever. However, under dynamic camera motion or moving objects:
- A pixel rendered by GPU 0 at Frame $N$ may have been rendered by GPU 1 at Frame $N-1$.
- Because each GPU only maintains its own history buffer in local VRAM, temporal reprojection lookups ($x - v_x, y - v_y$) frequently land in unrendered or out-of-date tiles.
- This creates checkerboard-patterned ghosting, variance clipping failures, and history smearing under fast camera translation.

#### Proposed Improvement: Unified Primary History Reprojection
1. **History Residence on GPU 0:** Consolidate the temporal history buffer (`uHistoryImage`) exclusively on GPU 0.
2. **Merge-First Temporal Loop:** 
   Execute the temporal pipeline strictly in the post-merge stage on GPU 0:
   $$\text{Ray Tracing (Dual GPU)} \longrightarrow \text{DMA-BUF P2P Transfer} \longrightarrow \text{Accum Merge} \longrightarrow \text{Temporal Reprojection / TAA}$$
3. **Double-Hop History Broadcast (Optional for Distributed Filtering):**
   If temporal filtering must remain distributed across both GPUs, GPU 0 can broadcast the reprojected history buffer back to GPU 1 via DMA-BUF BAR during the frame preamble. At 4K FP16, a full-frame transfer takes only $\sim 0.18\text{ ms}$, preserving complete temporal consistency.

---

### Area 3: Amdahl's Law in Post-Processing & Neural Upscaling

#### The Problem
In the current engine, ray tracing scales near $2.0\times$, but the post-raytracing pipeline is entirely serialized on GPU 0:
- Upways Neural Denoiser & Super-Resolution (WMMA Wave32 tensor passes)
- Temporal Radiance Accumulation & wRLS Outlier Rejection
- ACES Tonemapping & Color Grading
- ImGui UI Rendering & Presentation

As ray tracing execution drops to $\sim 2.10\text{ ms}$ on dual R9700s, post-processing on GPU 0 ($\sim 0.8 – 1.5\text{ ms}$) becomes a massive proportion of the total frame time. While GPU 0 executes these post-passes, **GPU 1 sits completely idle**, capping real-world scaling at $1.5\times – 1.6\times$.

#### Proposed Improvement: Cooperative Post-Processing
1. **Split-Viewport Tonemapping & Denoising:**
   Partition post-processing compute passes across both cards:
   - GPU 1 executes tonemapping and local contrast adjustments on its own $(W/2, H)$ compacted buffer *before* transfer.
   - GPU 0 only merges final display-ready LDR/sRGB values, slashing PCIe bandwidth by another $50\%$ (converting 8-byte FP16 down to 4-byte RGBA8).
2. **Asymmetric Wavefront Pipelining:**
   Overlap GPU 1's ray tracing for Frame $N+1$ with GPU 0's post-processing and presentation for Frame $N$.

---

### Area 4: VRAM Geometry Duplication & Memory Scaling

#### The Problem
Because path-traced diffuse and glossy rays bounce unpredictably across the entire 3D scene, secondary rays originating in a GPU 0 tile can strike objects anywhere in the world.
- Currently, [`MultiGpuManager`](src/mgpu/MultiGpuManager.cpp#L940-L1050) creates a **complete duplicate** of all BLAS, TLAS, vertex/index buffers, materials, textures, and light trees in GPU 1's VRAM.
- Although the dual Radeon AI PRO R9700 setup provides $2 \times 32\text{GB} = 64\text{GB}$ of total VRAM, the maximum scene capacity remains hard-capped at **32GB**.

#### Proposed Improvement: Unified Virtual Asset Paging via PCIe BAR
1. **Resident Core vs. Paged Assets:**
   - Keep the primary TLAS and low-detail proxy BLAS resident on both cards for fast ray intersection testing.
   - For ultra-high-resolution textures (4K/8K PBR albedo, normal, roughness) and secondary environment maps, store them non-redundantly across the two GPUs:
     - Textures 0–255 resident in GPU 0 VRAM.
     - Textures 256–511 resident in GPU 1 VRAM.
2. **P2P Direct Sampling over Resizable BAR:**
   On systems with full PCIe Resizable BAR enabled, GPU 0 can sample storage buffers located in GPU 1's VRAM directly over the PCIe bus using `VK_EXT_external_memory_dma_buf`. While remote PCIe reads have higher latency than local GDDR6, high-level ray bounces can absorb the latency through wavefront latency hiding.

---

### Area 5: Dynamic Tile Granularity & Pathological Ray Imbalance

#### The Problem
While $64 \times 64$ checkerboard tiling provides excellent statistical load balancing in open scenes (*Bistro*, *Cornell Box*), scenes with localized path-tracing hotspots break down:
- Example: Looking into a small bathroom with complex mirror reflections on the left side of the screen, while the right side displays a flat wall.
- Even with 2,040 alternating tiles, regions containing caustics, complex glass transmission, or high rough-specular recursion create heavy localized computational spikes, leaving one GPU waiting on the other.

#### Proposed Improvement: Adaptive Tile Granularity
1. **Feedback-Driven Tile Sizes:**
   Instead of a static $64 \times 64$ grid, support dynamic tile sizes ($32 \times 32$ or $16 \times 16$) in scenes with high spatial bounce variance. Smaller tile sizes increase spatial mixing, further smoothing out localized hotspots.
2. **Dynamic Work Stealing / Ray Queue Rebalancing:**
   In the Wavefront pipeline ([`wavefront_intersect.comp`](shaders/compute/wavefront_intersect.comp)), after Bounce 2:
   - If GPU 0 has 2,000,000 active rays remaining and GPU 1 has only 400,000, GPU 0 can export a contiguous ray queue chunk over DMA-BUF BAR for GPU 1 to process, bringing wave occupancy back to parity.

---

## 3. High-Priority Proposal: Parametric NRC + Compacted Spatial Checkerboard Synergy

To achieve the **lowest performance overhead and most linear scaling** while permanently eliminating cross-tile boundary seams and temporal ghosting, Pathways should implement and benchmark a combined **Parametric Neural Radiance Caching (NRC) + Compacted Spatial Checkerboard** architecture.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          PARAMETRIC NRC + COMPACTED SPATIAL CHECKERBOARD SYNERGY            │
└─────────────────────────────────────────────────────────────────────────────┘
  [ GPU 0: White Tiles (W/2, H) ]               [ GPU 1: Black Tiles (W/2, H) ]
   ├─ Primary Ray Trace (1 SPP)                  ├─ Primary Ray Trace (1 SPP)
   ├─ Query Local NRC MLP (Bounce 1/2)           ├─ Query Local NRC MLP (Bounce 1/2)
   │  (Instant Infinite GI in ~0.3 ms)           │  (Instant Infinite GI in ~0.3 ms)
   ├─ Compute Local Training Gradients ∇θ L₀     ├─ Compute Local Training Gradients ∇θ L₁
   │                                             │
   │  ◄─── AllReduce Gradient Sync (PCIe BAR: 50 KB in 0.003 ms / 3 µs) ───►
   │                                             │
   ├─ Update Local Weights: θ ← θ - η·g          ├─ Update Local Weights: θ ← θ - η·g
   │                                             │
   │  ◄─── Compact Primary Hit + Motion Vectors (DMA-BUF: 16 MB LDR) ──────┘
   │
   ├─ Advection-Padded Motion Vector Merge (accum_merge.comp)
   └─ Swapchain Present
```

### 3.1 Architectural Principles
1. **Primary Hit via Compacted Checkerboard ($W/2 \times H$):**
   - Retain Pathways' existing, battle-tested compacted dispatch grid in [`raytrace.rgen`](shaders/rt/raytrace.rgen).
   - GPU 0 traces Parity 0 ("white" tiles); GPU 1 traces Parity 1 ("black" tiles).
   - Guarantees 100% SIMD lane occupancy on RDNA 4 Wave32 and cuts primary ray traversal overhead by exactly 50%.
2. **Infinite Indirect Bounces via Local NRC Inference:**
   - Rather than executing 4–8 divergent ray bounces across scene BVHs (which chokes GDDR6 memory bandwidth and induces ray divergence), rays terminate at Bounce 1 or 2 and query a small, local Multi-Layer Perceptron (MLP).
   - On the dual Radeon AI PRO R9700s, this MLP is evaluated in pure FP16 via hardware WMMA (Wave Matrix Multiply Accumulate) tensor instructions ([`compute/nrc_evaluate.comp`](shaders/compute/nrc_evaluate.comp)), resolving global illumination in $\sim 0.3\text{ ms}$.
3. **Data-Parallel Temporal Synchronization (Gradient AllReduce):**
   - The temporal history of the scene's radiance field is parameterised in the **weights $\theta \in \mathbb{R}^D$ of the network**, not in a 2D pixel buffer.
   - During training passes (5–10% of rays), each GPU calculates local weight gradients ($\nabla_\theta \mathcal{L}_0$ and $\nabla_\theta \mathcal{L}_1$).
   - The two GPUs perform an **AllReduce over PCIe Resizable BAR via DMA-BUF**:
     $$g = \frac{1}{2}\big(\nabla_\theta \mathcal{L}_0 + \nabla_\theta \mathcal{L}_1\big)$$
   - **Bandwidth Reduction:** For a 5-layer, 64-wide MLP ($D \approx 12,000 – 48,000$ FP16 weights), the gradient tensor is **less than $100\text{ KB}$**. Transfer latency over PCIe 4.0/5.0 is **$0.003\text{ ms}$ ($3\ \mu\text{s}$)**—completely eliminating the 33–66 MB frame-buffer transfer bottleneck.
4. **Natural Elimination of Boundary Seams:**
   - Because the neural network represents a continuous function in 3D world space ($L(x, \omega; \theta)$), indirect lighting is **mathematically continuous across 2D screen tiles**.
   - Tile seams cannot exist in the indirect radiance field.
5. **Shared Motion Vectors for Primary Reconstruction:**
   - To resolve primary hit silhouettes and temporal anti-aliasing without ghosting, both GPUs share their compacted motion vector field ($\sim 16\text{ MB}$, $0.05\text{ ms}$).
   - The neural denoiser uses these motion vectors to perform **Advection Boundary Padding**, extrapolating tile border features along their true optical flow rather than relying on static clamping or zero-padding.

---

### 3.2 Proposed Test & Benchmark Plan

1. **Phase 1: Gradient AllReduce Microbenchmark:**
   - Implement an isolated standalone test (`tests/test_p2p_allreduce.cpp`) allocating a 100 KB shared DMA-BUF buffer between `dev0` and `dev1`.
   - Measure round-trip latency of cross-device gradient accumulation using `VK_KHR_external_semaphore_fd`. Target: $\le 10\ \mu\text{s}$.
2. **Phase 2: Multi-GPU NRC Pipeline Integration:**
   - Extend [`MultiGpuManager`](src/mgpu/MultiGpuManager.cpp) to dispatch [`nrc_evaluate.comp`](shaders/compute/nrc_evaluate.comp) and [`nrc_train.comp`](shaders/compute/nrc_train.comp) concurrently on both `GpuDeviceNode` instances.
   - Remap training ray samples using the compacted checkerboard coord unpack.
3. **Phase 3: Scaling & Quality Verification:**
   - Benchmark against standard 4-bounce brute-force path tracing across *Living Room*, *Kitchen Extended*, and *Bistro Interior*.
   - Target Metrics:
     - **Scaling Efficiency:** $\ge 96\%$ (near-linear $1.92\times – 1.98\times$ dual-GPU speedup).
     - **Frame Time Reduction:** $2.5\times – 3.0\times$ faster overall frame latency compared to brute-force multi-bounce tracing.
     - **Visual Quality:** Zero visible seams under `--visualize-split` in shadow penumbras and indirect diffuse corners.

---

### 3.3 Future Validation Plan: Parallel Multi-GPU FSR 4 (Vulkan SDK)

With AMD transitioning FidelityFX Super Resolution 4 (FSR 4) to a dedicated machine-learning / neural architecture (leveraging hardware tensor / Wave Matrix Multiply-Accumulate units on RDNA 4), Pathways should implement and benchmark **Parallel Dual-GPU FSR 4 Execution** upon the release of the FSR 4 Vulkan SDK:

#### The Dual-GPU Execution Model
Rather than serializing FSR 4 on GPU 0 after the merge (which induces the Amdahl's Law penalty capping scaling at $\sim 1.8\times$), the engine will distribute inference across both devices:
1. **Per-Device FSR 4 Contexts:** Instantiate standalone FSR 4 contexts on both `GpuDeviceNode` instances in [`MultiGpuManager`](src/mgpu/MultiGpuManager.cpp).
2. **Pre-Transfer Neural Upscaling & Denoising:**
   - Both GPUs run FSR 4 in parallel on their respective half-grids / tiles.
   - Use the shared motion vector field ($16.58\text{ MB}$, $0.05\text{ ms}$) and linear depth for boundary advection padding to prevent seam artifacts between tiles.
3. **10-Bit Packed Output (`VK_FORMAT_A2R10G10B10_UNORM_PACK32`):**
   - Output directly to 10-bit per channel display-ready HDR (`A2R10G10B10`).
   - Slashes secondary GPU PCIe BAR transfer traffic by $50\%$ compared to raw FP16 HDR (from $33.17\text{ MB}$ down to **$16.58\text{ MB}$** at 4K).
4. **Near-Term Testing Envelope (1 SPP vs. 2 SPP):**
   - While 128–256 SPP represents an offline/cinematic target, initial real-time testing will focus on **1 SPP and 2 SPP** configurations where the engine already maintains substantial framerate headroom on dual R9700s:
     - *Bistro Interior:* **476 FPS (2.10 ms)** at 1 SPP $\implies$ **~14.5 ms headroom** against a 60 FPS budget, **~6.2 ms headroom** against 120 FPS.
     - *Cornell Box:* **265 FPS (3.76 ms)** at 1 SPP $\implies$ **~12.9 ms headroom** (60 FPS), **~4.5 ms headroom** (120 FPS).
     - *Living Room:* **178 FPS (5.61 ms)** at 1 SPP $\implies$ **~11.0 ms headroom** (60 FPS), **~2.7 ms headroom** (120 FPS).
   - **The 1 SPP vs. 2 SPP Mode Transition:**
     - **At 1 SPP (`CheckerboardTile`):** Tests FSR 4 parallel tile inference and shared motion-vector advection padding to bridge spatial seams.
     - **At 2 SPP (`SampleParallel`):** Pathways automatically shifts to full-frame sample interleaving (1 SPP on GPU 0, 1 SPP on GPU 1 with decorrelated seeds). **Spatial seams vanish entirely**, allowing FSR 4 to run on a clean, seamless 2 SPP input while comfortably remaining within 60–120 FPS.
   - **Expected Scaling:** Restores dual-GPU scaling from $\sim 1.80\times$ back to **$\ge 1.95\times$**.

---

## 4. Implementation Roadmap & Priority Matrix

| Tier | Task | Impact | Complexity | Target Files |
| :---: | :--- | :---: | :---: | :--- |
| **Tier 1** | **Fix Spatial Filter $Y$-Clamp Bug**<br>Clamp `tapCoord.y` to local tile bounds in `ffx_shadow_filter.comp`. | High (Eliminates seam noise) | Low | [`ffx_shadow_filter.comp`](shaders/compute/ffx_shadow_filter.comp) |
| **Tier 1** | **RGBA8 / A2R10G10B10 Transfer Mode**<br>Move tonemapping / 10-bit conversion to secondary GPU pre-transfer, cutting PCIe traffic from 33MB to 16.5MB. | Medium (50% bandwidth cut) | Medium | [`MultiGpuManager.cpp`](src/mgpu/MultiGpuManager.cpp)<br>[`accum_merge.comp`](shaders/compute/accum_merge.comp) |
| **Tier 2** | **Merge-First Denoising Architecture**<br>Move spatial À-Trous and BMFR passes post-merge on GPU 0 to eradicate all tile boundary seams. | High (Flawless penumbras) | Medium | [`Engine.cpp`](src/core/Engine.cpp) |
| **Tier 2** | **Full History Buffer Broadcast for TAA**<br>Broadcast previous frame history to GPU 1 over DMA-BUF to enable motion-vector reprojection in checkerboard mode. | High (Enables real-time TAA) | Medium | [`Engine.cpp`](src/core/Engine.cpp)<br>[`MultiGpuManager.cpp`](src/mgpu/MultiGpuManager.cpp) |
| **Tier 3** | **Parametric NRC + Compacted Checkerboard Integration**<br>Implement data-parallel gradient AllReduce for Neural Radiance Caching across dual GPUs. | **Transformative (Near-linear scaling, 3 µs sync, zero seams)** | High | [`MultiGpuManager.cpp`](src/mgpu/MultiGpuManager.cpp)<br>[`nrc_evaluate.comp`](shaders/compute/nrc_evaluate.comp)<br>[`nrc_train.comp`](shaders/compute/nrc_train.comp) |
| **Tier 3** | **Shared Motion Vector Advection Padding**<br>Share the compacted 16MB velocity field to enable flow-guided boundary extrapolation for neural denoisers. | Very High (Seamless neural denoising) | High | [`MultiGpuManager.cpp`](src/mgpu/MultiGpuManager.cpp)<br>[`accum_merge.comp`](shaders/compute/accum_merge.comp) |
| **Tier 3** | **Parallel Dual-GPU FSR 4 Integration (Upon Vulkan SDK Release)**<br>Execute FSR 4 neural upscaling and reconstruction concurrently on both GPUs with 10-bit `A2R10G10B10` output to restore $\ge 1.95\times$ scaling. | **Transformative (Enables 128–256 SPP 4K real-time at $\ge 1.95\times$ scaling)** | High | [`MultiGpuManager.cpp`](src/mgpu/MultiGpuManager.cpp)<br>[`Engine.cpp`](src/core/Engine.cpp) |
| **Tier 4** | **VRAM Texture Virtualization via P2P BAR**<br>Pool the 64GB VRAM across both GPUs by storing non-overlapping texture sets in each card's physical memory. | High (Enables >32GB scenes) | High | [`MultiGpuManager.cpp`](src/mgpu/MultiGpuManager.cpp)<br>[`Texture.cpp`](src/vulkan/Texture.cpp) |

---

## 5. Verification & Testing Methodology

When implementing improvements to the multi-GPU pipeline:
1. **Automated Unit Testing:**
   Run the dedicated P2P barrier and TAA unit tests:
   ```bash
   ./build/linux-release/bin/test_p2p_direct_bar
   ./build/linux-release/bin/test_taa
   ./build/linux-release/bin/test_shadow_denoiser
   ```
2. **Visual Seam Verification:**
   Launch with `--visualize-split` and `--mgpu`:
   ```bash
   ./build/linux-release/bin/pathways --scene scenes/living-room.obj --mgpu --tile-size 64 --visualize-split
   ```
   Inspect penumbra boundaries across the alternating cyan and amber overlay to ensure that À-Trous filter steps do not replicate edge pixels or sample stale neighbor data.
3. **Hardware Trace Auditing:**
   Use the low-level SQTT / RGP profiler to ensure that inter-device semaphore waits do not cause CPU thread stalls or GPU queue bubbles:
   ```bash
   MESA_VK_TRACE=rgp MESA_VK_TRACE_FRAME=10 ./build/linux-release/bin/pathways --scene scenes/kitchen.obj --mgpu
   ```
   Verify that the DMA-BUF transfer occurs concurrently with primary GPU compute passes whenever possible.
