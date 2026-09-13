# Neural Radiance Caching (NRC) in Pathways

## 1. Overview & Operational Policy

Pathways integrates an experimental on-device, hardware-accelerated **Neural Radiance Caching (NRC)** subsystem built on **`VK_KHR_cooperative_matrix` (Wave32 WMMA)** for AMD RDNA 4 (`gfx1201`, AMD Radeon AI PRO R9700).

> [!IMPORTANT]
> **Configuration Policy**: NRC is strictly an **optional, opt-in feature (`--nrc`)**, **disabled by default**.
> For standard real-time interactive navigation at 1 SPP with low bounce counts ($\le 4$ bounces), pure DGC Wavefront path tracing provides superior raw framerate and sharper dynamic response. NRC is designed for deep-bounce ($\ge 8$ to 16 bounces), diffuse-dominated architectural scenes and progressive accumulation.

---

## 2. Architecture & Pipeline Integration

The NRC subsystem is implemented in `src/rt/NRCManager.cpp`, `src/rt/NRCManager.hpp`, and compute shaders:

```
                              4K / 1080p Primary Viewport
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │    Bounce 0: Primary  │  (Hardware Ray Queries:
                              │    Geometric Hit      │   100% geometric sharpness)
                              └───────────┬───────────┘
                                          │
                                          ▼
                              ┌───────────────────────┐
                              │    Bounce 1: Direct   │  (Direct lighting +
                              │    & Specular Hit     │   Shadow microkernel)
                              └───────────┬───────────┘
                                          │
              ┌───────────────────────────┴───────────────────────────┐
              │ (97% of Rays)                                         │ (3% Continuation Rays)
              ▼                                                       ▼
  ┌───────────────────────┐                               ┌───────────────────────┐
  │  Emit NRCQuery        │                               │  Emit NRCTrainingRec  │
  │  (Pos, Normal, V, Alb)│                               │  (Continue Tracing)   │
  └───────────┬───────────┘                               └───────────┬───────────┘
              │                                                       │
              ▼                                                       ▼
  ┌───────────────────────┐                               ┌───────────────────────┐
  │  nrc_encode_infer     │                               │  nrc_train.comp       │
  │  Wave32 WMMA 16x16x16 │                               │  Rel L1 Loss + Adam   │
  └───────────┬───────────┘                               └───────────┬───────────┘
              │                                                       │
              ▼                                                       ▼
  ┌───────────────────────┐                               ┌───────────────────────┐
  │ Indirect Radiance     │                               │ Updated MLP Weights   │
  │ Added to Accum Image  │                               │ for Next Frame        │
  └───────────────────────┘                               └───────────────────────┘
```

### 2.1 Wave32 WMMA Cooperative Matrix Neural Pipeline
- **Hardware Target**: AMD RDNA 4 (`gfx1201`) Wave32 SIMD matrix tensor units.
- **Subgroup Control**: Explicitly locked to Wave32 via `VkPipelineShaderStageRequiredSubgroupSizeCreateInfo` (`requiredSubgroupSize = 32`), satisfying Vulkan 1.4 rule `VUID-VkPipelineShaderStageCreateInfo-module-08987`.
- **ISA Analysis**: Verified with Radeon GPU Analyzer (RGA): emits native `v_wmma_f32_16x16x16_f16` instructions with **0 scratch memory spills** and **0 VGPR spills**.
- **MLP Topology**:
  - Layer 0: 64 inputs $\to$ 64 hidden units (LeakyReLU)
  - Layer 1: 64 $\to$ 64 hidden units (LeakyReLU)
  - Layer 2: 64 $\to$ 16 outputs (First 3 channels = positive RGB radiance, linear)
  - Memory Footprint: 9,216 half-precision weights ($18.4\text{ KB}$, permanently resident in GPU L1 cache).

### 2.2 Multi-Frequency Sinusoidal Positional Encoding
Rather than using an uncoalesced global memory hash grid (which incurred 41M memory stalls and $>9.8\text{ ms}$ latency at 4K), spatial encoding computes 12 frequency octaves directly in vector ALU registers:
$$f_k(x) = \left[ \sin(2^k \pi x), \cos(2^k \pi x) \right], \quad k \in [0, 11]$$
Combined with surface normal (3), view direction (3), roughness (1), and base albedo (3), this forms the 64-channel feature vector with **zero VRAM bandwidth**.

### 2.3 Online Adam Training Subsystem
- **Stochastic Sampling**: Continues $\sim 3\%$ of paths to collect ground-truth training radiance.
- **Relative $\ell_1$ Loss**: Dampens fireflies and high-dynamic-range variance while maintaining color fidelity:
  $$\mathcal{L}(y, \hat{y}) = \frac{|y - \hat{y}|}{y + \epsilon}, \quad \frac{\partial \mathcal{L}}{\partial y} = \frac{\text{sign}(y - \hat{y})}{y + \epsilon} \quad (\epsilon = 0.01)$$
- **Reverse-Mode Backpropagation & Adam Optimizer**: Dynamically tracks first ($M$) and second ($V$) moments per parameter to adapt to camera translations and dynamic lighting.

---

## 3. Empirical Profiling (4K Living Room: 1 SPP, 4 Bounces)

Benchmarked on dual **AMD Radeon AI PRO R9700** GPUs (32GB VRAM each) at 4K ($3840 \times 2160$):

| Stage Breakdown | Tier 3 DGC Wavefront Baseline | Tier 4 NRC Wavefront | Latency Delta |
| :--- | :--- | :--- | :--- |
| **Primary RayGen (Classify)** | 1.159 ms | 1.198 ms | +0.039 ms |
| **Bounce 0 (Direct Lighting)** | 6.412 ms | 6.506 ms | +0.094 ms |
| **Bounce 1 (1st Indirect)** | 4.641 ms (1.85M rays) | 5.057 ms | +0.416 ms [VRAM Queue Write] |
| **Bounce 2 (2nd Indirect)** | 1.459 ms (850k rays) | 1.470 ms (127k rays) | +0.011 ms |
| **Bounce 3 (3rd Indirect)** | 0.379 ms (13k rays) | 0.109 ms (8k rays) | **-0.270 ms** [Ray Traversal Savings] |
| **NRC WMMA Inference Dispatch** | N/A | 1.250 ms | +1.250 ms [Compute Overhead] |
| **NRC Adam Training & Barrier** | N/A | 0.630 ms | +0.630 ms [Compute Overhead] |
| **Tonemap & Resolve** | 0.115 ms | 0.117 ms | +0.002 ms |
| **Total Frame Time** | **12.062 ms** (82.9 FPS) | **14.283 ms** (70.0 FPS) | **+2.221 ms** (18% slowdown) |

---

## 4. Multi-SPP & Deep-Bounce Analysis: Why 8 SPP + 16 Bounces Appears Noisier

When testing multi-sample workloads (e.g., 8 SPP with 16 bounces), users may observe that enabling NRC makes the image significantly noisier and degraded compared to baseline pure Monte Carlo. Waiting for accumulation does not eliminate this noise.

### 4.1 Root Causes

#### 1. Query Queue Overflow & Silent Ray Dropping
- In `src/rt/NRCManager.cpp`, the query queue buffer is sized to single-sample viewport capacity:
  $$\text{Queue Capacity} = \text{Width} \times \text{Height} \quad (2{,}073{,}600 \text{ records at 1080p})$$
- At **1 SPP**, surviving rays reaching Bounce 2 ($\sim 460{,}000$) fit comfortably.
- At **8 SPP**, 8 samples per pixel generate $8 \times 460{,}000 \approx \mathbf{3{,}680{,}000}$ **secondary rays**.
- In `shaders/compute/wavefront_shade_diffuse.comp`:
  ```glsl
  uint qIdx = atomicAdd(nrcCounters.queryCount, 1u);
  if (qIdx < pc.maxQueueCapacity) {
      nrcQueries[qIdx] = ...; // Enqueued
  }
  pathTerminated = true; // Ray is always terminated!
  ```
- By sample 4 or 5, `queryCount` exceeds `maxQueueCapacity`. For all remaining samples, the path is terminated from tracing further bounces, but **silently dropped from the query queue**.
- The dropped samples contribute **$0.0$ indirect radiance**, causing massive sample-to-sample variance, dark holes, and severe blotchiness.

#### 2. Non-Atomic Read-Modify-Write Data Race on `uAccumImage` [RESOLVED - CRIT-05]
- At **1 SPP**, each pixel $(x, y)$ has at most one query at bounce 2.
- At **8 SPP**, each pixel $(x, y)$ has up to 8 separate queries in the queue.
- *Historical Issue*: In previous revisions of `shaders/compute/nrc_encode_infer.comp`, inference waves executed unsynchronized `imageLoad` and `imageStore` on `uAccumImage`. Multiple GPU compute waves targeting the same pixel coordinates collided concurrently without atomic operations, overwriting each other's radiance contributions and creating high-frequency salt-and-pepper noise and flickering.
- *Resolution (CRIT-05)*: Replaced the storage image load/store with a 32-bit Q16.16 fixed-point atomic buffer (`AtomicAccumBuffer`) where inference threads execute hardware integer `atomicAdd(nrcAtomicBuffer[base + c], fixedRad[c])` scaled by $65,536.0$. A dedicated single-pass resolve microkernel (`shaders/compute/nrc_resolve.comp`) executes one thread per pixel to read the accumulated fixed-point values, scale back to float, apply stochastic FP16 dithering onto `uAccumImage`, and cleanly reset the atomic buffer back to zero in L2/MALL cache.


#### 3. Monte Carlo Physical Convergence vs. Under-Trained Neural Approximation
- **Pure Monte Carlo**: At 8 SPP and 16 bounces, each pixel fires 8 true physical rays traversing the full BVH hierarchy. Variance decreases monotonically following the central limit theorem:
  $$\sigma \propto \frac{1}{\sqrt{N_{\text{samples}}}}$$
- **NRC**: Ray traversal stops at Bounce 2. The remaining 14 bounces are approximated by an online MLP. The network is initialized from scratch and trains on a small stream ($1{,}024$ samples/frame) targeting direct lighting (`secDirectL`) rather than the fully integrated multi-bounce tail. It cannot match the physical fidelity or smoothness of 8 true Monte Carlo random walks.

---

## 5. Usage Recommendations & Guidelines

| Configuration | Behavior with NRC | Recommendation |
| :--- | :--- | :--- |
| **1 SPP + $\ge 8$ to 16 Bounces** | Clean (no buffer overflow, no write collisions, fast constant inference) | **RECOMMENDED for NRC** |
| **1 SPP + 4 Bounces** | Slower than baseline (+2.2 ms overhead due to VRAM writes and dispatch overhead) | **Use Baseline Wavefront** |
| **8 SPP + any bounces** | Queue saturates at sample 4; concurrent write data races on `uAccumImage` | **Use Baseline Wavefront** (Do not enable NRC) |
| **Interactive Camera Motion** | Cache requires accumulation frames to adapt; 1-frame contribution is imperceptible | **Use Baseline Wavefront** |

---

## 6. CLI & UI Controls Reference

### CLI Options (Strict Single-Option Design)
- `--nrc`: Enables Tier 4 Neural Radiance Caching.
- `--nrc-bounce <N>`: Sets the cutoff bounce depth (default: `2`). Bounces $0 \dots (N-1)$ are fully traced; bounce $N$ queries the neural cache.
- `--nrc-train-ratio <F>`: Fraction of rays continued for ground-truth training collection (default: `0.03`, i.e. 3%).

### Live ImGui Controls
Located under **Lighting & Shading Components**:
- **Enable Neural Radiance Cache (Wave32 WMMA)**: Checkbox toggling runtime inference and training passes.
- **Cutoff Bounce**: Slider $[1, 4]$ selecting the bounce index where NRC evaluation begins.
- **Training Continuation Ratio**: Slider $[0.01, 0.10]$ controlling the fraction of paths kept alive for ground-truth training.

---

## 7. Future Architectural Roadmap for Multi-SPP NRC

To enable high-quality NRC at $\text{SPP} > 1$, the following architectural improvements are planned:
1. **Dynamic Multi-SPP Buffer Allocation**: Size `m_queryQueue` dynamically as $\text{Width} \times \text{Height} \times \text{SPP}$, or dispatch NRC inference per sample index rather than once per frame.
2. **Intermediate Atomic Accumulation**: Accumulate NRC inference contributions into a 32-bit fixed-point integer atomic buffer or per-pixel staging buffer to eliminate the non-atomic `imageLoad`/`imageStore` race condition. **[COMPLETED - CRIT-05 via Q16.16 AtomicAccumBuffer & nrc_resolve.comp]**
3. **Multi-Bounce Training Ground Truth**: Route downstream continuation ray radiance back into `nrcTrainRecords.targetRadiance` across bounces $3 \dots 16$ to train the MLP on true multi-bounce equilibrium.
4. **Indirect Dispatch Sizing**: Use `vkCmdDispatchIndirect` reading from `counters.queryCount` to eliminate launching idle wavegroups.
