# Unified 3-Axis Dynamic Sample Rate (DSR) & Quality Governor: Architecture & Implementation Plan

## 1. Executive Summary & Vision

Pathways currently delivers real-time path tracing at extreme frame rates on modern GPU architectures such as Dual AMD Radeon AI PRO R9700 GPUs (RDNA 4, gfx1201). When configured at baseline 1 primary ray + 4 bounce rays, the engine achieves:
- **`DamagedHelmet` (4K Native):** 614 FPS (1.6 ms) single-GPU, 880 FPS (1.1 ms) dual-GPU.
- **`DragonAttenuation` (4K Native):** 138–148 FPS (6.7–7.2 ms) single-GPU, 232–248 FPS (4.0–4.3 ms) dual-GPU.
- **`living-room` (4K Native):** 80–90 FPS (11.1–12.4 ms) single-GPU, 155–166 FPS (6.0–6.4 ms) dual-GPU.

While these benchmarks validate high ray tracing throughput, running uncapped at 1 SPP introduces two major limitations:
1. **Perceptual Noise During Interactive Navigation:** Because progressive accumulation resets whenever the camera moves (`m_frameIndex = 0`), the user perceives a noisy, grainy 1-SPP image during camera movement despite the hardware running at full power.
2. **Surplus Headroom Waste:** In typical 60 Hz or 120 Hz displays, excess frames rendered beyond the display refresh rate are discarded or presented redundantly. That surplus GPU compute could instead be invested into multi-sample quality (2–14 SPP) within a guaranteed frame budget.

**Objective:** Transform Pathways from a static-budget renderer into an autonomous, self-optimizing path tracer. Given a user-defined target frame rate (e.g., 60 FPS / 16.67 ms, 120 FPS / 8.33 ms, or 30 FPS / 33.33 ms), the engine dynamically modulates sample counts, multi-GPU allocation, and bounce depths to maximize visual fidelity while strictly honoring the frame time budget.

```
                  ┌────────────────────────────────────────────────────────┐
                  │             Target Frame Budget (e.g. 60 FPS)          │
                  │             Total Frame Time: 16.667 ms                │
                  └───────────────────────────┬────────────────────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     │ Fixed Overhead (~0.5 - 1.0 ms):                 │
                     │ - ACES Tonemapping Compute Pass                 │
                     │ - Swapchain Blit & Presentation                 │
                     │ - Dear ImGui Interface Overlay                  │
                     │ - Command Buffer Recording & Queue Submit       │
                     └────────────────────────┬────────────────────────┘
                                              │
                                              ▼
                  ┌────────────────────────────────────────────────────────┐
                  │      Path Tracing Budget: 15.0 ms - 15.5 ms            │
                  │      Managed by 3-Axis Dynamic Quality Governor        │
                  └────────────────────────────────────────────────────────┘
```

---

## 2. The 3 Orthogonal Control Axes

The fundamental problem of dynamic sample rate scaling in path tracing is **the discrete integer jump problem**: stepping from $1 \to 2\text{ SPP}$ jumps ray tracing workload by $+100\%$, and $2 \to 3\text{ SPP}$ increases workload by $+50\%$. A single integer knob is too coarse, causing controllers to oscillate ("hunt") across the budget boundary.

Pathways solves this by combining three orthogonal scaling axes into a continuous quality curve:

```
                          ▲ [Axis 1: Spatiotemporal Checkerboard]
                          │ (0.5 SPP Steps: Pixel Density)
                          │
                          │        / [Axis 2: Dual-GPU Allocation]
                          │       /    (Device Sample Balancing)
                          │      /
                          │     /
                          │    /
                          │   /
                          │  /
                          │ /
                          └──────────────────────────► [Axis 3: Dynamic Bounce Depth]
                                                          (Continuous ~5-15% Vernier)
```

### Axis 1: Spatiotemporal Checkerboard Sampling (0.5 SPP Steps)

- **Mechanism:** On any given frame, exactly 50% of the pixels launch ray paths based on a spatial checkerboard parity pattern:
  $$\text{parity} = (x + y + \text{frameIndex}) \pmod 2$$
- **Hardware Compaction (Wave32 Coherence):** A naive `if (((x+y)&1) != parity) return;` shader pattern idles 50% of SIMD lanes, forfeiting traversal throughput. Pathways dispatches a compacted grid:
  $$\text{Dispatch Dimensions} = \left(\frac{W}{2}, H\right)$$
  The ray generation shader mathematically unpacks coordinates:
  ```glsl
  ivec2 launchID = ivec2(gl_LaunchIDEXT.xy);
  int x = launchID.x * 2 + ((launchID.y + frameParity) & 1);
  int y = launchID.y;
  ivec2 pixelCoord = ivec2(x, y);
  ```
  Every lane in every Wave32 wavefront executes active ray traversal, yielding a true $2.0\times$ speedup per checkerboard pass.
- **Granularity:** Allows half-integer steps: **0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0 SPP**.

### Axis 2: Dual-GPU Sample Parallel Balancing (Device Partitioning)

- **Challenge:** If GPU 0 runs 1 SPP and GPU 1 runs 2 SPP on full frames, GPU 0 completes in ~4.0 ms and sits completely idle waiting for GPU 1 to finish at ~8.0 ms. This wastes 50% of GPU 0's compute capacity.
- **Solution (Dual-GPU Balanced Interleaving):** Both GPUs execute identical workloads to achieve target odd sample counts:
  - To achieve **3.0 SPP total**:
    - GPU 0 (Primary): Traces 1 full pass (1.0 SPP) + Parity A checkerboard (0.5 SPP) $\implies$ **1.5 SPP total work**.
    - GPU 1 (Secondary): Traces 1 full pass (1.0 SPP) + Parity B checkerboard (0.5 SPP) $\implies$ **1.5 SPP total work**.
    - Both GPUs run for the exact same duration ($\sim 6.0\text{ ms}$). **Zero GPU idle time.**
  - **Display Overhead Compensation:** GPU 0 handles swapchain acquisition, Dear ImGui overlay rendering, ACES tonemapping, and presentation (~0.8 ms total). The governor slightly skews the ray tracing workload (e.g. GPU 0 traces 1.3 SPP while GPU 1 traces 1.5 SPP) so both GPUs finish concurrently at the exact same millisecond.

### Axis 3: Dynamic Bounce Depth (Continuous Analog Vernier)

- **Mechanism:** In unidirectional path tracing, rays terminate early via Russian Roulette and luminance energy thresholding (`lum < 0.001` in `raytrace.rgen`).
- **Cost Distribution per Bounce:**
  - Primary Hit (Bounce 0): $100\%$ base cost.
  - Secondary Bounce (Bounce 1): $\sim 40\%$ of rays active.
  - Tertiary Bounce (Bounce 2): $\sim 15\%$ of rays active.
  - Quaternary Bounce (Bounce 3): $\sim 5\%$ of rays active.
  - Bounces 4–8: $\sim 1\text{--}2\%$ of rays active.
- **Role in the Governor:** Modulating `max_bounces` between 2 and 8 scales total frame time in continuous $\sim 5\%\text{--}15\%$ increments. It acts as an analog vernier dial, smoothing out the steps between SPP tiers and absorbing remaining idle headroom right up to the 16.67 ms presentation boundary.

---

## 3. The Continuous Performance Matrix (4K Native Dual-GPU)

The following matrix models how combinations of (SPP Tier, Multi-GPU Split, Bounce Depth) form a monotonic, continuous scaling curve targeting a **60 FPS (16.67 ms total / 15.5 ms RT)** budget on `DragonAttenuation`:

| Tier | Effective SPP | Multi-GPU Split (GPU0 / GPU1) | Max Bounces | Workload Multiplier | Estimated Frame Time | Headroom to 15.5 ms | Status |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0** | **0.50 SPP** | 0.25 SPP / 0.25 SPP (Quarter) | 2 | 0.38x | ~1.6 ms | +13.9 ms | Extreme Low-Power |
| **1** | **1.00 SPP** | 0.50 SPP / 0.50 SPP (Checker) | 2 | 0.75x | ~3.1 ms | +12.4 ms | High Headroom |
| **2** | **1.00 SPP** | 0.50 SPP / 0.50 SPP (Checker) | 4 | 1.00x | ~4.1 ms | +11.4 ms | Current Baseline |
| **3** | **1.00 SPP** | 0.50 SPP / 0.50 SPP (Checker) | 8 | 1.15x | ~4.7 ms | +10.8 ms | Headroom Available |
| **4** | **1.50 SPP** | 0.75 SPP / 0.75 SPP | 2 | 1.12x | ~4.6 ms | +10.9 ms | Headroom Available |
| **5** | **1.50 SPP** | 0.75 SPP / 0.75 SPP | 4 | 1.50x | ~6.1 ms | +9.4 ms | Headroom Available |
| **6** | **2.00 SPP** | 1.00 SPP / 1.00 SPP (Sample) | 2 | 1.50x | ~6.1 ms | +9.4 ms | Headroom Available |
| **7** | **2.00 SPP** | 1.00 SPP / 1.00 SPP (Sample) | 4 | 2.00x | ~8.2 ms | +7.3 ms | Headroom Available |
| **8** | **2.00 SPP** | 1.00 SPP / 1.00 SPP (Sample) | 8 | 2.30x | ~9.4 ms | +6.1 ms | Headroom Available |
| **9** | **2.50 SPP** | 1.25 SPP / 1.25 SPP | 4 | 2.50x | ~10.3 ms | +5.2 ms | Headroom Available |
| **10** | **3.00 SPP** | 1.50 SPP / 1.50 SPP | 4 | 3.00x | ~12.3 ms | +3.2 ms | Headroom Available |
| **11** | **3.00 SPP** | 1.50 SPP / 1.50 SPP | 6 | 3.25x | ~13.3 ms | +2.2 ms | Near Target |
| **12** | **3.50 SPP** | 1.75 SPP / 1.75 SPP | 4 | 3.50x | ~14.3 ms | +1.2 ms | Near Target |
| **13** | **3.50 SPP** | 1.75 SPP / 1.75 SPP | 6 | 3.75x | ~15.4 ms | +0.1 ms | **Target Budget Hit** |
| **14** | **4.00 SPP** | 2.00 SPP / 2.00 SPP | 4 | 4.00x | ~16.4 ms | -0.9 ms | Slight Overshoot |

*Result:* The governor automatically converges on **Tier 13 (3.5 SPP at 6 bounces)**, soaking up 99.3% of the allowable 15.5 ms ray tracing budget, rendering 3.5x fewer noise artifacts per frame without dropping a single frame from the 60 FPS presentation cadence.

---

## 4. Control Architecture: The Dynamic Governor Engine

```
┌────────────────────────────────────────────────────────────────────────┐
│            1. Hardware Sensor: GPU Timestamp Queries                  │
│    vkGetQueryPoolResults -> gpuRtMs, secGpuMs, tonemapMs (ns precision)│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            2. Damped Exponential Moving Average (EMA)                  │
│               t_avg = alpha * t_current + (1 - alpha) * t_avg          │
│               alpha = 0.10 (smooths frame-to-frame BVH jitter)         │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            3. Headroom Evaluation vs Target Budget                     │
│               Delta = TargetRTBudget (15.5 ms) - t_avg                 │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
         ┌──────────────────────────┼──────────────────────────┐
         │                          │                          │
         ▼                          ▼                          ▼
 [Delta < -0.8 ms]          [|Delta| <= 1.2 ms]        [Delta > +2.5 ms]
 Emergency Downgrade        Vernier Fine-Tuning        Conservative Upgrade
 - Immediate Tier -1        - Adjust max_bounces       - Check cooldown timer
 - Bypass cooldown            (+/- 1 bounce)             (requires 20 frames)
 - Protects 60 FPS          - Locks SPP tier           - Step Tier +1
         │                          │                          │
         └──────────────────────────┼──────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            4. Zero-Overhead Hardware Dispatch Execution                │
│    - Update CameraUniform (spp, maxBounces) -> memcpy host UBO         │
│    - Update Push Constants (tileOffsetX, tileOffsetY, accumulate)      │
│    - Dispatch TraceRaysKHR (Zero pipeline rebuilds, zero stalls)       │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│            5. High-Precision Frame Pacer & Swapchain Present           │
│    - Sleep until target timestamp: std::this_thread::sleep_until()     │
│    - Guarantees rock-solid 16.667 ms presentation pacing               │
└────────────────────────────────────────────────────────────────────────┘
```

### Safety & Stability Rules:
1. **Asymmetric Cooldowns:**
   - **Downgrade:** Zero frames cooldown (triggers immediately upon detecting any frame exceeding $0.95 \times \text{Budget}$ to prevent UI lag or presentation stutter).
   - **Upgrade:** Strict 20-frame cooldown (prevents rapid oscillation when the camera moves through dynamic lighting or geometry transitions).
2. **Camera Motion State Awareness:**
   - During continuous camera motion, the governor prioritizes **SPP over Bounces** (1 bounce less, 1 SPP more) to maximize anti-aliasing and specular cleanliness.
   - When the camera is stationary, the governor allows progressive accumulation to take over, shifting headroom into **higher bounce depths** to resolve long-tail diffuse and specular transport.

---

## 5. Implementation Roadmap & Technical Specifications

### Phase 1: Quality Governor Core Class (`core/QualityGovernor.hpp`, `core/QualityGovernor.cpp`)

Create a standalone governor class decoupled from Vulkan handles to facilitate deterministic testing:

```cpp
namespace pathways {

struct GovernorConfig {
    uint32_t targetFps = 60;          // 0 = uncapped
    bool enabled = false;             // Toggle dynamic regulation
    float targetBudgetMs = 15.2f;     // Path tracing budget (ms)
    uint32_t minSpp = 1;
    uint32_t maxSpp = 16;
    uint32_t minBounces = 2;
    uint32_t maxBounces = 8;
    bool enableCheckerboard = true;
    bool enableDualGpuBalance = true;
};

struct GovernorState {
    uint32_t currentSpp = 1;
    uint32_t currentBounces = 4;
    bool checkerboardActive = false;
    uint32_t primSpp = 1;
    uint32_t secSpp = 1;
    float filteredRtMs = 0.0f;
    float headroomPercent = 0.0f;
};

class QualityGovernor {
public:
    void init(const GovernorConfig& cfg);
    void update(float measuredRtMs, float fixedOverheadMs, bool cameraMoving);
    const GovernorState& getState() const { return m_state; }
    void setTargetFps(uint32_t fps);

private:
    GovernorConfig m_config;
    GovernorState m_state;
    float m_emaRtMs = 0.0f;
    uint32_t m_cooldownFrames = 0;
};

} // namespace pathways
```

### Phase 2: Compact Checkerboard Ray Generation (`shaders/rt/raytrace.rgen`)

Update `raytrace.rgen` push constants to support fractional 0.5 SPP checkerboard dispatches:
- Add `uint checkerboardMode` (0 = Full frame, 1 = Parity A, 2 = Parity B).
- In `RTPipeline::traceRays()`, when checkerboard mode is active:
  - Set dispatch width to `(width + 1) / 2`.
  - In-shader remapping maps `gl_LaunchIDEXT` to the active checkerboard diamond without wave divergence.

### Phase 3: Dual-GPU Load Compensator (`mgpu/MultiGpuManager.cpp`)

Extend `launchSecondaryWork()` in `MultiGpuManager.cpp`:
- Dynamically assign asymmetric sample allocations:
  ```cpp
  uint32_t primSpp = governor.getState().primSpp;
  uint32_t secSpp = governor.getState().secSpp;
  uboPrimary.spp = primSpp;
  uboSecondary.spp = secSpp;
  ```
- In `accum_merge.comp`, update the sample accumulation weighting to sum `primSpp + secSpp` correctly without radiance bleaching or dimming.

### Phase 4: High-Precision CPU Frame Pacer (`core/Engine.cpp`)

Implement high-resolution pacing before swapchain presentation in `Engine::renderFrame()`:

```cpp
if (m_config.target_fps > 0) {
    auto targetDuration = std::chrono::duration<double, std::milli>(1000.0 / m_config.target_fps);
    auto nextFrameTime = m_lastFrameTimestamp + targetDuration;
    
    // Precise hybrid sleep + spin-lock wait
    auto now = std::chrono::high_resolution_clock::now();
    if (now < nextFrameTime) {
        auto sleepTime = nextFrameTime - now - std::chrono::microseconds(200);
        if (sleepTime > std::chrono::microseconds(0)) {
            std::this_thread::sleep_for(sleepTime);
        }
        while (std::chrono::high_resolution_clock::now() < nextFrameTime) {
            #if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
            #endif
        }
    }
    m_lastFrameTimestamp = std::chrono::high_resolution_clock::now();
}
```

### Phase 5: Dear ImGui Overlay & CLI Integration (`ui/GuiManager.cpp`, `core/Config.cpp`)

1. **CLI Flags:**
   - `--target-fps <int>` (Default: 0 = uncapped).
   - `--adaptive-spp` (Enable 3-axis quality governor).
   - `--min-spp <int>`, `--max-spp <int>`.
   - `--min-bounces <int>`, `--max-bounces <int>`.
2. **ImGui Telemetry Panel:**
   - Interactive Target FPS slider (`Off`, `30`, `60`, `90`, `120`, `144`).
   - Dynamic Governor status bar:
     ```
     [Target: 60 FPS (16.67 ms)] | Status: LOCKED
     Current Workload: 3.5 SPP (Checkerboard) | 6 Bounces
     Dual-GPU Distribution: GPU 0 [1.75 SPP] | GPU 1 [1.75 SPP]
     GPU RT Duration: 14.82 ms | Tonemap/UI: 0.71 ms | Headroom: +1.14 ms (+7.3%)
     ```

---

## 6. Verification & Validation Protocol

The feature must pass the following rigorous test cases before merging:

1. **Static Convergence Invariance:**
   - When stationary, total accumulated radiance must match fixed 1 SPP ground truth exactly without color banding, scaling bias, or brightness distortion. Verified via `compare_images.py` ($\text{PSNR} > 60\text{ dB}$, $\text{MAE} < 0.05$).
2. **Camera Motion Stress Test:**
   - Continuous mouse rotation (`--camera-motion`) must hold frame times within $\pm 0.5\text{ ms}$ of the 16.67 ms target line with zero dropped presentation frames.
3. **Dynamic Scene Transition Robustness:**
   - Switching scenes live between `DamagedHelmet` (lightweight), `DragonAttenuation` (medium), and `living-room` (heavy) must adapt within $<30$ frames without GPU crashes or pipeline stalls.
4. **Vulkan Validation Cleanliness:**
   - Zero Vulkan validation layer warnings or memory barriers errors under standard headless automated test suites (`scripts/run_headless_tests.sh`).
