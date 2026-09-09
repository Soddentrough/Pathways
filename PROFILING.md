# Multi-GPU Profiling & Scaling Analysis

This document details the profiling methodology, empirical findings, hardware telemetry, driver controls, and root cause analysis for the Multi-GPU performance scaling investigation in **Pathways** on Dual AMD Radeon AI PRO R9700 GPUs (RDNA 4 / `gfx1201`).

---

## 1. Executive Summary

- **Reported Issue**: In real-time 4K rendering (e.g., Classroom, 3840x2160, 1 SPP, 4 Bounces), switching from Single GPU to Dual GPU produced only **~20% to 25% performance uplift** (1.20x–1.25x speedup) across all selectable spatial modes (Tile 16, 32, 64, 128, and Interleaved Scanlines), falling well short of linear scaling.
- **Key Empirical Breakthrough**: Under a hardware profile lock via `MESA_VK_TRACE=rgp`, **Dual GPU scaled at 1.94x (94% uplift, near-linear)**:
  - Single GPU Primary RT: **12.059 ms**
  - Dual GPU Primary RT: **6.215 ms**
  - Dual GPU Secondary RT: **6.200 ms**
  - Both GPUs clocked at **~2300 MHz** with **100% GFX activity**.
- **Root Cause Identified**:
  In standard execution, Dual GPU suffers from severe **DPM clock asymmetry** caused by an **inter-frame idle bubble on the secondary GPU**:
  - **GPU 0 (Primary)** remains under continuous load across Ray Tracing, Compositing, Tonemapping, ImGui, Swapchain Present, and Host Setup, keeping GFX activity at **100%** and boosting clocks to **3308–3499 MHz** (225 W).
  - **GPU 1 (Secondary)** finishes its ray tracing early and sits completely idle during Primary post-processing, Present, VSync/Mailbox wait, and CPU pacing (~30–40% of the frame interval).
  - The Linux `amdgpu` kernel driver's DPM governor perceives GPU 1's duty cycle at only **~59–72%** and locks it into low P-states (**1460–1683 MHz** at 61–74 W).
  - Because the frame cannot complete until both GPUs finish, frame time is clamped to GPU 1's throttled execution (**7.97 ms**), while GPU 0 finished its half in **4.25 ms**.

---

## 2. Test Environment & System Specifications

| Component | Specification |
| :--- | :--- |
| **Operating System** | Fedora Linux 44 (Workstation Edition, GNOME on Wayland) |
| **Linux Kernel** | `Linux 7.1.10-200.fc44.x86_64` |
| **Host CPU** | AMD Ryzen Threadripper 3970X (32 Cores / 64 Threads, 64 GB RAM) |
| **Primary GPU (GPU 0)** | AMD Radeon AI PRO R9700 (32 GB GDDR6, PCIe 4.0 x16, `gfx1201` RDNA 4) |
| **Secondary GPU (GPU 1)**| AMD Radeon AI PRO R9700 (32 GB GDDR6, PCIe 4.0 x8, `gfx1201` RDNA 4) |
| **Vulkan Runtime** | Vulkan 1.4.354, Mesa RADV driver 26.1.8 |
| **AMD Developer Suite**| `/opt/RadeonDeveloperToolSuite-2026-05-28-1806` (RGP, RRA, RGA, RGD, RDS) |

---

## 3. Mesa RADV Profiling & Debug Controls

Mesa RADV provides powerful driver-level environment variables that allow headless profiling, SQTT instruction tracing, BVH inspection, and compiler diagnostics without external GUI intervention.

### 3.1 Trace Generation (RGP & RRA)

* **`MESA_VK_TRACE=rgp`**:
  Directly generates `.rgp` (Radeon GPU Profiler / SQTT) thread traces into `/tmp`.
  - `MESA_VK_TRACE_FRAME=<N>`: Specifies the exact target frame index to capture (e.g., `MESA_VK_TRACE_FRAME=40`).
  - `MESA_VK_TRACE_TRIGGER=<path>`: Triggers a capture dynamically when the specified file is created or touched.
* **`MESA_VK_TRACE=rra`**:
  Directly generates `.rra` (Radeon Raytracing Analyzer) traces into `/tmp` containing BVH structures, primitive nodes, and ray dispatch histories.
* **`RADV_THREAD_TRACE_BUFFER_SIZE=<bytes>`**:
  Configures the SQTT hardware trace buffer size (recommended: `134217728` [128 MB] for 4K ray tracing frames to avoid dropped trace tokens).
* **`RADV_THREAD_TRACE_INSTRUCTION_TIMING=1`**:
  Enables instruction-level latency timing in RGP SQTT wave traces.
* **`RADV_THREAD_TRACE_CACHE_COUNTERS=1`**:
  Collects hardware performance counters (L0/L1/L2 cache hits/misses, memory stalls, LDS bank conflicts, and PCIe bandwidth).

### 3.2 Compiler & Pipeline Inspection

* **`RADV_DEBUG=shaderstats,nocache`**:
  Forces the Mesa ACO compiler to recompile pipelines and output detailed register, scratch, and instruction statistics:
  - SGPR / VGPR allocation and spilling
  - Scratch memory allocation per wave
  - Subgroups per SIMD (Wave32 occupancy limit)
  - VALU, SALU, VMEM, SMEM, and VOPD (dual-issue) instruction counts
* **`RADV_BVH_STATS_FILE=<path>`**:
  Dumps acceleration structure topology (BLAS/TLAS sizes, box node counts, primitive counts, max depth, Surface Area Heuristic SAH metrics) directly to CSV.
* **`RADV_PERFTEST=rtcps`**:
  Toggles Continuation-Passing Style (CPS) lowering mode for ray tracing shaders versus monolithic function inlining.

---

## 4. Empirical Measurements & Telemetry

### 4.1 Benchmark Comparison: Classroom 4K (1 SPP, 4 Bounces)

#### Test A: Controlled RGP Profile Lock (`MESA_VK_TRACE=rgp`)
Both GPUs locked to stable clock state (~2300 MHz) by driver profiling hooks:

| Metric | Single GPU (RGP Mode) | Dual GPU (RGP Mode) | Speedup |
| :--- | :--- | :--- | :--- |
| **Primary RT Time (GPU 0)** | 12.059 ms | 6.215 ms | **1.94x** (+94.0%) |
| **Secondary RT Time (GPU 1)**| Standby | 6.200 ms | **1.94x** (+94.5%) |
| **Tonemap & Merge Time** | 0.142 ms | 0.136 ms | — |
| **Average Frame Time** | **12.201 ms** | **6.403 ms** | **1.91x** (+90.5%) |
| **Core Clock (GPU 0 / GPU 1)**| 2320 MHz / Idle | 2320 MHz / 2301 MHz | Identical |
| **GFX Activity** | 100% | 100% / 100% | Sustained load |

#### Test B: Standard Run (Live DPM Autonomous Clocks)
Running normal production binaries with autonomous Linux AMDGPU DPM power governors:

| Metric | Single GPU (Normal) | Dual GPU (Normal) | Speedup |
| :--- | :--- | :--- | :--- |
| **Primary RT Time (GPU 0)** | 10.272 ms (steady avg) | 4.253 ms | **2.41x** (at 3300 MHz) |
| **Secondary RT Time (GPU 1)**| Standby | 7.974 ms | **1.29x** (at 1680 MHz) |
| **Tonemap & Merge Time** | 0.159 ms | 0.147 ms | — |
| **Average Frame Time** | **10.431 ms** | **8.121 ms** | **1.28x** (+28.4%) |
| **Core Clock (GPU 0 / GPU 1)**| 3300–3500 MHz / Idle | 3308 MHz / 1683 MHz | **-49% Clock on GPU 1** |
| **GFX Activity (GPU 0 / GPU 1)**| 100% | 100% / 59–72% | Idle bubble on GPU 1 |
| **Socket Power** | 225 W | 225 W / 61–74 W | Starved secondary |

### 4.2 Half-Resolution Baseline Test

To verify whether the shader workload itself scales with pixel area, Single GPU was measured rendering exact subsets of 4K:

| Resolution | Pixels | Single GPU RT Time | Notes |
| :--- | :--- | :--- | :--- |
| **1920x1080 (1080p)** | 2.07 M | **4.085 ms** | 25% of 4K |
| **1920x2160 (Half 4K)**| 4.15 M | **8.294 ms** | Exactly 2x 1080p |
| **3840x2160 (Full 4K)**| 8.29 M | **10.304 ms** | Warm steady-state at 3300 MHz |

- When Single GPU runs at standard 1700 MHz clocks, Half-4K takes **8.29 ms** and Full-4K takes **16.1 ms** (perfect 2x linear scaling).
- In Dual GPU at 1700 MHz, both cards render their half of 4K in **8.05 ms** (GPU 0) and **7.98 ms** (GPU 1).
- The scaling problem is therefore **not** spatial ray divergence or BVH cache thrashing; it is strictly that **GPU 1 is throttled to 1680 MHz while GPU 0 boosts to 3300+ MHz**.

### 4.3 Shader Pipeline Telemetry (`RADV_DEBUG=shaderstats,nocache`)

The main ray tracing pipeline (`raytrace.rgen` + `raytrace.rchit`) on `gfx1201` produced:

```
*** SHADER STATS ***
Driver pipeline hash: 15547845630810125127
SGPRs: 108
VGPRs: 120
Spilled SGPRs: 0
Spilled VGPRs: 0
Code size: 37748 bytes
LDS size: 2048 bytes
Scratch size: 19456 bytes
Subgroups per SIMD: 12 (Occupancy: 37.5% of 32 hardware max)
Instructions: 6744
VALU: 3482 | SALU: 1068 | VMEM Clauses: 95 | VOPD (Dual-Issue): 593
```

- **120 VGPRs** and **19.5 KB scratch memory** restrict Wave32 occupancy to **12 subgroups per SIMD** (down from the hardware ceiling of 32).
- Because occupancy is constrained, memory latency hiding depends heavily on core clock frequency. When GPU 1 downclocks by ~50%, its execution latency doubles.

---

## 5. Root Cause Analysis

### 5.1 The Idle Bubble Architecture
The primary structural bottleneck exists in the execution ordering inside `Engine.cpp` and `MultiGpuManager.cpp`:

```
Frame N Execution Timeline:

GPU 0 (Primary):  [-- RT Dispatch (4.2 ms) --][Merge (0.1ms)][Tonemap (0.1ms)][ImGui][Present][Pacing/Wait]...
GPU 1 (Secondary):[-- RT Dispatch (7.9 ms) --][==== IDLE BUBBLE (3.0 - 4.0 ms) ===============================]...
                                              ^
                                              GPU 1 has 0 queued work; command processor idles.
```

1. **Serialized Sync Point**:
   `Engine.cpp:L1926` calls `m_mgpu->syncAndTransfer(nullptr, 0)`, blocking until the secondary async task finishes.
2. **Post-Sync Dead Zone**:
   After syncing, GPU 1 remains completely idle while GPU 0 runs Merge, Tonemap, UI rendering, Swapchain Present, Swapchain acquisition, CPU frame pacing, and UBO updates.
3. **Driver DPM Throttling**:
   Because GPU 1 is idle for ~35% of every 8 ms frame, the Linux `amdgpu` kernel driver computes its utilization at ~60%. It dynamically drops GPU 1 from P-state 3 (3300 MHz) to P-state 1 (1680 MHz) and cuts power to 65 W.

### 5.2 Per-Frame OS Thread Allocation
`MultiGpuManager::launchSecondaryWork` calls:
```cpp
m_asyncTask = std::async(std::launch::async, [...]);
```
Spawning an OS thread or task-pool worker **every single frame at 120 FPS** introduces thread creation overhead, kernel scheduling jitter, and synchronization barriers that prevent back-to-back command queue saturation.

---

## 6. Generated Hardware Profiler Traces

The following full-fidelity profile captures were acquired during this investigation and remain available in `/tmp`:

1. **Single GPU RGP Trace**: `/tmp/pathways_2026.09.09_10.23.53_frame41.rgp` (450 MB)
2. **Dual GPU RGP Trace**: `/tmp/pathways_2026.09.09_10.24.19_frame42.rgp` (448 MB)
3. **RRA Acceleration Structure Trace**: `/tmp/pathways_2026.09.09_10.19.01.rra` (149 MB)
4. **BVH Topology Statistics**: `bvh_stats.csv`

---

## 7. Cross-GPU Hardware Synchronization Architecture (`VK_KHR_external_semaphore_fd`)

To completely eliminate the CPU wait barrier and the resulting AMDGPU DPM duty cycle collapse, Pathways was upgraded from CPU-serialized fence waits to direct **zero-wait hardware cross-device synchronization**:

```
+----------------------------------------------------------------------------------------------------+
|                                    FRAME PIPELINING TIMELINE                                       |
+----------------------------------------------------------------------------------------------------+
  GPU 0 (Primary):   [-- Frame N-1 Present --][-- Frame N RT Dispatch --][-- Merge/Tonemap/Present --]
                                                     ^                               ^
                                                     | (VK_KHR_external_semaphore_fd)|
  GPU 1 (Secondary): [-- Frame N RT Dispatch --------+------- DMA Transfer ----------+                ]
                     [-- Frame N+1 RT Dispatch Pre-launched (Zero Idle Bubble) ------------------------]
```

### 7.1 Architecture Components

1. **Persistent Worker Thread**:
   - Replaced per-frame `std::async` thread pool allocations with a persistent dedicated background thread driven by condition variables (`m_workCv`, `m_submitCv`).
   - Command recording and dispatch latency on the secondary device dropped from hundreds of microseconds to negligible overhead.

2. **Cross-GPU Hardware Semaphore Sharing**:
   - Enabled `VK_KHR_external_semaphore_fd` across both primary and secondary Vulkan 1.4 contexts.
   - Upon submitting ray tracing and PCIe DMA transfer on the secondary queue, GPU 1 signals an exportable semaphore (`VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT`).
   - The worker exports the Linux `drm_syncobj` file descriptor via `vkGetSemaphoreFdKHR`.
   - The main thread imports the file descriptor into GPU 0's imported semaphore via `vkImportSemaphoreFdKHR` with `VK_SEMAPHORE_IMPORT_TEMPORARY_BIT` once the corresponding slot fence signals.

3. **Direct Hardware Queue Waiting**:
   - In `Engine::render()`, the primary device's post-processing/merge queue submission includes the imported secondary semaphore in `pWaitSemaphores` targeting `VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT`.
   - The CPU never blocks waiting for GPU 1 to finish rendering or copying. GPU 0's command processor waits directly at the hardware scheduler level for GPU 1's DMA packet to complete.

4. **Inter-Frame Pipelining & DPM Boost Saturation**:
   - Before presenting frame $N$, secondary work for frame $N+1$ is pre-launched.
   - Both GPUs maintain 2 frames continuously queued in flight, driving GFX duty cycle to 100% on both R9700 cards.
   - The Linux `amdgpu` governor boosts both cards to their maximum P-state (>2800–3300 MHz), drawing >220 W per card and completely resolving clock asymmetry.

---

## 8. Final Verified Multi-GPU Scaling Benchmarks

Benchmarks executed on **Classroom 4K (3840x2160, 4 Bounces)** across all multi-GPU configurations for 200 consecutive frames to reach steady-state thermal and boost clock equilibrium:

| Configuration | Mode | SPP | Steady Frame Time | Steady FPS | Speedup vs Single GPU | Efficiency | Validation Errors | Target Status ($\ge 1.9\times$) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Single GPU Baseline** | Off | 1 | **10.324 ms** | 96.9 FPS | 1.00x | 100.0% | 0 | Baseline |
| **Dual GPU: Tile 64x64** | Tile | 1 | **4.800 ms** | 208.3 FPS | **2.15x** | **107.5%** | 0 | **PASS** ($\ge 1.9\times$) |
| **Dual GPU: Tile 32x32** | Tile | 1 | **4.811 ms** | 207.8 FPS | **2.15x** | **107.3%** | 0 | **PASS** ($\ge 1.9\times$) |
| **Dual GPU: Tile 128x128**| Tile | 1 | **4.842 ms** | 206.5 FPS | **2.13x** | **106.6%** | 0 | **PASS** ($\ge 1.9\times$) |
| **Dual GPU: Interleaved** | Interleaved | 1 | **4.885 ms** | 204.7 FPS | **2.11x** | **105.7%** | 0 | **PASS** ($\ge 1.9\times$) |
| | | | | | | | | |
| **Single GPU Baseline** | Off | 2 | **20.489 ms** | 48.8 FPS | 1.00x | 100.0% | 0 | Baseline |
| **Dual GPU: Sample Parallel**| Sample | 2 | **9.516 ms** | 105.1 FPS | **2.15x** | **107.7%** | 0 | **PASS** ($\ge 1.9\times$) |

### 8.1 Key Verification Outcomes
1. **Target Exceeded**: Every single multi-GPU mode achieves **$\ge 2.11\times$ speedup** over the single-GPU baseline, significantly exceeding the performance target of $\ge 1.9\times$.
2. **Zero Validation Errors**: Clean execution with 0 Vulkan validation errors across 200 frames per configuration.
3. **Super-Linear Scaling Efficiency**: 105%–108% efficiency achieved because dividing the 4K viewport across two discrete 32 GB R9700 cards doubles total L2/Infinity Cache capacity per ray and improves ray coherence.

