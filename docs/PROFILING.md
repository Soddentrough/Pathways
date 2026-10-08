# Performance Profiling & Hardware Telemetry Guide

This document defines the comprehensive performance profiling methodology, telemetry tools, and diagnostic workflows for **Pathways** on AMD RDNA hardware (specifically targeting the Dual AMD Radeon AI PRO R9700 / `gfx1201` architecture on Linux).

---

## 1. The Three-Tier Profiling Strategy

Diagnosing rendering and compute performance bottlenecks requires analyzing the system at three distinct abstraction layers:

```
+----------------------------------------------------------------------------------------------------+
|                                    PROFILING ABSTRACTION LAYERS                                    |
+----------------------------------------------------------------------------------------------------+
  1. System & Board Telemetry (`amd-smi`)
     - Scope: Macro hardware limits, clocks, power, thermals, memory controllers, PCIe bus, multi-GPU
     - Answers: "Is the GPU compute-bound, memory-bandwidth-bound, PCIe-bound, CPU-starved, or throttled?"

  2. Driver & API Diagnostics (`RADV_PERFEST`, `RADV_DEBUG`)
     - Scope: Mesa RADV driver implementation, ACO compiler stats, API validation, shader pipeline layouts
     - Answers: "Are shaders spilling registers? Are render targets uncompressed? Is the API used optimally?"

  3. GPU Microarchitecture & Instruction Tracing (RGP, RRA, RGA)
     - Scope: Wavefront occupancy, SIMD lane divergence, VALU/SALU/VMEM latency, LDS bank conflicts, BVH
     - Answers: "Which specific instruction, loop, or memory access pattern is stalling the compute units?"
+----------------------------------------------------------------------------------------------------+
```

---

## 2. Hardware Bottleneck Identification with AMD-SMI

While **Radeon GPU Profiler (RGP)** provides microsecond-level instruction traces and **`RADV_PERFEST`** flags driver-level optimization gaps, `/opt/rocm/core-10.0/bin/amd-smi` provides real-time operational telemetry from the kernel driver (`amdgpu`) and hardware System Management Unit (SMU).

### Quick Telemetry Commands

```bash
# Live dashboard of clocks, power, thermals, engine %, memory %:
/opt/rocm/core-10.0/bin/amd-smi monitor -u -m -p -t -v -w 1

# Comprehensive hardware snapshot (JSON or tabular):
/opt/rocm/core-10.0/bin/amd-smi metric -g all
/opt/rocm/core-10.0/bin/amd-smi metric --json

# Process-level resource and engine breakdown:
/opt/rocm/core-10.0/bin/amd-smi process -G -e

# Multi-GPU PCIe & P2P topology:
/opt/rocm/core-10.0/bin/amd-smi topology -a -t -d -b
```

---

### 2.1 Engine Activity: Distinguishing CPU, Memory & Compute Bottlenecks

Inspect using:
```bash
/opt/rocm/core-10.0/bin/amd-smi metric -u
# or
/opt/rocm/core-10.0/bin/amd-smi monitor -u -m -w 1
```

| Metric | Telemetry Pattern | Root Cause / Diagnosis | Actionable Next Step |
| :--- | :--- | :--- | :--- |
| **`GFX_ACTIVITY`** | Low (<85–90%) during uncapped frames | **CPU Bottleneck or Host Stall**: GPU is starved waiting for draw/dispatch submission, host sync (`vkWaitForFences`), descriptor updates, or swapchain presentation (`vkQueuePresentKHR`). | Profile host CPU loop; verify async worker thread dispatch; ensure double/triple buffering in-flight frames. |
| **`GFX_ACTIVITY`** | Sustained ~100% | **GPU-Bound**: Workload fully saturates GPU execution resources. | Inspect `UMC_ACTIVITY` and RGP wave occupancy to isolate ALU vs. memory constraints. |
| **`UMC_ACTIVITY`** | High (>75–85%) with moderate GFX% | **External VRAM Bandwidth Bound**: Unified Memory Controller bus is saturated by off-chip GDDR6 traffic (textures, 4K ray payload queues, uncompressed framebuffers). | Compact payload structures (e.g. SoA queues), enable DCC / texture compression, or improve spatial cache locality. |
| **`UMC_ACTIVITY`** | Low (<30%) with ~100% GFX% | **Compute / ALU Bound**: Core shader execution units are saturated with math instructions, ray traversal loops, or transcendental operations. | Inspect ACO shader stats; optimize VALU instructions; check for loop unrolling and dual-issue VOPD opportunities. |
| **`VCN_ACTIVITY` / `DMA`** | Stalled or alternating | **Queue Serialization**: Async compute or DMA transfer queues are blocking behind graphics barriers rather than running concurrently. | Review `VkDependencyInfo` / pipeline barriers; decouple async transfer queues from the primary render queue. |

---

### 2.2 VRAM Subsystem, ReBAR & System Memory Spilling

Inspect using:
```bash
/opt/rocm/core-10.0/bin/amd-smi metric -m
```

* **`USED_VISIBLE_VRAM` vs. `TOTAL_VISIBLE_VRAM` (Host-Visible BAR)**:
  * When **Resizable BAR (Smart Access Memory)** is enabled, `TOTAL_VISIBLE_VRAM` matches the full device VRAM capacity (e.g., 32 GB on R9700).
  * If ReBAR is disabled in motherboard firmware, `TOTAL_VISIBLE_VRAM` drops to **256 MB**.
  * **Bottleneck**: If host-visible staging buffers, dynamic uniform buffers, or acceleration structure scratch spaces exceed 256 MB, the kernel driver must dynamically re-map or page allocations through system DRAM, causing severe periodic hitching.
* **`USED_GTT` (Graphics Translation Table / System RAM Spilling)**:
  * Measures GPU-accessible system RAM allocations over PCIe.
  * **Bottleneck**: If `USED_GTT` spikes during rendering, device-local VRAM has been oversubscribed. Any shader reading or writing GTT memory incurs massive PCIe roundtrip latency (~32–64 GB/s PCIe vs. ~640 GB/s GDDR6).

---

### 2.3 Dynamic Clock Scaling & DPM Performance States

Inspect using:
```bash
/opt/rocm/core-10.0/bin/amd-smi metric -c
```

* **`GFX_0 CLK` vs. `MAX_CLK`**:
  * On RDNA 4 (`gfx1201`), target boost clocks reach **2350–3300+ MHz**.
  * **Bottleneck (The Idle Bubble)**: If core clocks fluctuate down into low P-states (e.g. 500–1600 MHz) or enter `DEEP_SLEEP` during active execution, the driver's power governor is detecting intermittent GPU idle periods between frame submissions.
  * In Multi-GPU setups, if a secondary GPU finishes its partition early and waits on host synchronization, its duty cycle drops, causing the driver to lock it into a low P-state (halving its execution speed).
* **Fixing Clock Fluctuation for Benchmarking**:
  ```bash
  # Pin GPU DPM performance level to peak clocks during benchmarks:
  /opt/rocm/core-10.0/bin/amd-smi set -l high -g all
  # Return to auto governor post-benchmark:
  /opt/rocm/core-10.0/bin/amd-smi set -l auto -g all
  ```

---

### 2.4 Thermal and Power Throttling

Inspect using:
```bash
/opt/rocm/core-10.0/bin/amd-smi metric -p -t -v
```

* **`HOTSPOT` vs. `EDGE` Temperature**:
  * Modern AMD GPUs govern throttling based on the **`HOTSPOT`** junction (die center / compute clusters), which typically runs 15–25°C higher than `EDGE`.
  * If `HOTSPOT` reaches thermal thresholds (~100–105°C), clocks throttle down immediately even if `EDGE` reports a deceptive 65–70°C.
* **`SOCKET_POWER` vs. Power Cap**:
  * Indicates package power consumption in Watts. If sustained power equals the board limit (e.g., 225 W), the SMU clamps clock speeds to maintain package thermal dissipation limits (PPT).
* **Throttle Accumulators & Violations (`-v`)**:
  * Tracks cumulative throttle events:
    * `PPT_ACCUMULATED`: Power limit throttling.
    * `SOCKET_THERMAL_ACCUMULATED` / `VR_THERMAL_ACCUMULATED`: Thermal or voltage-regulator limit throttling.
    * `LOW_UTILIZATION_ACCUMULATED`: Clock reduction due to GPU starvation.

---

### 2.5 PCIe Link Health and Bus Saturation

Inspect using:
```bash
/opt/rocm/core-10.0/bin/amd-smi metric -P
```

* **`SPEED` and `WIDTH`**:
  * Verifies negotiated link configuration (e.g., `16 GT/s` / `WIDTH: 16` for PCIe 4.0 x16).
  * If a card silently negotiates down to `x8`, `x4`, or `2.5 GT/s` (PCIe 1.0) due to PCIe power saving, motherboard lane bifurcation, or physical slot degradation, host streaming and readback bandwidth drops proportionally.
* **`REPLAY_COUNT` and `NAK_SENT_COUNT`**:
  * Monitors PCIe physical layer packet retransmissions.
  * **Bottleneck**: Non-zero or rapidly incrementing replay counts indicate PCIe bus signal degradation (common with riser cables or high PCIe clock jitter), resulting in micro-stutters and transfer stalls.

---

### 2.6 Multi-GPU Interconnect & P2P Topology

Inspect using:
```bash
/opt/rocm/core-10.0/bin/amd-smi topology -a -t -d -b
```

* **Direct DMA (`-d`) & Accessibility (`-a`)**:
  * Validates whether GPU 0 and GPU 1 can perform direct Peer-to-Peer (P2P) DMA over the PCIe root complex or xGMI bridge without CPU memory staging.
  * If P2P DMA is unavailable (`False`), cross-GPU frame assembly must bounce through host RAM, consuming CPU memory bandwidth and introducing PCIe roundtrips.
* **Link Type (`-t`) & NUMA Bandwidth (`-b`)**:
  * Reports interconnect hops and maximum unidirectional bandwidth between devices.

---

### 2.7 Process Contention and Engine Sharing

Inspect using:
```bash
/opt/rocm/core-10.0/bin/amd-smi process -G -e
```

* Identifies all processes allocating VRAM or executing on graphics/compute engines.
* Ensures background processes (GNOME Wayland / `mutter`, web browsers, video decoders) do not contend for memory bandwidth or introduce scheduling jitter during benchmarking.

---

## 3. Driver Diagnostics: Mesa RADV Controls

Mesa RADV provides driver-level environment variables to diagnose compilation, memory layout, and execution pipeline issues:

### 3.1 Driver Performance Hints (`RADV_PERFTEST`)

| Variable | Target & Purpose | Bottleneck Diagnosed |
| :--- | :--- | :--- |
| **`RADV_PERFEST=1`** | Prints driver-level performance warnings to `stderr`. | Flags uncompressed DCC fallback, non-optimal image layouts, sub-optimal blits, and excessive descriptor updates. |
| **`RADV_PERFTEST=rtcps`** | Toggles Continuation-Passing Style (CPS) lowering for ray tracing. | Compares monolithic function inlining against CPS shader scheduling on complex ray-tracing pipelines. |
| **`RADV_PERFTEST=cswave32`** | Forces Wave32 execution for compute shaders. | Compares Wave32 vs. Wave64 register pressure and SIMD utilization. |
| **`RADV_PERFTEST=nogttspill`** | Restricts allocations strictly to VRAM. | Isolates whether performance drops are caused by transparent GTT system memory spilling. |

### 3.2 Compiler & Register Diagnostics (`RADV_DEBUG`)

```bash
# Dump ACO compiler statistics for all compiled pipelines:
RADV_DEBUG=shaderstats,nocache ./build/bin/pathways --headless --frames 10
```

Key compiler metrics to examine:
* **`VGPRs` & `SGPRs`**:
  * Target: $\le 64$ VGPRs for Wave32 to maintain 16+ waves per SIMD on RDNA 4.
  * If VGPRs exceed 104–128, wave occupancy drops to 8 or fewer subgroups per SIMD.
* **`Scratch Size (Spill)`**:
  * **Target: 0 Bytes**.
  * Any non-zero scratch allocation means VGPR pressure forced register contents to spill into VRAM scratch memory, introducing memory latency into ALU loops.
* **`VOPD (Dual-Issue)`**:
  * Measures RDNA dual-issue ALU instruction generation. Higher VOPD counts indicate optimal instruction scheduling by ACO.

---

## 4. Hardware Instruction & Ray Tracing Profilers (RDTS)

Tool binaries located at `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/`:

### 4.1 Radeon GPU Profiler (RGP) via Mesa RADV

Capture SQTT wave execution traces:
```bash
MESA_VK_TRACE=rgp \
MESA_VK_TRACE_FRAME=50 \
RADV_THREAD_TRACE_BUFFER_SIZE=134217728 \
RADV_THREAD_TRACE_INSTRUCTION_TIMING=1 \
RADV_THREAD_TRACE_CACHE_COUNTERS=1 \
./build/bin/pathways --headless --frames 60
```
Open the generated `/tmp/pathways_*.rgp` trace in `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonGPUProfiler`:
* **Wavefront Occupancy Tab**: Identify compute unit idling, pipeline barrier stalls, and queue bubbles.
* **Instruction Details Tab**: Pinpoint high-latency VMEM clauses, LDS bank conflicts, and branch divergence.

### 4.2 Radeon Raytracing Analyzer (RRA)

Capture acceleration structure and ray query traces:
```bash
MESA_VK_TRACE=rra \
MESA_VK_TRACE_FRAME=50 \
./build/bin/pathways --headless --frames 60
```
Open `/tmp/pathways_*.rra` in `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonRaytracingAnalyzer`:
* Inspect BLAS/TLAS box node counts, primitive surface area heuristics (SAH), and memory footprints.
* Analyze ray traversal depth and candidate testing costs.

### 4.3 Radeon GPU Analyzer (RGA)

Perform static offline ISA disassembly and VGPR auditing:
```bash
/opt/RadeonDeveloperToolSuite-2026-05-28-1806/rga \
  -s vk-spv-offline \
  --asic gfx1201 \
  --compute shaders/rt/wavefront_shade.comp.spv \
  -a output/rga_analysis.txt
```

---

## 5. Diagnostic Decision Matrix

Use this matrix to rapidly isolate the root cause when performance targets are missed:

| Observation / Symptom | Primary Telemetry Indicator | Secondary Diagnostic Tool | Root Cause | Remediation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| Low framerate, GPU clocks drop, power low (<100W) | `GFX_ACTIVITY` < 80% in `amd-smi` | CPU flamegraph / OS trace | **CPU-Bound or Sync Stall**: Host draw submission or fence wait blocking GPU. | Migrate to async workers; use Synchronization 2 (`vkQueueSubmit2`); pipeline multiple frames in flight. |
| GPU at 100% GFX, but frame time higher than expected | `UMC_ACTIVITY` > 80% in `amd-smi` | RGP Memory Stalls counter | **Memory Bandwidth Bottleneck**: Excessive GDDR6 traffic. | Structure-of-Arrays (SoA) ray payload compaction; enable DCC image compression; optimize cache locality. |
| GPU at 100% GFX, UMC% low (<30%), low framerate | `UMC_ACTIVITY` < 30%, `GFX%` 100% | `RADV_DEBUG=shaderstats` | **Compute / ALU Bottleneck**: Math or register pressure stall. | Reduce VGPR usage; eliminate scratch memory spilling; leverage Wave32 SIMD compaction. |
| Sudden frame hitching during buffer/texture updates | `USED_VISIBLE_VRAM` at 256 MB cap | `amd-smi metric -m` | **ReBAR Disabled**: Staging buffers spilling out of 256MB aperture. | Enable Resizable BAR in UEFI firmware; throttle concurrent host-visible staging allocations. |
| Massive frame drop (>50%) on high-resolution scenes | `USED_GTT` non-zero & increasing | `amd-smi metric -m` | **VRAM Oversubscription**: Textures or BVHs spilled to host RAM. | Implement texture streaming, mip clamping, or geometry LOD; verify allocation sizes. |
| Inconsistent frame times during sustained benchmark runs | `HOTSPOT` > 100°C or `PPT_ACCUMULATED` incrementing | `amd-smi metric -p -t -v` | **Thermal or Power Throttling**: Clock clamped by SMU. | Pin DPM clock state via `amd-smi set -l high`; verify chassis cooling and fan profiles. |
| Micro-stutters during host-device asset streaming | `REPLAY_COUNT` incrementing in `amd-smi -P` | `dmesg \| grep amdgpu` | **PCIe Link Signal Degradation**: Bus packet retransmissions. | Inspect PCIe riser cable, motherboard PCIe slot configuration, or bifurcation settings. |
| Poor Dual-GPU scaling (<1.5x on 2 GPUs) | Clock asymmetry (GPU 0 at 3300MHz, GPU 1 at 1600MHz) | `amd-smi topology -d` | **Inter-Frame Idle Bubble**: Secondary GPU idling during present/pacing. | Implement cross-GPU hardware semaphores (`VK_KHR_external_semaphore_fd`) and inter-frame pipelining. |
