# Project: Pathways 4K Deep Profiling, Hardware Telemetry & Synchronization Optimization

## Architecture
Pathways is a real-time Vulkan 1.4 path tracer optimized for Dual AMD Radeon AI PRO R9700 (`gfx1201`, RDNA 4) GPUs and Mesa RADV.
This project executes comprehensive 4K performance characterization, deep hardware profiling, and targeted synchronization/shader optimizations across 5 milestones:
1. **4K Multi-Scene Benchmark Matrix & Telemetry Automation**: Native 3840x2160 benchmarking across all 15 scenes in Single-GPU Monolithic, Single-GPU DGC Sorting, Dual-GPU Tile Parallelism, and Dual-GPU Sample Parallelism, with AMD-SMI power and clock metrics.
2. **ACO Compiler Diagnostics & Shader Register Optimization**: Full register utilization audit (SGPR, VGPR, scratch spills, Wave32/64 occupancy, instruction mix) via Mesa ACO (`RADV_DEBUG=shaderstats,nocache`) and AMD RGA, with targeted register pressure optimization for spilling kernels.
3. **Hardware Cache Telemetry & DGC Autonomous Execution Validation**: Collection of L0/L1/L2 cache hit/miss rates, memory stalls, and bandwidth via Mesa RADV trace controls (`RADV_THREAD_TRACE_CACHE_COUNTERS=1`), coupled with formal verification of GPU-autonomous `VK_EXT_device_generated_commands` execution without CPU intervention or silent fallbacks.
4. **Pipeline Barrier & Synchronization Overhead Optimization**: Audit and refactoring of pipeline barriers and synchronization primitives across wavefront bounces and cross-GPU transfers (DMA-BUF / zero-copy host memory) to eliminate redundant pipeline flushes and over-broad stage masks.
5. **Final Regression Verification & Comprehensive Optimization Report**: Full execution of `./scripts/run_headless_tests.sh` and `python3 tests/test_image_quality.py`, culminating in the authoritative `output/deep_profile/comprehensive_perf_report.md`.

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| F1 | 15-Scene Asset & Config Catalog Integration | Integrate all 15 showcase scenes into `scripts/deep_profile_scenes.py` including Cyber City and Kitchen USD | M1 | Survey 1 |
| F2 | Native 4K 4-Mode Execution Matrix | Support Monolithic, DGC sort, Dual-GPU Tile, and Dual-GPU Sample modes at 4K in benchmark runner | M1 | Survey 1 |
| F3 | AMD-SMI Power & Clock Telemetry | Harvest GPU socket power (W), core clocks (MHz), and temperatures from `/opt/rocm/core-10.0/bin/amd-smi` into JSON | M1 | Survey 1 |
| F4 | Automated Shader Statistics Extractor | Script/tool to automate Mesa ACO (`RADV_DEBUG=shaderstats,nocache`) and RGA shader inspection | M2 | Survey 2 |
| F5 | Shader Compiler Register Table | Generate comprehensive table of SGPR/VGPR counts, scratch allocation, occupancy, and instruction mix | M2 | Survey 2 |
| F6 | Shader Register Pressure Remediation | Optimize shader logic in spilling kernels (`nrc_train.comp`, `fsr3_upscale.comp`) to fit register budgets | M2 | Survey 2 |
| F7 | DGC Autonomous Execution Validation | Verify `vkCmdExecuteGeneratedCommandsEXT` executes without CPU intervention or fallbacks, verifying ring buffer safety | M3 | Survey 3 |
| F8 | Hardware Cache Telemetry Collection | Collect and tabulate L0/L1/L2 hit/miss ratios, memory stalls, and bandwidth using Mesa RADV SQTT/cache counters | M3 | Survey 3 |
| F9 | Engine Redundant Memory Barrier Pruning | Remove redundant post-raytracing `memBarrier` in `src/core/Engine.cpp` | M4 | Survey 3 |
| F10 | Multi-GPU Barrier Stage Mask Tightening | Tighten over-broad `VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT` in `MultiGpuManager.cpp` to compute/RT stages | M4 | Survey 3 |
| F11 | Cross-GPU Barrier Batching & Semaphore Optimization | Batch image layout transitions and tighten secondary semaphore signal stages | M4 | Survey 3 |
| F12 | Full Regression & Image Quality Validation | Validate engine with `./scripts/run_headless_tests.sh` and `python3 tests/test_image_quality.py` | M5 | Survey 1/2/3 |
| F13 | Comprehensive 4K Optimization Report | Authoritative report at `output/deep_profile/comprehensive_perf_report.md` | M5 | Survey 1/2/3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | 4K Multi-Scene Benchmark Matrix & Automated Profiling | F1, F2, F3 | none | DONE |
| M2 | ACO Compiler Shader Statistics & Register Optimization | F4, F5, F6 | none | DONE |
| M3 | Hardware Cache Telemetry & DGC Autonomous Validation | F7, F8 | M1 | DONE |
| M4 | Pipeline Barrier & Synchronization Overhead Optimization | F9, F10, F11 | M1 | DONE |
| M5 | Final Regression Testing & Comprehensive Report | F12, F13 | M1, M2, M3, M4 | DONE |

## Interface Contracts
### Benchmark Runner ↔ Engine CLI
- Execution binary: `./build/bin/pathways`
- Headless invocation: `--headless --width 3840 --height 2160 --warmup-frames 15 --frames 45 --dump-stats <json_path>`
- Mode 1 (Monolithic): `--wavefront-sort none --mgpu-mode off`
- Mode 2 (DGC Sort): `--wavefront-sort archetype --mgpu-mode off`
- Mode 3 (Dual-GPU Tile): `--mgpu-mode tile --tile-size 64 --wavefront-sort archetype`
- Mode 4 (Dual-GPU Sample): `--mgpu-mode sample --wavefront-sort archetype`
- Telemetry output: JSON containing timestamp metrics (`primary_gpu_time_ms`, `secondary_gpu_time_ms`, `tonemap_and_merge_time_ms`, `avg_frame_time_ms`, `avg_fps`, `gigarays_per_second`), plus `amd_smi` power and clock readings.

### Shader Compilation ↔ Profiling Pipeline
- Mesa ACO statistics: `RADV_DEBUG=shaderstats,nocache` redirected to log.
- AMD RGA binary: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/rga -s compute -c gfx1201 --isa <path.isa> --analysis <path.csv> <shader.spv>`.
- Core invariant: Traversal and shading microkernels must maintain 0 scratch memory spills.

### Pipeline Barriers ↔ Cross-GPU Synchronization
- Pre-PCIe transfer barrier: `srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR`, `srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT`.
- Secondary completion semaphore: signal stage `VK_PIPELINE_STAGE_2_TRANSFER_BIT`.
- Post-raytracing barrier in `Engine.cpp`: Guarded to avoid redundant execution when `WavefrontPipeline::recordFrame` already emitted `finalBarrier`.

## Code Layout
- `scripts/deep_profile_scenes.py`: Automated 4K multi-scene benchmark runner.
- `scripts/extract_shader_stats.py`: Automated Mesa ACO and AMD RGA compiler statistics extraction and tabulator.
- `src/core/Engine.cpp`: Main engine loop, frame recording, barrier synchronization.
- `src/mgpu/MultiGpuManager.cpp`: Multi-GPU tile and sample orchestration, cross-GPU image transitions, and semaphores.
- `src/rt/WavefrontPipeline.cpp`: Wavefront path tracing dispatch, barrier management, and DGC execution.
- `src/rt/DGCManager.cpp`: DGC indirect commands layouts, preprocessing ring buffer allocation and recording.
- `shaders/compute/`: Microkernel shaders (`wavefront_*.comp`, `nrc_*.comp`, `fsr3_*.comp`).
- `output/deep_profile/`: Destination for benchmark JSONs, shader statistics tables, RGP traces, and final report.
