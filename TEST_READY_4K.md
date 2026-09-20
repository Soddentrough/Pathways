# TEST_READY_4K: Pathways 4K Performance, DGC Autonomous Execution & Hardware Telemetry E2E Test Suite

## 1. Test Runner Invocation Commands

The automated multi-tier 4K test suite is implemented in `tests/e2e/test_4k_deep_profile.py`. It supports both standard `pytest` orchestration and standalone execution via the system Python runtime.

### Standard Pytest Execution
```bash
# Execute entire 4K test suite (all 22 tests across Tiers 1–4)
pytest tests/e2e/test_4k_deep_profile.py -v

# Execute specific tier via pytest keyword filter
pytest tests/e2e/test_4k_deep_profile.py -k "tier1" -v
pytest tests/e2e/test_4k_deep_profile.py -k "tier2" -v
pytest tests/e2e/test_4k_deep_profile.py -k "tier3" -v
pytest tests/e2e/test_4k_deep_profile.py -k "tier4" -v
```

### Standalone CLI Execution (Direct Python)
```bash
# Run all tiers with formatted console output and JSON report export:
python3 tests/e2e/test_4k_deep_profile.py

# Run specific tier:
python3 tests/e2e/test_4k_deep_profile.py --tier 1
python3 tests/e2e/test_4k_deep_profile.py --tier 2
python3 tests/e2e/test_4k_deep_profile.py --tier 3
python3 tests/e2e/test_4k_deep_profile.py --tier 4

# Custom output directory:
python3 tests/e2e/test_4k_deep_profile.py --output-dir output/deep_profile
```

---

## 2. Test Execution Summary & Verification Matrix

The test suite was executed against the Pathways engine binary (`./build/bin/pathways`) on Dual AMD Radeon AI PRO R9700 GPUs (`gfx1201`, RDNA 4) running Fedora Linux 44 under Vulkan 1.4 / Mesa RADV:

| Tier | Focus & Scope | Tests | PASS | FAIL | Pass Rate | Total Execution Time |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| **Tier 1** | Functional & Mode Verification (Monolithic, DGC sort, Dual Tile, Dual Sample at 4K) | 6 | 6 | 0 | **100.0%** | ~21.0s |
| **Tier 2** | Boundary & Corner Cases (Warmup filtering, Single vs Dual consistency, Cyber City, Bistro) | 6 | 6 | 0 | **100.0%** | ~30.7s |
| **Tier 3** | DGC & Autonomous Execution Validation (Tier-1 explicit preprocess, ring buffer, zero CPU fallbacks) | 5 | 5 | 0 | **100.0%** | ~9.8s |
| **Tier 4** | Physical Telemetry & Performance Verification (SMI power/clock, RT stage timings, mGPU scaling) | 5 | 5 | 0 | **100.0%** | ~0.3s |
| **TOTAL** | **Comprehensive 4K Deep Profile & Optimization Test Suite** | **22** | **22** | **0** | **100.0%** | **~70.9s** |

---

## 3. Test Inventory & Pass/Fail Semantics

### Tier 1: Functional & Mode Verification (4K Native: 3840x2160)
- **`F01-T01` (`test_tier1_mode1_4k_monolithic`)**:
  - *CLI*: `--headless --width 3840 --height 2160 --wavefront-sort none --mgpu-mode off --warmup-frames 5 --frames 10 --no-accumulation`
  - *Pass/Fail Semantics*: Process returncode == 0; telemetry JSON output exists and validates `resolution == [3840, 2160]`, `material_sort_mode == "none"`, `validation_errors == 0`, `avg_frame_time_ms > 0`, `avg_fps > 0`, `gigarays_per_second > 0`; 4K frame dump exists, dimensions exactly (3840, 2160), non-black, zero NaN/Inf pixels.
- **`F02-T01` (`test_tier1_mode2_4k_dgc_archetype_sort`)**:
  - *CLI*: `--headless --width 3840 --height 2160 --wavefront-sort archetype --mgpu-mode off --warmup-frames 5 --frames 10 --no-accumulation`
  - *Pass/Fail Semantics*: Returncode == 0; stdout confirms DGC activation; telemetry validates `material_sort_mode == "archetype"`, `validation_errors == 0`; 4K frame dump passes integrity verification.
- **`F02-T02` (`test_tier1_mode2_4k_dgc_dual_sort`)**:
  - *CLI*: `--headless --width 3840 --height 2160 --wavefront-sort dual --mgpu-mode off --warmup-frames 5 --frames 10 --no-accumulation`
  - *Pass/Fail Semantics*: Returncode == 0; confirms dual sort path executes without validation errors.
- **`F03-T01` (`test_tier1_mode3_4k_dual_gpu_tile`)**:
  - *CLI*: `--headless --width 3840 --height 2160 --mgpu-mode tile --tile-size 64 --wavefront-sort archetype --warmup-frames 5 --frames 10 --no-accumulation`
  - *Pass/Fail Semantics*: Returncode == 0; telemetry confirms `secondary_gpu.active == true`, `secondary_gpu.mgpu_mode == "checkerboard_tile"`; GPU breakdown confirms concurrent execution (`primary_gpu_time_ms > 0`, `secondary_gpu_time_ms > 0`, `tonemap_and_merge_time_ms > 0`); 4K stitched frame passes validation.
- **`F04-T01` (`test_tier1_mode4_4k_dual_gpu_sample`)**:
  - *CLI*: `--headless --width 3840 --height 2160 --mgpu-mode sample --spp 2 --wavefront-sort archetype --warmup-frames 5 --frames 10 --no-accumulation`
  - *Pass/Fail Semantics*: Returncode == 0; telemetry confirms `secondary_gpu.active == true`, `secondary_gpu.mgpu_mode == "sample_parallel"`, `spp == 2`; both GPUs contribute ray samples.
- **`F05-T01` (`test_tier1_4k_representative_scenes`)**:
  - *CLI*: `--headless --width 3840 --height 2160 --scene <path>` across Cornell Caustic and Glass of Water.
  - *Pass/Fail Semantics*: Returncode == 0; `validation_errors == 0`; triangle count > 0; non-zero ray throughput.

### Tier 2: Boundary & Corner Cases
- **`F06-T01` (`test_tier2_warmup_frames_filtering`)**:
  - *CLI*: Compares `--warmup-frames 0 --frames 20` vs `--warmup-frames 10 --frames 20`.
  - *Pass/Fail Semantics*: Verifies that `frame_count` in configurations breakdown strictly equals the 20 measured frames; verifies `total_frames` is 20 and 30 respectively; verifies warmup filters out cold driver and AS creation spikes.
- **`F07-T01` (`test_tier2_single_vs_dual_gpu_consistency_tile`)**:
  - *CLI*: Compares 15-frame accumulated 4K renders between Single-GPU and Dual-GPU Tile mode.
  - *Pass/Fail Semantics*: Vectorized SSIM > 0.88 (empirically 0.973), PSNR > 25.0 dB (empirically 31.35 dB), Pearson correlation > 0.95. Confirms checkerboard tile boundary stitching has zero missing tiles or seam tearing.
- **`F07-T02` (`test_tier2_single_vs_dual_gpu_consistency_sample`)**:
  - *CLI*: Compares 15-frame accumulated 4K renders between Single-GPU 2 SPP and Dual-GPU Sample 2 SPP.
  - *Pass/Fail Semantics*: Pearson correlation > 0.85 (empirically 0.922), SSIM > 0.65; confirms sample parallel accumulation converges to identical lighting solution.
- **`F08-T01` (`test_tier2_extreme_geometry_cyber_city`)**:
  - *CLI*: `--headless --width 3840 --height 2160 --scene procedural:cyber-city`
  - *Pass/Fail Semantics*: Returncode == 0; generates 4,000 instances across 19 modular BLAS prototypes; allocated VRAM < 16 GB (well within 32 GB capacity); 0 validation errors.
- **`F09-T01` (`test_tier2_extreme_geometry_bistro_interior`)**:
  - *CLI*: `--headless --width 3840 --height 2160 --scene scenes/bistro/bistro_interior.glb`
  - *Pass/Fail Semantics*: Returncode == 0; parses > 1,300,000 triangles, 74 materials; ray throughput > 0.40 GigaRays/s; 0 validation errors.
- **`F10-T01` (`test_tier2_bounce_depth_boundary`)**:
  - *CLI*: Compares `--max-bounces 1` vs `--max-bounces 8` at 4K.
  - *Pass/Fail Semantics*: 1-bounce run produces exactly 1 bounce record in telemetry; 8-bounce run produces >= 4 bounces with monotonically decreasing active ray counts.

### Tier 3: DGC & Autonomous Execution Validation
- **`F11-T01` (`test_tier3_dgc_tier1_explicit_preprocess_enabled`)**:
  - *CLI*: Standard DGC execution check.
  - *Pass/Fail Semantics*: Stdout confirms `[INFO] DGC Tier 1 explicit preprocessing enabled.`, indirect commands layout flags `0x3` (`VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT | VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_EXT`), dispatch token stride 12 bytes, material execution set stride 16 bytes.
- **`F11-T02` (`test_tier3_dgc_implicit_preprocess_fallback`)**:
  - *CLI*: `--no-dgc-preprocess`
  - *Pass/Fail Semantics*: Stdout confirms clean fallback `DGC baseline active (implicit preprocessing)` with layout flags `0x0`; exit code 0.
- **`F12-T01` (`test_tier3_dgc_ring_buffer_health`)**:
  - *CLI*: Extended multi-frame 4K execution (`--warmup-frames 5 --frames 35`).
  - *Pass/Fail Semantics*: Confirms preprocess buffer allocation (32 slices @ 4096 bytes = 131,072 bytes total, align 256 bytes); confirms 40 frames execute with zero memory corruption, zero device lost errors, and 0 validation errors.
- **`F13-T01` (`test_tier3_autonomous_gpu_execution_zero_cpu_fallback`)**:
  - *CLI*: Standard 4K DGC run with telemetry inspection.
  - *Pass/Fail Semantics*: Confirms absence of CPU fallback warnings; confirms GPU timestamp profiler records active GPU durations (`classify_time_ms > 0`, `shade_ms > 0`, `intersect_ms > 0`).
- **`F14-T01` (`test_tier3_secondary_gpu_dgc_autonomy`)**:
  - *CLI*: Dual GPU execution.
  - *Pass/Fail Semantics*: Confirms Secondary GPU initializes and executes independent DGC pipeline on Peer Compute node without CPU stalls.

### Tier 4: Physical Telemetry & Performance Verification
- **`F15-T01` (`test_tier4_physical_gpu_clocks_and_temperatures`)**:
  - *CLI*: Telemetry validation against RDNA 4 hardware limits.
  - *Pass/Fail Semantics*: Primary GPU clock between 400 MHz and 3500 MHz; temperature between 20 °C and 95 °C; allocated VRAM between 100 MB and 32768 MB.
- **`F16-T01` (`test_tier4_amd_smi_power_draw_telemetry`)**:
  - *CLI*: Invokes `/opt/rocm/core-10.0/bin/amd-smi metric --power --clock -t --json`.
  - *Pass/Fail Semantics*: Returncode == 0; detects 2 GPUs; socket power for GPU 0 and GPU 1 > 10 W and <= 350 W; genuine hardware telemetry.
- **`F17-T01` (`test_tier4_wavefront_rt_stage_timings`)**:
  - *CLI*: Telemetry profiler breakdown check.
  - *Pass/Fail Semantics*: `primary_gpu_time_ms > 0`, `tonemap_and_merge_time_ms > 0`, `total_wavefront_time_ms > 0`.
- **`F18-T01` (`test_tier4_positive_multigpu_scaling_tile`)**:
  - *CLI*: Single-GPU Monolithic ($T_1$) vs Dual-GPU Tile ($T_2$).
  - *Pass/Fail Semantics*: Speedup $T_1 / T_2 > 1.15\times$ (empirically $1.57\times - 2.14\times$); Secondary GPU performs >= 30% of total RT workload.
- **`F18-T02` (`test_tier4_positive_multigpu_scaling_sample`)**:
  - *CLI*: Single-GPU 1 SPP ($GRays_1$) vs Dual-GPU Sample 2 SPP ($GRays_2$).
  - *Pass/Fail Semantics*: Throughput scaling ratio $GRays_2 / GRays_1 > 1.30\times$ (empirically $1.98\times$).

---

## 4. Output Artifacts & Inspection

All test execution telemetry and frame captures are output to `output/test_4k_deep_profile/`:
- `test_4k_deep_profile_report.json`: Comprehensive execution summary with per-test timing and status.
- `t1_mode1_monolithic.png` & `.json`: 4K Single-GPU Monolithic render.
- `t1_mode2_dgc_arch.png` & `.json`: 4K Single-GPU DGC Archetype Sort render.
- `t1_mode3_dual_tile.png` & `.json`: 4K Dual-GPU 64x64 Checkerboard Tile render.
- `t1_mode4_dual_sample.png` & `.json`: 4K Dual-GPU 2 SPP Sample Parallel render.
- `t2_cyber_city.png` & `.json`: 4K Procedural Cyber City extreme stress render.
- `t2_bistro_interior.json`: 4K Bistro Interior (1.3M tris) telemetry.
- `t2_accum_single.png` & `t2_accum_tile.png`: 15-frame accumulated images demonstrating 0.9733 SSIM consistency.

All tests are self-contained, deterministic, and verifiable on the Dual AMD Radeon AI PRO R9700 system.
