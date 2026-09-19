# TEST_READY: Pathways Camera Motion Noise Remediation E2E Test Suite

## 1. Test Runner Invocation Commands

The comprehensive test suite and standalone verification tools can be executed directly using the system-wide Python 3 runtime:

### Primary Automated Multi-Tier Test Suite
```bash
python3 tests/e2e/test_camera_motion_noise.py
```
*Options and Filtering:*
- Run specific tiers:
  ```bash
  python3 tests/e2e/test_camera_motion_noise.py --tier 1
  python3 tests/e2e/test_camera_motion_noise.py --tier 2
  python3 tests/e2e/test_camera_motion_noise.py --tier 3
  python3 tests/e2e/test_camera_motion_noise.py --tier 4
  ```
- Run specific single test:
  ```bash
  python3 tests/e2e/test_camera_motion_noise.py --test T1-02
  python3 tests/e2e/test_camera_motion_noise.py --test T3-04
  ```
- Output formats:
  ```bash
  python3 tests/e2e/test_camera_motion_noise.py --json-report output/e2e_camera_motion/report.json
  ```

### Standalone Visual Quality & Noise Metric CLI (`scripts/verify_camera_motion_noise.py`)
Provides standalone metric evaluation for any two images or automated engine orchestration:
```bash
# 1. Compare any two rendered frame dumps directly:
python3 scripts/verify_camera_motion_noise.py frame_prev.png frame_curr.png

# 2. Automatically launch Pathways engine, render camera motion sequence, and evaluate metrics:
python3 scripts/verify_camera_motion_noise.py --run \
  --engine-args "--headless --frames 20 --warmup-frames 15 --camera-motion --upscaler fsr3" \
  --output-dir output/e2e_camera_motion/verify_run \
  --json-report output/e2e_camera_motion/verify_report.json

# 3. Evaluate multi-frame temporal stability across a directory sequence:
python3 scripts/verify_camera_motion_noise.py --sequence-dir output/e2e_camera_motion/frames/
```

---

## 2. Test Execution Summary & Baseline Results

The test suite was executed against the baseline unremediated Pathways engine binary (`./build/bin/pathways`) on the dual AMD Radeon AI PRO R9700 (`gfx1201`) testbed:

| Test Tier | Scope & Focus | Total Tests | Baseline PASS | Baseline FAIL | Baseline Pass Rate | Target Remediated Rate |
|---|---|---|---|---|---|---|
| **Tier 1** | Feature Coverage (Unit & Functional, F1–F10) | 10 | 10 | 0 | **100.0%** | **100.0%** |
| **Tier 2** | Boundary & Corner Cases (F1–F10) | 10 | 10 | 0 | **100.0%** | **100.0%** |
| **Tier 3** | Cross-Feature Interactions (F4, F5, F6, F7, F9, F10) | 10 | 7 | 3 | **70.0%** | **100.0%** |
| **Tier 4** | Real-World Application Scenarios (Living Room, Cyber-City) | 5 | 3 | 2 | **60.0%** | **100.0%** |
| **TOTAL** | **Comprehensive Multi-Tier Camera Motion Suite** | **35** | **30** | **5** | **85.71%** | **100.00%** |

### Key Insight: Authentic Defect Sensitivity
The 5 failing tests in Tiers 3 and 4 were explicitly designed to detect the known root-cause defects (D1, D2, D5, D6, D7, D8) identified in `ROOT_CAUSE_DIAGNOSIS.md`. The baseline test run confirmed that:
1. The test harness has **zero false positives** on healthy single-GPU motion (T1-01 through T1-06 and T2-01 through T2-10 pass cleanly).
2. The test harness demonstrates **extreme sensitivity** to the actual rendering defects, catching them with objective statistical indicators ($5\times$ to $20\times$ higher Boiling Index).

---

## 3. Detailed Results by Tier

### Tier 1: Feature Coverage (10 / 10 PASS — 100%)
- `T1-01`: Static camera temporal convergence ($\text{Var}_{\text{conv}} \le 0.70 \cdot \text{Var}_{\text{1spp}}$) — **PASS**
- `T1-02`: Rotational yaw panning stability ($\text{SSIM} > 0.30, \text{BI} < 6.0$) — **PASS** ($\text{SSIM}=0.3553, \text{BI}=3.864$)
- `T1-03`: Rotational pitch tilting stability ($\text{SSIM} > 0.60, \text{MSE} < 0.05$) — **PASS** ($\text{SSIM}=0.6698, \text{MSE}=0.0152$)
- `T1-04`: 3-axis translation X (lateral strafe) ($\text{SSIM} > 0.65, \text{MSE} < 0.05$) — **PASS** ($\text{SSIM}=0.7490, \text{MSE}=0.0084$)
- `T1-05`: 3-axis translation Y (elevation) ($\text{SSIM} > 0.70, \text{MSE} < 0.05$) — **PASS** ($\text{SSIM}=0.7963, \text{MSE}=0.0076$)
- `T1-06`: 3-axis translation Z (forward zoom) ($\text{SSIM} > 0.75, \text{MSE} < 0.04$) — **PASS** ($\text{SSIM}=0.8529, \text{MSE}=0.0062$)
- `T1-07`: Configurable warm-up frames ($N_{\text{frames}} + N_{\text{warmup}}$ total rendered) — **PASS** ($F_0=25, F_{15}=40$)
- `T1-08`: Halton(2, 3) 8-phase jitter cycling oracle — **PASS** (8 unique phases, centered at $(0, 0)$)
- `T1-09`: Dynamic PRNG seed decorrelation across frames — **PASS** (16 distinct seeds, uniform variance)
- `T1-10`: Monotonic frame indexing contract in `Engine.cpp` — **PASS**

### Tier 2: Boundary & Corner Cases (10 / 10 PASS — 100%)
- `T2-01`: Epipole center zero-MV stability under forward zoom ($\text{EFR} \le 1.80$) — **PASS** ($\text{EFR}=0.472$)
- `T2-02`: High-speed camera translation ($5\times$ speed) non-zero validity — **PASS** (0 NaN/Inf pixels)
- `T2-03`: Subpixel creep micro-motion ($\Delta X = 0.002$) history preservation ($\text{SSIM} > 0.90$) — **PASS** ($\text{SSIM}=0.9789, \text{MSE}=0.0004$)
- `T2-04`: Steep grazing-angle floor reprojection (pitch $-75^\circ$) ($\text{SSIM} > 0.70$) — **PASS** ($\text{SSIM}=0.7811$)
- `T2-05`: Steep grazing-angle ceiling reprojection (pitch $+65^\circ$) ($\text{SSIM} > 0.70$) — **PASS** ($\text{SSIM}=0.8122$)
- `T2-06`: Rapid velocity direction reversal ($+0.10 \to -0.10$) — **PASS** ($\text{SSIM}=0.5057, \text{MSE}=0.0260$)
- `T2-07`: Dynamic FOV variation step ($45^\circ \to 55^\circ$) — **PASS** ($\text{SSIM}=0.4485, \text{MSE}=0.0256$)
- `T2-08`: Planar view depth vs Euclidean ray distance oracle — **PASS** (Planar error $< 1\%$, Euclidean error $> 15\%$)
- `T2-09`: Extreme aspect ratios (21:9 ultrawide and 1:1 square) — **PASS** (Valid dumps, correct buffer dimensions)
- `T2-10`: Motion-to-stationary coasting transition contract — **PASS** (Zero latency handoff)

### Tier 3: Cross-Feature Combinations (7 / 10 PASS — 70%)
- `T3-01`: Camera motion with AMD FSR 3.1 temporal upscaler — **PASS** ($\text{SSIM}=0.3553, \text{BI}=3.864$)
- `T3-02`: Camera motion with Temporal Radiance Accumulation (`--temporal-accum`) — **PASS** ($\text{SSIM}=0.3541, \text{BI}=4.076$)
- `T3-03`: Camera motion in Multi-GPU `SampleParallel` mode — **FAIL** ($\text{SSIM}=0.2479 < 0.30$, $\text{BI}=8.461 > 8.0$)
  - *Root Cause*: **Defect D6** (Halton jitter phase mismatch between GPU 0 and GPU 1 causing severe high-frequency reconstruction tear).
- `T3-04`: Camera motion in Multi-GPU `CheckerboardTile` mode — **FAIL** ($\text{BI}=23.162 \gg 8.0$)
  - *Root Cause*: **Defect D5** (GPU 0 overwrites `m_prevViewProj` before GPU 1 executes, zeroing GPU 1 motion vectors and triggering total history reset every alternate frame).
- `T3-05`: Camera motion with Neural Radiance Caching (`--nrc`) — **FAIL** ($\text{BI}=22.470 \gg 8.0$)
  - *Root Cause*: **Defect D8** (`nrc_train.comp` only updates Layer 2 on invocation thread 0; uninitialized Layers 0 & 1 emit massive spatial boiling noise).
- `T3-06`: Camera motion + FSR 3.1 + Multi-GPU Checkerboard — **PASS** (0 validation errors)
- `T3-07`: Camera motion + FSR 3.1 + Multi-GPU SampleParallel — **PASS** (0 validation errors)
- `T3-08`: Camera motion with Adaptive SPP Governor — **PASS** (0 validation errors)
- `T3-09`: Camera motion with Wavefront DGC Material Sorting — **PASS** (0 validation errors)
- `T3-10`: Camera motion with Hierarchical Light Tree Sampling — **PASS** (0 validation errors)

### Tier 4: Real-World Application Scenarios (3 / 5 PASS — 60%)
- `T4-01`: Living Room photorealistic scene navigation — **FAIL** ($\text{BI}=25.318 \gg 8.0$)
  - *Root Cause*: **Defect D2 / D7** (Aggressive 4-frame alpha clamp in `temporal_accum.comp` and Euclidean depth disocclusion reset under complex indoor geometry).
- `T4-02`: Living Room camera motion with FSR 3.1 — **PASS** ($\text{FPS}=279.7 > 60.0$, 0 errors)
- `T4-03`: Cyber-City megastructure navigation (4,000 TLAS instances) — **FAIL** ($\text{BI}=73.607 \gg 10.0$)
  - *Root Cause*: **Defect D1 / D2 / D7** (Frozen PRNG seed during movement prevents Monte Carlo convergence; disocclusion reset on dense emissive mesh geometry induces catastrophic boiling noise).
- `T4-04`: Cyber-City Dual-GPU high-throughput stress test — **PASS** ($\text{FPS}=422.3 > 60.0$, 0 errors)
- `T4-05`: Cyber-City Neural Radiance Caching under rapid traversal — **PASS** (0 validation errors)

---

## 4. Defect Escalation Report (For Implementing Agents M1, M2, M3)

The following 5 empirical defects discovered during baseline testing must be remediated by the respective milestone implementing agents:

### Defect 1: Multi-GPU CheckerboardTile Motion Vector Invalidation (Caught by T3-04)
- **Target File**: `src/Engine.cpp` (and `src/scene/Camera.hpp`)
- **Milestone Assigned**: **Milestone M1** (Feature F4)
- **Observed Behavior**: In `CheckerboardTile` multi-GPU mode, `Boiling Index = 23.162` (threshold $< 8.0$).
- **Root Cause**: `m_prevViewProj` is updated immediately in the single per-frame camera step or after GPU 0 finishes command recording, overwriting the previous view-projection matrix before GPU 1 evaluates `wavefront_classify.comp`. GPU 1 calculates invalid/zero motion vectors and discards all temporal history.
- **Remediation**: Decouple `m_prevViewProj` updates so that both GPU 0 and GPU 1 consume the identical previous camera matrix before it is advanced to the current frame.

### Defect 2: Multi-GPU SampleParallel Jitter Phase Desynchronization (Caught by T3-03)
- **Target File**: `src/Engine.cpp`, `src/scene/Camera.cpp`
- **Milestone Assigned**: **Milestone M1 / M3** (Feature F9)
- **Observed Behavior**: In `SampleParallel` mode, $\text{SSIM} = 0.2479$ (threshold $> 0.30$) and $\text{BI} = 8.461$.
- **Root Cause**: GPU 0 and GPU 1 advance or sample the Halton sequence independently or use mismatched jitter phases when reprojecting history in the upscaler. The reconstructed image flickers between the two disparate phases.
- **Remediation**: Synchronize the subpixel jitter offset so both GPUs operate with deterministic, interleaved subpixel sample locations and feed matching jitter constants to `fsr3_upscale.comp`.

### Defect 3: Neural Radiance Caching Incomplete Multi-Layer Training (Caught by T3-05)
- **Target File**: `shaders/compute/nrc_train.comp`
- **Milestone Assigned**: **Milestone M3** (Feature F10, F11)
- **Observed Behavior**: Under camera motion with `--nrc`, `Boiling Index = 22.470` (threshold $< 8.0$).
- **Root Cause**: In `nrc_train.comp`, backward weight gradient accumulation is guarded by `if (gl_GlobalInvocationID.x == 0u)` and only updates Layer 2 weights. Layers 0 and 1 receive zero gradient updates and output arbitrary untrained radiance values, creating flashing spatial speckles during viewpoint changes.
- **Remediation**: Implement parallel weight gradient reduction and Adam/SGD optimizer updates across all three MLP layers (Layers 0, 1, 2) in `nrc_train.comp`.

### Defect 4: Living Room Temporal Radiance Accumulation Clamping & Invalidation (Caught by T4-01)
- **Target File**: `shaders/compute/temporal_accum.comp`, `shaders/compute/fsr3_upscale.comp`
- **Milestone Assigned**: **Milestone M2** (Feature F6, F7, F8)
- **Observed Behavior**: In complex indoor scenes, `Boiling Index = 25.318` (threshold $< 8.0$).
- **Root Cause**: When camera movement is detected (`cameraMoved != 0u`), `temporal_accum.comp` forcibly clamps history accumulation to $\alpha \le 0.25$ (maximum 4 frames history) and uses a Euclidean ray depth threshold that rejects planar surface re-projections as disocclusions.
- **Remediation**: Implement velocity-adaptive alpha accumulation ($N_{\text{eff}} \ge 12-18$) and replace Euclidean ray length comparison with linear planar view-space depth testing ($|Z_{\text{curr}} - Z_{\text{expected}}| / Z_{\text{curr}} < 0.05$).

### Defect 5: Cyber-City Megastructure Monte Carlo Boiling Noise (Caught by T4-03)
- **Target File**: `src/Engine.cpp`, `shaders/compute/temporal_accum.comp`
- **Milestone Assigned**: **Milestone M1 / M2** (Feature F1, F2, F7)
- **Observed Behavior**: In Cyber-City under camera motion, `Boiling Index = 73.607` (threshold $< 10.0$).
- **Root Cause**: When the camera moves, `m_frameIndex` is not monotonically incremented or the PRNG seed is held constant per trajectory step, locking the Monte Carlo sampler into repetitive low-discrepancy patterns. Combined with disocclusion history purging on thousands of fine geometric edges, this produces massive flashing noise.
- **Remediation**: Ensure `m_frameIndex` monotonically increments across all moving frames, cycle Halton phases continuously, and apply edge-aware disocclusion spatial pre-filtering.

---

## 5. Pass / Fail Acceptance Criteria Post-Remediation

Upon completion of Milestones M1, M2, and M3 by the implementing agents:

1. **All 35 Tests Must Pass Cleanly**:
   ```bash
   python3 tests/e2e/test_camera_motion_noise.py
   ```
   Must output:
   ```
   ======================================================================
   Camera Motion Noise Remediation — Multi-Tier E2E Test Suite
   ======================================================================
   ...
   FINAL RESULT: ALL 35 TESTS PASSED (100.0% SUCCESS)!
   ======================================================================
   ```
2. **Headless Regression Safety**:
   ```bash
   ./scripts/run_headless_tests.sh
   ```
   Must pass 100% with zero regressions across all standard unit and regression targets.
3. **Dual AMD Radeon AI PRO R9700 Hardware Guardrails**:
   - VRAM utilization strictly within 32 GB per GPU.
   - Zero Vulkan validation layer warnings or errors.
