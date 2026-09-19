# TEST_INFRA: Comprehensive Multi-Tier Testing Infrastructure & Specification
# Pathways Vulkan 1.4 Path Tracer — Camera Motion Noise Remediation (Features F1–F15)

## 1. Executive Summary & Testing Philosophy

This document specifies the complete testing architecture, interface contracts, mathematical reference models (oracles), and evaluation metrics for validating the remediation of camera motion noise in the Pathways Vulkan 1.4 path tracer across dual AMD Radeon AI PRO R9700 (`gfx1201`) GPUs.

The test infrastructure establishes a **4-Tier Progressive Verification Hierarchy**:
- **Tier 1: Feature Coverage (Unit & Functional)** — Core camera navigation, 3-axis translation (X, Y, Z), rotational yaw/pitch panning, steady-state temporal convergence ($\ge 18$ frames), configurable warm-up frames, Halton sequence cycling, and dynamic PRNG seed advancement (10 tests: T1-01 to T1-10).
- **Tier 2: Boundary & Corner Cases** — Focus of expansion (epipole) zero-MV stability during Z-zoom, high-speed camera translation ($5\times$ speed), subpixel creep, steep grazing-angle floor/ceiling reprojection ($> 75^\circ$), rapid direction reversals, dynamic FOV variation, linear planar view depth vs Euclidean distance oracle, extreme aspect ratios (21:9, 1:1), and zero-velocity stationary transition responsiveness (10 tests: T2-01 to T2-10).
- **Tier 3: Cross-Feature Combinations** — Motion with AMD FSR 3.1 temporal upscaling, motion with Temporal Radiance Accumulation (`temporal_accum.comp`), motion in Multi-GPU `SampleParallel` mode (dual R9700 at 1 SPP each), motion in Multi-GPU `CheckerboardTile` mode, motion with Neural Radiance Caching (`--nrc`), combined FSR3 + Multi-GPU modes, dynamic SPP governor, Wavefront DGC material sorting, and hierarchical light tree sampling (10 tests: T3-01 to T3-10).
- **Tier 4: Real-World Application Scenarios** — Complex photorealistic navigation in `scenes/living-room/living_room_extended.glb` (143k triangles), rotational panning with FSR 3.1, procedural `Cyber-City` megastructure navigation (4,000 TLAS instances, hundreds of emissive mesh lights), dual-GPU frame pacing stress ($\ge 60$ FPS), and neural cache temporal stability under rapid viewpoint translation (5 tests: T4-01 to T4-05).

Total Suite: **35 Dedicated, Authoritative Tests**

---

## 2. Feature Inventory & Interface Contracts

| ID | Feature Name | Target Component | Architectural Role | Milestone |
|---|---|---|---|---|
| **F1** | Continuous Frame Indexing | `Engine.hpp`, `Engine.cpp` | Ensure `m_frameIndex` monotonically increments during camera motion; eliminate static PRNG seed locking | M1 |
| **F2** | Continuous Halton Jitter Cycling | `Camera.hpp`, `Camera.cpp` | Cycle Halton sequence phases across moving frames instead of freezing at phase 0 | M1 |
| **F3** | Motion Vector Jitter Alignment | `wavefront_classify.comp` | Eliminate subpixel jitter bias in motion vectors so reprojected UV coordinates match previous surface positions | M1 |
| **F4** | Multi-GPU PrevViewProj Decoupling | `Engine.cpp`, `Camera.cpp` | Decouple `m_prevViewProj` updates so GPU 0 does not overwrite previous matrix before GPU 1 query | M1 |
| **F5** | Elimination of False Zero-MV Disocclusion | `fsr3_upscale.comp` | Remove `dot(mv, mv) < 1e-8 && pc.cameraMoved != 0u` history invalidation in `fsr3_upscale.comp` | M2 |
| **F6** | Planar View-Space Depth Disocclusion Heuristic | `temporal_accum.comp`, `fsr3_upscale.comp` | Replace Euclidean ray length delta with linear planar view depth reprojection test | M2 |
| **F7** | Velocity-Adaptive Alpha Smoothing | `temporal_accum.comp`, `fsr3_upscale.comp` | Retain high temporal accumulation ($N_{\text{eff}} \ge 12-18$) during camera movement | M2 |
| **F8** | Disocclusion Spatial Pre-Filtering & Clamping | `temporal_accum.comp`, `fsr3_upscale.comp` | Provide edge-aware neighborhood filtering to suppress raw 2-SPP variance in true disoccluded regions | M2 |
| **F9** | Multi-GPU Jitter Phase Synchronization | `Engine.cpp`, `MultiGpuManager.cpp` | Pass correct jitter offsets to GPU 1 upscaler and reconcile SampleParallel reconstruction | M3 |
| **F10** | NRC Multi-Layer Parallel Training | `nrc_train.comp` | Implement complete training for Layers 0, 1, and 2 in `nrc_train.comp` with proper gradient scaling | M3 |
| **F11** | Decoupled Positional Encoding | `nrc_encode_infer.comp` | Replace coupled `sin(p.x+p.y)` with independent axis encoding in `nrc_encode_infer.comp` | M3 |
| **F12** | NRC Multi-Bounce Target Conditioning | `wavefront_shade_diffuse.comp` | Condition NRC training targets on validated multi-bounce radiance without unshadowed direct light bias | M3 |
| **F13** | Dynamic Camera Trajectory Test Suite | `tests/e2e/test_camera_motion_noise.py` | Executable test harness executing static, yaw/pitch panning, and XYZ translation with warm-up frames | M4 |
| **F14** | Visual Quality & Noise Verification | `scripts/verify_camera_motion_noise.py` | Standalone verification tool computing MAD, MSE, PSNR, SSIM, and boiling noise index | M4 |
| **F15** | Headless Test Suite Regression Safety | `scripts/run_headless_tests.sh` | Validate that full headless regression suite passes 100% cleanly with zero regressions | M4 |

---

## 3. Mathematical Metrics, Invariants & Oracles

### 3.1 Mathematical Metrics for Camera Motion Noise

1. **Mean Absolute Difference (MAD)**:
   $$\text{MAD} = \frac{1}{W \cdot H} \sum_{x,y} |I_t(x, y) - I_{t-1}(x, y)|$$
2. **Mean Squared Error (MSE)**:
   $$\text{MSE} = \frac{1}{3 \cdot W \cdot H} \sum_{c \in \{R,G,B\}} \sum_{x,y} (I_t^c(x, y) - I_{t-1}^c(x, y))^2$$
3. **Peak Signal-to-Noise Ratio (PSNR)**:
   $$\text{PSNR} = 10 \cdot \log_{10}\left(\frac{1.0}{\max(\text{MSE}, 10^{-12})}\right) \quad [\text{dB}]$$
4. **Structural Similarity Index (SSIM)**:
   $$\text{SSIM}(x, y) = \frac{(2\mu_x\mu_y + C_1)(2\sigma_{xy} + C_2)}{(\mu_x^2 + \mu_y^2 + C_1)(\sigma_x^2 + \sigma_y^2 + C_2)}$$
   where $C_1 = 0.01^2, C_2 = 0.03^2$.
5. **Boiling Noise Index ($\text{BI}$)**:
   In real-time path tracing, "boiling" or "speckle flashing" is characterized by high spatial variance in the frame-to-frame difference map and extreme outliers in disoccluded or zero-MV pixels:
   $$\text{BI} = 100.0 \cdot \text{Var}[\Delta_t] \cdot \left(\frac{P_{99}[\Delta_t]}{\mu_{\Delta_t} + 10^{-5}}\right)$$
   where $\Delta_t = |L_t - L_{t-1}|$ is the pixel luminance difference, $P_{99}$ is the 99th percentile, and $\text{Var}[\Delta_t]$ is the spatial variance.
   - **Ground-Truth Motion Optical Flow Baseline**: $\text{BI} \approx 3.0 \dots 5.0$.
   - **Severe Boiling Noise (Defects D1/D5/D8)**: $\text{BI} \ge 20.0 \dots 75.0$ ($5\times - 20\times$ increase).
6. **Focus of Expansion (Epipole) Flash Ratio ($\text{EFR}$)**:
   During forward/backward camera translation along the optical axis ($Z$), the vanishing point center has $\vec{v} \to 0$.
   $$\text{EFR} = \frac{\mu_{\Delta_t}(\text{Center Box } [0.4W..0.6W, 0.4H..0.6H])}{\max(\mu_{\Delta_t}(\text{Surround}), 10^{-5})}$$
   - **Healthy Reprojection**: $\text{EFR} \le 1.8$ (center diff is equal to or lower than surround flow).
   - **Defect D2 Flash Spike**: $\text{EFR} \gg 2.0$ (center vanishing point flashes raw 1-SPP boiling noise while surround reprojects).

### 3.2 Authoritative Oracles

#### Oracle 1: Halton(2, 3) 8-Phase Jitter Sequence
```
Phase 0: (-0.000, -0.167)    Phase 4: (-0.438, -0.389)
Phase 1: (+0.250, +0.167)    Phase 5: (+0.062, -0.056)
Phase 2: (-0.250, -0.389)    Phase 6: (-0.188, +0.278)
Phase 3: (+0.375, -0.056)    Phase 7: (+0.312, +0.056)
```
- Subpixel coverage: All phases strictly bounded in $[-0.5, 0.5]^2$.
- Mean offset: $\approx (0, 0)$ across full cycle.

#### Oracle 2: Linear Planar View Depth vs. Euclidean Ray Distance
- Surface at $Z_{\text{view}} = 2.0$, camera translation $\Delta Z = 0.4$ (20% movement):
  - **Euclidean Ray Distance**: $D_{t-1} = 2.07\text{m} \to D_t = 1.66\text{m}$. Relative difference:
    $$\frac{|D_t - D_{t-1}|}{D_t} = \frac{0.41}{1.66} = 25.0\% > 15.0\% \implies \text{FALSE DISOCCLUSION RESETS ENTIRE SCREEN}$$
  - **Linear Planar View Depth**: $Z_{\text{curr}} = 1.60\text{m}$, $Z_{\text{expected}} = Z_{\text{prev}} - \Delta Z = 1.60\text{m}$.
    $$\frac{|Z_{\text{curr}} - Z_{\text{expected}}|}{Z_{\text{curr}}} = 0.0\% \implies \text{ACCURATE GEOMETRIC HISTORY PRESERVED}$$

#### Oracle 3: Ground-Truth Optical Flow Shift Model
- For a 0.2° yaw pan (~3 pixels displacement at 720p on high-contrast edges):
  - Expected ground-truth shift SSIM: $\approx 0.32$
  - Expected ground-truth shift MSE: $\approx 0.017$
  - Expected ground-truth shift Boiling Index: $\approx 3.89$

---

## 4. 4-Tier Test Matrix Specification

### Tier 1: Feature Coverage (Unit & Functional)
| Test ID | Feature | Test Description | Command / Invocation | Authoritative Oracle / Criteria | Threshold |
|---|---|---|---|---|---|
| **T1-01** | F1, F7 | Static Camera Temporal Convergence | `--frames 20` vs `--frames 1` stationary | Laplacian variance drops $\ge 3\times$ | $\text{Var}_{\text{conv}} \le 0.70 \cdot \text{Var}_{\text{1spp}}$ |
| **T1-02** | F1, F3 | Rotational Yaw Panning Stability | `--camera-motion --frames 15` vs `16` | Ground-truth 3-px optical flow model | $\text{SSIM} > 0.30, \text{BI} < 6.0$ |
| **T1-03** | F1, F2 | Rotational Pitch Tilting Stability | `--camera 0,1,3,0,1.27,2` vs `0,1,3,0,1.35,2` | Clamped pitch $[-89^\circ, 89^\circ]$ flow | $\text{SSIM} > 0.60, \text{MSE} < 0.05$ |
| **T1-04** | F1, F3 | 3-Axis Translation X (Strafe) | `--camera 0.0,1,3` vs `0.08,1,3` | Lateral horizontal flow | $\text{SSIM} > 0.65, \text{MSE} < 0.05$ |
| **T1-05** | F1, F3 | 3-Axis Translation Y (Elevation) | `--camera 0,1.00,3` vs `0,1.08,3` | Vertical elevation flow | $\text{SSIM} > 0.70, \text{MSE} < 0.05$ |
| **T1-06** | F1, F6 | 3-Axis Translation Z (Zoom) | `--camera 0,1,3.00` vs `0,1,2.90` | Forward optical expansion | $\text{SSIM} > 0.75, \text{MSE} < 0.04$ |
| **T1-07** | F13 | Configurable Warm-Up Frames | `--warmup-frames 0` vs `15` | Total rendered: $N_{\text{frames}} + N_{\text{warmup}}$ | $F_0 = 25, F_{15} = 40$ |
| **T1-08** | F2 | Halton(2, 3) Jitter Cycling Oracle | Math verification of 8 phases | Radical inverse sequence | 8 distinct phases, mean $\approx 0$ |
| **T1-09** | F1 | Dynamic PRNG Seed Decorrelation | Math verification across 16 frames | Hash distribution in $[0, 1]$ | 16 unique seeds, $\sigma^2 > 0.04$ |
| **T1-10** | F1 | Monotonic Frame Indexing Contract | Code inspection of `Engine.cpp` | Decoupled accumulation tracking | `m_frameIndex` monotonic |

### Tier 2: Boundary & Corner Cases
| Test ID | Feature | Test Description | Command / Invocation | Authoritative Oracle / Criteria | Threshold |
|---|---|---|---|---|---|
| **T2-01** | F5 | Epipole Zero-MV Stability | `--camera 0,1,3.0` vs `0,1,2.9` TRA | Vanishing point zero-MV flow | $\text{EFR} \le 1.80$ |
| **T2-02** | F7 | High-Speed Camera Translation | `--camera 0.0,1,3` vs `0.5,1,3` ($5\times$ speed) | Velocity-adaptive alpha clamping | 0 NaN/Inf pixels |
| **T2-03** | F3 | Subpixel Creep Micro-Motion | `--camera 0.000,1,3` vs `0.002,1,3` | Subpixel history preservation | $\text{SSIM} > 0.90, \text{MSE} < 0.01$ |
| **T2-04** | F6 | Grazing-Angle Floor Reprojection | Pitch $-75^\circ$ looking at floor plane | Planar view depth consistency | $\text{SSIM} > 0.70, \text{MSE} < 0.05$ |
| **T2-05** | F6 | Grazing-Angle Ceiling Reprojection | Pitch $+65^\circ$ looking at ceiling | Normal gradient stability | $\text{SSIM} > 0.70, \text{MSE} < 0.05$ |
| **T2-06** | F7 | Rapid Velocity Direction Reversal | Camera $+0.10 \to -0.10$ translation | Ghost-free inversion response | $\text{SSIM} > 0.40, \text{MSE} < 0.08$ |
| **T2-07** | F3 | Dynamic FOV Variation Boundary | FOV $45^\circ \to 55^\circ$ zoom step | Projection matrix scaling | $\text{SSIM} > 0.35, \text{MSE} < 0.08$ |
| **T2-08** | F6 | Planar View Depth vs Euclid Oracle | Numerical simulation of $\Delta Z$ step | Geometric disocclusion math | Euclid $> 15\%$, Planar $< 1\%$ |
| **T2-09** | F3 | Extreme Aspect Ratio Boundaries | 21:9 (1680x720) vs 1:1 (800x800) | Screen UV normalization | Valid dumps, correct resolution |
| **T2-10** | F1 | Motion-to-Stationary Transition | Code inspection of `Camera.hpp` | Immediate accumulation handoff | Coasting latency $< 50\text{ ms}$ |

### Tier 3: Cross-Feature Combinations
| Test ID | Feature | Test Description | Command / Invocation | Target Defect Caught | Threshold |
|---|---|---|---|---|---|
| **T3-01** | F5, F7 | Motion + AMD FSR 3.1 Upscaling | `--camera-motion --upscaler fsr3` | Lanczos reprojection quality | $\text{SSIM} > 0.35, \text{BI} < 5.0$ |
| **T3-02** | F6, F7 | Motion + Temporal Radiance Accum | `--camera-motion --temporal-accum` | TRA planar depth tracking | $\text{SSIM} > 0.30, \text{BI} < 6.0$ |
| **T3-03** | F9 | Motion in Multi-GPU SampleParallel | `--camera-motion --mgpu --mgpu-mode sample` | **Defect D6** (Jitter Phase Mismatch) | $\text{SSIM} > 0.30, \text{BI} < 8.0$ |
| **T3-04** | F4 | Motion in Multi-GPU CheckerboardTile | `--camera-motion --mgpu --mgpu-mode tile` | **Defect D5** (GPU 1 Zero MV) | $\text{SSIM} > 0.30, \text{BI} < 8.0$ |
| **T3-05** | F10, F11 | Motion with Neural Radiance Caching | `--camera-motion --nrc` | **Defect D8** (NRC Unlearned Weights) | $\text{SSIM} > 0.30, \text{BI} < 8.0$ |
| **T3-06** | F4, F5 | Motion + FSR3 + MGPU Checkerboard | `--camera-motion --mgpu --upscaler fsr3` | Combined upscaling/tiling | 0 validation errors |
| **T3-07** | F5, F9 | Motion + FSR3 + MGPU SampleParallel | `--camera-motion --mgpu-mode sample --upscaler fsr3` | Combined sample upscaling | 0 validation errors |
| **T3-08** | F1 | Motion with Adaptive SPP Governor | `--camera-motion --adaptive-spp` | Dynamic governor stability | 0 validation errors |
| **T3-09** | F3 | Motion with Wavefront DGC Sorting | `--camera-motion --wavefront-sort dual` | DGC microkernel MV integrity | 0 validation errors |
| **T3-10** | F1 | Motion with Hierarchical Light Tree | `--camera-motion --light-tree` | Directional sampling stability | 0 validation errors |

### Tier 4: Real-World Application Scenarios
| Test ID | Feature | Test Description | Command / Invocation | Target Defect Caught | Threshold |
|---|---|---|---|---|---|
| **T4-01** | F6, F7 | Living Room Navigation | `--scene living-room --camera-motion` | **Defect D2/D7** (Living Room Noise) | $\text{SSIM} > 0.30, \text{BI} < 8.0$ |
| **T4-02** | F5 | Living Room FSR 3.1 Panning | `--scene living-room --upscaler fsr3` | Framerate & upscaling polish | $\text{FPS} > 60.0, 0\text{ errors}$ |
| **T4-03** | F1, F6 | Cyber-City Megastructure Navigation | `--scene cyber-city --camera-motion` | **Defect D1/D2/D7** (Cyber Boiling) | $\text{SSIM} > 0.25, \text{BI} < 10.0$ |
| **T4-04** | F4 | Cyber-City Dual-GPU Stress Test | `--scene cyber-city --mgpu --camera-motion` | Dual R9700 throughput | $\text{FPS} \ge 60.0, 0\text{ errors}$ |
| **T4-05** | F10 | Cyber-City Neural Radiance Caching | `--scene cyber-city --nrc --camera-motion` | NRC megastructure cache | 0 validation errors |

---

## 5. Test Artifacts Summary

1. **`tests/e2e/test_camera_motion_noise.py`**: Automated multi-tier E2E test suite with 35 authoritative tests, `--tier` filtering, colored CLI output, and structured JSON reporting.
2. **`scripts/verify_camera_motion_noise.py`**: Standalone verification and metric computation CLI script supporting image pair comparison, dynamic `--run` orchestration, and folder sequence analysis.
3. **`TEST_INFRA.md`**: Architectural specification of test hierarchy, mathematical formulas, and contract test matrix (this document).
4. **`TEST_READY.md`**: Test execution guide, baseline pass/fail status, and escalated implementation defects report.
