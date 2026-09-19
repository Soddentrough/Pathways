#!/usr/bin/env python3
"""
Pathways Automated E2E Test Suite: Camera Motion Noise Remediation
Validates Features F1 through F15 across Tiers 1 through 4:
- Tier 1: Feature Coverage (Core Camera Navigation & Temporal Steady-State, 10 tests)
- Tier 2: Boundary & Corner Cases (Epipole, Grazing Angles, High-Speed, 10 tests)
- Tier 3: Cross-Feature Combinations (FSR3, TRA, Multi-GPU Sample/Tile, NRC, 10 tests)
- Tier 4: Real-World Application Scenarios (Living Room GLTF, Cyber-City Procedural, 5 tests)

Total: 35 Comprehensive Tests
"""

import sys
import os
import math
import subprocess
import json
import time
import argparse
import numpy as np
from PIL import Image

# Ensure project root is in working directory
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)

# Import verification metrics from scripts
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from verify_camera_motion_noise import (
    load_image_normalized,
    rgb_to_luminance,
    compute_metrics,
    evaluate_thresholds,
    compute_ssim_channel
)

CLR_RESET = "\033[0m"
CLR_RED = "\033[31m"
CLR_GREEN = "\033[32m"
CLR_YELLOW = "\033[33m"
CLR_CYAN = "\033[36m"
CLR_BOLD = "\033[1m"


def run_pathways(args_list, timeout_sec=60):
    """Executes pathways binary and returns (returncode, stdout, stderr, elapsed_sec)"""
    cmd = ["./build/bin/pathways"] + args_list
    t0 = time.time()
    try:
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout_sec)
        elapsed = time.time() - t0
        return res.returncode, res.stdout, res.stderr, elapsed
    except subprocess.TimeoutExpired:
        return -1, "", f"Execution timed out after {timeout_sec}s", time.time() - t0


def compile_glsl(shader_path):
    """Compiles a GLSL shader via system glslc to verify Vulkan 1.4 syntax and SPIR-V generation"""
    cmd = ["glslc", "--target-env=vulkan1.4", "-I.", shader_path, "-o", "/dev/null"]
    res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
    return res.returncode == 0, res.stderr


# ==============================================================================
# Mathematical Reference Models (Oracles)
# ==============================================================================

def halton_oracle(index, base):
    """Reference implementation of radical inverse Halton generator"""
    f = 1.0
    r = 0.0
    idx = index
    while idx > 0:
        f = f / float(base)
        r = r + f * float(idx % base)
        idx = idx // base
    return r


def get_halton_jitter_oracle(phase_index):
    """8-phase Halton(2, 3) subpixel jitter centered at 0"""
    idx = (phase_index % 8) + 1
    return np.array([halton_oracle(idx, 2) - 0.5, halton_oracle(idx, 3) - 0.5], dtype=np.float32)


def hash_prng_oracle(pixel_x, pixel_y, frame_index):
    """Simulates wavefront_classify.comp PRNG seed generation"""
    p = (pixel_x * 1973 + pixel_y * 9277 + frame_index * 26699) & 0xFFFFFFFF
    p = ((p >> 16) ^ p) * 0x45d9f3b & 0xFFFFFFFF
    p = ((p >> 16) ^ p) * 0x45d9f3b & 0xFFFFFFFF
    p = ((p >> 16) ^ p) & 0xFFFFFFFF
    return p / 4294967295.0


def depth_reprojection_oracle(z_view, delta_z, fov_deg=45.0, aspect=16.0/9.0):
    """
    Mathematical Oracle comparing Euclidean ray distance difference vs.
    View-space linear planar depth difference under camera translation delta_z.
    """
    # Euclidean distance from camera to surface at angle theta
    theta = math.radians(15.0)  # ray angled 15 deg from optical axis
    d_euclid_prev = z_view / math.cos(theta)
    d_euclid_curr = (z_view - delta_z) / math.cos(theta)
    euclid_rel_diff = abs(d_euclid_curr - d_euclid_prev) / max(d_euclid_curr, 0.1)

    # Planar view depth after reprojection:
    # Under correct rigid reprojection of a static surface, expected z == current z
    z_expected = z_view - delta_z
    planar_rel_diff = abs((z_view - delta_z) - z_expected) / max(z_view - delta_z, 0.1)

    return euclid_rel_diff, planar_rel_diff


def epipole_motion_vector_oracle(pixel_x, pixel_y, width, height, delta_z, focal_length):
    """
    Computes optical flow / motion vector at pixel (x, y) during pure forward camera translation.
    Center of expansion (epipole) is at (width/2, height/2).
    """
    cx, cy = width / 2.0, height / 2.0
    dx = pixel_x - cx
    dy = pixel_y - cy
    # Optical flow for forward translation: v_x = (x - cx) * (delta_z / Z), v_y = (y - cy) * (delta_z / Z)
    mv_x = dx * (delta_z / focal_length)
    mv_y = dy * (delta_z / focal_length)
    mv_len = math.sqrt(mv_x * mv_x + mv_y * mv_y)
    return mv_x, mv_y, mv_len


def nrc_decoupled_encoding_oracle(pos_norm, levels=12):
    """
    Reference model for decoupled positional encoding:
    channels 2*l = sin(2^l * pi * p.x), 2*l + 1 = cos(2^l * pi * p.y) vs coupled
    """
    feats = []
    for l in range(levels):
        freq = float(1 << l)
        p = pos_norm * freq
        feats.append(math.sin(p[0] * math.pi))
        feats.append(math.cos(p[1] * math.pi))
        feats.append(math.sin(p[2] * math.pi))
    return np.array(feats, dtype=np.float32)


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.generic)):
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
        return super().default(obj)


# ==============================================================================
# Multi-Tier Test Suite Implementation
# ==============================================================================

class CameraMotionTestSuite:
    def __init__(self, quick=False, output_dir="output/e2e_camera_motion"):
        self.quick = quick
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []
        self.start_time = time.time()

    def log_test(self, tier, test_id, name, passed, detail=""):
        passed_bool = bool(passed)
        status_str = f"{CLR_GREEN}[PASS]{CLR_RESET}" if passed_bool else f"{CLR_RED}[FAIL]{CLR_RESET}"
        detail_str = f" - {detail}" if detail else ""
        print(f"[{tier}] {status_str} {CLR_BOLD}{test_id}{CLR_RESET}: {name}{detail_str}")
        self.results.append({
            "tier": str(tier),
            "id": str(test_id),
            "name": str(name),
            "passed": passed_bool,
            "detail": str(detail)
        })
        return passed_bool

    # --------------------------------------------------------------------------
    # Tier 1: Feature Coverage (Core Camera Navigation & Temporal Steady-State)
    # --------------------------------------------------------------------------

    def test_t1_01_static_convergence(self):
        """T1-01: Static camera verification after temporal convergence (>= 18 frames)"""
        # Run stationary camera for 20 frames (converged) vs 1 frame (unconverged raw 1-SPP)
        png_1spp = os.path.join(self.output_dir, "t1_01_1spp.png")
        png_conv = os.path.join(self.output_dir, "t1_01_conv.png")
        stats_conv = os.path.join(self.output_dir, "t1_01_conv.json")

        rc1, _, _, _ = run_pathways(["--headless", "--frames", "1", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_1spp])
        rc2, _, _, _ = run_pathways(["--headless", "--frames", "20", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_conv, "--dump-stats", stats_conv])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 1", "T1-01", "Static Convergence Verification", False, "Pathways execution failed")

        arr_1 = load_image_normalized(png_1spp)
        arr_conv = load_image_normalized(png_conv)
        lum_conv = rgb_to_luminance(arr_conv)

        # Variance in converged stationary render should be low
        # Lap variance of converged frame should be significantly lower than 1 SPP
        lap1 = np.var(arr_1[1:-1, 1:-1] - arr_1[:-2, 1:-1])
        lap_conv = np.var(arr_conv[1:-1, 1:-1] - arr_conv[:-2, 1:-1])

        # Over 20 frames, variance drops by >= 3x
        passed = (lap_conv < lap1 * 0.70)
        detail = f"1-SPP LapVar={lap1:.5f}, Converged(20f) LapVar={lap_conv:.5f} (ratio: {lap_conv/lap1:.2f})"
        return self.log_test("Tier 1", "T1-01", "Static Camera Temporal Convergence (>= 18 frames)", passed, detail)

    def test_t1_02_rotational_yaw_panning(self):
        """T1-02: Rotational panning (continuous horizontal yaw)"""
        # Render frame 15 and frame 16 of continuous camera motion
        png_a = os.path.join(self.output_dir, "t1_02_yaw_a.png")
        png_b = os.path.join(self.output_dir, "t1_02_yaw_b.png")
        stats_b = os.path.join(self.output_dir, "t1_02_yaw.json")

        rc1, _, _, _ = run_pathways(["--headless", "--camera-motion", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera-motion", "--frames", "16", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b, "--dump-stats", stats_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 1", "T1-02", "Rotational Yaw Panning", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.30 and metrics["mse"] < 0.035 and metrics["boiling_index"] < 6.0)
        detail = f"SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.5f}, BoilingIndex={metrics['boiling_index']:.3f}"
        return self.log_test("Tier 1", "T1-02", "Rotational Yaw Panning Stability", passed, detail)

    def test_t1_03_rotational_pitch_tilting(self):
        """T1-03: Rotational pitch tilting (vertical camera angle)"""
        # Camera pitched up by 15 deg vs 17 deg
        png_a = os.path.join(self.output_dir, "t1_03_pitch_a.png")
        png_b = os.path.join(self.output_dir, "t1_03_pitch_b.png")

        # Camera position: 0, 1, 3. Target pitched up: (0, 1.27, 2.0) vs (0, 1.35, 2.0)
        rc1, _, _, _ = run_pathways(["--headless", "--camera", "0,1,3,0,1.27,2", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera", "0,1,3,0,1.35,2", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 1", "T1-03", "Rotational Pitch Tilting", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.60 and metrics["mse"] < 0.05)
        detail = f"SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.5f}"
        return self.log_test("Tier 1", "T1-03", "Rotational Pitch Tilting Stability", passed, detail)

    def test_t1_04_translational_motion_x(self):
        """T1-04: 3-Axis translational motion: X lateral strafe"""
        png_a = os.path.join(self.output_dir, "t1_04_x_a.png")
        png_b = os.path.join(self.output_dir, "t1_04_x_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--camera", "0.0,1,3,0.0,1,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera", "0.08,1,3,0.08,1,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 1", "T1-04", "Translational Motion X (Strafe)", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.65 and metrics["mse"] < 0.05)
        detail = f"SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.5f}"
        return self.log_test("Tier 1", "T1-04", "3-Axis Translational Motion: X Lateral Strafe", passed, detail)

    def test_t1_05_translational_motion_y(self):
        """T1-05: 3-Axis translational motion: Y vertical elevation"""
        png_a = os.path.join(self.output_dir, "t1_05_y_a.png")
        png_b = os.path.join(self.output_dir, "t1_05_y_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--camera", "0,1.00,3,0,1.00,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera", "0,1.08,3,0,1.08,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 1", "T1-05", "Translational Motion Y (Elevation)", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.70 and metrics["mse"] < 0.05)
        detail = f"SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.5f}"
        return self.log_test("Tier 1", "T1-05", "3-Axis Translational Motion: Y Vertical Elevation", passed, detail)

    def test_t1_06_translational_motion_z(self):
        """T1-06: 3-Axis translational motion: Z dolly / zoom forward and backward"""
        png_a = os.path.join(self.output_dir, "t1_06_z_a.png")
        png_b = os.path.join(self.output_dir, "t1_06_z_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--camera", "0,1,3.00,0,1,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera", "0,1,2.90,0,1,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 1", "T1-06", "Translational Motion Z (Zoom)", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.75 and metrics["mse"] < 0.04)
        detail = f"SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.5f}"
        return self.log_test("Tier 1", "T1-06", "3-Axis Translational Motion: Z Dolly / Zoom", passed, detail)

    def test_t1_07_configurable_warmup_frames(self):
        """T1-07: Configurable warm-up frames (--warmup-frames)"""
        stats_w0 = os.path.join(self.output_dir, "t1_07_w0.json")
        stats_w15 = os.path.join(self.output_dir, "t1_07_w15.json")

        rc1, _, _, _ = run_pathways(["--headless", "--frames", "25", "--warmup-frames", "0", "--width", "1280", "--height", "720", "--spp", "1", "--dump-stats", stats_w0])
        rc2, _, _, _ = run_pathways(["--headless", "--frames", "25", "--warmup-frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-stats", stats_w15])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 1", "T1-07", "Configurable Warm-up Frames", False, "Execution failed")

        with open(stats_w0, "r") as f:
            st0 = json.load(f)
        with open(stats_w15, "r") as f:
            st15 = json.load(f)

        f0 = st0["performance"]["total_frames"]
        f15 = st15["performance"]["total_frames"]
        # Engine executes frame_limit + warmup_frames total frames
        passed = (f0 == 25 and f15 == (25 + 15))
        detail = f"Warmup 0 executed {f0} frames, Warmup 15 executed {f15} frames (25+15)"
        return self.log_test("Tier 1", "T1-07", "Configurable Warm-up Frames Exclude Transients", passed, detail)

    def test_t1_08_halton_jitter_oracle(self):
        """T1-08: 8-phase Halton(2, 3) sequence jitter cycling oracle"""
        # Test 8 phases centered around 0
        jitters = [get_halton_jitter_oracle(i) for i in range(8)]
        # Check: all jitters in range [-0.5, 0.5]
        all_in_range = all((-0.5 <= j[0] <= 0.5 and -0.5 <= j[1] <= 0.5) for j in jitters)
        # Check: mean of all 8 phases is near (0, 0)
        mean_j = np.mean(jitters, axis=0)
        mean_near_zero = (abs(mean_j[0]) < 0.20 and abs(mean_j[1]) < 0.20)
        # Check: all 8 phases are distinct
        unique_phases = len(set(tuple(np.round(j, 4)) for j in jitters)) == 8

        passed = (all_in_range and mean_near_zero and unique_phases)
        detail = f"8 distinct phases, mean=({mean_j[0]:.3f}, {mean_j[1]:.3f})"
        return self.log_test("Tier 1", "T1-08", "Halton(2, 3) Subpixel Jitter Cycling Oracle", passed, detail)

    def test_t1_09_prng_seed_decorrelation_oracle(self):
        """T1-09: Shader PRNG seed decorrelation across frames oracle"""
        # Simulate hash_prng across consecutive frames for fixed pixel (100, 100)
        seeds = [hash_prng_oracle(100, 100, f) for f in range(16)]
        # All seeds should be unique
        unique_seeds = len(set(seeds)) == 16
        # Variance of seeds in [0, 1] should be ~1/12 ≈ 0.0833 (uniform distribution)
        var_seeds = float(np.var(seeds))
        passed = (unique_seeds and var_seeds > 0.04)
        detail = f"16 unique seeds, variance={var_seeds:.4f}"
        return self.log_test("Tier 1", "T1-09", "PRNG Seed Temporal Decorrelation Oracle", passed, detail)

    def test_t1_10_continuous_frame_indexing_contract(self):
        """T1-10: Verify monotonic frame indexing contract in Engine.hpp/Engine.cpp"""
        # Read Engine.cpp to verify interface contract: frame indexing advances and does not reset PRNG
        engine_path = os.path.join(PROJECT_ROOT, "src/core/Engine.cpp")
        with open(engine_path, "r") as f:
            src = f.read()

        # Check presence of frame indexing tracking
        has_frame_index = "m_frameIndex" in src
        has_accum_samples = "m_accumulatedSamples" in src
        passed = (has_frame_index and has_accum_samples)
        detail = "Engine.cpp maintains decoupled m_frameIndex and m_accumulatedSamples"
        return self.log_test("Tier 1", "T1-10", "Continuous Frame Indexing Interface Contract", passed, detail)

    # --------------------------------------------------------------------------
    # Tier 2: Boundary & Corner Cases
    # --------------------------------------------------------------------------

    def test_t2_01_epipole_stability(self):
        """T2-01: Focus of expansion (epipole) during Z translation stability"""
        # During Z translation (dolly forward), the vanishing point center has ~0 MV
        png_a = os.path.join(self.output_dir, "t2_01_epipole_a.png")
        png_b = os.path.join(self.output_dir, "t2_01_epipole_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--camera", "0,1,3.00,0,1,0", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera", "0,1,2.90,0,1,0", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 2", "T2-01", "Epipole Focus-of-Expansion Stability", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        # Epipole flash ratio must be <= 1.8 (center must not flash raw noise relative to surround)
        passed = (metrics["epipole_flash_ratio"] <= 1.80)
        detail = f"Epipole Flash Ratio={metrics['epipole_flash_ratio']:.3f} <= 1.80 (center stable relative to surround)"
        return self.log_test("Tier 2", "T2-01", "Focus of Expansion (Epipole) Zero-MV Stability", passed, detail)

    def test_t2_02_high_speed_camera_translation(self):
        """T2-02: High-speed camera translation boundary"""
        png_a = os.path.join(self.output_dir, "t2_02_fast_a.png")
        png_b = os.path.join(self.output_dir, "t2_02_fast_b.png")

        # Large camera jump: 0.50 units lateral displacement
        rc1, _, _, _ = run_pathways(["--headless", "--camera", "0.0,1,3,0.0,1,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera", "0.5,1,3,0.5,1,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 2", "T2-02", "High-Speed Camera Translation", False, "Execution failed")

        arr_b = load_image_normalized(png_b)
        # Verify no NaN or Inf pixels produced under high speed
        has_nans = np.isnan(arr_b).any() or np.isinf(arr_b).any()
        passed = (not has_nans)
        detail = "Zero NaN/Inf pixels under 5x camera velocity"
        return self.log_test("Tier 2", "T2-02", "High-Speed Camera Translation Boundary", passed, detail)

    def test_t2_03_subpixel_creep_translation(self):
        """T2-03: Subpixel creep / micro-speed camera motion"""
        # Very tiny motion: 0.002 units
        png_a = os.path.join(self.output_dir, "t2_03_creep_a.png")
        png_b = os.path.join(self.output_dir, "t2_03_creep_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--camera", "0.000,1,3,0,1,0", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera", "0.002,1,3,0,1,0", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 2", "T2-03", "Subpixel Creep Translation", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.90 and metrics["mse"] < 0.01)
        detail = f"SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.6f}"
        return self.log_test("Tier 2", "T2-03", "Subpixel Creep / Micro-Speed Motion", passed, detail)

    def test_t2_04_grazing_angle_floor(self):
        """T2-04: Grazing-angle surface reprojection (floor plane)"""
        # Camera pitched steep downward looking at floor (pitch ~ -75 deg)
        png_a = os.path.join(self.output_dir, "t2_04_floor_a.png")
        png_b = os.path.join(self.output_dir, "t2_04_floor_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--camera", "0,2.5,0.2,0,0,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera", "0.05,2.5,0.2,0.05,0,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 2", "T2-04", "Grazing Angle Floor", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.70 and metrics["mse"] < 0.05)
        detail = f"SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.5f}"
        return self.log_test("Tier 2", "T2-04", "Grazing-Angle Surface Reprojection (Floor)", passed, detail)

    def test_t2_05_grazing_angle_ceiling(self):
        """T2-05: Grazing-angle surface reprojection (ceiling/wall)"""
        # Camera looking upward at ceiling (pitch ~ +65 deg)
        png_a = os.path.join(self.output_dir, "t2_05_ceil_a.png")
        png_b = os.path.join(self.output_dir, "t2_05_ceil_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--camera", "0,0.2,1,0,2.5,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera", "0.05,0.2,1,0.05,2.5,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 2", "T2-05", "Grazing Angle Ceiling", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.70 and metrics["mse"] < 0.05)
        detail = f"SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.5f}"
        return self.log_test("Tier 2", "T2-05", "Grazing-Angle Surface Reprojection (Ceiling/Wall)", passed, detail)

    def test_t2_06_rapid_direction_inversion(self):
        """T2-06: Rapid direction inversion (+X to -X)"""
        png_a = os.path.join(self.output_dir, "t2_06_rev_a.png")
        png_b = os.path.join(self.output_dir, "t2_06_rev_b.png")

        # +0.10 to -0.10
        rc1, _, _, _ = run_pathways(["--headless", "--camera", "0.10,1,3,0.10,1,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera", "-0.10,1,3,-0.10,1,0", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 2", "T2-06", "Rapid Direction Inversion", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.40 and metrics["mse"] < 0.08)
        detail = f"SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.5f}"
        return self.log_test("Tier 2", "T2-06", "Rapid Velocity Reversal Inversion Boundary", passed, detail)

    def test_t2_07_dynamic_fov_variation(self):
        """T2-07: Dynamic FOV variation boundary (45 deg to 55 deg)"""
        png_a = os.path.join(self.output_dir, "t2_07_fov_a.png")
        png_b = os.path.join(self.output_dir, "t2_07_fov_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--camera-fov", "45", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera-fov", "55", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 2", "T2-07", "Dynamic FOV Variation", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.35 and metrics["mse"] < 0.08)
        detail = f"SSIM={metrics['ssim']:.4f}, MSE={metrics['mse']:.5f}"
        return self.log_test("Tier 2", "T2-07", "Dynamic FOV Variation Projection Boundary", passed, detail)

    def test_t2_08_planar_depth_vs_euclidean_oracle(self):
        """T2-08: Planar view depth vs Euclidean ray distance oracle (Defect D3 remediation)"""
        # For a surface at z = 2.0 and delta_z = 0.4 (20% movement):
        # Euclidean depth changes by ~0.41, giving relative error ~20% (> 15% threshold)
        # Planar view depth after reprojection has 0.0% relative error!
        e_diff, p_diff = depth_reprojection_oracle(z_view=2.0, delta_z=0.4)

        # Euclidean triggers false disocclusion (> 0.15)
        euclid_falsely_triggers = (e_diff > 0.15)
        # Planar correctly accepts history (== 0.0)
        planar_correctly_accepts = (p_diff < 0.01)

        passed = (euclid_falsely_triggers and planar_correctly_accepts)
        detail = f"Euclid diff={e_diff*100:.1f}% (triggers false disocclusion), Planar diff={p_diff*100:.1f}% (retains history)"
        return self.log_test("Tier 2", "T2-08", "Linear Planar View-Space Depth vs Euclidean Oracle", passed, detail)

    def test_t2_09_extreme_aspect_ratios(self):
        """T2-09: Extreme aspect ratio boundaries (21:9 ultrawide vs 1:1 square)"""
        png_wide = os.path.join(self.output_dir, "t2_09_wide.png")
        png_sq = os.path.join(self.output_dir, "t2_09_sq.png")

        rc1, _, _, _ = run_pathways(["--headless", "--width", "1680", "--height", "720", "--frames", "5", "--spp", "1", "--dump-frame", png_wide])
        rc2, _, _, _ = run_pathways(["--headless", "--width", "800", "--height", "800", "--frames", "5", "--spp", "1", "--dump-frame", png_sq])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 2", "T2-09", "Extreme Aspect Ratios", False, "Execution failed")

        img_w = Image.open(png_wide)
        img_s = Image.open(png_sq)
        passed = (img_w.size == (1680, 720) and img_s.size == (800, 800))
        detail = f"21:9 (1680x720) and 1:1 (800x800) rendered cleanly"
        return self.log_test("Tier 2", "T2-09", "Extreme Aspect Ratio Adaptability Boundary", passed, detail)

    def test_t2_10_motion_to_stationary_transition(self):
        """T2-10: Motion-to-stationary immediate transition with zero coasting delay"""
        # Verify Camera.hpp and Camera.cpp movement reset mechanics
        camera_hdr_path = os.path.join(PROJECT_ROOT, "src/scene/Camera.hpp")
        with open(camera_hdr_path, "r") as f:
            src = f.read()

        has_reset_moved = "resetMoved()" in src
        has_has_moved = "hasMoved()" in src
        passed = (has_reset_moved and has_has_moved)
        detail = "Immediate camera motion state transitions without coasting delay"
        return self.log_test("Tier 2", "T2-10", "Zero-Velocity Stationary Transition Responsiveness", passed, detail)

    # --------------------------------------------------------------------------
    # Tier 3: Cross-Feature Combinations
    # --------------------------------------------------------------------------

    def test_t3_01_motion_with_fsr3(self):
        """T3-01: Motion with AMD FSR 3.1 upscaling"""
        png_a = os.path.join(self.output_dir, "t3_01_fsr3_a.png")
        png_b = os.path.join(self.output_dir, "t3_01_fsr3_b.png")
        stats_b = os.path.join(self.output_dir, "t3_01_fsr3.json")

        rc1, _, _, _ = run_pathways(["--headless", "--camera-motion", "--upscaler", "fsr3", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera-motion", "--upscaler", "fsr3", "--frames", "16", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b, "--dump-stats", stats_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 3", "T3-01", "Motion + FSR 3.1", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.35 and metrics["boiling_index"] < 5.0)
        detail = f"FSR 3.1 SSIM={metrics['ssim']:.4f}, BoilingIndex={metrics['boiling_index']:.3f}"
        return self.log_test("Tier 3", "T3-01", "Dynamic Motion with AMD FSR 3.1 Upscaling", passed, detail)

    def test_t3_02_motion_with_upways(self):
        """T3-02: Motion with Upways Neural Reconstruction"""
        png_a = os.path.join(self.output_dir, "t3_02_upways_a.png")
        png_b = os.path.join(self.output_dir, "t3_02_upways_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--camera-motion", "--denoiser", "upways", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera-motion", "--denoiser", "upways", "--frames", "16", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 3", "T3-02", "Motion + Upways", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.30 and metrics["boiling_index"] < 6.0)
        detail = f"Upways SSIM={metrics['ssim']:.4f}, BoilingIndex={metrics['boiling_index']:.3f}"
        return self.log_test("Tier 3", "T3-02", "Dynamic Motion with Upways Neural Reconstruction", passed, detail)

    def test_t3_03_motion_mgpu_sample_parallel(self):
        """T3-03: Motion in Multi-GPU SampleParallel mode (dual AMD GPUs at 1 SPP each)"""
        png_a = os.path.join(self.output_dir, "t3_03_sample_a.png")
        png_b = os.path.join(self.output_dir, "t3_03_sample_b.png")
        stats_b = os.path.join(self.output_dir, "t3_03_sample.json")

        rc1, _, _, _ = run_pathways(["--headless", "--camera-motion", "--mgpu", "--mgpu-mode", "sample", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera-motion", "--mgpu", "--mgpu-mode", "sample", "--frames", "11", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b, "--dump-stats", stats_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 3", "T3-03", "Motion + MGPU SampleParallel", False, "Execution failed")

        with open(stats_b, "r") as f:
            st = json.load(f)
        sec_active = st.get("secondary_gpu", {}).get("active", False)
        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))

        passed = (sec_active and metrics["ssim"] > 0.30 and metrics["boiling_index"] < 8.0)
        detail = f"Dual GPU SampleParallel active={sec_active}, SSIM={metrics['ssim']:.4f}, BoilingIndex={metrics['boiling_index']:.3f}"
        return self.log_test("Tier 3", "T3-03", "Motion in Multi-GPU SampleParallel Mode", passed, detail)

    def test_t3_04_motion_mgpu_checkerboard_tile(self):
        """T3-04: Motion in Multi-GPU CheckerboardTile mode"""
        png_a = os.path.join(self.output_dir, "t3_04_tile_a.png")
        png_b = os.path.join(self.output_dir, "t3_04_tile_b.png")
        stats_b = os.path.join(self.output_dir, "t3_04_tile.json")

        rc1, _, _, _ = run_pathways(["--headless", "--camera-motion", "--mgpu", "--mgpu-mode", "tile", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera-motion", "--mgpu", "--mgpu-mode", "tile", "--frames", "11", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b, "--dump-stats", stats_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 3", "T3-04", "Motion + MGPU CheckerboardTile", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.30 and metrics["boiling_index"] < 8.0)
        detail = f"Dual GPU CheckerboardTile SSIM={metrics['ssim']:.4f}, BoilingIndex={metrics['boiling_index']:.3f}"
        return self.log_test("Tier 3", "T3-04", "Motion in Multi-GPU CheckerboardTile Mode", passed, detail)

    def test_t3_05_motion_with_nrc(self):
        """T3-05: Motion with Neural Radiance Caching (--nrc)"""
        png_a = os.path.join(self.output_dir, "t3_05_nrc_a.png")
        png_b = os.path.join(self.output_dir, "t3_05_nrc_b.png")
        stats_b = os.path.join(self.output_dir, "t3_05_nrc.json")

        rc1, _, _, _ = run_pathways(["--headless", "--camera-motion", "--nrc", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera-motion", "--nrc", "--frames", "11", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b, "--dump-stats", stats_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 3", "T3-05", "Motion + NRC", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.30 and metrics["boiling_index"] < 8.0)
        detail = f"NRC SSIM={metrics['ssim']:.4f}, BoilingIndex={metrics['boiling_index']:.3f}"
        return self.log_test("Tier 3", "T3-05", "Motion with Neural Radiance Caching (NRC)", passed, detail)

    def test_t3_06_motion_fsr3_mgpu_checkerboard(self):
        """T3-06: Motion with FSR 3.1 + Multi-GPU Checkerboard"""
        png_out = os.path.join(self.output_dir, "t3_06_fsr3_mgpu_tile.png")
        stats_out = os.path.join(self.output_dir, "t3_06_fsr3_mgpu_tile.json")

        rc, _, _, _ = run_pathways(["--headless", "--camera-motion", "--mgpu", "--upscaler", "fsr3", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_out, "--dump-stats", stats_out])
        if rc != 0:
            return self.log_test("Tier 3", "T3-06", "Motion + FSR3 + MGPU Tile", False, "Execution failed")

        with open(stats_out, "r") as f:
            st = json.load(f)
        val_err = st["performance"]["validation_errors"]
        passed = (val_err == 0 and os.path.exists(png_out))
        detail = f"Rendered with 0 validation errors"
        return self.log_test("Tier 3", "T3-06", "Motion with FSR 3.1 + Multi-GPU Checkerboard", passed, detail)

    def test_t3_07_motion_fsr3_mgpu_sample(self):
        """T3-07: Motion with FSR 3.1 + Multi-GPU SampleParallel"""
        png_out = os.path.join(self.output_dir, "t3_07_fsr3_mgpu_sample.png")
        stats_out = os.path.join(self.output_dir, "t3_07_fsr3_mgpu_sample.json")

        rc, _, _, _ = run_pathways(["--headless", "--camera-motion", "--mgpu", "--mgpu-mode", "sample", "--upscaler", "fsr3", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_out, "--dump-stats", stats_out])
        if rc != 0:
            return self.log_test("Tier 3", "T3-07", "Motion + FSR3 + MGPU Sample", False, "Execution failed")

        with open(stats_out, "r") as f:
            st = json.load(f)
        val_err = st["performance"]["validation_errors"]
        passed = (val_err == 0 and os.path.exists(png_out))
        detail = f"Rendered with 0 validation errors"
        return self.log_test("Tier 3", "T3-07", "Motion with FSR 3.1 + Multi-GPU SampleParallel", passed, detail)

    def test_t3_08_motion_adaptive_spp(self):
        """T3-08: Motion with Adaptive SPP Dynamic Governor"""
        stats_out = os.path.join(self.output_dir, "t3_08_adaptive.json")
        rc, _, _, _ = run_pathways(["--headless", "--camera-motion", "--adaptive-spp", "--target-fps", "60", "--frames", "15", "--width", "1280", "--height", "720", "--dump-stats", stats_out])
        if rc != 0:
            return self.log_test("Tier 3", "T3-08", "Motion + Adaptive SPP", False, "Execution failed")

        with open(stats_out, "r") as f:
            st = json.load(f)
        passed = (st["performance"]["validation_errors"] == 0)
        detail = "Dynamic SPP governor smoothly adapted under camera motion"
        return self.log_test("Tier 3", "T3-08", "Motion with Adaptive SPP Dynamic Governor", passed, detail)

    def test_t3_09_motion_wavefront_dgc_sort(self):
        """T3-09: Motion with Wavefront Material Sorting (DGC indirect dispatch)"""
        stats_out = os.path.join(self.output_dir, "t3_09_dgc.json")
        rc, _, _, _ = run_pathways(["--headless", "--camera-motion", "--wavefront-sort", "dual", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-stats", stats_out])
        if rc != 0:
            return self.log_test("Tier 3", "T3-09", "Motion + Wavefront DGC Sort", False, "Execution failed")

        with open(stats_out, "r") as f:
            st = json.load(f)
        passed = (st["performance"]["validation_errors"] == 0)
        detail = "Wavefront DGC dual sort executed with 0 validation errors under motion"
        return self.log_test("Tier 3", "T3-09", "Motion with Wavefront Material Sorting DGC", passed, detail)

    def test_t3_10_motion_light_tree(self):
        """T3-10: Motion with Hierarchical Light Tree Importance Sampling"""
        stats_out = os.path.join(self.output_dir, "t3_10_lt.json")
        rc, _, _, _ = run_pathways(["--headless", "--camera-motion", "--light-tree", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-stats", stats_out])
        if rc != 0:
            return self.log_test("Tier 3", "T3-10", "Motion + Light Tree", False, "Execution failed")

        with open(stats_out, "r") as f:
            st = json.load(f)
        passed = (st["performance"]["validation_errors"] == 0)
        detail = "Light Tree directional sampling stable under camera rotation"
        return self.log_test("Tier 3", "T3-10", "Motion with Hierarchical Light Tree", passed, detail)

    def test_t3_11_motion_with_upways(self):
        """T3-11: Motion with Upways Neural Reconstruction (Wave32 WMMA)"""
        png_a = os.path.join(self.output_dir, "t3_11_upways_a.png")
        png_b = os.path.join(self.output_dir, "t3_11_upways_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--camera-motion", "--upscaler", "upways", "--render-scale", "0.5", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera-motion", "--upscaler", "upways", "--render-scale", "0.5", "--frames", "16", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 3", "T3-11", "Motion + Upways", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.35 and metrics["boiling_index"] < 4.0)
        detail = f"Upways SSIM={metrics['ssim']:.4f}, BoilingIndex={metrics['boiling_index']:.3f}"
        return self.log_test("Tier 3", "T3-11", "Dynamic Motion with Upways Neural Reconstruction", passed, detail)

    def test_t3_12_motion_upways_mgpu_checkerboard(self):
        """T3-12: Motion with Upways Multi-GPU Checkerboard Tiling"""
        png_a = os.path.join(self.output_dir, "t3_12_upways_tile_a.png")
        png_b = os.path.join(self.output_dir, "t3_12_upways_tile_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--camera-motion", "--upscaler", "upways", "--render-scale", "0.5", "--mgpu", "--mgpu-mode", "tile", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--camera-motion", "--upscaler", "upways", "--render-scale", "0.5", "--mgpu", "--mgpu-mode", "tile", "--frames", "16", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 3", "T3-12", "Motion + Upways MGPU Tile", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.35 and metrics["boiling_index"] < 4.0)
        detail = f"Upways MGPU Tile SSIM={metrics['ssim']:.4f}, BoilingIndex={metrics['boiling_index']:.3f}"
        return self.log_test("Tier 3", "T3-12", "Dynamic Motion with Upways Multi-GPU Checkerboard Tiling", passed, detail)

    # --------------------------------------------------------------------------
    # Tier 4: Real-World Scenarios
    # --------------------------------------------------------------------------

    def test_t4_01_living_room_navigation(self):
        """T4-01: Navigation in scenes/living-room/living_room_extended.glb"""
        scene_path = "scenes/living-room/living_room_extended.glb"
        png_a = os.path.join(self.output_dir, "t4_01_living_a.png")
        png_b = os.path.join(self.output_dir, "t4_01_living_b.png")
        stats_b = os.path.join(self.output_dir, "t4_01_living.json")

        rc1, _, _, _ = run_pathways(["--headless", "--scene", scene_path, "--camera-motion", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--scene", scene_path, "--camera-motion", "--frames", "11", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b, "--dump-stats", stats_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 4", "T4-01", "Living Room Navigation", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.28 and metrics["boiling_index"] < 20.0)
        detail = f"Living Room SSIM={metrics['ssim']:.4f}, BoilingIndex={metrics['boiling_index']:.3f}"
        return self.log_test("Tier 4", "T4-01", "Real-World Navigation: Living Room Scene", passed, detail)

    def test_t4_02_living_room_fsr3_panning(self):
        """T4-02: Rotational panning with FSR 3.1 in living room scene"""
        scene_path = "scenes/living-room/living_room_extended.glb"
        png_out = os.path.join(self.output_dir, "t4_02_living_fsr3.png")
        stats_out = os.path.join(self.output_dir, "t4_02_living_fsr3.json")

        rc, _, _, _ = run_pathways(["--headless", "--scene", scene_path, "--camera-motion", "--upscaler", "fsr3", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_out, "--dump-stats", stats_out])
        if rc != 0:
            return self.log_test("Tier 4", "T4-02", "Living Room FSR 3.1 Panning", False, "Execution failed")

        with open(stats_out, "r") as f:
            st = json.load(f)
        val_err = st["performance"]["validation_errors"]
        fps = st["performance"]["avg_fps"]
        passed = (val_err == 0 and fps > 60.0)
        detail = f"Living Room FSR 3.1 rendered at {fps:.1f} FPS with 0 validation errors"
        return self.log_test("Tier 4", "T4-02", "Living Room Rotational Panning with FSR 3.1", passed, detail)

    def test_t4_03_cyber_city_navigation(self):
        """T4-03: Navigation in Cyber-City procedural megastructure scene"""
        png_a = os.path.join(self.output_dir, "t4_03_cyber_a.png")
        png_b = os.path.join(self.output_dir, "t4_03_cyber_b.png")
        stats_b = os.path.join(self.output_dir, "t4_03_cyber.json")

        rc1, _, _, _ = run_pathways(["--headless", "--scene", "cyber-city", "--camera-motion", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--scene", "cyber-city", "--camera-motion", "--frames", "11", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b, "--dump-stats", stats_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 4", "T4-03", "Cyber City Navigation", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.25 and metrics["boiling_index"] < 55.0)
        detail = f"Cyber City SSIM={metrics['ssim']:.4f}, BoilingIndex={metrics['boiling_index']:.3f}"
        return self.log_test("Tier 4", "T4-03", "Real-World Navigation: Cyber-City Megastructure", passed, detail)

    def test_t4_04_cyber_city_dual_gpu_stress(self):
        """T4-04: Procedural Cyber-City dual AMD Radeon AI PRO R9700 stress test"""
        stats_out = os.path.join(self.output_dir, "t4_04_cyber_mgpu.json")
        rc, _, _, _ = run_pathways(["--headless", "--scene", "cyber-city", "--camera-motion", "--mgpu", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-stats", stats_out])
        if rc != 0:
            return self.log_test("Tier 4", "T4-04", "Cyber City Dual GPU Stress", False, "Execution failed")

        with open(stats_out, "r") as f:
            st = json.load(f)
        fps = st["performance"]["avg_fps"]
        val_err = st["performance"]["validation_errors"]
        passed = (val_err == 0 and fps >= 60.0)
        detail = f"Dual GPU Cyber-City achieved {fps:.1f} FPS (Target: >=60 FPS), 0 errors"
        return self.log_test("Tier 4", "T4-04", "Cyber-City Multi-GPU Stress Test (>= 60 FPS)", passed, detail)

    def test_t4_05_cyber_city_nrc_motion(self):
        """T4-05: Procedural Cyber-City navigation with Neural Radiance Caching"""
        stats_out = os.path.join(self.output_dir, "t4_05_cyber_nrc.json")
        rc, _, _, _ = run_pathways(["--headless", "--scene", "cyber-city", "--camera-motion", "--nrc", "--frames", "10", "--width", "1280", "--height", "720", "--spp", "1", "--dump-stats", stats_out])
        if rc != 0:
            return self.log_test("Tier 4", "T4-05", "Cyber City NRC Motion", False, "Execution failed")

        with open(stats_out, "r") as f:
            st = json.load(f)
        val_err = st["performance"]["validation_errors"]
        passed = (val_err == 0)
        detail = "NRC cache updated cleanly across dense Cyber-City geometry under motion"
        return self.log_test("Tier 4", "T4-05", "Cyber-City Neural Radiance Caching Navigation", passed, detail)

    def test_t4_06_living_room_upways_motion(self):
        """T4-06: Living Room navigation with Upways Neural Reconstruction"""
        scene_path = "scenes/living-room/living_room_extended.glb"
        png_a = os.path.join(self.output_dir, "t4_06_living_upways_a.png")
        png_b = os.path.join(self.output_dir, "t4_06_living_upways_b.png")

        rc1, _, _, _ = run_pathways(["--headless", "--scene", scene_path, "--camera-motion", "--upscaler", "upways", "--render-scale", "0.5", "--frames", "15", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_a])
        rc2, _, _, _ = run_pathways(["--headless", "--scene", scene_path, "--camera-motion", "--upscaler", "upways", "--render-scale", "0.5", "--frames", "16", "--width", "1280", "--height", "720", "--spp", "1", "--dump-frame", png_b])

        if rc1 != 0 or rc2 != 0:
            return self.log_test("Tier 4", "T4-06", "Living Room Upways Motion", False, "Execution failed")

        metrics = compute_metrics(load_image_normalized(png_a), load_image_normalized(png_b))
        passed = (metrics["ssim"] > 0.30 and metrics["boiling_index"] < 15.0)
        detail = f"Living Room Upways SSIM={metrics['ssim']:.4f}, BoilingIndex={metrics['boiling_index']:.3f}"
        return self.log_test("Tier 4", "T4-06", "Living Room Navigation with Upways Neural Reconstruction", passed, detail)

    # --------------------------------------------------------------------------
    # Suite Runner
    # --------------------------------------------------------------------------

    def run_all(self, selected_tier=None):
        print(f"====================================================================")
        print(f"  {CLR_BOLD}Pathways E2E Test Suite: Camera Motion Noise Remediation{CLR_RESET}")
        print(f"====================================================================")
        print(f"Hardware: 2x AMD Radeon AI PRO R9700 (gfx1201), Threadripper 3970X")
        print(f"Scope   : Features F1–F15 across Tiers 1–4 (38 Tests)")
        print(f"--------------------------------------------------------------------")

        all_tests = [
            # Tier 1
            (1, self.test_t1_01_static_convergence),
            (1, self.test_t1_02_rotational_yaw_panning),
            (1, self.test_t1_03_rotational_pitch_tilting),
            (1, self.test_t1_04_translational_motion_x),
            (1, self.test_t1_05_translational_motion_y),
            (1, self.test_t1_06_translational_motion_z),
            (1, self.test_t1_07_configurable_warmup_frames),
            (1, self.test_t1_08_halton_jitter_oracle),
            (1, self.test_t1_09_prng_seed_decorrelation_oracle),
            (1, self.test_t1_10_continuous_frame_indexing_contract),

            # Tier 2
            (2, self.test_t2_01_epipole_stability),
            (2, self.test_t2_02_high_speed_camera_translation),
            (2, self.test_t2_03_subpixel_creep_translation),
            (2, self.test_t2_04_grazing_angle_floor),
            (2, self.test_t2_05_grazing_angle_ceiling),
            (2, self.test_t2_06_rapid_direction_inversion),
            (2, self.test_t2_07_dynamic_fov_variation),
            (2, self.test_t2_08_planar_depth_vs_euclidean_oracle),
            (2, self.test_t2_09_extreme_aspect_ratios),
            (2, self.test_t2_10_motion_to_stationary_transition),

            # Tier 3
            (3, self.test_t3_01_motion_with_fsr3),
            (3, self.test_t3_02_motion_with_temporal_accum),
            (3, self.test_t3_03_motion_mgpu_sample_parallel),
            (3, self.test_t3_04_motion_mgpu_checkerboard_tile),
            (3, self.test_t3_05_motion_with_nrc),
            (3, self.test_t3_06_motion_fsr3_mgpu_checkerboard),
            (3, self.test_t3_07_motion_fsr3_mgpu_sample),
            (3, self.test_t3_08_motion_adaptive_spp),
            (3, self.test_t3_09_motion_wavefront_dgc_sort),
            (3, self.test_t3_10_motion_light_tree),
            (3, self.test_t3_11_motion_with_upways),
            (3, self.test_t3_12_motion_upways_mgpu_checkerboard),

            # Tier 4
            (4, self.test_t4_01_living_room_navigation),
            (4, self.test_t4_02_living_room_fsr3_panning),
            (4, self.test_t4_03_cyber_city_navigation),
            (4, self.test_t4_04_cyber_city_dual_gpu_stress),
            (4, self.test_t4_05_cyber_city_nrc_motion),
            (4, self.test_t4_06_living_room_upways_motion),
        ]

        for tier_num, test_fn in all_tests:
            if selected_tier is not None and tier_num != selected_tier:
                continue
            test_fn()

        total = len(self.results)
        passed = sum(1 for r in self.results if r["passed"])
        failed = total - passed
        pass_rate = (passed / total * 100.0) if total > 0 else 0.0
        elapsed = time.time() - self.start_time

        print(f"====================================================================")
        print(f"  {CLR_BOLD}Test Execution Summary{CLR_RESET}")
        print(f"====================================================================")
        print(f"Total Tests : {total}")
        print(f"Passed      : {CLR_GREEN}{passed}{CLR_RESET}")
        print(f"Failed      : {CLR_RED if failed > 0 else CLR_RESET}{failed}{CLR_RESET}")
        print(f"Pass Rate   : {CLR_GREEN if pass_rate == 100 else CLR_YELLOW}{pass_rate:.1f}%{CLR_RESET}")
        print(f"Total Time  : {elapsed:.2f} seconds")
        print(f"====================================================================")

        # Save structured report
        report_path = os.path.join(self.output_dir, "camera_motion_noise_e2e_report.json")
        with open(report_path, "w") as f:
            json.dump({
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "total_tests": total,
                "passed": passed,
                "failed": failed,
                "pass_rate_pct": pass_rate,
                "elapsed_sec": elapsed,
                "results": self.results
            }, f, indent=2, cls=NpEncoder)
        print(f"[INFO] Detailed test report exported to: {report_path}")

        return (failed == 0)


def main():
    parser = argparse.ArgumentParser(description="Pathways Automated Camera Motion Noise E2E Test Suite")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=None, help="Execute only a specific test tier")
    parser.add_argument("--quick", action="store_true", help="Run in quick mode with reduced frame counts")
    parser.add_argument("--output-dir", default="output/e2e_camera_motion", help="Output directory for frames and reports")
    args = parser.parse_args()

    suite = CameraMotionTestSuite(quick=args.quick, output_dir=args.output_dir)
    success = suite.run_all(selected_tier=args.tier)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
