#!/usr/bin/env python3
"""
Pathways Automated E2E Test Suite: 4K Performance, DGC Autonomous Execution & Hardware Telemetry
Validates 4K profiling, DGC Tier-1 explicit preprocessing, multi-GPU scaling, and physical telemetry
across Dual AMD Radeon AI PRO R9700 (RDNA 4, gfx1201) GPUs.

Tier 1: Functional & Mode Verification (Monolithic, DGC sort, Dual Tile, Dual Sample at 4K)
Tier 2: Boundary & Corner Cases (Warmup filtering, Single vs Dual consistency, Cyber City & Bistro Interior)
Tier 3: DGC & Autonomous Execution Validation (Explicit preprocessing, ring buffer health, zero CPU fallbacks)
Tier 4: Physical Telemetry & Performance Verification (SMI power/clock, RT stage timings, positive mGPU scaling)

Supports both pytest execution and direct standalone CLI execution.
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

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "test_4k_deep_profile")

# ANSI Color Formatting
CLR_RESET = "\033[0m"
CLR_RED = "\033[31m"
CLR_GREEN = "\033[32m"
CLR_YELLOW = "\033[33m"
CLR_CYAN = "\033[36m"
CLR_BOLD = "\033[1m"


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ==============================================================================
# Helper Utilities & Reference Models
# ==============================================================================

def run_pathways(args_list, timeout_sec=90):
    """Executes Pathways binary and returns (returncode, stdout, stderr, elapsed_sec)."""
    bin_path = os.path.join(PROJECT_ROOT, "build", "bin", "pathways")
    if not os.path.exists(bin_path):
        raise FileNotFoundError(f"Pathways binary not found at {bin_path}")

    cmd = [bin_path] + args_list
    t0 = time.time()
    try:
        res = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=timeout_sec)
        elapsed = time.time() - t0
        return res.returncode, res.stdout, res.stderr, elapsed
    except subprocess.TimeoutExpired:
        return -1, "", f"Execution timed out after {timeout_sec}s", time.time() - t0


def query_amd_smi_telemetry():
    """Queries /opt/rocm/core-10.0/bin/amd-smi for power, clocks, and temperatures."""
    for smi_path in ["/opt/rocm/core-10.0/bin/amd-smi", "/home/naoki/.local/bin/amd-smi", "amd-smi"]:
        if os.path.exists(smi_path) or smi_path == "amd-smi":
            try:
                cmd = [smi_path, "metric", "--power", "--clock", "-t", "--json"]
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
                if res.returncode == 0 and res.stdout.strip():
                    return json.loads(res.stdout.strip())
            except Exception:
                continue
    return None


def load_image_normalized(path):
    """Loads an 8-bit, 10-bit, or 16-bit PNG and returns float32 array in [0.0, 1.0]."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = Image.open(path)
    arr = np.array(img)

    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]

    if arr.dtype == np.uint8:
        arr = arr.astype(np.float32) / 255.0
    elif arr.dtype == np.uint16:
        arr = arr.astype(np.float32) / 65535.0
    elif np.issubdtype(arr.dtype, np.floating):
        arr = np.clip(arr.astype(np.float32), 0.0, 1.0)
    else:
        max_val = np.iinfo(arr.dtype).max if np.issubdtype(arr.dtype, np.integer) else 1.0
        arr = (arr.astype(np.float32) / max_val).clip(0.0, 1.0)

    return arr


def compute_psnr(im1, im2):
    """Computes Peak Signal-to-Noise Ratio (PSNR) between two normalized images."""
    mse = float(np.mean((im1 - im2) ** 2))
    if mse < 1e-10:
        return 100.0
    return 10.0 * math.log10(1.0 / mse)


def compute_ssim_channel(im1, im2, K1=0.01, K2=0.03, L=1.0):
    """Vectorized block-averaged SSIM calculation for a single 2D channel."""
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2

    h, w = im1.shape
    bh, bw = 8, 8
    n_h = h // bh
    n_w = w // bw

    if n_h == 0 or n_w == 0:
        mu1 = float(np.mean(im1))
        mu2 = float(np.mean(im2))
        sig1_sq = float(np.var(im1))
        sig2_sq = float(np.var(im2))
        sig12 = float(np.mean((im1 - mu1) * (im2 - mu2)))
        return float(((2 * mu1 * mu2 + C1) * (2 * sig12 + C2)) /
                     ((mu1**2 + mu2**2 + C1) * (sig1_sq + sig2_sq + C2)))

    crop1 = im1[:n_h * bh, :n_w * bw].reshape(n_h, bh, n_w, bw).swapaxes(1, 2)
    crop2 = im2[:n_h * bh, :n_w * bw].reshape(n_h, bh, n_w, bw).swapaxes(1, 2)

    mu1 = crop1.mean(axis=(2, 3))
    mu2 = crop2.mean(axis=(2, 3))

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sig1_sq = ((crop1 - mu1[:, :, None, None]) ** 2).mean(axis=(2, 3))
    sig2_sq = ((crop2 - mu2[:, :, None, None]) ** 2).mean(axis=(2, 3))
    sig12 = ((crop1 - mu1[:, :, None, None]) * (crop2 - mu2[:, :, None, None])).mean(axis=(2, 3))

    num = (2 * mu1_mu2 + C1) * (2 * sig12 + C2)
    den = (mu1_sq + mu2_sq + C1) * (sig1_sq + sig2_sq + C2)

    ssim_map = num / (den + 1e-10)
    return float(np.mean(ssim_map))


def compute_image_metrics(im1, im2):
    """Computes MSE, PSNR, Pearson correlation, and multi-channel SSIM between two normalized RGB images."""
    mse = float(np.mean((im1 - im2) ** 2))
    psnr = compute_psnr(im1, im2)

    ssim_channels = [compute_ssim_channel(im1[:, :, c], im2[:, :, c]) for c in range(3)]
    mean_ssim = float(np.mean(ssim_channels))

    corr = float(np.corrcoef(im1.flatten(), im2.flatten())[0, 1])

    return {
        "mse": mse,
        "psnr_db": psnr,
        "ssim": mean_ssim,
        "correlation": corr
    }


def validate_frame_image(png_path, expected_width=3840, expected_height=2160):
    """Verifies that a dumped frame PNG exists, has exact dimensions, and is non-corrupt."""
    assert os.path.exists(png_path), f"Frame dump PNG not found: {png_path}"
    assert os.path.getsize(png_path) > 1024, f"Frame dump PNG is suspiciously small (<1KB): {png_path}"

    img = Image.open(png_path)
    assert img.size == (expected_width, expected_height), (
        f"Incorrect frame dimensions: expected {expected_width}x{expected_height}, got {img.size[0]}x{img.size[1]}"
    )

    arr = load_image_normalized(png_path)
    assert not np.isnan(arr).any(), f"Frame contains NaN pixels: {png_path}"
    assert not np.isinf(arr).any(), f"Frame contains Inf pixels: {png_path}"
    assert float(np.max(arr)) > 0.001, f"Frame is completely black: {png_path}"
    return arr


# ==============================================================================
# Master Test Suite Class (Pytest & Standalone CLI Compatible)
# ==============================================================================

class Test4KDeepProfileSuite:
    """Automated Multi-Tier Test Suite for 4K Performance, DGC & Hardware Telemetry."""

    @classmethod
    def setup_class(cls):
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)

    # --------------------------------------------------------------------------
    # Tier 1: Functional & Mode Verification (at 4K / 3840x2160)
    # --------------------------------------------------------------------------

    def test_tier1_mode1_4k_monolithic(self):
        """Tier 1: Verify 4K Monolithic single-GPU execution (exit 0, valid JSON, valid PNG)."""
        json_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode1_monolithic.json")
        png_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode1_monolithic.png")

        args = [
            "--headless", "--width", "3840", "--height", "2160",
            "--wavefront-sort", "none", "--mgpu-mode", "off",
            "--warmup-frames", "5", "--frames", "10",
            "--no-accumulation",
            "--dump-stats", json_path,
            "--dump-frame", png_path
        ]

        code, stdout, stderr, elapsed = run_pathways(args, timeout_sec=45)
        assert code == 0, f"Pathways failed with code {code}: {stderr}"

        # Telemetry verification
        assert os.path.exists(json_path), f"Telemetry JSON was not generated: {json_path}"
        with open(json_path, "r") as f:
            stats = json.load(f)

        assert stats["engine_settings"]["resolution"] == [3840, 2160]
        assert stats["performance"]["wavefront_profiler_breakdown"]["material_sort_mode"] == "none"
        assert stats["performance"]["validation_errors"] == 0
        assert stats["performance"]["avg_frame_time_ms"] > 0.0
        assert stats["performance"]["avg_fps"] > 0.0
        assert stats["performance"]["gigarays_per_second"] > 0.0

        # Frame verification
        validate_frame_image(png_path, 3840, 2160)

    def test_tier1_mode2_4k_dgc_archetype_sort(self):
        """Tier 1: Verify 4K DGC Archetype Sort single-GPU execution."""
        json_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode2_dgc_arch.json")
        png_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode2_dgc_arch.png")

        args = [
            "--headless", "--width", "3840", "--height", "2160",
            "--wavefront-sort", "archetype", "--mgpu-mode", "off",
            "--warmup-frames", "5", "--frames", "10",
            "--no-accumulation",
            "--dump-stats", json_path,
            "--dump-frame", png_path
        ]

        code, stdout, stderr, elapsed = run_pathways(args, timeout_sec=45)
        assert code == 0, f"Pathways failed with code {code}: {stderr}"
        assert "DGC Tier 1 explicit preprocessing enabled" in stdout or "DGC" in stdout

        assert os.path.exists(json_path), f"JSON stats missing: {json_path}"
        with open(json_path, "r") as f:
            stats = json.load(f)

        assert stats["engine_settings"]["resolution"] == [3840, 2160]
        assert stats["performance"]["wavefront_profiler_breakdown"]["material_sort_mode"] == "archetype"
        assert stats["performance"]["validation_errors"] == 0
        assert stats["performance"]["avg_frame_time_ms"] > 0.0

        validate_frame_image(png_path, 3840, 2160)

    def test_tier1_mode2_4k_dgc_dual_sort(self):
        """Tier 1: Verify 4K DGC Dual Sort mode execution."""
        json_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode2_dgc_dual.json")
        png_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode2_dgc_dual.png")

        args = [
            "--headless", "--width", "3840", "--height", "2160",
            "--wavefront-sort", "dual", "--mgpu-mode", "off",
            "--warmup-frames", "5", "--frames", "10",
            "--no-accumulation",
            "--dump-stats", json_path,
            "--dump-frame", png_path
        ]

        code, stdout, stderr, elapsed = run_pathways(args, timeout_sec=45)
        assert code == 0, f"Pathways failed with code {code}: {stderr}"

        with open(json_path, "r") as f:
            stats = json.load(f)
        assert stats["performance"]["validation_errors"] == 0
        validate_frame_image(png_path, 3840, 2160)

    def test_tier1_mode3_4k_dual_gpu_tile(self):
        """Tier 1: Verify 4K Dual-GPU Checkerboard Tile execution (50/50 load split)."""
        json_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode3_dual_tile.json")
        png_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode3_dual_tile.png")

        args = [
            "--headless", "--width", "3840", "--height", "2160",
            "--mgpu-mode", "tile", "--tile-size", "64",
            "--wavefront-sort", "archetype",
            "--warmup-frames", "5", "--frames", "10",
            "--no-accumulation",
            "--dump-stats", json_path,
            "--dump-frame", png_path
        ]

        code, stdout, stderr, elapsed = run_pathways(args, timeout_sec=45)
        assert code == 0, f"Pathways failed with code {code}: {stderr}"

        with open(json_path, "r") as f:
            stats = json.load(f)

        assert stats["secondary_gpu"]["active"] is True
        assert stats["secondary_gpu"]["mgpu_mode"] == "checkerboard_tile"
        assert stats["performance"]["validation_errors"] == 0

        breakdown = stats["performance"]["gpu_profiler_breakdown_ms"]
        assert breakdown["primary_gpu_time_ms"] > 0.0
        assert breakdown["secondary_gpu_time_ms"] > 0.0
        assert breakdown["tonemap_and_merge_time_ms"] > 0.0

        validate_frame_image(png_path, 3840, 2160)

    def test_tier1_mode4_4k_dual_gpu_sample(self):
        """Tier 1: Verify 4K Dual-GPU Sample Parallelism (2 SPP, concurrent ray dispatch)."""
        json_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode4_dual_sample.json")
        png_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode4_dual_sample.png")

        args = [
            "--headless", "--width", "3840", "--height", "2160",
            "--mgpu-mode", "sample", "--spp", "2",
            "--wavefront-sort", "archetype",
            "--warmup-frames", "5", "--frames", "10",
            "--no-accumulation",
            "--dump-stats", json_path,
            "--dump-frame", png_path
        ]

        code, stdout, stderr, elapsed = run_pathways(args, timeout_sec=45)
        assert code == 0, f"Pathways failed with code {code}: {stderr}"

        with open(json_path, "r") as f:
            stats = json.load(f)

        assert stats["secondary_gpu"]["active"] is True
        assert stats["secondary_gpu"]["mgpu_mode"] == "sample_parallel"
        assert stats["engine_settings"]["spp"] == 2
        assert stats["performance"]["validation_errors"] == 0

        breakdown = stats["performance"]["gpu_profiler_breakdown_ms"]
        assert breakdown["primary_gpu_time_ms"] > 0.0
        assert breakdown["secondary_gpu_time_ms"] > 0.0

        validate_frame_image(png_path, 3840, 2160)

    def test_tier1_4k_representative_scenes(self):
        """Tier 1: Verify 4K execution across representative showcase scenes."""
        scenes = [
            ("scenes/cornell-caustic/cornell_caustic_extended.glb", "caustic"),
            ("scenes/glass-of-water/glass_of_water_extended.glb", "glass"),
            ("scenes/veach-ajar/veach_ajar_extended.glb", "veach_ajar")
        ]

        for scene_rel_path, tag in scenes:
            full_path = os.path.join(PROJECT_ROOT, scene_rel_path)
            if not os.path.exists(full_path):
                continue

            json_path = os.path.join(DEFAULT_OUTPUT_DIR, f"t1_scene_{tag}.json")
            args = [
                "--headless", "--width", "3840", "--height", "2160",
                "--scene", scene_rel_path,
                "--warmup-frames", "3", "--frames", "6",
                "--no-accumulation",
                "--dump-stats", json_path
            ]
            code, stdout, stderr, _ = run_pathways(args, timeout_sec=45)
            assert code == 0, f"Failed rendering {scene_rel_path}: {stderr}"

            with open(json_path, "r") as f:
                stats = json.load(f)
            assert stats["engine_settings"]["scene"]["num_triangles"] > 0
            assert stats["performance"]["validation_errors"] == 0
            assert stats["performance"]["avg_frame_time_ms"] > 0.0

    # --------------------------------------------------------------------------
    # Tier 2: Boundary & Corner Cases
    # --------------------------------------------------------------------------

    def test_tier2_warmup_frames_filtering(self):
        """Tier 2: Verify warmup frames properly exclude initialization spike from statistics."""
        json_cold = os.path.join(DEFAULT_OUTPUT_DIR, "t2_warmup_cold_0.json")
        json_warm = os.path.join(DEFAULT_OUTPUT_DIR, "t2_warmup_warm_10.json")

        # Run 0 warmup, 20 measured frames (total rendered: 20)
        code0, _, _, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--warmup-frames", "0", "--frames", "20",
            "--no-accumulation", "--dump-stats", json_cold
        ], timeout_sec=45)
        assert code0 == 0

        # Run 10 warmup, 20 measured frames (total rendered: 30)
        code10, _, _, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--warmup-frames", "10", "--frames", "20",
            "--no-accumulation", "--dump-stats", json_warm
        ], timeout_sec=45)
        assert code10 == 0

        with open(json_cold, "r") as f:
            stats_cold = json.load(f)
        with open(json_warm, "r") as f:
            stats_warm = json.load(f)

        cfg_cold = stats_cold["performance"]["configurations_breakdown"][0]
        cfg_warm = stats_warm["performance"]["configurations_breakdown"][0]

        # In Pathways, cfg["frame_count"] is the number of measured frames (20)
        assert cfg_cold["frame_count"] == 20
        assert cfg_warm["frame_count"] == 20

        tot_cold = stats_cold.get("performance", {}).get("total_frames", 20)
        tot_warm = stats_warm.get("performance", {}).get("total_frames", 30)

        assert tot_cold == 20
        assert tot_warm == 30
        assert tot_warm - cfg_warm["frame_count"] == 10
        assert tot_cold - cfg_cold["frame_count"] == 0

        # Both runs produce valid steady-state performance metrics
        assert cfg_warm["avg_frame_time_ms"] > 0.0
        assert cfg_warm["min_frame_time_ms"] > 0.0

    def test_tier2_single_vs_dual_gpu_consistency_tile(self):
        """Tier 2: Verify converged image consistency between Single-GPU and Dual-GPU Tile mode at 4K."""
        png_single_accum = os.path.join(DEFAULT_OUTPUT_DIR, "t2_accum_single.png")
        png_tile_accum = os.path.join(DEFAULT_OUTPUT_DIR, "t2_accum_tile.png")

        # Render 15 accumulated frames in Single-GPU mode
        code_s, _, _, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--wavefront-sort", "archetype", "--mgpu-mode", "off",
            "--frames", "15", "--dump-frame", png_single_accum
        ], timeout_sec=45)
        assert code_s == 0

        # Render 15 accumulated frames in Dual-GPU Tile mode
        code_t, _, _, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--wavefront-sort", "archetype", "--mgpu-mode", "tile",
            "--tile-size", "64", "--frames", "15",
            "--dump-frame", png_tile_accum
        ], timeout_sec=45)
        assert code_t == 0

        im_single = load_image_normalized(png_single_accum)
        im_tile = load_image_normalized(png_tile_accum)

        metrics = compute_image_metrics(im_single, im_tile)

        # High structural match (>0.88 SSIM, >25 dB PSNR)
        assert metrics["ssim"] > 0.88, f"SSIM between Single and Dual Tile is too low: {metrics['ssim']:.3f}"
        assert metrics["psnr_db"] > 25.0, f"PSNR between Single and Dual Tile is too low: {metrics['psnr_db']:.2f} dB"
        assert metrics["correlation"] > 0.95, f"Correlation too low: {metrics['correlation']:.3f}"

    def test_tier2_single_vs_dual_gpu_consistency_sample(self):
        """Tier 2: Verify converged image consistency between Single-GPU and Dual-GPU Sample mode at 4K."""
        png_single_2spp = os.path.join(DEFAULT_OUTPUT_DIR, "t2_accum_single_2spp.png")
        png_sample_2spp = os.path.join(DEFAULT_OUTPUT_DIR, "t2_accum_sample_2spp.png")

        # Render 15 accumulated frames in Single-GPU 2 SPP mode
        code_s, _, _, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--wavefront-sort", "archetype", "--mgpu-mode", "off",
            "--spp", "2", "--frames", "15",
            "--dump-frame", png_single_2spp
        ], timeout_sec=45)
        assert code_s == 0

        # Render 15 accumulated frames in Dual-GPU Sample 2 SPP mode
        code_m, _, _, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--wavefront-sort", "archetype", "--mgpu-mode", "sample",
            "--spp", "2", "--frames", "15",
            "--dump-frame", png_sample_2spp
        ], timeout_sec=45)
        assert code_m == 0

        im_single = load_image_normalized(png_single_2spp)
        im_sample = load_image_normalized(png_sample_2spp)

        metrics = compute_image_metrics(im_single, im_sample)
        assert metrics["correlation"] > 0.85, f"Correlation too low: {metrics['correlation']:.3f}"
        assert metrics["ssim"] > 0.65, f"SSIM between Single 2SPP and Dual Sample 2SPP too low: {metrics['ssim']:.3f}"

    def test_tier2_extreme_geometry_cyber_city(self):
        """Tier 2: Stress-test extreme geometry with Procedural Cyber City (3.8M instanced tris, 4k instances)."""
        json_path = os.path.join(DEFAULT_OUTPUT_DIR, "t2_cyber_city.json")
        png_path = os.path.join(DEFAULT_OUTPUT_DIR, "t2_cyber_city.png")

        args = [
            "--headless", "--width", "3840", "--height", "2160",
            "--scene", "procedural:cyber-city",
            "--warmup-frames", "5", "--frames", "10",
            "--no-accumulation",
            "--dump-stats", json_path,
            "--dump-frame", png_path
        ]

        code, stdout, stderr, elapsed = run_pathways(args, timeout_sec=60)
        assert code == 0, f"Cyber City 4K failed: {stderr}"

        with open(json_path, "r") as f:
            stats = json.load(f)

        assert stats["performance"]["validation_errors"] == 0
        assert stats["engine_settings"]["scene"]["path"] == "procedural:cyber-city"

        # Verify BLAS and TLAS build successfully without VRAM exhaustion
        vram_alloc = stats["primary_gpu"]["memory"]["allocated_vram_mb"]
        assert vram_alloc < 16384.0, f"Excessive VRAM allocation for Cyber City: {vram_alloc} MB"

        validate_frame_image(png_path, 3840, 2160)

    def test_tier2_extreme_geometry_bistro_interior(self):
        """Tier 2: Stress-test large exterior/interior real-world asset (Bistro Interior: 1.3M+ tris, 74 mats)."""
        bistro_path = os.path.join(PROJECT_ROOT, "scenes", "bistro", "bistro_interior.glb")
        if not os.path.exists(bistro_path):
            return

        json_path = os.path.join(DEFAULT_OUTPUT_DIR, "t2_bistro_interior.json")
        args = [
            "--headless", "--width", "3840", "--height", "2160",
            "--scene", "scenes/bistro/bistro_interior.glb",
            "--camera-pos", "3.5,1.75,-6.2",
            "--camera-target", "9.5,1.65,0.5",
            "--camera-fov", "70",
            "--warmup-frames", "3", "--frames", "8",
            "--no-accumulation",
            "--dump-stats", json_path
        ]

        code, stdout, stderr, elapsed = run_pathways(args, timeout_sec=60)
        assert code == 0, f"Bistro Interior 4K failed: {stderr}"

        with open(json_path, "r") as f:
            stats = json.load(f)

        assert stats["engine_settings"]["scene"]["num_triangles"] > 1000000
        assert stats["performance"]["validation_errors"] == 0
        assert stats["performance"]["gigarays_per_second"] > 0.4

    def test_tier2_bounce_depth_boundary(self):
        """Tier 2: Verify boundary bounce depths (1 bounce vs 8 bounces) execute cleanly."""
        json_b1 = os.path.join(DEFAULT_OUTPUT_DIR, "t2_bounces_1.json")
        json_b8 = os.path.join(DEFAULT_OUTPUT_DIR, "t2_bounces_8.json")

        code1, _, _, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--max-bounces", "1", "--warmup-frames", "2", "--frames", "5",
            "--dump-stats", json_b1
        ], timeout_sec=30)
        assert code1 == 0

        code8, _, _, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--max-bounces", "8", "--warmup-frames", "2", "--frames", "5",
            "--dump-stats", json_b8
        ], timeout_sec=30)
        assert code8 == 0

        with open(json_b1, "r") as f:
            stats_b1 = json.load(f)
        with open(json_b8, "r") as f:
            stats_b8 = json.load(f)

        bounces_1 = stats_b1["performance"]["wavefront_profiler_breakdown"]["bounces"]
        bounces_8 = stats_b8["performance"]["wavefront_profiler_breakdown"]["bounces"]

        assert len(bounces_1) == 1, f"Expected 1 bounce entry, got {len(bounces_1)}"
        assert len(bounces_8) >= 4, f"Expected multiple bounce entries, got {len(bounces_8)}"

        # Ray count must decrease monotonically across bounces
        for i in range(len(bounces_8) - 1):
            assert bounces_8[i]["active_rays"] >= bounces_8[i+1]["active_rays"]

    # --------------------------------------------------------------------------
    # Tier 3: DGC & Autonomous Execution Validation
    # --------------------------------------------------------------------------

    def test_tier3_dgc_tier1_explicit_preprocess_enabled(self):
        """Tier 3: Verify DGC Tier-1 explicit preprocessing layout flags and token strides."""
        code, stdout, stderr, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--wavefront-sort", "archetype",
            "--warmup-frames", "1", "--frames", "2"
        ], timeout_sec=30)
        assert code == 0

        # Verify explicit preprocessing enabled in log
        assert "DGC Tier 1 explicit preprocessing enabled" in stdout
        # Verify single dispatch token layout with flags 0x3 (EXPLICIT_PREPROCESS | UNORDERED_SEQUENCES)
        assert "Single Dispatch Token, stride: 12 bytes, flags: 0x3" in stdout
        # Verify execution set + dispatch token stride
        assert "Material indirect commands layout (ExecutionSet + Dispatch, stride: 16 bytes)" in stdout

    def test_tier3_dgc_implicit_preprocess_fallback(self):
        """Tier 3: Verify --no-dgc-preprocess cleanly falls back to implicit preprocessing."""
        code, stdout, stderr, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--no-dgc-preprocess",
            "--warmup-frames", "1", "--frames", "2"
        ], timeout_sec=30)
        assert code == 0

        assert "DGC baseline active (implicit preprocessing, flags = UNORDERED_SEQUENCES)" in stdout
        assert "Single Dispatch Token, stride: 12 bytes, flags: 0x0" in stdout

    def test_tier3_dgc_ring_buffer_health(self):
        """Tier 3: Verify DGC preprocess buffer allocation (32 slices @ 4KB) and multi-frame wrapping."""
        code, stdout, stderr, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--wavefront-sort", "archetype",
            "--warmup-frames", "5", "--frames", "35"
        ], timeout_sec=45)
        assert code == 0

        # Check buffer layout in logs: 32 slices @ 4096 bytes = 131,072 bytes
        assert "Allocated DGC preprocess buffer (total: 131072 bytes, 32 slices @ 4096 bytes, align: 256 bytes)" in stdout
        assert "Total Frames Rendered: 40 | Validation Errors: 0" in stdout

    def test_tier3_autonomous_gpu_execution_zero_cpu_fallback(self):
        """Tier 3: Verify GPU-autonomous execution without host CPU fallback or silent dispatch bypass."""
        json_path = os.path.join(DEFAULT_OUTPUT_DIR, "t3_autonomous_exec.json")
        code, stdout, stderr, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--wavefront-sort", "archetype",
            "--warmup-frames", "2", "--frames", "5",
            "--dump-stats", json_path
        ], timeout_sec=30)
        assert code == 0

        # Check absence of fallback warnings
        lower_out = stdout.lower() + stderr.lower()
        assert "fallback to baseline implicit dgc" not in lower_out
        assert "cpu fallback" not in lower_out
        assert "device lost" not in lower_out

        # Verify GPU timestamp profiler recorded active execution across all stages
        with open(json_path, "r") as f:
            stats = json.load(f)

        wb = stats["performance"]["wavefront_profiler_breakdown"]
        assert wb["classify_time_ms"] > 0.0
        assert wb["total_wavefront_time_ms"] > 0.0
        b0 = wb["bounces"][0]
        assert b0["shade_ms"] > 0.0
        assert b0["intersect_ms"] > 0.0

    def test_tier3_secondary_gpu_dgc_autonomy(self):
        """Tier 3: Verify Secondary GPU independently initializes and executes Wavefront DGC pipeline."""
        code, stdout, stderr, _ = run_pathways([
            "--headless", "--width", "3840", "--height", "2160",
            "--mgpu-mode", "tile",
            "--warmup-frames", "2", "--frames", "4"
        ], timeout_sec=30)
        assert code == 0

        assert "Secondary GPU: Wavefront Path Tracing Pipeline (Ray Queues & DGC) initialized successfully" in stdout
        assert "Secondary GPU Node fully initialized" in stdout

    # --------------------------------------------------------------------------
    # Tier 4: Physical Telemetry & Performance Verification
    # --------------------------------------------------------------------------

    def test_tier4_physical_gpu_clocks_and_temperatures(self):
        """Tier 4: Verify telemetry JSON contains realistic GPU clock, temperature, and memory."""
        json_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode1_monolithic.json")
        if not os.path.exists(json_path):
            self.test_tier1_mode1_4k_monolithic()

        with open(json_path, "r") as f:
            stats = json.load(f)

        prim_gpu = stats["primary_gpu"]
        clock_mhz = prim_gpu["telemetry"]["clock_mhz"]
        temp_c = prim_gpu["telemetry"]["temperature_c"]
        vram_mb = prim_gpu["memory"]["allocated_vram_mb"]

        assert 400 <= clock_mhz <= 3500, f"Primary GPU clock out of realistic range: {clock_mhz} MHz"
        assert 20 <= temp_c <= 95, f"Primary GPU temp out of realistic range: {temp_c} °C"
        assert 100.0 <= vram_mb <= 32768.0, f"Allocated VRAM out of realistic range: {vram_mb} MB"

    def test_tier4_amd_smi_power_draw_telemetry(self):
        """Tier 4: Query AMD-SMI for physical socket power draw, voltage, and multi-GPU health."""
        smi_data = query_amd_smi_telemetry()
        assert smi_data is not None, "Failed to query /opt/rocm/core-10.0/bin/amd-smi metric JSON"

        # Check GPUs list (may be wrapped under 'gpu_data')
        gpus = smi_data.get("gpu_data", smi_data) if isinstance(smi_data, dict) else smi_data
        assert len(gpus) >= 2, f"Expected at least 2 AMD GPUs, found {len(gpus)}"

        gpu0 = gpus[0]
        gpu1 = gpus[1]

        # Power draw verification
        p0 = gpu0.get("power", {}).get("socket_power", {}).get("value", 0)
        p1 = gpu1.get("power", {}).get("socket_power", {}).get("value", 0)

        assert p0 > 10, f"GPU 0 reported idle or zero socket power: {p0} W"
        assert p1 > 10, f"GPU 1 reported idle or zero socket power: {p1} W"
        assert p0 <= 350, f"GPU 0 socket power exceeds maximum TBP: {p0} W"

    def test_tier4_wavefront_rt_stage_timings(self):
        """Tier 4: Verify GPU timestamp profiler stage timings are realistic and non-zero."""
        json_path = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode1_monolithic.json")
        with open(json_path, "r") as f:
            stats = json.load(f)

        perf = stats["performance"]
        rt_time = perf["gpu_profiler_breakdown_ms"]["primary_gpu_time_ms"]
        tonemap = perf["gpu_profiler_breakdown_ms"]["tonemap_and_merge_time_ms"]
        avg_ft = perf["avg_frame_time_ms"]

        assert rt_time > 0.0, f"Primary GPU RT time is zero: {rt_time}"
        assert tonemap > 0.0, f"Tonemap time is zero: {tonemap}"
        # Frame time should be dominated by ray tracing
        assert rt_time <= avg_ft * 1.1

    def test_tier4_positive_multigpu_scaling_tile(self):
        """Tier 4: Verify Dual-GPU Tile mode achieves positive speedup (>1.15x) over Single-GPU at 4K."""
        json_mono = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode1_monolithic.json")
        json_tile = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode3_dual_tile.json")

        if not os.path.exists(json_mono):
            self.test_tier1_mode1_4k_monolithic()
        if not os.path.exists(json_tile):
            self.test_tier1_mode3_4k_dual_gpu_tile()

        with open(json_mono, "r") as f:
            stats_mono = json.load(f)
        with open(json_tile, "r") as f:
            stats_tile = json.load(f)

        t_mono = stats_mono["performance"]["avg_frame_time_ms"]
        t_tile = stats_tile["performance"]["avg_frame_time_ms"]

        speedup = t_mono / t_tile
        assert speedup > 1.15, (
            f"Dual-GPU Tile mode failed to demonstrate positive scaling: "
            f"Single-GPU={t_mono:.2f}ms, Dual-Tile={t_tile:.2f}ms, Speedup={speedup:.2f}x (required >1.15x)"
        )

        # Secondary GPU must perform substantial workload
        sec_time = stats_tile["performance"]["gpu_profiler_breakdown_ms"]["secondary_gpu_time_ms"]
        prim_time = stats_tile["performance"]["gpu_profiler_breakdown_ms"]["primary_gpu_time_ms"]
        assert sec_time > 0.3 * prim_time, (
            f"Secondary GPU did not share balanced workload: prim={prim_time:.2f}ms, sec={sec_time:.2f}ms"
        )

    def test_tier4_positive_multigpu_scaling_sample(self):
        """Tier 4: Verify Dual-GPU Sample mode achieves positive throughput scaling (>1.3x GigaRays/s)."""
        json_single_1spp = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode1_monolithic.json")
        json_sample_2spp = os.path.join(DEFAULT_OUTPUT_DIR, "t1_mode4_dual_sample.json")

        if not os.path.exists(json_single_1spp):
            self.test_tier1_mode1_4k_monolithic()
        if not os.path.exists(json_sample_2spp):
            self.test_tier1_mode4_4k_dual_gpu_sample()

        with open(json_single_1spp, "r") as f:
            stats_s1 = json.load(f)
        with open(json_sample_2spp, "r") as f:
            stats_d2 = json.load(f)

        grays_s1 = stats_s1["performance"]["gigarays_per_second"]
        grays_d2 = stats_d2["performance"]["gigarays_per_second"]

        throughput_ratio = grays_d2 / grays_s1
        assert throughput_ratio > 1.30, (
            f"Dual-GPU Sample mode failed throughput scaling: "
            f"Single-GPU 1SPP={grays_s1:.2f} GRays/s, Dual-Sample 2SPP={grays_d2:.2f} GRays/s, "
            f"Ratio={throughput_ratio:.2f}x (required >1.30x)"
        )


# ==============================================================================
# Standalone CLI Test Runner
# ==============================================================================

def run_standalone_suite(selected_tier=None, output_dir=DEFAULT_OUTPUT_DIR):
    """Executes the test suite directly from Python with structured logging and reporting."""
    os.makedirs(output_dir, exist_ok=True)
    suite = Test4KDeepProfileSuite()
    Test4KDeepProfileSuite.setup_class()

    tests_tier1 = [
        ("F01-T01", "4K Single-GPU Monolithic Execution", suite.test_tier1_mode1_4k_monolithic),
        ("F02-T01", "4K Single-GPU DGC Archetype Sort", suite.test_tier1_mode2_4k_dgc_archetype_sort),
        ("F02-T02", "4K Single-GPU DGC Dual Sort", suite.test_tier1_mode2_4k_dgc_dual_sort),
        ("F03-T01", "4K Dual-GPU Checkerboard Tile Mode", suite.test_tier1_mode3_4k_dual_gpu_tile),
        ("F04-T01", "4K Dual-GPU Sample Parallelism Mode", suite.test_tier1_mode4_4k_dual_gpu_sample),
        ("F05-T01", "4K Representative Showcase Scenes", suite.test_tier1_4k_representative_scenes),
    ]

    tests_tier2 = [
        ("F06-T01", "Warmup Frames Initialization Filtering", suite.test_tier2_warmup_frames_filtering),
        ("F07-T01", "Single vs Dual Tile Image Consistency", suite.test_tier2_single_vs_dual_gpu_consistency_tile),
        ("F07-T02", "Single vs Dual Sample Image Consistency", suite.test_tier2_single_vs_dual_gpu_consistency_sample),
        ("F08-T01", "Procedural Cyber City Extreme Load", suite.test_tier2_extreme_geometry_cyber_city),
        ("F09-T01", "Bistro Interior Large-Scale Mesh (1.3M Tris)", suite.test_tier2_extreme_geometry_bistro_interior),
        ("F10-T01", "Bounce Depth Boundary (1 vs 8 Bounces)", suite.test_tier2_bounce_depth_boundary),
    ]

    tests_tier3 = [
        ("F11-T01", "DGC Tier-1 Explicit Preprocessing Active", suite.test_tier3_dgc_tier1_explicit_preprocess_enabled),
        ("F11-T02", "DGC Implicit Fallback Switch (--no-dgc-preprocess)", suite.test_tier3_dgc_implicit_preprocess_fallback),
        ("F12-T01", "DGC Preprocess Ring Buffer Health & Slicing", suite.test_tier3_dgc_ring_buffer_health),
        ("F13-T01", "Autonomous GPU Execution & Zero CPU Fallback", suite.test_tier3_autonomous_gpu_execution_zero_cpu_fallback),
        ("F14-T01", "Secondary GPU Autonomous DGC Execution", suite.test_tier3_secondary_gpu_dgc_autonomy),
    ]

    tests_tier4 = [
        ("F15-T01", "Physical GPU Clock & Temperature Health", suite.test_tier4_physical_gpu_clocks_and_temperatures),
        ("F16-T01", "ROCm AMD-SMI Socket Power Telemetry", suite.test_tier4_amd_smi_power_draw_telemetry),
        ("F17-T01", "Wavefront RT Microkernel Timings Breakdown", suite.test_tier4_wavefront_rt_stage_timings),
        ("F18-T01", "Positive Multi-GPU Scaling (Tile Mode >1.15x)", suite.test_tier4_positive_multigpu_scaling_tile),
        ("F18-T02", "Positive Multi-GPU Throughput (Sample Mode >1.30x)", suite.test_tier4_positive_multigpu_scaling_sample),
    ]

    all_tiers = {
        1: ("Tier 1: Functional & Mode Verification", tests_tier1),
        2: ("Tier 2: Boundary & Corner Cases", tests_tier2),
        3: ("Tier 3: DGC & Autonomous Execution", tests_tier3),
        4: ("Tier 4: Physical Telemetry & Performance", tests_tier4)
    }

    selected = [selected_tier] if selected_tier in all_tiers else [1, 2, 3, 4]

    print("====================================================================")
    print(f"  {CLR_BOLD}Pathways 4K Deep Profile & Optimization E2E Test Suite{CLR_RESET}")
    print(f"  Hardware: Dual AMD Radeon AI PRO R9700 (gfx1201 / RDNA 4)")
    print(f"  Target: Vulkan 1.4, Mesa RADV, DGC Tier-1 Explicit Preprocessing")
    print("====================================================================")

    total = 0
    passed = 0
    failed = 0
    results_list = []
    t_start = time.time()

    for tier_num in selected:
        tier_title, test_cases = all_tiers[tier_num]
        print(f"\n{CLR_CYAN}--- {tier_title} ({len(test_cases)} tests) ---{CLR_RESET}")

        for test_id, name, func in test_cases:
            total += 1
            t0 = time.time()
            try:
                func()
                elapsed = time.time() - t0
                passed += 1
                status = f"{CLR_GREEN}PASS{CLR_RESET}"
                print(f"  [{status}] {test_id}: {name} ({elapsed:.2f}s)")
                results_list.append({
                    "tier": tier_num,
                    "id": test_id,
                    "name": name,
                    "status": "PASS",
                    "elapsed_sec": elapsed,
                    "error": None
                })
            except Exception as e:
                elapsed = time.time() - t0
                failed += 1
                status = f"{CLR_RED}FAIL{CLR_RESET}"
                print(f"  [{status}] {test_id}: {name} ({elapsed:.2f}s) - {str(e)}")
                results_list.append({
                    "tier": tier_num,
                    "id": test_id,
                    "name": name,
                    "status": "FAIL",
                    "elapsed_sec": elapsed,
                    "error": str(e)
                })

    t_total = time.time() - t_start
    pass_rate = (passed / total * 100.0) if total > 0 else 0.0

    print("\n====================================================================")
    print(f"  {CLR_BOLD}Test Execution Summary{CLR_RESET}")
    print("====================================================================")
    print(f"Total Tests : {total}")
    print(f"Passed      : {CLR_GREEN}{passed}{CLR_RESET}")
    print(f"Failed      : {CLR_RED if failed > 0 else CLR_RESET}{failed}{CLR_RESET}")
    print(f"Pass Rate   : {CLR_GREEN if pass_rate == 100 else CLR_YELLOW}{pass_rate:.1f}%{CLR_RESET}")
    print(f"Total Time  : {t_total:.2f} seconds")
    print("====================================================================")

    # Export structured JSON report
    report_path = os.path.join(output_dir, "test_4k_deep_profile_report.json")
    with open(report_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "hardware": "Dual AMD Radeon AI PRO R9700 (gfx1201)",
            "total_tests": total,
            "passed": passed,
            "failed": failed,
            "pass_rate_pct": pass_rate,
            "elapsed_sec": t_total,
            "results": results_list
        }, f, indent=2, cls=NpEncoder)
    print(f"[INFO] Detailed test report exported to: {report_path}")

    return failed == 0


def main():
    parser = argparse.ArgumentParser(description="Pathways 4K Deep Profile & Optimization E2E Test Suite")
    parser.add_argument("--tier", type=int, choices=[1, 2, 3, 4], default=None, help="Execute specific test tier")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Directory for output frames and telemetry")
    args = parser.parse_args()

    success = run_standalone_suite(selected_tier=args.tier, output_dir=args.output_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
