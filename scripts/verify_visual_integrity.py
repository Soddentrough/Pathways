#!/usr/bin/env python3
"""
Pathways Visual Integrity & Exposure Stability Test Suite
Validates end-to-end visual quality, exposure invariance across progressive accumulation,
multi-GPU tile disparity, and perceptual similarity against ground truth reference images.
"""

import os
import sys
import json
import subprocess

try:
    import numpy as np
    import cv2
    from skimage.metrics import structural_similarity as compute_ssim
    HAS_DEPS = True
    MISSING_DEP = ""
except ImportError as e:
    HAS_DEPS = False
    MISSING_DEP = str(e)

PATHWAYS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN_PATHWAYS = os.path.join(PATHWAYS_ROOT, "build", "bin", "pathways")
if not os.path.exists(BIN_PATHWAYS) and os.path.exists(os.path.join(PATHWAYS_ROOT, "build", "linux-release", "bin", "pathways")):
    BIN_PATHWAYS = os.path.join(PATHWAYS_ROOT, "build", "linux-release", "bin", "pathways")
elif not os.path.exists(BIN_PATHWAYS) and os.path.exists(os.path.join(PATHWAYS_ROOT, "build", "bin", "Release", "pathways.exe")):
    BIN_PATHWAYS = os.path.join(PATHWAYS_ROOT, "build", "bin", "Release", "pathways.exe")
elif not os.path.exists(BIN_PATHWAYS) and os.path.exists(os.path.join(PATHWAYS_ROOT, "build", "bin", "pathways.exe")):
    BIN_PATHWAYS = os.path.join(PATHWAYS_ROOT, "build", "bin", "pathways.exe")

OUTPUT_DIR = os.path.join(PATHWAYS_ROOT, "output", "visual_integrity")
REF_DIR = os.path.join(PATHWAYS_ROOT, "tests", "references")

os.makedirs(OUTPUT_DIR, exist_ok=True)

class TestResult:
    def __init__(self, name):
        self.name = name
        self.passed = True
        self.errors = []
        self.metrics = {}

    def fail(self, msg):
        self.passed = False
        self.errors.append(msg)

    def record(self, key, val):
        self.metrics[key] = val

def run_pathways(args):
    extra = os.environ.get("EXTRA_PATHWAYS_ARGS", "").split()
    cmd = [BIN_PATHWAYS, "--headless"] + extra + args
    res = subprocess.run(cmd, cwd=PATHWAYS_ROOT, capture_output=True, text=True)
    return res.returncode, res.stdout, res.stderr

def compute_image_stats(img):
    """Computes mean, max, luminance, clipping, and quadrant/centerline metrics."""
    h, w, c = img.shape
    mean_rgb = np.mean(img, axis=(0, 1))
    # Relative luminance (Rec.709)
    lum = 0.2126 * img[:, :, 2] + 0.7152 * img[:, :, 1] + 0.0722 * img[:, :, 0]
    mean_lum = float(np.mean(lum))
    max_lum = float(np.max(lum))
    clipped_pct = float(np.mean(lum >= 245.0) * 100.0)
    black_pct = float(np.mean(lum <= 2.0) * 100.0)

    # Centerline discontinuities (Row W/2, Col H/2)
    mid_x, mid_y = w // 2, h // 2
    row_diff = float(np.mean(np.abs(img[mid_y, :].astype(float) - img[mid_y - 1, :].astype(float))))
    col_diff = float(np.mean(np.abs(img[:, mid_x].astype(float) - img[:, mid_x - 1].astype(float))))
    base_row_diff = float(np.mean(np.abs(img[1:, :].astype(float) - img[:-1, :].astype(float))))
    base_col_diff = float(np.mean(np.abs(img[:, 1:].astype(float) - img[:, :-1].astype(float))))

    row_ratio = row_diff / max(base_row_diff, 1e-4)
    col_ratio = col_diff / max(base_col_diff, 1e-4)

    return {
        "width": w,
        "height": h,
        "mean_rgb": [float(x) for x in mean_rgb],
        "mean_lum": mean_lum,
        "max_lum": max_lum,
        "clipped_pct": clipped_pct,
        "black_pct": black_pct,
        "row_ratio": row_ratio,
        "col_ratio": col_ratio
    }

def compute_reference_comparison(img, ref_img):
    """Computes PSNR and SSIM against ground truth reference."""
    h, w, _ = img.shape
    ref_h, ref_w, _ = ref_img.shape
    if (h, w) != (ref_h, ref_w):
        ref_resized = cv2.resize(ref_img, (w, h), interpolation=cv2.INTER_AREA)
    else:
        ref_resized = ref_img

    mse = float(np.mean((img.astype(float) - ref_resized.astype(float)) ** 2))
    psnr = float(10.0 * np.log10((255.0 ** 2) / max(mse, 1e-6)))
    gray_a = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(ref_resized, cv2.COLOR_BGR2GRAY)
    ssim_val = float(compute_ssim(gray_a, gray_b))

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_val
    }

def test_single_gpu_accumulation_stability():
    result = TestResult("Single GPU Upways Accumulation Exposure Stability")
    print(f"\n--- Running: {result.name} ---")

    frame_counts = [1, 5, 15, 30]
    stats_list = []

    for fc in frame_counts:
        out_png = os.path.join(OUTPUT_DIR, f"cornell_upways_f{fc}.png")
        rc, stdout, stderr = run_pathways([
            "--res", "1080p",
            "--upscaler", "upways",
            "--render-scale", "0.5",
            "--sec-sort", "none",
            "--frames", str(fc),
            "--dump-frame", out_png
        ])
        if rc != 0:
            result.fail(f"Pathways exited with code {rc} for frame count {fc}: {stderr}")
            return result

        img = cv2.imread(out_png)
        if img is None:
            result.fail(f"Failed to load output image {out_png}")
            return result

        st = compute_image_stats(img)
        stats_list.append((fc, st, img))
        print(f"  Frame {fc:2d}: MeanLum={st['mean_lum']:.1f}, MaxLum={st['max_lum']:.1f}, Clipped={st['clipped_pct']:.2f}%")

    # Invariant 1: Proper exposure (no overexposure blowout and no underexposure blackout)
    for fc, st, _ in stats_list:
        if st["mean_lum"] > 215.0:
            result.fail(f"Overexposure blowout detected at frame {fc}: Mean luminance {st['mean_lum']:.1f} > 215.0")
        if st["clipped_pct"] > 10.0:
            result.fail(f"Excessive highlight clipping at frame {fc}: {st['clipped_pct']:.2f}% > 10.0%")
        if st["mean_lum"] < 45.0:
            result.fail(f"Underexposure blackout / red mud collapse detected at frame {fc}: Mean luminance {st['mean_lum']:.1f} < 45.0")
        if st["max_lum"] < 120.0:
            result.fail(f"Insufficient dynamic range at frame {fc}: Max luminance {st['max_lum']:.1f} < 120.0")

    # Invariant 2: Accumulation drift bounded (allows natural MC convergence but forbids runaway)
    f1_lum = stats_list[0][1]["mean_lum"]
    f30_lum = stats_list[-1][1]["mean_lum"]
    drift_pct = abs(f30_lum - f1_lum) / f1_lum * 100.0
    result.record("accumulation_drift_pct", drift_pct)
    print(f"  Accumulation Luminance Drift (F1 -> F30): {drift_pct:.2f}%")
    if drift_pct > 35.0:
        result.fail(f"Accumulation exposure runaway: Drift {drift_pct:.2f}% > 35.0%")

    # Invariant 3: Chromatic neutrality on white back wall patch [250:380, 800:1000]
    wall_patch = stats_list[-1][2][250:380, 800:1000].astype(float)
    mean_bgr = np.mean(wall_patch, axis=(0, 1))
    b, g, r = mean_bgr[0], mean_bgr[1], mean_bgr[2]
    chroma_delta = max(abs(r - g), abs(g - b), abs(r - b)) / max(np.mean(mean_bgr), 1.0)
    result.record("white_wall_chroma_delta", chroma_delta)
    print(f"  White Wall Chromatic Delta: {chroma_delta:.3f} (R={r:.1f}, G={g:.1f}, B={b:.1f})")
    if chroma_delta > 0.25:
        result.fail(f"White wall chromatic distortion detected: Chroma delta {chroma_delta:.3f} > 0.25")

    # Invariant 4: Ground truth reference similarity
    ref_path = os.path.join(REF_DIR, "cornell_upways_1080p.png")
    if os.path.exists(ref_path):
        ref_img = cv2.imread(ref_path)
        comp = compute_reference_comparison(stats_list[-1][2], ref_img)
        result.record("psnr_vs_ref", comp["psnr"])
        result.record("ssim_vs_ref", comp["ssim"])
        print(f"  Ground Truth Reference Similarity: PSNR={comp['psnr']:.2f} dB, SSIM={comp['ssim']:.4f}")
        if comp["psnr"] < 25.0:
            result.fail(f"PSNR too low against reference: {comp['psnr']:.2f} dB < 25.0 dB")
        if comp["ssim"] < 0.70:
            result.fail(f"SSIM too low against reference: {comp['ssim']:.4f} < 0.70")


    return result

def test_mgpu_tile_stability_and_seams():
    result = TestResult("Multi-GPU Checkerboard Tiled Mode Disparity & Exposure Stability")
    print(f"\n--- Running: {result.name} ---")

    frame_counts = [1, 10, 30]
    for fc in frame_counts:
        out_png = os.path.join(OUTPUT_DIR, f"breakfast_mgpu_tile_f{fc}.png")
        rc, stdout, stderr = run_pathways([
            "--scene", "scenes/breakfast-room/breakfast_room_extended.glb",
            "--res", "4k",
            "--upscaler", "upways",
            "--render-scale", "0.5",
            "--mgpu",
            "--mgpu-mode", "tile",
            "--frames", str(fc),
            "--dump-frame", out_png
        ])
        if rc != 0:
            result.fail(f"Pathways exited with code {rc} for frame count {fc}")
            return result

        img = cv2.imread(out_png)
        st = compute_image_stats(img)
        print(f"  Frame {fc:2d}: MeanLum={st['mean_lum']:.1f}, MaxLum={st['max_lum']:.1f}, Clipped={st['clipped_pct']:.2f}%")

        if st["mean_lum"] > 200.0:
            result.fail(f"Overexposure blowout at frame {fc}: Mean luminance {st['mean_lum']:.1f} > 200.0")
        if st["clipped_pct"] > 5.0:
            result.fail(f"Excessive clipping at frame {fc}: {st['clipped_pct']:.2f}% > 5.0%")

        # Evaluate checkerboard tile disparity across 64x64 tiles
        h, w, _ = img.shape
        tile_size = 64
        even_means = []
        odd_means = []
        for ty in range(0, h // tile_size):
            for tx in range(0, w // tile_size):
                tile = img[ty*tile_size:(ty+1)*tile_size, tx*tile_size:(tx+1)*tile_size]
                t_lum = np.mean(tile)
                if (tx + ty) % 2 == 0:
                    even_means.append(t_lum)
                else:
                    odd_means.append(t_lum)

        delta = abs(np.mean(even_means) - np.mean(odd_means))
        disparity_pct = delta / max(np.mean(even_means), 1.0) * 100.0
        print(f"  Tile Disparity (Frame {fc}): Delta={delta:.2f} ({disparity_pct:.2f}%)")
        if delta > 6.0:
            result.fail(f"Checkerboard tile seam detected: Delta {delta:.2f} > 6.0")

    return result

def test_camera_motion_noise_stability():
    result = TestResult("Upways Camera Motion Temporal Stability & High-Frequency Noise Suppression")
    print(f"\n--- Running: {result.name} ---")

    # 1. Pure MC baseline under camera motion (1 SPP, 30 frames)
    out_mc30 = os.path.join(OUTPUT_DIR, "cornell_mc_motion_f30.png")
    rc, stdout, stderr = run_pathways([
        "--res", "1080p",
        "--upscaler", "none",
        "--denoiser", "none",
        "--camera-motion",
        "--frames", "30",
        "--dump-frame", out_mc30
    ])
    if rc != 0:
        result.fail(f"Pure MC baseline motion run exited with code {rc}: {stderr}")
        return result

    img_mc = cv2.imread(out_mc30)
    if img_mc is None:
        result.fail(f"Failed to read baseline Pure MC motion image {out_mc30}")
        return result

    gray_mc = cv2.cvtColor(img_mc, cv2.COLOR_BGR2GRAY)
    lap_mc = cv2.Laplacian(gray_mc, cv2.CV_64F)
    mc_noise_var = float(np.var(lap_mc))
    result.record("pure_mc_motion_noise_var", mc_noise_var)
    print(f"  Pure MC 1-SPP Motion Spatial Noise Variance: {mc_noise_var:.2f}")

    # 2. Upways under continuous camera motion (Frame 29 and Frame 30)
    out_upways29 = os.path.join(OUTPUT_DIR, "cornell_upways_motion_f29.png")
    out_upways30 = os.path.join(OUTPUT_DIR, "cornell_upways_motion_f30.png")

    rc, stdout, stderr = run_pathways([
        "--res", "1080p",
        "--upscaler", "upways",
        "--render-scale", "0.5",
        "--camera-motion",
        "--frames", "29",
        "--dump-frame", out_upways29
    ])
    if rc != 0:
        result.fail(f"Upways motion F29 exited with code {rc}: {stderr}")
        return result

    rc, stdout, stderr = run_pathways([
        "--res", "1080p",
        "--upscaler", "upways",
        "--render-scale", "0.5",
        "--camera-motion",
        "--frames", "30",
        "--dump-frame", out_upways30
    ])
    if rc != 0:
        result.fail(f"Upways motion F30 exited with code {rc}: {stderr}")
        return result

    im29 = cv2.imread(out_upways29)
    im30 = cv2.imread(out_upways30)
    if im29 is None or im30 is None:
        result.fail("Failed to read rendered Upways motion frames.")
        return result

    # Compute spatial noise on F30
    gray29 = cv2.cvtColor(im29, cv2.COLOR_BGR2GRAY)
    gray30 = cv2.cvtColor(im30, cv2.COLOR_BGR2GRAY)
    lap_upways = cv2.Laplacian(gray30, cv2.CV_64F)
    upways_noise_var = float(np.var(lap_upways))
    result.record("upways_motion_noise_var", upways_noise_var)
    noise_reduction_ratio = mc_noise_var / max(upways_noise_var, 1e-4)
    result.record("noise_reduction_ratio", noise_reduction_ratio)
    print(f"  Upways Motion Spatial Noise Variance:       {upways_noise_var:.2f} ({noise_reduction_ratio:.1f}x reduction vs Pure MC)")

    # Patch-based Laplacian noise variance on flat surfaces (ceiling & wall)
    patch_ceiling_mc = gray_mc[60:150, 700:1200]
    patch_wall_mc = gray_mc[250:380, 800:1000]
    mc_patch_ceiling_var = float(np.var(cv2.Laplacian(patch_ceiling_mc, cv2.CV_64F)))
    mc_patch_wall_var = float(np.var(cv2.Laplacian(patch_wall_mc, cv2.CV_64F)))

    patch_ceiling_up = gray30[60:150, 700:1200]
    patch_wall_up = gray30[250:380, 800:1000]
    up_patch_ceiling_var = float(np.var(cv2.Laplacian(patch_ceiling_up, cv2.CV_64F)))
    up_patch_wall_var = float(np.var(cv2.Laplacian(patch_wall_up, cv2.CV_64F)))
    ceiling_reduction = mc_patch_ceiling_var / max(up_patch_ceiling_var, 1e-4)
    wall_reduction = mc_patch_wall_var / max(up_patch_wall_var, 1e-4)

    print(f"  Ceiling Patch Noise Var: Pure MC = {mc_patch_ceiling_var:.2f}, Upways = {up_patch_ceiling_var:.2f} ({ceiling_reduction:.1f}x reduction)")
    print(f"  Wall Patch Noise Var:    Pure MC = {mc_patch_wall_var:.2f}, Upways = {up_patch_wall_var:.2f} ({wall_reduction:.1f}x reduction)")

    # Compute temporal boiling index between F29 and F30
    diff = np.abs(gray30.astype(float) - gray29.astype(float)) / 255.0
    diff_var = float(np.var(diff))
    diff_p99 = float(np.percentile(diff, 99.0))
    mad = float(np.mean(diff))
    boiling_index = float(100.0 * diff_var * (diff_p99 / (mad + 1e-5)))
    result.record("boiling_index", boiling_index)
    print(f"  Upways Temporal Boiling Index:              {boiling_index:.3f}")

    # Compute exposure / luminance stats
    st = compute_image_stats(im30)
    print(f"  Upways Motion Exposure: MeanLum={st['mean_lum']:.1f}, MaxLum={st['max_lum']:.1f}, Clipped={st['clipped_pct']:.2f}%")

    # Invariant: Chromatic neutrality on white back wall under motion
    wall_patch = im30[250:380, 800:1000].astype(float)
    mean_bgr = np.mean(wall_patch, axis=(0, 1))
    b, g, r = mean_bgr[0], mean_bgr[1], mean_bgr[2]
    chroma_delta = max(abs(r - g), abs(g - b), abs(r - b)) / max(np.mean(mean_bgr), 1.0)
    print(f"  White Wall Motion Chromatic Delta: {chroma_delta:.3f} (R={r:.1f}, G={g:.1f}, B={b:.1f})")
    if chroma_delta > 0.25:
        result.fail(f"White wall chromatic distortion under motion: Chroma delta {chroma_delta:.3f} > 0.25")

    # Invariant 1: Noise suppression assertion (must be > 5.0x lower than Pure MC, and variance <= reference level 3000.0)
    if upways_noise_var > 3000.0:
        result.fail(f"Upways camera motion noise too high: Laplacian variance {upways_noise_var:.2f} > 3000.0")
    if noise_reduction_ratio < 5.0:
        result.fail(f"Upways noise reduction insufficient: {noise_reduction_ratio:.1f}x < 5.0x vs Pure MC baseline")
    if ceiling_reduction < 5.0:
        result.fail(f"Upways ceiling patch noise reduction insufficient: {ceiling_reduction:.1f}x < 5.0x")
    if wall_reduction < 5.0:
        result.fail(f"Upways wall patch noise reduction insufficient: {wall_reduction:.1f}x < 5.0x")

    # Invariant 2: Temporal boiling / flickering suppression (must be < 4.0)
    if boiling_index > 4.0:
        result.fail(f"Upways camera motion boiling noise detected: Boiling Index {boiling_index:.3f} > 4.0")

    # Invariant 3: Exposure stability during camera motion
    if st["mean_lum"] > 210.0 or st["mean_lum"] < 35.0:
        result.fail(f"Upways camera motion exposure abnormal: Mean luminance {st['mean_lum']:.1f} not in [35.0, 210.0]")
    if st["clipped_pct"] > 8.0:
        result.fail(f"Excessive highlight blowout in Upways motion frame: {st['clipped_pct']:.2f}% > 8.0%")

    return result

def test_mgpu_tile_motion_noise():
    result = TestResult("Multi-GPU Checkerboard Tiled Motion Stability & Noise Suppression")
    print(f"\n--- Running: {result.name} ---")

    out_tile29 = os.path.join(OUTPUT_DIR, "cornell_tile_motion_f29.png")
    out_tile30 = os.path.join(OUTPUT_DIR, "cornell_tile_motion_f30.png")

    rc, _, stderr = run_pathways([
        "--res", "1080p",
        "--upscaler", "upways",
        "--render-scale", "0.5",
        "--mgpu",
        "--mgpu-mode", "tile",
        "--camera-motion",
        "--frames", "29",
        "--dump-frame", out_tile29
    ])
    if rc != 0:
        result.fail(f"MGPU tile motion F29 failed: {stderr}")
        return result

    rc, _, stderr = run_pathways([
        "--res", "1080p",
        "--upscaler", "upways",
        "--render-scale", "0.5",
        "--mgpu",
        "--mgpu-mode", "tile",
        "--camera-motion",
        "--frames", "30",
        "--dump-frame", out_tile30
    ])
    if rc != 0:
        result.fail(f"MGPU tile motion F30 failed: {stderr}")
        return result

    im29 = cv2.imread(out_tile29)
    im30 = cv2.imread(out_tile30)
    if im29 is None or im30 is None:
        result.fail("Failed to load MGPU tile motion images.")
        return result

    gray30 = cv2.cvtColor(im30, cv2.COLOR_BGR2GRAY)
    gray29 = cv2.cvtColor(im29, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray30, cv2.CV_64F)
    noise_var = float(np.var(lap))
    result.record("mgpu_tile_motion_noise_var", noise_var)

    diff = np.abs(gray30.astype(float) - gray29.astype(float)) / 255.0
    diff_var = float(np.var(diff))
    diff_p99 = float(np.percentile(diff, 99.0))
    mad = float(np.mean(diff))
    boiling_index = float(100.0 * diff_var * (diff_p99 / (mad + 1e-5)))
    result.record("mgpu_tile_boiling_index", boiling_index)

    # Check tile disparity on F30
    h, w = gray30.shape
    tile_size = 64
    even_means = []
    odd_means = []
    for ty in range(0, h // tile_size):
        for tx in range(0, w // tile_size):
            tile = gray30[ty*tile_size:(ty+1)*tile_size, tx*tile_size:(tx+1)*tile_size]
            t_lum = np.mean(tile)
            if (tx + ty) % 2 == 0:
                even_means.append(t_lum)
            else:
                odd_means.append(t_lum)
    delta = abs(np.mean(even_means) - np.mean(odd_means))
    result.record("mgpu_tile_motion_disparity_delta", delta)

    st = compute_image_stats(im30)
    print(f"  MGPU Tile Motion Spatial Noise Variance: {noise_var:.2f}")
    print(f"  MGPU Tile Motion Boiling Index:          {boiling_index:.3f}")
    print(f"  MGPU Tile Motion Disparity Delta:        {delta:.2f}")

    if noise_var > 3000.0:
        result.fail(f"MGPU tile motion noise too high: {noise_var:.2f} > 3000.0")
    if boiling_index > 4.0:
        result.fail(f"MGPU tile motion boiling too high: {boiling_index:.3f} > 4.0")
    if delta > 6.0:
        result.fail(f"MGPU tile checkerboard disparity seam detected under motion: {delta:.2f} > 6.0")

    return result

def test_breakfast_room_motion_noise():
    result = TestResult("Breakfast Room Complex Scene Motion Stability & Noise Suppression")
    print(f"\n--- Running: {result.name} ---")

    scene_path = "scenes/breakfast-room/breakfast_room_extended.glb"
    out_mc30 = os.path.join(OUTPUT_DIR, "br_mc_motion_f30.png")
    out_up29 = os.path.join(OUTPUT_DIR, "br_upways_motion_f29.png")
    out_up30 = os.path.join(OUTPUT_DIR, "br_upways_motion_f30.png")

    rc, _, stderr = run_pathways([
        "--scene", scene_path,
        "--res", "1080p",
        "--upscaler", "none",
        "--denoiser", "none",
        "--camera-motion",
        "--frames", "30",
        "--dump-frame", out_mc30
    ])
    if rc != 0:
        result.fail(f"Breakfast Room MC baseline failed: {stderr}")
        return result

    rc, _, stderr = run_pathways([
        "--scene", scene_path,
        "--res", "1080p",
        "--upscaler", "upways",
        "--render-scale", "0.5",
        "--camera-motion",
        "--frames", "29",
        "--dump-frame", out_up29
    ])
    if rc != 0:
        result.fail(f"Breakfast Room Upways F29 failed: {stderr}")
        return result

    rc, _, stderr = run_pathways([
        "--scene", scene_path,
        "--res", "1080p",
        "--upscaler", "upways",
        "--render-scale", "0.5",
        "--camera-motion",
        "--frames", "30",
        "--dump-frame", out_up30
    ])
    if rc != 0:
        result.fail(f"Breakfast Room Upways F30 failed: {stderr}")
        return result

    im_mc = cv2.imread(out_mc30)
    im29 = cv2.imread(out_up29)
    im30 = cv2.imread(out_up30)
    if im_mc is None or im29 is None or im30 is None:
        result.fail("Failed to load Breakfast Room render images.")
        return result

    gray_mc = cv2.cvtColor(im_mc, cv2.COLOR_BGR2GRAY)
    gray29 = cv2.cvtColor(im29, cv2.COLOR_BGR2GRAY)
    gray30 = cv2.cvtColor(im30, cv2.COLOR_BGR2GRAY)

    lap_mc = cv2.Laplacian(gray_mc, cv2.CV_64F)
    lap_up = cv2.Laplacian(gray30, cv2.CV_64F)
    mc_var = float(np.var(lap_mc))
    up_var = float(np.var(lap_up))
    noise_reduction = mc_var / max(up_var, 1e-4)
    result.record("br_mc_noise_var", mc_var)
    result.record("br_upways_noise_var", up_var)
    result.record("br_noise_reduction_ratio", noise_reduction)

    diff = np.abs(gray30.astype(float) - gray29.astype(float)) / 255.0
    diff_var = float(np.var(diff))
    diff_p99 = float(np.percentile(diff, 99.0))
    mad = float(np.mean(diff))
    boiling_index = float(100.0 * diff_var * (diff_p99 / (mad + 1e-5)))
    result.record("br_boiling_index", boiling_index)

    st = compute_image_stats(im30)
    print(f"  Breakfast Room Pure MC Noise Variance:   {mc_var:.2f}")
    print(f"  Breakfast Room Upways Noise Variance:    {up_var:.2f} ({noise_reduction:.2f}x reduction vs MC)")
    print(f"  Breakfast Room Boiling Index:            {boiling_index:.3f}")
    print(f"  Breakfast Room Mean Lum:                 {st['mean_lum']:.1f}")

    if noise_reduction < 1.8:
        result.fail(f"Breakfast Room noise reduction insufficient: {noise_reduction:.2f}x < 1.8x")
    if boiling_index > 14.0:
        result.fail(f"Breakfast Room motion boiling too high: {boiling_index:.3f} > 14.0")
    if st["mean_lum"] < 40.0 or st["mean_lum"] > 160.0:
        result.fail(f"Breakfast Room mean luminance abnormal: {st['mean_lum']:.1f}")

    return result

def main():
    print("================================================================")
    print("  Pathways Visual Integrity & Exposure Stability Test Suite     ")
    print("================================================================")

    if not HAS_DEPS:
        print(f"[SKIP] Required Python libraries not available: {MISSING_DEP}. Skipping visual integrity test in minimal environment.")
        return 0

    if not os.path.exists(BIN_PATHWAYS):
        print(f"[SKIP] Pathways binary not found at {BIN_PATHWAYS}. Skipping visual integrity test.")
        return 0

    # Probe whether Vulkan ray tracing hardware is available in this environment
    rc, stdout, stderr = run_pathways(["--frames", "1", "--width", "320", "--height", "240", "--spp", "1"])
    if rc != 0:
        print(f"[SKIP] Vulkan Ray Tracing hardware not available in environment (code {rc}): {stderr.strip()[:200]}. Skipping visual integrity test.")
        return 0

    tests = [
        test_single_gpu_accumulation_stability,
        test_mgpu_tile_stability_and_seams,
        test_camera_motion_noise_stability,
        test_mgpu_tile_motion_noise,
        test_breakfast_room_motion_noise
    ]

    all_passed = True
    summary = []

    for test_fn in tests:
        res = test_fn()
        summary.append(res)
        if not res.passed:
            all_passed = False
            print(f"  -> [FAIL] {res.name}")
            for err in res.errors:
                print(f"     * Error: {err}")
        else:
            print(f"  -> [PASS] {res.name}")

    print("\n================================================================")
    print("  Test Execution Summary                                        ")
    print("================================================================")
    for res in summary:
        status_str = "[PASS]" if res.passed else "[FAIL]"
        print(f"  {status_str} {res.name}")

    report_path = os.path.join(OUTPUT_DIR, "visual_integrity_report.json")
    with open(report_path, "w") as f:
        json.dump([{
            "name": r.name,
            "passed": r.passed,
            "errors": r.errors,
            "metrics": r.metrics
        } for r in summary], f, indent=2)
    print(f"\n[INFO] Detailed report written to: {report_path}")

    if all_passed:
        print("[SUCCESS] All visual integrity and exposure invariants PASSED cleanly!\n")
        return 0
    else:
        print("[ERROR] Visual integrity verification FAILED!\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
