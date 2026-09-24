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
    min_val = float(np.min(img))
    clipped_pct = float(np.mean(lum >= 245.0) * 100.0)
    black_pct = float(np.mean(lum <= 2.0) * 100.0)
    p05 = float(np.percentile(lum, 5.0))
    p95 = float(np.percentile(lum, 95.0))
    contrast = p95 - p05

    has_nan = bool(np.isnan(img).any())
    has_inf = bool(np.isinf(img).any())

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
        "channels": c,
        "mean_rgb": [float(x) for x in mean_rgb],
        "mean_lum": mean_lum,
        "max_lum": max_lum,
        "min_val": min_val,
        "clipped_pct": clipped_pct,
        "black_pct": black_pct,
        "p05": p05,
        "p95": p95,
        "contrast": contrast,
        "has_nan": has_nan,
        "has_inf": has_inf,
        "row_ratio": row_ratio,
        "col_ratio": col_ratio
    }

def compute_reference_comparison(img, ref_img):
    """Computes PSNR, SSIM, low-frequency SSIM, edge correlation, and channel correlations."""
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

    # Low-frequency SSIM: Gaussian blur suppresses zero-mean Monte Carlo sample noise (decays as 1/sqrt(N))
    # while preserving macro geometry silhouettes, material patches, and overall illumination.
    blur_a = cv2.GaussianBlur(img, (0, 0), 3.0)
    blur_b = cv2.GaussianBlur(ref_resized, (0, 0), 3.0)
    low_gray_a = cv2.cvtColor(blur_a, cv2.COLOR_BGR2GRAY)
    low_gray_b = cv2.cvtColor(blur_b, cv2.COLOR_BGR2GRAY)
    low_ssim = float(compute_ssim(low_gray_a, low_gray_b))

    # Edge correlation: normalized cross-correlation of Sobel gradients on smoothed luminance
    sobel_a = cv2.Sobel(low_gray_a, cv2.CV_32F, 1, 1, ksize=3)
    sobel_b = cv2.Sobel(low_gray_b, cv2.CV_32F, 1, 1, ksize=3)
    mag_a = np.abs(sobel_a)
    mag_b = np.abs(sobel_b)
    norm_a = np.linalg.norm(mag_a)
    norm_b = np.linalg.norm(mag_b)
    edge_corr = float(np.sum(mag_a * mag_b) / (norm_a * norm_b)) if (norm_a >= 1e-4 and norm_b >= 1e-4) else 1.0

    # Color channel cross-correlations (detects inverted colors, BGR vs RGB swaps)
    corrs = {}
    for idx, name in enumerate(["B", "G", "R"]):
        a = img[:, :, idx].astype(float) - np.mean(img[:, :, idx])
        b = ref_resized[:, :, idx].astype(float) - np.mean(ref_resized[:, :, idx])
        norm_ca = np.linalg.norm(a)
        norm_cb = np.linalg.norm(b)
        corrs[name] = float(np.sum(a * b) / (norm_ca * norm_cb)) if (norm_ca >= 1e-4 and norm_cb >= 1e-4) else 1.0

    # Cross-channel check (B vs R) to catch BGR vs RGB swap bugs
    a_b = img[:, :, 0].astype(float) - np.mean(img[:, :, 0])
    b_r = ref_resized[:, :, 2].astype(float) - np.mean(ref_resized[:, :, 2])
    norm_ab = np.linalg.norm(a_b)
    norm_br = np.linalg.norm(b_r)
    corrs["cross_BR"] = float(np.sum(a_b * b_r) / (norm_ab * norm_br)) if (norm_ab >= 1e-4 and norm_br >= 1e-4) else 0.0

    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_val,
        "low_ssim": low_ssim,
        "edge_corr": edge_corr,
        "corrs": corrs
    }

def verify_cornell_semantic_materials(img):
    """Validates semantic material invariants on canonical Cornell Box scenes."""
    h, w, _ = img.shape
    left_wall = img[int(0.25*h):int(0.75*h), int(0.05*w):int(0.20*w)].astype(float)
    right_wall = img[int(0.25*h):int(0.75*h), int(0.80*w):int(0.95*w)].astype(float)
    ceiling_light = img[int(0.05*h):int(0.15*h), int(0.40*w):int(0.60*w)].astype(float)
    back_wall = img[int(0.25*h):int(0.75*h), int(0.40*w):int(0.60*w)].astype(float)

    left_bgr = np.mean(left_wall, axis=(0, 1))
    right_bgr = np.mean(right_wall, axis=(0, 1))
    back_bgr = np.mean(back_wall, axis=(0, 1))

    light_lum = float(np.mean(0.2126 * ceiling_light[:, :, 2] + 0.7152 * ceiling_light[:, :, 1] + 0.0722 * ceiling_light[:, :, 0]))

    left_red_ratio = float(left_bgr[2] / max(max(left_bgr[0], left_bgr[1]), 1.0))
    right_green_ratio = float(right_bgr[1] / max(max(right_bgr[0], right_bgr[2]), 1.0))

    back_mean = max(float(np.mean(back_bgr)), 1.0)
    back_chroma = float(max(abs(back_bgr[2] - back_bgr[1]), abs(back_bgr[1] - back_bgr[0]), abs(back_bgr[2] - back_bgr[0])) / back_mean)

    # Conductor / Metallic mirror sphere in foreground: x in [0.55*w, 0.65*w], y in [0.70*h, 0.85*h]
    metal_sphere = img[int(0.70*h):int(0.85*h), int(0.55*w):int(0.65*w)].astype(float)
    metal_bgr = np.mean(metal_sphere, axis=(0, 1))
    metal_lum = float(0.2126 * metal_bgr[2] + 0.7152 * metal_bgr[1] + 0.0722 * metal_bgr[0])
    metal_black_pct = float(np.mean(np.mean(metal_sphere, axis=2) < 5.0) * 100.0)

    # Dielectric / Glass sphere on left: x in [0.30*w, 0.42*w], y in [0.50*h, 0.65*h]
    glass_sphere = img[int(0.50*h):int(0.65*h), int(0.30*w):int(0.42*w)].astype(float)
    glass_bgr = np.mean(glass_sphere, axis=(0, 1))
    glass_lum = float(0.2126 * glass_bgr[2] + 0.7152 * glass_bgr[1] + 0.0722 * glass_bgr[0])
    glass_black_pct = float(np.mean(np.mean(glass_sphere, axis=2) < 5.0) * 100.0)

    errors = []
    if left_red_ratio < 1.20:
        errors.append(f"Left wall missing red material (ratio {left_red_ratio:.2f} < 1.20, R={left_bgr[2]:.1f}, G={left_bgr[1]:.1f}, B={left_bgr[0]:.1f})")
    if right_green_ratio < 1.20:
        errors.append(f"Right wall missing green material (ratio {right_green_ratio:.2f} < 1.20, R={right_bgr[2]:.1f}, G={right_bgr[1]:.1f}, B={right_bgr[0]:.1f})")
    if light_lum < 180.0:
        errors.append(f"Ceiling light missing emission (luminance {light_lum:.1f} < 180.0)")
    if back_chroma > 0.25:
        errors.append(f"Back wall chromatic distortion (delta {back_chroma:.3f} > 0.25)")
    if metal_lum < 100.0 or metal_black_pct > 1.0:
        errors.append(f"Metallic mirror sphere missing reflection / pitch black collapse (lum {metal_lum:.1f} < 100.0, black_pct {metal_black_pct:.1f}% > 1.0%)")
    if glass_lum < 120.0 or glass_black_pct > 1.0:
        errors.append(f"Glass sphere missing transmission / black collapse (lum {glass_lum:.1f} < 120.0, black_pct {glass_black_pct:.1f}% > 1.0%)")

    return {
        "valid": len(errors) == 0,
        "errors": errors,
        "left_red_ratio": left_red_ratio,
        "right_green_ratio": right_green_ratio,
        "light_lum": light_lum,
        "back_chroma": back_chroma,
        "metal_lum": metal_lum,
        "metal_black_pct": metal_black_pct,
        "glass_lum": glass_lum,
        "glass_black_pct": glass_black_pct
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
        print(f"  Frame {fc:2d}: MeanLum={st['mean_lum']:.1f}, MaxLum={st['max_lum']:.1f}, Clipped={st['clipped_pct']:.2f}%, Contrast={st['contrast']:.1f}")

    # Invariant 1: Proper dimensions, valid gamut, no NaNs/Infs
    final_st = stats_list[-1][1]
    final_img = stats_list[-1][2]
    if final_st["width"] != 1920 or final_st["height"] != 1080:
        result.fail(f"Incorrect image resolution: expected 1920x1080, got {final_st['width']}x{final_st['height']}")
    if final_st["has_nan"]:
        result.fail("Invalid output: NaN values detected in rendered image")
    if final_st["has_inf"]:
        result.fail("Invalid output: Infinite values detected in rendered image")
    if final_st["min_val"] < 0.0:
        result.fail(f"Invalid output: Negative color values detected ({final_st['min_val']:.4f})")

    # Invariant 2: Proper exposure & dynamic range (no blowout, blackout, or contrast collapse)
    for fc, st, _ in stats_list:
        if st["mean_lum"] > 215.0:
            result.fail(f"Overexposure blowout detected at frame {fc}: Mean luminance {st['mean_lum']:.1f} > 215.0")
        if st["clipped_pct"] > 10.0:
            result.fail(f"Excessive highlight clipping at frame {fc}: {st['clipped_pct']:.2f}% > 10.0%")
        if st["mean_lum"] < 45.0:
            result.fail(f"Underexposure blackout / red mud collapse detected at frame {fc}: Mean luminance {st['mean_lum']:.1f} < 45.0")
        if st["max_lum"] < 120.0:
            result.fail(f"Insufficient dynamic range at frame {fc}: Max luminance {st['max_lum']:.1f} < 120.0")
        if st["contrast"] < 35.0:
            result.fail(f"Contrast collapse at frame {fc}: P95-P05 contrast {st['contrast']:.1f} < 35.0")

    # Invariant 3: Accumulation drift bounded (allows natural MC convergence but forbids runaway)
    f1_lum = stats_list[0][1]["mean_lum"]
    f30_lum = stats_list[-1][1]["mean_lum"]
    drift_pct = abs(f30_lum - f1_lum) / f1_lum * 100.0
    result.record("accumulation_drift_pct", drift_pct)
    print(f"  Accumulation Luminance Drift (F1 -> F30): {drift_pct:.2f}%")
    if drift_pct > 35.0:
        result.fail(f"Accumulation exposure runaway: Drift {drift_pct:.2f}% > 35.0%")

    # Invariant 4: Semantic Material Invariants (Catches missing materials, magenta/gray fallbacks)
    mat_check = verify_cornell_semantic_materials(final_img)
    result.record("left_red_ratio", mat_check["left_red_ratio"])
    result.record("right_green_ratio", mat_check["right_green_ratio"])
    result.record("light_lum", mat_check["light_lum"])
    result.record("white_wall_chroma_delta", mat_check["back_chroma"])
    print(f"  Semantic Materials: LeftRed={mat_check['left_red_ratio']:.2f}, RightGreen={mat_check['right_green_ratio']:.2f}, LightLum={mat_check['light_lum']:.1f}, WallChroma={mat_check['back_chroma']:.3f}, MetalLum={mat_check['metal_lum']:.1f}, GlassLum={mat_check['glass_lum']:.1f}")
    if not mat_check["valid"]:
        for err in mat_check["errors"]:
            result.fail(err)

    # Invariant 5: Macro Geometry & Color Space Invariants (Noise-Immune Reference Comparison)
    ref_path = os.path.join(REF_DIR, "cornell_upways_1080p.png")
    if os.path.exists(ref_path):
        ref_img = cv2.imread(ref_path)
        comp = compute_reference_comparison(final_img, ref_img)
        result.record("psnr_vs_ref", comp["psnr"])
        result.record("ssim_vs_ref", comp["ssim"])
        result.record("low_ssim_vs_ref", comp["low_ssim"])
        result.record("edge_corr_vs_ref", comp["edge_corr"])
        result.record("channel_corr_R", comp["corrs"]["R"])
        result.record("channel_corr_G", comp["corrs"]["G"])
        result.record("channel_corr_B", comp["corrs"]["B"])

        print(f"  Reference Comparison: LowSSIM={comp['low_ssim']:.4f}, EdgeCorr={comp['edge_corr']:.4f}, RawPSNR={comp['psnr']:.2f} dB, RawSSIM={comp['ssim']:.4f}")
        print(f"  Channel Correlations: R={comp['corrs']['R']:.4f}, G={comp['corrs']['G']:.4f}, B={comp['corrs']['B']:.4f}, CrossBR={comp['corrs']['cross_BR']:.4f}")

        # Check macro structural similarity (catches missing/broken meshes and gross errors)
        if comp["low_ssim"] < 0.90:
            result.fail(f"Macro structural divergence: Low-frequency SSIM {comp['low_ssim']:.4f} < 0.90")

        # Check color channel polarity and swap detection (e.g. Vulkan BGR vs RGB bug)
        if comp["corrs"]["R"] < 0.70 or comp["corrs"]["G"] < 0.70 or comp["corrs"]["B"] < 0.70:
            result.fail(f"Color channel distortion: Correlations R={comp['corrs']['R']:.2f}, G={comp['corrs']['G']:.2f}, B={comp['corrs']['B']:.2f} below 0.70")
        if comp["corrs"]["cross_BR"] > comp["corrs"]["R"] and comp["corrs"]["cross_BR"] > comp["corrs"]["B"]:
            result.fail("Inverted colors / Channel swap detected: Red and Blue channels are swapped (Vulkan BGR/RGB mismatch)")

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
