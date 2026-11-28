#!/usr/bin/env python3
"""Pathways P0 Continuous-Motion Ground-Truth Quality Gate.

Eliminates static-only verification blind spots by evaluating live mid-motion
frames (Cornell Box pan+dolly at checkpoint k=30, t=29/60=0.48333333s) against
an unbiased 4-bounce high-SPP static ground truth at the exact same camera pose.

Checks enforced:
  1. Unbiased 4-bounce GT integrity:
       dynamic_bounces == max_bounces (4), 0 validation errors.
  2. Default static MC parity (no --reference):
       dynamic_bounces == max_bounces (4) and brightness within +/-2% of GT
       (permanently catches tail-megakernel dynamic_bounces 4 -> 2 truncation).
  3. Continuous-motion frame k=30 across core modes:
     - default_1spp:     raw PT variance ceilings (Zero%, FF5x%), dynamic_bounces == 4
     - restir_nrc_1spp:  |GlassBias_ub| < 0.04 and MirZero% < 0.5%
     - upways_auto_1spp: Auto selects V5 ("5.0"), PSNR_ub >= 25.0 dB,
                         HDR relMSE <= 0.05 (when EXR readable),
                         white_wall LocStd <= 0.003, MirZero% < 0.5%, 0 VVL errors
     - upways5_1spp:     Explicit V5 ("5.0"), PSNR_ub >= 25.0 dB,
                         HDR relMSE <= 0.05 (when EXR readable),
                         white_wall LocStd <= 0.003, MirZero% < 0.5%, 0 VVL errors
     - reblur_1spp:      Analytical ReBLUR ("ReBLUR"), PSNR_ub >= 27.5 dB,
                         HDR relMSE <= 0.035 (when EXR readable),
                         white_wall LocStd <= 0.002, mirror_sphere LocStd <= 0.0190,
                         MirZero% < 0.5%, 0 VVL errors

Usage:
  python3 scripts/motion_quality_gate.py [--binary PATH] [--gpu 1]
      [--out-dir DIR] [--ref-frames 1024] [--cache-ref] [--quick]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAMERA_PATH = os.path.join(REPO_ROOT, "tests", "data", "cornell_motion_path.json")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "output", "motion_quality_gate")

BASE_WIDTH = 1280
BASE_HEIGHT = 720
FPS = 60.0
DEFAULT_CHECKPOINT_K = 30
EXPECTED_MAX_BOUNCES = 4
MIN_MEM_AVAILABLE_GIB = 16.0

# Verified ROIs on 1280x720 k=30 Cornell Box close-up framing: (x0, y0, x1, y1)
BASE_ROIS = {
    "white_wall":     (370,  30,  630, 230),
    "red_wall":       (40,   80,  260, 480),
    "blue_box":       (700,  40,  935, 350),
    "green_wall":     (1030, 40, 1250, 260),
    "glass_sphere":   (385, 315,  610, 540),
    "mirror_sphere":  (615, 420,  900, 670),
    "caustic_shadow": (290, 535,  560, 625),
    "floor_bright":   (120, 635,  380, 710),
}

# Authoritative P0 gate thresholds
THRESHOLDS = {
    "static_mc_luma_parity_tol": 0.02,      # |luma_ratio - 1.0| <= 0.02 (within 2% of unbiased GT)
    "default_1spp_max_zero_pct": 8.0,       # catches raw PT variance regression (was 11.35%)
    "default_1spp_max_ff5x_pct": 1.5,       # catches bounce-1 shadow RR firefly regression (was 2.85%)
    "restir_nrc_max_abs_glass_bias": 0.04,  # catches milky opaque glass (+0.082 ub / +0.214 mc)
    "restir_nrc_max_mirror_zero_pct": 0.5,  # catches mirror black crescent (was 0.91%)
    "upways_min_psnr_ub": 25.0,             # V5 >= 25.0 dB vs unbiased 4-bounce GT (V4 was 24.38 dB)
    "upways_max_hdr_relmse": 0.05,          # V5 <= 0.05 HDR relMSE (V4 was 0.064)
    "upways_max_white_wall_loc_std": 0.003, # V5 <= 0.003 (V4 was 0.0062, V5 ~0.0011)
    "upways_max_mirror_zero_pct": 0.5,      # no black crescent on mirror
    "upways_expected_version": "5.0",       # Auto must select V5, never dead-head V4
    "reblur_min_psnr_ub": 27.5,             # ReBLUR >= 27.5 dB vs unbiased 4-bounce GT (tighten after GPU 1 run)
    "reblur_max_hdr_relmse": 0.035,         # ReBLUR <= 0.035 HDR relMSE
    "reblur_max_white_wall_loc_std": 0.002, # ReBLUR <= 0.002 white wall LocStd
    "reblur_max_mirror_loc_std": 0.0190,    # ReBLUR <= 0.0190 mirror sphere LocStd (GT=0.0307, Upways V5=0.0207)
    "reblur_max_mirror_zero_pct": 0.5,      # no black crescent on mirror
}


def detect_default_binary():
    env_bin = os.environ.get("PATHWAYS_BIN")
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    candidates = [
        os.path.join(REPO_ROOT, "build", "linux-release", "bin", "pathways"),
        os.path.join(REPO_ROOT, "build", "bin", "pathways"),
        os.path.join(REPO_ROOT, "build", "windows-clang-release", "bin", "pathways.exe"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return candidates[0]


def check_mem_available_gib():
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        return None
    return None


def scale_rois(width, height):
    if width == BASE_WIDTH and height == BASE_HEIGHT:
        return dict(BASE_ROIS)
    sx = width / float(BASE_WIDTH)
    sy = height / float(BASE_HEIGHT)
    scaled = {}
    for name, (x0, y0, x1, y1) in BASE_ROIS.items():
        scaled[name] = (
            max(0, min(width - 1, int(round(x0 * sx)))),
            max(0, min(height - 1, int(round(y0 * sy)))),
            max(1, min(width, int(round(x1 * sx)))),
            max(1, min(height, int(round(y1 * sy)))),
        )
    return scaled


def load_png_rgb(path):
    """Load 8-bit or 16-bit PNG as float32 [0, 1] RGB array."""
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    elif arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 65535.0
    if arr.dtype == np.uint8:
        return arr.astype(np.float32) / 255.0
    return np.clip(arr.astype(np.float32), 0.0, 1.0)


def load_exr_luma(path):
    """Load linear HDR EXR luminance via OpenImageIO if installed (None otherwise)."""
    if not path or not os.path.isfile(path):
        return None, "missing"
    try:
        import OpenImageIO as oiio  # type: ignore
    except ImportError:
        return None, "oiio_unavailable"
    try:
        buf = oiio.ImageBuf(path)
        if buf.has_error:
            return None, f"oiio_error: {buf.geterror()}"
        img = np.asarray(buf.get_pixels(oiio.FLOAT))
        rgb = np.nan_to_num(img[..., :3].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
        luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)
        return luma, "ok"
    except Exception as exc:  # noqa: BLE001
        return None, f"oiio_exception: {exc}"


def to_luma(rgb):
    return rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722


def fast_box_mean(img2d, k=5):
    """Uniform kxk box mean using an integral image (no SciPy, low RAM)."""
    pad = k // 2
    padded = np.pad(img2d.astype(np.float64), pad, mode="reflect")
    cs = np.pad(padded, ((1, 0), (1, 0)), mode="constant").cumsum(axis=0).cumsum(axis=1)
    h, w = img2d.shape
    total = cs[k : k + h, k : k + w] - cs[:h, k : k + w] - cs[k : k + h, :w] + cs[:h, :w]
    return (total / float(k * k)).astype(np.float32)


def local_5x5_median(img2d):
    """Exact 5x5 local median via shifted slices (~92 MB peak at 720p, freed immediately)."""
    padded = np.pad(img2d, 2, mode="reflect")
    h, w = img2d.shape
    shifts = [padded[dy : dy + h, dx : dx + w] for dy in range(5) for dx in range(5)]
    stack = np.stack(shifts, axis=-1)
    med = np.median(stack, axis=-1).astype(np.float32)
    del stack, shifts, padded
    return med


def compute_ssim_rgb(a, b):
    """8x8 block SSIM averaged over RGB channels."""
    c1_const = 0.01 ** 2
    c2_const = 0.03 ** 2
    h, w, _ = a.shape
    nh, nw = h // 8, w // 8
    vals = []
    for ch in range(3):
        b1 = a[: nh * 8, : nw * 8, ch].reshape(nh, 8, nw, 8).swapaxes(1, 2)
        b2 = b[: nh * 8, : nw * 8, ch].reshape(nh, 8, nw, 8).swapaxes(1, 2)
        mu1 = b1.mean(axis=(2, 3))
        mu2 = b2.mean(axis=(2, 3))
        v1 = ((b1 - mu1[:, :, None, None]) ** 2).mean(axis=(2, 3))
        v2 = ((b2 - mu2[:, :, None, None]) ** 2).mean(axis=(2, 3))
        cov = ((b1 - mu1[:, :, None, None]) * (b2 - mu2[:, :, None, None])).mean(axis=(2, 3))
        s = ((2.0 * mu1 * mu2 + c1_const) * (2.0 * cov + c2_const)) / (
            (mu1 ** 2 + mu2 ** 2 + c1_const) * (v1 + v2 + c2_const) + 1e-8
        )
        vals.append(float(s.mean()))
    return float(sum(vals) / 3.0)


def region_metrics(rgb, ref_rgb, luma, ref_luma, loc_std_map, med5_map, bbox=None):
    if bbox is not None:
        x0, y0, x1, y1 = bbox
        r = rgb[y0:y1, x0:x1]
        rr = ref_rgb[y0:y1, x0:x1]
        l = luma[y0:y1, x0:x1]
        rl = ref_luma[y0:y1, x0:x1]
        ls = loc_std_map[y0:y1, x0:x1]
        m5 = med5_map[y0:y1, x0:x1]
    else:
        r, rr, l, rl, ls, m5 = rgb, ref_rgb, luma, ref_luma, loc_std_map, med5_map

    diff = r - rr
    mse = float(np.mean(diff * diff))
    rmse = math.sqrt(mse)
    psnr = 10.0 * math.log10(1.0 / mse) if mse > 1e-12 else 100.0
    mae = float(np.mean(np.abs(diff)))

    lap = 4.0 * l[1:-1, 1:-1] - l[:-2, 1:-1] - l[2:, 1:-1] - l[1:-1, :-2] - l[1:-1, 2:]
    lap_sigma = float(np.std(lap) / math.sqrt(20.0))

    zero_frac = float(np.mean(l < 0.02))
    firefly_5x = float(np.mean(l > np.maximum(5.0 * m5, 0.15)))

    mean_l = float(np.mean(l))
    mean_rl = float(np.mean(rl))
    luma_bias = mean_l - mean_rl
    luma_ratio = mean_l / max(mean_rl, 1e-6)

    return {
        "rmse": round(rmse, 5),
        "psnr": round(psnr, 2),
        "mae": round(mae, 5),
        "lap_sigma": round(lap_sigma, 5),
        "local_std": round(float(np.mean(ls)), 5),
        "zero_pct": round(zero_frac * 100.0, 3),
        "firefly_5x_pct": round(firefly_5x * 100.0, 3),
        "mean_luma": round(mean_l, 5),
        "luma_bias": round(luma_bias, 5),
        "luma_ratio": round(luma_ratio, 4),
    }


def parse_stats(stats_path):
    if not stats_path or not os.path.isfile(stats_path):
        return None
    with open(stats_path, "r", encoding="utf-8") as f:
        st = json.load(f)
    perf = st.get("performance", {})
    gpu_bd = perf.get("gpu_profiler_breakdown_ms", {})
    es = st.get("engine_settings", {})
    return {
        "resolution": es.get("resolution", []),
        "max_bounces": int(es.get("max_bounces", 0)),
        "dynamic_bounces": int(es.get("dynamic_bounces", 0)),
        "denoiser_mode": str(es.get("denoiser_mode", "")),
        "upscaler_mode": str(es.get("upscaler_mode", "")),
        "upways_version": str(es.get("upways_version", "")),
        "upways_precision": str(es.get("upways_precision", "")),
        "enable_nrc": bool(es.get("enable_nrc", False)),
        "spp": int(es.get("spp", 0)),
        "total_frames": int(perf.get("total_frames", 0)),
        "accum_samples": int(perf.get("total_accumulated_samples", 0)),
        "validation_errors": int(perf.get("validation_errors", -1)),
        "avg_frame_ms": round(float(perf.get("avg_frame_time_ms", 0.0)), 3),
        "gpu_rt_ms": round(float(gpu_bd.get("primary_gpu_time_ms", 0.0)), 3),
        "denoise_sr_ms": round(float(gpu_bd.get("denoiser_and_upscale_time_ms", 0.0)), 3),
        "tonemap_ms": round(float(gpu_bd.get("tonemap_and_merge_time_ms", 0.0)), 3),
    }


def analyze_render(png_path, ref_rgb, ref_luma, rois, exr_path=None, ref_exr_luma=None, stats_path=None):
    rgb = load_png_rgb(png_path)
    luma = to_luma(rgb)

    mean5 = fast_box_mean(luma, 5)
    mean5_sq = fast_box_mean(luma * luma, 5)
    loc_std_map = np.sqrt(np.maximum(mean5_sq - mean5 * mean5, 0.0))
    med5_map = local_5x5_median(luma)

    full = region_metrics(rgb, ref_rgb, luma, ref_luma, loc_std_map, med5_map, bbox=None)
    full["ssim"] = round(compute_ssim_rgb(rgb, ref_rgb), 4)

    rois_out = {}
    for rname, bbox in rois.items():
        rois_out[rname] = region_metrics(rgb, ref_rgb, luma, ref_luma, loc_std_map, med5_map, bbox=bbox)

    hdr = None
    if exr_path and ref_exr_luma is not None:
        xl, exr_status = load_exr_luma(exr_path)
        if xl is not None and xl.shape == ref_exr_luma.shape:
            err = (xl - ref_exr_luma) ** 2 / (ref_exr_luma ** 2 + 1e-2)
            trimmed = np.sort(err.ravel())[: int(err.size * 0.999)]
            hdr = {
                "relmse": round(float(err.mean()), 4),
                "relmse_trim": round(float(trimmed.mean()), 4),
                "hdr_lum_ratio": round(float(xl.mean() / max(ref_exr_luma.mean(), 1e-9)), 4),
                "status": exr_status,
            }
        else:
            hdr = {"status": exr_status}

    stats = parse_stats(stats_path)
    return {"full": full, "rois": rois_out, "hdr": hdr, "stats": stats}


def run_render(cmd, tag, log_path, png_path, stats_path):
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        lf.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT)
    wall_s = time.time() - t0
    ok = (proc.returncode == 0) and os.path.isfile(png_path) and os.path.isfile(stats_path)
    status = "OK" if ok else f"FAIL(rc={proc.returncode})"
    print(f"  [{status:12s}] {tag:24s} ({wall_s:.2f}s)", flush=True)
    return ok, proc.returncode, round(wall_s, 3)


def can_reuse_cached_ref(png_p, exr_p, stats_p, expected_frames, width, height):
    if not (os.path.isfile(png_p) and os.path.isfile(stats_p)):
        return False
    st = parse_stats(stats_p)
    if st is None:
        return False
    if st["total_frames"] != expected_frames:
        return False
    if st["resolution"] != [width, height]:
        return False
    if st["max_bounces"] != EXPECTED_MAX_BOUNCES or st["dynamic_bounces"] != EXPECTED_MAX_BOUNCES:
        return False
    # If EXR is missing, only require it when OpenImageIO is actually available
    try:
        import OpenImageIO  # noqa: F401
        if not os.path.isfile(exr_p):
            return False
    except ImportError:
        pass
    return True


def record_check(checks, mode, name, passed, detail):
    checks.append({
        "mode": mode,
        "check": name,
        "passed": bool(passed),
        "detail": detail,
    })
    return bool(passed)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--binary", default=detect_default_binary(), help="Path to pathways binary")
    ap.add_argument("--gpu", type=int, default=1, help="Vulkan GPU index (default: 1, non-display GPU)")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output directory for frames, logs, and report")
    ap.add_argument("--camera-path", default=DEFAULT_CAMERA_PATH, help="Deterministic camera path JSON")
    ap.add_argument("--ref-frames", type=int, default=None, help="Unbiased GT frame count (default: 1024, or 256 with --quick)")
    ap.add_argument("--static-check-frames", type=int, default=None, help="Default static MC parity frame count (default: 256, or 128 with --quick)")
    ap.add_argument("--cache-ref", action="store_true", help="Reuse cached unbiased GT if matching ref-frames & resolution exist")
    ap.add_argument("--quick", action="store_true", help="Faster reference sample counts (256 GT / 128 static check); runs all motion modes and gates")
    ap.add_argument("--no-validation", action="store_true", help="Pass --no-validation on motion runs too (default: validation ON for 30-frame motion runs)")
    ap.add_argument("--width", type=int, default=BASE_WIDTH)
    ap.add_argument("--height", type=int, default=BASE_HEIGHT)
    ap.add_argument("--checkpoint-k", type=int, default=DEFAULT_CHECKPOINT_K, help="Motion checkpoint frame number (default: 30)")
    args = ap.parse_args()

    ref_frames = args.ref_frames if args.ref_frames is not None else (256 if args.quick else 1024)
    static_frames = args.static_check_frames if args.static_check_frames is not None else (128 if args.quick else 256)
    t_checkpoint = (args.checkpoint_k - 1) / FPS

    if not os.path.isfile(args.binary):
        print(f"[ERROR] Pathways binary not found: {args.binary}", file=sys.stderr)
        return 2
    if not os.path.isfile(args.camera_path):
        print(f"[ERROR] Camera path JSON not found: {args.camera_path}", file=sys.stderr)
        return 2

    mem_gib = check_mem_available_gib()
    if mem_gib is not None and mem_gib < MIN_MEM_AVAILABLE_GIB:
        print(f"[ERROR] MemAvailable {mem_gib:.1f} GiB < {MIN_MEM_AVAILABLE_GIB:.1f} GiB safety floor.", file=sys.stderr)
        return 2

    os.makedirs(args.out_dir, exist_ok=True)
    rois = scale_rois(args.width, args.height)

    print("==============================================================================")
    print("  Pathways P0 Continuous-Motion Ground-Truth Quality Gate")
    print("==============================================================================")
    print(f"  Binary      : {args.binary}")
    print(f"  GPU         : {args.gpu}")
    print(f"  Resolution  : {args.width}x{args.height}")
    print(f"  Camera Path : {args.camera_path}")
    print(f"  Checkpoint  : k={args.checkpoint_k} (t={t_checkpoint:.8f}s)")
    print(f"  Ref Frames  : unbiased={ref_frames}, static_mc={static_frames} (cache_ref={args.cache_ref})")
    if mem_gib is not None:
        print(f"  MemAvailable: {mem_gib:.1f} GiB")
    print("", flush=True)

    base_common = [
        args.binary,
        "--headless",
        "--gpu", str(args.gpu),
        "--width", str(args.width),
        "--height", str(args.height),
        "--max-bounces", str(EXPECTED_MAX_BOUNCES),
        "--camera-path", args.camera_path,
    ]

    checks = []
    report = {
        "config": {
            "binary": args.binary,
            "gpu": args.gpu,
            "width": args.width,
            "height": args.height,
            "camera_path": args.camera_path,
            "checkpoint_k": args.checkpoint_k,
            "t_checkpoint": round(t_checkpoint, 8),
            "ref_frames": ref_frames,
            "static_check_frames": static_frames,
            "quick": args.quick,
            "validation_on_motion": not args.no_validation,
        },
        "thresholds": THRESHOLDS,
        "references": {},
        "modes": {},
        "checks": checks,
        "passed": False,
    }

    # ------------------------------------------------------------------
    # 1. Unbiased 4-Bounce Ground Truth at k=30 (t = 29/60 = 0.48333333s)
    # ------------------------------------------------------------------
    print("[1/3] Ground-Truth & Static Bounce-Integrity References...", flush=True)
    gt_tag = f"ref_unbiased{ref_frames}_k{args.checkpoint_k}"
    gt_png = os.path.join(args.out_dir, f"{gt_tag}.png")
    gt_exr = os.path.join(args.out_dir, f"{gt_tag}.exr")
    gt_stats = os.path.join(args.out_dir, f"{gt_tag}.stats.json")
    gt_log = os.path.join(args.out_dir, f"{gt_tag}.log")

    if args.cache_ref and can_reuse_cached_ref(gt_png, gt_exr, gt_stats, ref_frames, args.width, args.height):
        print(f"  [CACHED      ] {gt_tag:24s} (reusing {gt_png})", flush=True)
        gt_ok, gt_rc, gt_wall = True, 0, 0.0
    else:
        gt_cmd = [
            *base_common,
            "--no-validation",
            "--spp", "1",
            "--reference",
            "--accum-format", "fp32",
            "--accum-cutoff", "0",
            "--camera-path-time", f"{t_checkpoint:.8f}",
            "--frames", str(ref_frames),
            "--dump-frame", gt_png,
            "--dump-hdr", gt_exr,
            "--dump-stats", gt_stats,
        ]
        gt_ok, gt_rc, gt_wall = run_render(gt_cmd, gt_tag, gt_log, gt_png, gt_stats)

    record_check(checks, "ref_unbiased", "render_ok", gt_ok, f"rc={gt_rc}, wall={gt_wall}s")
    if not gt_ok:
        report_path = os.path.join(args.out_dir, "motion_gate_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\n[FAIL] Unbiased GT render failed; see {gt_log}", file=sys.stderr)
        return 1

    ref_rgb = load_png_rgb(gt_png)
    ref_luma = to_luma(ref_rgb)
    ref_exr_luma, ref_exr_status = load_exr_luma(gt_exr)
    if ref_exr_luma is None:
        print(f"  [NOTE] HDR EXR loading skipped ({ref_exr_status}); falling back to PNG metrics only.", flush=True)

    gt_st = parse_stats(gt_stats)
    record_check(
        checks, "ref_unbiased", "dynamic_bounces_eq_max",
        gt_st is not None and gt_st["dynamic_bounces"] == gt_st["max_bounces"] == EXPECTED_MAX_BOUNCES,
        f"dynamic_bounces={gt_st['dynamic_bounces'] if gt_st else None}, max_bounces={gt_st['max_bounces'] if gt_st else None}",
    )
    record_check(
        checks, "ref_unbiased", "validation_errors_zero",
        gt_st is not None and gt_st["validation_errors"] == 0,
        f"validation_errors={gt_st['validation_errors'] if gt_st else None}",
    )
    gt_mean_luma = float(np.mean(ref_luma))
    record_check(
        checks, "ref_unbiased", "non_black_gt",
        gt_mean_luma > 0.05,
        f"mean_luma={gt_mean_luma:.4f} (expected > 0.05)",
    )
    report["references"]["unbiased_gt"] = {
        "tag": gt_tag,
        "wall_s": gt_wall,
        "mean_luma": round(gt_mean_luma, 5),
        "exr_status": ref_exr_status,
        "stats": gt_st,
    }

    # ------------------------------------------------------------------
    # 2. Default Static MC Parity Check (catches dynamic_bounces 4 -> 2)
    # ------------------------------------------------------------------
    st_tag = f"ref_mc{static_frames}_k{args.checkpoint_k}"
    st_png = os.path.join(args.out_dir, f"{st_tag}.png")
    st_exr = os.path.join(args.out_dir, f"{st_tag}.exr")
    st_stats = os.path.join(args.out_dir, f"{st_tag}.stats.json")
    st_log = os.path.join(args.out_dir, f"{st_tag}.log")

    st_cmd = [
        *base_common,
        "--no-validation",
        "--spp", "1",
        "--accum-cutoff", "0",
        "--camera-path-time", f"{t_checkpoint:.8f}",
        "--frames", str(static_frames),
        "--dump-frame", st_png,
        "--dump-hdr", st_exr,
        "--dump-stats", st_stats,
    ]
    st_ok, st_rc, st_wall = run_render(st_cmd, st_tag, st_log, st_png, st_stats)
    record_check(checks, "ref_mc_static", "render_ok", st_ok, f"rc={st_rc}, wall={st_wall}s")
    if st_ok:
        st_analysis = analyze_render(st_png, ref_rgb, ref_luma, rois, st_exr, ref_exr_luma, st_stats)
        st_analysis["wall_s"] = st_wall
        report["references"]["default_static_mc"] = st_analysis
        s_st = st_analysis["stats"]
        luma_ratio = st_analysis["full"]["luma_ratio"]
        tol = THRESHOLDS["static_mc_luma_parity_tol"]
        record_check(
            checks, "ref_mc_static", "dynamic_bounces_eq_max",
            s_st is not None and s_st["dynamic_bounces"] == s_st["max_bounces"] == EXPECTED_MAX_BOUNCES,
            f"dynamic_bounces={s_st['dynamic_bounces'] if s_st else None}, max_bounces={s_st['max_bounces'] if s_st else None}",
        )
        record_check(
            checks, "ref_mc_static", "brightness_parity_within_2pct",
            abs(luma_ratio - 1.0) <= tol,
            f"luma_ratio={luma_ratio:.4f} (allowed [{1.0 - tol:.2f}, {1.0 + tol:.2f}])",
        )
        record_check(
            checks, "ref_mc_static", "validation_errors_zero",
            s_st is not None and s_st["validation_errors"] == 0,
            f"validation_errors={s_st['validation_errors'] if s_st else None}",
        )

    # ------------------------------------------------------------------
    # 3. Continuous-Motion Modes at Frame k=30
    # ------------------------------------------------------------------
    print("\n[2/3] Rendering Continuous-Motion Frame k={} Core Modes...".format(args.checkpoint_k), flush=True)
    motion_modes = [
        ("default_1spp",     ["--spp", "1"]),
        ("restir_nrc_1spp",  ["--spp", "1", "--restir", "--nrc"]),
        ("upways_auto_1spp", ["--spp", "1", "--denoiser", "upways"]),
        ("upways5_1spp",     ["--spp", "1", "--denoiser", "upways", "--upways-version", "5"]),
        ("reblur_1spp",      ["--spp", "1", "--denoiser", "reblur"]),
    ]

    for mode_name, extra_args in motion_modes:
        tag = f"{mode_name}_motion_k{args.checkpoint_k}"
        png_p = os.path.join(args.out_dir, f"{tag}.png")
        exr_p = os.path.join(args.out_dir, f"{tag}.exr")
        stats_p = os.path.join(args.out_dir, f"{tag}.stats.json")
        log_p = os.path.join(args.out_dir, f"{tag}.log")

        cmd = [*base_common]
        if args.no_validation:
            cmd.append("--no-validation")
        cmd += [
            *extra_args,
            "--frames", str(args.checkpoint_k),
            "--dump-frame", png_p,
            "--dump-hdr", exr_p,
            "--dump-stats", stats_p,
        ]
        ok, rc, wall_s = run_render(cmd, tag, log_p, png_p, stats_p)
        record_check(checks, mode_name, "render_ok", ok, f"rc={rc}, wall={wall_s}s")
        if not ok:
            report["modes"][mode_name] = {"render_ok": False, "rc": rc, "log": log_p}
            continue

        analysis = analyze_render(png_p, ref_rgb, ref_luma, rois, exr_p, ref_exr_luma, stats_p)
        analysis["wall_s"] = wall_s
        report["modes"][mode_name] = analysis

        full = analysis["full"]
        r_rois = analysis["rois"]
        hdr = analysis["hdr"]
        st = analysis["stats"]

        record_check(
            checks, mode_name, "validation_errors_zero",
            st is not None and st["validation_errors"] == 0,
            f"validation_errors={st['validation_errors'] if st else None}",
        )

        if mode_name == "default_1spp":
            record_check(
                checks, mode_name, "dynamic_bounces_eq_max",
                st is not None and st["dynamic_bounces"] == st["max_bounces"] == EXPECTED_MAX_BOUNCES,
                f"dynamic_bounces={st['dynamic_bounces'] if st else None}, max_bounces={st['max_bounces'] if st else None}",
            )
            record_check(
                checks, mode_name, "zero_pct_ceiling",
                full["zero_pct"] <= THRESHOLDS["default_1spp_max_zero_pct"],
                f"Zero%={full['zero_pct']:.3f}% <= {THRESHOLDS['default_1spp_max_zero_pct']:.1f}%",
            )
            record_check(
                checks, mode_name, "ff5x_pct_ceiling",
                full["firefly_5x_pct"] <= THRESHOLDS["default_1spp_max_ff5x_pct"],
                f"FF5x%={full['firefly_5x_pct']:.3f}% <= {THRESHOLDS['default_1spp_max_ff5x_pct']:.1f}%",
            )

        elif mode_name == "restir_nrc_1spp":
            glass_bias = r_rois["glass_sphere"]["luma_bias"]
            mir_zero = r_rois["mirror_sphere"]["zero_pct"]
            record_check(
                checks, mode_name, "glass_bias_ub_bound",
                abs(glass_bias) < THRESHOLDS["restir_nrc_max_abs_glass_bias"],
                f"|GlassBias_ub|={abs(glass_bias):.4f} (signed {glass_bias:+.4f}) < {THRESHOLDS['restir_nrc_max_abs_glass_bias']:.2f}",
            )
            record_check(
                checks, mode_name, "mirror_zero_pct_bound",
                mir_zero < THRESHOLDS["restir_nrc_max_mirror_zero_pct"],
                f"MirZero%={mir_zero:.3f}% < {THRESHOLDS['restir_nrc_max_mirror_zero_pct']:.2f}%",
            )

        elif mode_name in ("upways_auto_1spp", "upways5_1spp"):
            expected_ver = THRESHOLDS["upways_expected_version"]
            record_check(
                checks, mode_name, "denoiser_is_upways",
                st is not None and st["denoiser_mode"] == "Upways",
                f"denoiser_mode={st['denoiser_mode'] if st else None}",
            )
            record_check(
                checks, mode_name, "upways_version_is_v5",
                st is not None and st["upways_version"] == expected_ver,
                f"upways_version={st['upways_version'] if st else None} (expected '{expected_ver}')",
            )
            record_check(
                checks, mode_name, "psnr_ub_floor",
                full["psnr"] >= THRESHOLDS["upways_min_psnr_ub"],
                f"PSNR_ub={full['psnr']:.2f} dB >= {THRESHOLDS['upways_min_psnr_ub']:.1f} dB",
            )
            if hdr is not None and "relmse" in hdr:
                record_check(
                    checks, mode_name, "hdr_relmse_ceiling",
                    hdr["relmse"] <= THRESHOLDS["upways_max_hdr_relmse"],
                    f"HDR relMSE={hdr['relmse']:.4f} <= {THRESHOLDS['upways_max_hdr_relmse']:.3f}",
                )
            ww_loc_std = r_rois["white_wall"]["local_std"]
            record_check(
                checks, mode_name, "white_wall_loc_std_ceiling",
                ww_loc_std <= THRESHOLDS["upways_max_white_wall_loc_std"],
                f"white_wall LocStd={ww_loc_std:.5f} <= {THRESHOLDS['upways_max_white_wall_loc_std']:.4f}",
            )
            mir_zero = r_rois["mirror_sphere"]["zero_pct"]
            record_check(
                checks, mode_name, "mirror_zero_pct_bound",
                mir_zero < THRESHOLDS["upways_max_mirror_zero_pct"],
                f"MirZero%={mir_zero:.3f}% < {THRESHOLDS['upways_max_mirror_zero_pct']:.2f}%",
            )

        elif mode_name == "reblur_1spp":
            record_check(
                checks, mode_name, "denoiser_is_reblur",
                st is not None and st["denoiser_mode"] == "ReBLUR",
                f"denoiser_mode={st['denoiser_mode'] if st else None}",
            )
            record_check(
                checks, mode_name, "psnr_ub_floor",
                full["psnr"] >= THRESHOLDS["reblur_min_psnr_ub"],
                f"PSNR_ub={full['psnr']:.2f} dB >= {THRESHOLDS['reblur_min_psnr_ub']:.1f} dB",
            )
            if hdr is not None and "relmse" in hdr:
                record_check(
                    checks, mode_name, "hdr_relmse_ceiling",
                    hdr["relmse"] <= THRESHOLDS["reblur_max_hdr_relmse"],
                    f"HDR relMSE={hdr['relmse']:.4f} <= {THRESHOLDS['reblur_max_hdr_relmse']:.3f}",
                )
            ww_loc_std = r_rois["white_wall"]["local_std"]
            record_check(
                checks, mode_name, "white_wall_loc_std_ceiling",
                ww_loc_std <= THRESHOLDS["reblur_max_white_wall_loc_std"],
                f"white_wall LocStd={ww_loc_std:.5f} <= {THRESHOLDS['reblur_max_white_wall_loc_std']:.4f}",
            )
            mir_loc_std = r_rois["mirror_sphere"]["local_std"]
            record_check(
                checks, mode_name, "mirror_sphere_loc_std_ceiling",
                mir_loc_std <= THRESHOLDS["reblur_max_mirror_loc_std"],
                f"mirror_sphere LocStd={mir_loc_std:.5f} <= {THRESHOLDS['reblur_max_mirror_loc_std']:.4f}",
            )
            mir_zero = r_rois["mirror_sphere"]["zero_pct"]
            record_check(
                checks, mode_name, "mirror_zero_pct_bound",
                mir_zero < THRESHOLDS["reblur_max_mirror_zero_pct"],
                f"MirZero%={mir_zero:.3f}% < {THRESHOLDS['reblur_max_mirror_zero_pct']:.2f}%",
            )

    # ------------------------------------------------------------------
    # 4. Print Crisp Summary Tables & Write JSON Report
    # ------------------------------------------------------------------
    print("\n[3/3] P0 Motion Quality Gate Summary (vs Unbiased 4-Bounce GT @ k={})".format(args.checkpoint_k))
    hdr_line = (
        f"{'Mode':18s} | {'Ver':>4s} {'DynB':>4s} | {'PSNR_ub':>7s} {'RMSE_ub':>7s} {'SSIM':>6s} | "
        f"{'relMSE':>6s} {'rTrim':>6s} | {'LapSig':>6s} {'WW_Std':>6s} | "
        f"{'Zero%':>6s} {'FF5x%':>5s} | {'GlsBias':>8s} {'MirZ%':>6s} | {'TotMs':>5s} {'VVL':>3s}"
    )
    print(hdr_line)
    print("-" * len(hdr_line))

    rows = []
    if "default_static_mc" in report["references"]:
        rows.append(("ref_mc_static", report["references"]["default_static_mc"]))
    for m_name, _ in motion_modes:
        if m_name in report["modes"] and "full" in report["modes"][m_name]:
            rows.append((m_name, report["modes"][m_name]))

    for label, data in rows:
        fl = data["full"]
        rr = data["rois"]
        hd = data.get("hdr") or {}
        st = data.get("stats") or {}
        rel_s = f"{hd['relmse']:6.4f}" if "relmse" in hd else "   n/a"
        trm_s = f"{hd['relmse_trim']:6.4f}" if "relmse_trim" in hd else "   n/a"
        ver = st.get("upways_version", "-")
        if ver in ("none", ""):
            ver = "-"
        print(
            f"{label:18s} | {ver:>4s} {st.get('dynamic_bounces', 0):4d} | "
            f"{fl['psnr']:7.2f} {fl['rmse']:7.4f} {fl['ssim']:6.4f} | "
            f"{rel_s} {trm_s} | "
            f"{fl['lap_sigma']:6.4f} {rr['white_wall']['local_std']:6.4f} | "
            f"{fl['zero_pct']:6.2f} {fl['firefly_5x_pct']:5.2f} | "
            f"{rr['glass_sphere']['luma_bias']:+8.4f} {rr['mirror_sphere']['zero_pct']:6.2f} | "
            f"{st.get('avg_frame_ms', 0.0):5.2f} {st.get('validation_errors', -1):3d}"
        )

    print("\nPer-ROI Breakdown (LocStd / PSNR_ub dB):")
    roi_names = list(BASE_ROIS.keys())
    roi_hdr = f"{'Mode':18s} | " + " | ".join(f"{rn:13s}" for rn in roi_names)
    print(roi_hdr)
    print("-" * len(roi_hdr))
    for label, data in rows:
        rr = data["rois"]
        cells = " | ".join(f"{rr[rn]['local_std']:.4f}/{rr[rn]['psnr']:5.2f}" for rn in roi_names)
        print(f"{label:18s} | {cells}")

    failed_checks = [c for c in checks if not c["passed"]]
    all_passed = len(failed_checks) == 0
    report["passed"] = all_passed
    report["summary"] = {
        "total_checks": len(checks),
        "passed_checks": len(checks) - len(failed_checks),
        "failed_checks": len(failed_checks),
    }

    report_path = os.path.join(args.out_dir, "motion_gate_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"\nGate Checks: {len(checks) - len(failed_checks)}/{len(checks)} passed | Report: {report_path}")
    if failed_checks:
        print("\nFAILED GATES:")
        for fc in failed_checks:
            print(f"  [FAIL] {fc['mode']}.{fc['check']}: {fc['detail']}")
        return 1

    print("[PASS] All P0 continuous-motion quality gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
