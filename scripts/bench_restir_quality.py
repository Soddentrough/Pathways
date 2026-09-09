#!/usr/bin/env python3
"""
Pathways ReSTIR DI Quality & Frame Pacing Benchmark
Compares Baseline (Uniform Direct Light NEE) against ReSTIR DI under:
1. Locked Frame Rates (120 FPS, 60 FPS) with QualityGovernor pacing
2. Single-Frame 1-SPP instant direct illumination quality & variance
3. Multi-frame continuous camera motion real-time quality
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np

def compute_local_variance(img_np, box_size=3):
    """
    Computes average local variance across 3x3 windows on luminance channel.
    Lower variance indicates cleaner, less noisy rendering.
    """
    if img_np.ndim == 3:
        # Rec. 709 Luminance
        gray = 0.2126 * img_np[:, :, 0] + 0.7152 * img_np[:, :, 1] + 0.0722 * img_np[:, :, 2]
    else:
        gray = img_np

    try:
        from scipy.ndimage import uniform_filter
        mean = uniform_filter(gray, size=box_size, mode='reflect')
        mean_sq = uniform_filter(gray**2, size=box_size, mode='reflect')
    except ImportError:
        # Pure numpy fallback box filter via 2D cumsum
        pad = box_size // 2
        padded = np.pad(gray, pad, mode='reflect')
        cs_x = np.cumsum(padded, axis=0)
        bx = cs_x[box_size:, :] - cs_x[:-box_size, :]
        cs_y = np.cumsum(bx, axis=1)
        mean = (cs_y[:, box_size:] - cs_y[:, :-box_size]) / (box_size * box_size)

        padded_sq = np.pad(gray**2, pad, mode='reflect')
        cs_sq_x = np.cumsum(padded_sq, axis=0)
        bx_sq = cs_sq_x[box_size:, :] - cs_sq_x[:-box_size, :]
        cs_sq_y = np.cumsum(bx_sq, axis=1)
        mean_sq = (cs_sq_y[:, box_size:] - cs_sq_y[:, :-box_size]) / (box_size * box_size)

    var_map = np.maximum(0.0, mean_sq - mean**2)
    return float(np.mean(var_map))

def run_test(bin_path, scene, restir, target_fps, frames, spp, camera_motion, resolution, out_prefix, warmup_frames=30):
    png_path = f"{out_prefix}.png"
    json_path = f"{out_prefix}.json"

    cmd = [
        str(bin_path),
        "--headless",
        "--scene", scene,
        "--res", resolution,
        "--frames", str(frames),
        "--spp", str(spp),
        "--dump-frame", png_path,
        "--dump-stats", json_path,
    ]

    if frames > 1:
        cmd.extend(["--warmup-frames", str(warmup_frames), "--no-accumulation"])

    if restir:
        cmd.append("--restir-di")
    else:
        cmd.append("--no-restir-di")

    if target_fps > 0:
        cmd.extend(["--target-fps", str(target_fps)])
    else:
        cmd.extend(["--target-fps", "0"])

    if camera_motion:
        cmd.append("--camera-motion")

    t_start = time.perf_counter()
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    t_end = time.perf_counter()

    if proc.returncode != 0:
        print(f"Error running command: {' '.join(cmd)}")
        print(proc.stderr[-500:])
        return None

    # Parse dumped stats
    stats_data = {}
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                stats_data = json.load(f)
        except Exception as e:
            print(f"Warning: Could not read {json_path}: {e}")

    # Parse image quality
    local_var = None
    if os.path.exists(png_path):
        try:
            im = Image.open(png_path).convert("RGB")
            im_arr = np.array(im, dtype=np.float32) / 255.0
            local_var = compute_local_variance(im_arr)
        except Exception as e:
            print(f"Warning: Could not analyze {png_path}: {e}")

    perf = stats_data.get("performance", {})
    eng = stats_data.get("engine_settings", {})

    return {
        "restir": restir,
        "target_fps": target_fps,
        "frames": frames,
        "spp": spp,
        "camera_motion": camera_motion,
        "png_path": png_path,
        "json_path": json_path,
        "execution_time_s": t_end - t_start,
        "current_frame_time_ms": perf.get("current_frame_time_ms", 0.0),
        "current_fps": perf.get("current_fps", 0.0),
        "avg_frame_time_ms": perf.get("avg_frame_time_ms", 0.0),
        "dynamic_spp": eng.get("dynamic_spp", spp),
        "dynamic_bounces": eng.get("dynamic_bounces", 4),
        "local_variance": local_var,
    }

def main():
    parser = argparse.ArgumentParser(description="Pathways ReSTIR DI Quality & Frame Pacing Benchmark")
    parser.add_argument("--bin", default="./build/bin/pathways", help="Path to pathways binary")
    parser.add_argument("--scene", default="many-lights", help="Scene name or path (default: many-lights)")
    parser.add_argument("--res", default="1280x720", help="Resolution WxH (default: 1280x720)")
    parser.add_argument("--out-dir", default="./output/bench_restir", help="Output directory")
    parser.add_argument("--warmup-frames", type=int, default=30, help="Number of warmup frames to exclude from stats (default: 30)")
    args = parser.parse_args()

    bin_path = Path(args.bin).resolve()
    if not bin_path.exists():
        print(f"Error: pathways binary not found at {bin_path}")
        sys.exit(1)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("================================================================================")
    print("      Pathways ReSTIR DI Quality & Locked Frame Rate Benchmark                  ")
    print("================================================================================")
    print(f"Scene:         {args.scene}")
    print(f"Resolution:    {args.res}")
    print(f"Warmup Frames: {args.warmup_frames}")
    print(f"Binary:        {bin_path}")
    print(f"Output:        {out_dir}")
    print("--------------------------------------------------------------------------------\n")

    test_configs = [
        # Name, target_fps, frames, spp, camera_motion
        ("1-SPP Static Quality", 0, 1, 1, False),
        ("Locked 120 FPS (8.33ms)", 120, 240, 1, True),
        ("Locked 60 FPS (16.67ms)", 60, 180, 1, True),
        ("Uncapped Motion Quality", 0, 300, 1, True),
    ]

    results = []

    for name, target_fps, frames, spp, motion in test_configs:
        print(f"--> Running Benchmark Scenario: {name}")
        base_prefix = str(out_dir / f"{name.replace(' ', '_').lower()}_baseline")
        restir_prefix = str(out_dir / f"{name.replace(' ', '_').lower()}_restir")

        print("    Executing Baseline (Uniform NEE)...", end="", flush=True)
        res_base = run_test(bin_path, args.scene, False, target_fps, frames, spp, motion, args.res, base_prefix, args.warmup_frames)
        print(" Done.")

        print("    Executing ReSTIR DI...", end="", flush=True)
        res_restir = run_test(bin_path, args.scene, True, target_fps, frames, spp, motion, args.res, restir_prefix, args.warmup_frames)
        print(" Done.")

        if res_base and res_restir:
            var_base = res_base.get("local_variance")
            var_restir = res_restir.get("local_variance")
            var_reduction = 0.0
            if var_base and var_restir and var_base > 0:
                var_reduction = ((var_restir - var_base) / var_base) * 100.0

            # Compute PSNR/MSE between the two output images
            mse, psnr = 0.0, 0.0
            if os.path.exists(res_base["png_path"]) and os.path.exists(res_restir["png_path"]):
                im1 = np.array(Image.open(res_base["png_path"]).convert("RGB"), dtype=np.float32) / 255.0
                im2 = np.array(Image.open(res_restir["png_path"]).convert("RGB"), dtype=np.float32) / 255.0
                diff = im1 - im2
                mse = float(np.mean(diff ** 2))
                psnr = float(10 * np.log10(1.0 / mse)) if mse > 0 else 999.0

            scenario_res = {
                "scenario": name,
                "target_fps": target_fps,
                "target_frame_time_ms": (1000.0 / target_fps) if target_fps > 0 else 0.0,
                "baseline": res_base,
                "restir": res_restir,
                "var_reduction_pct": var_reduction,
                "mse": mse,
                "psnr_db": psnr,
            }
            results.append(scenario_res)

            print(f"    [Result] Baseline Variance: {var_base:.6f} | ReSTIR Variance: {var_restir:.6f}")
            print(f"    [Result] Variance Reduction: {var_reduction:+.2f}%")
            if target_fps > 0:
                target_ms = 1000.0 / target_fps
                base_ms = res_base['current_frame_time_ms']
                restir_ms = res_restir['current_frame_time_ms']
                print(f"    [Result] Target Interval: {target_ms:.2f} ms | Baseline: {base_ms:.2f} ms | ReSTIR: {restir_ms:.2f} ms")
        print()

    # Summary Report Table
    print("=========================================================================================================")
    print(f"{'Scenario':<26} | {'Mode':<10} | {'Frame Time':<14} | {'FPS':<8} | {'Variance':<12} | {'Noise Change'}")
    print("---------------------------------------------------------------------------------------------------------")
    for r in results:
        sc = r["scenario"]
        b = r["baseline"]
        rst = r["restir"]
        var_red = r["var_reduction_pct"]

        print(f"{sc:<26} | {'Baseline':<10} | {b['current_frame_time_ms']:>6.2f} ms      | {b['current_fps']:>5.1f}  | {b['local_variance']:>10.6f} | Ref")
        print(f"{'':<26} | {'ReSTIR':<10} | {rst['current_frame_time_ms']:>6.2f} ms      | {rst['current_fps']:>5.1f}  | {rst['local_variance']:>10.6f} | {var_red:>+6.2f}%")
        print("---------------------------------------------------------------------------------------------------------")

    # Save summary JSON
    summary_path = out_dir / "bench_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[PASS] Detailed benchmark results saved to {summary_path}")

if __name__ == "__main__":
    main()
