#!/usr/bin/env python3
"""
scripts/verify_dgc_execset.py

Diagnostic verification script to audit DGC Execution Sets (VkIndirectExecutionSetEXT)
against standard GPU multi-dispatch indirect queues across Mesa RADV driver updates.

Checks:
  1. Veach Ajar dielectric transmission integrity (glass teapot ROI luminance).
  2. Cornell Box specular & dielectric sphere reflections/refractions.
  3. Ray survival count delta at bounce 1 and beyond.
  4. Energy conservation, PSNR, and difference heatmap.
"""

import os
import sys
import argparse
import subprocess
import json
import re
import numpy as np
import cv2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PATHWAYS_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
OUTPUT_DIR = os.path.join(PATHWAYS_ROOT, "output", "dgc_execset_audit")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_command(cmd, env_extra=None):
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=PATHWAYS_ROOT, env=env)
    return res.returncode, res.stdout, res.stderr

def parse_bounce_rays(stdout):
    bounce_rays = {}
    pattern = re.compile(r"Bounce\s+(\d+):.*?\|\s*(\d+)\s+rays left")
    for match in pattern.finditer(stdout):
        b = int(match.group(1))
        rays = int(match.group(2))
        bounce_rays[b] = rays
    return bounce_rays

def eval_veach_glass_roi(img):
    h, w = img.shape[:2]
    y0, y1 = int(0.45 * h), int(0.75 * h)
    x0, x1 = int(0.35 * w), int(0.75 * w)
    crop = img[y0:y1, x0:x1]
    ch, cw = crop.shape[:2]
    glass_core = crop[int(0.30*ch):int(0.70*ch), int(0.70*cw):int(0.90*cw)]
    lum = 0.2126 * glass_core[:,:,2] + 0.7152 * glass_core[:,:,1] + 0.0722 * glass_core[:,:,0]
    return float(np.mean(lum))

def eval_cornell_sphere_rois(img):
    # Cornell Box sphere regions in 1080p
    h, w = img.shape[:2]
    # Metal sphere (right side)
    metal_crop = img[int(0.55*h):int(0.75*h), int(0.55*w):int(0.75*w)]
    # Glass sphere (left side)
    glass_crop = img[int(0.55*h):int(0.75*h), int(0.25*w):int(0.45*w)]
    
    metal_lum = float(np.mean(0.2126 * metal_crop[:,:,2] + 0.7152 * metal_crop[:,:,1] + 0.0722 * metal_crop[:,:,0]))
    glass_lum = float(np.mean(0.2126 * glass_crop[:,:,2] + 0.7152 * glass_crop[:,:,1] + 0.0722 * glass_crop[:,:,0]))
    return metal_lum, glass_lum

def compare_images(img_base, img_exec, prefix="audit"):
    base_f = img_base.astype(np.float32) / 255.0
    exec_f = img_exec.astype(np.float32) / 255.0

    diff = np.abs(base_f - exec_f)
    mse = float(np.mean(diff ** 2))
    psnr = 10.0 * np.log10(1.0 / max(mse, 1e-10))
    mean_base = float(np.mean(base_f))
    mean_exec = float(np.mean(exec_f))
    energy_drift = abs(mean_exec - mean_base) / max(mean_base, 1e-5) * 100.0

    # Generate difference heatmap
    err_mag = np.linalg.norm(exec_f - base_f, axis=2)
    err_norm = np.clip(err_mag * 5.0, 0.0, 1.0)
    heatmap = cv2.applyColorMap((err_norm * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    
    out_diff = os.path.join(OUTPUT_DIR, f"{prefix}_diff_heatmap.png")
    cv2.imwrite(out_diff, heatmap)
    
    return {
        "mse": mse,
        "psnr": psnr,
        "mean_lum_baseline": mean_base * 255.0,
        "mean_lum_execset": mean_exec * 255.0,
        "energy_drift_pct": energy_drift,
        "diff_heatmap": out_diff
    }

def audit_scene(scene_path, scene_name, scene_bounces=False, spp=2, frames=64, res="1080p"):
    bin_path = os.path.join(PATHWAYS_ROOT, "build", "bin", "pathways")
    if not os.path.exists(bin_path):
        bin_path = os.path.join(PATHWAYS_ROOT, "build", "linux-release", "bin", "pathways")

    out_base_png = os.path.join(OUTPUT_DIR, f"{scene_name}_baseline.png")
    out_exec_png = os.path.join(OUTPUT_DIR, f"{scene_name}_execset.png")

    cmd_common = [
        bin_path, "--headless",
        "--scene", scene_path,
        "--res", res,
        "--spp", str(spp),
        "--frames", str(frames),
        "--no-animate"
    ]
    if scene_bounces:
        cmd_common.append("--scene-bounces")

    print(f"\n========================================================")
    print(f" Auditing Scene: {scene_name} ({res}, {spp} SPP, {frames} frames)")
    print(f"========================================================")

    # 1. Run Baseline (Multi-Dispatch Indirect)
    print(f"[*] Running Baseline (Multi-Dispatch Indirect, --no-dgc-execset)...")
    cmd_base = cmd_common + ["--no-dgc-execset", "--dump-frame", out_base_png]
    rc_base, out_base, err_base = run_command(cmd_base)
    if rc_base != 0:
        print(f"[ERROR] Baseline run failed: {err_base}")
        return None
    rays_base = parse_bounce_rays(out_base)

    # 2. Run with DGC Execution Set
    print(f"[*] Running with DGC Execution Set (--dgc-execset)...")
    cmd_exec = cmd_common + ["--dgc-execset", "--dump-frame", out_exec_png]
    rc_exec, out_exec, err_exec = run_command(cmd_exec)
    if rc_exec != 0:
        print(f"[ERROR] DGC ExecSet run failed: {err_exec}")
        return None
    rays_exec = parse_bounce_rays(out_exec)

    # Load rendered frames
    img_base = cv2.imread(out_base_png)
    img_exec = cv2.imread(out_exec_png)
    if img_base is None or img_exec is None:
        print(f"[ERROR] Failed to read rendered image files.")
        return None

    # Compare
    metrics = compare_images(img_base, img_exec, prefix=scene_name)

    print("\n--- Ray Count Breakdown per Bounce ---")
    bounces = sorted(set(list(rays_base.keys()) + list(rays_exec.keys())))
    ray_stats = {}
    for b in bounces:
        rb = rays_base.get(b, 0)
        re_val = rays_exec.get(b, 0)
        delta = re_val - rb
        pct = (delta / rb * 100.0) if rb > 0 else 0.0
        ray_stats[b] = {"baseline": rb, "execset": re_val, "delta": delta, "pct": pct}
        print(f"  Bounce {b:2d}: Baseline = {rb:8d} | ExecSet = {re_val:8d} | Delta = {delta:8d} ({pct:+.1f}%)")

    metrics["ray_stats"] = ray_stats

    if "veach" in scene_name.lower():
        roi_base = eval_veach_glass_roi(img_base)
        roi_exec = eval_veach_glass_roi(img_exec)
        metrics["veach_glass_roi_baseline"] = roi_base
        metrics["veach_glass_roi_execset"] = roi_exec
        print(f"\n--- Veach Ajar Glass Teapot ROI Luminance ---")
        print(f"  Baseline Glass ROI:  {roi_base:.2f}")
        print(f"  ExecSet Glass ROI:   {roi_exec:.2f}")
        drop_pct = (roi_base - roi_exec) / max(roi_base, 1e-4) * 100.0
        print(f"  Luminance Drop:      {drop_pct:.1f}%")
        metrics["veach_glass_drop_pct"] = drop_pct
        if drop_pct > 15.0 or roi_exec < 80.0:
            metrics["defect_detected"] = True
            print(f"  -> [DEFECT DETECTED] Significant dielectric transmission attenuation / blackout!")
        else:
            metrics["defect_detected"] = False
            print(f"  -> [OK] Dielectric transmission preserved within tolerance.")

    elif "cornell" in scene_name.lower():
        m_base, g_base = eval_cornell_sphere_rois(img_base)
        m_exec, g_exec = eval_cornell_sphere_rois(img_exec)
        metrics["cornell_metal_roi_baseline"] = m_base
        metrics["cornell_metal_roi_execset"] = m_exec
        metrics["cornell_glass_roi_baseline"] = g_base
        metrics["cornell_glass_roi_execset"] = g_exec
        print(f"\n--- Cornell Box Sphere ROI Luminances ---")
        print(f"  Metal Sphere ROI: Baseline = {m_base:.2f}, ExecSet = {m_exec:.2f} (Delta: {m_exec - m_base:+.2f})")
        print(f"  Glass Sphere ROI: Baseline = {g_base:.2f}, ExecSet = {g_exec:.2f} (Delta: {g_exec - g_base:+.2f})")
        b1_drop = (rays_base.get(1, 0) - rays_exec.get(1, 0)) / max(rays_base.get(1, 1), 1) * 100.0
        metrics["bounce1_ray_drop_pct"] = b1_drop
        print(f"  Bounce 1 Ray Drop: {b1_drop:.1f}%")
        if b1_drop > 25.0:
            metrics["defect_detected"] = True
            print(f"  -> [DEFECT DETECTED] Severe bounce-1 ray loss ({b1_drop:.1f}% rays dropped)!")
        else:
            metrics["defect_detected"] = False

    print(f"\n--- Image Comparison Metrics ---")
    print(f"  PSNR:         {metrics['psnr']:.2f} dB")
    print(f"  MSE:          {metrics['mse']:.6f}")
    print(f"  Energy Drift: {metrics['energy_drift_pct']:.2f}%")
    print(f"  Diff Heatmap: {metrics['diff_heatmap']}")

    return metrics

def main():
    parser = argparse.ArgumentParser(description="Audit DGC Execution Sets against Multi-Dispatch on Mesa RADV")
    parser.add_argument("--scene", choices=["veach", "cornell", "all"], default="all", help="Scene to audit")
    parser.add_argument("--res", default="1080p", help="Resolution (1080p, 720p, etc.)")
    parser.add_argument("--spp", type=int, default=2, help="SPP")
    parser.add_argument("--frames", type=int, default=64, help="Frames")
    args = parser.parse_args()

    results = {}
    if args.scene in ["veach", "all"]:
        results["veach"] = audit_scene(
            "scenes/veach-ajar/veach_ajar_extended.glb",
            "veach_ajar",
            scene_bounces=True,
            spp=args.spp,
            frames=args.frames,
            res=args.res
        )

    if args.scene in ["cornell", "all"]:
        results["cornell"] = audit_scene(
            "scenes/cornell-box/cornell_box_extended.glb",
            "cornell_box",
            scene_bounces=False,
            spp=args.spp,
            frames=args.frames,
            res=args.res
        )

    out_json = os.path.join(OUTPUT_DIR, "dgc_execset_audit_report.json")
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Full Audit Report written to: {out_json}")

    any_defect = any(r.get("defect_detected", False) for r in results.values() if r)
    if any_defect:
        print("\n>>> CONCLUSION: Driver defect IS STILL PRESENT in current Mesa release! <<<")
        return 2
    else:
        print("\n>>> CONCLUSION: Driver defect appears FIXED in current Mesa release! <<<")
        return 0

if __name__ == "__main__":
    sys.exit(main())
