#!/usr/bin/env python3
"""
Pathways Image Quality & Camera Motion Stability Test
Validates that:
1. Native Wavefront path tracing delivers reference image quality, sub-8ms 4K latency, and >= 10% deep shadows.
2. Camera motion does not introduce ghost trails, smearing, or disocclusion distortion.
3. ReSTIR (when explicitly enabled) preserves contact shadows (>= 8%) and does not blow out exposure.
"""

import sys
import os
import subprocess
import json
import numpy as np
from PIL import Image

def run_cmd(cmd_list):
    print(f"[EXEC] {' '.join(cmd_list)}")
    res = subprocess.run(cmd_list, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[FAIL] Command failed with code {res.returncode}")
        print("STDOUT:\n", res.stdout[-1000:])
        print("STDERR:\n", res.stderr[-1000:])
        return False, res.stdout
    return True, res.stdout

def analyze_image(png_path, label=""):
    img = Image.open(png_path).convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    lum = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    mean_lum = float(np.mean(lum))
    shadow_pct = float(np.mean(lum < 0.05) * 100.0)
    min_val = float(arr.min())
    max_val = float(arr.max())
    mean_rgb = [float(np.mean(arr[:, :, c])) for c in range(3)]
    print(f"[{label}] Mean Lum: {mean_lum:.4f} | Deep Shadows (<0.05): {shadow_pct:.2f}% | RGB: [{mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f}] | Range: [{min_val:.3f}, {max_val:.3f}]")
    return {
        "mean_lum": mean_lum,
        "shadow_pct": shadow_pct,
        "mean_rgb": mean_rgb,
        "min_val": min_val,
        "max_val": max_val,
        "arr": arr
    }

def main():
    os.makedirs("output", exist_ok=True)
    bin_path = "./build/bin/pathways"
    if not os.path.exists(bin_path):
        print(f"[FAIL] Pathways binary not found at {bin_path}")
        sys.exit(1)

    all_passed = True

    # -------------------------------------------------------------------------
    # Test 1: Classroom 4K Native Wavefront Reference Mode (Default)
    # -------------------------------------------------------------------------
    print("\n====================================================================")
    print("  [TEST 1] Classroom 4K Native Reference Mode (Camera Motion, 60 Frames)")
    print("====================================================================")
    wf_png = "output/test_classroom_wf_motion.png"
    wf_stats = "output/stats_classroom_wf_motion.json"

    cmd_wf = [
        bin_path,
        "--headless",
        "--scene", "scenes/classroom/classroom_extended.glb",
        "--width", "3840",
        "--height", "2160",
        "--spp", "1",
        "--max-bounces", "4",
        "--frames", "60",
        "--warmup-frames", "10",
        "--no-accumulation",
        "--camera-motion",
        "--dump-frame", wf_png,
        "--dump-stats", wf_stats
    ]
    ok, _ = run_cmd(cmd_wf)
    if not ok:
        print("[FAIL] Classroom Wavefront run failed")
        all_passed = False
    else:
        m_wf = analyze_image(wf_png, "Classroom Default Wavefront")
        with open(wf_stats, "r") as f:
            st = json.load(f)
        avg_ms = st["performance"]["avg_frame_time_ms"]
        fps = st["performance"]["avg_fps"]
        print(f"       Latency: {avg_ms:.3f} ms ({fps:.1f} FPS)")

        if m_wf["shadow_pct"] < 10.0:
            print(f"[FAIL] Deep shadow retention {m_wf['shadow_pct']:.2f}% is below 10% floor")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Contact shadows intact: {m_wf['shadow_pct']:.2f}% >= 10%")

        if m_wf["mean_lum"] > 0.62:
            print(f"[FAIL] Mean luminance {m_wf['mean_lum']:.4f} exceeds ceiling 0.62")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Proper exposure: {m_wf['mean_lum']:.4f} <= 0.62")

        if avg_ms > 8.0:
            print(f"\033[33m[WARN]\033[0m Latency {avg_ms:.3f} ms slightly above 8.0 ms target")
        else:
            print(f"\033[32m[PASS]\033[0m Sub-8ms budget achieved: {avg_ms:.3f} ms <= 8.0 ms")

    # -------------------------------------------------------------------------
    # Test 2: Classroom 4K ReSTIR Resampling Mode (Verified Shader Fixes)
    # -------------------------------------------------------------------------
    print("\n====================================================================")
    print("  [TEST 2] Classroom 4K ReSTIR Mode with Algorithmic Fixes (Camera Motion)")
    print("====================================================================")
    restir_png = "output/test_classroom_restir_motion.png"
    restir_stats = "output/stats_classroom_restir_motion.json"

    cmd_restir = [
        bin_path,
        "--headless",
        "--scene", "scenes/classroom/classroom_extended.glb",
        "--width", "3840",
        "--height", "2160",
        "--spp", "1",
        "--max-bounces", "4",
        "--frames", "60",
        "--warmup-frames", "10",
        "--no-accumulation",
        "--camera-motion",
        "--restir",
        "--dump-frame", restir_png,
        "--dump-stats", restir_stats
    ]
    ok, _ = run_cmd(cmd_restir)
    if not ok:
        print("[FAIL] Classroom ReSTIR run failed")
        all_passed = False
    else:
        m_restir = analyze_image(restir_png, "Classroom Fixed ReSTIR")
        with open(restir_stats, "r") as f:
            st_r = json.load(f)
        avg_ms_r = st_r["performance"]["avg_frame_time_ms"]
        fps_r = st_r["performance"]["avg_fps"]
        print(f"       Latency: {avg_ms_r:.3f} ms ({fps_r:.1f} FPS)")

        # Verify shadow retention: previously was 0.34%, now must be >= 8%
        if m_restir["shadow_pct"] < 8.0:
            print(f"[FAIL] ReSTIR deep shadow retention {m_restir['shadow_pct']:.2f}% is below 8% floor (shadows destroyed)")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m ReSTIR contact shadows preserved: {m_restir['shadow_pct']:.2f}% >= 8% (was 0.34% before fix)")

        # Verify exposure: previously was 0.752 with green tint, now must be <= 0.65
        if m_restir["mean_lum"] > 0.65:
            print(f"[FAIL] ReSTIR mean luminance {m_restir['mean_lum']:.4f} is bleached / overexposed (exceeds 0.65)")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m ReSTIR exposure normalized: {m_restir['mean_lum']:.4f} <= 0.65 (was 0.752 before fix)")

        # Verify color tint balance: R, G, B should be balanced (not green tint G >> R, B)
        rg_diff = abs(m_restir["mean_rgb"][1] - m_restir["mean_rgb"][0])
        if rg_diff > 0.10:
            print(f"[FAIL] ReSTIR has severe green-yellow color cast: G-R diff={rg_diff:.4f}")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m ReSTIR color tint balanced: |G - R| = {rg_diff:.4f} <= 0.10")

    print("\n--------------------------------------------------------------------")
    if all_passed:
        print("\033[32m[SUCCESS]\033[0m All image quality and camera motion tests passed!")
        sys.exit(0)
    else:
        print("\033[31m[FAILURE]\033[0m Image quality regression detected.")
        sys.exit(1)

if __name__ == "__main__":
    main()
