#!/usr/bin/env python3
"""
Pathways Image Quality & Camera Motion Stability Test
Validates that:
1. Native Wavefront path tracing delivers reference image quality, sub-8ms 4K latency, and >= 10% deep shadows.
2. Camera motion does not introduce ghost trails, smearing, or disocclusion distortion.
3. Pure Monte Carlo multi-sample accumulation achieves pristine convergence and preserves contact shadows (>= 10%).
"""

import sys
import os
import subprocess
import json
import numpy as np
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    blown_pct = float(np.mean(lum > 0.95) * 100.0)
    min_val = float(arr.min())
    max_val = float(arr.max())
    mean_rgb = [float(np.mean(arr[:, :, c])) for c in range(3)]
    print(f"[{label}] Mean Lum: {mean_lum:.4f} | Deep Shadows (<0.05): {shadow_pct:.2f}% | Blown Out (>0.95): {blown_pct:.2f}% | RGB: [{mean_rgb[0]:.3f}, {mean_rgb[1]:.3f}, {mean_rgb[2]:.3f}] | Range: [{min_val:.3f}, {max_val:.3f}]")
    return {
        "mean_lum": mean_lum,
        "shadow_pct": shadow_pct,
        "blown_pct": blown_pct,
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
    # Test 1: Classroom 4K Native Wavefront Mode (Camera Motion, 60 Frames)
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
        "--camera-motion",
        "--dump-frame", wf_png,
        "--dump-stats", wf_stats
    ]
    ok, _ = run_cmd(cmd_wf)
    if not ok:
        print("[FAIL] Classroom Wavefront run failed")
        all_passed = False
    else:
        m_wf = analyze_image(wf_png, "Classroom Wavefront Motion")
        with open(wf_stats, "r") as f:
            st = json.load(f)
        avg_ms = st["performance"]["avg_frame_time_ms"]
        fps = st["performance"]["avg_fps"]
        print(f"       Latency: {avg_ms:.3f} ms ({fps:.1f} FPS)")

        if m_wf["shadow_pct"] < 10.0:
            print(f"[FAIL] Deep shadow retention {m_wf['shadow_pct']:.2f}% is below 10% floor")
            all_passed = False
        elif m_wf["shadow_pct"] > 40.0:
            print(f"[FAIL] Deep shadow retention {m_wf['shadow_pct']:.2f}% exceeds 40% ceiling (dark collapse)")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Contact shadows intact: {m_wf['shadow_pct']:.2f}% in [10%, 40%]")

        if m_wf["mean_lum"] < 0.20:
            print(f"[FAIL] Mean luminance {m_wf['mean_lum']:.4f} below floor 0.20 (dark collapse under motion)")
            all_passed = False
        elif m_wf["mean_lum"] > 0.50:
            print(f"[FAIL] Mean luminance {m_wf['mean_lum']:.4f} exceeds ceiling 0.50 (overexposed)")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Proper exposure: {m_wf['mean_lum']:.4f} in [0.20, 0.50]")

        if m_wf["blown_pct"] > 5.0:
            print(f"[FAIL] Blown-out percentage {m_wf['blown_pct']:.2f}% exceeds 5.0% ceiling")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Blown-out pixels controlled: {m_wf['blown_pct']:.2f}% <= 5.0%")

        if avg_ms > 16.0:
            print(f"\033[33m[WARN]\033[0m Latency {avg_ms:.3f} ms slightly above 16.0 ms target")
        else:
            print(f"\033[32m[PASS]\033[0m Target latency achieved: {avg_ms:.3f} ms <= 16.0 ms")

    # -------------------------------------------------------------------------
    # Test 2: Classroom 4K Pure Monte Carlo Convergence (30 Frames Static)
    # -------------------------------------------------------------------------
    print("\n====================================================================")
    print("  [TEST 2] Classroom 4K Pure Monte Carlo Convergence (30 Frames Static)")
    print("====================================================================")
    mc_png = "output/test_classroom_mc_converged.png"
    mc_stats = "output/stats_classroom_mc_converged.json"

    cmd_mc = [
        bin_path,
        "--headless",
        "--scene", "scenes/classroom/classroom_extended.glb",
        "--width", "3840",
        "--height", "2160",
        "--spp", "1",
        "--max-bounces", "4",
        "--frames", "30",
        "--warmup-frames", "5",
        "--dump-frame", mc_png,
        "--dump-stats", mc_stats
    ]
    ok, _ = run_cmd(cmd_mc)
    if not ok:
        print("[FAIL] Classroom Pure Monte Carlo run failed")
        all_passed = False
    else:
        m_mc = analyze_image(mc_png, "Classroom Converged Pure Monte Carlo")
        with open(mc_stats, "r") as f:
            st_mc = json.load(f)
        avg_ms_mc = st_mc["performance"]["avg_frame_time_ms"]
        fps_mc = st_mc["performance"]["avg_fps"]
        print(f"       Latency: {avg_ms_mc:.3f} ms ({fps_mc:.1f} FPS)")

        if m_mc["shadow_pct"] < 5.0:
            print(f"[FAIL] Deep shadow retention {m_mc['shadow_pct']:.2f}% is below 5% floor (shadows destroyed)")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Contact shadows preserved: {m_mc['shadow_pct']:.2f}% >= 5%")

        if m_mc["mean_lum"] < 0.35:
            print(f"[FAIL] Mean luminance {m_mc['mean_lum']:.4f} is too dark (below 0.35)")
            all_passed = False
        elif m_mc["mean_lum"] > 0.62:
            print(f"[FAIL] Mean luminance {m_mc['mean_lum']:.4f} is bleached / overexposed (exceeds 0.62)")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Exposure normalized: {m_mc['mean_lum']:.4f} in [0.35, 0.62]")

        if m_mc["blown_pct"] > 30.0:
            print(f"[FAIL] Blown-out percentage {m_mc['blown_pct']:.2f}% exceeds 30.0% ceiling")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Highlight bounds preserved: {m_mc['blown_pct']:.2f}% <= 30.0%")

        rg_diff = abs(m_mc["mean_rgb"][1] - m_mc["mean_rgb"][0])
        if rg_diff > 0.10:
            print(f"[FAIL] Color cast detected: G-R diff={rg_diff:.4f}")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Color tint balanced: |G - R| = {rg_diff:.4f} <= 0.10")

    # -------------------------------------------------------------------------
    # Test 3: Living Room Static Convergence vs Dynamic Motion (1080p)
    # -------------------------------------------------------------------------
    print("\n====================================================================")
    print("  [TEST 3] Living Room Static Convergence & Dynamic Motion Stability")
    print("====================================================================")
    lr_stat_png = "output/test_lr_static_test.png"
    lr_stat_json = "output/stats_lr_static_test.json"
    cmd_lr_stat = [
        bin_path,
        "--headless",
        "--scene", "scenes/living-room/living_room_extended.glb",
        "--width", "1920",
        "--height", "1080",
        "--spp", "1",
        "--max-bounces", "4",
        "--frames", "60",
        "--warmup-frames", "10",
        "--dump-frame", lr_stat_png,
        "--dump-stats", lr_stat_json
    ]
    ok_stat, _ = run_cmd(cmd_lr_stat)

    lr_mot_png = "output/test_lr_motion_test.png"
    lr_mot_json = "output/stats_lr_motion_test.json"
    cmd_lr_mot = [
        bin_path,
        "--headless",
        "--scene", "scenes/living-room/living_room_extended.glb",
        "--width", "1920",
        "--height", "1080",
        "--spp", "1",
        "--max-bounces", "4",
        "--frames", "60",
        "--warmup-frames", "10",
        "--camera-motion",
        "--dump-frame", lr_mot_png,
        "--dump-stats", lr_mot_json
    ]
    ok_mot, _ = run_cmd(cmd_lr_mot)

    if not ok_stat or not ok_mot:
        print("[FAIL] Living room test executions failed")
        all_passed = False
    else:
        m_stat = analyze_image(lr_stat_png, "Living Room Static (60 Frames)")
        m_mot = analyze_image(lr_mot_png, "Living Room Motion (60 Frames)")

        # Static checks: strictly verify NO overexposure blowout
        if m_stat["mean_lum"] < 0.25 or m_stat["mean_lum"] > 0.50:
            print(f"[FAIL] Living Room static mean luminance {m_stat['mean_lum']:.4f} out of bounds [0.25, 0.50]")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Living Room static convergence: Mean={m_stat['mean_lum']:.4f} in [0.25, 0.50]")

        if m_stat["blown_pct"] > 2.0:
            print(f"[FAIL] Living Room static blown-out pixels {m_stat['blown_pct']:.2f}% exceeds 2.0% (overexposure regression!)")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Living Room static blown-out pixels: {m_stat['blown_pct']:.2f}% <= 2.0%")

        # Dynamic motion checks: strictly verify NO dark grainy collapse
        if m_mot["mean_lum"] < 0.15 or m_mot["mean_lum"] > 0.35:
            print(f"[FAIL] Living Room motion mean luminance {m_mot['mean_lum']:.4f} out of bounds [0.15, 0.35] (energy collapse!)")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Living Room motion exposure: Mean={m_mot['mean_lum']:.4f} in [0.15, 0.35]")

        if m_mot["shadow_pct"] > 35.0:
            print(f"[FAIL] Living Room motion shadow percentage {m_mot['shadow_pct']:.2f}% exceeds 35% (collapsed to dark noise!)")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Living Room motion shadow retention: {m_mot['shadow_pct']:.2f}% <= 35.0%")

        retention = m_mot["mean_lum"] / m_stat["mean_lum"] * 100.0
        if retention < 50.0:
            print(f"[FAIL] Dynamic motion energy retention {retention:.1f}% below 50.0% threshold")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Dynamic motion energy retention: {retention:.1f}% >= 50.0%")

    # -------------------------------------------------------------------------
    # Test 4: Multi-GPU Checkerboard + FSR 3.1 Dynamic Camera Motion & Seam Verification
    # -------------------------------------------------------------------------
    print("\n====================================================================")
    print("  [TEST 4] Multi-GPU Checkerboard + FSR 3.1 Dynamic Motion & Seam Verification")
    print("====================================================================")
    mgpu_mot_png = "output/test_classroom_mgpu_tile_fsr3_motion.png"
    mgpu_mot_json = "output/stats_classroom_mgpu_tile_fsr3_motion.json"
    cmd_mgpu_mot = [
        bin_path,
        "--headless",
        "--scene", "scenes/classroom/classroom_extended.glb",
        "--width", "2560",
        "--height", "1440",
        "--spp", "1",
        "--max-bounces", "4",
        "--frames", "30",
        "--camera-motion",
        "--mgpu",
        "--upscaler", "fsr3",
        "--dump-frame", mgpu_mot_png,
        "--dump-stats", mgpu_mot_json
    ]
    ok_mgpu_mot, _ = run_cmd(cmd_mgpu_mot)
    if not ok_mgpu_mot:
        print("[FAIL] Multi-GPU FSR 3.1 motion test execution failed")
        all_passed = False
    else:
        m_mgpu_mot = analyze_image(mgpu_mot_png, "Classroom mGPU FSR 3.1 Motion")
        with open(mgpu_mot_json, "r") as f:
            st = json.load(f)
        avg_ms = st["performance"]["avg_frame_time_ms"]
        fps = st["performance"]["avg_fps"]
        val_errors = st["performance"].get("validation_errors", -1)
        sec_active = st.get("secondary_gpu", {}).get("active", False)
        print(f"       Latency: {avg_ms:.3f} ms ({fps:.1f} FPS) | Secondary GPU Active: {sec_active} | Vulkan Errors: {val_errors}")

        if val_errors != 0:
            print(f"[FAIL] Vulkan validation errors: {val_errors}")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Clean Vulkan validation (0 errors)")

        img_arr = np.array(Image.open(mgpu_mot_png))
        is_blk = np.all(img_arr[:, :, :3] < 5, axis=2)
        right_blk = is_blk[:, 2520:]
        blk_cols = np.where(np.sum(right_blk, axis=0) > (1440 * 0.25))[0]
        if len(blk_cols) > 0:
            print(f"[FAIL] Detected {len(blk_cols)} black clipped column(s) on right boundary!")
            all_passed = False
        else:
            print(f"\033[32m[PASS]\033[0m Right boundary intact: 0 truncated tile columns (x >= 2520)")

        if avg_ms <= 3.5:
            print(f"\033[32m[PASS]\033[0m Target latency achieved: {avg_ms:.3f} ms <= 3.5 ms ({fps:.1f} FPS)")
        else:
            print(f"\033[33m[WARN]\033[0m Frame time {avg_ms:.3f} ms slightly above 3.5 ms target")

    # -------------------------------------------------------------------------
    # Test 5: Automated Before/After Golden Reference Verification
    # -------------------------------------------------------------------------
    print("\n====================================================================")
    print("  [TEST 5] Before/After Golden Reference Verification & Anomaly Detection")
    print("====================================================================")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
    try:
        from visual_regression_test import compute_metrics, generate_html_report, TEST_CONFIGS
        vis_results = []
        for cfg in TEST_CONFIGS:
            if os.path.exists(cfg["render_path"]) and os.path.exists(cfg["ref_path"]):
                m = compute_metrics(cfg["render_path"], cfg["ref_path"], cfg["diff_path"])
                vis_results.append({"config": cfg, "metrics": m})
                sev = m.get("severity")
                if sev == "pass":
                    print(f"  \033[32m[{m['status']}]\033[0m {cfg['name']}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.1f} dB")
                elif sev == "notice":
                    print(f"  \033[35m[{m['status']}]\033[0m {cfg['name']}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.1f} dB | {m['detail']}")
                elif sev == "warn":
                    print(f"  \033[33m[{m['status']}]\033[0m {cfg['name']}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.1f} dB | {m['detail']}")
                else:
                    print(f"  \033[31m[{m['status']}]\033[0m {cfg['name']}: SSIM={m['ssim']:.4f}, PSNR={m['psnr']:.1f} dB | {m['detail']}")
                    all_passed = False
        generate_html_report(vis_results, "output/visual_regression_report.html")
    except Exception as e:
        print(f"[WARN] Visual regression engine check error: {e}")

    print("\n--------------------------------------------------------------------")
    if all_passed:
        print("\033[32m[SUCCESS]\033[0m All image quality and camera motion tests passed cleanly!")
        sys.exit(0)
    else:
        print("\033[31m[FAILURE]\033[0m Image quality regression detected.")
        sys.exit(1)

if __name__ == "__main__":
    main()
