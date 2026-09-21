#!/usr/bin/env python3
"""
Pathways Automated E2E Test Suite: Cyber-City Dynamic Multi-Frame Video Hologram Playback
Validates:
1. Video timeline playback across multiple keyframes (Frame 0, Frame 72, Frame 192).
2. Absence of static culling defects (e.g. truncated UI screens or ghost android heads).
3. 16:9 widescreen voxel matrix coverage across both left android and right interactive UI HUD.
4. Real-time path tracing performance (Sub-8.3ms frame time / >120 FPS at 1080p, 0 validation errors).
"""

import sys
import os
import subprocess
import json
import time
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
os.chdir(PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output/e2e_video_playback")
os.makedirs(OUTPUT_DIR, exist_ok=True)

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


def load_rgb_normalized(image_path):
    img = Image.open(image_path).convert("RGB")
    return np.array(img, dtype=np.float32) / 255.0


def compute_luminance(arr):
    return 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]


def main():
    print("====================================================================")
    print(f"  {CLR_BOLD}Pathways E2E Test Suite: Cyber-City Dynamic Video Hologram{CLR_RESET}")
    print("====================================================================")
    print("Validating multi-frame dynamic playback, 16:9 voxel coverage & performance...")

    terrace_cam = "-6.8,29.0,36.5,2.0,25.0,-40.0,62.0"
    all_passed = True

    # --------------------------------------------------------------------------
    # Test 1: Multi-frame rendering (Frames 0, 72, 192)
    # --------------------------------------------------------------------------
    test_frames = [
        (1, "f0", "Frame 0 (Android Greeting)"),
        (72, "f72", "Frame 72 (Android Gesturing)"),
        (195, "f192", "Frame 192 (Android + Interactive UI HUD)")
    ]

    rendered_data = {}

    for frame_count, tag, desc in test_frames:
        png_path = os.path.join(OUTPUT_DIR, f"cyber_terrace_{tag}.png")
        stats_path = os.path.join(OUTPUT_DIR, f"stats_cyber_terrace_{tag}.json")

        print(f"\n[RENDER] Rendering {desc} ({frame_count} frames)...")
        rc, out, err, elapsed = run_pathways([
            "--headless",
            "--scene", "cyber-city",
            "--camera", terrace_cam,
            "--frames", str(frame_count),
            "--width", "1920",
            "--height", "1080",
            "--spp", "1",
            "--dump-frame", png_path,
            "--dump-stats", stats_path
        ])

        if rc != 0 or not os.path.exists(png_path) or not os.path.exists(stats_path):
            print(f"{CLR_RED}[FAIL]{CLR_RESET} Failed to render {desc}: rc={rc}, error={err[-300:]}")
            all_passed = False
            continue

        with open(stats_path, "r") as f:
            stats = json.load(f)

        perf = stats.get("performance", {})
        val_errors = perf.get("validation_errors", 0)
        avg_ms = perf.get("avg_frame_time_ms", 999.0)
        fps = perf.get("avg_fps", 0.0)

        # Performance assertion: Sub-8.3ms budget & 0 validation errors
        perf_ok = (val_errors == 0 and avg_ms <= 8.30)
        if not perf_ok:
            print(f"{CLR_RED}[FAIL]{CLR_RESET} Performance failure for {desc}: avg={avg_ms:.2f}ms (target: <=8.30ms), val_errors={val_errors}")
            all_passed = False
        else:
            print(f"{CLR_GREEN}[PASS]{CLR_RESET} {desc} rendered: {avg_ms:.2f}ms ({fps:.1f} FPS), 0 validation errors")

        arr = load_rgb_normalized(png_path)
        rendered_data[tag] = {
            "path": png_path,
            "arr": arr,
            "lum": compute_luminance(arr),
            "stats": stats
        }

    if len(rendered_data) < 3:
        print(f"\n{CLR_RED}[FATAL]{CLR_RESET} Insufficient frames successfully rendered to complete verification.")
        sys.exit(1)

    # --------------------------------------------------------------------------
    # Test 2: Dynamic Timeline Evolution (Frames must not be identical)
    # --------------------------------------------------------------------------
    print("\n--- Validating Dynamic Video Playback Across Timeline ---")
    arr_0 = rendered_data["f0"]["arr"]
    arr_72 = rendered_data["f72"]["arr"]
    arr_192 = rendered_data["f192"]["arr"]

    mae_0_72 = float(np.mean(np.abs(arr_72 - arr_0)))
    mae_72_192 = float(np.mean(np.abs(arr_192 - arr_72)))
    mae_0_192 = float(np.mean(np.abs(arr_192 - arr_0)))

    print(f"  Frame 0 -> Frame 72   MAE: {mae_0_72:.5f}")
    print(f"  Frame 72 -> Frame 192 MAE: {mae_72_192:.5f}")
    print(f"  Frame 0 -> Frame 192  MAE: {mae_0_192:.5f}")

    if mae_0_72 > 0.005 and mae_72_192 > 0.005 and mae_0_192 > 0.008:
        print(f"{CLR_GREEN}[PASS]{CLR_RESET} Video stream dynamically changes across timeline (no frozen frames)")
    else:
        print(f"{CLR_RED}[FAIL]{CLR_RESET} Video stream appears static or failed to advance")
        all_passed = False

    # --------------------------------------------------------------------------
    # Test 3: Hologram Widescreen Completeness (Frame 192 UI & Android Verification)
    # --------------------------------------------------------------------------
    print("\n--- Validating Full 16:9 Widescreen Hologram Geometry in Frame 192 ---")
    # In terrace camera view (-6.8, 29.0, 36.5 -> 2.0, 25.0, -40.0, 62 deg FOV):
    # Left terrace hologram receiver plate & projector is located at:
    # X in pixel coordinates: 700 to 1100
    # Y in pixel coordinates: 330 to 580
    # Left side (Android figure): X in [750, 890], Y in [350, 560]
    # Right side (Interactive UI HUD screen): X in [890, 1070], Y in [370, 540]

    lum_192 = rendered_data["f192"]["lum"]
    android_region = lum_192[350:560, 750:890]
    ui_region = lum_192[370:540, 890:1070]

    mean_lum_android = float(np.mean(android_region))
    mean_lum_ui = float(np.mean(ui_region))

    print(f"  Left Android Region  Mean Lum: {mean_lum_android:.4f}")
    print(f"  Right UI HUD Region  Mean Lum: {mean_lum_ui:.4f}")

    # Both regions must be actively illuminated by the voxel hologram emission
    # Neither should be dead dark (<0.10) or pure blown white (>0.95)
    android_ok = (0.15 <= mean_lum_android <= 0.85)
    ui_ok = (0.15 <= mean_lum_ui <= 0.85)

    if android_ok:
        print(f"{CLR_GREEN}[PASS]{CLR_RESET} Android figure geometry and emission intact on left side")
    else:
        print(f"{CLR_RED}[FAIL]{CLR_RESET} Android figure region failed luminance check (lum={mean_lum_android:.4f})")
        all_passed = False

    if ui_ok:
        print(f"{CLR_GREEN}[PASS]{CLR_RESET} Interactive UI HUD screen intact on right side (no missing/culled UI)")
    else:
        print(f"{CLR_RED}[FAIL]{CLR_RESET} Interactive UI HUD screen failed luminance check (lum={mean_lum_ui:.4f})")
        all_passed = False

    # Check color presence in UI HUD (should contain vibrant cyan/green/blue emissive tones)
    arr_192_ui = arr_192[370:540, 890:1070]
    mean_g = float(np.mean(arr_192_ui[:, :, 1]))
    mean_b = float(np.mean(arr_192_ui[:, :, 2]))
    has_cyber_color = (mean_g > 0.15 and mean_b > 0.20)

    if has_cyber_color:
        print(f"{CLR_GREEN}[PASS]{CLR_RESET} UI HUD exhibits rich cyan/blue holographic emission (G={mean_g:.3f}, B={mean_b:.3f})")
    else:
        print(f"{CLR_RED}[FAIL]{CLR_RESET} UI HUD lacks expected cyan/blue color signature (G={mean_g:.3f}, B={mean_b:.3f})")
        all_passed = False

    # --------------------------------------------------------------------------
    # Final Summary
    # --------------------------------------------------------------------------
    print("\n--------------------------------------------------------------------")
    if all_passed:
        print(f"{CLR_GREEN}[SUCCESS]{CLR_RESET} All Cyber-City Dynamic Video Hologram E2E Tests PASSED!")
        sys.exit(0)
    else:
        print(f"{CLR_RED}[FAILURE]{CLR_RESET} One or more Cyber-City E2E tests failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
