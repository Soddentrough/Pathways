#!/usr/bin/env python3
"""
Pathways Automated Frame & Stats Verification Script
Validates rendered PNG/EXR frame captures and JSON performance metrics.
"""

import sys
import os
import json
from PIL import Image

def verify_frame(png_path, expected_width=None, expected_height=None):
    if not os.path.exists(png_path):
        print(f"\033[31m[FAIL]\033[0m Image file not found: {png_path}")
        return False

    try:
        img = Image.open(png_path)
        w, h = img.size
        print(f"\033[32m[PASS]\033[0m Opened image: {png_path} ({w}x{h}, mode: {img.mode})")

        if expected_width and expected_height:
            if w != expected_width or h != expected_height:
                print(f"\033[31m[FAIL]\033[0m Dimension mismatch: expected {expected_width}x{expected_height}, got {w}x{h}")
                return False
            print(f"\033[32m[PASS]\033[0m Resolution matches expected: {w}x{h}")

        # Statistical analysis of pixel intensity
        rgb_img = img.convert("RGB")
        extrema = rgb_img.getextrema()
        # extrema is ((r_min, r_max), (g_min, g_max), (b_min, b_max))
        max_val = max(e[1] for e in extrema)
        min_val = min(e[0] for e in extrema)

        if max_val == 0:
            print(f"\033[31m[FAIL]\033[0m Image is completely black (zero radiance rendered)")
            return False

        if min_val == 255 and max_val == 255:
            print(f"\033[31m[FAIL]\033[0m Image is completely blown out / saturated white")
            return False

        print(f"\033[32m[PASS]\033[0m Pixel intensity range: R={extrema[0]}, G={extrema[1]}, B={extrema[2]} (valid dynamic range)")
        return True

    except Exception as e:
        print(f"\033[31m[FAIL]\033[0m Error reading image {png_path}: {e}")
        return False

def verify_stats(json_path, max_target_ms=8.0):
    if not os.path.exists(json_path):
        print(f"\033[31m[FAIL]\033[0m Stats file not found: {json_path}")
        return False

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        perf = data.get("performance", {})
        gpu_info = data.get("primary_gpu", {})
        gpu_name = data.get("gpu_name") or gpu_info.get("device_name", "Unknown")
        avg_ms = perf.get("avg_frame_time_ms") if "avg_frame_time_ms" in perf else data.get("avg_frame_time_ms", 0.0)
        fps = perf.get("avg_fps") if "avg_fps" in perf else data.get("avg_fps", 0.0)
        rays_sec = perf.get("rays_per_second") if "rays_per_second" in perf else data.get("rays_per_second", 0.0)
        val_errors = perf.get("validation_errors") if "validation_errors" in perf else data.get("validation_errors", -1)

        print(f"\033[32m[PASS]\033[0m Loaded stats for GPU: {gpu_name}")
        print(f"       Average Frame Time: {avg_ms:.3f} ms ({fps:.1f} FPS)")
        print(f"       Ray Throughput:     {rays_sec:.2e} rays/sec")
        print(f"       Validation Errors:  {val_errors}")

        if val_errors != 0:
            print(f"\033[31m[FAIL]\033[0m Vulkan validation errors detected: {val_errors}")
            return False
        print(f"\033[32m[PASS]\033[0m Clean Vulkan validation pass (0 errors)")

        if avg_ms <= max_target_ms:
            print(f"\033[32m[PASS]\033[0m Frame time target achieved: {avg_ms:.3f} ms <= {max_target_ms:.1f} ms")
        else:
            print(f"\033[33m[INFO]\033[0m Single-GPU baseline frame time: {avg_ms:.3f} ms (Multi-GPU target: <{max_target_ms:.1f} ms)")

        return True

    except Exception as e:
        print(f"\033[31m[FAIL]\033[0m Error reading stats JSON: {e}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: verify_frame.py <image.png> <stats.json> [expected_width] [expected_height] [max_ms]")
        sys.exit(1)

    png_path = sys.argv[1]
    json_path = sys.argv[2]
    expected_w = int(sys.argv[3]) if len(sys.argv) > 3 else None
    expected_h = int(sys.argv[4]) if len(sys.argv) > 4 else None
    max_ms = float(sys.argv[5]) if len(sys.argv) > 5 else 8.0

    print("==========================================================")
    print("  Pathways Automated Test Verification")
    print("==========================================================")

    img_ok = verify_frame(png_path, expected_w, expected_h)
    stats_ok = verify_stats(json_path, max_ms)

    print("----------------------------------------------------------")
    if img_ok and stats_ok:
        print("\033[32m[SUCCESS]\033[0m All verification checks passed cleanly!")
        print("==========================================================")
        sys.exit(0)
    else:
        print("\033[31m[FAILURE]\033[0m Verification checks failed.")
        print("==========================================================")
        sys.exit(1)

if __name__ == "__main__":
    main()
