#!/usr/bin/env python3
"""
Pathways Caustics Verification Script
Validates that enabling --caustics produces real, focused photon irradiance
on receiver surfaces compared to a non-caustic baseline render.
"""

import sys
import os
import json
import numpy as np
from PIL import Image

def main():
    if len(sys.argv) < 4:
        print(f"Usage: {sys.argv[0]} <baseline_png> <caustics_png> <stats_json>")
        sys.exit(1)

    baseline_path = sys.argv[1]
    caustics_path = sys.argv[2]
    stats_path = sys.argv[3]

    print("==========================================================")
    print("  Pathways: Real-Time Caustics Verification               ")
    print("==========================================================")
    print(f"Baseline frame: {baseline_path}")
    print(f"Caustics frame: {caustics_path}")
    print(f"Stats JSON:     {stats_path}")

    # 1. Validate Telemetry & Gating in stats.json
    if not os.path.exists(stats_path):
        print(f"[FAIL] Stats file missing: {stats_path}")
        sys.exit(1)

    try:
        with open(stats_path, "r") as f:
            stats = json.load(f)
    except Exception as e:
        print(f"[FAIL] Failed to parse {stats_path}: {e}")
        sys.exit(1)

    caustics_sec = stats.get("engine_settings", {}).get("caustics", stats.get("caustics", {}))
    is_active = caustics_sec.get("caustics_active", False)
    has_dielectrics = caustics_sec.get("has_dielectrics", False)
    photon_count = caustics_sec.get("caustic_photons", 0)

    print(f"[Telemetry] caustics_active={is_active}, has_dielectrics={has_dielectrics}, photons={photon_count}")

    if not is_active:
        print("[FAIL] Telemetry error: caustics_active is FALSE despite --caustics flag")
        sys.exit(1)
    if not has_dielectrics:
        print("[FAIL] Telemetry error: has_dielectrics is FALSE for dielectric scene")
        sys.exit(1)

    # 2. Validate Image Disparity & Focal Energy Injection
    if not os.path.exists(baseline_path):
        print(f"[FAIL] Baseline image missing: {baseline_path}")
        sys.exit(1)
    if not os.path.exists(caustics_path):
        print(f"[FAIL] Caustics image missing: {caustics_path}")
        sys.exit(1)

    img_off = np.array(Image.open(baseline_path)).astype(np.float64)
    img_on = np.array(Image.open(caustics_path)).astype(np.float64)

    if img_off.shape != img_on.shape:
        print(f"[FAIL] Resolution mismatch: {img_off.shape} vs {img_on.shape}")
        sys.exit(1)

    # Handle both 8-bit and 16-bit PNG dumps
    max_scale = 65535.0 if (img_off.max() > 256.0 or img_on.max() > 256.0) else 255.0
    norm_off = img_off / max_scale
    norm_on = img_on / max_scale

    diff = np.abs(norm_on - norm_off)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    diff_pixels = int(np.sum(diff > 0.01))
    total_pixels = img_off.shape[0] * img_off.shape[1]

    print(f"[Disparity] Max Diff: {max_diff * 100.0:.2f}% | Mean Diff: {mean_diff * 100.0:.4f}% | Affected Pixels: {diff_pixels} / {total_pixels} ({diff_pixels / total_pixels * 100.0:.2f}%)")

    # Guardrails:
    # 1. A caustic focal hot spot must produce at least 15% peak intensity boost
    if max_diff < 0.15:
        print(f"[FAIL] Caustic focal hot-spot missing or negligible (max diff {max_diff * 100.0:.2f}% < 15.0%)")
        sys.exit(1)

    # 2. Overall scene must receive real photon energy (> 0.05% average screen luminance)
    if mean_diff < 0.0005:
        print(f"[FAIL] Caustic energy injection absent (mean diff {mean_diff * 100.0:.4f}% < 0.05%)")
        sys.exit(1)

    print("[SUCCESS] Real-time photon caustics verified successfully!")
    sys.exit(0)

if __name__ == "__main__":
    main()
