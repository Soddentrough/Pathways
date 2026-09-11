#!/usr/bin/env python3
"""
Pathways Automated Multi-GPU Frame Pacing Verification Script
Validates multi-GPU execution telemetry, ensuring smooth interactive frame pacing
and preventing PCIe BAR stall regressions during camera motion.
"""

import sys
import os
import json

def verify_mgpu_pacing(json_path, max_avg_ms=6.0, max_peak_ms=10.0, min_frames=30):
    if not os.path.exists(json_path):
        print(f"\033[31m[FAIL]\033[0m Stats file not found: {json_path}")
        return False

    try:
        with open(json_path, "r") as f:
            data = json.load(f)

        perf = data.get("performance", {})
        sec_gpu = data.get("secondary_gpu", {})

        gpu_name_prim = data.get("primary_gpu", {}).get("device_name", "Unknown")
        gpu_name_sec = sec_gpu.get("device_name", "None")
        sec_active = sec_gpu.get("active", False)

        avg_ms = perf.get("avg_frame_time_ms", 0.0)
        min_ms = perf.get("min_frame_time_ms", 0.0)
        max_ms = perf.get("max_frame_time_ms", 0.0)
        fps = perf.get("avg_fps", 0.0)
        total_frames = perf.get("total_frames", 0)
        val_errors = perf.get("validation_errors", -1)

        print("==========================================================")
        print("  Multi-GPU Frame Pacing Telemetry Verification")
        print("==========================================================")
        print(f"Primary GPU:    {gpu_name_prim}")
        print(f"Secondary GPU:  {gpu_name_sec} (Active: {sec_active})")
        print(f"Frames Sampled: {total_frames} (Min Required: {min_frames})")
        print(f"Avg Frame Time: {avg_ms:.3f} ms ({fps:.1f} FPS) [Target: <= {max_avg_ms:.1f} ms]")
        print(f"Min Frame Time: {min_ms:.3f} ms")
        print(f"Max Frame Time: {max_ms:.3f} ms [Peak Budget: <= {max_peak_ms:.1f} ms]")
        print(f"Vulkan Errors:  {val_errors}")
        print("----------------------------------------------------------")

        success = True

        # 1. Vulkan validation errors check
        if val_errors != 0:
            print(f"\033[31m[FAIL]\033[0m Vulkan validation errors detected: {val_errors}")
            success = False
        else:
            print(f"\033[32m[PASS]\033[0m Clean Vulkan validation pass (0 errors)")

        # 2. Secondary GPU active check
        if not sec_active:
            print(f"\033[31m[FAIL]\033[0m Secondary GPU was not active during multi-GPU test!")
            success = False
        else:
            print(f"\033[32m[PASS]\033[0m Dual discrete GPUs active and collaborating")

        # 3. Frame count check
        if total_frames < min_frames:
            print(f"\033[31m[FAIL]\033[0m Insufficient frame count: rendered {total_frames}, expected >= {min_frames}")
            success = False
        else:
            print(f"\033[32m[PASS]\033[0m Sufficient frame count ({total_frames} frames)")

        # 4. Average frame time check (catches throughput degradation)
        if avg_ms > max_avg_ms:
            print(f"\033[31m[FAIL]\033[0m Average frame time regression: {avg_ms:.3f} ms > {max_avg_ms:.1f} ms")
            success = False
        else:
            print(f"\033[32m[PASS]\033[0m Average frame time within budget: {avg_ms:.3f} ms <= {max_avg_ms:.1f} ms")

        # 5. Peak frame time check (catches PCIe BAR stall spikes and stutter)
        if max_ms > max_peak_ms:
            print(f"\033[31m[FAIL]\033[0m Peak frame time stall detected: max frame took {max_ms:.3f} ms (budget: {max_peak_ms:.1f} ms)")
            print(f"       This indicates inter-GPU PCIe bus contention, uncached BAR reads, or lockup.")
            success = False
        else:
            print(f"\033[32m[PASS]\033[0m Peak frame pacing smooth: {max_ms:.3f} ms <= {max_peak_ms:.1f} ms (no PCIe stalls)")

        print("==========================================================")
        if success:
            print("\033[32m[SUCCESS]\033[0m Multi-GPU frame pacing verified smoothly!")
            return True
        else:
            print("\033[31m[FAILURE]\033[0m Multi-GPU frame pacing verification failed.")
            return False

    except Exception as e:
        print(f"\033[31m[FAIL]\033[0m Error reading stats JSON: {e}")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: verify_mgpu_pacing.py <stats.json> [max_avg_ms] [max_peak_ms] [min_frames]")
        sys.exit(1)

    json_path = sys.argv[1]
    max_avg_ms = float(sys.argv[2]) if len(sys.argv) > 2 else 6.0
    max_peak_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    min_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 30

    if verify_mgpu_pacing(json_path, max_avg_ms, max_peak_ms, min_frames):
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()
