#!/usr/bin/env python3
"""
Pathways Multi-GPU Scaling Verification Script
Computes the exact speedup S = T_single / T_multi and verifies S >= 1.80x.
"""

import sys
import os
import json

def verify_scaling(single_stats_path, multi_stats_path, min_speedup=1.80):
    if not os.path.exists(single_stats_path):
        print(f"\033[31m[FAIL]\033[0m Single GPU stats file not found: {single_stats_path}")
        return False
    if not os.path.exists(multi_stats_path):
        print(f"\033[31m[FAIL]\033[0m Multi-GPU stats file not found: {multi_stats_path}")
        return False

    with open(single_stats_path, "r") as f:
        single = json.load(f)
    with open(multi_stats_path, "r") as f:
        multi = json.load(f)

    perf_single = single.get("performance", {})
    perf_multi = multi.get("performance", {})
    t_single = perf_single.get("avg_frame_time_ms") if "avg_frame_time_ms" in perf_single else single.get("avg_frame_time_ms", 0.0)
    t_multi = perf_multi.get("avg_frame_time_ms") if "avg_frame_time_ms" in perf_multi else multi.get("avg_frame_time_ms", 0.0)
    fps_single = perf_single.get("avg_fps") if "avg_fps" in perf_single else single.get("avg_fps", 0.0)
    fps_multi = perf_multi.get("avg_fps") if "avg_fps" in perf_multi else multi.get("avg_fps", 0.0)
    rays_single = perf_single.get("rays_per_second") if "rays_per_second" in perf_single else single.get("rays_per_second", 0.0)
    rays_multi = perf_multi.get("rays_per_second") if "rays_per_second" in perf_multi else multi.get("rays_per_second", 0.0)

    if t_single <= 0.0 or t_multi <= 0.0:
        print(f"\033[31m[FAIL]\033[0m Invalid frame times (single: {t_single} ms, multi: {t_multi} ms)")
        return False

    speedup = t_single / t_multi
    throughput_gain = rays_multi / rays_single if rays_single > 0 else 0.0

    settings_single = single.get("engine_settings", {})
    settings_multi = multi.get("engine_settings", {})
    res_single = settings_single.get("resolution") if "resolution" in settings_single else single.get("resolution", [0, 0])
    res_multi = settings_multi.get("resolution") if "resolution" in settings_multi else multi.get("resolution", [0, 0])
    spp_single = settings_single.get("spp") if "spp" in settings_single else single.get("spp", 0)
    sec_gpu = multi.get("secondary_gpu", {})
    mgpu_mode = sec_gpu.get("mgpu_mode") if "mgpu_mode" in sec_gpu else multi.get("mgpu_mode", "mgpu")

    print("==========================================================")
    print("  Pathways: Multi-GPU Performance Scaling Verification")
    print("==========================================================")
    print(f"  Configuration: {res_single[0]}x{res_single[1]} @ {spp_single} SPP (Mode: {mgpu_mode})")
    print(f"  Single GPU:    {t_single:.3f} ms ({fps_single:.1f} FPS) | {rays_single:.2e} rays/s")
    print(f"  Dual GPU:      {t_multi:.3f} ms ({fps_multi:.1f} FPS) | {rays_multi:.2e} rays/s")
    print("----------------------------------------------------------")
    print(f"  Measured Multi-GPU Speedup:       {speedup:.3f}x")
    print(f"  Measured Ray Throughput Gain:     {throughput_gain:.3f}x")
    print(f"  Target Scaling Threshold:         >={min_speedup:.2f}x")
    print("----------------------------------------------------------")

    if speedup >= min_speedup:
        print(f"\033[32m[PASS]\033[0m Multi-GPU scaling requirement SATISFIED ({speedup:.3f}x >= {min_speedup:.2f}x)")
        print("==========================================================")
        return True
    else:
        print(f"\033[31m[FAIL]\033[0m Multi-GPU scaling below target ({speedup:.3f}x < {min_speedup:.2f}x)")
        print("==========================================================")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: verify_scaling.py <single_stats.json> <multi_stats.json> [min_speedup]")
        sys.exit(1)

    min_s = float(sys.argv[3]) if len(sys.argv) > 3 else 1.80
    success = verify_scaling(sys.argv[1], sys.argv[2], min_s)
    sys.exit(0 if success else 1)
