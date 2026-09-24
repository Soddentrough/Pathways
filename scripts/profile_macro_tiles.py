#!/usr/bin/env python3
"""
Pathways Macro-Tile Cache & Low-Level Hardware Profiler
Analyzes L0/L1/L2 cache hit rates, memory stalls, VRAM traffic, and stage timings
across different macro-tile counts (1, 2, 4, 8) at 4K resolution on AMD RDNA 4.
"""

import os
import sys
import glob
import json
import time
import struct
import subprocess
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "macro_tile_profile")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Import parse_rgp_spm from profile_cache_and_dgc
sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
from profile_cache_and_dgc import parse_rgp_spm

def run_profile_tile(macro_tiles, scene_args=None, warmup=5, frames=12, trace_frame=10):
    if scene_args is None:
        scene_args = []
    
    bin_path = os.path.join(PROJECT_ROOT, "build", "linux-release", "bin", "pathways")
    if not os.path.exists(bin_path):
        bin_path = os.path.join(PROJECT_ROOT, "build", "bin", "pathways")
        
    stats_json = os.path.join(OUTPUT_DIR, f"stats_tiles_{macro_tiles}.json")
    if os.path.exists(stats_json):
        os.remove(stats_json)

    # Clean old /tmp/pathways_*.rgp
    for old_rgp in glob.glob("/tmp/pathways_*.rgp"):
        try:
            os.remove(old_rgp)
        except OSError:
            pass

    cmd = [
        bin_path,
        "--headless",
        "--width", "3840",
        "--height", "2160",
        "--benchmark",
        "--warmup-frames", str(warmup),
        "--frames", str(frames),
        "--macro-tiles", str(macro_tiles),
        "--dump-stats", stats_json
    ] + scene_args

    env = os.environ.copy()
    env["MESA_VK_TRACE"] = "rgp"
    env["MESA_VK_TRACE_FRAME"] = str(trace_frame)
    env["MESA_VK_TRACE_PER_SUBMIT"] = "1"
    env["RADV_THREAD_TRACE_BUFFER_SIZE"] = "134217728"
    env["RADV_THREAD_TRACE_CACHE_COUNTERS"] = "1"
    env["RADV_THREAD_TRACE_INSTRUCTION_TIMING"] = "1"

    print(f"\n=======================================================")
    print(f"  Profiling --macro-tiles {macro_tiles} (Frame {trace_frame} Trace)")
    print(f"=======================================================")
    print("Command:", " ".join(cmd))

    t0 = time.time()
    res = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=PROJECT_ROOT)
    elapsed = time.time() - t0

    if res.returncode != 0:
        print(f"Error running pathways: returncode={res.returncode}")
        print("STDERR:", res.stderr)
        return None

    # Parse stdout for execution summary
    stdout_lines = res.stdout.splitlines()
    summary = {}
    for line in stdout_lines:
        if "Average Frame Time:" in line:
            summary["avg_frame_time"] = line.split("Average Frame Time:")[1].strip()
        elif "GPU Breakdown:" in line:
            summary["gpu_breakdown"] = line.split("GPU Breakdown:")[1].strip()
        elif "Classify (Primary RayGen):" in line:
            summary["classify"] = line.split("Classify (Primary RayGen):")[1].strip()
        elif "Bounce 0:" in line:
            summary["bounce_0"] = line.split("Bounce 0:")[1].strip()
        elif "Bounce 1:" in line:
            summary["bounce_1"] = line.split("Bounce 1:")[1].strip()
        elif "Bounce 2:" in line:
            summary["bounce_2"] = line.split("Bounce 2:")[1].strip()
        elif "Bounce 3:" in line:
            summary["bounce_3"] = line.split("Bounce 3:")[1].strip()
        elif "Tonemap / Resolve:" in line:
            summary["tonemap"] = line.split("Tonemap / Resolve:")[1].strip()
        elif "Ray Throughput:" in line:
            summary["throughput"] = line.split("Ray Throughput:")[1].strip()

    # Find the largest RGP trace (the main render frame submit)
    traces = glob.glob("/tmp/pathways_*.rgp")
    largest_trace = None
    max_size = 0
    for t in traces:
        sz = os.path.getsize(t)
        if sz > max_size:
            max_size = sz
            largest_trace = t

    spm_data = None
    if largest_trace:
        print(f"Found primary RGP trace: {largest_trace} ({max_size / (1024*1024):.2f} MB)")
        spm_data = parse_rgp_spm(largest_trace)
        # Archive trace
        dst_trace = os.path.join(OUTPUT_DIR, f"macro_tiles_{macro_tiles}.rgp")
        try:
            import shutil
            shutil.copyfile(largest_trace, dst_trace)
        except Exception:
            pass
    else:
        print("Warning: No RGP trace found!")

    return {
        "macro_tiles": macro_tiles,
        "elapsed_sec": elapsed,
        "engine_summary": summary,
        "spm": spm_data
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", nargs="*", default=[], help="Optional scene arguments")
    parser.add_argument("--tiles", nargs="+", type=int, default=[1, 2, 4, 8], help="List of tile counts to test")
    args = parser.parse_args()

    results = []
    for count in args.tiles:
        r = run_profile_tile(count, scene_args=args.scene)
        if r:
            results.append(r)

    # Save results to json
    report_json_path = os.path.join(OUTPUT_DIR, "macro_tiles_comparison.json")
    with open(report_json_path, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*80)
    print("                LOW-LEVEL HARDWARE ANALYSIS SUMMARY")
    print("="*80)
    print(f"{'Macro Tiles':<12} | {'Frame Time':<12} | {'L0 Hit %':<10} | {'L1 Hit %':<10} | {'L2 Hit %':<10} | {'Mem Stall %':<12} | {'VRAM Read GB/s':<14}")
    print("-" * 80)
    for r in results:
        tiles = r["macro_tiles"]
        ft = r["engine_summary"].get("avg_frame_time", "N/A").split("(")[0].strip()
        if r["spm"]:
            l0 = f"{r['spm']['l0_tcp']['hit_ratio_pct']}%"
            l1 = f"{r['spm']['l1_gl1c']['hit_ratio_pct']}%"
            l2 = f"{r['spm']['l2_gl2c']['hit_ratio_pct']}%"
            mstall = f"{r['spm']['stalls']['memory_unit_stalled_pct']}%"
            vram_bw = f"{r['spm']['bandwidth']['vram_read_bandwidth_gbps']} GB/s"
        else:
            l0 = l1 = l2 = mstall = vram_bw = "N/A"
        print(f"{tiles:<12} | {ft:<12} | {l0:<10} | {l1:<10} | {l2:<10} | {mstall:<12} | {vram_bw:<14}")
    print("="*80)

if __name__ == "__main__":
    main()
