#!/usr/bin/env python3
"""
Head-to-Head Performance Benchmark:
Main (Full-Frame Macro Tiling) vs. Hybrid Primary Tiling (perf-uma-cache-opt)
"""

import os
import sys
import json
import subprocess
import time

PATHWAYS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN_MAIN = os.path.join(PATHWAYS_ROOT, "build", "linux-release", "bin", "pathways_main")
BIN_HYBRID = os.path.join(PATHWAYS_ROOT, "build", "linux-release", "bin", "pathways")
OUT_DIR = os.path.join(PATHWAYS_ROOT, "output", "approach_comparison")
os.makedirs(OUT_DIR, exist_ok=True)

SCENES = [
    {
        "name": "Cornell Box",
        "scene_arg": [],
        "width": 3840,
        "height": 2160,
        "bounces": 4,
    },
    {
        "name": "Damaged Helmet (PBR)",
        "scene_arg": ["--scene", "scenes/DamagedHelmet.glb"],
        "width": 3840,
        "height": 2160,
        "bounces": 4,
    },
    {
        "name": "Breakfast Room (GI)",
        "scene_arg": ["--scene", "scenes/breakfast-room/breakfast_room_extended.glb"],
        "width": 3840,
        "height": 2160,
        "bounces": 4,
    }
]

def run_benchmark(bin_path, extra_args, dump_file):
    if os.path.exists(dump_file):
        os.remove(dump_file)
    cmd = [
        bin_path,
        "--headless",
        "--benchmark",
        "--warmup-frames", "5",
        "--frames", "15",
        "--dump-stats", dump_file
    ] + extra_args
    res = subprocess.run(cmd, cwd=PATHWAYS_ROOT, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(dump_file):
        print(f"Error running {bin_path} with {extra_args}: {res.stderr}")
        return None
    with open(dump_file, "r") as f:
        return json.load(f)

def parse_run_stats(data):
    perf = data["performance"]
    wf = perf["wavefront_profiler_breakdown"]
    bounces = wf.get("bounces", [])
    
    b0_shade = bounces[0]["shade_ms"] if len(bounces) > 0 else 0.0
    b0_isect = bounces[0]["intersect_ms"] if len(bounces) > 0 else 0.0
    b1_shade = bounces[1]["shade_ms"] if len(bounces) > 1 else 0.0
    b1_isect = bounces[1]["intersect_ms"] if len(bounces) > 1 else 0.0
    b2_shade = bounces[2]["shade_ms"] if len(bounces) > 2 else 0.0
    b2_isect = bounces[2]["intersect_ms"] if len(bounces) > 2 else 0.0
    b3_shade = bounces[3]["shade_ms"] if len(bounces) > 3 else 0.0
    
    return {
        "frame_time_ms": perf["avg_frame_time_ms"],
        "fps": perf["avg_fps"],
        "wf_total_ms": wf["total_wavefront_time_ms"],
        "classify_ms": wf["classify_time_ms"],
        "b0_shade_ms": b0_shade,
        "b0_isect_ms": b0_isect,
        "b1_total_ms": b1_shade + b1_isect,
        "b2_total_ms": b2_shade + b2_isect,
        "b3_total_ms": b3_shade,
    }

def main():
    print("=" * 80)
    print("  PATHWAYS ARCHITECTURE COMPARISON: MAIN VS HYBRID PRIMARY TILING")
    print("=" * 80)
    print(f"Main binary:   {BIN_MAIN}")
    print(f"Hybrid binary: {BIN_HYBRID}")
    print()

    all_results = {}

    for sc in SCENES:
        sname = sc["name"]
        print(f"\n>>> Benchmarking Scene: {sname} ({sc['width']}x{sc['height']}, {sc['bounces']} Bounces)")
        all_results[sname] = {}
        common = sc["scene_arg"] + ["--width", str(sc["width"]), "--height", str(sc["height"]), "--max-bounces", str(sc["bounces"])]

        # 1. Monolithic Baseline (1 Tile)
        dump = os.path.join(OUT_DIR, f"{sname.replace(' ', '_')}_baseline.json")
        data = run_benchmark(BIN_HYBRID, common + ["--macro-tiles", "1"], dump)
        base_stats = parse_run_stats(data) if data else None
        all_results[sname]["Baseline (1 Tile)"] = base_stats
        if base_stats:
            print(f"  [Baseline 1 Tile]     : {base_stats['frame_time_ms']:.2f} ms ({base_stats['fps']:.1f} FPS) | Classify: {base_stats['classify_ms']:.2f}ms | B0 Shade: {base_stats['b0_shade_ms']:.2f}ms | B1: {base_stats['b1_total_ms']:.2f}ms | B2: {base_stats['b2_total_ms']:.2f}ms")

        # 2. Main Full-Frame Tiling (2 Tiles & 4 Tiles)
        # 2 tiles on main: macroTile = 2160
        dump = os.path.join(OUT_DIR, f"{sname.replace(' ', '_')}_main_2t.json")
        data = run_benchmark(BIN_MAIN, common + ["--macro-tile", "2160"], dump)
        main_2t = parse_run_stats(data) if data else None
        all_results[sname]["Main (2 Tiles)"] = main_2t
        if main_2t:
            print(f"  [Main 2 Tiles]        : {main_2t['frame_time_ms']:.2f} ms ({main_2t['fps']:.1f} FPS) | Classify: {main_2t['classify_ms']:.2f}ms | B0 Shade: {main_2t['b0_shade_ms']:.2f}ms | B1: {main_2t['b1_total_ms']:.2f}ms | B2: {main_2t['b2_total_ms']:.2f}ms")

        # 4 tiles on main: macroTile = 1920
        dump = os.path.join(OUT_DIR, f"{sname.replace(' ', '_')}_main_4t.json")
        data = run_benchmark(BIN_MAIN, common + ["--macro-tile", "1920"], dump)
        main_4t = parse_run_stats(data) if data else None
        all_results[sname]["Main (4 Tiles)"] = main_4t
        if main_4t:
            print(f"  [Main 4 Tiles]        : {main_4t['frame_time_ms']:.2f} ms ({main_4t['fps']:.1f} FPS) | Classify: {main_4t['classify_ms']:.2f}ms | B0 Shade: {main_4t['b0_shade_ms']:.2f}ms | B1: {main_4t['b1_total_ms']:.2f}ms | B2: {main_4t['b2_total_ms']:.2f}ms")

        # 3. Hybrid Primary Tiling (2 Tiles, 4 Tiles, 8 Tiles)
        dump = os.path.join(OUT_DIR, f"{sname.replace(' ', '_')}_hybrid_2t.json")
        data = run_benchmark(BIN_HYBRID, common + ["--macro-tiles", "2"], dump)
        hyb_2t = parse_run_stats(data) if data else None
        all_results[sname]["Hybrid (2 Tiles)"] = hyb_2t
        if hyb_2t:
            print(f"  [Hybrid 2 Tiles]      : {hyb_2t['frame_time_ms']:.2f} ms ({hyb_2t['fps']:.1f} FPS) | Classify: {hyb_2t['classify_ms']:.2f}ms | B0 Shade: {hyb_2t['b0_shade_ms']:.2f}ms | B1: {hyb_2t['b1_total_ms']:.2f}ms | B2: {hyb_2t['b2_total_ms']:.2f}ms")

        dump = os.path.join(OUT_DIR, f"{sname.replace(' ', '_')}_hybrid_4t.json")
        data = run_benchmark(BIN_HYBRID, common + ["--macro-tiles", "4"], dump)
        hyb_4t = parse_run_stats(data) if data else None
        all_results[sname]["Hybrid (4 Tiles)"] = hyb_4t
        if hyb_4t:
            print(f"  [Hybrid 4 Tiles]      : {hyb_4t['frame_time_ms']:.2f} ms ({hyb_4t['fps']:.1f} FPS) | Classify: {hyb_4t['classify_ms']:.2f}ms | B0 Shade: {hyb_4t['b0_shade_ms']:.2f}ms | B1: {hyb_4t['b1_total_ms']:.2f}ms | B2: {hyb_4t['b2_total_ms']:.2f}ms")

        dump = os.path.join(OUT_DIR, f"{sname.replace(' ', '_')}_hybrid_8t.json")
        data = run_benchmark(BIN_HYBRID, common + ["--macro-tiles", "8"], dump)
        hyb_8t = parse_run_stats(data) if data else None
        all_results[sname]["Hybrid (8 Tiles)"] = hyb_8t
        if hyb_8t:
            print(f"  [Hybrid 8 Tiles]      : {hyb_8t['frame_time_ms']:.2f} ms ({hyb_8t['fps']:.1f} FPS) | Classify: {hyb_8t['classify_ms']:.2f}ms | B0 Shade: {hyb_8t['b0_shade_ms']:.2f}ms | B1: {hyb_8t['b1_total_ms']:.2f}ms | B2: {hyb_8t['b2_total_ms']:.2f}ms")

    # Save summary report
    summary_path = os.path.join(OUT_DIR, "comparison_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nAll benchmark results saved to {summary_path}")

if __name__ == "__main__":
    main()
