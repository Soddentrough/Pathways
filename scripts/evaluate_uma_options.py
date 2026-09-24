#!/usr/bin/env python3
"""
Pathways UMA Architecture & Driver Isolation Evaluation Script
Systematically evaluates architectural and driver options in isolation against baseline
on AMD Strix Halo (Radeon 8060S / gfx1151).
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN_PATH = os.path.join(PROJECT_ROOT, "build", "linux-release", "bin", "pathways")
if not os.path.exists(BIN_PATH):
    BIN_PATH = os.path.join(PROJECT_ROOT, "build", "bin", "pathways")

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "uma_eval")
os.makedirs(OUTPUT_DIR, exist_ok=True)

TARGET_SCENES = [
    {
        "id": "cornell_box",
        "name": "Cornell Box",
        "category": "Baseline Diffuse",
        "args": ["--scene", "procedural:cornell-box"]
    },
    {
        "id": "damaged_helmet",
        "name": "Damaged Helmet",
        "category": "Canonical PBR",
        "args": ["--scene", "scenes/DamagedHelmet.glb"]
    },
    {
        "id": "breakfast_room",
        "name": "Breakfast Room",
        "category": "Heavy Occlusion Architectural GI",
        "args": ["--scene", "scenes/breakfast-room/breakfast_room_extended.glb"]
    },
    {
        "id": "cyber_city",
        "name": "Cyber City",
        "category": "High-Density Multi-BLAS Instancing",
        "args": ["--scene", "cyber-city"]
    },
    {
        "id": "veach_ajar",
        "name": "Veach Ajar",
        "category": "Lighting Variance & Portal GI",
        "args": ["--scene", "scenes/veach-ajar/veach_ajar_extended.glb"]
    }
]

# Each option isolated against baseline
TEST_OPTIONS = [
    {
        "id": "baseline",
        "name": "0. Baseline (Linear Raster, Dual Sort, Inline Shadows, Default RADV)",
        "args": [],
        "env": {}
    },
    {
        "id": "opt1_morton2d",
        "name": "1. Spatial Ray Ordering: Morton 2D Z-Curve (--use-morton)",
        "args": ["--use-morton"],
        "env": {}
    },
    {
        "id": "opt2_archetype_sort",
        "name": "2. Wavefront Material Sorting: Archetype Sort (--wavefront-sort archetype)",
        "args": ["--wavefront-sort", "archetype"],
        "env": {}
    },
    {
        "id": "opt3_directional_sec_sort",
        "name": "3. Directional Secondary Ray Sort (--sec-sort directional)",
        "args": ["--sec-sort", "directional"],
        "env": {}
    },
    {
        "id": "opt4_vram_shadow_queue",
        "name": "4. Shadow Evaluation: VRAM Shadow Queue (--no-inline-shadows)",
        "args": ["--no-inline-shadows"],
        "env": {}
    },
    {
        "id": "opt5_cswave32",
        "name": "5. Wave Scheduling: RADV_PERFTEST=cswave32",
        "args": [],
        "env": {"RADV_PERFTEST": "cswave32"}
    },
    {
        "id": "opt6_rtwave64",
        "name": "6. Wave Scheduling: RADV_PERFTEST=rtwave64",
        "args": [],
        "env": {"RADV_PERFTEST": "rtwave64"}
    },
    {
        "id": "opt7_nogttspill",
        "name": "7. Memory Allocation: RADV_PERFTEST=nogttspill",
        "args": [],
        "env": {"RADV_PERFTEST": "nogttspill"}
    }
]


def run_benchmark(scene, option, width=3840, height=2074, warmup=4, frames=12):
    """Executes a benchmark configuration and extracts telemetry."""
    stats_file = os.path.join(OUTPUT_DIR, f"{scene['id']}_{option['id']}.json")
    if os.path.exists(stats_file):
        os.remove(stats_file)

    cmd = [
        BIN_PATH,
        "--headless",
        "--width", str(width),
        "--height", str(height),
        "--warmup-frames", str(warmup),
        "--frames", str(frames),
        "--no-accumulation",
        "--dump-stats", stats_file
    ] + scene["args"] + option["args"]

    env = os.environ.copy()
    env["DISPLAY"] = ""
    env["WAYLAND_DISPLAY"] = ""
    for k, v in option["env"].items():
        env[k] = v

    t0 = time.time()
    try:
        res = subprocess.run(cmd, env=env, cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=90)
    except subprocess.TimeoutExpired:
        print(f"      [TIMEOUT] {scene['name']} with {option['name']}")
        return None

    elapsed = time.time() - t0
    if res.returncode != 0 or not os.path.exists(stats_file):
        print(f"      [FAILED code {res.returncode}] {scene['name']} with {option['name']}")
        return None

    with open(stats_file) as f:
        data = json.load(f)

    perf = data.get("performance", {})
    bd = perf.get("gpu_profiler_breakdown_ms", {})
    wb = perf.get("wavefront_profiler_breakdown", {})
    bounces = wb.get("bounces", [])

    b0_shade = bounces[0].get("shade_ms", 0.0) if len(bounces) > 0 else 0.0
    b0_isect = bounces[0].get("intersect_ms", 0.0) if len(bounces) > 0 else 0.0
    b1_shade = bounces[1].get("shade_ms", 0.0) if len(bounces) > 1 else 0.0
    b1_isect = bounces[1].get("intersect_ms", 0.0) if len(bounces) > 1 else 0.0

    return {
        "avg_frame_time_ms": perf.get("avg_frame_time_ms", 0.0),
        "min_frame_time_ms": perf.get("min_frame_time_ms", 0.0),
        "max_frame_time_ms": perf.get("max_frame_time_ms", 0.0),
        "fps": perf.get("avg_fps", 0.0),
        "gigarays_per_sec": perf.get("gigarays_per_second", 0.0),
        "primary_gpu_ms": bd.get("primary_gpu_time_ms", 0.0),
        "tonemap_ms": bd.get("tonemap_and_merge_time_ms", 0.0),
        "classify_ms": wb.get("classify_time_ms", 0.0),
        "b0_shade_ms": b0_shade,
        "b0_isect_ms": b0_isect,
        "b1_shade_ms": b1_shade,
        "b1_isect_ms": b1_isect,
        "total_wavefront_ms": wb.get("total_wavefront_time_ms", 0.0),
        "elapsed_sec": round(elapsed, 2)
    }


def main():
    print("=" * 80)
    print("  Pathways: Strix Halo (gfx1151 / Radeon 8060S) UMA Optimization Benchmark")
    print(f"  Target Resolution: 3840x2074 (4K) | Timestamp: {datetime.now().isoformat()}")
    print("=" * 80)

    results = {}

    for scene in TARGET_SCENES:
        print(f"\n>>> Benchmarking Scene: {scene['name']} ({scene['category']})")
        results[scene["id"]] = {}

        baseline_metric = None
        for option in TEST_OPTIONS:
            print(f"    -> Evaluating: {option['name']}...")
            metrics = run_benchmark(scene, option)
            if metrics:
                results[scene["id"]][option["id"]] = metrics
                if option["id"] == "baseline":
                    baseline_metric = metrics["avg_frame_time_ms"]
                    delta_str = "0.00% (Baseline)"
                else:
                    if baseline_metric:
                        delta = ((metrics["avg_frame_time_ms"] - baseline_metric) / baseline_metric) * 100.0
                        delta_str = f"{delta:+.2f}% {'(Faster)' if delta < 0 else '(Slower)'}"
                    else:
                        delta_str = "N/A"

                print(f"       Avg Time: {metrics['avg_frame_time_ms']:.3f} ms | "
                      f"FPS: {metrics['fps']:.1f} | "
                      f"GRays/s: {metrics['gigarays_per_sec']:.2f} | "
                      f"Delta: {delta_str}")

    out_summary_json = os.path.join(OUTPUT_DIR, "uma_isolation_summary.json")
    with open(out_summary_json, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n[+] Saved full summary results to: {out_summary_json}")


if __name__ == "__main__":
    main()
