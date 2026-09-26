#!/usr/bin/env python3
"""
Pathways Comparative Benchmarking Suite
Evaluates performance across all 19 canonical scenes at 1080p and 4K UHD.
Measures:
  - Frame Time (ms) & FPS
  - RT GPU Time (ms)
  - Ray Throughput (GigaRays/sec)
  - Bounce 0 Shade / Shadow / Intersect / Gap3 (ms)
  - BLAS Build Time & Size
  - TLAS Build Time & Instance Count
"""

import sys
import os
import subprocess
import json
import time

CANONICAL_SCENES = [
    {"name": "Cornell Box", "path": "cornell-box"},
    {"name": "Many-Lights (64 Lights)", "path": "many-lights"},
    {"name": "Cyber City", "path": "procedural:cyber-city"},
    {"name": "Infinity Mirror", "path": "infinity-mirror"},
    {"name": "Damaged Helmet", "path": "scenes/DamagedHelmet.glb"},
    {"name": "Dragon Attenuation", "path": "scenes/DragonAttenuation.glb"},
    {"name": "Dragon Dispersion", "path": "scenes/DragonDispersion.glb"},
    {"name": "BMW M6", "path": "scenes/bmw-m6/bmw_m6_extended.glb"},
    {"name": "Breakfast Room", "path": "scenes/breakfast-room/breakfast_room_extended.glb"},
    {"name": "Classroom", "path": "scenes/classroom/classroom_extended.glb"},
    {"name": "Cornell Caustic", "path": "scenes/cornell-caustic/cornell_caustic_extended.glb"},
    {"name": "Glass Of Water", "path": "scenes/glass-of-water/glass_of_water_extended.glb"},
    {"name": "Living Room", "path": "scenes/living-room/living_room_extended.glb"},
    {"name": "Modern Hall", "path": "scenes/modern-hall/modern_hall_extended.glb"},
    {"name": "Veach Ajar", "path": "scenes/veach-ajar/veach_ajar_extended.glb"},
    {"name": "Buick Riviera", "path": "scenes/BuickRiviera/BuickRiviera.usdc"},
    {"name": "Coffee Maker", "path": "scenes/coffee-maker/coffee_maker.usda"},
    {"name": "Kitchen Set", "path": "scenes/Kitchen_set/Kitchen_set.usd"},
    {"name": "Point Instanced Med City", "path": "scenes/PointInstancedMedCity/PointInstancedMedCity.usd"}
]

def run_benchmark(bin_path, width, height, scene_path, extra_args, out_json):
    cmd = [
        bin_path,
        "--headless",
        "--width", str(width),
        "--height", str(height),
        "--spp", "1",
        "--bounces", "8",
        "--frames", "60",
        "--warmup-frames", "15",
        "--scene", scene_path,
        "--dump-stats", out_json
    ] + extra_args

    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0

    if res.returncode != 0:
        print(f"FAILED (code {res.returncode}): {scene_path}")
        print(res.stderr[-500:])
        return None

    if not os.path.exists(out_json):
        print(f"FAILED (no stats json): {scene_path}")
        return None

    with open(out_json, "r") as f:
        data = json.load(f)
    return data

def main():
    bin_path = "./build/linux-release/bin/pathways"
    os.makedirs("output/benchmarks", exist_ok=True)

    target_res = sys.argv[1] if len(sys.argv) > 1 else "1080p"
    mode = sys.argv[2] if len(sys.argv) > 2 else "both" # baseline, optimized, both

    if target_res == "1080p":
        width, height = 1920, 1080
    elif target_res == "4k":
        width, height = 3840, 2160
    else:
        print(f"Unknown resolution: {target_res}")
        return

    orig_telemetry = {}
    if os.path.exists("pathways_telemetry_20260926_111527.json"):
        with open("pathways_telemetry_20260926_111527.json", "r") as f:
            t_data = json.load(f)
            for c in t_data.get("performance", {}).get("configurations_breakdown", []):
                clean_name = c["label"].split("]")[0].replace("[", "").strip().lower()
                orig_telemetry[clean_name] = c

    results = {}

    for idx, sc in enumerate(CANONICAL_SCENES):
        name = sc["name"]
        path = sc["path"]
        print(f"[{idx+1:02d}/{len(CANONICAL_SCENES)}] Benchmarking {name} ({target_res})...", flush=True)

        sc_res = {}

        # 1. Baseline
        if target_res == "4k":
            # Match with original telemetry baseline
            matched = None
            for k, v in orig_telemetry.items():
                if k in name.lower() or name.lower() in k:
                    matched = v
                    break
            if matched:
                sc_res["baseline"] = matched
        else:
            out_base = f"output/benchmarks/{target_res}_base_{idx+1:02d}.json"
            base_data = run_benchmark(bin_path, width, height, path, ["--secondary-sort", "octant"], out_base)
            if base_data:
                sc_res["baseline"] = base_data["performance"]

        # 2. Optimized (Current engine with zero-gap and DirectCoherent defaults)
        out_opt = f"output/benchmarks/{target_res}_opt_{idx+1:02d}.json"
        opt_data = run_benchmark(bin_path, width, height, path, [], out_opt)
        if opt_data:
            sc_res["optimized"] = opt_data["performance"]

        results[name] = sc_res

    combined_out = f"output/benchmarks/benchmark_summary_{target_res}.json"
    with open(combined_out, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAll benchmark results written to: {combined_out}")

if __name__ == "__main__":
    main()
