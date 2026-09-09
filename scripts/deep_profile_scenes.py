#!/usr/bin/env python3
"""
Pathways Deep Profiling Suite
Profiles key research scenes across Single-GPU and Dual-GPU (Interleaved Scanlines & Sample Parallelism)
Extracts detailed GPU hardware timings, ray throughput, load balance ratios, and scaling factors.
"""

import sys
import os
import subprocess
import json
import time

SCENES = [
    {
        "name": "Cornell Box (Reference)",
        "path": "scenes/cornell_box.gltf"
    },
    {
        "name": "Classroom (Interior Arch)",
        "path": "scenes/classroom/classroom_extended.glb"
    },
    {
        "name": "Veach MIS (Direct Lighting)",
        "path": "scenes/veach-mis/veach_mis_extended.glb"
    },
    {
        "name": "Dragon (High Polygon Density & Attenuation)",
        "path": "scenes/DragonAttenuation.glb"
    },
    {
        "name": "Living Room (Complex Materials)",
        "path": "scenes/living-room/living_room_extended.glb"
    }
]

def check_gpu():
    amd_smi = "/home/naoki/.local/bin/amd-smi"
    if not os.path.exists(amd_smi):
        amd_smi = "amd-smi"
    try:
        res = subprocess.run([amd_smi], capture_output=True, text=True, check=False)
        print(res.stdout)
    except Exception as e:
        print(f"Warning: could not query amd-smi: {e}")

def run_bench(scene_path, width, height, spp, mgpu_mode, frames, output_json):
    cmd = [
        "./build/bin/pathways",
        "--headless",
        "--scene", scene_path,
        "--width", str(width),
        "--height", str(height),
        "--spp", str(spp),
        "--max-bounces", "4",
        "--frames", str(frames),
        "--warmup-frames", "30",
        "--no-accumulation",
        "--dump-stats", output_json
    ]
    if mgpu_mode == "off":
        cmd += ["--mgpu-mode", "off"]
    elif mgpu_mode == "interleaved":
        cmd += ["--mgpu"]
    elif mgpu_mode == "sample":
        cmd += ["--mgpu-mode", "sample"]
    elif mgpu_mode == "tile":
        cmd += ["--mgpu-mode", "tile"]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"Benchmark failed for {scene_path}: {res.stderr}")
        return None

    if not os.path.exists(output_json):
        print(f"Output JSON not found: {output_json}")
        return None

    with open(output_json, "r") as f:
        data = json.load(f)
    return data

def main():
    print("================================================================================")
    print("  Pathways: Dual AMD Radeon AI PRO R9700 Deep Profiling Benchmark Suite")
    print("================================================================================")
    print("\n[Step 1] Initial GPU Health & Utilization Check:")
    check_gpu()

    os.makedirs("output/deep_profile", exist_ok=True)
    results = []

    for idx, scene in enumerate(SCENES, 1):
        sname = scene["name"]
        spath = scene["path"]
        base_id = sname.lower().split()[0].replace("(", "").replace(")", "")

        print(f"\n[{idx}/{len(SCENES)}] Profiling Scene: {sname} ({spath})")
        if not os.path.exists(spath):
            print(f"  [ERROR] Scene file {spath} does not exist. Skipping.")
            continue

        # 1. 4K UHD @ 1 SPP - Single GPU Baseline
        json_4k_single = f"output/deep_profile/{base_id}_4k_single.json"
        data_4k_single = run_bench(spath, 3840, 2160, 1, "off", 200, json_4k_single)

        # 2. 4K UHD @ 1 SPP - Dual GPU Interleaved Scanlines
        json_4k_interleaved = f"output/deep_profile/{base_id}_4k_interleaved.json"
        data_4k_interleaved = run_bench(spath, 3840, 2160, 1, "interleaved", 200, json_4k_interleaved)

        # 3. 1080p @ 16 SPP - Single GPU Baseline
        json_1080p_single = f"output/deep_profile/{base_id}_1080p_single.json"
        data_1080p_single = run_bench(spath, 1920, 1080, 16, "off", 150, json_1080p_single)

        # 4. 1080p @ 16 SPP - Dual GPU Sample Parallelism
        json_1080p_sample = f"output/deep_profile/{base_id}_1080p_sample.json"
        data_1080p_sample = run_bench(spath, 1920, 1080, 16, "sample", 150, json_1080p_sample)

        if not (data_4k_single and data_4k_interleaved and data_1080p_single and data_1080p_sample):
            print(f"  [WARNING] Incomplete data for {sname}.")
            continue

        # Extract Scene Metadata
        sc_info = data_4k_single.get("engine_settings", {}).get("scene", {})
        tris = sc_info.get("num_triangles", 0)
        mats = sc_info.get("num_materials", 0)
        lights = sc_info.get("num_lights", 0)

        # Extract 4K Metrics
        perf_4k_s = data_4k_single.get("performance", {})
        t_4k_s = perf_4k_s.get("avg_frame_time_ms", 0.0)
        fps_4k_s = perf_4k_s.get("avg_fps", 0.0)
        grays_4k_s = perf_4k_s.get("gigarays_per_second", 0.0)

        perf_4k_i = data_4k_interleaved.get("performance", {})
        t_4k_i = perf_4k_i.get("avg_frame_time_ms", 0.0)
        fps_4k_i = perf_4k_i.get("avg_fps", 0.0)
        grays_4k_i = perf_4k_i.get("gigarays_per_second", 0.0)
        speedup_4k = t_4k_s / t_4k_i if t_4k_i > 0 else 0.0

        gpu_breakdown = perf_4k_i.get("gpu_profiler_breakdown_ms", {})
        prim_ms = gpu_breakdown.get("primary_gpu_time_ms", 0.0)
        sec_ms = gpu_breakdown.get("secondary_gpu_time_ms", 0.0)
        merge_ms = gpu_breakdown.get("tonemap_and_merge_time_ms", 0.0)
        total_gpu_time = prim_ms + sec_ms
        prim_pct = (prim_ms / total_gpu_time * 100.0) if total_gpu_time > 0 else 50.0
        sec_pct = (sec_ms / total_gpu_time * 100.0) if total_gpu_time > 0 else 50.0

        # Extract 1080p 16SPP Metrics
        perf_1080_s = data_1080p_single.get("performance", {})
        t_1080_s = perf_1080_s.get("avg_frame_time_ms", 0.0)
        fps_1080_s = perf_1080_s.get("avg_fps", 0.0)

        perf_1080_m = data_1080p_sample.get("performance", {})
        t_1080_m = perf_1080_m.get("avg_frame_time_ms", 0.0)
        fps_1080_m = perf_1080_m.get("avg_fps", 0.0)
        speedup_1080 = t_1080_s / t_1080_m if t_1080_m > 0 else 0.0

        scene_result = {
            "name": sname,
            "path": spath,
            "triangles": tris,
            "materials": mats,
            "lights": lights,
            "4k_1spp": {
                "single_gpu_ms": t_4k_s,
                "single_gpu_fps": fps_4k_s,
                "single_gpu_grays": grays_4k_s,
                "dual_gpu_ms": t_4k_i,
                "dual_gpu_fps": fps_4k_i,
                "dual_gpu_grays": grays_4k_i,
                "speedup": speedup_4k,
                "primary_gpu_ms": prim_ms,
                "secondary_gpu_ms": sec_ms,
                "primary_gpu_pct": prim_pct,
                "secondary_gpu_pct": sec_pct,
                "merge_tonemap_ms": merge_ms
            },
            "1080p_16spp": {
                "single_gpu_ms": t_1080_s,
                "single_gpu_fps": fps_1080_s,
                "dual_gpu_ms": t_1080_m,
                "dual_gpu_fps": fps_1080_m,
                "speedup": speedup_1080
            }
        }
        results.append(scene_result)

        print(f"  Geometry: {tris:,} Triangles | {mats} Materials | {lights} Lights")
        print(f"  4K UHD 1 SPP:")
        print(f"    Single GPU: {t_4k_s:.2f} ms ({fps_4k_s:.1f} FPS) | {grays_4k_s:.2f} G-rays/s")
        print(f"    Dual GPU:   {t_4k_i:.2f} ms ({fps_4k_i:.1f} FPS) | {grays_4k_i:.2f} G-rays/s -> {speedup_4k:.2f}x Speedup")
        print(f"    Load Balance: GPU 0 = {prim_ms:.2f} ms ({prim_pct:.1f}%) vs GPU 1 = {sec_ms:.2f} ms ({sec_pct:.1f}%) [Merge: {merge_ms:.3f} ms]")
        print(f"  1080p 16 SPP:")
        print(f"    Single GPU: {t_1080_s:.2f} ms ({fps_1080_s:.1f} FPS)")
        print(f"    Dual GPU:   {t_1080_m:.2f} ms ({fps_1080_m:.1f} FPS) -> {speedup_1080:.2f}x Speedup")

    # Save complete JSON report
    report_file = "output/deep_profile/deep_profiling_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Saved full deep profiling results to: {report_file}")

if __name__ == "__main__":
    main()
