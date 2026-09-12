#!/usr/bin/env python3
"""
Pathways Deep Profiling & Extreme Cases Regression Testing Suite
Evaluates extreme path tracing workloads across Single-GPU and Dual-GPU modes
on Dual AMD Radeon AI PRO R9700 (RDNA 4, gfx1201).

Profiles:
  1. 4K Native (3840x2160, 1 SPP, 4 Bounces) - Single-GPU Monolithic
  2. 4K Native (3840x2160, 1 SPP, 4 Bounces) - Single-GPU Index Material Sort
  3. 4K Native (3840x2160, 1 SPP, 4 Bounces) - Dual-GPU Checkerboard Tile (50/50)
  4. 1080p (1920x1080, 16 SPP, 4 Bounces)   - Single-GPU Baseline
  5. 1080p (1920x1080, 16 SPP, 4 Bounces)   - Dual-GPU Sample Parallelism
"""

import sys
import os
import subprocess
import json
import time

SCENES = [
    {
        "name": "Cornell Box (Procedural)",
        "category": "Baseline",
        "path": "",
        "desc": "Uniform diffuse geometry; fixed barrier and scheduling overhead."
    },
    {
        "name": "Cornell Caustic",
        "category": "Specular & Transmission",
        "path": "scenes/cornell-caustic/cornell_caustic_extended.glb",
        "desc": "Extreme dielectric vs diffuse divergence, total internal reflection & caustics."
    },
    {
        "name": "Glass of Water",
        "category": "Extreme Refraction",
        "path": "scenes/glass-of-water/glass_of_water_extended.glb",
        "desc": "Curved dielectric transmission & refraction with 406k triangles."
    },
    {
        "name": "Dragon Attenuation",
        "category": "High Poly & Absorption",
        "path": "scenes/DragonAttenuation.glb",
        "desc": "High geometry density with physical Beer-Lambert volumetric absorption."
    },
    {
        "name": "Damaged Helmet",
        "category": "Complex PBR",
        "path": "scenes/DamagedHelmet.glb",
        "desc": "Canonical metallic-roughness PBR model with full normal, AO & emissive maps."
    },
    {
        "name": "Coffee Maker",
        "category": "Multi-Material Stress",
        "path": "scenes/coffee-maker/coffee_maker_extended.glb",
        "desc": "High-frequency glossy conductors, dielectric glass, and diffuse bodies."
    },
    {
        "name": "Classroom",
        "category": "Dense Occlusion & GI",
        "path": "scenes/classroom/classroom_extended.glb",
        "desc": "Complex interior architectural GI with high occlusion and multiple light sources."
    },
    {
        "name": "Kitchen Extended",
        "category": "Heavy Shading (72% RT)",
        "path": "scenes/kitchen/kitchen_extended.glb",
        "desc": "Diverse materials (stainless steel, wood, marble, ceramic) with 1.44M triangles."
    },
    {
        "name": "Living Room",
        "category": "Interior Architectural",
        "path": "scenes/living-room/living_room_extended.glb",
        "desc": "Varied interior fabrics, wood, glass, and multiple emissive fixtures."
    },
    {
        "name": "Veach Ajar",
        "category": "Lighting Variance & MIS",
        "path": "scenes/veach-ajar/veach_ajar_extended.glb",
        "desc": "Extreme lighting variance stress test through a narrow door portal."
    },
    {
        "name": "Bistro Interior",
        "category": "Massive Scale (>1.3M Triangles)",
        "path": "scenes/bistro/bistro_interior.glb",
        "camera_args": ["--camera-pos", "-0.5,2.1,-1.5", "--camera-target", "6.0,1.8,-3.5", "--fov", "70"],
        "desc": "Massive geometry (1.32M tris), 74 materials, 64 lights, and 207 textures."
    }
]

def check_gpu():
    for p in ["/opt/rocm/core-10.0/bin/amd-smi", "/home/naoki/.local/bin/amd-smi"]:
        if os.path.exists(p):
            try:
                res = subprocess.run([p], capture_output=True, text=True, check=False)
                print(res.stdout)
                return
            except Exception as e:
                print(f"Warning querying {p}: {e}")
    print("Warning: amd-smi not available")

def run_bench(scene_path, width, height, spp, mgpu_mode, frames, output_json, sort_mode="none", warmup_frames=20, camera_args=None):
    cmd = [
        "./build/bin/pathways",
        "--headless",
        "--width", str(width),
        "--height", str(height),
        "--spp", str(spp),
        "--max-bounces", "4",
        "--frames", str(frames),
        "--warmup-frames", str(warmup_frames),
        "--no-accumulation",
        "--dump-stats", output_json
    ]
    if scene_path:
        cmd += ["--scene", scene_path]
    if camera_args:
        cmd += camera_args
    if mgpu_mode == "off":
        cmd += ["--mgpu-mode", "off"]
    elif mgpu_mode == "interleaved":
        cmd += ["--mgpu"]
    elif mgpu_mode == "sample":
        cmd += ["--mgpu-mode", "sample"]
    elif mgpu_mode == "tile":
        cmd += ["--mgpu-mode", "tile"]
    if sort_mode:
        cmd += ["--wavefront-sort", sort_mode]

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"  [ERROR] Benchmark failed for {scene_path} ({mgpu_mode}, sort={sort_mode}): {res.stderr[:200]}")
        return None

    if not os.path.exists(output_json):
        print(f"  [ERROR] Output JSON not found: {output_json}")
        return None

    with open(output_json, "r") as f:
        data = json.load(f)
    return data

def main():
    print("========================================================================================")
    print("  Pathways: Extreme Cases & Multi-GPU Performance Regression Battery")
    print("  Target: Dual AMD Radeon AI PRO R9700 (gfx1201 / RDNA 4) on ROCm 10 / Vulkan 1.4")
    print("========================================================================================")
    print("\n[Step 1] Telemetry Baseline: Initial GPU Health & Utilization:")
    check_gpu()

    os.makedirs("output/deep_profile", exist_ok=True)
    results = []

    total_scenes = len(SCENES)
    for idx, scene in enumerate(SCENES, 1):
        sname = scene["name"]
        scat = scene["category"]
        spath = scene["path"]
        sdesc = scene["desc"]
        cam_args = scene.get("camera_args")
        base_id = sname.lower().split()[0].replace("(", "").replace(")", "").replace("/", "_")

        print(f"\n[{idx}/{total_scenes}] Profiling Extreme Case: {sname} [{scat}]")
        print(f"      Description: {sdesc}")
        if spath and not os.path.exists(spath):
            print(f"      [SKIP] Scene file '{spath}' not found.")
            continue

        # Adjust frame counts for heavy scenes (e.g. Bistro) to balance speed and accuracy
        frames_4k = 40 if "bistro" in base_id or "kitchen" in base_id else 60
        warmup_4k = 10 if "bistro" in base_id or "kitchen" in base_id else 15
        frames_1080p = 35 if "bistro" in base_id or "kitchen" in base_id else 50
        warmup_1080p = 10 if "bistro" in base_id or "kitchen" in base_id else 15

        # 1. 4K UHD @ 1 SPP - Single GPU Monolithic (sort: none)
        json_4k_mono = f"output/deep_profile/{base_id}_4k_mono.json"
        data_4k_mono = run_bench(spath, 3840, 2160, 1, "off", frames_4k, json_4k_mono, sort_mode="none", warmup_frames=warmup_4k, camera_args=cam_args)

        # 2. 4K UHD @ 1 SPP - Single GPU Index Material Sort (sort: archetype)
        json_4k_arch = f"output/deep_profile/{base_id}_4k_arch.json"
        data_4k_arch = run_bench(spath, 3840, 2160, 1, "off", frames_4k, json_4k_arch, sort_mode="archetype", warmup_frames=warmup_4k, camera_args=cam_args)

        # 3. 4K UHD @ 1 SPP - Dual GPU Checkerboard Tile (50/50 Balanced)
        json_4k_tile = f"output/deep_profile/{base_id}_4k_tile.json"
        data_4k_tile = run_bench(spath, 3840, 2160, 1, "tile", frames_4k, json_4k_tile, sort_mode="none", warmup_frames=warmup_4k, camera_args=cam_args)

        # 4. 1080p @ 16 SPP - Single GPU Baseline
        json_1080p_single = f"output/deep_profile/{base_id}_1080p_single.json"
        data_1080p_single = run_bench(spath, 1920, 1080, 16, "off", frames_1080p, json_1080p_single, sort_mode="none", warmup_frames=warmup_1080p, camera_args=cam_args)

        # 5. 1080p @ 16 SPP - Dual GPU Sample Parallelism
        json_1080p_sample = f"output/deep_profile/{base_id}_1080p_sample.json"
        data_1080p_sample = run_bench(spath, 1920, 1080, 16, "sample", frames_1080p, json_1080p_sample, sort_mode="none", warmup_frames=warmup_1080p, camera_args=cam_args)

        if not (data_4k_mono and data_4k_tile and data_1080p_single and data_1080p_sample):
            print(f"      [WARNING] Incomplete benchmark telemetry for {sname}.")
            continue

        # Extract Scene Metadata
        sc_info = data_4k_mono.get("engine_settings", {}).get("scene", {})
        tris = sc_info.get("num_triangles", 0)
        mats = sc_info.get("num_materials", 0)
        lights = sc_info.get("num_lights", 0)
        textures = sc_info.get("num_textures", 0)

        # Extract 4K Monolithic Metrics
        perf_4k_mono = data_4k_mono.get("performance", {})
        t_4k_mono = perf_4k_mono.get("avg_frame_time_ms", 0.0)
        fps_4k_mono = perf_4k_mono.get("avg_fps", 0.0)
        grays_4k_mono = perf_4k_mono.get("gigarays_per_second", 0.0)

        # Extract 4K Archetype Sort Metrics
        perf_4k_arch = data_4k_arch.get("performance", {}) if data_4k_arch else {}
        t_4k_arch = perf_4k_arch.get("avg_frame_time_ms", 0.0)
        fps_4k_arch = perf_4k_arch.get("avg_fps", 0.0)
        grays_4k_arch = perf_4k_arch.get("gigarays_per_second", 0.0)
        arch_speedup = (t_4k_mono / t_4k_arch) if (t_4k_arch > 0 and t_4k_mono > 0) else 1.0

        # Extract 4K Multi-GPU Tile Metrics
        perf_4k_tile = data_4k_tile.get("performance", {})
        t_4k_tile = perf_4k_tile.get("avg_frame_time_ms", 0.0)
        fps_4k_tile = perf_4k_tile.get("avg_fps", 0.0)
        grays_4k_tile = perf_4k_tile.get("gigarays_per_second", 0.0)
        speedup_4k = t_4k_mono / t_4k_tile if t_4k_tile > 0 else 0.0

        gpu_breakdown_tile = perf_4k_tile.get("gpu_profiler_breakdown_ms", {})
        prim_ms_tile = gpu_breakdown_tile.get("primary_gpu_time_ms", 0.0)
        sec_ms_tile = gpu_breakdown_tile.get("secondary_gpu_time_ms", 0.0)
        merge_ms_tile = gpu_breakdown_tile.get("tonemap_and_merge_time_ms", 0.0)
        tot_ms_tile = prim_ms_tile + sec_ms_tile
        prim_pct = (prim_ms_tile / tot_ms_tile * 100.0) if tot_ms_tile > 0 else 50.0
        sec_pct = (sec_ms_tile / tot_ms_tile * 100.0) if tot_ms_tile > 0 else 50.0

        # Extract 1080p 16SPP Metrics
        perf_1080_s = data_1080p_single.get("performance", {})
        t_1080_s = perf_1080_s.get("avg_frame_time_ms", 0.0)
        fps_1080_s = perf_1080_s.get("avg_fps", 0.0)
        grays_1080_s = perf_1080_s.get("gigarays_per_second", 0.0)

        perf_1080_m = data_1080p_sample.get("performance", {})
        t_1080_m = perf_1080_m.get("avg_frame_time_ms", 0.0)
        fps_1080_m = perf_1080_m.get("avg_fps", 0.0)
        grays_1080_m = perf_1080_m.get("gigarays_per_second", 0.0)
        speedup_1080 = t_1080_s / t_1080_m if t_1080_m > 0 else 0.0

        scene_result = {
            "name": sname,
            "category": scat,
            "path": spath,
            "description": sdesc,
            "geometry": {
                "triangles": tris,
                "materials": mats,
                "lights": lights,
                "textures": textures
            },
            "4k_single_mono": {
                "frame_time_ms": t_4k_mono,
                "fps": fps_4k_mono,
                "grays_per_sec": grays_4k_mono
            },
            "4k_single_archetype": {
                "frame_time_ms": t_4k_arch,
                "fps": fps_4k_arch,
                "grays_per_sec": grays_4k_arch,
                "speedup_vs_mono": arch_speedup
            },
            "4k_dual_tile": {
                "frame_time_ms": t_4k_tile,
                "fps": fps_4k_tile,
                "grays_per_sec": grays_4k_tile,
                "mgpu_speedup": speedup_4k,
                "primary_gpu_ms": prim_ms_tile,
                "secondary_gpu_ms": sec_ms_tile,
                "primary_gpu_pct": prim_pct,
                "secondary_gpu_pct": sec_pct,
                "merge_tonemap_ms": merge_ms_tile
            },
            "1080p_16spp": {
                "single_gpu_ms": t_1080_s,
                "single_gpu_fps": fps_1080_s,
                "single_gpu_grays": grays_1080_s,
                "dual_gpu_ms": t_1080_m,
                "dual_gpu_fps": fps_1080_m,
                "dual_gpu_grays": grays_1080_m,
                "mgpu_speedup": speedup_1080
            }
        }
        results.append(scene_result)

        print(f"      Geometry: {tris:,} Triangles | {mats} Materials | {lights} Lights | {textures} Textures")
        print(f"      [4K 1 SPP Single-GPU] Monolithic: {t_4k_mono:.2f} ms ({fps_4k_mono:.1f} FPS) | Archetype Sort: {t_4k_arch:.2f} ms ({fps_4k_arch:.1f} FPS) [Rel: {arch_speedup:.2f}x]")
        print(f"      [4K 1 SPP Dual-GPU]   Tile Mode:  {t_4k_tile:.2f} ms ({fps_4k_tile:.1f} FPS) -> Multi-GPU Speedup: {speedup_4k:.2f}x")
        print(f"                            Load Split: GPU 0 = {prim_ms_tile:.2f} ms ({prim_pct:.1f}%) | GPU 1 = {sec_ms_tile:.2f} ms ({sec_pct:.1f}%) [Merge: {merge_ms_tile:.3f} ms]")
        print(f"      [1080p 16 SPP]        Single GPU: {t_1080_s:.2f} ms ({fps_1080_s:.1f} FPS) -> Dual GPU (Sample): {t_1080_m:.2f} ms ({fps_1080_m:.1f} FPS) [{speedup_1080:.2f}x]")

    # Save JSON report
    report_file = "output/deep_profile/deep_profiling_report.json"
    with open(report_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Saved full deep profiling telemetry to: {report_file}")

    # Generate Markdown Summary
    md_file = "output/deep_profile/deep_profiling_summary.md"
    with open(md_file, "w") as f:
        f.write("# Pathways Extreme Scenes Multi-GPU Regression Telemetry Report\n\n")
        f.write("**Hardware**: Dual AMD Radeon AI PRO R9700 (32GB GDDR6 each, gfx1201 / RDNA 4)\n")
        f.write("**CPU**: AMD Ryzen Threadripper 3970X (32C/64T), 64GB RAM\n")
        f.write(f"**Date / Time**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 1. 4K UHD (3840x2160, 1 SPP, 4 Bounces) Benchmark Matrix\n\n")
        f.write("| Scene | Category | Triangles | Single Mono (ms) | Single Arch (ms) | Arch vs Mono | Dual Tile (ms) | Dual FPS | mGPU Speedup | GPU0/1 Split |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            g = r["geometry"]
            s = r["4k_single_mono"]
            a = r["4k_single_archetype"]
            d = r["4k_dual_tile"]
            f.write(f"| **{r['name']}** | {r['category']} | {g['triangles']:,} | {s['frame_time_ms']:.2f} ms | {a['frame_time_ms']:.2f} ms | {a['speedup_vs_mono']:.2f}x | **{d['frame_time_ms']:.2f} ms** | {d['fps']:.1f} | **{d['mgpu_speedup']:.2f}x** | {d['primary_gpu_pct']:.0f}% / {d['secondary_gpu_pct']:.0f}% |\n")

        f.write("\n## 2. 1080p High-Sample (1920x1080, 16 SPP, 4 Bounces) Scaling Battery\n\n")
        f.write("| Scene | Category | Single GPU (ms) | Single FPS | Dual Sample (ms) | Dual FPS | mGPU Scaling Speedup |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            p = r["1080p_16spp"]
            f.write(f"| **{r['name']}** | {r['category']} | {p['single_gpu_ms']:.2f} ms | {p['single_gpu_fps']:.1f} | **{p['dual_gpu_ms']:.2f} ms** | **{p['dual_gpu_fps']:.1f}** | **{p['mgpu_speedup']:.2f}x** |\n")

    print(f"[Done] Saved Markdown summary to: {md_file}")
    print("\n[Step 3] Final Telemetry: Post-Run GPU Health Check:")
    check_gpu()

if __name__ == "__main__":
    main()
