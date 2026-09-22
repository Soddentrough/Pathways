#!/usr/bin/env python3
"""
Pathways 4K Multi-Scene Deep Profiling & Telemetry Benchmark Suite
Evaluates extreme path tracing workloads across Single-GPU and Dual-GPU modes
on Dual AMD Radeon AI PRO R9700 (RDNA 4, gfx1201) under Vulkan 1.4.

Execution Modes (Native 4K UHD 3840x2160, 1 SPP, 4 Bounces):
  1. Single-GPU Monolithic         (--wavefront-sort none --mgpu-mode off)
  2. Single-GPU DGC Material Sort  (--wavefront-sort archetype --mgpu-mode off)
  3. Dual-GPU Tile Parallelism     (--mgpu-mode tile --tile-size 64 --wavefront-sort archetype)
  4. Dual-GPU Sample Parallelism   (--mgpu-mode sample --wavefront-sort archetype)

Hardware Telemetry:
  Real-time GPU socket power (W), core clocks (MHz), clock stability, and temperatures
  sampled via `/opt/rocm/core-10.0/bin/amd-smi metric -p -c -t --json`.
"""

import sys
import os
import subprocess
import json
import time
import math
import argparse
import threading

AMD_SMI_PATHS = [
    "/opt/rocm/core-10.0/bin/amd-smi",
    "/home/naoki/.local/bin/amd-smi"
]

SCENES = [
    {
        "name": "Cornell Box",
        "category": "Baseline Diffuse",
        "path": "procedural:cornell-box",
        "desc": "Uniform diffuse geometry; baseline barrier and scheduling overhead."
    },
    {
        "name": "Cornell Caustic",
        "category": "Specular & Transmission",
        "path": "scenes/cornell-caustic/cornell_caustic_extended.glb",
        "desc": "Extreme dielectric vs diffuse divergence, total internal reflection & caustics."
    },
    {
        "name": "Glass of Water",
        "category": "Extreme Refraction & Dielectric",
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
        "name": "Dragon Dispersion",
        "category": "Dielectric Dispersion & Transmission",
        "path": "scenes/DragonDispersion.glb",
        "desc": "Dense dragon model testing KHR_materials_dispersion and dielectric transmission."
    },
    {
        "name": "BMW M6",
        "category": "Automotive PBR & Clearcoat",
        "path": "scenes/bmw-m6/bmw_m6_extended.glb",
        "desc": "Complex automotive model showcasing clearcoat, chrome conductors, and detailed materials."
    },
    {
        "name": "Breakfast Room",
        "category": "Interior Architectural GI",
        "path": "scenes/breakfast-room/breakfast_room_extended.glb",
        "desc": "Complex interior architectural GI with high multi-bounce occlusion and subtle daylighting."
    },
    {
        "name": "Damaged Helmet",
        "category": "Canonical Metallic-Roughness PBR",
        "path": "scenes/DamagedHelmet.glb",
        "desc": "Canonical metallic-roughness PBR model with full normal, AO & emissive maps."
    },
    {
        "name": "Coffee Maker",
        "category": "Multi-Material Stress (OpenUSD)",
        "path": "scenes/coffee-maker/coffee_maker.usda",
        "desc": "High-frequency glossy conductors, dielectric glass, and diffuse bodies in native OpenUSD."
    },
    {
        "name": "Classroom",
        "category": "Dense Occlusion & Complex GI",
        "path": "scenes/classroom/classroom_extended.glb",
        "desc": "Complex interior architectural GI with high occlusion and multiple light sources."
    },
    {
        "name": "Kitchen Extended",
        "category": "Heavy Shading & Complex Architecture",
        "path": "scenes/Kitchen_set/Kitchen_set.usd",
        "desc": "Diverse materials (stainless steel, wood, marble, ceramic) with 537k triangles via OpenUSD."
    },
    {
        "name": "Living Room",
        "category": "Interior Architectural Divergence",
        "path": "scenes/living-room/living_room_extended.glb",
        "desc": "Varied interior fabrics, wood, glass, and multiple emissive fixtures."
    },
    {
        "name": "Veach Ajar",
        "category": "Lighting Variance & MIS Portal",
        "path": "scenes/veach-ajar/veach_ajar_extended.glb",
        "desc": "Extreme lighting variance stress test through a narrow door portal."
    },
    {
        "name": "Bistro Interior",
        "category": "Large-Scale (>1.3M Triangles)",
        "path": "scenes/bistro/bistro_interior.glb",
        "camera_args": ["--camera-pos", "3.5,1.75,-6.2", "--camera-target", "9.5,1.65,0.5", "--fov", "70"],
        "desc": "Complex geometry (1.32M tris), 74 materials, 64 lights, and 207 textures."
    },
    {
        "name": "Cyber City",
        "category": "High-Density Multi-BLAS Instancing",
        "path": "cyber-city",
        "desc": "Complex procedural sci-fi megastructure with 4,000 TLAS instances, 256 lights, and diverse materials."
    }
]


def find_amd_smi():
    for p in AMD_SMI_PATHS:
        if os.path.exists(p):
            return p
    return None


def check_gpu():
    smi = find_amd_smi()
    if smi:
        try:
            res = subprocess.run([smi], capture_output=True, text=True, check=False)
            print(res.stdout)
            return
        except Exception as e:
            print(f"Warning querying {smi}: {e}")
    print("Warning: amd-smi not available")


class AmdSmiMonitor:
    """
    Background sampler for live AMD-SMI hardware telemetry during benchmark runs.
    Extracts socket power (W), core clock (MHz), clock stability, and temperatures.
    """
    def __init__(self, smi_path=None, interval_sec=0.2):
        self.smi_path = smi_path or find_amd_smi()
        self.interval_sec = interval_sec
        self.running = False
        self.thread = None
        self.samples = []

    def _sample(self):
        if not self.smi_path:
            return None
        try:
            res = subprocess.run(
                [self.smi_path, "metric", "-p", "-c", "-t", "--json"],
                capture_output=True,
                text=True,
                check=False
            )
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                gpu_data = data.get("gpu_data", [])
                sample = {"timestamp": time.time(), "gpus": {}}

                def safe_float(v):
                    try:
                        return float(v)
                    except (ValueError, TypeError):
                        return None

                for entry in gpu_data:
                    gpu_id = entry.get("gpu")
                    power_val = entry.get("power", {}).get("socket_power", {}).get("value")
                    clock_val = entry.get("clock", {}).get("gfx_0", {}).get("clk", {}).get("value")
                    temp_edge = entry.get("temperature", {}).get("edge", {}).get("value")
                    temp_hotspot = entry.get("temperature", {}).get("hotspot", {}).get("value")
                    temp_mem = entry.get("temperature", {}).get("mem", {}).get("value")

                    sample["gpus"][gpu_id] = {
                        "power_w": safe_float(power_val),
                        "clock_mhz": safe_float(clock_val),
                        "temp_edge_c": safe_float(temp_edge),
                        "temp_hotspot_c": safe_float(temp_hotspot),
                        "temp_mem_c": safe_float(temp_mem)
                    }
                return sample
        except Exception:
            pass
        return None

    def _worker(self):
        while self.running:
            s = self._sample()
            if s:
                self.samples.append(s)
            time.sleep(self.interval_sec)

    def start(self):
        self.samples = []
        first_s = self._sample()
        if first_s:
            self.samples.append(first_s)
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        last_s = self._sample()
        if last_s:
            self.samples.append(last_s)

    def compute_stats(self, fallback_engine_telemetry=None):
        stats = {}
        for gpu_id in [0, 1]:
            powers = [s["gpus"][gpu_id]["power_w"] for s in self.samples if gpu_id in s["gpus"] and s["gpus"][gpu_id]["power_w"] is not None]
            clocks = [s["gpus"][gpu_id]["clock_mhz"] for s in self.samples if gpu_id in s["gpus"] and s["gpus"][gpu_id]["clock_mhz"] is not None]
            edges = [s["gpus"][gpu_id]["temp_edge_c"] for s in self.samples if gpu_id in s["gpus"] and s["gpus"][gpu_id]["temp_edge_c"] is not None]
            hotspots = [s["gpus"][gpu_id]["temp_hotspot_c"] for s in self.samples if gpu_id in s["gpus"] and s["gpus"][gpu_id]["temp_hotspot_c"] is not None]
            mems = [s["gpus"][gpu_id]["temp_mem_c"] for s in self.samples if gpu_id in s["gpus"] and s["gpus"][gpu_id]["temp_mem_c"] is not None]

            # Filter out idle clocks (<300 MHz) if active clocks exist
            active_clocks = [c for c in clocks if c >= 300.0]
            eval_clocks = active_clocks if active_clocks else clocks

            avg_power = sum(powers) / len(powers) if powers else 0.0
            max_power = max(powers) if powers else 0.0
            min_power = min(powers) if powers else 0.0

            avg_clock = sum(eval_clocks) / len(eval_clocks) if eval_clocks else 0.0
            max_clock = max(eval_clocks) if eval_clocks else 0.0
            min_clock = min(eval_clocks) if eval_clocks else 0.0

            if len(eval_clocks) > 1 and avg_clock > 0:
                variance = sum((c - avg_clock) ** 2 for c in eval_clocks) / len(eval_clocks)
                stddev_clock = math.sqrt(variance)
                stability_pct = max(0.0, min(100.0, (1.0 - (stddev_clock / avg_clock)) * 100.0))
            else:
                stddev_clock = 0.0
                stability_pct = 100.0 if avg_clock > 0 else 0.0

            avg_edge = sum(edges) / len(edges) if edges else 0.0
            max_edge = max(edges) if edges else 0.0
            avg_hotspot = sum(hotspots) / len(hotspots) if hotspots else 0.0
            max_hotspot = max(hotspots) if hotspots else 0.0
            avg_mem = sum(mems) / len(mems) if mems else 0.0

            # Fallback to engine telemetry if amd-smi values were not captured
            if avg_clock == 0.0 and fallback_engine_telemetry:
                gpu_key = "primary_gpu" if gpu_id == 0 else "secondary_gpu"
                fb = fallback_engine_telemetry.get(gpu_key, {}).get("telemetry", {})
                avg_clock = float(fb.get("clock_mhz", 0.0))
                max_clock = avg_clock
                min_clock = avg_clock
                avg_edge = float(fb.get("temperature_c", 0.0))
                max_edge = avg_edge
                stability_pct = 100.0 if avg_clock > 0 else 0.0

            stats[f"gpu{gpu_id}"] = {
                "socket_power_w": round(avg_power, 1),
                "peak_power_w": round(max_power, 1),
                "min_power_w": round(min_power, 1),
                "core_clock_mhz": round(avg_clock, 1),
                "peak_clock_mhz": round(max_clock, 1),
                "min_clock_mhz": round(min_clock, 1),
                "clock_stddev_mhz": round(stddev_clock, 2),
                "clock_stability_pct": round(stability_pct, 1),
                "temp_edge_c": round(avg_edge, 1),
                "max_temp_edge_c": round(max_edge, 1),
                "temp_hotspot_c": round(avg_hotspot, 1),
                "max_temp_hotspot_c": round(max_hotspot, 1),
                "temp_mem_c": round(avg_mem, 1),
                "samples_count": len(powers)
            }
        return stats


def is_scene_available(path):
    if not path or path.startswith("procedural:") or path in ["cyber-city", "cornell-box"]:
        return True
    return os.path.exists(path)


def run_bench(scene_path, width, height, spp, mgpu_mode, frames, output_json,
              sort_mode="archetype", warmup_frames=15, camera_args=None, tile_size=64):
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
    elif mgpu_mode == "tile":
        cmd += ["--mgpu-mode", "tile", "--tile-size", str(tile_size)]
    elif mgpu_mode == "sample":
        cmd += ["--mgpu-mode", "sample"]
    if sort_mode:
        cmd += ["--wavefront-sort", sort_mode]

    monitor = AmdSmiMonitor(interval_sec=0.2)
    monitor.start()

    env = os.environ.copy()
    env["RADV_PROFILE_PSTATE"] = "peak"
    env["RADV_PERFTEST"] = "nogttspill"
    res = subprocess.run(cmd, env=env, capture_output=True, text=True)

    monitor.stop()

    if res.returncode != 0:
        print(f"  [ERROR] Benchmark failed for {scene_path} ({mgpu_mode}, sort={sort_mode}): {res.stderr[:200]}")
        return None, None

    if not os.path.exists(output_json):
        print(f"  [ERROR] Output JSON not found: {output_json}")
        return None, None

    with open(output_json, "r") as f:
        data = json.load(f)

    telemetry_stats = monitor.compute_stats(fallback_engine_telemetry=data)
    return data, telemetry_stats


def main():
    parser = argparse.ArgumentParser(description="Pathways 4K Multi-Scene Deep Profiling & Telemetry Benchmark Suite")
    parser.add_argument("--output", default="output/deep_profile/deep_profiling_report.json",
                        help="Path to output JSON report (default: output/deep_profile/deep_profiling_report.json)")
    parser.add_argument("--summary", default="output/deep_profile/deep_profiling_summary.md",
                        help="Path to output Markdown summary (default: output/deep_profile/deep_profiling_summary.md)")
    parser.add_argument("--frames", type=int, default=60,
                        help="Frame count for standard scenes (default: 60)")
    parser.add_argument("--warmup-frames", type=int, default=15,
                        help="Warmup frame count for standard scenes (default: 15)")
    parser.add_argument("--heavy-frames", type=int, default=40,
                        help="Frame count for heavy scenes (default: 40)")
    parser.add_argument("--heavy-warmup", type=int, default=10,
                        help="Warmup frame count for heavy scenes (default: 10)")
    parser.add_argument("--scene", type=str, default="",
                        help="Filter to scenes matching substring (optional)")
    parser.add_argument("--skip-gpu-check", action="store_true",
                        help="Skip initial amd-smi health check printout")

    args = parser.parse_args()

    print("========================================================================================")
    print("  Pathways: Native 4K Multi-Scene Benchmark & Telemetry Matrix (Milestone 1 / R1)")
    print("  Target: Dual AMD Radeon AI PRO R9700 (gfx1201 / RDNA 4) on Vulkan 1.4 / RADV")
    print("========================================================================================")
    
    if not args.skip_gpu_check:
        print("\n[Step 1] Initial GPU Health & Utilization Baseline:")
        check_gpu()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    results = []

    active_scenes = [s for s in SCENES if not args.scene or args.scene.lower() in s["name"].lower()]
    total_scenes = len(active_scenes)

    print(f"\n[Step 2] Executing 4K Multi-Scene Benchmark Matrix ({total_scenes} Scenes x 4 Modes):")

    for idx, scene in enumerate(active_scenes, 1):
        sname = scene["name"]
        scat = scene["category"]
        spath = scene["path"]
        sdesc = scene["desc"]
        cam_args = scene.get("camera_args")
        base_id = sname.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")

        print(f"\n[{idx}/{total_scenes}] Profiling Scene: {sname} [{scat}]")
        print(f"      Description: {sdesc}")

        if not is_scene_available(spath):
            print(f"      [SKIP] Scene file '{spath}' not found.")
            continue

        is_heavy = ("bistro" in base_id) or ("cyber_city" in base_id) or ("kitchen" in base_id)
        frames_4k = args.heavy_frames if is_heavy else args.frames
        warmup_4k = args.heavy_warmup if is_heavy else args.warmup_frames

        # Mode 1: Single-GPU Monolithic
        print("      -> Mode 1/4: Single-GPU Monolithic (4K UHD 1 SPP)...", end="", flush=True)
        json_mono = f"output/deep_profile/{base_id}_4k_mono.json"
        data_mono, telem_mono = run_bench(
            spath, 3840, 2160, 1, "off", frames_4k, json_mono,
            sort_mode="none", warmup_frames=warmup_4k, camera_args=cam_args
        )
        if data_mono:
            t_m = data_mono.get("performance", {}).get("avg_frame_time_ms", 0.0)
            fps_m = data_mono.get("performance", {}).get("avg_fps", 0.0)
            p_w = telem_mono["gpu0"]["socket_power_w"]
            print(f" Done ({t_m:.2f} ms, {fps_m:.1f} FPS, {p_w} W)")
        else:
            print(" FAILED")

        # Mode 2: Single-GPU DGC Dual Sorting
        print("      -> Mode 2/4: Single-GPU DGC Dual Sorting (4K UHD 1 SPP)...", end="", flush=True)
        json_arch = f"output/deep_profile/{base_id}_4k_arch.json"
        data_arch, telem_arch = run_bench(
            spath, 3840, 2160, 1, "off", frames_4k, json_arch,
            sort_mode="dual", warmup_frames=warmup_4k, camera_args=cam_args
        )
        if data_arch:
            t_a = data_arch.get("performance", {}).get("avg_frame_time_ms", 0.0)
            fps_a = data_arch.get("performance", {}).get("avg_fps", 0.0)
            p_w = telem_arch["gpu0"]["socket_power_w"]
            print(f" Done ({t_a:.2f} ms, {fps_a:.1f} FPS, {p_w} W)")
        else:
            print(" FAILED")

        # Mode 3: Dual-GPU Tile Parallelism
        print("      -> Mode 3/4: Dual-GPU Tile Parallelism (4K UHD 1 SPP)...", end="", flush=True)
        json_tile = f"output/deep_profile/{base_id}_4k_tile.json"
        data_tile, telem_tile = run_bench(
            spath, 3840, 2160, 1, "tile", frames_4k, json_tile,
            sort_mode="dual", warmup_frames=warmup_4k, camera_args=cam_args, tile_size=64
        )
        if data_tile:
            t_t = data_tile.get("performance", {}).get("avg_frame_time_ms", 0.0)
            fps_t = data_tile.get("performance", {}).get("avg_fps", 0.0)
            p0 = telem_tile["gpu0"]["socket_power_w"]
            p1 = telem_tile["gpu1"]["socket_power_w"]
            print(f" Done ({t_t:.2f} ms, {fps_t:.1f} FPS, GPU0: {p0}W, GPU1: {p1}W)")
        else:
            print(" FAILED")

        # Mode 4: Dual-GPU Sample Parallelism
        print("      -> Mode 4/4: Dual-GPU Sample Parallelism (4K UHD 1 SPP)...", end="", flush=True)
        json_sample = f"output/deep_profile/{base_id}_4k_sample.json"
        data_sample, telem_sample = run_bench(
            spath, 3840, 2160, 1, "sample", frames_4k, json_sample,
            sort_mode="dual", warmup_frames=warmup_4k, camera_args=cam_args
        )
        if data_sample:
            t_s = data_sample.get("performance", {}).get("avg_frame_time_ms", 0.0)
            fps_s = data_sample.get("performance", {}).get("avg_fps", 0.0)
            p0 = telem_sample["gpu0"]["socket_power_w"]
            p1 = telem_sample["gpu1"]["socket_power_w"]
            print(f" Done ({t_s:.2f} ms, {fps_s:.1f} FPS, GPU0: {p0}W, GPU1: {p1}W)")
        else:
            print(" FAILED")

        if not (data_mono and data_arch and data_tile and data_sample):
            print(f"      [WARNING] Incomplete benchmark telemetry for {sname}.")
            continue

        # Extract Scene Metadata
        sc_info = data_mono.get("engine_settings", {}).get("scene", {})
        tris = sc_info.get("num_triangles", 0)
        inst_tris = sc_info.get("num_instanced_triangles", tris)
        mats = sc_info.get("num_materials", 0)
        lights = sc_info.get("num_lights", 0)
        textures = sc_info.get("num_textures", 0)

        # Mode 1 Metrics: Single-GPU Monolithic
        perf_m = data_mono.get("performance", {})
        t_mono = perf_m.get("avg_frame_time_ms", 0.0)
        fps_mono = perf_m.get("avg_fps", 0.0)
        grays_mono = perf_m.get("gigarays_per_second", 0.0)
        gpu_m = perf_m.get("gpu_profiler_breakdown_ms", {})
        prim_ms_mono = gpu_m.get("primary_gpu_time_ms", 0.0)
        tonemap_ms_mono = gpu_m.get("tonemap_and_merge_time_ms", 0.0)

        # Mode 2 Metrics: Single-GPU Archetype DGC
        perf_a = data_arch.get("performance", {})
        t_arch = perf_a.get("avg_frame_time_ms", 0.0)
        fps_arch = perf_a.get("avg_fps", 0.0)
        grays_arch = perf_a.get("gigarays_per_second", 0.0)
        gpu_a = perf_a.get("gpu_profiler_breakdown_ms", {})
        prim_ms_arch = gpu_a.get("primary_gpu_time_ms", 0.0)
        tonemap_ms_arch = gpu_a.get("tonemap_and_merge_time_ms", 0.0)
        arch_speedup = round((t_mono / t_arch), 3) if (t_arch > 0 and t_mono > 0) else 1.0

        # Mode 3 Metrics: Dual-GPU Tile Parallelism
        perf_t = data_tile.get("performance", {})
        t_tile = perf_t.get("avg_frame_time_ms", 0.0)
        fps_tile = perf_t.get("avg_fps", 0.0)
        grays_tile = perf_t.get("gigarays_per_second", 0.0)
        gpu_t = perf_t.get("gpu_profiler_breakdown_ms", {})
        prim_ms_tile = gpu_t.get("primary_gpu_time_ms", 0.0)
        sec_ms_tile = gpu_t.get("secondary_gpu_time_ms", 0.0)
        merge_ms_tile = gpu_t.get("tonemap_and_merge_time_ms", 0.0)
        tot_ms_tile = prim_ms_tile + sec_ms_tile
        prim_pct_tile = round((prim_ms_tile / tot_ms_tile * 100.0), 1) if tot_ms_tile > 0 else 50.0
        sec_pct_tile = round((sec_ms_tile / tot_ms_tile * 100.0), 1) if tot_ms_tile > 0 else 50.0
        speedup_tile = round(t_mono / t_tile, 3) if t_tile > 0 else 0.0
        tile_efficiency = round((speedup_tile / 2.0) * 100.0, 1)

        # Mode 4 Metrics: Dual-GPU Sample Parallelism
        perf_s = data_sample.get("performance", {})
        t_sample = perf_s.get("avg_frame_time_ms", 0.0)
        fps_sample = perf_s.get("avg_fps", 0.0)
        grays_sample = perf_s.get("gigarays_per_second", 0.0)
        gpu_s = perf_s.get("gpu_profiler_breakdown_ms", {})
        prim_ms_sample = gpu_s.get("primary_gpu_time_ms", 0.0)
        sec_ms_sample = gpu_s.get("secondary_gpu_time_ms", 0.0)
        merge_ms_sample = gpu_s.get("tonemap_and_merge_time_ms", 0.0)
        tot_ms_sample = prim_ms_sample + sec_ms_sample
        prim_pct_sample = round((prim_ms_sample / tot_ms_sample * 100.0), 1) if tot_ms_sample > 0 else 50.0
        sec_pct_sample = round((sec_ms_sample / tot_ms_sample * 100.0), 1) if tot_ms_sample > 0 else 50.0
        speedup_sample = round(t_mono / t_sample, 3) if t_sample > 0 else 0.0
        sample_efficiency = round((speedup_sample / 2.0) * 100.0, 1)

        scene_result = {
            "name": sname,
            "category": scat,
            "path": spath,
            "description": sdesc,
            "geometry": {
                "triangles": tris,
                "instanced_triangles": inst_tris,
                "materials": mats,
                "lights": lights,
                "textures": textures
            },
            "4k_single_mono": {
                "frame_time_ms": t_mono,
                "fps": fps_mono,
                "primary_gpu_ms": prim_ms_mono,
                "tonemap_ms": tonemap_ms_mono,
                "grays_per_sec": grays_mono,
                "power_watts": telem_mono["gpu0"]["socket_power_w"],
                "clock_mhz": telem_mono["gpu0"]["core_clock_mhz"],
                "clock_stability_pct": telem_mono["gpu0"]["clock_stability_pct"],
                "temperature_c": telem_mono["gpu0"]["temp_edge_c"],
                "telemetry": telem_mono
            },
            "4k_single_archetype": {
                "frame_time_ms": t_arch,
                "fps": fps_arch,
                "primary_gpu_ms": prim_ms_arch,
                "tonemap_ms": tonemap_ms_arch,
                "grays_per_sec": grays_arch,
                "speedup_vs_mono": arch_speedup,
                "power_watts": telem_arch["gpu0"]["socket_power_w"],
                "clock_mhz": telem_arch["gpu0"]["core_clock_mhz"],
                "clock_stability_pct": telem_arch["gpu0"]["clock_stability_pct"],
                "temperature_c": telem_arch["gpu0"]["temp_edge_c"],
                "telemetry": telem_arch
            },
            "4k_dual_tile": {
                "frame_time_ms": t_tile,
                "fps": fps_tile,
                "primary_gpu_ms": prim_ms_tile,
                "secondary_gpu_ms": sec_ms_tile,
                "primary_gpu_pct": prim_pct_tile,
                "secondary_gpu_pct": sec_pct_tile,
                "merge_tonemap_ms": merge_ms_tile,
                "grays_per_sec": grays_tile,
                "mgpu_speedup": speedup_tile,
                "scaling_efficiency_pct": tile_efficiency,
                "power_watts_gpu0": telem_tile["gpu0"]["socket_power_w"],
                "power_watts_gpu1": telem_tile["gpu1"]["socket_power_w"],
                "clock_mhz_gpu0": telem_tile["gpu0"]["core_clock_mhz"],
                "clock_mhz_gpu1": telem_tile["gpu1"]["core_clock_mhz"],
                "clock_stability_pct_gpu0": telem_tile["gpu0"]["clock_stability_pct"],
                "clock_stability_pct_gpu1": telem_tile["gpu1"]["clock_stability_pct"],
                "temperature_c_gpu0": telem_tile["gpu0"]["temp_edge_c"],
                "temperature_c_gpu1": telem_tile["gpu1"]["temp_edge_c"],
                "telemetry": telem_tile
            },
            "4k_dual_sample": {
                "frame_time_ms": t_sample,
                "fps": fps_sample,
                "primary_gpu_ms": prim_ms_sample,
                "secondary_gpu_ms": sec_ms_sample,
                "primary_gpu_pct": prim_pct_sample,
                "secondary_gpu_pct": sec_pct_sample,
                "merge_tonemap_ms": merge_ms_sample,
                "grays_per_sec": grays_sample,
                "mgpu_speedup": speedup_sample,
                "scaling_efficiency_pct": sample_efficiency,
                "power_watts_gpu0": telem_sample["gpu0"]["socket_power_w"],
                "power_watts_gpu1": telem_sample["gpu1"]["socket_power_w"],
                "clock_mhz_gpu0": telem_sample["gpu0"]["core_clock_mhz"],
                "clock_mhz_gpu1": telem_sample["gpu1"]["core_clock_mhz"],
                "clock_stability_pct_gpu0": telem_sample["gpu0"]["clock_stability_pct"],
                "clock_stability_pct_gpu1": telem_sample["gpu1"]["clock_stability_pct"],
                "temperature_c_gpu0": telem_sample["gpu0"]["temp_edge_c"],
                "temperature_c_gpu1": telem_sample["gpu1"]["temp_edge_c"],
                "telemetry": telem_sample
            }
        }
        results.append(scene_result)

        print(f"      [Summary] Mono: {t_mono:.2f} ms | Arch: {t_arch:.2f} ms ({arch_speedup:.2f}x) | Tile: {t_tile:.2f} ms ({speedup_tile:.2f}x) | Sample: {t_sample:.2f} ms ({speedup_sample:.2f}x)")

    # Save JSON report
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Done] Saved full deep profiling telemetry to: {args.output}")

    # Generate Markdown Summary
    with open(args.summary, "w") as f:
        f.write("# Pathways 4K Native Multi-Scene Performance & Telemetry Matrix\n\n")
        f.write("**Hardware**: Dual AMD Radeon AI PRO R9700 (32GB GDDR6 each, gfx1201 / RDNA 4)\n")
        f.write("**Platform**: AMD Ryzen Threadripper 3970X (32C/64T), 64GB RAM, Fedora Linux 44, Vulkan 1.4 / RADV\n")
        f.write(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## 1. 4K Native (3840x2160, 1 SPP, 4 Bounces) Execution Matrix\n\n")
        f.write("| Scene | Category | Triangles | Single Mono (ms) | Single Arch (ms) | Arch Rel | Dual Tile (ms) | Tile Speedup | Dual Sample (ms) | Sample Speedup |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            g = r["geometry"]
            m = r["4k_single_mono"]
            a = r["4k_single_archetype"]
            t = r["4k_dual_tile"]
            s = r["4k_dual_sample"]
            f.write(f"| **{r['name']}** | {r['category']} | {g['triangles']:,} | {m['frame_time_ms']:.2f} ms | {a['frame_time_ms']:.2f} ms | {a['speedup_vs_mono']:.2f}x | **{t['frame_time_ms']:.2f} ms** | **{t['mgpu_speedup']:.2f}x** | **{s['frame_time_ms']:.2f} ms** | **{s['mgpu_speedup']:.2f}x** |\n")

        f.write("\n## 2. Hardware Telemetry & Power Matrix (AMD-SMI / RDNA 4 gfx1201)\n\n")
        f.write("| Scene | Mode | GPU 0 Power (W) | GPU 1 Power (W) | GPU 0 Clock (MHz) | GPU 1 Clock (MHz) | Stability (%) | GPU 0 Temp (°C) | GPU 1 Temp (°C) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            m = r["4k_single_mono"]
            a = r["4k_single_archetype"]
            t = r["4k_dual_tile"]
            s = r["4k_dual_sample"]
            f.write(f"| {r['name']} | Mono | {m['power_watts']:.1f} W | 0.0 W | {m['clock_mhz']:.0f} MHz | - | {m['clock_stability_pct']:.1f}% | {m['temperature_c']:.1f}°C | - |\n")
            f.write(f"| {r['name']} | Arch | {a['power_watts']:.1f} W | 0.0 W | {a['clock_mhz']:.0f} MHz | - | {a['clock_stability_pct']:.1f}% | {a['temperature_c']:.1f}°C | - |\n")
            f.write(f"| {r['name']} | Tile | {t['power_watts_gpu0']:.1f} W | {t['power_watts_gpu1']:.1f} W | {t['clock_mhz_gpu0']:.0f} MHz | {t['clock_mhz_gpu1']:.0f} MHz | {t['clock_stability_pct_gpu0']:.1f}% / {t['clock_stability_pct_gpu1']:.1f}% | {t['temperature_c_gpu0']:.1f}°C | {t['temperature_c_gpu1']:.1f}°C |\n")
            f.write(f"| {r['name']} | Sample | {s['power_watts_gpu0']:.1f} W | {s['power_watts_gpu1']:.1f} W | {s['clock_mhz_gpu0']:.0f} MHz | {s['clock_mhz_gpu1']:.0f} MHz | {s['clock_stability_pct_gpu0']:.1f}% / {s['clock_stability_pct_gpu1']:.1f}% | {s['temperature_c_gpu0']:.1f}°C | {s['temperature_c_gpu1']:.1f}°C |\n")

        f.write("\n## 3. Ray Tracing Latency & Tonemapping Breakdown\n\n")
        f.write("| Scene | Mode | Primary Ray Tracing (ms) | Secondary RT (ms) | Tonemap & Merge (ms) | Throughput (GRays/s) |\n")
        f.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for r in results:
            m = r["4k_single_mono"]
            t = r["4k_dual_tile"]
            s = r["4k_dual_sample"]
            f.write(f"| {r['name']} | Mono | {m['primary_gpu_ms']:.2f} ms | - | {m['tonemap_ms']:.3f} ms | {m['grays_per_sec']:.2f} |\n")
            f.write(f"| {r['name']} | Tile | {t['primary_gpu_ms']:.2f} ms | {t['secondary_gpu_ms']:.2f} ms | {t['merge_tonemap_ms']:.3f} ms | {t['grays_per_sec']:.2f} |\n")
            f.write(f"| {r['name']} | Sample | {s['primary_gpu_ms']:.2f} ms | {s['secondary_gpu_ms']:.2f} ms | {s['merge_tonemap_ms']:.3f} ms | {s['grays_per_sec']:.2f} |\n")

    print(f"[Done] Saved Markdown summary to: {args.summary}")

    if not args.skip_gpu_check:
        print("\n[Step 3] Post-Run GPU Health Check:")
        check_gpu()


if __name__ == "__main__":
    main()
