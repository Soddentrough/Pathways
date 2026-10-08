#!/usr/bin/env python3
"""
Pathways Automated Performance Regression Matrix
Executes and validates the complete 21-Scene x 2-Resolution (1080p, 4K) x 2-GPU (Single, Dual)
matrix (84 configurations total) against golden reference baselines.

Gate semantics (Oct 8 2026, review finding A2):
- Baselines are hardware-keyed. The baseline JSON carries a "metadata" block recording the
  device name, driver version, and git commit it was captured on. If the current host's
  device does not match the baseline device, the matrix SKIPS (exit 0) with a printed
  reason instead of reporting false regressions. Legacy baselines (flat dict, no metadata)
  also skip, prompting a re-cut with --update-baseline on the reference hardware.
- Multi-GPU profiles (*_mgpu) are skipped automatically when the engine's --mgpu run
  reports no active secondary GPU (i.e. fewer than 2 qualifying RT/DGC devices).
- The harness parses no engine stdout: device identity, driver, and dual-GPU capability
  are all read from the engine's own --dump-stats JSON (the engine is the source of truth).
- An idle-system guard checks amdgpu gpu_busy_percent before measuring; a busy GPU
  (e.g. another workload sharing a UMA APU) aborts the run with a printed reason.
- RADV_PROFILE_PSTATE=peak is pinned for all benchmark subprocesses so Mesa does not
  downclock between runs.
"""

import sys
import os
import argparse
import subprocess
import json
import time
import tempfile

PATHWAYS_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BIN_CANDIDATES = [
    os.path.join(PATHWAYS_ROOT, "build", "bin", "pathways"),
    os.path.join(PATHWAYS_ROOT, "build", "linux-release", "bin", "pathways"),
    os.path.join(PATHWAYS_ROOT, "build", "bin", "Release", "pathways.exe"),
    os.path.join(PATHWAYS_ROOT, "build", "bin", "pathways.exe"),
]

def find_binary():
    for cand in BIN_CANDIDATES:
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None

CANONICAL_SCENES = [
    # Procedural (4)
    {"id": "cornell_box", "name": "Cornell Box", "path": "cornell-box", "bounces": 4},
    {"id": "many_lights", "name": "Many-Lights (64 Lights)", "path": "many-lights", "bounces": 4},
    {"id": "cyber_city", "name": "Cyber City", "path": "cyber-city", "bounces": 4},
    {"id": "infinity_mirror", "name": "Infinity Mirror", "path": "infinity-mirror", "bounces": 8},
    # Showcase GLTF (3)
    {"id": "damaged_helmet", "name": "Damaged Helmet", "path": "scenes/DamagedHelmet.glb", "bounces": 4},
    {"id": "dragon_attenuation", "name": "Dragon Attenuation", "path": "scenes/DragonAttenuation.glb", "bounces": 4},
    {"id": "dragon_dispersion", "name": "Dragon Dispersion", "path": "scenes/DragonDispersion.glb", "bounces": 4},
    # Research GLTF (8)
    {"id": "bmw_m6", "name": "BMW M6", "path": "scenes/bmw-m6/bmw_m6_extended.glb", "bounces": 4},
    {"id": "breakfast_room", "name": "Breakfast Room", "path": "scenes/breakfast-room/breakfast_room_extended.glb", "bounces": 4},
    {"id": "classroom", "name": "Classroom", "path": "scenes/classroom/classroom_extended.glb", "bounces": 4},
    {"id": "cornell_caustic", "name": "Cornell Caustic", "path": "scenes/cornell-caustic/cornell_caustic_extended.glb", "bounces": 4},
    {"id": "glass_of_water", "name": "Glass of Water", "path": "scenes/glass-of-water/glass_of_water_extended.glb", "bounces": 4},
    {"id": "living_room", "name": "Living Room", "path": "scenes/living-room/living_room_extended.glb", "bounces": 4},
    {"id": "modern_hall", "name": "Modern Hall", "path": "scenes/modern-hall/modern_hall_extended.glb", "bounces": 4},
    {"id": "veach_ajar", "name": "Veach Ajar", "path": "scenes/veach-ajar/veach_ajar_extended.glb", "bounces": 4},
    # OpenUSD (4)
    {"id": "buick_riviera", "name": "Buick Riviera", "path": "scenes/BuickRiviera/BuickRiviera.usdc", "bounces": 4},
    {"id": "coffee_maker", "name": "Coffee Maker", "path": "scenes/coffee-maker/coffee_maker.usda", "bounces": 4},
    {"id": "kitchen_set", "name": "Kitchen Set", "path": "scenes/Kitchen_set/Kitchen_set.usd", "bounces": 4},
    {"id": "med_city", "name": "Point Instanced Med City", "path": "scenes/PointInstancedMedCity/PointInstancedMedCity.usd", "bounces": 4},
    # Custom Bistro (2)
    {"id": "bistro_exterior", "name": "Bistro Exterior", "path": "scenes/bistro/bistro_exterior.glb", "bounces": 4},
    {"id": "bistro_interior", "name": "Bistro Interior", "path": "scenes/bistro/bistro_interior.glb", "bounces": 4}
]

PROFILES = [
    {"id": "1080p_single", "label": "1080p Single", "res": "1080p", "mgpu": False},
    {"id": "1080p_mgpu",   "label": "1080p Dual",   "res": "1080p", "mgpu": True},
    {"id": "4k_single",    "label": "4K Single",    "res": "4k",    "mgpu": False},
    {"id": "4k_mgpu",      "label": "4K Dual",      "res": "4k",    "mgpu": True}
]

def skip(reason):
    print(f"\033[33m[SKIP]\033[0m {reason}")
    return 0

def normalize_device(name):
    """Normalize a device name for cross-run comparison (case/punct insensitive)."""
    import re
    return re.sub(r"[^a-z0-9]+", "", (name or "").lower())

def benchmark_env():
    """Subprocess environment with the Mesa P-state pinned for stable timing."""
    env = os.environ.copy()
    env.setdefault("RADV_PROFILE_PSTATE", "peak")
    return env

def probe_hardware(bin_path):
    """Run one tiny --mgpu headless frame with --dump-stats and read device identity,
    driver version, and dual-GPU capability straight from the engine's own JSON.

    A single run suffices: the primary-GPU block is identical with or without --mgpu,
    and --mgpu additionally tells us whether a second GPU is actually active
    (secondary_gpu.active). No stdout is parsed — the engine is the source of truth.
    """
    probe_json = os.path.join(tempfile.gettempdir(), "pathways_perf_probe.json")
    cmd = [bin_path, "--headless", "--frames", "1", "--width", "320", "--height", "240",
           "--spp", "1", "--mgpu", "--dump-stats", probe_json]
    try:
        probe_res = subprocess.run(cmd, cwd=PATHWAYS_ROOT, capture_output=True, text=True,
                                   timeout=20, env=benchmark_env())
    except subprocess.TimeoutExpired:
        return None, None, False, "Hardware probe timed out after 20s (software emulation)"

    if probe_res.returncode != 0:
        return None, None, False, (f"Vulkan Ray Tracing hardware not available in environment "
                                   f"(code {probe_res.returncode}): {probe_res.stderr.strip()[:200]}")

    try:
        with open(probe_json, "r") as f:
            data = json.load(f)
    except Exception as e:
        return None, None, False, f"Failed to read probe stats JSON: {e}"
    finally:
        try:
            os.remove(probe_json)
        except OSError:
            pass

    pg = data.get("primary_gpu", {})
    device = pg.get("device_name", "").split(" (")[0].strip()
    driver = pg.get("vulkan_api_version", "")
    if device and any(sw in device.lower() for sw in ["llvmpipe", "lavapipe", "software rasterizer", "cpu device"]):
        return None, None, False, "Software/CPU Vulkan renderer detected"
    mgpu_available = bool(data.get("secondary_gpu", {}).get("active", False))
    return device or None, driver, mgpu_available, None

def gpu_busy_percentages():
    """Read amdgpu gpu_busy_percent for all DRM cards (empty list if unavailable)."""
    busy = []
    for card in sorted(os.listdir("/sys/class/drm")):
        if not card.startswith("card") or not card[4:].isdigit():
            continue
        path = f"/sys/class/drm/{card}/device/gpu_busy_percent"
        try:
            with open(path) as f:
                busy.append((card, int(f.read().strip())))
        except (OSError, ValueError):
            continue
    return busy

def git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=PATHWAYS_ROOT,
                              capture_output=True, text=True, timeout=5).stdout.strip()
    except Exception:
        return "unknown"

def load_baseline(baseline_path):
    """Return (results_dict_or_None, skip_reason_or_None, metadata_or_None)."""
    if not os.path.isfile(baseline_path):
        return None, "No baseline file found (run with --update-baseline to create one)", None
    try:
        with open(baseline_path, "r") as f:
            data = json.load(f)
    except Exception as e:
        return None, f"Failed to load baseline ({e})", None
    if isinstance(data, dict) and "metadata" in data and "results" in data:
        return data.get("results", {}), None, data.get("metadata", {})
    # Legacy flat format: no hardware keying -> cannot be trusted on arbitrary hosts.
    return None, ("Baseline is in the legacy flat format (no hardware metadata). "
                  "Re-cut it on the reference hardware with --update-baseline."), None

def run_single_benchmark(bin_path, scene, profile, frames, warmup, out_json):
    cmd = [
        bin_path,
        "--headless",
        "--scene", scene["path"],
        "--res", profile["res"],
        "--spp", "1",
        "--max-bounces", str(scene["bounces"]),
        "--frames", str(frames),
        "--warmup-frames", str(warmup),
        "--benchmark",
        "--no-accumulation",
        "--dump-stats", out_json
    ]
    if profile["mgpu"]:
        cmd.append("--mgpu")

    res = subprocess.run(cmd, cwd=PATHWAYS_ROOT, capture_output=True, text=True, env=benchmark_env())
    if res.returncode != 0:
        return None, f"Exited with code {res.returncode}: {res.stderr.strip()[:300]}"

    if not os.path.exists(out_json):
        return None, "Stats JSON file was not generated"

    try:
        with open(out_json, "r") as f:
            data = json.load(f)
    except Exception as e:
        return None, f"Failed to parse stats JSON: {e}"

    perf_blocks = data.get("performance", {}).get("configurations_breakdown", [])
    if not perf_blocks:
        return None, "No performance configuration breakdown in stats JSON"

    cfg = perf_blocks[0]
    val_errors = data.get("performance", {}).get("validation_errors", 0)

    stages = cfg.get("pipeline_stages_ms", {})
    bounces = stages.get("bounces", [])
    shade_sum = sum(b.get("shade_ms", 0.0) for b in bounces)
    shadow_sum = sum(b.get("shadow_ms", 0.0) for b in bounces)
    intersect_sum = sum(b.get("intersect_ms", 0.0) for b in bounces)

    result = {
        "avg_frame_time_ms": cfg.get("avg_frame_time_ms", 0.0),
        "avg_fps": cfg.get("avg_fps", 0.0),
        "gigarays_per_second": cfg.get("gigarays_per_second", 0.0),
        "classify_ms": stages.get("classify_ms", 0.0),
        "shade_sum_ms": shade_sum,
        "shadow_sum_ms": shadow_sum,
        "intersect_sum_ms": intersect_sum,
        "tonemap_ms": stages.get("tonemap_ms", 0.0),
        "validation_errors": val_errors
    }
    return result, None

def main():
    parser = argparse.ArgumentParser(description="Pathways Performance Regression Matrix (21 Scenes x 4 Profiles = 84 Runs)")
    parser.add_argument("--quick", action="store_true", help="Quick mode (5 measured frames, 2 warmup frames)")
    parser.add_argument("--frames", type=int, default=None, help="Measured frames per test point")
    parser.add_argument("--warmup", type=int, default=None, help="Warmup frames per test point")
    parser.add_argument("--scenes", type=str, default="", help="Comma-separated scene IDs or regex filter")
    parser.add_argument("--profiles", type=str, default="", help="Comma-separated profile IDs or filter (e.g. '4k_mgpu' or '4k_single,4k_mgpu')")
    parser.add_argument("--update-baseline", action="store_true", help="Generate or update reference baseline JSON")
    parser.add_argument("--baseline-path", type=str, default="", help="Custom baseline JSON path")
    parser.add_argument("--tolerance", type=float, default=None, help="Regression threshold tolerance (default: 0.15, or 0.20 in --quick mode)")
    parser.add_argument("--output-dir", type=str, default="output/benchmark_matrix", help="Output directory for reports")
    parser.add_argument("--force", action="store_true",
                        help="Bypass the idle-system guard and baseline device-match check (for re-cutting baselines on a loaded or non-reference host)")
    parser.add_argument("--max-gpu-busy", type=int, default=15,
                        help="Idle-system guard: skip if any amdgpu card reports higher busy%% (default: 15)")
    args = parser.parse_args()

    bin_path = find_binary()
    if not bin_path:
        return skip("Pathways binary not found! Skipping performance matrix.")

    # In CI environments (GitHub Actions) without dedicated hardware benchmarking GPUs, skip regression matrix
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return skip("Detected CI environment without dedicated benchmarking hardware. Skipping performance matrix.")

    # Probe hardware: device identity, driver, and dual-GPU capability
    device, driver, mgpu_available, probe_err = probe_hardware(bin_path)
    if probe_err:
        return skip(probe_err)

    # Idle-system guard: a GPU shared with other workloads (e.g. a 22 GB LLM server on a
    # UMA APU) produces meaningless timings. Measure only on an idle GPU.
    if not args.force:
        busy = gpu_busy_percentages()
        hot = [(c, p) for c, p in busy if p > args.max_gpu_busy]
        if hot:
            detail = ", ".join(f"{c} {p}%" for c, p in hot)
            return skip(f"GPU is not idle ({detail}; threshold {args.max_gpu_busy}%). "
                        f"Another workload is contending for the GPU — timings would be meaningless. "
                        f"Re-run when the system is idle, or use --force to override.")
        if not busy:
            print("\033[33m[WARN]\033[0m gpu_busy_percent unavailable (non-amdgpu?); idle guard disabled.")
    else:
        print("\033[33m[WARN]\033[0m --force: idle-system guard and device-match checks bypassed.")

    frames = args.frames if args.frames is not None else (5 if args.quick else 15)
    warmup = args.warmup if args.warmup is not None else (4 if args.quick else 5)

    baseline_path = args.baseline_path if args.baseline_path else os.path.join(PATHWAYS_ROOT, "tests", "references", "performance_baseline.json")
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.dirname(baseline_path), exist_ok=True)

    # Baseline load + hardware keying (review finding A2)
    baseline_data = {}
    baseline_meta = None
    if not args.update_baseline:
        baseline_data, skip_reason, baseline_meta = load_baseline(baseline_path)
        if skip_reason:
            return skip(f"{skip_reason} Skipping to avoid false regressions.")
        base_device = (baseline_meta or {}).get("device_name", "")
        if base_device and device and normalize_device(base_device) != normalize_device(device) and not args.force:
            return skip(f"Baseline was recorded on '{base_device}' but this host is '{device}'. "
                        f"Performance numbers are not comparable across hardware. "
                        f"Run on the reference hardware, or re-cut with --update-baseline.")
        if baseline_meta:
            print(f"[INFO] Baseline device: {baseline_meta.get('device_name', '?')} "
                  f"(driver {baseline_meta.get('driver_version', '?')}, git {baseline_meta.get('git_commit', '?')}, "
                  f"{baseline_meta.get('date', '?')}) | Current: {device} (driver {driver})")

    # Filter profiles if requested
    profile_filter = [p.strip().lower() for p in args.profiles.split(",") if p.strip()]
    profiles_to_run = []
    for prof in PROFILES:
        if not profile_filter or any(f in prof["id"].lower() for f in profile_filter):
            profiles_to_run.append(prof)

    # Multi-GPU capability gate (review finding A2): skip *_mgpu profiles when the engine
    # cannot actually form a Dual GPU configuration on this host.
    mgpu_profiles = [p for p in profiles_to_run if p["mgpu"]]
    if mgpu_profiles and not mgpu_available:
        skipped = ", ".join(p["id"] for p in mgpu_profiles)
        print(f"\033[33m[WARN]\033[0m No Dual GPU configuration available on this host (only 1 qualifying "
              f"RT/DGC device); skipping profiles: {skipped}")
        profiles_to_run = [p for p in profiles_to_run if not p["mgpu"]]
    if not profiles_to_run:
        return skip("No profiles left to run after capability filtering.")

    # Filter scenes if requested
    scene_filter = [s.strip().lower() for s in args.scenes.split(",") if s.strip()]
    scenes_to_run = []
    for sc in CANONICAL_SCENES:
        if not scene_filter:
            scenes_to_run.append(sc)
        else:
            sc_id = sc["id"].lower()
            sc_name = sc["name"].lower()
            if any(f in sc_id or f in sc_name for f in scene_filter):
                scenes_to_run.append(sc)

    tolerance = args.tolerance if args.tolerance is not None else (0.20 if args.quick else 0.15)
    total_runs = len(scenes_to_run) * len(profiles_to_run)
    print("==========================================================================================")
    print(f"  Pathways Automated Performance Regression Matrix ({len(scenes_to_run)} Scenes x {len(profiles_to_run)} Profiles = {total_runs} Runs)")
    print(f"  Binary:   {os.path.relpath(bin_path, PATHWAYS_ROOT)}")
    print(f"  Device:   {device} (driver {driver}) | Dual GPU: {'yes' if mgpu_available else 'no'}")
    print(f"  Frames:   {frames} measured (+ {warmup} warmup) per configuration")
    print(f"  Mode:     {'Update Golden Baseline' if args.update_baseline else 'Regression Evaluation (Tolerance: ' + str(int(tolerance * 100)) + '%)'}")
    print("==========================================================================================")

    matrix_results = {}
    regressions = []
    failures = []
    improvements = []
    run_idx = 0

    print(f"\n{'Idx':<4} | {'Scene':<22} | {'Profile':<14} | {'Time (ms)':<9} | {'FPS':<7} | {'GRays/s':<8} | {'Baseline':<9} | {'Delta':<7} | {'Status'}")
    print("-" * 105)

    for sc in scenes_to_run:
        sc_id = sc["id"]
        matrix_results[sc_id] = {}

        for prof in profiles_to_run:
            run_idx += 1
            prof_id = prof["id"]
            tag = f"{sc_id}_{prof_id}"
            stat_file = os.path.join(out_dir, f"stats_{tag}.json")

            res, err = run_single_benchmark(bin_path, sc, prof, frames, warmup, stat_file)
            if err or not res:
                failures.append((tag, err or "Unknown execution error"))
                print(f"[{run_idx:02d}/{total_runs:02d}] | {sc['name']:<22} | {prof['label']:<14} | FAILED: {err[:40]}")
                continue

            matrix_results[sc_id][prof_id] = res

            cur_time = res["avg_frame_time_ms"]
            cur_fps = res["avg_fps"]
            cur_grays = res["gigarays_per_second"]
            val_errs = res["validation_errors"]

            status = "\033[32mPASS\033[0m"
            delta_str = "0.0%"
            base_str = "N/A"

            if val_errs > 0:
                status = f"\033[31mFAIL (Val: {val_errs})\033[0m"
                failures.append((tag, f"{val_errs} Vulkan validation errors"))
            elif tag in baseline_data:
                base_time = baseline_data[tag].get("avg_frame_time_ms", 0.0)
                if base_time > 0.0:
                    base_str = f"{base_time:.2f} ms"
                    ratio = cur_time / base_time
                    delta_pct = (ratio - 1.0) * 100.0
                    delta_str = f"{delta_pct:+.1f}%"

                    if ratio > 1.0 + tolerance:
                        status = "\033[31mREGRESSION\033[0m"
                        regressions.append((tag, f"{cur_time:.2f}ms vs baseline {base_time:.2f}ms ({delta_pct:+.1f}%)"))
                    elif ratio < 0.95:
                        status = "\033[36mIMPROVED\033[0m"
                        improvements.append((tag, f"{cur_time:.2f}ms vs baseline {base_time:.2f}ms ({delta_pct:+.1f}%)"))

            print(f"[{run_idx:02d}/{total_runs:02d}] | {sc['name']:<22} | {prof['label']:<14} | {cur_time:7.2f} ms | {cur_fps:5.0f} | {cur_grays:6.2f} | {base_str:<9} | {delta_str:<7} | {status}")

    # Save full matrix report
    report_json_path = os.path.join(out_dir, "performance_matrix_report.json")
    with open(report_json_path, "w") as f:
        json.dump({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": device,
            "driver": driver,
            "dual_gpu": mgpu_available,
            "total_configurations": total_runs,
            "regressions": regressions,
            "failures": failures,
            "improvements": improvements,
            "results": matrix_results
        }, f, indent=2)

    # Save or update baseline if requested (hardware-keyed format, review finding A2)
    if args.update_baseline:
        flat_baseline = {}
        for sc_id, profs in matrix_results.items():
            for prof_id, data in profs.items():
                flat_baseline[f"{sc_id}_{prof_id}"] = {
                    "avg_frame_time_ms": round(data["avg_frame_time_ms"], 3),
                    "avg_fps": round(data["avg_fps"], 1),
                    "gigarays_per_second": round(data["gigarays_per_second"], 3)
                }
        if baseline_meta and baseline_meta.get("device_name") and device \
                and normalize_device(baseline_meta["device_name"]) != normalize_device(device):
            print(f"\033[33m[WARN]\033[0m Re-keying baseline from '{baseline_meta['device_name']}' to '{device}'.")
        baseline_doc = {
            "metadata": {
                "device_name": device,
                "driver_version": driver,
                "git_commit": git_commit(),
                "date": time.strftime("%Y-%m-%d", time.gmtime()),
                "frames": frames,
                "warmup": warmup,
                "dual_gpu": mgpu_available,
                "note": "Hardware-keyed golden baseline. Regenerate on the reference hardware with --update-baseline."
            },
            "results": flat_baseline
        }
        with open(baseline_path, "w") as f:
            json.dump(baseline_doc, f, indent=2)
        print(f"\n\033[32m[SUCCESS]\033[0m Updated golden baseline ({len(flat_baseline)} entries, device: {device}) at: {baseline_path}")

    # Print summary
    print("\n==========================================================================================")
    print("  Performance Matrix Execution Summary")
    print("==========================================================================================")
    print(f"  Total Configurations: {total_runs}")
    print(f"  Failures / Errors:    {len(failures)}")
    print(f"  Regressions (> {int(tolerance*100)}%):  {len(regressions)}")
    print(f"  Improvements (> 5%):  {len(improvements)}")
    print(f"  Report written to:    {report_json_path}")

    if regressions:
        print("\n\033[31m[REGRESSION ALERTS]\033[0m:")
        for tag, msg in regressions:
            print(f"  * {tag}: {msg}")

    if failures:
        print("\n\033[31m[FAILURE ALERTS]\033[0m:")
        for tag, msg in failures:
            print(f"  * {tag}: {msg}")

    if not regressions and not failures:
        print("\n\033[32m[SUCCESS] All performance matrix configurations passed regression checks cleanly!\033[0m\n")
        return 0
    else:
        print("\n\033[31m[FAILURE] Performance matrix detected regressions or errors!\033[0m\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
