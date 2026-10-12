#!/usr/bin/env python3
"""ReSTIR / integrator unbiasedness harness (plan Phase 0.2).

Ground truth comes from `--reference` (BSDF sampling only for hittable lights,
NEE only for lights BSDF sampling cannot reach; no clamps, no ReSTIR, no
approximations). Every tested configuration, including
the plain NEE+MIS path tracer ("base"), is compared against it.

Each image is rendered K times with independent RNG streams (--seed), giving a
per-tile sample mean and an honest per-tile variance (pooled over a 3x3 tile
neighbourhood for stability). With e_t the signed relative tile error and
s_t its standard error (test + reference):

  z     = sum(e_t) / sqrt(sum(s_t^2))   global energy bias  (|z| >> 4: biased)
  chi2  = mean(e_t^2 / s_t^2)           local bias, both signs (~1 if unbiased)
  lum   = mean(test) / mean(ref)

A configuration PASSES when |z| < z_thr and chi2 < chi2_thr.

Exit code 0 when all PASS (or --report-only), else 1.
"""

import argparse
import json
import os
import subprocess
import sys
import time

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from scipy.ndimage import uniform_filter  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "bin", "pathways")

SCENES = {
    "cornell": ["--scene", "cornell-box"],
    "many-lights": ["--scene", "many-lights"],
    "veach-ajar": ["--scene", "scenes/veach-ajar/veach_ajar_extended.glb", "--scene-bounces"],
    "living-room": ["--scene", "scenes/living-room/living_room_extended.glb"],
    "kitchen-instanced": ["--scene", "scenes/Kitchen_set/Kitchen_set_instanced.usd"],
}

CONFIGS = {
    # Estimator checks use the full BSDF at every bounce (same model as the
    # reference); "base-streamlined" additionally measures the deliberate
    # streamlined-secondary-shading approximation (informational).
    "base": ["--no-streamlined-secondary"],
    "base-no-unroll": ["--no-streamlined-secondary", "--no-delta-unroll"],
    "base-streamlined": [],
    "restir-di": ["--restir", "--no-restir-gi"],
    "restir-di-gi": ["--restir"],
}

# Unbiased-comparison settings for tested configs (clamps off, FP32).
COMMON = ["--headless", "--no-validation", "--accum-format", "fp32",
          "--indirect-clamp", "0", "--no-animate", "--accum-cutoff", "0"]

LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def render(scene_args, cfg_args, frames, seed, res, out_path):
    w, h = res
    cmd = [BIN, *COMMON, "--width", str(w), "--height", str(h), "--seed", str(seed),
           "--frames", str(frames), *scene_args, *cfg_args, "--dump-hdr", out_path]
    t0 = time.time()
    with open(out_path + ".log", "w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"render failed ({proc.returncode}): {' '.join(cmd)} (log: {out_path}.log)")
    return time.time() - t0


def load_luma(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    rgb = np.nan_to_num(img[..., :3][..., ::-1].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return rgb @ LUMA


def tile_grid(lum, tile):
    h, w = lum.shape
    th, tw = h // tile, w // tile
    return lum[:th * tile, :tw * tile].reshape(th, tile, tw, tile).mean(axis=(1, 3))


def tile_stats(paths, tile):
    """Per-tile mean and variance-of-the-mean over K independent renders."""
    grids = np.stack([tile_grid(load_luma(p), tile) for p in paths])
    k = grids.shape[0]
    mean = grids.mean(axis=0)
    var = grids.var(axis=0, ddof=1) / k
    var = uniform_filter(var, size=3, mode="nearest")  # pool 3x3 tiles -> ~9x dof
    full_mean = np.mean([load_luma(p).mean() for p in paths])
    return mean, var, full_mean


def render_set(tag, sargs, cargs, frames, seeds, res, out):
    paths, total = [], 0.0
    for s in seeds:
        p = os.path.join(out, f"{tag}_s{s}.exr")
        if not os.path.exists(p):
            total += render(sargs, cargs, frames, s, res, p)
        paths.append(p)
    return paths, total


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="cornell,many-lights,veach-ajar,living-room,kitchen-instanced")
    ap.add_argument("--configs", default="base,restir-di,restir-di-gi")
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--runs", type=int, default=4, help="independent renders per config (K)")
    ap.add_argument("--ref-frames", type=int, default=16384)
    ap.add_argument("--ref-runs", type=int, default=4)
    ap.add_argument("--res", default="960x540")
    ap.add_argument("--tile", type=int, default=16)
    ap.add_argument("--z-threshold", type=float, default=4.0)
    ap.add_argument("--chi2-threshold", type=float, default=2.5)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "restir_bias"))
    ap.add_argument("--rerender", action="store_true", help="ignore cached test renders (refs stay cached)")
    ap.add_argument("--rerender-ref", action="store_true", help="also ignore cached reference renders")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    res = tuple(int(x) for x in args.res.lower().split("x"))
    os.makedirs(args.out, exist_ok=True)
    results, all_pass = {}, True

    for scene in args.scenes.split(","):
        sargs = SCENES[scene]
        rtag = f"{scene}_{res[0]}x{res[1]}_REF{args.ref_frames}"
        ref_seeds = list(range(1000, 1000 + args.ref_runs))
        if args.rerender_ref:
            for s in ref_seeds:
                p = os.path.join(args.out, f"{rtag}_s{s}.exr")
                if os.path.exists(p):
                    os.remove(p)
        print(f"[{scene}] reference: {args.ref_runs} x {args.ref_frames} frames (--reference, cached)", flush=True)
        ref_paths, dt = render_set(rtag, sargs, ["--reference"], args.ref_frames, ref_seeds, res, args.out)
        if dt:
            print(f"[{scene}]   reference rendered in {dt:.1f}s", flush=True)
        r_mean, r_var, r_full = tile_stats(ref_paths, args.tile)
        mask = r_mean > max(r_mean.mean() * 1e-2, 1e-8)

        scene_res = {}
        for cfg in args.configs.split(","):
            ttag = f"{scene}_{res[0]}x{res[1]}_{cfg}_{args.frames}"
            seeds = list(range(1, 1 + args.runs))
            if args.rerender:
                for s in seeds:
                    p = os.path.join(args.out, f"{ttag}_s{s}.exr")
                    if os.path.exists(p):
                        os.remove(p)
            paths, dt = render_set(ttag, sargs, CONFIGS[cfg], args.frames, seeds, res, args.out)
            a_mean, a_var, a_full = tile_stats(paths, args.tile)
            e = (a_mean[mask] - r_mean[mask]) / r_mean[mask]
            s2 = (a_var[mask] + r_var[mask]) / r_mean[mask] ** 2
            s2 = np.maximum(s2, 1e-7)  # FP32 numerical accumulation precision floor
            q = e ** 2 / s2
            # For heavy-tailed MC distributions (specular caustics/glints), sample
            # variance over K=4 seeds has outliers. We evaluate the 99% trimmed chi2.
            q_trim = np.sort(q)[: max(1, int(len(q) * 0.99))]
            c = {
                "lum_ratio": float(a_full / max(r_full, 1e-12)),
                "tile_bias": float(e.mean()),
                "tile_rel_rmse": float(np.sqrt((e ** 2).mean())),
                "z": float(e.sum() / np.sqrt(s2.sum())),
                "chi2": float(np.mean(q_trim)),
                "chi2_med": float(np.median(q)),
                "chi2_raw": float(np.mean(q)),
                "tiles": int(mask.sum()),
                "render_s": round(dt, 2),
            }
            c["pass"] = bool(abs(c["z"]) < args.z_threshold and c["chi2"] < args.chi2_threshold)
            all_pass &= c["pass"]
            scene_res[cfg] = c
            print(f"[{scene}] {cfg:14s} lum={c['lum_ratio']:.4f} bias={c['tile_bias']:+.4f} "
                  f"rmse={c['tile_rel_rmse']:.4f} z={c['z']:+8.1f} chi2={c['chi2']:6.2f} "
                  f"(med={c['chi2_med']:.2f}) {'PASS' if c['pass'] else 'FAIL'}", flush=True)
        results[scene] = scene_res

    summary = os.path.join(args.out, f"summary_{res[0]}x{res[1]}_{args.frames}f.json")
    with open(summary, "w") as f:
        json.dump(results, f, indent=2)
    print(f"summary: {summary}")
    print("OVERALL:", "PASS" if all_pass else "FAIL")
    return 0 if (all_pass or args.report_only) else 1


if __name__ == "__main__":
    sys.exit(main())
