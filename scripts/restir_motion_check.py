#!/usr/bin/env python3
"""ReSTIR / denoiser quality under camera motion (plan Phase 0.4).

Plays a deterministic camera path headless (frame k is rendered at path time
k/60 s) with progressive accumulation OFF, so each dumped frame is exactly
what the user sees mid-motion. The frame after K frames of motion is compared
to a static high-sample reference rendered at the same pose
(--camera-path-time).

Metrics per (scene, checkpoint, config):
  relmse     : mean((x - r)^2 / (r^2 + 1e-2))   on luminance
  tile_bias  : mean signed relative 16px tile error (systematic bias)
  lum_ratio  : mean(x) / mean(r)

Usage: python3 scripts/restir_motion_check.py [--checkpoints 30,90] [--configs ...]
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BIN = os.path.join(ROOT, "build", "bin", "pathways")
FPS = 60.0

SCENES = {
    "living-room": (["--scene", "scenes/living-room/living_room_extended.glb"],
                    "scenes/living-room/living_room_flight.json"),
    "classroom": (["--scene", "scenes/classroom/classroom_extended.glb"],
                  "scenes/classroom/classroom_flight.json"),
}

CONFIGS = {
    "base": [],
    "restir-di": ["--restir", "--no-restir-gi"],
    "restir-di-gi": ["--restir"],
}

COMMON = ["--headless", "--no-validation", "--accum-format", "fp32",
          "--indirect-clamp", "0", "--no-animate"]
LUMA = np.array([0.2126, 0.7152, 0.0722], dtype=np.float64)


def run(cmd, log_path, out_path):
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode != 0 or not os.path.exists(out_path):
        raise RuntimeError(f"render failed ({proc.returncode}): {' '.join(cmd)} (log: {log_path})")
    return time.time() - t0


def load_luma(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read {path}")
    rgb = np.nan_to_num(img[..., :3][..., ::-1].astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return rgb @ LUMA


def tile_means(lum, tile=16):
    h, w = lum.shape
    th, tw = h // tile, w // tile
    return lum[:th * tile, :tw * tile].reshape(th, tile, tw, tile).mean(axis=(1, 3)).ravel()


def metrics(x, r):
    tx, tr = tile_means(x), tile_means(r)
    m = tr > max(r.mean() * 1e-3, 1e-8)
    err = (x - r) ** 2 / (r ** 2 + 1e-2)
    trimmed = np.sort(err.ravel())[: int(err.size * 0.999)]
    bx, br = tile_means(x, 8), tile_means(r, 8)
    return {
        "relmse": float(err.mean()),
        "relmse_trim": float(trimmed.mean()),          # top 0.1% fireflies removed
        "relmse_8x": float(np.mean((bx - br) ** 2 / (br ** 2 + 1e-2))),  # low-frequency error
        "tile_bias": float(np.mean((tx[m] - tr[m]) / tr[m])),
        "lum_ratio": float(x.mean() / max(r.mean(), 1e-12)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes", default="living-room")
    ap.add_argument("--configs", default="base,restir-di,restir-di-gi")
    ap.add_argument("--checkpoints", default="30,90", help="frame counts of motion before the dump")
    ap.add_argument("--ref-frames", type=int, default=4096)
    ap.add_argument("--res", default="960x540")
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "restir_motion"))
    args = ap.parse_args()

    w, h = (int(v) for v in args.res.lower().split("x"))
    os.makedirs(args.out, exist_ok=True)
    results = {}
    for scene in args.scenes.split(","):
        sargs, path = SCENES[scene]
        base = [BIN, *COMMON, "--width", str(w), "--height", str(h), *sargs, "--camera-path", path]
        for k in (int(c) for c in args.checkpoints.split(",")):
            t = (k - 1) / FPS  # last rendered frame index is k-1
            tag = f"{scene}_{w}x{h}_k{k}"
            ref_p = os.path.join(args.out, f"{tag}_ref{args.ref_frames}.exr")
            if not os.path.exists(ref_p):
                print(f"[{tag}] reference at t={t:.4f}s ...", flush=True)
                run([*base, "--camera-path-time", f"{t:.6f}", "--accum-cutoff", "0",
                     "--frames", str(args.ref_frames), "--dump-hdr", ref_p], ref_p + ".log", ref_p)
            ref = load_luma(ref_p)
            for cfg in args.configs.split(","):
                p = os.path.join(args.out, f"{tag}_{cfg}.exr")
                dt = run([*base, *CONFIGS[cfg], "--realtime", "--frames", str(k), "--dump-hdr", p], p + ".log", p)
                m = metrics(load_luma(p), ref)
                m["render_s"] = round(dt, 2)
                results.setdefault(scene, {}).setdefault(f"k{k}", {})[cfg] = m
                print(f"[{tag}] {cfg:14s} relMSE={m['relmse']:.4f} trim={m['relmse_trim']:.4f} "
                      f"8x={m['relmse_8x']:.5f} tile_bias={m['tile_bias']:+.4f} "
                      f"lum_ratio={m['lum_ratio']:.4f}", flush=True)

    out = os.path.join(args.out, f"summary_{w}x{h}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"summary: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
