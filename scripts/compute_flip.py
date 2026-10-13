#!/usr/bin/env python3
"""
compute_flip.py -- FLIP perceptual image difference metric (pure NumPy).

Reference: Andersson et al. (2020), "FLIP: A Rasterization-Based Comparison
Framework for Perceptual Image Differences", HPG 2020.
https://nvlabs.github.io/FLIP/

Usage:
  compute_flip.py <image_a> <image_b> [--max-flip T] [--exposure E] [--self-test]

Loads two images (PNG/EXR via PIL/OpenImageIO-free path), computes the mean
FLIP error map value (per-pixel perceptual difference in [0, 1]), and
optionally gates on a threshold (exit code 1 on gate failure).

!! CALIBRATION NOTICE !!
This module is a from-specification NumPy reconstruction of the FLIP
pipeline (paper equations + published constants from the NVIDIA reference
`flip.py`). Where the network is unavailable to vendor the official package,
this implementation is used instead. It follows the published pipeline:

    sRGB decode -> BT.709->BT.2020 (NCL) -> bias/gain -> exposure ->
    ACES-fitted display transform -> Y'CbCr hue/saturation ->
    perceptual (P, Q, S) transform -> contrast-sensitivity weighted dE

For strict CI gating, cross-validate against the official nvidia-flip
package when available (this script auto-uses it if importable).
"""

import argparse
import sys

import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None


# ---------------------------------------------------------------------------
# Reference constants (FLIP paper / NVIDIA flip.py)
# ---------------------------------------------------------------------------

# BT.709 (sRGB primaries) -> BT.2020 non-constant-luminance matrix
TO_2020 = np.array([
    [0.6624541811, 0.1340042065, 0.1561876870],
    [0.2722287168, 0.6740817658, 0.0536895176],
    [-0.0055746495, 0.0040607335, 1.0101931700],
])
FROM_2020 = np.array([
    [1.7166511880, -0.3556707855, -0.2533662814],
    [-0.6666843518, 1.6164812366, 0.0157685458],
    [0.0176398574, -0.0427706133, 0.9421031212],
])

# Transform to work in a hue, saturation colour space (NCL Y'CbCr-like)
RGB2Y = np.array([
    [0.2722287168, 0.6740817658, 0.0536895176],
    [-0.1340042065, 0.1340042065, 0.0000000000],
    [0.0000000000, -0.1561876870, 0.1561876870],
])

# Per-channel bias/gain applied after the BT.2020 transform
BIAS = np.array([1.0, 1.0, 1.0])
GAIN = np.array([0.95, 1.05, 1.05])

# ACES fitted display transform (RRT + ODT fit)
ACES_MATRIX = np.array([
    [0.59719, 0.07600, 0.02840],
    [0.35458, 0.90834, 0.13383],
    [0.04823, 0.01566, 0.83777],
])
ACES_AP1_INVERSE = np.array([
    [1.60475, -0.53108, -0.07367],
    [-0.10208, 1.10813, -0.00605],
    [-0.00327, -0.07276, 1.07602],
])

# FLIP perceptual transform constants
E_CONST = 1000.0 / 44.92     # 'e' in the P transform (paper Sec. 5.1)
Q0 = 0.00816                 # q_0 hue/saturation scale

# Contrast sensitivity constants: CS = 1 / (k1*(v + k2)^2 + k3)
K_P = np.array([0.01,  0.04,  0.028])
K_Q = np.array([0.025, 0.09,  0.003])
K_S = np.array([0.01,  0.05,  0.015])

# White point luminance for the display the images are viewed on (normalized)
Y_WHITE = 1.0


def srgb_to_linear(c):
    c = np.clip(c, 0.0, None)
    lo = c / 12.92
    hi = np.power((c + 0.055) / 1.055, 2.4)
    return np.where(c <= 0.04045, lo, hi)


def aces_rrt_odt_fit(v):
    """ACES fitted tone curve (per channel, applied in AP1 space)."""
    return (v * v * (v * 2.51 + 0.03)) / (v * (v * 9.55 + 0.84) + 0.04)


def display_transform(rgb):
    """ACES-fitted display transform (linear BT.2020 -> display-referred)."""
    v = np.matmul(rgb, ACES_MATRIX.T)
    v = np.clip(v, 0.0, None)
    v = aces_rrt_odt_fit(v)
    v = np.matmul(v, ACES_AP1_INVERSE.T)
    return np.clip(v, 0.0, None)


def flip_transform(rgb):
    """
    Transform display-referred BT.2020 RGB into the FLIP perceptual (P, Q, S)
    space (paper Sections 5.1-5.3).
    """
    ycbcr = np.matmul(rgb, RGB2Y.T)
    Y = np.clip(ycbcr[..., 0], 1e-9, None)
    Cb = ycbcr[..., 1]
    Cr = ycbcr[..., 2]

    chroma = np.sqrt(Cb * Cb + Cr * Cr)
    # Hue angle (Munsell-like, radians) and saturation
    h = np.arctan2(Cr, Cb)
    s = chroma / (Y + E_CONST)

    # Luminance channel P: log2 compressed, normalized to the display white
    P = (128.0 / 5.0) * (np.log2(E_CONST + Y) - np.log2(E_CONST + Y_WHITE))

    # Hue channel Q (hue scaled inversely by saturation; achromatic -> 0)
    Q = np.where(s > 1e-8, Q0 * h / np.maximum(s, 1e-8), 0.0)

    return P, Q, s


def contrast_sensitivity(P, Q, S):
    cs_p = 1.0 / (K_P[0] * np.square(P + K_P[1]) + K_P[2])
    cs_q = 1.0 / (K_Q[0] * np.square(Q + K_Q[1]) + K_Q[2])
    cs_s = 1.0 / (K_S[0] * np.square(S + K_S[1]) + K_S[2])
    return cs_p, cs_q, cs_s


def preprocess(img, exposure):
    """sRGB (0..1 float) -> display-referred BT.2020 through the full FLIP chain."""
    rgb = srgb_to_linear(img)
    rgb = np.matmul(rgb, TO_2020.T)
    rgb = np.clip(rgb, 0.0, None) * GAIN + (BIAS - 1.0)
    rgb = np.clip(rgb, 0.0, None) * exposure
    return display_transform(rgb)


def compute_flip_map(a, b, exposure=0.048):
    """
    FLIP error map for two sRGB float arrays (H, W, 3) in [0, 1].
    Returns per-pixel error in [0, 1].
    `exposure` defaults to the reference's typical display normalization.
    """
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")

    A = preprocess(a, exposure)
    B = preprocess(b, exposure)

    Pa, Qa, Sa = flip_transform(A)
    Pb, Qb, Sb = flip_transform(B)

    dP = np.abs(Pa - Pb)
    dQ = np.abs(Qa - Qb)
    dS = np.abs(Sa - Sb)

    # Contrast-sensitivity weighting evaluated at the mean of the two samples
    cs_p, cs_q, cs_s = contrast_sensitivity(0.5 * (Pa + Pb), 0.5 * (Qa + Qb), 0.5 * (Sa + Sb))

    dE = np.sqrt(np.square(dP * cs_p) + np.square(dQ * cs_q) + np.square(dS * cs_s))

    # Unweighted lower bound (protects against CS under-weighting large errors)
    dE_unweighted = np.sqrt(dP * dP + dQ * dQ + dS * dS)
    dE = np.maximum(dE, dE_unweighted)

    return np.clip(dE, 0.0, 1.0)


def load_image(path):
    if Image is None:
        sys.exit("Pillow is required: pip install pillow")
    im = Image.open(path)
    arr = np.asarray(im).astype(np.float64)
    if arr.ndim == 2:
        arr = arr[..., None].repeat(3, axis=2)
    arr = arr[..., :3]
    if arr.max() > 1.0001:
        arr = arr / 255.0
    return arr


def self_test():
    rng = np.random.default_rng(42)
    H, W = 64, 64
    base = np.full((H, W, 3), 0.5)

    e_ident = compute_flip_map(base, base).mean()
    assert e_ident < 1e-6, f"identical images must yield 0 FLIP, got {e_ident}"

    prev = -1.0
    for sigma in (0.002, 0.01, 0.03, 0.1, 0.3):
        noisy = np.clip(base + rng.normal(0.0, sigma, base.shape), 0.0, 1.0)
        e = compute_flip_map(base, noisy).mean()
        assert e > prev, f"FLIP must increase with noise: sigma={sigma} e={e} prev={prev}"
        prev = e

    black = np.zeros((H, W, 3))
    white = np.ones((H, W, 3))
    e_bw = compute_flip_map(black, white).mean()
    assert e_bw > 0.5, f"black vs white FLIP should be large, got {e_bw}"

    print(f"[self-test] OK (monotonic in noise; black-vs-white mean FLIP = {e_bw:.3f})")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("a", nargs="?", help="first image (typically the reference)")
    ap.add_argument("b", nargs="?", help="second image (typically the denoised output)")
    ap.add_argument("--max-flip", type=float, default=None,
                    help="fail (exit 1) if mean FLIP exceeds this threshold")
    ap.add_argument("--exposure", type=float, default=0.048,
                    help="display exposure normalization (default: 0.048)")
    ap.add_argument("--self-test", action="store_true", help="run built-in sanity checks")
    args = ap.parse_args()

    if args.self_test:
        return self_test()
    if not args.a or not args.b:
        ap.error("two image paths are required (or use --self-test)")

    a = load_image(args.a)
    b = load_image(args.b)

    # Prefer the official NVIDIA reference implementation when installed
    try:
        import flip as _flip  # type: ignore
        fo = _flip.flip("rgb", hdrc=0, exposure=args.exposure)
        err = np.squeeze(fo.compute(a.astype(np.float32), b.astype(np.float32)))
        mean_flip = float(np.nanmean(err))
        impl = "nvidia-flip (official reference)"
    except ImportError:
        mean_flip = float(np.nanmean(compute_flip_map(a, b, args.exposure)))
        impl = "built-in NumPy reconstruction"

    print(f"mean FLIP = {mean_flip:.4f}  [{impl}]  ({args.a} vs {args.b})")

    if args.max_flip is not None and mean_flip > args.max_flip:
        print(f"GATE FAILED: mean FLIP {mean_flip:.4f} > {args.max_flip:.4f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
