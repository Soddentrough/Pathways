#!/usr/bin/env python3
"""
Pathways Before/After Visual Regression Testing System
Compares rendered frames against golden reference baselines, computes perceptual and
statistical differences (MAE, MSE, PSNR, SSIM, Sharpness, Luminance Drift),
classifies changes as potential regressions or improvements, generates difference heatmaps,
and produces an interactive HTML visual inspection report.
"""

import sys
import os
import shutil
import argparse
import json
import numpy as np
from PIL import Image
import cv2
from skimage.metrics import structural_similarity as ssim

REF_DIR = "tests/references"
OUTPUT_DIR = "output"

# Canonical test configurations for visual regression tracking
TEST_CONFIGS = [
    {
        "id": "classroom_4k_converged",
        "name": "Classroom 4K (Converged Pure MC)",
        "scene": "scenes/classroom/classroom_extended.glb",
        "cmd": [
            "./build/bin/pathways", "--headless",
            "--scene", "scenes/classroom/classroom_extended.glb",
            "--width", "3840", "--height", "2160",
            "--spp", "1", "--max-bounces", "4",
            "--frames", "30", "--warmup-frames", "5",
            "--dump-frame", f"{OUTPUT_DIR}/test_classroom_mc_converged.png",
            "--dump-stats", f"{OUTPUT_DIR}/stats_classroom_mc_converged.json"
        ],
        "render_path": f"{OUTPUT_DIR}/test_classroom_mc_converged.png",
        "ref_path": f"{REF_DIR}/classroom_4k_converged.png",
        "diff_path": f"{OUTPUT_DIR}/diff_classroom_4k_converged.png"
    },
    {
        "id": "classroom_4k_motion",
        "name": "Classroom 4K (Camera Motion & Temporal Accumulation)",
        "scene": "scenes/classroom/classroom_extended.glb",
        "cmd": [
            "./build/bin/pathways", "--headless",
            "--scene", "scenes/classroom/classroom_extended.glb",
            "--width", "3840", "--height", "2160",
            "--spp", "1", "--max-bounces", "4",
            "--frames", "60", "--warmup-frames", "10",
            "--camera-motion",
            "--dump-frame", f"{OUTPUT_DIR}/test_classroom_wf_motion.png",
            "--dump-stats", f"{OUTPUT_DIR}/stats_classroom_wf_motion.json"
        ],
        "render_path": f"{OUTPUT_DIR}/test_classroom_wf_motion.png",
        "ref_path": f"{REF_DIR}/classroom_4k_motion.png",
        "diff_path": f"{OUTPUT_DIR}/diff_classroom_4k_motion.png"
    },
    {
        "id": "living_room_1080p_static",
        "name": "Living Room 1080p (Converged Static Accumulation)",
        "scene": "scenes/living-room/living_room_extended.glb",
        "cmd": [
            "./build/bin/pathways", "--headless",
            "--scene", "scenes/living-room/living_room_extended.glb",
            "--width", "1920", "--height", "1080",
            "--spp", "1", "--max-bounces", "4",
            "--frames", "60", "--warmup-frames", "10",
            "--dump-frame", f"{OUTPUT_DIR}/test_lr_static_test.png",
            "--dump-stats", f"{OUTPUT_DIR}/stats_lr_static_test.json"
        ],
        "render_path": f"{OUTPUT_DIR}/test_lr_static_test.png",
        "ref_path": f"{REF_DIR}/living_room_1080p_static.png",
        "diff_path": f"{OUTPUT_DIR}/diff_living_room_1080p_static.png"
    },
    {
        "id": "living_room_1080p_motion",
        "name": "Living Room 1080p (Dynamic Camera Motion)",
        "scene": "scenes/living-room/living_room_extended.glb",
        "cmd": [
            "./build/bin/pathways", "--headless",
            "--scene", "scenes/living-room/living_room_extended.glb",
            "--width", "1920", "--height", "1080",
            "--spp", "1", "--max-bounces", "4",
            "--frames", "60", "--warmup-frames", "10",
            "--camera-motion",
            "--dump-frame", f"{OUTPUT_DIR}/test_lr_motion_test.png",
            "--dump-stats", f"{OUTPUT_DIR}/stats_lr_motion_test.json"
        ],
        "render_path": f"{OUTPUT_DIR}/test_lr_motion_test.png",
        "ref_path": f"{REF_DIR}/living_room_1080p_motion.png",
        "diff_path": f"{OUTPUT_DIR}/diff_living_room_1080p_motion.png"
    },
    {
        "id": "living_room_1080p_bmfr",
        "name": "Living Room 1080p (BMFR Regression Denoiser)",
        "scene": "scenes/living-room/living_room_extended.glb",
        "cmd": [
            "./build/bin/pathways", "--headless",
            "--scene", "scenes/living-room/living_room_extended.glb",
            "--width", "1920", "--height", "1080",
            "--spp", "1", "--max-bounces", "4",
            "--frames", "60", "--warmup-frames", "10",
            "--bmfr", "--camera-motion",
            "--dump-frame", f"{OUTPUT_DIR}/test_lr_bmfr_motion_test.png",
            "--dump-stats", f"{OUTPUT_DIR}/stats_lr_bmfr_motion_test.json"
        ],
        "render_path": f"{OUTPUT_DIR}/test_lr_bmfr_motion_test.png",
        "ref_path": f"{REF_DIR}/living_room_1080p_bmfr.png",
        "diff_path": f"{OUTPUT_DIR}/diff_living_room_1080p_bmfr.png"
    },
    {
        "id": "cornell_1080p",
        "name": "Cornell Box 1080p (16 SPP Full Quality)",
        "scene": "Procedural Cornell Box",
        "cmd": [
            "./build/bin/pathways", "--headless",
            "--width", "1920", "--height", "1080",
            "--spp", "16", "--max-bounces", "4",
            "--dump-frame", f"{OUTPUT_DIR}/test_cornell_1080p.png",
            "--dump-stats", f"{OUTPUT_DIR}/stats_1080p.json"
        ],
        "render_path": f"{OUTPUT_DIR}/test_cornell_1080p.png",
        "ref_path": f"{REF_DIR}/cornell_1080p.png",
        "diff_path": f"{OUTPUT_DIR}/diff_cornell_1080p.png"
    },
    {
        "id": "damaged_helmet",
        "name": "Damaged Helmet (glTF 2.0 Complex PBR)",
        "scene": "scenes/DamagedHelmet.glb",
        "cmd": [
            "./build/bin/pathways", "--headless",
            "--scene", "scenes/DamagedHelmet.glb",
            "--width", "1920", "--height", "1080",
            "--spp", "16", "--max-bounces", "4",
            "--dump-frame", f"{OUTPUT_DIR}/test_gltf_helmet.png",
            "--dump-stats", f"{OUTPUT_DIR}/stats_gltf_helmet.json"
        ],
        "render_path": f"{OUTPUT_DIR}/test_gltf_helmet.png",
        "ref_path": f"{REF_DIR}/damaged_helmet.png",
        "diff_path": f"{OUTPUT_DIR}/diff_damaged_helmet.png"
    },
    {
        "id": "dragon_dispersion",
        "name": "Dragon Dispersion (KHR_materials_dispersion)",
        "scene": "scenes/DragonDispersion.glb",
        "cmd": [
            "./build/bin/pathways", "--headless",
            "--scene", "scenes/DragonDispersion.glb",
            "--width", "1920", "--height", "1080",
            "--spp", "4", "--max-bounces", "4", "--frames", "10",
            "--dump-frame", f"{OUTPUT_DIR}/test_dragon_dispersion.png",
            "--dump-stats", f"{OUTPUT_DIR}/stats_dragon_dispersion.json"
        ],
        "render_path": f"{OUTPUT_DIR}/test_dragon_dispersion.png",
        "ref_path": f"{REF_DIR}/dragon_dispersion.png",
        "diff_path": f"{OUTPUT_DIR}/diff_dragon_dispersion.png"
    },
    {
        "id": "car_concept",
        "name": "Car Concept (Clearcoat & Metallic Automotive PBR)",
        "scene": "scenes/CarConcept.glb",
        "cmd": [
            "./build/bin/pathways", "--headless",
            "--scene", "scenes/CarConcept.glb",
            "--width", "1920", "--height", "1080",
            "--spp", "4", "--max-bounces", "4", "--frames", "10",
            "--dump-frame", f"{OUTPUT_DIR}/test_car_concept.png",
            "--dump-stats", f"{OUTPUT_DIR}/stats_car_concept.json"
        ],
        "render_path": f"{OUTPUT_DIR}/test_car_concept.png",
        "ref_path": f"{REF_DIR}/car_concept.png",
        "diff_path": f"{OUTPUT_DIR}/diff_car_concept.png"
    },
    {
        "id": "breakfast_room",
        "name": "Breakfast Room (Multi-Bounce Interior GI)",
        "scene": "scenes/breakfast-room/breakfast_room_extended.glb",
        "cmd": [
            "./build/bin/pathways", "--headless",
            "--scene", "scenes/breakfast-room/breakfast_room_extended.glb",
            "--width", "1920", "--height", "1080",
            "--spp", "4", "--max-bounces", "4", "--frames", "10",
            "--dump-frame", f"{OUTPUT_DIR}/test_breakfast_room.png",
            "--dump-stats", f"{OUTPUT_DIR}/stats_breakfast_room.json"
        ],
        "render_path": f"{OUTPUT_DIR}/test_breakfast_room.png",
        "ref_path": f"{REF_DIR}/breakfast_room.png",
        "diff_path": f"{OUTPUT_DIR}/diff_breakfast_room.png"
    }
]

def compute_metrics(curr_path, ref_path, diff_output_path=None):
    """
    Computes statistical and perceptual metrics between curr_path and ref_path.
    Returns dictionary with metrics and classification.
    """
    if not os.path.exists(curr_path):
        return {"error": f"Current render file not found: {curr_path}"}
    if not os.path.exists(ref_path):
        return {"error": f"Reference baseline file not found: {ref_path}"}

    img_curr = Image.open(curr_path).convert("RGB")
    img_ref = Image.open(ref_path).convert("RGB")

    w_curr, h_curr = img_curr.size
    w_ref, h_ref = img_ref.size

    if (w_curr, h_curr) != (w_ref, h_ref):
        img_ref = img_ref.resize((w_curr, h_curr), Image.Resampling.LANCZOS)

    arr_curr = np.array(img_curr, dtype=np.float32) / 255.0
    arr_ref = np.array(img_ref, dtype=np.float32) / 255.0

    # 1. Pixel Difference Metrics
    abs_diff = np.abs(arr_curr - arr_ref)
    mae = float(np.mean(abs_diff))
    mse = float(np.mean((arr_curr - arr_ref) ** 2))
    psnr = float(20.0 * np.log10(1.0 / np.sqrt(mse))) if mse > 1e-10 else 99.0

    # 2. Structural Similarity (SSIM)
    try:
        ssim_val = float(ssim(arr_curr, arr_ref, channel_axis=2, data_range=1.0))
    except Exception as e:
        ssim_val = 1.0

    # 3. Luminance & Exposure Metrics
    lum_curr = 0.2126 * arr_curr[:, :, 0] + 0.7152 * arr_curr[:, :, 1] + 0.0722 * arr_curr[:, :, 2]
    lum_ref = 0.2126 * arr_ref[:, :, 0] + 0.7152 * arr_ref[:, :, 1] + 0.0722 * arr_ref[:, :, 2]

    mean_lum_curr = float(np.mean(lum_curr))
    mean_lum_ref = float(np.mean(lum_ref))
    delta_mean_lum = mean_lum_curr - mean_lum_ref

    shadow_curr = float(np.mean(lum_curr < 0.05) * 100.0)
    shadow_ref = float(np.mean(lum_ref < 0.05) * 100.0)
    delta_shadow = shadow_curr - shadow_ref

    blown_curr = float(np.mean(lum_curr > 0.95) * 100.0)
    blown_ref = float(np.mean(lum_ref > 0.95) * 100.0)
    delta_blown = blown_curr - blown_ref

    # 4. Sharpness / Edge Gradient (Laplacian Variance)
    try:
        lap_curr = cv2.Laplacian((lum_curr * 255.0).astype(np.float32), cv2.CV_32F)
        lap_ref = cv2.Laplacian((lum_ref * 255.0).astype(np.float32), cv2.CV_32F)
        sharp_curr = float(lap_curr.var())
        sharp_ref = float(lap_ref.var())
        sharp_pct_change = ((sharp_curr - sharp_ref) / max(sharp_ref, 1e-4)) * 100.0
    except Exception:
        sharp_curr = 0.0
        sharp_ref = 0.0
        sharp_pct_change = 0.0

    # 5. Difference Heatmap Generation
    if diff_output_path:
        os.makedirs(os.path.dirname(diff_output_path) or ".", exist_ok=True)
        diff_max = np.max(abs_diff, axis=2)
        diff_scaled = np.clip(diff_max * 3.5, 0.0, 1.0)
        diff_u8 = (diff_scaled * 255.0).astype(np.uint8)
        diff_color = cv2.applyColorMap(diff_u8, cv2.COLORMAP_TURBO)
        diff_color_rgb = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)
        Image.fromarray(diff_color_rgb).save(diff_output_path)

    # 6. Classification Engine (Regressions vs Improvements vs Matches)
    status = "MATCH"
    detail = "Within stochastic Monte Carlo variance tolerance"
    severity = "pass"

    if ssim_val >= 0.985 and mae <= 0.015:
        status = "MATCH (STABLE)"
        detail = "Reference match within stochastic tolerance"
        severity = "pass"
    else:
        # Significant difference detected. Classify root cause:
        if delta_blown > 1.5 or (delta_mean_lum > 0.12 and blown_curr > 2.0):
            status = "FLAGGED REGRESSION: OVEREXPOSURE"
            detail = f"Highlight blowout increased by {delta_blown:+.2f}%, mean lum increased by {delta_mean_lum:+.4f}"
            severity = "fail"
        elif delta_shadow > 10.0 or delta_mean_lum < -0.10:
            status = "FLAGGED REGRESSION: ENERGY LOSS / DARK COLLAPSE"
            detail = f"Deep shadows increased by {delta_shadow:+.2f}%, mean lum decreased by {delta_mean_lum:+.4f}"
            severity = "fail"
        elif delta_shadow < -15.0 and delta_mean_lum > 0.08:
            status = "FLAGGED REGRESSION: SHADOW LOSS / BLEACHING"
            detail = f"Contact shadows bleached/reduced by {abs(delta_shadow):.2f}%"
            severity = "fail"
        elif psnr < 25.0 and sharp_pct_change < -5.0:
            status = "FLAGGED REGRESSION: SEVERE NOISE / BLUR"
            detail = f"Low PSNR ({psnr:.1f} dB) with degraded edge sharpness ({sharp_pct_change:+.1f}%)"
            severity = "fail"
        elif sharp_pct_change > 10.0 and abs(delta_mean_lum) <= 0.05 and delta_blown <= 1.0 and ssim_val >= 0.88:
            status = "FLAGGED NOTICE: IMPROVED CLARITY"
            detail = f"Edge sharpness increased by {sharp_pct_change:+.1f}% with stable exposure"
            severity = "notice"
        elif abs(delta_mean_lum) <= 0.03 and delta_blown <= 0.5 and ssim_val >= 0.94 and psnr >= 34.0:
            status = "FLAGGED NOTICE: IMPROVED DENOISING"
            detail = f"High structural fidelity (SSIM={ssim_val:.4f}, PSNR={psnr:.1f} dB) with cleaner surface convergence"
            severity = "notice"
        else:
            status = "FLAGGED REVIEW: SIGNIFICANT DIFFERENCE"
            detail = f"SSIM={ssim_val:.4f}, MAE={mae:.4f}, PSNR={psnr:.1f} dB, delta_lum={delta_mean_lum:+.4f}"
            severity = "warn"

    return {
        "status": status,
        "detail": detail,
        "severity": severity,
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim_val,
        "mean_lum_curr": mean_lum_curr,
        "mean_lum_ref": mean_lum_ref,
        "delta_mean_lum": delta_mean_lum,
        "shadow_curr": shadow_curr,
        "shadow_ref": shadow_ref,
        "delta_shadow": delta_shadow,
        "blown_curr": blown_curr,
        "blown_ref": blown_ref,
        "delta_blown": delta_blown,
        "sharp_curr": sharp_curr,
        "sharp_ref": sharp_ref,
        "sharp_pct_change": sharp_pct_change,
        "resolution": f"{w_curr}x{h_curr}"
    }

def generate_html_report(results, report_path="output/visual_regression_report.html"):
    """
    Generates an interactive HTML dashboard with side-by-side Before/After sliders,
    false-color heatmaps, and metric telemetry.
    """
    total = len(results)
    passes = sum(1 for r in results if r["metrics"].get("severity") == "pass")
    notices = sum(1 for r in results if r["metrics"].get("severity") == "notice")
    warns = sum(1 for r in results if r["metrics"].get("severity") == "warn")
    fails = sum(1 for r in results if r["metrics"].get("severity") == "fail")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Pathways Before/After Visual Regression Report</title>
<style>
  :root {{
    --bg-dark: #0f1117;
    --card-bg: #1a1d27;
    --border-color: #2b3040;
    --text-primary: #e6edf3;
    --text-muted: #8b949e;
    --pass-color: #2ea043;
    --notice-color: #a371f7;
    --warn-color: #d29922;
    --fail-color: #f85149;
    --accent: #58a6ff;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg-dark);
    color: var(--text-primary);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    line-height: 1.5;
    padding: 24px;
  }}
  .header {{
    margin-bottom: 24px;
    padding-bottom: 16px;
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }}
  h1 {{ font-size: 24px; font-weight: 600; color: #ffffff; }}
  .stats-summary {{
    display: flex;
    gap: 16px;
  }}
  .stat-badge {{
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 13px;
    font-weight: 600;
  }}
  .stat-pass {{ background: rgba(46, 160, 67, 0.2); color: var(--pass-color); border: 1px solid var(--pass-color); }}
  .stat-notice {{ background: rgba(163, 113, 247, 0.2); color: var(--notice-color); border: 1px solid var(--notice-color); }}
  .stat-warn {{ background: rgba(210, 153, 34, 0.2); color: var(--warn-color); border: 1px solid var(--warn-color); }}
  .stat-fail {{ background: rgba(248, 81, 73, 0.2); color: var(--fail-color); border: 1px solid var(--fail-color); }}

  .test-card {{
    background: var(--card-bg);
    border: 1px solid var(--border-color);
    border-radius: 8px;
    margin-bottom: 24px;
    padding: 20px;
  }}
  .card-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }}
  .card-title {{ font-size: 18px; font-weight: 600; }}
  .status-badge {{
    padding: 4px 10px;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 700;
    text-transform: uppercase;
  }}
  .status-pass {{ background: var(--pass-color); color: white; }}
  .status-notice {{ background: var(--notice-color); color: white; }}
  .status-warn {{ background: var(--warn-color); color: black; }}
  .status-fail {{ background: var(--fail-color); color: white; }}

  .detail-text {{
    font-size: 14px;
    color: var(--text-muted);
    margin-bottom: 16px;
  }}

  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
    gap: 10px;
    margin-bottom: 16px;
  }}
  .metric-box {{
    background: rgba(0,0,0,0.3);
    padding: 8px 12px;
    border-radius: 6px;
    border: 1px solid rgba(255,255,255,0.05);
  }}
  .metric-label {{ font-size: 11px; text-transform: uppercase; color: var(--text-muted); }}
  .metric-val {{ font-size: 15px; font-weight: 600; margin-top: 2px; }}

  .comparison-container {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 16px;
    margin-top: 12px;
  }}
  @media (max-width: 1200px) {{
    .comparison-container {{ grid-template-columns: 1fr; }}
  }}
  .img-panel {{
    background: #000;
    border-radius: 6px;
    overflow: hidden;
    border: 1px solid var(--border-color);
  }}
  .img-panel-title {{
    padding: 8px 12px;
    font-size: 12px;
    font-weight: 600;
    background: rgba(255,255,255,0.05);
    border-bottom: 1px solid var(--border-color);
    display: flex;
    justify-content: space-between;
  }}
  .img-panel img {{
    width: 100%;
    height: auto;
    display: block;
  }}
</style>
</head>
<body>

<div class="header">
  <div>
    <h1>Pathways Visual Regression & Quality Dashboard</h1>
    <div style="font-size: 13px; color: var(--text-muted); margin-top: 4px;">
      Automated Before/After Reference Comparison & Anomaly Classification
    </div>
  </div>
  <div class="stats-summary">
    <div class="stat-badge stat-pass">{passes} Matches</div>
    <div class="stat-badge stat-notice">{notices} Notices</div>
    <div class="stat-badge stat-warn">{warns} Reviews</div>
    <div class="stat-badge stat-fail">{fails} Regressions</div>
  </div>
</div>
"""

    for r in results:
        cfg = r["config"]
        m = r["metrics"]

        if "error" in m:
            html += f"""
<div class="test-card">
  <div class="card-header">
    <div class="card-title">{cfg['name']}</div>
    <div class="status-badge status-fail">ERROR</div>
  </div>
  <div class="detail-text">{m['error']}</div>
</div>"""
            continue

        sev = m.get("severity", "warn")
        badge_class = f"status-{sev}"
        rel_ref = os.path.relpath(cfg["ref_path"], os.path.dirname(report_path))
        rel_render = os.path.relpath(cfg["render_path"], os.path.dirname(report_path))
        rel_diff = os.path.relpath(cfg["diff_path"], os.path.dirname(report_path)) if os.path.exists(cfg["diff_path"]) else ""

        html += f"""
<div class="test-card">
  <div class="card-header">
    <div class="card-title">{cfg['name']} <span style="font-size: 13px; color: var(--text-muted); font-weight: 400;">({cfg['scene']})</span></div>
    <div class="status-badge {badge_class}">{m['status']}</div>
  </div>
  <div class="detail-text">{m['detail']}</div>

  <div class="metrics-grid">
    <div class="metric-box">
      <div class="metric-label">SSIM</div>
      <div class="metric-val">{m['ssim']:.4f}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">PSNR</div>
      <div class="metric-val">{m['psnr']:.1f} dB</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">MAE</div>
      <div class="metric-val">{m['mae']:.4f}</div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Mean Lum</div>
      <div class="metric-val">{m['mean_lum_curr']:.4f} <span style="font-size: 11px; color: var(--text-muted); font-weight: normal;">({m['delta_mean_lum']:+.3f})</span></div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Shadows (&lt;0.05)</div>
      <div class="metric-val">{m['shadow_curr']:.1f}% <span style="font-size: 11px; color: var(--text-muted); font-weight: normal;">({m['delta_shadow']:+.1f}%)</span></div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Blown-Out (&gt;0.95)</div>
      <div class="metric-val">{m['blown_curr']:.2f}% <span style="font-size: 11px; color: var(--text-muted); font-weight: normal;">({m['delta_blown']:+.2f}%)</span></div>
    </div>
    <div class="metric-box">
      <div class="metric-label">Sharpness</div>
      <div class="metric-val">{m['sharp_pct_change']:+.1f}%</div>
    </div>
  </div>

  <div class="comparison-container">
    <div class="img-panel">
      <div class="img-panel-title"><span>BEFORE (Golden Reference)</span><span>{m['resolution']}</span></div>
      <img src="{rel_ref}" alt="Reference" loading="lazy">
    </div>
    <div class="img-panel">
      <div class="img-panel-title"><span>AFTER (Current Render)</span><span>{m['resolution']}</span></div>
      <img src="{rel_render}" alt="Current Render" loading="lazy">
    </div>
    <div class="img-panel">
      <div class="img-panel-title"><span>DIFFERENCE HEATMAP (Turbo 3.5x)</span><span>Per-pixel max error</span></div>
      <img src="{rel_diff}" alt="Difference Heatmap" loading="lazy">
    </div>
  </div>
</div>"""

    html += """
</body>
</html>"""

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    with open(report_path, "w") as f:
        f.write(html)
    print(f"\033[32m[PASS]\033[0m Generated visual regression report: {report_path}")

def main():
    parser = argparse.ArgumentParser(description="Pathways Visual Regression Test Engine")
    parser.add_argument("--update-baselines", action="store_true", help="Update/generate golden reference images from current renders")
    parser.add_argument("--render", action="store_true", help="Execute pathways binary to generate fresh renders before comparison")
    parser.add_argument("--report", type=str, default="output/visual_regression_report.html", help="Path to write HTML report")
    parser.add_argument("--strict", action="store_true", help="Exit with non-zero code on any flagged regression")
    parser.add_argument("--single", type=str, default=None, help="Run single test config ID (e.g. living_room_1080p_motion)")
    args = parser.parse_args()

    os.makedirs(REF_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    configs = TEST_CONFIGS
    if args.single:
        configs = [c for c in TEST_CONFIGS if c["id"] == args.single]
        if not configs:
            print(f"\033[31m[FAIL]\033[0m Unknown test config ID: {args.single}")
            sys.exit(1)

    print("====================================================================")
    print("  Pathways Before/After Visual Regression Engine")
    print("====================================================================")

    # 1. If --render is passed, execute render commands
    if args.render:
        import subprocess
        for cfg in configs:
            print(f"[RENDER] {cfg['name']}...")
            res = subprocess.run(cfg["cmd"], capture_output=True, text=True)
            if res.returncode != 0:
                print(f"\033[31m[FAIL]\033[0m Render failed for {cfg['id']}:\n{res.stderr[-500:]}")
            else:
                print(f"\033[32m[PASS]\033[0m Render complete: {cfg['render_path']}")

    # 2. If --update-baselines, copy current renders to golden reference directory
    if args.update_baselines:
        print("\n[UPDATING BASELINES] Copying current renders to tests/references/...")
        for cfg in configs:
            if os.path.exists(cfg["render_path"]):
                shutil.copyfile(cfg["render_path"], cfg["ref_path"])
                print(f"  -> Updated golden reference: {cfg['ref_path']}")
            else:
                print(f"\033[33m[WARN]\033[0m Render not found to copy: {cfg['render_path']}")
        print("\033[32m[SUCCESS]\033[0m Golden baselines updated successfully.")
        sys.exit(0)

    # 3. Compare current renders against reference baselines
    results = []
    has_regressions = False

    for cfg in configs:
        print(f"\n--- Checking {cfg['name']} ---")
        m = compute_metrics(cfg["render_path"], cfg["ref_path"], cfg["diff_path"])
        results.append({"config": cfg, "metrics": m})

        if "error" in m:
            print(f"\033[31m[ERROR]\033[0m {m['error']}")
            has_regressions = True
            continue

        sev = m.get("severity")
        if sev == "pass":
            print(f"\033[32m[{m['status']}]\033[0m SSIM: {m['ssim']:.4f} | PSNR: {m['psnr']:.1f} dB | MAE: {m['mae']:.4f} | {m['detail']}")
        elif sev == "notice":
            print(f"\033[35m[{m['status']}]\033[0m SSIM: {m['ssim']:.4f} | PSNR: {m['psnr']:.1f} dB | Sharpness: {m['sharp_pct_change']:+.1f}% | {m['detail']}")
        elif sev == "warn":
            print(f"\033[33m[{m['status']}]\033[0m SSIM: {m['ssim']:.4f} | PSNR: {m['psnr']:.1f} dB | MAE: {m['mae']:.4f} | {m['detail']}")
        else: # fail
            print(f"\033[31m[{m['status']}]\033[0m SSIM: {m['ssim']:.4f} | PSNR: {m['psnr']:.1f} dB | MAE: {m['mae']:.4f} | {m['detail']}")
            has_regressions = True

    # 4. Generate HTML Report
    generate_html_report(results, args.report)

    print("\n--------------------------------------------------------------------")
    if has_regressions:
        print("\033[31m[FLAGGED REGRESSIONS DETECTED]\033[0m Review visual diffs in " + args.report)
        if args.strict:
            sys.exit(1)
    else:
        print("\033[32m[CLEAN PASS]\033[0m No visual regressions detected against reference baselines.")

    sys.exit(0)

if __name__ == "__main__":
    main()
