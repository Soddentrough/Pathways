#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "=========================================================="
echo "  Pathways: Automated Headless Test Suite & Frame Capture"
echo "=========================================================="

# 1. Check GPU utilization & VRAM status per user rules
echo "[1/4] Checking AMD GPU metrics via amd-smi..."
if [ -x /opt/rocm/core-10.0/bin/amd-smi ]; then
    /opt/rocm/core-10.0/bin/amd-smi || true
elif [ -x /home/naoki/.local/bin/amd-smi ]; then
    /home/naoki/.local/bin/amd-smi || true
elif command -v amd-smi &> /dev/null; then
    amd-smi || true
else
    echo "amd-smi not found in known paths, continuing..."
fi

BUILD_DIR="${BUILD_DIR:-build}"
if [ ! -f "${BUILD_DIR}/build.ninja" ] && [ -f "build/linux-release/build.ninja" ]; then
    BUILD_DIR="build/linux-release"
fi
PATHWAYS_BIN="./${BUILD_DIR}/bin/pathways"

# 2. Build / ensure binaries are up to date (limiting threads for Threadripper 3750X)
echo ""
echo "[2/4] Verifying build with ninja (-j16)..."
ninja -C "${BUILD_DIR}" -j16

# Create output directory
mkdir -p output

# 2b. Run CTest unit test suites (Camera controls, ImGui headless, Shadow denoiser, TAA/A-Trous, Telemetry)
echo ""
echo "[2b] Running CTest Unit Test Suites..."
ctest --test-dir "${BUILD_DIR}" --output-on-failure

# 2c. Run Dynamic Scene Switching Tests
echo ""
echo "[2c] Running Dynamic Scene Switching Tests..."
"${PATHWAYS_BIN}" --test-scene-switching

# 3. Test Suite 1: 1080p 16 SPP Full Quality Verification
echo ""
echo "[3/7] Running Test Suite 1: 1080p @ 16 SPP (PNG + OpenEXR + Stats)..."
"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 16 \
    --max-bounces 4 \
    --dump-frame output/test_cornell_1080p.png \
    --dump-hdr output/test_cornell_1080p.exr \
    --dump-stats output/stats_1080p.json

python3 scripts/verify_frame.py output/test_cornell_1080p.png output/stats_1080p.json 1920 1080 40.0 --max-mean-lum 0.85 --max-blown-pct 20.0

# 4. Test Suite 2: 4K Native Real-Time Benchmark (<8ms Target)
echo ""
echo "[4/5] Running Test Suite 2: 4K Native (3840x2160) @ 1 SPP (Benchmark Mode)..."
"${PATHWAYS_BIN}" \
    --headless \
    --width 3840 \
    --height 2160 \
    --spp 1 \
    --max-bounces 4 \
    --frames 200 \
    --warmup-frames 30 \
    --no-accumulation \
    --benchmark \
    --dump-frame output/test_cornell_4k.png \
    --dump-stats output/stats_4k.json

python3 scripts/verify_frame.py output/test_cornell_4k.png output/stats_4k.json 3840 2160 10.0 --max-mean-lum 0.85 --max-blown-pct 12.0

# 4b. Test Suite 2b: 4K Native Multi-GPU Interleaved Scanlines (1 SPP, Sub-8ms Target)
echo ""
echo "[4b] Running Test Suite 2b: 4K Native Interleaved Scanlines (Dual R9700, 1 SPP)..."
"${PATHWAYS_BIN}" \
    --headless \
    --width 3840 \
    --height 2160 \
    --spp 1 \
    --max-bounces 4 \
    --frames 200 \
    --warmup-frames 30 \
    --no-accumulation \
    --mgpu \
    --dump-frame output/test_cornell_4k_mgpu_interleaved.png \
    --dump-stats output/stats_4k_mgpu_interleaved.json

python3 scripts/verify_frame.py output/test_cornell_4k_mgpu_interleaved.png output/stats_4k_mgpu_interleaved.json 3840 2160 8.0 --max-mean-lum 0.85 --max-blown-pct 12.0

# 4c. Test Suite 2c: 4K Native Multi-GPU Frame Pacing & Camera Motion Regression Test
echo ""
echo "[4c] Running Test Suite 2c: 4K Native Multi-GPU Frame Pacing (Camera Motion, Host Zero-Copy)..."
"${PATHWAYS_BIN}" \
    --headless \
    --width 3840 \
    --height 2160 \
    --spp 1 \
    --max-bounces 4 \
    --frames 60 \
    --camera-motion \
    --mgpu \
    --dump-frame output/test_cornell_4k_mgpu_motion.png \
    --dump-stats output/stats_4k_mgpu_motion.json

python3 scripts/verify_mgpu_pacing.py output/stats_4k_mgpu_motion.json 8.0 10.0 60

# 4d. Test Suite 2d: Multi-GPU + FSR 3.1 Dynamic Camera Motion & Seam Test (1440p Quality)
echo ""
echo "[4d] Running Test Suite 2d: Multi-GPU + FSR 3.1 Dynamic Camera Motion (1440p Quality, Dual R9700)..."
"${PATHWAYS_BIN}" \
    --headless \
    --scene scenes/classroom/classroom_extended.glb \
    --width 2560 \
    --height 1440 \
    --spp 1 \
    --max-bounces 4 \
    --frames 30 \
    --camera-motion \
    --mgpu \
    --upscaler fsr3 \
    --dump-frame output/test_classroom_mgpu_tile_fsr3_motion.png \
    --dump-stats output/stats_classroom_mgpu_tile_fsr3_motion.json

python3 scripts/verify_mgpu_fsr3_motion.py output/test_classroom_mgpu_tile_fsr3_motion.png output/stats_classroom_mgpu_tile_fsr3_motion.json 2560 1440 3.5

# 5. Test Suite 3: Multi-GPU Sample Parallelism (Dual Radeon AI PRO R9700)
echo ""
echo "[4/5] Running Test Suite 3: Multi-GPU Sample Parallelism (Dual R9700 @ PCIe 5.0 x16)..."
"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 8 \
    --max-bounces 4 \
    --mgpu-mode sample \
    --dump-frame output/test_mgpu_sample.png \
    --dump-stats output/stats_mgpu_sample.json

python3 scripts/verify_frame.py output/test_mgpu_sample.png output/stats_mgpu_sample.json 1920 1080 30.0 --max-mean-lum 0.85 --max-blown-pct 12.0

# 6. Test Suite 4: Multi-GPU Scaling Benchmark (Verify >=1.8x Speedup)
echo ""
echo "[5/6] Running Test Suite 4: Multi-GPU Scaling Verification (Single vs Dual GPU >= 1.8x)..."
"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 16 \
    --max-bounces 4 \
    --frames 100 \
    --warmup-frames 20 \
    --no-accumulation \
    --scene cornell-box \
    --mgpu-mode off \
    --dump-stats output/stats_scaling_single.json

"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 16 \
    --max-bounces 4 \
    --frames 100 \
    --warmup-frames 20 \
    --no-accumulation \
    --scene cornell-box \
    --mgpu-mode sample \
    --dump-stats output/stats_scaling_multi.json

# Realistic scaling threshold: PCIe 4.0 x8 secondary link + cross-GPU merge overhead
python3 scripts/verify_scaling.py output/stats_scaling_single.json output/stats_scaling_multi.json 1.65

# 7. Test Suite 5: glTF 2.0 Ingestion Pipeline & Auto-Framing Verification
echo ""
echo "[6a] Running Test Suite 5: glTF 2.0 Ingestion & Verification (Damaged Helmet)..."

"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 16 \
    --max-bounces 4 \
    --scene scenes/DamagedHelmet.glb \
    --dump-frame output/test_gltf_helmet.png \
    --dump-stats output/stats_gltf_helmet.json

python3 scripts/verify_frame.py output/test_gltf_helmet.png output/stats_gltf_helmet.json 1920 1080 45.0 --max-mean-lum 0.65 --max-blown-pct 2.0

# 8. Test Suite 6: Extreme Scenes & Dielectric Transmission Stress Test (Cornell Caustic & Glass of Water)
echo ""
echo "[6b] Running Test Suite 6: Extreme Scenes Regression Verification (Caustics & Glass of Water)..."
"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 4 \
    --max-bounces 4 \
    --frames 10 \
    --scene scenes/cornell-caustic/cornell_caustic_extended.glb \
    --dump-frame output/test_cornell_caustic.png \
    --dump-stats output/stats_cornell_caustic.json

python3 scripts/verify_frame.py output/test_cornell_caustic.png output/stats_cornell_caustic.json 1920 1080 30.0 --max-mean-lum 0.85 --max-blown-pct 15.0

"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 4 \
    --max-bounces 4 \
    --frames 10 \
    --mgpu-mode sample \
    --scene scenes/glass-of-water/glass_of_water_extended.glb \
    --dump-frame output/test_glass_of_water.png \
    --dump-stats output/stats_glass_of_water.json

python3 scripts/verify_frame.py output/test_glass_of_water.png output/stats_glass_of_water.json 1920 1080 30.0 --max-mean-lum 0.85 --max-blown-pct 10.0

# 9. Test Suite 6c: Curated glTF 2.0 Extensions & Research Scenes (Dragon Dispersion, Car Concept, Breakfast Room, Veach Ajar)
echo ""
echo "[6c] Running Test Suite 6c: Curated glTF 2.0 Extensions & Research Scenes..."
"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 4 \
    --max-bounces 4 \
    --frames 10 \
    --scene scenes/DragonDispersion.glb \
    --dump-frame output/test_dragon_dispersion.png \
    --dump-stats output/stats_dragon_dispersion.json

python3 scripts/verify_frame.py output/test_dragon_dispersion.png output/stats_dragon_dispersion.json 1920 1080 30.0 --max-mean-lum 0.85 --max-blown-pct 25.0

"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 4 \
    --max-bounces 4 \
    --frames 10 \
    --scene scenes/bmw-m6/bmw_m6_extended.glb \
    --dump-frame output/test_bmw_m6.png \
    --dump-stats output/stats_bmw_m6.json

python3 scripts/verify_frame.py output/test_bmw_m6.png output/stats_bmw_m6.json 1920 1080 20.0 --max-mean-lum 0.95 --max-blown-pct 50.0

"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 4 \
    --max-bounces 4 \
    --frames 10 \
    --scene scenes/breakfast-room/breakfast_room_extended.glb \
    --dump-frame output/test_breakfast_room.png \
    --dump-stats output/stats_breakfast_room.json

python3 scripts/verify_frame.py output/test_breakfast_room.png output/stats_breakfast_room.json 1920 1080 35.0 --max-mean-lum 0.50 --max-blown-pct 8.0

"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 4 \
    --max-bounces 4 \
    --frames 10 \
    --scene scenes/veach-ajar/veach_ajar_extended.glb \
    --dump-frame output/test_veach_ajar.png \
    --dump-stats output/stats_veach_ajar.json

python3 scripts/verify_frame.py output/test_veach_ajar.png output/stats_veach_ajar.json 1920 1080 30.0 --max-mean-lum 0.85 --max-blown-pct 25.0

# 9b. Test Suite 6d: Many-Lights (64 Lights) Procedural Cornell Box (Alias Table & Local RIS)
echo ""
echo "[6d] Running Test Suite 6d: Many-Lights Scene (64 Lights)..."
"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 4 \
    --max-bounces 4 \
    --scene many-lights \
    --dump-frame output/test_many_lights.png \
    --dump-stats output/stats_many_lights.json

python3 scripts/verify_frame.py output/test_many_lights.png output/stats_many_lights.json 1920 1080 40.0

# 9c. Test Suite 6e: Hierarchical Light Tree (FEAT-02) Many-Lights Verification
echo ""
echo "[6e] Running Test Suite 6e: Many-Lights Scene with Hierarchical Light Tree (--light-tree)..."
"${PATHWAYS_BIN}" \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 4 \
    --max-bounces 4 \
    --scene many-lights \
    --light-tree \
    --dump-frame output/test_many_lights_tree.png \
    --dump-stats output/stats_many_lights_tree.json

python3 scripts/verify_frame.py output/test_many_lights_tree.png output/stats_many_lights_tree.json 1920 1080 40.0

# 9d. Test Suite 6f: OpenUSD Scene Ingestion & Regression Verification (Buick Riviera)
if [ -f "scenes/BuickRiviera/BuickRiviera.usdc" ]; then
    echo ""
    echo "[6f] Running Test Suite 6f: OpenUSD Scene Ingestion & Path Tracing (Buick Riviera)..."
    "${PATHWAYS_BIN}" \
        --headless \
        --width 1920 \
        --height 1080 \
        --spp 4 \
        --max-bounces 4 \
        --frames 2 \
        --scene scenes/BuickRiviera/BuickRiviera.usdc \
        --dump-frame output/test_buick_usd.png \
        --dump-stats output/stats_buick_usd.json

    python3 scripts/verify_frame.py output/test_buick_usd.png output/stats_buick_usd.json 1920 1080 30.0 --max-mean-lum 0.85 --max-blown-pct 15.0
fi

# 9e. Test Suite 6g: Procedural Cyber-City Megastructure & Multi-GPU 4K Real-Time Benchmark
echo ""
echo "[6g] Running Test Suite 6g: Procedural Cyber-City Multi-GPU 4K Real-Time Benchmark..."
"${PATHWAYS_BIN}" \
    --headless \
    --scene cyber-city \
    --width 3840 \
    --height 2160 \
    --frames 60 \
    --benchmark \
    --mgpu \
    --dump-frame output/test_cyber_city_4k.png \
    --dump-stats output/stats_cyber_city_4k.json

python3 scripts/verify_frame.py output/test_cyber_city_4k.png output/stats_cyber_city_4k.json 3840 2160 10.0 --max-mean-lum 0.85 --max-blown-pct 25.0

# 9f. Test Suite 6h: Procedural Cyber-City Dynamic Multi-Frame Video Hologram E2E Verification
echo ""
echo "[6h] Running Test Suite 6h: Cyber-City Dynamic Multi-Frame Video Hologram Playback & Terrace View..."
"${PATHWAYS_BIN}" \
    --headless \
    --scene cyber-city \
    --camera -6.8,29.0,36.5,2.0,25.0,-40.0,62.0 \
    --width 1920 \
    --height 1080 \
    --frames 195 \
    --dump-frame output/test_cyber_city_terrace.png \
    --dump-stats output/stats_cyber_city_terrace.json

python3 scripts/verify_frame.py output/test_cyber_city_terrace.png output/stats_cyber_city_terrace.json 1920 1080 6.0

python3 tests/e2e/test_cyber_city_video_playback.py

# 10. Test Suite 7: Image Quality, Shadow Retention & Camera Motion Stability
echo ""
echo "[7/8] Running Test Suite 7: Image Quality, Shadow Retention & Motion Stability..."
python3 tests/test_image_quality.py

# 11. Test Suite 8: Automated Before/After Visual Regression Verification
echo ""
echo "[8/9] Running Test Suite 8: Visual Regression Verification against Golden References..."
python3 scripts/visual_regression_test.py --strict

# 12. Test Suite 9: Visual Integrity, Accumulation Exposure Stability & Multi-GPU Seams
echo ""
echo "[9/9] Running Test Suite 9: Visual Integrity & Exposure Stability..."
python3 scripts/verify_visual_integrity.py

echo ""
echo "=========================================================="
echo "  Pathways: All Automated Headless Tests Passed Successfully!"
echo "=========================================================="

