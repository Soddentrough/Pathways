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
if command -v amd-smi &> /dev/null; then
    amd-smi
elif [ -x /home/naoki/.local/bin/amd-smi ]; then
    /home/naoki/.local/bin/amd-smi
elif [ -x /opt/rocm/core-10.0/bin/amd-smi ]; then
    /opt/rocm/core-10.0/bin/amd-smi
elif [ -x /opt/rocm/bin/amd-smi ]; then
    /opt/rocm/bin/amd-smi
else
    echo "amd-smi not found in known paths, continuing..."
fi

# 2. Build / ensure binaries are up to date (limiting threads for Threadripper 3750X)
echo ""
echo "[2/4] Verifying build with ninja (-j8)..."
ninja -C build -j8

# Create output directory
mkdir -p output

# 2b. Run camera controls unit test suite
echo ""
echo "[2b] Running Camera Controls Unit Tests..."
./build/bin/test_camera_controls

# 2c. Run Dynamic Scene Switching Tests
echo ""
echo "[2c] Running Dynamic Scene Switching Tests..."
./build/bin/pathways --test-scene-switching

# 3. Test Suite 1: 1080p 16 SPP Full Quality Verification
echo ""
echo "[3/7] Running Test Suite 1: 1080p @ 16 SPP (PNG + OpenEXR + Stats)..."
./build/bin/pathways \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 16 \
    --max-bounces 4 \
    --dump-frame output/test_cornell_1080p.png \
    --dump-hdr output/test_cornell_1080p.exr \
    --dump-stats output/stats_1080p.json

python3 scripts/verify_frame.py output/test_cornell_1080p.png output/stats_1080p.json 1920 1080 40.0

# 4. Test Suite 2: 4K Native Real-Time Benchmark (<8ms Target)
echo ""
echo "[4/5] Running Test Suite 2: 4K Native (3840x2160) @ 1 SPP (Benchmark Mode)..."
./build/bin/pathways \
    --headless \
    --width 3840 \
    --height 2160 \
    --spp 1 \
    --max-bounces 4 \
    --frames 5 \
    --benchmark \
    --dump-frame output/test_cornell_4k.png \
    --dump-stats output/stats_4k.json

python3 scripts/verify_frame.py output/test_cornell_4k.png output/stats_4k.json 3840 2160 10.0

# 4b. Test Suite 2b: 4K Native Multi-GPU Interleaved Scanlines (1 SPP, Sub-8ms Target)
echo ""
echo "[4b] Running Test Suite 2b: 4K Native Interleaved Scanlines (Dual R9700, 1 SPP)..."
./build/bin/pathways \
    --headless \
    --width 3840 \
    --height 2160 \
    --spp 1 \
    --max-bounces 4 \
    --frames 5 \
    --mgpu \
    --dump-frame output/test_cornell_4k_mgpu_interleaved.png \
    --dump-stats output/stats_4k_mgpu_interleaved.json

python3 scripts/verify_frame.py output/test_cornell_4k_mgpu_interleaved.png output/stats_4k_mgpu_interleaved.json 3840 2160 8.0

# 5. Test Suite 3: Multi-GPU Sample Parallelism (Dual Radeon AI PRO R9700)
echo ""
echo "[4/5] Running Test Suite 3: Multi-GPU Sample Parallelism (Dual R9700 @ PCIe 5.0 x16)..."
./build/bin/pathways \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 8 \
    --max-bounces 4 \
    --mgpu-mode sample \
    --dump-frame output/test_mgpu_sample.png \
    --dump-stats output/stats_mgpu_sample.json

python3 scripts/verify_frame.py output/test_mgpu_sample.png output/stats_mgpu_sample.json 1920 1080 30.0

# 6. Test Suite 4: Multi-GPU Scaling Benchmark (Verify >=1.8x Speedup)
echo ""
echo "[5/6] Running Test Suite 4: Multi-GPU Scaling Verification (Single vs Dual GPU >= 1.8x)..."
./build/bin/pathways \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 16 \
    --max-bounces 4 \
    --frames 10 \
    --mgpu-mode off \
    --dump-stats output/stats_scaling_single.json

./build/bin/pathways \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 16 \
    --max-bounces 4 \
    --frames 10 \
    --mgpu-mode sample \
    --dump-stats output/stats_scaling_multi.json

python3 scripts/verify_scaling.py output/stats_scaling_single.json output/stats_scaling_multi.json 1.80

# 7. Test Suite 5: glTF 2.0 Ingestion Pipeline & Auto-Framing Verification
echo ""
echo "[6/6] Running Test Suite 5: glTF 2.0 Ingestion & Verification (Cornell & Shapes)..."
python3 scripts/generate_test_gltf.py

./build/bin/pathways \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 16 \
    --max-bounces 4 \
    --scene scenes/cornell_box.gltf \
    --dump-frame output/test_gltf_cornell.png \
    --dump-stats output/stats_gltf_cornell.json

python3 scripts/verify_frame.py output/test_gltf_cornell.png output/stats_gltf_cornell.json 1920 1080 45.0

./build/bin/pathways \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 16 \
    --max-bounces 4 \
    --mgpu-mode sample \
    --scene scenes/test_shapes.gltf \
    --dump-frame output/test_gltf_shapes.png \
    --dump-stats output/stats_gltf_shapes.json

python3 scripts/verify_frame.py output/test_gltf_shapes.png output/stats_gltf_shapes.json 1920 1080 15.0

# 8. Test Suite 6: glTF 2.0 Full PBR Pipeline (MR, Normal, AO, Emissive, Alpha Mask, Glass)
echo ""
echo "[7/7] Running Test Suite 6: Full glTF 2.0 PBR Pipeline Verification (Showcase Scene)..."
python3 scripts/generate_pbr_test_scene.py

./build/bin/pathways \
    --headless \
    --width 1920 \
    --height 1080 \
    --spp 16 \
    --max-bounces 4 \
    --scene scenes/pbr_showcase.gltf \
    --dump-frame output/test_pbr_showcase.png \
    --dump-stats output/stats_pbr_showcase.json

python3 scripts/verify_frame.py output/test_pbr_showcase.png output/stats_pbr_showcase.json 1920 1080 35.0

echo ""
echo "=========================================================="
echo "  Pathways: All Automated Headless Tests Passed Successfully!"
echo "=========================================================="
