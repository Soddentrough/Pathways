#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${ROOT_DIR}"

echo "======================================================================"
echo "  Pathways: Scanlands & Point Instancing E2E Automated Test Suite"
echo "======================================================================"

# 1. Inspect Single-GPU VRAM utilization and metrics via amd-smi per user rules
echo ""
echo "[1/4] Checking AMD GPU Hardware Metrics & Single-GPU VRAM Status..."
if [ -x /opt/rocm/core-10.0/bin/amd-smi ]; then
    /opt/rocm/core-10.0/bin/amd-smi metric --mem-usage
elif [ -x /home/naoki/.local/bin/amd-smi ]; then
    /home/naoki/.local/bin/amd-smi metric --mem-usage
elif command -v amd-smi &> /dev/null; then
    amd-smi metric --mem-usage
else
    echo "amd-smi not found in known paths, continuing..."
fi

# 2. Verify project build (capping parallel threads at -j16 per user rules)
echo ""
echo "[2/4] Verifying build with ninja (-j16)..."
ninja -C build -j16

# 3. Execute CTest Unit Test Suite
echo ""
echo "[3/4] Running CTest Unit Test Suite (UsdLoader, ProceduralCyberCity, Camera, Denoisers)..."
ctest --test-dir build --output-on-failure

# 4. Execute Multi-Tier E2E Capabilities Test Suite (Tiers 1-4, 145 Tests)
echo ""
echo "[4/4] Executing Comprehensive Multi-Tier E2E Test Suite (F1-F13)..."
python3 tests/e2e/test_scanlands_capabilities.py

echo ""
echo "======================================================================"
echo "  Pathways: All Scanlands E2E Tests Completed Successfully (100%)"
echo "======================================================================"
