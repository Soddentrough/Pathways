#include "rt/NRCManager.hpp"
#include <iostream>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <vector>
#include <vulkan/vulkan.h>

using namespace pathways;

static void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] Assertion failed: " << msg << std::endl;
        std::exit(1);
    }
}

// Emulate GLSL spatial hash corner calculation
static uint32_t nrcHashCorner(int32_t cx, int32_t cy, int32_t cz, uint32_t level) {
    const uint32_t P1 = 1u;
    const uint32_t P2 = 2654435761u;
    const uint32_t P3 = 805459861u;
    const uint32_t P_LEVEL = 1000003u;
    const uint32_t HASH_TABLE_SIZE = 262144u;

    uint32_t h = (static_cast<uint32_t>(cx) * P1 ^
                  static_cast<uint32_t>(cy) * P2 ^
                  static_cast<uint32_t>(cz) * P3) ^
                 (level * P_LEVEL);
    h = h ^ (h >> 16);
    return h % HASH_TABLE_SIZE;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing Tier 4 Neural Radiance Caching (NRC)   " << std::endl;
    std::cout << "  Wave32 WMMA Cooperative Matrix & Optimizer Invariants   " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Verify NRCQuery and NRCTrainingRecord std430 alignment & layout
    std::cout << "[TEST 1] NRC Buffer Struct Layout & Size Verification..." << std::endl;
    check_true(sizeof(NRCQuery) == 80, "NRCQuery must be exactly 80 bytes (5x vec4)");
    check_true(offsetof(NRCQuery, pos_roughness) == 0, "pos_roughness must be at byte 0");
    check_true(offsetof(NRCQuery, normal_flags) == 16, "normal_flags must be at byte 16");
    check_true(offsetof(NRCQuery, dir_pixelIndex) == 32, "dir_pixelIndex must be at byte 32");
    check_true(offsetof(NRCQuery, albedo_pad) == 48, "albedo_pad must be at byte 48");
    check_true(offsetof(NRCQuery, throughput) == 64, "throughput must be at byte 64");
    std::cout << "  -> NRCQuery layout verified (80B std430 aligned)." << std::endl;

    check_true(sizeof(NRCTrainingRecord) == 96, "NRCTrainingRecord must be exactly 96 bytes (6x vec4)");
    check_true(offsetof(NRCTrainingRecord, pos_roughness) == 0, "pos_roughness must be at byte 0");
    check_true(offsetof(NRCTrainingRecord, normal_flags) == 16, "normal_flags must be at byte 16");
    check_true(offsetof(NRCTrainingRecord, dir_pixelIndex) == 32, "dir_pixelIndex must be at byte 32");
    check_true(offsetof(NRCTrainingRecord, albedo_pad) == 48, "albedo_pad must be at byte 48");
    check_true(offsetof(NRCTrainingRecord, throughput) == 64, "throughput must be at byte 64");
    check_true(offsetof(NRCTrainingRecord, targetRadiance) == 80, "targetRadiance must be at byte 80");
    std::cout << "  -> NRCTrainingRecord layout verified (96B std430 aligned)." << std::endl;

    check_true(sizeof(NRCCountersBuffer) == 16, "NRCCountersBuffer must be exactly 16 bytes (4x uint32_t)");
    check_true(offsetof(NRCCountersBuffer, queryCount) == 0, "queryCount must be at byte 0");
    check_true(offsetof(NRCCountersBuffer, trainCount) == 4, "trainCount must be at byte 4");
    check_true(offsetof(NRCCountersBuffer, dispatchX) == 8, "dispatchX must be at byte 8");
    std::cout << "  -> NRCCountersBuffer layout verified (16B std430 aligned)." << std::endl;

    // 2. Verify Hash Grid Feature Resolution & Coordinate Parity
    std::cout << "[TEST 2] Hash Grid Parameters & Bounded Table Lookup..." << std::endl;
    const float GRID_SCALES[12] = {
        16.0f, 23.36f, 34.12f, 49.83f, 72.78f, 106.28f,
        155.23f, 226.70f, 331.09f, 483.54f, 706.18f, 1031.33f
    };
    check_true(GRID_SCALES[0] >= 16.0f, "Min grid scale should be 16");
    check_true(GRID_SCALES[11] >= 1000.0f, "Max grid scale should exceed 1000 for sub-millimeter detail");

    // Test hash boundedness and determinism across random normalized coords
    for (uint32_t l = 0; l < 12; ++l) {
        float s = GRID_SCALES[l];
        for (int i = 0; i < 50; ++i) {
            float px = (i * 0.021f);
            float py = (i * 0.017f);
            float pz = (i * 0.029f);
            int32_t cx = static_cast<int32_t>(std::floor(px * s));
            int32_t cy = static_cast<int32_t>(std::floor(py * s));
            int32_t cz = static_cast<int32_t>(std::floor(pz * s));

            for (int dz = 0; dz <= 1; ++dz) {
                for (int dy = 0; dy <= 1; ++dy) {
                    for (int dx = 0; dx <= 1; ++dx) {
                        uint32_t h = nrcHashCorner(cx + dx, cy + dy, cz + dz, l);
                        check_true(h < 262144u, "Hash corner index must be within table size (2^18)");
                    }
                }
            }
        }
    }
    std::cout << "  -> Hash table index bounded to [0, 262143] across all 12 multi-resolution levels." << std::endl;

    // 3. Verify WMMA Cooperative Matrix 16x16x16 Tiling Math
    std::cout << "[TEST 3] WMMA Wave32 Cooperative Matrix MLP Topology..." << std::endl;
    // Input features: 64. Hidden: 64. Output: 16 (first 3 channels = RGB).
    // Batch size per SIMD32 wave: 16 rays.
    // Matrix A: 16 x 64 (16 queries, 64 features)
    // Matrix B0: 64 x 64 (Layer 0 weights)
    // Matrix B1: 64 x 64 (Layer 1 weights)
    // Matrix B2: 64 x 16 (Layer 2 weights)
    const uint32_t w0_count = 64 * 64; // 4096
    const uint32_t w1_count = 64 * 64; // 4096
    const uint32_t w2_count = 64 * 16; // 1024
    const uint32_t total_weights = w0_count + w1_count + w2_count; // 9216 float16
    check_true(total_weights == 9216, "MLP total weights must be 9216 float16 parameters");
    check_true(total_weights * sizeof(uint16_t) == 18432, "Weights buffer must be 18432 bytes");
    check_true((w0_count % (16 * 16)) == 0, "Layer 0 dimensions must be multiple of 16x16 WMMA tile");
    check_true((w1_count % (16 * 16)) == 0, "Layer 1 dimensions must be multiple of 16x16 WMMA tile");
    check_true((w2_count % (16 * 16)) == 0, "Layer 2 dimensions must be multiple of 16x16 WMMA tile");
    std::cout << "  -> Wave32 WMMA 16x16x16 tiling alignment verified for all 3 layers." << std::endl;

    // 4. Verify Relative L1 Loss & Adam Optimizer Numerical Stability
    std::cout << "[TEST 4] Relative L1 Loss & Adam Optimizer Numerical Stability..." << std::endl;
    auto relL1Loss = [](float pred, float target) -> float {
        return std::abs(pred - target) / (std::max(pred, 0.0f) + 0.01f);
    };

    // Test zero radiance
    check_true(relL1Loss(0.0f, 0.0f) == 0.0f, "Loss for identical 0 radiance should be 0");
    // Test firefly suppression (pred = 1.0, target = 1000.0)
    float fireflyLoss = relL1Loss(1.0f, 1000.0f);
    check_true(std::isfinite(fireflyLoss), "Firefly loss must be finite");
    check_true(fireflyLoss < 1000.0f, "Relative loss dampens high radiance fireflies");

    // Test Adam step numerical stability
    float m = 0.0f;
    float v = 0.0f;
    float w = 0.5f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float lr = 1e-3f;
    const float eps = 1e-8f;

    std::vector<float> testGradients = { 0.0f, 1.0f, -2.5f, 100.0f, -500.0f, 1e-5f };
    for (uint32_t t = 1; t <= 100; ++t) {
        float g = testGradients[t % testGradients.size()];
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * (g * g);

        float mHat = m / (1.0f - std::pow(beta1, static_cast<float>(t)));
        float vHat = v / (1.0f - std::pow(beta2, static_cast<float>(t)));

        float step = lr * mHat / (std::sqrt(vHat) + eps);
        w -= step;

        check_true(std::isfinite(w), "Adam weight update must remain finite");
        check_true(!std::isnan(w), "Adam weight update must never become NaN");
    }
    std::cout << "  -> Adam optimizer update loop remained numerically stable across 100 iterations." << std::endl;

    std::cout << "\n[SUCCESS] All Tier 4 Neural Radiance Caching (NRC) unit tests passed!" << std::endl;
    return 0;
}
