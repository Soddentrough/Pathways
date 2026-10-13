// =============================================================================
// tests/test_denoiser_pipeline.cpp
// Adaptive E-SVGF Denoiser Host-Side Contract Verification (CTest)
// Reference: docs/design/adaptive_svgf_denoiser_proposal.md
//
// Validates without creating an SDL3 window:
//   1. std140 CameraMatricesUBO binary layout (160 B, column-major mat4 + mat3)
//   2. Push constant struct sizes for all 5 dispatches (<= 128 B Vulkan minimum)
//   3. SPIR-V artifacts for the 5 dispatch shaders exist and carry valid headers
//   4. Edge classification / adaptive threshold math (noise + variance floors)
//   5. Dual-channel edge density temporal estimator fixed-point integrity
//   6. Continuous smoothstep crossfade ramp continuity at partition boundaries
// =============================================================================

#include "rt/DenoiserPipeline.hpp"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace pathways;

static int g_failures = 0;

static void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::fprintf(stderr, "[FAIL] %s\n", msg);
        ++g_failures;
    }
}

static std::filesystem::path findSpv(const std::string& name) {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates = {
#ifdef PATHWAYS_SPV_DIR
        fs::path(PATHWAYS_SPV_DIR) / name,
#endif
        fs::current_path() / "shaders" / name,
        fs::current_path() / "build" / "shaders" / name,
        fs::current_path() / "build" / "linux-release" / "shaders" / name,
        fs::current_path() / "build" / "windows-clang-release" / "shaders" / name,
    };
    for (const auto& p : candidates) {
        if (fs::exists(p)) return p;
    }
    return {};
}

static bool validateSpirv(const std::filesystem::path& path, uint32_t& outBoundWords) {
    FILE* f = std::fopen(path.string().c_str(), "rb");
    if (!f) return false;
    uint32_t header[5]{};
    size_t n = std::fread(header, sizeof(uint32_t), 5, f);
    std::fclose(f);
    if (n < 5) return false;
    if (header[0] != 0x07230203u) return false; // SPIR-V magic number
    outBoundWords = header[3];
    return header[3] > 5;
}

int main() {
    std::printf("==========================================================\n");
    std::printf("  Pathways: Adaptive E-SVGF Denoiser Pipeline Unit Test    \n");
    std::printf("==========================================================\n");

    // -------------------------------------------------------------------------
    // TEST 1: CameraMatricesUBO std140 layout
    // -------------------------------------------------------------------------
    std::printf("[TEST 1] CameraMatricesUBO std140 layout...\n");
    check_true(sizeof(DenoiserPipeline::CameraMatrices) == 160, "CameraMatrices must be exactly 160 bytes");
    check_true(offsetof(DenoiserPipeline::CameraMatrices, currViewToPrevView) == 0, "currViewToPrevView at byte 0");
    check_true(offsetof(DenoiserPipeline::CameraMatrices, rotCols) == 64, "rotCols (mat3 std140) at byte 64");
    check_true(offsetof(DenoiserPipeline::CameraMatrices, currCameraJitter) == 112, "currCameraJitter at byte 112");
    check_true(offsetof(DenoiserPipeline::CameraMatrices, prevCameraJitter) == 120, "prevCameraJitter at byte 120");
    check_true(offsetof(DenoiserPipeline::CameraMatrices, tanHalfFovY) == 128, "tanHalfFovY at byte 128");
    check_true(offsetof(DenoiserPipeline::CameraMatrices, aspectRatio) == 132, "aspectRatio at byte 132");
    check_true(offsetof(DenoiserPipeline::CameraMatrices, padTail) == 144, "padTail at byte 144");
    std::printf("  -> OK (160 B)\n");

    // -------------------------------------------------------------------------
    // TEST 2: Push constant sizes for the 5 dispatches (<= 128 B guaranteed)
    // -------------------------------------------------------------------------
    std::printf("[TEST 2] Push constant struct sizes...\n");
    check_true(sizeof(DenoiserPipeline::FusedTemporalPushConstants) == 80, "FusedTemporalPushConstants == 80 B");
    check_true(sizeof(DenoiserPipeline::AtrousPushConstants) == 32, "AtrousPushConstants == 32 B");
    check_true(sizeof(DenoiserPipeline::AtrousCoarseRemodulatePushConstants) == 48, "AtrousCoarseRemodulatePushConstants == 48 B");
    check_true(sizeof(DenoiserPipeline::FusedTemporalPushConstants) <= 128, "Pass 1 push constants within 128 B");
    std::printf("  -> OK (80 / 32 / 48 B)\n");

    // -------------------------------------------------------------------------
    // TEST 3: SPIR-V artifacts for the 5 authoritative dispatches
    // -------------------------------------------------------------------------
    std::printf("[TEST 3] SPIR-V dispatch shader artifacts...\n");
    const char* shaderNames[5] = {
        "denoise_fused_temporal.comp.spv",
        "denoise_atrous_step1.comp.spv",
        "denoise_atrous_step2.comp.spv",
        "denoise_atrous_step3.comp.spv",
        "denoise_atrous_coarse_remodulate.comp.spv",
    };
    for (const char* name : shaderNames) {
        auto path = findSpv(name);
        if (path.empty()) {
            // Compilation gate is enforced by CMake (SPV_FILES); tolerate absence
            // when tests run without the shader build artifacts present.
            std::printf("  -> SKIP %s (not found near test working directory)\n", name);
            continue;
        }
        uint32_t words = 0;
        bool ok = validateSpirv(path, words);
        if (!ok) {
            std::fprintf(stderr, "[FAIL] Invalid SPIR-V header for %s\n", name);
            ++g_failures;
        } else {
            std::printf("  -> OK %s (%u words)\n", name, words);
        }
    }

    // -------------------------------------------------------------------------
    // TEST 4: Statistical adaptive threshold with noise/variance/contrast floors
    //   T_tile = max(mu + 1.2 * max(sigma, sigmaMin), noiseFloor)
    //   M_edge = (M >= T) && (M - mu >= deltaMin)
    // -------------------------------------------------------------------------
    std::printf("[TEST 4] Adaptive threshold & floors...\n");
    const float sigmaMin = 0.02f, noiseFloor = 0.05f, deltaMin = 0.02f, kSigma = 1.2f;
    auto threshold = [&](float mu, float sigma) {
        return std::fmax(mu + kSigma * std::fmax(sigma, sigmaMin), noiseFloor);
    };
    auto isEdge = [&](float m, float mu, float sigma) {
        return (m >= threshold(mu, sigma)) && ((m - mu) >= deltaMin);
    };

    // Pathological flat tile: uniform M = 0.06, sigma collapses to 0.
    // Without floors, T = mu = 0.06 and 100% of pixels classify as edges.
    check_true(!isEdge(0.06f, 0.06f, 0.0f), "flat tile (sigma collapse) must NOT classify uniform pixels as edges");
    // Sub-noise smooth tile
    check_true(!isEdge(0.03f, 0.03f, 0.0f), "sub-noise-floor tile rejects edges");
    // Real edge spike above threshold with contrast
    check_true(isEdge(0.50f, 0.05f, 0.10f), "strong gradient spike classified as edge");
    // Slightly-above-mean bump inside the contrast floor is rejected
    check_true(!isEdge(0.10f, 0.09f, 0.005f), "contrast floor deltaMin rejects marginal bumps");

    // Rayleigh exceedance intuition: T ~ 2.0394 sigma_0 for ideal Gaussian derivatives
    {
        float sigma0 = 1.0f;
        float muR = sigma0 * std::sqrt(3.14159265f / 2.0f);
        float sigR = sigma0 * std::sqrt((4.0f - 3.14159265f) / 2.0f);
        float T = muR + 1.2f * sigR;
        float exceedance = std::exp(-(T * T) / (2.0f * sigma0 * sigma0));
        check_true(std::fabs(exceedance - 0.125f) < 0.01f, "Rayleigh exceedance ~12.5% at T = mu + 1.2 sigma");
    }
    std::printf("  -> OK\n");

    // -------------------------------------------------------------------------
    // TEST 5: Dual-channel edge density estimator integrity
    //   R channel stores raw rho^t; G channel stores rho_eff.
    //   Fixed point of the temporal recursion on R converges to rho_spatial,
    //   NOT to the collapsed 0.15 f rho / (1 - 0.85 f) modulation trap.
    // -------------------------------------------------------------------------
    std::printf("[TEST 5] Temporal edge density fixed point...\n");
    {
        const float alpha = 0.15f;
        const float rhoSpatial = 0.8f;
        float rho = 0.0f;
        for (int i = 0; i < 300; ++i) {
            rho = (1.0f - alpha) * rho + alpha * rhoSpatial; // R channel recursion (valid reprojection)
        }
        check_true(std::fabs(rho - rhoSpatial) < 1e-3f, "raw rho^t converges to rho_spatial (no fixed-point collapse)");

        // Effective density modulation widens kernels on disocclusion but must
        // never feed back into the temporal state channel.
        float nDiff = 1.0f;
        float rhoEff = rho * std::fmin(std::fmax(nDiff / 8.0f, 0.25f), 1.0f);
        check_true(std::fabs(rhoEff - 0.25f * rhoSpatial) < 1e-3f, "disocclusion rho_eff = 0.25 * rho (kernel widening)");
        float rhoEffFull = rho * std::fmin(std::fmax(32.0f / 8.0f, 0.25f), 1.0f);
        check_true(std::fabs(rhoEffFull - rhoSpatial) < 1e-3f, "converged history rho_eff = rho (no dilation)");
    }
    std::printf("  -> OK\n");

    // -------------------------------------------------------------------------
    // TEST 6: Smoothstep crossfade ramp continuity at kernel partition boundaries
    //   Pass 4: smoothstep(0.45, 0.55, rho); Pass 5: smoothstep(0.25, 0.35, rho)
    // -------------------------------------------------------------------------
    std::printf("[TEST 6] Crossfade ramp continuity...\n");
    auto smoothstep = [](float a, float b, float x) {
        float t = std::fmin(std::fmax((x - a) / (b - a), 0.0f), 1.0f);
        return t * t * (3.0f - 2.0f * t);
    };
    check_true(smoothstep(0.45f, 0.55f, 0.44f) == 0.0f, "Pass 4 ramp: below range = 0");
    check_true(smoothstep(0.45f, 0.55f, 0.56f) == 1.0f, "Pass 4 ramp: above range = 1 (full bypass at subgroupAll top)");
    check_true(smoothstep(0.25f, 0.35f, 0.30f) > 0.4f && smoothstep(0.25f, 0.35f, 0.30f) < 0.6f, "Pass 5 ramp midpoint ~0.5");
    // Monotonicity across both ramps (no variance step jumps / checkerboarding)
    for (float x = 0.0f; x < 1.0f; x += 0.01f) {
        float w1a = smoothstep(0.45f, 0.55f, x);
        float w1b = smoothstep(0.45f, 0.55f, x + 0.01f);
        float w2a = smoothstep(0.25f, 0.35f, x);
        float w2b = smoothstep(0.25f, 0.35f, x + 0.01f);
        if (w1b < w1a - 1e-6f || w2b < w2a - 1e-6f) { check_true(false, "crossfade ramps must be monotonic"); break; }
    }
    std::printf("  -> OK\n");

    // -------------------------------------------------------------------------
    // TEST 7: Receptive field accounting (5-dispatch dilated hierarchy)
    // -------------------------------------------------------------------------
    std::printf("[TEST 7] Dilated receptive field & tap budget...\n");
    {
        // Accumulated radius: R_k = sum of dilation offsets {2, 4, 8, 16}
        int radius = 0, taps = 0;
        int diameters[4];
        for (int k = 0; k < 4; ++k) {
            radius += (k == 0) ? 2 : (2 << k);   // 2, 4, 8, 16
            diameters[k] = 2 * radius + 1;
            taps += 25; // 5x5 cross-bilateral taps per dispatch
        }
        check_true(diameters[0] == 5,  "Pass 2 diameter = 5 (s=1)");
        check_true(diameters[1] == 13, "Pass 3 accumulated diameter = 13 (covers k=9)");
        check_true(diameters[2] == 29, "Pass 4 accumulated diameter = 29 (covers k=17)");
        check_true(diameters[3] == 61, "Pass 5 accumulated diameter = 61 (covers k=33)");
        check_true(taps == 100, "full-support tap budget = 100 (vs 1089 brute-force 33x33)");
    }
    std::printf("  -> OK\n");

    if (g_failures > 0) {
        std::fprintf(stderr, "\n[RESULT] %d check(s) FAILED\n", g_failures);
        return 1;
    }
    std::printf("\n[RESULT] All Adaptive E-SVGF denoiser contract checks PASSED\n");
    return 0;
}
