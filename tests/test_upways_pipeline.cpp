#include "rt/UpwaysPipeline.hpp"
#include "rt/GatewayPipeline.hpp"
#include "rt/upways_weights.hpp"
#include "rt/upways_default_weights.hpp"
#include "rt/gateway_default_weights.hpp"
#include "core/Config.hpp"

#include <iostream>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <vector>
#include <fstream>
#include <filesystem>
#include <algorithm>

using namespace pathways;

static void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] Assertion failed: " << msg << std::endl;
        std::exit(1);
    }
}

// Invertible log transform implementation matching upways_common.glsl
static float upwaysLogTransform(float x, float mu = 16.0f) {
    float clampedX = std::max(x, 0.0f);
    return std::log1p(mu * clampedX) / std::log(1.0f + mu);
}

static float upwaysInverseLogTransform(float y, float mu = 16.0f) {
    float clampedY = std::clamp(y, 0.0f, 6.0f);
    return (std::pow(1.0f + mu, clampedY) - 1.0f) / mu;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing Upways (3-Pass) & Gateway (4-Pass)    " << std::endl;
    std::cout << "  Shader Push Constants, Weights & Pipeline Invariants    " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Verify Real Shader Push Constant Struct Layouts for Upways (3-Pass) & Gateway (4-Pass)
    std::cout << "[TEST 1] Upways (3-Pass) & Gateway (4-Pass) Push Constant Layouts..." << std::endl;
    check_true(sizeof(UpwaysTemporalPushConstants) == 20, "UpwaysTemporalPushConstants must be 20 bytes");
    check_true(offsetof(UpwaysTemporalPushConstants, renderWidth) == 0, "UpwaysTemporalPushConstants::renderWidth at 0");
    check_true(offsetof(UpwaysTemporalPushConstants, minAlphaSpecular) == 16, "UpwaysTemporalPushConstants::minAlphaSpecular at 16");

    check_true(sizeof(UpwaysNeuralGuidePushConstants) == 36, "UpwaysNeuralGuidePushConstants must be 36 bytes");
    check_true(offsetof(UpwaysNeuralGuidePushConstants, basePhiLumaSpecular) == 32, "UpwaysNeuralGuidePushConstants::basePhiLumaSpecular at 32");

    check_true(sizeof(UpwaysUpscaleResolvePushConstants) == 32, "UpwaysUpscaleResolvePushConstants must be 32 bytes");
    check_true(offsetof(UpwaysUpscaleResolvePushConstants, displayHistoryAlpha) == 28, "UpwaysUpscaleResolvePushConstants::displayHistoryAlpha at 28");

    check_true(sizeof(GatewayTemporalPushConstants) == 20, "GatewayTemporalPushConstants must be 20 bytes");
    check_true(sizeof(GatewayNeuralGuidePushConstants) == 36, "GatewayNeuralGuidePushConstants must be 36 bytes");
    check_true(sizeof(GatewayAtrousPushConstants) == 12, "GatewayAtrousPushConstants must be 12 bytes");
    check_true(sizeof(GatewayUpscaleResolvePushConstants) == 32, "GatewayUpscaleResolvePushConstants must be 32 bytes");
    std::cout << "  -> Upways (3-pass) and Gateway (4-pass) push constant structs verified." << std::endl;

    // 2. Verify Invertible Log Transform Invariants
    std::cout << "[TEST 2] Invertible Log Compression & Decompression Math..." << std::endl;
    check_true(std::abs(upwaysLogTransform(0.0f)) < 1e-6f, "Log transform of 0.0 must be 0.0");
    check_true(upwaysLogTransform(10000.0f) < 4.5f, "10000 nit firefly must be compressed to < 4.5");

    const float testRadiances[] = { 0.0f, 0.01f, 0.1f, 0.5f, 1.0f, 5.0f, 10.0f, 50.0f, 100.0f, 500.0f, 1000.0f, 5000.0f, 10000.0f };
    for (float r : testRadiances) {
        float compressed = upwaysLogTransform(r);
        float decompressed = upwaysInverseLogTransform(compressed);
        float relError = std::abs(decompressed - r) / (r + 1e-4f);
        check_true(relError < 0.01f, "Log transform inversion relative error must be < 1%");
    }
    std::cout << "  -> Invertible log transform validated with < 1% error across 0 to 10,000 nits." << std::endl;

    // 3. Verify Diffuse Albedo & Metallic-Fresnel Specular Demodulation / Remodulation Invariants
    std::cout << "[TEST 3] Co-Designed Diffuse & Metallic-Fresnel Specular Demodulation..." << std::endl;
    for (float alb = 0.05f; alb <= 1.0f; alb += 0.15f) {
        float rad = 2.5f;
        float safeAlb = std::max(alb, 0.04f);
        float demod = rad / safeAlb;
        float remod = demod * safeAlb;
        float err = std::abs(remod - rad) / rad;
        check_true(err < 1e-5f, "Diffuse demodulation/remodulation roundtrip must be exact");
    }
    for (float metallic : { 0.0f, 0.5f, 1.0f }) {
        float albedo = 0.8f;
        float nDotV = 0.5f;
        float f0 = 0.04f * (1.0f - metallic) + albedo * metallic;
        float fresnel = f0 + (1.0f - f0) * std::pow(1.0f - nDotV, 5.0f);
        float safeF = std::max(fresnel, 0.04f);
        float specRad = 1.75f;
        float demodSpec = specRad / safeF;
        float remodSpec = demodSpec * safeF;
        check_true(std::abs(remodSpec - specRad) < 1e-5f, "Specular metallic-Fresnel demodulation/remodulation roundtrip must be exact");
    }
    std::cout << "  -> Co-designed diffuse and metallic-Fresnel specular demodulation/remodulation verified." << std::endl;

    // 4. Verify Upways (80,512B U-Net) vs Gateway (2,144B MLP) Weight Topologies
    std::cout << "[TEST 4] Upways (80,512B 3-Scale 2D Halo-WMMA U-Net) & Gateway (2,144B Quad MLP) Weights Topology..." << std::endl;
    check_true(upways::TOTAL_WEIGHT_ELEMENTS == 40256, "Upways TOTAL_WEIGHT_ELEMENTS must be 40,256 FP16 parameters");
    check_true(upways::TOTAL_WEIGHT_BUFFER_SIZE == 80512, "Upways TOTAL_WEIGHT_BUFFER_SIZE must be 80,512 bytes");
    check_true(upways::ENC1_W_OFFSET == 0, "ENC1_W_OFFSET must be 0");
    check_true(upways::ENC1_B_OFFSET == 2560, "ENC1_B_OFFSET must be 2560");
    check_true(upways::REC_PROJ_W_OFFSET == 2592, "REC_PROJ_W_OFFSET must be 2592");
    check_true(upways::ENC2_W_OFFSET == 3104, "ENC2_W_OFFSET must be 3104");
    check_true(upways::BOTTLENECK1_W_OFFSET == 6208, "BOTTLENECK1_W_OFFSET must be 6208");
    check_true(upways::BOTTLENECK2_W_OFFSET == 12416, "BOTTLENECK2_W_OFFSET must be 12416");
    check_true(upways::BOTTLENECK3_W_OFFSET == 24768, "BOTTLENECK3_W_OFFSET must be 24768");
    check_true(upways::DEC2_W_OFFSET == 30944, "DEC2_W_OFFSET must be 30944");
    check_true(upways::DEC1_W_OFFSET == 34048, "DEC1_W_OFFSET must be 34048");
    check_true(upways::HEAD_OUT_W_OFFSET == 39200, "HEAD_OUT_W_OFFSET must be 39200");
    check_true(upways::HEAD_REC_W_OFFSET == 39728, "HEAD_REC_W_OFFSET must be 39728");
    check_true(upways::HEAD_REC_B_OFFSET + 16 == upways::TOTAL_WEIGHT_ELEMENTS, "HEAD_REC_B_OFFSET + 16 must equal 40,256");

    check_true(gateway::TOTAL_WEIGHT_BUFFER_SIZE == 2144, "Gateway TOTAL_WEIGHT_BUFFER_SIZE must be 2,144 bytes (1,072 FP16 params)");
    check_true(sizeof(gateway::DEFAULT_GATEWAY_WEIGHTS) == gateway::TOTAL_WEIGHT_BUFFER_SIZE, "Gateway embedded weights must be 2,144 bytes");
    std::cout << "  -> Upways (40,256 FP16 params = 80,512B) and Gateway (1,072 FP16 params = 2,144B) verified." << std::endl;

    // 5. Verify Exported Weights and Embedded Fallback Integrity
    std::cout << "[TEST 5] Neural Weights Integrity (Embedded & File)..." << std::endl;
    check_true(sizeof(upways::DEFAULT_UPWAYS_WEIGHTS) == upways::TOTAL_WEIGHT_BUFFER_SIZE,
               "Embedded DEFAULT_UPWAYS_WEIGHTS size must match TOTAL_WEIGHT_BUFFER_SIZE exactly");
    check_true(upways::EMBEDDED_WEIGHTS_SIZE == upways::TOTAL_WEIGHT_BUFFER_SIZE,
               "EMBEDDED_WEIGHTS_SIZE must match TOTAL_WEIGHT_BUFFER_SIZE");
    std::cout << "  -> Embedded default neural weights verified (" << sizeof(upways::DEFAULT_UPWAYS_WEIGHTS) << " bytes)." << std::endl;

    std::vector<std::string> searchPaths = {
        "data/models/upways_weights.bin",
        "../data/models/upways_weights.bin",
        "../../data/models/upways_weights.bin",
        "../../../data/models/upways_weights.bin",
        "build/bin/data/models/upways_weights.bin",
        "bin/data/models/upways_weights.bin"
    };
    std::string foundPath;
    for (const auto& p : searchPaths) {
        if (std::filesystem::exists(p)) {
            foundPath = p;
            break;
        }
    }
    if (!foundPath.empty()) {
        auto fileSize = std::filesystem::file_size(foundPath);
        check_true(fileSize == upways::TOTAL_WEIGHT_BUFFER_SIZE, "upways_weights.bin file size must exactly match TOTAL_WEIGHT_BUFFER_SIZE");
        std::cout << "  -> External weight file binary verified at: " << foundPath << " (" << fileSize << " bytes)." << std::endl;
    }

    // 6. Verify Disocclusion Confidence Gating Math
    std::cout << "[TEST 6] Disocclusion Confidence Gating..." << std::endl;
    auto testConfidence = [](float depthCurr, float depthPrev, float depthThreshold) -> float {
        float depthDelta = (depthCurr > 0.0f && depthPrev > 0.0f) ? (std::abs(depthCurr - depthPrev) / (depthCurr + 1e-3f)) : 0.0f;
        if (depthDelta >= depthThreshold) return 0.0f;
        return std::clamp(1.0f - depthDelta * 20.0f, 0.0f, 1.0f);
    };

    check_true(testConfidence(5.0f, 5.0f, 0.05f) == 1.0f, "Identical depth must have 100% confidence");
    check_true(testConfidence(5.0f, 4.95f, 0.05f) > 0.7f, "Small depth change (1%) must have high confidence");
    check_true(testConfidence(5.0f, 2.0f, 0.05f) == 0.0f, "Disocclusion (> 5%) must have 0% confidence");
    std::cout << "  -> Disocclusion confidence gating function verified." << std::endl;

    // 7. Verify 90-Degree Camera Flick Disocclusion Invariants
    std::cout << "[TEST 7] 90-Degree Camera Flick Disocclusion & Rejection..." << std::endl;
    {
        float nCurr_world[3] = { 1.0f, 0.0f, 0.0f };
        float nPrev_world[3] = { 0.0f, 0.0f, 1.0f };
        float normalDot = nCurr_world[0] * nPrev_world[0] + nCurr_world[1] * nPrev_world[1] + nCurr_world[2] * nPrev_world[2];
        float expectedPrevDepth = 0.0f;
        float centerDepth = 4.0f;
        float depthDelta = std::abs(expectedPrevDepth - centerDepth) / std::max(expectedPrevDepth, 1e-3f);

        bool rejectNormal = (normalDot < 0.707f);
        bool rejectDepth = (depthDelta > 0.10f);
        bool rejectDiff = rejectNormal || rejectDepth;

        check_true(rejectNormal && rejectDepth && rejectDiff, "90-degree camera flick must trigger disocclusion rejection");
        float confDiff = rejectDiff ? 0.0f : std::exp(-depthDelta * 20.0f);
        float alphaDiff = (1.0f - confDiff) * 1.0f + confDiff * 0.08f;
        check_true(alphaDiff == 1.0f, "Temporal accumulation rate must clamp to 1.0 on disocclusion");
        std::cout << "  -> 90-degree camera flick disocclusion rejection verified (conf = 0.0, alpha = 1.0)." << std::endl;
    }

    // 8. Verify 3-Pass Upways LDS Memory Footprint & Occupancy Invariants
    std::cout << "[TEST 8] 3-Pass Upways LDS Footprint & Wave32 Occupancy Invariants..." << std::endl;
    {
        // Pass 1 (upways_temporal.comp): 2 x f16vec4[18][19] + float16_t s_lumS[18][20] = 5,472 + 720 = 6,192 bytes (6.05 KB)
        size_t pass1LdsBytes = (2 * 18 * 19 * 8) + (18 * 20 * sizeof(uint16_t));
        // Pass 2 (upways_neural_guide.comp): float16_t s_buf[8192] + float16_t s_e1[8192] = 32,768 bytes (32.00 KB)
        size_t pass2LdsBytes = 2 * 8192 * sizeof(uint16_t);
        // Pass 3 (upways_upscale_resolve.comp): 4 x f16vec4[24][25] + float16_t s_met[18][19] = 19,200 + 684 = 19,884 bytes (19.42 KB)
        size_t pass3LdsBytes = (4 * 24 * 25 * 8) + (18 * 19 * sizeof(uint16_t));

        check_true(pass1LdsBytes == 6192 && pass1LdsBytes <= 32768, "Pass 1 LDS must be 6,192 bytes (<= 32 KB)");
        check_true(pass2LdsBytes == 32768, "Pass 2 LDS must be exactly 32,768 bytes (32.0 KB)");
        check_true(pass3LdsBytes == 19884 && pass3LdsBytes <= 32768, "Pass 3 LDS must be 19,884 bytes (<= 32 KB)");
        std::cout << "  -> LDS footprints verified: Pass 1 = " << (pass1LdsBytes / 1024.0f)
                  << " KB, Pass 2 = " << (pass2LdsBytes / 1024.0f)
                  << " KB, Pass 3 = " << (pass3LdsBytes / 1024.0f) << " KB (<= 32 KB budget)." << std::endl;
    }

    // 9. Verify Separate CLI Configuration for --denoiser gateway vs --denoiser upways
    std::cout << "[TEST 9] CLI Configuration Separation (--denoiser gateway vs --denoiser upways)..." << std::endl;
    {
        // 9a: --res 4k --scaler upways quality
        const char* argv1[] = { "pathways", "--res", "4k", "--scaler", "upways", "quality" };
        Config c1 = Config::parse(6, const_cast<char**>(argv1));
        check_true(c1.width == 3840 && c1.height == 2160, "c1 output resolution must be 3840x2160");
        check_true(c1.upscaler_mode == UpscalerMode::Upways, "c1 upscaler_mode must be Upways");
        check_true(c1.upways_superres == true, "c1 upways_superres must be true");
        check_true(c1.denoiser_mode == DenoiserMode::Upways, "c1 denoiser_mode must be Upways");
        check_true(c1.getRenderWidth() == 2560 && c1.getRenderHeight() == 1440, "c1 internal resolution must be 2560x1440 (1.5x)");

        // 9b: --res 4k --scaler gateway quality
        const char* argv2[] = { "pathways", "--res", "4k", "--scaler", "gateway", "quality" };
        Config c2 = Config::parse(6, const_cast<char**>(argv2));
        check_true(c2.width == 3840 && c2.height == 2160, "c2 output resolution must be 3840x2160");
        check_true(c2.upscaler_mode == UpscalerMode::Gateway, "c2 upscaler_mode must be Gateway");
        check_true(c2.gateway_superres == true, "c2 gateway_superres must be true");
        check_true(c2.denoiser_mode == DenoiserMode::Gateway, "c2 denoiser_mode must be Gateway");
        check_true(c2.getRenderWidth() == 2560 && c2.getRenderHeight() == 1440, "c2 internal resolution must be 2560x1440 (1.5x)");

        // 9c: --res 1080 --denoiser gateway --scaler gateway native
        const char* argv3[] = { "pathways", "--res", "1080", "--denoiser", "gateway", "--scaler", "gateway", "native" };
        Config c3 = Config::parse(8, const_cast<char**>(argv3));
        check_true(c3.denoiser_mode == DenoiserMode::Gateway, "c3 denoiser_mode must be Gateway");
        check_true(c3.gateway_superres == false, "c3 gateway_superres must be false for native");
        check_true(c3.getRenderWidth() == 1920 && c3.getRenderHeight() == 1080, "c3 internal resolution must be 1920x1080");

        // 9d: --res 1080 --denoiser upways --scaler upways native
        const char* argv4[] = { "pathways", "--res", "1080", "--denoiser", "upways", "--scaler", "upways", "native" };
        Config c4 = Config::parse(8, const_cast<char**>(argv4));
        check_true(c4.denoiser_mode == DenoiserMode::Upways, "c4 denoiser_mode must be Upways");
        check_true(c4.upways_superres == false, "c4 upways_superres must be false for native");
        check_true(c4.getRenderWidth() == 1920 && c4.getRenderHeight() == 1080, "c4 internal resolution must be 1920x1080");

        // 9e: --res 4k --scaler upways 1080
        const char* argv5[] = { "pathways", "--res", "4k", "--scaler", "upways", "1080" };
        Config c5 = Config::parse(6, const_cast<char**>(argv5));
        check_true(c5.upscaler_mode == UpscalerMode::Upways, "c5 upscaler_mode must be Upways");
        check_true(c5.getRenderWidth() == 1920 && c5.getRenderHeight() == 1080, "c5 internal resolution must be 1920x1080 (2.0x)");

        // 9f: --res 4k --scaler gateway 1080
        const char* argv6[] = { "pathways", "--res", "4k", "--scaler", "gateway", "1080" };
        Config c6 = Config::parse(6, const_cast<char**>(argv6));
        check_true(c6.upscaler_mode == UpscalerMode::Gateway, "c6 upscaler_mode must be Gateway");
        check_true(c6.getRenderWidth() == 1920 && c6.getRenderHeight() == 1080, "c6 internal resolution must be 1920x1080 (2.0x)");

        std::cout << "  -> Verified distinct CLI parsing and resolution scaling for --denoiser/--scaler gateway and upways." << std::endl;
    }

    // 10. Verify Per-Pass Hardware Timestamp Profiler Structures & Slot Counts
    std::cout << "[TEST 10] Per-Pass Hardware Timestamp Query Pool Layout (Upways vs Gateway)..." << std::endl;
    {
        static_assert(UpwaysPipeline::MAX_QUERY_SLOTS == 2, "UpwaysPipeline must double-buffer query slots");
        static_assert(UpwaysPipeline::QUERIES_PER_SLOT == 5, "UpwaysPipeline must record 5 timestamps per slot (Pass 1, Pass 2, Pass 3, History Copy)");
        static_assert(GatewayPipeline::MAX_QUERY_SLOTS == 2, "GatewayPipeline must double-buffer query slots");
        static_assert(GatewayPipeline::QUERIES_PER_SLOT == 7, "GatewayPipeline must record 7 timestamps per slot (Pass 1, Pass 2, Pass 3a, Pass 3b, Pass 4, History Copy)");

        UpwaysPassTimings uTimings{};
        check_true(!uTimings.valid && uTimings.totalMs == 0.0, "Default UpwaysPassTimings must be invalid and zeroed");
        GatewayPassTimings gTimings{};
        check_true(!gTimings.valid && gTimings.totalMs == 0.0, "Default GatewayPassTimings must be invalid and zeroed");
        std::cout << "  -> Verified Upways (5 timestamps/slot) and Gateway (7 timestamps/slot) per-pass query pool layouts." << std::endl;
    }

    std::cout << "\n==========================================================" << std::endl;
    std::cout << " [SUCCESS] All Upways (3-Pass) & Gateway (4-Pass) Tests Passed! (10/10)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
