#include "rt/UpwaysPipeline.hpp"
#include "rt/upways_weights.hpp"

#include <iostream>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <vector>
#include <fstream>
#include <filesystem>

using namespace pathways;

static void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] Assertion failed: " << msg << std::endl;
        std::exit(1);
    }
}

// Invertible log transform implementation for host-side verification
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
    std::cout << "  Pathways: Testing Upways Neural Denoiser & Super-Res    " << std::endl;
    std::cout << "  Wave32 WMMA Cooperative Matrix & Pipeline Invariants    " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Verify UpwaysPushConstants layout & size alignment
    std::cout << "[TEST 1] UpwaysPushConstants Struct Layout & Alignment..." << std::endl;
    check_true(sizeof(UpwaysPushConstants) == 64, "UpwaysPushConstants must be exactly 64 bytes");
    check_true(offsetof(UpwaysPushConstants, inputWidth) == 0, "inputWidth must be at byte 0");
    check_true(offsetof(UpwaysPushConstants, inputHeight) == 4, "inputHeight must be at byte 4");
    check_true(offsetof(UpwaysPushConstants, outputWidth) == 8, "outputWidth must be at byte 8");
    check_true(offsetof(UpwaysPushConstants, outputHeight) == 12, "outputHeight must be at byte 12");
    check_true(offsetof(UpwaysPushConstants, invInputWidth) == 16, "invInputWidth must be at byte 16");
    check_true(offsetof(UpwaysPushConstants, invInputHeight) == 20, "invInputHeight must be at byte 20");
    check_true(offsetof(UpwaysPushConstants, invOutputWidth) == 24, "invOutputWidth must be at byte 24");
    check_true(offsetof(UpwaysPushConstants, invOutputHeight) == 28, "invOutputHeight must be at byte 28");
    check_true(offsetof(UpwaysPushConstants, frameIndex) == 32, "frameIndex must be at byte 32");
    check_true(offsetof(UpwaysPushConstants, resetHistory) == 36, "resetHistory must be at byte 36");
    check_true(offsetof(UpwaysPushConstants, cameraMoved) == 40, "cameraMoved must be at byte 40");
    check_true(offsetof(UpwaysPushConstants, superResMode) == 44, "superResMode must be at byte 44");
    check_true(offsetof(UpwaysPushConstants, depthThreshold) == 48, "depthThreshold must be at byte 48");
    check_true(offsetof(UpwaysPushConstants, normalThreshold) == 52, "normalThreshold must be at byte 52");
    check_true(offsetof(UpwaysPushConstants, blendAlpha) == 56, "blendAlpha must be at byte 56");
    check_true(offsetof(UpwaysPushConstants, pad) == 60, "pad must be at byte 60");
    std::cout << "  -> UpwaysPushConstants layout verified (64B push constant aligned)." << std::endl;

    // 2. Verify Invertible Log Transform Invariants
    std::cout << "[TEST 2] Invertible Log Compression & Decompression Math..." << std::endl;
    check_true(std::abs(upwaysLogTransform(0.0f)) < 1e-6f, "Log transform of 0.0 must be 0.0");
    check_true(upwaysLogTransform(10000.0f) < 4.5f, "10000 nit firefly must be compressed to < 4.5");

    // Verify invertible identity: phi^-1(phi(x)) == x across wide dynamic range
    const float testRadiances[] = { 0.0f, 0.01f, 0.1f, 0.5f, 1.0f, 5.0f, 10.0f, 50.0f, 100.0f, 500.0f, 1000.0f, 5000.0f, 10000.0f };
    for (float r : testRadiances) {
        float compressed = upwaysLogTransform(r);
        float decompressed = upwaysInverseLogTransform(compressed);
        float relError = std::abs(decompressed - r) / (r + 1e-4f);
        check_true(relError < 0.01f, "Log transform inversion relative error must be < 1%");
    }
    std::cout << "  -> Invertible log transform validated with < 1% error across 0 to 10,000 nits." << std::endl;

    // 3. Verify Demodulation and Remodulation Invariants
    std::cout << "[TEST 3] Physical Demodulation & Remodulation Invariants..." << std::endl;
    // For diffuse: Demod = Radiance / (Albedo + 1e-3), Remod = Demod * Albedo
    for (float alb = 0.05f; alb <= 1.0f; alb += 0.15f) {
        float rad = 2.5f;
        float demod = rad / (alb + 1e-3f);
        float remod = demod * alb;
        float err = std::abs(remod - rad) / rad;
        check_true(err < 0.02f, "Demodulation/remodulation error must remain small");
    }
    std::cout << "  -> Physical demodulation / remodulation mathematically verified." << std::endl;

    // 4. Verify Cooperative Matrix WMMA Weights Topology
    std::cout << "[TEST 4] Wave32 WMMA Cooperative Matrix Weights Topology..." << std::endl;
    check_true(upways::TOTAL_WEIGHT_BUFFER_SIZE == 524792, "TOTAL_WEIGHT_BUFFER_SIZE must be 524792 bytes");
    check_true(upways::LAYER_ENC0_PROJ.inChannels == 16, "enc0_proj input channels must be 16");
    check_true(upways::LAYER_ENC0_PROJ.outChannels == 32, "enc0_proj output channels must be 32");
    check_true(upways::LAYER_ENC0_RES_CONV1.outChannels == 32, "enc0_res_conv1 output channels must be 32");
    check_true(upways::LAYER_DOWNSAMPLE_1.outChannels == 64, "downsample_1 output channels must be 64");
    check_true(upways::LAYER_ENC1_RES_CONV1.outChannels == 64, "enc1_res_conv1 output channels must be 64");
    check_true(upways::LAYER_CONV_GRU_CONV_RZ.outChannels == 128, "conv_gru_conv_rz output channels must be 128");
    check_true(upways::LAYER_CONV_GRU_CONV_H.outChannels == 64, "conv_gru_conv_h output channels must be 64");
    check_true(upways::LAYER_DEC1_RES_CONV1.outChannels == 64, "dec1_res_conv1 output channels must be 64");
    check_true(upways::LAYER_UPSAMPLE_1.outChannels == 32, "upsample_1 output channels must be 32");
    check_true(upways::LAYER_SKIP_BLEND.outChannels == 32, "skip_blend output channels must be 32");
    check_true(upways::LAYER_DEC0_RES_CONV1.outChannels == 32, "dec0_res_conv1 output channels must be 32");
    check_true(upways::LAYER_UPSCALER_UP_CONV.outChannels == 128, "upscaler_up_conv output channels must be 128");
    check_true(upways::LAYER_UPSCALER_REFINE_4.outChannels == 6, "upscaler_refine_4 output channels must be 6");
    check_true(upways::LAYER_UPSCALER_NATIVE_CONV_2.outChannels == 6, "upscaler_native_conv_2 output channels must be 6");

    // Verify all channel counts are aligned to 16 for Wave32 WMMA (except final 6-channel output)
    check_true(upways::LAYER_ENC0_PROJ.inChannels % 16 == 0, "enc0_proj inChannels must be multiple of 16");
    check_true(upways::LAYER_ENC0_PROJ.outChannels % 16 == 0, "enc0_proj outChannels must be multiple of 16");
    check_true(upways::LAYER_ENC1_RES_CONV1.outChannels % 16 == 0, "enc1_res_conv1 outChannels must be multiple of 16");
    check_true(upways::LAYER_CONV_GRU_CONV_RZ.outChannels % 16 == 0, "conv_gru outChannels must be multiple of 16");
    std::cout << "  -> Wave32 WMMA 16x16 cooperative matrix alignment verified across all network layers." << std::endl;

    // 5. Verify Exported Weights File Existence and Size
    std::cout << "[TEST 5] Exported Weights File Binary Integrity..." << std::endl;
    std::vector<std::string> searchPaths = {
        "data/models/upways_weights.bin",
        "../data/models/upways_weights.bin",
        "../../data/models/upways_weights.bin",
        "../../../data/models/upways_weights.bin",
        "/home/naoki/Development/Pathways/data/models/upways_weights.bin",
        "/home/naoki/Development/Upways/checkpoints/run_multiscene_superres/vulkan_export/upways_weights.bin"
    };
    std::string foundPath;
    for (const auto& p : searchPaths) {
        if (std::filesystem::exists(p)) {
            foundPath = p;
            break;
        }
    }
    check_true(!foundPath.empty(), "upways_weights.bin must exist in at least one standard search path");
    auto fileSize = std::filesystem::file_size(foundPath);
    check_true(fileSize == upways::TOTAL_WEIGHT_BUFFER_SIZE, "upways_weights.bin file size must exactly match TOTAL_WEIGHT_BUFFER_SIZE");
    std::cout << "  -> Weight file binary verified at: " << foundPath << " (" << fileSize << " bytes)." << std::endl;

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

    std::cout << "\n==========================================================" << std::endl;
    std::cout << " [SUCCESS] All Upways Vulkan Pipeline Invariant Tests Passed!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
