#pragma once

// Auto-generated Upways 3D Kinematic Dual-Head WMMA Reconstructor FP16 Weights Header
// Matching Pass 2 shaders/compute/upways_reconstruct.comp
#include <cstdint>
#include <cstddef>

namespace upways {

// =========================================================================
// Diffuse Head Offsets (Elements in weights[])
// =========================================================================
constexpr uint32_t DIFF_FC1_WEIGHT_ELEM = 0;
constexpr uint32_t DIFF_FC1_BIAS_ELEM   = 1024;
constexpr uint32_t DIFF_FC2_WEIGHT_ELEM = 1088;
constexpr uint32_t DIFF_FC2_BIAS_ELEM   = 5184;
constexpr uint32_t DIFF_FC3_WEIGHT_ELEM = 5248;
constexpr uint32_t DIFF_FC3_BIAS_ELEM   = 6272;

// =========================================================================
// Specular Head Offsets (Elements in weights[])
// =========================================================================
constexpr uint32_t SPEC_OFFSET_BASE     = 6288;
constexpr uint32_t SPEC_FC1_WEIGHT_ELEM = SPEC_OFFSET_BASE + 0;
constexpr uint32_t SPEC_FC1_BIAS_ELEM   = SPEC_OFFSET_BASE + 1024;
constexpr uint32_t SPEC_FC2_WEIGHT_ELEM = SPEC_OFFSET_BASE + 1088;
constexpr uint32_t SPEC_FC2_BIAS_ELEM   = SPEC_OFFSET_BASE + 5184;
constexpr uint32_t SPEC_FC3_WEIGHT_ELEM = SPEC_OFFSET_BASE + 5248;
constexpr uint32_t SPEC_FC3_BIAS_ELEM   = SPEC_OFFSET_BASE + 6272;

constexpr uint32_t TOTAL_WEIGHT_ELEMENTS = 12576;
constexpr uint32_t TOTAL_WEIGHT_BUFFER_SIZE = 25152; // 25152 bytes

struct LayerDescriptor {
    uint32_t weightOffset; // Byte offset from start of SSBO
    uint32_t weightSize;   // Size in bytes
    uint32_t biasOffset;   // Bias byte offset (16-byte aligned)
    uint32_t biasSize;     // Bias size in bytes
    uint32_t inChannels;   // C_in
    uint32_t outChannels;  // C_out
    uint32_t stride;       // Stride (16 for cooperative matrix tiles)
};

// Diffuse Layer Descriptors
inline constexpr LayerDescriptor LAYER_DIFF_FC1 = {
    .weightOffset = DIFF_FC1_WEIGHT_ELEM * 2, .weightSize = 1024 * 2,
    .biasOffset = DIFF_FC1_BIAS_ELEM * 2, .biasSize = 64 * 2,
    .inChannels = 16, .outChannels = 64, .stride = 16
};

inline constexpr LayerDescriptor LAYER_DIFF_FC2 = {
    .weightOffset = DIFF_FC2_WEIGHT_ELEM * 2, .weightSize = 4096 * 2,
    .biasOffset = DIFF_FC2_BIAS_ELEM * 2, .biasSize = 64 * 2,
    .inChannels = 64, .outChannels = 64, .stride = 16
};

inline constexpr LayerDescriptor LAYER_DIFF_FC3 = {
    .weightOffset = DIFF_FC3_WEIGHT_ELEM * 2, .weightSize = 1024 * 2,
    .biasOffset = DIFF_FC3_BIAS_ELEM * 2, .biasSize = 16 * 2,
    .inChannels = 64, .outChannels = 16, .stride = 16
};

// Specular Layer Descriptors
inline constexpr LayerDescriptor LAYER_SPEC_FC1 = {
    .weightOffset = SPEC_FC1_WEIGHT_ELEM * 2, .weightSize = 1024 * 2,
    .biasOffset = SPEC_FC1_BIAS_ELEM * 2, .biasSize = 64 * 2,
    .inChannels = 16, .outChannels = 64, .stride = 16
};

inline constexpr LayerDescriptor LAYER_SPEC_FC2 = {
    .weightOffset = SPEC_FC2_WEIGHT_ELEM * 2, .weightSize = 4096 * 2,
    .biasOffset = SPEC_FC2_BIAS_ELEM * 2, .biasSize = 64 * 2,
    .inChannels = 64, .outChannels = 64, .stride = 16
};

inline constexpr LayerDescriptor LAYER_SPEC_FC3 = {
    .weightOffset = SPEC_FC3_WEIGHT_ELEM * 2, .weightSize = 1024 * 2,
    .biasOffset = SPEC_FC3_BIAS_ELEM * 2, .biasSize = 16 * 2,
    .inChannels = 64, .outChannels = 16, .stride = 16
};

} // namespace upways

namespace upways3 = upways;
