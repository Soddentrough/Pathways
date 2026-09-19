#pragma once

// Auto-generated Pure Neural Reconstructor FP16 WMMA Weights Header
// Matrix Layout: Transposed (C_in, C_out) Row-Major with stride C_out
#include <cstdint>

namespace upways {

struct LayerDescriptor {
    uint32_t weightOffset; // Byte offset from start of SSBO
    uint32_t weightSize;   // Size in bytes
    uint32_t biasOffset;   // Bias byte offset (16-byte aligned)
    uint32_t biasSize;     // Bias size in bytes
    uint32_t outChannels;  // C_out
    uint32_t inChannels;   // C_in
    uint32_t stride;       // Row stride (C_out)
};

constexpr uint32_t TOTAL_WEIGHT_BUFFER_SIZE = 22944;

// Layer: fc1
inline constexpr LayerDescriptor LAYER_FC1 = {
    .weightOffset = 0,
    .weightSize   = 4096,
    .biasOffset   = 4096,
    .biasSize     = 128,
    .outChannels  = 64,
    .inChannels   = 32,
    .stride       = 64
};

// Layer: fc2
inline constexpr LayerDescriptor LAYER_FC2 = {
    .weightOffset = 4224,
    .weightSize   = 8192,
    .biasOffset   = 12416,
    .biasSize     = 128,
    .outChannels  = 64,
    .inChannels   = 64,
    .stride       = 64
};

// Layer: fc3
inline constexpr LayerDescriptor LAYER_FC3 = {
    .weightOffset = 12544,
    .weightSize   = 8192,
    .biasOffset   = 20736,
    .biasSize     = 128,
    .outChannels  = 64,
    .inChannels   = 64,
    .stride       = 64
};

// Layer: fc4
inline constexpr LayerDescriptor LAYER_FC4 = {
    .weightOffset = 20864,
    .weightSize   = 2048,
    .biasOffset   = 22912,
    .biasSize     = 32,
    .outChannels  = 16,
    .inChannels   = 64,
    .stride       = 16
};

} // namespace upways

namespace upways3 = upways;
