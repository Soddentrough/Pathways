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

constexpr uint32_t TOTAL_WEIGHT_BUFFER_SIZE = 2121536;

// Layer: guide_branch_proj
inline constexpr LayerDescriptor LAYER_GUIDE_BRANCH_PROJ = {
    .weightOffset = 0,
    .weightSize   = 4608,
    .biasOffset   = 4608,
    .biasSize     = 32,
    .outChannels  = 16,
    .inChannels   = 16,
    .stride       = 16
};

// Layer: guide_branch_block1_dwconv
inline constexpr LayerDescriptor LAYER_GUIDE_BRANCH_BLOCK1_DWCONV = {
    .weightOffset = 4640,
    .weightSize   = 288,
    .biasOffset   = 4928,
    .biasSize     = 32,
    .outChannels  = 16,
    .inChannels   = 1,
    .stride       = 16
};

// Layer: guide_branch_block1_pwconv1
inline constexpr LayerDescriptor LAYER_GUIDE_BRANCH_BLOCK1_PWCONV1 = {
    .weightOffset = 4960,
    .weightSize   = 1024,
    .biasOffset   = 5984,
    .biasSize     = 64,
    .outChannels  = 32,
    .inChannels   = 16,
    .stride       = 32
};

// Layer: guide_branch_block1_pwconv2
inline constexpr LayerDescriptor LAYER_GUIDE_BRANCH_BLOCK1_PWCONV2 = {
    .weightOffset = 6048,
    .weightSize   = 1024,
    .biasOffset   = 7072,
    .biasSize     = 32,
    .outChannels  = 16,
    .inChannels   = 32,
    .stride       = 16
};

// Layer: guide_branch_up_proj
inline constexpr LayerDescriptor LAYER_GUIDE_BRANCH_UP_PROJ = {
    .weightOffset = 7104,
    .weightSize   = 1024,
    .biasOffset   = 8128,
    .biasSize     = 64,
    .outChannels  = 32,
    .inChannels   = 16,
    .stride       = 32
};

// Layer: guide_branch_block2_dwconv
inline constexpr LayerDescriptor LAYER_GUIDE_BRANCH_BLOCK2_DWCONV = {
    .weightOffset = 8192,
    .weightSize   = 576,
    .biasOffset   = 8768,
    .biasSize     = 64,
    .outChannels  = 32,
    .inChannels   = 1,
    .stride       = 32
};

// Layer: guide_branch_block2_pwconv1
inline constexpr LayerDescriptor LAYER_GUIDE_BRANCH_BLOCK2_PWCONV1 = {
    .weightOffset = 8832,
    .weightSize   = 4096,
    .biasOffset   = 12928,
    .biasSize     = 128,
    .outChannels  = 64,
    .inChannels   = 32,
    .stride       = 64
};

// Layer: guide_branch_block2_pwconv2
inline constexpr LayerDescriptor LAYER_GUIDE_BRANCH_BLOCK2_PWCONV2 = {
    .weightOffset = 13056,
    .weightSize   = 4096,
    .biasOffset   = 17152,
    .biasSize     = 64,
    .outChannels  = 32,
    .inChannels   = 64,
    .stride       = 32
};

// Layer: enc1_proj
inline constexpr LayerDescriptor LAYER_ENC1_PROJ = {
    .weightOffset = 17216,
    .weightSize   = 3072,
    .biasOffset   = 20288,
    .biasSize     = 96,
    .outChannels  = 48,
    .inChannels   = 32,
    .stride       = 48
};

// Layer: enc1_pconv_dwconv
inline constexpr LayerDescriptor LAYER_ENC1_PCONV_DWCONV = {
    .weightOffset = 20384,
    .weightSize   = 288,
    .biasOffset   = 20672,
    .biasSize     = 32,
    .outChannels  = 16,
    .inChannels   = 1,
    .stride       = 16
};

// Layer: enc1_pconv_pwconv
inline constexpr LayerDescriptor LAYER_ENC1_PCONV_PWCONV = {
    .weightOffset = 20704,
    .weightSize   = 4608,
    .biasOffset   = 25312,
    .biasSize     = 96,
    .outChannels  = 48,
    .inChannels   = 48,
    .stride       = 48
};

// Layer: enc1_block_dwconv
inline constexpr LayerDescriptor LAYER_ENC1_BLOCK_DWCONV = {
    .weightOffset = 25408,
    .weightSize   = 864,
    .biasOffset   = 26272,
    .biasSize     = 96,
    .outChannels  = 48,
    .inChannels   = 1,
    .stride       = 48
};

// Layer: enc1_block_pwconv1
inline constexpr LayerDescriptor LAYER_ENC1_BLOCK_PWCONV1 = {
    .weightOffset = 26368,
    .weightSize   = 9216,
    .biasOffset   = 35584,
    .biasSize     = 192,
    .outChannels  = 96,
    .inChannels   = 48,
    .stride       = 96
};

// Layer: enc1_block_pwconv2
inline constexpr LayerDescriptor LAYER_ENC1_BLOCK_PWCONV2 = {
    .weightOffset = 35776,
    .weightSize   = 9216,
    .biasOffset   = 44992,
    .biasSize     = 96,
    .outChannels  = 48,
    .inChannels   = 96,
    .stride       = 48
};

// Layer: down1
inline constexpr LayerDescriptor LAYER_DOWN1 = {
    .weightOffset = 45088,
    .weightSize   = 30720,
    .biasOffset   = 75808,
    .biasSize     = 160,
    .outChannels  = 80,
    .inChannels   = 48,
    .stride       = 80
};

// Layer: enc2_block1_dwconv
inline constexpr LayerDescriptor LAYER_ENC2_BLOCK1_DWCONV = {
    .weightOffset = 75968,
    .weightSize   = 7840,
    .biasOffset   = 83808,
    .biasSize     = 160,
    .outChannels  = 80,
    .inChannels   = 1,
    .stride       = 80
};

// Layer: enc2_block1_pwconv1
inline constexpr LayerDescriptor LAYER_ENC2_BLOCK1_PWCONV1 = {
    .weightOffset = 83968,
    .weightSize   = 25600,
    .biasOffset   = 109568,
    .biasSize     = 320,
    .outChannels  = 160,
    .inChannels   = 80,
    .stride       = 160
};

// Layer: enc2_block1_pwconv2
inline constexpr LayerDescriptor LAYER_ENC2_BLOCK1_PWCONV2 = {
    .weightOffset = 109888,
    .weightSize   = 25600,
    .biasOffset   = 135488,
    .biasSize     = 160,
    .outChannels  = 80,
    .inChannels   = 160,
    .stride       = 80
};

// Layer: enc2_block2_dwconv
inline constexpr LayerDescriptor LAYER_ENC2_BLOCK2_DWCONV = {
    .weightOffset = 135648,
    .weightSize   = 7840,
    .biasOffset   = 143488,
    .biasSize     = 160,
    .outChannels  = 80,
    .inChannels   = 1,
    .stride       = 80
};

// Layer: enc2_block2_pwconv1
inline constexpr LayerDescriptor LAYER_ENC2_BLOCK2_PWCONV1 = {
    .weightOffset = 143648,
    .weightSize   = 25600,
    .biasOffset   = 169248,
    .biasSize     = 320,
    .outChannels  = 160,
    .inChannels   = 80,
    .stride       = 160
};

// Layer: enc2_block2_pwconv2
inline constexpr LayerDescriptor LAYER_ENC2_BLOCK2_PWCONV2 = {
    .weightOffset = 169568,
    .weightSize   = 25600,
    .biasOffset   = 195168,
    .biasSize     = 160,
    .outChannels  = 80,
    .inChannels   = 160,
    .stride       = 80
};

// Layer: down2
inline constexpr LayerDescriptor LAYER_DOWN2 = {
    .weightOffset = 195328,
    .weightSize   = 71680,
    .biasOffset   = 267008,
    .biasSize     = 224,
    .outChannels  = 112,
    .inChannels   = 80,
    .stride       = 112
};

// Layer: bottleneck_gru_gate_conv
inline constexpr LayerDescriptor LAYER_BOTTLENECK_GRU_GATE_CONV = {
    .weightOffset = 267232,
    .weightSize   = 903168,
    .biasOffset   = 1170400,
    .biasSize     = 448,
    .outChannels  = 224,
    .inChannels   = 224,
    .stride       = 224
};

// Layer: bottleneck_gru_cand_conv
inline constexpr LayerDescriptor LAYER_BOTTLENECK_GRU_CAND_CONV = {
    .weightOffset = 1170848,
    .weightSize   = 451584,
    .biasOffset   = 1622432,
    .biasSize     = 224,
    .outChannels  = 112,
    .inChannels   = 224,
    .stride       = 112
};

// Layer: bottleneck_gru_infill_conv
inline constexpr LayerDescriptor LAYER_BOTTLENECK_GRU_INFILL_CONV = {
    .weightOffset = 1622656,
    .weightSize   = 225792,
    .biasOffset   = 1848448,
    .biasSize     = 224,
    .outChannels  = 112,
    .inChannels   = 112,
    .stride       = 112
};

// Layer: bottleneck_block_dwconv
inline constexpr LayerDescriptor LAYER_BOTTLENECK_BLOCK_DWCONV = {
    .weightOffset = 1848672,
    .weightSize   = 10976,
    .biasOffset   = 1859648,
    .biasSize     = 224,
    .outChannels  = 112,
    .inChannels   = 1,
    .stride       = 112
};

// Layer: bottleneck_block_pwconv1
inline constexpr LayerDescriptor LAYER_BOTTLENECK_BLOCK_PWCONV1 = {
    .weightOffset = 1859872,
    .weightSize   = 50176,
    .biasOffset   = 1910048,
    .biasSize     = 448,
    .outChannels  = 224,
    .inChannels   = 112,
    .stride       = 224
};

// Layer: bottleneck_block_pwconv2
inline constexpr LayerDescriptor LAYER_BOTTLENECK_BLOCK_PWCONV2 = {
    .weightOffset = 1910496,
    .weightSize   = 50176,
    .biasOffset   = 1960672,
    .biasSize     = 224,
    .outChannels  = 112,
    .inChannels   = 224,
    .stride       = 112
};

// Layer: dec2_fuse
inline constexpr LayerDescriptor LAYER_DEC2_FUSE = {
    .weightOffset = 1960896,
    .weightSize   = 25600,
    .biasOffset   = 1986496,
    .biasSize     = 160,
    .outChannels  = 80,
    .inChannels   = 160,
    .stride       = 80
};

// Layer: dec2_block_dwconv
inline constexpr LayerDescriptor LAYER_DEC2_BLOCK_DWCONV = {
    .weightOffset = 1986656,
    .weightSize   = 7840,
    .biasOffset   = 1994496,
    .biasSize     = 160,
    .outChannels  = 80,
    .inChannels   = 1,
    .stride       = 80
};

// Layer: dec2_block_pwconv1
inline constexpr LayerDescriptor LAYER_DEC2_BLOCK_PWCONV1 = {
    .weightOffset = 1994656,
    .weightSize   = 25600,
    .biasOffset   = 2020256,
    .biasSize     = 320,
    .outChannels  = 160,
    .inChannels   = 80,
    .stride       = 160
};

// Layer: dec2_block_pwconv2
inline constexpr LayerDescriptor LAYER_DEC2_BLOCK_PWCONV2 = {
    .weightOffset = 2020576,
    .weightSize   = 25600,
    .biasOffset   = 2046176,
    .biasSize     = 160,
    .outChannels  = 80,
    .inChannels   = 160,
    .stride       = 80
};

// Layer: dec1_fuse
inline constexpr LayerDescriptor LAYER_DEC1_FUSE = {
    .weightOffset = 2046336,
    .weightSize   = 12288,
    .biasOffset   = 2058624,
    .biasSize     = 96,
    .outChannels  = 48,
    .inChannels   = 128,
    .stride       = 48
};

// Layer: dec1_block_dwconv
inline constexpr LayerDescriptor LAYER_DEC1_BLOCK_DWCONV = {
    .weightOffset = 2058720,
    .weightSize   = 864,
    .biasOffset   = 2059584,
    .biasSize     = 96,
    .outChannels  = 48,
    .inChannels   = 1,
    .stride       = 48
};

// Layer: dec1_block_pwconv1
inline constexpr LayerDescriptor LAYER_DEC1_BLOCK_PWCONV1 = {
    .weightOffset = 2059680,
    .weightSize   = 9216,
    .biasOffset   = 2068896,
    .biasSize     = 192,
    .outChannels  = 96,
    .inChannels   = 48,
    .stride       = 96
};

// Layer: dec1_block_pwconv2
inline constexpr LayerDescriptor LAYER_DEC1_BLOCK_PWCONV2 = {
    .weightOffset = 2069088,
    .weightSize   = 9216,
    .biasOffset   = 2078304,
    .biasSize     = 96,
    .outChannels  = 48,
    .inChannels   = 96,
    .stride       = 48
};

// Layer: out_head_0
inline constexpr LayerDescriptor LAYER_OUT_HEAD_0 = {
    .weightOffset = 2078400,
    .weightSize   = 41472,
    .biasOffset   = 2119872,
    .biasSize     = 96,
    .outChannels  = 48,
    .inChannels   = 48,
    .stride       = 48
};

// Layer: out_head_2
inline constexpr LayerDescriptor LAYER_OUT_HEAD_2 = {
    .weightOffset = 2119968,
    .weightSize   = 1536,
    .biasOffset   = 2121504,
    .biasSize     = 32,
    .outChannels  = 16,
    .inChannels   = 48,
    .stride       = 16
};

} // namespace upways

namespace upways3 = upways;
