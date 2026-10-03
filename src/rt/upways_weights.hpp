#pragma once

// Auto-generated Upways Wave32 WMMA KPN Reconstructor FP16 Weights Header
// Matching shaders/compute/neural_reconstruct.comp
#include <cstdint>

namespace upways {

struct KPNWeightsBuffer {
    uint16_t w_l1[32 * 128]; // Offset 0, 8192 bytes
    uint16_t w_l2[128 * 32]; // Offset 8192, 8192 bytes
    uint16_t b_l1[128];      // Offset 16384, 256 bytes
    uint16_t b_l2[32];      // Offset 16640, 64 bytes
};

constexpr uint32_t KPN_W_L1_OFFSET = 0;
constexpr uint32_t KPN_W_L1_SIZE   = 8192;
constexpr uint32_t KPN_W_L2_OFFSET = 8192;
constexpr uint32_t KPN_W_L2_SIZE   = 8192;
constexpr uint32_t KPN_B_L1_OFFSET = 16384;
constexpr uint32_t KPN_B_L1_SIZE   = 256;
constexpr uint32_t KPN_B_L2_OFFSET = 16640;
constexpr uint32_t KPN_B_L2_SIZE   = 64;

constexpr uint32_t TOTAL_KPN_WEIGHT_BUFFER_SIZE = 16704;
constexpr uint32_t TOTAL_WEIGHT_BUFFER_SIZE = 16704;

struct LayerDescriptor {
    uint32_t weightOffset;
    uint32_t weightSize;
    uint32_t biasOffset;
    uint32_t biasSize;
    uint32_t outChannels;
    uint32_t inChannels;
    uint32_t stride;
};

// Layer: fc1
inline constexpr LayerDescriptor LAYER_FC1 = {
    .weightOffset = KPN_W_L1_OFFSET,
    .weightSize   = KPN_W_L1_SIZE,
    .biasOffset   = KPN_B_L1_OFFSET,
    .biasSize     = KPN_B_L1_SIZE,
    .outChannels  = 128,
    .inChannels   = 32,
    .stride       = 128
};

// Layer: fc2
inline constexpr LayerDescriptor LAYER_FC2 = {
    .weightOffset = KPN_W_L2_OFFSET,
    .weightSize   = KPN_W_L2_SIZE,
    .biasOffset   = KPN_B_L2_OFFSET,
    .biasSize     = KPN_B_L2_SIZE,
    .outChannels  = 32,
    .inChannels   = 128,
    .stride       = 32
};

} // namespace upways

namespace upways3 = upways;
