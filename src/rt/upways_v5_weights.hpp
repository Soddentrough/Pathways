#pragma once
// Upways 5.0 Pure-ML Neural Reconstructor Weight Layout Metadata (158,400 params, 190,208 B).
// Matches Upways/upways/export.py & Upways/checkpoints/upways_weights.hpp strictly 1:1.

#include <cstddef>
#include <cstdint>

namespace upways_v5 {

inline constexpr size_t FP8_WEIGHT_BYTES = 150016u;
inline constexpr size_t INT8_WEIGHT_BYTES = 150016u;
inline constexpr size_t FP32_WEIGHT_BYTES = 40192u;
inline constexpr size_t TOTAL_WEIGHT_BYTES = 190208u;
inline constexpr size_t TOTAL_WEIGHT_BUFFER_SIZE = 190208u;
inline constexpr size_t FP8_WEIGHT_SECTION_OFFSET = 0u;
inline constexpr size_t FP8_WEIGHT_SECTION_BYTES = 150016u;
inline constexpr size_t FP32_WEIGHT_SECTION_OFFSET = 150016u;
inline constexpr size_t FP32_WEIGHT_SECTION_BYTES = 40192u;
inline constexpr size_t TOTAL_PARAMETER_COUNT = 158400u;
inline constexpr size_t ACTIVE_PARAMETER_COUNT = 158400u;

// Pointwise 1x1 16x16 col-major tile offsets (in uint32 elements within 150,016 B section)
inline constexpr uint32_t U32_W_ENC1_PW1 = 0u;
inline constexpr uint32_t U32_W_ENC1_PW2 = 1536u;
inline constexpr uint32_t U32_W_ENC2_PW1 = 2112u;
inline constexpr uint32_t U32_W_ENC2_PW2 = 3072u;
inline constexpr uint32_t U32_W_BOT3_PW1 = 4672u;
inline constexpr uint32_t U32_W_BOT3_PW2 = 7232u;
inline constexpr uint32_t U32_W_BOT3_PW3 = 11328u;
inline constexpr uint32_t U32_W_BOT3_PW4 = 15424u;
inline constexpr uint32_t U32_W_BOT3_KHEAD = 18496u;
inline constexpr uint32_t U32_W_DEC2_PW1 = 19264u;
inline constexpr uint32_t U32_W_DEC2_PW2 = 23488u;
inline constexpr uint32_t U32_W_DEC2_KHEAD = 25408u;
inline constexpr uint32_t U32_W_DEC1_PW1 = 26048u;
inline constexpr uint32_t U32_W_DEC1_PW2 = 28096u;
inline constexpr uint32_t U32_W_DEC1_KHEAD = 29120u;
inline constexpr uint32_t U32_W_DEN_HEAD = 29632u;
inline constexpr uint32_t U32_W_SR_PW1 = 31680u;
inline constexpr uint32_t U32_W_SR_PW2 = 32704u;
inline constexpr uint32_t U32_W_SR_PW3 = 33728u;
inline constexpr uint32_t U32_W_SR_PW4 = 34752u;
inline constexpr uint32_t U32_W_SR_HEAD = 35776u;
inline constexpr uint32_t U32_W_HR_POS_MLP = 37312u;
inline constexpr uint32_t U32_W_HR_TEMP_MLP1 = 37376u;
inline constexpr uint32_t U32_W_HR_TEMP_MLP2 = 37440u;

} // namespace upways_v5
