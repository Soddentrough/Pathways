#pragma once

#include <cstdint>
#include <cstddef>
#include <glm/glm.hpp>

namespace pathways {

// 32-byte ReSTIR DI reservoir. Must match `DiReservoir` in
// shaders/compute/restir_common.glsl (scalar-only std430 layout).
struct DiReservoirGPU {
    uint32_t idxM;        // [15:0] light index, [31:16] M
    float    wSum;        // RIS weight sum
    float    targetPdf;   // p-hat of the selected sample at the storing pixel
    uint32_t flags;       // [7:0] age, [8] valid, [9] trans lobe, [25:10] oct16(x1 normal)
    float    hitDist;     // x1 -> light-sample distance
    uint32_t lightPacked; // [15:0] oct16(light normal), [31:16] reserved
    uint32_t x1Misc;      // [15:0] fp16 cos(theta at light), [31:16] fp16 primary hit distance
    uint32_t lightUv16;   // packUnorm2x16(luv) — full 16-bit area-light sample UV
};
static_assert(sizeof(DiReservoirGPU) == 32, "DiReservoirGPU must be exactly 32 bytes (GLSL std430 scalar layout)");

// 16-byte Hit atom: canonical geometry hit traded across trace, shift, and shade kernels.
struct HitGPU {
    uint32_t instanceId;  // TLAS instance custom index
    uint32_t primitiveId; // Triangle index in BLAS geometry (or sphere index)
    uint32_t baryPacked;  // packUnorm2x16(bary.xy) or oct32 normal
    uint32_t flags;       // [0]: isTriangle (0: tri, 1: sphere), [1]: frontFace, [6:2]: lobe/mat flags
};
static_assert(sizeof(HitGPU) == 16, "HitGPU must be exactly 16 bytes");

// 32-byte ReSTIR GI path-space secondary reconnection reservoir (world-space x2).
// Must match `GiReservoir` in shaders/compute/restir_common.glsl.
struct GiReservoirGPU {
    uint32_t posXBits;      // floatBitsToUint(x2.x) world-space reconnection vertex
    uint32_t posYBits;      // floatBitsToUint(x2.y)
    uint32_t posZBits;      // floatBitsToUint(x2.z)
    uint32_t wSumBits;      // fp32 bits (RIS running weight sum)
    uint32_t targetCosBits; // [31:8] fp32(target)[31:8], [7:0] cosSrc (unorm8)
    uint32_t radRgb9e5;     // RGB9E5 shared-exponent radiance L2 at x2
    uint32_t normalsPacked; // [15:0] oct16(n2), [31:16] oct16(x1Normal)
    uint32_t dSrcFlagsM;    // [15:0] fp16 |x2 - x1|, [16] valid, [17] hasSurface, [23:18] age (0..63), [31:24] M (0..255)
};
static_assert(sizeof(GiReservoirGPU) == 32, "GiReservoirGPU must be exactly 32 bytes");

// 24-byte per-pixel BRDF context at the primary hit x1, written at bounce 0
// and consumed at bounce 1 (and restir_debug_view). Must match `X1Context` in GLSL.
struct X1ContextGPU {
    uint32_t n1Oct;       // oct32(x1 world shading normal)
    uint32_t albedoRG;    // fp16 (baseColor.r, baseColor.g)
    uint32_t albedoB;     // [15:0] fp16 baseColor.b, [31:16] fp16 metallic
    uint32_t flags;       // [0] valid, [1] reserved, [31:2] qSelf fp32[31:2] (BSDF mixture pdf at x1)
    uint32_t woOct;       // oct32(x1 -> camera direction)
    uint32_t f0Roughness; // [15:0] fp16 dielectric scalar F0, [31:16] fp16 roughness
};
static_assert(sizeof(X1ContextGPU) == 24, "X1ContextGPU must be exactly 24 bytes");

} // namespace pathways
