#pragma once

#include <cstdint>
#include <cstddef>
#include <glm/glm.hpp>

namespace pathways {

// 28-byte ReSTIR DI reservoir. Must match `DiReservoir` in
// shaders/compute/restir_common.glsl (scalar-only std430 layout).
struct DiReservoirGPU {
    uint32_t idxM;        // [15:0] light index, [31:16] M
    float    wSum;        // RIS weight sum
    float    targetPdf;   // p-hat of the selected sample at the storing pixel
    uint32_t flags;       // [7:0] age, [8] valid, [9] trans lobe, [25:10] oct16(x1 normal)
    float    hitDist;     // x1 -> light-sample distance
    uint32_t lightPacked; // [15:0] oct16(light normal), [23:16] u8, [31:24] v8
    uint32_t x1Misc;      // [15:0] fp16 cos(theta at light), [31:16] fp16 primary hit distance
};
static_assert(sizeof(DiReservoirGPU) == 28, "DiReservoirGPU must be exactly 28 bytes (GLSL std430 scalar layout)");

// 16-byte Hit atom: canonical geometry hit traded across trace, shift, and shade kernels.
struct HitGPU {
    uint32_t instanceId;  // TLAS instance custom index
    uint32_t primitiveId; // Triangle index in BLAS geometry (or sphere index)
    uint32_t baryPacked;  // packUnorm2x16(bary.xy) or oct32 normal
    uint32_t flags;       // [0]: isTriangle (0: tri, 1: sphere), [1]: frontFace, [6:2]: lobe/mat flags
};
static_assert(sizeof(HitGPU) == 16, "HitGPU must be exactly 16 bytes");

// 32-byte ShadeRequest: emitted by shift/trace kernels to request BSDF evaluation (Approach A).
struct ShadeRequestGPU {
    HitGPU   hit;          // 16 bytes: hit atom at the evaluated vertex
    uint32_t wiOct;        // 4 bytes: octahedral incoming direction (x_{k-1} -> x_k)
    uint32_t woOct;        // 4 bytes: octahedral outgoing direction (x_k -> x_{k+1})
    uint32_t resultSlot;   // 4 bytes: linear index into ShadeResult buffer
    uint32_t matIdFlags;   // 4 bytes: [23:0] materialId, [31:24] lobeMask
};
static_assert(sizeof(ShadeRequestGPU) == 32, "ShadeRequestGPU must be exactly 32 bytes");

// 16-byte ShadeResult: output of binned material shading kernels (Approach A).
struct ShadeResultGPU {
    uint32_t fRG;          // 4 bytes: fp16 (f.r, f.g) where f = BSDF * |cos|
    uint32_t fB_pdfFwd;    // 4 bytes: fp16 f.b, fp16 forward pdf
    uint32_t pdfRev_flags; // 4 bytes: fp16 reverse pdf, [31:16] lobe flags
    uint32_t pad;          // 4 bytes: alignment / reserved
};
static_assert(sizeof(ShadeResultGPU) == 16, "ShadeResultGPU must be exactly 16 bytes");

// 32-byte ReSTIR GI reservoir (path-space, docs/RESTIR_PT_DESIGN.md).
// Must match `GiReservoir` in shaders/compute/restir_common.glsl.
struct GiReservoirGPU {
    uint32_t hitInstanceId;  // TLAS instance custom index
    uint32_t hitPrimitiveId; // Triangle index in BLAS geometry (or sphere index)
    uint32_t hitBaryPacked;  // packUnorm2x16(bary.xy) or oct32 normal
    uint32_t wSum;           // fp32 bits
    uint32_t target;         // fp32 bits, p-hat at the recording pixel
    uint32_t radRG;          // fp16 (L2.r, L2.g)
    uint32_t radB_dSrc;      // fp16 L2.b, fp16 |x2 - x1| at recording time
    uint32_t flagsM;         // [0] valid, [8:1] age, [18:9] M, [20:19] hitFlags, [28:21] cosSrc (unorm8), [31:29] reserved
};
static_assert(sizeof(GiReservoirGPU) == 32, "GiReservoirGPU must be exactly 32 bytes");

// 24-byte per-pixel BRDF context at the primary hit x1, written at bounce 0
// and consumed at bounce 1 of the same frame. Must match `X1Context` in GLSL.
struct X1ContextGPU {
    uint32_t n1Oct;       // oct32(x1 normal)
    uint32_t albedoRG;    // fp16 (albedo.r, albedo.g)
    uint32_t albedoB;     // fp16 albedo.b
    uint32_t flags;       // [0] valid (diffuse cosine reflection lobe)
    uint32_t woOct;       // oct32(x1 -> camera direction)
    uint32_t f0Roughness; // [15:0] fp16 scalar F0, [31:16] fp16 roughness
};
static_assert(sizeof(X1ContextGPU) == 24, "X1ContextGPU must be exactly 24 bytes");

// 32-byte PathVertex atom: geometric vertex in a path skeleton (Approach B: Fully Inverted Path-Database).
struct PathVertexGPU {
    HitGPU   hit;        // 16 bytes: TLAS instance, BLAS primitive, unorm16 barycentrics, flags
    uint32_t woOct;      // 4 bytes: octahedral outgoing direction towards predecessor
    uint32_t matId;      // 4 bytes: integer material ID for deferred gather routing
    uint32_t vertexType; // 4 bytes: 0: CAMERA, 1: SURFACE_DIFFUSE, 2: SURFACE_GLOSSY, 3: SURFACE_DELTA, 4: LIGHT
    uint32_t flags;      // 4 bytes: [7:0] subpath index, [15:8] lobe flags, [31:16] reserved
};
static_assert(sizeof(PathVertexGPU) == 32, "PathVertexGPU must be exactly 32 bytes (std430 aligned)");

// Maximum vertex depth supported in the path skeleton pool.
// Typically 4 vertices: x1 (primary), x2 (secondary), x3 (tertiary), x4 (quaternary/light).
inline constexpr uint32_t PATH_SAMPLE_MAX_VERTICES = 4;

// 160-byte PathSample: compact geometric subpath skeleton with stateless RNG replay seeds.
struct PathSampleGPU {
    uint32_t      vertexCount;      // 4 bytes: effective length of subpath (e.g. 2 to 4)
    uint32_t      rngSeedPrefix;    // 4 bytes: replay seed for vertices before reconnection
    uint32_t      rngSeedSuffix;    // 4 bytes: replay seed for vertices after reconnection
    uint32_t      strategyId;       // 4 bytes: sampling provenance (unidirectional, NEE, BSDF)
    uint32_t      flags;            // 4 bytes: [0]: isComplete, [1]: hasLightVertex, [31:2]: reserved
    uint32_t      rcIndex;          // 4 bytes: reconnection vertex index k (default 2)
    uint32_t      reserved[2];      // 8 bytes: alignment / reserved
    PathVertexGPU vertices[PATH_SAMPLE_MAX_VERTICES]; // 4 * 32 = 128 bytes
};
static_assert(sizeof(PathSampleGPU) == 160, "PathSampleGPU must be exactly 160 bytes (16-byte aligned)");

// 32-byte ReSTIR PT reservoir (Approach B: Fully Inverted Path-Database).
struct ReservoirPTGPU {
    float    weightSum;    // 4 bytes: sum of candidate sample weights (w_sum)
    float    W;            // 4 bytes: unbiased contribution weight: W = w_sum / (M * targetProxy)
    float    M;            // 4 bytes: confidence counter (candidate count)
    float    targetProxy;  // 4 bytes: geometric target proxy p_hat(X)
    uint32_t pathIndex;    // 4 bytes: index into PathSample pool
    uint32_t age;          // 4 bytes: temporal age in frames
    uint32_t flags;        // 4 bytes: [0]: valid, [1]: isSpecular, [31:2]: reserved
    uint32_t pad;          // 4 bytes: 32-byte cache-line alignment
};
static_assert(sizeof(ReservoirPTGPU) == 32, "ReservoirPTGPU must be exactly 32 bytes");

} // namespace pathways
