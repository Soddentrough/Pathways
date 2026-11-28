// Upways 4.0 shared GLSL definitions, layout constants, FP8/FP16 helpers, and explicit bilinear warps.
// Must remain strictly identical between Upways and Pathways repositories.

#ifndef UPWAYS_PRECISION_MODE
#ifdef UPWAYS_FP16_WMMA
#define UPWAYS_PRECISION_MODE 2
#else
#define UPWAYS_PRECISION_MODE 0
#endif
#endif

#define UPWAYS_IS_FP8  (UPWAYS_PRECISION_MODE == 0)
#define UPWAYS_IS_INT8 (UPWAYS_PRECISION_MODE == 1)
#define UPWAYS_IS_FP16 (UPWAYS_PRECISION_MODE == 2)

#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#if UPWAYS_IS_FP8
#extension GL_EXT_float_e4m3 : require
#endif

#include "rgb9e5.glsl"

// -----------------------------------------------------------------------------
// Push Constants (128 bytes exact)
// -----------------------------------------------------------------------------
layout(push_constant) uniform UpwaysPC {
    uint  lrWidth;        // 0
    uint  lrHeight;       // 4
    uint  hrWidth;        // 8
    uint  hrHeight;       // 12
    float jitterX;        // 16 (LR pixels, Camera::ubo.jitterOffset.x)
    float jitterY;        // 20 (LR pixels, Camera::ubo.jitterOffset.y)
    uint  frameIndex;     // 24
    uint  flags;          // 28: bit 0 = resetHistory, bit 1 = enableSuperRes, bit 2 = zeroResidualMode
    uint  offX0;          // 32 (uint32 element offset in DenoiseActFp8Buf)
    uint  offHw0;         // 36 (shared with offSrIn)
    uint  offS0;          // 40
    uint  offB1;          // 44
    uint  offB2;          // 48
    float exposureScale;  // 52
    uint  bandLrY0;       // 56 (band start Y in full LR frame)
    uint  bandHrY0;       // 60 (band start Y in full HR frame)
    uint  fullLrHeight;   // 64 (full LR frame height)
    uint  fullHrHeight;   // 68 (full HR frame height)
    uint  keepHrYStart;   // 72 (first band-local HR row written to uOutputColor)
    uint  keepHrYEnd;     // 76 (exclusive band-local HR row written to uOutputColor)
    vec4  prevToCurrViewQuat;   // 80  (xyzw unit quat rotating prev view -> curr view)
    vec4  currToPrevTransView;  // 96  (xyz = (prevView * inverse(currView))[3].xyz, w = tanHalfFovY)
    vec4  camMotionParams;      // 112 (x = aspectRatio, y = camMotionValid [1.0 when live camera], z/w = 0)
} pc;

const uint UPWAYS_FLAG_RESET_HISTORY      = 1u;
const uint UPWAYS_FLAG_ENABLE_SUPERRES    = 2u;
const uint UPWAYS_FLAG_ZERO_RESIDUAL_MODE = 4u;
const uint UPWAYS_FLAG_FUSED_BAND_PACK    = 8u;

// -----------------------------------------------------------------------------
// Weight Offsets (Total = 24,064 bytes: 15,616 B FP8 + 8,448 B FP32)
// -----------------------------------------------------------------------------
// FP8 E4M3fn Pointwise matrices: uint32 offsets inside WeightFp8Buf (3,904 uint32s)
// Each 16x16 (K=16, N=16) col-major tile occupies 64 uint32s (256 B), stride = 4 uint32s.
const uint U32_W_ENC1_PW1 = 0u;     // 96 -> 32  (6x2 tiles = 768 u32)
const uint U32_W_ENC1_PW2 = 768u;   // 32 -> 32  (2x2 tiles = 256 u32)
const uint U32_W_BOT2_PW1 = 1024u;  // 32 -> 64  (2x4 tiles = 512 u32)
const uint U32_W_BOT2_PW2 = 1536u;  // 64 -> 32  (4x2 tiles = 512 u32)
const uint U32_W_DEC1_PW1 = 2048u;  // 64 -> 48  (4x3 tiles = 768 u32)
const uint U32_W_DEN_HEAD = 2816u;  // 48 -> 48  (3x3 tiles = 576 u32, 4x12 active outputs)
const uint U32_W_SR_PW1   = 3392u;  // 16 -> 32  (1x2 tiles = 128 u32)
const uint U32_W_SR_PW2   = 3520u;  // 32 -> 32  (2x2 tiles = 256 u32)
const uint U32_W_SR_HEAD  = 3776u;  // 32 -> 16  (2x1 tiles = 128 u32)
const uint TOTAL_WEIGHT_FP8_U32 = 3904u;

// FP32 Scales, Biases, and Depthwise 3x3 ([9][C]): float offsets inside WeightFp32Buf (2,112 floats)
const uint F32_S_ENC1_PW1 = 0u;
const uint F32_S_ENC1_PW2 = 32u;
const uint F32_S_BOT2_PW1 = 64u;
const uint F32_S_BOT2_PW2 = 128u;
const uint F32_S_DEC1_PW1 = 160u;
const uint F32_S_DEN_HEAD = 208u;
const uint F32_S_SR_PW1   = 256u;
const uint F32_S_SR_PW2   = 288u;
const uint F32_S_SR_HEAD  = 320u;

const uint F32_B_ENC1_PW1 = 336u;
const uint F32_B_ENC1_PW2 = 368u;
const uint F32_B_BOT2_PW1 = 400u;
const uint F32_B_BOT2_PW2 = 464u;
const uint F32_B_DEC1_PW1 = 496u;
const uint F32_B_DEN_HEAD = 544u;
const uint F32_B_SR_PW1   = 592u;
const uint F32_B_SR_PW2   = 624u;
const uint F32_B_SR_HEAD  = 656u;

const uint F32_DW_ENC1_W  = 672u;   // 9 * 32 = 288
const uint F32_DW_BOT2_W  = 960u;   // 9 * 32 = 288
const uint F32_DW_DEC1_W  = 1248u;  // 9 * 64 = 576
const uint F32_DW_SR1_W   = 1824u;  // 9 * 16 = 144

const uint F32_DW_ENC1_B  = 1968u;  // 32
const uint F32_DW_BOT2_B  = 2000u;  // 32
const uint F32_DW_DEC1_B  = 2032u;  // 64
const uint F32_DW_SR1_B   = 2096u;  // 16
const uint TOTAL_WEIGHT_FP32_F32 = 2112u;

// -----------------------------------------------------------------------------
// Scalar & Math Helpers
// -----------------------------------------------------------------------------
float upwaysLrelu(float x) {
    return x >= 0.0 ? x : 0.125 * x;
}

vec4 upwaysLrelu4(vec4 v) {
    return mix(0.125 * v, v, greaterThanEqual(v, vec4(0.0)));
}

float upwaysLuma(vec3 c) {
    return dot(max(c, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
}

float safeLog1pF(float x) {
    return log(1.0 + clamp(x, 0.0, 65504.0));
}

vec3 safeLog1p(vec3 v) {
    return log(vec3(1.0) + clamp(v, vec3(0.0), vec3(65504.0)));
}

vec3 safeExpm1(vec3 v) {
    return max(exp(clamp(v, vec3(0.0), vec3(11.0898))) - vec3(1.0), vec3(0.0));
}

vec3 clampFp16Rgb(vec3 c) {
    return clamp(c, vec3(0.0), vec3(65504.0));
}

vec3 sanitizeRgb(vec3 c) {
    return mix(vec3(0.0), c, lessThan(floatBitsToUint(c), uvec3(0x7F800000u)));
}

float sanitizeFloat(float v, float fallback) {
    return ((floatBitsToUint(v) & 0x7FFFFFFFu) < 0x7F800000u) ? v : fallback;
}

vec3 rotateByQuatUpways(vec4 q, vec3 v) {
    vec3 t = 2.0 * cross(q.xyz, v);
    return v + q.w * t + cross(q.xyz, t);
}

bool isSkyOrEmissiveOnly(float linearDepth, float matClass, vec3 aDiff, vec3 aSpec) {
    if (linearDepth >= 9999.0) {
        return true;
    }
    float maxA = max(max(aDiff.r, max(aDiff.g, aDiff.b)), max(aSpec.r, max(aSpec.g, aSpec.b)));
    return (matClass < 0.05 && maxA < 1e-5);
}

// -----------------------------------------------------------------------------
// Software / Hardware OCP E4M3fn Pack & Unpack (bit-exact with floate4m3_t)
// -----------------------------------------------------------------------------
uint encodeE4M3Byte(float x) {
#if UPWAYS_IS_FP8
    float xc = clamp(x, -448.0, 448.0);
    return uint(uint8_t(floate4m3BitsToIntEXT(floate4m3_t(xc))));
#else
    if (isnan(x)) {
        return 0x7Fu;
    }
    uint signBit = (x < 0.0) ? 0x80u : 0x00u;
    float ax = min(abs(x), 448.0);
    if (ax < 0.0009765625) {
        return signBit; // < 2^-10 rounds to +-0
    }
    if (ax < 0.015625) {
        // Subnormals: step 2^-9, nearest-even
        float scaled = ax * 512.0;
        uint q = uint(roundEven(scaled));
        if (q >= 8u) {
            return signBit | (1u << 3u);
        }
        return signBit | (q & 0x7u);
    }
    int exp2;
    float mant = frexp(ax, exp2); // ax = mant * 2^exp2, mant in [0.5, 1)
    int e = exp2 - 1 + 7;         // unbiased = exp2 - 1, bias = 7
    float frac = (mant * 2.0 - 1.0) * 8.0;
    uint m = uint(roundEven(frac));
    if (m >= 8u) {
        m = 0u;
        e += 1;
    }
    if (e > 15 || (e == 15 && m > 6u)) {
        e = 15;
        m = 6u; // 448.0 max finite E4M3fn (0x7E)
    }
    return signBit | (uint(e) << 3u) | (m & 0x7u);
#endif
}

float decodeE4M3Byte(uint b) {
#if UPWAYS_IS_FP8
    return float(intBitsToFloate4m3EXT(int8_t(uint8_t(b & 0xFFu))));
#else
    uint u = b & 0xFFu;
    float s = ((u & 0x80u) != 0u) ? -1.0 : 1.0;
    uint e = (u >> 3u) & 0xFu;
    uint m = u & 0x7u;
    if (e == 0u) {
        return s * (float(m) / 512.0);
    }
    if (e == 15u && m == 7u) {
        return 0.0;
    }
    return s * exp2(float(int(e) - 7)) * (1.0 + float(m) / 8.0);
#endif
}

uint packFp8Vec4(vec4 v) {
#if UPWAYS_IS_FP8
    vec4 vc = clamp(v, vec4(-448.0), vec4(448.0));
    return pack32(floate4m3BitsToUintEXT(fe4m3vec4(vc)));
#else
    return encodeE4M3Byte(v.x)
         | (encodeE4M3Byte(v.y) << 8u)
         | (encodeE4M3Byte(v.z) << 16u)
         | (encodeE4M3Byte(v.w) << 24u);
#endif
}

vec4 unpackFp8Vec4(uint u) {
#if UPWAYS_IS_FP8
    return vec4(uintBitsToFloate4m3EXT(unpack8(u)));
#else
    return vec4(
        decodeE4M3Byte(u),
        decodeE4M3Byte(u >> 8u),
        decodeE4M3Byte(u >> 16u),
        decodeE4M3Byte(u >> 24u)
    );
#endif
}

vec4 quantFp8Sim4(vec4 v) {
    return unpackFp8Vec4(packFp8Vec4(v));
}

// Fixed symmetric INT8 static bounds for pre-serialized g_act slots & temporal state
const float UPWAYS_INT8_ACT_MAX_FEAT   = 12.0;
const float UPWAYS_INT8_ACT_MAX_HIDDEN = 4.0;
const float UPWAYS_INT8_ACT_MAX_SKYMAT = 1.0;
const float UPWAYS_INT8_ACT_MAX_TERR   = 4.0;

const vec4 UPWAYS_X0_INT8_MAX_0   = vec4(12.0, 12.0, 12.0, 4.0);
const vec4 UPWAYS_X0_INT8_MAX_4   = vec4(4.0,  4.0,  12.0, 4.0);
const vec4 UPWAYS_X0_INT8_MAX_5   = vec4(4.0,  4.0,  4.0,  4.0);

const vec4 UPWAYS_SRIN_INT8_MAX_0 = vec4(12.0, 12.0, 12.0, 4.0);
const vec4 UPWAYS_SRIN_INT8_MAX_2 = vec4(4.0,  4.0,  4.0,  4.0);
const vec4 UPWAYS_SRIN_INT8_MAX_3 = vec4(12.0, 4.0,  4.0,  4.0);

uint packInt8Vec4Inv(vec4 v, vec4 invScale) {
    ivec4 q = clamp(ivec4(roundEven(v * invScale)), ivec4(-127), ivec4(127));
    uvec4 u = uvec4(q) & uvec4(0xFFu);
    return u.x | (u.y << 8u) | (u.z << 16u) | (u.w << 24u);
}

uint packInt8Vec4(vec4 v, vec4 maxAbs) {
    return packInt8Vec4Inv(v, vec4(127.0) / maxAbs);
}

vec4 unpackInt8Vec4Scale(uint u, vec4 scale) {
    ivec4 q = ivec4(
        bitfieldExtract(int(u), 0, 8),
        bitfieldExtract(int(u), 8, 8),
        bitfieldExtract(int(u), 16, 8),
        bitfieldExtract(int(u), 24, 8)
    );
    return vec4(q) * scale;
}

vec4 unpackInt8Vec4(uint u, vec4 maxAbs) {
    return unpackInt8Vec4Scale(u, maxAbs * (1.0 / 127.0));
}

float unpackInt8Byte(uint shifted, float maxAbs) {
    int q = bitfieldExtract(int(shifted), 0, 8);
    return float(q) * (maxAbs * (1.0 / 127.0));
}

uint packHiddenVec4(vec4 v) {
#if UPWAYS_IS_FP8
    return packFp8Vec4(v);
#else
    return packInt8Vec4(v, vec4(UPWAYS_INT8_ACT_MAX_HIDDEN));
#endif
}

vec4 unpackHiddenVec4(uint u) {
#if UPWAYS_IS_FP8
    return unpackFp8Vec4(u);
#else
    return unpackInt8Vec4(u, vec4(UPWAYS_INT8_ACT_MAX_HIDDEN));
#endif
}

uint packSkyMat4(vec4 v) {
#if UPWAYS_IS_FP8
    return packFp8Vec4(v);
#else
    return packInt8Vec4(v, vec4(UPWAYS_INT8_ACT_MAX_SKYMAT));
#endif
}

vec4 unpackSkyMat4(uint u) {
#if UPWAYS_IS_FP8
    return unpackFp8Vec4(u);
#else
    return unpackInt8Vec4(u, vec4(UPWAYS_INT8_ACT_MAX_SKYMAT));
#endif
}

float unpackSkyMatByte(uint shifted) {
#if UPWAYS_IS_FP8
    return decodeE4M3Byte(shifted);
#else
    return unpackInt8Byte(shifted, UPWAYS_INT8_ACT_MAX_SKYMAT);
#endif
}

uint packTempErr4(vec4 tErr4) {
#if UPWAYS_IS_FP8
    return packFp8Vec4(tErr4 * 16.0);
#else
    return packInt8Vec4(tErr4, vec4(UPWAYS_INT8_ACT_MAX_TERR));
#endif
}

float unpackTempErrByte(uint shifted) {
#if UPWAYS_IS_FP8
    return decodeE4M3Byte(shifted) * 0.0625;
#else
    return unpackInt8Byte(shifted, UPWAYS_INT8_ACT_MAX_TERR);
#endif
}

float dynInt8AbsMax4(vec4 v) {
    vec4 a = abs(v);
    return max(max(a.x, a.y), max(a.z, a.w));
}

uvec2 packHalf4(vec4 v) {
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}

vec4 unpackHalf4(uvec2 u) {
    return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y));
}

// Bilinear tap computation matching PyTorch & HLSL integer imageLoad
struct BilinearTaps {
    ivec2 i00;
    ivec2 i10;
    ivec2 i01;
    ivec2 i11;
    vec4  w; // (w00, w10, w01, w11)
};

BilinearTaps computeBilinearTaps(vec2 coord, ivec2 dims) {
    vec2 f = floor(coord);
    vec2 frac = clamp(coord - f, vec2(0.0), vec2(1.0));
    int x0 = int(f.x);
    int y0 = int(f.y);
    int x1 = x0 + 1;
    int y1 = y0 + 1;
    ivec2 hi = dims - ivec2(1);
    BilinearTaps t;
    t.i00 = ivec2(clamp(x0, 0, hi.x), clamp(y0, 0, hi.y));
    t.i10 = ivec2(clamp(x1, 0, hi.x), clamp(y0, 0, hi.y));
    t.i01 = ivec2(clamp(x0, 0, hi.x), clamp(y1, 0, hi.y));
    t.i11 = ivec2(clamp(x1, 0, hi.x), clamp(y1, 0, hi.y));
    t.w = vec4(
        (1.0 - frac.x) * (1.0 - frac.y),
        frac.x * (1.0 - frac.y),
        (1.0 - frac.x) * frac.y,
        frac.x * frac.y
    );
    return t;
}

vec3 demodulateMlSafe(vec3 rad, vec3 demodFactor) {
    bool valid = any(greaterThan(demodFactor, vec3(1e-4)));
    vec3 safeDenom = valid ? max(demodFactor, vec3(0.02)) : vec3(1.0);
    return min(rad / safeDenom, vec3(65000.0));
}

vec3 octDecodeUpways(vec2 f) {
    f = clamp(f, vec2(-1.0), vec2(1.0));
    vec3 n = vec3(f.x, f.y, 1.0 - abs(f.x) - abs(f.y));
    float t = max(-n.z, 0.0);
    n.x += (n.x >= 0.0) ? -t : t;
    n.y += (n.y >= 0.0) ? -t : t;
    float l2 = dot(n, n);
    return (l2 > 1e-12) ? (n * inversesqrt(l2)) : vec3(0.0, 0.0, 1.0);
}

vec3 unpackOct32Upways(uint p) {
    return octDecodeUpways(unpackSnorm2x16(p));
}

void unpackCompactNormalDepth(uvec4 pg, out vec3 viewNormal, out float linDepth) {
    viewNormal = unpackOct32Upways(pg.x);
    float rawD = unpackHalf2x16(pg.y).x;
    linDepth = ((floatBitsToUint(rawD) & 0x7FFFFFFFu) < 0x7F800000u) ? clamp(rawD, 0.0, 10000.0) : 10000.0;
}

void unpackCompactPrimary(
    uvec4 pg,
    out vec3 viewNormal,
    out float linDepth,
    out float diffHitDist,
    out float reactiveMask,
    out vec3 aDiff,
    out float roughness,
    out vec3 aSpec,
    out float matClass
) {
    unpackCompactNormalDepth(pg, viewNormal, linDepth);
    diffHitDist  = float((pg.y >> 16u) & 0xFFu) * (1.0 / 255.0);
    reactiveMask = float((pg.y >> 24u) & 0xFFu) * (1.0 / 255.0);
    vec4 albR = unpackUnorm4x8(pg.z);
    vec4 sAlb = unpackUnorm4x8(pg.w);
    aDiff     = albR.rgb;
    roughness = albR.a;
    aSpec     = sAlb.rgb;
    matClass  = sAlb.a;
}

bool isCompactPrimaryTransmissive(uvec4 pg) {
    return ((pg.y >> 16u) & 1u) != 0u;
}

vec3 unpackCompactSkip(uint skipPack) {
    return (skipPack != 0u) ? unpackRGB9E5(skipPack) : vec3(0.0);
}

void unpackCompactMotionAndSpec(
    uint surfMvPack,
    uvec2 specVmvRaw,
    float linDepth,
    bool isPrimaryTrans,
    out vec2 surfaceMv,
    out vec2 specMv,
    out float specInvDepthSigned,
    out float specHitDist
) {
    surfaceMv = unpackHalf2x16(surfMvPack);
    if (specVmvRaw.y != 0u) {
        specMv = unpackHalf2x16(specVmvRaw.x);
        vec2 zDist = unpackHalf2x16(specVmvRaw.y);
        specInvDepthSigned = zDist.x;
        specHitDist        = zDist.y;
    } else {
        specMv = surfaceMv;
        float dSpec = (linDepth >= 9999.0)
            ? (isPrimaryTrans ? 1e-4 : 0.0)
            : (1.0 / (1.0 + max(linDepth, 0.0)));
        specInvDepthSigned = isPrimaryTrans ? -dSpec : dSpec;
        specHitDist        = 0.0;
    }
}
