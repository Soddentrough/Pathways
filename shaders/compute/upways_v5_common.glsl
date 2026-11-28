// Upways 5.0 Pure-ML (158,400 params, 190,208 B) shared GLSL definitions.
// Strictly bit-faithful with Upways/upways/models/{denoiser,upscaler,ops,quantization}.py & export.py.

#ifndef UPWAYS_V5_COMMON_GLSL
#define UPWAYS_V5_COMMON_GLSL

#ifndef UPWAYS_PRECISION_MODE
#define UPWAYS_PRECISION_MODE 0
#endif

#define UPWAYS_IS_FP8  (UPWAYS_PRECISION_MODE == 0)
#define UPWAYS_IS_INT8 (UPWAYS_PRECISION_MODE == 1)

#extension GL_EXT_control_flow_attributes : enable
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
// Push Constants (128 bytes exact, shared across all Upways 5.0 shaders)
// -----------------------------------------------------------------------------
layout(push_constant) uniform UpwaysV5PC {
    uint  lrWidth;        // 0  (active band/full LR width)
    uint  lrHeight;       // 4  (active band/full LR height)
    uint  hrWidth;        // 8  (active band/full HR width)
    uint  hrHeight;       // 12 (active band/full HR height)
    float jitterX;        // 16 (LR pixels, Camera::ubo.jitterOffset.x)
    float jitterY;        // 20 (LR pixels, Camera::ubo.jitterOffset.y)
    uint  frameIndex;     // 24
    uint  flags;          // 28: bit 0=RESET, bit 1=SR, bit 2=ZERO_RESIDUAL, bit 3=FUSED_BAND_PACK
    uint  inW;            // 32 (stage input width)
    uint  inH;            // 36 (stage input height)
    uint  outW;           // 40 (stage output width)
    uint  outH;           // 44 (stage output height)
    uint  off0;           // 48 (stage-specific param / offset 0)
    uint  off1;           // 52 (stage-specific param / offset 1)
    uint  bandLrY0;       // 56 (band start Y in full LR frame)
    uint  bandHrY0;       // 60 (band start Y in full HR frame)
    uint  fullLrHeight;   // 64 (full LR frame height)
    uint  fullHrHeight;   // 68 (full HR frame height)
    uint  keepHrYStart;   // 72 (first band-local output row for band pack / uOutputColor)
    uint  keepHrYEnd;     // 76 (exclusive band-local output row)
    vec4  prevToCurrViewQuat;   // 80  (xyzw unit quat rotating prev view -> curr view)
    vec4  currToPrevTransView;  // 96  (xyz = (prevView * inverse(currView))[3].xyz, w = tanHalfFovY)
    vec4  camMotionParams;      // 112 (x = aspectRatio, y = camMotionValid [1.0 when live camera], z/w = 0)
} pc;

const uint UPWAYS_V5_FLAG_RESET_HISTORY      = 1u;
const uint UPWAYS_V5_FLAG_ENABLE_SUPERRES    = 2u;
const uint UPWAYS_V5_FLAG_ZERO_RESIDUAL_MODE = 4u;
const uint UPWAYS_V5_FLAG_FUSED_BAND_PACK    = 8u;

// -----------------------------------------------------------------------------
// Weight Offsets (190,208 B total: 150,016 B 8-bit PW + 40,192 B FP32)
// -----------------------------------------------------------------------------
const uint U32_W5_ENC1_PW1     = 0u;      // 128 -> 48
const uint U32_W5_ENC1_PW2     = 1536u;   // 48  -> 48
const uint U32_W5_ENC2_PW1     = 2112u;   // 48  -> 80
const uint U32_W5_ENC2_PW2     = 3072u;   // 80  -> 80
const uint U32_W5_BOT3_PW1     = 4672u;   // 80  -> 128
const uint U32_W5_BOT3_PW2     = 7232u;   // 128 -> 128
const uint U32_W5_BOT3_PW3     = 11328u;  // 128 -> 128
const uint U32_W5_BOT3_PW4     = 15424u;  // 128 -> 96
const uint U32_W5_BOT3_KHEAD   = 18496u;  // 96  -> 32
const uint U32_W5_DEC2_PW1     = 19264u;  // 176 -> 96
const uint U32_W5_DEC2_PW2     = 23488u;  // 96  -> 80
const uint U32_W5_DEC2_KHEAD   = 25408u;  // 80  -> 32
const uint U32_W5_DEC1_PW1     = 26048u;  // 128 -> 64
const uint U32_W5_DEC1_PW2     = 28096u;  // 64  -> 64
const uint U32_W5_DEC1_KHEAD   = 29120u;  // 64  -> 32
const uint U32_W5_DEN_HEAD     = 29632u;  // 64  -> 128
const uint U32_W5_SR_PW1       = 31680u;  // 64  -> 64
const uint U32_W5_SR_PW2       = 32704u;  // 64  -> 64
const uint U32_W5_SR_PW3       = 33728u;  // 64  -> 64
const uint U32_W5_SR_PW4       = 34752u;  // 64  -> 64
const uint U32_W5_SR_HEAD      = 35776u;  // 64  -> 96
const uint U32_W5_HR_POS_MLP   = 37312u;  // 16  -> 16
const uint U32_W5_HR_TEMP_MLP1 = 37376u;  // 16  -> 16
const uint U32_W5_HR_TEMP_MLP2 = 37440u;  // 16  -> 16

// FP32 vec4 offsets inside WeightFp32Buf (2,512 vec4s = 40,192 B)
const uint V4_S5_ENC1_PW1      = 0u;
const uint V4_S5_ENC1_PW2      = 12u;
const uint V4_S5_ENC2_PW1      = 24u;
const uint V4_S5_ENC2_PW2      = 44u;
const uint V4_S5_BOT3_PW1      = 64u;
const uint V4_S5_BOT3_PW2      = 96u;
const uint V4_S5_BOT3_PW3      = 128u;
const uint V4_S5_BOT3_PW4      = 160u;
const uint V4_S5_BOT3_KHEAD    = 184u;
const uint V4_S5_DEC2_PW1      = 192u;
const uint V4_S5_DEC2_PW2      = 216u;
const uint V4_S5_DEC2_KHEAD    = 236u;
const uint V4_S5_DEC1_PW1      = 244u;
const uint V4_S5_DEC1_PW2      = 260u;
const uint V4_S5_DEC1_KHEAD    = 276u;
const uint V4_S5_DEN_HEAD      = 284u;
const uint V4_S5_SR_PW1        = 316u;
const uint V4_S5_SR_PW2        = 332u;
const uint V4_S5_SR_PW3        = 348u;
const uint V4_S5_SR_PW4        = 364u;
const uint V4_S5_SR_HEAD       = 380u;
const uint V4_S5_HR_POS_MLP    = 404u;
const uint V4_S5_HR_TEMP_MLP1  = 408u;
const uint V4_S5_HR_TEMP_MLP2  = 412u;

const uint V4_B5_OFFSET        = 416u;

const uint V4_DW5_ENC1_W       = 832u;   // 9 * 12 = 108 vec4
const uint V4_DW5_ENC2_W       = 940u;   // 9 * 20 = 180 vec4
const uint V4_DW5_BOT3_1_W     = 1120u;  // 9 * 32 = 288 vec4
const uint V4_DW5_BOT3_2_W     = 1408u;  // 9 * 32 = 288 vec4
const uint V4_DW5_DEC2_W       = 1696u;  // 9 * 24 = 216 vec4
const uint V4_DW5_DEC1_W       = 1912u;  // 9 * 16 = 144 vec4
const uint V4_DW5_SR1_W        = 2056u;  // 9 * 16 = 144 vec4
const uint V4_DW5_SR2_W        = 2200u;  // 9 * 16 = 144 vec4

const uint V4_DW5_ENC1_B       = 2344u;  // 12 vec4
const uint V4_DW5_ENC2_B       = 2356u;  // 20 vec4
const uint V4_DW5_BOT3_1_B     = 2376u;  // 32 vec4
const uint V4_DW5_BOT3_2_B     = 2408u;  // 32 vec4
const uint V4_DW5_DEC2_B       = 2440u;  // 24 vec4
const uint V4_DW5_DEC1_B       = 2464u;  // 16 vec4
const uint V4_DW5_SR1_B        = 2480u;  // 16 vec4
const uint V4_DW5_SR2_B        = 2496u;  // 16 vec4

// -----------------------------------------------------------------------------
// Scalar, Math & Pure-ML Gate Helpers
// -----------------------------------------------------------------------------
float upwaysLrelu(float x) {
    return (x >= 0.0) ? x : (0.125 * x);
}

vec4 upwaysLrelu4(vec4 v) {
    return mix(0.125 * v, v, greaterThanEqual(v, vec4(0.0)));
}

float zeroCenteredGate(float z) {
    float zp = max(z, 0.0);
    float zp2 = zp * zp;
    return zp2 / (1.0 + zp2);
}

vec2 zeroCenteredGate2(vec2 z) {
    vec2 zp = max(z, vec2(0.0));
    vec2 zp2 = zp * zp;
    return zp2 / (vec2(1.0) + zp2);
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

vec3 clampFp16Rgb(vec3 c) {
    return clamp(c, vec3(0.0), vec3(65504.0));
}

vec3 sanitizeRgb(vec3 c) {
    return mix(vec3(0.0), max(c, vec3(0.0)), lessThan(floatBitsToUint(c), uvec3(0x7F800000u)));
}

float sanitizeFloat(float v, float fallback) {
    return ((floatBitsToUint(v) & 0x7FFFFFFFu) < 0x7F800000u) ? v : fallback;
}

vec3 normalizeNormalV5(vec3 n) {
    float l2 = dot(n, n);
    return (l2 > 1e-10) ? (n * inversesqrt(l2)) : vec3(0.0, 0.0, 1.0);
}

vec3 rotateByQuatV5(vec4 q, vec3 v) {
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
// OCP E4M3fn FP8 & Symmetric INT8 Quantization Helpers
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
        return signBit;
    }
    if (ax < 0.015625) {
        float scaled = ax * 512.0;
        uint q = uint(roundEven(scaled));
        if (q >= 8u) {
            return signBit | (1u << 3u);
        }
        return signBit | (q & 0x7u);
    }
    int exp2v;
    float mant = frexp(ax, exp2v);
    int e = exp2v - 1 + 7;
    float frac = (mant * 2.0 - 1.0) * 8.0;
    uint m = uint(roundEven(frac));
    if (m >= 8u) {
        m = 0u;
        e += 1;
    }
    if (e > 15 || (e == 15 && m > 6u)) {
        e = 15;
        m = 6u;
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

const float UPWAYS_V5_INT8_MAX_HIDDEN = 4.0;
const float UPWAYS_V5_INT8_DYN_FLOOR  = 1e-6;

// Static INT8 bounds matching quantization.py (X0_INT8_MAX_32 & SRIN_INT8_MAX_16)
const vec4 V5_X0_INT8_MAX_0   = vec4(12.0, 12.0, 12.0, 12.0); // ch 0..3
const vec4 V5_X0_INT8_MAX_1   = vec4(12.0, 12.0, 12.0, 12.0); // ch 4..7
const vec4 V5_X0_INT8_MAX_2   = vec4(12.0, 12.0, 12.0, 12.0); // ch 8..11
const vec4 V5_X0_INT8_MAX_3   = vec4(12.0, 12.0, 12.0, 4.0);  // ch 12..15
const vec4 V5_X0_INT8_MAX_4   = vec4(4.0,  4.0,  12.0, 4.0);  // ch 16..19
const vec4 V5_X0_INT8_MAX_5   = vec4(4.0,  4.0,  4.0,  4.0);  // ch 20..23
const vec4 V5_X0_INT8_MAX_6   = vec4(4.0,  4.0,  4.0,  4.0);  // ch 24..27
const vec4 V5_X0_INT8_MAX_7   = vec4(4.0,  4.0,  4.0,  4.0);  // ch 28..31

const vec4 V5_SRIN_INT8_MAX_0 = vec4(12.0, 12.0, 12.0, 4.0);  // ch 0..3
const vec4 V5_SRIN_INT8_MAX_1 = vec4(4.0,  4.0,  12.0, 4.0);  // ch 4..7
const vec4 V5_SRIN_INT8_MAX_2 = vec4(4.0,  4.0,  4.0,  4.0);  // ch 8..11
const vec4 V5_SRIN_INT8_MAX_3 = vec4(4.0,  4.0,  4.0,  4.0);  // ch 12..15

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

float dynInt8AbsMax4(vec4 v) {
    vec4 a = abs(v);
    return max(max(a.x, a.y), max(a.z, a.w));
}

uint packHiddenVec4(vec4 v) {
#if UPWAYS_IS_FP8
    return packFp8Vec4(v);
#else
    return packInt8Vec4(v, vec4(UPWAYS_V5_INT8_MAX_HIDDEN));
#endif
}

vec4 unpackHiddenVec4(uint u) {
#if UPWAYS_IS_FP8
    return unpackFp8Vec4(u);
#else
    return unpackInt8Vec4(u, vec4(UPWAYS_V5_INT8_MAX_HIDDEN));
#endif
}

uvec2 packHalf4(vec4 v) {
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}

vec4 unpackHalf4(uvec2 u) {
    return vec4(unpackHalf2x16(u.x), unpackHalf2x16(u.y));
}

// -----------------------------------------------------------------------------
// Bilinear Gather Helper & Compact 41 B/px G-Buffer Unpacking
// -----------------------------------------------------------------------------
struct BilinearTaps {
    ivec2 i00;
    ivec2 i10;
    ivec2 i01;
    ivec2 i11;
    vec4  w;
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
    out float reactiveMask,
    out vec3 aDiff,
    out float roughness,
    out vec3 aSpec,
    out float matClass,
    out bool transBit
) {
    unpackCompactNormalDepth(pg, viewNormal, linDepth);
    transBit     = (pg.y & (1u << 16u)) != 0u;
    reactiveMask = float((pg.y >> 24u) & 0xFFu) * (1.0 / 255.0);
    vec4 albR = unpackUnorm4x8(pg.z);
    vec4 sAlb = unpackUnorm4x8(pg.w);
    aDiff     = albR.rgb;
    roughness = albR.a;
    aSpec     = sAlb.rgb;
    matClass  = sAlb.a;
}

vec3 unpackCompactSkip(uint skipWord) {
    return (skipWord != 0u) ? sanitizeRgb(unpackRGB9E5(skipWord)) : vec3(0.0);
}

void unpackCompactSpecVmv(
    uvec2 specVmvRaw,
    vec2 surfMv,
    float linDepth,
    bool transBit,
    bool isSky,
    out vec2 specMv,
    out float specInvZ,
    out float specDist
) {
    if (specVmvRaw.y != 0u) {
        vec2 sMv = unpackHalf2x16(specVmvRaw.x);
        vec2 sYz = unpackHalf2x16(specVmvRaw.y);
        specMv   = vec2(sanitizeFloat(sMv.x, surfMv.x), sanitizeFloat(sMv.y, surfMv.y));
        specInvZ = sanitizeFloat(sYz.x, 0.0);
        specDist = clamp(sanitizeFloat(sYz.y, 0.0), 0.0, 10000.0);
    } else {
        specMv   = surfMv;
        specInvZ = isSky ? (transBit ? -1e-4 : 0.0) : ((transBit ? -1.0 : 1.0) / (1.0 + max(linDepth, 0.0)));
        specDist = 0.0;
    }
}

void demodulateAndFoldV5(
    uvec2 irr,
    vec3 aDiff,
    vec3 aSpec,
    vec3 skipRad,
    float linDepth,
    float matClass,
    bool transBit,
    float specInvZ,
    out vec3 rawDiff,
    out vec3 rawSpec,
    out vec3 aDiffEff,
    out vec3 aSpecEff,
    out vec3 skipEff,
    out bool isSky,
    out bool isDiel
) {
    vec3 irrD = (irr.x != 0u) ? sanitizeRgb(unpackRGB9E5(irr.x)) : vec3(0.0);
    vec3 irrS = (irr.y != 0u) ? sanitizeRgb(unpackRGB9E5(irr.y)) : vec3(0.0);
    rawDiff = (irr.x != 0u) ? clampFp16Rgb(demodulateMlSafe(irrD, aDiff)) : vec3(0.0);
    rawSpec = (irr.y != 0u) ? clampFp16Rgb(demodulateMlSafe(irrS, aSpec)) : vec3(0.0);
    isSky   = isSkyOrEmissiveOnly(linDepth, matClass, aDiff, aSpec);
    isDiel  = (!isSky) && (transBit || (specInvZ < -1e-5) || ((matClass > 0.55) && (matClass < 0.79)));
    if (isDiel) {
        vec3 dContrib = any(greaterThan(aDiff, vec3(1e-4))) ? (aDiff * rawDiff) : irrD;
        vec3 sContrib = any(greaterThan(aSpec, vec3(1e-4))) ? (aSpec * rawSpec) : irrS;
        vec3 dielRad  = clampFp16Rgb(dContrib + sContrib + sanitizeRgb(skipRad));
        rawDiff  = vec3(0.0);
        rawSpec  = dielRad;
        aDiffEff = vec3(0.0);
        aSpecEff = vec3(1.0);
        skipEff  = vec3(0.0);
    } else {
        aDiffEff = aDiff;
        aSpecEff = aSpec;
        skipEff  = (isSky && transBit)
            ? clampFp16Rgb(irrD + irrS + sanitizeRgb(skipRad))
            : sanitizeRgb(skipRad);
    }
}

bool computeBandWarpCoord(
    ivec2 localPos,
    vec2 mvUv,
    int bandW,
    int bandH,
    ivec2 prevDims,
    uint bandY0,
    uint fullH_u,
    bool allowSample,
    out vec2 localCoord
) {
    int fullH = int(max(fullH_u, uint(bandH)));
    vec2 fullSizeF = vec2(float(bandW), float(fullH));
    vec2 currFullUV = (vec2(float(localPos.x), float(localPos.y + int(bandY0))) + vec2(0.5)) / fullSizeF;
    vec2 prevFullUV = currFullUV - vec2(sanitizeFloat(mvUv.x, 0.0), sanitizeFloat(mvUv.y, 0.0));
    float sx = float(prevDims.x) / float(max(bandW, 1));
    float sy = float(prevDims.y) / float(max(bandH, 1));
    localCoord = vec2(
        prevFullUV.x * float(bandW) * sx,
        (prevFullUV.y * float(fullH) - float(bandY0)) * sy
    ) - vec2(0.5);
    return allowSample
        && all(greaterThanEqual(prevFullUV, vec2(0.0)))
        && all(lessThan(prevFullUV, vec2(1.0)))
        && (localCoord.y >= -0.5)
        && (localCoord.y < float(prevDims.y) - 0.5);
}

#endif // UPWAYS_V5_COMMON_GLSL
