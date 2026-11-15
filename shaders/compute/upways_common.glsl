// Upways 4.0 shared GLSL definitions, layout constants, FP8/FP16 helpers, and explicit bilinear warps.
// Must remain strictly identical between Upways and Pathways repositories.

#extension GL_KHR_memory_scope_semantics : require
#extension GL_KHR_cooperative_matrix : require
#extension GL_KHR_shader_subgroup_basic : require
#extension GL_EXT_shader_explicit_arithmetic_types : require
#extension GL_EXT_shader_explicit_arithmetic_types_float16 : require
#extension GL_EXT_shader_explicit_arithmetic_types_int8 : require
#ifndef UPWAYS_FP16_WMMA
#extension GL_EXT_float_e4m3 : require
#endif

// -----------------------------------------------------------------------------
// Push Constants (56 bytes exact)
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
} pc;

const uint UPWAYS_FLAG_RESET_HISTORY      = 1u;
const uint UPWAYS_FLAG_ENABLE_SUPERRES    = 2u;
const uint UPWAYS_FLAG_ZERO_RESIDUAL_MODE = 4u;

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
#ifndef UPWAYS_FP16_WMMA
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
#ifndef UPWAYS_FP16_WMMA
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
#ifndef UPWAYS_FP16_WMMA
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
#ifndef UPWAYS_FP16_WMMA
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
