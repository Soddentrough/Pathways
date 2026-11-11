// upways_common.glsl — Shared Vulkan 1.4 + RDNA 4 (gfx1201) FP8 definitions, offsets, and math helpers.
// Mirrors upways/data/protocol.py, upways/marshaling.py, upways/network.py, upways/upscaler.py, and upways/export.py 1:1.

#ifndef UPWAYS_COMMON_GLSL
#define UPWAYS_COMMON_GLSL

const float SKY_DEPTH_THRESHOLD       = 9999.0;
const float SPEC_SKY_DIST_THRESHOLD   = 999.0;
const float EMISSIVE_ALBEDO_SQ_EPS    = 1e-5;
const float MIN_DEMOD_ALBEDO          = 0.01;
const float MAX_LOG1P_IRRADIANCE      = 11.0903549; // log1p(65504.0)
const float INV_MAX_LOG1P_IRRADIANCE  = 1.0 / 11.0903549;
const float FP8_E4M3_MAX              = 448.0;
const float FP8_ACT_SCALE             = 16.0;
const float FP8_ACT_INV_SCALE         = 0.0625;

// Exact Section 1 (FP8 E4M3 WMMA 16x16 blocks) & Section 2 (FP32 scales, biases, 3x3 DW, skip_scale) offsets
// Generated deterministically by upways.export.compute_shader_weight_offsets() (89,536 bytes total)
const uint ENC0_PW1_W_FP8_OFFSET      = 0u;
const uint ENC0_PW1_S_F32_OFFSET      = 0u;
const uint ENC0_PW1_B_F32_OFFSET      = 32u;
const uint REC_PROJ_W_FP8_OFFSET      = 1024u;
const uint REC_PROJ_S_F32_OFFSET      = 64u;
const uint ENC0_PW2_W_FP8_OFFSET      = 1536u;
const uint ENC0_PW2_S_F32_OFFSET      = 96u;
const uint ENC0_PW2_B_F32_OFFSET      = 128u;
const uint ENC1_PW1_W_FP8_OFFSET      = 2560u;
const uint ENC1_PW1_S_F32_OFFSET      = 160u;
const uint ENC1_PW1_B_F32_OFFSET      = 224u;
const uint ENC1_PW2_W_FP8_OFFSET      = 4608u;
const uint ENC1_PW2_S_F32_OFFSET      = 288u;
const uint ENC1_PW2_B_F32_OFFSET      = 352u;
const uint BOT1_PW1_W_FP8_OFFSET      = 8704u;
const uint BOT1_PW1_S_F32_OFFSET      = 416u;
const uint BOT1_PW1_B_F32_OFFSET      = 544u;
const uint BOT1_PW2_W_FP8_OFFSET      = 16896u;
const uint BOT1_PW2_S_F32_OFFSET      = 672u;
const uint BOT1_PW2_B_F32_OFFSET      = 800u;
const uint BOT2_PW1_W_FP8_OFFSET      = 33280u;
const uint BOT2_PW1_S_F32_OFFSET      = 928u;
const uint BOT2_PW1_B_F32_OFFSET      = 1056u;
const uint BOT2_PW2_W_FP8_OFFSET      = 49664u;
const uint BOT2_PW2_S_F32_OFFSET      = 1184u;
const uint BOT2_PW2_B_F32_OFFSET      = 1248u;
const uint DEC1_PW1_W_FP8_OFFSET      = 57856u;
const uint DEC1_PW1_S_F32_OFFSET      = 1312u;
const uint DEC1_PW1_B_F32_OFFSET      = 1376u;
const uint DEC1_PW2_W_FP8_OFFSET      = 61952u;
const uint DEC1_PW2_S_F32_OFFSET      = 1440u;
const uint DEC1_PW2_B_F32_OFFSET      = 1472u;
const uint DEC0_PW1_W_FP8_OFFSET      = 64000u;
const uint DEC0_PW1_S_F32_OFFSET      = 1504u;
const uint DEC0_PW1_B_F32_OFFSET      = 1568u;
const uint DEC0_PW2_W_FP8_OFFSET      = 66048u;
const uint DEC0_PW2_S_F32_OFFSET      = 1632u;
const uint DEC0_PW2_B_F32_OFFSET      = 1664u;
const uint HEAD_OUT_W_FP8_OFFSET      = 68096u;
const uint HEAD_OUT_S_F32_OFFSET      = 1696u;
const uint HEAD_OUT_B_F32_OFFSET      = 1712u;
const uint HEAD_REC_W_FP8_OFFSET      = 68608u;
const uint HEAD_REC_S_F32_OFFSET      = 1728u;
const uint HEAD_REC_B_F32_OFFSET      = 1760u;

const uint ENC0_DW_W_F32_OFFSET       = 1792u;
const uint ENC1_DW_W_F32_OFFSET       = 2080u;
const uint BOT1_DW_W_F32_OFFSET       = 2368u;
const uint BOT2_DW_W_F32_OFFSET       = 2944u;
const uint DEC1_DW_W_F32_OFFSET       = 4096u;
const uint DEC0_DW_W_F32_OFFSET       = 4672u;
const uint SKIP_SCALE_F32_OFFSET      = 4960u;

layout(push_constant) uniform UpwaysPushConstants {
    uint renderWidth;
    uint renderHeight;
    uint displayWidth;
    uint displayHeight;
    float jitterX;
    float jitterY;
    float preExposure;
    uint resetHistory;
    uint preserveJitterGuide;
    uint stageId;
    uint srcWidth;
    uint srcHeight;
    uint dstWidth;
    uint dstHeight;
} pc;

struct PyramidLayout {
    uint w0, h0, n0;
    uint w1, h1, n1;
    uint w2, h2, n2;
    uint offX0;
    uint offHw0;
    uint offS0;
    uint offL1In;
    uint offS1;
    uint offL2In;
    uint offB1;
    uint offB2;
    uint offD1;
    uint totalBytes;
};

uint pyrAlign256(uint bytes) {
    return (bytes + 255u) & ~255u;
}

PyramidLayout computePyramidLayout(uint w0, uint h0) {
    PyramidLayout pl;
    pl.w0 = max(w0, 1u);
    pl.h0 = max(h0, 1u);
    pl.n0 = pl.w0 * pl.h0;

    pl.w1 = (pl.w0 + 1u) >> 1u;
    pl.h1 = (pl.h0 + 1u) >> 1u;
    pl.n1 = pl.w1 * pl.h1;

    pl.w2 = (pl.w1 + 1u) >> 1u;
    pl.h2 = (pl.h1 + 1u) >> 1u;
    pl.n2 = pl.w2 * pl.h2;

    uint cur = 0u;
    pl.offX0   = cur; cur += pyrAlign256(32u * pl.n0);
    pl.offHw0  = cur; cur += pyrAlign256(16u * pl.n0);
    pl.offS0   = cur; cur += pyrAlign256(32u * pl.n0);
    pl.offL1In = cur; cur += pyrAlign256(32u * pl.n1);
    pl.offS1   = cur; cur += pyrAlign256(64u * pl.n1);
    pl.offL2In = cur; cur += pyrAlign256(64u * pl.n2);
    pl.offB1   = cur; cur += pyrAlign256(128u * pl.n2);
    pl.offB2   = cur; cur += pyrAlign256(64u * pl.n2);
    pl.offD1   = cur; cur += pyrAlign256(32u * pl.n1);
    pl.totalBytes = max(cur, 256u);
    return pl;
}

float sanitizeScalar(float x, float posInfReplacement) {
    if (isnan(x)) return 0.0;
    if (isinf(x)) return (x > 0.0) ? posInfReplacement : 0.0;
    return x;
}

vec3 sanitizeVec3Input(vec3 v) {
    return vec3(
        sanitizeScalar(v.x, 65504.0),
        sanitizeScalar(v.y, 65504.0),
        sanitizeScalar(v.z, 65504.0)
    );
}

vec4 sanitizeVec4Input(vec4 v) {
    return vec4(
        sanitizeScalar(v.x, 65504.0),
        sanitizeScalar(v.y, 65504.0),
        sanitizeScalar(v.z, 65504.0),
        sanitizeScalar(v.w, 65504.0)
    );
}

vec3 sanitizeRadiance(vec3 v) {
    vec3 c = vec3(
        (isnan(v.x) || isinf(v.x)) ? 0.0 : v.x,
        (isnan(v.y) || isinf(v.y)) ? 0.0 : v.y,
        (isnan(v.z) || isinf(v.z)) ? 0.0 : v.z
    );
    return clamp(c, vec3(0.0), vec3(65504.0));
}

float upwaysLog1p(float x) {
    float u = 1.0 + x;
    if (u <= 1.0) {
        return (x == 0.0) ? 0.0 : log(max(u, 1e-38));
    }
    return log(u) * (x / (u - 1.0));
}

vec3 upwaysLog1pVec3(vec3 v) {
    return vec3(upwaysLog1p(v.x), upwaysLog1p(v.y), upwaysLog1p(v.z));
}

float upwaysExpm1(float x) {
    if (abs(x) < 1e-4) {
        return x + 0.5 * x * x;
    }
    return exp(x) - 1.0;
}

vec3 upwaysExpm1Vec3(vec3 v) {
    return vec3(upwaysExpm1(v.x), upwaysExpm1(v.y), upwaysExpm1(v.z));
}

float upwaysSoftplus(float x) {
    float clamped = clamp(x, -40.0, 20.0);
    return (x > 20.0) ? x : upwaysLog1p(exp(clamped));
}

float leakyRelu01(float x) {
    return (x >= 0.0) ? x : (0.1 * x);
}

float safeTanh(float x) {
    float c = clamp(x, -16.0, 16.0);
    float e2x = exp(-2.0 * abs(c));
    float t = (1.0 - e2x) / (1.0 + e2x);
    return (c >= 0.0) ? t : -t;
}

float safeSigmoid(float x) {
    float c = clamp(x, -20.0, 20.0);
    return 1.0 / (1.0 + exp(-c));
}

void decodeMetallicAndEmissive(float rawCh7, vec3 albedo, out float metallic, out float isEmissive) {
    float albedoSq = dot(albedo, albedo);
    bool isPureEmissive = (rawCh7 > 0.5) && (albedoSq <= EMISSIVE_ALBEDO_SQ_EPS);
    bool isEmissiveMask = (rawCh7 > 1.5) || isPureEmissive;
    isEmissive = isEmissiveMask ? 1.0 : 0.0;
    metallic = isPureEmissive ? 0.0 : clamp(rawCh7, 0.0, 1.0);
}

vec3 computeDemodulationFactor(vec3 albedo, float linearDepth) {
    float albedoSq = dot(albedo, albedo);
    bool isSkyOrEmissive = (linearDepth >= SKY_DEPTH_THRESHOLD) || (albedoSq <= EMISSIVE_ALBEDO_SQ_EPS);
    return isSkyOrEmissive ? vec3(1.0) : max(albedo, vec3(MIN_DEMOD_ALBEDO));
}

vec3 computeSpecularModulation(vec3 albedo, float metallic, float roughness, float nViewZ, float linearDepth) {
    float albedoSq = dot(albedo, albedo);
    bool isBypass = (linearDepth >= SKY_DEPTH_THRESHOLD) || (metallic <= 1e-4) || (albedoSq <= EMISSIVE_ALBEDO_SQ_EPS);
    if (isBypass) {
        return vec3(1.0);
    }
    vec3 f0Metal = max(clamp(albedo, vec3(0.0), vec3(1.0)), vec3(0.04));
    float cosTheta = clamp(abs(nViewZ), 0.0, 1.0);
    float omc = 1.0 - cosTheta;
    float omc2 = omc * omc;
    float fresnel = omc2 * omc2 * omc * (1.0 - clamp(roughness, 0.0, 1.0));
    float wFres = 0.25 * fresnel;
    vec3 metalMod = f0Metal * (1.0 - wFres) + vec3(wFres);
    float mClamped = clamp(metallic, 0.0, 1.0);
    return vec3(1.0 - mClamped) + mClamped * metalMod;
}

// 4-quadrant bilinear footprint summary for display->render history downsampling when scale > 2.0x.
// Matches warp_area_4tap_uv() in upways/marshaling.py 1:1.
vec4 sampleHistoryAreaLr(sampler2D tex, vec2 prevUv, uvec2 renderSize) {
    uvec2 srcSize = uvec2(textureSize(tex, 0));
    float scaleX = float(srcSize.x) / float(max(renderSize.x, 1u));
    float scaleY = float(srcSize.y) / float(max(renderSize.y, 1u));
    float offX = clamp((scaleX - 2.0) * 0.5, 0.0, 1.0) / float(max(srcSize.x, 1u));
    float offY = clamp((scaleY - 2.0) * 0.5, 0.0, 1.0) / float(max(srcSize.y, 1u));
    if (offX == 0.0 && offY == 0.0) {
        return textureLod(tex, prevUv, 0.0);
    }
    vec4 s00 = textureLod(tex, prevUv + vec2(-offX, -offY), 0.0);
    vec4 s01 = textureLod(tex, prevUv + vec2( offX, -offY), 0.0);
    vec4 s10 = textureLod(tex, prevUv + vec2(-offX,  offY), 0.0);
    vec4 s11 = textureLod(tex, prevUv + vec2( offX,  offY), 0.0);
    return 0.25 * (s00 + s01 + s10 + s11);
}

// 5-tap Jimenez Catmull-Rom bicubic 4-channel backward warp with anti-ringing min/max clamp.
// Matches warp_catmull_rom_5tap() in upways/marshaling.py 1:1.
vec4 sampleCatmullRom5Tap(
    sampler2D tex,
    ivec2 dstPos,
    uvec2 dstSize,
    uvec2 srcSize,
    vec2 mvUv,
    out float inBoundsMask
) {
    vec2 uv = (vec2(dstPos) + 0.5) / vec2(dstSize);
    vec2 prevUv = uv - mvUv;
    bool inBounds = (prevUv.x >= 0.0) && (prevUv.x <= 1.0) && (prevUv.y >= 0.0) && (prevUv.y <= 1.0);
    inBoundsMask = inBounds ? 1.0 : 0.0;
    if (!inBounds) {
        return vec4(0.0);
    }

    vec2 clampedUv = clamp(prevUv, vec2(0.0), vec2(1.0));
    vec2 srcF = vec2(srcSize);
    vec2 pos = clampedUv * srcF;
    vec2 tc1 = floor(pos - 0.5) + 0.5;
    vec2 f = pos - tc1;
    vec2 f2 = f * f;
    vec2 f3 = f2 * f;

    vec2 w0 = -0.5 * f3 + f2 - 0.5 * f;
    vec2 w1 =  1.5 * f3 - 2.5 * f2 + 1.0;
    vec2 w2 = -1.5 * f3 + 2.0 * f2 + 0.5 * f;
    vec2 w3 =  0.5 * f3 - 0.5 * f2;

    vec2 w12 = w1 + w2;
    vec2 off12 = w2 / w12;

    vec2 tc0  = (tc1 - 1.0) / srcF;
    vec2 tc12 = (tc1 + off12) / srcF;
    vec2 tc3  = (tc1 + 2.0) / srcF;

    float wt0 = w12.x * w0.y;
    float wt1 = w0.x  * w12.y;
    float wt2 = w12.x * w12.y;
    float wt3 = w3.x  * w12.y;
    float wt4 = w12.x * w3.y;
    float wtSum = max(wt0 + wt1 + wt2 + wt3 + wt4, 1e-6);

    vec4 c = textureLod(tex, vec2(tc12.x, tc0.y),  0.0);
    vec4 raw = c * wt0;
    f16vec4 cMin = f16vec4(c);
    f16vec4 cMax = f16vec4(c);

    c = textureLod(tex, vec2(tc0.x,  tc12.y), 0.0);
    raw += c * wt1;
    cMin = min(cMin, f16vec4(c));
    cMax = max(cMax, f16vec4(c));

    c = textureLod(tex, vec2(tc12.x, tc12.y), 0.0);
    raw += c * wt2;
    cMin = min(cMin, f16vec4(c));
    cMax = max(cMax, f16vec4(c));

    c = textureLod(tex, vec2(tc3.x,  tc12.y), 0.0);
    raw += c * wt3;
    cMin = min(cMin, f16vec4(c));
    cMax = max(cMax, f16vec4(c));

    c = textureLod(tex, vec2(tc12.x, tc3.y),  0.0);
    raw += c * wt4;
    cMin = min(cMin, f16vec4(c));
    cMax = max(cMax, f16vec4(c));

    vec4 res = clamp(raw / wtSum, vec4(cMin), vec4(cMax));
    return vec4(clamp(res.rgb, vec3(0.0), vec3(MAX_LOG1P_IRRADIANCE)), clamp(res.a, 0.0, 1.0));
}

#endif // UPWAYS_COMMON_GLSL
