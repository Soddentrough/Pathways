#ifndef RESTIR_COMMON_GLSL
#define RESTIR_COMMON_GLSL

#include "wavefront_common.glsl"

// =============================================================================
// ReSTIR DI: spatiotemporal reservoir resampling for direct illumination.
// =============================================================================
// One reservoir per primary-hit pixel stores a single light sample
// (light index + UV on area lights) selected by RIS from:
//   * a fresh M-candidate stream per shaded point,
//   * one temporal tap (MV-reprojected center of the previous frame's grid),
//   * four spatial taps (rotated cross neighborhood of the reprojected pixel,
//     read from the previous frame's grid — no cross-lane shuffles, so reuse
//     is legal under divergent wavefront control flow).
//
// Target function p̂ (scalar, solid-angle measure at the primary hit x1):
//
//   p̂ = lum(effective_emission * proxy_BRDF) * NdotL_eff
//
// For AREA lights the effective emission deliberately omits 1/d² and the
// cosine at the light; those factors live in the source pdf expressed in
// per-vertex solid-angle measure: q(ω) = d² / (A·cosL) · selectPdf. Because
// q depends on the vertex, reusing a stored weight requires the ratio
// q_src / q_curr — that is exactly what evalDirectReuseJacobian() returns for
// area lights. For point-like lights (spot / point / directional) the 1/d²
// attenuation is folded into the stored target itself and q is
// position-independent, so the ratio is exactly 1.0. (Regression note: a
// previous revision multiplied point lights by an extra (d_src/d_curr)² and
// biased spot reuse — do not "harmonize" the branches.)
//
// All clamps and gates below are deliberate, documented bias sources (see
// docs/reports/restir_review_2026_10_10.md §4.3): keep them centralized here.
//
// Path-space ReSTIR PT (GI reservoirs with path replay) is a separate roadmap
// item — see docs/RESTIR_PT_DESIGN.md. Do NOT re-add speculative PT helpers
// here without a live call site.
// =============================================================================

// -----------------------------------------------------------------------------
// Reservoir layout: 8 × 4 B = 32 bytes, matches DiReservoirGPU in ReSTIRTypes.hpp
// -----------------------------------------------------------------------------
struct DiReservoir {
    uint  idxM;        // [15:0] light index, [31:16] M
    float wSum;        // RIS weight sum
    float targetPdf;   // p̂ of the selected sample at the storing pixel
    uint  flags;       // [7:0] age, [8] valid, [9] transmission lobe, [25:10] oct16(x1 normal)
    float hitDist;     // x1 -> light-sample distance (1e4 for directional lights)
    uint  lightPacked; // [15:0] oct16(light normal), [31:16] reserved
    uint  x1Misc;      // [15:0] fp16 cos(theta at light, stored config), [31:16] fp16 primary hit distance
    uint  lightUv16;   // packUnorm2x16(luv) — full 16-bit area-light sample UV
};

// Tunables / bias gates
const uint  RESTIR_MAX_AGE              = 255u;   // 8-bit age ceiling (disocclusion and motion handle invalidation)
const float RESTIR_TAP_NORMAL_MIN_DOT   = 0.70f;  // disocclusion guard on stored x1 normal
const float RESTIR_TAP_DEPTH_REL_GATE   = 0.15f;  // disocclusion guard on primary hit distance
const float RESTIR_JACOBIAN_CLAMP_MIN   = 0.05f;
const float RESTIR_JACOBIAN_CLAMP_MAX   = 10.0f;

const ivec2 RESTIR_TAP_OFFSETS[32] = ivec2[32](
    ivec2( 2,  0), ivec2( 0,  2), ivec2(-2,  0), ivec2( 0, -2),
    ivec2( 2,  2), ivec2(-2,  2), ivec2(-2, -2), ivec2( 2, -2),
    ivec2( 4,  0), ivec2( 0,  4), ivec2(-4,  0), ivec2( 0, -4),
    ivec2( 3,  3), ivec2(-3,  3), ivec2(-3, -3), ivec2( 3, -3),
    ivec2( 5,  1), ivec2(-1,  5), ivec2(-5, -1), ivec2( 1, -5),
    ivec2( 4,  4), ivec2(-4,  4), ivec2(-4, -4), ivec2( 4, -4),
    ivec2( 6,  0), ivec2( 0,  6), ivec2(-6,  0), ivec2( 0, -6),
    ivec2( 6,  6), ivec2(-6,  6), ivec2(-6, -6), ivec2( 6, -6)
);

// -----------------------------------------------------------------------------
// Bit packing
// -----------------------------------------------------------------------------
uint restirPackIdxM(uint id, uint M) {
    return (id & 0xFFFFu) | ((M & 0xFFFFu) << 16u);
}

uint restirGetLightIdx(uint idxM) {
    return idxM & 0xFFFFu;
}

uint restirGetM(uint idxM) {
    return (idxM >> 16u) & 0xFFFFu;
}

uint restirPackOct16(vec3 v) {
    vec2 enc = octEncode(normalize(v)) * 0.5 + 0.5;
    uint u8 = uint(clamp(enc.x * 255.0, 0.0, 255.0));
    uint v8 = uint(clamp(enc.y * 255.0, 0.0, 255.0));
    return (u8 & 0xFFu) | ((v8 & 0xFFu) << 8u);
}

vec3 restirUnpackOct16(uint p) {
    float u = float(p & 0xFFu) / 255.0;
    float v = float((p >> 8u) & 0xFFu) / 255.0;
    return octDecode(vec2(u, v) * 2.0 - 1.0);
}

uint restirPackFlags(uint age, bool valid, bool trans, vec3 x1Normal) {
    return (age & 0xFFu) |
           ((valid ? 1u : 0u) << 8u) |
           ((trans ? 1u : 0u) << 9u) |
           (restirPackOct16(x1Normal) << 10u);
}

uint restirGetAge(uint flags) {
    return flags & 0xFFu;
}

bool restirIsValid(uint flags) {
    return ((flags >> 8u) & 1u) != 0u;
}

vec3 restirGetX1Normal(uint flags) {
    return restirUnpackOct16((flags >> 10u) & 0xFFFFu);
}

uint restirPackF16Pair(float a, float b) {
    return packHalf2x16(vec2(a, b));
}

vec2 restirUnpackF16Pair(uint p) {
    return unpackHalf2x16(p);
}

uint restirPackUv8(vec2 uv) {
    uint u = uint(clamp(uv.x * 255.0 + 0.5, 0.0, 255.0));
    uint v = uint(clamp(uv.y * 255.0 + 0.5, 0.0, 255.0));
    return (u & 0xFFu) | ((v & 0xFFu) << 8u);
}

vec2 restirUnpackUv8(uint packedUv) {
    float u = float(packedUv & 0xFFu) * (1.0 / 255.0);
    float v = float((packedUv >> 8u) & 0xFFu) * (1.0 / 255.0);
    return vec2(u, v);
}

DiReservoir invalidDiReservoir() {
    DiReservoir r;
    r.idxM = 0u;
    r.wSum = 0.0;
    r.targetPdf = 0.0;
    r.flags = 0u;
    r.hitDist = 0.0;
    r.lightPacked = 0u;
    r.x1Misc = 0u;
    r.lightUv16 = 0u;
    return r;
}

// The RIS stage functions below require `lights`, `lightNodes`, `triangles`
// and `ubo` to be declared before this include. Passes that only need the
// reservoir type for invalidation writes skip them by not defining
// RESTIR_INCLUDE_SAMPLER_HELPERS.
#ifdef RESTIR_INCLUDE_SAMPLER_HELPERS

#include "mis_common.glsl"

// -----------------------------------------------------------------------------
// Direct light path reconnection for a stored (lightIndex, luv) sample
// -----------------------------------------------------------------------------
bool reconnectDirectPath(
    vec3 hitPoint,
    Light light,
    vec2 luv,
    out vec3 lightDir,
    out float lightDist,
    out vec3 lightEmission,
    out float cosLight
) {
    lightEmission = light.emission.rgb;
    cosLight = 1.0;
    uint lightType = uint(light.position.w);
    if (lightType == 0u || lightType == 3u /* AREA QUAD / TRIANGLE */) {
        vec2 uv = luv;
        if (lightType == 3u) {
            float r1 = sqrt(luv.x);
            uv = vec2(1.0 - r1, luv.y * r1);
        }
        vec3 lightPoint = light.position.xyz + uv.x * light.u.xyz + uv.y * light.v.xyz;
        vec3 toLight = lightPoint - hitPoint;
        lightDist = length(toLight);
        if (lightDist < 1e-4) return false;
        lightDir = toLight / lightDist;
        cosLight = dot(-lightDir, light.normal.xyz);
        if (cosLight <= 0.0) return false;
        return true;
    } else if (lightType == 1u /* SPOT */) {
        vec3 toLight = light.position.xyz - hitPoint;
        lightDist = length(toLight);
        if (lightDist < 1e-4) return false;
        lightDir = toLight / lightDist;
        float cosSpot = dot(-lightDir, light.normal.xyz);
        if (cosSpot < light.v.w) return false;
        float spotFactor = clamp((cosSpot - light.v.w) / max(light.u.w - light.v.w, 1e-4), 0.0, 1.0);
        lightEmission *= spotFactor / max(lightDist * lightDist, 1e-4);
        return true;
    } else if (uint(light.position.w) == 2u /* DIRECTIONAL */) {
        lightDir = normalize(light.normal.xyz);
        lightDist = 10000.0;
        return true;
    }
    return false;
}

// -----------------------------------------------------------------------------
// Measure-conversion factor for reuse: q_src / q_curr (see file header).
//   AREA:  q(ω) = d²/(A·cosL)·selectPdf  ->  (d_src/d_curr)² · (cosL_curr/cosL_src)
//   SPOT / DIRECTIONAL: q is position-independent -> 1.0 (NOT a distance ratio!)
// The clamp is a deliberate bias gate against near-singularity reconnects.
// -----------------------------------------------------------------------------
float evalDirectReuseJacobian(
    uint lightType,
    float cosLightCurr,
    float distCurr,
    float cosLightOrig,
    float distOrig
) {
    if (lightType == 1u || lightType == 2u /* SPOT / DIRECTIONAL */) {
        return 1.0;
    }
    float distRatio = distOrig / max(distCurr, 1e-4);
    float J = (distRatio * distRatio) * (cosLightCurr / max(cosLightOrig, 1e-4));
    return clamp(J, RESTIR_JACOBIAN_CLAMP_MIN, RESTIR_JACOBIAN_CLAMP_MAX);
}

// -----------------------------------------------------------------------------
// Selected-sample record shared between the RIS stages and finalization
// -----------------------------------------------------------------------------
struct DiSample {
    uint lightIdx;
    vec3 dir;
    float dist;
    vec3 emission;    // effective emission (includes 1/d² for spot lights)
    float selectPdf;  // light-selection pdf (alias / light tree)
    float cosLight;   // cosine at the light's own normal (area lights)
    float ndotL;      // NdotL (or transmission-adjusted) at x1
    float target;     // p̂
    vec2 luv;
    bool isTrans;
    bool has;
};

// -----------------------------------------------------------------------------
// Fresh candidate stream: M light samples, classical 1-sample stream RIS.
// Builds the initial reservoir including the current-surface context (x1).
// -----------------------------------------------------------------------------
void restirDiFreshStream(
    uint numLights,
    bool useLightTree,
    vec3 hitPoint,
    vec3 hitNormal,
    float hitT,
    f16vec3 proxyLumWeight,
    float diffTrans,
    uint M,
    inout uint seed,
    out DiReservoir r,
    out DiSample sel
) {
    float totalWeightSum = 0.0;
    sel.has = false;
    sel.lightIdx = 0u;
    sel.dir = vec3(0.0);
    sel.dist = 0.0;
    sel.emission = vec3(0.0);
    sel.selectPdf = 1.0;
    sel.cosLight = 1.0;
    sel.ndotL = 0.0;
    sel.target = 0.0;
    sel.luv = vec2(0.5);
    sel.isTrans = false;

    for (uint c = 0u; c < M; ++c) {
        uint lightIdx = 0u;
        float lightSelectPdf = 1.0;
        if (useLightTree && numLights > 1u) {
            SAMPLE_LIGHT_TREE(numLights, hitPoint, hitNormal, seed, lightIdx, lightSelectPdf);
        } else {
            SAMPLE_LIGHT_ALIAS(numLights, seed, lightIdx, lightSelectPdf);
        }

        Light light = lights[lightIdx];
        vec3 lightDir = vec3(0.0);
        float lightDist = 0.0;
        float lightPdf = 1.0;
        vec3 lightEmission = light.emission.rgb;
        bool validLightSample = false;
        vec2 luv = vec2(0.5);
        float cosLight = 1.0;

        uint lightType = uint(light.position.w);
        if (lightType == 0u || lightType == 3u /* AREA QUAD / TRIANGLE */) {
            luv = randVec2(seed);
            vec2 uv = luv;
            if (lightType == 3u) {
                float r1 = sqrt(luv.x);
                uv = vec2(1.0 - r1, luv.y * r1);
            }
            vec3 lightPoint = light.position.xyz + uv.x * light.u.xyz + uv.y * light.v.xyz;
            vec3 toLight = lightPoint - hitPoint;
            lightDist = length(toLight);
            lightDir = toLight / max(lightDist, 1e-4);

            cosLight = dot(-lightDir, light.normal.xyz);
            if (cosLight > 0.0) {
                float area = light.emission.w;
                lightPdf = (lightDist * lightDist) / (area * cosLight);
                validLightSample = true;
            }
        } else if (lightType == 1u /* SPOT */) {
            vec3 toLight = light.position.xyz - hitPoint;
            lightDist = length(toLight);
            lightDir = toLight / max(lightDist, 1e-4);

            float cosSpot = dot(-lightDir, light.normal.xyz);
            float innerCos = light.u.w;
            float outerCos = light.v.w;
            if (cosSpot >= outerCos) {
                float spotFactor = clamp((cosSpot - outerCos) / max(innerCos - outerCos, 1e-4), 0.0, 1.0);
                lightEmission *= spotFactor / max(lightDist * lightDist, 1e-4);
                lightPdf = 1.0;
                validLightSample = true;
            }
        } else if (uint(light.position.w) == 2u /* DIRECTIONAL */) {
            lightDir = normalize(light.normal.xyz);
            lightDist = 10000.0;
            lightPdf = 1.0;
            validLightSample = true;
        }

        float rawNdotL = dot(hitNormal, lightDir);
        bool isTrans = (rawNdotL < 0.0 && diffTrans > 0.001);
        float NdotL = isTrans ? -rawNdotL : clamp(rawNdotL, 0.0, 1.0);
        if (validLightSample && NdotL > 0.0 && (rawNdotL > 0.0 || isTrans)) {
            float transScale = isTrans ? diffTrans : (1.0 - diffTrans);
            float target = max(dot(lightEmission, vec3(proxyLumWeight)) * (NdotL * transScale), 1e-6);
            float sourcePdf = max(lightPdf * lightSelectPdf, 1e-20);
            float weight = target / sourcePdf;

            totalWeightSum += weight;
            if (randFloat(seed) * totalWeightSum < weight) {
                sel.lightIdx = lightIdx;
                sel.dir = lightDir;
                sel.dist = lightDist;
                sel.emission = lightEmission;
                sel.selectPdf = lightSelectPdf;
                sel.cosLight = cosLight;
                sel.ndotL = NdotL;
                sel.target = target;
                sel.luv = luv;
                sel.isTrans = isTrans;
                sel.has = true;
            }
        }
    }

    if (sel.has) {
        r.idxM = restirPackIdxM(sel.lightIdx, M);
        r.wSum = totalWeightSum;
        r.targetPdf = sel.target;
        r.hitDist = sel.dist;
        r.lightPacked = restirPackOct16(lights[sel.lightIdx].normal.xyz);
        r.flags = restirPackFlags(0u, true, sel.isTrans, hitNormal);
        r.x1Misc = restirPackF16Pair(sel.cosLight, min(hitT, 60000.0));
        r.lightUv16 = packUnorm2x16(sel.luv);
    } else {
        // Unlit / backfacing candidates evaluated this frame: carries 0 weight, but M trials
        r.idxM = restirPackIdxM(0u, M);
        r.wSum = 0.0;
        r.targetPdf = 0.0;
        r.hitDist = 0.0;
        r.lightPacked = 0u;
        r.flags = restirPackFlags(0u, true, false, hitNormal);
        r.x1Misc = restirPackF16Pair(1.0, min(hitT, 60000.0));
        r.lightUv16 = 0u;
    }
}

// -----------------------------------------------------------------------------
// History tap: reconnect a reservoir's light sample at the current shading
// point, gate on surface similarity + reconnection validity, and stream-combine
// into r. mCap truncates the tap's history (weights scaled consistently).
// -----------------------------------------------------------------------------
void restirDiApplyTap(
    inout DiReservoir r,
    inout DiSample sel,
    in DiReservoir src,
    uint numLights,
    vec3 hitPoint,
    vec3 hitNormal,
    float hitT,
    f16vec3 proxyLumWeight,
    float diffTrans,
    uint mCap,
    uint ageNew,
    bool isOccluded,
    inout uint seed
) {
    if (!restirIsValid(src.flags)) return;
    if (restirGetAge(src.flags) >= RESTIR_MAX_AGE) return;

    uint mSrc = restirGetM(src.idxM);
    if (mSrc == 0u) return;

    uint srcLightIdx = restirGetLightIdx(src.idxM);
    if (srcLightIdx >= numLights) return;

    // Disocclusion / surface-similarity gates against the stored x1 context.
    if (dot(hitNormal, restirGetX1Normal(src.flags)) < RESTIR_TAP_NORMAL_MIN_DOT) return;
    vec2 srcMisc = restirUnpackF16Pair(src.x1Misc);
    float srcX1Dist = srcMisc.y;
    float relDepthDiff = abs(hitT - srcX1Dist) / max(max(hitT, srcX1Dist), 1e-4);
    if (relDepthDiff > RESTIR_TAP_DEPTH_REL_GATE) return;

    uint mTap = min(mSrc, mCap);

    // If source evaluated samples with 0 weight, it contributes 0 weight but carries mTap trials.
    if (src.targetPdf <= 0.0 || src.wSum <= 0.0) {
        uint combinedM = restirGetM(r.idxM) + mTap;
        r.idxM = restirPackIdxM(restirGetLightIdx(r.idxM), combinedM);
        return;
    }

    Light srcLight = lights[srcLightIdx];
    vec2 srcLuv = unpackUnorm2x16(src.lightUv16);
    vec3 reconDir;
    float reconDist;
    vec3 reconEmission;
    float cosLightNew;
    bool reconOk = reconnectDirectPath(hitPoint, srcLight, srcLuv, reconDir, reconDist, reconEmission, cosLightNew);

    float rawNdotL = reconOk ? dot(hitNormal, reconDir) : 0.0;
    bool isTrans = (rawNdotL < 0.0 && diffTrans > 0.001);
    float NdotL = isTrans ? -rawNdotL : clamp(rawNdotL, 0.0, 1.0);
    bool validGeom = !isOccluded && reconOk && NdotL > 0.0 && (rawNdotL > 0.0 || isTrans);

    if (!validGeom) {
        // Sample lands behind horizon or is occluded from current surface: carries 0 weight, but increments M
        uint combinedM = restirGetM(r.idxM) + mTap;
        r.idxM = restirPackIdxM(restirGetLightIdx(r.idxM), combinedM);
        return;
    }

    float transScale = isTrans ? diffTrans : (1.0 - diffTrans);
    float pHat = max(dot(reconEmission, vec3(proxyLumWeight)) * (NdotL * transScale), 1e-6);

    float J = evalDirectReuseJacobian(uint(srcLight.position.w), cosLightNew, reconDist, srcMisc.x, src.hitDist);

    // History M-truncation scales the stored sum by mTap/mSrc before re-weighting.
    float weight = (src.wSum / float(mSrc)) * float(mTap) * (pHat / max(src.targetPdf, 1e-6)) * J;

    r.wSum += weight;
    uint combinedM = restirGetM(r.idxM) + mTap;

    if (weight > 0.0 && (randFloat(seed) * r.wSum < weight || !sel.has)) {
        sel.lightIdx = srcLightIdx;
        sel.dir = reconDir;
        sel.dist = reconDist;
        sel.emission = reconEmission;
        sel.selectPdf = misLightSelectPdf(srcLightIdx, hitPoint, numLights);
        sel.cosLight = cosLightNew;
        sel.ndotL = NdotL;
        sel.target = pHat;
        sel.luv = srcLuv;
        sel.isTrans = isTrans;
        sel.has = true;

        r.idxM = restirPackIdxM(srcLightIdx, combinedM);
        r.targetPdf = pHat;
        r.hitDist = reconDist;
        r.lightPacked = restirPackOct16(srcLight.normal.xyz);
        // x1 context is re-stamped with the CURRENT surface: the reservoir
        // describes this pixel for the next frame's taps.
        r.flags = restirPackFlags(ageNew, true, isTrans, hitNormal);
        r.x1Misc = restirPackF16Pair(cosLightNew, min(hitT, 60000.0));
        r.lightUv16 = packUnorm2x16(srcLuv);
    } else {
        r.idxM = restirPackIdxM(restirGetLightIdx(r.idxM), combinedM);
    }
}

#endif // RESTIR_INCLUDE_SAMPLER_HELPERS

// =============================================================================
// ReSTIR PT Indirect Path-Space Resampling (Secondary Vertex Reconnection)
// =============================================================================
// Resamples the 1-bounce illumination integral at primary hit x1 over the
// triangle-surface domain Omega_GI (with full-sphere partition S^2 = Omega_GI u
// (S^2 \ Omega_GI) so sky, cutouts, spheres, dielectrics, and indexed emitters
// remain 100% unbiased in standard MC while contributing M=1, w=0 trials to GI).
//
// Stored as 8 scalar uints (32 bytes), matching GiReservoirGPU in ReSTIRTypes.hpp:
//   posXBits:      floatBitsToUint(x2.x) world-space reconnection vertex
//   posYBits:      floatBitsToUint(x2.y)
//   posZBits:      floatBitsToUint(x2.z)
//   wSumBits:      RIS running weight sum (fp32 bits)
//   targetCosBits: [31:8] fp32(target)[31:8], [7:0] cosSrc (unorm8)
//   radRgb9e5:     RGB9E5 shared-exponent radiance L̂2 at x2
//   normalsPacked: [15:0] oct16(n2), [31:16] oct16(x1Normal)
//   dSrcFlagsM:    [15:0] fp16 dSrc, [16] valid, [17] hasSurface,
//                  [23:18] age (0..63), [31:24] M (0..255)
struct GiReservoir {
    uint posXBits;
    uint posYBits;
    uint posZBits;
    uint wSumBits;
    uint targetCosBits;
    uint radRgb9e5;
    uint normalsPacked;
    uint dSrcFlagsM;
};

// Per-pixel BRDF context at the primary hit x1 (24 bytes, matches X1ContextGPU).
struct X1Context {
    uint n1Oct;     // oct32(x1 world shading normal)
    uint albedoRG;  // fp16 (baseColor.r, baseColor.g)
    uint albedoB;   // [15:0] fp16 baseColor.b, [31:16] fp16 metallic
    uint flags;     // [0] valid, [1] reserved, [31:2] qSelf fp32[31:2]
    uint woOct;     // oct32(x1 -> camera direction, V at primary hit)
    uint f0Pad;     // [15:0] fp16 dielectric scalar F0, [31:16] fp16 roughness
};

const uint  GI_MAX_AGE          = 63u;
const uint  GI_INVALID_INST_20  = 0xFFFFFu;
const float GI_TAP_MIN_COS      = 0.05;
const float GI_TAP_MIN_DIST     = 1e-3;
const float GI_WEIGHT_CLAMP_MAX = 500.0;

uint restirPackRgb9e5(vec3 rgb) {
    vec3 c = clamp(rgb, vec3(0.0), vec3(65408.0));
    float maxC = max(c.r, max(c.g, c.b));
    if (maxC < 1e-6) return 0u;
    int expShared = clamp(max(-15, int(floor(log2(maxC)))) + 16, 0, 31);
    float denom = exp2(float(expShared - 24));
    int maxM = int(floor(maxC / denom + 0.5));
    if (maxM == 512 && expShared < 31) {
        denom *= 2.0;
        expShared += 1;
    }
    uvec3 m = uvec3(clamp(ivec3(floor(c / denom + 0.5)), ivec3(0), ivec3(511)));
    return m.r | (m.g << 9u) | (m.b << 18u) | (uint(expShared) << 27u);
}

vec3 restirUnpackRgb9e5(uint p) {
    if (p == 0u) return vec3(0.0);
    int expShared = int((p >> 27u) & 0x1Fu);
    float scale = exp2(float(expShared - 24));
    return vec3(float(p & 0x1FFu), float((p >> 9u) & 0x1FFu), float((p >> 18u) & 0x1FFu)) * scale;
}

X1Context invalidX1Context() {
    X1Context c;
    c.n1Oct = 0u;
    c.albedoRG = 0u;
    c.albedoB = 0u;
    c.flags = 0u;
    c.woOct = 0u;
    c.f0Pad = 0u;
    return c;
}

X1Context makeX1Context(vec3 n1, vec3 baseColor, float metallic, vec3 wo, float f0Scalar, float roughness, float qSelf) {
    X1Context c;
    c.n1Oct = packOct32(normalize(n1));
    c.albedoRG = packHalf2x16(clamp(baseColor.rg, vec2(0.0), vec2(65000.0)));
    c.albedoB = packHalf2x16(vec2(clamp(baseColor.b, 0.0, 65000.0), clamp(metallic, 0.0, 1.0)));
    c.flags = (floatBitsToUint(max(qSelf, 1e-8)) & ~3u) | 1u;
    c.woOct = packOct32(normalize(wo));
    c.f0Pad = packHalf2x16(vec2(clamp(f0Scalar, 0.0, 1.0), clamp(roughness, 0.0, 1.0)));
    return c;
}

bool  x1ContextValid(X1Context c)     { return (c.flags & 1u) != 0u; }
vec3  x1ContextNormal(X1Context c)    { return unpackOct32(c.n1Oct); }
vec3  x1ContextBaseColor(X1Context c) {
    vec2 rg = unpackHalf2x16(c.albedoRG);
    return vec3(rg.x, rg.y, unpackHalf2x16(c.albedoB).x);
}
float x1ContextMetallic(X1Context c)  { return unpackHalf2x16(c.albedoB).y; }
float x1ContextQSelf(X1Context c)     { return uintBitsToFloat(c.flags & ~3u); }
vec3  x1ContextWo(X1Context c)        { return unpackOct32(c.woOct); }
float x1ContextF0(X1Context c)        { return unpackHalf2x16(c.f0Pad).x; }
float x1ContextRoughness(X1Context c) { return unpackHalf2x16(c.f0Pad).y; }

GiReservoir invalidGiReservoir() {
    GiReservoir r;
    r.posXBits = 0u;
    r.posYBits = 0u;
    r.posZBits = 0u;
    r.wSumBits = 0u;
    r.targetCosBits = 0u;
    r.radRgb9e5 = 0u;
    r.normalsPacked = 0u;
    r.dSrcFlagsM = 0u;
    return r;
}

bool  giIsValid(GiReservoir r)      { return ((r.dSrcFlagsM >> 16u) & 1u) != 0u; }
bool  giHasSurface(GiReservoir r)   { return ((r.dSrcFlagsM >> 17u) & 1u) != 0u; }
uint  giGetAge(GiReservoir r)       { return (r.dSrcFlagsM >> 18u) & 0x3Fu; }
uint  giGetM(GiReservoir r)         { return (r.dSrcFlagsM >> 24u) & 0xFFu; }
float giGetDSrc(GiReservoir r)      { return unpackHalf2x16(r.dSrcFlagsM & 0xFFFFu).x; }
float giGetCosSrc(GiReservoir r)    { return float(r.targetCosBits & 0xFFu) * (1.0 / 255.0); }
float giGetTarget(GiReservoir r)    { return uintBitsToFloat(r.targetCosBits & ~0xFFu); }
float giGetWSum(GiReservoir r)      { return uintBitsToFloat(r.wSumBits); }
vec3  giGetX2(GiReservoir r)        { return vec3(uintBitsToFloat(r.posXBits), uintBitsToFloat(r.posYBits), uintBitsToFloat(r.posZBits)); }
vec3  giGetN2(GiReservoir r)        { return restirUnpackOct16(r.normalsPacked & 0xFFFFu); }
vec3  giGetX1Normal(GiReservoir r)  { return restirUnpackOct16((r.normalsPacked >> 16u) & 0xFFFFu); }
vec3  giGetRadiance(GiReservoir r)  { return restirUnpackRgb9e5(r.radRgb9e5); }

GiReservoir makeGiReservoir(bool hasSurface, vec3 x2, vec3 n2, vec3 x1Normal, float wSum, uint M, float target, vec3 radiance, float dSrc, float cosSrc, uint age) {
    GiReservoir r = invalidGiReservoir();
    bool validSurf = hasSurface && (target > 0.0);
    r.posXBits = validSurf ? floatBitsToUint(x2.x) : 0u;
    r.posYBits = validSurf ? floatBitsToUint(x2.y) : 0u;
    r.posZBits = validSurf ? floatBitsToUint(x2.z) : 0u;
    r.wSumBits = floatBitsToUint(max(wSum, 0.0));
    uint uCos = validSurf ? uint(clamp(round(clamp(cosSrc, 0.0, 1.0) * 255.0), 0.0, 255.0)) : 0u;
    uint tBits = validSurf ? (floatBitsToUint(max(target, 0.0)) & ~0xFFu) : 0u;
    r.targetCosBits = tBits | uCos;
    r.radRgb9e5 = validSurf ? restirPackRgb9e5(radiance) : 0u;
    uint octN2 = validSurf ? restirPackOct16(n2) : 0u;
    uint octN1 = restirPackOct16(x1Normal);
    r.normalsPacked = octN2 | (octN1 << 16u);
    uint dHalf = validSurf ? (packHalf2x16(vec2(clamp(dSrc, 0.0, 65000.0), 0.0)) & 0xFFFFu) : 0u;
    r.dSrcFlagsM = dHalf |
                   (1u << 16u) |
                   ((validSurf ? 1u : 0u) << 17u) |
                   ((min(age, GI_MAX_AGE) & 0x3Fu) << 18u) |
                   ((min(M, 255u) & 0xFFu) << 24u);
    return r;
}

// Full analytical Jacobian for secondary vertex reconnection:
// J = (||x2 - x_src||^2 / ||x2 - x_new||^2) * (cos(theta_rc_new) / cos(theta_rc_src))
float evalGIJacobian(float dSrc, float dNew, float cosRcSrc, float cosRcNew) {
    if (dSrc >= 5000.0 || dNew >= 5000.0) return 1.0;
    float dSqSrc = dSrc * dSrc;
    float dSqNew = max(dNew * dNew, 1e-4);
    float cosSrcEff = max(cosRcSrc, 1e-3);
    float cosNewEff = max(cosRcNew, 1e-4);
    float J = (dSqSrc / dSqNew) * (cosNewEff / cosSrcEff);
    return clamp(J, RESTIR_JACOBIAN_CLAMP_MIN, RESTIR_JACOBIAN_CLAMP_MAX);
}

#endif // RESTIR_COMMON_GLSL
