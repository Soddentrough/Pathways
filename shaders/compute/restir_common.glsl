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
// Reservoir layout: 6 x 4 B = 24 bytes, matches DiReservoirGPU in ReSTIRTypes.hpp
// -----------------------------------------------------------------------------
struct DiReservoir {
    uint  idxM;          // [23:0] light index (up to 16.7M lights), [31:24] M (0..255)
    float wSum;          // RIS weight sum
    uint  targetCosBits; // [31:8] fp32(targetPdf)[31:8], [7:0] unorm8 cos(theta at light)
    uint  flags;         // [7:0] age (transient diffTrans pre-reuse), [8] valid, [9] trans lobe, [25:10] oct16(x1 normal)
    uint  distHitT;      // [15:0] fp16 hitDist, [31:16] fp16 primary hit distance
    uint  lightUv16;     // packUnorm2x16(luv) — full 16-bit area-light sample UV
};

// Tunables / bias gates
const uint  RESTIR_MAX_AGE              = 255u;   // 8-bit age ceiling (disocclusion and motion handle invalidation)
const float RESTIR_TAP_NORMAL_MIN_DOT   = 0.90f;  // disocclusion guard on stored x1 normal
const float RESTIR_TAP_DEPTH_REL_GATE   = 0.10f;  // spatial tap disocclusion guard on view depth
const float RESTIR_JACOBIAN_CLAMP_MIN   = 0.25f;
const float RESTIR_JACOBIAN_CLAMP_MAX   = 4.0f;

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
    return (id & 0xFFFFFFu) | (min(M, 255u) << 24u);
}

uint restirGetLightIdx(uint idxM) {
    return idxM & 0xFFFFFFu;
}

uint restirGetM(uint idxM) {
    return (idxM >> 24u) & 0xFFu;
}

uint restirPackOct16(vec3 v) {
    vec2 enc = octEncode(normalize(v)) * 0.5 + 0.5;
    uint u8 = uint(clamp(enc.x * 255.0, 0.0, 255.0));
    uint v8 = uint(clamp(enc.y * 255.0, 0.0, 255.0));
    return (u8 & 0xFFu) | ((v8 & 0xFFu) << 8u);
}

uint restirPackOct16FromOct32(uint oct32) {
    vec2 enc = clamp(unpackSnorm2x16(oct32), vec2(-1.0), vec2(1.0)) * 0.5 + 0.5;
    uint u8 = uint(clamp(enc.x * 255.0, 0.0, 255.0));
    uint v8 = uint(clamp(enc.y * 255.0, 0.0, 255.0));
    return (u8 & 0xFFu) | ((v8 & 0xFFu) << 8u);
}

vec3 restirUnpackOct16(uint p) {
    float u = float(p & 0xFFu) / 255.0;
    float v = float((p >> 8u) & 0xFFu) / 255.0;
    return octDecode(vec2(u, v) * 2.0 - 1.0);
}

uint restirPackLightNormalAndSelectPdf(vec3 lightNormal, float selectPdf) {
    uint bf16Bits = floatBitsToUint(max(selectPdf, 0.0)) & 0xFFFF0000u;
    return bf16Bits | (restirPackOct16(lightNormal) & 0xFFFFu);
}

float restirUnpackSelectPdf(uint lightPacked) {
    uint bf16Bits = lightPacked & 0xFFFF0000u;
    return (bf16Bits != 0u) ? uintBitsToFloat(bf16Bits) : -1.0;
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

// Transient ReSTIR DI control flags packed into bits [31:26] of DiReservoir.flags
// (bits [25:0] store age, valid, trans, oct16(x1Normal)).
const uint DI_FLAG_ACTIVE           = (1u << 26u); // bounce-0 diffuse/complex surface ran ReSTIR DI
const uint DI_FLAG_FRESH_VISIBLE    = (1u << 27u); // fresh DI sample passed shadow ray test
const uint DI_FLAG_NEED_FRESH_VIS   = (1u << 28u); // shadow ray was queued for fresh DI sample
const uint DI_FLAG_DI_MIS_ACTIVE    = (1u << 29u); // DI ReSTIR MIS enabled on primary hit
const uint DI_FLAG_HAS_CONT         = (1u << 30u); // surface also has indirect bounce-1 continuation
const uint DI_FLAG_MERGE_MATTE_SPEC = (1u << 31u); // fold matte spec into ML diffuse output
const uint DI_FLAG_HISTORY_MASK     = 0x03FFFFFFu; // persistent history bits [25:0]

uint restirDiPackTargetCos(float targetPdf, float cosLight) {
    uint tBits = (targetPdf > 0.0) ? (floatBitsToUint(targetPdf) & ~0xFFu) : 0u;
    uint c8 = uint(clamp(round(clamp(cosLight, 0.0, 1.0) * 255.0), 0.0, 255.0));
    return tBits | c8;
}

float restirDiGetTarget(uint targetCosBits) {
    return uintBitsToFloat(targetCosBits & ~0xFFu);
}

float restirDiGetCosLight(uint targetCosBits) {
    return float(targetCosBits & 0xFFu) * (1.0 / 255.0);
}

uint restirDiPackDistHitT(float hitDist, float hitT) {
    return packHalf2x16(vec2(clamp(hitDist, 0.0, 60000.0), clamp(hitT, 0.0, 60000.0)));
}

vec2 restirDiUnpackDistHitT(uint distHitT) {
    return unpackHalf2x16(distHitT);
}

DiReservoir invalidDiReservoir() {
    DiReservoir r;
    r.idxM = 0u;
    r.wSum = 0.0;
    r.targetCosBits = 0u;
    r.flags = 0u;
    r.distHitT = 0u;
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
bool reconnectEnvDirectPath(
    vec2 envUv,
    float envIntensity,
    out vec3 lightDir,
    out float lightDist,
    out vec3 lightEmission
) {
    if (!(envIntensity > 0.0)) {
        lightDir = vec3(0.0, 1.0, 0.0);
        lightDist = 0.0;
        lightEmission = vec3(0.0);
        return false;
    }
    lightDir = envUvToDir(envUv);
    lightDist = 10000.0;
    lightEmission = textureLod(environmentMap, envUv, 0.0).rgb * envIntensity;
    return true;
}

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
float restirEvalTargetPdf(
    vec3 lightEmission,
    vec3 lightDir,
    vec3 hitNormal,
    vec3 V,
    f16vec3 diffColor,
    f16vec3 F0,
    float alphaRoughness,
    float diffTrans,
    bool isTrans
) {
    float rawNdotL = dot(hitNormal, lightDir);
    float NdotL = isTrans ? max(-rawNdotL, 0.0) : max(rawNdotL, 0.0);
    if (NdotL <= 0.0) return 0.0;

    float transScale = isTrans ? diffTrans : (1.0 - diffTrans);
    vec3 f = vec3(diffColor) * (INV_PI * transScale);

    if (!isTrans) {
        float NdotV = max(dot(hitNormal, V), 1e-4);
        vec3 H = normalize(V + lightDir);
        float NdotH = max(dot(hitNormal, H), 0.0);
        float VdotH = max(dot(V, H), 0.0);
        float a = clamp(alphaRoughness, 0.12, 1.0);
        float D = distributionGGX(NdotH, a);
        float Vis = visibilitySmithGGXCorrelated(NdotV, NdotL, a);
        vec3 f0v = vec3(F0);
        vec3 F = f0v + (vec3(1.0) - f0v) * pow(clamp(1.0 - VdotH, 0.0, 1.0), 5.0);
        f += (D * Vis) * F;
    }

    const vec3 LUMA = vec3(0.2126, 0.7152, 0.0722);
    return max(dot(lightEmission * f, LUMA) * NdotL, 0.0);
}

void restirDiFreshStream(
    uint numLights,
    float envIntensity,
    bool useLightTree,
    vec3 hitPoint,
    vec3 hitNormal,
    float hitT,
    vec3 V,
    f16vec3 diffColor,
    f16vec3 F0,
    float alphaRoughness,
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

    float pEnv = (envNeeActive() && envIntensity > 0.0)
        ? ((numLights == 0u) ? 1.0 : envNeeSelectProb())
        : 0.0;
    if (numLights == 0u && pEnv <= 0.0) {
        r.idxM = restirPackIdxM(0u, M);
        r.wSum = 0.0;
        r.targetCosBits = restirDiPackTargetCos(0.0, 1.0);
        r.flags = restirPackFlags(0u, true, false, hitNormal);
        r.distHitT = restirDiPackDistHitT(0.0, hitT);
        r.lightUv16 = 0u;
        return;
    }

    for (uint c = 0u; c < M; ++c) {
        bool pickEnv = (numLights == 0u) || (pEnv > 0.0 && randFloat(seed) < pEnv);
        uint lightIdx = RESTIR_LIGHT_IDX_ENV;
        float lightSelectPdf = pEnv;
        vec3 lightDir = vec3(0.0);
        float lightDist = 0.0;
        float lightPdf = 1.0;
        vec3 lightEmission = vec3(0.0);
        bool validLightSample = false;
        vec2 luv = vec2(0.5);
        float cosLight = 1.0;

        if (pickEnv) {
            if (sampleEnvMapImportance(randVec3(seed), lightDir, luv, lightPdf)) {
                lightDist = 10000.0;
                lightEmission = textureLod(environmentMap, luv, 0.0).rgb * envIntensity;
                validLightSample = true;
            }
        } else {
            if (useLightTree && numLights > 1u) {
                SAMPLE_LIGHT_TREE(numLights, hitPoint, hitNormal, seed, lightIdx, lightSelectPdf);
            } else {
                SAMPLE_LIGHT_ALIAS(numLights, seed, lightIdx, lightSelectPdf);
            }
            lightSelectPdf *= max(1.0 - pEnv, 1e-20);

            Light light = lights[lightIdx];
            lightEmission = light.emission.rgb;
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
        }

        float rawNdotL = dot(hitNormal, lightDir);
        bool isTrans = (rawNdotL < 0.0 && diffTrans > 0.001);
        float NdotL = isTrans ? -rawNdotL : clamp(rawNdotL, 0.0, 1.0);
        if (validLightSample && NdotL > 0.0 && (rawNdotL > 0.0 || isTrans)) {
            float target = restirEvalTargetPdf(lightEmission, lightDir, hitNormal, V,
                                               diffColor, F0, alphaRoughness, diffTrans, isTrans);
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
        r.targetCosBits = restirDiPackTargetCos(sel.target, sel.cosLight);
        r.flags = restirPackFlags(0u, true, sel.isTrans, hitNormal);
        r.distHitT = restirDiPackDistHitT(sel.dist, hitT);
        r.lightUv16 = packUnorm2x16(sel.luv);
    } else {
        // Unlit / backfacing candidates evaluated this frame: carries 0 weight, but M trials
        r.idxM = restirPackIdxM(0u, M);
        r.wSum = 0.0;
        r.targetCosBits = restirDiPackTargetCos(0.0, 1.0);
        r.flags = restirPackFlags(0u, true, false, hitNormal);
        r.distHitT = restirDiPackDistHitT(0.0, hitT);
        r.lightUv16 = 0u;
    }
}

// -----------------------------------------------------------------------------
// History tap: reconnect a reservoir's light sample at the current shading
// point, gate on surface similarity + reconnection validity, and stream-combine
// into r. mCap truncates the tap's history (weights scaled consistently).
// -----------------------------------------------------------------------------
bool restirDiPrepareTapShadowRay(
    in DiReservoir src,
    uint numLights,
    float envIntensity,
    vec3 hitPoint,
    vec3 hitNormal,
    float expectedPrevHitT,
    float diffTrans,
    out vec3 outShadowDir,
    out float outShadowDist,
    out bool outIsTrans
) {
    outShadowDir = vec3(0.0, 1.0, 0.0);
    outShadowDist = 0.0;
    outIsTrans = false;
    if (!restirIsValid(src.flags)) return false;
    if (restirGetAge(src.flags) >= RESTIR_MAX_AGE) return false;
    uint mSrc = restirGetM(src.idxM);
    if (mSrc == 0u) return false;
    uint srcLightIdx = restirGetLightIdx(src.idxM);
    bool srcIsEnv = (srcLightIdx == RESTIR_LIGHT_IDX_ENV) && envNeeActive() && (envIntensity > 0.0);
    if (!srcIsEnv && srcLightIdx >= numLights) return false;
    if (dot(hitNormal, restirGetX1Normal(src.flags)) < RESTIR_TAP_NORMAL_MIN_DOT) return false;
    vec2 srcDistHitT = restirDiUnpackDistHitT(src.distHitT);
    float srcX1Dist = srcDistHitT.y;
    float relDepthDiff = abs(expectedPrevHitT - srcX1Dist) / max(max(expectedPrevHitT, srcX1Dist), 1e-4);
    float tapDepthGate = (numLights > 4096u) ? 0.04 : RESTIR_TAP_DEPTH_REL_GATE;
    if (relDepthDiff > tapDepthGate) return false;
    if (restirDiGetTarget(src.targetCosBits) <= 0.0 || src.wSum <= 0.0) return false;

    vec2 srcLuv = unpackUnorm2x16(src.lightUv16);
    vec3 reconEmission;
    float cosLightNew = 1.0;
    bool reconOk = srcIsEnv
        ? reconnectEnvDirectPath(srcLuv, envIntensity, outShadowDir, outShadowDist, reconEmission)
        : reconnectDirectPath(hitPoint, lights[srcLightIdx], srcLuv, outShadowDir, outShadowDist, reconEmission, cosLightNew);
    if (!reconOk) {
        return false;
    }
    float rawNdotL = dot(hitNormal, outShadowDir);
    outIsTrans = (rawNdotL < 0.0 && diffTrans > 0.001);
    float NdotL = outIsTrans ? -rawNdotL : clamp(rawNdotL, 0.0, 1.0);
    return (NdotL > 0.0 && (rawNdotL > 0.0 || outIsTrans));
}

struct DiTapEval {
    uint mTap;            // 0u if rejected by similarity gates, else min(mSrc, mCap)
    float weightOverPHat; // > 0.0 iff validGeom && weight > 0.0, equals wFactor * float(mTap)
    DiSample tapSample;
};

bool restirDiApplyTapEx(
    inout DiReservoir r,
    inout DiSample sel,
    in DiReservoir src,
    uint numLights,
    float envIntensity,
    vec3 hitPoint,
    vec3 hitNormal,
    float hitT,
    float expectedPrevHitT,
    vec3 V,
    f16vec3 diffColor,
    f16vec3 F0,
    float alphaRoughness,
    float diffTrans,
    uint mCap,
    uint ageNew,
    bool isOccluded,
    bool clampWeight,
    inout uint seed,
    out DiTapEval outEval
) {
    outEval.mTap = 0u;
    outEval.weightOverPHat = 0.0;
    outEval.tapSample.lightIdx = 0u;
    outEval.tapSample.dir = vec3(0.0, 1.0, 0.0);
    outEval.tapSample.dist = 0.0;
    outEval.tapSample.emission = vec3(0.0);
    outEval.tapSample.selectPdf = -1.0;
    outEval.tapSample.cosLight = 1.0;
    outEval.tapSample.ndotL = 0.0;
    outEval.tapSample.target = 0.0;
    outEval.tapSample.luv = vec2(0.0);
    outEval.tapSample.isTrans = false;
    outEval.tapSample.has = false;

    if (!restirIsValid(src.flags)) return false;
    if (restirGetAge(src.flags) >= RESTIR_MAX_AGE) return false;

    uint mSrc = restirGetM(src.idxM);
    if (mSrc == 0u) return false;

    uint srcLightIdx = restirGetLightIdx(src.idxM);
    bool srcIsEnv = (srcLightIdx == RESTIR_LIGHT_IDX_ENV) && envNeeActive() && (envIntensity > 0.0);
    if (!srcIsEnv && srcLightIdx >= numLights) return false;

    // Disocclusion / surface-similarity gates against the stored x1 context.
    if (dot(hitNormal, restirGetX1Normal(src.flags)) < RESTIR_TAP_NORMAL_MIN_DOT) return false;
    vec2 srcDistHitT = restirDiUnpackDistHitT(src.distHitT);
    float srcX1Dist = srcDistHitT.y;
    float relDepthDiff = abs(expectedPrevHitT - srcX1Dist) / max(max(expectedPrevHitT, srcX1Dist), 1e-4);
    float tapDepthGate = (numLights > 4096u) ? 0.04 : RESTIR_TAP_DEPTH_REL_GATE;
    if (relDepthDiff > tapDepthGate) return false;

    uint mTap = min(mSrc, mCap);
    if (mTap == 0u) return false;
    outEval.mTap = mTap;

    // If source evaluated samples with 0 weight, it contributes 0 weight but carries mTap trials.
    float srcTargetPdf = restirDiGetTarget(src.targetCosBits);
    if (srcTargetPdf <= 0.0 || src.wSum <= 0.0) {
        uint combinedM = min(restirGetM(r.idxM) + mTap, 255u);
        r.idxM = restirPackIdxM(restirGetLightIdx(r.idxM), combinedM);
        return false;
    }

    vec2 srcLuv = unpackUnorm2x16(src.lightUv16);
    vec3 reconDir;
    float reconDist;
    vec3 reconEmission;
    float cosLightNew = 1.0;
    bool reconOk = srcIsEnv
        ? reconnectEnvDirectPath(srcLuv, envIntensity, reconDir, reconDist, reconEmission)
        : reconnectDirectPath(hitPoint, lights[srcLightIdx], srcLuv, reconDir, reconDist, reconEmission, cosLightNew);

    float rawNdotL = reconOk ? dot(hitNormal, reconDir) : 0.0;
    bool isTrans = (rawNdotL < 0.0 && diffTrans > 0.001);
    float NdotL = isTrans ? -rawNdotL : clamp(rawNdotL, 0.0, 1.0);
    bool validGeom = !isOccluded && reconOk && NdotL > 0.0 && (rawNdotL > 0.0 || isTrans);

    if (!validGeom) {
        // Sample lands behind horizon or is occluded from current surface: carries 0 weight, but increments M
        uint combinedM = min(restirGetM(r.idxM) + mTap, 255u);
        r.idxM = restirPackIdxM(restirGetLightIdx(r.idxM), combinedM);
        return false;
    }

    float pHat = restirEvalTargetPdf(reconEmission, reconDir, hitNormal, V,
                                     diffColor, F0, alphaRoughness, diffTrans, isTrans);

    uint srcLightType = srcIsEnv ? 2u : uint(lights[srcLightIdx].position.w);
    float J = evalDirectReuseJacobian(srcLightType, cosLightNew, reconDist, restirDiGetCosLight(src.targetCosBits), srcDistHitT.x);

    // History M-truncation scales the stored sum by mTap/mSrc before re-weighting.
    float W_src = src.wSum / max(float(mSrc) * srcTargetPdf, 1e-6);
    float wFactor = W_src * J;
    if (clampWeight) {
        bool isAreaSrc = (srcLightType == 0u || srcLightType == 3u);
        bool manyLights = (numLights > 4096u);
        if (manyLights) {
            float J_capped = min(J, 6.0);
            wFactor = W_src * J_capped;
            float tapRad = pHat * wFactor;
            if (tapRad > 35.0 && pHat > 1e-8) {
                wFactor = 35.0 / pHat;
            }
        } else {
            float maxWSrc = isAreaSrc ? 100.0 : 250.0;
            float maxWFac = isAreaSrc ? 120.0 : 250.0;
            W_src = min(W_src, maxWSrc);
            wFactor = min(W_src * J, maxWFac);
        }
    }
    float weight = wFactor * float(mTap) * pHat;
    if (isnan(weight) || isinf(weight) || !(weight > 0.0)) {
        weight = 0.0;
    }

    float cachedSelectPdf = -1.0;
    if (weight > 0.0) {
        outEval.weightOverPHat = wFactor * float(mTap);
        outEval.tapSample.lightIdx = srcLightIdx;
        outEval.tapSample.dir = reconDir;
        outEval.tapSample.dist = reconDist;
        outEval.tapSample.emission = reconEmission;
        outEval.tapSample.selectPdf = cachedSelectPdf;
        outEval.tapSample.cosLight = cosLightNew;
        outEval.tapSample.ndotL = NdotL;
        outEval.tapSample.target = pHat;
        outEval.tapSample.luv = srcLuv;
        outEval.tapSample.isTrans = isTrans;
        outEval.tapSample.has = true;
    }

    r.wSum += weight;
    uint combinedM = min(restirGetM(r.idxM) + mTap, 255u);

    if (weight > 0.0 && (randFloat(seed) * r.wSum < weight || !sel.has)) {
        sel.lightIdx = srcLightIdx;
        sel.dir = reconDir;
        sel.dist = reconDist;
        sel.emission = reconEmission;
        sel.selectPdf = cachedSelectPdf;
        sel.cosLight = cosLightNew;
        sel.ndotL = NdotL;
        sel.target = pHat;
        sel.luv = srcLuv;
        sel.isTrans = isTrans;
        sel.has = true;

        r.idxM = restirPackIdxM(srcLightIdx, combinedM);
        r.targetCosBits = restirDiPackTargetCos(pHat, cosLightNew);
        // x1 context is re-stamped with the CURRENT surface: the reservoir
        // describes this pixel for the next frame's taps.
        r.flags = restirPackFlags(ageNew, true, isTrans, hitNormal);
        r.distHitT = restirDiPackDistHitT(reconDist, hitT);
        r.lightUv16 = packUnorm2x16(srcLuv);
        return true;
    } else {
        r.idxM = restirPackIdxM(restirGetLightIdx(r.idxM), combinedM);
        return false;
    }
}

bool restirDiApplyTap(
    inout DiReservoir r,
    inout DiSample sel,
    in DiReservoir src,
    uint numLights,
    float envIntensity,
    vec3 hitPoint,
    vec3 hitNormal,
    float hitT,
    float expectedPrevHitT,
    vec3 V,
    f16vec3 diffColor,
    f16vec3 F0,
    float alphaRoughness,
    float diffTrans,
    uint mCap,
    uint ageNew,
    bool isOccluded,
    bool clampWeight,
    inout uint seed
) {
    DiTapEval unusedEval;
    return restirDiApplyTapEx(
        r, sel, src, numLights, envIntensity, hitPoint, hitNormal, hitT, expectedPrevHitT,
        V, diffColor, F0, alphaRoughness, diffTrans,
        mCap, ageNew, isOccluded, clampWeight, seed, unusedEval
    );
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

// Per-pixel BRDF context at the primary hit x1 (16 bytes / 1 uvec4, matches X1ContextGPU).
struct X1Context {
    uvec4 data;
    // .x: n1Oct  - oct32(x1 world shading normal)
    // .y: alRG   - fp16 (baseColor.r, baseColor.g)
    // .z: alBM   - fp16 (baseColor.b, metallic)
    // .w: packed - [1:0] flags (valid, diRestirMis),
    //              [9:2] roughness (unorm8),
    //              [17:10] sqrt(f0Scalar) (unorm8; 0.04 -> 51/255 exact),
    //              [31:18] qSelf (positive fp32 bits [30:17])
};

const uint  GI_MAX_AGE          = 63u;
const uint  GI_INVALID_INST_20  = 0xFFFFFu;
const float GI_TAP_MIN_COS      = 0.05;
const float GI_TAP_MIN_DIST     = 1e-3;
const float GI_WEIGHT_CLAMP_MAX = 32.0;

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

const uint X1_FLAG_VALID         = 1u;
const uint X1_FLAG_DI_RESTIR_MIS = 2u;
const uint X1_FLAGS_MASK         = 3u;

uint packX1PackedW(uint flags, float f0Scalar, float roughness, float qSelf) {
    uint rQ = clamp(uint(roundEven(clamp(roughness, 0.0, 1.0) * 255.0)), 0u, 255u);
    uint fQ = clamp(uint(roundEven(sqrt(clamp(f0Scalar, 0.0, 1.0)) * 255.0)), 0u, 255u);
    uint qBits = (qSelf > 0.0) ? (((floatBitsToUint(qSelf) + (1u << 16u)) >> 17u) & 0x3FFFu) : 0u;
    return (flags & X1_FLAGS_MASK) | (rQ << 2u) | (fQ << 10u) | (qBits << 18u);
}

X1Context invalidX1Context() {
    X1Context c;
    c.data = uvec4(0u);
    return c;
}

X1Context makeDiMisOnlyX1Context() {
    X1Context c;
    c.data = uvec4(0u, 0u, 0u, X1_FLAG_DI_RESTIR_MIS);
    return c;
}

X1Context makeX1Context(vec3 n1, vec3 baseColor, float metallic, vec3 wo, float f0Scalar, float roughness, float qSelf, uint flags) {
    X1Context c;
    c.data.x = packOct32(normalize(n1));
    c.data.y = packHalf2x16(clamp(baseColor.rg, vec2(0.0), vec2(65000.0)));
    c.data.z = packHalf2x16(vec2(clamp(baseColor.b, 0.0, 65000.0), clamp(metallic, 0.0, 1.0)));
    c.data.w = packX1PackedW(flags, f0Scalar, roughness, max(qSelf, 1e-8));
    return c;
}

X1Context makePrepackedX1Context(vec3 n1, uint preY, uint preZ, uint preRF0Mask, float qSelf, uint flags) {
    X1Context c;
    float qEff = max(qSelf, 1e-8);
    uint qBits = ((floatBitsToUint(qEff) + (1u << 16u)) >> 17u) & 0x3FFFu;
    c.data.x = packOct32(normalize(n1));
    c.data.y = preY;
    c.data.z = preZ;
    c.data.w = (flags & X1_FLAGS_MASK) | (preRF0Mask & 0x3FFFCu) | (qBits << 18u);
    return c;
}

X1Context makeX1Context(vec3 n1, vec3 baseColor, float metallic, vec3 wo, float f0Scalar, float roughness, float qSelf) {
    return makeX1Context(n1, baseColor, metallic, wo, f0Scalar, roughness, qSelf, X1_FLAG_VALID);
}

bool  x1ContextValid(X1Context c)          { return (c.data.w & X1_FLAG_VALID) != 0u; }
bool  x1ContextHasDiRestirMis(X1Context c) { return (c.data.w & X1_FLAG_DI_RESTIR_MIS) != 0u; }
vec3  x1ContextNormal(X1Context c)         { return unpackOct32(c.data.x); }
vec3  x1ContextBaseColor(X1Context c) {
    vec2 rg = unpackHalf2x16(c.data.y);
    return vec3(rg.x, rg.y, unpackHalf2x16(c.data.z).x);
}
float x1ContextMetallic(X1Context c)  { return unpackHalf2x16(c.data.z).y; }
float x1ContextRoughness(X1Context c) { return float((c.data.w >> 2u) & 0xFFu) * (1.0 / 255.0); }
float x1ContextF0(X1Context c) {
    float s = float((c.data.w >> 10u) & 0xFFu) * (1.0 / 255.0);
    return s * s;
}
float x1ContextQSelf(X1Context c) {
    uint rawBits = (c.data.w & 0xFFFC0000u) >> 1u;
    return (rawBits != 0u) ? uintBitsToFloat(rawBits) : 0.0;
}

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

GiReservoir makeGiReservoirOct16(bool hasSurface, vec3 x2, vec3 n2, uint octN1, float wSum, uint M, float target, vec3 radiance, float dSrc, float cosSrc, uint age) {
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
    r.normalsPacked = octN2 | ((octN1 & 0xFFFFu) << 16u);
    uint dHalf = validSurf ? (packHalf2x16(vec2(clamp(dSrc, 0.0, 65000.0), 0.0)) & 0xFFFFu) : 0u;
    r.dSrcFlagsM = dHalf |
                   (1u << 16u) |
                   ((validSurf ? 1u : 0u) << 17u) |
                   ((min(age, GI_MAX_AGE) & 0x3Fu) << 18u) |
                   ((min(M, 255u) & 0xFFu) << 24u);
    return r;
}

GiReservoir makeGiReservoir(bool hasSurface, vec3 x2, vec3 n2, vec3 x1Normal, float wSum, uint M, float target, vec3 radiance, float dSrc, float cosSrc, uint age) {
    return makeGiReservoirOct16(hasSurface, x2, n2, restirPackOct16(x1Normal), wSum, M, target, radiance, dSrc, cosSrc, age);
}

// Planar SoA pack/unpack helpers for binding 38/39 (uvec2[4 * N] inside 32B/pixel allocation):
//   Plane 0 [0 .. N-1]:     GiMeta    (x = radRgb9e5, y = oct16(x1Normal) | flags | age | M)  [8 B/px]
//   Plane 1 [N .. 3N-1]:    GiPosN2   ([0] = (posX, posY), [1] = (posZ, oct16(n2) | fp16 dSrc)) [16 B/px]
//   Plane 2 [3N .. 4N-1]:   GiWeights (x = wSumBits, y = targetCosBits)                         [8 B/px]
bool  giMetaIsValid(uint metaY)     { return ((metaY >> 16u) & 1u) != 0u; }
bool  giMetaHasSurface(uint metaY)  { return ((metaY >> 17u) & 1u) != 0u; }
uint  giMetaGetAge(uint metaY)      { return (metaY >> 18u) & 0x3Fu; }
uint  giMetaGetM(uint metaY)        { return (metaY >> 24u) & 0xFFu; }
vec3  giMetaGetX1Normal(uint metaY) { return restirUnpackOct16(metaY & 0xFFFFu); }

uvec2 giPackEmptyMeta(uint octN1, uint M) {
    return uvec2(0u, (octN1 & 0xFFFFu) | (1u << 16u) | ((min(M, 255u) & 0xFFu) << 24u));
}

uvec2 giPackMeta(GiReservoir r) {
    return uvec2(r.radRgb9e5, (r.dSrcFlagsM & 0xFFFF0000u) | ((r.normalsPacked >> 16u) & 0xFFFFu));
}

uvec2 giPackPosN2_0(GiReservoir r) {
    return uvec2(r.posXBits, r.posYBits);
}

uvec2 giPackPosN2_1(GiReservoir r) {
    return uvec2(r.posZBits, (r.normalsPacked & 0xFFFFu) | ((r.dSrcFlagsM & 0xFFFFu) << 16u));
}

uvec2 giPackWeights(GiReservoir r) {
    return uvec2(r.wSumBits, r.targetCosBits);
}

GiReservoir giUnpackFromSoa(uvec2 meta, uvec2 p0, uvec2 p1, uvec2 w) {
    GiReservoir r;
    r.posXBits = p0.x;
    r.posYBits = p0.y;
    r.posZBits = p1.x;
    r.wSumBits = w.x;
    r.targetCosBits = w.y;
    r.radRgb9e5 = meta.x;
    r.normalsPacked = (p1.y & 0xFFFFu) | ((meta.y & 0xFFFFu) << 16u);
    r.dSrcFlagsM = (p1.y >> 16u) | (meta.y & 0xFFFF0000u);
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
