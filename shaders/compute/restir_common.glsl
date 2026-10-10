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
// Reservoir layout: 7 × 4 B = 28 bytes, matches DiReservoirGPU in ReSTIRManager.hpp
// -----------------------------------------------------------------------------
struct DiReservoir {
    uint  idxM;        // [15:0] light index, [31:16] M
    float wSum;        // RIS weight sum
    float targetPdf;   // p̂ of the selected sample at the storing pixel
    uint  flags;       // [7:0] age, [8] valid, [9] transmission lobe, [25:10] oct16(x1 normal)
    float hitDist;     // x1 -> light-sample distance (1e4 for directional lights)
    uint  lightPacked; // [15:0] oct16(light normal), [23:16] uv.u8, [31:24] uv.v8
    uint  x1Misc;      // [15:0] fp16 cos(theta at light, stored config), [31:16] fp16 primary hit distance
};

// Tunables / bias gates
const uint  RESTIR_MAX_AGE              = 30u;    // temporal truncation: taps at least this old are rejected
const float RESTIR_TAP_NORMAL_MIN_DOT   = 0.70f;  // disocclusion guard on stored x1 normal
const float RESTIR_TAP_DEPTH_REL_GATE   = 0.15f;  // disocclusion guard on primary hit distance
const float RESTIR_JACOBIAN_CLAMP_MIN   = 0.05f;
const float RESTIR_JACOBIAN_CLAMP_MAX   = 20.0f;

const ivec2 RESTIR_TAP_OFFSETS[4] = ivec2[4](ivec2( 2, 0), ivec2(0,  2), ivec2(-2, 0), ivec2(0, -2));

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
    return r;
}

// The RIS stage functions below require the `lights` and `lightNodes` storage
// buffers to be declared before this include. Passes that only need the
// reservoir type for invalidation writes (dielectric / emissive / passthrough /
// conductor) skip them by not defining RESTIR_INCLUDE_SAMPLER_HELPERS.
#ifdef RESTIR_INCLUDE_SAMPLER_HELPERS

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
    if (uint(light.position.w) == 0u /* AREA QUAD */) {
        vec3 lightPoint = light.position.xyz + luv.x * light.u.xyz + luv.y * light.v.xyz;
        vec3 toLight = lightPoint - hitPoint;
        lightDist = length(toLight);
        if (lightDist < 1e-4) return false;
        lightDir = toLight / lightDist;
        cosLight = dot(-lightDir, light.normal.xyz);
        if (cosLight <= 0.0) return false;
        return true;
    } else if (uint(light.position.w) == 1u /* SPOT */) {
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

        if (uint(light.position.w) == 0u /* AREA QUAD */) {
            luv = randVec2(seed);
            vec3 lightPoint = light.position.xyz + luv.x * light.u.xyz + luv.y * light.v.xyz;
            vec3 toLight = lightPoint - hitPoint;
            lightDist = length(toLight);
            lightDir = toLight / max(lightDist, 1e-4);

            cosLight = dot(-lightDir, light.normal.xyz);
            if (cosLight > 0.0) {
                float area = light.emission.w;
                lightPdf = (lightDist * lightDist) / (area * cosLight);
                validLightSample = true;
            }
        } else if (uint(light.position.w) == 1u /* SPOT */) {
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
            float sourcePdf = max(lightPdf * lightSelectPdf, 1e-6);
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
        r.lightPacked = restirPackOct16(lights[sel.lightIdx].normal.xyz) | (restirPackUv8(sel.luv) << 16u);
        r.flags = restirPackFlags(0u, true, sel.isTrans, hitNormal);
        r.x1Misc = restirPackF16Pair(sel.cosLight, min(hitT, 60000.0));
    } else {
        r = invalidDiReservoir();
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
    inout uint seed
) {
    if (!restirIsValid(src.flags) || src.targetPdf <= 0.0 || src.wSum <= 0.0) return;
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

    Light srcLight = lights[srcLightIdx];
    vec2 srcLuv = restirUnpackUv8(src.lightPacked >> 16u);
    vec3 reconDir;
    float reconDist;
    vec3 reconEmission;
    float cosLightNew;
    if (!reconnectDirectPath(hitPoint, srcLight, srcLuv, reconDir, reconDist, reconEmission, cosLightNew)) return;

    float rawNdotL = dot(hitNormal, reconDir);
    bool isTrans = (rawNdotL < 0.0 && diffTrans > 0.001);
    float NdotL = isTrans ? -rawNdotL : clamp(rawNdotL, 0.0, 1.0);
    if (NdotL <= 0.0 || !(rawNdotL > 0.0 || isTrans)) return;

    float transScale = isTrans ? diffTrans : (1.0 - diffTrans);
    float pHat = max(dot(reconEmission, vec3(proxyLumWeight)) * (NdotL * transScale), 1e-6);

    float J = evalDirectReuseJacobian(uint(srcLight.position.w), cosLightNew, reconDist, srcMisc.x, src.hitDist);

    uint mTap = min(mSrc, mCap);
    // History M-truncation scales the stored sum by mTap/mSrc before re-weighting.
    float weight = (src.wSum / float(mSrc)) * float(mTap) * (pHat / max(src.targetPdf, 1e-6)) * J;

    r.wSum += weight;
    uint combinedM = restirGetM(r.idxM) + mTap;

    if (weight > 0.0 && (randFloat(seed) * r.wSum < weight || !sel.has)) {
        sel.lightIdx = srcLightIdx;
        sel.dir = reconDir;
        sel.dist = reconDist;
        sel.emission = reconEmission;
        sel.selectPdf = 1.0;
        sel.cosLight = cosLightNew;
        sel.ndotL = NdotL;
        sel.target = pHat;
        sel.luv = srcLuv;
        sel.isTrans = isTrans;
        sel.has = true;

        r.idxM = restirPackIdxM(srcLightIdx, combinedM);
        r.targetPdf = pHat;
        r.hitDist = reconDist;
        r.lightPacked = restirPackOct16(srcLight.normal.xyz) | (restirPackUv8(srcLuv) << 16u);
        // x1 context is re-stamped with the CURRENT surface: the reservoir
        // describes this pixel for the next frame's taps.
        r.flags = restirPackFlags(ageNew, true, isTrans, hitNormal);
        r.x1Misc = restirPackF16Pair(cosLightNew, min(hitT, 60000.0));
    } else {
        r.idxM = restirPackIdxM(restirGetLightIdx(r.idxM), combinedM);
    }
}

#endif // RESTIR_INCLUDE_SAMPLER_HELPERS

#endif // RESTIR_COMMON_GLSL
