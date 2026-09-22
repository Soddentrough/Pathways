#ifndef RESTIR_COMMON_GLSL
#define RESTIR_COMMON_GLSL

#include "wavefront_common.glsl"

// Unified 32-byte ReSTIR PT Reservoir layout (Direct + 1-Bounce Indirect Path Resampling)
struct UnifiedReservoirPT {
    uint lightIndex_M;             // lower 16: lightIndex / candidate ID, upper 16: M sample count
    float wSum;                    // sum of candidate weights / final evaluation weight W
    uint flags_uv_age;             // [0..7] age, [8] valid, [9..10] pathLength (1=DI, 2=GI), [11] lobe, [16..31] packed UV/oct
    float targetPdf;               // scalar unshadowed target distribution p_hat
    float secondaryHitDist;        // distance from x0 to secondary vertex x1
    uint secondaryHitNormal;       // 32-bit octahedral packed normal of vertex x1
    uint secondaryRadianceRG;      // packed FP16 R and G radiance arriving from x1
    uint secondaryRadianceB_flags; // packed FP16 B radiance and pad
};

// Aliasing for compatibility across pipelines
#define ReservoirDI UnifiedReservoirPT
#define age_flags flags_uv_age

uint packLightM(uint lightIdx, uint M) {
    return (lightIdx & 0xFFFFu) | ((M & 0xFFFFu) << 16u);
}

uint getLightIndex(uint packed) {
    return packed & 0xFFFFu;
}

uint getM(uint packed) {
    return (packed >> 16u) & 0xFFFFu;
}

uint packUnifiedFlags(uint age, uint valid, uint pathLength, uint lobe, vec2 uv) {
    uint u8 = uint(clamp(uv.x * 255.0, 0.0, 255.0));
    uint v8 = uint(clamp(uv.y * 255.0, 0.0, 255.0));
    return (age & 0xFFu) |
           ((valid & 1u) << 8u) |
           ((pathLength & 3u) << 9u) |
           ((lobe & 1u) << 11u) |
           (u8 << 16u) |
           (v8 << 24u);
}

uint packAgeFlagsUV(uint age, uint valid, vec2 uv) {
    return packUnifiedFlags(age, valid, 1u /* DI */, 0u, uv);
}

uint packAgeFlags(uint age, uint valid) {
    return (age & 0xFFu) | ((valid & 1u) << 8u);
}

uint getPathLength(uint packed) {
    return (packed >> 9u) & 3u;
}

uint getAge(uint packed) {
    return packed & 0xFFu;
}

bool isValidReservoir(uint packed) {
    return ((packed >> 8u) & 1u) != 0u;
}

vec2 unpackUV(uint packed) {
    float u = float((packed >> 16u) & 0xFFu) / 255.0;
    float v = float((packed >> 24u) & 0xFFu) / 255.0;
    return vec2(u, v);
}

uint packOct16(vec3 v) {
    vec2 enc = octEncode(normalize(v)) * 0.5 + 0.5;
    uint u8 = uint(clamp(enc.x * 255.0, 0.0, 255.0));
    uint v8 = uint(clamp(enc.y * 255.0, 0.0, 255.0));
    return (u8 & 0xFFu) | ((v8 & 0xFFu) << 8u);
}

vec3 unpackOct16(uint p) {
    float u = float(p & 0xFFu) / 255.0;
    float v = float((p >> 8u) & 0xFFu) / 255.0;
    vec2 enc = vec2(u, v) * 2.0 - 1.0;
    return octDecode(enc);
}

uint packRadianceRG(f16vec2 rg) {
    return packHalf2x16(vec2(rg));
}

f16vec2 unpackRadianceRG(uint packed) {
    return f16vec2(unpackHalf2x16(packed));
}

uint packRadianceB(float16_t b, uint extraFlags) {
    return packHalf2x16(vec2(float(b), 0.0)) | (extraFlags << 16u);
}

float16_t unpackRadianceB(uint packed) {
    return float16_t(unpackHalf2x16(packed).x);
}

// Evaluate scalar unshadowed target distribution p_hat for light y at surface point x
float evalUnshadowedTargetDI(vec3 hitPoint, vec3 hitNormal, f16vec3 diffuseAlbedo, Light light, vec2 luv) {
    vec3 lightPoint = light.position.xyz;
    vec3 toLight;
    float lightDist = 0.0;
    float cosLight = 1.0;
    vec3 emission = light.emission.rgb;
    
    uint lightType = uint(light.position.w);
    if (lightType == 0u /* AREA QUAD */) {
        lightPoint = light.position.xyz + luv.x * light.u.xyz + luv.y * light.v.xyz;
        toLight = lightPoint - hitPoint;
        lightDist = length(toLight);
        vec3 lightDir = toLight / max(lightDist, 1e-4);
        cosLight = max(dot(-lightDir, light.normal.xyz), 0.0);
        if (cosLight <= 0.0) return 0.0;
    } else if (lightType == 1u /* SPOT */) {
        toLight = lightPoint - hitPoint;
        lightDist = length(toLight);
        vec3 lightDir = toLight / max(lightDist, 1e-4);
        float cosSpot = dot(-lightDir, light.normal.xyz);
        if (cosSpot < light.v.w) return 0.0;
        float spotFactor = clamp((cosSpot - light.v.w) / max(light.u.w - light.v.w, 1e-4), 0.0, 1.0);
        emission *= spotFactor;
    } else if (lightType == 2u /* DIRECTIONAL */) {
        vec3 lightDir = normalize(light.normal.xyz);
        float cosTheta = max(dot(hitNormal, lightDir), 0.0);
        float lum = dot(emission, vec3(0.2126, 0.7152, 0.0722));
        float albedoWeight = max(float(dot(diffuseAlbedo, f16vec3(0.3333))), 0.01);
        return lum * cosTheta * albedoWeight;
    }
    
    float distSq = max(lightDist * lightDist, 1e-4);
    vec3 lightDir = toLight / max(lightDist, 1e-4);
    float cosTheta = max(dot(hitNormal, lightDir), 0.0);
    if (cosTheta <= 0.0) return 0.0;
    
    float lum = dot(emission, vec3(0.2126, 0.7152, 0.0722));
    float albedoWeight = max(float(dot(diffuseAlbedo, f16vec3(0.3333))), 0.01);
    float area = (lightType == 0u) ? max(light.emission.w, 1e-4) : 1.0;
    return (lum * cosTheta * cosLight * albedoWeight * area) / distSq;
}

// Evaluate scalar unshadowed target distribution p_hat for secondary hit vertex x1 at primary hit x0
float evalUnshadowedTargetGI(vec3 hitPoint, vec3 hitNormal, f16vec3 diffuseAlbedo,
                            vec3 secHitPoint, vec3 secHitNormal, vec3 secRadiance) {
    vec3 toSec = secHitPoint - hitPoint;
    float dist = length(toSec);
    if (dist < 1e-4) return 0.0;
    vec3 dir = toSec / dist;
    float cosTheta0 = max(dot(hitNormal, dir), 0.0);
    if (cosTheta0 <= 0.0) return 0.0;
    float cosTheta1 = max(dot(secHitNormal, -dir), 0.0);
    if (cosTheta1 <= 0.0) return 0.0;
    float lum = dot(secRadiance, vec3(0.2126, 0.7152, 0.0722));
    if (lum <= 1e-6) return 0.0;
    float albedoWeight = max(float(dot(diffuseAlbedo, f16vec3(0.3333))), 0.01);

    // Optical infinity / Sky dome has no 1/r^2 geometric distance decay
    if (dist >= 5000.0) {
        return lum * cosTheta0 * albedoWeight;
    }

    float distSq = max(dist * dist, 1e-4);
    return (lum * cosTheta0 * cosTheta1 * albedoWeight) / distSq;
}

// Jacobian determinant for secondary path reconnection:
// J = (||x1 - x_orig||^2 / ||x1 - x_new||^2) * (cos(theta1_new) / cos(theta1_orig))
float evalGIJacobian(vec3 x_new, vec3 x_orig, vec3 x1, vec3 n1) {
    vec3 to_orig = x1 - x_orig;
    vec3 to_new = x1 - x_new;
    float dSq_orig = dot(to_orig, to_orig);
    float dSq_new = dot(to_new, to_new);
    if (dSq_orig >= 2.5e7 || dSq_new >= 2.5e7) return 1.0;
    float d_orig = sqrt(max(dSq_orig, 1e-4));
    float d_new = sqrt(max(dSq_new, 1e-4));
    float cos1_orig = abs(dot(n1, -to_orig / d_orig));
    float cos1_new  = abs(dot(n1, -to_new / d_new));
    float J = (dSq_orig / max(dSq_new, 1e-4)) * (cos1_new / max(cos1_orig, 1e-3));
    return clamp(J, 0.05, 10.0);
}

// Footprint-based reconnection criterion (ReSTIR PT Enhanced 2026)
bool validateReconnectionFootprint(vec3 pPos, vec3 pNorm, float pRough,
                                   vec3 qPos, vec3 qNorm, float qRough,
                                   vec3 secPos) {
    if (dot(pNorm, qNorm) < 0.75) return false;
    float pDist = length(secPos - pPos);
    float qDist = length(secPos - qPos);
    float distRatio = pDist / max(qDist, 1e-3);
    if (distRatio < 0.2 || distRatio > 5.0) return false;
    float minRough = min(pRough, qRough);
    if (minRough < 0.15) {
        vec3 dirP = normalize(secPos - pPos);
        vec3 dirQ = normalize(secPos - qPos);
        if (dot(dirP, dirQ) < (1.0 - minRough * minRough * 2.0)) return false;
    }
    return true;
}

// Chao's streaming algorithm for initial DI candidate stream
void updateReservoirDI(inout UnifiedReservoirPT r, uint lightIdx, float weight, float targetPdf, vec2 luv, inout uint seed) {
    r.wSum += weight;
    uint currentM = getM(r.lightIndex_M) + 1u;
    if (weight > 0.0 && (randFloat(seed) * r.wSum < weight || !isValidReservoir(r.flags_uv_age))) {
        r.lightIndex_M = packLightM(lightIdx, currentM);
        r.targetPdf = targetPdf;
        r.flags_uv_age = packUnifiedFlags(0u, 1u, 1u /* DI */, 0u, luv);
        r.secondaryHitDist = 0.0;
        r.secondaryHitNormal = 0u;
        r.secondaryRadianceRG = 0u;
        r.secondaryRadianceB_flags = 0u;
    } else {
        r.lightIndex_M = packLightM(getLightIndex(r.lightIndex_M), currentM);
    }
}

// Update unified reservoir with indirect path candidate (k = 2)
void updateReservoirGI(inout UnifiedReservoirPT r, uint pathSeed, float weight, float targetPdf,
                       vec3 secDir, float secDist, vec3 secNormal, vec3 secRadiance, inout uint seed) {
    r.wSum += weight;
    uint currentM = getM(r.lightIndex_M) + 1u;
    if (weight > 0.0 && (randFloat(seed) * r.wSum < weight || !isValidReservoir(r.flags_uv_age))) {
        r.lightIndex_M = packLightM(pathSeed, currentM);
        r.targetPdf = targetPdf;
        uint oct16 = packOct16(secDir);
        r.flags_uv_age = (0u & 0xFFu) | ((1u & 1u) << 8u) | ((2u & 3u) << 9u) | (oct16 << 16u);
        r.secondaryHitDist = secDist;
        r.secondaryHitNormal = packOct32(secNormal);
        r.secondaryRadianceRG = packRadianceRG(f16vec2(secRadiance.rg));
        r.secondaryRadianceB_flags = packRadianceB(float16_t(secRadiance.b), 0u);
    } else {
        r.lightIndex_M = packLightM(getLightIndex(r.lightIndex_M), currentM);
    }
}

// Combine an incoming reservoir rB into accumulator rA
void combineReservoirsDI(inout UnifiedReservoirPT rA, in UnifiedReservoirPT rB, float p_hat_B_at_A, inout uint seed) {
    if (!isValidReservoir(rB.flags_uv_age) || rB.targetPdf <= 0.0 || p_hat_B_at_A <= 0.0) return;
    float wB = rB.wSum * (p_hat_B_at_A / max(rB.targetPdf, 1e-6));
    rA.wSum += wB;
    uint combinedM = getM(rA.lightIndex_M) + getM(rB.lightIndex_M);
    if (!isValidReservoir(rA.flags_uv_age) || randFloat(seed) * rA.wSum < wB) {
        rA.lightIndex_M = packLightM(getLightIndex(rB.lightIndex_M), combinedM);
        rA.targetPdf = p_hat_B_at_A;
        uint ageB = getAge(rB.flags_uv_age);
        uint pLenB = getPathLength(rB.flags_uv_age);
        uint lobeB = (rB.flags_uv_age >> 11u) & 1u;
        uint high16 = (rB.flags_uv_age >> 16u) & 0xFFFFu;
        rA.flags_uv_age = (ageB & 0xFFu) | ((1u & 1u) << 8u) | ((pLenB & 3u) << 9u) | ((lobeB & 1u) << 11u) | (high16 << 16u);
        rA.secondaryHitDist = rB.secondaryHitDist;
        rA.secondaryHitNormal = rB.secondaryHitNormal;
        rA.secondaryRadianceRG = rB.secondaryRadianceRG;
        rA.secondaryRadianceB_flags = rB.secondaryRadianceB_flags;
    } else {
        rA.lightIndex_M = packLightM(getLightIndex(rA.lightIndex_M), combinedM);
    }
}

#define combineReservoirsPT combineReservoirsDI

// Compute the unbiased Monte Carlo evaluation weight W = wSum / (M * p_hat)
float computeUnbiasedWeightDI(in UnifiedReservoirPT r) {
    uint M = getM(r.lightIndex_M);
    if (M == 0u || r.targetPdf <= 0.0 || !isValidReservoir(r.flags_uv_age)) return 0.0;
    float W = r.wSum / (float(M) * r.targetPdf);
    return clamp(W, 0.0, 10000.0);
}

#endif // RESTIR_COMMON_GLSL
