#ifndef UPWAYS_COMMON_GLSL
#define UPWAYS_COMMON_GLSL

// Shared constants and physical helper functions for Upways Vulkan 1.4 shaders.
// Synchronized 1:1 with Pathways/shaders/compute/denoise_common.glsl and upways/model.py.

const vec3 UPWAYS_LUMA_WEIGHTS = vec3(0.2126, 0.7152, 0.0722);
const float UPWAYS_SKY_DEPTH = 9999.0;
const float UPWAYS_EMISSIVE_ALBEDO_SQ_EPS = 1e-5;
const float UPWAYS_MIN_DEMOD_ALBEDO = 0.01;
const float UPWAYS_MAX_RADIANCE = 65504.0;

float upwaysLuminance(vec3 c) {
    return max(dot(c, UPWAYS_LUMA_WEIGHTS), 0.0);
}

float sanitizeFloat(float v) {
    if (isnan(v) || isinf(v)) {
        return 0.0;
    }
    return clamp(v, 0.0, UPWAYS_MAX_RADIANCE);
}

vec3 sanitizeVec3(vec3 v) {
    return vec3(sanitizeFloat(v.r), sanitizeFloat(v.g), sanitizeFloat(v.b));
}

// Matches denoiseDemodulationFactor in Pathways/shaders/compute/denoise_common.glsl:16-21
vec3 denoiseDemodulationFactor(vec3 albedo, float linearDepth) {
    if (linearDepth >= UPWAYS_SKY_DEPTH || dot(albedo, albedo) <= UPWAYS_EMISSIVE_ALBEDO_SQ_EPS) {
        return vec3(1.0);
    }
    return max(albedo, vec3(UPWAYS_MIN_DEMOD_ALBEDO));
}

vec3 demodulateDiffuse(vec3 diffuseRad, vec3 albedo, float linearDepth) {
    vec3 factor = denoiseDemodulationFactor(albedo, linearDepth);
    return sanitizeVec3(diffuseRad) / factor;
}

vec3 remodulateDiffuse(vec3 demodIrradiance, vec3 albedo, float linearDepth) {
    vec3 factor = denoiseDemodulationFactor(albedo, linearDepth);
    return sanitizeVec3(demodIrradiance * factor);
}

// Co-designed Metallic-Fresnel Specular Demodulation/Remodulation factor matching upways/data/protocol.py
vec3 denoiseSpecularModulationFactor(
    vec3 albedo,
    float metallic,
    float roughness,
    float nViewZ,
    float linearDepth
) {
    if (linearDepth >= UPWAYS_SKY_DEPTH || metallic <= 1e-4 || dot(albedo, albedo) <= UPWAYS_EMISSIVE_ALBEDO_SQ_EPS) {
        return vec3(1.0);
    }
    vec3 f0Metal = max(clamp(albedo, vec3(0.0), vec3(1.0)), vec3(0.04));
    float cosTheta = clamp(abs(nViewZ), 0.0, 1.0);
    float oneMinusCos = 1.0 - cosTheta;
    float oneMinusCos2 = oneMinusCos * oneMinusCos;
    float fresnel = oneMinusCos2 * oneMinusCos2 * oneMinusCos * (1.0 - clamp(roughness, 0.0, 1.0));
    vec3 metalMod = mix(f0Metal, vec3(1.0), 0.25 * fresnel);
    return mix(vec3(1.0), metalMod, clamp(metallic, 0.0, 1.0));
}

vec3 demodulateSpecular(
    vec3 specularRad,
    vec3 albedo,
    float metallic,
    float roughness,
    float nViewZ,
    float linearDepth
) {
    vec3 factor = denoiseSpecularModulationFactor(albedo, metallic, roughness, nViewZ, linearDepth);
    return sanitizeVec3(specularRad) / factor;
}

vec3 remodulateSpecular(
    vec3 demodSpecular,
    vec3 albedo,
    float metallic,
    float roughness,
    float nViewZ,
    float linearDepth
) {
    vec3 factor = denoiseSpecularModulationFactor(albedo, metallic, roughness, nViewZ, linearDepth);
    return sanitizeVec3(demodSpecular * factor);
}

// Physical Schlick-GGX view-dependent Fresnel reflectance luminance for neural feature channel 14
float evaluateFresnelLuminance(
    vec3 albedo,
    float metallic,
    float roughness,
    float nViewZ,
    float linearDepth
) {
    if (linearDepth >= UPWAYS_SKY_DEPTH) {
        return 0.0;
    }
    vec3 f0 = mix(vec3(0.04), max(clamp(albedo, vec3(0.0), vec3(1.0)), vec3(0.04)), clamp(metallic, 0.0, 1.0));
    float cosTheta = clamp(abs(nViewZ), 0.0, 1.0);
    float u = 1.0 - cosTheta;
    float u2 = u * u;
    float u5 = u2 * u2 * u;
    vec3 fMax = max(vec3(1.0 - clamp(roughness, 0.0, 1.0)), f0);
    vec3 fEnv = f0 + (fMax - f0) * u5;
    return clamp(upwaysLuminance(fEnv), 0.0, 1.0);
}

float karisWeight(vec3 rad) {
    return 1.0 / (1.0 + upwaysLuminance(rad));
}

float evaluateNormalWeight(vec3 nCenter, vec3 nTap, float phiNormal) {
    float cosTheta = clamp(dot(nCenter, nTap), 0.0, 1.0);
    return pow(cosTheta, max(phiNormal, 1.0));
}

float evaluateNormalWeight16(vec3 nCenter, vec3 nTap) {
    float c  = clamp(dot(nCenter, nTap), 0.0, 1.0);
    float c2 = c * c;
    float c4 = c2 * c2;
    float c8 = c4 * c4;
    return c8 * c8;
}

float evaluateNormalWeight24(vec3 nCenter, vec3 nTap) {
    float c  = clamp(dot(nCenter, nTap), 0.0, 1.0);
    float c2 = c * c;
    float c4 = c2 * c2;
    float c8 = c4 * c4;
    return (c8 * c8) * c8;
}

float evaluateNormalWeight32(vec3 nCenter, vec3 nTap) {
    float c16 = evaluateNormalWeight16(nCenter, nTap);
    return c16 * c16;
}

// Metric tangent-plane depth weight matching Pathways evaluatePlaneDepthWeight
float evaluatePlaneDepthWeight(
    float zCenter,
    float zTap,
    vec2 dzdxy,
    vec2 offsetPx,
    float phiDepth
) {
    float zExpected = zCenter + dot(dzdxy, offsetPx);
    float absDiff = abs(zTap - zExpected);
    float denom = max(phiDepth * max(zCenter, 0.1), 1e-4);
    return exp(-clamp(absDiff / denom, 0.0, 20.0));
}

float evaluateRelativeDepthWeight(float zCenter, float zTap, float phiDepth) {
    float relDiff = abs(zCenter - zTap) / max(max(zCenter, zTap), 0.1);
    return exp(-clamp(relDiff / max(phiDepth, 1e-4), 0.0, 20.0));
}

// Soft logarithmic firefly rolloff above threshold T while preserving RGB chromaticity
vec3 softClampLuminance(vec3 rad, float maxLum) {
    vec3 clean = sanitizeVec3(rad);
    float lum = upwaysLuminance(clean);
    if (lum <= maxLum || lum <= 1e-6) {
        return clean;
    }
    float compressedLum = maxLum + log(1.0 + (lum - maxLum));
    return clean * (compressedLum / lum);
}

#endif // UPWAYS_COMMON_GLSL
