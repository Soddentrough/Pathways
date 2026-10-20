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
    return (isnan(v) || isinf(v)) ? 0.0 : clamp(v, 0.0, UPWAYS_MAX_RADIANCE);
}

vec3 sanitizeVec3(vec3 v) {
    return vec3(sanitizeFloat(v.r), sanitizeFloat(v.g), sanitizeFloat(v.b));
}

// Matches denoiseDemodulationFactor in Pathways/shaders/compute/denoise_common.glsl:16-21
vec3 denoiseDemodulationFactor(vec3 albedo, float linearDepth) {
    bool isIdentity = (linearDepth >= UPWAYS_SKY_DEPTH || dot(albedo, albedo) <= UPWAYS_EMISSIVE_ALBEDO_SQ_EPS);
    return isIdentity ? vec3(1.0) : max(albedo, vec3(UPWAYS_MIN_DEMOD_ALBEDO));
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

// Packs log1p irradiance in [0, 11.09035489] into 32-bit 11-11-10 using FP16(v + 1.0) - 0x3C00u.
// Since v + 1.0 is in [1.0, 12.09035489], its FP16 biased exponent is in 15..18 (2 bits), so
// (h - 0x3C00u) is a 12-bit integer preserving all 10 FP16 mantissa bits. Zero-initialized 0u unpacks to vec3(0.0).
uint packIrr11_11_10(vec3 v) {
    vec3 vp1 = clamp(v, vec3(0.0), vec3(11.09035489)) + vec3(1.0);
    uint hr_hg = packHalf2x16(vp1.rg);
    uint hb    = packHalf2x16(vec2(vp1.b, 0.0));
    uint r12 = clamp((hr_hg & 0xFFFFu) - 0x3C00u, 0u, 4095u);
    uint g12 = clamp((hr_hg >> 16u)    - 0x3C00u, 0u, 4095u);
    uint b12 = clamp((hb    & 0xFFFFu) - 0x3C00u, 0u, 4095u);
    uint r11 = (r12 + 1u) >> 1u;
    uint g11 = (g12 + 1u) >> 1u;
    uint b10 = (b12 + 2u) >> 2u;
    return (r11 & 0x7FFu) | ((g11 & 0x7FFu) << 11u) | ((b10 & 0x3FFu) << 22u);
}

vec3 unpackIrr11_11_10(uint p) {
    uint r12 = (p & 0x7FFu) << 1u;
    uint g12 = ((p >> 11u) & 0x7FFu) << 1u;
    uint b12 = ((p >> 22u) & 0x3FFu) << 2u;
    uint hr_hg = (r12 + 0x3C00u) | ((g12 + 0x3C00u) << 16u);
    uint hb    = (b12 + 0x3C00u);
    vec2 rg = unpackHalf2x16(hr_hg) - vec2(1.0);
    float b = unpackHalf2x16(hb).x - 1.0;
    return max(vec3(f16vec3(vec3(rg, b))), vec3(0.0));
}

uint packGuide11_11_10(vec3 g) {
    vec3 c = clamp(g, vec3(0.0), vec3(1.0));
    uint r  = uint(round(c.r * 2047.0));
    uint gr = uint(round(c.g * 2047.0));
    uint b  = uint(round(c.b * 1023.0));
    return (r & 0x7FFu) | ((gr & 0x7FFu) << 11u) | ((b & 0x3FFu) << 22u);
}

vec3 unpackGuide11_11_10(uint p) {
    float r = float(p & 0x7FFu) * (1.0 / 2047.0);
    float g = float((p >> 11u) & 0x7FFu) * (1.0 / 2047.0);
    float b = float((p >> 22u) & 0x3FFu) * (1.0 / 1023.0);
    return vec3(f16vec3(vec3(r, g, b)));
}

uint packSpecGuide29(vec3 g, bool isSurf, bool hasCustomSpecMv) {
    vec3 c = clamp(g, vec3(0.0), vec3(1.0));
    uint r  = uint(round(c.r * 1023.0)) & 0x3FFu;
    uint gr = uint(round(c.g * 1023.0)) & 0x3FFu;
    uint b  = uint(round(c.b * 511.0))  & 0x1FFu;
    uint flags = (isSurf ? (1u << 29u) : 0u)
               | (hasCustomSpecMv ? (1u << 31u) : 0u);
    return r | (gr << 10u) | (b << 20u) | flags;
}

vec3 unpackSpecGuide29(uint p, out float isSurf, out bool hasCustomSpecMv) {
    float r = float(p & 0x3FFu) * (1.0 / 1023.0);
    float g = float((p >> 10u) & 0x3FFu) * (1.0 / 1023.0);
    float b = float((p >> 20u) & 0x1FFu) * (1.0 / 511.0);
    isSurf          = ((p & (1u << 29u)) != 0u) ? 1.0 : 0.0;
    hasCustomSpecMv = ((p & (1u << 31u)) != 0u);
    return vec3(f16vec3(vec3(r, g, b)));
}

#endif // UPWAYS_COMMON_GLSL
