#ifndef UPWAYS_COMMON_V3_GLSL
#define UPWAYS_COMMON_V3_GLSL

// Upways 3.0 Ultra-Low-Latency Pure Vulkan 1.4 Neural Reconstructor
// Target Architecture: AMD RDNA 4 (gfx1201 / Wave32 WMMA v_wmma_f32_16x16x16_f16)

const float UPWAYS_MU = 16.0;
const float UPWAYS_LOG_DENOM = 2.833213344056216; // ln(1.0 + 16.0)

// Invertible log-space radiance compression: phi(x) = ln(1 + mu * x) / ln(1 + mu)
vec3 upwaysLogTransform(vec3 x) {
    vec3 clampedX = max(x, vec3(0.0));
    return log(vec3(1.0) + UPWAYS_MU * clampedX) / UPWAYS_LOG_DENOM;
}

float16_t upwaysLogTransformF16(float16_t x) {
    float16_t clampedX = max(x, float16_t(0.0));
    return log(float16_t(1.0) + float16_t(UPWAYS_MU) * clampedX) / float16_t(UPWAYS_LOG_DENOM);
}

// Exact inverse log-space decompression: phi^-1(y) = ((1 + mu)^y - 1) / mu
// Clamped at 3.8 in FP16 to strictly prevent 16-bit half-precision exponential overflow (17^3.8 = 48,707 < 65,504)
vec3 upwaysInverseLogTransform(vec3 y) {
    vec3 clampedY = clamp(y, vec3(0.0), vec3(31.0));
    return (pow(vec3(1.0 + UPWAYS_MU), clampedY) - vec3(1.0)) / UPWAYS_MU;
}

float16_t upwaysInverseLogTransformF16(float16_t y) {
    float16_t clampedY = clamp(y, float16_t(0.0), float16_t(3.8));
    return (pow(float16_t(1.0 + UPWAYS_MU), clampedY) - float16_t(1.0)) / float16_t(UPWAYS_MU);
}

// Fast approximate GELU activation: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
float16_t upwaysGelu(float16_t x) {
    const float16_t c1 = float16_t(0.79788456); // sqrt(2/pi)
    const float16_t c2 = float16_t(0.044715);
    float16_t x3 = x * x * x;
    return float16_t(0.5) * x * (float16_t(1.0) + tanh(c1 * (x + c2 * x3)));
}

// Normalized sample density for variable SPP conditioning: S_norm = 1.0 / sqrt(max(SPP, 1.0))
float upwaysComputeSnorm(float spp) {
    return 1.0 / sqrt(max(spp, 1.0));
}

float16_t upwaysComputeSnormF16(float16_t spp) {
    return float16_t(1.0) / sqrt(max(spp, float16_t(1.0)));
}

// Temperature parameter for convex KPN softmax: tau = clamp(S_norm, 0.08, 1.0)
float16_t upwaysComputeTau(float16_t s_norm) {
    return clamp(s_norm, float16_t(0.08), float16_t(1.0));
}

// Dielectric and Conductor Fresnel reflectance calculation
vec3 upwaysComputeF0(vec3 albedo, float roughness, float metallic) {
    vec3 dielectricF0 = vec3(0.04);
    return max(mix(dielectricF0, albedo, metallic), dielectricF0);
}

#endif // UPWAYS_COMMON_V3_GLSL
