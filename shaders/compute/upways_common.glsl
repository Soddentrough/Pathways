#ifndef UPWAYS_COMMON_GLSL
#define UPWAYS_COMMON_GLSL

// Upways Neural Denoiser & Continuous Super-Resolution Architecture
// Target: AMD RDNA 4 (gfx1201 / Wave32 WMMA), Vulkan 1.4

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
vec3 upwaysInverseLogTransform(vec3 y) {
    vec3 clampedY = clamp(y, vec3(0.0), vec3(6.0));
    return (pow(vec3(1.0 + UPWAYS_MU), clampedY) - vec3(1.0)) / UPWAYS_MU;
}

float16_t upwaysInverseLogTransformF16(float16_t y) {
    float16_t clampedY = clamp(y, float16_t(0.0), float16_t(4.89));
    return (pow(float16_t(1.0 + UPWAYS_MU), clampedY) - float16_t(1.0)) / float16_t(UPWAYS_MU);
}

// Fast approximate GELU activation: 0.5 * x * (1.0 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
float16_t upwaysGelu(float16_t x) {
    const float16_t c1 = float16_t(0.79788456); // sqrt(2/pi)
    const float16_t c2 = float16_t(0.044715);
    float16_t x3 = x * x * x;
    return float16_t(0.5) * x * (float16_t(1.0) + tanh(c1 * (x + c2 * x3)));
}

// Dielectric and Conductor Fresnel reflectance calculation
vec3 upwaysComputeF0(vec3 albedo, float roughness) {
    // Standard dielectric base F0 = 0.04; metals transition toward base albedo
    float lum = dot(albedo, vec3(0.2126, 0.7152, 0.0722));
    float metallicFactor = clamp(lum * 2.0 - 0.2, 0.0, 1.0) * (1.0 - roughness);
    return max(mix(vec3(0.04), albedo, metallicFactor), vec3(0.04));
}

#endif // UPWAYS_COMMON_GLSL
