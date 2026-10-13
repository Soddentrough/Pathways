#ifndef DENOISE_ATROUS_COMMON_GLSL
#define DENOISE_ATROUS_COMMON_GLSL

// =============================================================================
// Shared cross-bilateral A-Trous wavelet filtering for E-SVGF Passes 2-5.
//
// NOTE: These helpers reference the per-shader global uniforms uInDiffuse,
// uInSpecular and uNormalDepth directly (GLSL forbids passing formatted
// storage images as function parameters). Including shaders MUST declare:
//
//   layout(set = 0, binding = 0, rgba16f) readonly uniform image2D uInDiffuse;
//   layout(set = 0, binding = 1, rgba16f) readonly uniform image2D uInSpecular;
//   layout(set = 0, binding = 2, rgba16f) readonly uniform image2D uNormalDepth;
//
// Radiance accumulators are strictly FP32; bilateral edge-stopping weights
// are evaluated in native FP16 (explicit float16 arithmetic extension).
// =============================================================================

#include "denoise_common.glsl"

// Filters the DIFFUSE lobe (RGBA16F: RGB radiance, A variance) with the 25-tap
// dilated cross-bilateral kernel. Returns vec4(filteredRadiance, filteredVar)
// using SVGF w^2 variance propagation:
//   Var_{k+1} = sum(w_i^2 * Var_k(q_i)) / (sum(w_i))^2
vec4 atrousFilterDiffuse(
    ivec2 center,
    int stepSize,
    vec3 centerNormal,
    vec3 centerPos,
    float centerLum,
    float centerVar,
    float phiNormal,
    float phiPlaneDepth,
    float phiLuminance,
    float tanHalfFovY,
    float aspectRatio,
    ivec2 imgMax)
{
    vec3 radianceAcc = vec3(0.0);   // FP32 accumulators
    float weightAcc = 0.0;
    float varAcc = 0.0;

    vec2 resolution = vec2(float(imgMax.x + 1), float(imgMax.y + 1));

    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            float kernelWeight = ATROUS_KERNEL_1D[x + 2] * ATROUS_KERNEL_1D[y + 2];

            ivec2 tapCoord = clamp(center + ivec2(x, y) * stepSize, ivec2(0), imgMax);
            vec4 tapData = imageLoad(uInDiffuse, tapCoord);
            vec4 tapND = imageLoad(uNormalDepth, tapCoord);

            float16_t wNorm = float16_t(denoiseNormalWeight(centerNormal, tapND.xyz, phiNormal));

            vec3 tapPos = unprojectToView(tapCoord, tapND.w, tanHalfFovY, aspectRatio, resolution);
            float16_t wDepth = float16_t(evaluatePlaneDepthWeight(
                centerPos, tapPos, centerNormal, phiPlaneDepth, stepSize, tanHalfFovY, resolution.y));

            float tapLum = denoiseLuminance(tapData.rgb);
            float16_t wLum = float16_t(denoiseLuminanceWeight(centerLum, tapLum, centerVar, phiLuminance));

            float w = kernelWeight * float(wNorm * wDepth * wLum);

            radianceAcc += tapData.rgb * w;
            weightAcc += w;
            varAcc += w * w * max(tapData.a, 0.0);
        }
    }

    float invW = 1.0 / max(weightAcc, 1e-4);
    vec3 filtered = radianceAcc * invW;
    float filteredVar = varAcc * invW * invW;
    return vec4(filtered, filteredVar);
}

// Filters the SPECULAR lobe with continuous roughness modulation (Section 5.5):
//  * Continuous mirror bypass: 1 - smoothstep(0.02, 0.08, roughness)
//  * Roughness-squared dilation with monotonic non-collapsing tap offsets
//  * Normal power capped: min(phiNormal / max(r^2, 0.05), 256.0)
vec4 atrousFilterSpecular(
    ivec2 center,
    int stepSize,
    vec3 centerNormal,
    vec3 centerPos,
    float centerLum,
    float centerVar,
    float roughness,
    float phiNormal,
    float phiPlaneDepth,
    float phiLuminance,
    float tanHalfFovY,
    float aspectRatio,
    ivec2 imgMax)
{
    vec4 rawSpecData = imageLoad(uInSpecular, center);
    vec3 rawSpecular = rawSpecData.rgb;

    // Continuous mirror identity bypass: 0.0 = full filter, 1.0 = pure mirror
    float mirrorBypass = 1.0 - smoothstep(0.02, 0.08, roughness);
    if (mirrorBypass >= 0.999) {
        return rawSpecData; // Pure mirror: skip 25 taps entirely
    }

    float roughnessScale = clamp(roughness * roughness, 0.1, 1.0);
    float effectivePhiNormal = min(phiNormal / max(roughness * roughness, 0.05), 256.0);

    vec3 specAcc = vec3(0.0);   // FP32 accumulators (prevent HDR Inf overflow > 65504)
    float weightAcc = 0.0;
    float varAcc = 0.0;

    vec2 resolution = vec2(float(imgMax.x + 1), float(imgMax.y + 1));

    for (int y = -2; y <= 2; ++y) {
        for (int x = -2; x <= 2; ++x) {
            float kernelWeight = ATROUS_KERNEL_1D[x + 2] * ATROUS_KERNEL_1D[y + 2];

            // Monotonic non-collapsing tap offset: guarantees distinct non-zero
            // taps for non-center offsets at low roughness
            ivec2 unitOffset = ivec2(x, y);
            ivec2 absOffset = abs(unitOffset);
            ivec2 dilatedOffset = ivec2(round(vec2(absOffset) * float(stepSize) * roughnessScale));
            ivec2 tapOffset = ivec2(sign(vec2(unitOffset))) * max(absOffset, dilatedOffset);
            ivec2 tapCoord = clamp(center + tapOffset, ivec2(0), imgMax);

            vec4 tapData = imageLoad(uInSpecular, tapCoord);
            vec4 tapND = imageLoad(uNormalDepth, tapCoord);

            float16_t wNorm = float16_t(denoiseNormalWeight(centerNormal, tapND.xyz, effectivePhiNormal));

            vec3 tapPos = unprojectToView(tapCoord, tapND.w, tanHalfFovY, aspectRatio, resolution);
            float16_t wDepth = float16_t(evaluatePlaneDepthWeight(
                centerPos, tapPos, centerNormal, phiPlaneDepth, stepSize, tanHalfFovY, resolution.y));

            float tapLum = denoiseLuminance(tapData.rgb);
            float16_t wLum = float16_t(denoiseLuminanceWeight(centerLum, tapLum, centerVar, phiLuminance));

            float w = kernelWeight * float(wNorm * wDepth * wLum);

            specAcc += tapData.rgb * w;
            weightAcc += w;
            varAcc += w * w * max(tapData.a, 0.0);
        }
    }

    float invW = 1.0 / max(weightAcc, 1e-4);
    vec3 filtered = specAcc * invW;
    float filteredVar = varAcc * invW * invW;
    return vec4(mix(filtered, rawSpecular, mirrorBypass), filteredVar);
}

#endif // DENOISE_ATROUS_COMMON_GLSL
