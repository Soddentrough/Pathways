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
    vec4 centerDiffuse,
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
    if (centerPos.z <= 0.0 || centerPos.z >= 9999.0 || centerDiffuse.a <= 1e-6) {
        return centerDiffuse;
    }

    const float centerW = 0.140625; // ATROUS_KERNEL_1D[2] * ATROUS_KERNEL_1D[2] = (3/8)^2
    vec3 radianceAcc = centerDiffuse.rgb * centerW;
    float weightAcc = centerW;
    float varAcc = (centerW * centerW) * max(centerDiffuse.a, 0.0);

    vec2 resolution = vec2(float(imgMax.x + 1), float(imgMax.y + 1));
    vec2 invRes2 = 2.0 / resolution;
    vec2 viewScale = vec2(aspectRatio * tanHalfFovY, tanHalfFovY);
    vec3 nView = vec3(centerNormal.x, -centerNormal.y, -centerNormal.z);
    float pixelFootprint = 2.0 * centerPos.z * tanHalfFovY * (1.0 / max(resolution.y, 1.0));
    float fp16QuantizationFloor = max(centerPos.z * 0.0009765625, 1e-4);
    float invDepthTolerance = 1.0 / max(phiPlaneDepth * float(stepSize) * pixelFootprint, fp16QuantizationFloor);
    float invLumDenom = 1.0 / (phiLuminance * sqrt(max(centerVar, 1e-4)) + 1e-4);

    const int radius = 1;
    float centerPlaneDot = dot(centerPos, nView);

    for (int y = -radius; y <= radius; ++y) {
        float ky = ATROUS_KERNEL_1D[y + 2];
        for (int x = -radius; x <= radius; ++x) {
            if (x == 0 && y == 0) continue;

            ivec2 tapCoord = clamp(center + ivec2(x, y) * stepSize, ivec2(0), imgMax);
            vec4 tapND = imageLoad(uNormalDepth, tapCoord);
            if (tapND.w <= 0.0 || tapND.w >= 9999.0) continue;

            float cosAngle = max(dot(centerNormal, tapND.xyz), 0.0);
            if (cosAngle <= 0.15) continue;

            vec2 tapNdc = (vec2(tapCoord) + 0.5) * invRes2 - 1.0;
            vec3 tapPos = vec3(tapNdc * viewScale * tapND.w, tapND.w);
            float planeDist = abs(dot(tapPos, nView) - centerPlaneDot);
            float depthArg = planeDist * invDepthTolerance;
            if (depthArg >= 8.0) continue;

            vec4 tapData = imageLoad(uInDiffuse, tapCoord);
            float tapLum = denoiseLuminance(tapData.rgb);
            float lumArg = abs(centerLum - tapLum) * invLumDenom;

            float c2 = cosAngle * cosAngle;
            float c4 = c2 * c2;
            float c8 = c4 * c4;
            float c16 = c8 * c8;
            float16_t wNorm = float16_t((phiNormal >= 28.0) ? (c16 * c16) : (c16 * c8));
            float16_t wEdge = float16_t(exp2(-(depthArg + lumArg) * 1.44269504));
            float kernelWeight = ATROUS_KERNEL_1D[x + 2] * ky;
            float w = kernelWeight * float(wNorm * wEdge);

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
//  * Zero-energy / converged fast early-exit for diffuse/matte surfaces
//  * Roughness-squared dilation with monotonic non-collapsing tap offsets
//  * Normal power capped: min(phiNormal / max(r^2, 0.05), 256.0)
vec4 atrousFilterSpecular(
    ivec2 center,
    int stepSize,
    vec4 rawSpecData,
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
    if (centerPos.z <= 0.0 || centerPos.z >= 9999.0 || (centerLum <= 1e-4 && rawSpecData.a <= 1e-5)) {
        return rawSpecData;
    }

    // Continuous mirror identity bypass: 0.0 = full filter, 1.0 = pure mirror
    float mirrorBypass = 1.0 - smoothstep(0.02, 0.08, roughness);
    if (mirrorBypass >= 0.999) {
        return rawSpecData; // Pure mirror: skip taps entirely
    }

    vec3 rawSpecular = rawSpecData.rgb;
    float roughnessScale = clamp(roughness * roughness, 0.1, 1.0);
    float effectivePhiNormal = min(phiNormal / max(roughness * roughness, 0.05), 256.0);

    const float centerW = 0.140625;
    vec3 specAcc = rawSpecular * centerW;
    float weightAcc = centerW;
    float varAcc = (centerW * centerW) * max(rawSpecData.a, 0.0);

    vec2 resolution = vec2(float(imgMax.x + 1), float(imgMax.y + 1));
    vec2 invRes2 = 2.0 / resolution;
    vec2 viewScale = vec2(aspectRatio * tanHalfFovY, tanHalfFovY);
    vec3 nView = vec3(centerNormal.x, -centerNormal.y, -centerNormal.z);
    float pixelFootprint = 2.0 * centerPos.z * tanHalfFovY * (1.0 / max(resolution.y, 1.0));
    float fp16QuantizationFloor = max(centerPos.z * 0.0009765625, 1e-4);
    float invDepthTolerance = 1.0 / max(phiPlaneDepth * float(stepSize) * pixelFootprint, fp16QuantizationFloor);
    float invLumDenom = 1.0 / (phiLuminance * sqrt(max(centerVar, 1e-4)) + 1e-4);

    const int radius = 1;
    float centerPlaneDot = dot(centerPos, nView);

    for (int y = -radius; y <= radius; ++y) {
        float ky = ATROUS_KERNEL_1D[y + 2];
        for (int x = -radius; x <= radius; ++x) {
            if (x == 0 && y == 0) continue;

            // Monotonic non-collapsing tap offset: guarantees distinct non-zero
            // taps for non-center offsets at low roughness
            ivec2 unitOffset = ivec2(x, y);
            ivec2 absOffset = abs(unitOffset);
            ivec2 dilatedOffset = ivec2(round(vec2(absOffset) * float(stepSize) * roughnessScale));
            ivec2 tapOffset = ivec2(sign(vec2(unitOffset))) * max(absOffset, dilatedOffset);
            ivec2 tapCoord = clamp(center + tapOffset, ivec2(0), imgMax);

            vec4 tapND = imageLoad(uNormalDepth, tapCoord);
            if (tapND.w <= 0.0 || tapND.w >= 9999.0) continue;

            float cosAngle = max(dot(centerNormal, tapND.xyz), 0.0);
            if (cosAngle <= 0.25) continue;

            vec2 tapNdc = (vec2(tapCoord) + 0.5) * invRes2 - 1.0;
            vec3 tapPos = vec3(tapNdc * viewScale * tapND.w, tapND.w);
            float planeDist = abs(dot(tapPos, nView) - centerPlaneDot);
            float depthArg = planeDist * invDepthTolerance;
            if (depthArg >= 8.0) continue;

            vec4 tapData = imageLoad(uInSpecular, tapCoord);
            float tapLum = denoiseLuminance(tapData.rgb);
            float lumArg = abs(centerLum - tapLum) * invLumDenom;

            float16_t wNorm = float16_t(pow(cosAngle, effectivePhiNormal));
            float16_t wEdge = float16_t(exp2(-(depthArg + lumArg) * 1.44269504));
            float kernelWeight = ATROUS_KERNEL_1D[x + 2] * ky;
            float w = kernelWeight * float(wNorm * wEdge);

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
