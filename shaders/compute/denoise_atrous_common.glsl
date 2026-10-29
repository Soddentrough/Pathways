#ifndef DENOISE_ATROUS_COMMON_GLSL
#define DENOISE_ATROUS_COMMON_GLSL

// =============================================================================
// Shared cross-bilateral A-Trous wavelet filtering for E-SVGF Passes 2-5.
// =============================================================================

#include "denoise_common.glsl"

// Filters the DIFFUSE lobe (RGBA16F: RGB radiance, A variance).
// Fine passes (stepSize <= 2) evaluate the full 9-tap 3x3 B-spline wavelet kernel;
// coarse passes (stepSize >= 4) evaluate a wave-uniform 5-tap complementary
// axial/diagonal wavelet stencil (+ at s=4, x at s=8) since Passes 2-3 already
// produce a contiguous 13x13 2D B-spline footprint that bridges the 4/8-px stride.
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
    float cleanCenterVar = abs(centerDiffuse.a);
    if (centerPos.z <= 0.0 || centerPos.z >= 9999.0 || cleanCenterVar <= 1e-6) {
        return vec4(centerDiffuse.rgb, cleanCenterVar);
    }

    const float centerW = 0.140625; // (3/8)^2
    vec3 radianceAcc = centerDiffuse.rgb * centerW;
    float weightAcc = centerW;
    float varAcc = (centerW * centerW) * cleanCenterVar;

    vec2 resolution = vec2(float(imgMax.x + 1), float(imgMax.y + 1));
    vec3 nView = vec3(centerNormal.x, -centerNormal.y, -centerNormal.z);
    float kx = aspectRatio * tanHalfFovY * nView.x;
    float ky = tanHalfFovY * nView.y;
    float scaleX = (2.0 * kx) / resolution.x;
    float scaleY = (2.0 * ky) / resolution.y;
    float bias = nView.z - kx - ky;

    float pixelFootprint = 2.0 * centerPos.z * tanHalfFovY * (1.0 / max(resolution.y, 1.0));
    float fp16QuantizationFloor = max(centerPos.z * 0.0009765625, 1e-4);
    float invDepthTolerance = 1.0 / max(phiPlaneDepth * float(stepSize) * pixelFootprint, fp16QuantizationFloor);
    float centerPlaneDot = dot(centerPos, nView);

    if (stepSize >= 3) {
        const ivec2 AXIAL_TAPS[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
        const ivec2 DIAG_TAPS[4]  = ivec2[4](ivec2(-1, -1), ivec2(1, -1), ivec2(-1, 1), ivec2(1, 1));
        const float kWeight = (stepSize <= 4) ? 0.09375 : 0.0625;
        for (int i = 0; i < 4; ++i) {
            ivec2 offset = (stepSize <= 4) ? AXIAL_TAPS[i] : DIAG_TAPS[i];
            int tapX = clamp(center.x + offset.x * stepSize, 0, imgMax.x);
            int tapY = clamp(center.y + offset.y * stepSize, 0, imgMax.y);
            ivec2 tapCoord = ivec2(tapX, tapY);

            vec4 tapND = imageLoad(uNormalDepth, tapCoord);
            if (tapND.w <= 0.0 || tapND.w >= 9999.0) continue;

            float cosAngle = max(dot(centerNormal, tapND.xyz), 0.0);
            if (cosAngle <= 0.15) continue;

            float tapPlaneDot = tapND.w * ((float(tapX) + 0.5) * scaleX + (float(tapY) + 0.5) * scaleY + bias);
            float depthArg = abs(tapPlaneDot - centerPlaneDot) * invDepthTolerance;
            if (depthArg >= 8.0) continue;

            vec4 tapData = imageLoad(uInDiffuse, tapCoord);
            float tapVar = abs(tapData.a);
            float tapLum = denoiseLuminance(tapData.rgb);
            float lumArg = abs(centerLum - tapLum) / (phiLuminance * sqrt(max(max(centerVar, tapVar * 0.25), 1e-4)) + 1e-4);

            float c2 = cosAngle * cosAngle;
            float c4 = c2 * c2;
            float c8 = c4 * c4;
            float c16 = c8 * c8;
            float16_t wNorm = float16_t((phiNormal >= 28.0) ? (c16 * c16) : (c16 * c8));
            float16_t wEdge = float16_t(exp2(-(depthArg + lumArg) * 1.44269504));
            float w = kWeight * float(wNorm * wEdge);

            radianceAcc += tapData.rgb * w;
            weightAcc += w;
            varAcc += w * w * tapVar;
        }
    } else {
        for (int y = -1; y <= 1; ++y) {
            float wKy = ATROUS_KERNEL_1D[y + 2];
            int tapY = clamp(center.y + y * stepSize, 0, imgMax.y);
            float planeY = (float(tapY) + 0.5) * scaleY + bias;
            for (int x = -1; x <= 1; ++x) {
                if (x == 0 && y == 0) continue;

                int tapX = clamp(center.x + x * stepSize, 0, imgMax.x);
                ivec2 tapCoord = ivec2(tapX, tapY);
                vec4 tapND = imageLoad(uNormalDepth, tapCoord);
                if (tapND.w <= 0.0 || tapND.w >= 9999.0) continue;

                float cosAngle = max(dot(centerNormal, tapND.xyz), 0.0);
                if (cosAngle <= 0.15) continue;

                float tapPlaneDot = tapND.w * ((float(tapX) + 0.5) * scaleX + planeY);
                float depthArg = abs(tapPlaneDot - centerPlaneDot) * invDepthTolerance;
                if (depthArg >= 8.0) continue;

                vec4 tapData = imageLoad(uInDiffuse, tapCoord);
                float tapVar = abs(tapData.a);
                float tapLum = denoiseLuminance(tapData.rgb);
                float lumArg = abs(centerLum - tapLum) / (phiLuminance * sqrt(max(max(centerVar, tapVar * 0.25), 1e-4)) + 1e-4);

                float c2 = cosAngle * cosAngle;
                float c4 = c2 * c2;
                float c8 = c4 * c4;
                float c16 = c8 * c8;
                float16_t wNorm = float16_t((phiNormal >= 28.0) ? (c16 * c16) : (c16 * c8));
                float16_t wEdge = float16_t(exp2(-(depthArg + lumArg) * 1.44269504));
                float w = (ATROUS_KERNEL_1D[x + 2] * wKy) * float(wNorm * wEdge);

                radianceAcc += tapData.rgb * w;
                weightAcc += w;
                varAcc += w * w * tapVar;
            }
        }
    }

    float invW = 1.0 / max(weightAcc, 1e-4);
    return vec4(radianceAcc * invW, varAcc * invW * invW);
}

// Filters the SPECULAR lobe with continuous roughness modulation (Section 5.5)
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

    float mirrorBypass = 1.0 - smoothstep(0.02, 0.08, roughness);
    if (mirrorBypass >= 0.999) {
        return rawSpecData;
    }

    vec3 rawSpecular = rawSpecData.rgb;
    float roughnessScale = clamp(roughness * roughness, 0.1, 1.0);
    float effectivePhiNormal = min(phiNormal / max(roughness * roughness, 0.05), 256.0);
    float effPhiLum = (roughness >= 0.50) ? (phiLuminance * 2.0) : phiLuminance;

    const float centerW = 0.140625;
    vec3 specAcc = rawSpecular * centerW;
    float weightAcc = centerW;
    float varAcc = (centerW * centerW) * max(rawSpecData.a, 0.0);

    vec2 resolution = vec2(float(imgMax.x + 1), float(imgMax.y + 1));
    vec3 nView = vec3(centerNormal.x, -centerNormal.y, -centerNormal.z);
    float kx = aspectRatio * tanHalfFovY * nView.x;
    float ky = tanHalfFovY * nView.y;
    float scaleX = (2.0 * kx) / resolution.x;
    float scaleY = (2.0 * ky) / resolution.y;
    float bias = nView.z - kx - ky;

    float pixelFootprint = 2.0 * centerPos.z * tanHalfFovY * (1.0 / max(resolution.y, 1.0));
    float fp16QuantizationFloor = max(centerPos.z * 0.0009765625, 1e-4);
    float invDepthTolerance = 1.0 / max(phiPlaneDepth * float(stepSize) * pixelFootprint, fp16QuantizationFloor);
    float invLumDenom = 1.0 / (effPhiLum * sqrt(max(centerVar, 1e-4)) + 1e-4);

    float centerPlaneDot = dot(centerPos, nView);
    int specStep = max(1, int(round(float(stepSize) * roughnessScale)));

    if (specStep >= 3) {
        const ivec2 AXIAL_TAPS[4] = ivec2[4](ivec2(-1, 0), ivec2(1, 0), ivec2(0, -1), ivec2(0, 1));
        const ivec2 DIAG_TAPS[4]  = ivec2[4](ivec2(-1, -1), ivec2(1, -1), ivec2(-1, 1), ivec2(1, 1));
        const float kWeight = (specStep <= 4) ? 0.09375 : 0.0625;
        for (int i = 0; i < 4; ++i) {
            ivec2 offset = (specStep <= 4) ? AXIAL_TAPS[i] : DIAG_TAPS[i];
            int tapX = clamp(center.x + offset.x * specStep, 0, imgMax.x);
            int tapY = clamp(center.y + offset.y * specStep, 0, imgMax.y);
            ivec2 tapCoord = ivec2(tapX, tapY);

            vec4 tapND = imageLoad(uNormalDepth, tapCoord);
            if (tapND.w <= 0.0 || tapND.w >= 9999.0) continue;

            float cosAngle = max(dot(centerNormal, tapND.xyz), 0.0);
            if (cosAngle <= 0.25) continue;

            float tapPlaneDot = tapND.w * ((float(tapX) + 0.5) * scaleX + (float(tapY) + 0.5) * scaleY + bias);
            float depthArg = abs(tapPlaneDot - centerPlaneDot) * invDepthTolerance;
            if (depthArg >= 8.0) continue;

            vec4 tapData = imageLoad(uInSpecular, tapCoord);
            float tapLum = denoiseLuminance(tapData.rgb);
            float lumArg = abs(centerLum - tapLum) * invLumDenom;

            float16_t wNorm = float16_t(pow(cosAngle, effectivePhiNormal));
            float16_t wEdge = float16_t(exp2(-(depthArg + lumArg) * 1.44269504));
            float w = kWeight * float(wNorm * wEdge);

            specAcc += tapData.rgb * w;
            weightAcc += w;
            varAcc += w * w * max(tapData.a, 0.0);
        }
    } else {
        for (int y = -1; y <= 1; ++y) {
            float wKy = ATROUS_KERNEL_1D[y + 2];
            int tapY = clamp(center.y + y * specStep, 0, imgMax.y);
            float planeY = (float(tapY) + 0.5) * scaleY + bias;
            for (int x = -1; x <= 1; ++x) {
                if (x == 0 && y == 0) continue;

                int tapX = clamp(center.x + x * specStep, 0, imgMax.x);
                ivec2 tapCoord = ivec2(tapX, tapY);
                vec4 tapND = imageLoad(uNormalDepth, tapCoord);
                if (tapND.w <= 0.0 || tapND.w >= 9999.0) continue;

                float cosAngle = max(dot(centerNormal, tapND.xyz), 0.0);
                if (cosAngle <= 0.25) continue;

                float tapPlaneDot = tapND.w * ((float(tapX) + 0.5) * scaleX + planeY);
                float depthArg = abs(tapPlaneDot - centerPlaneDot) * invDepthTolerance;
                if (depthArg >= 8.0) continue;

                vec4 tapData = imageLoad(uInSpecular, tapCoord);
                float tapLum = denoiseLuminance(tapData.rgb);
                float lumArg = abs(centerLum - tapLum) * invLumDenom;

                float16_t wNorm = float16_t(pow(cosAngle, effectivePhiNormal));
                float16_t wEdge = float16_t(exp2(-(depthArg + lumArg) * 1.44269504));
                float w = (ATROUS_KERNEL_1D[x + 2] * wKy) * float(wNorm * wEdge);

                specAcc += tapData.rgb * w;
                weightAcc += w;
                varAcc += w * w * max(tapData.a, 0.0);
            }
        }
    }

    float invW = 1.0 / max(weightAcc, 1e-4);
    vec3 filtered = specAcc * invW;
    float filteredVar = varAcc * invW * invW;
    return vec4(mix(filtered, rawSpecular, mirrorBypass), filteredVar);
}

#endif // DENOISE_ATROUS_COMMON_GLSL
