#ifndef DENOISE_COMMON_GLSL
#define DENOISE_COMMON_GLSL

// =============================================================================
// Pathways Adaptive Edge-Aware SVGF (E-SVGF) Denoiser Common Functions
// Reference: docs/design/adaptive_svgf_denoiser_proposal.md
// =============================================================================

float denoiseLuminance(vec3 c) {
    return dot(max(c, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
}

// Computes safe albedo demodulation / remodulation factor from G-buffer albedo and linear depth.
// Sky pixels (depth >= 9999.0) and pure black-albedo emitters (dot(albedo, albedo) <= 1e-5)
// use identity vec3(1.0) so sky/emissive radiance is preserved without darkening.
vec3 denoiseDemodulationFactor(vec3 albedo, float linearDepth) {
    if (linearDepth >= 9999.0 || dot(albedo, albedo) <= 1e-5) {
        return vec3(1.0);
    }
    return max(albedo, vec3(0.01));
}

// B3-spline 5x5 wavelet filter 1D weights: [1/16, 1/4, 3/8, 1/4, 1/16]
const float ATROUS_KERNEL_1D[5] = float[5](
    0.0625, 0.2500, 0.3750, 0.2500, 0.0625
);

// Bilateral normal edge-stopping weight
float denoiseNormalWeight(vec3 nP, vec3 nQ, float phiN) {
    float cosAngle = max(dot(nP, nQ), 0.0);
    return pow(cosAngle, phiN);
}

// Bilateral depth edge-stopping weight (scale-invariant relative depth gradient)
float denoiseDepthWeight(float zP, float zQ, float phiZ) {
    float maxZ = max(max(abs(zP), abs(zQ)), 1e-3);
    float diffZ = abs(zP - zQ);
    return exp(-diffZ / max(phiZ * maxZ, 1e-4));
}

// Bilateral luminance variance edge-stopping weight.
// NOTE: True exponential falloff WITHOUT an artificial 3*sigma numerator floor
// (Section 5.2 mitigation: a 3-sigma floor forced w_L >= exp(-3/4) ~ 0.47,
// blurring across sharp shadow boundaries).
float denoiseLuminanceWeight(float lumP, float lumQ, float varianceP, float phiL) {
    float diffLum = abs(lumP - lumQ);
    float sigma = sqrt(max(varianceP, 1e-4));
    return exp(-diffLum / (phiL * sigma + 1e-4));
}

// -----------------------------------------------------------------------------
// Metric view-space unprojection (Section 5.1, tangent-plane depth weighting)
// -----------------------------------------------------------------------------

// Unprojects pixel coordinates and linear view depth into metric 3D view space
vec3 unprojectToView(ivec2 pixelCoord, float linearDepth, float tanHalfFovY, float aspectRatio, vec2 resolution) {
    if (isnan(linearDepth) || isinf(linearDepth) || linearDepth <= 0.0) {
        return vec3(0.0);
    }
    vec2 ndc = (vec2(pixelCoord) + 0.5) / resolution * 2.0 - 1.0;
    float x = ndc.x * aspectRatio * tanHalfFovY * linearDepth;
    float y = ndc.y * tanHalfFovY * linearDepth;
    return vec3(x, y, linearDepth);
}

// Metric tangent-plane distance depth weighting (Section 8 FM1 mitigation).
// For a continuous planar surface, planeDist evaluates to identically 0.0
// regardless of view distance or grazing angle, eliminating depth staircase
// false edges. Tolerance is bounded below by the FP16 quantization floor.
float evaluatePlaneDepthWeight(
    vec3 pViewPos,
    vec3 qViewPos,
    vec3 pNormal,
    float phiPlaneDepth,
    int stepSize,
    float tanHalfFovY,
    float screenHeight)
{
    // Guard against skybox / infinite depth singularities
    if (length(pViewPos) <= 1e-4 || length(qViewPos) <= 1e-4) {
        return 0.0;
    }

    vec3 disp = qViewPos - pViewPos;
    // uNormalDepth.xyz is in GLM right-handed view space (+X right, +Y up, -Z forward),
    // whereas unprojectToView() returns Vulkan-NDC positive-depth space (+X right, +Y down, +Z forward).
    vec3 nView = vec3(pNormal.x, -pNormal.y, -pNormal.z);
    float planeDist = abs(dot(disp, nView));

    float pixelFootprint = 2.0 * pViewPos.z * tanHalfFovY / max(screenHeight, 1.0);
    float fp16QuantizationFloor = max(pViewPos.z * 0.0009765625, 1e-4); // 2 * eps_fp16(z)
    float depthTolerance = max(phiPlaneDepth * float(stepSize) * pixelFootprint, fp16QuantizationFloor);

    return exp(-planeDist / depthTolerance);
}

#endif // DENOISE_COMMON_GLSL
