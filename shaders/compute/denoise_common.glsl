#ifndef DENOISE_COMMON_GLSL
#define DENOISE_COMMON_GLSL

// =============================================================================
// Pathways In-House ReLAX / A-SVGF Denoiser Common Functions & Kernels
// =============================================================================

float denoiseLuminance(vec3 c) {
    return dot(max(c, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
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

// Bilateral luminance variance edge-stopping weight
float denoiseLuminanceWeight(float lumP, float lumQ, float varianceP, float phiL) {
    float diffLum = abs(lumP - lumQ);
    float sigma = sqrt(max(varianceP, 0.0));
    return exp(-diffLum / max(phiL * sigma, 1e-2));
}

#endif // DENOISE_COMMON_GLSL
