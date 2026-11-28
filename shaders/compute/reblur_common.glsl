#ifndef REBLUR_COMMON_GLSL
#define REBLUR_COMMON_GLSL

// =============================================================================
// Pathways NRD ReBLUR-Grade Analytical Spatiotemporal Denoiser — Shared Helpers
// =============================================================================

const float REBLUR_SKY_DEPTH    = 9999.0;
const float REBLUR_ALBEDO_FLOOR = 0.02;
const float REBLUR_MAX_FP16     = 65000.0;
const float REBLUR_LOG2E        = 1.4426950408889634;

// Shared Camera UBO (std140, 224 bytes, set 0 binding 0)
#ifdef REBLUR_USE_CAMERA_UBO
layout(std140, set = 0, binding = 0) uniform ReblurCameraUBO {
    mat4  viewToWorld;         // offset 0   (64 B): camera viewInverse
    mat4  prevViewToWorld;     // offset 64  (64 B): camera prevViewInverse
    mat4  currViewToPrevView;  // offset 128 (64 B): S * (prevView * viewInverse) * S
    vec2  currJitterUv;        // offset 192 (8 B)
    vec2  prevJitterUv;        // offset 200 (8 B)
    float tanHalfFovY;         // offset 208 (4 B)
    float aspectRatio;         // offset 212 (4 B)
    float prevTanHalfFovY;     // offset 216 (4 B)
    float prevAspectRatio;     // offset 220 (4 B)
} uCam;
#endif

// -----------------------------------------------------------------------------
// Numeric sanitization, compact FP16 packing & color utilities
// -----------------------------------------------------------------------------

float reblurSanitizeScalar(float v, float fallbackVal) {
    return (isnan(v) || isinf(v)) ? fallbackVal : v;
}

vec2 reblurSanitizeVec2(vec2 v) {
    return vec2(
        (isnan(v.x) || isinf(v.x)) ? 0.0 : v.x,
        (isnan(v.y) || isinf(v.y)) ? 0.0 : v.y
    );
}

vec3 reblurSanitizeRgb(vec3 v) {
    bvec3 bad = bvec3(
        isnan(v.x) || isinf(v.x),
        isnan(v.y) || isinf(v.y),
        isnan(v.z) || isinf(v.z)
    );
    vec3 clean = vec3(
        bad.x ? 0.0 : max(v.x, 0.0),
        bad.y ? 0.0 : max(v.y, 0.0),
        bad.z ? 0.0 : max(v.z, 0.0)
    );
    return min(clean, vec3(REBLUR_MAX_FP16));
}

uvec2 reblurPackHalf4(vec4 v) {
    return uvec2(packHalf2x16(v.xy), packHalf2x16(v.zw));
}

vec4 reblurUnpackHalf4(uvec2 p) {
    return vec4(unpackHalf2x16(p.x), unpackHalf2x16(p.y));
}

float reblurLuminance(vec3 c) {
    return dot(max(c, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
}

bool reblurIsSky(float linearDepth) {
    return (isnan(linearDepth) || isinf(linearDepth) || linearDepth <= 0.0 || linearDepth >= REBLUR_SKY_DEPTH);
}

// Exact match to ffx_rr_prep.comp:89-92: floor 0.02 for non-zero albedo,
// identity vec3(1.0) on sky / zero-albedo emitters so demod/remod is energy-exact.
vec3 reblurEffectiveAlbedo(vec3 rawAlb, float linearDepth) {
    if (reblurIsSky(linearDepth)) {
        return vec3(1.0);
    }
    vec3 safeAlb = reblurSanitizeRgb(rawAlb);
    return any(greaterThan(safeAlb, vec3(1e-3))) ? max(safeAlb, vec3(REBLUR_ALBEDO_FLOOR)) : vec3(1.0);
}

// Reversible Reinhard tone-space mapping for HDR-safe unjittering & AABB clipping.
vec3 reblurTonemap(vec3 c) {
    vec3 safe = max(c, vec3(0.0));
    return safe / (1.0 + reblurLuminance(safe));
}

vec3 reblurInverseTonemap(vec3 tc) {
    vec3 safe = max(tc, vec3(0.0));
    return safe / max(1.0 - reblurLuminance(safe), 1e-4);
}

// -----------------------------------------------------------------------------
// Coordinate & geometric edge-stopping utilities
// -----------------------------------------------------------------------------

// Converts world-space unit normal to Vulkan-NDC positive-depth view space
// (+X right, +Y down, +Z forward) matching reblurUnprojectToView().
vec3 reblurWorldNormalToVkView(vec3 worldN, mat4 viewToWorld) {
    vec3 nGlm = transpose(mat3(viewToWorld)) * worldN;
    return vec3(nGlm.x, -nGlm.y, -nGlm.z);
}

// Unprojects pixel coordinates and positive linear depth into Vulkan-NDC
// positive-depth 3D view space (+X right, +Y down, +Z forward).
vec3 reblurUnprojectToView(ivec2 pixelCoord, float linearDepth, float tanHalfFovY, float aspectRatio, vec2 resolution) {
    if (reblurIsSky(linearDepth)) {
        return vec3(0.0);
    }
    vec2 ndc = (vec2(pixelCoord) + 0.5) / resolution * 2.0 - 1.0;
    float x = ndc.x * aspectRatio * tanHalfFovY * linearDepth;
    float y = ndc.y * tanHalfFovY * linearDepth;
    return vec3(x, y, linearDepth);
}

// Unprojects an arbitrary UV (e.g. unjittered UV) and positive linear depth
// into Vulkan-NDC positive-depth 3D view space.
vec3 reblurUnprojectUvToView(vec2 uv, float linearDepth, float tanHalfFovY, float aspectRatio) {
    if (reblurIsSky(linearDepth)) {
        return vec3(0.0);
    }
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * aspectRatio * tanHalfFovY * linearDepth,
                ndc.y * tanHalfFovY * linearDepth,
                linearDepth);
}

// Metric tangent-plane depth weight (zero false edges on oblique/grazing planes).
// nVkView must be in Vulkan-NDC positive-depth view space (+X right, +Y down, +Z forward).
float reblurEvaluatePlaneDepthWeight(
    vec3 pViewPos,
    vec3 qViewPos,
    vec3 nVkView,
    float phiPlaneDepth,
    int stepSize,
    float tanHalfFovY,
    float screenHeight)
{
    if (pViewPos.z <= 1e-4 || qViewPos.z <= 1e-4) {
        return 0.0;
    }
    float planeDist = abs(dot(qViewPos - pViewPos, nVkView));
    float pixelFootprint = 2.0 * pViewPos.z * tanHalfFovY / max(screenHeight, 1.0);
    float fp16Floor = max(pViewPos.z * 0.0009765625, 1e-4);
    float depthTol = max(phiPlaneDepth * float(max(stepSize, 1)) * pixelFootprint, fp16Floor);
    return exp2(-(planeDist / depthTol) * REBLUR_LOG2E);
}

// Fast cos^32 via 5 multiplies
float reblurPow32(float x) {
    float x2  = x * x;
    float x4  = x2 * x2;
    float x8  = x4 * x4;
    float x16 = x8 * x8;
    return x16 * x16;
}

// -----------------------------------------------------------------------------
// YCoCg & AABB clipping (Pass 5 temporal stabilization)
// -----------------------------------------------------------------------------

vec3 reblurRgbToYCoCg(vec3 c) {
    float co = c.r - c.b;
    float t  = c.b + co * 0.5;
    float cg = c.g - t;
    float y  = t + cg * 0.5;
    return vec3(y, co, cg);
}

vec3 reblurYCoCgToRgb(vec3 ycocg) {
    float t = ycocg.x - ycocg.z * 0.5;
    float g = ycocg.z + t;
    float b = t - ycocg.y * 0.5;
    float r = b + ycocg.y;
    return vec3(r, g, b);
}

// Ray-AABB clip from history sample toward neighborhood center in YCoCg space.
vec3 reblurClipAABB(vec3 aabbMin, vec3 aabbMax, vec3 hist, vec3 center) {
    vec3 pClip = 0.5 * (aabbMax + aabbMin);
    vec3 eClip = 0.5 * (aabbMax - aabbMin) + vec3(1e-5);
    vec3 vClip = hist - pClip;
    vec3 vUnit = vClip / eClip;
    vec3 aUnit = abs(vUnit);
    float maxUnit = max(aUnit.x, max(aUnit.y, aUnit.z));
    if (maxUnit > 1.0) {
        return pClip + vClip / maxUnit;
    }
    return hist;
}

// Sharp positive non-ringing Blackman-Harris approximation for 3x3 subpixel unjittering.
float reblurBlackmanHarrisWeight(vec2 deltaPx) {
    return exp2(-3.2 * REBLUR_LOG2E * dot(deltaPx, deltaPx));
}

// B3-spline 1D wavelet weights for 3x3 atrous (indices -1, 0, +1 -> 1/4, 3/8, 1/4).
const float REBLUR_ATROUS_1D[3] = float[3](0.25, 0.375, 0.25);

#endif // REBLUR_COMMON_GLSL
