#ifndef WAVEFRONT_RADIANCE_OUTPUT_GLSL
#define WAVEFRONT_RADIANCE_OUTPUT_GLSL

layout(binding = 42, rgba16f) uniform image2D uDirectDiffuseImage;
layout(binding = 43, rgba16f) uniform image2D uIndirectDiffuseImage;
layout(binding = 44, rgba16f) uniform image2D uDirectSpecularImage;
layout(binding = 45, rgba16f) uniform image2D uIndirectSpecularImage;

vec3 wfSanitizeRadiance(vec3 v) {
    bvec3 bad = bvec3(
        isnan(v.x) || isinf(v.x),
        isnan(v.y) || isinf(v.y),
        isnan(v.z) || isinf(v.z)
    );
    vec3 safe = mix(v, vec3(0.0), bad);
    return clamp(safe, vec3(0.0), vec3(65504.0));
}

void wfAddSplitDirect(ivec2 coord, vec3 diffRad, vec3 specRad, uint captureMlData, uint seed) {
    if ((captureMlData & ML_CAPTURE_SPLIT_4SIG_BIT) == 0u) {
        return;
    }
    uint localSeed = seed;
    vec3 d = wfSanitizeRadiance(diffRad);
    if (dot(d, d) > 1e-12) {
        vec4 prev = imageLoad(uDirectDiffuseImage, coord);
        vec3 sum = min(addFp16Stochastic(prev.rgb, d, localSeed), vec3(65504.0));
        imageStore(uDirectDiffuseImage, coord, vec4(sum, prev.a));
    }
    vec3 s = wfSanitizeRadiance(specRad);
    if (dot(s, s) > 1e-12) {
        vec4 prev = imageLoad(uDirectSpecularImage, coord);
        vec3 sum = min(addFp16Stochastic(prev.rgb, s, localSeed), vec3(65504.0));
        imageStore(uDirectSpecularImage, coord, vec4(sum, prev.a));
    }
}

void wfAddSplitIndirect(ivec2 coord, vec3 diffRad, vec3 specRad, uint captureMlData, uint seed) {
    if ((captureMlData & ML_CAPTURE_SPLIT_4SIG_BIT) == 0u) {
        return;
    }
    uint localSeed = seed;
    vec3 d = wfSanitizeRadiance(diffRad);
    if (dot(d, d) > 1e-12) {
        vec4 prev = imageLoad(uIndirectDiffuseImage, coord);
        vec3 sum = min(addFp16Stochastic(prev.rgb, d, localSeed), vec3(65504.0));
        imageStore(uIndirectDiffuseImage, coord, vec4(sum, prev.a));
    }
    vec3 s = wfSanitizeRadiance(specRad);
    if (dot(s, s) > 1e-12) {
        vec4 prev = imageLoad(uIndirectSpecularImage, coord);
        vec3 sum = min(addFp16Stochastic(prev.rgb, s, localSeed), vec3(65504.0));
        imageStore(uIndirectSpecularImage, coord, vec4(sum, prev.a));
    }
}

void wfStoreIndirectDiffuseHitDist(ivec2 coord, float hitDist, bool isMiss, uint captureMlData, uint sppSampleIndex) {
    if ((captureMlData & ML_CAPTURE_SPLIT_4SIG_BIT) == 0u || sppSampleIndex != 0u) {
        return;
    }
    vec4 prev = imageLoad(uIndirectDiffuseImage, coord);
    float rawHitT = isMiss ? 1000.0 : clamp(hitDist, 0.0, 10000.0);
    imageStore(uIndirectDiffuseImage, coord, vec4(prev.rgb, rawHitT));
}

void wfStoreIndirectSpecularHitDist(ivec2 coord, float hitDist, bool isMiss, uint captureMlData, uint sppSampleIndex) {
    if ((captureMlData & ML_CAPTURE_SPLIT_4SIG_BIT) == 0u || sppSampleIndex != 0u) {
        return;
    }
    vec4 prev = imageLoad(uIndirectSpecularImage, coord);
    float rawHitT = isMiss ? 1000.0 : clamp(hitDist, 0.0, 10000.0);
    imageStore(uIndirectSpecularImage, coord, vec4(prev.rgb, rawHitT));
}

// Compact live ML G-buffer attachments (41 B/px write-stage separated: zero 16-byte RMWs)
// Note: binding 46 is reserved for EnvImportanceBuffer (env_importance.glsl).
layout(binding = 47, rg32ui) uniform uimage2D uMlIrradianceImage;
layout(binding = 48, rgba32ui) uniform uimage2D uMlPrimaryGuideImage;
layout(binding = 49, r32ui) uniform uimage2D uMlSurfaceMvImage;
layout(binding = 50, r8) uniform image2D uMlDiffHitDistImage;
layout(binding = 51, rg32ui) uniform uimage2D uMlSpecVmvImage;
layout(binding = 52, r32ui) uniform uimage2D uMlSkipCompactImage;

void wfStoreCompactMlIrradiance(ivec2 coord, vec3 diffRad, vec3 specRad) {
    bool hasD = any(greaterThan(diffRad, vec3(0.0)));
    bool hasS = any(greaterThan(specRad, vec3(0.0)));
    if (!hasD && !hasS) {
        return;
    }
    uint px = hasD ? packRGB9E5(diffRad) : 0u;
    uint py = hasS ? packRGB9E5(specRad) : 0u;
    imageStore(uMlIrradianceImage, coord, uvec4(px, py, 0u, 0u));
}

void wfAddCompactMlIrradiance(ivec2 coord, vec3 addDiff, vec3 addSpec) {
    bool hasD = any(greaterThan(addDiff, vec3(0.0)));
    bool hasS = any(greaterThan(addSpec, vec3(0.0)));
    if (!hasD && !hasS) {
        return;
    }
    uvec2 cur = imageLoad(uMlIrradianceImage, coord).xy;
    if (hasD) {
        vec3 prevD = (cur.x != 0u) ? unpackRGB9E5(cur.x) : vec3(0.0);
        cur.x = packRGB9E5(prevD + addDiff);
    }
    if (hasS) {
        vec3 prevS = (cur.y != 0u) ? unpackRGB9E5(cur.y) : vec3(0.0);
        cur.y = packRGB9E5(prevS + addSpec);
    }
    imageStore(uMlIrradianceImage, coord, uvec4(cur, 0u, 0u));
}

void wfStoreCompactMlSkip(ivec2 coord, vec3 skipRad) {
    if (!any(greaterThan(skipRad, vec3(0.0)))) {
        return;
    }
    imageStore(uMlSkipCompactImage, coord, uvec4(packRGB9E5(skipRad), 0u, 0u, 0u));
}

void wfAddCompactMlSkip(ivec2 coord, vec3 addSkip) {
    if (!any(greaterThan(addSkip, vec3(0.0)))) {
        return;
    }
    uint cur = imageLoad(uMlSkipCompactImage, coord).x;
    vec3 prevSkip = (cur != 0u) ? unpackRGB9E5(cur) : vec3(0.0);
    imageStore(uMlSkipCompactImage, coord, uvec4(packRGB9E5(prevSkip + addSkip), 0u, 0u, 0u));
}

#endif // WAVEFRONT_RADIANCE_OUTPUT_GLSL
