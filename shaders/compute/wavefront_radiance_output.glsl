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

#endif // WAVEFRONT_RADIANCE_OUTPUT_GLSL
