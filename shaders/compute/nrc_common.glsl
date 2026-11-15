#ifndef NRC_COMMON_GLSL
#define NRC_COMMON_GLSL

// Neural Radiance Caching (NRC) Wave32 WMMA Architecture
// Target: AMD RDNA 4 (gfx1201), Vulkan 1.4

const uint NRC_W0_OFFSET = 0u;      // 64 * 64 = 4096 float16_t
const uint NRC_B0_OFFSET = 4096u;   // 64 float16_t
const uint NRC_W1_OFFSET = 4160u;   // 64 * 64 = 4096 float16_t
const uint NRC_B1_OFFSET = 8256u;   // 64 float16_t
const uint NRC_W2_OFFSET = 8320u;   // 64 * 16 = 1024 float16_t
const uint NRC_B2_OFFSET = 9344u;   // 16 float16_t
const uint NRC_TOTAL_WEIGHTS = 9360u;

const uint NRC_HASH_LEVELS = 12u;
const uint NRC_HASH_TABLE_SIZE = 262144u;

const float NRC_GRID_SCALES[12] = float[12](
    16.0, 23.36, 34.12, 49.83, 72.78, 106.28,
    155.23, 226.70, 331.09, 483.54, 706.18, 1031.33
);

const uint NRC_ACTIVE_HASH_LEVELS = 8u;
const float NRC_ACTIVE_SCALES[8] = float[8](
    16.0, 29.0, 52.6, 95.4, 173.0, 313.8, 569.1, 1031.33
);

uint nrcHashCorner(ivec3 c, uint level) {
    const uint P1 = 1u;
    const uint P2 = 2654435761u;
    const uint P3 = 805459861u;
    const uint P_LEVEL = 1000003u;
    uint h = (uint(c.x) * P1 ^ uint(c.y) * P2 ^ uint(c.z) * P3) ^ (level * P_LEVEL);
    h ^= (h >> 16u);
    return h & (NRC_HASH_TABLE_SIZE - 1u);
}

// Freudenthal 3D Simplex (tetrahedral) cell decomposition:
// Exact C0-continuous piecewise-linear interpolation across [0,1]^3 using 4 corners instead of 8.
void nrcSimplexCorners(vec3 p, out ivec3 c0, out ivec3 d1, out ivec3 d2, out vec4 weights4) {
    c0 = ivec3(floor(p));
    vec3 f = p - vec3(c0);
    float g0 = max(f.x, max(f.y, f.z));
    float g2 = min(f.x, min(f.y, f.z));
    float g1 = f.x + f.y + f.z - g0 - g2;
    d1 = (f.x >= f.y && f.x >= f.z) ? ivec3(1, 0, 0) : ((f.y >= f.z) ? ivec3(0, 1, 0) : ivec3(0, 0, 1));
    d2 = (f.z <= f.x && f.z <= f.y) ? ivec3(1, 1, 0) : ((f.y <= f.x) ? ivec3(1, 0, 1) : ivec3(0, 1, 1));
    weights4 = vec4(1.0 - g0, g0 - g1, g1 - g2, g2);
}

vec3 nrcClampChromaticity(vec3 c, float maxComp) {
    vec3 safe = max(c, vec3(0.0));
    float peak = max(safe.r, max(safe.g, safe.b));
    return (peak > maxComp) ? (safe * (maxComp / peak)) : safe;
}

vec3 nrcClipGradVector(vec3 g, float maxComp) {
    float peak = max(abs(g.r), max(abs(g.g), abs(g.b)));
    return (peak > maxComp) ? (g * (maxComp / peak)) : g;
}

vec3 nrcEnvBrdfApprox(vec3 F0, float roughness, float ndotv) {
    const vec4 c0 = vec4(-1.0, -0.0275, -0.572, 0.022);
    const vec4 c1 = vec4(1.0, 0.0425, 1.04, -0.04);
    vec4 r = roughness * c0 + c1;
    float a004 = min(r.x * r.x, exp2(-9.28 * ndotv)) * r.x + r.y;
    vec2 ab = vec2(-1.04, 1.04) * a004 + r.zw;
    float f0Scale = clamp(50.0 * max(F0.r, max(F0.g, F0.b)), 0.0, 1.0);
    return clamp(F0 * ab.x + ab.y * f0Scale, vec3(0.0), vec3(1.0));
}

void nrcSurfaceReflectance(
    vec3 diffuseColor,
    vec3 F0,
    float roughness,
    float clearcoat,
    float clearcoatRoughness,
    float ndotv,
    out vec3 outRefl,
    out float outSpecRatio
) {
    vec3 rDiff = max(diffuseColor, vec3(0.0));
    float ndv = clamp(ndotv, 0.0, 1.0);
    vec3 rSpec = nrcEnvBrdfApprox(max(F0, vec3(0.0)), clamp(roughness, 0.04, 1.0), ndv);
    if (clearcoat > 1e-4) {
        vec3 rCoat = nrcEnvBrdfApprox(vec3(0.04), clamp(clearcoatRoughness, 0.04, 1.0), ndv);
        float coatWeight = clamp(clearcoat, 0.0, 1.0) * 0.25;
        float fresnelCoat = 0.04 + 0.96 * pow(1.0 - ndv, 5.0);
        float baseScale = 1.0 - clamp(clearcoat * fresnelCoat, 0.0, 0.95);
        rDiff *= baseScale;
        rSpec = rSpec * baseScale + coatWeight * rCoat;
    }
    outRefl = clamp(rDiff + rSpec, vec3(0.0), vec3(1.0));
    float lumDiff = dot(rDiff, vec3(0.2126, 0.7152, 0.0722));
    float lumSpec = dot(rSpec, vec3(0.2126, 0.7152, 0.0722));
    outSpecRatio = clamp(lumSpec / max(lumDiff + lumSpec, 1e-4), 0.0, 1.0);
}

vec3 nrcDemodulateWeight(vec3 bsdfWeight, vec3 refl) {
    float lumW = dot(max(bsdfWeight, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
    float lumR = dot(max(refl, vec3(0.0)), vec3(0.2126, 0.7152, 0.0722));
    float geomW = (lumR > 1e-4) ? (lumW / lumR) : 0.0;
    vec3 safeR = max(refl, vec3(0.02));
    vec3 ratio = max(bsdfWeight, vec3(0.0)) / safeR;
    vec3 w = mix(vec3(geomW), ratio, clamp(refl / vec3(0.02), vec3(0.0), vec3(1.0)));
    return nrcClampChromaticity(w, 4.0);
}

struct NRCQuery {
    vec4 pos_roughness;    // pos.xyz, roughness (w)
    vec4 normal_flags;     // normal.xyz, flags (w)
    vec4 dir_pixelIndex;   // dir.xyz, pixelIndex (w as uintBitsToFloat)
    vec4 albedo_pad;       // albedo.rgb, pad (w)
    vec4 throughput;       // throughput.rgb, pad (w)
};

struct NRCTrainingRecord {
    vec4 pos_roughness;      // queryPos.xyz, queryRoughness (w)
    vec4 normal_flags;       // queryNormal.xyz, flags (w)
    vec4 dir_pixelIndex;     // queryDir.xyz, pixelIndex (w as uintBitsToFloat)
    vec4 albedo_pad;         // queryAlbedo.rgb, pad (w)
    vec4 throughput;         // tailBsdfWeight.rgb, hasTail (w: >0.5 to bootstrap MLP at tail vertex)
    vec4 targetRadiance;     // localSuffixDirectRadiance.rgb, pad (w)
    vec4 tailPos_roughness;  // tailPos.xyz, tailRoughness (w)
    vec4 tailPacked;         // oct(tailNormal), oct(tailDir), unorm4x8(tailAlbedo), pad
};

vec2 nrcOctEncode(vec3 n) {
    float l1 = abs(n.x) + abs(n.y) + abs(n.z);
    vec2 p = (l1 > 1e-6) ? (n.xy / l1) : vec2(0.0);
    if (n.z < 0.0) {
        vec2 s = vec2(p.x >= 0.0 ? 1.0 : -1.0, p.y >= 0.0 ? 1.0 : -1.0);
        p = (1.0 - abs(p.yx)) * s;
    }
    return clamp(p, -1.0, 1.0);
}

vec3 nrcOctDecode(vec2 e) {
    vec3 v = vec3(e.x, e.y, 1.0 - abs(e.x) - abs(e.y));
    if (v.z < 0.0) {
        vec2 s = vec2(v.x >= 0.0 ? 1.0 : -1.0, v.y >= 0.0 ? 1.0 : -1.0);
        v.xy = (1.0 - abs(v.yx)) * s;
    }
    return normalize(v);
}

uint nrcPackUnitVec3(vec3 v) {
    return packSnorm2x16(nrcOctEncode(v));
}

vec3 nrcUnpackUnitVec3(uint p) {
    return nrcOctDecode(unpackSnorm2x16(p));
}

struct NRCCountersBuffer {
    uint queryCount;
    uint trainCount;
    uint dispatchX;
    uint pad;
};

#endif // NRC_COMMON_GLSL
