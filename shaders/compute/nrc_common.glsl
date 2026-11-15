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
