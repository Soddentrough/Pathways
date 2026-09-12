#ifndef NRC_COMMON_GLSL
#define NRC_COMMON_GLSL

// Neural Radiance Caching (NRC) Wave32 WMMA Architecture
// Target: AMD RDNA 4 (gfx1201), Vulkan 1.4

const uint NRC_LEVELS = 12u;
const uint NRC_TABLE_SIZE = 262144u; // 2^18 entries per level
const uint NRC_TOTAL_ENTRIES = NRC_LEVELS * NRC_TABLE_SIZE; // 3,145,728 entries

const float NRC_GRID_SCALES[12] = float[12](
    16.0,
    24.8295,
    38.5316,
    59.7950,
    92.7928,
    144.0000,
    223.4658,
    346.7845,
    538.1565,
    835.1350,
    1295.9967,
    2011.1893
);

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
    vec4 pos_roughness;    // pos.xyz, roughness (w)
    vec4 normal_flags;     // normal.xyz, flags (w)
    vec4 dir_pixelIndex;   // dir.xyz, pixelIndex (w as uintBitsToFloat)
    vec4 albedo_pad;       // albedo.rgb, pad (w)
    vec4 throughput;       // throughput.rgb, pad (w)
    vec4 targetRadiance;   // targetRadiance.rgb, pad (w)
};

struct NRCCountersBuffer {
    uint queryCount;
    uint trainCount;
    uint dispatchX;
    uint pad;
};

// Spatial hash for voxel corner
uint nrcHashCorner(ivec3 p, uint level) {
    uint h = (uint(p.x) * 1u) ^ (uint(p.y) * 2654435761u) ^ (uint(p.z) * 805459861u);
    h = h & (NRC_TABLE_SIZE - 1u);
    return level * NRC_TABLE_SIZE + h;
}

#endif // NRC_COMMON_GLSL
