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

#endif // NRC_COMMON_GLSL
