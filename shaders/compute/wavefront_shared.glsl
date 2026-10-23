#ifndef WAVEFRONT_SHARED_GLSL
#define WAVEFRONT_SHARED_GLSL

// Shared C++ / GLSL ABI definitions for Wavefront Path Tracing (inspired by NoGraphicsAPI)
// Eliminates manual uint32_t pc[N] index marshaling and C++/GLSL struct drift.

#ifdef __cplusplus
#include <cstdint>
#include <cstddef>
namespace pathways {
using wf_uint = uint32_t;
#else
#define wf_uint uint
#endif

struct QueueCountersBuffer {
    wf_uint activeRayCount;
    wf_uint nextActiveCount;
    wf_uint totalProcessed;
    wf_uint shadowRayCount;
    wf_uint retiredWorkgroups;
    wf_uint currentShadowCount;
    wf_uint diffuseCount;       // Standard count
    wf_uint complexCount;
    wf_uint dielectricCount;
    wf_uint emissiveCount;
    wf_uint conductorCount;     // Reserved
    wf_uint alphamaskCount;     // Reserved
    wf_uint nextDiffuseCount;
    wf_uint nextComplexCount;
    wf_uint nextDielectricCount;
    wf_uint nextEmissiveCount;
    wf_uint nextConductorCount; // Reserved
    wf_uint nextAlphamaskCount; // Reserved
    wf_uint currentMaterialCounts[4];
    wf_uint padMat[2];
    wf_uint totalShadeWorkgroups;
    wf_uint octantCounts[8];
    wf_uint currentOctantCounts[8];
    wf_uint totalIntersectWorkgroups;
    wf_uint secondaryRayCount;
    wf_uint activeMaterialSequenceCount;
    wf_uint pad[20];
};

struct WavefrontClassifyPC {
    wf_uint numTriangles;
    wf_uint numSpheres;
    wf_uint numMaterials;
    wf_uint numLights;
    wf_uint width;
    wf_uint height;
    wf_uint useMorton;
    wf_uint hasEnvMap;
    float envMapIntensity;
    wf_uint maxQueueCapacity;
    wf_uint sppSampleIndex;
    wf_uint useHardwareRT;
    wf_uint tileOffsetX;
    wf_uint tileOffsetY;
    wf_uint tileSizeX;
    wf_uint tileSizeY;
    wf_uint sortMode;
    wf_uint numOpaqueTriangles;
    wf_uint fullWidth;
    wf_uint fullHeight;
    wf_uint captureMlData;
    wf_uint pad0;
};

struct WavefrontShadePC {
    wf_uint numTriangles;
    wf_uint numSpheres;
    wf_uint numMaterials;
    wf_uint numLights;
    wf_uint width;
    wf_uint height;
    wf_uint currentBounce;
    wf_uint maxBounces;
    wf_uint maxQueueCapacity;
    wf_uint sppSampleIndex;
    wf_uint hasEnvMap;
    float envMapIntensity;
    wf_uint useHardwareRT;
    wf_uint secondarySortMode;
    wf_uint enableNrc;
    wf_uint nrcBounce;
    float nrcTrainRatio;
    wf_uint frameIndex;
    wf_uint numOpaqueTriangles;
    wf_uint captureMlData;
    float indirectClamp;
    wf_uint deltaUnroll;
    wf_uint restirMCap;
    wf_uint minRayThreshold;
};

struct WavefrontShadowPC {
    wf_uint numTriangles;
    wf_uint numSpheres;
    wf_uint numMaterials;
    wf_uint numLights;
    wf_uint width;
    wf_uint height;
    wf_uint frameIndex;
    wf_uint sppSampleIndex;
    wf_uint numOpaqueTriangles;
    wf_uint captureMlData;
};

struct WavefrontIntersectPC {
    wf_uint numTriangles;
    wf_uint numSpheres;
    wf_uint numMaterials;
    wf_uint numLights;
    wf_uint width;
    wf_uint height;
    wf_uint currentBounce;
    wf_uint maxBounces;
    wf_uint maxQueueCapacity;
    wf_uint sppSampleIndex;
    wf_uint hasEnvMap;
    float envMapIntensity;
    wf_uint useHardwareRT;
    wf_uint sortMode;
    wf_uint numOpaqueTriangles;
    wf_uint secondarySortMode;
    wf_uint octantBin;
    float maxRayDistance;
    wf_uint minRayThreshold;
};

struct WavefrontTailPC {
    wf_uint numTriangles;
    wf_uint numSpheres;
    wf_uint numMaterials;
    wf_uint numLights;
    wf_uint width;
    wf_uint height;
    wf_uint currentBounce;
    wf_uint maxBounces;
    wf_uint maxQueueCapacity;
    wf_uint sppSampleIndex;
    wf_uint hasEnvMap;
    float envMapIntensity;
    wf_uint useHardwareRT;
    wf_uint secondarySortMode;
    wf_uint enableNrc;
    wf_uint nrcBounce;
    float nrcTrainRatio;
    wf_uint frameIndex;
    wf_uint numOpaqueTriangles;
    wf_uint captureMlData;
    float indirectClamp;
    float maxRayDistance;
};

#ifdef __cplusplus
static_assert(sizeof(QueueCountersBuffer) == 256, "QueueCountersBuffer must be 256 bytes");
static_assert(offsetof(QueueCountersBuffer, activeMaterialSequenceCount) == 172, "activeMaterialSequenceCount offset must be 172");
static_assert(sizeof(WavefrontClassifyPC) == 88, "WavefrontClassifyPC must be 88 bytes");
static_assert(sizeof(WavefrontShadePC) == 96, "WavefrontShadePC must be 96 bytes");
static_assert(sizeof(WavefrontShadowPC) == 40, "WavefrontShadowPC must be 40 bytes");
static_assert(sizeof(WavefrontIntersectPC) == 76, "WavefrontIntersectPC must be 76 bytes");
static_assert(sizeof(WavefrontTailPC) == 88, "WavefrontTailPC must be 88 bytes");
} // namespace pathways
#else
#undef wf_uint
#define WAVEFRONT_QUEUE_COUNTERS_BINDING(bind_idx) \
    layout(std430, binding = bind_idx) buffer QueueCounters { \
        QueueCountersBuffer queueCounters; \
    }
#endif

#endif // WAVEFRONT_SHARED_GLSL
