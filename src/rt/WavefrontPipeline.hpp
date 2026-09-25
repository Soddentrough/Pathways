#pragma once

#include <vulkan/vulkan.h>
#include "vulkan/Buffer.hpp"
#include "rt/DGCManager.hpp"
#include "scene/Camera.hpp"
#include <memory>
#include <vector>
#include <array>
#include <string>
#include <functional>

namespace pathways {

class Image;

struct QueueCountersBuffer {
    uint32_t activeRayCount;
    uint32_t nextActiveCount;
    uint32_t totalProcessed;
    uint32_t shadowRayCount;
    uint32_t retiredWorkgroups;
    uint32_t currentShadowCount;
    uint32_t diffuseCount; // Standard count
    uint32_t complexCount;
    uint32_t dielectricCount;
    uint32_t emissiveCount;
    uint32_t conductorCount; // Reserved
    uint32_t alphamaskCount; // Reserved
    uint32_t nextDiffuseCount;
    uint32_t nextComplexCount;
    uint32_t nextDielectricCount;
    uint32_t nextEmissiveCount;
    uint32_t nextConductorCount; // Reserved
    uint32_t nextAlphamaskCount; // Reserved
    uint32_t currentMaterialCounts[4];
    uint32_t padMat[2];
    uint32_t totalShadeWorkgroups;
    uint32_t octantCounts[8];
    uint32_t currentOctantCounts[8];
    uint32_t totalIntersectWorkgroups;
    uint32_t secondaryRayCount;
    uint32_t activeMaterialSequenceCount;
    uint32_t pad[20];
};
static_assert(sizeof(QueueCountersBuffer) == 256, "QueueCountersBuffer must be 256 bytes");
static_assert(offsetof(QueueCountersBuffer, activeMaterialSequenceCount) == 172, "activeMaterialSequenceCount offset must be 172");

struct WavefrontSceneData {
    uint32_t numTriangles = 0;
    uint32_t numSpheres = 0;
    uint32_t numMaterials = 0;
    uint32_t numLights = 0;
    uint32_t hasEnvMap = 0;
    float envMapIntensity = 1.0f;
    uint32_t useHardwareRT = 1;
    uint32_t frameIndex = 0;
    uint32_t useMorton = 1;
    uint32_t accumulateHistory = 1;
    uint32_t sortMode = 0; // 0: None, 1: Archetype, 2: BDA, 3: Dual
    uint32_t numOpaqueTriangles = 0;
    uint32_t secondarySortMode = 0; // 0: None, 1: Directional DGC, 2: Spatial Index
    uint32_t cameraFlags = 0;
    bool enableNrc = false;
    uint32_t nrcBounce = 2;
    float nrcTrainRatio = 0.03f;
    glm::vec3 boundsMin = glm::vec3(-1000.0f);
    glm::vec3 boundsMax = glm::vec3(1000.0f);
    bool streamlineSecondaryShading = true;
    bool enableDistanceClamping = true;
    float maxSecondaryRayDistance = 0.0f; // 0 = automatic scene bounding diameter * 1.25
    uint32_t tileOffsetX = 0;             // Multi-GPU Checkerboard: 0 = full frame / sample parallel, 1 = primary even tiles, 2 = secondary odd tiles
    uint32_t tileOffsetY = 0;             // Multi-GPU Checkerboard tile size (16, 32, 64, 128; defaults to 64 if 0)
    uint32_t fullWidth = 0;               // Full unclipped frame resolution width
    uint32_t fullHeight = 0;              // Full unclipped frame resolution height
    uint32_t captureMlData = 0;           // ML training data capture flag (demodulated buffers)
    float indirectClamp = 35.0f;          // Maximum indirect / secondary bounce radiance luminance (0 = disabled)
    bool inlineShadows = false;           // Detached shadow queue evaluation (Default: false for max occupancy & 0 LDS)
    uint32_t macroTileSize = 0;           // Legacy macro-tile cache panning (deprecated in favor of coarse batches)
    uint32_t batchCount = 0;              // Coarse batch count (0 = auto-detect, 1 = monolithic, 2, 4, 8...)
    uint32_t batchPixels = 0;             // Coarse batch ray budget in pixels (0 = auto-detect)
};

class WavefrontPipeline {
public:
    static constexpr uint32_t MAX_SCENE_TEXTURES = 512;
    static constexpr uint32_t MAX_WAVEFRONT_TIMESTAMP_QUERIES = 512;

    WavefrontPipeline(VkDevice device, VmaAllocator allocator,
                      uint32_t width, uint32_t height,
                      const std::vector<char>& classifyCode,
                      const std::vector<char>& intersectCode,
                      const std::vector<char>& shadeCode,
                      const std::vector<char>& shadowCode,
                      const std::vector<char>& shadeDiffuseCode = {},
                      const std::vector<char>& shadeDielectricCode = {},
                      const std::vector<char>& shadeConductorCode = {},
                      const std::vector<char>& shadeComplexCode = {},
                      const std::vector<char>& shadeEmissiveCode = {},
                      const std::vector<char>& shadePassthroughCode = {},
                      bool supportsExecutionSet = false,
                      const std::vector<char>& shadeDiffuseSecCode = {},
                      const std::vector<char>& shadeComplexSecCode = {},
                      bool enableDgcPreprocess = true,
                      bool supportsSubgroupSizeControl = true,
                      uint32_t maxBatchPixels = 0);
    ~WavefrontPipeline();

    WavefrontPipeline(const WavefrontPipeline&) = delete;
    WavefrontPipeline& operator=(const WavefrontPipeline&) = delete;

    void updateSceneDescriptors(uint32_t frameSlot,
                                VkImageView accumImageView,
                                VkBuffer cameraUBO,
                                VkBuffer triangleBuffer, VkDeviceSize triSize,
                                VkBuffer sphereBuffer, VkDeviceSize sphereSize,
                                VkBuffer matBuffer, VkDeviceSize matSize,
                                VkBuffer lightBuffer, VkDeviceSize lightSize,
                                VkAccelerationStructureKHR tlas,
                                VkDescriptorImageInfo envMapInfo,
                                const std::vector<VkDescriptorImageInfo>& sceneTexInfos,
                                VkBuffer nrcQueryBuffer = VK_NULL_HANDLE,
                                VkBuffer nrcTrainBuffer = VK_NULL_HANDLE,
                                VkBuffer nrcCountersBuffer = VK_NULL_HANDLE,
                                VkImageView motionVectorImageView = VK_NULL_HANDLE,
                                VkImageView normalDepthImageView = VK_NULL_HANDLE,
                                VkBuffer lightTreeBuffer = VK_NULL_HANDLE,
                                VkDeviceSize lightTreeSize = 0,
                                VkImageView mlAlbedoRoughnessImageView = VK_NULL_HANDLE,
                                VkImageView mlSpecularMotionImageView = VK_NULL_HANDLE,
                                VkImageView mlDiffuseImageView = VK_NULL_HANDLE,
                                VkImageView mlSpecularImageView = VK_NULL_HANDLE,
                                VkBuffer instanceBuffer = VK_NULL_HANDLE,
                                VkDeviceSize instanceSize = 0,
                                VkImageView causticImageView = VK_NULL_HANDLE,
                                VkBuffer restirReservoirBuffer = VK_NULL_HANDLE,
                                VkBuffer materialArchetypeBuffer = VK_NULL_HANDLE,
                                VkDeviceSize matArchetypeSize = 0,
                                VkBuffer shadeMaterialBuffer = VK_NULL_HANDLE,
                                VkDeviceSize shadeMaterialSize = 0);

    void resize(uint32_t width, uint32_t height, uint32_t maxBatchPixels = 0);
    uint32_t getMaxCapacity() const { return m_maxCapacity; }

    void recordFrame(VkCommandBuffer cmd, uint32_t frameSlot, uint32_t width, uint32_t height,
                     uint32_t spp, uint32_t maxBounces,
                     const WavefrontSceneData& sceneData);

    struct BounceProfilingData {
        uint32_t bounce = 0;
        double shadeMs = 0.0;
        double shadowMs = 0.0;
        double intersectMs = 0.0;
        uint32_t activeCount = 0;
        uint32_t shadowCount = 0;
        uint32_t nextCount = 0;
        uint32_t diffCount = 0;
        uint32_t dielCount = 0;
        uint32_t condCount = 0;
        uint32_t compCount = 0;
        uint32_t emisCount = 0;
        uint32_t passCount = 0;
    };

    struct WavefrontProfilingData {
        bool valid = false;
        double totalMs = 0.0;
        double classifyMs = 0.0;
        double resolveMs = 0.0;
        std::vector<BounceProfilingData> bounces;
        double queueMemoryFootprintMb = 0.0;
        double estimatedVramTrafficMb = 0.0;
        uint32_t sortMode = 0;
        uint32_t secondarySortMode = 0;
    };

    void printProfilingBreakdown(uint32_t frameSlot, double timestampPeriodNs, uint32_t maxBounces = 0);
    WavefrontProfilingData getProfilingData(uint32_t frameSlot, double timestampPeriodNs, uint32_t maxBounces = 0);
    double getQueueMemoryFootprintMb() const;

    VkPipelineLayout getPipelineLayout() const { return m_pipelineLayout; }
    DGCManager* getDGCManager() const { return m_dgcManager.get(); }
    bool supportsExecutionSet() const { return m_supportsExecutionSet; }

    Buffer* getRayGeomQueue(uint32_t slot) const { return m_rayGeomQueueA[slot].get(); }
    Buffer* getRayHitQueue(uint32_t slot) const { return m_rayHitQueue[slot].get(); }
    Buffer* getPixelToRayQueue(uint32_t slot) const { return m_pixelToRayQueue[slot].get(); }

    using PostClassifyCallback = std::function<void(VkCommandBuffer cmd, uint32_t frameSlot)>;
    void setPostClassifyCallback(PostClassifyCallback cb) { m_postClassifyCallback = std::move(cb); }

private:
    PostClassifyCallback m_postClassifyCallback = nullptr;
    void createDescriptorLayout();
    void allocateDescriptorSets();
    void allocateQueues(uint32_t capacity);
    void updateQueueDescriptors();
    void createPipelines(const std::vector<char>& classifyCode,
                         const std::vector<char>& intersectCode,
                         const std::vector<char>& shadeCode,
                         const std::vector<char>& shadowCode,
                         const std::vector<char>& shadeDiffuseCode,
                         const std::vector<char>& shadeDielectricCode,
                         const std::vector<char>& shadeConductorCode,
                         const std::vector<char>& shadeComplexCode,
                         const std::vector<char>& shadeEmissiveCode,
                         const std::vector<char>& shadePassthroughCode,
                         const std::vector<char>& shadeDiffuseSecCode = {},
                         const std::vector<char>& shadeComplexSecCode = {});

    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_maxCapacity = 0;
    uint32_t m_maxBatchPixels = 0;
    uint32_t m_sortMode = 0;
    uint32_t m_secondarySortMode = 0;
    bool m_supportsExecutionSet = false;
    bool m_enableDgcPreprocess = true;

    // Ray Work Queues & Counter SSBOs (SoA Layout) - Double-buffered per in-flight frame slot
    std::array<std::unique_ptr<Buffer>, 2> m_rayGeomQueueA;  // 16B RayGeometry
    std::array<std::unique_ptr<Buffer>, 2> m_rayGeomQueueB;  // 16B RayGeometry
    std::array<std::unique_ptr<Buffer>, 2> m_rayStateQueueA; // 16B PackedRayState
    std::array<std::unique_ptr<Buffer>, 2> m_rayStateQueueB; // 16B PackedRayState
    std::array<std::unique_ptr<Buffer>, 2> m_rayHitQueue;    // 16B PackedRayHit
    std::array<std::unique_ptr<Buffer>, 2> m_materialIndexQueue; // 4B index * 6 archetypes (Index-Based Material Queues)
    std::array<std::unique_ptr<Buffer>, 2> m_secondaryIndexQueue; // 4B index * 8 octants (Directional DGC Queues)
    std::array<std::unique_ptr<Buffer>, 2> m_shadowQueue;    // 32B PackedShadowRay
    std::array<std::unique_ptr<Buffer>, 2> m_queueCounters;
    std::array<std::unique_ptr<Buffer>, 2> m_indirectArgs;   // Double-buffered per in-flight frame slot
    std::array<std::unique_ptr<Buffer>, 2> m_dgcStream;
    std::array<std::unique_ptr<Buffer>, 2> m_pixelToRayQueue; // 4B pixel-to-ray mapping for ReSTIR DI

    // Descriptors
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descSetLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> m_descSetsEven = { VK_NULL_HANDLE, VK_NULL_HANDLE }; // In: A, Out: B
    std::array<VkDescriptorSet, 2> m_descSetsOdd = { VK_NULL_HANDLE, VK_NULL_HANDLE };  // In: B, Out: A

    // Pipelines
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_classifyPipeline = VK_NULL_HANDLE;
    VkPipeline m_intersectPipeline = VK_NULL_HANDLE;
    VkPipeline m_shadePipeline = VK_NULL_HANDLE;
    VkPipeline m_shadowPipeline = VK_NULL_HANDLE;
    VkPipeline m_shadeDiffusePipeline = VK_NULL_HANDLE;
    VkPipeline m_shadeDielectricPipeline = VK_NULL_HANDLE;
    VkPipeline m_shadeConductorPipeline = VK_NULL_HANDLE;
    VkPipeline m_shadeComplexPipeline = VK_NULL_HANDLE;
    VkPipeline m_shadeEmissivePipeline = VK_NULL_HANDLE;
    VkPipeline m_shadePassthroughPipeline = VK_NULL_HANDLE;
    VkPipeline m_shadeDiffuseSecPipeline = VK_NULL_HANDLE;
    VkPipeline m_shadeComplexSecPipeline = VK_NULL_HANDLE;

    std::vector<VkPipeline> m_primaryMatPipelines;
    std::vector<VkPipeline> m_secondaryMatPipelines;

    std::array<VkQueryPool, 2> m_queryPools = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<bool, 2> m_hasRecordedSlot = { false, false };
    std::array<uint32_t, 2> m_slotBounces = { 0, 0 };

    std::unique_ptr<DGCManager> m_dgcManager;
    std::unique_ptr<Image> m_dummyStorageImage;
    bool m_dummyImageTransitioned = false;
    bool m_supportsSubgroupSizeControl = true;
};

} // namespace pathways
