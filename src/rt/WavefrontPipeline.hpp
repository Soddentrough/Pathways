#pragma once

#include <vulkan/vulkan.h>
#include "vulkan/Buffer.hpp"
#include "rt/DGCManager.hpp"
#include "scene/Camera.hpp"
#include <memory>
#include <vector>
#include <array>
#include <string>

namespace pathways {

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
};

class WavefrontPipeline {
public:
    static constexpr uint32_t MAX_SCENE_TEXTURES = 512;
    static constexpr uint32_t MAX_WAVEFRONT_TIMESTAMP_QUERIES = 512;

    WavefrontPipeline(VkDevice device, VmaAllocator allocator,
                      uint32_t width, uint32_t height,
                      uint32_t tileSize,
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
                      const std::vector<char>& raySortCode = {},
                      bool supportsExecutionSet = false,
                      const std::vector<char>& shadeDiffuseSecCode = {},
                      const std::vector<char>& shadeComplexSecCode = {});
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
                                VkImageView normalDepthImageView = VK_NULL_HANDLE);

    void resize(uint32_t width, uint32_t height, uint32_t tileSize = 256);
    void setTileSize(uint32_t tileSize);
    uint32_t getTileSize() const { return m_tileSize; }

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

private:
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
                         const std::vector<char>& raySortCode,
                         const std::vector<char>& shadeDiffuseSecCode = {},
                         const std::vector<char>& shadeComplexSecCode = {});

    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_tileSize = 256;
    uint32_t m_maxCapacity = 0;
    uint32_t m_sortMode = 0;
    uint32_t m_secondarySortMode = 0;
    bool m_supportsExecutionSet = false;

    // Ray Work Queues & Counter SSBOs (SoA Layout)
    std::unique_ptr<Buffer> m_rayGeomQueueA;  // 32B RayGeometry
    std::unique_ptr<Buffer> m_rayGeomQueueB;  // 32B RayGeometry
    std::unique_ptr<Buffer> m_rayStateQueueA; // 32B RayState
    std::unique_ptr<Buffer> m_rayStateQueueB; // 32B RayState
    std::unique_ptr<Buffer> m_rayHitQueue;    // 32B RayHit
    std::unique_ptr<Buffer> m_materialIndexQueue; // 4B index * 4 archetypes (Index-Based Material Queues)
    std::unique_ptr<Buffer> m_secondaryIndexQueue; // 4B index * 8 octants (Secondary Ray Index Queue)
    std::unique_ptr<Buffer> m_shadowQueue;    // 32B PackedShadowRay
    std::unique_ptr<Buffer> m_queueCounters;
    std::array<std::unique_ptr<Buffer>, 2> m_indirectArgs; // Double-buffered per in-flight frame slot
    std::unique_ptr<Buffer> m_dgcStream;

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
    VkPipeline m_raySortPipeline = VK_NULL_HANDLE;

    std::vector<VkPipeline> m_primaryMatPipelines;
    std::vector<VkPipeline> m_secondaryMatPipelines;

    std::array<VkQueryPool, 2> m_queryPools = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<bool, 2> m_hasRecordedSlot = { false, false };
    std::array<uint32_t, 2> m_slotBounces = { 0, 0 };

    std::unique_ptr<DGCManager> m_dgcManager;
};

} // namespace pathways
