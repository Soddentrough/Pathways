#pragma once

#include "rt/WavefrontPipeline.hpp"
#include "rt/RTPipeline.hpp"
#include "rt/NRCManager.hpp"
#include "rt/ReSTIRManager.hpp"
#include "rt/CausticsPipeline.hpp"
#include "core/Config.hpp"
#include "scene/SceneManager.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "core/QualityGovernor.hpp"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <memory>
#include <vector>
#include <functional>

namespace pathways {

class VulkanContext;
class Camera;

struct RayTracingDispatchParams {
    uint32_t frameSlot = 0;
    uint32_t dispatchWidth = 0;
    uint32_t dispatchHeight = 0;
    uint32_t fullWidth = 0;
    uint32_t fullHeight = 0;
    uint32_t tileOffsetX = 0;
    uint32_t tileOffsetY = 0;
    uint32_t spp = 1;
    uint32_t bounces = 4;
    float fractionalSpp = 1.0f;
    uint32_t totalCompositeSpp = 0;
    uint32_t cameraFlags = 0;
    bool accumReset = false;
    bool isMultiGpu = false;
    MultiGpuMode mgpuMode = MultiGpuMode::Off;
    bool skipRayTracing = false;
    uint32_t frameIndex = 0;
    bool cameraMovedLastFrame = false;
    bool diagnosticHalfTiles = false;
    bool isRestirActive = false;
};

class RayTracingOrchestrator {
public:
    RayTracingOrchestrator(
        VkDevice device,
        VmaAllocator allocator,
        VkPhysicalDevice physicalDevice,
        VkPhysicalDeviceType deviceType
    );
    ~RayTracingOrchestrator();

    RayTracingOrchestrator(const RayTracingOrchestrator&) = delete;
    RayTracingOrchestrator& operator=(const RayTracingOrchestrator&) = delete;

    /// Initializes RTPipeline and its pipeline layout.
    void initRTPipeline(
        VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps,
        VkDescriptorSetLayout rtDescLayout,
        const std::vector<char>& rgenCode,
        const std::vector<char>& rmissCode,
        const std::vector<char>& shadowMissCode,
        const std::vector<char>& rchitCode,
        bool hasRtSubgroupSizeControl
    );

    /// Initializes WavefrontPipeline with all shader modules and DGC settings.
    void initWavefrontPipeline(
        uint32_t width,
        uint32_t height,
        const std::vector<char>& classifyCode,
        const std::vector<char>& intersectCode,
        const std::vector<char>& shadeCode,
        const std::vector<char>& shadowCode,
        const std::vector<char>& shadeDiffuseCode,
        const std::vector<char>& shadeDielectricCode,
        const std::vector<char>& shadeConductorCode,
        const std::vector<char>& shadeComplexCode,
        const std::vector<char>& shadeEmissiveCode,
        const std::vector<char>& shadePassthroughCode,
        bool hasDgcExecutionSet,
        const std::vector<char>& shadeDiffuseSecCode,
        const std::vector<char>& shadeComplexSecCode,
        bool enableDgcPreprocess,
        bool hasSubgroupSizeControl,
        uint32_t initialBatchPixels,
        const std::vector<char>& tailMegakernelCode
    );

    /// Initializes NRCManager.
    void initNRC(
        uint32_t width,
        uint32_t height,
        const std::vector<char>& inferCode,
        const std::vector<char>& trainCode,
        const std::vector<char>& resolveCode
    );

    /// Initializes ReSTIRManager.
    void createReSTIRResources(
        uint32_t width,
        uint32_t height,
        const std::vector<char>& temporalCode,
        const std::vector<char>& spatialCode
    );

    /// Destroys ReSTIRManager.
    void destroyReSTIRResources();

    /// Sizing calculations for Wavefront batches.
    uint32_t getTargetBatchPixels(const Config& config) const;
    uint32_t getEffectiveBatchCount(uint32_t renderW, uint32_t renderH, const Config& config) const;
    uint32_t getEffectiveBatchPixels(uint32_t renderW, uint32_t renderH, uint32_t batchCount) const;

    /// Evaluates divergent surface area ratio.
    float getDivergentAreaRatio(const SceneData& sceneData) const;
    void invalidateDivergentAreaCache() const noexcept { m_cachedDivergentAreaRatio = -1.0f; }

    /// Evaluates effective sorting mode.
    WavefrontSortMode getEffectiveWavefrontSortMode(const Config& config, const SceneData& sceneData) const;

    /// Resizes all ray tracing subsystems.
    void resize(uint32_t width, uint32_t height, const Config& config);

    /// Records ray tracing dispatches (Wavefront or RTPipeline).
    void recordRayTracing(
        VkCommandBuffer cmd,
        const RayTracingDispatchParams& params,
        Image* frameImage,
        Image* accumImage,
        Image* mlDiffuseImage,
        Image* mlSpecularImage,
        Image* normalDepthImage,
        Image* prevNormalDepthImage,
        Buffer* centerDepthBuffer,
        VkDescriptorSet rtDescSet,
        CausticsPipeline* causticsPipeline,
        QualityGovernor* governor,
        Camera* camera,
        const SceneData& sceneData,
        uint32_t numTriangles,
        uint32_t numSpheres,
        uint32_t numMaterials,
        uint32_t numLights,
        uint32_t numOpaqueTriangles,
        uint32_t hasEnvMap,
        float envIntensity,
        uint32_t useHwRT,
        const Config& config
    );

    // Getters
    [[nodiscard]] WavefrontPipeline* getWavefrontPipeline() const noexcept { return m_wavefrontPipeline.get(); }
    [[nodiscard]] RTPipeline* getRtpPipeline() const noexcept { return m_rtpKhrPipeline.get(); }
    [[nodiscard]] NRCManager* getNrcManager() const noexcept { return m_nrcManager.get(); }
    [[nodiscard]] ReSTIRManager* getRestirManager() const noexcept { return m_restirManager.get(); }
    [[nodiscard]] VkPipelineLayout getRtpPipelineLayout() const noexcept { return m_rtpPipelineLayout; }
    [[nodiscard]] uint32_t getCurrentBatchCount() const noexcept { return m_currentBatchCount; }
    [[nodiscard]] uint32_t getCurrentBatchPixels() const noexcept { return m_currentBatchPixels; }
    void setCurrentBatchParams(uint32_t count, uint32_t pixels) noexcept {
        m_currentBatchCount = count;
        m_currentBatchPixels = pixels;
    }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkPhysicalDeviceType m_deviceType = VK_PHYSICAL_DEVICE_TYPE_OTHER;

    VkPipelineLayout m_rtpPipelineLayout = VK_NULL_HANDLE;

    std::unique_ptr<RTPipeline> m_rtpKhrPipeline;
    std::unique_ptr<WavefrontPipeline> m_wavefrontPipeline;
    std::unique_ptr<NRCManager> m_nrcManager;
    std::unique_ptr<ReSTIRManager> m_restirManager;

    uint32_t m_currentBatchCount = 0;
    uint32_t m_currentBatchPixels = 0;
    mutable float m_cachedDivergentAreaRatio = -1.0f;
};

} // namespace pathways
