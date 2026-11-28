#pragma once

#include "rt/WavefrontPipeline.hpp"
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
    bool envNeeEnabled = false;
};

class RayTracingOrchestrator {
public:
    RayTracingOrchestrator(
        VkDevice device,
        VmaAllocator allocator,
        VkPhysicalDevice physicalDevice,
        VkPhysicalDeviceType deviceType,
        VkPipelineCache pipelineCache = VK_NULL_HANDLE
    );
    ~RayTracingOrchestrator();

    RayTracingOrchestrator(const RayTracingOrchestrator&) = delete;
    RayTracingOrchestrator& operator=(const RayTracingOrchestrator&) = delete;

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
        const std::vector<char>& shadeComplexCode,
        const std::vector<char>& shadeEmissiveCode,
        bool hasDgcExecutionSet,
        const std::vector<char>& shadeDiffuseSecCode,
        const std::vector<char>& shadeComplexSecCode,
        bool enableDgcPreprocess,
        bool hasSubgroupSizeControl,
        uint32_t initialBatchPixels,
        const std::vector<char>& tailMegakernelCode,
        const std::vector<char>& shadeDielectricSecCode = {}
    );

    void initWavefrontSpecializedVariants(
        const std::vector<char>& intersectAlphaCode,
        const std::vector<char>& shadeDiffuseRestirCode,
        const std::vector<char>& shadeDiffuseSecRestirCode,
        const std::vector<char>& shadeComplexRestirCode,
        const std::vector<char>& shadeComplexSecRestirCode,
        const std::vector<char>& shadeDielectricRestirCode,
        const std::vector<char>& shadeEmissiveRestirCode,
        const std::vector<char>& restirGiReuseCode = {},
        const std::vector<char>& restirDiReuseCode = {},
        const std::vector<char>& shadeDielectricSecRestirCode = {},
        const std::vector<char>& classifyAlphaCode = {},
        const std::vector<char>& shadowAlphaCode = {}
    );

    void initWavefrontAlphaPipeline(
        const std::vector<char>& intersectAlphaCode,
        const std::vector<char>& classifyAlphaCode = {},
        const std::vector<char>& shadowAlphaCode = {});
    void initWavefrontTailMegakernelPipeline(const std::vector<char>& tailMegakernelCode);
    void initWavefrontRestirVariants(
        const std::vector<char>& shadeDiffuseRestirCode,
        const std::vector<char>& shadeDiffuseSecRestirCode,
        const std::vector<char>& shadeComplexRestirCode,
        const std::vector<char>& shadeComplexSecRestirCode,
        const std::vector<char>& shadeDielectricRestirCode,
        const std::vector<char>& shadeEmissiveRestirCode,
        const std::vector<char>& restirGiReuseCode = {},
        const std::vector<char>& restirDiReuseCode = {},
        const std::vector<char>& shadeDielectricSecRestirCode = {}
    );
    void initWavefrontNrcVariants(
        const std::vector<char>& shadeDiffuseNrcCode,
        const std::vector<char>& shadeDiffuseSecNrcCode,
        const std::vector<char>& shadeComplexNrcCode,
        const std::vector<char>& shadeComplexSecNrcCode,
        const std::vector<char>& shadeDielectricNrcCode,
        const std::vector<char>& shadeDielectricSecNrcCode
    );

    /// Initializes NRCManager.
    void initNRC(
        uint32_t width,
        uint32_t height,
        const std::vector<char>& inferCode,
        const std::vector<char>& trainCode,
        const std::vector<char>& resolveCode
    );
    void releaseNRC() noexcept { m_nrcManager.reset(); }

    /// Initializes ReSTIRManager (reservoir grid allocation is deferred until
    /// ReSTIR is first enabled; see Engine lazy-init path).
    void createReSTIRResources();

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

    /// Records ray tracing dispatches (Wavefront).
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
        CausticsPipeline* causticsPipeline,
        QualityGovernor* governor,
        Camera* camera,
        const SceneData& sceneData,
        uint32_t numTriangles,
        uint32_t numMaterials,
        uint32_t numLights,
        uint32_t numOpaqueTriangles,
        uint32_t hasEnvMap,
        float envIntensity,
        const Config& config,
        Image* mlSkipImage = nullptr,
        Image* mlSpecAlbedoImage = nullptr,
        Image* mlDirectDiffuseImage = nullptr,
        Image* mlIndirectDiffuseImage = nullptr,
        Image* mlDirectSpecularImage = nullptr,
        Image* mlIndirectSpecularImage = nullptr,
        Image* mlIrradianceImage = nullptr,
        Image* mlPrimaryGuideImage = nullptr,
        Image* mlSurfaceMvImage = nullptr,
        Image* mlDiffHitDistImage = nullptr,
        Image* mlSpecVmvImage = nullptr,
        Image* mlSkipCompactImage = nullptr,
        Image* motionVectorImage = nullptr
    );

    // Getters
    [[nodiscard]] WavefrontPipeline* getWavefrontPipeline() const noexcept { return m_wavefrontPipeline.get(); }
    [[nodiscard]] NRCManager* getNrcManager() const noexcept { return m_nrcManager.get(); }
    [[nodiscard]] ReSTIRManager* getRestirManager() const noexcept { return m_restirManager.get(); }
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
    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;

    std::unique_ptr<WavefrontPipeline> m_wavefrontPipeline;
    std::unique_ptr<NRCManager> m_nrcManager;
    std::unique_ptr<ReSTIRManager> m_restirManager;

    uint32_t m_currentBatchCount = 0;
    uint32_t m_currentBatchPixels = 0;
    mutable float m_cachedDivergentAreaRatio = -1.0f;
};

} // namespace pathways
