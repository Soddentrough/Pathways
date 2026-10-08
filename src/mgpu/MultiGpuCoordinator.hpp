#pragma once

#include "core/Config.hpp"
#include "scene/Camera.hpp"
#include "scene/ProceduralScene.hpp"
#include "rt/RayTracingOrchestrator.hpp"
#include "rt/PostProcessPipeline.hpp"
#include "mgpu/MultiGpuManager.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"

#include <vulkan/vulkan.h>
#include <array>
#include <vector>
#include <functional>
#include <memory>

namespace pathways {

class QualityGovernor;
class CausticsPipeline;

struct MultiGpuFramePlan {
    MultiGpuMode activeMode = MultiGpuMode::Off;
    uint32_t mergeMode = 0;
    uint32_t primSpp = 1;
    uint32_t secSpp = 0;
    uint32_t totalCompositeSpp = 0;
    uint32_t secAccumHistory = 0;
    uint32_t tileOffsetX_prim = 0;
    uint32_t tileOffsetY_prim = 0;
    uint32_t tileOffsetX_sec = 0;
    uint32_t tileOffsetY_sec = 0;
    uint32_t dispatchWidth = 0;
    uint32_t dispatchHeight = 0;
    uint32_t primDispatchWidth = 0;
    uint32_t secDispatchWidth = 0;
    uint32_t mgpuBaseW = 0;
    uint32_t mgpuBaseH = 0;
    uint32_t formatMode = 0;
    size_t frameBytes = 0;
    void* dstHost = nullptr;
    bool isSampleBlendFsr3 = false;
    bool isCheckerboard = false;
    CameraUniform uboPrim{};
    CameraUniform uboSec{};
    uint32_t slot = 0;
};

struct MultiGpuPlanParams {
    Config& config;
    MultiGpuManager* mgpu = nullptr;
    QualityGovernor* governor = nullptr;
    Camera* camera = nullptr;
    Buffer* secTransferBuffer = nullptr;
    Buffer* cameraUbo = nullptr;
    const CameraUniform& ubo;
    uint32_t currentFrame = 0;
    uint32_t frameIndex = 0;
    uint32_t activeSpp = 1;
    uint32_t activeBounces = 4;
    float activeFractionalSpp = 0.0f;
    uint32_t flags = 0;
    bool accumReset = false;
    bool cameraMovedThisFrame = false;
    bool cameraMovedLastFrame = false;
    bool hardReset = false;
    bool skipRayTracing = false;
    uint32_t renderW = 1280;
    uint32_t renderH = 720;
    uint32_t numTriangles = 0;
    uint32_t numSpheres = 0;
    uint32_t numMaterials = 0;
    uint32_t numLights = 0;
    uint32_t numOpaqueTriangles = 0;
    uint32_t hasEnvMap = 0;
    float envIntensity = 1.0f;
    uint32_t useHwRT = 1;
    bool temporalResetRequested = false;
    uint32_t accumulatedSamples = 1;
    WavefrontSortMode effectiveWavefrontSortMode = WavefrontSortMode::None;
};

class MultiGpuCoordinator {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    MultiGpuCoordinator() = default;
    ~MultiGpuCoordinator();

    MultiGpuCoordinator(const MultiGpuCoordinator&) = delete;
    MultiGpuCoordinator& operator=(const MultiGpuCoordinator&) = delete;

    /// Initializes async compute command pool and merge command buffers/semaphores if supported.
    void init(VkDevice device, uint32_t asyncComputeQueueFamily, bool hasDedicatedAsyncCompute);

    /// Destroys Vulkan resources owned by the coordinator.
    void destroy(VkDevice device);

    /// Partitions workload between primary and secondary GPUs and launches secondary async execution.
    MultiGpuFramePlan planAndLaunchSecondary(
        const MultiGpuPlanParams& params,
        uint32_t& inOutAccumulatedSamples,
        bool& inOutAccumReset
    );

    /// Records primary ray tracing passes into cmd and submits to graphics queue signaling rtCompleteSemaphore.
    void recordPrimaryRayTracing(
        VkCommandBuffer cmd,
        VkQueue queue,
        VkDevice device,
        const MultiGpuFramePlan& plan,
        const MultiGpuPlanParams& params,
        uint32_t qBase,
        VkQueryPool queryPool,
        VkSemaphore rtCompleteSemaphore,
        RayTracingOrchestrator* rtOrchestrator,
        Image* frameImage,
        Image* accumImage,
        Image* mlDiffuseImage,
        Image* mlSpecularImage,
        Image* normalDepthImage,
        Image* prevNormalDepthImage,
        VkDescriptorSet rtDescSet,
        CausticsPipeline* causticsPipeline,
        const SceneData& sceneData,
        const std::function<void(VkCommandBuffer)>& onPreRecord
    );

    /// Records merge compute pass, memory barriers, FP16 radiance copy to G-Buffer, and running avg into activeCmd.
    void recordMergePass(
        VkCommandBuffer activeCmd,
        const MultiGpuFramePlan& plan,
        const MultiGpuPlanParams& params,
        PostProcessPipeline* postProcess,
        Image* frameImage,
        Image* mlDiffuseImage
    );

    /// Awaits secondary GPU completion and syncs host/PCIe staging buffer if host fallback is active.
    void syncSecondaryTransfer(MultiGpuManager* mgpu, const MultiGpuFramePlan& plan, bool skipRayTracing);

    /// Populates semaphore wait lists for the final post-processing/presentation queue submission.
    void populatePostWaitSemaphores(
        std::vector<VkSemaphoreSubmitInfo>& waitSemaphoreInfos,
        MultiGpuManager* mgpu,
        uint32_t currentFrame,
        VkSemaphore rtCompleteSemaphore,
        bool skipRayTracing
    );

    [[nodiscard]] VkCommandPool getAsyncComputeCommandPool() const noexcept { return m_asyncComputeCommandPool; }
    [[nodiscard]] VkCommandBuffer getMergeCommandBuffer(uint32_t slot) const noexcept { return m_mergeCommandBuffers[slot]; }
    [[nodiscard]] VkSemaphore getMergeCompleteSemaphore(uint32_t slot) const noexcept { return m_mergeCompleteSemaphores[slot]; }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkCommandPool m_asyncComputeCommandPool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> m_mergeCommandBuffers = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_mergeCompleteSemaphores = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    MultiGpuMode m_lastActiveMgpuMode = MultiGpuMode::Off;
};

} // namespace pathways
