#pragma once

#include "core/Config.hpp"
#include "core/Window.hpp"
#include "vulkan/VulkanContext.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "vulkan/Swapchain.hpp"
#include "scene/Camera.hpp"
#include "scene/ProceduralScene.hpp"
#include "utils/ImageDumper.hpp"
#include "rt/AccelerationStructure.hpp"
#include "rt/DGCManager.hpp"
#include "vulkan/Texture.hpp"

#include <memory>
#include <vector>
#include <chrono>

namespace pathways {

class Engine {
public:
    Engine(const Config& config);
    ~Engine();

    void run();
    void renderFrame();
    void dumpOutputFiles();
    FrameStats getStats() const;

    void setCameraMode(bool active);
    bool isCameraMode() const { return m_cameraMode; }

private:
    void initVulkan();
    void initScene();
    void initPipelines();
    void initWavefrontResources();
    void initWavefrontPipelines();
    void initSyncObjects();
    void initQueryPool();

    bool handleEvent(const SDL_Event& e);
    void updateInput();

    VkShaderModule createShaderModule(const std::vector<char>& code);
    std::vector<char> loadShaderSPIRV(const std::string& filename);
    void onResize(uint32_t newWidth, uint32_t newHeight);

    bool m_cameraMode = false;
    bool m_resetAccumulation = false;
    bool m_pendingToggleFullscreen = false;
    uint32_t m_pendingResizeW = 0;
    uint32_t m_pendingResizeH = 0;

    Config m_config;
    std::unique_ptr<Window> m_window;
    std::unique_ptr<VulkanContext> m_context;
    std::unique_ptr<Swapchain> m_swapchain;
    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<class GuiManager> m_gui;
    std::unique_ptr<class MultiGpuManager> m_mgpu;

    // GPU Buffers
    std::unique_ptr<Buffer> m_triangleBuffer;
    std::unique_ptr<Buffer> m_sphereBuffer;
    std::unique_ptr<Buffer> m_materialBuffer;
    std::unique_ptr<Buffer> m_lightBuffer;
    std::unique_ptr<Buffer> m_cameraUBO;
    std::unique_ptr<Buffer> m_uiDumpBuffer;

    // Hardware Acceleration Structures (VK_KHR_ray_query)
    std::unique_ptr<Buffer> m_asVertexBuffer;
    std::unique_ptr<AccelerationStructureManager> m_asManager;
    std::unique_ptr<AccelerationStructure> m_blas;
    std::unique_ptr<AccelerationStructure> m_tlas;

    // Device-Generated Commands (VK_EXT_device_generated_commands)
    std::unique_ptr<DGCManager> m_dgc;
    std::unique_ptr<Buffer> m_dgcArgumentBuffer;

    // Textures & Environment Map (Bindings 7 & 8)
    static constexpr uint32_t MAX_SCENE_TEXTURES = 64;
    std::unique_ptr<Texture> m_dummyWhite;
    std::unique_ptr<Texture> m_dummyNormal;
    std::unique_ptr<Texture> m_environmentMap;
    std::vector<std::unique_ptr<Texture>> m_sceneTextures;

    // Render Targets
    std::unique_ptr<Image> m_accumImage;
    std::unique_ptr<Image> m_outputImage;

    // Descriptors & Pipelines
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_rtDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_tonemapDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_rtDescSet = VK_NULL_HANDLE;
    VkDescriptorSet m_tonemapDescSet = VK_NULL_HANDLE;

    VkPipelineLayout m_rtPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_tonemapPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_rtPipeline = VK_NULL_HANDLE;
    VkPipeline m_tonemapPipeline = VK_NULL_HANDLE;

    // Wavefront Compaction Queues & Buffers
    std::unique_ptr<Buffer> m_rayQueueA;
    std::unique_ptr<Buffer> m_rayQueueB;
    std::unique_ptr<Buffer> m_wavefrontCounters;
    std::unique_ptr<Buffer> m_wavefrontIndirectCmd;
    std::unique_ptr<Buffer> m_wavefrontDgcStream;
    std::unique_ptr<Buffer> m_wavefrontDgcCount;

    // Wavefront Pipelines & Descriptors
    VkDescriptorSetLayout m_wfClassifyDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_wfResolveDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_wfShadeDescLayout = VK_NULL_HANDLE;

    VkDescriptorSet m_wfClassifyDescSet = VK_NULL_HANDLE;
    VkDescriptorSet m_wfResolveDescSet = VK_NULL_HANDLE;
    VkDescriptorSet m_wfShadeDescSetA = VK_NULL_HANDLE; // Reads A, writes B
    VkDescriptorSet m_wfShadeDescSetB = VK_NULL_HANDLE; // Reads B, writes A

    VkPipelineLayout m_wfClassifyPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_wfResolvePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_wfShadePipelineLayout = VK_NULL_HANDLE;

    VkPipeline m_wfClassifyPipeline = VK_NULL_HANDLE;
    VkPipeline m_wfResolvePipeline = VK_NULL_HANDLE;
    VkPipeline m_wfShadePipeline = VK_NULL_HANDLE;

    // Commands & Synchronization
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    VkCommandBuffer m_commandBuffer = VK_NULL_HANDLE;
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    VkFence m_inFlightFence = VK_NULL_HANDLE;
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    uint32_t m_currentFrame = 0;
    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;

    // Timestamp Profiling
    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    float m_timestampPeriod = 1.0f; // ns per tick

    // Multi-GPU Transfer & Merge Resources
    std::unique_ptr<Buffer> m_secTransferBuffer;
    VkDescriptorSetLayout m_mergeDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_mergeDescSet = VK_NULL_HANDLE;
    VkPipelineLayout m_mergePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_mergePipeline = VK_NULL_HANDLE;
    VkFence m_rtFence = VK_NULL_HANDLE;

    // Scene metadata
    SceneData m_sceneData;
    uint32_t m_numTriangles = 0;
    uint32_t m_numSpheres = 0;
    uint32_t m_numMaterials = 0;
    uint32_t m_numLights = 0;

    // Frame tracking
    uint32_t m_frameIndex = 0;
    uint32_t m_totalFramesRendered = 0;
    std::vector<double> m_frameTimesMs;
    double m_lastFrameTimeMs = 0.0;
    double m_lastGpuRtMs = 0.0;
    double m_lastSecGpuMs = 0.0;
    double m_lastTonemapMs = 0.0;
    std::chrono::high_resolution_clock::time_point m_startTime;
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    std::chrono::steady_clock::time_point m_lastLogTime;
};

} // namespace pathways
