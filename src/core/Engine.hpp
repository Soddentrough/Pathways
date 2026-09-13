#pragma once

#include "core/Config.hpp"
#include "core/ConfigTally.hpp"
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
#include "rt/RTPipeline.hpp"
#include "rt/WavefrontPipeline.hpp"
#include "rt/NRCManager.hpp"
#include "vulkan/Texture.hpp"
#include "scene/SceneRegistry.hpp"
#include "core/QualityGovernor.hpp"

#include <memory>
#include <vector>
#include <array>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <future>

namespace pathways {

class Engine {
public:
    Engine(const Config& config);
    ~Engine();

    void run();
    void renderFrame();
    void dumpOutputFiles();
    void printExecutionSummary() const;
    const std::vector<ConfigStatsTally>& getConfigTallies() const { return m_configTallies; }
    FrameStats getStats() const;
    std::string exportTelemetry(const std::string& customPath = "");

    void setCameraMode(bool active);
    bool isCameraMode() const { return m_cameraMode; }
    void setMgpuMode(MultiGpuMode mode);

    bool loadScene(const std::string& filepath);
    bool applyLoadedScene(SceneData newScene, const std::string& filepath);
    void requestSceneChange(const std::string& filepath);
    bool isSceneLoading() const { return m_isSceneLoading.load(); }
    const std::string& getLoadingSceneName() const { return m_loadingSceneName; }
    const std::vector<SceneEntry>& getAvailableScenes() const { return m_availableScenes; }
    int getCurrentSceneIndex() const { return m_currentSceneIndex; }
    std::string getActiveSceneName() const;
    Window* getWindow() const { return m_window.get(); }
    Swapchain* getSwapchain() const { return m_swapchain.get(); }
    QualityGovernor* getGovernor() const { return m_governor.get(); }
    NRCManager* getNrcManager() const { return m_nrcManager.get(); }

private:
    void initVulkan();
    void initScene();
    void initPipelines();
    void initSyncObjects();
    void initQueryPool();

    bool handleEvent(const SDL_Event& e);
    void updateInput();

    VkShaderModule createShaderModule(const std::vector<char>& code);
    std::vector<char> loadShaderSPIRV(const std::string& filename);
    void onResize(uint32_t newWidth, uint32_t newHeight, bool forceRecreate = false);

    bool m_cameraMode = false;
    bool m_resetAccumulation = false;
    bool m_cameraMovedLastFrame = false;
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

    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    // GPU Buffers
    std::unique_ptr<Buffer> m_triangleBuffer;
    std::unique_ptr<Buffer> m_sphereBuffer;
    std::unique_ptr<Buffer> m_materialBuffer;
    std::unique_ptr<Buffer> m_lightBuffer;
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_cameraUBOs;
    std::unique_ptr<Buffer> m_uiDumpBuffer;

    // Hardware Acceleration Structures (VK_KHR_ray_query)
    std::unique_ptr<Buffer> m_asVertexBuffer;
    std::unique_ptr<AccelerationStructureManager> m_asManager;
    std::unique_ptr<AccelerationStructure> m_blas;
    std::unique_ptr<AccelerationStructure> m_tlas;

    // GPU-Timeline TLAS Instance & Scratch Buffers (Tier 3)
    std::unique_ptr<Buffer> m_tlasInstanceBuffer;
    std::unique_ptr<Buffer> m_tlasInputInstancesBuffer;
    std::unique_ptr<Buffer> m_tlasScratchBuffer;
    uint32_t m_tlasInstanceCount = 0;
    bool m_tlasNeedsGpuUpdate = false;
    uint32_t m_tlasGpuUpdateCount = 0;



    // Textures & Environment Map (Bindings 7 & 8)
    static constexpr uint32_t MAX_SCENE_TEXTURES = 512;
    std::unique_ptr<Texture> m_dummyWhite;
    std::unique_ptr<Texture> m_dummyNormal;
    std::unique_ptr<Texture> m_blueNoiseTexture;
    std::unique_ptr<Texture> m_environmentMap;
    std::vector<std::unique_ptr<Texture>> m_sceneTextures;

    // Render Targets
    std::unique_ptr<Image> m_accumImage;
    std::unique_ptr<Image> m_outputImage;

    // Descriptors & Pipelines
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_rtDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_tonemapDescLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_rtDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet m_tonemapDescSet = VK_NULL_HANDLE;

    VkPipelineLayout m_rtpPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_tonemapPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_tonemapPipeline = VK_NULL_HANDLE;
    std::unique_ptr<RTPipeline> m_rtpKhrPipeline;
    std::unique_ptr<WavefrontPipeline> m_wavefrontPipeline;
    std::unique_ptr<NRCManager> m_nrcManager;
    WavefrontPipeline::WavefrontProfilingData m_lastWavefrontProfile;

    // GPU-Timeline TLAS Instance Update Pipeline (Tier 3)
    VkDescriptorSetLayout m_updateTlasDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_updateTlasDescPool = VK_NULL_HANDLE;
    VkDescriptorSet m_updateTlasDescSet = VK_NULL_HANDLE;
    VkPipelineLayout m_updateTlasPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_updateTlasPipeline = VK_NULL_HANDLE;
    void initTlasBuffers(uint32_t instanceCount);
    void initTlasUpdatePipeline();
    void recordGpuTlasUpdate(VkCommandBuffer cmd, bool updateMode = true);
    void updateInstanceTransform(uint32_t index, const glm::mat4& transform);
    void markTlasDirty() { m_tlasNeedsGpuUpdate = true; }

    // Commands & Synchronization
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> m_commandBuffers = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> m_postCommandBuffers = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> m_inFlightFences = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_rtCompleteSemaphores = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    uint32_t m_currentFrame = 0;
    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;

    // Timestamp Profiling
    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    float m_timestampPeriod = 1.0f; // ns per tick

    // Multi-GPU Transfer & Merge Resources
    std::unique_ptr<Buffer> m_secTransferBuffer;
    VkDescriptorSetLayout m_mergeDescLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> m_mergeDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout m_mergePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_mergePipeline = VK_NULL_HANDLE;
    void updateMergeDescriptors();
    void updateAllImageDescriptors();
    void updateSceneDescriptors();
    void updateWavefrontSceneDescriptors();

    std::unique_ptr<Image> m_directLightImage;
    std::unique_ptr<Image> m_normalDepthImage;
    std::unique_ptr<Image> m_prevNormalDepthImage;
    std::unique_ptr<Image> m_shadowFilterPingImage;
    std::unique_ptr<Image> m_momentsImages[2];
    std::unique_ptr<Image> m_depthImages[2];
    std::unique_ptr<Buffer> m_tileMetaDataBuffer;

    VkDescriptorSetLayout m_shadowClassifyDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_shadowFilterDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_shadowClassifyDescSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet m_shadowFilterDescSet = VK_NULL_HANDLE;
    uint32_t m_shadowPingPongIndex = 0;
    VkPipelineLayout m_shadowClassifyPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_shadowFilterPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_shadowClassifyPipeline = VK_NULL_HANDLE;
    VkPipeline m_shadowFilterPipeline = VK_NULL_HANDLE;

    void createShadowDenoiserPipelines();
    void createShadowDenoiserResources();
    void destroyShadowDenoiserResources();
    void destroyShadowDenoiserPipelines();
    void updateShadowDenoiserDescriptors();
    double m_lastShadowDenoiserTimeMs = 0.0;

    // Screen-Space Motion Vectors (used by ray tracer and temporal reconstruction passes)
    std::unique_ptr<Image> m_motionVectorImage;

    // Temporal Radiance Accumulation & wRLS Outlier Rejection
    std::array<std::unique_ptr<Image>, 2> m_temporalHistory;
    uint32_t m_temporalPingPong = 0;
    bool m_temporalResetRequested = true;
    VkDescriptorSetLayout m_temporalAccumDescSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_temporalAccumDescPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> m_temporalAccumDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout m_temporalAccumPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_temporalAccumPipeline = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> m_tonemapTemporalDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    void createTemporalAccumPipelines();
    void createTemporalAccumResources();
    void destroyTemporalAccumResources();
    void destroyTemporalAccumPipelines();
    void updateTemporalAccumDescriptors();
    uint32_t dispatchTemporalAccum(VkCommandBuffer cmd, bool resetHistory);

    // Blockwise Multi-Order Feature Regression (BMFR)
    std::unique_ptr<Image> m_bmfrOutputImage;
    VkDescriptorSetLayout m_bmfrDescSetLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_bmfrDescPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> m_bmfrDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet m_bmfrRawDescSet = VK_NULL_HANDLE;
    VkDescriptorSet m_tonemapBmfrDescSet = VK_NULL_HANDLE;
    VkPipelineLayout m_bmfrPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_bmfrPipeline = VK_NULL_HANDLE;

    void createBmfrPipelines();
    void createBmfrResources();
    void destroyBmfrResources();
    void destroyBmfrPipelines();
    void updateBmfrDescriptors();
    bool dispatchBmfr(VkCommandBuffer cmd, uint32_t temporalSlot);

    // Deferred GUI configuration actions
    bool m_pendingSceneChange = false;
    std::string m_pendingScenePath = "";
    std::future<SceneData> m_sceneLoadingFuture;
    std::atomic<bool> m_isSceneLoading{false};
    std::string m_loadingScenePath = "";
    std::string m_loadingSceneName = "";
    bool m_pendingMgpuModeChange = false;
    MultiGpuMode m_newMgpuMode = MultiGpuMode::Off;
    bool m_pendingAccumFormatChange = false;
    AccumFormat m_newAccumFormat = AccumFormat::RGBA16_SFLOAT;
    bool m_pendingDoubleBufferChange = false;
    bool m_newDoubleBuffer = true;
    bool m_pendingTileSizeChange = false;
    uint32_t m_newTileSize = 64;

    // Available scenes and dynamic selection
    std::vector<SceneEntry> m_availableScenes;
    int m_currentSceneIndex = -1;

    // Scene metadata
    SceneData m_sceneData;
    uint32_t m_numTriangles = 0;
    uint32_t m_numOpaqueTriangles = 0;
    uint32_t m_numSpheres = 0;
    uint32_t m_numMaterials = 0;
    uint32_t m_numLights = 0;
    bool m_sceneHasNonOpaque = false;
    void updateSceneTransparencyFlag();
    void partitionSceneGeometry();

    // Frame tracking & Quality Governor
    std::unique_ptr<QualityGovernor> m_governor;
    uint32_t m_accumulatedSamples = 0;
    bool m_accumulationComplete = false;
    std::chrono::high_resolution_clock::time_point m_currentFrameStartTime;
    std::chrono::high_resolution_clock::time_point m_lastWallFrameStartTime;
    double m_lastPresentationTimeMs = 0.0;
    std::vector<double> m_presentationTimesMs;
    uint32_t m_frameIndex = 0;
    uint32_t m_totalFramesRendered = 0;
    MultiGpuMode m_lastActiveMgpuMode = MultiGpuMode::Off;
    std::vector<double> m_frameTimesMs;
    double m_lastFrameTimeMs = 0.0;
    double m_lastGpuRtMs = 0.0;
    double m_lastSecGpuMs = 0.0;
    double m_lastTonemapMs = 0.0;
    std::chrono::high_resolution_clock::time_point m_startTime;
    std::chrono::high_resolution_clock::time_point m_lastFrameTime;
    std::chrono::steady_clock::time_point m_lastLogTime;

    // Per-configuration tallied statistics
    std::vector<ConfigStatsTally> m_configTallies;
    void recordFrameTally(double frameTimeMs, double primRtMs, double secRtMs, double tonemapMs,
                          const WavefrontStageSample* wfSample = nullptr);

    // Hardware Sensors & Telemetry (Infrequent background sampler)
    void startHwMonThread();
    void stopHwMonThread();
    void sampleHwSensors();
    void refreshPciStatus();

    std::atomic<bool> m_hwMonRunning{false};
    std::thread m_hwMonThread;
    std::mutex m_hwMonMutex;
    std::condition_variable m_hwMonCv;
    std::atomic<uint32_t> m_gpu0ClockMhz{0};
    std::atomic<uint32_t> m_gpu0TempC{0};
    std::atomic<uint32_t> m_gpu1ClockMhz{0};
    std::atomic<uint32_t> m_gpu1TempC{0};
};

} // namespace pathways
