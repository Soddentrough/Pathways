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
#include "rt/ReSTIRManager.hpp"
#include "rt/UpwaysPipeline.hpp"
#include "rt/Fsr3Upscaler.hpp"
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

class VideoDecoder;

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
    UpwaysPipeline* getUpwaysPipeline() const { return m_upwaysPipeline.get(); }
    Fsr3Upscaler* getFsr3Upscaler() const { return m_fsr3Upscaler.get(); }

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
    std::unique_ptr<Buffer> m_materialArchetypeBuffer;
    std::unique_ptr<Buffer> m_shadeMaterialBuffer;
    std::unique_ptr<Buffer> m_lightBuffer;
    std::unique_ptr<Buffer> m_lightTreeBuffer;
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_cameraUBOs;
    std::unique_ptr<Buffer> m_uiDumpBuffer;
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_centerDepthBuffers;
    float m_gpuCenterDepth = 0.0f;
    bool m_hasGpuCenterDepth = false;

    // Hardware Acceleration Structures (VK_KHR_ray_query)
    std::unique_ptr<Buffer> m_positionBuffer;
    std::unique_ptr<Buffer> m_asIndexBuffer;
    std::unique_ptr<Buffer> m_instanceBuffer;
    std::unique_ptr<AccelerationStructureManager> m_asManager;
    std::unique_ptr<AccelerationStructure> m_blas;
    std::vector<std::unique_ptr<AccelerationStructure>> m_blases;
    std::unique_ptr<AccelerationStructure> m_tlas;

    void createAccelerationStructures();
    void uploadToDeviceBuffer(Buffer& dstBuffer, const void* srcData, VkDeviceSize dataSize);
    void uploadIndexBuffer(Buffer& dstBuffer, uint32_t triangleCount);

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
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_frameImages;
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
    uint32_t m_currentBatchCount = 0;
    uint32_t m_currentBatchPixels = 0;
    uint32_t getTargetBatchPixels() const;
    uint32_t getEffectiveBatchCount(uint32_t renderW, uint32_t renderH) const;
    uint32_t getEffectiveBatchPixels(uint32_t renderW, uint32_t renderH, uint32_t batchCount) const;

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
    static constexpr uint32_t QUERIES_PER_FRAME = 6;
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

    // Running Average Accumulation Pipeline (FP16 Frame -> FP32 Persistent History)
    VkDescriptorSetLayout m_accumRunningAvgDescLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_accumRunningAvgDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout m_accumRunningAvgPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_accumRunningAvgPipeline = VK_NULL_HANDLE;
    void createAccumRunningAvgPipeline();
    void updateAccumRunningAvgDescriptors();

    // Fused Accumulation & Tonemapping Pipeline (Single-Dispatch Pass Fusion)
    VkDescriptorSetLayout m_accumTonemapDescLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_accumTonemapDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout m_accumTonemapPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_accumTonemapPipeline = VK_NULL_HANDLE;
    void createAccumTonemapPipeline();
    void updateAccumTonemapDescriptors();

    // G-Buffer Resources (used by ray tracer, direct lighting, and FSR / Upways)
    std::unique_ptr<Image> m_directLightImage;
    std::unique_ptr<Image> m_normalDepthImage;
    std::unique_ptr<Image> m_prevNormalDepthImage;
    // Screen-Space Motion Vectors (used by ray tracer and temporal reconstruction passes)
    std::unique_ptr<Image> m_motionVectorImage;
    void createGBufferResources();
    void destroyGBufferResources();

    // ML Training Data Capture Targets (Upways neural denoiser & continuous upscaler)
    std::unique_ptr<Image> m_mlAlbedoRoughnessImage;
    std::unique_ptr<Image> m_mlSpecularMotionImage;
    std::unique_ptr<Image> m_mlDiffuseImage;
    std::unique_ptr<Image> m_mlSpecularImage;
    void runTrainingDataCapture();
    void captureTrainingFrame(uint32_t frameIdx, bool isReference, uint32_t spp);
    void updateGamingChoreography(Camera* camera, uint32_t frameIdx, uint32_t totalFrames, const std::string& sceneName);

    bool m_choreoInitialized = false;
    glm::vec3 m_choreoInitialPos{0.0f};
    float m_choreoInitialYaw = 0.0f;
    float m_choreoInitialPitch = 0.0f;
    float m_choreoInitialFov = 45.0f;

    bool m_temporalResetRequested = true;

    // Post-processing descriptor pool (for upscalers / tonemapping sets)
    VkDescriptorPool m_postProcessDescPool = VK_NULL_HANDLE;
    void createPostProcessDescPool();
    void destroyPostProcessDescPool();

    // Upways Neural Denoiser & Super-Resolution (Wave32 WMMA)
    std::unique_ptr<UpwaysPipeline> m_upwaysPipeline;
    VkDescriptorSet m_tonemapUpwaysDescSet = VK_NULL_HANDLE;

    void createUpwaysPipelines();
    void createUpwaysResources();
    void destroyUpwaysResources();
    void destroyUpwaysPipelines();
    void updateUpwaysDescriptors();
    bool dispatchUpways(VkCommandBuffer cmd, bool resetHistory);

    // AMD FidelityFX Super Resolution 3.1
    std::unique_ptr<Fsr3Upscaler> m_fsr3Upscaler;
    std::unique_ptr<Image> m_secAccumImage;
    VkDescriptorSet m_tonemapFsr3DescSet = VK_NULL_HANDLE;

    // FSR 3.1 Multi-GPU SampleBlend 4K resolve
    VkDescriptorSetLayout m_fsr3BlendDescLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_fsr3BlendPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_fsr3BlendPipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_fsr3BlendDescPool = VK_NULL_HANDLE;
    VkDescriptorSet m_fsr3BlendDescSet = VK_NULL_HANDLE;

    void createFsr3Pipelines();
    void createFsr3Resources();
    void destroyFsr3Resources();
    void destroyFsr3Pipelines();
    void updateFsr3Descriptors();
    bool dispatchFsr3(VkCommandBuffer cmd, bool resetHistory);

    // Real-Time Caustics (Photon Injection + Atomic Splatting + Bilateral Filter)
    std::unique_ptr<Buffer> m_causticPhotonBuffer;
    std::unique_ptr<Buffer> m_causticAtomicBuffer;
    std::unique_ptr<Image> m_filteredCausticImage;
    std::unique_ptr<Image> m_prevCausticImage;

    VkDescriptorSetLayout m_causticTraceDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_causticSplatDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_causticFilterDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_causticDescPool = VK_NULL_HANDLE;

    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_causticTraceDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_causticSplatDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_causticFilterDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    VkPipelineLayout m_causticTracePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_causticSplatPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_causticFilterPipelineLayout = VK_NULL_HANDLE;

    VkPipeline m_causticTracePipeline = VK_NULL_HANDLE;
    VkPipeline m_causticSplatPipeline = VK_NULL_HANDLE;
    VkPipeline m_causticFilterPipeline = VK_NULL_HANDLE;

    void createCausticsPipelines();
    void createCausticsResources();
    void destroyCausticsResources();
    void destroyCausticsPipelines();
    void updateCausticsDescriptors();
    void dispatchCausticTrace(VkCommandBuffer cmd, uint32_t frameSlot);
    void dispatchCausticSplatAndFilter(VkCommandBuffer cmd, uint32_t frameSlot);
    void dispatchCaustics(VkCommandBuffer cmd, uint32_t frameSlot);

    // ReSTIR DI Subsystem
    std::unique_ptr<ReSTIRManager> m_restirManager;
    void createReSTIRResources();
    void destroyReSTIRResources();

    // Deferred GUI configuration actions
    bool m_pendingSceneChange = false;
    std::string m_pendingScenePath = "";
    std::future<SceneData> m_sceneLoadingFuture;
    std::atomic<bool> m_isSceneLoading{false};
    std::string m_loadingScenePath = "";
    std::string m_loadingSceneName = "";
    std::chrono::steady_clock::time_point m_sceneLoadingStartTime;
    bool m_pendingMgpuModeChange = false;
    MultiGpuMode m_newMgpuMode = MultiGpuMode::Off;
    bool m_pendingMgpuUpscaleModeChange = false;
    MgpuUpscaleMode m_newMgpuUpscaleMode = MgpuUpscaleMode::PostMerge;
    bool m_pendingAccumFormatChange = false;
    AccumFormat m_newAccumFormat = AccumFormat::RGBA16_SFLOAT;
    bool m_pendingDoubleBufferChange = false;
    bool m_newDoubleBuffer = true;
    bool m_pendingTileSizeChange = false;
    uint32_t m_newTileSize = 64;

    // Active configuration tracking for descriptor / resource resynchronization
    float m_lastRenderScale = 1.0f;
    UpscalerMode m_lastUpscalerMode = UpscalerMode::None;
    uint32_t m_lastTileSize = 64;

    // Available scenes and dynamic selection
    std::vector<SceneEntry> m_availableScenes;
    int m_currentSceneIndex = -1;

    // Scene metadata
    SceneData m_sceneData;
    uint32_t m_numTriangles = 0;
    uint64_t m_numInstancedTriangles = 0;
    uint32_t m_numInstances = 1;
    uint32_t m_numOpaqueTriangles = 0;
    uint32_t m_numSpheres = 0;
    uint32_t m_numMaterials = 0;
    uint32_t m_numLights = 0;
    bool m_sceneHasNonOpaque = false;
    bool m_sceneHasAlphaMask = false;
    void updateSceneTransparencyFlag();
    void partitionSceneGeometry();

    // Frame tracking & Quality Governor
    std::unique_ptr<QualityGovernor> m_governor;
    uint32_t m_dynamicWavefrontBounces = 0;
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
    double m_lastActiveRenderFrameTimeMs = 0.0;
    double m_lastGpuRtMs = 0.0;
    double m_lastSecGpuMs = 0.0;
    double m_lastTonemapMs = 0.0;
    double m_lastUpwaysMs = 0.0;
    std::array<bool, MAX_FRAMES_IN_FLIGHT> m_slotSkippedRayTracing = {false, false};
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

    // State & Asynchronous Workers
    bool m_isMinimized = false;
    std::thread m_telemetryWorker;

    // Gamepad Navigation Support (FEAT-01)
    SDL_Gamepad* m_gamepad = nullptr;
    float m_gamepadLeftX = 0.0f;
    float m_gamepadLeftY = 0.0f;
    float m_gamepadRightX = 0.0f;
    float m_gamepadRightY = 0.0f;
    float m_gamepadLeftTrigger = 0.0f;
    float m_gamepadRightTrigger = 0.0f;
    bool m_gamepadBtnA = false;
    bool m_gamepadBtnB = false;

    // Video billboard decoder & dynamic staging buffers
    std::unique_ptr<VideoDecoder> m_videoDecoder;
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_videoStagingBuffers;
    void initVideoBillboardDecoder(const std::string& scenePath);
    void updateVideoBillboards(VkCommandBuffer cmd);
};

} // namespace pathways
