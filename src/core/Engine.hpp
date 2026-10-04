#pragma once

#include "core/Config.hpp"
#include "core/ConfigTally.hpp"
#include "core/Window.hpp"
#include "vulkan/VulkanContext.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "vulkan/Swapchain.hpp"
#include "scene/Camera.hpp"
#include "scene/CameraPath.hpp"
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
#include "rt/SuperResolutionManager.hpp"
#include "rt/CausticsPipeline.hpp"
#include "rt/RayTracingOrchestrator.hpp"
#include "rt/PostProcessPipeline.hpp"
#include "core/InputController.hpp"
#include "core/HwMonitor.hpp"
#include "rt/TrainingCaptureManager.hpp"
#include "video/VideoBillboardManager.hpp"
#include "scene/SceneGeometryPipeline.hpp"
#include "rt/GpuTlasUpdatePipeline.hpp"
#include "rt/AccelerationStructurePipeline.hpp"
#include "core/TelemetryReporter.hpp"
#include "vulkan/RenderTargetManager.hpp"
#include "vulkan/Texture.hpp"
#include "scene/SceneRegistry.hpp"
#include "scene/SceneManager.hpp"
#include "vulkan/PresentationManager.hpp"
#include "core/QualityGovernor.hpp"
#include "vulkan/EngineDescriptorManager.hpp"
#include "mgpu/MultiGpuCoordinator.hpp"

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
    const std::vector<ConfigStatsTally>& getConfigTallies() const {
        static const std::vector<ConfigStatsTally> s_empty;
        return m_telemetryReporter ? m_telemetryReporter->getConfigTallies() : s_empty;
    }
    FrameStats getStats() const;
    std::string exportTelemetry(const std::string& customPath = "");

    void setCameraMode(bool active);
    bool isCameraMode() const;
    void setMgpuMode(MultiGpuMode mode);

    bool loadScene(const std::string& filepath);
    bool applyLoadedScene(SceneData newScene, const std::string& filepath);
    void requestSceneChange(const std::string& filepath);
    bool isSceneLoading() const { return m_sceneManager ? m_sceneManager->isSceneLoading() : false; }
    const std::string& getLoadingSceneName() const {
        static const std::string s_empty;
        return m_sceneManager ? m_sceneManager->getLoadingSceneName() : s_empty;
    }
    float getSceneLoadingElapsedSec() const {
        return m_sceneManager ? m_sceneManager->getLoadingElapsedSec() : 0.0f;
    }
    const std::vector<SceneEntry>& getAvailableScenes() const {
        static const std::vector<SceneEntry> s_empty;
        return m_sceneManager ? m_sceneManager->getAvailableScenes() : m_availableScenes;
    }
    int getCurrentSceneIndex() const {
        return m_sceneManager ? m_sceneManager->getCurrentSceneIndex() : m_currentSceneIndex;
    }
    std::string getActiveSceneName() const;
    Window* getWindow() const { return m_window.get(); }
    Swapchain* getSwapchain() const { return m_swapchain; }
    PresentationManager* getPresentationManager() const { return m_presentation.get(); }
    QualityGovernor* getGovernor() const { return m_governor.get(); }
    NRCManager* getNrcManager() const { return m_nrcManager; }
    UpwaysPipeline* getUpwaysPipeline() const { return m_superResolution ? m_superResolution->getUpwaysPipeline() : nullptr; }
    Fsr3Upscaler* getFsr3Upscaler() const { return m_superResolution ? m_superResolution->getFsr3Upscaler() : nullptr; }
    SuperResolutionManager* getSuperResolution() const { return m_superResolution.get(); }
    CausticsPipeline* getCausticsPipeline() const { return m_causticsPipeline.get(); }
    PostProcessPipeline* getPostProcess() const { return m_postProcess.get(); }
    InputController* getInputController() const { return m_inputController.get(); }
    HwMonitor* getHwMonitor() const { return m_hwMonitor.get(); }
    VideoBillboardManager* getVideoBillboard() const { return m_videoBillboard.get(); }
    SceneGeometryPipeline* getGeometryPipeline() const { return m_geometryPipeline.get(); }
    GpuTlasUpdatePipeline* getTlasUpdatePipeline() const { return m_tlasUpdatePipeline.get(); }
    TelemetryReporter* getTelemetryReporter() const { return m_telemetryReporter.get(); }
    RenderTargetManager* getRenderTargets() const { return m_renderTargets.get(); }
    SceneManager* getSceneManager() const { return m_sceneManager.get(); }
    AccelerationStructurePipeline* getAsPipeline() const { return m_asPipeline.get(); }
    EngineDescriptorManager* getDescriptorManager() const { return m_descriptorManager.get(); }
    MultiGpuCoordinator* getMgpuCoordinator() const { return m_mgpuCoordinator.get(); }
    void refreshPciStatus();

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

    bool m_resetAccumulation = false;
    bool m_cameraMovedLastFrame = false;
    bool m_instanceMovedLastFrame = false;
    bool m_pendingToggleFullscreen = false;
    uint32_t m_pendingResizeW = 0;
    uint32_t m_pendingResizeH = 0;

    Config m_config;
    std::unique_ptr<Window> m_window;
    std::unique_ptr<VulkanContext> m_context;
    std::unique_ptr<PresentationManager> m_presentation;
    Swapchain* m_swapchain = nullptr;
    Buffer* m_uiDumpBuffer = nullptr;
    void syncPresentationPointers();
    std::unique_ptr<Camera> m_camera;
    std::unique_ptr<CameraPath> m_cameraPath;
    float m_cameraPathTime = 0.0f;
    std::unique_ptr<class GuiManager> m_gui;
    std::unique_ptr<class MultiGpuManager> m_mgpu;

    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    // GPU Buffers (Observing pointers to resources owned by SceneManager)
    Buffer* m_triangleBuffer = nullptr;
    Buffer* m_sphereBuffer = nullptr;
    Buffer* m_materialBuffer = nullptr;
    Buffer* m_materialArchetypeBuffer = nullptr;
    Buffer* m_shadeMaterialBuffer = nullptr;
    Buffer* m_lightBuffer = nullptr;
    Buffer* m_lightTreeBuffer = nullptr;
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_cameraUBOs;
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_centerDepthBuffers;
    float m_gpuCenterDepth = 0.0f;
    bool m_hasGpuCenterDepth = false;

    // Hardware Acceleration Structures (VK_KHR_ray_query)
    friend class AccelerationStructurePipeline;
    std::unique_ptr<AccelerationStructurePipeline> m_asPipeline;
    Buffer* m_positionBuffer = nullptr;
    Buffer* m_asIndexBuffer = nullptr;
    Buffer* m_instanceBuffer = nullptr;
    AccelerationStructureManager* m_asManager = nullptr;
    AccelerationStructure* m_tlas = nullptr;
    void syncAsPointers();

    void createAccelerationStructures();
    void uploadToDeviceBuffer(Buffer& dstBuffer, const void* srcData, VkDeviceSize dataSize);

    // Textures & Environment Map (Observing pointers to resources owned by SceneManager)
    static constexpr uint32_t MAX_SCENE_TEXTURES = 512;
    Texture* m_dummyWhite = nullptr;
    Texture* m_dummyNormal = nullptr;
    Texture* m_blueNoiseTexture = nullptr;
    Texture* m_environmentMap = nullptr;
    const std::vector<std::unique_ptr<Texture>>& getSceneTextures() const {
        static const std::vector<std::unique_ptr<Texture>> s_empty;
        return m_sceneManager ? m_sceneManager->getSceneTextures() : s_empty;
    }

    // Render Targets Subsystem
    friend class RenderTargetManager;
    std::unique_ptr<RenderTargetManager> m_renderTargets;
    Image* m_accumImage = nullptr;
    Image* m_outputImage = nullptr;
    Image* m_motionVectorImage = nullptr;
    std::array<Image*, MAX_FRAMES_IN_FLIGHT> m_frameImages = { nullptr, nullptr };

    // Descriptors & Pipelines Subsystem
    friend class EngineDescriptorManager;
    std::unique_ptr<EngineDescriptorManager> m_descriptorManager;
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_rtDescLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_rtDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    Buffer* m_secTransferBuffer = nullptr;
    void syncDescriptorPointers();

    // Ray Tracing Pipeline Subsystem (Wavefront, RTPipeline, NRC, ReSTIR)
    friend class RayTracingOrchestrator;
    std::unique_ptr<RayTracingOrchestrator> m_rtOrchestrator;
    VkPipelineLayout m_rtpPipelineLayout = VK_NULL_HANDLE;
    RTPipeline* m_rtpKhrPipeline = nullptr;
    WavefrontPipeline* m_wavefrontPipeline = nullptr;
    NRCManager* m_nrcManager = nullptr;
    void syncRayTracingPointers();

    WavefrontPipeline::WavefrontProfilingData m_lastWavefrontProfile;
    uint32_t m_currentBatchCount = 0;
    uint32_t m_currentBatchPixels = 0;
    uint32_t getTargetBatchPixels() const;
    uint32_t getEffectiveBatchCount(uint32_t renderW, uint32_t renderH) const;
    uint32_t getEffectiveBatchPixels(uint32_t renderW, uint32_t renderH, uint32_t batchCount) const;
    float getDivergentAreaRatio() const;
    WavefrontSortMode getEffectiveWavefrontSortMode() const;

    // GPU-Timeline TLAS Instance Update Pipeline & Dynamic Simulation (Tier 3)
    std::unique_ptr<GpuTlasUpdatePipeline> m_tlasUpdatePipeline;
    void initTlasBuffers(const std::vector<ASInstanceInput>& asInstances);
    void initTlasBuffers(uint32_t instanceCount);
    void initTlasUpdatePipeline();
    void recordGpuTlasUpdate(VkCommandBuffer cmd, bool updateMode = true);
    void updateInstanceTransform(uint32_t index, const glm::mat4& transform);
    void markTlasDirty();
    void updateAnimatedInstances(float frameDelta);

    // Commands & Synchronization
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> m_commandBuffers = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> m_postCommandBuffers = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_rtCompleteSemaphores = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    uint32_t m_currentFrame = 0;

    // Timestamp Profiling
    static constexpr uint32_t QUERIES_PER_FRAME = 6;
    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    float m_timestampPeriod = 1.0f; // ns per tick

    // Multi-GPU Transfer & Merge Resources (Coordinated by MultiGpuCoordinator)
    friend class MultiGpuCoordinator;
    std::unique_ptr<MultiGpuCoordinator> m_mgpuCoordinator;
    VkCommandPool m_asyncComputeCommandPool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> m_mergeCommandBuffers = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkSemaphore, MAX_FRAMES_IN_FLIGHT> m_mergeCompleteSemaphores = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    void syncMgpuCoordinatorPointers();
    std::vector<uint32_t> getConcurrentQueueFamilies() const;
    void updateMergeDescriptors();
    void updateAllImageDescriptors();
    void updateSceneDescriptors();
    void updateWavefrontSceneDescriptors();

    // Post-Processing Subsystem (ACES Tonemap, Fused Accum/Tonemap, Running Avg, Blend4K, Multi-GPU Merge)
    std::unique_ptr<PostProcessPipeline> m_postProcess;

    // G-Buffer & ML Observing Pointers (resources owned by RenderTargetManager)
    Image* m_directLightImage = nullptr;
    Image* m_normalDepthImage = nullptr;
    Image* m_prevNormalDepthImage = nullptr;
    Image* m_mlAlbedoRoughnessImage = nullptr;
    Image* m_mlSpecularMotionImage = nullptr;
    Image* m_mlDiffuseImage = nullptr;
    Image* m_mlSpecularImage = nullptr;
    void createGBufferResources();
    void destroyGBufferResources();
    void syncRenderTargetPointers();

    // ML Training Data Capture Subsystem (Upways neural denoiser & continuous upscaler)
    friend class TrainingCaptureManager;
    std::unique_ptr<TrainingCaptureManager> m_trainingCapture;
    void runTrainingDataCapture();

    bool m_temporalResetRequested = true;

    // Super-Resolution & Neural Reconstruction Subsystem (Upways & AMD FSR 3.1)
    friend class SuperResolutionManager;
    std::unique_ptr<SuperResolutionManager> m_superResolution;
    void updateUpwaysDescriptors();
    void updateFsr3Descriptors();

    // Real-Time Caustics Subsystem
    std::unique_ptr<CausticsPipeline> m_causticsPipeline;
    void initCaustics();
    void updateCausticsDescriptors();

    // ReSTIR DI Subsystem
    ReSTIRManager* m_restirManager = nullptr;
    void createReSTIRResources();
    void destroyReSTIRResources();
    [[nodiscard]] bool isRestirActive() const noexcept {
        return m_config.enable_restir_di && (m_numLights >= m_config.restir_min_lights);
    }

    // Scene Management Subsystem
    friend class SceneManager;
    std::unique_ptr<SceneManager> m_sceneManager;
    void syncScenePointers();

    // Deferred GUI configuration actions
    bool m_pendingSceneChange = false;
    std::string m_pendingScenePath = "";
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
    std::unique_ptr<SceneGeometryPipeline> m_geometryPipeline;
    void updateSceneTransparencyFlag();
    void partitionSceneGeometry();
    void clusterInstancesToMacroBlas(SceneData& scene);

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

    // Telemetry & Reporting Subsystem
    friend class TelemetryReporter;
    std::unique_ptr<TelemetryReporter> m_telemetryReporter;
    void recordFrameTally(double frameTimeMs, double primRtMs, double secRtMs, double tonemapMs,
                          const WavefrontStageSample* wfSample = nullptr);

    // Hardware Sensors & Telemetry Subsystem
    std::unique_ptr<HwMonitor> m_hwMonitor;

    // State
    bool m_isMinimized = false;

    // Input & Interaction Subsystem (Keyboard, Mouse, Gamepad)
    std::unique_ptr<InputController> m_inputController;

    // Video Billboard Subsystem
    std::unique_ptr<VideoBillboardManager> m_videoBillboard;
};

} // namespace pathways
