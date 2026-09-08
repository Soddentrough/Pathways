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
#include "vulkan/Texture.hpp"
#include "scene/SceneRegistry.hpp"

#include <memory>
#include <vector>
#include <array>
#include <chrono>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>

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

    bool loadScene(const std::string& filepath);
    const std::vector<SceneEntry>& getAvailableScenes() const { return m_availableScenes; }
    int getCurrentSceneIndex() const { return m_currentSceneIndex; }
    std::string getActiveSceneName() const;
    Window* getWindow() const { return m_window.get(); }
    Swapchain* getSwapchain() const { return m_swapchain.get(); }

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
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_rtDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet m_tonemapDescSet = VK_NULL_HANDLE;

    VkPipelineLayout m_rtpPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_tonemapPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_tonemapPipeline = VK_NULL_HANDLE;
    std::unique_ptr<RTPipeline> m_rtpKhrPipeline;

    // Commands & Synchronization
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> m_commandBuffers = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    std::array<VkFence, MAX_FRAMES_IN_FLIGHT> m_inFlightFences = { VK_NULL_HANDLE, VK_NULL_HANDLE };
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
    VkFence m_rtFence = VK_NULL_HANDLE;
    void updateMergeDescriptors();
    void updateAllImageDescriptors();
    void updateSceneDescriptors();

    // Deferred GUI configuration actions
    bool m_pendingSceneChange = false;
    std::string m_pendingScenePath = "";
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

    // Per-configuration tallied statistics
    std::vector<ConfigStatsTally> m_configTallies;
    void recordFrameTally(double frameTimeMs, double primRtMs, double secRtMs, double tonemapMs);

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
