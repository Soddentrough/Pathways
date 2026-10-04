#include "core/Engine.hpp"
#include "core/Logger.hpp"
#include "ui/GuiManager.hpp"
#include "mgpu/MultiGpuManager.hpp"
#include "scene/GltfLoader.hpp"
#include "scene/UsdLoader.hpp"
#include "scene/LightTree.hpp"
#include "scene/Material.hpp"
#include "video/VideoBillboardManager.hpp"
#include "scene/SceneGeometryPipeline.hpp"
#include <glm/detail/type_half.hpp>
#include <glm/gtc/packing.hpp>

#include <fstream>
#include <filesystem>
#include <format>
#include <numeric>
#include <algorithm>
#include <thread>
#include <bit>
#include <cstring>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif
#ifdef __linux__
    #include <sys/utsname.h>
#endif

namespace pathways {

namespace {
inline float halfToFloat(uint16_t h) {
    uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exp = (h & 0x7C00u) >> 10;
    uint32_t mant = (h & 0x03FFu);
    if (exp == 0) {
        if (mant == 0) return std::bit_cast<float>(sign);
        while ((mant & 0x0400u) == 0) { mant <<= 1; exp--; }
        exp++;
        mant &= ~0x0400u;
    } else if (exp == 31) {
        return std::bit_cast<float>(sign | 0x7F800000u | (mant << 13));
    }
    exp = exp + (127 - 15);
    mant = mant << 13;
    return std::bit_cast<float>(sign | (exp << 23) | mant);
}
} // namespace

Engine::Engine(const Config& config) : m_config(config) {
    m_startTime = std::chrono::high_resolution_clock::now();
    m_lastFrameTime = m_startTime;
    m_lastLogTime = std::chrono::steady_clock::now();

    Logger::info("Initializing Pathways Engine...");

    // Point instancing CLI options and environment variable parsing
    {
        std::ifstream cmdline("/proc/self/cmdline", std::ios::binary);
        if (cmdline) {
            std::string arg;
            std::vector<std::string> args;
            while (std::getline(cmdline, arg, '\0')) {
                args.push_back(arg);
            }
            for (size_t i = 1; i < args.size(); ++i) {
                if ((args[i] == "--instance-density" || args[i] == "--density") && i + 1 < args.size()) {
                    m_config.instance_density = std::clamp(std::stof(args[++i]), 0.0f, 1.0f);
                } else if (args[i].starts_with("--instance-density=")) {
                    m_config.instance_density = std::clamp(std::stof(args[i].substr(args[i].find('=') + 1)), 0.0f, 1.0f);
                } else if (args[i].starts_with("--density=")) {
                    m_config.instance_density = std::clamp(std::stof(args[i].substr(args[i].find('=') + 1)), 0.0f, 1.0f);
                } else if ((args[i] == "--cull-distance" || args[i] == "--cull-dist") && i + 1 < args.size()) {
                    m_config.cull_distance = std::max(0.0f, std::stof(args[++i]));
                } else if (args[i].starts_with("--cull-distance=") || args[i].starts_with("--cull-dist=")) {
                    m_config.cull_distance = std::max(0.0f, std::stof(args[i].substr(args[i].find('=') + 1)));
                }
            }
        }
        if (const char* envDensity = std::getenv("PATHWAYS_INSTANCE_DENSITY")) {
            m_config.instance_density = std::clamp(std::stof(envDensity), 0.0f, 1.0f);
        }
        if (const char* envCull = std::getenv("PATHWAYS_CULL_DISTANCE")) {
            m_config.cull_distance = std::max(0.0f, std::stof(envCull));
        }
        if (m_config.instance_density < 0.999f || m_config.cull_distance > 0.0f) {
            Logger::info("Point Instancing Configuration: instance-density={:.2f}, cull-distance={:.1f}m",
                         m_config.instance_density, m_config.cull_distance);
        }
    }
    m_window = std::make_unique<Window>(m_config);

    if (!m_config.headless) {
        m_config.width = m_window->getWidth();
        m_config.height = m_window->getHeight();
    }

    if (!m_config.headless) {
        m_window->setResizeCallback([this](uint32_t w, uint32_t h) {
            onResize(w, h);
        });
    }

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!m_config.headless) {
        // Create surface before context physical device selection
        // Create temporary instance or pass window to context
        // In our architecture, context creates instance then window creates surface
    }

    m_context = std::make_unique<VulkanContext>(m_config, VK_NULL_HANDLE, "Primary GPU / Display");

    m_presentation = std::make_unique<PresentationManager>(
        m_window.get(), m_context.get(), m_config, MAX_FRAMES_IN_FLIGHT
    );
    syncPresentationPointers();

    float aspect = static_cast<float>(m_config.width) / static_cast<float>(m_config.height);
    m_camera = std::make_unique<Camera>(
        glm::vec3(0.0f, 1.0f, 3.4f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        45.0f,
        aspect
    );
    m_camera->setAspect(aspect);
    m_camera->setDynamicScaling(m_config.adaptive_speed);

    initVulkan();
    m_videoBillboard = std::make_unique<VideoBillboardManager>();
    initScene();


    if (!m_config.camera_path.empty()) {
        m_cameraPath = CameraPath::loadFromFile(m_config.camera_path);
        if (m_cameraPath && m_cameraPath->isValid()) {
            Logger::info("Initialized CameraPath trajectory: '{}' ({} keyframes, {:.2f}s duration, loop: {})",
                         m_cameraPath->getName(), m_cameraPath->getKeyframeCount(),
                         m_cameraPath->getDuration(), (m_config.camera_path_loop || m_cameraPath->isLoop()));
            CameraSample initSample = m_cameraPath->evaluate(0.0f, m_config.camera_path_loop);
            m_camera->lookAt(initSample.position, initSample.target, initSample.up);
            if (initSample.fov > 1.0f && initSample.fov < 170.0f) {
                m_camera->setFov(initSample.fov);
            }
        } else {
            Logger::error("Failed to load valid CameraPath from: {}", m_config.camera_path);
        }
    }
    auto physicalDevices = VulkanContext::enumeratePhysicalDevices(m_context->getInstance());
    uint32_t hwDeviceCount = 0;
    for (auto pd : physicalDevices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
            hwDeviceCount++;
        }
    }
    if (hwDeviceCount >= 2 && m_config.mgpu_mode != MultiGpuMode::Off) {
        m_mgpu = std::make_unique<MultiGpuManager>(m_config, m_context.get(), m_sceneData);
    }
    // Reclaim host memory used for scene geometry ingestion (now safely resident in device VRAM)
    if (m_sceneManager) {
        m_sceneManager->reclaimHostTriangles();
    }
    m_sceneData.triangles.clear();
    m_sceneData.triangles.shrink_to_fit();
    initPipelines();
    initSyncObjects();
    initQueryPool();

    if (!m_config.headless && m_swapchain) {
        m_gui = std::make_unique<GuiManager>(
            m_window->getSDLWindow(),
            m_context->getInstance(),
            m_context->getPhysicalDevice(),
            m_context->getDevice(),
            m_context->getGraphicsQueueFamily(),
            m_context->getGraphicsQueue(),
            m_swapchain->getFormat(),
            2,
            m_swapchain->getImageCount()
        );

        m_window->setEventCallback([this](const SDL_Event& e) -> bool {
            return handleEvent(e);
        });
    }

    m_inputController = std::make_unique<InputController>(m_window.get(), m_gui.get(), m_config);
    m_inputController->setCamera(m_camera.get());
    m_inputController->setSceneTarget(m_sceneData.centralTarget, m_sceneData.focalRadius);
    m_inputController->setFullscreenToggleCallback([this]() {
        m_pendingToggleFullscreen = true;
    });
    m_inputController->setScreenshotCallback([this]() {
        m_pendingScreenshot = true;
    });
    if (!m_config.headless && m_swapchain) {
        m_inputController->init();
    }

    m_hwMonitor = std::make_unique<HwMonitor>(m_context.get(), [this]() -> VulkanContext* {
        return (m_mgpu && m_mgpu->getSecondaryContext()) ? m_mgpu->getSecondaryContext() : nullptr;
    });
    m_hwMonitor->start();

    m_trainingCapture = std::make_unique<TrainingCaptureManager>(this);
    m_telemetryReporter = std::make_unique<TelemetryReporter>(this);

    m_lastRenderScale = m_config.render_scale;
    m_lastUpscalerMode = m_config.upscaler_mode;
    m_lastTileSize = m_config.tile_size;
    m_dynamicWavefrontBounces = m_config.max_bounces;

    // Initialize Dynamic Quality Governor
    GovernorConfig govCfg{};
    govCfg.targetFps = m_config.target_fps;
    govCfg.enabled = m_config.adaptive_spp;
    govCfg.minSpp = m_config.min_spp;
    govCfg.maxSpp = m_config.max_spp;
    govCfg.minBounces = m_config.min_bounces;
    govCfg.maxBounces = m_config.max_dynamic_bounces;
    m_governor = std::make_unique<QualityGovernor>(govCfg);
    if (m_config.target_fps > 0) {
        if (m_config.adaptive_spp && m_governor->getState().active) {
            Logger::info("Dynamic Quality Governor ACTIVE: Target {} FPS (Budget: {:.2f} ms), SPP Range [{}..{}], Bounces [{}..{}]",
                         m_config.target_fps, 1000.0f / static_cast<float>(m_config.target_fps),
                         m_config.min_spp, m_config.max_spp, m_config.min_bounces, m_config.max_dynamic_bounces);
        } else {
            Logger::info("Frame Rate Limiter ACTIVE: Target {} FPS (Budget: {:.2f} ms)",
                         m_config.target_fps, 1000.0f / static_cast<float>(m_config.target_fps));
        }
    }

    Logger::info("Pathways Engine initialization complete. Ready to render.");
}

Engine::~Engine() {
    Logger::info("Shutting down Pathways Engine...");
    if (m_pendingScreenshotFuture.valid()) {
        m_pendingScreenshotFuture.wait();
    }
    if (m_hwMonitor) {
        m_hwMonitor->stop();
    }
    VkDevice device = m_context->getDevice();
    vkDeviceWaitIdle(device);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_rtCompleteSemaphores[i]) vkDestroySemaphore(device, m_rtCompleteSemaphores[i], nullptr);
    }
    m_mgpuCoordinator.reset();
    syncMgpuCoordinatorPointers();
    m_geometryPipeline.reset();
    if (m_commandPool) vkDestroyCommandPool(device, m_commandPool, nullptr);

    if (m_queryPool) vkDestroyQueryPool(device, m_queryPool, nullptr);

    m_rtOrchestrator.reset();
    syncRayTracingPointers();
    destroyGBufferResources();
    m_superResolution.reset();
    m_causticsPipeline.reset();
    m_renderTargets.reset();
    syncRenderTargetPointers();
    m_postProcess.reset();
    m_telemetryReporter.reset();

    m_descriptorManager.reset();
    syncDescriptorPointers();

    m_tlasUpdatePipeline.reset();
    m_asPipeline.reset();
    syncAsPointers();

    m_sceneManager.reset();
    syncScenePointers();

    m_gui.reset();
    m_presentation.reset();
    syncPresentationPointers();
}

void Engine::syncDescriptorPointers() {
    if (m_descriptorManager) {
        m_descriptorPool = m_descriptorManager->getPool();
        m_rtDescLayout = m_descriptorManager->getRtDescLayout();
        m_rtDescSets = m_descriptorManager->getRtDescSets();
        m_secTransferBuffer = m_descriptorManager->getSecTransferBuffer();
    } else {
        m_descriptorPool = VK_NULL_HANDLE;
        m_rtDescLayout = VK_NULL_HANDLE;
        m_rtDescSets.fill(VK_NULL_HANDLE);
        m_secTransferBuffer = nullptr;
    }
}

void Engine::syncMgpuCoordinatorPointers() {
    if (m_mgpuCoordinator) {
        m_asyncComputeCommandPool = m_mgpuCoordinator->getAsyncComputeCommandPool();
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            m_mergeCommandBuffers[i] = m_mgpuCoordinator->getMergeCommandBuffer(i);
            m_mergeCompleteSemaphores[i] = m_mgpuCoordinator->getMergeCompleteSemaphore(i);
        }
    } else {
        m_asyncComputeCommandPool = VK_NULL_HANDLE;
        m_mergeCommandBuffers.fill(VK_NULL_HANDLE);
        m_mergeCompleteSemaphores.fill(VK_NULL_HANDLE);
    }
}

void Engine::refreshPciStatus() {
    if (m_hwMonitor) {
        m_hwMonitor->refreshPciStatus();
    }
}

void Engine::initVulkan() {
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();

    // Command pool
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_context->getGraphicsQueueFamily();
    vkCreateCommandPool(device, &poolInfo, nullptr, &m_commandPool);

    m_geometryPipeline = std::make_unique<SceneGeometryPipeline>(
        device,
        m_context->getGraphicsQueue(),
        allocator,
        m_commandPool
    );

    m_tlasUpdatePipeline = std::make_unique<GpuTlasUpdatePipeline>(
        device,
        allocator,
        m_context->hasSubgroupSizeControl()
    );

    // Command buffers (Double-buffered)
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    vkAllocateCommandBuffers(device, &allocInfo, m_commandBuffers.data());
    vkAllocateCommandBuffers(device, &allocInfo, m_postCommandBuffers.data());

    m_mgpuCoordinator = std::make_unique<MultiGpuCoordinator>();
    m_mgpuCoordinator->init(device, m_context->getAsyncComputeQueueFamily(), m_context->hasDedicatedAsyncCompute());
    syncMgpuCoordinatorPointers();
    if (m_context->hasDedicatedAsyncCompute()) {
        Logger::info("Async Compute Merge initialized (Family: {})", m_context->getAsyncComputeQueueFamily());
    }

    auto concurrentQueues = getConcurrentQueueFamilies();

    m_renderTargets = std::make_unique<RenderTargetManager>(device, allocator, m_context->getGraphicsQueue());
    m_renderTargets->createRenderTargets(
        m_config.width, m_config.height, m_config.accum_format,
        m_swapchain && !m_config.headless && m_swapchain->isHdr(),
        m_swapchain ? m_swapchain->getFormat() : VK_FORMAT_UNDEFINED,
        concurrentQueues,
        m_commandBuffers[0]
    );
    syncRenderTargetPointers();
}

void Engine::syncScenePointers() {
    if (m_sceneManager) {
        m_triangleBuffer = m_sceneManager->getTriangleBuffer();
        m_sphereBuffer = m_sceneManager->getSphereBuffer();
        m_materialBuffer = m_sceneManager->getMaterialBuffer();
        m_materialArchetypeBuffer = m_sceneManager->getMaterialArchetypeBuffer();
        m_shadeMaterialBuffer = m_sceneManager->getShadeMaterialBuffer();
        m_lightBuffer = m_sceneManager->getLightBuffer();
        m_lightTreeBuffer = m_sceneManager->getLightTreeBuffer();
        m_positionBuffer = m_sceneManager->getPositionBuffer();

        m_dummyWhite = m_sceneManager->getDummyWhite();
        m_dummyNormal = m_sceneManager->getDummyNormal();
        m_blueNoiseTexture = m_sceneManager->getBlueNoiseTexture();
        m_environmentMap = m_sceneManager->getEnvironmentMap();

        m_numTriangles = m_sceneManager->getNumTriangles();
        m_numInstancedTriangles = m_sceneManager->getNumInstancedTriangles();
        m_numInstances = m_sceneManager->getNumInstances();
        m_numOpaqueTriangles = m_sceneManager->getNumOpaqueTriangles();
        m_numSpheres = m_sceneManager->getNumSpheres();
        m_numMaterials = m_sceneManager->getNumMaterials();
        m_numLights = m_sceneManager->getNumLights();
        m_sceneHasNonOpaque = m_sceneManager->hasNonOpaque();
        m_sceneHasAlphaMask = m_sceneManager->hasAlphaMask();
        m_sceneData = m_sceneManager->getSceneData();
        m_currentSceneIndex = m_sceneManager->getCurrentSceneIndex();
        m_availableScenes = m_sceneManager->getAvailableScenes();
    } else {
        m_triangleBuffer = nullptr;
        m_sphereBuffer = nullptr;
        m_materialBuffer = nullptr;
        m_materialArchetypeBuffer = nullptr;
        m_shadeMaterialBuffer = nullptr;
        m_lightBuffer = nullptr;
        m_lightTreeBuffer = nullptr;
        m_positionBuffer = nullptr;

        m_dummyWhite = nullptr;
        m_dummyNormal = nullptr;
        m_blueNoiseTexture = nullptr;
        m_environmentMap = nullptr;

        m_numTriangles = 0;
        m_numInstancedTriangles = 0;
        m_numInstances = 1;
        m_numOpaqueTriangles = 0;
        m_numSpheres = 0;
        m_numMaterials = 0;
        m_numLights = 0;
        m_sceneHasNonOpaque = false;
        m_sceneHasAlphaMask = false;
    }
}

void Engine::uploadToDeviceBuffer(Buffer& dstBuffer, const void* srcData, VkDeviceSize dataSize) {
    if (m_geometryPipeline) {
        m_geometryPipeline->uploadToDeviceBuffer(dstBuffer, srcData, dataSize);
    }
}

void Engine::syncPresentationPointers() {
    if (m_presentation) {
        m_swapchain = m_presentation->getSwapchain();
        m_surface = m_presentation->getSurface();
        m_uiDumpBuffer = m_presentation->getUiDumpBuffer();
    } else {
        m_swapchain = nullptr;
        m_surface = VK_NULL_HANDLE;
        m_uiDumpBuffer = nullptr;
    }
}

void Engine::syncAsPointers() {
    if (m_asPipeline) {
        m_tlas = m_asPipeline->getTlas();
        m_asManager = m_asPipeline->getAsManager();
        m_instanceBuffer = m_asPipeline->getInstanceBuffer();
        m_asIndexBuffer = m_asPipeline->getIndexBuffer();
    } else {
        m_tlas = nullptr;
        m_asManager = nullptr;
        m_instanceBuffer = nullptr;
        m_asIndexBuffer = nullptr;
    }
}

void Engine::syncRayTracingPointers() {
    if (m_rtOrchestrator) {
        m_wavefrontPipeline = m_rtOrchestrator->getWavefrontPipeline();
        m_rtpKhrPipeline = m_rtOrchestrator->getRtpPipeline();
        m_nrcManager = m_rtOrchestrator->getNrcManager();
        m_restirManager = m_rtOrchestrator->getRestirManager();
        m_rtpPipelineLayout = m_rtOrchestrator->getRtpPipelineLayout();
    } else {
        m_wavefrontPipeline = nullptr;
        m_rtpKhrPipeline = nullptr;
        m_nrcManager = nullptr;
        m_restirManager = nullptr;
        m_rtpPipelineLayout = VK_NULL_HANDLE;
    }
}

void Engine::createAccelerationStructures() {
    if (!m_context->hasRayTracing()) {
        throw std::runtime_error("Hardware Ray Tracing is required under Vulkan 1.4 baseline but is not available.");
    }

    if (!m_asPipeline) {
        m_asPipeline = std::make_unique<AccelerationStructurePipeline>(
            m_context->getDevice(),
            m_context->getAllocator(),
            m_context->getGraphicsQueue(),
            m_context->getGraphicsQueueFamily()
        );
    }
    m_asPipeline->build(m_sceneData, m_numTriangles, m_numOpaqueTriangles, m_positionBuffer, m_geometryPipeline.get());
    syncAsPointers();

    initTlasBuffers(m_asPipeline->getAsInstances());
    Logger::info("Hardware Ray Tracing Pipeline Active. Extensions in use: VK_KHR_ray_query, VK_KHR_acceleration_structure, VK_KHR_buffer_device_address, VK_KHR_deferred_host_operations (SPIR-V: GL_EXT_ray_query)");
}

void Engine::initScene() {
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();
    VkQueue queue = m_context->getGraphicsQueue();
    VkCommandPool pool = m_commandPool;

    m_sceneManager = std::make_unique<SceneManager>(device, allocator, queue, pool);
    m_sceneManager->discoverScenes();

    std::string scenePath = m_config.scene_path;
    std::string resolvedScene = m_sceneManager->resolveScenePath(scenePath);

    UsdLoadOptions options{};
    options.instanceDensity = m_config.instance_density;
    options.cullDistance = m_config.cull_distance;
    options.cameraPosOverride = m_config.camera_pos;
    options.viewportAspect = (m_config.height > 0) ? (static_cast<float>(m_config.width) / static_cast<float>(m_config.height)) : (16.0f / 9.0f);

    SceneData loadedData = SceneManager::loadRawSceneData(resolvedScene, options);
    m_sceneManager->ingestSceneData(std::move(loadedData), resolvedScene, m_config, m_geometryPipeline.get());
    m_sceneManager->uploadGpuBuffers(m_geometryPipeline.get());
    m_sceneManager->updateCamera(m_camera.get(), m_config);

    // Camera UBOs (Double-buffered)
    VkDeviceSize uboSize = sizeof(CameraUniform);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_cameraUBOs[i] = std::make_unique<Buffer>(
            allocator, uboSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        );
    }

    // Center pixel G-buffer depth readback buffers (Double-buffered, 16 bytes for 1 RGBA16F texel)
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_centerDepthBuffers[i] = std::make_unique<Buffer>(
            allocator, 16,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_TO_CPU,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
        );
    }

    syncScenePointers();

    // Hardware Acceleration Structures (VK_KHR_ray_query)
    createAccelerationStructures();

    // Textures & HDRI Environment Map Initialization
    m_sceneManager->loadTextures(m_config.hdri_path, m_config.scene_path);

    syncScenePointers();

    if (m_videoBillboard) {
        m_videoBillboard->init(m_config.scene_path, m_sceneData, allocator);
    }
}

void Engine::requestSceneChange(const std::string& filepath) {
    if (m_sceneManager) {
        m_sceneManager->requestSceneChange(filepath, m_config);
    }
}

bool Engine::applyLoadedScene(SceneData newScene, const std::string& filepath) {
    if (newScene.triangles.empty() && newScene.spheres.empty()) {
        Logger::warn("Loaded scene '{}' contains no renderable geometry! Keeping current scene.", filepath);
        return false;
    }

    VkDevice device = m_context->getDevice();
    vkDeviceWaitIdle(device);
    if (m_mgpu && m_mgpu->getSecondaryContext()) {
        vkDeviceWaitIdle(m_mgpu->getSecondaryContext()->getDevice());
    }

    m_config.scene_path = filepath;
    if (!m_config.custom_hdri) {
        m_config.hdri_path = m_sceneManager ? m_sceneManager->detectSceneHdri(filepath) : "";
    }
    if (m_mgpu) {
        m_mgpu->setConfig(m_config);
    }

    if (m_sceneManager) {
        m_sceneManager->ingestSceneData(std::move(newScene), filepath, m_config, m_geometryPipeline.get());
        m_sceneManager->uploadGpuBuffers(m_geometryPipeline.get());
    }

    syncScenePointers();

    // Rebuild Acceleration Structures (BLAS & TLAS)
    createAccelerationStructures();

    // Reload scene textures & environment map
    if (m_sceneManager) {
        m_sceneManager->loadTextures(m_config.hdri_path, filepath);
    }

    syncScenePointers();

    // Update primary descriptor sets
    updateSceneDescriptors();

    // Secondary GPU reload
    if (m_mgpu && m_mgpu->isSecondaryInitialized()) {
        m_mgpu->loadScene(m_sceneData, filepath);
    }

    // Update camera framing
    if (m_sceneManager) {
        m_sceneManager->updateCamera(m_camera.get(), m_config);
    }

    if (m_inputController) {
        m_inputController->setSceneTarget(m_sceneData.centralTarget, m_sceneData.focalRadius);
    }

    // Reset frame accumulation
    m_frameIndex = 0;
    m_resetAccumulation = true;
    m_frameTimesMs.clear();
    if (m_rtOrchestrator) {
        m_rtOrchestrator->invalidateDivergentAreaCache();
    }

    // Reclaim host memory used for scene geometry ingestion (now safely resident in device VRAM)
    if (m_sceneManager) {
        m_sceneManager->reclaimHostTriangles();
    }
    m_sceneData.triangles.clear();
    m_sceneData.triangles.shrink_to_fit();

    VmaAllocator allocator = m_context->getAllocator();
    if (m_videoBillboard) {
        m_videoBillboard->init(filepath, m_sceneData, allocator);
    }

    Logger::info("Scene successfully switched to: {} (Index: {})", filepath, m_currentSceneIndex);
    return true;
}

bool Engine::loadScene(const std::string& filepath) {
    m_dynamicWavefrontBounces = m_config.max_bounces;
    UsdLoadOptions options{};
    options.instanceDensity = m_config.instance_density;
    options.cullDistance = m_config.cull_distance;
    options.cameraPosOverride = m_config.camera_pos;
    options.viewportAspect = (m_config.height > 0) ? (static_cast<float>(m_config.width) / static_cast<float>(m_config.height)) : (16.0f / 9.0f);

    SceneData newScene = SceneManager::loadRawSceneData(filepath, options);
    return applyLoadedScene(std::move(newScene), filepath);
}

void Engine::updateSceneTransparencyFlag() {
    if (m_geometryPipeline) {
        m_geometryPipeline->updateSceneTransparency(m_sceneData.materials, m_sceneHasNonOpaque, m_sceneHasAlphaMask);
    }
}

void Engine::clusterInstancesToMacroBlas(SceneData& scene) {
    if (m_geometryPipeline) {
        m_geometryPipeline->clusterInstancesToMacroBlas(scene, m_config);
    }
}

void Engine::partitionSceneGeometry() {
    if (m_geometryPipeline) {
        m_geometryPipeline->partitionSceneGeometry(m_sceneData, m_numOpaqueTriangles);
    }
}

std::vector<char> Engine::loadShaderSPIRV(const std::string& filename) {
    std::filesystem::path exeDir;
#ifdef _WIN32
    char exePathBuf[MAX_PATH] = {0};
    if (GetModuleFileNameA(NULL, exePathBuf, MAX_PATH)) {
        exeDir = std::filesystem::path(exePathBuf).parent_path();
    }
#elif defined(__linux__)
    std::error_code ec;
    auto p = std::filesystem::canonical("/proc/self/exe", ec);
    if (!ec) exeDir = p.parent_path();
#endif

    std::vector<std::string> searchPaths = {
        filename,
        std::string("shaders/") + filename,
    };

    if (!exeDir.empty()) {
        searchPaths.push_back((exeDir / "shaders" / filename).string());
        searchPaths.push_back((exeDir / filename).string());
        searchPaths.push_back((exeDir / ".." / "share" / "pathways" / "shaders" / filename).string());
    }

    searchPaths.push_back(std::string(SHADER_DIR) + "/" + filename);
    searchPaths.push_back(std::string("/usr/share/pathways/shaders/") + filename);
    searchPaths.push_back(std::string("/usr/local/share/pathways/shaders/") + filename);
    searchPaths.push_back(std::string("build/shaders/") + filename);
    searchPaths.push_back(std::string("../build/shaders/") + filename);

    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (file.is_open()) {
                size_t fileSize = static_cast<size_t>(file.tellg());
                std::vector<char> buffer(fileSize);
                file.seekg(0);
                file.read(buffer.data(), fileSize);
                Logger::debug("Loaded shader SPIR-V from: {}", path);
                return buffer;
            }
        }
    }
    throw std::runtime_error("Could not find compiled SPIR-V file: " + filename);
}

VkShaderModule Engine::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    VkResult res = vkCreateShaderModule(m_context->getDevice(), &createInfo, nullptr, &shaderModule);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module!");
    }
    return shaderModule;
}

uint32_t Engine::getTargetBatchPixels() const {
    return m_rtOrchestrator ? m_rtOrchestrator->getTargetBatchPixels(m_config) : 0u;
}

uint32_t Engine::getEffectiveBatchCount(uint32_t renderW, uint32_t renderH) const {
    return m_rtOrchestrator ? m_rtOrchestrator->getEffectiveBatchCount(renderW, renderH, m_config) : 1u;
}

uint32_t Engine::getEffectiveBatchPixels(uint32_t renderW, uint32_t renderH, uint32_t batchCount) const {
    return m_rtOrchestrator ? m_rtOrchestrator->getEffectiveBatchPixels(renderW, renderH, batchCount) : (renderW * renderH);
}

float Engine::getDivergentAreaRatio() const {
    return m_rtOrchestrator ? m_rtOrchestrator->getDivergentAreaRatio(m_sceneData) : 0.0f;
}

WavefrontSortMode Engine::getEffectiveWavefrontSortMode() const {
    return m_rtOrchestrator ? m_rtOrchestrator->getEffectiveWavefrontSortMode(m_config, m_sceneData) : WavefrontSortMode::None;
}

void Engine::initPipelines() {
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();

    // 1. Primary Descriptors Subsystem
    m_descriptorManager = std::make_unique<EngineDescriptorManager>(device, allocator);
    m_descriptorManager->init(m_frameImages, m_cameraUBOs);
    syncDescriptorPointers();

    updateSceneDescriptors();

    m_rtOrchestrator = std::make_unique<RayTracingOrchestrator>(
        device, m_context->getAllocator(),
        m_context->getPhysicalDevice(),
        m_context->getDeviceProperties().deviceType
    );

    // 6. Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline)
    auto rgenCode = loadShaderSPIRV("raytrace.rgen.spv");
    auto rmissCode = loadShaderSPIRV("raytrace.rmiss.spv");
    auto shadowMissCode = loadShaderSPIRV("shadow.rmiss.spv");
    auto rchitCode = loadShaderSPIRV("raytrace.rchit.spv");

    m_rtOrchestrator->initRTPipeline(
        m_context->getRayTracingPipelineProperties(),
        m_rtDescLayout,
        rgenCode, rmissCode, shadowMissCode, rchitCode,
        m_context->hasRtSubgroupSizeControl()
    );

    // 6b. Wavefront Path Tracing Pipeline (Ray Queues & DGC)
    auto wfClassifyCode = loadShaderSPIRV("wavefront_classify.comp.spv");
    auto wfIntersectCode = loadShaderSPIRV("wavefront_intersect.comp.spv");
    auto wfShadeCode = loadShaderSPIRV("wavefront_shade.comp.spv");
    auto wfShadowCode = loadShaderSPIRV("wavefront_shadow.comp.spv");
    auto wfShadeDiffuseCode = loadShaderSPIRV("wavefront_shade_diffuse.comp.spv");
    auto wfShadeDielectricCode = loadShaderSPIRV("wavefront_shade_dielectric.comp.spv");
    auto wfShadeConductorCode = loadShaderSPIRV("wavefront_shade_conductor.comp.spv");
    auto wfShadeComplexCode = loadShaderSPIRV("wavefront_shade_complex.comp.spv");
    auto wfShadeEmissiveCode = loadShaderSPIRV("wavefront_shade_emissive.comp.spv");
    auto wfShadePassthroughCode = loadShaderSPIRV("wavefront_shade_passthrough.comp.spv");
    auto wfShadeDiffuseSecCode = loadShaderSPIRV("wavefront_shade_diffuse_sec.comp.spv");
    auto wfShadeComplexSecCode = loadShaderSPIRV("wavefront_shade_complex_sec.comp.spv");
    auto wfTailMegakernelCode = loadShaderSPIRV("wavefront_tail_megakernel.comp.spv");

    uint32_t initBatchCount = getEffectiveBatchCount(m_config.width, m_config.height);
    uint32_t initBatchPixels = getEffectiveBatchPixels(m_config.width, m_config.height, initBatchCount);
    if (m_config.mgpu_mode != MultiGpuMode::Off) {
        initBatchCount = 1;
        initBatchPixels = m_config.width * m_config.height;
    }
    m_currentBatchCount = initBatchCount;
    m_currentBatchPixels = initBatchPixels;

    m_rtOrchestrator->initWavefrontPipeline(
        m_config.width, m_config.height,
        wfClassifyCode, wfIntersectCode, wfShadeCode, wfShadowCode,
        wfShadeDiffuseCode, wfShadeDielectricCode, wfShadeConductorCode, wfShadeComplexCode,
        wfShadeEmissiveCode, wfShadePassthroughCode,
        m_context->hasDgcExecutionSet(),
        wfShadeDiffuseSecCode, wfShadeComplexSecCode,
        m_config.dgc_preprocess && !m_context->isRDNA4(), // enableDgcPreprocess (bypass on RDNA4/GFX1201 due to RADV illegal opcode bug)
        m_context->hasSubgroupSizeControl(),
        initBatchPixels,
        wfTailMegakernelCode
    );

    // 6c. Neural Radiance Caching (NRC) Manager (Wave32 WMMA)
    try {
        auto nrcInferCode = loadShaderSPIRV("nrc_encode_infer.comp.spv");
        auto nrcTrainCode = loadShaderSPIRV("nrc_train.comp.spv");
        auto nrcResolveCode = loadShaderSPIRV("nrc_resolve.comp.spv");
        m_rtOrchestrator->initNRC(
            m_config.width, m_config.height,
            nrcInferCode, nrcTrainCode, nrcResolveCode
        );
    } catch (const std::exception& e) {
        if (m_config.enable_nrc) {
            throw std::runtime_error(std::string("NRC was explicitly requested (--nrc) but initialization failed: ") + e.what());
        }
        Logger::warn("NRCManager initialization failed: {}", e.what());
    }

    // 6d. Ultra-Lean ReSTIR DI Subsystem
    createReSTIRResources();
    syncRayTracingPointers();

    // 7. Post-Processing Pipeline Subsystem (ACES Tonemap, Fused Accum/Tonemap, Running Avg, Blend4K, Multi-GPU Merge)
    auto tonemapCode = loadShaderSPIRV("tonemap_aces.comp.spv");
    auto fusedCode = loadShaderSPIRV("accum_tonemap_fused.comp.spv");
    auto runningAvgCode = loadShaderSPIRV("accum_running_avg.comp.spv");
    auto blendCode = loadShaderSPIRV("fsr3_blend.comp.spv");
    auto mergeCode = loadShaderSPIRV("accum_merge.comp.spv");

    m_postProcess = std::make_unique<PostProcessPipeline>(
        device, allocator,
        m_context->hasSubgroupSizeControl(),
        m_descriptorPool,
        tonemapCode, fusedCode, runningAvgCode, blendCode, mergeCode
    );
    updateMergeDescriptors();

    // 8. G-Buffer Resources (Direct Light & Surface Normals/Depth)
    createGBufferResources();

    // 9. Super-Resolution & Neural Reconstruction Subsystem (Upways & AMD FSR 3.1)
    m_superResolution = std::make_unique<SuperResolutionManager>(
        device, m_context->getPhysicalDevice(), allocator,
        m_context->getGraphicsQueue(),
        [this](const std::string& name) { return loadShaderSPIRV(name); }
    );
    m_superResolution->init(m_config, m_postProcess.get(), m_commandBuffers[0]);

    // 11. GPU TLAS Instance Writer & Refit Pipeline (Tier 3)
    initTlasUpdatePipeline();

    // 12. Real-Time Caustics Subsystem
    initCaustics();

    if (m_wavefrontPipeline) {
        m_wavefrontPipeline->setPostClassifyCallback([this](VkCommandBuffer cmd, uint32_t frameSlot) {
            if (m_causticsPipeline && m_config.enable_caustics && m_sceneData.hasDielectrics && m_numLights > 0) {
                m_causticsPipeline->recordSplatAndFilter(
                    cmd, frameSlot,
                    m_normalDepthImage,
                    m_camera ? m_camera->getFov() : 45.0f,
                    m_frameIndex,
                    m_config.progressive_accumulation,
                    m_cameraMovedLastFrame,
                    m_sceneData.hasDielectrics,
                    m_sceneData.dielectricBoundsMin,
                    m_sceneData.dielectricBoundsMax
                );
            }
            if (isRestirActive() && m_restirManager) {
                uint32_t rw = m_config.width;
                uint32_t rh = m_config.height;
                Buffer* rayGeom = m_wavefrontPipeline->getRayGeomQueue(frameSlot);
                Buffer* rayHit = m_wavefrontPipeline->getRayHitQueue(frameSlot);
                Buffer* pixelToRay = m_wavefrontPipeline->getPixelToRayQueue(frameSlot);
                Buffer* camUBO = m_cameraUBOs[frameSlot].get();
                Buffer* lightsBuf = m_lightBuffer;
                Buffer* matsBuf = m_materialBuffer;
                Buffer* ltBuf = m_lightTreeBuffer;
                VkImageView mvView = m_motionVectorImage ? m_motionVectorImage->getImageView() : VK_NULL_HANDLE;
                VkImageView ndView = m_normalDepthImage ? m_normalDepthImage->getImageView() : VK_NULL_HANDLE;
                VkImageView prevNdView = m_prevNormalDepthImage ? m_prevNormalDepthImage->getImageView() : ndView;
                UpwaysPipeline* upways = getUpwaysPipeline();
                VkImageView confView = (upways && upways->getConfidenceImage())
                    ? upways->getConfidenceImage()->getImageView()
                    : VK_NULL_HANDLE;

                bool hasLt = m_config.enable_light_tree || (!m_sceneData.lightTreeNodes.empty() && ltBuf != nullptr);
                m_restirManager->recordFrame(cmd, frameSlot, rw, rh,
                                             m_numLights, static_cast<uint32_t>(m_sceneData.triangles.size()), hasLt,
                                             m_frameIndex, m_config.restir_di_m_cap,
                                             rayGeom, rayHit, pixelToRay,
                                             lightsBuf, matsBuf,
                                             camUBO, ltBuf,
                                             mvView, ndView, prevNdView,
                                             confView);
            }
        });
    }

    updateAllImageDescriptors();
    updateSceneDescriptors();
}

void Engine::updateAllImageDescriptors() {
    if (!m_accumImage || !m_outputImage || !m_descriptorManager) return;

    ImageDescriptorParams imgParams{};
    imgParams.accumImage = m_accumImage;
    imgParams.outputImage = m_outputImage;
    imgParams.directLightImage = m_directLightImage;
    imgParams.normalDepthImage = m_normalDepthImage;
    imgParams.motionVectorImage = m_motionVectorImage;
    imgParams.filteredCausticImage = (m_causticsPipeline && m_causticsPipeline->getFilteredCausticImage())
        ? m_causticsPipeline->getFilteredCausticImage() : nullptr;
    imgParams.frameImages = m_frameImages;

    m_descriptorManager->updatePrimaryImageDescriptors(imgParams);

    if (m_postProcess) {
        m_postProcess->updateTonemapDescriptors(m_postProcess->getTonemapDescSet(), m_accumImage, m_outputImage);
        if (m_renderTargets) {
            m_postProcess->updateRunningAvgDescriptors(m_renderTargets->getFrameImages(), m_accumImage);
            m_postProcess->updateFusedAccumTonemapDescriptors(m_renderTargets->getFrameImages(), m_accumImage, m_outputImage);
        }
    }

    updateWavefrontSceneDescriptors();
    updateUpwaysDescriptors();
    updateFsr3Descriptors();
    updateCausticsDescriptors();
}

void Engine::createGBufferResources() {
    auto concurrentQueues = getConcurrentQueueFamilies();
    if (m_renderTargets) {
        m_renderTargets->createGBufferResources(m_config.width, m_config.height, concurrentQueues, m_commandBuffers[0]);
    }
    syncRenderTargetPointers();
    updateMergeDescriptors();
}

void Engine::destroyGBufferResources() {
    if (m_renderTargets) {
        m_renderTargets->destroyGBufferResources();
    }
    syncRenderTargetPointers();
}

void Engine::syncRenderTargetPointers() {
    if (m_renderTargets) {
        m_accumImage = m_renderTargets->getAccumImage();
        m_outputImage = m_renderTargets->getOutputImage();
        m_motionVectorImage = m_renderTargets->getMotionVectorImage();
        m_directLightImage = m_renderTargets->getDirectLightImage();
        m_normalDepthImage = m_renderTargets->getNormalDepthImage();
        m_prevNormalDepthImage = m_renderTargets->getPrevNormalDepthImage();
        m_mlAlbedoRoughnessImage = m_renderTargets->getMlAlbedoRoughnessImage();
        m_mlSpecularMotionImage = m_renderTargets->getMlSpecularMotionImage();
        m_mlDiffuseImage = m_renderTargets->getMlDiffuseImage();
        m_mlSpecularImage = m_renderTargets->getMlSpecularImage();
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            m_frameImages[i] = m_renderTargets->getFrameImage(i);
        }
    } else {
        m_accumImage = nullptr;
        m_outputImage = nullptr;
        m_motionVectorImage = nullptr;
        m_directLightImage = nullptr;
        m_normalDepthImage = nullptr;
        m_prevNormalDepthImage = nullptr;
        m_mlAlbedoRoughnessImage = nullptr;
        m_mlSpecularMotionImage = nullptr;
        m_mlDiffuseImage = nullptr;
        m_mlSpecularImage = nullptr;
        m_frameImages.fill(nullptr);
    }
}

void Engine::updateUpwaysDescriptors() {
    if (m_superResolution) {
        m_superResolution->updateDescriptors(
            m_postProcess.get(), m_accumImage, m_outputImage,
            m_normalDepthImage, m_motionVectorImage,
            m_mlAlbedoRoughnessImage, m_mlSpecularMotionImage,
            m_mlDiffuseImage, m_mlSpecularImage, m_frameImages[0]
        );
    }
}

void Engine::updateFsr3Descriptors() {
    if (m_superResolution) {
        m_superResolution->updateDescriptors(
            m_postProcess.get(), m_accumImage, m_outputImage,
            m_normalDepthImage, m_motionVectorImage,
            m_mlAlbedoRoughnessImage, m_mlSpecularMotionImage,
            m_mlDiffuseImage, m_mlSpecularImage, m_frameImages[0]
        );
    }
}



// === REAL-TIME CAUSTICS SUBSYSTEM ===

void Engine::initCaustics() {
    if (m_causticsPipeline) {
        return;
    }
    if (!m_config.enable_caustics) {
        return;
    }

    try {
        auto traceSpv = loadShaderSPIRV("caustic_photon_trace.comp.spv");
        auto splatSpv = loadShaderSPIRV("caustic_splat.comp.spv");
        auto filterSpv = loadShaderSPIRV("caustic_filter.comp.spv");

        m_causticsPipeline = std::make_unique<CausticsPipeline>(
            m_context->getDevice(),
            m_context->getAllocator(),
            m_context->hasSubgroupSizeControl(),
            m_config.width,
            m_config.height,
            m_config.caustic_photons,
            traceSpv, splatSpv, filterSpv,
            [this](auto recordFn) {
                VkCommandBuffer cmd = m_commandBuffers[0];
                vkResetCommandBuffer(cmd, 0);
                VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                vkBeginCommandBuffer(cmd, &bi);
                recordFn(cmd);
                vkEndCommandBuffer(cmd);
                VkCommandBufferSubmitInfo csi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
                csi.commandBuffer = cmd;
                VkSubmitInfo2 si{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
                si.commandBufferInfoCount = 1;
                si.pCommandBufferInfos = &csi;
                vkQueueSubmit2(m_context->getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
                vkQueueWaitIdle(m_context->getGraphicsQueue());
            }
        );

        updateCausticsDescriptors();
        updateWavefrontSceneDescriptors();
    } catch (const std::exception& e) {
        Logger::warn("CausticsPipeline initialization failed: {}", e.what());
    }
}

void Engine::createReSTIRResources() {
    if (!m_rtOrchestrator) return;
    try {
        auto temporalCode  = loadShaderSPIRV("restir_di_temporal.comp.spv");
        auto spatialCode   = loadShaderSPIRV("restir_di_spatial.comp.spv");
        m_rtOrchestrator->createReSTIRResources(
            m_config.width, m_config.height,
            temporalCode, spatialCode
        );
        syncRayTracingPointers();
    } catch (const std::exception& e) {
        Logger::warn("ReSTIRManager initialization failed: {}", e.what());
    }
}

void Engine::destroyReSTIRResources() {
    if (m_rtOrchestrator) {
        m_rtOrchestrator->destroyReSTIRResources();
        syncRayTracingPointers();
    }
}

void Engine::updateCausticsDescriptors() {
    if (!m_causticsPipeline) {
        return;
    }
    VkAccelerationStructureKHR tlasHandle = m_tlas ? m_tlas->getHandle() : VK_NULL_HANDLE;
    m_causticsPipeline->updateDescriptors(
        m_triangleBuffer,
        m_sphereBuffer,
        m_materialBuffer,
        m_lightBuffer,
        tlasHandle,
        getSceneTextures(),
        m_dummyWhite,
        m_instanceBuffer,
        m_cameraUBOs,
        m_normalDepthImage,
        m_motionVectorImage
    );
}

void Engine::updateSceneDescriptors() {
    if (!m_descriptorManager || !m_triangleBuffer || !m_sphereBuffer || !m_materialBuffer || !m_lightBuffer) return;

    SceneDescriptorParams sceneParams{};
    sceneParams.triangleBuffer = m_triangleBuffer;
    sceneParams.sphereBuffer = m_sphereBuffer;
    sceneParams.materialBuffer = m_materialBuffer;
    sceneParams.lightBuffer = m_lightBuffer;
    sceneParams.tlasHandle = m_tlas ? m_tlas->getHandle() : VK_NULL_HANDLE;
    sceneParams.environmentMap = m_environmentMap;
    sceneParams.dummyWhite = m_dummyWhite;
    sceneParams.blueNoiseTexture = m_blueNoiseTexture;
    const auto& sceneTextures = getSceneTextures();
    sceneParams.sceneTextures = &sceneTextures;

    m_descriptorManager->updatePrimarySceneDescriptors(sceneParams);

    updateWavefrontSceneDescriptors();
    updateCausticsDescriptors();
}

void Engine::updateWavefrontSceneDescriptors() {
    if (!m_wavefrontPipeline || !m_accumImage || !m_triangleBuffer || !m_descriptorManager) return;

    WavefrontDescriptorParams wfParams{};
    wfParams.wavefrontPipeline = m_wavefrontPipeline;
    wfParams.nrcManager = m_nrcManager;
    wfParams.restirManager = m_restirManager;
    wfParams.accumImage = m_accumImage;
    wfParams.frameImages = m_frameImages;
    wfParams.cameraUBOs = &m_cameraUBOs;
    wfParams.triangleBuffer = m_triangleBuffer;
    wfParams.sphereBuffer = m_sphereBuffer;
    wfParams.materialBuffer = m_materialBuffer;
    wfParams.lightBuffer = m_lightBuffer;
    wfParams.lightTreeBuffer = m_lightTreeBuffer;
    wfParams.instanceBuffer = m_instanceBuffer;
    wfParams.materialArchetypeBuffer = m_materialArchetypeBuffer;
    wfParams.shadeMaterialBuffer = m_shadeMaterialBuffer;
    wfParams.tlasHandle = m_tlas ? m_tlas->getHandle() : VK_NULL_HANDLE;
    wfParams.environmentMap = m_environmentMap;
    wfParams.dummyWhite = m_dummyWhite;
    const auto& wfSceneTextures = getSceneTextures();
    wfParams.sceneTextures = &wfSceneTextures;
    wfParams.motionVectorImage = m_motionVectorImage;
    wfParams.normalDepthImage = m_normalDepthImage;
    wfParams.mlAlbedoRoughnessImage = m_mlAlbedoRoughnessImage;
    wfParams.mlSpecularMotionImage = m_mlSpecularMotionImage;
    wfParams.mlDiffuseImage = m_mlDiffuseImage;
    wfParams.mlSpecularImage = m_mlSpecularImage;
    wfParams.filteredCausticImage = (m_causticsPipeline && m_causticsPipeline->getFilteredCausticImage())
        ? m_causticsPipeline->getFilteredCausticImage() : nullptr;

    m_descriptorManager->updateWavefrontDescriptors(wfParams);
}

std::vector<uint32_t> Engine::getConcurrentQueueFamilies() const {
    if (m_context && m_context->hasDedicatedAsyncCompute()) {
        return { m_context->getGraphicsQueueFamily(), m_context->getAsyncComputeQueueFamily() };
    }
    return {};
}

void Engine::updateMergeDescriptors() {
    if (!m_postProcess || !m_motionVectorImage || !m_normalDepthImage || !m_descriptorManager) {
        return;
    }

    MergeDescriptorParams mergeParams{};
    mergeParams.postProcess = m_postProcess.get();
    mergeParams.mgpu = m_mgpu.get();
    mergeParams.motionVectorImage = m_motionVectorImage;
    mergeParams.normalDepthImage = m_normalDepthImage;
    mergeParams.frameImages = m_frameImages;
    mergeParams.width = m_config.width;
    mergeParams.height = m_config.height;
    mergeParams.accumFormat = m_config.accum_format;
    mergeParams.concurrentQueues = getConcurrentQueueFamilies();

    m_descriptorManager->updateMergeDescriptors(mergeParams);
    syncDescriptorPointers();
}

void Engine::initTlasBuffers(const std::vector<ASInstanceInput>& asInstances) {
    if (m_tlasUpdatePipeline) {
        m_tlasUpdatePipeline->initBuffers(asInstances, m_asManager);
    }
}

void Engine::initTlasBuffers(uint32_t instanceCount) {
    if (m_tlasUpdatePipeline) {
        VkDeviceAddress defaultBlasAddr = m_asPipeline ? m_asPipeline->getDefaultBlasAddress() : 0;
        m_tlasUpdatePipeline->initBuffers(instanceCount, m_asManager, defaultBlasAddr);
    }
}

void Engine::initTlasUpdatePipeline() {
    if (m_tlasUpdatePipeline) {
        auto compCode = loadShaderSPIRV("update_tlas_instances.comp.spv");
        m_tlasUpdatePipeline->createPipeline(compCode);
    }
}

void Engine::recordGpuTlasUpdate(VkCommandBuffer cmd, bool updateMode) {
    if (m_tlasUpdatePipeline && m_asManager && m_tlas) {
        m_tlasUpdatePipeline->recordGpuTlasUpdate(cmd, m_asManager, m_tlas, updateMode);
    }
}

void Engine::updateInstanceTransform(uint32_t index, const glm::mat4& transform) {
    if (m_tlasUpdatePipeline) {
        m_tlasUpdatePipeline->updateInstanceTransform(index, transform, m_sceneData.instances);
    }
}

void Engine::markTlasDirty() {
    if (m_tlasUpdatePipeline) {
        m_tlasUpdatePipeline->markDirty();
    }
}

void Engine::updateAnimatedInstances(float frameDelta) {
    if (m_tlasUpdatePipeline) {
        m_tlasUpdatePipeline->updateAnimatedInstances(
            frameDelta,
            m_config.animate_objects,
            m_config.animation_speed,
            m_sceneData.animatedInstances,
            m_sceneData.instances
        );
    }
}



void Engine::initSyncObjects() {
    VkDevice device = m_context->getDevice();
    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkCreateSemaphore(device, &semInfo, nullptr, &m_rtCompleteSemaphores[i]);
    }
}

void Engine::initQueryPool() {
    VkDevice device = m_context->getDevice();
    VkQueryPoolCreateInfo queryInfo{};
    queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = QUERIES_PER_FRAME * MAX_FRAMES_IN_FLIGHT; // 6 timestamps per frame in flight
    vkCreateQueryPool(device, &queryInfo, nullptr, &m_queryPool);

    m_timestampPeriod = m_context->getDeviceProperties().limits.timestampPeriod;
    Logger::info("GPU Timestamp Profiler initialized (period: {:.2f} ns/tick, {} queries/frame)", m_timestampPeriod, QUERIES_PER_FRAME);
}

void Engine::setCameraMode(bool active) {
    if (m_inputController) {
        m_inputController->setCameraMode(active);
    }
}

bool Engine::isCameraMode() const {
    return m_inputController ? m_inputController->isCameraMode() : false;
}

void Engine::setMgpuMode(MultiGpuMode mode) {
    m_pendingMgpuModeChange = true;
    m_newMgpuMode = mode;
}

bool Engine::handleEvent(const SDL_Event& e) {
    if (e.type == SDL_EVENT_QUIT) {
        return false;
    }
    if (e.type == SDL_EVENT_WINDOW_MINIMIZED || e.type == SDL_EVENT_WINDOW_OCCLUDED) {
        m_isMinimized = true;
    } else if (e.type == SDL_EVENT_WINDOW_RESTORED || e.type == SDL_EVENT_WINDOW_EXPOSED) {
        m_isMinimized = false;
    }

    if (m_inputController) {
        return m_inputController->handleEvent(e);
    }
    return false;
}

void Engine::updateInput() {
    auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float>(now - m_lastFrameTime).count();
    m_lastFrameTime = now;
    if (dt <= 0.0f) {
        dt = 0.016f;
    } else if (dt > 0.1f) {
        dt = 0.1f;
    }

    if (m_inputController) {
        m_inputController->update(dt, m_gpuCenterDepth, m_hasGpuCenterDepth);
    }
}

void Engine::renderFrame() {
    if (m_presentation) {
        m_presentation->beginFrame(m_currentFrame, m_totalFramesRendered, m_config);
        m_lastPresentationTimeMs = m_presentation->getLastPresentationTimeMs();
        m_presentationTimesMs = m_presentation->getPresentationTimesMs();
        m_currentFrameStartTime = m_presentation->getCurrentFrameStartTime();
    }
    VkDevice device = m_context->getDevice();
    VkQueue queue = m_context->getGraphicsQueue();

    // Read back center pixel G-buffer depth from completed frame slot
    if (m_totalFramesRendered >= 1 && m_centerDepthBuffers[m_currentFrame]) {
        m_centerDepthBuffers[m_currentFrame]->invalidate();
        const uint16_t* raw16 = static_cast<const uint16_t*>(m_centerDepthBuffers[m_currentFrame]->map());
        if (raw16) {
            float depth = halfToFloat(raw16[3]);
            if (depth > 0.05f && depth < 9000.0f) {
                m_gpuCenterDepth = depth;
                m_hasGpuCenterDepth = true;
            } else {
                m_hasGpuCenterDepth = false;
            }
            m_centerDepthBuffers[m_currentFrame]->unmap();
        }
    }


    // Read back GPU query timestamps from slot m_currentFrame's completed frame
    if (m_totalFramesRendered >= MAX_FRAMES_IN_FLIGHT) {
        uint32_t qBase = m_currentFrame * QUERIES_PER_FRAME;
        uint64_t timestamps[QUERIES_PER_FRAME] = {0};
        vkGetQueryPoolResults(device, m_queryPool, qBase, QUERIES_PER_FRAME, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        double gpuRtMs = 0.0;
        if (timestamps[1] > timestamps[0]) {
            gpuRtMs = (timestamps[1] - timestamps[0]) * m_timestampPeriod * 1e-6;
        }
        double gpuTonemapMs = 0.0;
        if (timestamps[3] > timestamps[2]) {
            gpuTonemapMs = (timestamps[3] - timestamps[2]) * m_timestampPeriod * 1e-6;
        }
        double gpuUpwaysMs = 0.0;
        if (timestamps[5] > timestamps[4]) {
            gpuUpwaysMs = (timestamps[5] - timestamps[4]) * m_timestampPeriod * 1e-6;
        }
        if (gpuRtMs > 10000.0) gpuRtMs = 0.0;
        if (gpuTonemapMs > 10000.0) gpuTonemapMs = 0.0;
        if (gpuUpwaysMs > 10000.0) gpuUpwaysMs = 0.0;


        double secGpuMs = 0.0;
        if (m_mgpu && m_mgpu->isMultiGpuActive()) {
            secGpuMs = m_mgpu->getSecondaryGpuTimeMs();
        }
        double totalGpuMs = (m_mgpu && m_mgpu->isMultiGpuActive())
            ? (std::max(gpuRtMs, secGpuMs) + gpuTonemapMs + gpuUpwaysMs)
            : (gpuRtMs + gpuTonemapMs + gpuUpwaysMs);

        bool completedSlotSkipped = m_slotSkippedRayTracing[m_currentFrame];
        m_lastTonemapMs = gpuTonemapMs;
        m_lastUpwaysMs = gpuUpwaysMs;
        bool isMgpuActive = m_mgpu && m_mgpu->isMultiGpuActive();

        if (!completedSlotSkipped) {
            m_lastGpuRtMs = gpuRtMs;
            m_lastSecGpuMs = secGpuMs;

            if (m_config.pipeline_type == PipelineType::Wavefront && m_wavefrontPipeline) {
                if (m_totalFramesRendered >= MAX_FRAMES_IN_FLIGHT) {
                    m_lastWavefrontProfile = m_wavefrontPipeline->getProfilingData(m_currentFrame, m_timestampPeriod, m_config.max_bounces);
                    if (m_lastWavefrontProfile.valid && !m_lastWavefrontProfile.bounces.empty()) {
                        uint32_t activeBouncesCount = static_cast<uint32_t>(m_lastWavefrontProfile.bounces.size());
                        m_dynamicWavefrontBounces = activeBouncesCount;
                    }
                    static int wfProfCount = 0;
                    bool isBenchmarkMilestone = m_config.benchmark && (++wfProfCount == 10 || (m_config.frame_limit > 0 && m_totalFramesRendered + 1 >= m_config.frame_limit));
                    if (isBenchmarkMilestone || getenv("PATHWAYS_PROFILE_WF")) {
                        m_wavefrontPipeline->printProfilingBreakdown(m_currentFrame, m_timestampPeriod, m_config.max_bounces);
                    }
                }
            }

            if (totalGpuMs > 0.01) {
                m_lastActiveRenderFrameTimeMs = totalGpuMs;
                m_lastFrameTimeMs = totalGpuMs;
                if (m_totalFramesRendered >= m_config.warmup_frames + MAX_FRAMES_IN_FLIGHT) {
                    m_frameTimesMs.push_back(m_lastFrameTimeMs);
                    if (!m_config.headless && m_frameTimesMs.size() > 60) {
                        m_frameTimesMs.erase(m_frameTimesMs.begin());
                    }
                    WavefrontStageSample wfSample;
                    if (m_lastWavefrontProfile.valid && m_config.pipeline_type == PipelineType::Wavefront) {
                        wfSample.classifyMs = m_lastWavefrontProfile.classifyMs;
                        wfSample.tailMegakernelMs = m_lastWavefrontProfile.tailMegakernelMs;
                        wfSample.tailMegakernelBounce = m_lastWavefrontProfile.tailMegakernelBounce;
                        uint32_t traceW = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
                            static_cast<uint32_t>(m_config.width * m_config.render_scale) : m_config.width;
                        uint32_t traceH = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
                            static_cast<uint32_t>(m_config.height * m_config.render_scale) : m_config.height;
                        wfSample.primaryRays = static_cast<uint64_t>(traceW) * traceH * m_config.spp;
                        for (const auto& bp : m_lastWavefrontProfile.bounces) {
                            wfSample.bounces.push_back({bp.shadeMs, bp.shadowMs, bp.intersectMs, bp.gapBeforeShadeMs, bp.gapBeforeShadowMs, bp.gapBeforeIntersectMs, bp.activeCount, bp.nextCount, bp.shadowCount});
                        }
                    }
                    recordFrameTally(totalGpuMs, gpuRtMs, secGpuMs, gpuTonemapMs, (wfSample.bounces.empty() && wfSample.tailMegakernelMs <= 0.0) ? nullptr : &wfSample);
                }
            }
        } else if (m_lastActiveRenderFrameTimeMs > 0.01) {
            // Keep reported frame time locked to the last active rendering frame
            m_lastFrameTimeMs = m_lastActiveRenderFrameTimeMs;
        }

        // Update Dynamic Quality Governor with measured GPU timings
        if (m_governor && m_config.adaptive_spp) {
            bool isMgpuSample = (m_mgpu && m_mgpu->isMultiGpuActive() &&
                                (m_config.mgpu_mode == MultiGpuMode::SampleParallel));
            float activeRtMs = static_cast<float>((m_mgpu && m_mgpu->isMultiGpuActive()) ? std::max(gpuRtMs, secGpuMs) : gpuRtMs);
            m_governor->update(m_currentFrame, activeRtMs, static_cast<float>(gpuTonemapMs), m_cameraMovedLastFrame, isMgpuSample);
        }
    }

    // Check if background asynchronous scene loading completed
    if (m_sceneManager) {
        SceneData loadedData;
        std::string loadedPath;
        if (m_sceneManager->pollAsyncLoading(loadedData, loadedPath)) {
            applyLoadedScene(std::move(loadedData), loadedPath);
        }
    }

    if (m_pendingSceneChange) {
        std::string targetPath = m_pendingScenePath;
        m_pendingSceneChange = false;
        requestSceneChange(targetPath);
    }

    // Process deferred UI reconfiguration actions safely at frame boundary (before recording)
    bool scalingOrUpscalerChanged = (m_config.render_scale != m_lastRenderScale) ||
                                    (m_config.upscaler_mode != m_lastUpscalerMode) ||
                                    (m_config.tile_size != m_lastTileSize);

    if (m_pendingMgpuModeChange || m_pendingMgpuUpscaleModeChange || m_pendingAccumFormatChange ||
        m_pendingDoubleBufferChange || m_pendingTileSizeChange || scalingOrUpscalerChanged) {
        VkDevice dev = m_context->getDevice();
        VmaAllocator alloc = m_context->getAllocator();
        vkDeviceWaitIdle(dev);
        if (m_mgpu && m_mgpu->getSecondaryContext()) {
            vkDeviceWaitIdle(m_mgpu->getSecondaryContext()->getDevice());
        }

        m_lastRenderScale = m_config.render_scale;
        m_lastUpscalerMode = m_config.upscaler_mode;
        m_lastTileSize = m_config.tile_size;

        if (m_pendingMgpuUpscaleModeChange) {
            m_config.mgpu_upscale_mode = m_newMgpuUpscaleMode;
            m_pendingMgpuUpscaleModeChange = false;
        }

        if (m_pendingAccumFormatChange) {
            m_config.accum_format = m_newAccumFormat;
            m_pendingAccumFormatChange = false;

            auto concurrentQueues = getConcurrentQueueFamilies();
            if (m_renderTargets) {
                m_renderTargets->recreateAccumImages(m_config.width, m_config.height, m_config.accum_format, concurrentQueues, m_commandBuffers[0]);
            }
            syncRenderTargetPointers();

            updateAllImageDescriptors();
            if (m_mgpu) {
                m_mgpu->setFormat(m_config.accum_format);
                if (m_mgpu->isMultiGpuActive()) {
                    m_mgpu->resize(m_config.width, m_config.height);
                }
            }
        }

        if (m_pendingMgpuModeChange) {
            vkDeviceWaitIdle(device);
            if (m_mgpu && m_mgpu->getSecondaryContext()) {
                vkDeviceWaitIdle(m_mgpu->getSecondaryContext()->getDevice());
            }
            m_config.mgpu_mode = m_newMgpuMode;
            m_pendingMgpuModeChange = false;

            if (m_config.mgpu_mode != MultiGpuMode::Off) {
                if (!m_mgpu) {
                    m_mgpu = std::make_unique<MultiGpuManager>(m_config, m_context.get(), m_sceneData);
                } else {
                    m_mgpu->setMode(m_config.mgpu_mode);
                    m_mgpu->setFormat(m_config.accum_format);
                    m_mgpu->resize(m_config.width, m_config.height);
                }
                if (!m_mgpu->isSecondaryInitialized()) {
                    m_config.mgpu_mode = MultiGpuMode::Off;
                }
            } else if (m_mgpu) {
                m_mgpu->setMode(MultiGpuMode::Off);
            }
            m_resetAccumulation = true;

            VkCommandBufferBeginInfo clearBegin{};
            clearBegin.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            clearBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkResetCommandBuffer(m_commandBuffers[0], 0);
            vkBeginCommandBuffer(m_commandBuffers[0], &clearBegin);
            VkClearColorValue clearColor = { { 0.0f, 0.0f, 0.0f, 0.0f } };
            VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
            vkCmdClearColorImage(m_commandBuffers[0], m_accumImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &clearRange);
            vkEndCommandBuffer(m_commandBuffers[0]);
            VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
            cmdSubmitInfo.commandBuffer = m_commandBuffers[0];

            VkSubmitInfo2 clearSubmit{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
            clearSubmit.commandBufferInfoCount = 1;
            clearSubmit.pCommandBufferInfos = &cmdSubmitInfo;
            vkQueueSubmit2(m_context->getGraphicsQueue(), 1, &clearSubmit, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_context->getGraphicsQueue());
            if (m_superResolution) {
                updateUpwaysDescriptors();
            }
        }

        if (m_pendingDoubleBufferChange) {
            m_config.double_buffered_shared_mem = m_newDoubleBuffer;
            m_pendingDoubleBufferChange = false;
        }

        if (m_pendingTileSizeChange) {
            m_config.tile_size = m_newTileSize;
            m_lastTileSize = m_newTileSize;
            m_pendingTileSizeChange = false;
        }

        if (m_config.upscaler_mode == UpscalerMode::Upways && m_config.render_scale >= 1.0f) {
            m_config.render_scale = 0.5f;
            m_lastRenderScale = 0.5f;
        }

        if (m_config.upscaler_mode == UpscalerMode::Upways || m_config.denoiser_mode == DenoiserMode::Upways) {
            if (m_superResolution) {
                m_superResolution->resize(
                    m_config.width, m_config.height, m_config.render_scale,
                    m_config.upscaler_mode, m_config.denoiser_mode,
                    m_config.upways_superres, m_config.accum_format,
                    m_postProcess.get(), m_commandBuffers[0]
                );
            }
        }

        updateMergeDescriptors();
        m_resetAccumulation = true;
        m_frameTimesMs.clear();
    }

    bool sceneLoadingActive = isSceneLoading() || m_pendingSceneChange;

    // Update smooth continuous FPS keyboard navigation or camera trajectory
    bool pathActive = (m_cameraPath && m_cameraPath->isValid());
    bool pathMoving = false;
    if (!sceneLoadingActive) {
        if (pathActive && m_camera) {
            float dt = 1.0f / (m_config.target_fps > 0 ? static_cast<float>(m_config.target_fps) : 60.0f);
            if (!m_config.headless && m_lastPresentationTimeMs > 0.01 && m_lastPresentationTimeMs < 1000.0) {
                dt = static_cast<float>(m_lastPresentationTimeMs * 0.001);
            }
            m_cameraPathTime += dt * m_config.camera_path_speed;
            float simTime = m_config.headless ? (static_cast<float>(m_totalFramesRendered) * dt * m_config.camera_path_speed) : m_cameraPathTime;
            CameraSample sample = m_cameraPath->evaluate(simTime, m_config.camera_path_loop);
            m_camera->setAnimatedPose(sample.position, sample.target, sample.up, sample.fov);
            if (sample.isStationary) {
                m_camera->resetMoved();
                pathMoving = false;
            } else {
                pathMoving = true;
            }
        } else {
            updateInput();

            if (m_config.camera_motion && m_camera) {
                m_camera->processMouseMovement(2.0f, 0.0f);
            }
        }
    }

    // Advance video billboard decoder if active
    if (m_videoBillboard && m_videoBillboard->isOpen()) {
        float dt = 1.0f / 24.0f;
        if (!m_config.headless && m_lastPresentationTimeMs > 0.01 && m_lastPresentationTimeMs < 1000.0) {
            dt = static_cast<float>(m_lastPresentationTimeMs * 0.001);
        }
        m_videoBillboard->update(dt);
    }

    // Advance dynamic kinematic object animations
    float animDt = 1.0f / 60.0f;
    if (!m_config.headless && m_lastPresentationTimeMs > 0.01 && m_lastPresentationTimeMs < 1000.0) {
        animDt = static_cast<float>(m_lastPresentationTimeMs * 0.001);
    }
    bool isAnimating = m_config.animate_objects && (m_config.animation_speed > 0.0001f) && !m_sceneData.animatedInstances.empty();
    if (isAnimating) {
        updateAnimatedInstances(animDt);
    }
    bool instanceMovedThisFrame = isAnimating && (m_totalFramesRendered > 0);
    bool instanceJustStopped = (!isAnimating && m_instanceMovedLastFrame);

    // Reset accumulation if camera moved, camera just came to a stop, dynamic instances moved/stopped, or UI settings changed
    bool cameraMovedThisFrame = !sceneLoadingActive && ((m_camera && m_camera->hasMoved() && m_totalFramesRendered > 0) || m_config.camera_motion || pathMoving);
    bool cameraJustStopped = (!cameraMovedThisFrame && m_cameraMovedLastFrame);
    bool hardReset = m_resetAccumulation || (m_totalFramesRendered == 0);
    bool accumReset = cameraMovedThisFrame || cameraJustStopped || instanceMovedThisFrame || instanceJustStopped || hardReset;
    if (accumReset || cameraMovedThisFrame) {
        m_dynamicWavefrontBounces = m_config.max_bounces;
    }
    if (accumReset && !sceneLoadingActive) {
        m_accumulatedSamples = 0;
        if (m_camera) m_camera->resetMoved();
        m_resetAccumulation = false;
    }
    if (!sceneLoadingActive) {
        m_cameraMovedLastFrame = cameraMovedThisFrame;
        m_instanceMovedLastFrame = isAnimating;
    }

    if (m_governor) {
        if (m_governor->getConfig().targetFps != m_config.target_fps) {
            m_governor->setTargetFps(m_config.target_fps);
        }
        if (m_governor->getConfig().enabled != m_config.adaptive_spp) {
            m_governor->setEnabled(m_config.adaptive_spp);
        }
        m_governor->setBounds(m_config.min_spp, m_config.max_spp, m_config.min_bounces, m_config.max_dynamic_bounces);
    }

    uint32_t activeSpp = m_config.spp;
    float activeFractionalSpp = 0.0f;
    uint32_t activeBounces = m_config.max_bounces;
    if (m_config.pipeline_type == PipelineType::Wavefront && m_dynamicWavefrontBounces > 0 && m_totalFramesRendered >= m_config.warmup_frames) {
        activeBounces = std::clamp(m_dynamicWavefrontBounces + 1u, std::min(1u, m_config.max_bounces), m_config.max_bounces);
    }
    if (m_governor && (m_config.adaptive_spp || m_config.target_fps > 0) && m_governor->getState().active) {
        activeSpp = m_governor->getState().currentSpp;
        activeFractionalSpp = m_governor->getState().fractionalSpp;
        activeBounces = std::min(activeBounces, m_governor->getState().currentBounces);
    }
    bool accumReachedCutoff = (m_config.progressive_accumulation &&
                               !isAnimating &&
                               m_config.max_accum_frames > 0 &&
                               m_accumulatedSamples >= m_config.max_accum_frames);
    bool skipRayTracing = accumReachedCutoff || sceneLoadingActive;
    m_accumulationComplete = accumReachedCutoff;
    m_slotSkippedRayTracing[m_currentFrame] = skipRayTracing;
    if (m_config.progressive_accumulation && !isAnimating) {
        if (!skipRayTracing) {
            if (m_config.max_accum_frames == 0 || m_accumulatedSamples < m_config.max_accum_frames) {
                m_accumulatedSamples++;
            }
        }
    } else {
        m_accumulatedSamples = 1;
    }

    // Update Camera Uniform
    uint32_t flags = 0;
    if (m_config.enable_direct_light)   flags |= (1 << 0);
    if (m_config.enable_indirect_light) flags |= (1 << 1);
    flags |= (1 << 2); // Specular
    if (m_config.enable_refraction)     flags |= (1 << 3);
    if (m_config.enable_shadows)        flags |= (1 << 4);
    if (m_sceneHasNonOpaque)            flags |= (1 << 5);
    if (m_sceneHasAlphaMask)            flags |= (1 << 10);
    if (m_config.inline_primary_shadows) flags |= (1 << 6);
    if (m_config.enable_caustics && m_sceneData.hasDielectrics && m_numLights > 0) {
        if (!m_causticsPipeline) {
            initCaustics();
        }
        flags |= (1 << 8);
    }
    if (m_config.enable_delta_unroll) flags |= (1 << 11);
    if (m_videoBillboard && m_videoBillboard->hasNewFrame()) {
        flags |= (1 << 12); // Dynamic video bypass flag
    }
    if (accumReset || m_cameraMovedLastFrame) {
        flags |= (1 << 23); // Camera motion / history reset flag
    }

    uint32_t imageIndex = 0;
    if (m_presentation) {
        if (!m_presentation->acquireNextImage(m_currentFrame, imageIndex, m_config, [this](uint32_t w, uint32_t h, bool f) { onResize(w, h, f); })) {
            return;
        }
    }

    bool isStationaryAccum = m_config.progressive_accumulation && !m_cameraMovedLastFrame && (m_accumulatedSamples > 1);
    bool isUpscalerActive = (m_config.upscaler_mode == UpscalerMode::FSR3) || (m_config.upscaler_mode == UpscalerMode::Upways) || m_config.upways_superres;
    bool enableJitter = isUpscalerActive;
    uint32_t renderW = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
        static_cast<uint32_t>(m_config.width * m_config.render_scale) : m_config.width;
    uint32_t renderH = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
        static_cast<uint32_t>(m_config.height * m_config.render_scale) : m_config.height;

    CameraUniform ubo = m_camera->getUniformData(m_frameIndex, activeSpp, activeBounces, flags,
                                                 enableJitter, renderW, renderH, 0, /*updatePrev=*/false);
    m_cameraUBOs[m_currentFrame]->copyFrom(&ubo, sizeof(CameraUniform));

    uint32_t groupsX = (m_config.width + 15) / 16;
    uint32_t groupsY = (m_config.height + 15) / 16;
    uint32_t rtGroupsX = (renderW + 7) / 8;
    uint32_t rtGroupsY = (renderH + 3) / 4;

    PostProcessPipeline::TonemapPushConstants tonemapConstants;
    tonemapConstants.exposure = m_config.exposure;
    tonemapConstants.applyACES = m_config.aces_tonemap ? 1 : 0;
    tonemapConstants.visualizeSplit = 0;
    tonemapConstants.tileSize = m_config.tile_size;
    tonemapConstants.displayMode = (m_swapchain && !m_config.headless) ? static_cast<uint32_t>(m_swapchain->getHdrMode()) : 0u;
    tonemapConstants.peakNits = m_config.hdr_peak_nits;
    tonemapConstants.paperWhiteNits = m_config.hdr_paper_white_nits;

    bool isMgpu = (m_mgpu && m_mgpu->isMultiGpuActive() && m_config.mgpu_mode != MultiGpuMode::Off);
    bool useAsyncComputeMerge = false;
    bool asyncPost = false;
    bool hasPostSubmission = true;


    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    VkCommandBuffer activeCmd = cmd;

    uint32_t useHwRT = 1;
    uint32_t hasEnvMap = m_environmentMap ? 1 : 0;
    float envIntensity = 1.0f;
    MultiGpuFramePlan mgpuPlan{};
    uint32_t qBase = m_currentFrame * QUERIES_PER_FRAME;

    if (!isMgpu) {
        // --- Single GPU Execution Path ---
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginInfo);

        // Refined frame-start barrier: target exact read/write hazards on m_accumImage across frames
        if (m_config.progressive_accumulation && !accumReset && m_accumImage) {
            VkImageMemoryBarrier2 accumStartBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            accumStartBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            accumStartBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            accumStartBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            accumStartBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            accumStartBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            accumStartBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            accumStartBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            accumStartBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            accumStartBarrier.image = m_accumImage->getImage();
            accumStartBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo frameStartDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            frameStartDep.imageMemoryBarrierCount = 1;
            frameStartDep.pImageMemoryBarriers = &accumStartBarrier;
            vkCmdPipelineBarrier2(cmd, &frameStartDep);
        }

        vkCmdResetQueryPool(cmd, m_queryPool, qBase, QUERIES_PER_FRAME);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_queryPool, qBase + 0);

        // GPU-Timeline TLAS Update / Refit (Tier 3)
        if (m_tlasUpdatePipeline && m_tlasUpdatePipeline->needsGpuUpdate() && m_tlas) {
            recordGpuTlasUpdate(cmd, true);
        }

        // Update animated video billboard texture if a new frame is ready
        if (m_videoBillboard && !getSceneTextures().empty()) {
            m_videoBillboard->uploadFrame(cmd, m_currentFrame, getSceneTextures()[0].get());
        }

        if (!skipRayTracing && m_rtOrchestrator) {
            RayTracingDispatchParams rtParams{};
            rtParams.frameSlot = m_currentFrame;
            rtParams.dispatchWidth = renderW;
            rtParams.dispatchHeight = renderH;
            rtParams.fullWidth = renderW;
            rtParams.fullHeight = renderH;
            rtParams.tileOffsetX = 0;
            rtParams.tileOffsetY = 0;
            rtParams.spp = activeSpp;
            rtParams.bounces = activeBounces;
            rtParams.fractionalSpp = activeFractionalSpp;
            rtParams.totalCompositeSpp = 0;
            rtParams.cameraFlags = flags;
            rtParams.accumReset = accumReset;
            rtParams.isMultiGpu = false;
            rtParams.mgpuMode = MultiGpuMode::Off;
            rtParams.skipRayTracing = false;
            rtParams.frameIndex = m_frameIndex;
            rtParams.cameraMovedLastFrame = m_cameraMovedLastFrame;
            rtParams.diagnosticHalfTiles = m_config.diagnostic_half_tiles;
            rtParams.isRestirActive = isRestirActive();

            m_rtOrchestrator->recordRayTracing(
                cmd, rtParams,
                m_frameImages[m_currentFrame],
                m_accumImage,
                m_mlDiffuseImage,
                m_mlSpecularImage,
                m_normalDepthImage,
                m_prevNormalDepthImage,
                m_centerDepthBuffers[m_currentFrame].get(),
                m_rtDescSets[m_currentFrame],
                m_causticsPipeline.get(),
                m_governor.get(),
                m_camera.get(),
                m_sceneData,
                m_numTriangles, m_numSpheres, m_numMaterials, m_numLights, m_numOpaqueTriangles,
                hasEnvMap, envIntensity, useHwRT,
                m_config
            );
        }

        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_queryPool, qBase + 1);

        bool needUpscaler = (m_config.denoiser_mode == DenoiserMode::Upways ||
                             m_config.upscaler_mode == UpscalerMode::Upways ||
                             m_config.upscaler_mode == UpscalerMode::FSR3);

        if (!needUpscaler && m_postProcess && m_postProcess->hasFusedPipeline() && !skipRayTracing) {
            // Fused Accumulation Running Average + ACES Tonemapping (Zero-Copy Register Pass Fusion)
            // Eliminates intermediate compute pipeline barrier and 132.7 MB round-trip VRAM read of m_accumImage.
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 2);

            PostProcessPipeline::FusedAccumTonemapPushConstants fusedPC{};
            fusedPC.width = renderW;
            fusedPC.height = renderH;
            fusedPC.sampleCount = (m_config.progressive_accumulation && !accumReset) ? m_accumulatedSamples : 1u;
            fusedPC.invSpp = 1.0f;
            fusedPC.exposure = m_config.exposure;
            fusedPC.applyACES = m_config.aces_tonemap ? 1u : 0u;
            fusedPC.displayMode = (m_swapchain && !m_config.headless) ? static_cast<uint32_t>(m_swapchain->getHdrMode()) : 0u;
            fusedPC.peakNits = m_config.hdr_peak_nits;
            fusedPC.paperWhiteNits = m_config.hdr_paper_white_nits;
            fusedPC.pad = 0;

            m_postProcess->recordFusedAccumTonemap(cmd, m_currentFrame, fusedPC);

            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 3);
        } else {
            // Decoupled Path (Used when Upways or FSR3 is active or RT was skipped)
            // Running Average Accumulation Pass (FP16 Frame -> FP32 Persistent History)
            if (!skipRayTracing && m_postProcess && m_postProcess->hasRunningAvgPipeline()) {
                PostProcessPipeline::RunningAvgPushConstants avgPC{};
                avgPC.width = renderW;
                avgPC.height = renderH;
                avgPC.sampleCount = (m_config.progressive_accumulation && !accumReset) ? m_accumulatedSamples : 1u;
                avgPC.invSpp = 1.0f;

                m_postProcess->recordRunningAvg(cmd, m_currentFrame, avgPC);

                VkMemoryBarrier2 avgBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                avgBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                avgBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                avgBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                avgBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

                VkDependencyInfo avgDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                avgDep.memoryBarrierCount = 1;
                avgDep.pMemoryBarriers = &avgBarrier;
                vkCmdPipelineBarrier2(cmd, &avgDep);
            }

            // Upways Neural Denoiser & Super-Resolution (Wave32 WMMA)
            bool resetTemporal = hardReset || m_temporalResetRequested;
            m_temporalResetRequested = false;
            bool upwaysRun = false;
            if (m_config.denoiser_mode == DenoiserMode::Upways || m_config.upscaler_mode == UpscalerMode::Upways) {
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 4);
                if (m_superResolution) {
                    upwaysRun = m_superResolution->dispatchUpways(
                        cmd, resetTemporal, m_config, m_camera.get(),
                        m_frameIndex, m_cameraMovedLastFrame, m_accumulatedSamples,
                        m_mlAlbedoRoughnessImage, m_normalDepthImage,
                        m_postProcess.get(), m_outputImage
                    );
                }
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 5);
            }

            // AMD FidelityFX Super Resolution 3.1
            bool fsr3Run = false;
            if (!upwaysRun && m_config.upscaler_mode == UpscalerMode::FSR3) {
                if (m_superResolution) {
                    fsr3Run = m_superResolution->dispatchFsr3(
                        cmd, resetTemporal, m_config, m_frameIndex,
                        m_cameraMovedLastFrame, m_accumulatedSamples,
                        m_accumImage, m_normalDepthImage, m_motionVectorImage,
                        m_mgpu.get(), m_secTransferBuffer, m_currentFrame,
                        m_postProcess.get(), m_outputImage,
                        [this]() { updateFsr3Descriptors(); }
                    );
                }
            }

            // Tonemapping
            VkDescriptorSet tmSet = m_postProcess ? m_postProcess->getTonemapDescSet() : VK_NULL_HANDLE;
            if (m_superResolution) {
                if (fsr3Run || (m_config.upscaler_mode == UpscalerMode::FSR3 && m_superResolution->getFsr3Upscaler())) {
                    tmSet = m_superResolution->getTonemapFsr3DescSet();
                    tonemapConstants.totalSamples = 1u;
                } else if (upwaysRun || (m_config.upscaler_mode == UpscalerMode::Upways && m_superResolution->getUpwaysPipeline())) {
                    tmSet = m_superResolution->getTonemapUpwaysDescSet();
                    tonemapConstants.totalSamples = 1u;
                } else {
                    tonemapConstants.totalSamples = 1u;
                }
            }

            tonemapConstants.visualizeSplit = 0;
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 2);
            if (m_postProcess) {
                m_postProcess->recordTonemap(cmd, tmSet, m_config.width, m_config.height, tonemapConstants);
            }
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 3);
        }
    } else {
        // --- Multi-GPU Path (Checkerboard Tiling or Sample Parallelism) ---
        hasPostSubmission = true;

        MultiGpuPlanParams planParams{
            m_config,
            m_mgpu.get(),
            m_governor.get(),
            m_camera.get(),
            m_secTransferBuffer,
            m_cameraUBOs[m_currentFrame].get(),
            ubo,
            m_currentFrame,
            m_frameIndex,
            activeSpp,
            activeBounces,
            activeFractionalSpp,
            flags,
            accumReset,
            cameraMovedThisFrame,
            m_cameraMovedLastFrame,
            hardReset,
            skipRayTracing,
            renderW,
            renderH,
            m_numTriangles,
            m_numSpheres,
            m_numMaterials,
            m_numLights,
            m_numOpaqueTriangles,
            hasEnvMap,
            envIntensity,
            useHwRT,
            m_temporalResetRequested,
            m_accumulatedSamples
        };

        if (m_mgpuCoordinator) {
            mgpuPlan = m_mgpuCoordinator->planAndLaunchSecondary(planParams, m_accumulatedSamples, accumReset);

            auto onPreRecord = [this](VkCommandBuffer c) {
                // GPU-Timeline TLAS Update / Refit (Tier 3)
                if (m_tlasUpdatePipeline && m_tlasUpdatePipeline->needsGpuUpdate() && m_tlas) {
                    recordGpuTlasUpdate(c, true);
                }
                // Update animated video billboard texture if a new frame is ready
                if (m_videoBillboard && !getSceneTextures().empty()) {
                    m_videoBillboard->uploadFrame(c, m_currentFrame, getSceneTextures()[0].get());
                }
            };

            m_mgpuCoordinator->recordPrimaryRayTracing(
                cmd, queue, device,
                mgpuPlan, planParams,
                qBase, m_queryPool,
                m_rtCompleteSemaphores[m_currentFrame],
                m_rtOrchestrator.get(),
                m_frameImages[m_currentFrame],
                m_accumImage,
                m_mlDiffuseImage,
                m_mlSpecularImage,
                m_normalDepthImage,
                m_prevNormalDepthImage,
                m_rtDescSets[m_currentFrame],
                m_causticsPipeline.get(),
                m_sceneData,
                onPreRecord
            );

            // Record Merge commands on primary GPU into postCmd
            activeCmd = m_postCommandBuffers[m_currentFrame];
            vkResetCommandBuffer(activeCmd, 0);
            VkCommandBufferBeginInfo postBeginInfo{};
            postBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
            vkBeginCommandBuffer(activeCmd, &postBeginInfo);

            m_mgpuCoordinator->recordMergePass(
                activeCmd, mgpuPlan, planParams,
                m_postProcess.get(),
                m_frameImages[mgpuPlan.slot],
                m_mlDiffuseImage
            );
        }

        // Post-processing passes on activeCmd: Upways / FSR3 / Tonemap
        bool resetTemporal = hardReset || m_temporalResetRequested;
        m_temporalResetRequested = false;
        bool upwaysRun = false;
        if (m_config.denoiser_mode == DenoiserMode::Upways || m_config.upscaler_mode == UpscalerMode::Upways) {
            vkCmdWriteTimestamp2(activeCmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 4);
            if (m_superResolution) {
                upwaysRun = m_superResolution->dispatchUpways(
                    activeCmd, resetTemporal, m_config, m_camera.get(),
                    m_frameIndex, m_cameraMovedLastFrame, m_accumulatedSamples,
                    m_mlAlbedoRoughnessImage, m_normalDepthImage,
                    m_postProcess.get(), m_outputImage
                );
            }
            vkCmdWriteTimestamp2(activeCmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 5);
        }

        bool fsr3Run = false;
        if (!upwaysRun && m_config.upscaler_mode == UpscalerMode::FSR3) {
            if (m_superResolution) {
                fsr3Run = m_superResolution->dispatchFsr3(
                    activeCmd, resetTemporal, m_config, m_frameIndex,
                    m_cameraMovedLastFrame, m_accumulatedSamples,
                    m_accumImage, m_normalDepthImage, m_motionVectorImage,
                    m_mgpu.get(), m_secTransferBuffer, m_currentFrame,
                    m_postProcess.get(), m_outputImage,
                    [this]() { updateFsr3Descriptors(); }
                );
            }
        }

        VkDescriptorSet tmSet = m_postProcess ? m_postProcess->getTonemapDescSet() : VK_NULL_HANDLE;
        if (m_superResolution) {
            if (fsr3Run || (m_config.upscaler_mode == UpscalerMode::FSR3 && m_superResolution->getFsr3Upscaler())) {
                tmSet = m_superResolution->getTonemapFsr3DescSet();
                tonemapConstants.totalSamples = 1u;
            } else if (upwaysRun || (m_config.upscaler_mode == UpscalerMode::Upways && m_superResolution->getUpwaysPipeline())) {
                tmSet = m_superResolution->getTonemapUpwaysDescSet();
                tonemapConstants.totalSamples = 1u;
            } else {
                tonemapConstants.totalSamples = 1u;
            }
        }

        bool isCheckerboard = (m_mgpu && m_mgpu->isMultiGpuActive() && m_config.mgpu_mode != MultiGpuMode::Off);
        if (isCheckerboard) {
            MultiGpuMode effectiveMode = m_config.mgpu_mode;
            if (effectiveMode == MultiGpuMode::Auto) {
                effectiveMode = (m_config.spp > 1) ? MultiGpuMode::SampleParallel : MultiGpuMode::CheckerboardTile;
            }
            if (effectiveMode == MultiGpuMode::SampleParallel) {
                isCheckerboard = false;
            }
        }
        tonemapConstants.visualizeSplit = (m_config.visualize_mgpu_split && isCheckerboard) ? 1u : 0u;
        tonemapConstants.tileSize = m_config.tile_size;
        tonemapConstants.displayMode = (m_swapchain && !m_config.headless) ? static_cast<uint32_t>(m_swapchain->getHdrMode()) : 0u;
        tonemapConstants.peakNits = m_config.hdr_peak_nits;
        tonemapConstants.paperWhiteNits = m_config.hdr_paper_white_nits;
        vkCmdWriteTimestamp2(activeCmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 2);
        if (m_postProcess) {
            m_postProcess->recordTonemap(activeCmd, tmSet, m_config.width, m_config.height, tonemapConstants);
        }
        vkCmdWriteTimestamp2(activeCmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 3);
    }

    // 3. Interactive Blit & Dear ImGui Overlay
    if (m_presentation) {
        GuiActions guiActions{};
        if (m_presentation->recordPresentation(
            activeCmd, imageIndex, m_outputImage, m_gui.get(), m_config, getStats(),
            isCameraMode(), m_camera.get(), &guiActions, m_availableScenes, m_currentSceneIndex
        )) {
            m_resetAccumulation = true;
            if (!m_config.headless) m_frameTimesMs.clear();
        }
        m_uiDumpBuffer = m_presentation->getUiDumpBuffer();
        if (guiActions.sceneChanged) {
            m_pendingSceneChange = true;
            m_pendingScenePath = guiActions.newScenePath;
        }
        if (guiActions.mgpuModeChanged) {
            m_pendingMgpuModeChange = true;
            m_newMgpuMode = guiActions.newMgpuMode;
        }
        if (guiActions.mgpuUpscaleModeChanged) {
            m_pendingMgpuUpscaleModeChange = true;
            m_newMgpuUpscaleMode = guiActions.newMgpuUpscaleMode;
        }
        if (guiActions.accumFormatChanged) {
            m_pendingAccumFormatChange = true;
            m_newAccumFormat = guiActions.newAccumFormat;
        }
        if (guiActions.doubleBufferChanged) {
            m_pendingDoubleBufferChange = true;
            m_newDoubleBuffer = guiActions.newDoubleBuffer;
        }
        if (guiActions.tileSizeChanged) {
            m_pendingTileSizeChange = true;
            m_newTileSize = guiActions.newTileSize;
        }
        if (guiActions.resetAccumulation) {
            m_resetAccumulation = true;
            if (!m_config.headless) m_frameTimesMs.clear();
        }
        if (guiActions.exportTelemetry) {
            exportTelemetry(guiActions.exportTelemetryPath);
        }
        if (guiActions.refreshPciStatus) {
            refreshPciStatus();
        }
        if (guiActions.toggleFullscreen) {
            m_pendingToggleFullscreen = true;
        }
        if (guiActions.requestedWidth > 0 && guiActions.requestedHeight > 0) {
            m_pendingResizeW = guiActions.requestedWidth;
            m_pendingResizeH = guiActions.requestedHeight;
        }
        if (guiActions.takeScreenshot) {
            m_pendingScreenshot = true;
        }
    }

    if (hasPostSubmission) {
        vkEndCommandBuffer(activeCmd);
    }

    // Wait for secondary GPU completion of slot and PCIe transfer (if MGPU)
    if (isMgpu && !skipRayTracing && m_mgpuCoordinator) {
        m_mgpuCoordinator->syncSecondaryTransfer(m_mgpu.get(), mgpuPlan, skipRayTracing);
    }

    if (hasPostSubmission) {
        // Submit Work
        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = activeCmd;

        std::vector<VkSemaphoreSubmitInfo> waitSemaphoreInfos;
        std::vector<VkSemaphoreSubmitInfo> signalSemaphoreInfos;

        if (isMgpu && m_mgpuCoordinator) {
            m_mgpuCoordinator->populatePostWaitSemaphores(waitSemaphoreInfos, m_mgpu.get(), m_currentFrame, m_rtCompleteSemaphores[m_currentFrame], skipRayTracing);
        }

        if (!m_config.headless && m_swapchain && m_presentation) {
            VkSemaphoreSubmitInfo waitImg{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            waitImg.semaphore = m_presentation->getImageAvailableSemaphore(m_currentFrame);
            waitImg.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
            waitSemaphoreInfos.push_back(waitImg);

            VkSemaphoreSubmitInfo sigRender{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            sigRender.semaphore = m_presentation->getRenderFinishedSemaphore(imageIndex);
            sigRender.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
            signalSemaphoreInfos.push_back(sigRender);
        }

        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        submitInfo.waitSemaphoreInfoCount = static_cast<uint32_t>(waitSemaphoreInfos.size());
        submitInfo.pWaitSemaphoreInfos = waitSemaphoreInfos.data();
        submitInfo.signalSemaphoreInfoCount = static_cast<uint32_t>(signalSemaphoreInfos.size());
        submitInfo.pSignalSemaphoreInfos = signalSemaphoreInfos.data();

        VkFence inFlightFence = m_presentation ? m_presentation->getInFlightFence(m_currentFrame) : VK_NULL_HANDLE;
        if (inFlightFence != VK_NULL_HANDLE) {
            vkResetFences(device, 1, &inFlightFence);
        }
        vkQueueSubmit2(queue, 1, &submitInfo, inFlightFence);
    }


    if (m_presentation) {
        m_presentation->present(queue, imageIndex, [this](uint32_t w, uint32_t h, bool f) { onResize(w, h, f); });
    }

    // Execute pending window resolution / fullscreen actions outside of command buffer recording
    if (m_pendingToggleFullscreen) {
        m_pendingToggleFullscreen = false;
        m_window->toggleFullscreen();
    }
    if (m_pendingResizeW > 0 && m_pendingResizeH > 0) {
        uint32_t w = m_pendingResizeW;
        uint32_t h = m_pendingResizeH;
        m_pendingResizeW = 0;
        m_pendingResizeH = 0;
        m_window->setWindowResolution(w, h);
    }

    // High-precision frame pacing if target FPS is set
    if (m_presentation) {
        m_presentation->paceFrame(m_config, m_governor.get(), accumReachedCutoff);
    }

    // Screenshot capture trigger
    if (m_pendingScreenshot) {
        m_pendingScreenshot = false;
        saveScreenshot();
    }

    if (m_screenshotNotificationTimer > 0.0f) {
        float dt = (m_lastPresentationTimeMs > 0.01 && m_lastPresentationTimeMs < 1000.0)
            ? static_cast<float>(m_lastPresentationTimeMs * 0.001)
            : 0.016f;
        m_screenshotNotificationTimer = std::max(0.0f, m_screenshotNotificationTimer - dt);
    }

    m_frameIndex++;
    m_totalFramesRendered++;
    m_currentFrame = (m_currentFrame + 1) % MAX_FRAMES_IN_FLIGHT;

    bool shouldLog = false;
    if (m_config.benchmark) {
        shouldLog = true;
    } else if (m_config.log_interval_sec > 0.0f) {
        auto now = std::chrono::steady_clock::now();
        double elapsedSec = std::chrono::duration<double>(now - m_lastLogTime).count();
        if (elapsedSec >= m_config.log_interval_sec) {
            shouldLog = true;
            m_lastLogTime = now;
        }
    }

    if (shouldLog) {
        double gpuRtMs = m_lastGpuRtMs;
        double secGpuMs = m_lastSecGpuMs;
        double gpuTonemapMs = m_lastTonemapMs;
        double totalGpuMs = (m_mgpu && m_mgpu->isMultiGpuActive()) ? (std::max(gpuRtMs, secGpuMs) + gpuTonemapMs) : (gpuRtMs + gpuTonemapMs);
        if (m_governor && m_config.adaptive_spp && m_governor->getState().active) {
            uint32_t curSpp = m_governor->getState().currentSpp;
            uint32_t curBounces = m_governor->getState().currentBounces;
            if (m_mgpu && m_mgpu->isMultiGpuActive()) {
                Logger::info("Frame {:3d} | Dual-GPU: {:.3f} ms (GPU 0: {:.3f} ms [{} SPP], GPU 1: {:.3f} ms [{} SPP]) | Dynamic: {} SPP, {} Bounces | Target {} FPS",
                             m_totalFramesRendered, totalGpuMs, gpuRtMs, m_governor->getState().primSpp, secGpuMs, m_governor->getState().secSpp,
                             curSpp, curBounces, m_config.target_fps);
            } else {
                Logger::info("Frame {:3d} | Single GPU: {:.3f} ms (RT: {:.3f} ms, Tonemap: {:.3f} ms) | Dynamic: {} SPP, {} Bounces | Target {} FPS",
                             m_totalFramesRendered, totalGpuMs, gpuRtMs, gpuTonemapMs,
                             curSpp, curBounces, m_config.target_fps);
            }
        } else if (m_mgpu && m_mgpu->isMultiGpuActive()) {
            Logger::info("Frame {:3d} | Dual-GPU Total: {:.3f} ms (GPU 0: {:.3f} ms, GPU 1: {:.3f} ms, Tonemap: {:.3f} ms) | Target <{:.1f}ms: {}",
                         m_totalFramesRendered, totalGpuMs, gpuRtMs, secGpuMs, gpuTonemapMs,
                         m_config.target_frame_time_ms,
                         totalGpuMs < m_config.target_frame_time_ms ? "\033[32mPASS\033[0m" : "\033[33mCHECK\033[0m");
        } else {
            Logger::info("Frame {:3d} | Single GPU Total: {:.3f} ms (RT: {:.3f} ms, Tonemap: {:.3f} ms) | Target <{:.1f}ms: {}",
                         m_totalFramesRendered, totalGpuMs, gpuRtMs, gpuTonemapMs,
                         m_config.target_frame_time_ms,
                         totalGpuMs < m_config.target_frame_time_ms ? "\033[32mPASS\033[0m" : "\033[33mCHECK\033[0m");
        }
    }
}

void Engine::dumpOutputFiles() {
    if (m_telemetryReporter) {
        m_telemetryReporter->dumpOutputFiles();
    }
}

FrameStats Engine::getStats() const {
    if (m_telemetryReporter) {
        return m_telemetryReporter->getStats();
    }
    return FrameStats{};
}

std::string Engine::exportTelemetry(const std::string& customPath) {
    if (m_hwMonitor) {
        return m_hwMonitor->exportTelemetry(customPath, getStats());
    }
    return "";
}

void Engine::onResize(uint32_t newWidth, uint32_t newHeight, bool forceRecreate) {
    if (m_config.headless || !m_swapchain) return;
    if (newWidth == 0 || newHeight == 0) return;

    newWidth = std::max(64u, newWidth);
    newHeight = std::max(64u, newHeight);

    if (!forceRecreate &&
        newWidth == m_config.width && newHeight == m_config.height &&
        m_swapchain->getExtent().width == newWidth && m_swapchain->getExtent().height == newHeight) {
        m_resetAccumulation = true;
        return;
    }

    if (m_presentation) {
        m_presentation->onResize(newWidth, newHeight, m_config, forceRecreate);
        syncPresentationPointers();
    }

    VkDevice device = m_context->getDevice();
    VkQueue queue = m_context->getGraphicsQueue();

    vkDeviceWaitIdle(device);

    auto concurrentQueues = getConcurrentQueueFamilies();

    // 3. Recreate Accumulation & Output Images
    if (m_renderTargets) {
        m_renderTargets->resize(
            m_config.width, m_config.height, m_config.accum_format,
            m_swapchain && !m_config.headless && m_swapchain->isHdr(),
            m_swapchain ? m_swapchain->getFormat() : VK_FORMAT_UNDEFINED,
            concurrentQueues,
            m_commandBuffers[0]
        );
    }
    syncRenderTargetPointers();

    // 5. Resize secondary GPU if active before updating merge descriptor set
    if (m_mgpu && m_mgpu->isMultiGpuActive()) {
        m_mgpu->resize(m_config.width, m_config.height);
    }

    destroyGBufferResources();
    createGBufferResources();
    if (m_superResolution) {
        m_superResolution->resize(
            m_config.width, m_config.height, m_config.render_scale,
            m_config.upscaler_mode, m_config.denoiser_mode,
            m_config.upways_superres, m_config.accum_format,
            m_postProcess.get(), m_commandBuffers[0]
        );
    }
    if (m_causticsPipeline) {
        m_causticsPipeline->resize(m_config.width, m_config.height, [this](auto recordFn) {
            VkCommandBuffer cmd = m_commandBuffers[0];
            vkResetCommandBuffer(cmd, 0);
            VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            vkBeginCommandBuffer(cmd, &bi);
            recordFn(cmd);
            vkEndCommandBuffer(cmd);
            VkCommandBufferSubmitInfo csi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
            csi.commandBuffer = cmd;
            VkSubmitInfo2 si{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
            si.commandBufferInfoCount = 1;
            si.pCommandBufferInfos = &csi;
            vkQueueSubmit2(m_context->getGraphicsQueue(), 1, &si, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_context->getGraphicsQueue());
        });
    }

    if (m_rtOrchestrator) {
        m_rtOrchestrator->resize(m_config.width, m_config.height, m_config);
        m_currentBatchCount = m_rtOrchestrator->getCurrentBatchCount();
        m_currentBatchPixels = m_rtOrchestrator->getCurrentBatchPixels();
    }

    updateAllImageDescriptors();
    updateMergeDescriptors();

    // 7. Adapt Camera aspect ratio
    if (m_camera) {
        float aspect = static_cast<float>(m_config.width) / static_cast<float>(m_config.height);
        m_camera->setAspect(aspect);
    }

    // 9. Invalidate accumulation
    m_frameIndex = 0;
    m_resetAccumulation = true;

    // 10. Prune transient startup tallies if window layout resizes during engine initialization settling phase
    if (m_totalFramesRendered <= m_config.warmup_frames + MAX_FRAMES_IN_FLIGHT + 2) {
        if (m_telemetryReporter) {
            m_telemetryReporter->getConfigTallies().clear();
        }
        m_frameTimesMs.clear();
    }
}

std::string Engine::getActiveSceneName() const {
    if (m_sceneManager) {
        return m_sceneManager->getActiveSceneName(m_config.scene_path);
    }
    return "Procedural Cornell Box";
}

void Engine::recordFrameTally(double frameTimeMs, double primRtMs, double secRtMs, double tonemapMs,
                              const WavefrontStageSample* wfSample) {
    if (m_telemetryReporter) {
        m_telemetryReporter->recordFrameTally(frameTimeMs, primRtMs, secRtMs, tonemapMs, wfSample);
    }
}

void Engine::printExecutionSummary() const {
    if (m_telemetryReporter) {
        m_telemetryReporter->printExecutionSummary();
    }
}

std::string Engine::generateScreenshotFilename() const {
    std::string sceneSlug;
    if (!m_config.scene_path.empty() && !SceneManager::isProceduralCornellBoxPath(m_config.scene_path)) {
        sceneSlug = std::filesystem::path(m_config.scene_path).stem().string();
    } else {
        sceneSlug = getActiveSceneName();
    }

    std::string cleanScene;
    for (char c : sceneSlug) {
        if (std::isalnum(static_cast<unsigned char>(c))) {
            cleanScene += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        } else if (c == ' ' || c == '-' || c == '_') {
            if (!cleanScene.empty() && cleanScene.back() != '_') {
                cleanScene += '_';
            }
        }
    }
    while (!cleanScene.empty() && cleanScene.back() == '_') {
        cleanScene.pop_back();
    }
    if (cleanScene.empty()) {
        cleanScene = "scene";
    }

    uint32_t activeSpp = m_config.spp;
    uint32_t activeBounces = (m_dynamicWavefrontBounces > 0) ? m_dynamicWavefrontBounces : m_config.max_bounces;
    if (m_governor && m_config.adaptive_spp && m_governor->getState().active) {
        activeSpp = m_governor->getState().currentSpp;
        activeBounces = m_governor->getState().currentBounces;
    }

    uint32_t accumSamples = m_accumulatedSamples;
    if (accumSamples == 0) {
        accumSamples = 1;
    }

    std::time_t now = std::time(nullptr);
    std::tm tmNow{};
#ifdef _WIN32
    localtime_s(&tmNow, &now);
#else
    localtime_r(&now, &tmNow);
#endif
    char timeBuf[32];
    std::strftime(timeBuf, sizeof(timeBuf), "%Y%m%d_%H%M%S", &tmNow);

    std::string filename = std::format("pw_{}_{}spp_{}bounce_{}accum_{}.png",
                                       cleanScene, activeSpp, activeBounces, accumSamples, timeBuf);
    return (std::filesystem::path("screenshots") / filename).string();
}

bool Engine::saveScreenshot(const std::string& customPath) {
    if (!m_outputImage || !m_context) {
        Logger::error("saveScreenshot failed: outputImage or VulkanContext is null");
        return false;
    }

    std::string filepath = customPath.empty() ? generateScreenshotFilename() : customPath;

    try {
        std::filesystem::path p(filepath);
        if (p.has_parent_path()) {
            std::filesystem::create_directories(p.parent_path());
        }
    } catch (const std::exception& e) {
        Logger::error("Failed to create directory for screenshot {}: {}", filepath, e.what());
    }

    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();
    VkQueue queue = m_context->getGraphicsQueue();

    vkDeviceWaitIdle(device);

    VkFormat outFmt = m_outputImage->getFormat();
    size_t bpp = (outFmt == VK_FORMAT_R16G16B16A16_SFLOAT) ? 8 : 4;
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(m_config.width) * m_config.height * bpp;
    Buffer staging(allocator, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    vkResetCommandBuffer(m_commandBuffers[0], 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(m_commandBuffers[0], &beginInfo);

    m_outputImage->transitionLayout(
        m_commandBuffers[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
    );

    VkBufferImageCopy copyRegion{};
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageExtent = { m_config.width, m_config.height, 1 };

    vkCmdCopyImageToBuffer(m_commandBuffers[0], m_outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.getBuffer(), 1, &copyRegion);

    m_outputImage->transitionLayout(
        m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    );

    vkEndCommandBuffer(m_commandBuffers[0]);

    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = m_commandBuffers[0];

    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
    vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    staging.invalidate();

    // Ensure any previously pending background screenshot has completed
    if (m_pendingScreenshotFuture.valid()) {
        m_pendingScreenshotFuture.wait();
    }

    if (outFmt == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || outFmt == VK_FORMAT_A2R10G10B10_UNORM_PACK32) {
        const uint32_t* src32 = static_cast<const uint32_t*>(staging.map());
        bool isRgb = (outFmt == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
        bool force8bit = m_config.dump_8bit_png;
        if (force8bit) {
            std::vector<uint8_t> rgba8(static_cast<size_t>(m_config.width) * m_config.height * 4);
            for (size_t p = 0; p < static_cast<size_t>(m_config.width) * m_config.height; ++p) {
                uint32_t px = src32[p];
                uint32_t c0 = (px >> 20) & 0x3FF;
                uint32_t c1 = (px >> 10) & 0x3FF;
                uint32_t c2 = px & 0x3FF;
                uint32_t a2 = (px >> 30) & 0x03;
                uint32_t r10 = isRgb ? c0 : c2;
                uint32_t g10 = c1;
                uint32_t b10 = isRgb ? c2 : c0;
                rgba8[p * 4 + 0] = static_cast<uint8_t>((r10 * 255 + 511) / 1023);
                rgba8[p * 4 + 1] = static_cast<uint8_t>((g10 * 255 + 511) / 1023);
                rgba8[p * 4 + 2] = static_cast<uint8_t>((b10 * 255 + 511) / 1023);
                rgba8[p * 4 + 3] = static_cast<uint8_t>((a2 * 255) / 3);
            }
            staging.unmap();
            m_pendingScreenshotFuture = ImageDumper::savePNGAsync(filepath, m_config.width, m_config.height, std::move(rgba8));
        } else {
            std::vector<uint16_t> rgba16(static_cast<size_t>(m_config.width) * m_config.height * 4);
            for (size_t p = 0; p < static_cast<size_t>(m_config.width) * m_config.height; ++p) {
                uint32_t px = src32[p];
                uint32_t c0 = (px >> 20) & 0x3FF;
                uint32_t c1 = (px >> 10) & 0x3FF;
                uint32_t c2 = px & 0x3FF;
                uint32_t a2 = (px >> 30) & 0x03;
                uint32_t r10 = isRgb ? c0 : c2;
                uint32_t g10 = c1;
                uint32_t b10 = isRgb ? c2 : c0;
                rgba16[p * 4 + 0] = static_cast<uint16_t>((r10 * 65535 + 511) / 1023);
                rgba16[p * 4 + 1] = static_cast<uint16_t>((g10 * 65535 + 511) / 1023);
                rgba16[p * 4 + 2] = static_cast<uint16_t>((b10 * 65535 + 511) / 1023);
                rgba16[p * 4 + 3] = static_cast<uint16_t>((a2 * 65535 + 1) / 3);
            }
            staging.unmap();
            m_pendingScreenshotFuture = ImageDumper::savePNG16Async(filepath, m_config.width, m_config.height, std::move(rgba16));
        }
    } else if (outFmt == VK_FORMAT_R16G16B16A16_SFLOAT) {
        const uint16_t* halfPixels = static_cast<const uint16_t*>(staging.map());
        bool force8bit = m_config.dump_8bit_png;
        float invPaperWhite = 80.0f / (m_config.hdr_paper_white_nits > 0.0f ? m_config.hdr_paper_white_nits : 200.0f);
        if (force8bit) {
            std::vector<uint8_t> rgba8(static_cast<size_t>(m_config.width) * m_config.height * 4);
            for (size_t p = 0; p < static_cast<size_t>(m_config.width) * m_config.height; ++p) {
                for (int c = 0; c < 3; ++c) {
                    float val = glm::detail::toFloat32(halfPixels[p * 4 + c]);
                    float srgb = std::pow(std::clamp(val * invPaperWhite, 0.0f, 1.0f), 1.0f / 2.2f);
                    rgba8[p * 4 + c] = static_cast<uint8_t>(std::clamp(srgb * 255.0f + 0.5f, 0.0f, 255.0f));
                }
                rgba8[p * 4 + 3] = 255;
            }
            staging.unmap();
            m_pendingScreenshotFuture = ImageDumper::savePNGAsync(filepath, m_config.width, m_config.height, std::move(rgba8));
        } else {
            std::vector<uint16_t> rgba16(static_cast<size_t>(m_config.width) * m_config.height * 4);
            for (size_t p = 0; p < static_cast<size_t>(m_config.width) * m_config.height; ++p) {
                for (int c = 0; c < 3; ++c) {
                    float val = glm::detail::toFloat32(halfPixels[p * 4 + c]);
                    float srgb = std::pow(std::clamp(val * invPaperWhite, 0.0f, 1.0f), 1.0f / 2.2f);
                    rgba16[p * 4 + c] = static_cast<uint16_t>(std::clamp(srgb * 65535.0f + 0.5f, 0.0f, 65535.0f));
                }
                rgba16[p * 4 + 3] = 65535;
            }
            staging.unmap();
            m_pendingScreenshotFuture = ImageDumper::savePNG16Async(filepath, m_config.width, m_config.height, std::move(rgba16));
        }
    } else {
        const uint8_t* rawPixels = static_cast<const uint8_t*>(staging.map());
        std::vector<uint8_t> rgba8(rawPixels, rawPixels + m_config.width * m_config.height * 4);
        staging.unmap();
        m_pendingScreenshotFuture = ImageDumper::savePNGAsync(filepath, m_config.width, m_config.height, std::move(rgba8));
    }

    m_lastScreenshotPath = filepath;
    m_screenshotNotificationTimer = 3.5f;
    Logger::info("Screenshot captured and saving asynchronously to: {}", filepath);
    return true;
}



void Engine::runTrainingDataCapture() {
    if (m_trainingCapture) {
        m_trainingCapture->run();
    }
}

void Engine::run() {
    if (!m_config.capture_training_data_dir.empty()) {
        runTrainingDataCapture();
        return;
    }
    Logger::info("Starting Pathways render loop...");

    while (!m_window->shouldClose()) {
        m_window->pollEvents();

        // Check if minimized/occluded: throttle render loop to prevent runaway GPU power draw (OPT-04)
        if (m_isMinimized || (m_window && m_window->isMinimized())) {
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            continue;
        }

        // Execute pending window resolution / fullscreen actions outside of command buffer recording
        if (m_pendingToggleFullscreen) {
            m_pendingToggleFullscreen = false;
            m_window->toggleFullscreen();
        }
        if (m_pendingResizeW > 0 && m_pendingResizeH > 0) {
            uint32_t w = m_pendingResizeW;
            uint32_t h = m_pendingResizeH;
            m_pendingResizeW = 0;
            m_pendingResizeH = 0;
            m_window->setWindowResolution(w, h);
        }

        renderFrame();

        if (m_config.frame_limit > 0 && m_totalFramesRendered >= (m_config.frame_limit + m_config.warmup_frames)) {
            if (m_config.warmup_frames > 0) {
                Logger::info("Reached frame limit of {} frames ({} measured + {} warmup). Terminating loop.",
                             m_config.frame_limit + m_config.warmup_frames, m_config.frame_limit, m_config.warmup_frames);
            } else {
                Logger::info("Reached frame limit of {} frames. Terminating loop.", m_config.frame_limit);
            }
            break;
        }
    }

    dumpOutputFiles();
}

} // namespace pathways
