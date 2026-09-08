#include "core/Engine.hpp"
#include "core/Logger.hpp"
#include "ui/GuiManager.hpp"
#include "mgpu/MultiGpuManager.hpp"
#include "scene/GltfLoader.hpp"

#include <fstream>
#include <filesystem>
#include <numeric>
#include <algorithm>
#include <thread>
#include <bit>

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
std::string queryOperatingSystem() {
    std::ifstream osRelease("/etc/os-release");
    if (osRelease.is_open()) {
        std::string line;
        while (std::getline(osRelease, line)) {
            if (line.starts_with("PRETTY_NAME=")) {
                std::string val = line.substr(12);
                if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
                    val = val.substr(1, val.size() - 2);
                }
                return val;
            }
        }
    }
#ifdef __linux__
    struct utsname uts{};
    if (uname(&uts) == 0) {
        return std::string(uts.sysname) + " " + uts.release + " (" + uts.machine + ")";
    }
#endif
    return "Linux";
}

std::string queryCpuModel() {
    std::ifstream cpuInfo("/proc/cpuinfo");
    if (cpuInfo.is_open()) {
        std::string line;
        while (std::getline(cpuInfo, line)) {
            if (line.starts_with("model name")) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    size_t first = line.find_first_not_of(" \t", colon + 1);
                    if (first != std::string::npos) {
                        return line.substr(first);
                    }
                }
            }
        }
    }
    return "AMD Threadripper Processor";
}

uint64_t queryTotalRamMB() {
    std::ifstream memInfo("/proc/meminfo");
    if (memInfo.is_open()) {
        std::string line;
        while (std::getline(memInfo, line)) {
            if (line.starts_with("MemTotal:")) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    uint64_t kb = std::strtoull(line.c_str() + colon + 1, nullptr, 10);
                    return kb / 1024;
                }
            }
        }
    }
    return 65536;
}

const std::string& getCachedOS() {
    static const std::string s_os = queryOperatingSystem();
    return s_os;
}

const std::string& getCachedCPU() {
    static const std::string s_cpu = queryCpuModel();
    return s_cpu;
}

uint64_t getCachedRAM() {
    static const uint64_t s_ram = queryTotalRamMB();
    return s_ram;
}
} // namespace

Engine::Engine(const Config& config) : m_config(config) {
    m_startTime = std::chrono::high_resolution_clock::now();
    m_lastFrameTime = m_startTime;
    m_lastLogTime = std::chrono::steady_clock::now();

    Logger::info("Initializing Pathways Engine...");
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

    m_context = std::make_unique<VulkanContext>(m_config);

    if (!m_config.headless && m_window) {
        m_window->setTitle(std::format("Pathways - Vulkan 1.4 Path Tracer ({})", m_context->getShortArchName()));
        m_surface = m_window->createSurface(m_context->getInstance());
        m_swapchain = std::make_unique<Swapchain>(
            m_context->getDevice(),
            m_context->getPhysicalDevice(),
            m_surface,
            m_window->getWidth(),
            m_window->getHeight(),
            m_context->getGraphicsQueueFamily()
        );
        m_config.width = m_swapchain->getExtent().width;
        m_config.height = m_swapchain->getExtent().height;
    }

    float aspect = static_cast<float>(m_config.width) / static_cast<float>(m_config.height);
    m_camera = std::make_unique<Camera>(
        glm::vec3(0.0f, 1.0f, 3.4f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        45.0f,
        aspect
    );
    m_camera->adaptFovForAspect(aspect);

    initVulkan();
    initScene();
    auto physicalDevices = VulkanContext::enumeratePhysicalDevices(m_context->getInstance());
    if (physicalDevices.size() >= 2) {
        m_mgpu = std::make_unique<MultiGpuManager>(m_config, m_context.get(), m_sceneData);
    }
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

    startHwMonThread();

    Logger::info("Pathways Engine initialization complete. Ready to render.");
}

Engine::~Engine() {
    Logger::info("Shutting down Pathways Engine...");
    stopHwMonThread();
    VkDevice device = m_context->getDevice();
    vkDeviceWaitIdle(device);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_inFlightFences[i]) vkDestroyFence(device, m_inFlightFences[i], nullptr);
    }
    if (m_rtFence) vkDestroyFence(device, m_rtFence, nullptr);
    if (m_commandPool) vkDestroyCommandPool(device, m_commandPool, nullptr);

    if (m_queryPool) vkDestroyQueryPool(device, m_queryPool, nullptr);

    for (auto sem : m_imageAvailableSemaphores) {
        if (sem) vkDestroySemaphore(device, sem, nullptr);
    }
    m_imageAvailableSemaphores.clear();

    for (auto sem : m_renderFinishedSemaphores) {
        if (sem) vkDestroySemaphore(device, sem, nullptr);
    }
    m_renderFinishedSemaphores.clear();

    m_rtpKhrPipeline.reset();
    if (m_tonemapPipeline) vkDestroyPipeline(device, m_tonemapPipeline, nullptr);
    if (m_mergePipeline) vkDestroyPipeline(device, m_mergePipeline, nullptr);

    if (m_rtpPipelineLayout) vkDestroyPipelineLayout(device, m_rtpPipelineLayout, nullptr);
    if (m_tonemapPipelineLayout) vkDestroyPipelineLayout(device, m_tonemapPipelineLayout, nullptr);
    if (m_mergePipelineLayout) vkDestroyPipelineLayout(device, m_mergePipelineLayout, nullptr);

    if (m_rtDescLayout) vkDestroyDescriptorSetLayout(device, m_rtDescLayout, nullptr);
    if (m_tonemapDescLayout) vkDestroyDescriptorSetLayout(device, m_tonemapDescLayout, nullptr);
    if (m_mergeDescLayout) vkDestroyDescriptorSetLayout(device, m_mergeDescLayout, nullptr);

    if (m_descriptorPool) vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    m_secTransferBuffer.reset();

    m_gui.reset();
    m_swapchain.reset();
    if (m_surface) {
        vkDestroySurfaceKHR(m_context->getInstance(), m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }
}

static uint64_t readSysfsUint64(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return 0;
    uint64_t val = 0;
    if (file >> val) return val;
    return 0;
}

void Engine::startHwMonThread() {
    m_hwMonRunning = true;
    m_hwMonThread = std::thread([this]() {
        // Initial sample immediately on startup
        sampleHwSensors();

        while (m_hwMonRunning) {
            std::unique_lock<std::mutex> lock(m_hwMonMutex);
            if (m_hwMonCv.wait_for(lock, std::chrono::milliseconds(2000), [this] { return !m_hwMonRunning.load(); })) {
                break;
            }
            sampleHwSensors();
        }
    });
}

void Engine::stopHwMonThread() {
    if (m_hwMonRunning) {
        m_hwMonRunning = false;
        m_hwMonCv.notify_all();
        if (m_hwMonThread.joinable()) {
            m_hwMonThread.join();
        }
    }
}

void Engine::sampleHwSensors() {
    // GPU 0
    if (m_context) {
        const auto& pciInfo = m_context->getPciLinkInfo();
        if (!pciInfo.hwmonPath.empty()) {
            uint64_t rawFreq = readSysfsUint64(pciInfo.hwmonPath + "/freq1_input");
            uint64_t rawTemp = readSysfsUint64(pciInfo.hwmonPath + "/temp1_input");
            if (rawFreq > 0) {
                m_gpu0ClockMhz.store(static_cast<uint32_t>(rawFreq / 1000000ULL), std::memory_order_relaxed);
            }
            if (rawTemp > 0) {
                m_gpu0TempC.store(static_cast<uint32_t>(rawTemp / 1000ULL), std::memory_order_relaxed);
            }
        }
    }

    // GPU 1
    if (m_mgpu && m_mgpu->getSecondaryContext()) {
        const auto& secPciInfo = m_mgpu->getSecondaryContext()->getPciLinkInfo();
        if (!secPciInfo.hwmonPath.empty()) {
            uint64_t rawFreq = readSysfsUint64(secPciInfo.hwmonPath + "/freq1_input");
            uint64_t rawTemp = readSysfsUint64(secPciInfo.hwmonPath + "/temp1_input");
            if (rawFreq > 0) {
                m_gpu1ClockMhz.store(static_cast<uint32_t>(rawFreq / 1000000ULL), std::memory_order_relaxed);
            }
            if (rawTemp > 0) {
                m_gpu1TempC.store(static_cast<uint32_t>(rawTemp / 1000ULL), std::memory_order_relaxed);
            }
        }
    }
}

void Engine::refreshPciStatus() {
    if (m_context) {
        m_context->refreshPciLinkInfo();
    }
    if (m_mgpu && m_mgpu->getSecondaryContext()) {
        m_mgpu->getSecondaryContext()->refreshPciLinkInfo();
    }
    sampleHwSensors();
    Logger::info("Manually refreshed PCIe status.");
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

    // Command buffers (Double-buffered)
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    vkAllocateCommandBuffers(device, &allocInfo, m_commandBuffers.data());

    // Create Render Target Images
    VkFormat accumFmt = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    m_accumImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        accumFmt,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_outputImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // Transition layouts to GENERAL
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_commandBuffers[0], &beginInfo);

    m_accumImage->transitionLayout(
        m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    m_outputImage->transitionLayout(
        m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    vkEndCommandBuffer(m_commandBuffers[0]);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffers[0];
    vkQueueSubmit(m_context->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_context->getGraphicsQueue());
}

void Engine::initScene() {
    VmaAllocator allocator = m_context->getAllocator();

    m_availableScenes = SceneRegistry::scan("scenes");
    m_currentSceneIndex = 0;

    if (!m_config.scene_path.empty()) {
        Logger::info("Loading user specified scene: {}", m_config.scene_path);
        m_sceneData = GltfLoader::loadSceneData(m_config.scene_path);
        m_currentSceneIndex = -1;
        for (size_t i = 0; i < m_availableScenes.size(); ++i) {
            std::error_code ec;
            if (m_availableScenes[i].filepath == m_config.scene_path ||
                (!m_availableScenes[i].filepath.empty() &&
                 std::filesystem::exists(m_availableScenes[i].filepath) &&
                 std::filesystem::exists(m_config.scene_path) &&
                 std::filesystem::equivalent(m_availableScenes[i].filepath, m_config.scene_path, ec))) {
                m_currentSceneIndex = static_cast<int>(i);
                break;
            }
        }
    } else {
        m_sceneData = ProceduralScene::createCornellBox();
        m_currentSceneIndex = 0;
    }

    if (m_sceneData.triangles.empty() && m_sceneData.spheres.empty()) {
        Logger::warn("Loaded scene contains no renderable geometry! Falling back to procedural Cornell Box.");
        m_sceneData = ProceduralScene::createCornellBox();
    }

    m_numTriangles = static_cast<uint32_t>(m_sceneData.triangles.size());
    m_numSpheres = static_cast<uint32_t>(m_sceneData.spheres.size());
    m_numMaterials = static_cast<uint32_t>(m_sceneData.materials.size());
    m_numLights = static_cast<uint32_t>(m_sceneData.lights.size());
    updateSceneTransparencyFlag();

    Logger::info("Active Scene: {} Triangles, {} Spheres, {} Materials, {} Lights (Non-Opaque: {})",
                 m_numTriangles, m_numSpheres, m_numMaterials, m_numLights, m_sceneHasNonOpaque ? "YES" : "NO");

    if (m_camera) {
        m_camera->setSceneScale(m_sceneData.sceneRadius, m_sceneData.focalDistance, m_sceneData.centralTarget);
        if (m_sceneData.hasCamera) {
            m_camera->lookAt(m_sceneData.cameraPosition, m_sceneData.cameraTarget, m_sceneData.cameraUp);
            m_camera->setFov(m_sceneData.cameraFov);
            m_camera->setDefaultFraming(m_sceneData.cameraPosition, m_sceneData.cameraTarget, m_sceneData.cameraFov);
        }
    }

    // Triangle buffer
    VkDeviceSize triSize = std::max(sizeof(TriangleGPU) * m_sceneData.triangles.size(), sizeof(TriangleGPU));
    m_triangleBuffer = std::make_unique<Buffer>(
        allocator, triSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.triangles.empty()) {
        m_triangleBuffer->copyFrom(m_sceneData.triangles.data(), sizeof(TriangleGPU) * m_sceneData.triangles.size());
    }

    // Sphere buffer
    VkDeviceSize sphereSize = std::max(sizeof(SphereGPU) * m_sceneData.spheres.size(), sizeof(SphereGPU));
    m_sphereBuffer = std::make_unique<Buffer>(
        allocator, sphereSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.spheres.empty()) {
        m_sphereBuffer->copyFrom(m_sceneData.spheres.data(), sizeof(SphereGPU) * m_sceneData.spheres.size());
    }

    // Material buffer
    VkDeviceSize matSize = std::max(sizeof(MaterialGPU) * m_sceneData.materials.size(), sizeof(MaterialGPU));
    m_materialBuffer = std::make_unique<Buffer>(
        allocator, matSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.materials.empty()) {
        m_materialBuffer->copyFrom(m_sceneData.materials.data(), sizeof(MaterialGPU) * m_sceneData.materials.size());
    }

    // Light buffer
    VkDeviceSize lightSize = std::max(sizeof(LightGPU) * m_sceneData.lights.size(), sizeof(LightGPU));
    m_lightBuffer = std::make_unique<Buffer>(
        allocator, lightSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.lights.empty()) {
        m_lightBuffer->copyFrom(m_sceneData.lights.data(), sizeof(LightGPU) * m_sceneData.lights.size());
    }

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

    // Hardware Acceleration Structures (VK_KHR_ray_query)
    if (m_context->hasRayTracing()) {
        std::vector<Vertex> asVertices;
        if (!m_sceneData.triangles.empty()) {
            asVertices.reserve(m_sceneData.triangles.size() * 3);
            for (const auto& tri : m_sceneData.triangles) {
                asVertices.push_back(tri.v0);
                asVertices.push_back(tri.v1);
                asVertices.push_back(tri.v2);
            }
        } else {
            Vertex v{};
            asVertices.assign(3, v);
        }

        VkDeviceSize vertexBufferSize = sizeof(Vertex) * asVertices.size();
        m_asVertexBuffer = std::make_unique<Buffer>(
            allocator, vertexBufferSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        );
        m_asVertexBuffer->copyFrom(asVertices.data(), vertexBufferSize);

        m_asManager = std::make_unique<AccelerationStructureManager>(
            m_context->getDevice(), allocator,
            m_context->getGraphicsQueue(), m_context->getGraphicsQueueFamily()
        );

        ASGeometryInput geom{};
        geom.vertexBufferAddress = m_asVertexBuffer->getDeviceAddress(m_context->getDevice());
        geom.indexBufferAddress = 0;
        geom.vertexCount = static_cast<uint32_t>(asVertices.size());
        geom.triangleCount = static_cast<uint32_t>(asVertices.size() / 3);
        geom.vertexStride = sizeof(Vertex);
        geom.indexType = VK_INDEX_TYPE_NONE_KHR;
        bool hasAlphaMask = false;
        for (const auto& mat : m_sceneData.materials) {
            if (mat.alphaMode == ALPHA_MODE_MASK) {
                hasAlphaMask = true;
                break;
            }
        }
        geom.isOpaque = !hasAlphaMask;

        m_blas = m_asManager->buildBLAS({ geom });

        ASInstanceInput inst{};
        inst.blasAddress = m_blas->getDeviceAddress();
        inst.transform = glm::mat4(1.0f);
        inst.customIndex = 0;
        inst.mask = 0xFF;
        inst.hitGroupId = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

        m_tlas = m_asManager->buildTLAS({ inst });
        if (!m_tlas) {
            throw std::runtime_error("Hardware Ray Tracing TLAS build failed.");
        }
        Logger::info("Hardware Ray Tracing Acceleration Structures initialized successfully (BLAS & TLAS).");
        Logger::info("Hardware Ray Tracing Pipeline Active. Extensions in use: VK_KHR_ray_query, VK_KHR_acceleration_structure, VK_KHR_buffer_device_address, VK_KHR_deferred_host_operations (SPIR-V: GL_EXT_ray_query)");
    }

    // Textures & HDRI Environment Map Initialization
    VkDevice device = m_context->getDevice();
    VkQueue queue = m_context->getGraphicsQueue();
    VkCommandPool pool = m_commandPool;

    m_dummyWhite = Texture::createDummyWhite(device, allocator, queue, pool);
    m_dummyNormal = Texture::createDummyNormal(device, allocator, queue, pool);

    if (!m_config.hdri_path.empty() && std::filesystem::exists(m_config.hdri_path)) {
        m_environmentMap = Texture::loadFromFile(device, allocator, queue, pool, m_config.hdri_path);
    }
    if (!m_environmentMap) {
        m_environmentMap = Texture::createProceduralHdrSky(device, allocator, queue, pool);
        Logger::info("Generated physical procedural HDRI sky dome (512x256, 32-bit Float).");
    }

    m_sceneTextures.clear();
    for (const auto& texData : m_sceneData.textures) {
        if (!texData.pixels.empty() && texData.width > 0 && texData.height > 0) {
            VkFormat fmt = texData.isSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
            auto tex = Texture::createFromPixels(
                device, allocator, queue, pool,
                texData.width, texData.height,
                fmt, texData.pixels.data(),
                texData.pixels.size(), false
            );
            m_sceneTextures.push_back(std::move(tex));
        } else {
            m_sceneTextures.push_back(Texture::createDummyWhite(device, allocator, queue, pool));
        }
    }
    Logger::info("Scene textures loaded: {} texture(s).", m_sceneTextures.size());
}

bool Engine::loadScene(const std::string& filepath) {
    VkDevice device = m_context->getDevice();
    vkDeviceWaitIdle(device);
    if (m_mgpu && m_mgpu->getSecondaryContext()) {
        vkDeviceWaitIdle(m_mgpu->getSecondaryContext()->getDevice());
    }

    SceneData newScene;
    if (filepath.empty() || filepath == "__procedural_cornell_box__") {
        Logger::info("Dynamic Scene Switch: Loading Procedural Cornell Box...");
        newScene = ProceduralScene::createCornellBox();
    } else {
        Logger::info("Dynamic Scene Switch: Loading '{}'...", filepath);
        newScene = GltfLoader::loadSceneData(filepath);
    }
    if (newScene.triangles.empty() && newScene.spheres.empty()) {
        Logger::warn("Loaded scene '{}' contains no renderable geometry! Keeping current scene.", filepath);
        return false;
    }

    m_sceneData = std::move(newScene);
    m_numTriangles = static_cast<uint32_t>(m_sceneData.triangles.size());
    m_numSpheres = static_cast<uint32_t>(m_sceneData.spheres.size());
    m_numMaterials = static_cast<uint32_t>(m_sceneData.materials.size());
    m_numLights = static_cast<uint32_t>(m_sceneData.lights.size());
    updateSceneTransparencyFlag();

    Logger::info("Active Scene: {} Triangles, {} Spheres, {} Materials, {} Lights (Non-Opaque: {})",
                 m_numTriangles, m_numSpheres, m_numMaterials, m_numLights, m_sceneHasNonOpaque ? "YES" : "NO");

    // Recreate primary buffers
    VmaAllocator allocator = m_context->getAllocator();

    VkDeviceSize triSize = std::max(sizeof(TriangleGPU) * m_sceneData.triangles.size(), sizeof(TriangleGPU));
    m_triangleBuffer = std::make_unique<Buffer>(
        allocator, triSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.triangles.empty()) {
        m_triangleBuffer->copyFrom(m_sceneData.triangles.data(), sizeof(TriangleGPU) * m_sceneData.triangles.size());
    }

    VkDeviceSize sphereSize = std::max(sizeof(SphereGPU) * m_sceneData.spheres.size(), sizeof(SphereGPU));
    m_sphereBuffer = std::make_unique<Buffer>(
        allocator, sphereSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.spheres.empty()) {
        m_sphereBuffer->copyFrom(m_sceneData.spheres.data(), sizeof(SphereGPU) * m_sceneData.spheres.size());
    }

    VkDeviceSize matSize = std::max(sizeof(MaterialGPU) * m_sceneData.materials.size(), sizeof(MaterialGPU));
    m_materialBuffer = std::make_unique<Buffer>(
        allocator, matSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.materials.empty()) {
        m_materialBuffer->copyFrom(m_sceneData.materials.data(), sizeof(MaterialGPU) * m_sceneData.materials.size());
    }

    VkDeviceSize lightSize = std::max(sizeof(LightGPU) * m_sceneData.lights.size(), sizeof(LightGPU));
    m_lightBuffer = std::make_unique<Buffer>(
        allocator, lightSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.lights.empty()) {
        m_lightBuffer->copyFrom(m_sceneData.lights.data(), sizeof(LightGPU) * m_sceneData.lights.size());
    }

    // Rebuild Acceleration Structures (BLAS & TLAS)
    if (m_context->hasRayTracing()) {
        m_tlas.reset();
        m_blas.reset();
        m_asVertexBuffer.reset();
        m_asManager.reset();

        std::vector<Vertex> asVertices;
        if (!m_sceneData.triangles.empty()) {
            asVertices.reserve(m_sceneData.triangles.size() * 3);
            for (const auto& tri : m_sceneData.triangles) {
                asVertices.push_back(tri.v0);
                asVertices.push_back(tri.v1);
                asVertices.push_back(tri.v2);
            }
        } else {
            Vertex v{};
            asVertices.assign(3, v);
        }

        VkDeviceSize vertexBufferSize = sizeof(Vertex) * asVertices.size();
        m_asVertexBuffer = std::make_unique<Buffer>(
            allocator, vertexBufferSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        );
        m_asVertexBuffer->copyFrom(asVertices.data(), vertexBufferSize);

        m_asManager = std::make_unique<AccelerationStructureManager>(
            device, allocator,
            m_context->getGraphicsQueue(), m_context->getGraphicsQueueFamily()
        );

        ASGeometryInput geom{};
        geom.vertexBufferAddress = m_asVertexBuffer->getDeviceAddress(device);
        geom.indexBufferAddress = 0;
        geom.vertexCount = static_cast<uint32_t>(asVertices.size());
        geom.triangleCount = static_cast<uint32_t>(asVertices.size() / 3);
        geom.vertexStride = sizeof(Vertex);
        geom.indexType = VK_INDEX_TYPE_NONE_KHR;
        bool hasAlphaMask = false;
        for (const auto& mat : m_sceneData.materials) {
            if (mat.alphaMode == ALPHA_MODE_MASK) {
                hasAlphaMask = true;
                break;
            }
        }
        geom.isOpaque = !hasAlphaMask;

        m_blas = m_asManager->buildBLAS({ geom });

        ASInstanceInput inst{};
        inst.blasAddress = m_blas->getDeviceAddress();
        inst.transform = glm::mat4(1.0f);
        inst.customIndex = 0;
        inst.mask = 0xFF;
        inst.hitGroupId = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

        m_tlas = m_asManager->buildTLAS({ inst });
        if (!m_tlas) {
            throw std::runtime_error("Hardware Ray Tracing TLAS rebuild failed.");
        }
    }

    // Reload scene textures
    VkQueue queue = m_context->getGraphicsQueue();
    VkCommandPool pool = m_commandPool;

    m_sceneTextures.clear();
    for (const auto& texData : m_sceneData.textures) {
        if (!texData.pixels.empty() && texData.width > 0 && texData.height > 0) {
            VkFormat fmt = texData.isSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
            auto tex = Texture::createFromPixels(
                device, allocator, queue, pool,
                texData.width, texData.height,
                fmt, texData.pixels.data(),
                texData.pixels.size(), false
            );
            m_sceneTextures.push_back(std::move(tex));
        } else {
            m_sceneTextures.push_back(Texture::createDummyWhite(device, allocator, queue, pool));
        }
    }

    // Update primary descriptor sets
    updateSceneDescriptors();

    // Secondary GPU reload
    if (m_mgpu && m_mgpu->isSecondaryInitialized()) {
        m_mgpu->loadScene(m_sceneData);
    }

    // Update camera framing
    if (m_camera) {
        m_camera->setSceneScale(m_sceneData.sceneRadius, m_sceneData.focalDistance, m_sceneData.centralTarget);
        m_camera->lookAt(m_sceneData.cameraPosition, m_sceneData.cameraTarget, m_sceneData.cameraUp);
        m_camera->setFov(m_sceneData.cameraFov);
        m_camera->setDefaultFraming(m_sceneData.cameraPosition, m_sceneData.cameraTarget, m_sceneData.cameraFov);
    }

    // Update scene path and current index
    m_config.scene_path = filepath;
    m_currentSceneIndex = 0;
    if (!filepath.empty()) {
        for (size_t i = 0; i < m_availableScenes.size(); ++i) {
            std::error_code ec;
            if (m_availableScenes[i].filepath == filepath ||
                (!m_availableScenes[i].filepath.empty() &&
                 std::filesystem::exists(m_availableScenes[i].filepath) &&
                 std::filesystem::exists(filepath) &&
                 std::filesystem::equivalent(m_availableScenes[i].filepath, filepath, ec))) {
                m_currentSceneIndex = static_cast<int>(i);
                break;
            }
        }
    }

    // Reset frame accumulation
    m_frameIndex = 0;
    m_resetAccumulation = true;
    m_frameTimesMs.clear();

    Logger::info("Scene successfully switched to: {} (Index: {})", filepath, m_currentSceneIndex);
    return true;
}

void Engine::updateSceneTransparencyFlag() {
    m_sceneHasNonOpaque = false;
    for (const auto& mat : m_sceneData.materials) {
        if (mat.alphaMode != 0 || mat.transmission > 0.05f || mat.type == 2) {
            m_sceneHasNonOpaque = true;
            break;
        }
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

void Engine::initPipelines() {
    VkDevice device = m_context->getDevice();

    // 0. Initialize ReSTIR DI Buffers
    initReSTIRBuffers();

    // 1. Descriptor Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 128 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 32 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 64;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool);

    // 2. Ray Tracing Descriptor Set Layout (VK_KHR_ray_tracing_pipeline)
    VkShaderStageFlags rtStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

    std::vector<VkDescriptorSetLayoutBinding> rtBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, rtStages, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, rtStages, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, rtStages, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_SCENE_TEXTURES, rtStages, nullptr },
        { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo rtLayoutInfo{};
    rtLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    rtLayoutInfo.bindingCount = static_cast<uint32_t>(rtBindings.size());
    rtLayoutInfo.pBindings = rtBindings.data();
    vkCreateDescriptorSetLayout(device, &rtLayoutInfo, nullptr, &m_rtDescLayout);

    // 3. Tonemap Descriptor Set Layout
    std::vector<VkDescriptorSetLayoutBinding> tonemapBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo tonemapLayoutInfo{};
    tonemapLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    tonemapLayoutInfo.bindingCount = static_cast<uint32_t>(tonemapBindings.size());
    tonemapLayoutInfo.pBindings = tonemapBindings.data();
    vkCreateDescriptorSetLayout(device, &tonemapLayoutInfo, nullptr, &m_tonemapDescLayout);

    // 4. Allocate Descriptor Sets
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        VkDescriptorSetAllocateInfo rtAllocInfo{};
        rtAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        rtAllocInfo.descriptorPool = m_descriptorPool;
        rtAllocInfo.descriptorSetCount = 1;
        rtAllocInfo.pSetLayouts = &m_rtDescLayout;
        vkAllocateDescriptorSets(device, &rtAllocInfo, &m_rtDescSets[i]);
    }

    VkDescriptorSetAllocateInfo tonemapAllocInfo{};
    tonemapAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    tonemapAllocInfo.descriptorPool = m_descriptorPool;
    tonemapAllocInfo.descriptorSetCount = 1;
    tonemapAllocInfo.pSetLayouts = &m_tonemapDescLayout;
    vkAllocateDescriptorSets(device, &tonemapAllocInfo, &m_tonemapDescSet);

    // 5. Update Descriptor Sets
    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageView = m_accumImage->getImageView();
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageView = m_outputImage->getImageView();
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::vector<VkDescriptorBufferInfo> uboBufferInfos(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        uboBufferInfos[i] = { m_cameraUBOs[i]->getBuffer(), 0, sizeof(CameraUniform) };
    }

    std::vector<VkWriteDescriptorSet> writes;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboBufferInfos[i], nullptr });
    }
    // Tonemap set
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_tonemapDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr });
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_tonemapDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputImageInfo, nullptr, nullptr });
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    updateSceneDescriptors();

    // 6. Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline)
    VkPushConstantRange rtpPushConstant{};
    rtpPushConstant.stageFlags = rtStages;
    rtpPushConstant.offset = 0;
    rtpPushConstant.size = sizeof(uint32_t) * 12;

    VkPipelineLayoutCreateInfo rtpPipeLayoutInfo{};
    rtpPipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    rtpPipeLayoutInfo.setLayoutCount = 1;
    rtpPipeLayoutInfo.pSetLayouts = &m_rtDescLayout;
    rtpPipeLayoutInfo.pushConstantRangeCount = 1;
    rtpPipeLayoutInfo.pPushConstantRanges = &rtpPushConstant;
    vkCreatePipelineLayout(device, &rtpPipeLayoutInfo, nullptr, &m_rtpPipelineLayout);

    VmaAllocator allocator = m_context->getAllocator();

    auto rgenCode = loadShaderSPIRV("raytrace.rgen.spv");
    auto rmissCode = loadShaderSPIRV("raytrace.rmiss.spv");
    auto shadowMissCode = loadShaderSPIRV("shadow.rmiss.spv");
    auto rchitCode = loadShaderSPIRV("raytrace.rchit.spv");

    m_rtpKhrPipeline = std::make_unique<RTPipeline>(
        device, allocator,
        m_context->getRayTracingPipelineProperties(),
        m_rtpPipelineLayout,
        rgenCode, rmissCode, shadowMissCode, rchitCode
    );
    Logger::info("Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline) created successfully.");

    // 7. ACES Tonemapping Compute Pipeline (Wave32 execution mode on RDNA4)
    VkPushConstantRange tonemapPushConstant{};
    tonemapPushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tonemapPushConstant.offset = 0;
    tonemapPushConstant.size = sizeof(float) + sizeof(uint32_t) * 7; // exposure, totalSamples, applyACES, visualizeSplit, splitY, padding[3] (32 bytes)

    VkPipelineLayoutCreateInfo tonemapPipeLayoutInfo{};
    tonemapPipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    tonemapPipeLayoutInfo.setLayoutCount = 1;
    tonemapPipeLayoutInfo.pSetLayouts = &m_tonemapDescLayout;
    tonemapPipeLayoutInfo.pushConstantRangeCount = 1;
    tonemapPipeLayoutInfo.pPushConstantRanges = &tonemapPushConstant;
    vkCreatePipelineLayout(device, &tonemapPipeLayoutInfo, nullptr, &m_tonemapPipelineLayout);

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    auto tonemapCode = loadShaderSPIRV("tonemap_aces.comp.spv");
    VkShaderModule tonemapModule = createShaderModule(tonemapCode);

    VkComputePipelineCreateInfo tonemapPipelineInfo{};
    tonemapPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    tonemapPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    tonemapPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    tonemapPipelineInfo.stage.module = tonemapModule;
    tonemapPipelineInfo.stage.pName = "main";
    if (m_context->hasSubgroupSizeControl()) {
        tonemapPipelineInfo.stage.pNext = &subgroupSize32;
    }
    tonemapPipelineInfo.layout = m_tonemapPipelineLayout;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &tonemapPipelineInfo, nullptr, &m_tonemapPipeline);
    vkDestroyShaderModule(device, tonemapModule, nullptr);

    Logger::info("ACES Tonemapping Pipeline (Wave32) created successfully.");

    // Multi-GPU Merge Pipeline & Descriptors (Created unconditionally so mode switches never fault)
    std::vector<VkDescriptorSetLayoutBinding> mergeBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo mergeLayoutInfo{};
    mergeLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    mergeLayoutInfo.bindingCount = static_cast<uint32_t>(mergeBindings.size());
    mergeLayoutInfo.pBindings = mergeBindings.data();
    vkCreateDescriptorSetLayout(device, &mergeLayoutInfo, nullptr, &m_mergeDescLayout);

    std::array<VkDescriptorSetLayout, 2> mergeLayouts = { m_mergeDescLayout, m_mergeDescLayout };
    VkDescriptorSetAllocateInfo mergeAllocInfo{};
    mergeAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    mergeAllocInfo.descriptorPool = m_descriptorPool;
    mergeAllocInfo.descriptorSetCount = 2;
    mergeAllocInfo.pSetLayouts = mergeLayouts.data();
    vkAllocateDescriptorSets(device, &mergeAllocInfo, m_mergeDescSets.data());

    updateMergeDescriptors();

    VkPushConstantRange mergePushConstant{};
    mergePushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    mergePushConstant.offset = 0;
    mergePushConstant.size = sizeof(uint32_t) * 6; // width, height, spp, mode, formatMode, dynamicRatioPermille

    VkPipelineLayoutCreateInfo mergePipeLayoutInfo{};
    mergePipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    mergePipeLayoutInfo.setLayoutCount = 1;
    mergePipeLayoutInfo.pSetLayouts = &m_mergeDescLayout;
    mergePipeLayoutInfo.pushConstantRangeCount = 1;
    mergePipeLayoutInfo.pPushConstantRanges = &mergePushConstant;
    vkCreatePipelineLayout(device, &mergePipeLayoutInfo, nullptr, &m_mergePipelineLayout);

    auto mergeCode = loadShaderSPIRV("accum_merge.comp.spv");
    VkShaderModule mergeModule = createShaderModule(mergeCode);

    VkComputePipelineCreateInfo mergePipelineInfo{};
    mergePipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    mergePipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    mergePipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    mergePipelineInfo.stage.module = mergeModule;
    mergePipelineInfo.stage.pName = "main";
    if (m_context->hasSubgroupSizeControl()) {
        mergePipelineInfo.stage.pNext = &subgroupSize32;
    }
    mergePipelineInfo.layout = m_mergePipelineLayout;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &mergePipelineInfo, nullptr, &m_mergePipeline);
    vkDestroyShaderModule(device, mergeModule, nullptr);

    Logger::info("Multi-GPU merge compute pipeline (Wave32) created successfully.");
}

void Engine::updateAllImageDescriptors() {
    VkDevice device = m_context->getDevice();
    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageView = m_accumImage->getImageView();
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageView = m_outputImage->getImageView();
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::vector<VkWriteDescriptorSet> writes;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_rtDescSets[i] != VK_NULL_HANDLE) {
            VkWriteDescriptorSet w0{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w0.dstSet = m_rtDescSets[i];
            w0.dstBinding = 0;
            w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w0.descriptorCount = 1;
            w0.pImageInfo = &accumImageInfo;
            writes.push_back(w0);
        }
    }

    if (m_tonemapDescSet != VK_NULL_HANDLE) {
        VkWriteDescriptorSet w1{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w1.dstSet = m_tonemapDescSet;
        w1.dstBinding = 0;
        w1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1.descriptorCount = 1;
        w1.pImageInfo = &accumImageInfo;
        writes.push_back(w1);

        VkWriteDescriptorSet w2{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w2.dstSet = m_tonemapDescSet;
        w2.dstBinding = 1;
        w2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w2.descriptorCount = 1;
        w2.pImageInfo = &outputImageInfo;
        writes.push_back(w2);
    }


    if (!writes.empty()) {
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void Engine::updateSceneDescriptors() {
    VkDevice device = m_context->getDevice();

    VkDescriptorBufferInfo triBufferInfo{ m_triangleBuffer->getBuffer(), 0, m_triangleBuffer->getSize() };
    VkDescriptorBufferInfo sphereBufferInfo{ m_sphereBuffer->getBuffer(), 0, m_sphereBuffer->getSize() };
    VkDescriptorBufferInfo matBufferInfo{ m_materialBuffer->getBuffer(), 0, m_materialBuffer->getSize() };
    VkDescriptorBufferInfo lightBufferInfo{ m_lightBuffer->getBuffer(), 0, m_lightBuffer->getSize() };

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    VkAccelerationStructureKHR tlasHandle = m_tlas ? m_tlas->getHandle() : VK_NULL_HANDLE;
    asInfo.pAccelerationStructures = &tlasHandle;

    VkDescriptorImageInfo envInfo = m_environmentMap ? m_environmentMap->getDescriptorInfo() : m_dummyWhite->getDescriptorInfo();

    std::vector<VkDescriptorImageInfo> texInfos(MAX_SCENE_TEXTURES);
    for (size_t i = 0; i < MAX_SCENE_TEXTURES; ++i) {
        if (i < m_sceneTextures.size() && m_sceneTextures[i]) {
            texInfos[i] = m_sceneTextures[i]->getDescriptorInfo();
        } else {
            texInfos[i] = m_dummyWhite->getDescriptorInfo();
        }
    }

    VkDescriptorBufferInfo res0Info{};
    VkDescriptorBufferInfo res1Info{};
    if (m_restirReservoirs[0] && m_restirReservoirs[1]) {
        res0Info = { m_restirReservoirs[0]->getBuffer(), 0, m_restirReservoirs[0]->getSize() };
        res1Info = { m_restirReservoirs[1]->getBuffer(), 0, m_restirReservoirs[1]->getSize() };
    }

    std::vector<VkWriteDescriptorSet> writes;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_rtDescSets[i] == VK_NULL_HANDLE) continue;
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sphereBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, m_rtDescSets[i], 6, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 8, 0, MAX_SCENE_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texInfos.data(), nullptr, nullptr });
        if (m_restirReservoirs[0] && m_restirReservoirs[1]) {
            writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &res0Info, nullptr });
            writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &res1Info, nullptr });
        }
    }
    if (!writes.empty()) {
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void Engine::initReSTIRBuffers() {
    VkDeviceSize resSize = static_cast<VkDeviceSize>(m_config.width) * m_config.height * sizeof(ReservoirGPU);
    VmaAllocator allocator = m_context->getAllocator();

    for (uint32_t i = 0; i < 2; ++i) {
        m_restirReservoirs[i] = std::make_unique<Buffer>(
            allocator, resSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );
    }
    m_restirPingPongIndex = 0;

    // Clear reservoir buffers to 0 using a one-time command
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_commandBuffers[0], &beginInfo);
    vkCmdFillBuffer(m_commandBuffers[0], m_restirReservoirs[0]->getBuffer(), 0, resSize, 0);
    vkCmdFillBuffer(m_commandBuffers[0], m_restirReservoirs[1]->getBuffer(), 0, resSize, 0);
    vkEndCommandBuffer(m_commandBuffers[0]);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffers[0];
    vkQueueSubmit(m_context->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_context->getGraphicsQueue());
}

void Engine::updateReSTIRDescriptors(uint32_t frameSlot) {
    if (m_rtDescSets[frameSlot] == VK_NULL_HANDLE || !m_restirReservoirs[0] || !m_restirReservoirs[1]) {
        return;
    }
    VkDevice device = m_context->getDevice();
    VkDeviceSize resSize = static_cast<VkDeviceSize>(m_config.width) * m_config.height * sizeof(ReservoirGPU);

    VkDescriptorBufferInfo curInfo{ m_restirReservoirs[m_restirPingPongIndex]->getBuffer(), 0, resSize };
    VkDescriptorBufferInfo histInfo{ m_restirReservoirs[1 - m_restirPingPongIndex]->getBuffer(), 0, resSize };

    std::vector<VkWriteDescriptorSet> writes = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[frameSlot], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &curInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[frameSlot], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &histInfo, nullptr }
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void Engine::updateMergeDescriptors() {
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();

    uint32_t bytesPerPixel = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? (4 * sizeof(uint16_t)) : (4 * sizeof(float));
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(m_config.width) * m_config.height * bytesPerPixel;

    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageView = m_accumImage->getImageView();
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    for (uint32_t slot = 0; slot < 2; ++slot) {
        if (m_mergeDescSets[slot] == VK_NULL_HANDLE) continue;

        VkBuffer secBuffer = VK_NULL_HANDLE;
        VkDeviceSize curSize = bufferSize;

        if (m_mgpu && m_mgpu->isZeroCopyActive()) {
            secBuffer = m_mgpu->getPrimarySharedBuffer(slot);
            curSize = m_mgpu->getSharedBufferSize();
        } else {
            if (!m_secTransferBuffer || m_secTransferBuffer->getSize() < bufferSize) {
                m_secTransferBuffer = std::make_unique<Buffer>(
                    allocator, bufferSize,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VMA_MEMORY_USAGE_AUTO,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
                );
            }
            secBuffer = m_secTransferBuffer->getBuffer();
            curSize = m_secTransferBuffer->getSize();
        }

        VkDescriptorBufferInfo secBufInfo{ secBuffer, 0, curSize };

        std::vector<VkWriteDescriptorSet> mergeWrites = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secBufInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secBufInfo, nullptr }
        };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(mergeWrites.size()), mergeWrites.data(), 0, nullptr);
    }
}



void Engine::initSyncObjects() {
    VkDevice device = m_context->getDevice();

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkCreateFence(device, &fenceInfo, nullptr, &m_inFlightFences[i]);
    }
    vkCreateFence(device, &fenceInfo, nullptr, &m_rtFence);

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    m_imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkCreateSemaphore(device, &semInfo, nullptr, &m_imageAvailableSemaphores[i]);
    }

    uint32_t numSwapImages = m_swapchain ? m_swapchain->getImageCount() : MAX_FRAMES_IN_FLIGHT;
    m_renderFinishedSemaphores.resize(numSwapImages);
    for (uint32_t i = 0; i < numSwapImages; ++i) {
        vkCreateSemaphore(device, &semInfo, nullptr, &m_renderFinishedSemaphores[i]);
    }
}

void Engine::initQueryPool() {
    VkDevice device = m_context->getDevice();
    VkQueryPoolCreateInfo queryInfo{};
    queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = 4 * MAX_FRAMES_IN_FLIGHT; // 4 timestamps per frame in flight
    vkCreateQueryPool(device, &queryInfo, nullptr, &m_queryPool);

    m_timestampPeriod = m_context->getDeviceProperties().limits.timestampPeriod;
    Logger::info("GPU Timestamp Profiler initialized (period: {:.2f} ns/tick)", m_timestampPeriod);
}

void Engine::setCameraMode(bool active) {
    if (m_cameraMode == active) return;
    m_cameraMode = active;
    if (m_window) {
        m_window->setRelativeMouseMode(m_cameraMode);
    }
    Logger::info("Interaction Mode: {}", m_cameraMode ? "FPS Scene Navigation (Mouse grabbed, WASD active)" : "UI Control Panel (Mouse released)");
}

bool Engine::handleEvent(const SDL_Event& e) {
    if (e.type == SDL_EVENT_QUIT) {
        return false; // Allow default Window handling to set shouldClose
    }

    // 1. F11 key: toggle fullscreen (intercepted first so ImGui never swallows F11)
    if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F11 && !e.key.repeat) {
        Logger::info("Received SDLK_F11 -> queueing fullscreen toggle.");
        m_pendingToggleFullscreen = true;
        return true;
    }
    if (e.type == SDL_EVENT_KEY_UP && e.key.key == SDLK_F11) {
        return true;
    }

    // 2. Window-level events: allow ImGui to track focus/cursor, then forward to Window handler
    if (e.type >= SDL_EVENT_WINDOW_FIRST && e.type <= SDL_EVENT_WINDOW_LAST) {
        if (e.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
            if (m_cameraMode) {
                setCameraMode(false);
            }
        }
        if (m_gui) {
            m_gui->processEvent(e);
        }
        return false;
    }

    // 3. TAB key toggles between UI control panel and FPS scene navigation
    if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_TAB) {
        setCameraMode(!m_cameraMode);
        return true; // Consume event so ImGui navigation does not swallow TAB
    }

    // 4. Release mouse if window focus is lost while in camera mode
    if (e.type == SDL_EVENT_WINDOW_FOCUS_LOST) {
        if (m_cameraMode) {
            setCameraMode(false);
        }
        return false;
    }

    // 5. ESC key: if in camera mode, return to UI mode; if in UI mode, close window
    if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) {
        if (m_cameraMode) {
            setCameraMode(false);
            return true;
        }
        return false; // Let Window close application
    }

    // 4. In Camera Mode (FPS navigation)
    if (m_cameraMode) {
        if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_F) {
            if (m_camera) {
                m_camera->focusOnTarget(m_sceneData.centralTarget, m_sceneData.focalRadius);
            }
            return true;
        }
        if (e.type == SDL_EVENT_MOUSE_MOTION) {
            if (m_camera) {
                const bool* keyState = SDL_GetKeyboardState(nullptr);
                bool ctrl = keyState && (keyState[SDL_SCANCODE_LCTRL] || keyState[SDL_SCANCODE_RCTRL]);
                m_camera->processMouseMovement(e.motion.xrel, e.motion.yrel, ctrl);
            }
            return true;
        }
        if (e.type == SDL_EVENT_MOUSE_WHEEL) {
            if (m_camera) {
                m_camera->adjustSpeedByWheel(e.wheel.y);
            }
            return true;
        }
        // Consume mouse clicks in FPS mode so they don't hit underlying ImGui elements
        if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN || e.type == SDL_EVENT_MOUSE_BUTTON_UP) {
            return true;
        }
        return false;
    }

    // 5. In UI Mode: route events to ImGui
    if (m_gui) {
        // If mouse wheel happened outside ImGui windows, adjust camera speed
        if (e.type == SDL_EVENT_MOUSE_WHEEL && !m_gui->wantCaptureMouse()) {
            if (m_camera) {
                m_camera->adjustSpeedByWheel(e.wheel.y);
            }
            return true;
        }

        bool handled = m_gui->processEvent(e);
        if (handled) {
            return true;
        }
        // If left click happened outside ImGui windows, capture mouse for FPS navigation
        if (e.type == SDL_EVENT_MOUSE_BUTTON_DOWN && e.button.button == SDL_BUTTON_LEFT) {
            if (!m_gui->wantCaptureMouse()) {
                setCameraMode(true);
                return true;
            }
        }
    }

    return false;
}

void Engine::updateInput() {
    auto now = std::chrono::high_resolution_clock::now();
    float dt = std::chrono::duration<float>(now - m_lastFrameTime).count();
    m_lastFrameTime = now;
    if (dt <= 0.0f || dt > 0.1f) dt = 0.016f;

    if (!m_cameraMode || m_config.headless || !m_camera) {
        return;
    }

    const bool* keyState = SDL_GetKeyboardState(nullptr);
    if (!keyState) return;

    float forward = 0.0f;
    float strafe = 0.0f;
    float vertical = 0.0f;

    if (keyState[SDL_SCANCODE_W]) forward += 1.0f;
    if (keyState[SDL_SCANCODE_S]) forward -= 1.0f;
    if (keyState[SDL_SCANCODE_D]) strafe += 1.0f;
    if (keyState[SDL_SCANCODE_A]) strafe -= 1.0f;
    if (keyState[SDL_SCANCODE_SPACE] || keyState[SDL_SCANCODE_E]) vertical += 1.0f;
    if (keyState[SDL_SCANCODE_C] || keyState[SDL_SCANCODE_Q]) vertical -= 1.0f;

    bool sprint = keyState[SDL_SCANCODE_LSHIFT] || keyState[SDL_SCANCODE_RSHIFT];
    bool crawl = keyState[SDL_SCANCODE_LALT] || keyState[SDL_SCANCODE_RALT];
    bool ctrl = keyState[SDL_SCANCODE_LCTRL] || keyState[SDL_SCANCODE_RCTRL];

    if (ctrl) {
        if (!m_camera->isOrbiting()) {
            // Find target object along view ray
            float hitDist = 0.0f;
            glm::vec3 hitPoint;
            std::string hitName;
            glm::vec3 camPos = m_camera->getPosition();
            glm::vec3 camFront = m_camera->getFront();
            if (m_sceneData.raycast(camPos, camFront, 5000.0f, hitDist, hitPoint, &hitName)) {
                m_camera->startOrbit(hitPoint);
            } else {
                // Fallback: project centralTarget or use focal distance along view ray
                glm::vec3 toCenter = m_sceneData.centralTarget - camPos;
                float proj = glm::dot(toCenter, camFront);
                float dist = (proj > 0.1f) ? proj : m_camera->getFocalDistance();
                m_camera->startOrbit(camPos + camFront * dist);
            }
        }
    } else {
        if (m_camera->isOrbiting()) {
            m_camera->endOrbit();
        }
    }

    m_camera->processFpsInput(forward, strafe, vertical, dt, sprint, crawl, ctrl);
}

void Engine::renderFrame() {
    VkDevice device = m_context->getDevice();
    VkQueue queue = m_context->getGraphicsQueue();

    vkWaitForFences(device, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &m_inFlightFences[m_currentFrame]);

    // Read back GPU query timestamps from slot m_currentFrame's completed frame
    if (m_totalFramesRendered >= MAX_FRAMES_IN_FLIGHT) {
        uint32_t qBase = m_currentFrame * 4;
        uint64_t timestamps[4] = {0, 0, 0, 0};
        vkGetQueryPoolResults(device, m_queryPool, qBase, 4, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        double gpuRtMs = 0.0;
        if (timestamps[1] > timestamps[0]) {
            gpuRtMs = (timestamps[1] - timestamps[0]) * m_timestampPeriod * 1e-6;
        }
        double gpuTonemapMs = 0.0;
        if (timestamps[3] > timestamps[2]) {
            gpuTonemapMs = (timestamps[3] - timestamps[2]) * m_timestampPeriod * 1e-6;
        }
        if (gpuRtMs > 10000.0) gpuRtMs = 0.0;
        if (gpuTonemapMs > 10000.0) gpuTonemapMs = 0.0;

        double secGpuMs = 0.0;
        if (m_mgpu && m_mgpu->isMultiGpuActive()) {
            secGpuMs = m_mgpu->getSecondaryGpuTimeMs();
        }
        double totalGpuMs = (m_mgpu && m_mgpu->isMultiGpuActive()) ? (std::max(gpuRtMs, secGpuMs) + gpuTonemapMs) : (gpuRtMs + gpuTonemapMs);

        m_lastGpuRtMs = gpuRtMs;
        m_lastSecGpuMs = secGpuMs;
        m_lastTonemapMs = gpuTonemapMs;
        if (totalGpuMs > 0.01) {
            m_lastFrameTimeMs = totalGpuMs;
            m_frameTimesMs.push_back(m_lastFrameTimeMs);
            if (!m_config.headless && m_frameTimesMs.size() > 60) {
                m_frameTimesMs.erase(m_frameTimesMs.begin());
            }
            recordFrameTally(totalGpuMs, gpuRtMs, secGpuMs, gpuTonemapMs);
        }
    }

    // Process deferred UI reconfiguration actions safely at frame boundary (before recording)
    if (m_pendingSceneChange || m_pendingMgpuModeChange || m_pendingAccumFormatChange || m_pendingDoubleBufferChange) {
        VkDevice dev = m_context->getDevice();
        VmaAllocator alloc = m_context->getAllocator();
        vkDeviceWaitIdle(dev);
        if (m_mgpu && m_mgpu->getSecondaryContext()) {
            vkDeviceWaitIdle(m_mgpu->getSecondaryContext()->getDevice());
        }

        if (m_pendingSceneChange) {
            loadScene(m_pendingScenePath);
            m_pendingSceneChange = false;
        }

        if (m_pendingAccumFormatChange) {
            m_config.accum_format = m_newAccumFormat;
            m_pendingAccumFormatChange = false;

            VkFormat accumFmt = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
            m_accumImage = std::make_unique<Image>(
                dev, alloc, m_config.width, m_config.height,
                accumFmt,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
            );

            // Transition m_accumImage to GENERAL
            vkResetCommandBuffer(m_commandBuffers[0], 0);
            VkCommandBufferBeginInfo transBegin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            transBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(m_commandBuffers[0], &transBegin);
            m_accumImage->transitionLayout(
                m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
            );
            vkEndCommandBuffer(m_commandBuffers[0]);
            VkSubmitInfo transSubmit{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            transSubmit.commandBufferCount = 1;
            transSubmit.pCommandBuffers = &m_commandBuffers[0];
            vkQueueSubmit(m_context->getGraphicsQueue(), 1, &transSubmit, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_context->getGraphicsQueue());

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
            } else if (m_mgpu) {
                m_mgpu->setMode(MultiGpuMode::Off);
            }
            m_resetAccumulation = true;
        }

        if (m_pendingDoubleBufferChange) {
            m_config.double_buffered_shared_mem = m_newDoubleBuffer;
            m_pendingDoubleBufferChange = false;
        }

        if (m_pendingTileSizeChange) {
            m_config.tile_size = m_newTileSize;
            m_pendingTileSizeChange = false;
        }

        updateMergeDescriptors();
        m_resetAccumulation = true;
        m_frameTimesMs.clear();
    }

    // Update smooth continuous FPS keyboard navigation
    updateInput();

    if (m_config.camera_motion && m_camera) {
        m_camera->processMouseMovement(2.0f, 0.0f);
    }

    // Reset accumulation if camera moved or UI settings changed
    if (m_camera->hasMoved() || m_resetAccumulation) {
        m_frameIndex = 0;
        m_camera->resetMoved();
        m_resetAccumulation = false;
    }

    // Update Camera Uniform
    uint32_t flags = 0;
    if (m_config.enable_direct_light)   flags |= (1 << 0);
    if (m_config.enable_indirect_light) flags |= (1 << 1);
    flags |= (1 << 2); // Specular
    if (m_config.enable_refraction)     flags |= (1 << 3);
    if (m_config.enable_shadows)        flags |= (1 << 4);
    if (m_sceneHasNonOpaque)            flags |= (1 << 5);
    if (m_config.enable_restir_di)      flags |= (1 << 6);

    CameraUniform ubo = m_camera->getUniformData(m_frameIndex, m_config.spp, m_config.max_bounces, flags);
    m_cameraUBOs[m_currentFrame]->copyFrom(&ubo, sizeof(CameraUniform));

    uint32_t imageIndex = 0;
    if (!m_config.headless && m_swapchain) {
        VkResult res = m_swapchain->acquireNextImage(m_imageAvailableSemaphores[m_currentFrame], &imageIndex);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR ||
            m_config.width != m_swapchain->getExtent().width ||
            m_config.height != m_swapchain->getExtent().height) {
            int curW = 0, curH = 0;
            SDL_GetWindowSizeInPixels(m_window->getSDLWindow(), &curW, &curH);
            uint32_t targetW = (curW > 0) ? static_cast<uint32_t>(curW) : m_window->getWidth();
            uint32_t targetH = (curH > 0) ? static_cast<uint32_t>(curH) : m_window->getHeight();
            onResize(targetW, targetH, /*forceRecreate=*/true);
            return;
        }
    }

    // Ping-pong ReSTIR DI reservoir buffers if enabled
    if (m_config.enable_restir_di) {
        m_restirPingPongIndex = 1 - m_restirPingPongIndex;
    }
    updateReSTIRDescriptors(m_currentFrame);

    uint32_t groupsX = (m_config.width + 15) / 16;
    uint32_t groupsY = (m_config.height + 15) / 16;
    uint32_t rtGroupsX = (m_config.width + 7) / 8;
    uint32_t rtGroupsY = (m_config.height + 3) / 4;

    struct TonemapPushConstants {
        float exposure = 1.0f;
        uint32_t totalSamples = 1;
        uint32_t applyACES = 1;
        uint32_t visualizeSplit = 0;
        uint32_t tileSize = 64;
        uint32_t padding[3] = {0, 0, 0};
    } tonemapConstants;
    tonemapConstants.exposure = 1.0f;
    tonemapConstants.applyACES = m_config.aces_tonemap ? 1 : 0;
    tonemapConstants.visualizeSplit = 0;
    tonemapConstants.tileSize = m_config.tile_size;

    bool isMgpu = (m_mgpu && m_mgpu->isMultiGpuActive() && m_config.mgpu_mode != MultiGpuMode::Off);

    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];

    if (!isMgpu) {
        // --- Single GPU Execution Path ---
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &beginInfo);

        // Frame-start pipeline barrier to synchronize compute and transfer writes across frames
        VkMemoryBarrier2 frameStartBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        frameStartBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT;
        frameStartBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        frameStartBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        frameStartBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;

        VkDependencyInfo frameStartDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        frameStartDep.memoryBarrierCount = 1;
        frameStartDep.pMemoryBarriers = &frameStartBarrier;
        vkCmdPipelineBarrier2(cmd, &frameStartDep);

        uint32_t qBase = m_currentFrame * 4;
        vkCmdResetQueryPool(cmd, m_queryPool, qBase, 4);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, qBase + 0);

        uint32_t useHwRT = 1;
        uint32_t hasEnvMap = m_environmentMap ? 1 : 0;
        float envIntensity = 1.0f;
        uint32_t envIntensityBits = std::bit_cast<uint32_t>(envIntensity);

        // === DEDICATED HARDWARE RAY TRACING PIPELINE (VK_KHR_ray_tracing_pipeline) ===
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtpKhrPipeline->getPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtpPipelineLayout, 0, 1, &m_rtDescSets[m_currentFrame], 0, nullptr);

        uint32_t rtPushConstants[12] = {
            m_numTriangles, m_numSpheres, m_numMaterials, m_numLights,
            0, 0, m_config.width, m_config.height,
            useHwRT,
            hasEnvMap,
            envIntensityBits,
            1u // accumulateHistory
        };
        VkShaderStageFlags rtpStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        vkCmdPushConstants(cmd, m_rtpPipelineLayout, rtpStages, 0, sizeof(rtPushConstants), rtPushConstants);

        m_rtpKhrPipeline->traceRays(cmd, m_config.width, m_config.height, 1);

        VkMemoryBarrier2 memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);

        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 1);

        // Tonemapping
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &m_tonemapDescSet, 0, nullptr);

        tonemapConstants.visualizeSplit = 0;
        tonemapConstants.totalSamples = (m_frameIndex + 1) * m_config.spp;
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, qBase + 2);
        vkCmdPushConstants(cmd, m_tonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tonemapConstants), &tonemapConstants);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 3);
    } else {
        // --- Multi-GPU Path (Checkerboard Tiling or Sample Parallelism) ---
        uint32_t useHwRT = 1;
        uint32_t hasEnvMap = m_environmentMap ? 1 : 0;
        float envIntensity = 1.0f;
        uint32_t envIntensityBits = std::bit_cast<uint32_t>(envIntensity);

        uint32_t formatMode = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? 0u : 1u;
        uint32_t bytesPerPixel = (formatMode == 0u) ? 8 : 16;
        size_t frameBytes = 0;
        void* dstHost = nullptr;
        if (!m_mgpu->isZeroCopyActive()) {
            frameBytes = static_cast<size_t>(m_config.width) * m_config.height * bytesPerPixel;
            dstHost = m_secTransferBuffer ? m_secTransferBuffer->map() : nullptr;
        }

        MultiGpuMode activeMode = m_config.mgpu_mode;
        if (activeMode == MultiGpuMode::Auto) {
            activeMode = (m_config.spp > 1) ? MultiGpuMode::SampleParallel : MultiGpuMode::CheckerboardTile;
        }

        uint32_t tileOffsetX_sec = 2u;
        uint32_t tileOffsetY_sec = 0u;
        uint32_t tileOffsetX_prim = 1u;
        uint32_t tileOffsetY_prim = 0u;
        uint32_t dispatchWidth = m_config.width;
        uint32_t dispatchHeight = (m_config.height + 1) / 2;
        uint32_t secAccumHistory = 1u;
        uint32_t mergeMode = 0u; // 0 = InterleavedScanline, 1 = CheckerboardTile, 2 = SampleParallel

        CameraUniform uboSec = ubo;

        if (activeMode == MultiGpuMode::InterleavedScanline) {
            mergeMode = 0u;
            tileOffsetX_sec = 2u;
            tileOffsetY_sec = 0u;
            tileOffsetX_prim = 1u;
            tileOffsetY_prim = 0u;
            dispatchWidth = m_config.width;
            dispatchHeight = (m_config.height + 1) / 2;
            secAccumHistory = 1u;
        } else if (activeMode == MultiGpuMode::CheckerboardTile) {
            mergeMode = 1u;
            tileOffsetX_sec = 2u;
            tileOffsetY_sec = m_config.tile_size;
            tileOffsetX_prim = 1u;
            tileOffsetY_prim = m_config.tile_size;
            dispatchWidth = (m_config.width + 1) / 2;
            dispatchHeight = m_config.height;
            secAccumHistory = 1u;
        } else if (activeMode == MultiGpuMode::SampleParallel) {
            mergeMode = 2u;
            tileOffsetX_sec = 0u;
            tileOffsetY_sec = 0u;
            tileOffsetX_prim = 0u;
            tileOffsetY_prim = 0u;
            dispatchWidth = m_config.width;
            dispatchHeight = m_config.height;
            secAccumHistory = 0u; // Secondary only renders current frame's delta; Primary accumulates

            // Split SPP: e.g. spp = 2 -> prim: 1, sec: 1; spp = 4 -> prim: 2, sec: 2
            uint32_t primSpp = (m_config.spp + 1) / 2;
            uint32_t secSpp = m_config.spp / 2;
            ubo.spp = primSpp;
            uboSec.spp = secSpp;

            // De-correlate secondary PRNG seed from primary
            uboSec.frameIndex = m_frameIndex + 1000003u;

            // Re-upload primary camera UBO with primSpp
            m_cameraUBOs[m_currentFrame]->copyFrom(&ubo, sizeof(CameraUniform));
        }

        if (!m_mgpu->isZeroCopyActive()) {
            frameBytes = static_cast<size_t>(dispatchWidth) * dispatchHeight * bytesPerPixel;
            dstHost = m_secTransferBuffer ? m_secTransferBuffer->map() : nullptr;
        }

        uint32_t slot = m_config.double_buffered_shared_mem ? (m_currentFrame % 2) : 0;

        // 1. Launch secondary GPU asynchronously
        m_mgpu->launchSecondaryWork(uboSec, slot, tileOffsetX_sec, tileOffsetY_sec, m_config.width, m_config.height,
                                   m_numTriangles, m_numSpheres, m_numMaterials, m_numLights, useHwRT,
                                   hasEnvMap, envIntensity, secAccumHistory, dstHost, frameBytes);

        // 2. Concurrently record and execute primary GPU
        vkWaitForFences(device, 1, &m_rtFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &m_rtFence);

        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo rtBeginInfo{};
        rtBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &rtBeginInfo);

        uint32_t qBase = m_currentFrame * 4;
        vkCmdResetQueryPool(cmd, m_queryPool, qBase, 4);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, qBase + 0);

        uint32_t rtPushConstants[12] = {
            m_numTriangles, m_numSpheres, m_numMaterials, m_numLights,
            tileOffsetX_prim,
            tileOffsetY_prim,
            m_config.width, m_config.height,
            useHwRT,
            hasEnvMap,
            envIntensityBits,
            1u // accumulateHistory
        };

        // Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtpKhrPipeline->getPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtpPipelineLayout, 0, 1, &m_rtDescSets[m_currentFrame], 0, nullptr);
        VkShaderStageFlags rtpStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        vkCmdPushConstants(cmd, m_rtpPipelineLayout, rtpStages, 0, sizeof(rtPushConstants), rtPushConstants);
        m_rtpKhrPipeline->traceRays(cmd, dispatchWidth, dispatchHeight, 1);

        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_queryPool, qBase + 1);
        vkEndCommandBuffer(cmd);

        VkSubmitInfo rtSubmit{};
        rtSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        rtSubmit.commandBufferCount = 1;
        rtSubmit.pCommandBuffers = &cmd;
        vkQueueSubmit(queue, 1, &rtSubmit, m_rtFence);

        // 3. Wait for primary GPU raytracing and secondary GPU completion + PCIe transfer
        vkWaitForFences(device, 1, &m_rtFence, VK_TRUE, UINT64_MAX);
        m_mgpu->syncAndTransfer(nullptr, 0);

        // 4. Record Merge & Tonemapping commands on primary GPU
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo postBeginInfo{};
        postBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &postBeginInfo);

        // Barrier: Ensure primary RT writes to m_accumImage and secondary DMA host writes are visible before merge compute reads/writes
        VkMemoryBarrier2 rtToMergeBarrier{};
        rtToMergeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        rtToMergeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
        rtToMergeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
        rtToMergeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rtToMergeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo rtToMergeDep{};
        rtToMergeDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        rtToMergeDep.memoryBarrierCount = 1;
        rtToMergeDep.pMemoryBarriers = &rtToMergeBarrier;
        vkCmdPipelineBarrier2(cmd, &rtToMergeDep);

        // Merge Pass
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_mergePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_mergePipelineLayout, 0, 1, &m_mergeDescSets[slot], 0, nullptr);

        uint32_t mergePC[6] = { m_config.width, m_config.height, m_config.spp, m_config.tile_size, formatMode, mergeMode };
        vkCmdPushConstants(cmd, m_mergePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mergePC), mergePC);

        uint32_t mergeGroupsX = (m_config.width + 15) / 16;
        uint32_t mergeGroupsY = (mergeMode == 0u) ? (((m_config.height + 1) / 2 + 15) / 16) : ((m_config.height + 15) / 16);
        vkCmdDispatch(cmd, mergeGroupsX, mergeGroupsY, 1);

        VkMemoryBarrier2 mergeBarrier{};
        mergeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        mergeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mergeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        mergeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        mergeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo mergeDep{};
        mergeDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        mergeDep.memoryBarrierCount = 1;
        mergeDep.pMemoryBarriers = &mergeBarrier;
        vkCmdPipelineBarrier2(cmd, &mergeDep);

        // Tonemapping
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &m_tonemapDescSet, 0, nullptr);

        tonemapConstants.visualizeSplit = m_config.visualize_mgpu_split ? 1u : 0u;
        tonemapConstants.tileSize = (activeMode == MultiGpuMode::InterleavedScanline) ? 0u : m_config.tile_size;
        tonemapConstants.totalSamples = (m_frameIndex + 1) * m_config.spp;
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, qBase + 2);
        vkCmdPushConstants(cmd, m_tonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tonemapConstants), &tonemapConstants);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);

        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 3);
    }

    // 3. Interactive Blit & Dear ImGui Overlay
    if (!m_config.headless && m_swapchain) {
        VkImage swapImage = m_swapchain->getImage(imageIndex);
        VkImageView swapView = m_swapchain->getImageView(imageIndex);

        // Transition m_outputImage to TRANSFER_SRC_OPTIMAL
        m_outputImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
        );

        // Transition swapImage from UNDEFINED to TRANSFER_DST_OPTIMAL
        VkImageMemoryBarrier2 toDst{};
        toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toDst.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        toDst.srcAccessMask = VK_ACCESS_2_NONE;
        toDst.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        toDst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toDst.image = swapImage;
        toDst.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo depToDst{};
        depToDst.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depToDst.imageMemoryBarrierCount = 1;
        depToDst.pImageMemoryBarriers = &toDst;
            vkCmdPipelineBarrier2(cmd, &depToDst);

        // Copy or blit output image to swapchain image
        bool extentsMatch = (m_config.width == m_swapchain->getExtent().width &&
                             m_config.height == m_swapchain->getExtent().height);
        bool formatsMatch = (m_outputImage->getFormat() == m_swapchain->getFormat());

        if (extentsMatch && formatsMatch) {
            VkImageCopy copyRegion{};
            copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.extent = { m_config.width, m_config.height, 1 };
            vkCmdCopyImage(cmd, m_outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
        } else {
            VkImageBlit blitRegion{};
            blitRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blitRegion.srcOffsets[0] = { 0, 0, 0 };
            blitRegion.srcOffsets[1] = { static_cast<int32_t>(m_config.width), static_cast<int32_t>(m_config.height), 1 };
            blitRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blitRegion.dstOffsets[0] = { 0, 0, 0 };
            blitRegion.dstOffsets[1] = { static_cast<int32_t>(m_swapchain->getExtent().width), static_cast<int32_t>(m_swapchain->getExtent().height), 1 };
            VkFilter filter = extentsMatch ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
            vkCmdBlitImage(cmd, m_outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, filter);
        }

        // Transition m_outputImage back to GENERAL
        m_outputImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );

        // Transition swapImage to COLOR_ATTACHMENT_OPTIMAL for ImGui
        VkImageMemoryBarrier2 toColor{};
        toColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toColor.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        toColor.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toColor.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toColor.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toColor.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toColor.image = swapImage;
        toColor.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo depToColor{};
        depToColor.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depToColor.imageMemoryBarrierCount = 1;
        depToColor.pImageMemoryBarriers = &toColor;
        vkCmdPipelineBarrier2(cmd, &depToColor);

        // Render ImGui overlay
        if (m_gui) {
            m_gui->newFrame();
            bool prevMode = m_cameraMode;
            MultiGpuMode prevMgpuMode = m_config.mgpu_mode;
            GuiActions guiActions{};
            if (m_gui->render(cmd, swapView, m_swapchain->getExtent().width, m_swapchain->getExtent().height,
                              m_config, getStats(), m_cameraMode, m_camera.get(),
                              &m_window->getDisplayInfo(), m_window->isFullscreen(), &guiActions,
                              m_availableScenes, m_currentSceneIndex)) {
                m_resetAccumulation = true;
                if (!m_config.headless) m_frameTimesMs.clear();
            }
            if (guiActions.sceneChanged && !guiActions.newScenePath.empty()) {
                m_pendingSceneChange = true;
                m_pendingScenePath = guiActions.newScenePath;
            }
            if (guiActions.mgpuModeChanged) {
                m_pendingMgpuModeChange = true;
                m_newMgpuMode = guiActions.newMgpuMode;
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
            if (m_cameraMode != prevMode) {
                setCameraMode(m_cameraMode);
            }
        }

        // Transition swapImage to PRESENT_SRC_KHR (or copy to staging if dumping UI)
        if (!m_config.dump_ui_path.empty()) {
            VkImageMemoryBarrier2 toSrc{};
            toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            toSrc.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            toSrc.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            toSrc.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            toSrc.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            toSrc.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toSrc.image = swapImage;
            toSrc.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depToSrc{};
            depToSrc.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depToSrc.imageMemoryBarrierCount = 1;
            depToSrc.pImageMemoryBarriers = &toSrc;
            vkCmdPipelineBarrier2(cmd, &depToSrc);

            if (!m_uiDumpBuffer) {
                VkDeviceSize size = static_cast<VkDeviceSize>(m_swapchain->getExtent().width) * m_swapchain->getExtent().height * 4;
                m_uiDumpBuffer = std::make_unique<Buffer>(m_context->getAllocator(), size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                         VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
            }

            VkBufferImageCopy copyRegion{};
            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageExtent = { m_swapchain->getExtent().width, m_swapchain->getExtent().height, 1 };
            vkCmdCopyImageToBuffer(cmd, swapImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_uiDumpBuffer->getBuffer(), 1, &copyRegion);

            VkImageMemoryBarrier2 toPresent{};
            toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            toPresent.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            toPresent.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            toPresent.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            toPresent.dstAccessMask = VK_ACCESS_2_NONE;
            toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            toPresent.image = swapImage;
            toPresent.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depToPresent{};
            depToPresent.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depToPresent.imageMemoryBarrierCount = 1;
            depToPresent.pImageMemoryBarriers = &toPresent;
            vkCmdPipelineBarrier2(cmd, &depToPresent);
        } else {
            VkImageMemoryBarrier2 toPresent{};
            toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            toPresent.srcStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
            toPresent.srcAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
            toPresent.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
            toPresent.dstAccessMask = VK_ACCESS_2_NONE;
            toPresent.oldLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
            toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
            toPresent.image = swapImage;
            toPresent.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo depToPresent{};
            depToPresent.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depToPresent.imageMemoryBarrierCount = 1;
            depToPresent.pImageMemoryBarriers = &toPresent;
            vkCmdPipelineBarrier2(cmd, &depToPresent);
        }
    }

    vkEndCommandBuffer(cmd);

    // Submit Work
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    if (!m_config.headless && m_swapchain) {
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &m_imageAvailableSemaphores[m_currentFrame];
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &m_renderFinishedSemaphores[imageIndex];
    }

    vkQueueSubmit(queue, 1, &submitInfo, m_inFlightFences[m_currentFrame]);

    if (!m_config.headless && m_swapchain) {
        VkResult res = m_swapchain->queuePresent(queue, imageIndex, m_renderFinishedSemaphores[imageIndex]);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
            int curW = 0, curH = 0;
            SDL_GetWindowSizeInPixels(m_window->getSDLWindow(), &curW, &curH);
            uint32_t targetW = (curW > 0) ? static_cast<uint32_t>(curW) : m_window->getWidth();
            uint32_t targetH = (curH > 0) ? static_cast<uint32_t>(curH) : m_window->getHeight();
            onResize(targetW, targetH, /*forceRecreate=*/true);
        }
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
        if (m_mgpu && m_mgpu->isMultiGpuActive()) {
            Logger::info("Frame {:3d} | Dual-GPU Total: {:.3f} ms (GPU 0: {:.3f} ms, GPU 1: {:.3f} ms, Tonemap: {:.3f} ms) | Target <8ms: {}",
                         m_totalFramesRendered, totalGpuMs, gpuRtMs, secGpuMs, gpuTonemapMs,
                         totalGpuMs < 8.0 ? "\033[32mPASS\033[0m" : "\033[33mCHECK\033[0m");
        } else {
            Logger::info("Frame {:3d} | Single GPU Total: {:.3f} ms (RT: {:.3f} ms, Tonemap: {:.3f} ms) | Target <8ms: {}",
                         m_totalFramesRendered, totalGpuMs, gpuRtMs, gpuTonemapMs,
                         totalGpuMs < 8.0 ? "\033[32mPASS\033[0m" : "\033[33mCHECK\033[0m");
        }
    }
}

void Engine::dumpOutputFiles() {
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();
    VkQueue queue = m_context->getGraphicsQueue();

    vkDeviceWaitIdle(device);

    // Drain remaining in-flight queries after vkDeviceWaitIdle
    uint32_t drainCount = std::min(m_totalFramesRendered, MAX_FRAMES_IN_FLIGHT);
    for (uint32_t d = 0; d < drainCount; ++d) {
        uint32_t slot = (m_totalFramesRendered - drainCount + d) % MAX_FRAMES_IN_FLIGHT;
        uint32_t qBase = slot * 4;
        uint64_t timestamps[4] = {0, 0, 0, 0};
        vkGetQueryPoolResults(device, m_queryPool, qBase, 4, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        double gpuRtMs = 0.0;
        if (timestamps[1] > timestamps[0]) {
            gpuRtMs = (timestamps[1] - timestamps[0]) * m_timestampPeriod * 1e-6;
        }
        double gpuTonemapMs = 0.0;
        if (timestamps[3] > timestamps[2]) {
            gpuTonemapMs = (timestamps[3] - timestamps[2]) * m_timestampPeriod * 1e-6;
        }
        if (gpuRtMs > 10000.0) gpuRtMs = 0.0;
        if (gpuTonemapMs > 10000.0) gpuTonemapMs = 0.0;

        double secGpuMs = 0.0;
        if (m_mgpu && m_mgpu->isMultiGpuActive()) {
            secGpuMs = m_mgpu->getSecondaryGpuTimeMs();
        }
        double totalGpuMs = (m_mgpu && m_mgpu->isMultiGpuActive()) ? (std::max(gpuRtMs, secGpuMs) + gpuTonemapMs) : (gpuRtMs + gpuTonemapMs);

        if (totalGpuMs > 0.01) {
            m_lastGpuRtMs = gpuRtMs;
            m_lastSecGpuMs = secGpuMs;
            m_lastTonemapMs = gpuTonemapMs;
            m_lastFrameTimeMs = totalGpuMs;
            m_frameTimesMs.push_back(m_lastFrameTimeMs);
            recordFrameTally(totalGpuMs, gpuRtMs, secGpuMs, gpuTonemapMs);
        }
    }

    // 1. Dump LDR PNG
    if (!m_config.dump_frame_path.empty()) {
        VkDeviceSize bufferSize = m_config.width * m_config.height * 4;
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

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffers[0];
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        const uint8_t* pixels = static_cast<const uint8_t*>(staging.map());
        ImageDumper::savePNG(m_config.dump_frame_path, m_config.width, m_config.height, pixels);
        staging.unmap();
    }

    // 2. Dump HDR OpenEXR
    if (!m_config.dump_hdr_path.empty()) {
        VkDeviceSize bufferSize = m_config.width * m_config.height * 4 * sizeof(float);
        Buffer staging(allocator, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        vkResetCommandBuffer(m_commandBuffers[0], 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(m_commandBuffers[0], &beginInfo);

        m_accumImage->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
        );

        VkBufferImageCopy copyRegion{};
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = { m_config.width, m_config.height, 1 };

        vkCmdCopyImageToBuffer(m_commandBuffers[0], m_accumImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.getBuffer(), 1, &copyRegion);

        m_accumImage->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );

        vkEndCommandBuffer(m_commandBuffers[0]);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffers[0];
        vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        const float* floatPixels = static_cast<const float*>(staging.map());
        ImageDumper::saveEXR(m_config.dump_hdr_path, m_config.width, m_config.height, floatPixels);
        staging.unmap();
    }

    // 3. Dump UI Viewport Backbuffer
    if (!m_config.dump_ui_path.empty() && m_uiDumpBuffer && m_swapchain) {
        uint32_t w = m_swapchain->getExtent().width;
        uint32_t h = m_swapchain->getExtent().height;
        const uint8_t* raw = static_cast<const uint8_t*>(m_uiDumpBuffer->map());
        std::vector<uint8_t> rgba(w * h * 4);
        bool isBgra = (m_swapchain->getFormat() == VK_FORMAT_B8G8R8A8_UNORM || m_swapchain->getFormat() == VK_FORMAT_B8G8R8A8_SRGB);
        for (size_t i = 0; i < w * h; ++i) {
            if (isBgra) {
                rgba[i * 4 + 0] = raw[i * 4 + 2];
                rgba[i * 4 + 1] = raw[i * 4 + 1];
                rgba[i * 4 + 2] = raw[i * 4 + 0];
                rgba[i * 4 + 3] = raw[i * 4 + 3];
            } else {
                rgba[i * 4 + 0] = raw[i * 4 + 0];
                rgba[i * 4 + 1] = raw[i * 4 + 1];
                rgba[i * 4 + 2] = raw[i * 4 + 2];
                rgba[i * 4 + 3] = raw[i * 4 + 3];
            }
        }
        m_uiDumpBuffer->unmap();
        ImageDumper::savePNG(m_config.dump_ui_path, w, h, rgba.data());
    }

    // 4. Dump Stats JSON
    if (!m_config.dump_stats_path.empty()) {
        FrameStats stats = getStats();
        ImageDumper::saveStatsJSON(m_config.dump_stats_path, stats);
    }
}

FrameStats Engine::getStats() const {
    FrameStats stats;
    stats.gpu_name = m_context->getDeviceName();
    if (m_mgpu && m_mgpu->isMultiGpuActive()) {
        stats.topology_name = stats.gpu_name + " + " + m_mgpu->getSecondaryDeviceName();
    } else {
        stats.topology_name = stats.gpu_name;
    }
    stats.width = m_config.width;
    stats.height = m_config.height;
    stats.spp = m_config.spp;
    stats.total_frames = m_totalFramesRendered;
    stats.validation_errors = m_context->getValidationErrors();

    stats.current_frame_time_ms = m_lastFrameTimeMs;
    stats.current_fps = m_lastFrameTimeMs > 0.0001 ? (1000.0 / m_lastFrameTimeMs) : 0.0;

    if (!m_frameTimesMs.empty()) {
        double sum = std::accumulate(m_frameTimesMs.begin(), m_frameTimesMs.end(), 0.0);
        stats.avg_frame_time_ms = sum / m_frameTimesMs.size();
        stats.min_frame_time_ms = *std::min_element(m_frameTimesMs.begin(), m_frameTimesMs.end());
        stats.max_frame_time_ms = *std::max_element(m_frameTimesMs.begin(), m_frameTimesMs.end());
        stats.avg_fps = stats.avg_frame_time_ms > 0.0 ? 1000.0 / stats.avg_frame_time_ms : 0.0;
        stats.target_achieved = (stats.avg_frame_time_ms < 8.0);

        // Rays per second based on active throughput
        double frameTimeForThroughput = stats.current_frame_time_ms > 0.001 ? stats.current_frame_time_ms : stats.avg_frame_time_ms;
        double raysPerFrame = static_cast<double>(m_config.width) * m_config.height * m_config.spp * m_config.max_bounces;
        stats.rays_per_second = (frameTimeForThroughput > 0.0) ? (raysPerFrame / (frameTimeForThroughput / 1000.0)) : 0.0;
    }

    switch (m_config.mgpu_mode) {
        case MultiGpuMode::InterleavedScanline: stats.mgpu_mode_str = "interleaved_scanline"; break;
        case MultiGpuMode::CheckerboardTile: stats.mgpu_mode_str = "checkerboard_tile"; break;
        case MultiGpuMode::SampleParallel: stats.mgpu_mode_str = "sample_parallel"; break;
        case MultiGpuMode::Auto: stats.mgpu_mode_str = (m_config.spp > 1) ? "auto (sample_parallel)" : "auto (checkerboard_tile)"; break;
        default: stats.mgpu_mode_str = "single_gpu"; break;
    }

    stats.pipeline_type_str = "rtp";

    stats.primary_gpu_time_ms = m_lastGpuRtMs;
    stats.secondary_gpu_time_ms = m_lastSecGpuMs;
    stats.tonemap_time_ms = m_lastTonemapMs;
    stats.num_triangles = m_numTriangles;
    stats.num_spheres = m_numSpheres;
    stats.num_materials = m_numMaterials;
    stats.num_lights = m_numLights;
    stats.num_textures = static_cast<uint32_t>(m_sceneTextures.size());
    stats.width = m_config.width;
    stats.height = m_config.height;
    stats.spp = m_config.spp;
    stats.max_bounces = m_config.max_bounces;
    stats.render_scale = m_config.render_scale;
    stats.total_frames = m_totalFramesRendered;
    stats.validation_errors = m_context->getValidationErrors();
    // 1. Session & Host Platform Metadata
    stats.os_name = getCachedOS();
#ifdef __linux__
    struct utsname uts{};
    if (uname(&uts) == 0) {
        stats.kernel_version = std::string(uts.sysname) + " " + uts.release;
    }
#endif
    stats.cpu_model = getCachedCPU();
    stats.ram_total_gb = static_cast<double>(getCachedRAM()) / 1024.0;

    // 2. Physical & Driver Device Information
    stats.gpu_name = m_context->getDeviceName();
    stats.vendor_id = m_context->getVendorID();
    stats.device_id = m_context->getDeviceID();
    stats.arch_name = m_context->getArchitectureName();
    stats.short_arch = m_context->getShortArchName();
    stats.ray_accelerator_name = m_context->getRayAcceleratorName();
    stats.is_rdna3 = m_context->isRDNA3();
    stats.is_rdna4 = m_context->isRDNA4();
    uint32_t drvVer = m_context->getDriverVersion();
    stats.driver_version_str = std::format("{}.{}.{}", (drvVer >> 22) & 0x3FF, (drvVer >> 12) & 0x3FF, drvVer & 0xFFF);
    uint32_t apiVer = m_context->getApiVersion();
    stats.vulkan_api_str = std::format("{}.{}.{}", VK_API_VERSION_MAJOR(apiVer), VK_API_VERSION_MINOR(apiVer), VK_API_VERSION_PATCH(apiVer));
    stats.device_type_str = (m_context->getDeviceType() == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? "Discrete GPU" : "Integrated GPU";
    stats.total_vram_mb = static_cast<double>(m_context->getTotalVramBytes()) / (1024.0 * 1024.0);
    stats.vram_used_mb = static_cast<double>(m_context->getAllocatedVramBytes()) / (1024.0 * 1024.0);
    stats.vram_budget_mb = stats.total_vram_mb;

    // Primary GPU PCIe and Sensors
    const auto& primPci = m_context->getPciLinkInfo();
    stats.primary_pci_link = primPci.formattedLink;
    stats.primary_pci_speed = primPci.currentSpeed;
    stats.primary_pci_width = primPci.currentWidth;
    stats.primary_pci_max_speed = primPci.maxSpeed;
    stats.primary_pci_max_width = primPci.maxWidth;
    stats.primary_pci_degraded = primPci.isDegraded;
    stats.primary_pci_degraded_reason = primPci.degradationReason;
    stats.primary_gpu_clock_mhz = m_gpu0ClockMhz.load(std::memory_order_relaxed);
    stats.primary_gpu_temp_c = m_gpu0TempC.load(std::memory_order_relaxed);

    // 3. Secondary GPU Hardware & Driver
    if (m_mgpu && m_mgpu->isMultiGpuActive()) {
        stats.is_mgpu_active = true;
        stats.secondary_gpu_name = m_mgpu->getSecondaryDeviceName();
        if (m_mgpu->getSecondaryContext()) {
            const auto& secPci = m_mgpu->getSecondaryContext()->getPciLinkInfo();
            stats.secondary_arch_name = m_mgpu->getSecondaryContext()->getArchitectureName();
            stats.secondary_pci_link = secPci.formattedLink;
            if (!secPci.formattedLink.empty() && secPci.formattedLink != "PCIe N/A") {
                stats.mgpu_interconnect_str = secPci.formattedLink;
            }
            stats.secondary_pci_speed = secPci.currentSpeed;
            stats.secondary_pci_width = secPci.currentWidth;
            stats.secondary_pci_max_speed = secPci.maxSpeed;
            stats.secondary_pci_max_width = secPci.maxWidth;
            stats.secondary_pci_degraded = secPci.isDegraded;
            stats.secondary_pci_degraded_reason = secPci.degradationReason;
            stats.secondary_gpu_clock_mhz = m_gpu1ClockMhz.load(std::memory_order_relaxed);
            stats.secondary_gpu_temp_c = m_gpu1TempC.load(std::memory_order_relaxed);
        }
    } else if (m_mgpu && m_mgpu->isSecondaryInitialized()) {
        stats.is_mgpu_active = false;
        stats.secondary_gpu_name = m_mgpu->getSecondaryDeviceName();
        if (m_mgpu->getSecondaryContext()) {
            const auto& secPci = m_mgpu->getSecondaryContext()->getPciLinkInfo();
            stats.secondary_arch_name = m_mgpu->getSecondaryContext()->getArchitectureName();
            stats.secondary_pci_link = secPci.formattedLink;
            stats.secondary_pci_speed = secPci.currentSpeed;
            stats.secondary_pci_width = secPci.currentWidth;
            stats.secondary_pci_max_speed = secPci.maxSpeed;
            stats.secondary_pci_max_width = secPci.maxWidth;
            stats.secondary_pci_degraded = secPci.isDegraded;
            stats.secondary_pci_degraded_reason = secPci.degradationReason;
        }
        stats.secondary_gpu_clock_mhz = m_gpu1ClockMhz.load(std::memory_order_relaxed);
        stats.secondary_gpu_temp_c = m_gpu1TempC.load(std::memory_order_relaxed);
    } else {
        stats.is_mgpu_active = false;
        stats.secondary_gpu_name = "";
        stats.secondary_arch_name = "";
        stats.secondary_pci_link = "";
        stats.secondary_pci_speed = "";
        stats.secondary_pci_width = 0;
        stats.secondary_pci_max_speed = "";
        stats.secondary_pci_max_width = 0;
        stats.secondary_pci_degraded = false;
        stats.secondary_pci_degraded_reason = "";
        stats.secondary_gpu_clock_mhz = 0;
        stats.secondary_gpu_temp_c = 0;
    }

    // 4. Hardware Support Levels
    stats.has_hw_rt = (m_tlas != nullptr);
    stats.has_ray_query = true;
    stats.has_as = (m_tlas != nullptr);
    stats.has_bda = true;
    stats.has_dho = true;
    stats.has_rt_pipeline = (m_rtpKhrPipeline != nullptr);
    stats.has_dgc = (m_rtpKhrPipeline && m_rtpKhrPipeline->isIndirectSupported());
    stats.has_subgroup_control = m_context->hasSubgroupSizeControl();
    stats.subgroup_size = 32;
    stats.has_dynamic_rendering = true;
    stats.has_timeline_semaphores = true;
    stats.has_sync2 = true;

    // 5. Engine Settings & State
    stats.visualize_mgpu_split = m_config.visualize_mgpu_split;
    stats.render_scale = m_config.render_scale;
    stats.max_bounces = m_config.max_bounces;
    stats.checkerboard_tile_size = m_config.tile_size;
    stats.enable_direct_light = m_config.enable_direct_light;
    stats.enable_indirect_light = m_config.enable_indirect_light;
    stats.enable_refraction = m_config.enable_refraction;
    stats.enable_shadows = m_config.enable_shadows;
    stats.aces_tonemap = m_config.aces_tonemap;
    stats.scene_path = m_config.scene_path.empty() ? "Cornell Box + Specular/Refraction Spheres" : m_config.scene_path;
    stats.hdri_path = m_config.hdri_path;

    // 6. Active Camera Framing
    if (m_camera) {
        glm::vec3 pos = m_camera->getPosition();
        stats.cam_pos[0] = pos.x;
        stats.cam_pos[1] = pos.y;
        stats.cam_pos[2] = pos.z;
        stats.cam_yaw = m_camera->getYaw();
        stats.cam_pitch = m_camera->getPitch();
        stats.cam_fov = m_camera->getFov();
    }

    for (const auto& tally : m_configTallies) {
        FrameStats::ConfigTallySummary s;
        s.label = tally.label;
        s.frame_count = tally.frameCount;
        s.avg_frame_time_ms = tally.getAvgFrameTimeMs();
        s.min_frame_time_ms = tally.minFrameTimeMs;
        s.max_frame_time_ms = tally.maxFrameTimeMs;
        s.avg_fps = tally.getAvgFps();
        s.primary_gpu_time_ms = tally.getAvgPrimaryRtMs();
        s.secondary_gpu_time_ms = tally.getAvgSecondaryRtMs();
        s.tonemap_time_ms = tally.getAvgTonemapMs();
        s.gigarays_per_second = tally.getRayThroughput() * 1e-9;
        s.target_achieved = tally.isTargetAchieved();
        stats.configurations_breakdown.push_back(std::move(s));
    }

    return stats;
}

std::string Engine::exportTelemetry(const std::string& customPath) {
    std::string path = customPath.empty() ? ImageDumper::generateDefaultTelemetryPath() : customPath;
    FrameStats stats = getStats();
    if (ImageDumper::saveStatsJSON(path, stats)) {
        Logger::info("Exported comprehensive telemetry dataset to: {}", path);
        return path;
    } else {
        Logger::error("Failed to export telemetry dataset to: {}", path);
        return "";
    }
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

    Logger::info("Handling window resize: updating viewport from {}x{} to {}x{}", m_config.width, m_config.height, newWidth, newHeight);

    VkDevice device = m_context->getDevice();
    VkQueue queue = m_context->getGraphicsQueue();
    VmaAllocator allocator = m_context->getAllocator();

    vkDeviceWaitIdle(device);

    m_config.width = newWidth;
    m_config.height = newHeight;

    // 1. Recreate Swapchain (destroy old swapchain first so surface is released)
    m_swapchain.reset();
    m_swapchain = std::make_unique<Swapchain>(
        device,
        m_context->getPhysicalDevice(),
        m_surface,
        m_config.width,
        m_config.height,
        m_context->getGraphicsQueueFamily()
    );
    m_config.width = m_swapchain->getExtent().width;
    m_config.height = m_swapchain->getExtent().height;

    // 2. Recreate render finish semaphores for new swapchain image count
    for (auto sem : m_renderFinishedSemaphores) {
        if (sem != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, sem, nullptr);
        }
    }
    m_renderFinishedSemaphores.clear();

    VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    uint32_t numSwapImages = m_swapchain->getImageCount();
    m_renderFinishedSemaphores.resize(numSwapImages);
    for (size_t i = 0; i < numSwapImages; ++i) {
        vkCreateSemaphore(device, &semInfo, nullptr, &m_renderFinishedSemaphores[i]);
    }

    // 3. Recreate Accumulation & Output Images
    VkFormat accumFmt = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    m_accumImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        accumFmt,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_outputImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // Transition images to GENERAL layout
    vkResetCommandBuffer(m_commandBuffers[0], 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_commandBuffers[0], &beginInfo);

    m_accumImage->transitionLayout(
        m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    m_outputImage->transitionLayout(
        m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    vkEndCommandBuffer(m_commandBuffers[0]);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffers[0];
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    // 4. Recreate ReSTIR DI Buffers
    initReSTIRBuffers();
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        updateReSTIRDescriptors(i);
    }

    // 5. Resize secondary GPU if active before updating merge descriptor set
    if (m_mgpu && m_mgpu->isMultiGpuActive()) {
        m_mgpu->resize(m_config.width, m_config.height);
    }

    // 6. Update all image descriptors and multi-GPU merge descriptors
    updateAllImageDescriptors();
    updateMergeDescriptors();

    // 5. Reset UI dump buffer if allocated
    if (m_uiDumpBuffer) {
        m_uiDumpBuffer.reset();
    }

    // 7. Adapt Camera aspect ratio & FOV
    if (m_camera) {
        float aspect = static_cast<float>(m_config.width) / static_cast<float>(m_config.height);
        m_camera->adaptFovForAspect(aspect);
    }

    // 9. Invalidate accumulation
    m_frameIndex = 0;
    m_resetAccumulation = true;
}

std::string Engine::getActiveSceneName() const {
    if (m_currentSceneIndex >= 0 && m_currentSceneIndex < static_cast<int>(m_availableScenes.size())) {
        if (!m_availableScenes[m_currentSceneIndex].label.empty()) {
            return m_availableScenes[m_currentSceneIndex].label;
        }
    }
    if (m_config.scene_path.empty() || m_config.scene_path == "__procedural_cornell_box__") {
        return "Procedural Cornell Box";
    }
    std::filesystem::path p(m_config.scene_path);
    return SceneRegistry::formatSceneName(p.stem().string());
}

void Engine::recordFrameTally(double frameTimeMs, double primRtMs, double secRtMs, double tonemapMs) {
    ConfigKey key;
    key.scene_name = getActiveSceneName();
    key.mgpu_mode = (m_mgpu && m_mgpu->isMultiGpuActive() && m_config.mgpu_mode != MultiGpuMode::Off) ? m_config.mgpu_mode : MultiGpuMode::Off;
    key.width = m_config.width;
    key.height = m_config.height;
    key.spp = m_config.spp;
    key.max_bounces = m_config.max_bounces;
    key.accum_format = m_config.accum_format;
    key.tile_size = m_config.tile_size;

    for (auto& tally : m_configTallies) {
        if (tally.key == key) {
            tally.addSample(frameTimeMs, primRtMs, secRtMs, tonemapMs);
            return;
        }
    }
    ConfigStatsTally newTally(key);
    newTally.addSample(frameTimeMs, primRtMs, secRtMs, tonemapMs);
    m_configTallies.push_back(std::move(newTally));
}

void Engine::printExecutionSummary() const {
    Logger::info("========================================================================================");
    if (m_configTallies.empty()) {
        Logger::info("  Execution Summary: No frames rendered.");
        Logger::info("========================================================================================");
        return;
    }

    Logger::info("  Execution Summary (Tallied Across {} Unique Configuration{}):",
                 m_configTallies.size(), m_configTallies.size() == 1 ? "" : "s");
    Logger::info("----------------------------------------------------------------------------------------");

    for (size_t i = 0; i < m_configTallies.size(); ++i) {
        const auto& tally = m_configTallies[i];
        Logger::info("  [Config {}/{}] {}", i + 1, m_configTallies.size(), tally.label);
        Logger::info("    Rendered Frames:     {}", tally.frameCount);
        Logger::info("    Average Frame Time:  {:.3f} ms ({:.1f} FPS) [Min: {:.3f} ms, Max: {:.3f} ms]",
                     tally.getAvgFrameTimeMs(), tally.getAvgFps(), tally.minFrameTimeMs, tally.maxFrameTimeMs);

        if (tally.key.mgpu_mode != MultiGpuMode::Off && tally.getAvgSecondaryRtMs() > 0.001) {
            Logger::info("    GPU Breakdown:       GPU 0: {:.3f} ms | GPU 1: {:.3f} ms | Tonemap & Merge: {:.3f} ms",
                         tally.getAvgPrimaryRtMs(), tally.getAvgSecondaryRtMs(), tally.getAvgTonemapMs());

            // Look up single-GPU baseline for the same scene, resolution, spp, bounces, and format
            double baselineMs = 0.0;
            for (const auto& other : m_configTallies) {
                if (other.key.scene_name == tally.key.scene_name &&
                    other.key.width == tally.key.width &&
                    other.key.height == tally.key.height &&
                    other.key.spp == tally.key.spp &&
                    other.key.max_bounces == tally.key.max_bounces &&
                    other.key.accum_format == tally.key.accum_format &&
                    other.key.mgpu_mode == MultiGpuMode::Off &&
                    other.frameCount > 0) {
                    baselineMs = other.getAvgFrameTimeMs();
                    break;
                }
            }

            if (baselineMs > 0.001) {
                double speedup = baselineMs / tally.getAvgFrameTimeMs();
                double efficiency = (speedup / 2.0) * 100.0;
                Logger::info("    Multi-GPU Scaling:   {:.2f}x speedup vs Single GPU ({:.1f}% efficiency)", speedup, efficiency);
            }
        } else {
            Logger::info("    GPU Breakdown:       GPU 0 (Primary RT): {:.3f} ms | Tonemap: {:.3f} ms | GPU 1: Standby",
                         tally.getAvgPrimaryRtMs(), tally.getAvgTonemapMs());
        }

        Logger::info("    Ray Throughput:      {:.2f} GigaRays/sec", tally.getRayThroughput() * 1e-9);
        Logger::info("    Sub-8ms Budget:      {}", tally.isTargetAchieved() ? "\033[32mACHIEVED\033[0m" : "\033[33mEXCEEDED\033[0m");
        if (i + 1 < m_configTallies.size()) {
            Logger::info("  --------------------------------------------------------------------------------------");
        }
    }

    Logger::info("----------------------------------------------------------------------------------------");
    Logger::info("  Total Frames Rendered: {} | Validation Errors: {}", m_totalFramesRendered, m_context->getValidationErrors());
    Logger::info("========================================================================================");
}

void Engine::run() {
    Logger::info("Starting Pathways render loop...");

    while (!m_window->shouldClose()) {
        m_window->pollEvents();

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

        if (m_config.frame_limit > 0 && m_totalFramesRendered >= m_config.frame_limit) {
            Logger::info("Reached frame limit of {} frames. Terminating loop.", m_config.frame_limit);
            break;
        }
    }

    dumpOutputFiles();
}

} // namespace pathways
