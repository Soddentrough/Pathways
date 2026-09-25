#include "core/Engine.hpp"
#include "core/Logger.hpp"
#include "ui/GuiManager.hpp"
#include "mgpu/MultiGpuManager.hpp"
#include "scene/GltfLoader.hpp"
#include "scene/UsdLoader.hpp"
#include "scene/LightTree.hpp"
#include "video/VideoDecoder.hpp"
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

    if (!m_config.headless && m_window) {
        m_window->setTitle(std::format("Pathways - Vulkan 1.4 Path Tracer ({})", m_context->getShortArchName()));
        // Auto-adapt peak luminance to native display capabilities if not customized by user
        if (!m_config.custom_hdr_peak && m_window->getDisplayInfo().isDisplayHdrCapable &&
            m_window->getDisplayInfo().maxLuminanceNits > 0.0f) {
            m_config.hdr_peak_nits = m_window->getDisplayInfo().maxLuminanceNits;
        }

        m_surface = m_window->createSurface(m_context->getInstance());

        m_swapchain = std::make_unique<Swapchain>(
            m_context->getDevice(),
            m_context->getPhysicalDevice(),
            m_surface,
            m_window->getWidth(),
            m_window->getHeight(),
            m_context->getGraphicsQueueFamily(),
            m_config.enable_hdr,
            m_window->isFullscreen(),
            m_context.get(),
            &m_window->getDisplayInfo(),
            m_config.hdr_peak_nits,
            m_config.hdr_paper_white_nits
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
    m_camera->setAspect(aspect);
    m_camera->setDynamicScaling(m_config.adaptive_speed);

    initVulkan();
    initScene();
    auto physicalDevices = VulkanContext::enumeratePhysicalDevices(m_context->getInstance());
    uint32_t hwDeviceCount = 0;
    for (auto pd : physicalDevices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
            hwDeviceCount++;
        }
    }
    if (hwDeviceCount >= 2) {
        m_mgpu = std::make_unique<MultiGpuManager>(m_config, m_context.get(), m_sceneData);
    }
    // Reclaim host memory used for scene geometry ingestion (now safely resident in device VRAM)
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

    startHwMonThread();

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
    stopHwMonThread();
    if (m_telemetryWorker.joinable()) {
        m_telemetryWorker.join();
    }
    if (m_gamepad) {
        SDL_CloseGamepad(m_gamepad);
        m_gamepad = nullptr;
    }
    VkDevice device = m_context->getDevice();
    vkDeviceWaitIdle(device);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_inFlightFences[i]) vkDestroyFence(device, m_inFlightFences[i], nullptr);
        if (m_rtCompleteSemaphores[i]) vkDestroySemaphore(device, m_rtCompleteSemaphores[i], nullptr);
    }
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

    m_wavefrontPipeline.reset();
    m_rtpKhrPipeline.reset();
    destroyGBufferResources();
    destroyPostProcessDescPool();
    destroyUpwaysResources();
    destroyUpwaysPipelines();
    destroyFsr3Resources();
    destroyFsr3Pipelines();
    destroyCausticsResources();
    destroyCausticsPipelines();
    destroyReSTIRResources();
    m_motionVectorImage.reset();
    m_mlAlbedoRoughnessImage.reset();
    m_mlSpecularMotionImage.reset();
    m_mlDiffuseImage.reset();
    m_mlSpecularImage.reset();
    if (m_tonemapPipeline) vkDestroyPipeline(device, m_tonemapPipeline, nullptr);
    if (m_mergePipeline) vkDestroyPipeline(device, m_mergePipeline, nullptr);
    if (m_accumRunningAvgPipeline) vkDestroyPipeline(device, m_accumRunningAvgPipeline, nullptr);
    if (m_accumTonemapPipeline) vkDestroyPipeline(device, m_accumTonemapPipeline, nullptr);

    if (m_rtpPipelineLayout) vkDestroyPipelineLayout(device, m_rtpPipelineLayout, nullptr);
    if (m_tonemapPipelineLayout) vkDestroyPipelineLayout(device, m_tonemapPipelineLayout, nullptr);
    if (m_mergePipelineLayout) vkDestroyPipelineLayout(device, m_mergePipelineLayout, nullptr);
    if (m_accumRunningAvgPipelineLayout) vkDestroyPipelineLayout(device, m_accumRunningAvgPipelineLayout, nullptr);
    if (m_accumTonemapPipelineLayout) vkDestroyPipelineLayout(device, m_accumTonemapPipelineLayout, nullptr);

    if (m_rtDescLayout) vkDestroyDescriptorSetLayout(device, m_rtDescLayout, nullptr);
    if (m_tonemapDescLayout) vkDestroyDescriptorSetLayout(device, m_tonemapDescLayout, nullptr);
    if (m_mergeDescLayout) vkDestroyDescriptorSetLayout(device, m_mergeDescLayout, nullptr);
    if (m_accumRunningAvgDescLayout) vkDestroyDescriptorSetLayout(device, m_accumRunningAvgDescLayout, nullptr);
    if (m_accumTonemapDescLayout) vkDestroyDescriptorSetLayout(device, m_accumTonemapDescLayout, nullptr);

    for (auto& img : m_frameImages) img.reset();

    if (m_updateTlasPipeline) vkDestroyPipeline(device, m_updateTlasPipeline, nullptr);
    if (m_updateTlasPipelineLayout) vkDestroyPipelineLayout(device, m_updateTlasPipelineLayout, nullptr);
    if (m_updateTlasDescLayout) vkDestroyDescriptorSetLayout(device, m_updateTlasDescLayout, nullptr);
    if (m_updateTlasDescPool) vkDestroyDescriptorPool(device, m_updateTlasDescPool, nullptr);
    m_tlasInstanceBuffer.reset();
    m_tlasInputInstancesBuffer.reset();
    m_tlasScratchBuffer.reset();
    m_instanceBuffer.reset();
    m_blases.clear();

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
    vkAllocateCommandBuffers(device, &allocInfo, m_postCommandBuffers.data());

    // Create Render Target Images
    VkFormat frameFmt = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_frameImages[i] = std::make_unique<Image>(
            device, allocator, m_config.width, m_config.height,
            frameFmt,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
    }
    m_accumImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        frameFmt,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // Internal post-process and tonemapping target is strictly 10-bit (or 16-bit float for scRGB HDR).
    // The internal pipeline is never degraded to 8-bit; SDR 8-bit presentation occurs via blit to the swapchain.
    VkFormat outputFmt = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    if (m_swapchain && !m_config.headless && m_swapchain->isHdr() && m_swapchain->getFormat() == VK_FORMAT_R16G16B16A16_SFLOAT) {
        outputFmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    }
    m_outputImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        outputFmt,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_motionVectorImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // G-Buffer Albedo & Roughness (Required for ML / Upways passes)
    m_mlAlbedoRoughnessImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_mlSpecularMotionImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    m_mlDiffuseImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    m_mlSpecularImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // Transition layouts to GENERAL
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_commandBuffers[0], &beginInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_frameImages[i]->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }

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

    m_motionVectorImage->transitionLayout(
        m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    if (m_mlAlbedoRoughnessImage) {
        m_mlAlbedoRoughnessImage->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }
    if (m_mlSpecularMotionImage) {
        m_mlSpecularMotionImage->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
        m_mlDiffuseImage->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
        m_mlSpecularImage->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }

    vkEndCommandBuffer(m_commandBuffers[0]);

    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = m_commandBuffers[0];

    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
    vkQueueSubmit2(m_context->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_context->getGraphicsQueue());
}

void Engine::uploadToDeviceBuffer(Buffer& dstBuffer, const void* srcData, VkDeviceSize dataSize) {
    if (dataSize == 0 || !srcData) return;
    VkDevice device = m_context->getDevice();
    VkQueue queue = m_context->getGraphicsQueue();
    VmaAllocator allocator = m_context->getAllocator();

    const VkDeviceSize maxChunkSize = 64 * 1024 * 1024; // 64 MB bounded staging buffer
    VkDeviceSize stagingSize = std::min(dataSize, maxChunkSize);

    Buffer stagingBuffer(
        allocator, stagingSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    VkDeviceSize offset = 0;
    while (offset < dataSize) {
        VkDeviceSize currentChunk = std::min(maxChunkSize, dataSize - offset);
        std::memcpy(stagingBuffer.map(), static_cast<const uint8_t*>(srcData) + offset, currentChunk);
        stagingBuffer.flush(0, currentChunk);

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = offset;
        copyRegion.size = currentChunk;
        vkCmdCopyBuffer(cmd, stagingBuffer.getBuffer(), dstBuffer.getBuffer(), 1, &copyRegion);

        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = cmd;
        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        offset += currentChunk;
    }

    vkFreeCommandBuffers(device, m_commandPool, 1, &cmd);
}

void Engine::uploadIndexBuffer(Buffer& dstBuffer, uint32_t triangleCount) {
    if (triangleCount == 0) {
        uint32_t dummy[3] = { 0, 1, 2 };
        uploadToDeviceBuffer(dstBuffer, dummy, sizeof(dummy));
        return;
    }

    VkDevice device = m_context->getDevice();
    VkQueue queue = m_context->getGraphicsQueue();
    VmaAllocator allocator = m_context->getAllocator();

    const uint32_t chunkTriangles = 1048576; // 1M triangles = 12 MB chunk
    VkDeviceSize stagingSize = std::min(static_cast<VkDeviceSize>(triangleCount), static_cast<VkDeviceSize>(chunkTriangles)) * 3 * sizeof(uint32_t);

    Buffer stagingBuffer(
        allocator, stagingSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    uint32_t* mappedStaging = static_cast<uint32_t*>(stagingBuffer.map());

    uint32_t triOffset = 0;
    while (triOffset < triangleCount) {
        uint32_t currentChunkTriangles = std::min(chunkTriangles, triangleCount - triOffset);
        for (uint32_t i = 0; i < currentChunkTriangles; ++i) {
            uint32_t k = triOffset + i;
            mappedStaging[i * 3 + 0] = 3 * k + 0;
            mappedStaging[i * 3 + 1] = 3 * k + 1;
            mappedStaging[i * 3 + 2] = 3 * k + 2;
        }
        VkDeviceSize currentBytes = currentChunkTriangles * 3 * sizeof(uint32_t);
        stagingBuffer.flush(0, currentBytes);

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = static_cast<VkDeviceSize>(triOffset) * 3 * sizeof(uint32_t);
        copyRegion.size = currentBytes;
        vkCmdCopyBuffer(cmd, stagingBuffer.getBuffer(), dstBuffer.getBuffer(), 1, &copyRegion);

        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = cmd;
        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        triOffset += currentChunkTriangles;
    }

    vkFreeCommandBuffers(device, m_commandPool, 1, &cmd);
}

void Engine::createAccelerationStructures() {
    if (!m_context->hasRayTracing()) {
        throw std::runtime_error("Hardware Ray Tracing is required under Vulkan 1.4 baseline but is not available.");
    }

    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();

    m_tlas.reset();
    m_blas.reset();
    m_blases.clear();
    m_asIndexBuffer.reset();
    m_instanceBuffer.reset();
    m_asManager.reset();

    uint32_t numTriangles = static_cast<uint32_t>(m_numTriangles);
    VkDeviceSize indexBufferSize = std::max(static_cast<VkDeviceSize>(sizeof(uint32_t) * 3 * numTriangles), static_cast<VkDeviceSize>(sizeof(uint32_t) * 3));
    m_asIndexBuffer = std::make_unique<Buffer>(
        allocator, indexBufferSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    uploadIndexBuffer(*m_asIndexBuffer, numTriangles);

    m_asManager = std::make_unique<AccelerationStructureManager>(
        device, allocator,
        m_context->getGraphicsQueue(), m_context->getGraphicsQueueFamily()
    );

    // 1. Instance Buffer (std430, binding 30)
    std::vector<InstanceGPU> instanceUpload;
    if (!m_sceneData.instanceData.empty()) {
        instanceUpload = m_sceneData.instanceData;
    } else {
        InstanceGPU defaultInst{};
        defaultInst.firstTriangle = 0;
        defaultInst.numOpaqueTriangles = m_numOpaqueTriangles;
        defaultInst.materialOffset = 0;
        defaultInst.flags = 0;
        instanceUpload.push_back(defaultInst);
    }

    VkDeviceSize instanceBufferSize = sizeof(InstanceGPU) * instanceUpload.size();
    m_instanceBuffer = std::make_unique<Buffer>(
        allocator, instanceBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    uploadToDeviceBuffer(*m_instanceBuffer, instanceUpload.data(), instanceBufferSize);

    // 2. Acceleration Structures
    VkDeviceAddress vertexBaseAddr = m_positionBuffer ? m_positionBuffer->getDeviceAddress(device) : 0;
    VkDeviceAddress indexBaseAddr = m_asIndexBuffer->getDeviceAddress(device);

    if (!m_sceneData.blasRanges.empty()) {
        // Multi-BLAS path
        m_asManager->resetStats();
        for (const auto& range : m_sceneData.blasRanges) {
            std::vector<ASGeometryInput> geoms;
            if (range.numOpaqueTriangles > 0) {
                ASGeometryInput geomOpaque{};
                geomOpaque.vertexBufferAddress = vertexBaseAddr;
                geomOpaque.indexBufferAddress = indexBaseAddr + static_cast<VkDeviceSize>(range.firstTriangle) * 3 * sizeof(uint32_t);
                geomOpaque.vertexCount = 3 * (range.firstTriangle + range.numOpaqueTriangles);
                geomOpaque.triangleCount = range.numOpaqueTriangles;
                geomOpaque.vertexStride = sizeof(glm::vec4);
                geomOpaque.indexType = VK_INDEX_TYPE_UINT32;
                geomOpaque.isOpaque = true;
                geoms.push_back(geomOpaque);
            }
            uint32_t numNonOpaque = (range.triangleCount > range.numOpaqueTriangles) ? (range.triangleCount - range.numOpaqueTriangles) : 0;
            if (numNonOpaque > 0) {
                ASGeometryInput geomNonOpaque{};
                geomNonOpaque.vertexBufferAddress = vertexBaseAddr;
                geomNonOpaque.indexBufferAddress = indexBaseAddr + static_cast<VkDeviceSize>(range.firstTriangle + range.numOpaqueTriangles) * 3 * sizeof(uint32_t);
                geomNonOpaque.vertexCount = 3 * (range.firstTriangle + range.triangleCount);
                geomNonOpaque.triangleCount = numNonOpaque;
                geomNonOpaque.vertexStride = sizeof(glm::vec4);
                geomNonOpaque.indexType = VK_INDEX_TYPE_UINT32;
                geomNonOpaque.isOpaque = false;
                geoms.push_back(geomNonOpaque);
            }
            if (geoms.empty()) {
                ASGeometryInput dummyGeom{};
                dummyGeom.vertexBufferAddress = vertexBaseAddr;
                dummyGeom.indexBufferAddress = indexBaseAddr;
                dummyGeom.vertexCount = 3;
                dummyGeom.triangleCount = 1;
                dummyGeom.vertexStride = sizeof(glm::vec4);
                dummyGeom.indexType = VK_INDEX_TYPE_UINT32;
                dummyGeom.isOpaque = true;
                geoms.push_back(dummyGeom);
            }
            m_blases.push_back(m_asManager->buildBLAS(geoms));
        }

        std::vector<ASInstanceInput> asInstances;
        asInstances.reserve(m_sceneData.instances.size());
        for (const auto& inst : m_sceneData.instances) {
            ASInstanceInput asInst{};
            uint32_t bIdx = std::min(inst.blasIndex, static_cast<uint32_t>(m_blases.size() - 1));
            asInst.blasAddress = m_blases[bIdx]->getDeviceAddress();
            asInst.transform = inst.transform;
            asInst.customIndex = inst.customIndex;
            asInst.mask = 0xFF;
            asInst.hitGroupId = 0;
            asInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            asInstances.push_back(asInst);
        }
        m_tlas = m_asManager->buildTLAS(asInstances);
        if (!m_tlas) {
            throw std::runtime_error("Hardware Ray Tracing Multi-BLAS TLAS build failed.");
        }
        initTlasBuffers(static_cast<uint32_t>(asInstances.size()));
        Logger::info("Hardware Ray Tracing Multi-BLAS Acceleration Structures initialized successfully ({} BLASes, {} TLAS Instances).",
                     m_blases.size(), asInstances.size());
    } else {
        // Monolithic single-BLAS path
        std::vector<ASGeometryInput> geoms;
        if (m_numOpaqueTriangles > 0) {
            ASGeometryInput geomOpaque{};
            geomOpaque.vertexBufferAddress = vertexBaseAddr;
            geomOpaque.indexBufferAddress = indexBaseAddr;
            geomOpaque.vertexCount = 3 * m_numOpaqueTriangles;
            geomOpaque.triangleCount = m_numOpaqueTriangles;
            geomOpaque.vertexStride = sizeof(glm::vec4);
            geomOpaque.indexType = VK_INDEX_TYPE_UINT32;
            geomOpaque.isOpaque = true;
            geoms.push_back(geomOpaque);
        }

        uint32_t numNonOpaque = numTriangles - m_numOpaqueTriangles;
        if (numNonOpaque > 0) {
            ASGeometryInput geomNonOpaque{};
            geomNonOpaque.vertexBufferAddress = vertexBaseAddr;
            geomNonOpaque.indexBufferAddress = indexBaseAddr + static_cast<VkDeviceSize>(m_numOpaqueTriangles) * 3 * sizeof(uint32_t);
            geomNonOpaque.vertexCount = 3 * numTriangles;
            geomNonOpaque.triangleCount = numNonOpaque;
            geomNonOpaque.vertexStride = sizeof(glm::vec4);
            geomNonOpaque.indexType = VK_INDEX_TYPE_UINT32;
            geomNonOpaque.isOpaque = false;
            geoms.push_back(geomNonOpaque);
        }

        if (geoms.empty()) {
            ASGeometryInput dummyGeom{};
            dummyGeom.vertexBufferAddress = vertexBaseAddr;
            dummyGeom.indexBufferAddress = indexBaseAddr;
            dummyGeom.vertexCount = 3;
            dummyGeom.triangleCount = 1;
            dummyGeom.vertexStride = sizeof(glm::vec4);
            dummyGeom.indexType = VK_INDEX_TYPE_UINT32;
            dummyGeom.isOpaque = true;
            geoms.push_back(dummyGeom);
        }

        m_blas = m_asManager->buildBLAS(geoms);

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
        initTlasBuffers(1);
        Logger::info("Hardware Ray Tracing Acceleration Structures initialized successfully (Monolithic BLAS & TLAS).");
    }
    Logger::info("Hardware Ray Tracing Pipeline Active. Extensions in use: VK_KHR_ray_query, VK_KHR_acceleration_structure, VK_KHR_buffer_device_address, VK_KHR_deferred_host_operations (SPIR-V: GL_EXT_ray_query)");
}

void Engine::initScene() {
    VmaAllocator allocator = m_context->getAllocator();

    // Discover scenes directory with standard Linux FHS fallbacks
    std::filesystem::path scenesDir = "scenes";
    if (!std::filesystem::exists(scenesDir) || !std::filesystem::is_directory(scenesDir)) {
        std::filesystem::path exeDir;
#ifdef _WIN32
        char exePathBuf[MAX_PATH] = {0};
        if (GetModuleFileNameA(NULL, exePathBuf, MAX_PATH)) {
            exeDir = std::filesystem::path(exePathBuf).parent_path();
        }
#elif defined(__linux__) || defined(__unix__)
        std::error_code ec;
        auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
        if (!ec && !p.empty()) {
            exeDir = p.parent_path();
        } else {
            p = std::filesystem::canonical("/proc/self/exe", ec);
            if (!ec) exeDir = p.parent_path();
        }
#endif
        if (!exeDir.empty()) {
            auto adjacentScenes = exeDir / "scenes";
            auto parentScenes = exeDir / ".." / "scenes";
            auto relShare = exeDir / ".." / "share" / "pathways" / "scenes";
            auto devScenes = exeDir / ".." / ".." / "scenes";
            if (std::filesystem::exists(adjacentScenes) && std::filesystem::is_directory(adjacentScenes)) {
                scenesDir = adjacentScenes;
            } else if (std::filesystem::exists(parentScenes) && std::filesystem::is_directory(parentScenes)) {
                scenesDir = parentScenes;
            } else if (std::filesystem::exists(relShare) && std::filesystem::is_directory(relShare)) {
                scenesDir = relShare;
            } else if (std::filesystem::exists(devScenes) && std::filesystem::is_directory(devScenes)) {
                scenesDir = devScenes;
            }
        }
        if ((!std::filesystem::exists(scenesDir) || !std::filesystem::is_directory(scenesDir)) &&
            std::filesystem::exists("/usr/share/pathways/scenes")) {
            scenesDir = "/usr/share/pathways/scenes";
        }
    }

    if (std::filesystem::exists(scenesDir) && std::filesystem::is_directory(scenesDir)) {
        m_availableScenes = SceneRegistry::scan(scenesDir.string());
    } else {
        // Only procedural scenes if no scenes directory exists
        m_availableScenes = SceneRegistry::scan("");
    }
    m_currentSceneIndex = 0;

    if (!m_config.scene_path.empty()) {
        if (m_config.scene_path == "many-lights" || m_config.scene_path == "many_lights" || m_config.scene_path == "procedural:many-lights") {
            Logger::info("Loading Procedural Many-Lights Cornell Box (64 Lights)...");
            m_sceneData = ProceduralScene::createManyLightsScene();
            m_currentSceneIndex = 1;
        } else if (m_config.scene_path == "cyber-city" || m_config.scene_path == "cyber_city" || m_config.scene_path == "procedural:cyber-city" || m_config.scene_path == "procedural:cyber_city" || m_config.scene_path == "Procedural Cyber City") {
            Logger::info("Loading Procedural Cyber City Megastructure...");
            m_sceneData = ProceduralScene::createCyberCityScene();
            m_currentSceneIndex = -1;
            for (size_t i = 0; i < m_availableScenes.size(); ++i) {
                if (m_availableScenes[i].filepath == "procedural:cyber-city") {
                    m_currentSceneIndex = static_cast<int>(i);
                    break;
                }
            }
        } else if (m_config.scene_path == "infinity-mirror" || m_config.scene_path == "procedural:infinity-mirror" || m_config.scene_path == "infinity_mirror" || m_config.scene_path == "procedural:infinity_mirror" || m_config.scene_path == "Infinity Mirror" || m_config.scene_path == "Procedural Infinity Mirror") {
            Logger::info("Loading Infinity Mirror Corridor...");
            m_sceneData = ProceduralScene::createInfinityMirrorScene();
            m_currentSceneIndex = -1;
            for (size_t i = 0; i < m_availableScenes.size(); ++i) {
                if (m_availableScenes[i].filepath == "infinity-mirror" || m_availableScenes[i].filepath == "procedural:infinity-mirror") {
                    m_currentSceneIndex = static_cast<int>(i);
                    break;
                }
            }
        } else {
            std::string resolvedScene = m_config.scene_path;
            if (!std::filesystem::exists(resolvedScene)) {
                auto candidate = scenesDir / resolvedScene;
                if (std::filesystem::exists(candidate)) {
                    resolvedScene = candidate.string();
                } else if (resolvedScene.rfind("scenes/", 0) == 0) {
                    auto subCandidate = scenesDir / resolvedScene.substr(7);
                    if (std::filesystem::exists(subCandidate)) {
                        resolvedScene = subCandidate.string();
                    }
                } else {
                    // Try looking under scenesDir/<stem>/<filename>
                    std::string stem = std::filesystem::path(resolvedScene).stem().string();
                    auto nested = scenesDir / stem / resolvedScene;
                    if (std::filesystem::exists(nested)) {
                        resolvedScene = nested.string();
                    }
                }
            }

            // Map legacy ClassicCar aliases to BuickRiviera if ClassicCar does not exist as a valid scene
            if (!std::filesystem::exists(resolvedScene) || (std::filesystem::is_regular_file(resolvedScene) && std::filesystem::file_size(resolvedScene) < 100)) {
                if (resolvedScene.find("ClassicCar") != std::string::npos || resolvedScene.find("classic_car") != std::string::npos) {
                    auto buickCandidate = scenesDir / "BuickRiviera" / "BuickRiviera.usdc";
                    if (std::filesystem::exists(buickCandidate)) {
                        resolvedScene = buickCandidate.string();
                    }
                }
            }

            // If resolvedScene is a directory, find the primary scene file within it
            if (std::filesystem::exists(resolvedScene) && std::filesystem::is_directory(resolvedScene)) {
                std::filesystem::path dirPath(resolvedScene);
                std::string dirName = dirPath.filename().string();
                std::string underscoreDir = dirName;
                std::replace(underscoreDir.begin(), underscoreDir.end(), '-', '_');

                std::vector<std::filesystem::path> candidates = {
                    dirPath / (dirName + ".usd"),
                    dirPath / (underscoreDir + ".usd"),
                    dirPath / (dirName + "_extended.glb"),
                    dirPath / (underscoreDir + "_extended.glb"),
                    dirPath / (dirName + ".glb"),
                    dirPath / (underscoreDir + ".glb"),
                    dirPath / (dirName + ".usda"),
                    dirPath / (dirName + ".usdc")
                };
                for (const auto& c : candidates) {
                    if (std::filesystem::exists(c)) {
                        resolvedScene = c.string();
                        break;
                    }
                }
            }

            Logger::info("Loading user specified scene: {}", resolvedScene);
            if (UsdLoader::isUsdFile(resolvedScene)) {
                UsdLoadOptions options{};
                options.instanceDensity = m_config.instance_density;
                options.cullDistance = m_config.cull_distance;
                options.cameraPosOverride = m_config.camera_pos;
                options.viewportAspect = (m_config.height > 0) ? (static_cast<float>(m_config.width) / static_cast<float>(m_config.height)) : (16.0f / 9.0f);
                m_sceneData = UsdLoader::loadSceneData(resolvedScene, options);
            } else {
                m_sceneData = GltfLoader::loadSceneData(resolvedScene);
            }
            m_currentSceneIndex = -1;
            for (size_t i = 0; i < m_availableScenes.size(); ++i) {
                std::error_code ec;
                if (m_availableScenes[i].filepath == resolvedScene ||
                    (!m_availableScenes[i].filepath.empty() &&
                     std::filesystem::exists(m_availableScenes[i].filepath) &&
                     std::filesystem::exists(resolvedScene) &&
                     std::filesystem::equivalent(m_availableScenes[i].filepath, resolvedScene, ec))) {
                    m_currentSceneIndex = static_cast<int>(i);
                    break;
                }
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

    uint64_t totalInstTris = 0;
    if (!m_sceneData.instances.empty() && !m_sceneData.blasRanges.empty()) {
        for (const auto& inst : m_sceneData.instances) {
            if (inst.blasIndex < m_sceneData.blasRanges.size()) {
                totalInstTris += m_sceneData.blasRanges[inst.blasIndex].triangleCount;
            }
        }
        m_numInstances = static_cast<uint32_t>(m_sceneData.instances.size());
    } else {
        totalInstTris = m_numTriangles;
        m_numInstances = 1;
    }
    m_numInstancedTriangles = totalInstTris;

    updateSceneTransparencyFlag();
    partitionSceneGeometry();

    if (m_numInstancedTriangles > m_numTriangles) {
        Logger::info("Active Scene: {} Base Triangles ({} Instanced across {} Instances), {} Spheres, {} Materials, {} Lights (Non-Opaque: {})",
                     m_numTriangles, m_numInstancedTriangles, m_numInstances, m_numSpheres, m_numMaterials, m_numLights, m_sceneHasNonOpaque ? "YES" : "NO");
    } else {
        Logger::info("Active Scene: {} Triangles, {} Spheres, {} Materials, {} Lights (Non-Opaque: {})",
                     m_numTriangles, m_numSpheres, m_numMaterials, m_numLights, m_sceneHasNonOpaque ? "YES" : "NO");
    }

    if (m_camera) {
        m_camera->setSceneScale(m_sceneData.sceneRadius, m_sceneData.focalDistance, m_sceneData.centralTarget);

        glm::vec3 camPos = m_sceneData.cameraPosition;
        glm::vec3 camTarget = m_sceneData.cameraTarget;
        glm::vec3 camUp = m_sceneData.cameraUp;
        float camFov = m_sceneData.cameraFov;
        bool hasOverride = false;

        if (m_config.camera_pos.has_value()) {
            camPos = *m_config.camera_pos;
            hasOverride = true;
        }
        if (m_config.camera_target.has_value()) {
            camTarget = *m_config.camera_target;
            hasOverride = true;
        }
        if (m_config.camera_up.has_value()) {
            camUp = *m_config.camera_up;
            hasOverride = true;
        }
        if (m_config.camera_fov.has_value()) {
            camFov = *m_config.camera_fov;
            hasOverride = true;
        }

        if (m_sceneData.hasCamera || hasOverride) {
            m_camera->lookAt(camPos, camTarget, camUp);
            m_camera->setFov(camFov);
            if (m_config.camera_fov.has_value()) {
                m_camera->setAdaptiveFov(false);
            }
            m_camera->setDefaultFraming(camPos, camTarget, camFov);
            Logger::info("Active Camera: pos=({:.3f}, {:.3f}, {:.3f}), target=({:.3f}, {:.3f}, {:.3f}), up=({:.3f}, {:.3f}, {:.3f}), fov={:.1f}°{}",
                         camPos.x, camPos.y, camPos.z,
                         camTarget.x, camTarget.y, camTarget.z,
                         camUp.x, camUp.y, camUp.z,
                         camFov,
                         hasOverride ? " [CLI Override]" : "");
        }
    }

    // 1. Position buffer (pure DEVICE_LOCAL VRAM for BLAS building, 16-byte aligned)
    size_t numTris = m_sceneData.triangles.size();
    std::vector<glm::vec4> positions;
    positions.reserve(numTris * 3);
    std::vector<TriangleShadeGPU> shadeTriangles;
    shadeTriangles.reserve(numTris);

    for (const auto& tri : m_sceneData.triangles) {
        positions.push_back(glm::vec4(glm::vec3(tri.v0.position), 1.0f));
        positions.push_back(glm::vec4(glm::vec3(tri.v1.position), 1.0f));
        positions.push_back(glm::vec4(glm::vec3(tri.v2.position), 1.0f));

        TriangleShadeGPU s{};
        s.normal0_u0 = glm::vec4(glm::vec3(tri.v0.normal), tri.v0.position.w);
        s.normal1_u1 = glm::vec4(glm::vec3(tri.v1.normal), tri.v1.position.w);
        s.normal2_u2 = glm::vec4(glm::vec3(tri.v2.normal), tri.v2.position.w);
        s.tan0_v0    = glm::vec4(glm::vec3(tri.v0.tangent), tri.v0.normal.w);
        s.tan1_v1    = glm::vec4(glm::vec3(tri.v1.tangent), tri.v1.normal.w);
        s.tan2_v2    = glm::vec4(glm::vec3(tri.v2.tangent), tri.v2.normal.w);
        s.tanSigns   = glm::vec4(tri.v0.tangent.w, tri.v1.tangent.w, tri.v2.tangent.w, 0.0f);
        s.materialId = tri.materialId;
        s.padding[0] = 0;
        s.padding[1] = 0;
        s.padding[2] = 0;
        shadeTriangles.push_back(s);
    }

    VkDeviceSize posSize = std::max(sizeof(glm::vec4) * positions.size(), sizeof(glm::vec4) * 3);
    m_positionBuffer = std::make_unique<Buffer>(
        allocator, posSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (!positions.empty()) {
        uploadToDeviceBuffer(*m_positionBuffer, positions.data(), sizeof(glm::vec4) * positions.size());
    }

    // 2. Triangle shading buffer (128-byte cache-line aligned for ray hit resolution)
    VkDeviceSize triSize = std::max(sizeof(TriangleShadeGPU) * shadeTriangles.size(), sizeof(TriangleShadeGPU));
    m_triangleBuffer = std::make_unique<Buffer>(
        allocator, triSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (!shadeTriangles.empty()) {
        uploadToDeviceBuffer(*m_triangleBuffer, shadeTriangles.data(), sizeof(TriangleShadeGPU) * shadeTriangles.size());
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

    // Material Archetype buffer (compact 4-byte lookup table for L0/L1 cache efficiency)
    std::vector<uint32_t> matArchetypes(m_sceneData.materials.size());
    for (size_t i = 0; i < m_sceneData.materials.size(); ++i) {
        matArchetypes[i] = computeMaterialArchetype(m_sceneData.materials[i]);
    }
    VkDeviceSize archSize = std::max(sizeof(uint32_t) * matArchetypes.size(), sizeof(uint32_t));
    m_materialArchetypeBuffer = std::make_unique<Buffer>(
        allocator, archSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!matArchetypes.empty()) {
        m_materialArchetypeBuffer->copyFrom(matArchetypes.data(), sizeof(uint32_t) * matArchetypes.size());
    }

    // Compact 64-byte Shading Material buffer (2 materials per 128B RDNA 4 vector cache line)
    std::vector<ShadeMaterialGPU> shadeMaterials(m_sceneData.materials.size());
    for (size_t i = 0; i < m_sceneData.materials.size(); ++i) {
        shadeMaterials[i] = createShadeMaterial(m_sceneData.materials[i]);
    }
    VkDeviceSize shadeMatSize = std::max(sizeof(ShadeMaterialGPU) * shadeMaterials.size(), sizeof(ShadeMaterialGPU));
    m_shadeMaterialBuffer = std::make_unique<Buffer>(
        allocator, shadeMatSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!shadeMaterials.empty()) {
        m_shadeMaterialBuffer->copyFrom(shadeMaterials.data(), sizeof(ShadeMaterialGPU) * shadeMaterials.size());
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
        buildLightAliasTable(m_sceneData.lights);
        buildLightTree(m_sceneData.lights, m_sceneData.lightTreeNodes);
        m_lightBuffer->copyFrom(m_sceneData.lights.data(), sizeof(LightGPU) * m_sceneData.lights.size());
    }

    // Light tree buffer (FEAT-02)
    VkDeviceSize lightTreeSize = std::max(sizeof(LightTreeNodeGPU) * m_sceneData.lightTreeNodes.size(), sizeof(LightTreeNodeGPU));
    m_lightTreeBuffer = std::make_unique<Buffer>(
        allocator, lightTreeSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.lightTreeNodes.empty()) {
        m_lightTreeBuffer->copyFrom(m_sceneData.lightTreeNodes.data(), sizeof(LightTreeNodeGPU) * m_sceneData.lightTreeNodes.size());
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

    // Center pixel G-buffer depth readback buffers (Double-buffered, 16 bytes for 1 RGBA16F texel)
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_centerDepthBuffers[i] = std::make_unique<Buffer>(
            allocator, 16,
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_GPU_TO_CPU,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
        );
    }

    // Hardware Acceleration Structures (VK_KHR_ray_query)
    createAccelerationStructures();


    // Textures & HDRI Environment Map Initialization
    VkDevice device = m_context->getDevice();
    VkQueue queue = m_context->getGraphicsQueue();
    VkCommandPool pool = m_commandPool;

    m_dummyWhite = Texture::createDummyWhite(device, allocator, queue, pool);
    m_dummyNormal = Texture::createDummyNormal(device, allocator, queue, pool);
    m_blueNoiseTexture = Texture::createBlueNoise64(device, allocator, queue, pool);

    m_environmentMap = Texture::createSceneEnvironmentMap(
        device, allocator, queue, pool,
        m_config.hdri_path, m_config.scene_path, m_sceneData.domeLightHdriPath
    );

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
    initVideoBillboardDecoder(m_config.scene_path);
}

void Engine::requestSceneChange(const std::string& filepath) {
    if (m_isSceneLoading.load()) {
        Logger::warn("Scene loading already in progress; ignoring request for '{}'", filepath);
        return;
    }
    m_loadingScenePath = filepath;

    std::string prettyName;
    for (const auto& sc : m_availableScenes) {
        if (sc.filepath == filepath) {
            prettyName = sc.label;
            break;
        }
    }
    if (prettyName.empty()) {
        std::string stem = std::filesystem::path(filepath).stem().string();
        if (stem.empty() || filepath == "__procedural_cornell_box__") {
            prettyName = "Procedural Cornell Box";
        } else if (filepath == "procedural:many-lights" || filepath == "many-lights" || filepath == "many_lights") {
            prettyName = "Procedural Many-Lights";
        } else if (filepath == "procedural:cyber-city" || filepath == "procedural:cyber_city" || filepath == "cyber-city" || filepath == "cyber_city" || filepath == "Procedural Cyber City") {
            prettyName = "Cyber City";
        } else if (filepath == "procedural:infinity-mirror" || filepath == "procedural:infinity_mirror" || filepath == "infinity-mirror" || filepath == "infinity_mirror" || filepath == "Procedural Infinity Mirror" || filepath == "Infinity Mirror") {
            prettyName = "Infinity Mirror";
        } else {
            prettyName = SceneRegistry::formatSceneName(stem);
        }
    }
    m_loadingSceneName = prettyName;
    m_sceneLoadingStartTime = std::chrono::steady_clock::now();
    m_isSceneLoading.store(true);
    Logger::info("Initiating asynchronous scene load for '{}' ({})...", filepath, m_loadingSceneName);

    UsdLoadOptions usdOptions{};
    usdOptions.instanceDensity = m_config.instance_density;
    usdOptions.cullDistance = m_config.cull_distance;
    usdOptions.cameraPosOverride = m_config.camera_pos;
    usdOptions.viewportAspect = (m_config.height > 0) ? (static_cast<float>(m_config.width) / static_cast<float>(m_config.height)) : (16.0f / 9.0f);

    m_sceneLoadingFuture = std::async(std::launch::async, [filepath, usdOptions]() -> SceneData {
        if (filepath.empty() || filepath == "__procedural_cornell_box__") {
            Logger::info("Dynamic Scene Switch: Loading Procedural Cornell Box...");
            return ProceduralScene::createCornellBox();
        } else if (filepath == "procedural:many-lights" || filepath == "many-lights" || filepath == "many_lights") {
            Logger::info("Dynamic Scene Switch: Loading Procedural Many-Lights Cornell Box (64 Lights)...");
            return ProceduralScene::createManyLightsScene();
        } else if (filepath == "procedural:cyber-city" || filepath == "procedural:cyber_city" || filepath == "cyber-city" || filepath == "cyber_city" || filepath == "Procedural Cyber City") {
            Logger::info("Dynamic Scene Switch: Loading Procedural Cyber City Megastructure...");
            return ProceduralScene::createCyberCityScene();
        } else if (filepath == "procedural:infinity-mirror" || filepath == "procedural:infinity_mirror" || filepath == "infinity-mirror" || filepath == "infinity_mirror" || filepath == "Procedural Infinity Mirror" || filepath == "Infinity Mirror") {
            Logger::info("Dynamic Scene Switch: Loading Infinity Mirror Corridor...");
            return ProceduralScene::createInfinityMirrorScene();
        } else {
            Logger::info("Dynamic Scene Switch: Loading '{}'...", filepath);
            if (UsdLoader::isUsdFile(filepath)) {
                return UsdLoader::loadSceneData(filepath, usdOptions);
            } else {
                return GltfLoader::loadSceneData(filepath);
            }
        }
    });
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
        m_config.hdri_path = "";
        std::filesystem::path sp(filepath);
        std::filesystem::path sceneDir = sp.has_parent_path() ? sp.parent_path() : std::filesystem::current_path();
        std::vector<std::filesystem::path> hdriCandidates = {
            sceneDir / "textures and hdri" / "golden_gate_hills_2k.exr",
            sceneDir / "textures" / "studio_small_08_4k.exr",
            sceneDir / "textures" / "studio_small_08_4k.hdr",
            sceneDir / ".." / "textures" / "studio_small_08_4k.exr",
            sceneDir / "textures and hdri" / "golden_gate_hills_2k.hdr",
        };
        for (const auto& cand : hdriCandidates) {
            std::error_code ec;
            if (std::filesystem::exists(cand, ec)) {
                m_config.hdri_path = cand.string();
                Logger::info("Engine: Auto-detected scene HDRI environment map: '{}'", m_config.hdri_path);
                break;
            }
        }
    }
    if (m_mgpu) {
        m_mgpu->setConfig(m_config);
    }

    m_sceneData = std::move(newScene);
    m_numTriangles = static_cast<uint32_t>(m_sceneData.triangles.size());
    m_numSpheres = static_cast<uint32_t>(m_sceneData.spheres.size());
    m_numMaterials = static_cast<uint32_t>(m_sceneData.materials.size());
    m_numLights = static_cast<uint32_t>(m_sceneData.lights.size());

    uint64_t totalInstTris = 0;
    if (!m_sceneData.instances.empty() && !m_sceneData.blasRanges.empty()) {
        for (const auto& inst : m_sceneData.instances) {
            if (inst.blasIndex < m_sceneData.blasRanges.size()) {
                totalInstTris += m_sceneData.blasRanges[inst.blasIndex].triangleCount;
            }
        }
        m_numInstances = static_cast<uint32_t>(m_sceneData.instances.size());
    } else {
        totalInstTris = m_numTriangles;
        m_numInstances = 1;
    }
    m_numInstancedTriangles = totalInstTris;

    updateSceneTransparencyFlag();
    partitionSceneGeometry();

    if (m_numInstancedTriangles > m_numTriangles) {
        Logger::info("Active Scene: {} Base Triangles ({} Instanced across {} Instances), {} Spheres, {} Materials, {} Lights (Non-Opaque: {})",
                     m_numTriangles, m_numInstancedTriangles, m_numInstances, m_numSpheres, m_numMaterials, m_numLights, m_sceneHasNonOpaque ? "YES" : "NO");
    } else {
        Logger::info("Active Scene: {} Triangles, {} Spheres, {} Materials, {} Lights (Non-Opaque: {})",
                     m_numTriangles, m_numSpheres, m_numMaterials, m_numLights, m_sceneHasNonOpaque ? "YES" : "NO");
    }

    // Recreate primary buffers
    VmaAllocator allocator = m_context->getAllocator();

    // 1. Position buffer (pure DEVICE_LOCAL VRAM for BLAS building, 16-byte aligned)
    size_t numTris = m_sceneData.triangles.size();
    std::vector<glm::vec4> positions;
    positions.reserve(numTris * 3);
    std::vector<TriangleShadeGPU> shadeTriangles;
    shadeTriangles.reserve(numTris);

    for (const auto& tri : m_sceneData.triangles) {
        positions.push_back(glm::vec4(glm::vec3(tri.v0.position), 1.0f));
        positions.push_back(glm::vec4(glm::vec3(tri.v1.position), 1.0f));
        positions.push_back(glm::vec4(glm::vec3(tri.v2.position), 1.0f));

        TriangleShadeGPU s{};
        s.normal0_u0 = glm::vec4(glm::vec3(tri.v0.normal), tri.v0.position.w);
        s.normal1_u1 = glm::vec4(glm::vec3(tri.v1.normal), tri.v1.position.w);
        s.normal2_u2 = glm::vec4(glm::vec3(tri.v2.normal), tri.v2.position.w);
        s.tan0_v0    = glm::vec4(glm::vec3(tri.v0.tangent), tri.v0.normal.w);
        s.tan1_v1    = glm::vec4(glm::vec3(tri.v1.tangent), tri.v1.normal.w);
        s.tan2_v2    = glm::vec4(glm::vec3(tri.v2.tangent), tri.v2.normal.w);
        s.tanSigns   = glm::vec4(tri.v0.tangent.w, tri.v1.tangent.w, tri.v2.tangent.w, 0.0f);
        s.materialId = tri.materialId;
        s.padding[0] = 0;
        s.padding[1] = 0;
        s.padding[2] = 0;
        shadeTriangles.push_back(s);
    }

    VkDeviceSize posSize = std::max(sizeof(glm::vec4) * positions.size(), sizeof(glm::vec4) * 3);
    m_positionBuffer = std::make_unique<Buffer>(
        allocator, posSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (!positions.empty()) {
        uploadToDeviceBuffer(*m_positionBuffer, positions.data(), sizeof(glm::vec4) * positions.size());
    }

    // 2. Triangle shading buffer (128-byte cache-line aligned for ray hit resolution)
    VkDeviceSize triSize = std::max(sizeof(TriangleShadeGPU) * shadeTriangles.size(), sizeof(TriangleShadeGPU));
    m_triangleBuffer = std::make_unique<Buffer>(
        allocator, triSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (!shadeTriangles.empty()) {
        uploadToDeviceBuffer(*m_triangleBuffer, shadeTriangles.data(), sizeof(TriangleShadeGPU) * shadeTriangles.size());
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

    // Material Archetype buffer (compact 4-byte lookup table for L0/L1 cache efficiency)
    std::vector<uint32_t> matArchetypes(m_sceneData.materials.size());
    for (size_t i = 0; i < m_sceneData.materials.size(); ++i) {
        matArchetypes[i] = computeMaterialArchetype(m_sceneData.materials[i]);
    }
    VkDeviceSize archSize = std::max(sizeof(uint32_t) * matArchetypes.size(), sizeof(uint32_t));
    m_materialArchetypeBuffer = std::make_unique<Buffer>(
        allocator, archSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!matArchetypes.empty()) {
        m_materialArchetypeBuffer->copyFrom(matArchetypes.data(), sizeof(uint32_t) * matArchetypes.size());
    }

    // Compact 64-byte Shading Material buffer (2 materials per 128B RDNA 4 vector cache line)
    std::vector<ShadeMaterialGPU> shadeMaterials(m_sceneData.materials.size());
    for (size_t i = 0; i < m_sceneData.materials.size(); ++i) {
        shadeMaterials[i] = createShadeMaterial(m_sceneData.materials[i]);
    }
    VkDeviceSize shadeMatSize = std::max(sizeof(ShadeMaterialGPU) * shadeMaterials.size(), sizeof(ShadeMaterialGPU));
    m_shadeMaterialBuffer = std::make_unique<Buffer>(
        allocator, shadeMatSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!shadeMaterials.empty()) {
        m_shadeMaterialBuffer->copyFrom(shadeMaterials.data(), sizeof(ShadeMaterialGPU) * shadeMaterials.size());
    }

    VkDeviceSize lightSize = std::max(sizeof(LightGPU) * m_sceneData.lights.size(), sizeof(LightGPU));
    m_lightBuffer = std::make_unique<Buffer>(
        allocator, lightSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.lights.empty()) {
        buildLightAliasTable(m_sceneData.lights);
        buildLightTree(m_sceneData.lights, m_sceneData.lightTreeNodes);
        m_lightBuffer->copyFrom(m_sceneData.lights.data(), sizeof(LightGPU) * m_sceneData.lights.size());
    }

    // Light tree buffer (FEAT-02)
    VkDeviceSize lightTreeSize = std::max(sizeof(LightTreeNodeGPU) * m_sceneData.lightTreeNodes.size(), sizeof(LightTreeNodeGPU));
    m_lightTreeBuffer = std::make_unique<Buffer>(
        allocator, lightTreeSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.lightTreeNodes.empty()) {
        m_lightTreeBuffer->copyFrom(m_sceneData.lightTreeNodes.data(), sizeof(LightTreeNodeGPU) * m_sceneData.lightTreeNodes.size());
    }

    // Rebuild Acceleration Structures (BLAS & TLAS)
    createAccelerationStructures();


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

    // Unified HDRI sky dome synchronization across scenes
    m_environmentMap = Texture::createSceneEnvironmentMap(
        device, allocator, queue, pool,
        m_config.hdri_path, filepath, m_sceneData.domeLightHdriPath
    );

    // Update primary descriptor sets
    updateSceneDescriptors();

    // Secondary GPU reload
    if (m_mgpu && m_mgpu->isSecondaryInitialized()) {
        m_mgpu->loadScene(m_sceneData, filepath);
    }

    // Update camera framing
    if (m_camera) {
        m_camera->setSceneScale(m_sceneData.sceneRadius, m_sceneData.focalDistance, m_sceneData.centralTarget);

        glm::vec3 camPos = m_sceneData.cameraPosition;
        glm::vec3 camTarget = m_sceneData.cameraTarget;
        glm::vec3 camUp = m_sceneData.cameraUp;
        float camFov = m_sceneData.cameraFov;
        bool hasOverride = false;

        if (m_config.camera_pos.has_value()) {
            camPos = *m_config.camera_pos;
            hasOverride = true;
        }
        if (m_config.camera_target.has_value()) {
            camTarget = *m_config.camera_target;
            hasOverride = true;
        }
        if (m_config.camera_up.has_value()) {
            camUp = *m_config.camera_up;
            hasOverride = true;
        }
        if (m_config.camera_fov.has_value()) {
            camFov = *m_config.camera_fov;
            hasOverride = true;
        }

        m_camera->lookAt(camPos, camTarget, camUp);
        m_camera->setFov(camFov);
        if (m_config.camera_fov.has_value()) {
            m_camera->setAdaptiveFov(false);
        }
        m_camera->setDefaultFraming(camPos, camTarget, camFov);
        Logger::info("Active Camera: pos=({:.3f}, {:.3f}, {:.3f}), target=({:.3f}, {:.3f}, {:.3f}), up=({:.3f}, {:.3f}, {:.3f}), fov={:.1f}°{}",
                     camPos.x, camPos.y, camPos.z,
                     camTarget.x, camTarget.y, camTarget.z,
                     camUp.x, camUp.y, camUp.z,
                     camFov,
                     hasOverride ? " [CLI Override]" : "");
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

    // Reclaim host memory used for scene geometry ingestion (now safely resident in device VRAM)
    m_sceneData.triangles.clear();
    m_sceneData.triangles.shrink_to_fit();

    initVideoBillboardDecoder(filepath);

    Logger::info("Scene successfully switched to: {} (Index: {})", filepath, m_currentSceneIndex);
    return true;
}

bool Engine::loadScene(const std::string& filepath) {
    m_dynamicWavefrontBounces = m_config.max_bounces;
    SceneData newScene;
    if (filepath.empty() || filepath == "__procedural_cornell_box__") {
        Logger::info("Loading Procedural Cornell Box...");
        newScene = ProceduralScene::createCornellBox();
    } else if (filepath == "procedural:many-lights" || filepath == "many-lights" || filepath == "many_lights") {
        Logger::info("Loading Procedural Many-Lights Cornell Box (64 Lights)...");
        newScene = ProceduralScene::createManyLightsScene();
    } else if (filepath == "procedural:cyber-city" || filepath == "procedural:cyber_city" || filepath == "cyber-city" || filepath == "cyber_city" || filepath == "Procedural Cyber City") {
        Logger::info("Loading Procedural Cyber City Megastructure...");
        newScene = ProceduralScene::createCyberCityScene();
    } else if (filepath == "procedural:infinity-mirror" || filepath == "infinity-mirror" || filepath == "procedural:infinity_mirror" || filepath == "infinity_mirror" || filepath == "Procedural Infinity Mirror" || filepath == "Infinity Mirror") {
        Logger::info("Loading Infinity Mirror Corridor...");
        newScene = ProceduralScene::createInfinityMirrorScene();
    } else {
        if (UsdLoader::isUsdFile(filepath)) {
            Logger::info("Loading OpenUSD scene '{}'...", filepath);
            UsdLoadOptions options{};
            options.instanceDensity = m_config.instance_density;
            options.cullDistance = m_config.cull_distance;
            options.cameraPosOverride = m_config.camera_pos;
            options.viewportAspect = (m_config.height > 0) ? (static_cast<float>(m_config.width) / static_cast<float>(m_config.height)) : (16.0f / 9.0f);
            newScene = UsdLoader::loadSceneData(filepath, options);
        } else {
            Logger::info("Loading glTF scene '{}'...", filepath);
            newScene = GltfLoader::loadSceneData(filepath);
        }
    }
    return applyLoadedScene(std::move(newScene), filepath);
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

void Engine::partitionSceneGeometry() {
    if (m_sceneData.triangles.empty()) {
        m_numOpaqueTriangles = 0;
        m_sceneData.numOpaqueTriangles = 0;
        return;
    }

    auto isOpaqueTriangle = [this](const TriangleGPU& tri) {
        if (tri.materialId >= m_sceneData.materials.size()) {
            return true;
        }
        const auto& mat = m_sceneData.materials[tri.materialId];
        return (mat.alphaMode == ALPHA_MODE_OPAQUE && mat.transmission <= 0.05f && mat.type != MATERIAL_DIELECTRIC);
    };

    if (!m_sceneData.blasRanges.empty()) {
        uint32_t totalOpaque = 0;
        for (auto& range : m_sceneData.blasRanges) {
            if (range.firstTriangle + range.triangleCount <= m_sceneData.triangles.size()) {
                auto rangeBegin = m_sceneData.triangles.begin() + range.firstTriangle;
                auto rangeEnd = rangeBegin + range.triangleCount;
                auto it = std::stable_partition(rangeBegin, rangeEnd, isOpaqueTriangle);
                range.numOpaqueTriangles = static_cast<uint32_t>(std::distance(rangeBegin, it));
                totalOpaque += range.numOpaqueTriangles;
            }
        }
        for (size_t i = 0; i < m_sceneData.instances.size(); ++i) {
            uint32_t bIdx = m_sceneData.instances[i].blasIndex;
            if (bIdx < m_sceneData.blasRanges.size() && i < m_sceneData.instanceData.size()) {
                m_sceneData.instanceData[i].firstTriangle = m_sceneData.blasRanges[bIdx].firstTriangle;
                m_sceneData.instanceData[i].numOpaqueTriangles = m_sceneData.blasRanges[bIdx].numOpaqueTriangles;
            }
        }
        m_numOpaqueTriangles = totalOpaque;
        m_sceneData.numOpaqueTriangles = totalOpaque;
    } else {
        auto it = std::stable_partition(m_sceneData.triangles.begin(), m_sceneData.triangles.end(), isOpaqueTriangle);
        m_numOpaqueTriangles = static_cast<uint32_t>(std::distance(m_sceneData.triangles.begin(), it));
        m_sceneData.numOpaqueTriangles = m_numOpaqueTriangles;
    }

    Logger::info("Geometry Partitioning: {} Opaque triangles, {} Non-Opaque triangles (Total: {})",
                 m_numOpaqueTriangles, m_sceneData.triangles.size() - m_numOpaqueTriangles, m_sceneData.triangles.size());
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
    if (m_config.batch_pixels > 0) {
        return m_config.batch_pixels;
    }
    if (m_config.batch_count > 0) {
        uint32_t totalPixels = m_config.width * m_config.height;
        return (totalPixels + m_config.batch_count - 1) / m_config.batch_count;
    }

    // Programmatic hardware profile detection
    VkPhysicalDeviceType devType = m_context->getDeviceProperties().deviceType;
    if (devType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        // Profile A: UMA / APU (e.g. AMD Strix Halo, Phoenix)
        // High-end APUs like Strix Halo feature large unified memory (up to 128GB LPDDR5X) and 40 CUs.
        // A 2.0M pixel budget (~360 MB per queue slot) maintains high CU occupancy across multi-bounce GI
        // while cutting queue memory footprint by >73% from monolithic 4K.
        return 2000000u;
    }

    // Query device-local VRAM budget
    VkDeviceSize totalDeviceVram = 0;
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_context->getPhysicalDevice(), &memProps);
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            totalDeviceVram += memProps.memoryHeaps[i].size;
        }
    }

    if (totalDeviceVram <= 8ULL * 1024 * 1024 * 1024) {
        // Profile B: Mid dGPU (<= 8GB VRAM)
        return 1500000u; // 1.5M pixels (~246 MB queue set)
    }

    if (totalDeviceVram < 16ULL * 1024 * 1024 * 1024) {
        // Profile C: High-mid dGPU (8GB - 16GB VRAM)
        return 2000000u; // 2.0M pixels (~330 MB queue set)
    }

    // Profile D: Enthusiast / Workstation dGPU (>= 16GB VRAM, e.g. Dual Radeon AI PRO R9700)
    // Sizing queues for the entire 4K viewport (~1.3 GB) fits comfortably in VRAM, eliminating
    // inter-batch pipeline flushes, duplicate raygen dispatches, and wave fragmentation.
    return 0u; // Monolithic full-frame dispatch
}

uint32_t Engine::getEffectiveBatchCount(uint32_t renderW, uint32_t renderH) const {
    if (m_config.batch_count > 0) {
        return m_config.batch_count;
    }
    uint32_t totalPixels = renderW * renderH;
    uint32_t targetBatch = getTargetBatchPixels();
    if (targetBatch == 0 || totalPixels <= targetBatch) {
        return 1u; // Monolithic short-circuit (e.g. 1080p)
    }
    uint32_t count = (totalPixels + targetBatch - 1) / targetBatch;
    // Cap auto batch count to at most 4 when max_bounces > 2 to prevent secondary ray CU starvation
    if (m_config.max_bounces > 2 && count > 4u) {
        count = 4u;
    }
    return count;
}

uint32_t Engine::getEffectiveBatchPixels(uint32_t renderW, uint32_t renderH, uint32_t batchCount) const {
    if (batchCount <= 1) return renderW * renderH;
    float screenAspect = static_cast<float>(renderW) / static_cast<float>(renderH);
    float bestMetric = 1e9f;
    uint32_t bestNx = 1, bestNy = 1;
    for (uint32_t nx = 1; nx <= batchCount; ++nx) {
        if (batchCount % nx == 0) {
            uint32_t ny = batchCount / nx;
            float tileAspect = screenAspect * (static_cast<float>(ny) / static_cast<float>(nx));
            float metric = std::abs(std::log(tileAspect));
            if (metric < bestMetric) {
                bestMetric = metric;
                bestNx = nx;
                bestNy = ny;
            }
        }
    }
    uint32_t maxW = (renderW + bestNx - 1) / bestNx;
    uint32_t maxH = (renderH + bestNy - 1) / bestNy;
    return maxW * maxH;
}

void Engine::initPipelines() {
    VkDevice device = m_context->getDevice();

    // 1. Descriptor Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 256 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 64 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 32 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 256;
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
        { 11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, rtStages, nullptr },
        { 14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 15, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr }
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

    std::vector<VkDescriptorImageInfo> frameImageInfos(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        frameImageInfos[i].imageView = m_frameImages[i]->getImageView();
        frameImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    std::vector<VkDescriptorBufferInfo> uboBufferInfos(MAX_FRAMES_IN_FLIGHT);
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        uboBufferInfos[i] = { m_cameraUBOs[i]->getBuffer(), 0, sizeof(CameraUniform) };
    }

    std::vector<VkWriteDescriptorSet> writes;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &frameImageInfos[i], nullptr, nullptr });
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
    rtpPushConstant.size = sizeof(uint32_t) * 16;

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
        rgenCode, rmissCode, shadowMissCode, rchitCode,
        m_context->hasRtSubgroupSizeControl()
    );
    Logger::info("Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline) created successfully.");

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

    uint32_t initBatchCount = getEffectiveBatchCount(m_config.width, m_config.height);
    uint32_t initBatchPixels = getEffectiveBatchPixels(m_config.width, m_config.height, initBatchCount);
    if (m_config.mgpu_mode != MultiGpuMode::Off) {
        initBatchCount = 1;
        initBatchPixels = m_config.width * m_config.height;
    }
    m_currentBatchCount = initBatchCount;
    m_currentBatchPixels = initBatchPixels;

    m_wavefrontPipeline = std::make_unique<WavefrontPipeline>(
        device, allocator,
        m_config.width, m_config.height,
        wfClassifyCode, wfIntersectCode, wfShadeCode, wfShadowCode,
        wfShadeDiffuseCode, wfShadeDielectricCode, wfShadeConductorCode, wfShadeComplexCode,
        wfShadeEmissiveCode, wfShadePassthroughCode,
        m_context->hasDgcExecutionSet(),
        wfShadeDiffuseSecCode, wfShadeComplexSecCode,
        true, // enableDgcPreprocess
        m_context->hasSubgroupSizeControl(),
        initBatchPixels
    );
    Logger::info("Wavefront Path Tracing Pipeline (Ray Queues & DGC) initialized successfully.");

    // 6c. Neural Radiance Caching (NRC) Manager (Wave32 WMMA)
    try {
        auto nrcInferCode = loadShaderSPIRV("nrc_encode_infer.comp.spv");
        auto nrcTrainCode = loadShaderSPIRV("nrc_train.comp.spv");
        auto nrcResolveCode = loadShaderSPIRV("nrc_resolve.comp.spv");
        m_nrcManager = std::make_unique<NRCManager>(
            device, allocator,
            m_config.width, m_config.height,
            nrcInferCode, nrcTrainCode, nrcResolveCode
        );
        Logger::info("Neural Radiance Caching Subsystem (Wave32 WMMA & Atomic Buffer) initialized successfully.");
    } catch (const std::exception& e) {
        if (m_config.enable_nrc) {
            throw std::runtime_error(std::string("NRC was explicitly requested (--nrc) but initialization failed: ") + e.what());
        }
        Logger::warn("NRCManager initialization failed: {}", e.what());
    }

    // 6d. Ultra-Lean ReSTIR DI Subsystem
    createReSTIRResources();

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
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
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
    mergePushConstant.size = sizeof(uint32_t) * 8; // width, height, secondarySpp, tileSize, formatMode, mergeMode, primarySpp, pad

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

    // 7b. Running Average Accumulation Pipeline (FP16 -> FP32)
    createAccumRunningAvgPipeline();

    // 7c. Fused Accumulation & Tonemapping Pipeline (Pass Fusion)
    createAccumTonemapPipeline();

    // 8. G-Buffer Resources (Direct Light & Surface Normals/Depth)
    createGBufferResources();

    // 9. Post-Processing Descriptors (Upscalers & Tonemapping)
    createPostProcessDescPool();

    // 10a. Upways Neural Denoiser & Super-Resolution (Wave32 WMMA)
    createUpwaysPipelines();
    createUpwaysResources();

    // 10c. AMD FidelityFX Super Resolution 3.1
    createFsr3Pipelines();
    createFsr3Resources();

    // 11. GPU TLAS Instance Writer & Refit Pipeline (Tier 3)
    initTlasUpdatePipeline();

    // 12. Real-Time Caustics (Forward Photon Injection + Atomic Splatting + Cross-Bilateral Filtering)
    createCausticsPipelines();
    createCausticsResources();

    if (m_wavefrontPipeline) {
        m_wavefrontPipeline->setPostClassifyCallback([this](VkCommandBuffer cmd, uint32_t frameSlot) {
            if (m_config.enable_caustics && m_sceneData.hasDielectrics && m_numLights > 0) {
                dispatchCausticSplatAndFilter(cmd, frameSlot);
            }
            if (m_config.enable_restir_di && m_restirManager) {
                uint32_t rw = m_config.width;
                uint32_t rh = m_config.height;
                Buffer* rayGeom = m_wavefrontPipeline->getRayGeomQueue(frameSlot);
                Buffer* rayHit = m_wavefrontPipeline->getRayHitQueue(frameSlot);
                Buffer* pixelToRay = m_wavefrontPipeline->getPixelToRayQueue(frameSlot);
                Buffer* camUBO = m_cameraUBOs[frameSlot].get();
                Buffer* lightsBuf = m_lightBuffer.get();
                Buffer* matsBuf = m_materialBuffer.get();
                Buffer* ltBuf = m_lightTreeBuffer.get();
                VkImageView mvView = m_motionVectorImage ? m_motionVectorImage->getImageView() : VK_NULL_HANDLE;
                VkImageView ndView = m_normalDepthImage ? m_normalDepthImage->getImageView() : VK_NULL_HANDLE;
                VkImageView prevNdView = m_prevNormalDepthImage ? m_prevNormalDepthImage->getImageView() : ndView;

                VkImageView confView = (m_upwaysPipeline && m_upwaysPipeline->getConfidenceImage())
                    ? m_upwaysPipeline->getConfidenceImage()->getImageView()
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
    if (!m_accumImage || !m_outputImage) return;
    VkDevice device = m_context->getDevice();
    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageView = m_accumImage->getImageView();
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageView = m_outputImage->getImageView();
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo directLightInfo{};
    if (m_directLightImage) {
        directLightInfo.imageView = m_directLightImage->getImageView();
        directLightInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkDescriptorImageInfo normDepthInfo{};
    if (m_normalDepthImage) {
        normDepthInfo.imageView = m_normalDepthImage->getImageView();
        normDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }
    VkDescriptorImageInfo mvImageInfo{};
    if (m_motionVectorImage) {
        mvImageInfo.imageView = m_motionVectorImage->getImageView();
        mvImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    VkDescriptorImageInfo causticInfo{};
    causticInfo.imageView = m_filteredCausticImage ? m_filteredCausticImage->getImageView() : accumImageInfo.imageView;
    causticInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkDescriptorImageInfo, MAX_FRAMES_IN_FLIGHT> frameImageInfos;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_frameImages[i]) {
            frameImageInfos[i].sampler = VK_NULL_HANDLE;
            frameImageInfos[i].imageView = m_frameImages[i]->getImageView();
            frameImageInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }
    }

    std::vector<VkWriteDescriptorSet> writes;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_rtDescSets[i] != VK_NULL_HANDLE) {
            VkWriteDescriptorSet w0{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
            w0.dstSet = m_rtDescSets[i];
            w0.dstBinding = 0;
            w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            w0.descriptorCount = 1;
            w0.pImageInfo = &frameImageInfos[i];
            writes.push_back(w0);

            if (m_directLightImage && m_normalDepthImage) {
                VkWriteDescriptorSet w11{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                w11.dstSet = m_rtDescSets[i];
                w11.dstBinding = 11;
                w11.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w11.descriptorCount = 1;
                w11.pImageInfo = &directLightInfo;
                writes.push_back(w11);

                VkWriteDescriptorSet w12{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                w12.dstSet = m_rtDescSets[i];
                w12.dstBinding = 12;
                w12.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w12.descriptorCount = 1;
                w12.pImageInfo = &normDepthInfo;
                writes.push_back(w12);
            }

            if (m_motionVectorImage) {
                VkWriteDescriptorSet w14{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                w14.dstSet = m_rtDescSets[i];
                w14.dstBinding = 14;
                w14.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w14.descriptorCount = 1;
                w14.pImageInfo = &mvImageInfo;
                writes.push_back(w14);
            }

            if (m_filteredCausticImage) {
                VkWriteDescriptorSet w15{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
                w15.dstSet = m_rtDescSets[i];
                w15.dstBinding = 15;
                w15.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                w15.descriptorCount = 1;
                w15.pImageInfo = &causticInfo;
                writes.push_back(w15);
            }
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

    updateWavefrontSceneDescriptors();
    updateUpwaysDescriptors();
    updateFsr3Descriptors();
    updateCausticsDescriptors();
    updateAccumRunningAvgDescriptors();
    updateAccumTonemapDescriptors();
}

void Engine::createAccumRunningAvgPipeline() {
    VkDevice device = m_context->getDevice();

    // 1. Descriptor Set Layout
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_accumRunningAvgDescLayout);

    // 2. Allocate Descriptor Sets (one per frame in flight)
    std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts = {
        m_accumRunningAvgDescLayout, m_accumRunningAvgDescLayout
    };
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(device, &allocInfo, m_accumRunningAvgDescSets.data());

    updateAccumRunningAvgDescriptors();

    // 3. Pipeline Layout with Push Constants
    VkPushConstantRange pushConstant{};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(uint32_t) * 3 + sizeof(float); // width, height, sampleCount, invSpp (16 bytes)

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts = &m_accumRunningAvgDescLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges = &pushConstant;
    vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &m_accumRunningAvgPipelineLayout);

    // 4. Compute Pipeline
    auto shaderCode = loadShaderSPIRV("accum_running_avg.comp.spv");
    VkShaderModule shaderModule = createShaderModule(shaderCode);

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    if (m_context->hasSubgroupSizeControl()) {
        pipelineInfo.stage.pNext = &subgroupSize32;
    }
    pipelineInfo.layout = m_accumRunningAvgPipelineLayout;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_accumRunningAvgPipeline);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    Logger::info("Running average accumulation compute pipeline (FP16 -> FP32) created successfully.");
}

void Engine::updateAccumRunningAvgDescriptors() {
    if (!m_accumImage) return;
    VkDevice device = m_context->getDevice();

    VkDescriptorImageInfo historyInfo{};
    historyInfo.imageView = m_accumImage->getImageView();
    historyInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkDescriptorImageInfo, MAX_FRAMES_IN_FLIGHT> frameInfos;
    std::vector<VkWriteDescriptorSet> writes;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (!m_frameImages[i] || m_accumRunningAvgDescSets[i] == VK_NULL_HANDLE) continue;

        frameInfos[i].sampler = VK_NULL_HANDLE;
        frameInfos[i].imageView = m_frameImages[i]->getImageView();
        frameInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Binding 0: readonly uCurrentFrame
        VkWriteDescriptorSet w0{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w0.dstSet = m_accumRunningAvgDescSets[i];
        w0.dstBinding = 0;
        w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w0.descriptorCount = 1;
        w0.pImageInfo = &frameInfos[i];
        writes.push_back(w0);

        // Binding 1: uHistoryAccum
        VkWriteDescriptorSet w1{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w1.dstSet = m_accumRunningAvgDescSets[i];
        w1.dstBinding = 1;
        w1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1.descriptorCount = 1;
        w1.pImageInfo = &historyInfo;
        writes.push_back(w1);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void Engine::createAccumTonemapPipeline() {
    VkDevice device = m_context->getDevice();

    // 1. Descriptor Set Layout (3 bindings: frameIn, historyInOut, outputImage)
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_accumTonemapDescLayout);

    // 2. Allocate Descriptor Sets (one per frame in flight)
    std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts = {
        m_accumTonemapDescLayout, m_accumTonemapDescLayout
    };
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();
    vkAllocateDescriptorSets(device, &allocInfo, m_accumTonemapDescSets.data());

    updateAccumTonemapDescriptors();

    // 3. Pipeline Layout with Push Constants (40 bytes)
    VkPushConstantRange pushConstant{};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(uint32_t) * 3 + sizeof(float) * 2 + sizeof(uint32_t) * 2 + sizeof(float) * 2 + sizeof(uint32_t); // 40 bytes

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts = &m_accumTonemapDescLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges = &pushConstant;
    vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &m_accumTonemapPipelineLayout);

    // 4. Compute Pipeline (Wave32)
    auto shaderCode = loadShaderSPIRV("accum_tonemap_fused.comp.spv");
    VkShaderModule shaderModule = createShaderModule(shaderCode);

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    VkComputePipelineCreateInfo pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = shaderModule;
    pipelineInfo.stage.pName = "main";
    if (m_context->hasSubgroupSizeControl()) {
        pipelineInfo.stage.pNext = &subgroupSize32;
    }
    pipelineInfo.layout = m_accumTonemapPipelineLayout;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_accumTonemapPipeline);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    Logger::info("Fused Accumulation & ACES Tonemapping compute pipeline (Wave32) created successfully.");
}

void Engine::updateAccumTonemapDescriptors() {
    if (!m_accumImage || !m_outputImage) return;
    VkDevice device = m_context->getDevice();

    VkDescriptorImageInfo historyInfo{};
    historyInfo.imageView = m_accumImage->getImageView();
    historyInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo outputInfo{};
    outputInfo.imageView = m_outputImage->getImageView();
    outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkDescriptorImageInfo, MAX_FRAMES_IN_FLIGHT> frameInfos;
    std::vector<VkWriteDescriptorSet> writes;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (!m_frameImages[i] || m_accumTonemapDescSets[i] == VK_NULL_HANDLE) continue;

        frameInfos[i].sampler = VK_NULL_HANDLE;
        frameInfos[i].imageView = m_frameImages[i]->getImageView();
        frameInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        // Binding 0: readonly uCurrentFrame
        VkWriteDescriptorSet w0{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w0.dstSet = m_accumTonemapDescSets[i];
        w0.dstBinding = 0;
        w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w0.descriptorCount = 1;
        w0.pImageInfo = &frameInfos[i];
        writes.push_back(w0);

        // Binding 1: uHistoryAccum
        VkWriteDescriptorSet w1{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w1.dstSet = m_accumTonemapDescSets[i];
        w1.dstBinding = 1;
        w1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w1.descriptorCount = 1;
        w1.pImageInfo = &historyInfo;
        writes.push_back(w1);

        // Binding 2: writeonly uOutputImage
        VkWriteDescriptorSet w2{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w2.dstSet = m_accumTonemapDescSets[i];
        w2.dstBinding = 2;
        w2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w2.descriptorCount = 1;
        w2.pImageInfo = &outputInfo;
        writes.push_back(w2);
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void Engine::createGBufferResources() {
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();
    uint32_t w = m_config.width;
    uint32_t h = m_config.height;

    // Allocate Image Resources
    m_directLightImage = std::make_unique<Image>(device, allocator, w, h,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);

    m_normalDepthImage = std::make_unique<Image>(device, allocator, w, h,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    m_prevNormalDepthImage = std::make_unique<Image>(device, allocator, w, h,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    // Transition images to GENERAL layout
    VkCommandBuffer cmd = m_commandBuffers[0];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &beginInfo);
    m_directLightImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    m_normalDepthImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    m_prevNormalDepthImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkEndCommandBuffer(cmd);
    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = cmd;

    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
    vkQueueSubmit2(m_context->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_context->getGraphicsQueue());

    updateMergeDescriptors();
    Logger::info("G-Buffer resources allocated successfully.");
}

void Engine::destroyGBufferResources() {
    m_directLightImage.reset();
    m_normalDepthImage.reset();
    m_prevNormalDepthImage.reset();
}

void Engine::createPostProcessDescPool() {
    VkDevice device = m_context->getDevice();
    if (m_postProcessDescPool != VK_NULL_HANDLE) {
        return;
    }
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32 }
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 8;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_postProcessDescPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create post-processing descriptor pool");
    }
}

void Engine::destroyPostProcessDescPool() {
    VkDevice device = m_context ? m_context->getDevice() : VK_NULL_HANDLE;
    if (!device) return;

    if (m_postProcessDescPool) {
        vkDestroyDescriptorPool(device, m_postProcessDescPool, nullptr);
        m_postProcessDescPool = VK_NULL_HANDLE;
    }
    m_tonemapUpwaysDescSet = VK_NULL_HANDLE;
    m_tonemapFsr3DescSet = VK_NULL_HANDLE;
}

void Engine::createUpwaysPipelines() {
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();

    try {
        auto upwaysCode = loadShaderSPIRV("neural_reconstruct.comp.spv");
        if (upwaysCode.empty()) {
            upwaysCode = loadShaderSPIRV("upways_reconstruct.comp.spv");
        }
        VkFormat format = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
        bool isSuperRes = m_config.upways_superres || (m_config.upscaler_mode == UpscalerMode::Upways);
        uint32_t inW = (m_config.render_scale < 1.0f && (m_config.upscaler_mode != UpscalerMode::None || m_config.upways_superres)) ?
            static_cast<uint32_t>(m_config.width * m_config.render_scale) : m_config.width;
        uint32_t inH = (m_config.render_scale < 1.0f && (m_config.upscaler_mode != UpscalerMode::None || m_config.upways_superres)) ?
            static_cast<uint32_t>(m_config.height * m_config.render_scale) : m_config.height;
        uint32_t outW = m_config.width;
        uint32_t outH = m_config.height;
        isSuperRes = (inW < outW || inH < outH || isSuperRes);

        m_upwaysPipeline = std::make_unique<UpwaysPipeline>(
            device,
            m_context->getPhysicalDevice(),
            allocator,
            inW,
            inH,
            outW,
            outH,
            upwaysCode,
            m_config.upways_weights_path,
            isSuperRes,
            format
        );
        Logger::info("Upways Neural Reconstruction Pipeline initialized successfully.");
    } catch (const std::exception& e) {
        if (m_config.upscaler_mode == UpscalerMode::Upways || m_config.upways_superres || m_config.denoiser_mode == DenoiserMode::Upways) {
            throw std::runtime_error(std::string("Upways was explicitly requested but initialization failed: ") + e.what());
        }
        Logger::warn("UpwaysPipeline initialization failed: {}", e.what());
    }
}

void Engine::destroyUpwaysPipelines() {
    m_upwaysPipeline.reset();
    m_tonemapUpwaysDescSet = VK_NULL_HANDLE;
}

void Engine::createUpwaysResources() {
    VkDevice device = m_context->getDevice();

    if (m_postProcessDescPool && m_tonemapDescLayout && m_tonemapUpwaysDescSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo tmAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        tmAllocInfo.descriptorPool = m_postProcessDescPool;
        tmAllocInfo.descriptorSetCount = 1;
        tmAllocInfo.pSetLayouts = &m_tonemapDescLayout;
        if (vkAllocateDescriptorSets(device, &tmAllocInfo, &m_tonemapUpwaysDescSet) != VK_SUCCESS) {
            Logger::warn("Failed to allocate Tonemap Upways descriptor set");
            m_tonemapUpwaysDescSet = VK_NULL_HANDLE;
        }
    }

    if (m_upwaysPipeline) {
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(m_commandBuffers[0], &beginInfo);

        m_upwaysPipeline->transitionInitialLayouts(m_commandBuffers[0]);

        vkEndCommandBuffer(m_commandBuffers[0]);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = m_commandBuffers[0];

        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(m_context->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_context->getGraphicsQueue());
    }

    updateUpwaysDescriptors();
}

void Engine::destroyUpwaysResources() {
    // Note: Descriptor set m_tonemapUpwaysDescSet persists across resizes; reset occurs in destroyUpwaysPipelines().
}

void Engine::updateUpwaysDescriptors() {
    if (!m_upwaysPipeline || !m_accumImage || !m_outputImage) {
        return;
    }

    VkImageView normDepthView = m_normalDepthImage ? m_normalDepthImage->getImageView() : VK_NULL_HANDLE;
    VkImageView motionView = m_motionVectorImage ? m_motionVectorImage->getImageView() : VK_NULL_HANDLE;
    VkImageView albedoView = (m_config.pipeline_type == PipelineType::Wavefront && m_mlAlbedoRoughnessImage)
        ? m_mlAlbedoRoughnessImage->getImageView()
        : (m_directLightImage ? m_directLightImage->getImageView() : (m_mlAlbedoRoughnessImage ? m_mlAlbedoRoughnessImage->getImageView() : VK_NULL_HANDLE));
    VkImageView specMotionView = m_mlSpecularMotionImage ? m_mlSpecularMotionImage->getImageView() : motionView;
    bool isMgpuActive = (m_mgpu && m_mgpu->isSecondaryInitialized() && m_config.mgpu_mode != MultiGpuMode::Off);
    VkImageView diffView = m_mlDiffuseImage ? m_mlDiffuseImage->getImageView() : m_frameImages[0]->getImageView();
    VkImageView specView = isMgpuActive ? VK_NULL_HANDLE : (m_mlSpecularImage ? m_mlSpecularImage->getImageView() : VK_NULL_HANDLE);

    m_upwaysPipeline->updateDescriptors(
        diffView,
        specView,
        normDepthView,
        albedoView,
        motionView,
        specMotionView,
        specMotionView,
        albedoView,
        normDepthView
    );

    if (m_tonemapUpwaysDescSet != VK_NULL_HANDLE && m_upwaysPipeline->getOutputImage()) {
        VkDevice device = m_context->getDevice();
        VkDescriptorImageInfo inInfo{ VK_NULL_HANDLE, m_upwaysPipeline->getOutputImage()->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo outInfo{ VK_NULL_HANDLE, m_outputImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

        std::vector<VkWriteDescriptorSet> writes;
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_tonemapUpwaysDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &inInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_tonemapUpwaysDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outInfo, nullptr, nullptr });
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

bool Engine::dispatchUpways(VkCommandBuffer cmd, bool resetHistory) {
    if ((m_config.denoiser_mode != DenoiserMode::Upways && m_config.upscaler_mode != UpscalerMode::Upways) || !m_upwaysPipeline) {
        return false;
    }

    bool isSuperRes = m_config.upways_superres || (m_config.upscaler_mode == UpscalerMode::Upways);
    uint32_t inW = (m_config.render_scale < 1.0f && (m_config.upscaler_mode != UpscalerMode::None || m_config.upways_superres)) ?
        static_cast<uint32_t>(m_config.width * m_config.render_scale) : m_config.width;
    uint32_t inH = (m_config.render_scale < 1.0f && (m_config.upscaler_mode != UpscalerMode::None || m_config.upways_superres)) ?
        static_cast<uint32_t>(m_config.height * m_config.render_scale) : m_config.height;
    uint32_t outW = m_config.width;
    uint32_t outH = m_config.height;
    isSuperRes = (inW < outW || inH < outH || isSuperRes);

    if (m_upwaysPipeline->getInputWidth() != inW || m_upwaysPipeline->getInputHeight() != inH ||
        m_upwaysPipeline->getOutputWidth() != outW || m_upwaysPipeline->getOutputHeight() != outH ||
        m_upwaysPipeline->isSuperResEnabled() != isSuperRes) {
        vkDeviceWaitIdle(m_context->getDevice());
        m_upwaysPipeline->resize(inW, inH, outW, outH, isSuperRes);
        updateUpwaysDescriptors();
    }

    glm::mat4 currInvView(1.0f);
    glm::mat4 prevViewProj(1.0f);
    glm::mat4 invProj(1.0f);
    glm::mat4 prevView(1.0f);
    glm::vec2 jitter(0.0f);

    if (m_camera) {
        if (resetHistory) {
            m_camera->resetPrevViewProj();
        }
        bool enableJitter = (m_config.upscaler_mode == UpscalerMode::Upways || m_config.upways_superres);
        CameraUniform ubo = m_camera->getUniformData(
            m_frameIndex, 1, m_config.max_bounces, 0,
            enableJitter, inW, inH, 0, false
        );
        currInvView = ubo.viewInverse;
        prevViewProj = m_camera->getPrevViewProjMatrix();
        invProj = ubo.projInverse;
        prevView = m_camera->getPrevViewMatrix();
        jitter = glm::vec2(ubo.jitterOffset.x, ubo.jitterOffset.y);
    }

    uint32_t totalSamples = (m_config.progressive_accumulation && m_accumulatedSamples > 0) ? m_accumulatedSamples : 1u;

    m_upwaysPipeline->recordFrame(
        cmd, m_frameIndex, resetHistory, m_cameraMovedLastFrame,
        currInvView, prevViewProj, invProj, prevView, jitter, totalSamples
    );
    return true;
}


// === AMD FIDELITYFX SUPER RESOLUTION 3.1 ===

void Engine::createFsr3Pipelines() {
    if (m_config.upscaler_mode != UpscalerMode::FSR3) {
        return;
    }

    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();

    try {
        auto upscaleCode = loadShaderSPIRV("fsr3_upscale.comp.spv");
        auto rcasCode = loadShaderSPIRV("fsr3_rcas.comp.spv");
        VkFormat format = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
        uint32_t renderW = (m_config.render_scale < 1.0f) ?
            static_cast<uint32_t>(m_config.width * m_config.render_scale) :
            m_config.width;
        uint32_t renderH = (m_config.render_scale < 1.0f) ?
            static_cast<uint32_t>(m_config.height * m_config.render_scale) :
            m_config.height;

        m_fsr3Upscaler = std::make_unique<Fsr3Upscaler>(
            device,
            m_context->getPhysicalDevice(),
            allocator,
            renderW,
            renderH,
            m_config.width,
            m_config.height,
            upscaleCode,
            rcasCode,
            format,
            "[GPU 0 Primary Viewport]"
        );

        // 4K Blend Pipeline for SampleBlend multi-GPU resolve
        std::vector<VkDescriptorSetLayoutBinding> blendBindings = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
        };
        VkDescriptorSetLayoutCreateInfo blendLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        blendLayoutInfo.bindingCount = static_cast<uint32_t>(blendBindings.size());
        blendLayoutInfo.pBindings = blendBindings.data();
        if (vkCreateDescriptorSetLayout(device, &blendLayoutInfo, nullptr, &m_fsr3BlendDescLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create FSR3 blend descriptor set layout");
        }

        VkPushConstantRange blendPcRange{};
        blendPcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        blendPcRange.offset = 0;
        blendPcRange.size = sizeof(uint32_t) * 2 + sizeof(float) * 2;

        VkPipelineLayoutCreateInfo blendPipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        blendPipeLayoutInfo.setLayoutCount = 1;
        blendPipeLayoutInfo.pSetLayouts = &m_fsr3BlendDescLayout;
        blendPipeLayoutInfo.pushConstantRangeCount = 1;
        blendPipeLayoutInfo.pPushConstantRanges = &blendPcRange;
        if (vkCreatePipelineLayout(device, &blendPipeLayoutInfo, nullptr, &m_fsr3BlendPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create FSR3 blend pipeline layout");
        }

        auto blendCode = loadShaderSPIRV("fsr3_blend.comp.spv");
        VkShaderModule blendModule = createShaderModule(blendCode);

        VkComputePipelineCreateInfo blendPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        blendPipeInfo.layout = m_fsr3BlendPipelineLayout;
        blendPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        blendPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        blendPipeInfo.stage.module = blendModule;
        blendPipeInfo.stage.pName = "main";
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO };
        subgroupSize32.requiredSubgroupSize = 32;
        if (m_context->hasSubgroupSizeControl()) {
            blendPipeInfo.stage.pNext = &subgroupSize32;
        }

        if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &blendPipeInfo, nullptr, &m_fsr3BlendPipeline) != VK_SUCCESS) {
            vkDestroyShaderModule(device, blendModule, nullptr);
            throw std::runtime_error("Failed to create FSR3 blend compute pipeline");
        }
        vkDestroyShaderModule(device, blendModule, nullptr);

        VkDescriptorPoolSize blendPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 };
        VkDescriptorPoolCreateInfo blendPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
        blendPoolInfo.maxSets = 1;
        blendPoolInfo.poolSizeCount = 1;
        blendPoolInfo.pPoolSizes = &blendPoolSize;
        if (vkCreateDescriptorPool(device, &blendPoolInfo, nullptr, &m_fsr3BlendDescPool) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create FSR3 blend descriptor pool");
        }

        VkDescriptorSetAllocateInfo blendAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        blendAllocInfo.descriptorPool = m_fsr3BlendDescPool;
        blendAllocInfo.descriptorSetCount = 1;
        blendAllocInfo.pSetLayouts = &m_fsr3BlendDescLayout;
        if (vkAllocateDescriptorSets(device, &blendAllocInfo, &m_fsr3BlendDescSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate FSR3 blend descriptor set");
        }

        Logger::info("AMD FSR 3.1 Super-Resolution & SampleBlend Pipelines initialized successfully.");
    } catch (const std::exception& e) {
        Logger::warn("Fsr3Upscaler initialization failed: {}", e.what());
    }
}

void Engine::destroyFsr3Pipelines() {
    VkDevice device = m_context->getDevice();
    if (m_fsr3BlendPipeline) {
        vkDestroyPipeline(device, m_fsr3BlendPipeline, nullptr);
        m_fsr3BlendPipeline = VK_NULL_HANDLE;
    }
    if (m_fsr3BlendPipelineLayout) {
        vkDestroyPipelineLayout(device, m_fsr3BlendPipelineLayout, nullptr);
        m_fsr3BlendPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_fsr3BlendDescLayout) {
        vkDestroyDescriptorSetLayout(device, m_fsr3BlendDescLayout, nullptr);
        m_fsr3BlendDescLayout = VK_NULL_HANDLE;
    }
    if (m_fsr3BlendDescPool) {
        vkDestroyDescriptorPool(device, m_fsr3BlendDescPool, nullptr);
        m_fsr3BlendDescPool = VK_NULL_HANDLE;
    }
    m_fsr3BlendDescSet = VK_NULL_HANDLE;
    m_fsr3Upscaler.reset();
    m_tonemapFsr3DescSet = VK_NULL_HANDLE;
}

void Engine::createFsr3Resources() {
    if (m_config.upscaler_mode != UpscalerMode::FSR3) {
        return;
    }

    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();

    if (m_postProcessDescPool && m_tonemapDescLayout && m_tonemapFsr3DescSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo tmAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        tmAllocInfo.descriptorPool = m_postProcessDescPool;
        tmAllocInfo.descriptorSetCount = 1;
        tmAllocInfo.pSetLayouts = &m_tonemapDescLayout;
        if (vkAllocateDescriptorSets(device, &tmAllocInfo, &m_tonemapFsr3DescSet) != VK_SUCCESS) {
            Logger::warn("Failed to allocate Tonemap FSR3 descriptor set");
            m_tonemapFsr3DescSet = VK_NULL_HANDLE;
        }
    }

    VkFormat format = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;

    // m_secAccumImage holds the secondary GPU's transferred 4K upscaled frame in Approach 2 (Sample Parallelism)
    if (!m_secAccumImage || m_secAccumImage->getWidth() != m_config.width || m_secAccumImage->getHeight() != m_config.height) {
        m_secAccumImage = std::make_unique<Image>(
            device, allocator, m_config.width, m_config.height, format,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
        );
    }

    updateFsr3Descriptors();
}

void Engine::destroyFsr3Resources() {
    m_secAccumImage.reset();
}

void Engine::updateFsr3Descriptors() {
    if (!m_fsr3Upscaler || !m_outputImage) {
        return;
    }

    VkDevice device = m_context->getDevice();
    if (m_tonemapFsr3DescSet != VK_NULL_HANDLE && m_fsr3Upscaler->getOutputImage()) {
        VkDescriptorImageInfo inInfo{ VK_NULL_HANDLE, m_fsr3Upscaler->getOutputImage()->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo outInfo{ VK_NULL_HANDLE, m_outputImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

        std::vector<VkWriteDescriptorSet> writes;
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_tonemapFsr3DescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &inInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_tonemapFsr3DescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outInfo, nullptr, nullptr });
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    if (m_fsr3BlendDescSet != VK_NULL_HANDLE && m_fsr3Upscaler->getOutputImage() &&
        m_secAccumImage) {
        VkDescriptorImageInfo dstInfo{ VK_NULL_HANDLE, m_fsr3Upscaler->getOutputImage()->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo srcInfo{ VK_NULL_HANDLE, m_secAccumImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

        std::vector<VkWriteDescriptorSet> blendWrites;
        blendWrites.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_fsr3BlendDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dstInfo, nullptr, nullptr });
        blendWrites.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_fsr3BlendDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &srcInfo, nullptr, nullptr });
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(blendWrites.size()), blendWrites.data(), 0, nullptr);
    }
}

bool Engine::dispatchFsr3(VkCommandBuffer cmd, bool resetHistory) {
    if (m_config.upscaler_mode != UpscalerMode::FSR3) {
        return false;
    }
    if (!m_fsr3Upscaler) {
        createFsr3Pipelines();
        createFsr3Resources();
        updateFsr3Descriptors();
    }
    if (!m_fsr3Upscaler || !m_accumImage) {
        return false;
    }

    uint32_t renderW = (m_config.render_scale < 1.0f) ?
        static_cast<uint32_t>(m_config.width * m_config.render_scale) :
        m_config.width;
    uint32_t renderH = (m_config.render_scale < 1.0f) ?
        static_cast<uint32_t>(m_config.height * m_config.render_scale) :
        m_config.height;

    if (m_fsr3Upscaler->getRenderWidth() != renderW || m_fsr3Upscaler->getRenderHeight() != renderH ||
        m_fsr3Upscaler->getDisplayWidth() != m_config.width || m_fsr3Upscaler->getDisplayHeight() != m_config.height) {
        vkDeviceWaitIdle(m_context->getDevice());
        m_fsr3Upscaler->resize(renderW, renderH, m_config.width, m_config.height);
        updateFsr3Descriptors();
        updateMergeDescriptors();
    }

    VkBuffer secBuffer = VK_NULL_HANDLE;
    if (m_mgpu && m_mgpu->isZeroCopyActive()) {
        uint32_t slot = m_config.double_buffered_shared_mem ? (m_currentFrame % 2) : 0;
        secBuffer = m_mgpu->getPrimarySharedBuffer(slot);
    } else if (m_secTransferBuffer) {
        secBuffer = m_secTransferBuffer->getBuffer();
    }

    bool isSampleBlend = (m_mgpu && m_mgpu->isMultiGpuActive() && m_config.mgpu_mode != MultiGpuMode::Off &&
                          m_config.mgpu_upscale_mode == MgpuUpscaleMode::SampleBlend &&
                          m_secAccumImage && secBuffer != VK_NULL_HANDLE);

    if (isSampleBlend) {
        // Approach 2 (Sample Parallelism @ 1 SPP per GPU):
        // GPU 0 upscales its own internal frame to 4K.
        // GPU 1 upscales its own internal frame to 4K and sends it to GPU 0.
        // GPU 0 merges/averages the two 4K frames: L = (L_0 + L_1) / 2.

        // 1. Dispatch Primary GPU FSR 3.1 pass (1440p -> 4K on GPU 0)
        UpscalerDispatchDesc descPrim{};
        descPrim.cmd = cmd;
        descPrim.colorIn = m_accumImage->getImageView();
        descPrim.depthIn = m_normalDepthImage ? m_normalDepthImage->getImageView() : VK_NULL_HANDLE;
        descPrim.motionVectorsIn = m_motionVectorImage ? m_motionVectorImage->getImageView() : VK_NULL_HANDLE;
        descPrim.colorOut = m_fsr3Upscaler->getOutputImage()->getImageView();
        descPrim.renderWidth = renderW;
        descPrim.renderHeight = renderH;
        descPrim.displayWidth = m_config.width;
        descPrim.displayHeight = m_config.height;
        glm::vec2 jitter(0.0f);
        if (m_config.upscaler_mode == UpscalerMode::FSR3) {
            if (!m_config.progressive_accumulation || m_cameraMovedLastFrame || m_accumulatedSamples <= 1) {
                jitter = getHaltonJitter(m_frameIndex);
            }
        }
        descPrim.jitterX = jitter.x;
        descPrim.jitterY = jitter.y;
        descPrim.enableSharpening = m_config.upscaler_sharpening;
        descPrim.sharpness = m_config.upscaler_sharpening ? m_config.upscaler_sharpness : 0.0f;
        descPrim.resetHistory = resetHistory;
        descPrim.cameraMoved = m_cameraMovedLastFrame;
        descPrim.frameIndex = m_frameIndex;
        descPrim.inputIsNormalized = true;
        descPrim.totalSamples = (m_config.progressive_accumulation && m_accumulatedSamples > 0) ? m_accumulatedSamples : 1u;
        m_fsr3Upscaler->recordUpscale(descPrim);

        // 2. Copy secondary transferred 4K radiance buffer (upscaled on GPU 1) into m_secAccumImage
        m_secAccumImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
        );

        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copyRegion.imageOffset = { 0, 0, 0 };
        copyRegion.imageExtent = { m_config.width, m_config.height, 1 };

        vkCmdCopyBufferToImage(cmd, secBuffer, m_secAccumImage->getImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        m_secAccumImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        // 3. Barrier: ensure both primary FSR upscale and secondary buffer copy are complete
        VkMemoryBarrier2 f2bBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        f2bBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        f2bBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        f2bBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        f2bBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        f2bBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo f2bDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        f2bDep.memoryBarrierCount = 1;
        f2bDep.pMemoryBarriers = &f2bBarrier;
        vkCmdPipelineBarrier2(cmd, &f2bDep);

        // 4. 4K Blend Resolve Pass: merge the two 4K frames (50% / 50% averaging)
        if (m_fsr3BlendPipeline && m_fsr3BlendDescSet != VK_NULL_HANDLE) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fsr3BlendPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fsr3BlendPipelineLayout, 0, 1, &m_fsr3BlendDescSet, 0, nullptr);

            struct BlendPushConstants {
                uint32_t width;
                uint32_t height;
                float weightDst;
                float weightSrc;
            } blendPC;
            blendPC.width = m_config.width;
            blendPC.height = m_config.height;
            blendPC.weightDst = 0.5f;
            blendPC.weightSrc = 0.5f;

            vkCmdPushConstants(cmd, m_fsr3BlendPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(blendPC), &blendPC);
            uint32_t blendGroupsX = (m_config.width + 15) / 16;
            uint32_t blendGroupsY = (m_config.height + 15) / 16;
            vkCmdDispatch(cmd, blendGroupsX, blendGroupsY, 1);

            // Barrier: blend write -> tonemap compute read
            VkMemoryBarrier2 blendToTmBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            blendToTmBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            blendToTmBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            blendToTmBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            blendToTmBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            blendToTmBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

            VkDependencyInfo blendDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            blendDep.memoryBarrierCount = 1;
            blendDep.pMemoryBarriers = &blendToTmBarrier;
            vkCmdPipelineBarrier2(cmd, &blendDep);
        }
    } else {
        // Approach 1 (Tile Mode: Checkerboard / Scanline) OR Single-GPU:
        // In Approach 1, GPU 0 has already merged secondary GPU's tiles into m_accumImage at internal res.
        // GPU 0 upscales m_accumImage ONCE to display resolution (4K).
        UpscalerDispatchDesc desc{};
        desc.cmd = cmd;
        desc.colorIn = m_accumImage->getImageView();
        desc.depthIn = m_normalDepthImage ? m_normalDepthImage->getImageView() : VK_NULL_HANDLE;
        desc.motionVectorsIn = m_motionVectorImage ? m_motionVectorImage->getImageView() : VK_NULL_HANDLE;
        desc.colorOut = m_fsr3Upscaler->getOutputImage()->getImageView();
        desc.renderWidth = renderW;
        desc.renderHeight = renderH;
        desc.displayWidth = m_config.width;
        desc.displayHeight = m_config.height;
        glm::vec2 jitter(0.0f);
        if (m_config.upscaler_mode == UpscalerMode::FSR3) {
            if (!m_config.progressive_accumulation || m_cameraMovedLastFrame || m_accumulatedSamples <= 1) {
                glm::vec2 jitterPrim = getHaltonJitter(m_frameIndex);
                bool isMgpuSampleParallel = (m_mgpu && m_mgpu->isMultiGpuActive() && m_config.mgpu_mode != MultiGpuMode::Off &&
                    (m_config.mgpu_mode == MultiGpuMode::SampleParallel ||
                     (m_config.mgpu_mode == MultiGpuMode::Auto && (m_config.spp > 1 || (m_governor && m_config.adaptive_spp)))));
                if (isMgpuSampleParallel) {
                    glm::vec2 jitterSec = getHaltonJitter(m_frameIndex + 4);
                    jitter = (jitterPrim + jitterSec) * 0.5f;
                } else {
                    jitter = jitterPrim;
                }
            }
        }
        desc.jitterX = jitter.x;
        desc.jitterY = jitter.y;
        desc.enableSharpening = m_config.upscaler_sharpening;
        desc.sharpness = m_config.upscaler_sharpening ? m_config.upscaler_sharpness : 0.0f;
        desc.resetHistory = resetHistory;
        desc.cameraMoved = m_cameraMovedLastFrame;
        desc.frameIndex = m_frameIndex;
        desc.inputIsNormalized = true;
        desc.totalSamples = (m_config.progressive_accumulation && m_accumulatedSamples > 0) ? m_accumulatedSamples : 1u;

        m_fsr3Upscaler->recordUpscale(desc);
    }
    return true;
}

// === REAL-TIME CAUSTICS (PHOTON INJECTION + ATOMIC SPLATTING + BILATERAL FILTER) ===

void Engine::createCausticsPipelines() {
    VkDevice device = m_context->getDevice();

    // 1. Descriptor Set Layouts
    // 1a. Trace Layout: Triangles(2), Spheres(3), Materials(4), Lights(5), TLAS(6), Textures(8), Photons(20)
    std::vector<VkDescriptorSetLayoutBinding> traceBindings = {
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_SCENE_TEXTURES, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 30, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo traceLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    traceLayoutInfo.bindingCount = static_cast<uint32_t>(traceBindings.size());
    traceLayoutInfo.pBindings = traceBindings.data();
    if (vkCreateDescriptorSetLayout(device, &traceLayoutInfo, nullptr, &m_causticTraceDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic trace descriptor set layout");
    }

    // 1b. Splat Layout: CameraUBO(1), NormalDepth(12), Photons(20), AtomicBuffer(21)
    std::vector<VkDescriptorSetLayoutBinding> splatBindings = {
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 21, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo splatLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    splatLayoutInfo.bindingCount = static_cast<uint32_t>(splatBindings.size());
    splatLayoutInfo.pBindings = splatBindings.data();
    if (vkCreateDescriptorSetLayout(device, &splatLayoutInfo, nullptr, &m_causticSplatDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic splat descriptor set layout");
    }

    // 1c. Filter Layout: CameraUBO(1), NormalDepth(12), MotionVector(14), AtomicBuffer(21), FilteredCaustic(22), PrevCaustic(23)
    std::vector<VkDescriptorSetLayoutBinding> filterBindings = {
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 21, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 22, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 23, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo filterLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    filterLayoutInfo.bindingCount = static_cast<uint32_t>(filterBindings.size());
    filterLayoutInfo.pBindings = filterBindings.data();
    if (vkCreateDescriptorSetLayout(device, &filterLayoutInfo, nullptr, &m_causticFilterDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic filter descriptor set layout");
    }

    // 2. Descriptor Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 8 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2048 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16 },
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32 }
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 16;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_causticDescPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustics descriptor pool");
    }

    // Allocate Sets (MAX_FRAMES_IN_FLIGHT = 2 each)
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_causticDescPool;
        allocInfo.descriptorSetCount = 1;

        allocInfo.pSetLayouts = &m_causticTraceDescLayout;
        if (vkAllocateDescriptorSets(device, &allocInfo, &m_causticTraceDescSets[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate caustic trace descriptor set");
        }

        allocInfo.pSetLayouts = &m_causticSplatDescLayout;
        if (vkAllocateDescriptorSets(device, &allocInfo, &m_causticSplatDescSets[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate caustic splat descriptor set");
        }

        allocInfo.pSetLayouts = &m_causticFilterDescLayout;
        if (vkAllocateDescriptorSets(device, &allocInfo, &m_causticFilterDescSets[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate caustic filter descriptor set");
        }
    }

    // 3. Pipeline Layouts
    // 3a. Trace Pipeline Layout
    VkPushConstantRange tracePCRange{};
    tracePCRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tracePCRange.offset = 0;
    tracePCRange.size = sizeof(uint32_t) * 8 + sizeof(float) * 8; // 64 bytes
    VkPipelineLayoutCreateInfo tracePipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    tracePipeLayoutInfo.setLayoutCount = 1;
    tracePipeLayoutInfo.pSetLayouts = &m_causticTraceDescLayout;
    tracePipeLayoutInfo.pushConstantRangeCount = 1;
    tracePipeLayoutInfo.pPushConstantRanges = &tracePCRange;
    if (vkCreatePipelineLayout(device, &tracePipeLayoutInfo, nullptr, &m_causticTracePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic trace pipeline layout");
    }

    // 3b. Splat Pipeline Layout
    VkPushConstantRange splatPCRange{};
    splatPCRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    splatPCRange.offset = 0;
    splatPCRange.size = sizeof(uint32_t) * 3 + sizeof(float) + sizeof(glm::vec4) * 2; // 48 bytes
    VkPipelineLayoutCreateInfo splatPipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    splatPipeLayoutInfo.setLayoutCount = 1;
    splatPipeLayoutInfo.pSetLayouts = &m_causticSplatDescLayout;
    splatPipeLayoutInfo.pushConstantRangeCount = 1;
    splatPipeLayoutInfo.pPushConstantRanges = &splatPCRange;
    if (vkCreatePipelineLayout(device, &splatPipeLayoutInfo, nullptr, &m_causticSplatPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic splat pipeline layout");
    }

    // 3c. Filter Pipeline Layout
    VkPushConstantRange filterPCRange{};
    filterPCRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    filterPCRange.offset = 0;
    filterPCRange.size = sizeof(uint32_t) * 4; // 16 bytes
    VkPipelineLayoutCreateInfo filterPipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    filterPipeLayoutInfo.setLayoutCount = 1;
    filterPipeLayoutInfo.pSetLayouts = &m_causticFilterDescLayout;
    filterPipeLayoutInfo.pushConstantRangeCount = 1;
    filterPipeLayoutInfo.pPushConstantRanges = &filterPCRange;
    if (vkCreatePipelineLayout(device, &filterPipeLayoutInfo, nullptr, &m_causticFilterPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create caustic filter pipeline layout");
    }

    // 4. Compute Pipelines (Wave32 optimized for RDNA 4)
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
    };
    subgroupSize32.requiredSubgroupSize = 32;

    auto traceCode = loadShaderSPIRV("caustic_photon_trace.comp.spv");
    VkShaderModule traceMod = createShaderModule(traceCode);
    VkComputePipelineCreateInfo tracePipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    tracePipeInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, traceMod, "main", nullptr };
    if (m_context->hasSubgroupSizeControl()) tracePipeInfo.stage.pNext = &subgroupSize32;
    tracePipeInfo.layout = m_causticTracePipelineLayout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &tracePipeInfo, nullptr, &m_causticTracePipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, traceMod, nullptr);
        throw std::runtime_error("Failed to create caustic trace compute pipeline");
    }
    vkDestroyShaderModule(device, traceMod, nullptr);

    auto splatCode = loadShaderSPIRV("caustic_splat.comp.spv");
    VkShaderModule splatMod = createShaderModule(splatCode);
    VkComputePipelineCreateInfo splatPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    splatPipeInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, splatMod, "main", nullptr };
    if (m_context->hasSubgroupSizeControl()) splatPipeInfo.stage.pNext = &subgroupSize32;
    splatPipeInfo.layout = m_causticSplatPipelineLayout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &splatPipeInfo, nullptr, &m_causticSplatPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, splatMod, nullptr);
        throw std::runtime_error("Failed to create caustic splat compute pipeline");
    }
    vkDestroyShaderModule(device, splatMod, nullptr);

    auto filterCode = loadShaderSPIRV("caustic_filter.comp.spv");
    VkShaderModule filterMod = createShaderModule(filterCode);
    VkComputePipelineCreateInfo filterPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    filterPipeInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, filterMod, "main", nullptr };
    if (m_context->hasSubgroupSizeControl()) filterPipeInfo.stage.pNext = &subgroupSize32;
    filterPipeInfo.layout = m_causticFilterPipelineLayout;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &filterPipeInfo, nullptr, &m_causticFilterPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, filterMod, nullptr);
        throw std::runtime_error("Failed to create caustic filter compute pipeline");
    }
    vkDestroyShaderModule(device, filterMod, nullptr);

    Logger::info("Real-time Caustics pipelines (Wave32 trace, splat, bilateral filter) created successfully.");
}

void Engine::destroyCausticsPipelines() {
    VkDevice device = m_context ? m_context->getDevice() : VK_NULL_HANDLE;
    if (!device) return;

    if (m_causticTracePipeline) { vkDestroyPipeline(device, m_causticTracePipeline, nullptr); m_causticTracePipeline = VK_NULL_HANDLE; }
    if (m_causticSplatPipeline) { vkDestroyPipeline(device, m_causticSplatPipeline, nullptr); m_causticSplatPipeline = VK_NULL_HANDLE; }
    if (m_causticFilterPipeline) { vkDestroyPipeline(device, m_causticFilterPipeline, nullptr); m_causticFilterPipeline = VK_NULL_HANDLE; }

    if (m_causticTracePipelineLayout) { vkDestroyPipelineLayout(device, m_causticTracePipelineLayout, nullptr); m_causticTracePipelineLayout = VK_NULL_HANDLE; }
    if (m_causticSplatPipelineLayout) { vkDestroyPipelineLayout(device, m_causticSplatPipelineLayout, nullptr); m_causticSplatPipelineLayout = VK_NULL_HANDLE; }
    if (m_causticFilterPipelineLayout) { vkDestroyPipelineLayout(device, m_causticFilterPipelineLayout, nullptr); m_causticFilterPipelineLayout = VK_NULL_HANDLE; }

    if (m_causticDescPool) { vkDestroyDescriptorPool(device, m_causticDescPool, nullptr); m_causticDescPool = VK_NULL_HANDLE; }

    if (m_causticTraceDescLayout) { vkDestroyDescriptorSetLayout(device, m_causticTraceDescLayout, nullptr); m_causticTraceDescLayout = VK_NULL_HANDLE; }
    if (m_causticSplatDescLayout) { vkDestroyDescriptorSetLayout(device, m_causticSplatDescLayout, nullptr); m_causticSplatDescLayout = VK_NULL_HANDLE; }
    if (m_causticFilterDescLayout) { vkDestroyDescriptorSetLayout(device, m_causticFilterDescLayout, nullptr); m_causticFilterDescLayout = VK_NULL_HANDLE; }

    m_causticTraceDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    m_causticSplatDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    m_causticFilterDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
}

void Engine::createCausticsResources() {
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();
    uint32_t w = m_config.width;
    uint32_t h = m_config.height;

    // 1. Photon Buffer (64 bytes per CausticPhotonHit)
    uint32_t photonCount = std::max(m_config.caustic_photons, 65536u);
    VkDeviceSize photonBufferSize = static_cast<VkDeviceSize>(photonCount) * 64;
    m_causticPhotonBuffer = std::make_unique<Buffer>(
        allocator, photonBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    // 2. Atomic Accumulation Buffer (3x uint32_t per pixel: fixed-point Q16.16 RGB)
    VkDeviceSize atomicBufferSize = static_cast<VkDeviceSize>(w * h) * 3 * sizeof(uint32_t);
    m_causticAtomicBuffer = std::make_unique<Buffer>(
        allocator, atomicBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    // 3. Filtered Caustic Image & Previous History Image (RGBA16F)
    m_filteredCausticImage = std::make_unique<Image>(
        device, allocator, w, h,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_prevCausticImage = std::make_unique<Image>(
        device, allocator, w, h,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // Transition images to GENERAL layout and clear atomic buffer
    VkCommandBuffer cmd = m_commandBuffers[0];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd, &beginInfo);

    m_filteredCausticImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    m_prevCausticImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
    vkCmdFillBuffer(cmd, m_causticAtomicBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0);

    VkClearColorValue clearZero{};
    VkImageSubresourceRange sRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdClearColorImage(cmd, m_filteredCausticImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &sRange);
    vkCmdClearColorImage(cmd, m_prevCausticImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &sRange);

    vkEndCommandBuffer(cmd);
    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = cmd;
    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
    vkQueueSubmit2(m_context->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_context->getGraphicsQueue());

    updateCausticsDescriptors();
    Logger::info("Real-time Caustics GPU resources allocated successfully.");
}

void Engine::destroyCausticsResources() {
    m_causticPhotonBuffer.reset();
    m_causticAtomicBuffer.reset();
    m_filteredCausticImage.reset();
    m_prevCausticImage.reset();
}

void Engine::createReSTIRResources() {
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();
    try {
        auto temporalCode  = loadShaderSPIRV("restir_di_temporal.comp.spv");
        auto spatialCode   = loadShaderSPIRV("restir_di_spatial.comp.spv");
        m_restirManager = std::make_unique<ReSTIRManager>(
            device, allocator,
            m_config.width, m_config.height,
            temporalCode, spatialCode
        );
        Logger::info("Ultra-Lean ReSTIR PT Subsystem (32B Reservoirs, Fused Temporal & LDS Spatial Reuse) initialized successfully.");
    } catch (const std::exception& e) {
        Logger::warn("ReSTIRManager initialization failed: {}", e.what());
    }
}

void Engine::destroyReSTIRResources() {
    m_restirManager.reset();
}

void Engine::updateCausticsDescriptors() {
    VkDevice device = m_context->getDevice();
    if (!m_causticDescPool || !m_causticPhotonBuffer || !m_causticAtomicBuffer ||
        !m_filteredCausticImage || !m_prevCausticImage || !m_normalDepthImage ||
        !m_motionVectorImage || !m_triangleBuffer) {
        return;
    }

    VkDescriptorBufferInfo triInfo{ m_triangleBuffer->getBuffer(), 0, m_triangleBuffer->getSize() };
    VkDescriptorBufferInfo sphereInfo{ m_sphereBuffer->getBuffer(), 0, m_sphereBuffer->getSize() };
    VkDescriptorBufferInfo matInfo{ m_materialBuffer->getBuffer(), 0, m_materialBuffer->getSize() };
    VkDescriptorBufferInfo lightInfo{ m_lightBuffer->getBuffer(), 0, m_lightBuffer->getSize() };

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
    asInfo.accelerationStructureCount = 1;
    VkAccelerationStructureKHR tlasHandle = m_tlas ? m_tlas->getHandle() : VK_NULL_HANDLE;
    asInfo.pAccelerationStructures = &tlasHandle;

    std::vector<VkDescriptorImageInfo> texInfos(MAX_SCENE_TEXTURES);
    for (size_t i = 0; i < MAX_SCENE_TEXTURES; ++i) {
        if (i < m_sceneTextures.size() && m_sceneTextures[i]) {
            texInfos[i] = m_sceneTextures[i]->getDescriptorInfo();
        } else {
            texInfos[i] = m_dummyWhite->getDescriptorInfo();
        }
    }

    VkDescriptorBufferInfo photonBufInfo{ m_causticPhotonBuffer->getBuffer(), 0, m_causticPhotonBuffer->getSize() };
    VkDescriptorBufferInfo atomicBufInfo{ m_causticAtomicBuffer->getBuffer(), 0, m_causticAtomicBuffer->getSize() };

    VkDescriptorImageInfo ndImageInfo{ VK_NULL_HANDLE, m_normalDepthImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo mvImageInfo{ VK_NULL_HANDLE, m_motionVectorImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo filteredCausticInfo{ VK_NULL_HANDLE, m_filteredCausticImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo prevCausticInfo{ VK_NULL_HANDLE, m_prevCausticImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkDescriptorBufferInfo, MAX_FRAMES_IN_FLIGHT> camInfos;
    std::array<VkDescriptorBufferInfo, MAX_FRAMES_IN_FLIGHT> instanceInfos;
    VkBuffer actualInstanceBuffer = (m_instanceBuffer && m_instanceBuffer->getBuffer() != VK_NULL_HANDLE) ? m_instanceBuffer->getBuffer() : m_triangleBuffer->getBuffer();
    VkDeviceSize actualInstanceSize = (m_instanceBuffer && m_instanceBuffer->getSize() > 0) ? m_instanceBuffer->getSize() : m_triangleBuffer->getSize();

    std::vector<VkWriteDescriptorSet> writes;

    for (uint32_t slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
        if (!m_cameraUBOs[slot]) continue;
        camInfos[slot] = { m_cameraUBOs[slot]->getBuffer(), 0, sizeof(CameraUniform) };
        instanceInfos[slot] = { actualInstanceBuffer, 0, actualInstanceSize };

        // 1. Caustic Trace Set
        VkDescriptorSet traceSet = m_causticTraceDescSets[slot];
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sphereInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightInfo, nullptr });
        if (tlasHandle != VK_NULL_HANDLE) {
            writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, traceSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr });
        }
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 8, 0, MAX_SCENE_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texInfos.data(), nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 20, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &photonBufInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, traceSet, 30, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &instanceInfos[slot], nullptr });

        // 2. Caustic Splat Set
        VkDescriptorSet splatSet = m_causticSplatDescSets[slot];
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, splatSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &camInfos[slot], nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, splatSet, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ndImageInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, splatSet, 20, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &photonBufInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, splatSet, 21, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &atomicBufInfo, nullptr });

        // 3. Caustic Filter Set
        VkDescriptorSet filterSet = m_causticFilterDescSets[slot];
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &camInfos[slot], nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ndImageInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mvImageInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 21, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &atomicBufInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 22, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &filteredCausticInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, filterSet, 23, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &prevCausticInfo, nullptr, nullptr });
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void Engine::dispatchCausticTrace(VkCommandBuffer cmd, uint32_t frameSlot) {
    if (!m_causticTracePipeline || !m_causticPhotonBuffer || m_numLights == 0) return;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_causticTracePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_causticTracePipelineLayout, 0, 1, &m_causticTraceDescSets[frameSlot], 0, nullptr);

    struct CausticTracePC {
        uint32_t numTriangles;
        uint32_t numSpheres;
        uint32_t numMaterials;
        uint32_t numLights;
        uint32_t photonCount;
        uint32_t frameIndex;
        uint32_t numOpaqueTriangles;
        uint32_t maxBounces;
        glm::vec4 targetBBoxMin;
        glm::vec4 targetBBoxMax;
    } pc;

    pc.numTriangles = m_numTriangles;
    pc.numSpheres = m_numSpheres;
    pc.numMaterials = m_numMaterials;
    pc.numLights = m_numLights;
    pc.photonCount = m_config.caustic_photons;
    pc.frameIndex = m_frameIndex;
    pc.numOpaqueTriangles = m_numOpaqueTriangles;
    pc.maxBounces = 4u;
    pc.targetBBoxMin = glm::vec4(m_sceneData.dielectricBoundsMin, m_sceneData.hasDielectrics ? 1.0f : 0.0f);
    pc.targetBBoxMax = glm::vec4(m_sceneData.dielectricBoundsMax, 0.0f);

    vkCmdPushConstants(cmd, m_causticTracePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t groups = (m_config.caustic_photons + 255) / 256;
    vkCmdDispatch(cmd, groups, 1, 1);

    // Barrier: Photon buffer write -> Photon buffer read in Splat pass
    VkBufferMemoryBarrier2 photonBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    photonBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    photonBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    photonBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    photonBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    photonBarrier.buffer = m_causticPhotonBuffer->getBuffer();
    photonBarrier.offset = 0;
    photonBarrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.bufferMemoryBarrierCount = 1;
    dep.pBufferMemoryBarriers = &photonBarrier;
    vkCmdPipelineBarrier2(cmd, &dep);
}

void Engine::dispatchCausticSplatAndFilter(VkCommandBuffer cmd, uint32_t frameSlot) {
    if (!m_causticSplatPipeline || !m_causticFilterPipeline ||
        !m_causticAtomicBuffer || !m_causticPhotonBuffer || !m_filteredCausticImage) return;

    // 1. Fast GPU atomic accumulator buffer clear
    vkCmdFillBuffer(cmd, m_causticAtomicBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0);

    // 2. Barrier: FillBuffer -> Compute Shader Read/Write & NormalDepth Image Read Barrier
    VkBufferMemoryBarrier2 fillBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    fillBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    fillBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    fillBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    fillBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    fillBarrier.buffer = m_causticAtomicBuffer->getBuffer();
    fillBarrier.offset = 0;
    fillBarrier.size = VK_WHOLE_SIZE;

    VkImageMemoryBarrier2 ndBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    ndBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    ndBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    ndBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    ndBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    ndBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    ndBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    ndBarrier.image = m_normalDepthImage->getImage();
    ndBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo preSplatDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    preSplatDep.bufferMemoryBarrierCount = 1;
    preSplatDep.pBufferMemoryBarriers = &fillBarrier;
    preSplatDep.imageMemoryBarrierCount = 1;
    preSplatDep.pImageMemoryBarriers = &ndBarrier;
    vkCmdPipelineBarrier2(cmd, &preSplatDep);

    // 3. Dispatch Splat Pass
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_causticSplatPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_causticSplatPipelineLayout, 0, 1, &m_causticSplatDescSets[frameSlot], 0, nullptr);

    struct CausticSplatPC {
        uint32_t width;
        uint32_t height;
        uint32_t photonCount;
        float fov;
        glm::vec4 targetBBoxMin;
        glm::vec4 targetBBoxMax;
    } splatPC;
    splatPC.width = m_config.width;
    splatPC.height = m_config.height;
    splatPC.photonCount = m_config.caustic_photons;
    splatPC.fov = m_camera ? m_camera->getFov() : 45.0f;
    splatPC.targetBBoxMin = glm::vec4(m_sceneData.dielectricBoundsMin, m_sceneData.hasDielectrics ? 1.0f : 0.0f);
    splatPC.targetBBoxMax = glm::vec4(m_sceneData.dielectricBoundsMax, 0.0f);

    vkCmdPushConstants(cmd, m_causticSplatPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(splatPC), &splatPC);
    uint32_t splatGroups = (m_config.caustic_photons + 255) / 256;
    vkCmdDispatch(cmd, splatGroups, 1, 1);

    // 4. Barrier: Splat atomic writes -> Filter read
    VkBufferMemoryBarrier2 splatToFilterBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    splatToFilterBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    splatToFilterBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    splatToFilterBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    splatToFilterBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    splatToFilterBarrier.buffer = m_causticAtomicBuffer->getBuffer();
    splatToFilterBarrier.offset = 0;
    splatToFilterBarrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo filterDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    filterDep.bufferMemoryBarrierCount = 1;
    filterDep.pBufferMemoryBarriers = &splatToFilterBarrier;
    vkCmdPipelineBarrier2(cmd, &filterDep);

    // 5. Dispatch Bilateral Filter & Temporal Accumulation
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_causticFilterPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_causticFilterPipelineLayout, 0, 1, &m_causticFilterDescSets[frameSlot], 0, nullptr);

    uint32_t accumHist = 0u;
    if (m_config.progressive_accumulation && !m_cameraMovedLastFrame) {
        accumHist = 2u; // stationary progressive
    }

    struct CausticFilterPC {
        uint32_t width;
        uint32_t height;
        uint32_t frameIndex;
        uint32_t accumulateHistory;
    } filterPC;
    filterPC.width = m_config.width;
    filterPC.height = m_config.height;
    filterPC.frameIndex = m_frameIndex;
    filterPC.accumulateHistory = accumHist;

    vkCmdPushConstants(cmd, m_causticFilterPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(filterPC), &filterPC);
    uint32_t filterGroupsX = (m_config.width + 15) / 16;
    uint32_t filterGroupsY = (m_config.height + 15) / 16;
    vkCmdDispatch(cmd, filterGroupsX, filterGroupsY, 1);

    // 6. Barrier: Filter write -> Copy to Prev Image & Shading Read
    VkImageMemoryBarrier2 filterPostBarrier[2] = {};
    filterPostBarrier[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    filterPostBarrier[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    filterPostBarrier[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    filterPostBarrier[0].dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    filterPostBarrier[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    filterPostBarrier[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    filterPostBarrier[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    filterPostBarrier[0].image = m_filteredCausticImage->getImage();
    filterPostBarrier[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    filterPostBarrier[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    filterPostBarrier[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    filterPostBarrier[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    filterPostBarrier[1].dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    filterPostBarrier[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    filterPostBarrier[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    filterPostBarrier[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    filterPostBarrier[1].image = m_prevCausticImage->getImage();
    filterPostBarrier[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo copyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    copyDep.imageMemoryBarrierCount = 2;
    copyDep.pImageMemoryBarriers = filterPostBarrier;
    vkCmdPipelineBarrier2(cmd, &copyDep);

    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.extent = { m_config.width, m_config.height, 1 };
    vkCmdCopyImage(cmd, m_filteredCausticImage->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                   m_prevCausticImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);

    // Final barrier: m_filteredCausticImage ready for shader read in Bounce 0 shade
    VkImageMemoryBarrier2 finalBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    finalBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
    finalBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    finalBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    finalBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    finalBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    finalBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    finalBarrier.image = m_filteredCausticImage->getImage();
    finalBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo finalDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    finalDep.imageMemoryBarrierCount = 1;
    finalDep.pImageMemoryBarriers = &finalBarrier;
    vkCmdPipelineBarrier2(cmd, &finalDep);
}

void Engine::dispatchCaustics(VkCommandBuffer cmd, uint32_t frameSlot) {
    if (!m_config.enable_caustics || !m_sceneData.hasDielectrics || m_numLights == 0) return;
    dispatchCausticTrace(cmd, frameSlot);
    dispatchCausticSplatAndFilter(cmd, frameSlot);
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
        VkDescriptorImageInfo bnInfo = m_blueNoiseTexture ? m_blueNoiseTexture->getDescriptorInfo() : m_dummyWhite->getDescriptorInfo();
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSets[i], 13, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &bnInfo, nullptr, nullptr });
    }
    if (!writes.empty()) {
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    updateWavefrontSceneDescriptors();
    updateCausticsDescriptors();
}

void Engine::updateWavefrontSceneDescriptors() {
    if (!m_wavefrontPipeline || !m_accumImage || !m_triangleBuffer) return;

    VkAccelerationStructureKHR tlasHandle = m_tlas ? m_tlas->getHandle() : VK_NULL_HANDLE;
    VkDescriptorImageInfo envInfo = m_environmentMap ? m_environmentMap->getDescriptorInfo() : m_dummyWhite->getDescriptorInfo();

    std::vector<VkDescriptorImageInfo> texInfos(MAX_SCENE_TEXTURES);
    for (size_t i = 0; i < MAX_SCENE_TEXTURES; ++i) {
        if (i < m_sceneTextures.size() && m_sceneTextures[i]) {
            texInfos[i] = m_sceneTextures[i]->getDescriptorInfo();
        } else {
            texInfos[i] = m_dummyWhite->getDescriptorInfo();
        }
    }

    VkBuffer nrcQueryBuf = m_nrcManager ? m_nrcManager->getQueryQueue()->getBuffer() : VK_NULL_HANDLE;
    VkBuffer nrcTrainBuf = m_nrcManager ? m_nrcManager->getTrainQueue()->getBuffer() : VK_NULL_HANDLE;
    VkBuffer nrcCountBuf = m_nrcManager ? m_nrcManager->getCounters()->getBuffer() : VK_NULL_HANDLE;

    for (uint32_t slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
        if (!m_cameraUBOs[slot] || !m_frameImages[slot]) continue;
        VkBuffer restirReservoirBuf = m_restirManager ? m_restirManager->getSpatialReservoirBuffer(slot)->getBuffer() : VK_NULL_HANDLE;
        m_wavefrontPipeline->updateSceneDescriptors(
            slot,
            m_frameImages[slot]->getImageView(),
            m_cameraUBOs[slot]->getBuffer(),
            m_triangleBuffer->getBuffer(), m_triangleBuffer->getSize(),
            m_sphereBuffer->getBuffer(), m_sphereBuffer->getSize(),
            m_materialBuffer->getBuffer(), m_materialBuffer->getSize(),
            m_lightBuffer->getBuffer(), m_lightBuffer->getSize(),
            tlasHandle,
            envInfo,
            texInfos,
            nrcQueryBuf,
            nrcTrainBuf,
            nrcCountBuf,
            m_motionVectorImage ? m_motionVectorImage->getImageView() : VK_NULL_HANDLE,
            m_normalDepthImage ? m_normalDepthImage->getImageView() : VK_NULL_HANDLE,
            m_lightTreeBuffer ? m_lightTreeBuffer->getBuffer() : VK_NULL_HANDLE,
            m_lightTreeBuffer ? m_lightTreeBuffer->getSize() : 0,
            m_mlAlbedoRoughnessImage ? m_mlAlbedoRoughnessImage->getImageView() : VK_NULL_HANDLE,
            m_mlSpecularMotionImage ? m_mlSpecularMotionImage->getImageView() : VK_NULL_HANDLE,
            m_mlDiffuseImage ? m_mlDiffuseImage->getImageView() : VK_NULL_HANDLE,
            m_mlSpecularImage ? m_mlSpecularImage->getImageView() : VK_NULL_HANDLE,
            m_instanceBuffer ? m_instanceBuffer->getBuffer() : VK_NULL_HANDLE,
            m_instanceBuffer ? m_instanceBuffer->getSize() : 0,
            m_filteredCausticImage ? m_filteredCausticImage->getImageView() : VK_NULL_HANDLE,
            restirReservoirBuf,
            m_materialArchetypeBuffer ? m_materialArchetypeBuffer->getBuffer() : VK_NULL_HANDLE,
            m_materialArchetypeBuffer ? m_materialArchetypeBuffer->getSize() : 0,
            m_shadeMaterialBuffer ? m_shadeMaterialBuffer->getBuffer() : VK_NULL_HANDLE,
            m_shadeMaterialBuffer ? m_shadeMaterialBuffer->getSize() : 0
        );
    }

    if (m_nrcManager && m_frameImages[0]) {
        m_nrcManager->updateDescriptors(m_frameImages[0]->getImageView());
    }
}

void Engine::updateMergeDescriptors() {
    if (!m_accumImage || !m_motionVectorImage || !m_normalDepthImage) {
        return;
    }

    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();

    uint32_t bytesPerPixel = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? 24 : 32;
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(m_config.width) * m_config.height * bytesPerPixel;

    VkDescriptorImageInfo mvImageInfo{};
    mvImageInfo.imageView = m_motionVectorImage->getImageView();
    mvImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo normDepthInfo{};
    normDepthInfo.imageView = m_normalDepthImage->getImageView();
    normDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    uint32_t renderW = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
        static_cast<uint32_t>(m_config.width * m_config.render_scale) : m_config.width;
    uint32_t renderH = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
        static_cast<uint32_t>(m_config.height * m_config.render_scale) : m_config.height;

    uint32_t tileSize = (m_config.tile_size == 0u) ? 64u : m_config.tile_size;
    uint32_t numTilesX = (renderW + tileSize - 1u) / tileSize;
    uint32_t maxTilesPerGpuX = (numTilesX + 1u) / 2u;
    uint32_t dispatchWidth = maxTilesPerGpuX * tileSize;
    uint32_t dispatchHeight = renderH;

    uint32_t bpp = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? 8 : 16;
    VkDeviceSize radSize = static_cast<VkDeviceSize>(dispatchWidth) * dispatchHeight * bpp;
    VkDeviceSize radOffsetAligned = (radSize + 65535) & ~static_cast<VkDeviceSize>(65535);
    VkDeviceSize mvSize = static_cast<VkDeviceSize>(dispatchWidth) * dispatchHeight * 4; // RG16F
    VkDeviceSize mvOffsetAligned = (radOffsetAligned + mvSize + 65535) & ~static_cast<VkDeviceSize>(65535);

    for (uint32_t slot = 0; slot < 2; ++slot) {
        if (m_mergeDescSets[slot] == VK_NULL_HANDLE) continue;

        VkBuffer secBuffer = VK_NULL_HANDLE;
        VkDeviceSize curSize = bufferSize;

        if (m_mgpu && m_mgpu->isSecondaryInitialized() && (m_mgpu->isZeroCopyActive() || m_mgpu->isP2PDirectBarActive())) {
            secBuffer = m_mgpu->getPrimarySharedBuffer(slot);
            curSize = m_mgpu->getSharedBufferSize();
        }
        if (secBuffer == VK_NULL_HANDLE) {
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
        VkDescriptorBufferInfo secMvInfo{ secBuffer, radOffsetAligned, curSize > radOffsetAligned ? (curSize - radOffsetAligned) : VK_WHOLE_SIZE };
        VkDescriptorBufferInfo secNdInfo{ secBuffer, mvOffsetAligned, curSize > mvOffsetAligned ? (curSize - mvOffsetAligned) : VK_WHOLE_SIZE };

        VkDescriptorImageInfo frameInfo{};
        if (m_frameImages[slot]) {
            frameInfo.sampler = VK_NULL_HANDLE;
            frameInfo.imageView = m_frameImages[slot]->getImageView();
            frameInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        }

        std::vector<VkWriteDescriptorSet> mergeWrites = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &frameInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secBufInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secBufInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mvImageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normDepthInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secMvInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secNdInfo, nullptr }
        };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(mergeWrites.size()), mergeWrites.data(), 0, nullptr);
    }
}

void Engine::initTlasBuffers(uint32_t instanceCount) {
    if (!m_asManager || instanceCount == 0) return;
    m_tlasInstanceCount = instanceCount;
    VmaAllocator allocator = m_context->getAllocator();
    VkDevice device = m_context->getDevice();

    // 1. Device-local TLAS Instance Buffer (64 bytes per VkAccelerationStructureInstanceKHR)
    VkDeviceSize instanceBufferSize = sizeof(VkAccelerationStructureInstanceKHR) * instanceCount;
    m_tlasInstanceBuffer = std::make_unique<Buffer>(
        allocator, instanceBufferSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    // 2. Host-visible GPU Instance Data Buffer (96 bytes per ASInstanceGPUData)
    VkDeviceSize inputBufferSize = sizeof(ASInstanceGPUData) * instanceCount;
    m_tlasInputInstancesBuffer = std::make_unique<Buffer>(
        allocator, inputBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    // 3. Device-local Scratch Buffer (aligned to 256 bytes)
    auto sizeInfo = m_asManager->getTLASBuildSizes(instanceCount);
    VkDeviceSize scratchSize = std::max(sizeInfo.buildScratchSize, sizeInfo.updateScratchSize);
    if (scratchSize > 0) {
        scratchSize = (scratchSize + 255) & ~VkDeviceSize(255);
        m_tlasScratchBuffer = std::make_unique<Buffer>(
            allocator, scratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            0,
            256
        );
    }

    // 4. Initialize instance 0 with default transform and BLAS address
    VkDeviceAddress defaultBlasAddr = !m_blases.empty() ? m_blases[0]->getDeviceAddress() : (m_blas ? m_blas->getDeviceAddress() : 0);
    if (defaultBlasAddr != 0 && m_tlasInputInstancesBuffer) {
        ASInstanceGPUData initData{};
        initData.transform = glm::mat4(1.0f);
        initData.customIndex = 0;
        initData.mask = 0xFF;
        initData.hitGroupId = 0;
        initData.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        initData.blasAddress = defaultBlasAddr;
        initData.pad0 = 0;
        initData.pad1 = 0;
        m_tlasInputInstancesBuffer->copyFrom(&initData, sizeof(ASInstanceGPUData));
    }

    // 5. Update descriptor set if already created
    if (m_updateTlasDescSet != VK_NULL_HANDLE && m_tlasInputInstancesBuffer && m_tlasInstanceBuffer) {
        VkDescriptorBufferInfo inInfo{ m_tlasInputInstancesBuffer->getBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo outInfo{ m_tlasInstanceBuffer->getBuffer(), 0, VK_WHOLE_SIZE };

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_updateTlasDescSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &inInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_updateTlasDescSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &outInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void Engine::initTlasUpdatePipeline() {
    VkDevice device = m_context->getDevice();

    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &m_updateTlasDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS update descriptor set layout!");
    }

    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    if (vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_updateTlasDescPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS update descriptor pool!");
    }

    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = m_updateTlasDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_updateTlasDescLayout;
    if (vkAllocateDescriptorSets(device, &allocInfo, &m_updateTlasDescSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate TLAS update descriptor set!");
    }

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(uint32_t) * 2;

    VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_updateTlasDescLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(device, &plInfo, nullptr, &m_updateTlasPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS update pipeline layout!");
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    auto compCode = loadShaderSPIRV("update_tlas_instances.comp.spv");
    VkShaderModule compModule = createShaderModule(compCode);

    VkComputePipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = compModule;
    pipeInfo.stage.pName = "main";
    if (m_context->hasSubgroupSizeControl()) {
        pipeInfo.stage.pNext = &subgroupSize32;
    }
    pipeInfo.layout = m_updateTlasPipelineLayout;

    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_updateTlasPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(device, compModule, nullptr);
        throw std::runtime_error("Failed to create TLAS update compute pipeline!");
    }
    vkDestroyShaderModule(device, compModule, nullptr);

    if (m_tlasInputInstancesBuffer && m_tlasInstanceBuffer) {
        VkDescriptorBufferInfo inInfo{ m_tlasInputInstancesBuffer->getBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo outInfo{ m_tlasInstanceBuffer->getBuffer(), 0, VK_WHOLE_SIZE };

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_updateTlasDescSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &inInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_updateTlasDescSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &outInfo;

        vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    Logger::info("GPU TLAS Instance Writer & Refit Pipeline initialized successfully.");
}

void Engine::recordGpuTlasUpdate(VkCommandBuffer cmd, bool updateMode) {
    if (!m_updateTlasPipeline || !m_tlasInstanceBuffer || !m_tlasScratchBuffer || !m_tlas || m_tlasInstanceCount == 0) {
        return;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updateTlasPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updateTlasPipelineLayout, 0, 1, &m_updateTlasDescSet, 0, nullptr);

    struct {
        uint32_t instanceCount;
        uint32_t updateMode;
    } pc = { m_tlasInstanceCount, updateMode ? 1u : 0u };
    vkCmdPushConstants(cmd, m_updateTlasPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t groupCountX = (m_tlasInstanceCount + 63) / 64;
    vkCmdDispatch(cmd, groupCountX, 1, 1);

    m_asManager->recordBuildTLAS(cmd, m_tlasInstanceBuffer.get(), m_tlasInstanceCount, m_tlasScratchBuffer.get(), m_tlas.get(), updateMode);
    m_tlasGpuUpdateCount++;
}

void Engine::updateInstanceTransform(uint32_t index, const glm::mat4& transform) {
    if (!m_tlasInputInstancesBuffer || index >= m_tlasInstanceCount) {
        return;
    }
    VkDeviceSize offset = index * sizeof(ASInstanceGPUData) + offsetof(ASInstanceGPUData, transform);
    void* mapped = m_tlasInputInstancesBuffer->map();
    if (mapped) {
        std::memcpy(static_cast<char*>(mapped) + offset, &transform, sizeof(glm::mat4));
        vmaFlushAllocation(m_context->getAllocator(), m_tlasInputInstancesBuffer->getAllocation(), offset, sizeof(glm::mat4));
        m_tlasNeedsGpuUpdate = true;
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

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkCreateSemaphore(device, &semInfo, nullptr, &m_rtCompleteSemaphores[i]);
    }

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
    queryInfo.queryCount = QUERIES_PER_FRAME * MAX_FRAMES_IN_FLIGHT; // 6 timestamps per frame in flight
    vkCreateQueryPool(device, &queryInfo, nullptr, &m_queryPool);

    m_timestampPeriod = m_context->getDeviceProperties().limits.timestampPeriod;
    Logger::info("GPU Timestamp Profiler initialized (period: {:.2f} ns/tick, {} queries/frame)", m_timestampPeriod, QUERIES_PER_FRAME);
}

void Engine::setCameraMode(bool active) {
    if (m_cameraMode == active) return;
    m_cameraMode = active;
    if (!m_cameraMode) {
        m_gamepadLeftX = 0.0f;
        m_gamepadLeftY = 0.0f;
        m_gamepadRightX = 0.0f;
        m_gamepadRightY = 0.0f;
        m_gamepadLeftTrigger = 0.0f;
        m_gamepadRightTrigger = 0.0f;
        m_gamepadBtnA = false;
        m_gamepadBtnB = false;
    }
    if (m_window) {
        m_window->setRelativeMouseMode(m_cameraMode);
    }
    Logger::info("Interaction Mode: {}", m_cameraMode ? "FPS Scene Navigation (Mouse grabbed, WASD active)" : "UI Control Panel (Mouse released)");
}

void Engine::setMgpuMode(MultiGpuMode mode) {
    m_pendingMgpuModeChange = true;
    m_newMgpuMode = mode;
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
        if (e.type == SDL_EVENT_WINDOW_MINIMIZED || e.type == SDL_EVENT_WINDOW_OCCLUDED) {
            m_isMinimized = true;
        } else if (e.type == SDL_EVENT_WINDOW_RESTORED || e.type == SDL_EVENT_WINDOW_EXPOSED) {
            m_isMinimized = false;
        }
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

    // 5. ESC key: if in camera mode, return to UI mode; if in UI mode, close application
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

    // 6. Gamepad Hotplug & Dual-Analog Navigation Events (FEAT-01)
    if (e.type == SDL_EVENT_GAMEPAD_ADDED) {
        if (!m_gamepad) {
            m_gamepad = SDL_OpenGamepad(e.gdevice.which);
            if (m_gamepad) {
                Logger::info("Gamepad connected: {}", SDL_GetGamepadName(m_gamepad));
            }
        }
        return true;
    }
    if (e.type == SDL_EVENT_GAMEPAD_REMOVED) {
        if (m_gamepad && e.gdevice.which == SDL_GetGamepadID(m_gamepad)) {
            Logger::info("Gamepad disconnected.");
            SDL_CloseGamepad(m_gamepad);
            m_gamepad = nullptr;
            m_gamepadLeftX = 0.0f;
            m_gamepadLeftY = 0.0f;
            m_gamepadRightX = 0.0f;
            m_gamepadRightY = 0.0f;
            m_gamepadLeftTrigger = 0.0f;
            m_gamepadRightTrigger = 0.0f;
            m_gamepadBtnA = false;
            m_gamepadBtnB = false;
        }
        return true;
    }
    if (e.type == SDL_EVENT_GAMEPAD_AXIS_MOTION) {
        float deadzone = std::clamp(m_config.gamepad_deadzone, 0.01f, 0.50f);
        float rawVal = static_cast<float>(e.gaxis.value) / 32767.0f;
        float val = 0.0f;
        if (std::abs(rawVal) >= deadzone) {
            val = std::copysign((std::abs(rawVal) - deadzone) / (1.0f - deadzone), rawVal);
        }

        if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTX) m_gamepadLeftX = val;
        else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFTY) m_gamepadLeftY = -val;
        else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHTX) m_gamepadRightX = val;
        else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHTY) m_gamepadRightY = val;
        else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_LEFT_TRIGGER) m_gamepadLeftTrigger = std::max(0.0f, val);
        else if (e.gaxis.axis == SDL_GAMEPAD_AXIS_RIGHT_TRIGGER) m_gamepadRightTrigger = std::max(0.0f, val);
        return true;
    }
    if (e.type == SDL_EVENT_GAMEPAD_BUTTON_DOWN) {
        if (e.gbutton.button == SDL_GAMEPAD_BUTTON_START || e.gbutton.button == SDL_GAMEPAD_BUTTON_BACK) {
            setCameraMode(!m_cameraMode);
            return true;
        }
        if (e.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH) m_gamepadBtnA = true;
        if (e.gbutton.button == SDL_GAMEPAD_BUTTON_EAST) m_gamepadBtnB = true;
        return true;
    }
    if (e.type == SDL_EVENT_GAMEPAD_BUTTON_UP) {
        if (e.gbutton.button == SDL_GAMEPAD_BUTTON_SOUTH) m_gamepadBtnA = false;
        if (e.gbutton.button == SDL_GAMEPAD_BUTTON_EAST) m_gamepadBtnB = false;
        return true;
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
    if (dt <= 0.0f) {
        dt = 0.016f;
    } else if (dt > 0.1f) {
        dt = 0.1f;
    }

    if (!m_cameraMode || m_config.headless || !m_camera) {
        return;
    }

    const bool* keyState = SDL_GetKeyboardState(nullptr);
    if (!keyState) return;

    glm::vec3 camPos = m_camera->getPosition();
    glm::vec3 camFront = m_camera->getFront();
    bool hitGeometry = false;
    glm::vec3 hitPoint = camPos + camFront * m_camera->getFocalDistance();
    if (m_hasGpuCenterDepth && m_gpuCenterDepth > 0.05f) {
        m_camera->setLookDistance(m_gpuCenterDepth);
        hitGeometry = true;
        hitPoint = camPos + camFront * m_gpuCenterDepth;
    } else {
        // Looking at open sky or empty space: use cruising scale distance
        float fallbackDist = std::max(m_camera->getFocalDistance() * 2.5f, m_camera->getSceneScale());
        m_camera->setLookDistance(fallbackDist);
    }

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
            if (hitGeometry) {
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

    // Incorporate analog gamepad sticks & triggers (FEAT-01)
    forward += m_gamepadLeftY;
    strafe += m_gamepadLeftX;
    if (m_gamepadBtnA) vertical += 1.0f;
    if (m_gamepadBtnB) vertical -= 1.0f;
    if (m_gamepadRightTrigger > 0.1f) sprint = true;
    if (m_gamepadLeftTrigger > 0.1f) crawl = true;

    if (std::abs(m_gamepadRightX) > 0.05f || std::abs(m_gamepadRightY) > 0.05f) {
        constexpr float GAMEPAD_ROT_SPEED = 180.0f; // degrees per second
        m_camera->processMouseMovement(m_gamepadRightX * GAMEPAD_ROT_SPEED * dt, -m_gamepadRightY * GAMEPAD_ROT_SPEED * dt, ctrl);
    }

    m_camera->processFpsInput(forward, strafe, vertical, dt, sprint, crawl, ctrl);
    m_camera->update(dt);
}

void Engine::renderFrame() {
    auto frameNow = std::chrono::high_resolution_clock::now();
    if (m_totalFramesRendered > 0) {
        double wallIntervalMs = std::chrono::duration<double, std::milli>(frameNow - m_lastWallFrameStartTime).count();
        if (wallIntervalMs > 0.01 && wallIntervalMs < 1000.0) {
            m_lastPresentationTimeMs = wallIntervalMs;
            m_presentationTimesMs.push_back(wallIntervalMs);
            if (!m_config.headless && m_presentationTimesMs.size() > 60) {
                m_presentationTimesMs.erase(m_presentationTimesMs.begin());
            }
        }
    }
    m_lastWallFrameStartTime = frameNow;
    m_currentFrameStartTime = frameNow;
    VkDevice device = m_context->getDevice();
    VkQueue queue = m_context->getGraphicsQueue();

    vkWaitForFences(device, 1, &m_inFlightFences[m_currentFrame], VK_TRUE, UINT64_MAX);

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
                        if (m_lastWavefrontProfile.bounces.back().nextCount == 0) {
                            // All rays terminated at or before the last active bounce
                            m_dynamicWavefrontBounces = activeBouncesCount;
                        } else {
                            // Rays were still alive at cutoff, expand headroom up to max_bounces
                            m_dynamicWavefrontBounces = std::min(m_config.max_bounces, activeBouncesCount + 4);
                        }
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
                        uint32_t traceW = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
                            static_cast<uint32_t>(m_config.width * m_config.render_scale) : m_config.width;
                        uint32_t traceH = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
                            static_cast<uint32_t>(m_config.height * m_config.render_scale) : m_config.height;
                        wfSample.primaryRays = static_cast<uint64_t>(traceW) * traceH * m_config.spp;
                        for (const auto& bp : m_lastWavefrontProfile.bounces) {
                            wfSample.bounces.push_back({bp.shadeMs, bp.shadowMs, bp.intersectMs, bp.gapBeforeShadeMs, bp.gapBeforeShadowMs, bp.gapBeforeIntersectMs, bp.activeCount, bp.nextCount, bp.shadowCount});
                        }
                    }
                    recordFrameTally(totalGpuMs, gpuRtMs, secGpuMs, gpuTonemapMs, wfSample.bounces.empty() ? nullptr : &wfSample);
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
    if (m_isSceneLoading.load()) {
        if (m_sceneLoadingFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            SceneData loadedData = m_sceneLoadingFuture.get();
            applyLoadedScene(std::move(loadedData), m_loadingScenePath);
            m_isSceneLoading.store(false);
            m_loadingScenePath.clear();
            m_loadingSceneName.clear();
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

            VkFormat frameFmt = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
            for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                m_frameImages[i] = std::make_unique<Image>(
                    dev, alloc, m_config.width, m_config.height,
                    frameFmt,
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
                );
            }
            m_accumImage = std::make_unique<Image>(
                dev, alloc, m_config.width, m_config.height,
                frameFmt,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
            );

            // Transition m_frameImages and m_accumImage to GENERAL
            vkResetCommandBuffer(m_commandBuffers[0], 0);
            VkCommandBufferBeginInfo transBegin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            transBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(m_commandBuffers[0], &transBegin);
            for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
                m_frameImages[i]->transitionLayout(
                    m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                );
            }
            m_accumImage->transitionLayout(
                m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
            );
            vkEndCommandBuffer(m_commandBuffers[0]);
            VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
            cmdSubmitInfo.commandBuffer = m_commandBuffers[0];

            VkSubmitInfo2 transSubmit{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
            transSubmit.commandBufferInfoCount = 1;
            transSubmit.pCommandBufferInfos = &cmdSubmitInfo;
            vkQueueSubmit2(m_context->getGraphicsQueue(), 1, &transSubmit, VK_NULL_HANDLE);
            vkQueueWaitIdle(m_context->getGraphicsQueue());

            updateAccumTonemapDescriptors();
            updateAccumRunningAvgDescriptors();
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
            if (m_upwaysPipeline) {
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
            bool isSuperRes = m_config.upways_superres || (m_config.upscaler_mode == UpscalerMode::Upways);
            uint32_t inW = (m_config.render_scale < 1.0f && (m_config.upscaler_mode != UpscalerMode::None || m_config.upways_superres)) ?
                static_cast<uint32_t>(m_config.width * m_config.render_scale) : m_config.width;
            uint32_t inH = (m_config.render_scale < 1.0f && (m_config.upscaler_mode != UpscalerMode::None || m_config.upways_superres)) ?
                static_cast<uint32_t>(m_config.height * m_config.render_scale) : m_config.height;
            uint32_t outW = m_config.width;
            uint32_t outH = m_config.height;
            isSuperRes = (inW < outW || inH < outH || isSuperRes);

            if (m_upwaysPipeline) {
                m_upwaysPipeline->resize(inW, inH, outW, outH, isSuperRes);
                updateUpwaysDescriptors();
            }
        }

        updateMergeDescriptors();
        m_resetAccumulation = true;
        m_frameTimesMs.clear();
    }

    bool sceneLoadingActive = m_isSceneLoading.load() || m_pendingSceneChange;

    // Update smooth continuous FPS keyboard navigation
    if (!sceneLoadingActive) {
        updateInput();

        if (m_config.camera_motion && m_camera) {
            m_camera->processMouseMovement(2.0f, 0.0f);
        }
    }

    // Advance video billboard decoder if active
    if (m_videoDecoder && m_videoDecoder->isOpen()) {
        float dt = 1.0f / 24.0f;
        if (!m_config.headless && m_lastPresentationTimeMs > 0.01 && m_lastPresentationTimeMs < 1000.0) {
            dt = static_cast<float>(m_lastPresentationTimeMs * 0.001);
        }
        m_videoDecoder->update(dt);
    }

    // Reset accumulation if camera moved, camera just came to a stop, or UI settings changed
    bool cameraMovedThisFrame = !sceneLoadingActive && ((m_camera && m_camera->hasMoved() && m_totalFramesRendered > 0) || m_config.camera_motion);
    bool cameraJustStopped = (!cameraMovedThisFrame && m_cameraMovedLastFrame);
    bool hardReset = m_resetAccumulation || (m_totalFramesRendered == 0);
    bool accumReset = cameraMovedThisFrame || cameraJustStopped || hardReset;
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
    if (m_dynamicWavefrontBounces > 0) {
        activeBounces = std::min(activeBounces, m_dynamicWavefrontBounces);
    }
    if (m_governor && (m_config.adaptive_spp || m_config.target_fps > 0) && m_governor->getState().active) {
        activeSpp = m_governor->getState().currentSpp;
        activeFractionalSpp = m_governor->getState().fractionalSpp;
        activeBounces = std::min(activeBounces, m_governor->getState().currentBounces);
    }
    bool accumReachedCutoff = (m_config.progressive_accumulation &&
                               m_config.max_accum_frames > 0 &&
                               m_accumulatedSamples >= m_config.max_accum_frames);
    bool skipRayTracing = accumReachedCutoff || sceneLoadingActive;
    m_accumulationComplete = accumReachedCutoff;
    m_slotSkippedRayTracing[m_currentFrame] = skipRayTracing;
    if (m_config.progressive_accumulation) {
        if (!skipRayTracing) {
            m_accumulatedSamples++;
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
    if (m_config.inline_primary_shadows) flags |= (1 << 6);
    if (m_config.enable_light_tree || (!m_sceneData.lightTreeNodes.empty() && m_config.enable_restir_di)) flags |= (1 << 7);
    if (m_config.enable_caustics && m_sceneData.hasDielectrics && m_numLights > 0) flags |= (1 << 8);
    if (m_config.enable_restir_di) flags |= (1 << 9);
    if (accumReset || m_cameraMovedLastFrame) {
        flags |= (1 << 23); // Camera motion / history reset flag
    }

    uint32_t imageIndex = 0;
    if (!m_config.headless && m_swapchain) {
        int curW = 0, curH = 0;
        SDL_GetWindowSizeInPixels(m_window->getSDLWindow(), &curW, &curH);
        uint32_t targetW = (curW > 0) ? static_cast<uint32_t>(curW) : m_window->getWidth();
        uint32_t targetH = (curH > 0) ? static_cast<uint32_t>(curH) : m_window->getHeight();

        if (targetW != m_swapchain->getExtent().width ||
            targetH != m_swapchain->getExtent().height ||
            m_config.width != m_swapchain->getExtent().width ||
            m_config.height != m_swapchain->getExtent().height) {
            onResize(targetW, targetH, /*forceRecreate=*/true);
        }

        VkResult res = m_swapchain->acquireNextImage(m_imageAvailableSemaphores[m_currentFrame], &imageIndex);
        if (res == VK_ERROR_OUT_OF_DATE_KHR) {
            SDL_GetWindowSizeInPixels(m_window->getSDLWindow(), &curW, &curH);
            targetW = (curW > 0) ? static_cast<uint32_t>(curW) : m_window->getWidth();
            targetH = (curH > 0) ? static_cast<uint32_t>(curH) : m_window->getHeight();
            onResize(targetW, targetH, /*forceRecreate=*/true);
            return;
        }
        if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
            Logger::error("vkAcquireNextImageKHR failed with error: {}", static_cast<int>(res));
            return;
        }
    }

    bool isStationaryAccum = m_config.progressive_accumulation && !m_cameraMovedLastFrame && (m_accumulatedSamples > 1);
    bool enableJitter = ((m_config.upscaler_mode == UpscalerMode::FSR3) || (m_config.upscaler_mode == UpscalerMode::Upways)) && !isStationaryAccum;
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

    struct TonemapPushConstants {
        float exposure = 1.0f;
        uint32_t totalSamples = 1;
        uint32_t applyACES = 1;
        uint32_t visualizeSplit = 0;
        uint32_t tileSize = 64;
        uint32_t displayMode = 0;
        float peakNits = 1000.0f;
        float paperWhiteNits = 200.0f;
    } tonemapConstants;
    tonemapConstants.exposure = m_config.exposure;
    tonemapConstants.applyACES = m_config.aces_tonemap ? 1 : 0;
    tonemapConstants.visualizeSplit = 0;
    tonemapConstants.tileSize = m_config.tile_size;
    tonemapConstants.displayMode = (m_swapchain && !m_config.headless) ? static_cast<uint32_t>(m_swapchain->getHdrMode()) : 0u;
    tonemapConstants.peakNits = m_config.hdr_peak_nits;
    tonemapConstants.paperWhiteNits = m_config.hdr_paper_white_nits;

    bool isMgpu = (m_mgpu && m_mgpu->isMultiGpuActive() && m_config.mgpu_mode != MultiGpuMode::Off);

    VkCommandBuffer cmd = m_commandBuffers[m_currentFrame];
    VkCommandBuffer activeCmd = cmd;

    size_t frameBytes = 0;
    void* dstHost = nullptr;
    uint32_t tileOffsetX_sec = 0;
    uint32_t tileOffsetY_sec = 0;
    uint32_t secAccumHistory = 1;
    MultiGpuMode activeMode = m_config.mgpu_mode;
    CameraUniform uboSec = ubo;
    uint32_t useHwRT = 1;
    uint32_t hasEnvMap = m_environmentMap ? 1 : 0;
    float envIntensity = 1.0f;
    uint32_t envIntensityBits = std::bit_cast<uint32_t>(envIntensity);

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

        uint32_t qBase = m_currentFrame * QUERIES_PER_FRAME;
        vkCmdResetQueryPool(cmd, m_queryPool, qBase, QUERIES_PER_FRAME);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_queryPool, qBase + 0);

        // GPU-Timeline TLAS Update / Refit (Tier 3)
        if (m_tlasNeedsGpuUpdate && m_updateTlasPipeline && m_tlasInstanceBuffer && m_tlasScratchBuffer && m_tlas) {
            recordGpuTlasUpdate(cmd, true);
            m_tlasNeedsGpuUpdate = false;
        }

        // Update animated video billboard texture if a new frame is ready
        updateVideoBillboards(cmd);

        bool useWavefront = (m_config.pipeline_type == PipelineType::Wavefront && m_wavefrontPipeline);
        if (!skipRayTracing) {
            if (useWavefront) {
                if (m_config.enable_nrc && m_nrcManager) {
                    m_nrcManager->resetCounters(cmd);
                }

                // Clear per-frame ray tracing target (FP16)
                VkClearColorValue clearZero = { { 0.0f, 0.0f, 0.0f, 0.0f } };
                VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                vkCmdClearColorImage(cmd, m_frameImages[m_currentFrame]->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);

                bool needAccumReset = accumReset || !m_config.progressive_accumulation;
                if (needAccumReset) {
                    if (m_accumImage) {
                        vkCmdClearColorImage(cmd, m_accumImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
                    }
                    if (m_mlDiffuseImage) {
                        vkCmdClearColorImage(cmd, m_mlDiffuseImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
                    }
                    if (m_mlSpecularImage) {
                        vkCmdClearColorImage(cmd, m_mlSpecularImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
                    }
                }

                std::vector<VkImageMemoryBarrier2> clearBarriers;
                auto addClearBarrier = [&](Image* img) {
                    if (!img) return;
                    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                    b.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.image = img->getImage();
                    b.subresourceRange = clearRange;
                    clearBarriers.push_back(b);
                };
                addClearBarrier(m_frameImages[m_currentFrame].get());
                if (needAccumReset) {
                    addClearBarrier(m_accumImage.get());
                    addClearBarrier(m_mlDiffuseImage.get());
                    addClearBarrier(m_mlSpecularImage.get());
                }

                VkDependencyInfo clearDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                clearDep.imageMemoryBarrierCount = static_cast<uint32_t>(clearBarriers.size());
                clearDep.pImageMemoryBarriers = clearBarriers.data();
                vkCmdPipelineBarrier2(cmd, &clearDep);

            WavefrontSceneData wfSceneData{};
            wfSceneData.numTriangles = m_numTriangles;
            wfSceneData.numSpheres = m_numSpheres;
            wfSceneData.numMaterials = m_numMaterials;
            wfSceneData.numLights = m_numLights;
            wfSceneData.hasEnvMap = hasEnvMap;
            wfSceneData.envMapIntensity = envIntensity;
            wfSceneData.useHardwareRT = useHwRT;
            wfSceneData.frameIndex = m_frameIndex;
            wfSceneData.useMorton = m_config.use_morton ? 1u : 0u;
            wfSceneData.accumulateHistory = (m_config.progressive_accumulation && !accumReset) ? 1u : 0u;
            wfSceneData.sortMode = static_cast<uint32_t>(m_config.wavefront_sort_mode);
            wfSceneData.numOpaqueTriangles = m_numOpaqueTriangles;
            wfSceneData.secondarySortMode = static_cast<uint32_t>(m_config.secondary_sort_mode);
            wfSceneData.cameraFlags = flags;
            wfSceneData.enableNrc = m_config.enable_nrc;
            wfSceneData.nrcBounce = m_config.nrc_bounce;
            wfSceneData.nrcTrainRatio = m_config.nrc_train_ratio;
            wfSceneData.boundsMin = m_sceneData.boundsMin;
            wfSceneData.boundsMax = m_sceneData.boundsMax;
            wfSceneData.streamlineSecondaryShading = m_config.streamline_secondary_shading;
            wfSceneData.enableDistanceClamping = m_config.distance_clamping;
            wfSceneData.indirectClamp = m_config.indirect_clamp;
            uint32_t activeBatchCount = getEffectiveBatchCount(renderW, renderH);
            uint32_t activeBatchPixels = getEffectiveBatchPixels(renderW, renderH, activeBatchCount);
            if (activeBatchCount != m_currentBatchCount || activeBatchPixels != m_currentBatchPixels) {
                m_currentBatchCount = activeBatchCount;
                m_currentBatchPixels = activeBatchPixels;
                m_wavefrontPipeline->resize(renderW, renderH, activeBatchPixels);
            }

            wfSceneData.macroTileSize = m_config.macro_tile_size;
            wfSceneData.batchCount = activeBatchCount;
            wfSceneData.batchPixels = activeBatchPixels;
            wfSceneData.fullWidth = renderW;
            wfSceneData.captureMlData = (m_config.denoiser_mode == DenoiserMode::Upways ||
                                        m_config.upscaler_mode == UpscalerMode::Upways ||
                                        m_config.upways_superres ||
                                        !m_config.capture_training_data_dir.empty()) ? 1u : 0u;

            if (m_config.enable_caustics && m_sceneData.hasDielectrics && m_numLights > 0) {
                dispatchCausticTrace(cmd, m_currentFrame);
            }

            m_wavefrontPipeline->recordFrame(cmd, m_currentFrame, renderW, renderH,
                                             activeSpp, activeBounces, wfSceneData);

            if (m_config.enable_nrc && m_nrcManager) {
                m_nrcManager->recordInference(cmd, renderW, renderH,
                                              m_sceneData.boundsMin, m_sceneData.boundsMax);
                m_nrcManager->recordTraining(cmd, m_frameIndex,
                                             m_sceneData.boundsMin, m_sceneData.boundsMax,
                                             1e-3f, 1024);
            }

            if (m_config.enable_restir_di && m_normalDepthImage && m_prevNormalDepthImage) {
                VkImageCopy copyRegion{};
                copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                copyRegion.extent = { renderW, renderH, 1 };

                VkImageMemoryBarrier2 imgBarriers[2]{};
                imgBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                imgBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                imgBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                imgBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                imgBarriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                imgBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                imgBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                imgBarriers[0].image = m_normalDepthImage->getImage();
                imgBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

                imgBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                imgBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                imgBarriers[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                imgBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                imgBarriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                imgBarriers[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                imgBarriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
                imgBarriers[1].image = m_prevNormalDepthImage->getImage();
                imgBarriers[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

                VkDependencyInfo copyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                copyDep.imageMemoryBarrierCount = 2;
                copyDep.pImageMemoryBarriers = imgBarriers;
                vkCmdPipelineBarrier2(cmd, &copyDep);

                vkCmdCopyImage(cmd,
                               m_normalDepthImage->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                               m_prevNormalDepthImage->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                               1, &copyRegion);

                VkImageMemoryBarrier2 postBarrier{};
                postBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
                postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                postBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                postBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                postBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                postBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                postBarrier.image = m_prevNormalDepthImage->getImage();
                postBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

                VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                postDep.imageMemoryBarrierCount = 1;
                postDep.pImageMemoryBarriers = &postBarrier;
                vkCmdPipelineBarrier2(cmd, &postDep);
            }
        } else {
            if (m_config.enable_caustics && m_sceneData.hasDielectrics && m_numLights > 0) {
                dispatchCaustics(cmd, m_currentFrame);
            }
            // === DEDICATED HARDWARE RAY TRACING PIPELINE (VK_KHR_ray_tracing_pipeline) ===
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtpKhrPipeline->getPipeline());
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtpPipelineLayout, 0, 1, &m_rtDescSets[m_currentFrame], 0, nullptr);

            uint32_t traceW = renderW;
            uint32_t traceH = renderH;

            uint32_t fracSppBits = std::bit_cast<uint32_t>(activeFractionalSpp);
            uint32_t rtPushConstants[16] = {
                m_numTriangles, m_numSpheres, m_numMaterials, m_numLights,
                0, 0, traceW, traceH,
                useHwRT,
                hasEnvMap,
                envIntensityBits,
                (m_config.progressive_accumulation && !accumReset) ? 1u : 0u, // accumulateHistory
                fracSppBits,
                0u, m_numOpaqueTriangles, std::bit_cast<uint32_t>(m_config.indirect_clamp)
            };
            VkShaderStageFlags rtpStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
            vkCmdPushConstants(cmd, m_rtpPipelineLayout, rtpStages, 0, sizeof(rtPushConstants), rtPushConstants);

            m_rtpKhrPipeline->traceRays(cmd, traceW, traceH, 1);
        }
        if (m_governor) {
            m_governor->recordDispatch(m_currentFrame, activeSpp, activeBounces);
        }

        } // end if (!accumReachedCutoff)

        // Redundant post-RT barrier eliminated for wavefront mode: WavefrontPipeline::recordFrame
        // already terminates with a fully-scoped finalBarrier. Only emit for legacy RTP pipeline.
        if (!useWavefront) {
            VkMemoryBarrier2 memBarrier{};
            memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
            memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COPY_BIT;
            memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
            memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.memoryBarrierCount = 1;
            depInfo.pMemoryBarriers = &memBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_queryPool, qBase + 1);

        // 1-pixel GPU G-buffer depth readback of the center pixel for zero-overhead hardware camera adaptive speed
        if (!skipRayTracing && m_normalDepthImage && m_centerDepthBuffers[m_currentFrame]) {
            VkBufferImageCopy copyRegion{};
            copyRegion.bufferOffset = 0;
            copyRegion.bufferRowLength = 0;
            copyRegion.bufferImageHeight = 0;
            copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.imageOffset = { static_cast<int32_t>(renderW / 2), static_cast<int32_t>(renderH / 2), 0 };
            copyRegion.imageExtent = { 1, 1, 1 };

            VkImageMemoryBarrier2 preCopy{};
            preCopy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            preCopy.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            preCopy.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            preCopy.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            preCopy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            preCopy.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            preCopy.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            preCopy.image = m_normalDepthImage->getImage();
            preCopy.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo preDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            preDep.imageMemoryBarrierCount = 1;
            preDep.pImageMemoryBarriers = &preCopy;
            vkCmdPipelineBarrier2(cmd, &preDep);

            vkCmdCopyImageToBuffer(cmd, m_normalDepthImage->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                                   m_centerDepthBuffers[m_currentFrame]->getBuffer(), 1, &copyRegion);

            VkImageMemoryBarrier2 postCopy{};
            postCopy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            postCopy.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            postCopy.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            postCopy.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            postCopy.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            postCopy.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            postCopy.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            postCopy.image = m_normalDepthImage->getImage();
            postCopy.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            postDep.imageMemoryBarrierCount = 1;
            postDep.pImageMemoryBarriers = &postCopy;
            vkCmdPipelineBarrier2(cmd, &postDep);
        }

        bool needUpscaler = (m_config.denoiser_mode == DenoiserMode::Upways ||
                             m_config.upscaler_mode == UpscalerMode::Upways ||
                             m_config.upscaler_mode == UpscalerMode::FSR3);

        if (!needUpscaler && m_accumTonemapPipeline && !skipRayTracing) {
            // Fused Accumulation Running Average + ACES Tonemapping (Zero-Copy Register Pass Fusion)
            // Eliminates intermediate compute pipeline barrier and 132.7 MB round-trip VRAM read of m_accumImage.
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, qBase + 2);

            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_accumTonemapPipeline);
            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_accumTonemapPipelineLayout, 0, 1, &m_accumTonemapDescSets[m_currentFrame], 0, nullptr);

            struct {
                uint32_t width;
                uint32_t height;
                uint32_t sampleCount;
                float invSpp;
                float exposure;
                uint32_t applyACES;
                uint32_t displayMode;
                float peakNits;
                float paperWhiteNits;
                uint32_t pad;
            } fusedPC;

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

            vkCmdPushConstants(cmd, m_accumTonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(fusedPC), &fusedPC);
            vkCmdDispatch(cmd, (renderW + 15) / 16, (renderH + 15) / 16, 1);

            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 3);
        } else {
            // Decoupled Path (Used when Upways or FSR3 is active or RT was skipped)
            // Running Average Accumulation Pass (FP16 Frame -> FP32 Persistent History)
            if (!skipRayTracing && m_accumRunningAvgPipeline) {
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_accumRunningAvgPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_accumRunningAvgPipelineLayout, 0, 1, &m_accumRunningAvgDescSets[m_currentFrame], 0, nullptr);

                struct {
                    uint32_t width;
                    uint32_t height;
                    uint32_t sampleCount;
                    float invSpp;
                } avgPC;
                avgPC.width = renderW;
                avgPC.height = renderH;
                avgPC.sampleCount = (m_config.progressive_accumulation && !accumReset) ? m_accumulatedSamples : 1u;
                avgPC.invSpp = 1.0f;

                vkCmdPushConstants(cmd, m_accumRunningAvgPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(avgPC), &avgPC);
                vkCmdDispatch(cmd, (renderW + 15) / 16, (renderH + 15) / 16, 1);

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
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, qBase + 4);
                upwaysRun = dispatchUpways(cmd, resetTemporal);
                vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 5);
            }

            // AMD FidelityFX Super Resolution 3.1
            bool fsr3Run = false;
            if (!upwaysRun && m_config.upscaler_mode == UpscalerMode::FSR3) {
                fsr3Run = dispatchFsr3(cmd, resetTemporal);
            }

            // Tonemapping
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipeline);
            if (fsr3Run || (m_config.upscaler_mode == UpscalerMode::FSR3 && m_fsr3Upscaler)) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &m_tonemapFsr3DescSet, 0, nullptr);
                tonemapConstants.totalSamples = 1u;
            } else if (upwaysRun || (m_config.upscaler_mode == UpscalerMode::Upways && m_upwaysPipeline)) {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &m_tonemapUpwaysDescSet, 0, nullptr);
                tonemapConstants.totalSamples = 1u;
            } else {
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &m_tonemapDescSet, 0, nullptr);
                tonemapConstants.totalSamples = 1u;
            }

            uint32_t tmGroupsX = (m_config.width + 15) / 16;
            uint32_t tmGroupsY = (m_config.height + 15) / 16;

            tonemapConstants.visualizeSplit = 0;
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, qBase + 2);
            vkCmdPushConstants(cmd, m_tonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tonemapConstants), &tonemapConstants);
            vkCmdDispatch(cmd, tmGroupsX, tmGroupsY, 1);
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 3);
        }
    } else {
        // --- Multi-GPU Path (Checkerboard Tiling or Sample Parallelism) ---
        useHwRT = 1;
        hasEnvMap = m_environmentMap ? 1 : 0;
        envIntensity = 1.0f;
        envIntensityBits = std::bit_cast<uint32_t>(envIntensity);

        uint32_t formatMode = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? 0u : 1u;
        uint32_t bytesPerPixel = (formatMode == 0u) ? 8 : 16;
        frameBytes = 0;
        dstHost = nullptr;
        if (!m_mgpu->isZeroCopyActive()) {
            frameBytes = static_cast<size_t>(m_config.width) * m_config.height * bytesPerPixel;
            dstHost = m_secTransferBuffer ? m_secTransferBuffer->map() : nullptr;
        }

        activeMode = m_config.mgpu_mode;
        if (activeMode == MultiGpuMode::Auto) {
            activeMode = (activeSpp > 1) ? MultiGpuMode::SampleParallel : MultiGpuMode::CheckerboardTile;
        }
        if (activeMode != m_lastActiveMgpuMode) {
            accumReset = true;
            m_accumulatedSamples = 0;
            m_lastActiveMgpuMode = activeMode;
        }

        tileOffsetX_sec = 2u;
        tileOffsetY_sec = 0u;
        uint32_t tileOffsetX_prim = 1u;
        uint32_t tileOffsetY_prim = 0u;
        uint32_t mgpuBaseW = renderW;
        uint32_t mgpuBaseH = renderH;
        uint32_t dispatchWidth = mgpuBaseW;
        uint32_t dispatchHeight = (mgpuBaseH + 1) / 2;
        bool isSampleBlendFsr3 = (m_config.upscaler_mode == UpscalerMode::FSR3 && m_config.mgpu_upscale_mode == MgpuUpscaleMode::SampleBlend);
        uint32_t totalCompositeSpp = 0u;
        secAccumHistory = 0u;
        uint32_t mergeMode = 0u; // 0 = InterleavedScanline, 1 = CheckerboardTile, 2 = SampleParallel
        uint32_t primSpp = activeSpp;
        uint32_t secSpp = 0u;

        uboSec = ubo;

        uint32_t secDispatchWidth = dispatchWidth;
        uint32_t primDispatchWidth = dispatchWidth;

        if (activeMode == MultiGpuMode::CheckerboardTile) {
            mergeMode = 1u;
            tileOffsetX_sec = 2u;
            tileOffsetY_sec = m_config.tile_size;
            tileOffsetX_prim = 1u;
            tileOffsetY_prim = m_config.tile_size;
            uint32_t tileSize = (m_config.tile_size == 0u) ? 64u : m_config.tile_size;
            uint32_t numTilesX = (mgpuBaseW + tileSize - 1u) / tileSize;
            uint32_t maxTilesPerGpuX = (numTilesX + 1u) / 2u;
            secDispatchWidth = maxTilesPerGpuX * tileSize;
            bool needFullScreenPrimGbuffer = (m_config.upscaler_mode == UpscalerMode::FSR3 ||
                                              m_config.upscaler_mode == UpscalerMode::Upways ||
                                              m_config.denoiser_mode == DenoiserMode::Upways);
            primDispatchWidth = (needFullScreenPrimGbuffer && m_config.pipeline_type == PipelineType::Wavefront) ? mgpuBaseW : secDispatchWidth;
            dispatchWidth = primDispatchWidth;
            dispatchHeight = mgpuBaseH;
            secAccumHistory = 0u; // Secondary renders 1-frame delta; accum_running_avg accumulates merged frame
        } else if (activeMode == MultiGpuMode::SampleParallel) {
            mergeMode = 2u;
            tileOffsetX_sec = 0u;
            tileOffsetY_sec = 0u;
            tileOffsetX_prim = 0u;
            tileOffsetY_prim = 0u;
            secDispatchWidth = mgpuBaseW;
            primDispatchWidth = mgpuBaseW;
            dispatchWidth = mgpuBaseW;
            dispatchHeight = mgpuBaseH;
            secAccumHistory = isSampleBlendFsr3
                ? ((m_config.progressive_accumulation && !accumReset) ? 1u : 0u)
                : 0u; // Secondary only renders current frame's delta in PostMerge; accumulates in SampleBlend

            // Split SPP evenly: e.g. spp = 2 -> prim: 1, sec: 1; spp = 16 -> prim: 8, sec: 8
            uint32_t currentTotalSpp = activeSpp;
            if (isSampleBlendFsr3) {
                primSpp = std::max(1u, currentTotalSpp / 2u);
                secSpp = std::max(1u, currentTotalSpp / 2u);
            } else {
                primSpp = std::max(1u, (currentTotalSpp + 1) / 2);
                secSpp = std::max(1u, currentTotalSpp / 2);
            }
            if (m_governor && m_config.adaptive_spp && m_governor->getState().active) {
                primSpp = m_governor->getState().primSpp;
                secSpp = m_governor->getState().secSpp;
            }
            ubo.spp = primSpp;
            uboSec.spp = secSpp;

            // De-correlate secondary PRNG seed from primary
            uboSec.frameIndex = m_frameIndex + 1000003u;

            if (enableJitter) {
                uint32_t phaseOffset = (m_config.mgpu_mode == MultiGpuMode::SampleParallel) ? 4 : 0;
                uboSec = m_camera->getUniformData(m_frameIndex, secSpp, activeBounces, flags,
                                                  true, mgpuBaseW, mgpuBaseH, phaseOffset, false);
                uboSec.frameIndex = m_frameIndex + 1000003u;
            }

            totalCompositeSpp = (activeMode == MultiGpuMode::SampleParallel && !isSampleBlendFsr3) ? (primSpp + secSpp) : 0u;
            CameraUniform uboPrim = ubo;
            if (activeMode == MultiGpuMode::SampleParallel && m_config.pipeline_type == PipelineType::Wavefront && totalCompositeSpp > 0u) {
                uboPrim.spp = totalCompositeSpp;
            } else {
                uboPrim.spp = primSpp;
            }
            m_cameraUBOs[m_currentFrame]->copyFrom(&uboPrim, sizeof(CameraUniform));
        }

        if (m_mgpu) {
            m_mgpu->setConfig(m_config);
        }

        if (!m_mgpu->isZeroCopyActive()) {
            frameBytes = static_cast<size_t>(secDispatchWidth) * dispatchHeight * bytesPerPixel;
            dstHost = m_secTransferBuffer ? m_secTransferBuffer->map() : nullptr;
        }

        uint32_t slot = m_config.double_buffered_shared_mem ? (m_currentFrame % 2) : 0;

        // 1. Launch secondary GPU concurrently for current frame
        if (!skipRayTracing) {
            glm::vec2 jitterSec(0.0f);
            if (m_config.upscaler_mode == UpscalerMode::FSR3 || m_config.upscaler_mode == UpscalerMode::Upways) {
                if (!m_config.progressive_accumulation || m_cameraMovedLastFrame || m_accumulatedSamples <= 1) {
                    jitterSec = getHaltonJitter(m_frameIndex + 4);
                }
            }
            bool isSecMotion = cameraMovedThisFrame || m_cameraMovedLastFrame;
            m_mgpu->launchSecondaryWork(uboSec, slot, tileOffsetX_sec, tileOffsetY_sec, mgpuBaseW, mgpuBaseH,
                                       m_numTriangles, m_numSpheres, m_numMaterials, m_numLights, useHwRT,
                                       hasEnvMap, envIntensity, secAccumHistory, activeFractionalSpp, dstHost, frameBytes,
                                       totalCompositeSpp, m_numOpaqueTriangles,
                                       isSampleBlendFsr3, renderW, renderH, m_config.width, m_config.height,
                                       jitterSec, (hardReset || m_temporalResetRequested), isSecMotion, m_frameIndex,
                                       m_config.upscaler_sharpening, m_config.upscaler_sharpening ? m_config.upscaler_sharpness : 0.0f,
                                       (m_config.progressive_accumulation && m_accumulatedSamples > 0) ? m_accumulatedSamples : 1u);
        }

        // 2. Concurrently record and execute primary GPU ray tracing asynchronously
        vkResetCommandBuffer(cmd, 0);
        VkCommandBufferBeginInfo rtBeginInfo{};
        rtBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(cmd, &rtBeginInfo);

        uint32_t qBase = m_currentFrame * QUERIES_PER_FRAME;
        vkCmdResetQueryPool(cmd, m_queryPool, qBase, QUERIES_PER_FRAME);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_queryPool, qBase + 0);

        // GPU-Timeline TLAS Update / Refit (Tier 3)
        if (m_tlasNeedsGpuUpdate && m_updateTlasPipeline && m_tlasInstanceBuffer && m_tlasScratchBuffer && m_tlas) {
            recordGpuTlasUpdate(cmd, true);
            m_tlasNeedsGpuUpdate = false;
        }

        // Update animated video billboard texture if a new frame is ready
        updateVideoBillboards(cmd);

        uint32_t fracSppBits = std::bit_cast<uint32_t>(activeFractionalSpp);
        uint32_t rtPushConstants[16] = {
            m_numTriangles, m_numSpheres, m_numMaterials, m_numLights,
            tileOffsetX_prim,
            tileOffsetY_prim,
            mgpuBaseW, mgpuBaseH,
            useHwRT,
            hasEnvMap,
            envIntensityBits,
            (m_config.progressive_accumulation && !accumReset) ? 1u : 0u, // accumulateHistory
            fracSppBits,
            totalCompositeSpp,
            m_numOpaqueTriangles, std::bit_cast<uint32_t>(m_config.indirect_clamp)
        };

        // Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline) or Wavefront Pipeline
        if (!skipRayTracing) {
            bool useWavefront = (m_config.pipeline_type == PipelineType::Wavefront && m_wavefrontPipeline);
            uint32_t primDispatchSpp = (activeMode == MultiGpuMode::SampleParallel) ? primSpp : activeSpp;
            if (useWavefront) {
                if (m_config.enable_nrc && m_nrcManager) {
                    m_nrcManager->resetCounters(cmd);
                }

                // Clear per-frame ray tracing target (FP16)
                VkClearColorValue clearZero = { { 0.0f, 0.0f, 0.0f, 0.0f } };
                VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                vkCmdClearColorImage(cmd, m_frameImages[m_currentFrame]->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);

                bool needAccumReset = accumReset || !m_config.progressive_accumulation;
                if (needAccumReset) {
                    if (m_accumImage) {
                        vkCmdClearColorImage(cmd, m_accumImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
                    }
                    if (m_mlDiffuseImage) {
                        vkCmdClearColorImage(cmd, m_mlDiffuseImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
                    }
                    if (m_mlSpecularImage) {
                        vkCmdClearColorImage(cmd, m_mlSpecularImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
                    }
                }

                std::vector<VkImageMemoryBarrier2> clearBarriers;
                auto addClearBarrier = [&](Image* img) {
                    if (!img) return;
                    VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                    b.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                    b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                    b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                    b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                    b.image = img->getImage();
                    b.subresourceRange = clearRange;
                    clearBarriers.push_back(b);
                };
                addClearBarrier(m_frameImages[m_currentFrame].get());
                if (needAccumReset) {
                    addClearBarrier(m_accumImage.get());
                    addClearBarrier(m_mlDiffuseImage.get());
                    addClearBarrier(m_mlSpecularImage.get());
                }

                VkDependencyInfo clearDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                clearDep.imageMemoryBarrierCount = static_cast<uint32_t>(clearBarriers.size());
                clearDep.pImageMemoryBarriers = clearBarriers.data();
                vkCmdPipelineBarrier2(cmd, &clearDep);

                WavefrontSceneData wfSceneData{};
                wfSceneData.numTriangles = m_numTriangles;
                wfSceneData.numSpheres = m_numSpheres;
                wfSceneData.numMaterials = m_numMaterials;
                wfSceneData.numLights = m_numLights;
                wfSceneData.hasEnvMap = hasEnvMap;
                wfSceneData.envMapIntensity = envIntensity;
                wfSceneData.useHardwareRT = useHwRT;
                wfSceneData.frameIndex = m_frameIndex;
                wfSceneData.useMorton = m_config.use_morton ? 1u : 0u;
                wfSceneData.accumulateHistory = (m_config.progressive_accumulation && !accumReset) ? 1u : 0u;
                wfSceneData.sortMode = static_cast<uint32_t>(m_config.wavefront_sort_mode);
                wfSceneData.numOpaqueTriangles = m_numOpaqueTriangles;
                wfSceneData.secondarySortMode = static_cast<uint32_t>(m_config.secondary_sort_mode);
                wfSceneData.cameraFlags = flags;
                bool enableNrcThisFrame = m_config.enable_nrc;
                if (activeMode == MultiGpuMode::CheckerboardTile) {
                    enableNrcThisFrame = false; // Secondary GPU has no NRC inference network; disable to maintain tile parity
                }
                wfSceneData.enableNrc = enableNrcThisFrame;
                wfSceneData.nrcBounce = m_config.nrc_bounce;
                wfSceneData.nrcTrainRatio = m_config.nrc_train_ratio;
                wfSceneData.boundsMin = m_sceneData.boundsMin;
                wfSceneData.boundsMax = m_sceneData.boundsMax;
                wfSceneData.streamlineSecondaryShading = m_config.streamline_secondary_shading;
                wfSceneData.enableDistanceClamping = m_config.distance_clamping;
                wfSceneData.maxSecondaryRayDistance = m_config.max_secondary_distance;
                wfSceneData.indirectClamp = m_config.indirect_clamp;
                wfSceneData.inlineShadows = m_config.inline_primary_shadows;
                wfSceneData.tileOffsetX = tileOffsetX_prim;
                wfSceneData.tileOffsetY = tileOffsetY_prim;
                wfSceneData.fullWidth = mgpuBaseW;
                wfSceneData.captureMlData = (m_config.denoiser_mode == DenoiserMode::Upways ||
                                            m_config.upscaler_mode == UpscalerMode::Upways ||
                                            m_config.upways_superres ||
                                            !m_config.capture_training_data_dir.empty()) ? 1u : 0u;

                if (m_config.enable_caustics && m_sceneData.hasDielectrics && m_numLights > 0) {
                    dispatchCausticTrace(cmd, m_currentFrame);
                }

                uint32_t mgpuRequiredCapacity = dispatchWidth * dispatchHeight;
                if (m_wavefrontPipeline->getMaxCapacity() < mgpuRequiredCapacity) {
                    m_currentBatchPixels = mgpuRequiredCapacity;
                    m_currentBatchCount = 1;
                    m_wavefrontPipeline->resize(renderW, renderH, mgpuRequiredCapacity);
                }

                m_wavefrontPipeline->recordFrame(cmd, m_currentFrame, dispatchWidth, dispatchHeight,
                                                 primDispatchSpp, activeBounces, wfSceneData);

                if (enableNrcThisFrame && m_nrcManager) {
                    m_nrcManager->recordInference(cmd, mgpuBaseW, mgpuBaseH,
                                                  m_sceneData.boundsMin, m_sceneData.boundsMax);
                    m_nrcManager->recordTraining(cmd, m_frameIndex,
                                                 m_sceneData.boundsMin, m_sceneData.boundsMax,
                                                 1e-3f, 1024);
                }
            } else {
                if (m_config.enable_caustics && m_sceneData.hasDielectrics && m_numLights > 0) {
                    dispatchCaustics(cmd, m_currentFrame);
                }
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtpKhrPipeline->getPipeline());
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtpPipelineLayout, 0, 1, &m_rtDescSets[m_currentFrame], 0, nullptr);
                VkShaderStageFlags rtpStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
                vkCmdPushConstants(cmd, m_rtpPipelineLayout, rtpStages, 0, sizeof(rtPushConstants), rtPushConstants);
                m_rtpKhrPipeline->traceRays(cmd, dispatchWidth, dispatchHeight, 1);
            }
            if (m_governor) {
                m_governor->recordDispatch(m_currentFrame, primDispatchSpp, activeBounces);
            }

        } // end if (!accumReachedCutoff)

        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_queryPool, qBase + 1);
        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = cmd;

        VkSemaphoreSubmitInfo signalInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        signalInfo.semaphore = m_rtCompleteSemaphores[m_currentFrame];
        signalInfo.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

        VkSubmitInfo2 rtSubmit{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        rtSubmit.commandBufferInfoCount = 1;
        rtSubmit.pCommandBufferInfos = &cmdSubmitInfo;
        rtSubmit.signalSemaphoreInfoCount = 1;
        rtSubmit.pSignalSemaphoreInfos = &signalInfo;
        vkQueueSubmit2(queue, 1, &rtSubmit, VK_NULL_HANDLE);

        // 3. Concurrently record Merge & Tonemapping commands on primary GPU into postCmd
        activeCmd = m_postCommandBuffers[m_currentFrame];
        vkResetCommandBuffer(activeCmd, 0);
        VkCommandBufferBeginInfo postBeginInfo{};
        postBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(activeCmd, &postBeginInfo);

        // Barrier: Ensure primary RT writes to m_accumImage and secondary DMA host writes are visible before merge compute reads/writes
        VkMemoryBarrier2 rtToMergeBarrier{};
        rtToMergeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        rtToMergeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
        rtToMergeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
        rtToMergeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rtToMergeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo rtToMergeDep{};
        rtToMergeDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        rtToMergeDep.memoryBarrierCount = 1;
        rtToMergeDep.pMemoryBarriers = &rtToMergeBarrier;
        vkCmdPipelineBarrier2(activeCmd, &rtToMergeDep);

        // Merge Pass (Skipped for SampleBlend + FSR3 since both GPUs upscale to 4K independently before resolve)
        if (!skipRayTracing && !isSampleBlendFsr3) {
            vkCmdBindPipeline(activeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_mergePipeline);
            vkCmdBindDescriptorSets(activeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_mergePipelineLayout, 0, 1, &m_mergeDescSets[slot], 0, nullptr);

            uint32_t mergePC[8] = { mgpuBaseW, mgpuBaseH, secSpp, m_config.tile_size, formatMode, mergeMode, primSpp, secDispatchWidth };
            vkCmdPushConstants(activeCmd, m_mergePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mergePC), mergePC);

            uint32_t mergeGroupsX = (mgpuBaseW + 15) / 16;
            uint32_t mergeGroupsY = (mergeMode == 0u) ? (((mgpuBaseH + 1) / 2 + 15) / 16) : ((mgpuBaseH + 15) / 16);
            vkCmdDispatch(activeCmd, mergeGroupsX, mergeGroupsY, 1);
        }

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
        vkCmdPipelineBarrier2(activeCmd, &mergeDep);

        // In Multi-GPU mode, copy merged FP16 radiance into m_mlDiffuseImage for Upways neural reconstruction
        if (!skipRayTracing && m_mlDiffuseImage && (m_config.upscaler_mode == UpscalerMode::Upways || m_config.denoiser_mode == DenoiserMode::Upways)) {
            VkImageCopy copyRegion{};
            copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.extent = { mgpuBaseW, mgpuBaseH, 1 };
            vkCmdCopyImage(activeCmd, m_frameImages[slot]->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                           m_mlDiffuseImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);

            VkImageMemoryBarrier2 diffCopyBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            diffCopyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            diffCopyBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            diffCopyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            diffCopyBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            diffCopyBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            diffCopyBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            diffCopyBarrier.image = m_mlDiffuseImage->getImage();
            diffCopyBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo diffCopyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            diffCopyDep.imageMemoryBarrierCount = 1;
            diffCopyDep.pImageMemoryBarriers = &diffCopyBarrier;
            vkCmdPipelineBarrier2(activeCmd, &diffCopyDep);
        }

        // Running Average Accumulation Pass for Multi-GPU (FP16 Merged Frame -> FP32 Persistent History)
        if (!skipRayTracing && m_accumRunningAvgPipeline) {
            vkCmdBindPipeline(activeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_accumRunningAvgPipeline);
            vkCmdBindDescriptorSets(activeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_accumRunningAvgPipelineLayout, 0, 1, &m_accumRunningAvgDescSets[m_currentFrame], 0, nullptr);

            struct {
                uint32_t width;
                uint32_t height;
                uint32_t sampleCount;
                float invSpp;
            } avgPC;
            avgPC.width = mgpuBaseW;
            avgPC.height = mgpuBaseH;
            avgPC.sampleCount = (m_config.progressive_accumulation && !accumReset) ? m_accumulatedSamples : 1u;
            avgPC.invSpp = 1.0f;

            vkCmdPushConstants(activeCmd, m_accumRunningAvgPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(avgPC), &avgPC);
            vkCmdDispatch(activeCmd, (mgpuBaseW + 15) / 16, (mgpuBaseH + 15) / 16, 1);

            VkMemoryBarrier2 avgBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            avgBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            avgBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            avgBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            avgBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

            VkDependencyInfo avgDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            avgDep.memoryBarrierCount = 1;
            avgDep.pMemoryBarriers = &avgBarrier;
            vkCmdPipelineBarrier2(activeCmd, &avgDep);
        }

        // Upways Neural Denoiser & Super-Resolution (Wave32 WMMA)
        bool resetTemporal = hardReset || m_temporalResetRequested;
        m_temporalResetRequested = false;
        bool upwaysRun = false;
        if (m_config.denoiser_mode == DenoiserMode::Upways || m_config.upscaler_mode == UpscalerMode::Upways) {
            vkCmdWriteTimestamp2(activeCmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, qBase + 4);
            upwaysRun = dispatchUpways(activeCmd, resetTemporal);
            vkCmdWriteTimestamp2(activeCmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 5);
        }

        // AMD FidelityFX Super Resolution 3.1
        bool fsr3Run = false;
        if (!upwaysRun && m_config.upscaler_mode == UpscalerMode::FSR3) {
            fsr3Run = dispatchFsr3(activeCmd, resetTemporal);
        }

        // Tonemapping
        vkCmdBindPipeline(activeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipeline);
        if (fsr3Run || (m_config.upscaler_mode == UpscalerMode::FSR3 && m_fsr3Upscaler)) {
            vkCmdBindDescriptorSets(activeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &m_tonemapFsr3DescSet, 0, nullptr);
            tonemapConstants.totalSamples = 1u;
        } else if (upwaysRun || (m_config.upscaler_mode == UpscalerMode::Upways && m_upwaysPipeline)) {
            vkCmdBindDescriptorSets(activeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &m_tonemapUpwaysDescSet, 0, nullptr);
            tonemapConstants.totalSamples = 1u;
        } else {
            vkCmdBindDescriptorSets(activeCmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &m_tonemapDescSet, 0, nullptr);
            tonemapConstants.totalSamples = 1u;
        }

        uint32_t mgpuTmGroupsX = (m_config.width + 15) / 16;
        uint32_t mgpuTmGroupsY = (m_config.height + 15) / 16;

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
        vkCmdWriteTimestamp2(activeCmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, qBase + 2);
        vkCmdPushConstants(activeCmd, m_tonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tonemapConstants), &tonemapConstants);
        vkCmdDispatch(activeCmd, mgpuTmGroupsX, mgpuTmGroupsY, 1);

        vkCmdWriteTimestamp2(activeCmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 3);
    }

    // 3. Interactive Blit & Dear ImGui Overlay
    if (!m_config.headless && m_swapchain) {
        VkImage swapImage = m_swapchain->getImage(imageIndex);
        VkImageView swapView = m_swapchain->getImageView(imageIndex);

        // Transition m_outputImage to TRANSFER_SRC_OPTIMAL
        m_outputImage->transitionLayout(
            activeCmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
        vkCmdPipelineBarrier2(activeCmd, &depToDst);

        // Copy or blit output image to swapchain image
        bool extentsMatch = (m_config.width == m_swapchain->getExtent().width &&
                             m_config.height == m_swapchain->getExtent().height);
        bool formatsMatch = (m_outputImage->getFormat() == m_swapchain->getFormat());

        if (extentsMatch && formatsMatch) {
            VkImageCopy copyRegion{};
            copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.extent = { m_config.width, m_config.height, 1 };
            vkCmdCopyImage(activeCmd, m_outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
            vkCmdBlitImage(activeCmd, m_outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, filter);
        }

        // Transition m_outputImage back to GENERAL
        m_outputImage->transitionLayout(
            activeCmd, VK_IMAGE_LAYOUT_GENERAL,
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
        vkCmdPipelineBarrier2(activeCmd, &depToColor);

        // Render ImGui overlay
        if (m_gui) {
            m_gui->newFrame();
            bool prevMode = m_cameraMode;
            MultiGpuMode prevMgpuMode = m_config.mgpu_mode;
            GuiActions guiActions{};
            if (m_gui->render(activeCmd, swapView, m_swapchain->getExtent().width, m_swapchain->getExtent().height,
                              m_config, getStats(), m_cameraMode, m_camera.get(),
                              &m_window->getDisplayInfo(), m_window->isFullscreen(), &guiActions,
                              m_availableScenes, m_currentSceneIndex)) {
                m_resetAccumulation = true;
                if (!m_config.headless) m_frameTimesMs.clear();
            }
            if (guiActions.sceneChanged) {
                m_pendingSceneChange = true;
                m_pendingScenePath = guiActions.newScenePath;
                std::string targetLabel;
                for (const auto& entry : m_availableScenes) {
                    if (entry.filepath == guiActions.newScenePath) {
                        targetLabel = entry.label;
                        break;
                    }
                }
                if (targetLabel.empty()) {
                    if (guiActions.newScenePath.empty() || guiActions.newScenePath == "__procedural_cornell_box__") {
                        targetLabel = "Procedural Cornell Box";
                    } else if (guiActions.newScenePath == "procedural:many-lights" || guiActions.newScenePath == "many-lights" || guiActions.newScenePath == "many_lights") {
                        targetLabel = "Procedural Many-Lights";
                    } else if (guiActions.newScenePath == "procedural:cyber-city" || guiActions.newScenePath == "procedural:cyber_city" || guiActions.newScenePath == "cyber-city" || guiActions.newScenePath == "cyber_city" || guiActions.newScenePath == "Procedural Cyber City") {
                        targetLabel = "Procedural Cyber City";
                    } else if (guiActions.newScenePath == "procedural:infinity-mirror" || guiActions.newScenePath == "infinity-mirror" || guiActions.newScenePath == "procedural:infinity_mirror" || guiActions.newScenePath == "infinity_mirror" || guiActions.newScenePath == "Procedural Infinity Mirror" || guiActions.newScenePath == "Infinity Mirror") {
                        targetLabel = "Infinity Mirror";
                    } else {
                        targetLabel = SceneRegistry::formatSceneName(std::filesystem::path(guiActions.newScenePath).stem().string());
                    }
                }
                m_loadingSceneName = targetLabel;
                m_sceneLoadingStartTime = std::chrono::steady_clock::now();
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
            vkCmdPipelineBarrier2(activeCmd, &depToSrc);

            VkFormat swapFmt = m_swapchain->getFormat();
            size_t bpp = (swapFmt == VK_FORMAT_R16G16B16A16_SFLOAT) ? 8 : 4;
            if (!m_uiDumpBuffer) {
                VkDeviceSize size = static_cast<VkDeviceSize>(m_swapchain->getExtent().width) * m_swapchain->getExtent().height * bpp;
                m_uiDumpBuffer = std::make_unique<Buffer>(m_context->getAllocator(), size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                         VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
            }

            VkBufferImageCopy copyRegion{};
            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageExtent = { m_swapchain->getExtent().width, m_swapchain->getExtent().height, 1 };
            vkCmdCopyImageToBuffer(activeCmd, swapImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_uiDumpBuffer->getBuffer(), 1, &copyRegion);

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
            vkCmdPipelineBarrier2(activeCmd, &depToPresent);
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
            vkCmdPipelineBarrier2(activeCmd, &depToPresent);
        }
    }

    vkEndCommandBuffer(activeCmd);

    // Wait for secondary GPU completion of slot and PCIe transfer (if MGPU)
    if (isMgpu && !skipRayTracing) {
        uint32_t slot = m_config.double_buffered_shared_mem ? (m_currentFrame % 2) : 0;
        m_mgpu->syncAndTransfer(slot, dstHost, frameBytes);
    }

    // Submit Work
    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = activeCmd;

    std::vector<VkSemaphoreSubmitInfo> waitSemaphoreInfos;
    std::vector<VkSemaphoreSubmitInfo> signalSemaphoreInfos;

    if (isMgpu) {
        VkSemaphoreSubmitInfo waitRt{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        waitRt.semaphore = m_rtCompleteSemaphores[m_currentFrame];
        waitRt.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        waitSemaphoreInfos.push_back(waitRt);

        if (m_mgpu->isCrossGpuSyncActive() && !skipRayTracing) {
            uint32_t slot = m_config.double_buffered_shared_mem ? (m_currentFrame % 2) : 0;
            VkSemaphore secSem = m_mgpu->getImportedSemaphore(slot);
            if (secSem != VK_NULL_HANDLE) {
                VkSemaphoreSubmitInfo waitSec{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
                waitSec.semaphore = secSem;
                waitSec.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                waitSemaphoreInfos.push_back(waitSec);
            }
        }
    }

    if (!m_config.headless && m_swapchain) {
        VkSemaphoreSubmitInfo waitImg{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        waitImg.semaphore = m_imageAvailableSemaphores[m_currentFrame];
        waitImg.stageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT | VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT;
        waitSemaphoreInfos.push_back(waitImg);

        VkSemaphoreSubmitInfo sigRender{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        sigRender.semaphore = m_renderFinishedSemaphores[imageIndex];
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

    vkResetFences(device, 1, &m_inFlightFences[m_currentFrame]);
    vkQueueSubmit2(queue, 1, &submitInfo, m_inFlightFences[m_currentFrame]);

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

    // High-precision frame pacing if target FPS is set
    if (m_governor && m_config.target_fps > 0) {
        m_governor->paceFrame(m_currentFrameStartTime);
    } else if (accumReachedCutoff && !m_config.headless) {
        // When progressive accumulation cutoff is reached and scene is static, pace at 60 Hz
        // to prevent runaway CPU/GPU utilization presenting identical frames
        SDL_Delay(16);
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
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();
    VkQueue queue = m_context->getGraphicsQueue();

    vkDeviceWaitIdle(device);

    // Drain remaining in-flight queries after vkDeviceWaitIdle
    uint32_t drainCount = std::min(m_totalFramesRendered, MAX_FRAMES_IN_FLIGHT);
    for (uint32_t d = 0; d < drainCount; ++d) {
        uint32_t slot = (m_totalFramesRendered - drainCount + d) % MAX_FRAMES_IN_FLIGHT;
        uint32_t qBase = slot * QUERIES_PER_FRAME;
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

        uint32_t lastCompletedSlot = (m_totalFramesRendered - 1) % MAX_FRAMES_IN_FLIGHT;
        bool lastSlotSkipped = m_slotSkippedRayTracing[lastCompletedSlot];

        if (m_config.pipeline_type == PipelineType::Wavefront && m_wavefrontPipeline && m_totalFramesRendered > 0) {
            m_lastWavefrontProfile = m_wavefrontPipeline->getProfilingData(lastCompletedSlot, m_timestampPeriod, m_config.max_bounces);
        }

        if (!lastSlotSkipped && totalGpuMs > 0.01) {
            m_lastGpuRtMs = gpuRtMs;
            m_lastSecGpuMs = secGpuMs;
            m_lastTonemapMs = gpuTonemapMs;
            m_lastUpwaysMs = gpuUpwaysMs;
            m_lastFrameTimeMs = totalGpuMs;
            m_frameTimesMs.push_back(m_lastFrameTimeMs);

            WavefrontStageSample wfSample;
            if (m_lastWavefrontProfile.valid && m_config.pipeline_type == PipelineType::Wavefront) {
                wfSample.classifyMs = m_lastWavefrontProfile.classifyMs;
                uint32_t traceW = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
                    static_cast<uint32_t>(m_config.width * m_config.render_scale) : m_config.width;
                uint32_t traceH = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
                    static_cast<uint32_t>(m_config.height * m_config.render_scale) : m_config.height;
                wfSample.primaryRays = static_cast<uint64_t>(traceW) * traceH * m_config.spp;
                for (const auto& bp : m_lastWavefrontProfile.bounces) {
                    wfSample.bounces.push_back({bp.shadeMs, bp.shadowMs, bp.intersectMs, bp.gapBeforeShadeMs, bp.gapBeforeShadowMs, bp.gapBeforeIntersectMs, bp.activeCount, bp.nextCount, bp.shadowCount});
                }
            }
            recordFrameTally(totalGpuMs, gpuRtMs, secGpuMs, gpuTonemapMs, wfSample.bounces.empty() ? nullptr : &wfSample);
        }
    }

    // 1. Dump LDR PNG
    if (!m_config.dump_frame_path.empty()) {
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
                ImageDumper::savePNG(m_config.dump_frame_path, m_config.width, m_config.height, rgba8.data());
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
                ImageDumper::savePNG16(m_config.dump_frame_path, m_config.width, m_config.height, rgba16.data());
            }
            staging.unmap();
        } else if (outFmt == VK_FORMAT_R16G16B16A16_SFLOAT) {
            const uint16_t* halfPixels = static_cast<const uint16_t*>(staging.map());
            bool force8bit = m_config.dump_8bit_png;
            if (force8bit) {
                std::vector<uint8_t> rgba8(static_cast<size_t>(m_config.width) * m_config.height * 4);
                float invPaperWhite = 80.0f / (m_config.hdr_paper_white_nits > 0.0f ? m_config.hdr_paper_white_nits : 200.0f);
                for (size_t p = 0; p < static_cast<size_t>(m_config.width) * m_config.height; ++p) {
                    for (int c = 0; c < 3; ++c) {
                        float val = glm::detail::toFloat32(halfPixels[p * 4 + c]);
                        float srgb = std::pow(std::clamp(val * invPaperWhite, 0.0f, 1.0f), 1.0f / 2.2f);
                        rgba8[p * 4 + c] = static_cast<uint8_t>(std::clamp(srgb * 255.0f + 0.5f, 0.0f, 255.0f));
                    }
                    rgba8[p * 4 + 3] = 255;
                }
                ImageDumper::savePNG(m_config.dump_frame_path, m_config.width, m_config.height, rgba8.data());
            } else {
                std::vector<uint16_t> rgba16(static_cast<size_t>(m_config.width) * m_config.height * 4);
                float invPaperWhite = 80.0f / (m_config.hdr_paper_white_nits > 0.0f ? m_config.hdr_paper_white_nits : 200.0f);
                for (size_t p = 0; p < static_cast<size_t>(m_config.width) * m_config.height; ++p) {
                    for (int c = 0; c < 3; ++c) {
                        float val = glm::detail::toFloat32(halfPixels[p * 4 + c]);
                        float srgb = std::pow(std::clamp(val * invPaperWhite, 0.0f, 1.0f), 1.0f / 2.2f);
                        rgba16[p * 4 + c] = static_cast<uint16_t>(std::clamp(srgb * 65535.0f + 0.5f, 0.0f, 65535.0f));
                    }
                    rgba16[p * 4 + 3] = 65535;
                }
                ImageDumper::savePNG16(m_config.dump_frame_path, m_config.width, m_config.height, rgba16.data());
            }
            staging.unmap();
        } else {
            const uint8_t* pixels = static_cast<const uint8_t*>(staging.map());
            ImageDumper::savePNG(m_config.dump_frame_path, m_config.width, m_config.height, pixels);
            staging.unmap();
        }
    }

    // 2. Dump HDR OpenEXR
    if (!m_config.dump_hdr_path.empty()) {
        VkFormat accumFormat = m_accumImage ? m_accumImage->getFormat() : VK_FORMAT_R32G32B32A32_SFLOAT;
        bool isFp16 = (accumFormat == VK_FORMAT_R16G16B16A16_SFLOAT);
        VkDeviceSize bytesPerPixel = isFp16 ? (4 * sizeof(uint16_t)) : (4 * sizeof(float));
        VkDeviceSize bufferSize = m_config.width * m_config.height * bytesPerPixel;
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

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = m_commandBuffers[0];

        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        staging.invalidate();
        if (isFp16) {
            const uint16_t* halfPixels = static_cast<const uint16_t*>(staging.map());
            std::vector<float> floatPixels(static_cast<size_t>(m_config.width) * m_config.height * 4);
            for (size_t p = 0; p < floatPixels.size(); ++p) {
                floatPixels[p] = glm::detail::toFloat32(halfPixels[p]);
            }
            ImageDumper::saveEXR(m_config.dump_hdr_path, m_config.width, m_config.height, floatPixels.data());
            staging.unmap();
        } else {
            const float* floatPixels = static_cast<const float*>(staging.map());
            ImageDumper::saveEXR(m_config.dump_hdr_path, m_config.width, m_config.height, floatPixels);
            staging.unmap();
        }
    }

    // 3. Dump UI Viewport Backbuffer
    if (!m_config.dump_ui_path.empty() && m_uiDumpBuffer && m_swapchain) {
        uint32_t w = m_swapchain->getExtent().width;
        uint32_t h = m_swapchain->getExtent().height;
        VkFormat fmt = m_swapchain->getFormat();
        std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4);

        m_uiDumpBuffer->invalidate();
        if (fmt == VK_FORMAT_A2R10G10B10_UNORM_PACK32 || fmt == VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
            const uint32_t* raw32 = static_cast<const uint32_t*>(m_uiDumpBuffer->map());
            bool isRgb = (fmt == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
            float invPaperWhite = 1.0f / (m_config.hdr_paper_white_nits > 0.0f ? m_config.hdr_paper_white_nits : 200.0f);
            const float m1 = 0.1593017578125f;
            const float m2 = 78.84375f;
            const float c1 = 0.8359375f;
            const float c2 = 18.8515625f;
            const float c3 = 18.6875f;

            auto pqToSrgb = [&](float v) -> uint8_t {
                float v_pow = std::pow(std::max(v, 0.0f), 1.0f / m2);
                float num = std::max(v_pow - c1, 0.0f);
                float den = std::max(c2 - c3 * v_pow, 1e-6f);
                float linearNits = std::pow(num / den, 1.0f / m1) * 10000.0f;
                float srgb = std::pow(std::clamp(linearNits * invPaperWhite, 0.0f, 1.0f), 1.0f / 2.2f);
                return static_cast<uint8_t>(std::clamp(srgb * 255.0f + 0.5f, 0.0f, 255.0f));
            };

            for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
                uint32_t val = raw32[i];
                float c0 = static_cast<float>((val >> 20) & 0x3FF) / 1023.0f;
                float c1_val = static_cast<float>((val >> 10) & 0x3FF) / 1023.0f;
                float c2_val = static_cast<float>(val & 0x3FF) / 1023.0f;

                float r_norm = isRgb ? c0 : c2_val;
                float g_norm = c1_val;
                float b_norm = isRgb ? c2_val : c0;

                rgba[i * 4 + 0] = pqToSrgb(r_norm);
                rgba[i * 4 + 1] = pqToSrgb(g_norm);
                rgba[i * 4 + 2] = pqToSrgb(b_norm);
                rgba[i * 4 + 3] = 255;
            }
            m_uiDumpBuffer->unmap();
        } else if (fmt == VK_FORMAT_R16G16B16A16_SFLOAT) {
            const uint16_t* raw16 = static_cast<const uint16_t*>(m_uiDumpBuffer->map());
            float invPaperWhite = 80.0f / (m_config.hdr_paper_white_nits > 0.0f ? m_config.hdr_paper_white_nits : 200.0f);
            for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
                for (int c = 0; c < 3; ++c) {
                    float val = glm::detail::toFloat32(raw16[i * 4 + c]);
                    float srgb = std::pow(std::clamp(val * invPaperWhite, 0.0f, 1.0f), 1.0f / 2.2f);
                    rgba[i * 4 + c] = static_cast<uint8_t>(std::clamp(srgb * 255.0f + 0.5f, 0.0f, 255.0f));
                }
                rgba[i * 4 + 3] = 255;
            }
            m_uiDumpBuffer->unmap();
        } else {
            const uint8_t* raw = static_cast<const uint8_t*>(m_uiDumpBuffer->map());
            bool isBgra = (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_SRGB);
            for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
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
        }
        ImageDumper::savePNG(m_config.dump_ui_path, w, h, rgba.data());
    }

    // 4. Dump Stats JSON
    if (!m_config.dump_stats_path.empty()) {
        FrameStats stats = getStats();
        ImageDumper::saveStatsJSON(m_config.dump_stats_path, stats);
    }

    if (m_camera) {
        m_camera->advanceFrame();
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
    stats.total_samples = m_accumulatedSamples;
    stats.max_accum_frames = m_config.max_accum_frames;
    stats.accumulation_complete = m_accumulationComplete;
    stats.validation_errors = m_context->getValidationErrors();

    stats.is_scene_loading = m_isSceneLoading.load() || m_pendingSceneChange;
    stats.loading_scene_name = m_loadingSceneName;
    if (stats.is_scene_loading) {
        auto now = std::chrono::steady_clock::now();
        stats.loading_elapsed_sec = std::chrono::duration<float>(now - m_sceneLoadingStartTime).count();
    }

    stats.current_frame_time_ms = m_lastFrameTimeMs;
    stats.current_fps = m_lastFrameTimeMs > 0.0001 ? (1000.0 / m_lastFrameTimeMs) : 0.0;

    stats.presentation_time_ms = m_lastPresentationTimeMs;
    stats.presentation_fps = m_lastPresentationTimeMs > 0.0001 ? (1000.0 / m_lastPresentationTimeMs) : stats.current_fps;

    if (!m_presentationTimesMs.empty()) {
        double pSum = std::accumulate(m_presentationTimesMs.begin(), m_presentationTimesMs.end(), 0.0);
        double avgPresTime = pSum / m_presentationTimesMs.size();
        stats.avg_presentation_fps = avgPresTime > 0.0001 ? (1000.0 / avgPresTime) : stats.presentation_fps;
    } else {
        stats.avg_presentation_fps = stats.presentation_fps;
    }

    stats.target_fps = m_config.target_fps;
    stats.adaptive_spp = m_config.adaptive_spp;
    if (m_governor && m_config.adaptive_spp && m_governor->getState().active) {
        stats.dynamic_spp = m_governor->getState().currentSpp;
        stats.dynamic_bounces = m_governor->getState().currentBounces;
    } else {
        stats.dynamic_spp = m_config.spp;
        stats.dynamic_bounces = (m_dynamicWavefrontBounces > 0) ? m_dynamicWavefrontBounces : m_config.max_bounces;
    }

    if (!m_frameTimesMs.empty()) {
        double sum = std::accumulate(m_frameTimesMs.begin(), m_frameTimesMs.end(), 0.0);
        stats.avg_frame_time_ms = sum / m_frameTimesMs.size();
        stats.min_frame_time_ms = *std::min_element(m_frameTimesMs.begin(), m_frameTimesMs.end());
        stats.max_frame_time_ms = *std::max_element(m_frameTimesMs.begin(), m_frameTimesMs.end());
        stats.avg_fps = stats.avg_frame_time_ms > 0.0 ? 1000.0 / stats.avg_frame_time_ms : 0.0;
        stats.target_frame_time_ms = m_config.target_frame_time_ms;
        stats.target_achieved = (stats.avg_frame_time_ms < m_config.target_frame_time_ms);

        double frameTimeForThroughput = stats.current_frame_time_ms > 0.001 ? stats.current_frame_time_ms : stats.avg_frame_time_ms;
        uint32_t traceW = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
            static_cast<uint32_t>(m_config.width * m_config.render_scale) : m_config.width;
        uint32_t traceH = (m_config.render_scale < 1.0f && m_config.upscaler_mode != UpscalerMode::None) ?
            static_cast<uint32_t>(m_config.height * m_config.render_scale) : m_config.height;
        double raysPerFrame = static_cast<double>(traceW) * traceH * stats.dynamic_spp * stats.dynamic_bounces;
        stats.rays_per_second = (frameTimeForThroughput > 0.0) ? (raysPerFrame / (frameTimeForThroughput / 1000.0)) : 0.0;
    }

    switch (m_config.mgpu_mode) {
        case MultiGpuMode::CheckerboardTile: stats.mgpu_mode_str = "checkerboard_tile"; break;
        case MultiGpuMode::SampleParallel: stats.mgpu_mode_str = "sample_parallel"; break;
        case MultiGpuMode::Auto: stats.mgpu_mode_str = (m_config.spp > 1) ? "auto (sample_parallel)" : "auto (checkerboard_tile)"; break;
        default: stats.mgpu_mode_str = "single_gpu"; break;
    }

    if (m_mgpu && m_mgpu->isMultiGpuActive()) {
        stats.mgpu_transfer_mode_str = m_mgpu->getTransferModeString();
    }

    stats.pipeline_type_str = (m_config.pipeline_type == PipelineType::Wavefront) ? "wavefront" : "rtp";

    if (m_config.pipeline_type == PipelineType::Wavefront) {
        switch (m_config.wavefront_sort_mode) {
            case WavefrontSortMode::Archetype: stats.wavefront_stats.sort_mode_str = "archetype"; break;
            case WavefrontSortMode::Dual: stats.wavefront_stats.sort_mode_str = "dual"; break;
            default: stats.wavefront_stats.sort_mode_str = "none"; break;
        }
        switch (m_config.secondary_sort_mode) {
            case SecondarySortMode::DirectionalDGC: stats.wavefront_stats.secondary_sort_mode_str = "directional"; break;
            case SecondarySortMode::DirectCoherent: stats.wavefront_stats.secondary_sort_mode_str = "direct-coherent"; break;
            case SecondarySortMode::DirectCoherentK8: stats.wavefront_stats.secondary_sort_mode_str = "direct-coherent-k8"; break;
            default: stats.wavefront_stats.secondary_sort_mode_str = "none"; break;
        }
        if (m_lastWavefrontProfile.valid) {
            stats.wavefront_stats.valid = true;
            stats.wavefront_stats.total_ms = m_lastWavefrontProfile.totalMs;
            stats.wavefront_stats.classify_ms = m_lastWavefrontProfile.classifyMs;
            stats.wavefront_stats.resolve_ms = m_lastWavefrontProfile.resolveMs;
            stats.wavefront_stats.queue_memory_footprint_mb = m_lastWavefrontProfile.queueMemoryFootprintMb;
            stats.wavefront_stats.estimated_vram_traffic_mb = m_lastWavefrontProfile.estimatedVramTrafficMb;
            for (const auto& bp : m_lastWavefrontProfile.bounces) {
                FrameStats::BounceProfile bProf{};
                bProf.bounce = bp.bounce;
                bProf.shade_ms = bp.shadeMs;
                bProf.shadow_ms = bp.shadowMs;
                bProf.intersect_ms = bp.intersectMs;
                bProf.active_rays = bp.activeCount;
                bProf.shadow_rays = bp.shadowCount;
                bProf.next_rays = bp.nextCount;
                bProf.diff_rays = bp.diffCount;
                bProf.diel_rays = bp.dielCount;
                bProf.cond_rays = bp.condCount;
                bProf.comp_rays = bp.compCount;
                bProf.emis_rays = bp.emisCount;
                bProf.pass_rays = bp.passCount;
                stats.wavefront_stats.bounces.push_back(bProf);
            }
        }
    }

    stats.primary_gpu_time_ms = m_lastGpuRtMs;
    stats.secondary_gpu_time_ms = m_lastSecGpuMs;
    stats.tonemap_time_ms = m_lastTonemapMs;
    stats.upways_time_ms = m_lastUpwaysMs;
    stats.num_triangles = m_numTriangles;
    stats.num_instanced_triangles = m_numInstancedTriangles;
    stats.num_instances = m_numInstances;
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
    stats.dgc_preprocess = true;
    stats.has_subgroup_control = m_context->hasSubgroupSizeControl();
    stats.subgroup_size = 32;
    stats.has_dynamic_rendering = true;
    stats.has_timeline_semaphores = true;
    stats.has_sync2 = true;

    // 5. Engine Settings & State
    stats.visualize_mgpu_split = m_config.visualize_mgpu_split;
    switch (m_config.upscaler_mode) {
        case UpscalerMode::FSR3: stats.upscaler_mode_str = "FSR 3.1"; break;
        case UpscalerMode::Upways: stats.upscaler_mode_str = "Upways"; break;
        case UpscalerMode::None: default: stats.upscaler_mode_str = "None"; break;
    }
    stats.render_scale = m_config.render_scale;
    stats.max_bounces = m_config.max_bounces;
    stats.checkerboard_tile_size = m_config.tile_size;
    stats.enable_direct_light = m_config.enable_direct_light;
    stats.enable_indirect_light = m_config.enable_indirect_light;
    stats.enable_refraction = m_config.enable_refraction;
    stats.enable_shadows = m_config.enable_shadows;
    stats.aces_tonemap = m_config.aces_tonemap;
    if (m_swapchain && !m_config.headless) {
        stats.swapchain_format_str = m_swapchain->getFormatName();
        stats.swapchain_color_space_str = m_swapchain->getColorSpaceName();
        stats.is_hdr_display = m_swapchain->isHdr();
        stats.hdr_mode_str = (m_swapchain->getHdrMode() == HdrDisplayMode::scRGB) ? "scRGB Linear (16-bit Float)" :
                             ((m_swapchain->getHdrMode() == HdrDisplayMode::HDR10) ? "HDR10 PQ (10-bit Rec.2020)" : "SDR sRGB (8-bit)");
    } else {
        stats.swapchain_format_str = "R8G8B8A8_UNORM (Headless Offscreen)";
        stats.swapchain_color_space_str = "SRGB_NONLINEAR";
        stats.is_hdr_display = false;
        stats.hdr_mode_str = "Headless SDR";
    }
    stats.hdr_peak_nits = m_config.hdr_peak_nits;
    stats.hdr_paper_white_nits = m_config.hdr_paper_white_nits;
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
        if (tally.frameCount <= 2 && m_configTallies.size() > 1) {
            bool hasSubstantial = false;
            for (const auto& other : m_configTallies) {
                if (other.frameCount >= 5) {
                    hasSubstantial = true;
                    break;
                }
            }
            if (hasSubstantial) {
                continue;
            }
        }
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
        s.target_frame_time_ms = m_config.target_frame_time_ms;
        s.target_achieved = tally.isTargetAchieved(m_config.target_frame_time_ms);

        if (tally.hasWavefrontStages && tally.wavefrontSampleCount > 0) {
            s.pipeline_stages.is_wavefront = true;
            s.pipeline_stages.classify_ms = tally.getAvgClassifyMs();
            s.pipeline_stages.primary_rays = tally.getAvgPrimaryRays();
            s.pipeline_stages.tonemap_ms = tally.getAvgTonemapMs();
            auto bounces = tally.getAvgBounces();
            for (const auto& b : bounces) {
                FrameStats::StageBounceSummary sb;
                sb.bounce = b.bounce;
                sb.shade_ms = b.shadeMs;
                sb.shadow_ms = b.shadowMs;
                sb.intersect_ms = b.intersectMs;
                sb.total_bounce_ms = b.totalMs;
                sb.active_rays = b.activeCount;
                sb.rays_left = b.nextCount;
                sb.shadow_rays = b.shadowCount;
                s.pipeline_stages.bounces.push_back(sb);
            }
        } else {
            s.pipeline_stages.is_wavefront = false;
            s.pipeline_stages.ray_tracing_pass_ms = tally.getAvgPrimaryRtMs();
            s.pipeline_stages.tonemap_ms = tally.getAvgTonemapMs();
        }

        s.blas_build_time_ms = tally.blasBuildTimeMs;
        s.blas_size_kb = tally.blasSizeKb;
        s.blas_triangles = tally.blasTriangles;
        s.tlas_build_time_ms = tally.tlasBuildTimeMs;
        s.tlas_size_kb = tally.tlasSizeKb;
        s.tlas_instances = tally.tlasInstances;
        s.sec_blas_build_time_ms = tally.secBlasBuildTimeMs;
        s.sec_blas_size_kb = tally.secBlasSizeKb;
        s.sec_tlas_build_time_ms = tally.secTlasBuildTimeMs;
        s.sec_tlas_size_kb = tally.secTlasSizeKb;
        s.tlas_gpu_updates = tally.tlasGpuUpdateCount;

        stats.configurations_breakdown.push_back(std::move(s));
    }

    if (m_asManager) {
        stats.blas_build_time_ms = m_asManager->getLastBlasBuildTimeMs();
        stats.blas_size_kb = m_asManager->getBlasSizeKb();
        stats.blas_triangles = m_asManager->getBlasTriangles();
        stats.tlas_build_time_ms = m_asManager->getLastTlasBuildTimeMs();
        stats.tlas_size_kb = m_asManager->getTlasSizeKb();
        stats.tlas_instances = m_asManager->getTlasInstances();
    }
    if (m_mgpu && m_mgpu->getSecondaryAsManager()) {
        auto* secAs = m_mgpu->getSecondaryAsManager();
        stats.sec_blas_build_time_ms = secAs->getLastBlasBuildTimeMs();
        stats.sec_blas_size_kb = secAs->getBlasSizeKb();
        stats.sec_tlas_build_time_ms = secAs->getLastTlasBuildTimeMs();
        stats.sec_tlas_size_kb = secAs->getTlasSizeKb();
    }
    stats.tlas_gpu_updates = m_tlasGpuUpdateCount;

    return stats;
}

std::string Engine::exportTelemetry(const std::string& customPath) {
    std::string path = customPath.empty() ? ImageDumper::generateDefaultTelemetryPath() : customPath;
    FrameStats stats = getStats();
    if (m_telemetryWorker.joinable()) {
        m_telemetryWorker.join();
    }
    m_telemetryWorker = std::thread([path, stats]() {
        if (ImageDumper::saveStatsJSON(path, stats)) {
            Logger::info("Exported comprehensive telemetry dataset to: {}", path);
        } else {
            Logger::error("Failed to export telemetry dataset to: {}", path);
        }
    });
    return path;
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

    if (m_window && !m_config.custom_hdr_peak && m_window->getDisplayInfo().isDisplayHdrCapable &&
        m_window->getDisplayInfo().maxLuminanceNits > 0.0f) {
        m_config.hdr_peak_nits = m_window->getDisplayInfo().maxLuminanceNits;
    }

    // 1. Recreate Swapchain (reusing oldSwapchain for smooth compositor transition under Wayland)
    VkSwapchainKHR oldSwapchainHandle = m_swapchain ? m_swapchain->getSwapchain() : VK_NULL_HANDLE;
    auto newSwapchain = std::make_unique<Swapchain>(
        device,
        m_context->getPhysicalDevice(),
        m_surface,
        m_config.width,
        m_config.height,
        m_context->getGraphicsQueueFamily(),
        m_config.enable_hdr,
        m_window ? m_window->isFullscreen() : false,
        m_context.get(),
        m_window ? &m_window->getDisplayInfo() : nullptr,
        m_config.hdr_peak_nits,
        m_config.hdr_paper_white_nits,
        oldSwapchainHandle
    );
    m_swapchain = std::move(newSwapchain);
    m_config.width = m_swapchain->getExtent().width;
    m_config.height = m_swapchain->getExtent().height;

    // 2. Ensure render finished semaphores cover all swapchain images (never destroy in-use sync objects during resize)
    VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    uint32_t numSwapImages = m_swapchain->getImageCount();
    while (m_renderFinishedSemaphores.size() < numSwapImages) {
        VkSemaphore sem = VK_NULL_HANDLE;
        vkCreateSemaphore(device, &semInfo, nullptr, &sem);
        m_renderFinishedSemaphores.push_back(sem);
    }

    // 3. Recreate Accumulation & Output Images
    VkFormat frameFmt = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_frameImages[i] = std::make_unique<Image>(
            device, allocator, m_config.width, m_config.height,
            frameFmt,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
    }
    m_accumImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        frameFmt,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // Internal post-process and tonemapping target is strictly 10-bit (or 16-bit float for scRGB HDR).
    // The internal pipeline is never degraded to 8-bit; SDR 8-bit presentation occurs via blit to the swapchain.
    VkFormat outputFmt = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    if (m_swapchain && !m_config.headless && m_swapchain->isHdr() && m_swapchain->getFormat() == VK_FORMAT_R16G16B16A16_SFLOAT) {
        outputFmt = VK_FORMAT_R16G16B16A16_SFLOAT;
    }

    m_outputImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        outputFmt,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_motionVectorImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R16G16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_mlAlbedoRoughnessImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_mlSpecularMotionImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    m_mlDiffuseImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );
    m_mlSpecularImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // Transition images to GENERAL layout
    vkResetCommandBuffer(m_commandBuffers[0], 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_commandBuffers[0], &beginInfo);

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_frameImages[i]->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }

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

    m_motionVectorImage->transitionLayout(
        m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    if (m_mlAlbedoRoughnessImage) {
        m_mlAlbedoRoughnessImage->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }
    if (m_mlSpecularMotionImage) {
        m_mlSpecularMotionImage->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
        m_mlDiffuseImage->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
        m_mlSpecularImage->transitionLayout(
            m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }

    vkEndCommandBuffer(m_commandBuffers[0]);

    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = m_commandBuffers[0];

    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
    vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    // 5. Resize secondary GPU if active before updating merge descriptor set
    if (m_mgpu && m_mgpu->isMultiGpuActive()) {
        m_mgpu->resize(m_config.width, m_config.height);
    }

    destroyGBufferResources();
    createGBufferResources();
    destroyUpwaysResources();
    if (m_upwaysPipeline) {
        bool isSuperRes = m_config.upways_superres || (m_config.upscaler_mode == UpscalerMode::Upways);
        uint32_t inW = (m_config.render_scale < 1.0f && (m_config.upscaler_mode != UpscalerMode::None || m_config.upways_superres)) ?
            static_cast<uint32_t>(m_config.width * m_config.render_scale) : m_config.width;
        uint32_t inH = (m_config.render_scale < 1.0f && (m_config.upscaler_mode != UpscalerMode::None || m_config.upways_superres)) ?
            static_cast<uint32_t>(m_config.height * m_config.render_scale) : m_config.height;
        uint32_t outW = m_config.width;
        uint32_t outH = m_config.height;
        isSuperRes = (inW < outW || inH < outH || isSuperRes);
        m_upwaysPipeline->resize(inW, inH, outW, outH, isSuperRes);
    }
    createUpwaysResources();
    destroyFsr3Resources();
    if (m_fsr3Upscaler) {
        uint32_t renderW = (m_config.render_scale < 1.0f) ?
            static_cast<uint32_t>(m_config.width * m_config.render_scale) :
            m_config.width;
        uint32_t renderH = (m_config.render_scale < 1.0f) ?
            static_cast<uint32_t>(m_config.height * m_config.render_scale) :
            m_config.height;
        m_fsr3Upscaler->resize(renderW, renderH, m_config.width, m_config.height);
    }
    createFsr3Resources();
    destroyCausticsResources();
    createCausticsResources();

    if (m_nrcManager) {
        m_nrcManager->resize(m_config.width, m_config.height);
    }

    if (m_restirManager) {
        m_restirManager->resize(m_config.width, m_config.height);
    }

    if (m_wavefrontPipeline) {
        uint32_t batchCount = getEffectiveBatchCount(m_config.width, m_config.height);
        uint32_t batchPixels = getEffectiveBatchPixels(m_config.width, m_config.height, batchCount);
        m_currentBatchCount = batchCount;
        m_currentBatchPixels = batchPixels;
        m_wavefrontPipeline->resize(m_config.width, m_config.height, batchPixels);
    }

    updateAllImageDescriptors();
    updateMergeDescriptors();

    // 5. Reset UI dump buffer if allocated
    if (m_uiDumpBuffer) {
        m_uiDumpBuffer.reset();
    }

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
        m_configTallies.clear();
        m_frameTimesMs.clear();
    }
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

void Engine::recordFrameTally(double frameTimeMs, double primRtMs, double secRtMs, double tonemapMs,
                              const WavefrontStageSample* wfSample) {
    ConfigKey key;
    key.scene_name = getActiveSceneName();
    key.pipeline_type = m_config.pipeline_type;
    key.mgpu_mode = (m_mgpu && m_mgpu->isMultiGpuActive() && m_config.mgpu_mode != MultiGpuMode::Off) ? m_config.mgpu_mode : MultiGpuMode::Off;
    key.denoiser = (m_config.denoiser_mode == DenoiserMode::Upways) ? DenoiserMode::Upways : DenoiserMode::None;
    key.enable_nrc = m_config.enable_nrc;
    key.width = m_config.width;
    key.height = m_config.height;
    key.spp = (m_governor && m_config.adaptive_spp && m_governor->getState().active) ? m_governor->getState().currentSpp : m_config.spp;
    key.max_bounces = (m_governor && m_config.adaptive_spp && m_governor->getState().active) ? m_governor->getState().currentBounces : m_config.max_bounces;
    key.accum_format = m_config.accum_format;
    key.tile_size = m_config.tile_size;
    key.upscaler = m_config.upscaler_mode;
    key.mgpu_upscale_mode = m_config.mgpu_upscale_mode;
    key.render_scale = m_config.render_scale;

    auto updateAsMetrics = [this](ConfigStatsTally& t) {
        if (m_asManager) {
            t.blasBuildTimeMs = m_asManager->getLastBlasBuildTimeMs();
            t.blasSizeKb = m_asManager->getBlasSizeKb();
            t.uncompactedBlasSizeKb = m_asManager->getUncompactedBlasSizeKb();
            t.blasCompacted = m_asManager->isBlasCompacted();
            t.blasTriangles = m_asManager->getBlasTriangles();
            t.tlasBuildTimeMs = m_asManager->getLastTlasBuildTimeMs();
            t.tlasSizeKb = m_asManager->getTlasSizeKb();
            t.tlasInstances = m_asManager->getTlasInstances();
        }
        if (m_mgpu && m_mgpu->getSecondaryAsManager()) {
            auto* secAs = m_mgpu->getSecondaryAsManager();
            t.secBlasBuildTimeMs = secAs->getLastBlasBuildTimeMs();
            t.secBlasSizeKb = secAs->getBlasSizeKb();
            t.secTlasBuildTimeMs = secAs->getLastTlasBuildTimeMs();
            t.secTlasSizeKb = secAs->getTlasSizeKb();
        }
        t.tlasGpuUpdateCount = m_tlasGpuUpdateCount;
    };

    for (auto& tally : m_configTallies) {
        if (tally.key == key) {
            tally.addSample(frameTimeMs, primRtMs, secRtMs, tonemapMs, wfSample);
            updateAsMetrics(tally);
            return;
        }
    }
    ConfigStatsTally newTally(key);
    newTally.addSample(frameTimeMs, primRtMs, secRtMs, tonemapMs, wfSample);
    updateAsMetrics(newTally);
    m_configTallies.push_back(std::move(newTally));
}

void Engine::printExecutionSummary() const {
    Logger::info("========================================================================================");
    if (m_configTallies.empty()) {
        Logger::info("  Execution Summary: No frames rendered.");
        Logger::info("========================================================================================");
        return;
    }

    // Prune transient startup tallies (e.g. <= 2 frames sampled before window settled)
    // when a primary configuration exists
    std::vector<const ConfigStatsTally*> activeTallies;
    for (const auto& tally : m_configTallies) {
        if (tally.frameCount <= 2 && m_configTallies.size() > 1) {
            bool hasSubstantial = false;
            for (const auto& other : m_configTallies) {
                if (other.frameCount >= 5) {
                    hasSubstantial = true;
                    break;
                }
            }
            if (hasSubstantial) {
                continue; // Skip startup transient tally
            }
        }
        activeTallies.push_back(&tally);
    }

    if (activeTallies.empty()) {
        for (const auto& tally : m_configTallies) {
            activeTallies.push_back(&tally);
        }
    }

    Logger::info("  Pathways Hybrid Path Tracing Engine - Execution Summary ({} Configuration{})",
                 activeTallies.size(), activeTallies.size() == 1 ? "" : "s");
    Logger::info("========================================================================================");

    for (size_t i = 0; i < activeTallies.size(); ++i) {
        const auto& tally = *activeTallies[i];
        Logger::info("  [Config {}/{}] {}", i + 1, activeTallies.size(), tally.label);
        Logger::info("    Frames Sampled:      {}", tally.frameCount);
        Logger::info("    Average Frame Time:  {:.3f} ms ({:.1f} FPS)", tally.getAvgFrameTimeMs(), tally.getAvgFps());
        Logger::info("    Frame Time Range:    min: {:.3f} ms | max: {:.3f} ms", tally.minFrameTimeMs, tally.maxFrameTimeMs);
        Logger::info("    Acceleration Structures:");
        if (tally.blasCompacted) {
            double ratio = (1.0 - (tally.blasSizeKb / tally.uncompactedBlasSizeKb)) * 100.0;
            Logger::info("      - BLAS Build:        {:.3f} ms ({:.2f} KB compacted from {:.2f} KB, -{:.1f}%, {} Triangles)",
                         tally.blasBuildTimeMs, tally.blasSizeKb, tally.uncompactedBlasSizeKb, ratio, tally.blasTriangles);
        } else {
            Logger::info("      - BLAS Build:        {:.3f} ms ({:.2f} KB, {} Triangles)",
                         tally.blasBuildTimeMs, tally.blasSizeKb, tally.blasTriangles);
        }
        Logger::info("      - TLAS Build:        {:.3f} ms ({:.2f} KB, {} Instance{})",
                     tally.tlasBuildTimeMs, tally.tlasSizeKb, tally.tlasInstances, tally.tlasInstances == 1 ? "" : "s");
        if (tally.key.mgpu_mode != MultiGpuMode::Off && tally.secBlasBuildTimeMs > 0.0) {
            Logger::info("      - Secondary BLAS:    {:.3f} ms ({:.2f} KB)",
                         tally.secBlasBuildTimeMs, tally.secBlasSizeKb);
            Logger::info("      - Secondary TLAS:    {:.3f} ms ({:.2f} KB)",
                         tally.secTlasBuildTimeMs, tally.secTlasSizeKb);
        }
        Logger::info("      - GPU TLAS Updates:  {} update{}",
                     tally.tlasGpuUpdateCount, tally.tlasGpuUpdateCount == 1 ? "" : "s");

        if (tally.key.mgpu_mode != MultiGpuMode::Off) {
            Logger::info("    GPU Breakdown:       GPU 0 RT: {:.3f} ms | GPU 1 RT: {:.3f} ms | Tonemap: {:.3f} ms",
                         tally.getAvgPrimaryRtMs(), tally.getAvgSecondaryRtMs(), tally.getAvgTonemapMs());
            // Find single GPU baseline for speedup calculation
            double baselineMs = 0.0;
            for (const auto* other : activeTallies) {
                if (other->key.scene_name == tally.key.scene_name &&
                    other->key.pipeline_type == tally.key.pipeline_type &&
                    other->key.mgpu_mode == MultiGpuMode::Off &&
                    other->key.width == tally.key.width &&
                    other->key.height == tally.key.height &&
                    other->key.spp == tally.key.spp &&
                    other->key.max_bounces == tally.key.max_bounces &&
                    other->key.accum_format == tally.key.accum_format &&
                    other->key.denoiser == tally.key.denoiser &&
                    other->key.enable_nrc == tally.key.enable_nrc) {
                    baselineMs = other->getAvgFrameTimeMs();
                    break;
                }
            }
            if (baselineMs > 0.001) {
                double speedup = baselineMs / tally.getAvgFrameTimeMs();
                double efficiency = (speedup / 2.0) * 100.0;
                Logger::info("    Multi-GPU Scaling:   {:.2f}x speedup vs Single GPU ({:.1f}% efficiency)", speedup, efficiency);
            }
        } else {
            Logger::info("    GPU Breakdown:       GPU 0 RT: {:.3f} ms | Tonemap: {:.3f} ms | GPU 1: Standby",
                         tally.getAvgPrimaryRtMs(), tally.getAvgTonemapMs());
        }

        if (tally.hasWavefrontStages && tally.wavefrontSampleCount > 0) {
            bool inlineShadowsActive = m_config.inline_primary_shadows;
            Logger::info("    Pipeline Stages{}:", inlineShadowsActive ? " (Inline Hardware Shadows Active)" : "");
            uint64_t primaryRays = tally.getAvgPrimaryRays();
            auto bounces = tally.getAvgBounces();
            if (primaryRays == 0 && !bounces.empty() && bounces[0].activeCount > 0) {
                primaryRays = bounces[0].activeCount;
            }
            if (primaryRays > 0) {
                Logger::info("      - Classify (Primary RayGen): {:.3f} ms | {} rays left (100.0%)",
                             tally.getAvgClassifyMs(), formatRayCount(primaryRays));
            } else {
                Logger::info("      - Classify (Primary RayGen): {:.3f} ms", tally.getAvgClassifyMs());
            }
            for (const auto& b : bounces) {
                double pct = (primaryRays > 0) ? (100.0 * static_cast<double>(b.nextCount) / primaryRays) : 0.0;
                std::string shadowStr = (b.shadowMs > 0.0005) ? std::format("Shadow: {:.3f} ms", b.shadowMs)
                                      : (inlineShadowsActive ? "Shadow: Inline" : "Shadow: 0.000 ms");
                if (b.intersectMs > 0.0001) {
                    Logger::info("      - Bounce {}: Gap1: {:.3f}ms | Shade: {:.3f} ms | Gap2: {:.3f}ms | {} | Gap3: {:.3f}ms | Intersect: {:.3f} ms (Total: {:.3f} ms) | {} rays left ({:.1f}%)",
                                 b.bounce, b.gapBeforeShadeMs, b.shadeMs, b.gapBeforeShadowMs, shadowStr, b.gapBeforeIntersectMs, b.intersectMs, b.totalMs,
                                 formatRayCount(b.nextCount), pct);
                } else {
                    Logger::info("      - Bounce {}: Gap1: {:.3f}ms | Shade: {:.3f} ms | Gap2: {:.3f}ms | {} (Total: {:.3f} ms) | {} rays left ({:.1f}%)",
                                 b.bounce, b.gapBeforeShadeMs, b.shadeMs, b.gapBeforeShadowMs, shadowStr, b.totalMs,
                                 formatRayCount(b.nextCount), pct);
                }
            }
            Logger::info("      - Tonemap / Resolve:         {:.3f} ms", tally.getAvgTonemapMs());
        } else if (tally.key.pipeline_type == PipelineType::RTP) {
            Logger::info("    Pipeline Stages:");
            Logger::info("      - Ray Tracing Pass:          {:.3f} ms", tally.getAvgPrimaryRtMs());
            Logger::info("      - Tonemap / Resolve:         {:.3f} ms", tally.getAvgTonemapMs());
        }

        Logger::info("    Ray Throughput:      {:.2f} GigaRays/sec", tally.getRayThroughput() * 1e-9);
        Logger::info("    Sub-{:.1f}ms Budget:    {}", m_config.target_frame_time_ms,
                     tally.isTargetAchieved(m_config.target_frame_time_ms) ? "\033[32mACHIEVED\033[0m" : "\033[33mEXCEEDED\033[0m");
        if (i + 1 < activeTallies.size()) {
            Logger::info("  --------------------------------------------------------------------------------------");
        }
    }

    Logger::info("----------------------------------------------------------------------------------------");
    Logger::info("  Total Frames Rendered: {} | Validation Errors: {}", m_totalFramesRendered, m_context->getValidationErrors());
    Logger::info("========================================================================================");
}

void Engine::captureTrainingFrame(uint32_t frameIdx, bool isReference, uint32_t spp) {
    VkDevice device = m_context->getDevice();
    VkQueue queue = m_context->getGraphicsQueue();
    VmaAllocator allocator = m_context->getAllocator();
    uint32_t width = m_config.width;
    uint32_t height = m_config.height;

    if (!m_wavefrontPipeline || !m_accumImage || !m_camera) {
        Logger::error("Cannot capture training frame: Wavefront pipeline, accum image, or camera is null!");
        return;
    }

    uint32_t flags = 0;
    if (m_config.enable_direct_light)    flags |= (1 << 0);
    if (m_config.enable_indirect_light)  flags |= (1 << 1);
    flags |= (1 << 2); // Specular
    if (m_config.enable_refraction)      flags |= (1 << 3);
    if (m_config.enable_shadows)         flags |= (1 << 4);
    if (m_sceneHasNonOpaque)             flags |= (1 << 5);
    if (m_config.inline_primary_shadows) flags |= (1 << 6);
    if (m_config.enable_light_tree || (!m_sceneData.lightTreeNodes.empty() && m_config.enable_restir_di)) flags |= (1 << 7);
    if (m_config.enable_restir_di) flags |= (1 << 9);

    VkClearColorValue clearZero{};
    clearZero.float32[0] = 0.0f; clearZero.float32[1] = 0.0f; clearZero.float32[2] = 0.0f; clearZero.float32[3] = 0.0f;
    VkImageSubresourceRange colorRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    auto clearImage = [&](VkCommandBuffer cmd, Image* img) {
        if (!img) return;
        img->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkCmdClearColorImage(cmd, img->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearZero, 1, &colorRange);
        img->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    };

    Image* targetAccumImg = m_frameImages[0].get();
    bool isTargetFp16 = targetAccumImg && (targetAccumImg->getFormat() == VK_FORMAT_R16G16B16A16_SFLOAT);

    if (isReference) {
        // --- GROUND TRUTH REFERENCE CAPTURE PASS ---
        // 1. Clear targetAccumImg (m_frameImages[0]) to zero before progressive accumulation
        {
            VkCommandBuffer cmd = m_commandBuffers[0];
            vkResetCommandBuffer(cmd, 0);

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);
            clearImage(cmd, targetAccumImg);
            vkEndCommandBuffer(cmd);

            VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
            cmdSubmitInfo.commandBuffer = cmd;
            VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
            submitInfo.commandBufferInfoCount = 1;
            submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
            vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);
        }

        // 2. Accumulate in batches of 16 SPP
        uint32_t remainingSpp = spp;
        uint32_t currentSppOffset = 0;
        constexpr uint32_t BATCH_SIZE = 16;

        WavefrontSceneData wfSceneData{};
        wfSceneData.numTriangles = m_numTriangles;
        wfSceneData.numSpheres = m_numSpheres;
        wfSceneData.numMaterials = m_numMaterials;
        wfSceneData.numLights = m_numLights;
        wfSceneData.hasEnvMap = m_environmentMap ? 1u : 0u;
        wfSceneData.envMapIntensity = 1.0f;
        wfSceneData.useHardwareRT = 1u;
        wfSceneData.useMorton = m_config.use_morton ? 1u : 0u;
        wfSceneData.accumulateHistory = 1u;
        wfSceneData.sortMode = static_cast<uint32_t>(m_config.wavefront_sort_mode);
        wfSceneData.numOpaqueTriangles = m_numOpaqueTriangles;
        wfSceneData.secondarySortMode = static_cast<uint32_t>(m_config.secondary_sort_mode);
        wfSceneData.cameraFlags = flags;
        wfSceneData.boundsMin = m_sceneData.boundsMin;
        wfSceneData.boundsMax = m_sceneData.boundsMax;
        wfSceneData.streamlineSecondaryShading = m_config.streamline_secondary_shading;
        wfSceneData.enableDistanceClamping = m_config.distance_clamping;
        wfSceneData.maxSecondaryRayDistance = m_config.max_secondary_distance;
        wfSceneData.indirectClamp = m_config.indirect_clamp;
        wfSceneData.inlineShadows = m_config.inline_primary_shadows;
        wfSceneData.captureMlData = 0;

        uint32_t activeOfflineBounces = m_config.max_bounces;
        while (remainingSpp > 0) {
            uint32_t batchSpp = std::min(remainingSpp, BATCH_SIZE);
            wfSceneData.frameIndex = frameIdx * 10000 + currentSppOffset;

            // ubo with spp = 1 so samples accumulate full unscaled radiance; enable subpixel jitter for ground truth convergence
            CameraUniform ubo = m_camera->getUniformData(wfSceneData.frameIndex, 1, activeOfflineBounces, flags,
                                                         true, width, height, 0, /*updatePrev=*/false);
            m_cameraUBOs[0]->copyFrom(&ubo, sizeof(CameraUniform));

            VkCommandBuffer cmd = m_commandBuffers[0];
            vkResetCommandBuffer(cmd, 0);

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            m_wavefrontPipeline->recordFrame(cmd, 0, width, height, batchSpp, activeOfflineBounces, wfSceneData);

            vkEndCommandBuffer(cmd);

            VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
            cmdSubmitInfo.commandBuffer = cmd;
            VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
            submitInfo.commandBufferInfoCount = 1;
            submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
            vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);

            auto profData = m_wavefrontPipeline->getProfilingData(0, m_timestampPeriod, activeOfflineBounces);
            if (profData.valid && !profData.bounces.empty()) {
                uint32_t usedBounces = static_cast<uint32_t>(profData.bounces.size());
                if (profData.bounces.back().nextCount == 0) {
                    activeOfflineBounces = usedBounces;
                } else {
                    activeOfflineBounces = std::min(m_config.max_bounces, usedBounces + 4);
                }
            }

            remainingSpp -= batchSpp;
            currentSppOffset += batchSpp;
        }

        // 3. Read back targetAccumImg (m_frameImages[0]) and write PTTD reference file
        VkDeviceSize accumPixelBytes = isTargetFp16 ? (4 * sizeof(uint16_t)) : (4 * sizeof(float));
        VkDeviceSize stagingSize = static_cast<VkDeviceSize>(width) * height * accumPixelBytes;

        Buffer stagingAccum(allocator, stagingSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        {
            VkCommandBuffer cmd = m_commandBuffers[0];
            vkResetCommandBuffer(cmd, 0);

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);

            targetAccumImg->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

            VkBufferImageCopy copyRegion{};
            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageExtent = { width, height, 1 };

            vkCmdCopyImageToBuffer(cmd, targetAccumImg->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingAccum.getBuffer(), 1, &copyRegion);

            targetAccumImg->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            vkEndCommandBuffer(cmd);

            VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
            cmdSubmitInfo.commandBuffer = cmd;
            VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
            submitInfo.commandBufferInfoCount = 1;
            submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
            vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
            vkQueueWaitIdle(queue);
        }

        stagingAccum.invalidate();
        std::vector<uint16_t> refPayload(static_cast<size_t>(width) * height * 4);
        uint16_t sppFp16 = glm::detail::toFloat16(static_cast<float>(spp));

        if (isTargetFp16) {
            const uint16_t* src = static_cast<const uint16_t*>(stagingAccum.map());
            for (size_t p = 0; p < static_cast<size_t>(width) * height; ++p) {
                refPayload[p * 4 + 0] = src[p * 4 + 0];
                refPayload[p * 4 + 1] = src[p * 4 + 1];
                refPayload[p * 4 + 2] = src[p * 4 + 2];
                refPayload[p * 4 + 3] = sppFp16;
            }
            stagingAccum.unmap();
        } else {
            const float* src = static_cast<const float*>(stagingAccum.map());
            for (size_t p = 0; p < static_cast<size_t>(width) * height; ++p) {
                refPayload[p * 4 + 0] = glm::detail::toFloat16(src[p * 4 + 0]);
                refPayload[p * 4 + 1] = glm::detail::toFloat16(src[p * 4 + 1]);
                refPayload[p * 4 + 2] = glm::detail::toFloat16(src[p * 4 + 2]);
                refPayload[p * 4 + 3] = sppFp16;
            }
            stagingAccum.unmap();
        }

        glm::vec3 camPos = m_camera ? m_camera->getPosition() : glm::vec3(0.0f);
        float fov = m_camera ? m_camera->getFov() : 45.0f;
        float aspect = m_camera ? m_camera->getAspect() : (static_cast<float>(width) / height);
        float camPosArr[3] = { camPos.x, camPos.y, camPos.z };

        std::string refPath = std::format("{}/frame_{:05d}_reference.bin", m_config.capture_training_data_dir, frameIdx);
        ImageDumper::savePTTD(refPath, width, height, 4, 0 /* Float16 */,
                              frameIdx, spp, refPayload.data(), refPayload.size() * sizeof(uint16_t),
                              fov, aspect, camPosArr);
        Logger::info("  -> Saved reference: {} (4 channels, {} SPP)", refPath, spp);

    } else {
        // --- 1-SPP NOISY INPUT + ML FEATURE EXTRACTION PASS ---
        // 1. Clear targetAccumImg and ML images to zero
        VkCommandBuffer cmd = m_commandBuffers[0];
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        clearImage(cmd, targetAccumImg);
        clearImage(cmd, m_mlDiffuseImage.get());
        clearImage(cmd, m_mlSpecularImage.get());
        clearImage(cmd, m_mlAlbedoRoughnessImage.get());
        clearImage(cmd, m_mlSpecularMotionImage.get());
        clearImage(cmd, m_motionVectorImage.get());
        clearImage(cmd, m_normalDepthImage.get());

        // 2. Set camera UBO (advances m_prevViewProj to track true velocity)
        CameraUniform ubo = m_camera->getUniformData(frameIdx, 1, m_config.max_bounces, flags,
                                                     false, width, height, 0, /*updatePrev=*/true);
        m_cameraUBOs[0]->copyFrom(&ubo, sizeof(CameraUniform));

        // 3. Dispatch 1-SPP with captureMlData = 1
        WavefrontSceneData wfSceneData{};
        wfSceneData.numTriangles = m_numTriangles;
        wfSceneData.numSpheres = m_numSpheres;
        wfSceneData.numMaterials = m_numMaterials;
        wfSceneData.numLights = m_numLights;
        wfSceneData.hasEnvMap = m_environmentMap ? 1u : 0u;
        wfSceneData.envMapIntensity = 1.0f;
        wfSceneData.useHardwareRT = 1u;
        wfSceneData.frameIndex = frameIdx;
        wfSceneData.useMorton = m_config.use_morton ? 1u : 0u;
        wfSceneData.accumulateHistory = 0u;
        wfSceneData.sortMode = static_cast<uint32_t>(m_config.wavefront_sort_mode);
        wfSceneData.numOpaqueTriangles = m_numOpaqueTriangles;
        wfSceneData.secondarySortMode = static_cast<uint32_t>(m_config.secondary_sort_mode);
        wfSceneData.cameraFlags = flags;
        wfSceneData.boundsMin = m_sceneData.boundsMin;
        wfSceneData.boundsMax = m_sceneData.boundsMax;
        wfSceneData.streamlineSecondaryShading = m_config.streamline_secondary_shading;
        wfSceneData.enableDistanceClamping = m_config.distance_clamping;
        wfSceneData.maxSecondaryRayDistance = m_config.max_secondary_distance;
        wfSceneData.indirectClamp = m_config.indirect_clamp;
        wfSceneData.inlineShadows = m_config.inline_primary_shadows;
        wfSceneData.captureMlData = 1;

        m_wavefrontPipeline->recordFrame(cmd, 0, width, height, 1, m_config.max_bounces, wfSceneData);

        if (m_config.capture_channels >= 23 && m_restirManager && m_numLights > 0) {
            Buffer* rayGeom = m_wavefrontPipeline->getRayGeomQueue(0);
            Buffer* rayHit = m_wavefrontPipeline->getRayHitQueue(0);
            Buffer* pixelToRay = m_wavefrontPipeline->getPixelToRayQueue(0);
            Buffer* camUBO = m_cameraUBOs[0].get();
            Buffer* lightsBuf = m_lightBuffer.get();
            Buffer* matsBuf = m_materialBuffer.get();
            Buffer* ltBuf = m_lightTreeBuffer.get();
            VkImageView mvView = m_motionVectorImage ? m_motionVectorImage->getImageView() : VK_NULL_HANDLE;
            VkImageView ndView = m_normalDepthImage ? m_normalDepthImage->getImageView() : VK_NULL_HANDLE;
            VkImageView prevNdView = m_prevNormalDepthImage ? m_prevNormalDepthImage->getImageView() : ndView;
            VkImageView confView = (m_upwaysPipeline && m_upwaysPipeline->getConfidenceImage())
                ? m_upwaysPipeline->getConfidenceImage()->getImageView()
                : VK_NULL_HANDLE;

            bool hasLt = m_config.enable_light_tree || (!m_sceneData.lightTreeNodes.empty() && ltBuf != nullptr);
            m_restirManager->recordFrame(cmd, 0, width, height,
                                         m_numLights, static_cast<uint32_t>(m_sceneData.triangles.size()), hasLt,
                                         frameIdx, m_config.restir_di_m_cap,
                                         rayGeom, rayHit, pixelToRay,
                                         lightsBuf, matsBuf,
                                         camUBO, ltBuf,
                                         mvView, ndView, prevNdView,
                                         confView);
        }

        // 4. Staging copy for 6 ML images (and ReSTIR reservoir buffer if capture_channels >= 23)
        VkDeviceSize numPixels = static_cast<VkDeviceSize>(width) * height;
        VkDeviceSize rgba16Size = numPixels * 4 * sizeof(uint16_t);
        VkDeviceSize rg16Size   = numPixels * 2 * sizeof(uint16_t);

        VkDeviceSize offsetDiff = 0;
        VkDeviceSize offsetSpec = offsetDiff + rgba16Size;
        VkDeviceSize offsetAR   = offsetSpec + rgba16Size;
        VkDeviceSize offsetND   = offsetAR   + rgba16Size;
        VkDeviceSize offsetSM   = offsetND   + rgba16Size;
        VkDeviceSize offsetMV   = offsetSM   + rgba16Size;
        VkDeviceSize totalInputStagingSize = offsetMV + rg16Size;

        uint32_t outChannels = m_config.capture_channels;
        VkDeviceSize offsetRes = 0;
        VkDeviceSize resSize = 0;
        Buffer* resBuffer = nullptr;
        if (outChannels >= 23 && m_restirManager) {
            resBuffer = m_restirManager->getSpatialReservoirBuffer(0);
            if (resBuffer) {
                offsetRes = totalInputStagingSize;
                resSize = numPixels * sizeof(UnifiedReservoirPT);
                totalInputStagingSize += resSize;
            }
        }

        Buffer stagingInput(allocator, totalInputStagingSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        auto copyImgToBuffer = [&](Image* img, VkDeviceSize bufOffset) {
            if (!img) return;
            img->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

            VkBufferImageCopy region{};
            region.bufferOffset = bufOffset;
            region.bufferRowLength = width;
            region.bufferImageHeight = height;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = { width, height, 1 };
            vkCmdCopyImageToBuffer(cmd, img->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   stagingInput.getBuffer(), 1, &region);

            img->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        };

        copyImgToBuffer(m_mlDiffuseImage.get(), offsetDiff);
        copyImgToBuffer(m_mlSpecularImage.get(), offsetSpec);
        copyImgToBuffer(m_mlAlbedoRoughnessImage.get(), offsetAR);
        copyImgToBuffer(m_normalDepthImage.get(), offsetND);
        copyImgToBuffer(m_mlSpecularMotionImage.get(), offsetSM);
        copyImgToBuffer(m_motionVectorImage.get(), offsetMV);

        if (resBuffer) {
            VkBufferMemoryBarrier2 bufBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
            bufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bufBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            bufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            bufBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarrier.buffer = resBuffer->getBuffer();
            bufBarrier.offset = 0;
            bufBarrier.size = resSize;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.bufferMemoryBarrierCount = 1;
            depInfo.pBufferMemoryBarriers = &bufBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);

            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = offsetRes;
            copyRegion.size = resSize;
            vkCmdCopyBuffer(cmd, resBuffer->getBuffer(), stagingInput.getBuffer(), 1, &copyRegion);

            bufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            bufBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            bufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bufBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = cmd;
        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        stagingInput.invalidate();
        const uint8_t* basePtr = static_cast<const uint8_t*>(stagingInput.map());
        const uint16_t* diffPixels = reinterpret_cast<const uint16_t*>(basePtr + offsetDiff);
        const uint16_t* specPixels = reinterpret_cast<const uint16_t*>(basePtr + offsetSpec);
        const uint16_t* arPixels   = reinterpret_cast<const uint16_t*>(basePtr + offsetAR);
        const uint16_t* ndPixels   = reinterpret_cast<const uint16_t*>(basePtr + offsetND);
        const uint16_t* smPixels   = reinterpret_cast<const uint16_t*>(basePtr + offsetSM);
        const uint16_t* mvPixels   = reinterpret_cast<const uint16_t*>(basePtr + offsetMV);
        const UnifiedReservoirPT* resPixels = resBuffer ? reinterpret_cast<const UnifiedReservoirPT*>(basePtr + offsetRes) : nullptr;

        std::vector<uint16_t> inputPayload(static_cast<size_t>(numPixels) * outChannels);

        for (size_t p = 0; p < static_cast<size_t>(numPixels); ++p) {
            if (outChannels >= 20) {
                // Vector 0 (Ch 00..03): Diffuse Radiance RGB + Roughness
                inputPayload[p * outChannels + 0]  = diffPixels[p * 4 + 0]; // diffuse R
                inputPayload[p * outChannels + 1]  = diffPixels[p * 4 + 1]; // diffuse G
                inputPayload[p * outChannels + 2]  = diffPixels[p * 4 + 2]; // diffuse B
                inputPayload[p * outChannels + 3]  = arPixels[p * 4 + 3];   // roughness
                // Vector 1 (Ch 04..07): Specular Radiance RGB + Metallic
                inputPayload[p * outChannels + 4]  = specPixels[p * 4 + 0]; // specular R
                inputPayload[p * outChannels + 5]  = specPixels[p * 4 + 1]; // specular G
                inputPayload[p * outChannels + 6]  = specPixels[p * 4 + 2]; // specular B
                inputPayload[p * outChannels + 7]  = diffPixels[p * 4 + 3]; // metallic
                // Vector 2 (Ch 08..11): Base Color Albedo RGB + Linear Depth
                inputPayload[p * outChannels + 8]  = arPixels[p * 4 + 0];   // albedo R
                inputPayload[p * outChannels + 9]  = arPixels[p * 4 + 1];   // albedo G
                inputPayload[p * outChannels + 10] = arPixels[p * 4 + 2];   // albedo B
                inputPayload[p * outChannels + 11] = ndPixels[p * 4 + 3];   // linear depth
                // Vector 3 (Ch 12..15): Surface Motion XY + Specular Motion XY
                inputPayload[p * outChannels + 12] = mvPixels[p * 2 + 0];   // surface motion X
                inputPayload[p * outChannels + 13] = mvPixels[p * 2 + 1];   // surface motion Y
                inputPayload[p * outChannels + 14] = smPixels[p * 4 + 0];   // specular motion X
                inputPayload[p * outChannels + 15] = smPixels[p * 4 + 1];   // specular motion Y
                // Vector 4 (Ch 16..19): Surface Normal XYZ + Specular Hit Distance
                inputPayload[p * outChannels + 16] = ndPixels[p * 4 + 0];   // normal X
                inputPayload[p * outChannels + 17] = ndPixels[p * 4 + 1];   // normal Y
                inputPayload[p * outChannels + 18] = ndPixels[p * 4 + 2];   // normal Z
                inputPayload[p * outChannels + 19] = smPixels[p * 4 + 2];   // specular hit distance

                if (outChannels >= 23) {
                    if (resPixels) {
                        const auto& res = resPixels[p];
                        uint32_t M = (res.lightIndex_M >> 16) & 0xFFFFu;
                        float res_m = std::clamp(static_cast<float>(M) / 64.0f, 0.0f, 1.0f);
                        float pHat = std::max(res.targetPdf, 1e-4f);
                        float res_w = std::clamp(res.wSum / (std::max(static_cast<float>(M), 1.0f) * pHat), 0.0f, 4.0f) * 0.25f;
                        uint32_t pathLen = (res.flags_uv_age >> 9) & 0x3u;
                        float res_is_gi = (pathLen >= 2u) ? 1.0f : 0.0f;

                        inputPayload[p * outChannels + 20] = glm::packHalf1x16(res_m);
                        inputPayload[p * outChannels + 21] = glm::packHalf1x16(res_w);
                        inputPayload[p * outChannels + 22] = glm::packHalf1x16(res_is_gi);
                    } else {
                        inputPayload[p * outChannels + 20] = glm::packHalf1x16(0.04f);
                        inputPayload[p * outChannels + 21] = glm::packHalf1x16(0.04f);
                        inputPayload[p * outChannels + 22] = glm::packHalf1x16(0.0f);
                    }
                }
            } else {
                inputPayload[p * outChannels + 0]  = diffPixels[p * 4 + 0]; // diffuse R
                inputPayload[p * outChannels + 1]  = diffPixels[p * 4 + 1]; // diffuse G
                inputPayload[p * outChannels + 2]  = diffPixels[p * 4 + 2]; // diffuse B
                inputPayload[p * outChannels + 3]  = specPixels[p * 4 + 0]; // specular R
                inputPayload[p * outChannels + 4]  = specPixels[p * 4 + 1]; // specular G
                inputPayload[p * outChannels + 5]  = specPixels[p * 4 + 2]; // specular B
                inputPayload[p * outChannels + 6]  = arPixels[p * 4 + 0];   // albedo R
                inputPayload[p * outChannels + 7]  = arPixels[p * 4 + 1];   // albedo G
                inputPayload[p * outChannels + 8]  = arPixels[p * 4 + 2];   // albedo B
                inputPayload[p * outChannels + 9]  = arPixels[p * 4 + 3];   // roughness
                inputPayload[p * outChannels + 10] = ndPixels[p * 4 + 3];   // linear depth
                inputPayload[p * outChannels + 11] = smPixels[p * 4 + 2];   // specular hit distance
                inputPayload[p * outChannels + 12] = mvPixels[p * 2 + 0];   // surface motion X
                inputPayload[p * outChannels + 13] = mvPixels[p * 2 + 1];   // surface motion Y
                inputPayload[p * outChannels + 14] = smPixels[p * 4 + 0];   // specular motion X
                inputPayload[p * outChannels + 15] = smPixels[p * 4 + 1];   // specular motion Y
                if (outChannels >= 19) {
                    inputPayload[p * outChannels + 16] = ndPixels[p * 4 + 0]; // normal X
                    inputPayload[p * outChannels + 17] = ndPixels[p * 4 + 1]; // normal Y
                    inputPayload[p * outChannels + 18] = ndPixels[p * 4 + 2]; // normal Z
                }
            }
        }
        stagingInput.unmap();

        glm::vec3 camPos = m_camera ? m_camera->getPosition() : glm::vec3(0.0f);
        float fov = m_camera ? m_camera->getFov() : 45.0f;
        float aspect = m_camera ? m_camera->getAspect() : (static_cast<float>(width) / height);
        float camPosArr[3] = { camPos.x, camPos.y, camPos.z };

        std::string inpPath = std::format("{}/frame_{:05d}_input.bin", m_config.capture_training_data_dir, frameIdx);
        ImageDumper::savePTTD(inpPath, width, height, outChannels, 0 /* Float16 */,
                              frameIdx, 1, inputPayload.data(), inputPayload.size() * sizeof(uint16_t),
                              fov, aspect, camPosArr);
        Logger::info("  -> Saved input    : {} ({} channels, 1 SPP)", inpPath, outChannels);
    }
}

void Engine::updateGamingChoreography(Camera* camera, uint32_t frameIdx, uint32_t totalFrames, const std::string& sceneName) {
    if (!camera) return;

    if (!m_choreoInitialized) {
        m_choreoInitialPos = camera->getPosition();
        m_choreoInitialYaw = camera->getYaw();
        m_choreoInitialPitch = camera->getPitch();
        m_choreoInitialFov = camera->getFov();
        m_choreoInitialized = true;
        Logger::info("[Gaming Choreography] Initialized baseline camera at pos=({:.2f}, {:.2f}, {:.2f}), yaw={:.1f}°, pitch={:.1f}°, fov={:.1f}° for {}",
                     m_choreoInitialPos.x, m_choreoInitialPos.y, m_choreoInitialPos.z,
                     m_choreoInitialYaw, m_choreoInitialPitch, m_choreoInitialFov, sceneName);
    }

    if (totalFrames <= 1) return;

    // Basis vectors in local coordinate system
    float radYaw = glm::radians(m_choreoInitialYaw);
    float radPitch = glm::radians(m_choreoInitialPitch);
    glm::vec3 forward0(
        std::cos(radYaw) * std::cos(radPitch),
        std::sin(radPitch),
        std::sin(radYaw) * std::cos(radPitch)
    );
    forward0 = glm::normalize(forward0);

    glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
    glm::vec3 right0 = glm::normalize(glm::cross(forward0, worldUp));
    glm::vec3 groundForward0 = glm::normalize(glm::cross(worldUp, right0));

    glm::vec3 pos = m_choreoInitialPos;
    float yaw = m_choreoInitialYaw;
    float pitch = m_choreoInitialPitch;
    float fov = m_choreoInitialFov;

    constexpr float PI = 3.14159265358979323846f;

    // Normalize frame progress across 240 frames
    float normalizedFrame = (static_cast<float>(frameIdx) / static_cast<float>(totalFrames)) * 240.0f;

    // Landmark positions for phase continuity
    const float walkDist = 0.22f;
    const float sprintDist = 0.38f;
    const float jumpForwardDist = 0.15f;
    const float crawlDist = 0.08f;

    glm::vec3 pEndPhase1 = m_choreoInitialPos + groundForward0 * walkDist;
    glm::vec3 pEndPhase2 = pEndPhase1 + groundForward0 * sprintDist;
    glm::vec3 pEndPhase3 = pEndPhase2 + groundForward0 * jumpForwardDist;
    glm::vec3 pEndPhase5 = pEndPhase3 + groundForward0 * crawlDist;

    if (normalizedFrame < 40.0f) {
        // --- PHASE 1 (Frames 0..39): Stationary warmup (0..9) -> Walking acceleration (10..39) ---
        if (normalizedFrame >= 10.0f) {
            float u = (normalizedFrame - 10.0f) / 30.0f; // 0..1
            float easeU = u * u; // gentle acceleration
            float d = walkDist * easeU;
            float bob = 0.015f * std::sin(2.0f * PI * (normalizedFrame - 10.0f) / 12.0f) * u;
            float glance = 1.0f * std::sin(2.0f * PI * (normalizedFrame - 10.0f) / 24.0f) * u;

            pos = m_choreoInitialPos + groundForward0 * d + worldUp * bob;
            yaw = m_choreoInitialYaw + glance;
        }
    } else if (normalizedFrame < 90.0f) {
        // --- PHASE 2 (Frames 40..89): Sprint & Lateral Strafe + Sprint FOV (+5 deg) ---
        float u = (normalizedFrame - 40.0f) / 50.0f; // 0..1
        float d = sprintDist * u;
        float strafe = 0.16f * std::sin(2.0f * PI * u * 1.5f);
        float bob = 0.025f * std::sin(2.0f * PI * (normalizedFrame - 40.0f) / 8.0f);
        float dynamicFov = 5.0f * std::sin(PI * u);
        float counterYaw = -1.8f * std::sin(2.0f * PI * u * 1.5f);

        pos = pEndPhase1 + groundForward0 * d + right0 * strafe + worldUp * bob;
        yaw = m_choreoInitialYaw + counterYaw;
        fov = m_choreoInitialFov + dynamicFov;
    } else if (normalizedFrame < 130.0f) {
        // --- PHASE 3 (Frames 90..129): Vertical Jump & Landing Shock ---
        float fInPhase = normalizedFrame - 90.0f;
        float forwardProgress = (fInPhase / 40.0f) * jumpForwardDist;
        glm::vec3 baseP = pEndPhase2 + groundForward0 * forwardProgress;

        if (fInPhase < 24.0f) {
            // Parabolic Jump Ascent & Descent
            float tau = fInPhase / 24.0f; // 0..1
            float jumpHeight = 4.0f * 0.28f * tau * (1.0f - tau);
            float apexFloorLook = -4.0f * std::sin(PI * tau);

            pos = baseP + worldUp * jumpHeight;
            pitch = m_choreoInitialPitch + apexFloorLook;
        } else {
            // Landing Shock (Damped Spring Squish Oscillation)
            float tL = fInPhase - 24.0f;
            float decay = std::exp(-0.25f * tL);
            float squishY = -0.045f * decay * std::cos(2.0f * PI * tL / 5.0f);
            float chinNod = 3.5f * decay * std::sin(2.0f * PI * tL / 4.5f);

            pos = baseP + worldUp * squishY;
            pitch = m_choreoInitialPitch + chinNod;
        }
    } else if (normalizedFrame < 170.0f) {
        // --- PHASE 4 (Frames 130..169): Rapid Twitch Mouse Flick (+55 deg in 4 frames) & Snap Return ---
        pos = pEndPhase3;
        float fInPhase = normalizedFrame - 130.0f;

        float flickYaw = 0.0f;
        if (fInPhase < 4.0f) {
            // Pre-flick stationary pause
            flickYaw = 0.0f;
        } else if (fInPhase < 8.0f) {
            // 4-frame high-velocity flick (+13.75 deg / frame)
            float t = (fInPhase - 4.0f) / 4.0f;
            flickYaw = 55.0f * (t * t * (3.0f - 2.0f * t)); // smooth cubic flick
        } else if (fInPhase < 18.0f) {
            // Snap pause with micro-tremor
            float snapT = fInPhase - 8.0f;
            float microSway = 0.6f * std::sin(2.0f * PI * snapT / 5.0f);
            flickYaw = 55.0f + microSway;
        } else if (fInPhase < 22.0f) {
            // 4-frame rapid return flick back to 0 deg
            float t = (fInPhase - 18.0f) / 4.0f;
            float s = t * t * (3.0f - 2.0f * t);
            flickYaw = 55.0f * (1.0f - s);
        } else {
            // Stabilization
            flickYaw = 0.0f;
        }

        yaw = m_choreoInitialYaw + flickYaw;
    } else if (normalizedFrame < 210.0f) {
        // --- PHASE 5 (Frames 170..209): Aim Down Sights (ADS) Dynamic Zoom (45 -> 24 deg) & Tactical Crawl ---
        float fInPhase = normalizedFrame - 170.0f;
        float targetFov = 24.0f;

        if (fInPhase < 10.0f) {
            // Smooth ADS Zoom In
            float s = fInPhase / 10.0f;
            s = s * s * (3.0f - 2.0f * s);
            fov = m_choreoInitialFov - (m_choreoInitialFov - targetFov) * s;
            pos = pEndPhase3;
        } else if (fInPhase < 28.0f) {
            // Full ADS hold + tactical crawl + breathing sway
            fov = targetFov;
            float crawlT = (fInPhase - 10.0f) / 18.0f;
            float crawlProgress = crawlDist * crawlT;
            float breathYaw = 0.25f * std::sin(2.0f * PI * (fInPhase - 10.0f) / 14.0f);
            float breathPitch = 0.20f * std::cos(2.0f * PI * (fInPhase - 10.0f) / 14.0f);

            pos = pEndPhase3 + groundForward0 * crawlProgress;
            yaw = m_choreoInitialYaw + breathYaw;
            pitch = m_choreoInitialPitch + breathPitch;
        } else if (fInPhase < 38.0f) {
            // Smooth ADS Zoom Out
            float s = (fInPhase - 28.0f) / 10.0f;
            s = s * s * (3.0f - 2.0f * s);
            fov = targetFov + (m_choreoInitialFov - targetFov) * s;
            pos = pEndPhase5;
        } else {
            fov = m_choreoInitialFov;
            pos = pEndPhase5;
        }
    } else {
        // --- PHASE 6 (Frames 210..239): Arc-Strafe / Turntable Orbit (Non-uniform parallax flow) ---
        float fInPhase = normalizedFrame - 210.0f;
        float u = fInPhase / 29.0f; // 0..1
        float orbitR = 1.6f;
        glm::vec3 focalPoint = pEndPhase5 + groundForward0 * orbitR + worldUp * 0.05f;

        float maxAngleDeg = 16.0f;
        float curAngleDeg = maxAngleDeg * std::sin((PI * 0.5f) * u); // smooth ease-out arc
        float curAngleRad = glm::radians(curAngleDeg);

        // Orbit around focal point in horizontal plane
        glm::vec3 rel = -groundForward0 * std::cos(curAngleRad) + right0 * std::sin(curAngleRad);
        pos = focalPoint + rel * orbitR;

        // Camera gaze tracks focal point
        glm::vec3 lookDir = glm::normalize(focalPoint - pos);
        yaw = glm::degrees(std::atan2(lookDir.z, lookDir.x));
        pitch = glm::degrees(std::asin(std::clamp(lookDir.y, -0.999f, 0.999f)));
        fov = m_choreoInitialFov;
    }

    camera->setPose(pos, yaw, pitch);
    camera->setFov(fov);
}

void Engine::runTrainingDataCapture() {
    Logger::info("========================================================================================");
    Logger::info("  Pathways -> Upways Training Data Capture Pipeline (DGC Wavefront / Wave32)");
    Logger::info("  Target Directory : {}", m_config.capture_training_data_dir);
    Logger::info("  Frames to Capture: {}", m_config.capture_frames);
    Logger::info("  Reference SPP    : {}", m_config.capture_reference_spp);
    Logger::info("  Capture Channels : {} (PTTD v{})", m_config.capture_channels,
                 m_config.capture_channels >= 23 ? 3 : (m_config.capture_channels == 20 ? 2 : 1));
    Logger::info("  Capture Normals  : {}", m_config.capture_normals ? "YES" : "NO");
    Logger::info("========================================================================================");

    std::filesystem::create_directories(m_config.capture_training_data_dir);

    // Warm up camera & reset choreography state
    m_choreoInitialized = false;
    if (m_camera) {
        m_camera->resetMoved();
    }

    for (uint32_t f = 0; f < m_config.capture_frames; ++f) {
        Logger::info("--- Capturing Training Frame [{}/{}] ---", f + 1, m_config.capture_frames);

        // Update 6-DOF procedural choreography for frame f
        if (m_camera) {
            updateGamingChoreography(m_camera.get(), f, m_config.capture_frames, m_config.scene_path);
        }

        // 1. Render 1-SPP Noisy Input Frame FIRST (evaluates true inter-frame motion vectors from f-1 to f)
        captureTrainingFrame(f, /*isReference=*/false, 1);

        // 2. Render Ground Truth Reference Frame SECOND (same camera pose)
        captureTrainingFrame(f, /*isReference=*/true, m_config.capture_reference_spp);
    }

    Logger::info("========================================================================================");
    Logger::info("  Training Data Capture COMPLETE! {} frames written to {}",
                 m_config.capture_frames, m_config.capture_training_data_dir);
    Logger::info("========================================================================================");
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

void Engine::initVideoBillboardDecoder(const std::string& scenePath) {
    bool isCyberCity = (scenePath == "cyber-city" || scenePath == "cyber_city" ||
                        scenePath == "procedural:cyber-city" || scenePath == "procedural:cyber_city" ||
                        scenePath == "Procedural Cyber City");
    if (!isCyberCity) {
        for (const auto& mat : m_sceneData.materials) {
            if ((mat.type & MATERIAL_FLAG_HOLO_VIDEO) != 0) {
                isCyberCity = true;
                break;
            }
        }
    }

    if (!isCyberCity) {
        m_videoDecoder.reset();
        for (auto& sb : m_videoStagingBuffers) {
            sb.reset();
        }
        return;
    }

    std::filesystem::path exeDir;
#ifdef _WIN32
    char exePathBuf[MAX_PATH] = {0};
    if (GetModuleFileNameA(NULL, exePathBuf, MAX_PATH)) {
        exeDir = std::filesystem::path(exePathBuf).parent_path();
    }
#elif defined(__linux__) || defined(__unix__)
    std::error_code ec;
    auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec && !p.empty()) {
        exeDir = p.parent_path();
    } else {
        p = std::filesystem::canonical("/proc/self/exe", ec);
        if (!ec) exeDir = p.parent_path();
    }
#endif

    std::vector<std::filesystem::path> candidates = {
        "scenes/cyber_city/cyber_city_ad_1.mp4",
        "scenes/cyber_city_ad_1.mp4",
        "../scenes/cyber_city/cyber_city_ad_1.mp4",
        "../../scenes/cyber_city/cyber_city_ad_1.mp4",
    };
    if (!exeDir.empty()) {
        candidates.push_back(exeDir / "scenes" / "cyber_city" / "cyber_city_ad_1.mp4");
        candidates.push_back(exeDir / ".." / "scenes" / "cyber_city" / "cyber_city_ad_1.mp4");
        candidates.push_back(exeDir / ".." / ".." / "scenes" / "cyber_city" / "cyber_city_ad_1.mp4");
    }

    std::filesystem::path foundPath;
    for (const auto& c : candidates) {
        std::error_code err;
        if (std::filesystem::exists(c, err)) {
            foundPath = c;
            break;
        }
    }

    if (foundPath.empty()) {
        Logger::warn("Engine: Video billboard file 'cyber_city_ad_1.mp4' not found in candidate paths.");
        return;
    }

    m_videoDecoder = std::make_unique<VideoDecoder>();
    if (!m_videoDecoder->open(foundPath.string())) {
        Logger::error("Engine: Failed to open video billboard stream from '{}'", foundPath.string());
        m_videoDecoder.reset();
        return;
    }

    // Allocate per-frame-in-flight staging buffers for lock-free asynchronous GPU uploads
    VmaAllocator allocator = m_context->getAllocator();
    size_t rgbaSize = m_videoDecoder->getRgbaSize();
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_videoStagingBuffers[i] = std::make_unique<Buffer>(
            allocator, rgbaSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        );
    }

    Logger::info("Engine: Video billboard initialized ({}x{} @ {:.2f} fps) using '{}'",
                 m_videoDecoder->getWidth(), m_videoDecoder->getHeight(), m_videoDecoder->getFps(), foundPath.string());
}

void Engine::updateVideoBillboards(VkCommandBuffer cmd) {
    if (!m_videoDecoder || !m_videoDecoder->isOpen()) {
        return;
    }
    if (m_sceneTextures.empty() || !m_sceneTextures[0]) {
        return;
    }
    if (!m_videoDecoder->hasNewFrame()) {
        return;
    }

    const uint8_t* pixels = m_videoDecoder->getRgbaPixels();
    size_t size = m_videoDecoder->getRgbaSize();
    if (pixels && size > 0 && m_videoStagingBuffers[m_currentFrame]) {
        m_sceneTextures[0]->updatePixelsAsync(cmd, *m_videoStagingBuffers[m_currentFrame], pixels, size);
        m_videoDecoder->clearNewFrameFlag();
    }
}

} // namespace pathways
