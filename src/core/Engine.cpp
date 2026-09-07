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

namespace pathways {

Engine::Engine(const Config& config) : m_config(config) {
    m_startTime = std::chrono::high_resolution_clock::now();
    m_lastFrameTime = m_startTime;
    m_lastLogTime = std::chrono::steady_clock::now();

    Logger::info("Initializing Pathways Engine...");
    m_window = std::make_unique<Window>(m_config);

    if (!m_config.custom_resolution && !m_config.headless) {
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
    initPipelines();
    initWavefrontResources();
    initWavefrontPipelines();
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

    if (m_config.mgpu_mode != MultiGpuMode::Off) {
        m_mgpu = std::make_unique<MultiGpuManager>(m_config, m_context.get(), m_sceneData);
    }

    Logger::info("Pathways Engine initialization complete. Ready to render.");
}

Engine::~Engine() {
    Logger::info("Shutting down Pathways Engine...");
    VkDevice device = m_context->getDevice();
    vkDeviceWaitIdle(device);

    if (m_queryPool) vkDestroyQueryPool(device, m_queryPool, nullptr);
    if (m_inFlightFence) vkDestroyFence(device, m_inFlightFence, nullptr);
    if (m_commandPool) vkDestroyCommandPool(device, m_commandPool, nullptr);

    for (auto sem : m_imageAvailableSemaphores) {
        if (sem) vkDestroySemaphore(device, sem, nullptr);
    }
    m_imageAvailableSemaphores.clear();

    for (auto sem : m_renderFinishedSemaphores) {
        if (sem) vkDestroySemaphore(device, sem, nullptr);
    }
    m_renderFinishedSemaphores.clear();

    if (m_rtPipeline) vkDestroyPipeline(device, m_rtPipeline, nullptr);
    if (m_tonemapPipeline) vkDestroyPipeline(device, m_tonemapPipeline, nullptr);
    if (m_mergePipeline) vkDestroyPipeline(device, m_mergePipeline, nullptr);
    if (m_wfClassifyPipeline) vkDestroyPipeline(device, m_wfClassifyPipeline, nullptr);
    if (m_wfResolvePipeline) vkDestroyPipeline(device, m_wfResolvePipeline, nullptr);
    if (m_wfShadePipeline) vkDestroyPipeline(device, m_wfShadePipeline, nullptr);

    if (m_rtPipelineLayout) vkDestroyPipelineLayout(device, m_rtPipelineLayout, nullptr);
    if (m_tonemapPipelineLayout) vkDestroyPipelineLayout(device, m_tonemapPipelineLayout, nullptr);
    if (m_mergePipelineLayout) vkDestroyPipelineLayout(device, m_mergePipelineLayout, nullptr);
    if (m_wfClassifyPipelineLayout) vkDestroyPipelineLayout(device, m_wfClassifyPipelineLayout, nullptr);
    if (m_wfResolvePipelineLayout) vkDestroyPipelineLayout(device, m_wfResolvePipelineLayout, nullptr);
    if (m_wfShadePipelineLayout) vkDestroyPipelineLayout(device, m_wfShadePipelineLayout, nullptr);

    if (m_rtDescLayout) vkDestroyDescriptorSetLayout(device, m_rtDescLayout, nullptr);
    if (m_tonemapDescLayout) vkDestroyDescriptorSetLayout(device, m_tonemapDescLayout, nullptr);
    if (m_mergeDescLayout) vkDestroyDescriptorSetLayout(device, m_mergeDescLayout, nullptr);
    if (m_wfClassifyDescLayout) vkDestroyDescriptorSetLayout(device, m_wfClassifyDescLayout, nullptr);
    if (m_wfResolveDescLayout) vkDestroyDescriptorSetLayout(device, m_wfResolveDescLayout, nullptr);
    if (m_wfShadeDescLayout) vkDestroyDescriptorSetLayout(device, m_wfShadeDescLayout, nullptr);

    if (m_descriptorPool) vkDestroyDescriptorPool(device, m_descriptorPool, nullptr);
    if (m_rtFence) vkDestroyFence(device, m_rtFence, nullptr);
    m_secTransferBuffer.reset();

    m_rayQueueA.reset();
    m_rayQueueB.reset();
    m_wavefrontCounters.reset();
    m_wavefrontIndirectCmd.reset();
    m_wavefrontDgcStream.reset();
    m_wavefrontDgcCount.reset();

    m_gui.reset();
    m_swapchain.reset();
    if (m_surface) {
        vkDestroySurfaceKHR(m_context->getInstance(), m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
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

    // Command buffer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(device, &allocInfo, &m_commandBuffer);

    // Create Render Target Images
    m_accumImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
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
    vkBeginCommandBuffer(m_commandBuffer, &beginInfo);

    m_accumImage->transitionLayout(
        m_commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    m_outputImage->transitionLayout(
        m_commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    vkEndCommandBuffer(m_commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffer;
    vkQueueSubmit(m_context->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_context->getGraphicsQueue());
}

void Engine::initScene() {
    VmaAllocator allocator = m_context->getAllocator();

    if (!m_config.scene_path.empty()) {
        Logger::info("Loading user specified scene: {}", m_config.scene_path);
        m_sceneData = GltfLoader::loadSceneData(m_config.scene_path);
    } else {
        m_sceneData = ProceduralScene::createCornellBox();
    }

    if (m_sceneData.triangles.empty() && m_sceneData.spheres.empty()) {
        Logger::warn("Loaded scene contains no renderable geometry! Falling back to procedural Cornell Box.");
        m_sceneData = ProceduralScene::createCornellBox();
    }

    m_numTriangles = static_cast<uint32_t>(m_sceneData.triangles.size());
    m_numSpheres = static_cast<uint32_t>(m_sceneData.spheres.size());
    m_numMaterials = static_cast<uint32_t>(m_sceneData.materials.size());
    m_numLights = static_cast<uint32_t>(m_sceneData.lights.size());

    Logger::info("Active Scene: {} Triangles, {} Spheres, {} Materials, {} Lights",
                 m_numTriangles, m_numSpheres, m_numMaterials, m_numLights);

    if (m_sceneData.hasCamera && m_camera) {
        m_camera->lookAt(m_sceneData.cameraPosition, m_sceneData.cameraTarget, m_sceneData.cameraUp);
        m_camera->setFov(m_sceneData.cameraFov);
        m_camera->setDefaultFraming(m_sceneData.cameraPosition, m_sceneData.cameraTarget, m_sceneData.cameraFov);
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

    // Camera UBO
    VkDeviceSize uboSize = sizeof(CameraUniform);
    m_cameraUBO = std::make_unique<Buffer>(
        allocator, uboSize,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

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
        Logger::info("Hardware Ray Tracing Acceleration Structures initialized successfully (BLAS & TLAS).");
        if (m_config.enable_hardware_rt) {
            Logger::info("Hardware Ray Tracing Pipeline Active. Extensions in use: VK_KHR_ray_query, VK_KHR_acceleration_structure, VK_KHR_buffer_device_address, VK_KHR_deferred_host_operations (SPIR-V: GL_EXT_ray_query)");
        } else {
            Logger::info("Hardware Ray Tracing is DISABLED via config. Running Software Primitive Traversal (LDS/SSBO). HW RT extensions bypassed.");
        }
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
    }

    searchPaths.push_back(std::string(SHADER_DIR) + "/" + filename);
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

    // 1. Descriptor Pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 20 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 20 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 16 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 32;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &m_descriptorPool);

    // 2. Ray Tracing Descriptor Set Layout
    std::vector<VkDescriptorSetLayoutBinding> rtBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_SCENE_TEXTURES, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
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
    VkDescriptorSetAllocateInfo rtAllocInfo{};
    rtAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    rtAllocInfo.descriptorPool = m_descriptorPool;
    rtAllocInfo.descriptorSetCount = 1;
    rtAllocInfo.pSetLayouts = &m_rtDescLayout;
    vkAllocateDescriptorSets(device, &rtAllocInfo, &m_rtDescSet);

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

    VkDescriptorBufferInfo uboBufferInfo{ m_cameraUBO->getBuffer(), 0, sizeof(CameraUniform) };
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

    std::vector<VkWriteDescriptorSet> writes = {
        // RT set
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboBufferInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triBufferInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sphereBufferInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matBufferInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightBufferInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, m_rtDescSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_rtDescSet, 8, 0, MAX_SCENE_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texInfos.data(), nullptr, nullptr },
        // Tonemap set
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_tonemapDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_tonemapDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputImageInfo, nullptr, nullptr }
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // 6. Pipeline Layouts
    VkPushConstantRange rtPushConstant{};
    rtPushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    rtPushConstant.offset = 0;
    rtPushConstant.size = sizeof(uint32_t) * 12;

    VkPipelineLayoutCreateInfo rtPipeLayoutInfo{};
    rtPipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    rtPipeLayoutInfo.setLayoutCount = 1;
    rtPipeLayoutInfo.pSetLayouts = &m_rtDescLayout;
    rtPipeLayoutInfo.pushConstantRangeCount = 1;
    rtPipeLayoutInfo.pPushConstantRanges = &rtPushConstant;
    vkCreatePipelineLayout(device, &rtPipeLayoutInfo, nullptr, &m_rtPipelineLayout);

    VkPushConstantRange tonemapPushConstant{};
    tonemapPushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    tonemapPushConstant.offset = 0;
    tonemapPushConstant.size = sizeof(float) + sizeof(uint32_t) * 3; // exposure, totalSamples, applyACES, padding

    VkPipelineLayoutCreateInfo tonemapPipeLayoutInfo{};
    tonemapPipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    tonemapPipeLayoutInfo.setLayoutCount = 1;
    tonemapPipeLayoutInfo.pSetLayouts = &m_tonemapDescLayout;
    tonemapPipeLayoutInfo.pushConstantRangeCount = 1;
    tonemapPipeLayoutInfo.pPushConstantRanges = &tonemapPushConstant;
    vkCreatePipelineLayout(device, &tonemapPipeLayoutInfo, nullptr, &m_tonemapPipelineLayout);

    // 7. Compute Pipelines (Wave32 execution mode on RDNA4)
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    auto rtCode = loadShaderSPIRV("raytrace_comp.comp.spv");
    VkShaderModule rtModule = createShaderModule(rtCode);

    VkComputePipelineCreateInfo rtPipelineInfo{};
    rtPipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    rtPipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rtPipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    rtPipelineInfo.stage.module = rtModule;
    rtPipelineInfo.stage.pName = "main";
    if (m_context->hasSubgroupSizeControl()) {
        rtPipelineInfo.stage.pNext = &subgroupSize32;
    }
    rtPipelineInfo.layout = m_rtPipelineLayout;
    vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &rtPipelineInfo, nullptr, &m_rtPipeline);
    vkDestroyShaderModule(device, rtModule, nullptr);

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

    Logger::info("Compute pipelines (Path Tracer & ACES Tonemapping - Wave32) created successfully.");

    // Initialize Device-Generated Commands (VK_EXT_device_generated_commands) if supported
    VmaAllocator allocator = m_context->getAllocator();
    if (m_context->hasDGC()) {
        try {
            m_dgc = std::make_unique<DGCManager>(device, allocator, m_rtPipelineLayout);
            if (m_dgc->isSupported()) {
                m_dgcArgumentBuffer = std::make_unique<Buffer>(
                    allocator, sizeof(VkDispatchIndirectCommand),
                    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
                    256
                );
                VkDispatchIndirectCommand defaultCmd{};
                defaultCmd.x = (m_config.width + 7) / 8;
                defaultCmd.y = (m_config.height + 3) / 4;
                defaultCmd.z = 1;
                m_dgcArgumentBuffer->copyFrom(&defaultCmd, sizeof(VkDispatchIndirectCommand));
                Logger::info("Device-Generated Commands (DGC) initialized successfully.");
            }
        } catch (const std::exception& e) {
            Logger::warn("DGC initialization exception: {}", e.what());
        }
    }

    if (m_config.mgpu_mode != MultiGpuMode::Off) {
        VmaAllocator allocator = m_context->getAllocator();
        VkDeviceSize bufferSize = m_config.width * m_config.height * 4 * sizeof(float);
        m_secTransferBuffer = std::make_unique<Buffer>(
            allocator, bufferSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
        );

        // Descriptor Set Layout for accum_merge
        std::vector<VkDescriptorSetLayoutBinding> mergeBindings = {
            { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
            { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
        };
        VkDescriptorSetLayoutCreateInfo mergeLayoutInfo{};
        mergeLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        mergeLayoutInfo.bindingCount = static_cast<uint32_t>(mergeBindings.size());
        mergeLayoutInfo.pBindings = mergeBindings.data();
        vkCreateDescriptorSetLayout(device, &mergeLayoutInfo, nullptr, &m_mergeDescLayout);

        // Allocate Descriptor Set
        VkDescriptorSetAllocateInfo mergeAllocInfo{};
        mergeAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
        mergeAllocInfo.descriptorPool = m_descriptorPool;
        mergeAllocInfo.descriptorSetCount = 1;
        mergeAllocInfo.pSetLayouts = &m_mergeDescLayout;
        vkAllocateDescriptorSets(device, &mergeAllocInfo, &m_mergeDescSet);

        // Update Descriptor Set
        VkDescriptorImageInfo accumImageInfo{};
        accumImageInfo.imageView = m_accumImage->getImageView();
        accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkDescriptorBufferInfo secBufInfo{ m_secTransferBuffer->getBuffer(), 0, bufferSize };

        std::vector<VkWriteDescriptorSet> mergeWrites = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secBufInfo, nullptr }
        };
        vkUpdateDescriptorSets(device, static_cast<uint32_t>(mergeWrites.size()), mergeWrites.data(), 0, nullptr);

        // Pipeline Layout
        VkPushConstantRange mergePushConstant{};
        mergePushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        mergePushConstant.offset = 0;
        mergePushConstant.size = sizeof(uint32_t) * 4; // width, height, secondarySpp, padding

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
}

void Engine::initWavefrontResources() {
    VkDevice device = m_context->getDevice();
    VmaAllocator allocator = m_context->getAllocator();

    VkDeviceSize maxRays = static_cast<VkDeviceSize>(m_config.width) * m_config.height;
    VkDeviceSize rayPayloadSize = 96; // 6 * vec4 (24 floats)
    VkDeviceSize queueSize = maxRays * rayPayloadSize;

    // Ray Queue A
    m_rayQueueA = std::make_unique<Buffer>(
        allocator, queueSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    // Ray Queue B (ping-pong partner)
    m_rayQueueB = std::make_unique<Buffer>(
        allocator, queueSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    // Wavefront Counters: [activeRayCount, nextActiveCount, totalProcessed, padding]
    m_wavefrontCounters = std::make_unique<Buffer>(
        allocator, sizeof(uint32_t) * 4,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    // Indirect Dispatch Command: VkDispatchIndirectCommand { uint x, y, z }
    m_wavefrontIndirectCmd = std::make_unique<Buffer>(
        allocator, sizeof(VkDispatchIndirectCommand),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    // DGC Sequence Stream: sequence count (uint32) + padding + sequence items
    m_wavefrontDgcStream = std::make_unique<Buffer>(
        allocator, 256,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    m_wavefrontDgcCount = std::make_unique<Buffer>(
        allocator, 16,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    Logger::info("Wavefront Compaction Queues allocated: 2x {:.2f} MB ({:.2f}M ray capacity each).",
                 (queueSize / (1024.0 * 1024.0)), (maxRays / 1000000.0));
}

void Engine::initWavefrontPipelines() {
    VkDevice device = m_context->getDevice();

    // 1. Classify Descriptor Set Layout (Bindings 0-8 match RT, 9: OutRayQueue, 10: QueueCounters)
    std::vector<VkDescriptorSetLayoutBinding> classifyBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_SCENE_TEXTURES, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo classifyLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    classifyLayoutInfo.bindingCount = static_cast<uint32_t>(classifyBindings.size());
    classifyLayoutInfo.pBindings = classifyBindings.data();
    vkCreateDescriptorSetLayout(device, &classifyLayoutInfo, nullptr, &m_wfClassifyDescLayout);

    // 2. Resolve Descriptor Set Layout (Binding 0: QueueCounters, 1: IndirectCommand, 2: DGCStream)
    std::vector<VkDescriptorSetLayoutBinding> resolveBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo resolveLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    resolveLayoutInfo.bindingCount = static_cast<uint32_t>(resolveBindings.size());
    resolveLayoutInfo.pBindings = resolveBindings.data();
    vkCreateDescriptorSetLayout(device, &resolveLayoutInfo, nullptr, &m_wfResolveDescLayout);

    // 3. Shade Descriptor Set Layout (Bindings 0-8 match RT, 9: InRayQueue, 10: OutRayQueue, 11: QueueCounters)
    std::vector<VkDescriptorSetLayoutBinding> shadeBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_SCENE_TEXTURES, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo shadeLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    shadeLayoutInfo.bindingCount = static_cast<uint32_t>(shadeBindings.size());
    shadeLayoutInfo.pBindings = shadeBindings.data();
    vkCreateDescriptorSetLayout(device, &shadeLayoutInfo, nullptr, &m_wfShadeDescLayout);

    // 4. Allocate Descriptor Sets
    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = m_descriptorPool;

    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_wfClassifyDescLayout;
    vkAllocateDescriptorSets(device, &allocInfo, &m_wfClassifyDescSet);

    allocInfo.pSetLayouts = &m_wfResolveDescLayout;
    vkAllocateDescriptorSets(device, &allocInfo, &m_wfResolveDescSet);

    allocInfo.pSetLayouts = &m_wfShadeDescLayout;
    vkAllocateDescriptorSets(device, &allocInfo, &m_wfShadeDescSetA);
    vkAllocateDescriptorSets(device, &allocInfo, &m_wfShadeDescSetB);

    // 5. Populate and Write Descriptor Sets
    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageView = m_accumImage->getImageView();
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo uboBufferInfo{ m_cameraUBO->getBuffer(), 0, sizeof(CameraUniform) };
    VkDescriptorBufferInfo triBufferInfo{ m_triangleBuffer->getBuffer(), 0, m_triangleBuffer->getSize() };
    VkDescriptorBufferInfo sphereBufferInfo{ m_sphereBuffer->getBuffer(), 0, m_sphereBuffer->getSize() };
    VkDescriptorBufferInfo matBufferInfo{ m_materialBuffer->getBuffer(), 0, m_materialBuffer->getSize() };
    VkDescriptorBufferInfo lightBufferInfo{ m_lightBuffer->getBuffer(), 0, m_lightBuffer->getSize() };

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR };
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

    VkDescriptorBufferInfo queueABufInfo{ m_rayQueueA->getBuffer(), 0, m_rayQueueA->getSize() };
    VkDescriptorBufferInfo queueBBufInfo{ m_rayQueueB->getBuffer(), 0, m_rayQueueB->getSize() };
    VkDescriptorBufferInfo counterBufInfo{ m_wavefrontCounters->getBuffer(), 0, m_wavefrontCounters->getSize() };
    VkDescriptorBufferInfo indirectBufInfo{ m_wavefrontIndirectCmd->getBuffer(), 0, m_wavefrontIndirectCmd->getSize() };
    VkDescriptorBufferInfo dgcStreamBufInfo{ m_wavefrontDgcStream->getBuffer(), 0, m_wavefrontDgcStream->getSize() };

    std::vector<VkWriteDescriptorSet> writes;

    auto pushCommonBindings = [&](VkDescriptorSet dstSet) {
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dstSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dstSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dstSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dstSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sphereBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dstSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dstSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightBufferInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, dstSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dstSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dstSet, 8, 0, MAX_SCENE_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texInfos.data(), nullptr, nullptr });
    };

    // Classify Set: Common + binding 9 (queue A), binding 10 (counters)
    pushCommonBindings(m_wfClassifyDescSet);
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfClassifyDescSet, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &queueABufInfo, nullptr });
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfClassifyDescSet, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &counterBufInfo, nullptr });

    // Resolve Set: binding 0 (counters), binding 1 (indirect), binding 2 (dgc stream)
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfResolveDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &counterBufInfo, nullptr });
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfResolveDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &indirectBufInfo, nullptr });
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfResolveDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dgcStreamBufInfo, nullptr });

    // Shade Set A: Common + binding 9 (in: queue A), binding 10 (out: queue B), binding 11 (counters)
    pushCommonBindings(m_wfShadeDescSetA);
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfShadeDescSetA, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &queueABufInfo, nullptr });
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfShadeDescSetA, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &queueBBufInfo, nullptr });
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfShadeDescSetA, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &counterBufInfo, nullptr });

    // Shade Set B: Common + binding 9 (in: queue B), binding 10 (out: queue A), binding 11 (counters)
    pushCommonBindings(m_wfShadeDescSetB);
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfShadeDescSetB, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &queueBBufInfo, nullptr });
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfShadeDescSetB, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &queueABufInfo, nullptr });
    writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfShadeDescSetB, 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &counterBufInfo, nullptr });

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // 6. Pipeline Layouts
    VkPushConstantRange wfPushConstant{};
    wfPushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    wfPushConstant.offset = 0;
    wfPushConstant.size = sizeof(uint32_t) * 12;

    VkPipelineLayoutCreateInfo classifyPipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    classifyPipeLayoutInfo.setLayoutCount = 1;
    classifyPipeLayoutInfo.pSetLayouts = &m_wfClassifyDescLayout;
    classifyPipeLayoutInfo.pushConstantRangeCount = 1;
    classifyPipeLayoutInfo.pPushConstantRanges = &wfPushConstant;
    vkCreatePipelineLayout(device, &classifyPipeLayoutInfo, nullptr, &m_wfClassifyPipelineLayout);

    VkPushConstantRange resolvePushConstant{};
    resolvePushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    resolvePushConstant.offset = 0;
    resolvePushConstant.size = sizeof(uint32_t) * 4;

    VkPipelineLayoutCreateInfo resolvePipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    resolvePipeLayoutInfo.setLayoutCount = 1;
    resolvePipeLayoutInfo.pSetLayouts = &m_wfResolveDescLayout;
    resolvePipeLayoutInfo.pushConstantRangeCount = 1;
    resolvePipeLayoutInfo.pPushConstantRanges = &resolvePushConstant;
    vkCreatePipelineLayout(device, &resolvePipeLayoutInfo, nullptr, &m_wfResolvePipelineLayout);

    VkPipelineLayoutCreateInfo shadePipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    shadePipeLayoutInfo.setLayoutCount = 1;
    shadePipeLayoutInfo.pSetLayouts = &m_wfShadeDescLayout;
    shadePipeLayoutInfo.pushConstantRangeCount = 1;
    shadePipeLayoutInfo.pPushConstantRanges = &wfPushConstant;
    vkCreatePipelineLayout(device, &shadePipeLayoutInfo, nullptr, &m_wfShadePipelineLayout);

    // 7. Compute Pipelines with Wave32 Subgroup Size
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{ VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO };
    subgroupSize32.requiredSubgroupSize = 32;

    auto createPipeline = [&](const std::string& spvName, VkPipelineLayout layout) -> VkPipeline {
        auto code = loadShaderSPIRV(spvName);
        VkShaderModule mod = createShaderModule(code);

        VkComputePipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeInfo.stage.module = mod;
        pipeInfo.stage.pName = "main";
        if (m_context->hasSubgroupSizeControl()) {
            pipeInfo.stage.pNext = &subgroupSize32;
        }
        pipeInfo.layout = layout;

        VkPipeline pipe = VK_NULL_HANDLE;
        vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &pipe);
        vkDestroyShaderModule(device, mod, nullptr);
        return pipe;
    };

    m_wfClassifyPipeline = createPipeline("wavefront_classify.comp.spv", m_wfClassifyPipelineLayout);
    m_wfResolvePipeline  = createPipeline("wavefront_resolve.comp.spv", m_wfResolvePipelineLayout);
    m_wfShadePipeline    = createPipeline("wavefront_shade.comp.spv", m_wfShadePipelineLayout);

    if (m_context->hasDGC()) {
        try {
            m_dgc = std::make_unique<DGCManager>(device, m_context->getAllocator(), m_wfShadePipelineLayout);
        } catch (...) {
            Logger::warn("Failed to create DGCManager for wavefront shade pipeline.");
        }
    }

    Logger::info("Wavefront Compaction compute pipelines (Classify, Resolve, Shade - Wave32) created successfully.");
}

void Engine::initSyncObjects() {
    VkDevice device = m_context->getDevice();

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(device, &fenceInfo, nullptr, &m_inFlightFence);
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
    queryInfo.queryCount = 4; // 0: Start, 1: RT End, 2: Tonemap End
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
        if (e.type == SDL_EVENT_MOUSE_MOTION) {
            if (m_camera) {
                m_camera->processMouseMovement(e.motion.xrel, e.motion.yrel);
            }
            return true;
        }
        if (e.type == SDL_EVENT_MOUSE_WHEEL) {
            if (m_camera) {
                float newSpeed = std::clamp(m_camera->getSpeed() + e.wheel.y * 0.5f, 0.2f, 25.0f);
                m_camera->setSpeed(newSpeed);
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
    if (keyState[SDL_SCANCODE_LCTRL] || keyState[SDL_SCANCODE_RCTRL] || keyState[SDL_SCANCODE_C] || keyState[SDL_SCANCODE_Q]) vertical -= 1.0f;

    bool sprint = keyState[SDL_SCANCODE_LSHIFT] || keyState[SDL_SCANCODE_RSHIFT];

    m_camera->processFpsInput(forward, strafe, vertical, dt, sprint);
}

void Engine::renderFrame() {
    VkDevice device = m_context->getDevice();
    VkQueue queue = m_context->getGraphicsQueue();

    vkWaitForFences(device, 1, &m_inFlightFence, VK_TRUE, UINT64_MAX);
    vkResetFences(device, 1, &m_inFlightFence);

    auto frameStartTime = std::chrono::high_resolution_clock::now();

    // Update smooth continuous FPS keyboard navigation
    updateInput();

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

    CameraUniform ubo = m_camera->getUniformData(m_frameIndex, m_config.spp, m_config.max_bounces, flags);
    m_cameraUBO->copyFrom(&ubo, sizeof(CameraUniform));

    uint32_t imageIndex = 0;
    if (!m_config.headless && m_swapchain) {
        VkResult res = m_swapchain->acquireNextImage(m_imageAvailableSemaphores[m_currentFrame], &imageIndex);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR ||
            m_config.width != m_swapchain->getExtent().width ||
            m_config.height != m_swapchain->getExtent().height) {
            onResize(m_swapchain->getExtent().width, m_swapchain->getExtent().height);
            return;
        }
    }

    uint32_t groupsX = (m_config.width + 15) / 16;
    uint32_t groupsY = (m_config.height + 15) / 16;
    uint32_t rtGroupsX = (m_config.width + 7) / 8;
    uint32_t rtGroupsY = (m_config.height + 3) / 4;

    struct {
        float exposure = 1.0f;
        uint32_t totalSamples = 1;
        uint32_t applyACES = 1;
        uint32_t padding = 0;
    } tonemapConstants;
    tonemapConstants.exposure = 1.0f;
    tonemapConstants.applyACES = m_config.aces_tonemap ? 1 : 0;

    bool isMgpu = (m_mgpu && m_mgpu->isMultiGpuActive());

    if (!isMgpu) {
        // --- Single GPU Execution Path ---
        CameraUniform ubo = m_camera->getUniformData(m_frameIndex, m_config.spp, m_config.max_bounces, flags);
        m_cameraUBO->copyFrom(&ubo, sizeof(CameraUniform));

        vkResetCommandBuffer(m_commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(m_commandBuffer, &beginInfo);

        vkCmdResetQueryPool(m_commandBuffer, m_queryPool, 0, 4);
        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, 0);

        uint32_t useHwRT = (m_tlas != nullptr && m_config.enable_hardware_rt) ? 1 : 0;
        uint32_t hasEnvMap = m_environmentMap ? 1 : 0;
        float envIntensity = 1.0f;
        uint32_t envIntensityBits = std::bit_cast<uint32_t>(envIntensity);

        if (m_config.pipeline_type == PipelineType::Wavefront) {
            // === WAVEFRONT COMPACTION PIPELINE ===
            uint32_t maxCapacity = m_config.width * m_config.height;

            for (uint32_t s = 0; s < m_config.spp; ++s) {
                // 1. Clear Wavefront Counters
                vkCmdFillBuffer(m_commandBuffer, m_wavefrontCounters->getBuffer(), 0, sizeof(uint32_t) * 4, 0);

                VkMemoryBarrier2 clearBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
                clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

                VkDependencyInfo clearDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                clearDep.memoryBarrierCount = 1;
                clearDep.pMemoryBarriers = &clearBarrier;
                vkCmdPipelineBarrier2(m_commandBuffer, &clearDep);

                // 2. Classify Pass: Primary Ray Generation & Wave Compaction
                vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_wfClassifyPipeline);
                vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_wfClassifyPipelineLayout, 0, 1, &m_wfClassifyDescSet, 0, nullptr);

                uint32_t classifyPC[12] = {
                    m_numTriangles, m_numSpheres, m_numMaterials, m_numLights,
                    m_config.width, m_config.height,
                    m_config.enable_morton_order ? 1u : 0u,
                    hasEnvMap,
                    envIntensityBits,
                    maxCapacity,
                    s,
                    0u
                };
                vkCmdPushConstants(m_commandBuffer, m_wfClassifyPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(classifyPC), classifyPC);

                uint32_t classifyGroupsX = (m_config.width + 7) / 8;
                uint32_t classifyGroupsY = (m_config.height + 3) / 4;
                vkCmdDispatch(m_commandBuffer, classifyGroupsX, classifyGroupsY, 1);

                // Barrier: Classify writes queue A & counters -> Resolve reads counters
                VkMemoryBarrier2 classifyToResolveBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                classifyToResolveBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                classifyToResolveBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                classifyToResolveBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                classifyToResolveBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

                VkDependencyInfo classifyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                classifyDep.memoryBarrierCount = 1;
                classifyDep.pMemoryBarriers = &classifyToResolveBarrier;
                vkCmdPipelineBarrier2(m_commandBuffer, &classifyDep);

                // 3. Bounce Loop: Resolve -> Shade (DGC / Indirect)
                for (uint32_t bounce = 0; bounce < m_config.max_bounces; ++bounce) {
                    // A. Resolve Pass (1 Wave32 workgroup)
                    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_wfResolvePipeline);
                    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_wfResolvePipelineLayout, 0, 1, &m_wfResolveDescSet, 0, nullptr);

                    uint32_t resolveMode = (bounce == 0) ? 0u : 1u;
                    uint32_t resolvePC[4] = { resolveMode, 0, 0, 0 };
                    vkCmdPushConstants(m_commandBuffer, m_wfResolvePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(resolvePC), resolvePC);
                    vkCmdDispatch(m_commandBuffer, 1, 1, 1);

                    // Barrier: Resolve writes indirect cmd & resets counters -> Shade reads indirect cmd & queues
                    VkMemoryBarrier2 resolveToShadeBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                    resolveToShadeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    resolveToShadeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    resolveToShadeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    resolveToShadeBarrier.dstAccessMask = VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

                    VkDependencyInfo resolveDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                    resolveDep.memoryBarrierCount = 1;
                    resolveDep.pMemoryBarriers = &resolveToShadeBarrier;
                    vkCmdPipelineBarrier2(m_commandBuffer, &resolveDep);

                    // B. Shade Pass (Ping-Pong: even bounce reads A writes B; odd bounce reads B writes A)
                    VkDescriptorSet shadeSet = (bounce % 2 == 0) ? m_wfShadeDescSetA : m_wfShadeDescSetB;
                    vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_wfShadePipeline);
                    vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_wfShadePipelineLayout, 0, 1, &shadeSet, 0, nullptr);

                    uint32_t shadePC[12] = {
                        m_numTriangles, m_numSpheres, m_numMaterials, m_numLights,
                        m_config.width, m_config.height,
                        bounce,
                        m_config.max_bounces,
                        maxCapacity,
                        s,
                        hasEnvMap,
                        envIntensityBits
                    };
                    vkCmdPushConstants(m_commandBuffer, m_wfShadePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shadePC), shadePC);

                    if (m_dgc && m_dgc->isSupported()) {
                        m_dgc->recordExecute(m_commandBuffer, m_wfShadePipeline, m_wavefrontIndirectCmd.get(), 0, 1, m_wavefrontDgcStream.get(), 0);
                    } else {
                        vkCmdDispatchIndirect(m_commandBuffer, m_wavefrontIndirectCmd->getBuffer(), 0);
                    }

                    // Barrier: Shade writes downstream queue / accum image -> next Resolve or Tonemap
                    VkMemoryBarrier2 shadeToNextBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                    shadeToNextBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    shadeToNextBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    shadeToNextBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    shadeToNextBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

                    VkDependencyInfo shadeDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                    shadeDep.memoryBarrierCount = 1;
                    shadeDep.pMemoryBarriers = &shadeToNextBarrier;
                    vkCmdPipelineBarrier2(m_commandBuffer, &shadeDep);
                }
            }
        } else {
            // === MONOLITHIC MEGAKERNEL PIPELINE ===
            vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipeline);
            vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipelineLayout, 0, 1, &m_rtDescSet, 0, nullptr);

            uint32_t rtPushConstants[12] = {
                m_numTriangles, m_numSpheres, m_numMaterials, m_numLights,
                0, 0, m_config.width, m_config.height,
                useHwRT,
                hasEnvMap,
                envIntensityBits,
                1u // accumulateHistory
            };
            vkCmdPushConstants(m_commandBuffer, m_rtPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rtPushConstants), rtPushConstants);

            if (m_dgc && m_dgcArgumentBuffer) {
                VkDispatchIndirectCommand dgcCmd{};
                dgcCmd.x = rtGroupsX;
                dgcCmd.y = rtGroupsY;
                dgcCmd.z = 1;
                m_dgcArgumentBuffer->copyFrom(&dgcCmd, sizeof(VkDispatchIndirectCommand));
                m_dgc->recordIndirectDispatch(m_commandBuffer, m_dgcArgumentBuffer.get(), 0);
            } else {
                vkCmdDispatch(m_commandBuffer, rtGroupsX, rtGroupsY, 1);
            }
        }

        VkMemoryBarrier2 memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(m_commandBuffer, &depInfo);

        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, 1);

        // Tonemapping
        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipeline);
        vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &m_tonemapDescSet, 0, nullptr);

        tonemapConstants.totalSamples = (m_frameIndex + 1) * m_config.spp;
        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, 2);
        vkCmdPushConstants(m_commandBuffer, m_tonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tonemapConstants), &tonemapConstants);
        vkCmdDispatch(m_commandBuffer, groupsX, groupsY, 1);
        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, 3);
    } else if (m_config.mgpu_mode == MultiGpuMode::SampleParallel) {
        // --- Multi-GPU Sample Parallelism Path ---
        uint32_t spp0 = (m_config.spp > 1) ? (m_config.spp / 2) : 1;
        uint32_t spp1 = (m_config.spp > 1) ? (m_config.spp - spp0) : 1;

        CameraUniform ubo0 = m_camera->getUniformData(m_frameIndex * 2, spp0, m_config.max_bounces, flags);
        m_cameraUBO->copyFrom(&ubo0, sizeof(CameraUniform));

        CameraUniform ubo1 = m_camera->getUniformData(m_frameIndex * 2 + 1, spp1, m_config.max_bounces, flags);

        uint32_t useHwRT = (m_tlas != nullptr && m_config.enable_hardware_rt) ? 1 : 0;
        uint32_t hasEnvMap = m_environmentMap ? 1 : 0;
        float envIntensity = 1.0f;
        uint32_t envIntensityBits = std::bit_cast<uint32_t>(envIntensity);

        size_t transferBytes = static_cast<size_t>(m_config.width) * m_config.height * 4 * sizeof(float);
        void* dstHost = m_secTransferBuffer->map();

        // 1. Launch secondary GPU asynchronously in dedicated worker thread (with overlapped PCIe transfer)
        m_mgpu->launchSecondaryWork(ubo1, m_frameIndex * 2 + 1, 0, 0, m_config.width, m_config.height,
                                   m_numTriangles, m_numSpheres, m_numMaterials, m_numLights, useHwRT,
                                   hasEnvMap, envIntensity, 0u, dstHost, transferBytes);

        // 2. Concurrently record and execute primary GPU raytracing
        vkWaitForFences(device, 1, &m_rtFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &m_rtFence);

        vkResetCommandBuffer(m_commandBuffer, 0);
        VkCommandBufferBeginInfo rtBeginInfo{};
        rtBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(m_commandBuffer, &rtBeginInfo);

        vkCmdResetQueryPool(m_commandBuffer, m_queryPool, 0, 4);
        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, 0);

        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipeline);
        vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipelineLayout, 0, 1, &m_rtDescSet, 0, nullptr);

        uint32_t rtPushConstants[12] = {
            m_numTriangles, m_numSpheres, m_numMaterials, m_numLights,
            0, 0, m_config.width, m_config.height,
            useHwRT,
            hasEnvMap,
            envIntensityBits,
            1u // accumulateHistory: primary GPU maintains temporal master accumulation
        };
        vkCmdPushConstants(m_commandBuffer, m_rtPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rtPushConstants), rtPushConstants);
        if (m_dgc && m_dgcArgumentBuffer) {
            VkDispatchIndirectCommand dgcCmd{};
            dgcCmd.x = rtGroupsX;
            dgcCmd.y = rtGroupsY;
            dgcCmd.z = 1;
            m_dgcArgumentBuffer->copyFrom(&dgcCmd, sizeof(VkDispatchIndirectCommand));
            m_dgc->recordIndirectDispatch(m_commandBuffer, m_dgcArgumentBuffer.get(), 0);
        } else {
            vkCmdDispatch(m_commandBuffer, rtGroupsX, rtGroupsY, 1);
        }

        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, 1);
        vkEndCommandBuffer(m_commandBuffer);

        VkSubmitInfo rtSubmit{};
        rtSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        rtSubmit.commandBufferCount = 1;
        rtSubmit.pCommandBuffers = &m_commandBuffer;
        vkQueueSubmit(queue, 1, &rtSubmit, m_rtFence);

        // 3. Wait for primary GPU raytracing and secondary GPU completion + PCIe transfer
        vkWaitForFences(device, 1, &m_rtFence, VK_TRUE, UINT64_MAX);
        m_mgpu->syncAndTransfer(nullptr, 0);
        m_secTransferBuffer->unmap();

        // 4. Record Merge & Tonemapping commands on primary GPU
        vkResetCommandBuffer(m_commandBuffer, 0);
        VkCommandBufferBeginInfo postBeginInfo{};
        postBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(m_commandBuffer, &postBeginInfo);

        // Barrier: Ensure primary RT writes to m_accumImage are visible before merge compute reads/writes
        VkMemoryBarrier2 rtToMergeBarrier{};
        rtToMergeBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        rtToMergeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rtToMergeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        rtToMergeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        rtToMergeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo rtToMergeDep{};
        rtToMergeDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        rtToMergeDep.memoryBarrierCount = 1;
        rtToMergeDep.pMemoryBarriers = &rtToMergeBarrier;
        vkCmdPipelineBarrier2(m_commandBuffer, &rtToMergeDep);

        // Merge Pass
        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_mergePipeline);
        vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_mergePipelineLayout, 0, 1, &m_mergeDescSet, 0, nullptr);

        uint32_t mergePC[4] = { m_config.width, m_config.height, spp1, 0 };
        vkCmdPushConstants(m_commandBuffer, m_mergePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(mergePC), mergePC);
        vkCmdDispatch(m_commandBuffer, groupsX, groupsY, 1);

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
        vkCmdPipelineBarrier2(m_commandBuffer, &mergeDep);

        // Tonemapping
        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipeline);
        vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &m_tonemapDescSet, 0, nullptr);

        tonemapConstants.totalSamples = (m_frameIndex + 1) * (spp0 + spp1);
        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, 2);
        vkCmdPushConstants(m_commandBuffer, m_tonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tonemapConstants), &tonemapConstants);
        vkCmdDispatch(m_commandBuffer, groupsX, groupsY, 1);

        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, 3);
    } else {
        // --- Multi-GPU Split-Frame Tiling / Dynamic Work Queue Path ---
        uint32_t h0 = m_config.height / 2;
        uint32_t h1 = m_config.height - h0;

        CameraUniform ubo = m_camera->getUniformData(m_frameIndex, m_config.spp, m_config.max_bounces, flags);
        m_cameraUBO->copyFrom(&ubo, sizeof(CameraUniform));

        uint32_t useHwRT = (m_tlas != nullptr && m_config.enable_hardware_rt) ? 1 : 0;
        uint32_t hasEnvMap = m_environmentMap ? 1 : 0;
        float envIntensity = 1.0f;
        uint32_t envIntensityBits = std::bit_cast<uint32_t>(envIntensity);

        size_t tileBytes = static_cast<size_t>(m_config.width) * h1 * 4 * sizeof(float);
        void* dstHost = m_secTransferBuffer->map();

        // 1. Launch secondary GPU on bottom half asynchronously (accumulateHistory = 1: secondary retains bottom-half temporal accumulation)
        m_mgpu->launchSecondaryWork(ubo, m_frameIndex, 0, h0, m_config.width, h1,
                                   m_numTriangles, m_numSpheres, m_numMaterials, m_numLights, useHwRT,
                                   hasEnvMap, envIntensity, 1u, dstHost, tileBytes);

        // 2. Concurrently record and execute primary GPU on top half
        vkWaitForFences(device, 1, &m_rtFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &m_rtFence);

        vkResetCommandBuffer(m_commandBuffer, 0);
        VkCommandBufferBeginInfo rtBeginInfo{};
        rtBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(m_commandBuffer, &rtBeginInfo);

        vkCmdResetQueryPool(m_commandBuffer, m_queryPool, 0, 4);
        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, 0);

        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipeline);
        vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_rtPipelineLayout, 0, 1, &m_rtDescSet, 0, nullptr);

        uint32_t rtPushConstants[12] = {
            m_numTriangles, m_numSpheres, m_numMaterials, m_numLights,
            0, 0, m_config.width, h0,
            useHwRT,
            hasEnvMap,
            envIntensityBits,
            1u // accumulateHistory
        };
        vkCmdPushConstants(m_commandBuffer, m_rtPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rtPushConstants), rtPushConstants);

        uint32_t groupsY0 = (h0 + 3) / 4;
        vkCmdDispatch(m_commandBuffer, rtGroupsX, groupsY0, 1);

        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, 1);
        vkEndCommandBuffer(m_commandBuffer);

        VkSubmitInfo rtSubmit{};
        rtSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        rtSubmit.commandBufferCount = 1;
        rtSubmit.pCommandBuffers = &m_commandBuffer;
        vkQueueSubmit(queue, 1, &rtSubmit, m_rtFence);

        // 3. Wait for primary GPU raytracing and secondary GPU completion + PCIe transfer
        vkWaitForFences(device, 1, &m_rtFence, VK_TRUE, UINT64_MAX);
        m_mgpu->syncAndTransfer(nullptr, 0);
        m_secTransferBuffer->unmap();

        // 4. Record Buffer-to-Image Copy for bottom half & Tonemapping commands
        vkResetCommandBuffer(m_commandBuffer, 0);
        VkCommandBufferBeginInfo postBeginInfo{};
        postBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(m_commandBuffer, &postBeginInfo);

        // Copy secondary tile to accumImage bottom half
        m_accumImage->transitionLayout(
            m_commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
        );

        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageOffset = { 0, static_cast<int32_t>(h0), 0 };
        copyRegion.imageExtent = { m_config.width, h1, 1 };

        vkCmdCopyBufferToImage(m_commandBuffer, m_secTransferBuffer->getBuffer(),
                               m_accumImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        m_accumImage->transitionLayout(
            m_commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        // Tonemapping
        vkCmdBindPipeline(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipeline);
        vkCmdBindDescriptorSets(m_commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &m_tonemapDescSet, 0, nullptr);

        tonemapConstants.totalSamples = (m_frameIndex + 1) * m_config.spp;
        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPool, 2);
        vkCmdPushConstants(m_commandBuffer, m_tonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tonemapConstants), &tonemapConstants);
        vkCmdDispatch(m_commandBuffer, groupsX, groupsY, 1);

        vkCmdWriteTimestamp2(m_commandBuffer, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, 3);
    }

    // 3. Interactive Blit & Dear ImGui Overlay
    if (!m_config.headless && m_swapchain) {
        VkImage swapImage = m_swapchain->getImage(imageIndex);
        VkImageView swapView = m_swapchain->getImageView(imageIndex);

        // Transition m_outputImage to TRANSFER_SRC_OPTIMAL
        m_outputImage->transitionLayout(
            m_commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
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
        vkCmdPipelineBarrier2(m_commandBuffer, &depToDst);

        // Copy or blit output image to swapchain image
        if (m_config.width == m_swapchain->getExtent().width &&
            m_config.height == m_swapchain->getExtent().height) {
            VkImageCopy copyRegion{};
            copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.extent = { m_config.width, m_config.height, 1 };
            vkCmdCopyImage(m_commandBuffer, m_outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);
        } else {
            VkImageBlit blitRegion{};
            blitRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blitRegion.srcOffsets[0] = { 0, 0, 0 };
            blitRegion.srcOffsets[1] = { static_cast<int32_t>(m_config.width), static_cast<int32_t>(m_config.height), 1 };
            blitRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blitRegion.dstOffsets[0] = { 0, 0, 0 };
            blitRegion.dstOffsets[1] = { static_cast<int32_t>(m_swapchain->getExtent().width), static_cast<int32_t>(m_swapchain->getExtent().height), 1 };
            vkCmdBlitImage(m_commandBuffer, m_outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, VK_FILTER_LINEAR);
        }

        // Transition m_outputImage back to GENERAL
        m_outputImage->transitionLayout(
            m_commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
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
        vkCmdPipelineBarrier2(m_commandBuffer, &depToColor);

        // Render ImGui overlay
        if (m_gui) {
            m_gui->newFrame();
            bool prevMode = m_cameraMode;
            GuiActions guiActions{};
            if (m_gui->render(m_commandBuffer, swapView, m_swapchain->getExtent().width, m_swapchain->getExtent().height,
                              m_config, getStats(), m_cameraMode, m_camera.get(),
                              &m_window->getDisplayInfo(), m_window->isFullscreen(), &guiActions)) {
                m_resetAccumulation = true;
                if (!m_config.headless) m_frameTimesMs.clear();
            }
            if (guiActions.resetAccumulation) {
                m_resetAccumulation = true;
                if (!m_config.headless) m_frameTimesMs.clear();
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
            vkCmdPipelineBarrier2(m_commandBuffer, &depToSrc);

            if (!m_uiDumpBuffer) {
                VkDeviceSize size = static_cast<VkDeviceSize>(m_swapchain->getExtent().width) * m_swapchain->getExtent().height * 4;
                m_uiDumpBuffer = std::make_unique<Buffer>(m_context->getAllocator(), size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                         VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
            }

            VkBufferImageCopy copyRegion{};
            copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            copyRegion.imageSubresource.layerCount = 1;
            copyRegion.imageExtent = { m_swapchain->getExtent().width, m_swapchain->getExtent().height, 1 };
            vkCmdCopyImageToBuffer(m_commandBuffer, swapImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_uiDumpBuffer->getBuffer(), 1, &copyRegion);

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
            vkCmdPipelineBarrier2(m_commandBuffer, &depToPresent);
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
            vkCmdPipelineBarrier2(m_commandBuffer, &depToPresent);
        }
    }

    vkEndCommandBuffer(m_commandBuffer);

    // Submit Work
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffer;

    VkPipelineStageFlags waitStages[] = { VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT };
    if (!m_config.headless && m_swapchain) {
        submitInfo.waitSemaphoreCount = 1;
        submitInfo.pWaitSemaphores = &m_imageAvailableSemaphores[m_currentFrame];
        submitInfo.pWaitDstStageMask = waitStages;
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &m_renderFinishedSemaphores[imageIndex];
    }

    vkQueueSubmit(queue, 1, &submitInfo, m_inFlightFence);

    if (!m_config.headless && m_swapchain) {
        VkResult res = m_swapchain->queuePresent(queue, imageIndex, m_renderFinishedSemaphores[imageIndex]);
        if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
            onResize(m_window->getWidth(), m_window->getHeight());
        }
    } else {
        vkWaitForFences(device, 1, &m_inFlightFence, VK_TRUE, UINT64_MAX);
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

    auto frameEndTime = std::chrono::high_resolution_clock::now();
    double frameDurationMs = std::chrono::duration<double, std::milli>(frameEndTime - frameStartTime).count();

    // Query GPU Timestamps with underflow guards
    uint64_t timestamps[4] = {0, 0, 0, 0};
    vkGetQueryPoolResults(device, m_queryPool, 0, 4, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
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
    m_lastFrameTimeMs = totalGpuMs > 0.01 ? totalGpuMs : frameDurationMs;

    m_frameTimesMs.push_back(m_lastFrameTimeMs);
    if (!m_config.headless && m_frameTimesMs.size() > 60) {
        m_frameTimesMs.erase(m_frameTimesMs.begin());
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

    // 1. Dump LDR PNG
    if (!m_config.dump_frame_path.empty()) {
        VkDeviceSize bufferSize = m_config.width * m_config.height * 4;
        Buffer staging(allocator, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        vkResetCommandBuffer(m_commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(m_commandBuffer, &beginInfo);

        m_outputImage->transitionLayout(
            m_commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
        );

        VkBufferImageCopy copyRegion{};
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = { m_config.width, m_config.height, 1 };

        vkCmdCopyImageToBuffer(m_commandBuffer, m_outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.getBuffer(), 1, &copyRegion);

        m_outputImage->transitionLayout(
            m_commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );

        vkEndCommandBuffer(m_commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffer;
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

        vkResetCommandBuffer(m_commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(m_commandBuffer, &beginInfo);

        m_accumImage->transitionLayout(
            m_commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
        );

        VkBufferImageCopy copyRegion{};
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = { m_config.width, m_config.height, 1 };

        vkCmdCopyImageToBuffer(m_commandBuffer, m_accumImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.getBuffer(), 1, &copyRegion);

        m_accumImage->transitionLayout(
            m_commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );

        vkEndCommandBuffer(m_commandBuffer);

        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &m_commandBuffer;
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
    if (m_mgpu && m_mgpu->isMultiGpuActive()) {
        stats.gpu_name = m_context->getDeviceName() + " + " + m_mgpu->getSecondaryDeviceName();
    } else {
        stats.gpu_name = m_context->getDeviceName();
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
        case MultiGpuMode::SampleParallel: stats.mgpu_mode_str = "sample_parallel"; break;
        case MultiGpuMode::CheckerboardTile: stats.mgpu_mode_str = "checkerboard_tile"; break;
        case MultiGpuMode::DynamicWorkQueue: stats.mgpu_mode_str = "dynamic_work_queue"; break;
        default: stats.mgpu_mode_str = "single_gpu"; break;
    }

    stats.pipeline_type_str = (m_config.pipeline_type == PipelineType::Wavefront) ? "wavefront" : "megakernel";

    stats.primary_gpu_time_ms = m_lastGpuRtMs;
    stats.secondary_gpu_time_ms = m_lastSecGpuMs;
    stats.tonemap_time_ms = m_lastTonemapMs;
    stats.num_triangles = m_numTriangles;
    stats.num_spheres = m_numSpheres;
    stats.num_materials = m_numMaterials;
    stats.num_lights = m_numLights;
    stats.num_textures = static_cast<uint32_t>(m_sceneTextures.size());
    stats.has_hw_rt = (m_tlas != nullptr && m_config.enable_hardware_rt);
    stats.has_dgc = (m_dgc != nullptr);
    stats.is_rdna3 = m_context->isRDNA3();
    stats.is_rdna4 = m_context->isRDNA4();
    stats.arch_name = m_context->getArchitectureName();
    stats.short_arch = m_context->getShortArchName();
    stats.ray_accelerator_name = m_context->getRayAcceleratorName();

    return stats;
}

void Engine::onResize(uint32_t newWidth, uint32_t newHeight) {
    if (m_config.headless || !m_swapchain) return;
    if (newWidth == 0 || newHeight == 0) return;

    newWidth = std::max(64u, newWidth);
    newHeight = std::max(64u, newHeight);

    if (newWidth == m_config.width && newHeight == m_config.height &&
        m_swapchain->getExtent().width == newWidth && m_swapchain->getExtent().height == newHeight) {
        m_resetAccumulation = true;
        if (m_dgc && m_dgcArgumentBuffer) {
            VkDispatchIndirectCommand dgcCmd{};
            dgcCmd.x = (m_config.width + 7) / 8;
            dgcCmd.y = (m_config.height + 3) / 4;
            dgcCmd.z = 1;
            m_dgcArgumentBuffer->copyFrom(&dgcCmd, sizeof(VkDispatchIndirectCommand));
        }
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
    m_accumImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R32G32B32A32_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_outputImage = std::make_unique<Image>(
        device, allocator, m_config.width, m_config.height,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // Transition images to GENERAL layout
    vkResetCommandBuffer(m_commandBuffer, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(m_commandBuffer, &beginInfo);

    m_accumImage->transitionLayout(
        m_commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    m_outputImage->transitionLayout(
        m_commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    vkEndCommandBuffer(m_commandBuffer);

    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &m_commandBuffer;
    vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);

    // 4. Update descriptor sets with new image views
    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageView = m_accumImage->getImageView();
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo outputImageInfo{};
    outputImageInfo.imageView = m_outputImage->getImageView();
    outputImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::vector<VkWriteDescriptorSet> writes;
    // Update m_rtDescSet: binding 0 is m_accumImage (storage image)
    VkWriteDescriptorSet w0{};
    w0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w0.dstSet = m_rtDescSet;
    w0.dstBinding = 0;
    w0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w0.descriptorCount = 1;
    w0.pImageInfo = &accumImageInfo;
    writes.push_back(w0);

    // Update m_tonemapDescSet: binding 0 is m_accumImage, binding 1 is m_outputImage
    VkWriteDescriptorSet w1{};
    w1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w1.dstSet = m_tonemapDescSet;
    w1.dstBinding = 0;
    w1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w1.descriptorCount = 1;
    w1.pImageInfo = &accumImageInfo;
    writes.push_back(w1);

    VkWriteDescriptorSet w2{};
    w2.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    w2.dstSet = m_tonemapDescSet;
    w2.dstBinding = 1;
    w2.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w2.descriptorCount = 1;
    w2.pImageInfo = &outputImageInfo;
    writes.push_back(w2);

    // Multi-GPU merge descriptor set
    VkDescriptorBufferInfo secBufInfo{};
    if (m_secTransferBuffer) {
        VkDeviceSize bufferSize = static_cast<VkDeviceSize>(m_config.width) * m_config.height * 4 * sizeof(float);
        m_secTransferBuffer = std::make_unique<Buffer>(
            allocator, bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
        );
        secBufInfo = { m_secTransferBuffer->getBuffer(), 0, bufferSize };

        if (m_mergeDescSet != VK_NULL_HANDLE) {
            VkWriteDescriptorSet mw0{};
            mw0.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            mw0.dstSet = m_mergeDescSet;
            mw0.dstBinding = 0;
            mw0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            mw0.descriptorCount = 1;
            mw0.pImageInfo = &accumImageInfo;
            writes.push_back(mw0);

            VkWriteDescriptorSet mw1{};
            mw1.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            mw1.dstSet = m_mergeDescSet;
            mw1.dstBinding = 1;
            mw1.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            mw1.descriptorCount = 1;
            mw1.pBufferInfo = &secBufInfo;
            writes.push_back(mw1);
        }
    }

    // Wavefront descriptor sets update with new m_accumImage view
    if (m_wfClassifyDescSet != VK_NULL_HANDLE) {
        VkWriteDescriptorSet cw0{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        cw0.dstSet = m_wfClassifyDescSet;
        cw0.dstBinding = 0;
        cw0.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        cw0.descriptorCount = 1;
        cw0.pImageInfo = &accumImageInfo;
        writes.push_back(cw0);
    }
    if (m_wfShadeDescSetA != VK_NULL_HANDLE) {
        VkWriteDescriptorSet swA{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        swA.dstSet = m_wfShadeDescSetA;
        swA.dstBinding = 0;
        swA.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        swA.descriptorCount = 1;
        swA.pImageInfo = &accumImageInfo;
        writes.push_back(swA);
    }
    if (m_wfShadeDescSetB != VK_NULL_HANDLE) {
        VkWriteDescriptorSet swB{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        swB.dstSet = m_wfShadeDescSetB;
        swB.dstBinding = 0;
        swB.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        swB.descriptorCount = 1;
        swB.pImageInfo = &accumImageInfo;
        writes.push_back(swB);
    }

    // Reallocate wavefront ray queues if new resolution exceeds capacity
    VkDeviceSize requiredQueueSize = static_cast<VkDeviceSize>(m_config.width) * m_config.height * 96;
    VkDescriptorBufferInfo qAInfo{};
    VkDescriptorBufferInfo qBInfo{};
    if (m_rayQueueA && requiredQueueSize > m_rayQueueA->getSize()) {
        m_rayQueueA = std::make_unique<Buffer>(
            allocator, requiredQueueSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );
        m_rayQueueB = std::make_unique<Buffer>(
            allocator, requiredQueueSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );

        qAInfo = { m_rayQueueA->getBuffer(), 0, requiredQueueSize };
        qBInfo = { m_rayQueueB->getBuffer(), 0, requiredQueueSize };

        // Classify binding 9 (queue A)
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfClassifyDescSet, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &qAInfo, nullptr });
        // Shade A binding 9 (in: A), 10 (out: B)
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfShadeDescSetA, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &qAInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfShadeDescSetA, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &qBInfo, nullptr });
        // Shade B binding 9 (in: B), 10 (out: A)
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfShadeDescSetB, 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &qBInfo, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_wfShadeDescSetB, 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &qAInfo, nullptr });
    }

    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // 5. Reset UI dump buffer if allocated
    if (m_uiDumpBuffer) {
        m_uiDumpBuffer.reset();
    }

    // 6. Update DGC argument buffer with new dispatch dimensions
    if (m_dgc && m_dgcArgumentBuffer) {
        VkDispatchIndirectCommand dgcCmd{};
        dgcCmd.x = (m_config.width + 7) / 8;
        dgcCmd.y = (m_config.height + 3) / 4;
        dgcCmd.z = 1;
        m_dgcArgumentBuffer->copyFrom(&dgcCmd, sizeof(VkDispatchIndirectCommand));
        Logger::info("DGC indirect dispatch buffer updated: {}x{} workgroups ({}x{} pixels).",
                     dgcCmd.x, dgcCmd.y, m_config.width, m_config.height);
    }

    // 7. Resize secondary GPU if active
    if (m_mgpu && m_mgpu->isMultiGpuActive()) {
        m_mgpu->resize(m_config.width, m_config.height);
    }

    // 8. Adapt Camera aspect ratio & FOV
    if (m_camera) {
        float aspect = static_cast<float>(m_config.width) / static_cast<float>(m_config.height);
        m_camera->adaptFovForAspect(aspect);
    }

    // 9. Invalidate accumulation
    m_frameIndex = 0;
    m_resetAccumulation = true;

    Logger::info("Swapchain & render pipelines successfully recreated for {}x{}", m_config.width, m_config.height);
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
