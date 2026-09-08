#include "mgpu/MultiGpuManager.hpp"
#include "core/Logger.hpp"

#include <fstream>
#include <filesystem>
#include <cstring>
#include <bit>
#include <thread>
#include <vector>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

namespace pathways {

static void parallelMemcpy(void* dst, const void* src, size_t size, size_t numThreads = 8) {
    if (!dst || !src || size == 0 || dst == src) return;
    if (size < 1024 * 512 || numThreads <= 1) {
        std::memcpy(dst, src, size);
        return;
    }
    const size_t chunkSize = size / numThreads;
    std::vector<std::thread> workers;
    workers.reserve(numThreads - 1);
    for (size_t t = 1; t < numThreads; ++t) {
        size_t offset = t * chunkSize;
        size_t len = (t == numThreads - 1) ? (size - offset) : chunkSize;
        workers.emplace_back([=]() {
            std::memcpy(static_cast<char*>(dst) + offset, static_cast<const char*>(src) + offset, len);
        });
    }
    std::memcpy(dst, src, chunkSize);
    for (auto& w : workers) {
        w.join();
    }
}

GpuDeviceNode::~GpuDeviceNode() {
    if (!context) return;
    VkDevice device = context->getDevice();
    vkDeviceWaitIdle(device);

    rtpKhrPipeline.reset();
    if (rtpPipelineLayout) vkDestroyPipelineLayout(device, rtpPipelineLayout, nullptr);
    if (queryPool) vkDestroyQueryPool(device, queryPool, nullptr);
    if (rtPipeline) vkDestroyPipeline(device, rtPipeline, nullptr);
    if (rtPipelineLayout) vkDestroyPipelineLayout(device, rtPipelineLayout, nullptr);
    if (rtDescLayout) vkDestroyDescriptorSetLayout(device, rtDescLayout, nullptr);
    if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (renderFence) vkDestroyFence(device, renderFence, nullptr);
    if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);

    accumTarget.reset();
    p2pStagingBuffer.reset();
    triangleBuffer.reset();
    sphereBuffer.reset();
    materialBuffer.reset();
    lightBuffer.reset();
    cameraUBO.reset();
}

MultiGpuManager::MultiGpuManager(const Config& config, VulkanContext* primaryContext, const SceneData& scene)
    : m_primaryContext(primaryContext), m_mode(config.mgpu_mode), m_config(config) {

    if (m_mode == MultiGpuMode::Off) {
        Logger::info("Multi-GPU execution disabled (running in Single-GPU mode).");
        return;
    }

    auto devices = VulkanContext::enumeratePhysicalDevices(primaryContext->getInstance());
    if (devices.size() < 2) {
        Logger::warn("Multi-GPU requested but only {} physical Vulkan device(s) found. Falling back to single GPU.", devices.size());
        m_mode = MultiGpuMode::Off;
        return;
    }

    Logger::info("Initializing Multi-GPU Manager across {} discrete GPUs...", devices.size());
    initSecondaryDevice(config, scene);
}

MultiGpuManager::~MultiGpuManager() {
    if (m_asyncTask.valid()) {
        m_asyncTask.wait();
    }
    destroySharedHostBuffer();
    m_devices.clear();
}

double MultiGpuManager::getSecondaryGpuTimeMs() const {
    if (m_devices.empty()) return 0.0;
    return m_devices[0]->lastFrameTimeMs;
}

const std::string& MultiGpuManager::getSecondaryDeviceName() const {
    static const std::string empty;
    if (m_devices.empty()) return empty;
    return m_devices[0]->deviceName;
}

std::vector<char> MultiGpuManager::loadShaderSPIRV(const std::string& filename) {
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
                return buffer;
            }
        }
    }
    throw std::runtime_error("Could not find compiled SPIR-V file: " + filename);
}

VkShaderModule MultiGpuManager::createShaderModule(VkDevice device, const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    VkResult res = vkCreateShaderModule(device, &createInfo, nullptr, &shaderModule);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create secondary GPU shader module!");
    }
    return shaderModule;
}

void MultiGpuManager::destroySharedHostBuffer() {
    VkDevice dev0 = m_primaryContext ? m_primaryContext->getDevice() : VK_NULL_HANDLE;
    VkDevice dev1 = (!m_devices.empty() && m_devices[0]->context) ? m_devices[0]->context->getDevice() : VK_NULL_HANDLE;

    for (uint32_t slot = 0; slot < NUM_SHARED_BUFFERS; ++slot) {
        if (dev0 != VK_NULL_HANDLE) {
            if (m_sharedBufferPrimary[slot] != VK_NULL_HANDLE) {
                vkDestroyBuffer(dev0, m_sharedBufferPrimary[slot], nullptr);
                m_sharedBufferPrimary[slot] = VK_NULL_HANDLE;
            }
            if (m_sharedMemPrimary[slot] != VK_NULL_HANDLE) {
                vkFreeMemory(dev0, m_sharedMemPrimary[slot], nullptr);
                m_sharedMemPrimary[slot] = VK_NULL_HANDLE;
            }
        }
        if (dev1 != VK_NULL_HANDLE) {
            if (m_sharedBufferSecondary[slot] != VK_NULL_HANDLE) {
                vkDestroyBuffer(dev1, m_sharedBufferSecondary[slot], nullptr);
                m_sharedBufferSecondary[slot] = VK_NULL_HANDLE;
            }
            if (m_sharedMemSecondary[slot] != VK_NULL_HANDLE) {
                vkFreeMemory(dev1, m_sharedMemSecondary[slot], nullptr);
                m_sharedMemSecondary[slot] = VK_NULL_HANDLE;
            }
        }
        if (m_sharedHostPtr[slot]) {
            free(m_sharedHostPtr[slot]);
            m_sharedHostPtr[slot] = nullptr;
        }
    }
    m_sharedBufferSize = 0;
    m_useZeroCopyHost = false;
}

void MultiGpuManager::initSharedHostBuffer(VkDeviceSize bufferSize) {
    destroySharedHostBuffer();

    if (!m_primaryContext || !m_primaryContext->hasExternalMemoryHost() ||
        m_devices.empty() || !m_devices[0]->context || !m_devices[0]->context->hasExternalMemoryHost()) {
        Logger::warn("VK_EXT_external_memory_host not available on both devices. Falling back to CPU staging copy.");
        m_useZeroCopyHost = false;
        return;
    }

    m_sharedBufferSize = bufferSize;
    VkDevice dev0 = m_primaryContext->getDevice();
    VkPhysicalDevice phys0 = m_primaryContext->getPhysicalDevice();
    VkDevice dev1 = m_devices[0]->context->getDevice();
    VkPhysicalDevice phys1 = m_devices[0]->context->getPhysicalDevice();

    auto pfnGet0 = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev0, "vkGetMemoryHostPointerPropertiesEXT");
    auto pfnGet1 = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev1, "vkGetMemoryHostPointerPropertiesEXT");

    if (!pfnGet0 || !pfnGet1) {
        Logger::warn("Could not retrieve vkGetMemoryHostPointerPropertiesEXT function pointer.");
        m_useZeroCopyHost = false;
        return;
    }

    VkPhysicalDeviceMemoryProperties memProps0, memProps1;
    vkGetPhysicalDeviceMemoryProperties(phys0, &memProps0);
    vkGetPhysicalDeviceMemoryProperties(phys1, &memProps1);

    bool allSucceeded = true;
    for (uint32_t slot = 0; slot < NUM_SHARED_BUFFERS; ++slot) {
        if (posix_memalign(&m_sharedHostPtr[slot], 4096, m_sharedBufferSize) != 0 || !m_sharedHostPtr[slot]) {
            Logger::warn("Failed to allocate page-aligned host memory for slot {}.", slot);
            allSucceeded = false;
            break;
        }
        std::memset(m_sharedHostPtr[slot], 0, m_sharedBufferSize);

        VkMemoryHostPointerPropertiesEXT hostProps0{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        pfnGet0(dev0, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, m_sharedHostPtr[slot], &hostProps0);

        VkMemoryHostPointerPropertiesEXT hostProps1{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        pfnGet1(dev1, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, m_sharedHostPtr[slot], &hostProps1);

        uint32_t memIdx0 = UINT32_MAX, memIdx1 = UINT32_MAX;
        for (uint32_t i = 0; i < memProps0.memoryTypeCount; ++i) {
            if (hostProps0.memoryTypeBits & (1 << i)) { memIdx0 = i; break; }
        }
        for (uint32_t i = 0; i < memProps1.memoryTypeCount; ++i) {
            if (hostProps1.memoryTypeBits & (1 << i)) { memIdx1 = i; break; }
        }

        if (memIdx0 == UINT32_MAX || memIdx1 == UINT32_MAX) {
            Logger::warn("No compatible memory type found for host pointer import on slot {}.", slot);
            allSucceeded = false;
            break;
        }

        // Dev 0: Import memory and create buffer
        VkImportMemoryHostPointerInfoEXT import0{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
        import0.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        import0.pHostPointer = m_sharedHostPtr[slot];

        VkMemoryAllocateInfo alloc0{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc0.pNext = &import0;
        alloc0.allocationSize = m_sharedBufferSize;
        alloc0.memoryTypeIndex = memIdx0;

        VkResult resMem0 = vkAllocateMemory(dev0, &alloc0, nullptr, &m_sharedMemPrimary[slot]);

        VkExternalMemoryBufferCreateInfo extBufInfo0{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        extBufInfo0.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

        VkBufferCreateInfo bufInfo0{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufInfo0.pNext = &extBufInfo0;
        bufInfo0.size = m_sharedBufferSize;
        bufInfo0.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufInfo0.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkResult resBuf0 = vkCreateBuffer(dev0, &bufInfo0, nullptr, &m_sharedBufferPrimary[slot]);
        VkResult resBind0 = vkBindBufferMemory(dev0, m_sharedBufferPrimary[slot], m_sharedMemPrimary[slot], 0);

        // Dev 1: Import memory and create buffer
        VkImportMemoryHostPointerInfoEXT import1{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
        import1.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        import1.pHostPointer = m_sharedHostPtr[slot];

        VkMemoryAllocateInfo alloc1{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        alloc1.pNext = &import1;
        alloc1.allocationSize = m_sharedBufferSize;
        alloc1.memoryTypeIndex = memIdx1;

        VkResult resMem1 = vkAllocateMemory(dev1, &alloc1, nullptr, &m_sharedMemSecondary[slot]);

        VkExternalMemoryBufferCreateInfo extBufInfo1{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
        extBufInfo1.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

        VkBufferCreateInfo bufInfo1{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufInfo1.pNext = &extBufInfo1;
        bufInfo1.size = m_sharedBufferSize;
        bufInfo1.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufInfo1.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkResult resBuf1 = vkCreateBuffer(dev1, &bufInfo1, nullptr, &m_sharedBufferSecondary[slot]);
        VkResult resBind1 = vkBindBufferMemory(dev1, m_sharedBufferSecondary[slot], m_sharedMemSecondary[slot], 0);

        if (resMem0 != VK_SUCCESS || resBuf0 != VK_SUCCESS || resBind0 != VK_SUCCESS ||
            resMem1 != VK_SUCCESS || resBuf1 != VK_SUCCESS || resBind1 != VK_SUCCESS) {
            allSucceeded = false;
            break;
        }
    }

    if (allSucceeded) {
        m_useZeroCopyHost = true;
        Logger::info("Double-Buffered Zero-Copy Inter-GPU Host Buffers initialized via VK_EXT_external_memory_host (2x {:.2f} MB).",
                     static_cast<double>(m_sharedBufferSize) / (1024.0 * 1024.0));
    } else {
        Logger::warn("Failed to bind zero-copy host buffers on both GPUs. Falling back to CPU staging.");
        destroySharedHostBuffer();
        m_useZeroCopyHost = false;
    }
}

void MultiGpuManager::initSecondaryDevice(const Config& config, const SceneData& scene) {
    Config secConfig = config;
    secConfig.gpu_index = 1; // Explicit secondary GPU
    secConfig.headless = true; // Secondary GPU always runs headless compute

    auto secNode = std::make_unique<GpuDeviceNode>();
    secNode->deviceIndex = 1;
    secNode->context = std::make_unique<VulkanContext>(secConfig);
    secNode->deviceName = secNode->context->getDeviceName();
    secNode->timestampPeriod = secNode->context->getDeviceProperties().limits.timestampPeriod;

    VkDevice secDevice = secNode->context->getDevice();
    VmaAllocator secAlloc = secNode->context->getAllocator();

    // 1. Command Pool & Command Buffer
    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = secNode->context->getGraphicsQueueFamily();
    vkCreateCommandPool(secDevice, &poolInfo, nullptr, &secNode->commandPool);

    VkCommandBufferAllocateInfo cmdAllocInfo{};
    cmdAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAllocInfo.commandPool = secNode->commandPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;
    vkAllocateCommandBuffers(secDevice, &cmdAllocInfo, &secNode->commandBuffer);

    // 2. Fence & Query Pool
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    vkCreateFence(secDevice, &fenceInfo, nullptr, &secNode->renderFence);

    VkQueryPoolCreateInfo queryInfo{};
    queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = 2; // Start, End
    vkCreateQueryPool(secDevice, &queryInfo, nullptr, &secNode->queryPool);

    // 3. Render Targets on secondary device (half-width for CheckerboardTile to cut VRAM and PCIe footprint by 50%)
    uint32_t secWidth = (config.mgpu_mode == MultiGpuMode::CheckerboardTile) ? ((config.width + 1) / 2) : config.width;
    VkFormat accumFormat = (config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
    secNode->accumTarget = std::make_unique<Image>(
        secDevice, secAlloc, secWidth, config.height,
        accumFormat,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // 4. Scene Buffers on secondary device
    VkDeviceSize triSize = std::max(sizeof(TriangleGPU) * scene.triangles.size(), sizeof(TriangleGPU));
    secNode->triangleBuffer = std::make_unique<Buffer>(
        secAlloc, triSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.triangles.empty()) {
        secNode->triangleBuffer->copyFrom(scene.triangles.data(), sizeof(TriangleGPU) * scene.triangles.size());
    }

    VkDeviceSize sphereSize = std::max(sizeof(SphereGPU) * scene.spheres.size(), sizeof(SphereGPU));
    secNode->sphereBuffer = std::make_unique<Buffer>(
        secAlloc, sphereSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.spheres.empty()) {
        secNode->sphereBuffer->copyFrom(scene.spheres.data(), sizeof(SphereGPU) * scene.spheres.size());
    }

    VkDeviceSize matSize = std::max(sizeof(MaterialGPU) * scene.materials.size(), sizeof(MaterialGPU));
    secNode->materialBuffer = std::make_unique<Buffer>(
        secAlloc, matSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.materials.empty()) {
        secNode->materialBuffer->copyFrom(scene.materials.data(), sizeof(MaterialGPU) * scene.materials.size());
    }

    VkDeviceSize lightSize = std::max(sizeof(LightGPU) * scene.lights.size(), sizeof(LightGPU));
    secNode->lightBuffer = std::make_unique<Buffer>(
        secAlloc, lightSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.lights.empty()) {
        secNode->lightBuffer->copyFrom(scene.lights.data(), sizeof(LightGPU) * scene.lights.size());
    }

    VkDeviceSize uboSize = sizeof(CameraUniform);
    secNode->cameraUBO = std::make_unique<Buffer>(
        secAlloc, uboSize,
        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    // Hardware Acceleration Structures on secondary device (VK_KHR_ray_query)
    if (secNode->context->hasRayTracing()) {
        std::vector<Vertex> asVertices;
        if (!scene.triangles.empty()) {
            asVertices.reserve(scene.triangles.size() * 3);
            for (const auto& tri : scene.triangles) {
                asVertices.push_back(tri.v0);
                asVertices.push_back(tri.v1);
                asVertices.push_back(tri.v2);
            }
        } else {
            Vertex v{};
            asVertices.assign(3, v);
        }

        VkDeviceSize vertexBufferSize = sizeof(Vertex) * asVertices.size();
        secNode->asVertexBuffer = std::make_unique<Buffer>(
            secAlloc, vertexBufferSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        );
        secNode->asVertexBuffer->copyFrom(asVertices.data(), vertexBufferSize);

        secNode->asManager = std::make_unique<AccelerationStructureManager>(
            secDevice, secAlloc,
            secNode->context->getGraphicsQueue(), secNode->context->getGraphicsQueueFamily()
        );

        ASGeometryInput geom{};
        geom.vertexBufferAddress = secNode->asVertexBuffer->getDeviceAddress(secDevice);
        geom.indexBufferAddress = 0;
        geom.vertexCount = static_cast<uint32_t>(asVertices.size());
        geom.triangleCount = static_cast<uint32_t>(asVertices.size() / 3);
        geom.vertexStride = sizeof(Vertex);
        geom.indexType = VK_INDEX_TYPE_NONE_KHR;
        bool hasAlphaMask = false;
        for (const auto& mat : scene.materials) {
            if (mat.alphaMode == ALPHA_MODE_MASK) {
                hasAlphaMask = true;
                break;
            }
        }
        geom.isOpaque = !hasAlphaMask;

        secNode->blas = secNode->asManager->buildBLAS({ geom });

        ASInstanceInput inst{};
        inst.blasAddress = secNode->blas->getDeviceAddress();
        inst.transform = glm::mat4(1.0f);
        inst.customIndex = 0;
        inst.mask = 0xFF;
        inst.hitGroupId = 0;
        inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;

        secNode->tlas = secNode->asManager->buildTLAS({ inst });
        Logger::info("Secondary GPU Acceleration Structures initialized successfully (BLAS & TLAS).");
        Logger::info("Secondary GPU RT Pipeline Active. Extensions in use: VK_KHR_ray_query, VK_KHR_acceleration_structure, VK_KHR_buffer_device_address, VK_KHR_deferred_host_operations");
    }

    // Textures & HDRI Environment Map on secondary device
    VkQueue secQueue = secNode->context->getGraphicsQueue();
    VkCommandPool secPool = secNode->commandPool;

    secNode->dummyWhite = Texture::createDummyWhite(secDevice, secAlloc, secQueue, secPool);
    secNode->dummyNormal = Texture::createDummyNormal(secDevice, secAlloc, secQueue, secPool);

    if (!config.hdri_path.empty() && std::filesystem::exists(config.hdri_path)) {
        secNode->environmentMap = Texture::loadFromFile(secDevice, secAlloc, secQueue, secPool, config.hdri_path);
    }
    if (!secNode->environmentMap) {
        secNode->environmentMap = Texture::createProceduralHdrSky(secDevice, secAlloc, secQueue, secPool);
    }

    secNode->sceneTextures.clear();
    for (const auto& texData : scene.textures) {
        if (!texData.pixels.empty() && texData.width > 0 && texData.height > 0) {
            VkFormat fmt = texData.isSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
            auto tex = Texture::createFromPixels(
                secDevice, secAlloc, secQueue, secPool,
                texData.width, texData.height,
                fmt, texData.pixels.data(),
                texData.pixels.size(), false
            );
            secNode->sceneTextures.push_back(std::move(tex));
        } else {
            secNode->sceneTextures.push_back(Texture::createDummyWhite(secDevice, secAlloc, secQueue, secPool));
        }
    }

    // 5. Descriptor Pool & Sets on secondary device
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 4 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 512 }
    };
    VkDescriptorPoolCreateInfo descPoolInfo{};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    descPoolInfo.pPoolSizes = poolSizes.data();
    descPoolInfo.maxSets = 8;
    vkCreateDescriptorPool(secDevice, &descPoolInfo, nullptr, &secNode->descriptorPool);

    VkShaderStageFlags rtStages = VK_SHADER_STAGE_COMPUTE_BIT;
    if (secNode->context->hasRayTracing()) {
        rtStages |= VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    }

    std::vector<VkDescriptorSetLayoutBinding> rtBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, rtStages, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, rtStages, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, rtStages, nullptr },
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_SCENE_TEXTURES, rtStages, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo rtLayoutInfo{};
    rtLayoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    rtLayoutInfo.bindingCount = static_cast<uint32_t>(rtBindings.size());
    rtLayoutInfo.pBindings = rtBindings.data();
    vkCreateDescriptorSetLayout(secDevice, &rtLayoutInfo, nullptr, &secNode->rtDescLayout);

    VkDescriptorSetAllocateInfo descAllocInfo{};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = secNode->descriptorPool;
    descAllocInfo.descriptorSetCount = 1;
    descAllocInfo.pSetLayouts = &secNode->rtDescLayout;
    vkAllocateDescriptorSets(secDevice, &descAllocInfo, &secNode->rtDescSet);

    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageView = secNode->accumTarget->getImageView();
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo uboInfo{ secNode->cameraUBO->getBuffer(), 0, sizeof(CameraUniform) };
    VkDescriptorBufferInfo triInfo{ secNode->triangleBuffer->getBuffer(), 0, secNode->triangleBuffer->getSize() };
    VkDescriptorBufferInfo sphereInfo{ secNode->sphereBuffer->getBuffer(), 0, secNode->sphereBuffer->getSize() };
    VkDescriptorBufferInfo matInfo{ secNode->materialBuffer->getBuffer(), 0, secNode->materialBuffer->getSize() };
    VkDescriptorBufferInfo lightInfo{ secNode->lightBuffer->getBuffer(), 0, secNode->lightBuffer->getSize() };

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    VkAccelerationStructureKHR tlasHandle = secNode->tlas ? secNode->tlas->getHandle() : VK_NULL_HANDLE;
    asInfo.pAccelerationStructures = &tlasHandle;

    VkDescriptorImageInfo envInfo = secNode->environmentMap ? secNode->environmentMap->getDescriptorInfo() : secNode->dummyWhite->getDescriptorInfo();

    std::vector<VkDescriptorImageInfo> texInfos(MAX_SCENE_TEXTURES);
    for (size_t i = 0; i < MAX_SCENE_TEXTURES; ++i) {
        if (i < secNode->sceneTextures.size() && secNode->sceneTextures[i]) {
            texInfos[i] = secNode->sceneTextures[i]->getDescriptorInfo();
        } else {
            texInfos[i] = secNode->dummyWhite->getDescriptorInfo();
        }
    }

    std::vector<VkWriteDescriptorSet> writes = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sphereInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, secNode->rtDescSet, 6, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSet, 8, 0, MAX_SCENE_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texInfos.data(), nullptr, nullptr }
    };
    vkUpdateDescriptorSets(secDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

    // 6. Pipeline Layout & Compute Pipeline on secondary device
    VkPushConstantRange pushConstant{};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(uint32_t) * 12;

    VkPipelineLayoutCreateInfo pipeLayoutInfo{};
    pipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts = &secNode->rtDescLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges = &pushConstant;
    vkCreatePipelineLayout(secDevice, &pipeLayoutInfo, nullptr, &secNode->rtPipelineLayout);

    auto rtCode = loadShaderSPIRV("raytrace_comp.comp.spv");
    VkShaderModule rtModule = createShaderModule(secDevice, rtCode);

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    VkComputePipelineCreateInfo compPipeInfo{};
    compPipeInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    compPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compPipeInfo.stage.module = rtModule;
    compPipeInfo.stage.pName = "main";
    if (secNode->context->hasSubgroupSizeControl()) {
        compPipeInfo.stage.pNext = &subgroupSize32;
    }
    compPipeInfo.layout = secNode->rtPipelineLayout;
    vkCreateComputePipelines(secDevice, VK_NULL_HANDLE, 1, &compPipeInfo, nullptr, &secNode->rtPipeline);
    vkDestroyShaderModule(secDevice, rtModule, nullptr);

    // Initialize Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline) on Secondary GPU
    if (secNode->context->hasRayTracing()) {
        try {
            VkPushConstantRange rtpPushConstant{};
            rtpPushConstant.stageFlags = VK_SHADER_STAGE_RAYGEN_BIT_KHR |
                                         VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR |
                                         VK_SHADER_STAGE_MISS_BIT_KHR;
            rtpPushConstant.offset = 0;
            rtpPushConstant.size = sizeof(uint32_t) * 12;

            VkPipelineLayoutCreateInfo rtpPipeLayoutInfo{};
            rtpPipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
            rtpPipeLayoutInfo.setLayoutCount = 1;
            rtpPipeLayoutInfo.pSetLayouts = &secNode->rtDescLayout;
            rtpPipeLayoutInfo.pushConstantRangeCount = 1;
            rtpPipeLayoutInfo.pPushConstantRanges = &rtpPushConstant;
            vkCreatePipelineLayout(secDevice, &rtpPipeLayoutInfo, nullptr, &secNode->rtpPipelineLayout);

            auto rgenCode = loadShaderSPIRV("raytrace.rgen.spv");
            auto rmissCode = loadShaderSPIRV("raytrace.rmiss.spv");
            auto shadowMissCode = loadShaderSPIRV("shadow.rmiss.spv");
            auto rchitCode = loadShaderSPIRV("raytrace.rchit.spv");

            secNode->rtpKhrPipeline = std::make_unique<RTPipeline>(
                secDevice, secAlloc,
                secNode->context->getRayTracingPipelineProperties(),
                secNode->rtpPipelineLayout,
                rgenCode, rmissCode, shadowMissCode, rchitCode
            );
            Logger::info("Secondary GPU: Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline) created successfully.");
        } catch (const std::exception& e) {
            Logger::warn("Failed to initialize secondary GPU RTPipeline: {}", e.what());
        }
    }

    // 7. Transition secondary accumTarget to GENERAL layout
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(secNode->commandBuffer, &beginInfo);

    secNode->accumTarget->transitionLayout(
        secNode->commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    vkEndCommandBuffer(secNode->commandBuffer);

    VkSubmitInfo initSubmit{};
    initSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    initSubmit.commandBufferCount = 1;
    initSubmit.pCommandBuffers = &secNode->commandBuffer;
    vkQueueSubmit(secNode->context->getGraphicsQueue(), 1, &initSubmit, VK_NULL_HANDLE);
    vkQueueWaitIdle(secNode->context->getGraphicsQueue());

    Logger::info("Secondary GPU Node fully initialized: {} (PCIe 5.0 x16)", secNode->deviceName);
    m_devices.push_back(std::move(secNode));
    m_active = true;

    // Initialize zero-copy shared external memory host buffer across primary and secondary GPUs
    uint32_t bytesPerPixel = (config.accum_format == AccumFormat::RGBA16_SFLOAT) ? (4 * sizeof(uint16_t)) : (4 * sizeof(float));
    VkDeviceSize bufferSize = static_cast<VkDeviceSize>(config.width) * config.height * bytesPerPixel;
    initSharedHostBuffer(bufferSize);

    if (!m_useZeroCopyHost) {
        m_devices[0]->p2pStagingBuffer = std::make_unique<Buffer>(
            m_devices[0]->context->getAllocator(), bufferSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
        );
    }
}

void MultiGpuManager::launchSecondaryWork(const CameraUniform& cameraUniform,
                                         uint32_t frameIndex,
                                         uint32_t tileOffsetX, uint32_t tileOffsetY,
                                         uint32_t tileWidth, uint32_t tileHeight,
                                         uint32_t numTriangles, uint32_t numSpheres,
                                         uint32_t numMaterials, uint32_t numLights,
                                         uint32_t useHardwareRT,
                                         uint32_t hasEnvMap,
                                         float envMapIntensity,
                                         uint32_t accumulateHistory,
                                         void* dstHostPtr,
                                         size_t transferBytes) {
    if (!m_active || m_devices.empty()) return;

    m_asyncTask = std::async(std::launch::async, [this, cameraUniform, frameIndex,
                                                 tileOffsetX, tileOffsetY, tileWidth, tileHeight,
                                                 numTriangles, numSpheres, numMaterials, numLights,
                                                 useHardwareRT, hasEnvMap, envMapIntensity, accumulateHistory,
                                                 dstHostPtr, transferBytes]() {
        GpuDeviceNode* node = m_devices[0].get();
        VkDevice device = node->context->getDevice();
        VkQueue queue = node->context->getGraphicsQueue();

        // 1. Update Camera UBO on secondary GPU
        node->cameraUBO->copyFrom(&cameraUniform, sizeof(CameraUniform));

        // 2. Wait for previous execution to complete
        vkWaitForFences(device, 1, &node->renderFence, VK_TRUE, UINT64_MAX);
        vkResetFences(device, 1, &node->renderFence);

        // 3. Record secondary GPU commands
        vkResetCommandBuffer(node->commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(node->commandBuffer, &beginInfo);

        vkCmdResetQueryPool(node->commandBuffer, node->queryPool, 0, 2);
        vkCmdWriteTimestamp2(node->commandBuffer, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, node->queryPool, 0);

        uint32_t envBits = std::bit_cast<uint32_t>(envMapIntensity);
        uint32_t rtPushConstants[12] = {
            numTriangles, numSpheres, numMaterials, numLights,
            tileOffsetX, tileOffsetY, tileWidth, tileHeight,
            useHardwareRT,
            hasEnvMap,
            envBits,
            accumulateHistory
        };

        uint32_t dispatchWidth = (tileOffsetX == 1u || tileOffsetX == 2u) ? ((tileWidth + 1) / 2) : tileWidth;

        if (node->rtpKhrPipeline && node->rtpKhrPipeline->isSupported() && m_config.pipeline_type != PipelineType::Megakernel) {
            // High-Performance Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline)
            vkCmdBindPipeline(node->commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, node->rtpKhrPipeline->getPipeline());
            vkCmdBindDescriptorSets(node->commandBuffer, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, node->rtpPipelineLayout, 0, 1, &node->rtDescSet, 0, nullptr);

            VkShaderStageFlags rtpStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
            vkCmdPushConstants(node->commandBuffer, node->rtpPipelineLayout, rtpStages, 0, sizeof(rtPushConstants), rtPushConstants);

            node->rtpKhrPipeline->traceRays(node->commandBuffer, dispatchWidth, tileHeight, 1);
        } else {
            // Fallback Compute Pipeline
            vkCmdBindPipeline(node->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, node->rtPipeline);
            vkCmdBindDescriptorSets(node->commandBuffer, VK_PIPELINE_BIND_POINT_COMPUTE, node->rtPipelineLayout, 0, 1, &node->rtDescSet, 0, nullptr);
            vkCmdPushConstants(node->commandBuffer, node->rtPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rtPushConstants), rtPushConstants);

            uint32_t groupsX = (dispatchWidth + 7) / 8;
            uint32_t groupsY = (tileHeight + 3) / 4;
            vkCmdDispatch(node->commandBuffer, groupsX, groupsY, 1);
        }

        vkCmdWriteTimestamp2(node->commandBuffer, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, node->queryPool, 1);

        // Transition accumTarget to TRANSFER_SRC_OPTIMAL to copy to host-visible staging buffer
        node->accumTarget->transitionLayout(
            node->commandBuffer, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
        );

        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.mipLevel = 0;
        copyRegion.imageSubresource.baseArrayLayer = 0;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageOffset = { 0, 0, 0 };
        copyRegion.imageExtent = { dispatchWidth, tileHeight, 1 };

        uint32_t slot = m_config.double_buffered_shared_mem ? (frameIndex % NUM_SHARED_BUFFERS) : 0;
        VkBuffer targetBuffer = m_useZeroCopyHost ? m_sharedBufferSecondary[slot] : node->p2pStagingBuffer->getBuffer();

        vkCmdCopyImageToBuffer(node->commandBuffer, node->accumTarget->getImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               targetBuffer, 1, &copyRegion);

        node->accumTarget->transitionLayout(
            node->commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        vkEndCommandBuffer(node->commandBuffer);

        // 4. Submit secondary workload
        VkSubmitInfo submitInfo{};
        submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &node->commandBuffer;
        vkQueueSubmit(queue, 1, &submitInfo, node->renderFence);

        // 5. Wait for secondary fence and retrieve timestamps
        vkWaitForFences(device, 1, &node->renderFence, VK_TRUE, UINT64_MAX);

        uint64_t timestamps[2] = {0, 0};
        vkGetQueryPoolResults(device, node->queryPool, 0, 2, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        node->lastFrameTimeMs = (timestamps[1] - timestamps[0]) * node->timestampPeriod * 1e-6;

        // 6. Asynchronous PCIe transfer into host-mapped destination buffer (only for fallback when zero-copy disabled)
        if (!m_useZeroCopyHost && dstHostPtr != nullptr && transferBytes > 0) {
            void* srcPtr = node->p2pStagingBuffer->map();
            parallelMemcpy(dstHostPtr, srcPtr, transferBytes, 8);
        }
    });
}

void MultiGpuManager::syncAndTransfer(void* dstHostPtr, size_t byteSize) {
    if (!m_active || m_devices.empty()) return;

    if (m_asyncTask.valid()) {
        m_asyncTask.get();
    }

    if (!m_useZeroCopyHost && dstHostPtr != nullptr && byteSize > 0) {
        GpuDeviceNode* node = m_devices[0].get();
        void* srcPtr = node->p2pStagingBuffer->map();
        parallelMemcpy(dstHostPtr, srcPtr, byteSize, 8);
    }
}

void MultiGpuManager::resize(uint32_t width, uint32_t height) {
    if (!m_active || m_devices.empty()) return;

    if (m_asyncTask.valid()) {
        m_asyncTask.wait();
    }

    m_config.width = width;
    m_config.height = height;

    for (auto& node : m_devices) {
        VkDevice secDevice = node->context->getDevice();
        VmaAllocator secAlloc = node->context->getAllocator();
        vkDeviceWaitIdle(secDevice);

        uint32_t bytesPerPixel = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? (4 * sizeof(uint16_t)) : (4 * sizeof(float));
        VkFormat accumFormat = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
        uint32_t secWidth = (m_config.mgpu_mode == MultiGpuMode::CheckerboardTile) ? ((width + 1) / 2) : width;
        node->accumTarget = std::make_unique<Image>(
            secDevice, secAlloc, secWidth, height,
            accumFormat,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );

        // Transition accumTarget to GENERAL layout
        vkResetCommandBuffer(node->commandBuffer, 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(node->commandBuffer, &beginInfo);

        node->accumTarget->transitionLayout(
            node->commandBuffer, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        vkEndCommandBuffer(node->commandBuffer);

        VkSubmitInfo initSubmit{};
        initSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        initSubmit.commandBufferCount = 1;
        initSubmit.pCommandBuffers = &node->commandBuffer;
        vkQueueSubmit(node->context->getGraphicsQueue(), 1, &initSubmit, VK_NULL_HANDLE);
        vkQueueWaitIdle(node->context->getGraphicsQueue());

        VkDeviceSize bufferSize = static_cast<VkDeviceSize>(width) * height * bytesPerPixel;
        initSharedHostBuffer(bufferSize);

        if (!m_useZeroCopyHost) {
            node->p2pStagingBuffer = std::make_unique<Buffer>(
                secAlloc, bufferSize,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
            );
        }

        VkDescriptorImageInfo accumImageInfo{};
        accumImageInfo.imageView = node->accumTarget->getImageView();
        accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        VkWriteDescriptorSet write{};
        write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        write.dstSet = node->rtDescSet;
        write.dstBinding = 0;
        write.descriptorCount = 1;
        write.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        write.pImageInfo = &accumImageInfo;

        vkUpdateDescriptorSets(secDevice, 1, &write, 0, nullptr);
    }

    Logger::info("MultiGpuManager resized secondary GPU targets to {}x{}", width, height);
}

} // namespace pathways
