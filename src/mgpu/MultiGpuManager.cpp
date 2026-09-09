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
    for (uint32_t i = 0; i < NUM_IN_FLIGHT; ++i) {
        if (queryPools[i]) vkDestroyQueryPool(device, queryPools[i], nullptr);
        if (renderFences[i]) vkDestroyFence(device, renderFences[i], nullptr);
        if (secSemaphores[i]) vkDestroySemaphore(device, secSemaphores[i], nullptr);
        cameraUBOs[i].reset();
    }
    if (rtDescLayout) vkDestroyDescriptorSetLayout(device, rtDescLayout, nullptr);
    if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);

    accumTarget.reset();
    p2pStagingBuffer.reset();
    triangleBuffer.reset();
    sphereBuffer.reset();
    materialBuffer.reset();
    lightBuffer.reset();
}

MultiGpuManager::MultiGpuManager(const Config& config, VulkanContext* primaryContext, const SceneData& scene)
    : m_primaryContext(primaryContext), m_mode(config.mgpu_mode), m_config(config) {

    auto devices = VulkanContext::enumeratePhysicalDevices(primaryContext->getInstance());
    if (devices.size() < 2) {
        Logger::info("Multi-GPU: only {} physical Vulkan device(s) found. Multi-GPU unavailable.", devices.size());
        m_mode = MultiGpuMode::Off;
        return;
    }

    Logger::info("Initializing Multi-GPU Manager across {} discrete GPUs (Initial State: {})...",
                 devices.size(), m_mode == MultiGpuMode::Off ? "Standby (Single-GPU)" : "Active");
    initSecondaryDevice(config, scene);

    if (m_active) {
        m_workerThread = std::thread(&MultiGpuManager::workerLoop, this);
    }
}

MultiGpuManager::~MultiGpuManager() {
    {
        std::lock_guard<std::mutex> lock(m_workMutex);
        m_stopWorker = true;
        m_workCv.notify_all();
    }
    if (m_workerThread.joinable()) {
        m_workerThread.join();
    }
    if (!m_devices.empty() && m_devices[0]->context) {
        vkDeviceWaitIdle(m_devices[0]->context->getDevice());
    }
    if (m_primaryContext) {
        vkDeviceWaitIdle(m_primaryContext->getDevice());
        if (!m_devices.empty()) {
            VkDevice primDevice = m_primaryContext->getDevice();
            for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
                if (m_devices[0]->primImportedSemaphores[i] != VK_NULL_HANDLE) {
                    vkDestroySemaphore(primDevice, m_devices[0]->primImportedSemaphores[i], nullptr);
                    m_devices[0]->primImportedSemaphores[i] = VK_NULL_HANDLE;
                }
            }
        }
    }
    destroySharedHostBuffer();
    m_devices.clear();
}

double MultiGpuManager::getSecondaryGpuTimeMs() const {
    if (!isMultiGpuActive() || m_devices.empty()) return 0.0;
    return m_devices[0]->lastFrameTimeMs;
}

double MultiGpuManager::getSecondaryTransferTimeMs() const {
    if (!isMultiGpuActive() || m_devices.empty()) return 0.0;
    return m_devices[0]->lastTransferTimeMs;
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

    // Align buffer size to 64KB (satisfies minImportedHostPointerAlignment of 4096 or 65536)
    VkDeviceSize hostAlignment = 65536;
    m_sharedBufferSize = (bufferSize + hostAlignment - 1) & ~(hostAlignment - 1);
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
        if (posix_memalign(&m_sharedHostPtr[slot], hostAlignment, m_sharedBufferSize) != 0 || !m_sharedHostPtr[slot]) {
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
        Logger::info("{} Zero-Copy Inter-GPU Host Buffers initialized via VK_EXT_external_memory_host (2x {:.2f} MB).",
                     m_config.double_buffered_shared_mem ? "Double-Buffered" : "Single-Buffered (Double-Buffering Disabled)",
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
    cmdAllocInfo.commandBufferCount = GpuDeviceNode::NUM_IN_FLIGHT;
    vkAllocateCommandBuffers(secDevice, &cmdAllocInfo, secNode->commandBuffers.data());

    // 2. Fence & Query Pool (Double-buffered, 4 timestamps per pool)
    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
        vkCreateFence(secDevice, &fenceInfo, nullptr, &secNode->renderFences[i]);
    }

    VkQueryPoolCreateInfo queryInfo{};
    queryInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryInfo.queryCount = 4; // 0: Start RT, 1: End RT, 2: Start Copy, 3: End Copy
    for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
        vkCreateQueryPool(secDevice, &queryInfo, nullptr, &secNode->queryPools[i]);
    }

    // 2b. Cross-GPU Hardware Synchronization Semaphores (VK_KHR_external_semaphore_fd)
    VkDevice primDevice = m_primaryContext->getDevice();
    m_useCrossGpuSync = m_primaryContext->hasExternalSemaphoreFd() && secNode->context->hasExternalSemaphoreFd() &&
                        m_primaryContext->pfnImportSemaphoreFdKHR && secNode->context->pfnGetSemaphoreFdKHR;
    if (m_useCrossGpuSync) {
        for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
            VkExportSemaphoreCreateInfo exportInfo{ VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
            exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

            VkSemaphoreCreateInfo secSemInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            secSemInfo.pNext = &exportInfo;
            vkCreateSemaphore(secDevice, &secSemInfo, nullptr, &secNode->secSemaphores[i]);

            VkSemaphoreCreateInfo primSemInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
            vkCreateSemaphore(primDevice, &primSemInfo, nullptr, &secNode->primImportedSemaphores[i]);
            secNode->exportedFd[i] = -1;
            secNode->slotFdReady[i] = false;
        }
        Logger::info("Cross-GPU Hardware Synchronization active via VK_KHR_external_semaphore_fd (zero-wait GPU-to-GPU pipelining).");
    }

    // 3. Render Targets on secondary device (full-width to allow seamless dynamic switching between Checkerboard and SampleParallel)
    uint32_t secWidth = config.width;
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
    for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
        secNode->cameraUBOs[i] = std::make_unique<Buffer>(
            secAlloc, uboSize,
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        );
    }

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
    vkCreateDescriptorSetLayout(secDevice, &rtLayoutInfo, nullptr, &secNode->rtDescLayout);

    std::vector<VkDescriptorSetLayout> descLayouts(GpuDeviceNode::NUM_IN_FLIGHT, secNode->rtDescLayout);
    VkDescriptorSetAllocateInfo descAllocInfo{};
    descAllocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    descAllocInfo.descriptorPool = secNode->descriptorPool;
    descAllocInfo.descriptorSetCount = GpuDeviceNode::NUM_IN_FLIGHT;
    descAllocInfo.pSetLayouts = descLayouts.data();
    vkAllocateDescriptorSets(secDevice, &descAllocInfo, secNode->rtDescSets.data());

    // Allocate secondary ReSTIR reservoir ping-pong buffers
    VkDeviceSize resSize = static_cast<VkDeviceSize>(config.width) * config.height * sizeof(ReservoirGPU);
    for (uint32_t i = 0; i < 2; ++i) {
        secNode->restirReservoirs[i] = std::make_unique<Buffer>(
            secAlloc, resSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );
    }
    secNode->restirPingPongIndex = 0;

    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageView = secNode->accumTarget->getImageView();
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorBufferInfo triInfo{ secNode->triangleBuffer->getBuffer(), 0, secNode->triangleBuffer->getSize() };
    VkDescriptorBufferInfo sphereInfo{ secNode->sphereBuffer->getBuffer(), 0, secNode->sphereBuffer->getSize() };
    VkDescriptorBufferInfo matInfo{ secNode->materialBuffer->getBuffer(), 0, secNode->materialBuffer->getSize() };
    VkDescriptorBufferInfo lightInfo{ secNode->lightBuffer->getBuffer(), 0, secNode->lightBuffer->getSize() };
    VkDescriptorBufferInfo res0Info{ secNode->restirReservoirs[0]->getBuffer(), 0, resSize };
    VkDescriptorBufferInfo res1Info{ secNode->restirReservoirs[1]->getBuffer(), 0, resSize };

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

    for (uint32_t slot = 0; slot < GpuDeviceNode::NUM_IN_FLIGHT; ++slot) {
        VkDescriptorBufferInfo uboInfo{ secNode->cameraUBOs[slot]->getBuffer(), 0, sizeof(CameraUniform) };
        std::vector<VkWriteDescriptorSet> writes = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sphereInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, secNode->rtDescSets[slot], 6, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 8, 0, MAX_SCENE_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texInfos.data(), nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &res0Info, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &res1Info, nullptr }
        };
        vkUpdateDescriptorSets(secDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    // 6. Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline) on Secondary GPU
    VkPushConstantRange rtpPushConstant{};
    rtpPushConstant.stageFlags = rtStages;
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
    Logger::info("Secondary GPU: Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline) initialized.");

    // 7. Transition secondary accumTarget to GENERAL layout
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(secNode->commandBuffers[0], &beginInfo);

    secNode->accumTarget->transitionLayout(
        secNode->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    vkCmdFillBuffer(secNode->commandBuffers[0], secNode->restirReservoirs[0]->getBuffer(), 0, resSize, 0);
    vkCmdFillBuffer(secNode->commandBuffers[0], secNode->restirReservoirs[1]->getBuffer(), 0, resSize, 0);

    vkEndCommandBuffer(secNode->commandBuffers[0]);

    VkSubmitInfo initSubmit{};
    initSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    initSubmit.commandBufferCount = 1;
    initSubmit.pCommandBuffers = &secNode->commandBuffers[0];
    vkQueueSubmit(secNode->context->getGraphicsQueue(), 1, &initSubmit, VK_NULL_HANDLE);
    vkQueueWaitIdle(secNode->context->getGraphicsQueue());

    Logger::info("Secondary GPU Node fully initialized: {} ({})", secNode->deviceName, secNode->context->getPciLinkString());
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

void MultiGpuManager::workerLoop() {
    while (true) {
        SecondaryWorkPacket packet;
        {
            std::unique_lock<std::mutex> lock(m_workMutex);
            m_workCv.wait(lock, [this]() {
                return m_stopWorker || m_pendingWork.valid;
            });
            if (m_stopWorker) break;
            packet = m_pendingWork;
            m_pendingWork.valid = false;
            m_submitCv.notify_all();
        }

        executeSecondaryWork(packet);

        {
            std::lock_guard<std::mutex> lock(m_workMutex);
            uint32_t slot = packet.bufferSlot % GpuDeviceNode::NUM_IN_FLIGHT;
            m_workSubmitted = true;
            m_slotSubmitted[slot] = true;
            if (m_useCrossGpuSync && !m_devices.empty()) {
                m_devices[0]->slotFdReady[slot] = true;
            }
            m_submitCv.notify_all();
        }
    }
}

void MultiGpuManager::executeSecondaryWork(const SecondaryWorkPacket& packet) {
    if (m_devices.empty()) return;
    GpuDeviceNode* node = m_devices[0].get();
    VkDevice device = node->context->getDevice();
    VkQueue queue = node->context->getGraphicsQueue();
    uint32_t slot = packet.bufferSlot % GpuDeviceNode::NUM_IN_FLIGHT;

    // 1. Wait for previous execution using this slot to complete before recording
    vkWaitForFences(device, 1, &node->renderFences[slot], VK_TRUE, UINT64_MAX);

    // Read timestamp queries from completed execution of this slot (only if slot was previously executed)
    if (node->slotHasExecuted[slot]) {
        uint64_t slotTimestamps[4] = {0, 0, 0, 0};
        if (vkGetQueryPoolResults(device, node->queryPools[slot], 0, 4, sizeof(slotTimestamps), slotTimestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            if (slotTimestamps[1] > slotTimestamps[0]) {
                node->lastFrameTimeMs = (slotTimestamps[1] - slotTimestamps[0]) * node->timestampPeriod * 1e-6;
            }
            if (slotTimestamps[3] > slotTimestamps[2]) {
                node->lastTransferTimeMs = (slotTimestamps[3] - slotTimestamps[2]) * node->timestampPeriod * 1e-6;
            }
        }
    }
    vkResetFences(device, 1, &node->renderFences[slot]);

    // 2. Update Camera UBO for this slot
    node->cameraUBOs[slot]->copyFrom(&packet.cameraUniform, sizeof(CameraUniform));

    // 3. Ping-pong ReSTIR DI reservoir buffers if enabled
    if ((packet.cameraUniform.flags & (1 << 6)) && node->restirReservoirs[0] && node->restirReservoirs[1]) {
        node->restirPingPongIndex = 1 - node->restirPingPongIndex;
        VkDeviceSize resSize = static_cast<VkDeviceSize>(packet.tileWidth) * packet.tileHeight * sizeof(ReservoirGPU);
        VkDescriptorBufferInfo curInfo{ node->restirReservoirs[node->restirPingPongIndex]->getBuffer(), 0, resSize };
        VkDescriptorBufferInfo histInfo{ node->restirReservoirs[1 - node->restirPingPongIndex]->getBuffer(), 0, resSize };
        VkWriteDescriptorSet resWrites[2] = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &curInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &histInfo, nullptr }
        };
        vkUpdateDescriptorSets(device, 2, resWrites, 0, nullptr);
    }

    // 4. Record secondary GPU commands
    VkCommandBuffer cmd = node->commandBuffers[slot];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &beginInfo);

    vkCmdResetQueryPool(cmd, node->queryPools[slot], 0, 4);
    // Timestamp 0: RT Start
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, node->queryPools[slot], 0);

    uint32_t envBits = std::bit_cast<uint32_t>(packet.envMapIntensity);
    uint32_t rtPushConstants[12] = {
        packet.numTriangles, packet.numSpheres, packet.numMaterials, packet.numLights,
        packet.tileOffsetX, packet.tileOffsetY, packet.tileWidth, packet.tileHeight,
        packet.useHardwareRT,
        packet.hasEnvMap,
        envBits,
        packet.accumulateHistory
    };

    uint32_t dispatchWidth = packet.tileWidth;
    uint32_t dispatchHeight = packet.tileHeight;

    if (packet.tileOffsetY == 0u) {
        if (packet.tileOffsetX == 1u || packet.tileOffsetX == 2u) {
            dispatchWidth = packet.tileWidth;
            dispatchHeight = (packet.tileHeight + 1) / 2;
        }
    } else {
        if (packet.tileOffsetX == 1u || packet.tileOffsetX == 2u) {
            dispatchWidth = (packet.tileWidth + 1) / 2;
            dispatchHeight = packet.tileHeight;
        }
    }

    // Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline)
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, node->rtpKhrPipeline->getPipeline());
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, node->rtpPipelineLayout, 0, 1, &node->rtDescSets[slot], 0, nullptr);
    VkShaderStageFlags rtpStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
    vkCmdPushConstants(cmd, node->rtpPipelineLayout, rtpStages, 0, sizeof(rtPushConstants), rtPushConstants);

    node->rtpKhrPipeline->traceRays(cmd, dispatchWidth, dispatchHeight, 1);

    // Timestamp 1: RT End
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, node->queryPools[slot], 1);

    // Transition accumTarget to TRANSFER_SRC_OPTIMAL
    node->accumTarget->transitionLayout(
        cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
    );

    // Timestamp 2: Copy Start
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, node->queryPools[slot], 2);

    VkBufferImageCopy copyRegion{};
    copyRegion.bufferOffset = 0;
    copyRegion.bufferRowLength = 0;
    copyRegion.bufferImageHeight = 0;
    copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    copyRegion.imageSubresource.mipLevel = 0;
    copyRegion.imageSubresource.baseArrayLayer = 0;
    copyRegion.imageSubresource.layerCount = 1;
    copyRegion.imageOffset = { 0, 0, 0 };
    copyRegion.imageExtent = { dispatchWidth, dispatchHeight, 1 };

    uint32_t sharedSlot = m_config.double_buffered_shared_mem ? slot : 0;
    VkBuffer targetBuffer = m_useZeroCopyHost ? m_sharedBufferSecondary[sharedSlot] : node->p2pStagingBuffer->getBuffer();

    vkCmdCopyImageToBuffer(cmd, node->accumTarget->getImage(),
                           VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           targetBuffer, 1, &copyRegion);

    // Timestamp 3: Copy End
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, node->queryPools[slot], 3);

    node->accumTarget->transitionLayout(
        cmd, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    vkEndCommandBuffer(cmd);

    // 5. Submit secondary workload
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;
    if (m_useCrossGpuSync) {
        submitInfo.signalSemaphoreCount = 1;
        submitInfo.pSignalSemaphores = &node->secSemaphores[slot];
    }
    vkQueueSubmit(queue, 1, &submitInfo, node->renderFences[slot]);

    // Export semaphore FD from secondary device
    if (m_useCrossGpuSync) {
        VkDevice secDev = node->context->getDevice();
        VkSemaphoreGetFdInfoKHR getFdInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR };
        getFdInfo.semaphore = node->secSemaphores[slot];
        getFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        VkResult res = node->context->pfnGetSemaphoreFdKHR(secDev, &getFdInfo, &fd);
        if (res == VK_SUCCESS && fd >= 0) {
            node->exportedFd[slot] = fd;
        }
    }
    node->slotHasExecuted[slot] = true;
}

void MultiGpuManager::launchSecondaryWork(const CameraUniform& cameraUniform,
                                         uint32_t bufferSlot,
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

    {
        std::unique_lock<std::mutex> lock(m_workMutex);
        m_submitCv.wait(lock, [this]() { return !m_pendingWork.valid; });

        m_pendingWork.cameraUniform = cameraUniform;
        m_pendingWork.bufferSlot = bufferSlot;
        m_pendingWork.tileOffsetX = tileOffsetX;
        m_pendingWork.tileOffsetY = tileOffsetY;
        m_pendingWork.tileWidth = tileWidth;
        m_pendingWork.tileHeight = tileHeight;
        m_pendingWork.numTriangles = numTriangles;
        m_pendingWork.numSpheres = numSpheres;
        m_pendingWork.numMaterials = numMaterials;
        m_pendingWork.numLights = numLights;
        m_pendingWork.useHardwareRT = useHardwareRT;
        m_pendingWork.hasEnvMap = hasEnvMap;
        m_pendingWork.envMapIntensity = envMapIntensity;
        m_pendingWork.accumulateHistory = accumulateHistory;
        m_pendingWork.dstHostPtr = dstHostPtr;
        m_pendingWork.transferBytes = transferBytes;
        m_pendingWork.valid = true;
        uint32_t slot = bufferSlot % GpuDeviceNode::NUM_IN_FLIGHT;
        m_waitingSlot = slot;
        m_workSubmitted = false;
        m_slotSubmitted[slot] = false;
        if (m_useCrossGpuSync && !m_devices.empty()) {
            m_devices[0]->slotFdReady[slot] = false;
            m_devices[0]->exportedFd[slot] = -1;
        }

        m_workCv.notify_one();
    }
}

void MultiGpuManager::syncAndTransfer(uint32_t slot, void* dstHostPtr, size_t byteSize) {
    if (!m_active || m_devices.empty()) return;

    uint32_t s = slot % GpuDeviceNode::NUM_IN_FLIGHT;

    if (m_useCrossGpuSync) {
        // Wait until worker thread has exported the semaphore FD for this slot
        {
            std::unique_lock<std::mutex> lock(m_workMutex);
            m_submitCv.wait(lock, [this, s]() { return m_devices[0]->slotFdReady[s]; });
        }
        // Import FD into primary device semaphore on main thread (safe because frame N-2 fence signaled)
        GpuDeviceNode* node = m_devices[0].get();
        int fd = node->exportedFd[s];
        if (fd >= 0) {
            VkDevice primDev = m_primaryContext->getDevice();
            VkImportSemaphoreFdInfoKHR importInfo{ VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR };
            importInfo.semaphore = node->primImportedSemaphores[s];
            importInfo.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
            importInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
            importInfo.fd = fd;
            m_primaryContext->pfnImportSemaphoreFdKHR(primDev, &importInfo);
            node->exportedFd[s] = -1;
        }
        return;
    }

    // 1. Wait until the worker thread has submitted the command buffer for this slot
    {
        std::unique_lock<std::mutex> lock(m_workMutex);
        m_submitCv.wait(lock, [this, s]() { return m_slotSubmitted[s]; });
    }

    GpuDeviceNode* node = m_devices[0].get();
    VkDevice device = node->context->getDevice();

    // 2. Wait for secondary GPU completion of this slot
    vkWaitForFences(device, 1, &node->renderFences[s], VK_TRUE, UINT64_MAX);

    // 3. Read timestamp queries (0: Start RT, 1: End RT, 2: Start Copy, 3: End Copy)
    uint64_t timestamps[4] = {0, 0, 0, 0};
    vkGetQueryPoolResults(device, node->queryPools[s], 0, 4, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
    if (timestamps[1] > timestamps[0]) {
        node->lastFrameTimeMs = (timestamps[1] - timestamps[0]) * node->timestampPeriod * 1e-6;
    }
    if (timestamps[3] > timestamps[2]) {
        node->lastTransferTimeMs = (timestamps[3] - timestamps[2]) * node->timestampPeriod * 1e-6;
    }

    // 4. Asynchronous PCIe transfer into host-mapped destination buffer (fallback only when zero-copy disabled)
    if (!m_useZeroCopyHost && dstHostPtr != nullptr && byteSize > 0) {
        void* srcPtr = node->p2pStagingBuffer->map();
        parallelMemcpy(dstHostPtr, srcPtr, byteSize, 8);
    }
}

void MultiGpuManager::resize(uint32_t width, uint32_t height) {
    if (!m_active || m_devices.empty()) return;

    {
        std::unique_lock<std::mutex> lock(m_workMutex);
        m_submitCv.wait(lock, [this]() { return !m_pendingWork.valid; });
    }

    m_config.width = width;
    m_config.height = height;

    for (auto& node : m_devices) {
        VkDevice secDevice = node->context->getDevice();
        VmaAllocator secAlloc = node->context->getAllocator();
        vkDeviceWaitIdle(secDevice);

        uint32_t bytesPerPixel = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? (4 * sizeof(uint16_t)) : (4 * sizeof(float));
        VkFormat accumFormat = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT : VK_FORMAT_R32G32B32A32_SFLOAT;
        uint32_t secWidth = width;
        node->accumTarget = std::make_unique<Image>(
            secDevice, secAlloc, secWidth, height,
            accumFormat,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );

        // Transition accumTarget to GENERAL layout
        vkResetCommandBuffer(node->commandBuffers[0], 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(node->commandBuffers[0], &beginInfo);

        VkDeviceSize resSize = static_cast<VkDeviceSize>(width) * height * sizeof(ReservoirGPU);
        for (uint32_t i = 0; i < 2; ++i) {
            node->restirReservoirs[i] = std::make_unique<Buffer>(
                secAlloc, resSize,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
            );
        }
        node->restirPingPongIndex = 0;

        node->accumTarget->transitionLayout(
            node->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        vkCmdFillBuffer(node->commandBuffers[0], node->restirReservoirs[0]->getBuffer(), 0, resSize, 0);
        vkCmdFillBuffer(node->commandBuffers[0], node->restirReservoirs[1]->getBuffer(), 0, resSize, 0);

        vkEndCommandBuffer(node->commandBuffers[0]);

        VkSubmitInfo initSubmit{};
        initSubmit.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
        initSubmit.commandBufferCount = 1;
        initSubmit.pCommandBuffers = &node->commandBuffers[0];
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

        VkDescriptorBufferInfo res0Info{ node->restirReservoirs[0]->getBuffer(), 0, resSize };
        VkDescriptorBufferInfo res1Info{ node->restirReservoirs[1]->getBuffer(), 0, resSize };

        for (uint32_t slot = 0; slot < GpuDeviceNode::NUM_IN_FLIGHT; ++slot) {
            std::vector<VkWriteDescriptorSet> writes = {
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &res0Info, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &res1Info, nullptr }
            };
            vkUpdateDescriptorSets(secDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }

    Logger::info("MultiGpuManager resized secondary GPU targets to {}x{}", width, height);
}

bool MultiGpuManager::loadScene(const SceneData& scene) {
    if (m_devices.empty()) return false;
    auto& secNode = m_devices[0];
    if (!secNode || !secNode->context) return false;

    VkDevice secDevice = secNode->context->getDevice();
    VmaAllocator secAlloc = secNode->context->getAllocator();
    VkQueue secQueue = secNode->context->getGraphicsQueue();
    VkCommandPool secPool = secNode->commandPool;

    vkQueueWaitIdle(secQueue);

    // 1. Update scene buffers on secondary device
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

    // 2. Rebuild secondary AS
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
    }

    // 3. Upload scene textures on secondary device
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

    // 4. Update secondary descriptors
    VkDescriptorImageInfo accumImageInfo{};
    accumImageInfo.imageView = secNode->accumTarget->getImageView();
    accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

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

    VkDescriptorBufferInfo res0Info{};
    VkDescriptorBufferInfo res1Info{};
    if (secNode->restirReservoirs[0] && secNode->restirReservoirs[1]) {
        res0Info = { secNode->restirReservoirs[0]->getBuffer(), 0, secNode->restirReservoirs[0]->getSize() };
        res1Info = { secNode->restirReservoirs[1]->getBuffer(), 0, secNode->restirReservoirs[1]->getSize() };
    }

    for (uint32_t slot = 0; slot < GpuDeviceNode::NUM_IN_FLIGHT; ++slot) {
        VkDescriptorBufferInfo uboInfo{ secNode->cameraUBOs[slot]->getBuffer(), 0, sizeof(CameraUniform) };
        std::vector<VkWriteDescriptorSet> writes = {
            VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr },
            VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &uboInfo, nullptr },
            VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triInfo, nullptr },
            VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sphereInfo, nullptr },
            VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matInfo, nullptr },
            VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightInfo, nullptr },
            VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, secNode->rtDescSets[slot], 6, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr },
            VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envInfo, nullptr, nullptr },
            VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 8, 0, MAX_SCENE_TEXTURES, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, texInfos.data(), nullptr, nullptr }
        };
        if (secNode->restirReservoirs[0] && secNode->restirReservoirs[1]) {
            writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &res0Info, nullptr });
            writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &res1Info, nullptr });
        }
        vkUpdateDescriptorSets(secDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
    Logger::info("Secondary GPU Node reloaded scene successfully ({} triangles, {} materials).", scene.triangles.size(), scene.materials.size());
    return true;
}

} // namespace pathways

