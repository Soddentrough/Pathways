#include "mgpu/MultiGpuManager.hpp"
#include "core/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <filesystem>
#include <cstring>
#include <bit>
#include <thread>
#include <vector>
#include <array>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
    #include <malloc.h>
    #include <io.h>
#else
    #include <unistd.h>
    #include <sys/mman.h>
#endif

namespace pathways {

static inline void closeFileDescriptor(int fd) {
    if (fd >= 0) {
#if defined(_WIN32)
        _close(fd);
#else
        ::close(fd);
#endif
    }
}



static void uploadToDeviceBufferSec(GpuDeviceNode& secNode, Buffer& dstBuffer, const void* srcData, VkDeviceSize dataSize) {
    if (dataSize == 0 || !srcData) return;
    VkDevice device = secNode.context->getDevice();
    VkQueue queue = secNode.context->getGraphicsQueue();
    VmaAllocator allocator = secNode.context->getAllocator();

    const VkDeviceSize maxChunkSize = 64 * 1024 * 1024; // 64 MB bounded staging buffer
    VkDeviceSize stagingSize = std::min(dataSize, maxChunkSize);

    Buffer stagingBuffer(
        allocator, stagingSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = secNode.commandPool;
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

    vkFreeCommandBuffers(device, secNode.commandPool, 1, &cmd);
}

static void createSecondaryAccelerationStructures(GpuDeviceNode& secNode, const SceneData& scene, const std::vector<char>& updateTlasSpv) {
    if (!secNode.context->hasRayTracing()) return;

    VkDevice secDevice = secNode.context->getDevice();
    VmaAllocator secAlloc = secNode.context->getAllocator();

    secNode.tlasUpdatePipeline.reset();
    secNode.tlas.reset();
    secNode.blas.reset();
    secNode.blases.clear();
    secNode.instanceBuffer.reset();
    secNode.asManager.reset();

    uint32_t numTriangles = static_cast<uint32_t>(scene.triangles.size());

    secNode.asManager = std::make_unique<AccelerationStructureManager>(
        secDevice, secAlloc,
        secNode.context->getGraphicsQueue(), secNode.context->getGraphicsQueueFamily()
    );

    // 1. Instance buffer (std430, binding 30)
    std::vector<InstanceGPU> instanceUpload;
    struct MonoBucket {
        uint32_t firstTri = 0;
        uint32_t triCount = 0;
        uint32_t numOpaqueField = 0;
        bool isGeomOpaque = true;
        uint8_t mask = 0x01;
    };

    auto buildMonolithicBuckets = [&]() -> std::vector<MonoBucket> {
        std::vector<MonoBucket> buckets;
        const uint32_t numNonOpaque = (numTriangles > scene.numOpaqueTriangles) ? (numTriangles - scene.numOpaqueTriangles) : 0;
        uint32_t numSolidNonOpaque = 0;
        if (numNonOpaque > 0) {
            if (scene.numSolidNonOpaqueTriangles > 0) {
                numSolidNonOpaque = std::min(scene.numSolidNonOpaqueTriangles, numNonOpaque);
            } else if (!scene.triangles.empty()) {
                const uint32_t endTri = std::min<uint32_t>(numTriangles, static_cast<uint32_t>(scene.triangles.size()));
                for (uint32_t t = scene.numOpaqueTriangles; t < endTri; ++t) {
                    const uint32_t matId = scene.triangles[t].materialId;
                    if (matId < scene.materials.size() &&
                        scene.materials[matId].alphaMode != ALPHA_MODE_OPAQUE) {
                        break;
                    }
                    ++numSolidNonOpaque;
                }
            }
        }
        const uint32_t numAlphaNonOpaque = numNonOpaque - numSolidNonOpaque;

        if (numTriangles > 0 && scene.numOpaqueTriangles > 0) {
            buckets.push_back({ 0, scene.numOpaqueTriangles, scene.numOpaqueTriangles, true, 0x01 });
        }
        if (numSolidNonOpaque > 0) {
            buckets.push_back({ scene.numOpaqueTriangles, numSolidNonOpaque, 0, true, 0x02 });
        }
        if (numAlphaNonOpaque > 0) {
            buckets.push_back({ scene.numOpaqueTriangles + numSolidNonOpaque, numAlphaNonOpaque, 0, false, 0x02 });
        }
        if (buckets.empty()) {
            buckets.push_back({ 0, 0, 0, true, 0x02 });
        }
        return buckets;
    };

    const std::vector<MonoBucket> monoBuckets =
        (scene.instanceData.empty() || scene.blasRanges.empty()) ? buildMonolithicBuckets() : std::vector<MonoBucket>{};

    if (!scene.instanceData.empty()) {
        instanceUpload = scene.instanceData;
        for (size_t i = 0; i < instanceUpload.size(); ++i) {
            if (i < scene.instances.size()) {
                instanceUpload[i].transform = scene.instances[i].transform;
                instanceUpload[i].invTransform = scene.instances[i].transform;
            }
        }
    } else {
        instanceUpload.reserve(monoBuckets.size());
        for (const auto& b : monoBuckets) {
            InstanceGPU inst{};
            inst.firstTriangle = b.firstTri;
            inst.numOpaqueTriangles = b.numOpaqueField;
            inst.materialOffset = 0;
            inst.flags = 0;
            instanceUpload.push_back(inst);
        }
    }

    VkDeviceSize instanceBufferSize = sizeof(InstanceGPU) * instanceUpload.size();
    secNode.instanceBuffer = std::make_unique<Buffer>(
        secAlloc, instanceBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    uploadToDeviceBufferSec(secNode, *secNode.instanceBuffer, instanceUpload.data(), instanceBufferSize);

    // 2. Acceleration Structures (Indexless VK_INDEX_TYPE_NONE_KHR + VK_KHR_ray_tracing_position_fetch)
    VkDeviceAddress vertexBaseAddr = secNode.positionBuffer ? secNode.positionBuffer->getDeviceAddress(secDevice) : 0;
    constexpr VkDeviceSize kTriVertexBytes = 3 * sizeof(glm::vec4);

    auto subrangeHasAlphaCutout = [&scene](uint32_t firstTri, uint32_t count, uint32_t matOffset) -> bool {
        const uint32_t endTri = std::min<uint32_t>(firstTri + count, static_cast<uint32_t>(scene.triangles.size()));
        for (uint32_t t = firstTri; t < endTri; ++t) {
            const uint32_t matId = scene.triangles[t].materialId + matOffset;
            if (matId < scene.materials.size() &&
                scene.materials[matId].alphaMode != ALPHA_MODE_OPAQUE) {
                return true;
            }
        }
        return false;
    };

    std::vector<ASInstanceInput> asInstances;
    if (!scene.blasRanges.empty()) {
        // Multi-BLAS path: Batch all BLAS builds into a single GPU dispatch & compaction pass
        secNode.asManager->resetStats();
        std::vector<bool> blasHasAlphaCutout(scene.blasRanges.size(), false);
        for (size_t b = 0; b < scene.blasRanges.size(); ++b) {
            const auto& range = scene.blasRanges[b];
            const uint32_t numNonOpaque = (range.triangleCount > range.numOpaqueTriangles) ? (range.triangleCount - range.numOpaqueTriangles) : 0;
            if (numNonOpaque > 0 && subrangeHasAlphaCutout(range.firstTriangle + range.numOpaqueTriangles, numNonOpaque, 0)) {
                blasHasAlphaCutout[b] = true;
            }
        }
        for (size_t i = 0; i < scene.instances.size(); ++i) {
            const uint32_t bIdx = scene.instances[i].blasIndex;
            const uint32_t matOffset = (i < scene.instanceData.size()) ? scene.instanceData[i].materialOffset : 0;
            if (matOffset != 0 && bIdx < scene.blasRanges.size() && !blasHasAlphaCutout[bIdx]) {
                const auto& range = scene.blasRanges[bIdx];
                const uint32_t numNonOpaque = (range.triangleCount > range.numOpaqueTriangles) ? (range.triangleCount - range.numOpaqueTriangles) : 0;
                if (numNonOpaque > 0 && subrangeHasAlphaCutout(range.firstTriangle + range.numOpaqueTriangles, numNonOpaque, matOffset)) {
                    blasHasAlphaCutout[bIdx] = true;
                }
            }
        }

        std::vector<std::vector<ASGeometryInput>> allBlasGeoms;
        allBlasGeoms.reserve(scene.blasRanges.size());
        for (size_t b = 0; b < scene.blasRanges.size(); ++b) {
            const auto& range = scene.blasRanges[b];
            std::vector<ASGeometryInput> geoms;
            if (range.numOpaqueTriangles > 0) {
                ASGeometryInput geomOpaque{};
                geomOpaque.vertexBufferAddress = vertexBaseAddr + static_cast<VkDeviceSize>(range.firstTriangle) * kTriVertexBytes;
                geomOpaque.indexBufferAddress = 0;
                geomOpaque.vertexCount = 3 * range.numOpaqueTriangles - 1;
                geomOpaque.triangleCount = range.numOpaqueTriangles;
                geomOpaque.vertexStride = sizeof(glm::vec4);
                geomOpaque.indexType = VK_INDEX_TYPE_NONE_KHR;
                geomOpaque.isOpaque = true;
                geoms.push_back(geomOpaque);
            }
            uint32_t numNonOpaque = (range.triangleCount > range.numOpaqueTriangles) ? (range.triangleCount - range.numOpaqueTriangles) : 0;
            if (numNonOpaque > 0) {
                ASGeometryInput geomNonOpaque{};
                geomNonOpaque.vertexBufferAddress = vertexBaseAddr + static_cast<VkDeviceSize>(range.firstTriangle + range.numOpaqueTriangles) * kTriVertexBytes;
                geomNonOpaque.indexBufferAddress = 0;
                geomNonOpaque.vertexCount = 3 * numNonOpaque - 1;
                geomNonOpaque.triangleCount = numNonOpaque;
                geomNonOpaque.vertexStride = sizeof(glm::vec4);
                geomNonOpaque.indexType = VK_INDEX_TYPE_NONE_KHR;
                geomNonOpaque.isOpaque = !blasHasAlphaCutout[b];
                geoms.push_back(geomNonOpaque);
            }
            if (geoms.empty()) {
                ASGeometryInput dummyGeom{};
                dummyGeom.vertexBufferAddress = vertexBaseAddr;
                dummyGeom.indexBufferAddress = 0;
                dummyGeom.vertexCount = 2;
                dummyGeom.triangleCount = 1;
                dummyGeom.vertexStride = sizeof(glm::vec4);
                dummyGeom.indexType = VK_INDEX_TYPE_NONE_KHR;
                dummyGeom.isOpaque = true;
                geoms.push_back(dummyGeom);
            }
            allBlasGeoms.push_back(std::move(geoms));
        }
        secNode.blases = secNode.asManager->buildBLASBatch(allBlasGeoms);

        asInstances.reserve(scene.instances.size());
        for (const auto& inst : scene.instances) {
            ASInstanceInput asInst{};
            uint32_t bIdx = std::min(inst.blasIndex, static_cast<uint32_t>(secNode.blases.size() - 1));
            asInst.blasAddress = secNode.blases[bIdx]->getDeviceAddress();
            asInst.transform = inst.transform;
            asInst.customIndex = inst.customIndex;
            if (bIdx < scene.blasRanges.size()) {
                const auto& range = scene.blasRanges[bIdx];
                if (range.triangleCount == range.numOpaqueTriangles) {
                    asInst.mask = 0x01; // Pure opaque
                } else if (range.numOpaqueTriangles == 0) {
                    asInst.mask = 0x02; // Pure non-opaque / dielectric
                } else {
                    asInst.mask = 0x03; // Mixed
                }
            } else {
                asInst.mask = 0xFF;
            }
            asInst.hitGroupId = 0;
            asInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            asInstances.push_back(asInst);
        }
        secNode.tlas = secNode.asManager->buildTLAS(asInstances);
        Logger::info("Secondary GPU Multi-BLAS Acceleration Structures initialized successfully ({} BLASes, {} TLAS Instances).",
                     secNode.blases.size(), asInstances.size());
    } else {
        // Monolithic scene path (up to 3 BLAS buckets: Opaque, Solid Non-Opaque, Alpha Non-Opaque)
        secNode.asManager->resetStats();

        std::vector<std::vector<ASGeometryInput>> allMonoGeoms;
        allMonoGeoms.reserve(monoBuckets.size());
        for (const auto& b : monoBuckets) {
            std::vector<ASGeometryInput> geoms;
            ASGeometryInput geom{};
            if (b.triCount > 0) {
                geom.vertexBufferAddress = vertexBaseAddr + static_cast<VkDeviceSize>(b.firstTri) * kTriVertexBytes;
                geom.indexBufferAddress = 0;
                geom.vertexCount = 3 * b.triCount - 1;
                geom.triangleCount = b.triCount;
                geom.vertexStride = sizeof(glm::vec4);
                geom.indexType = VK_INDEX_TYPE_NONE_KHR;
                geom.isOpaque = b.isGeomOpaque;
            } else {
                geom.vertexBufferAddress = vertexBaseAddr;
                geom.indexBufferAddress = 0;
                geom.vertexCount = 2;
                geom.triangleCount = 1;
                geom.vertexStride = sizeof(glm::vec4);
                geom.indexType = VK_INDEX_TYPE_NONE_KHR;
                geom.isOpaque = true;
            }
            geoms.push_back(geom);
            allMonoGeoms.push_back(std::move(geoms));
        }

        if (allMonoGeoms.size() == 1) {
            secNode.blases.push_back(secNode.asManager->buildBLAS(allMonoGeoms[0]));
        } else {
            secNode.blases = secNode.asManager->buildBLASBatch(allMonoGeoms);
        }

        asInstances.reserve(monoBuckets.size());
        for (size_t idx = 0; idx < monoBuckets.size(); ++idx) {
            ASInstanceInput inst{};
            inst.blasAddress = secNode.blases[idx]->getDeviceAddress();
            inst.transform = glm::mat4(1.0f);
            inst.customIndex = static_cast<uint32_t>(idx);
            inst.mask = monoBuckets[idx].mask;
            inst.hitGroupId = 0;
            inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            asInstances.push_back(inst);
        }

        secNode.tlas = secNode.asManager->buildTLAS(asInstances);
        Logger::info("Secondary GPU Acceleration Structures initialized successfully (Monolithic {} BLASes & {} TLAS Instances).",
                     secNode.blases.size(), asInstances.size());
    }

    secNode.tlasUpdatePipeline = std::make_unique<GpuTlasUpdatePipeline>(
        secDevice, secAlloc, secNode.context->hasSubgroupSizeControl()
    );
    secNode.tlasUpdatePipeline->initShaderInstances(secNode.instanceBuffer.get(), instanceUpload);
    secNode.tlasUpdatePipeline->initBuffers(asInstances, secNode.asManager.get());
    if (!updateTlasSpv.empty()) {
        secNode.tlasUpdatePipeline->createPipeline(updateTlasSpv);
    }
}

GpuDeviceNode::~GpuDeviceNode() {
    if (!context) return;
    VkDevice device = context->getDevice();
    vkDeviceWaitIdle(device);

    causticsPipeline.reset();
    tlasUpdatePipeline.reset();
    rtpKhrPipeline.reset();
    wavefrontPipeline.reset();
    if (rtpPipelineLayout) vkDestroyPipelineLayout(device, rtpPipelineLayout, nullptr);
    for (uint32_t i = 0; i < NUM_IN_FLIGHT; ++i) {
        if (queryPools[i]) vkDestroyQueryPool(device, queryPools[i], nullptr);
        if (renderFences[i]) vkDestroyFence(device, renderFences[i], nullptr);
        if (secSemaphores[i]) vkDestroySemaphore(device, secSemaphores[i], nullptr);
        cameraUBOs[i].reset();
        lightStaging[i].reset();
        lightTreeStaging[i].reset();
        videoStagingBuffers[i].reset();
    }
    if (secTimelineSemaphore) vkDestroySemaphore(device, secTimelineSemaphore, nullptr);
    if (rtDescLayout) vkDestroyDescriptorSetLayout(device, rtDescLayout, nullptr);
    if (descriptorPool) vkDestroyDescriptorPool(device, descriptorPool, nullptr);
    if (commandPool) vkDestroyCommandPool(device, commandPool, nullptr);

    for (uint32_t i = 0; i < NUM_IN_FLIGHT; ++i) {
        accumTargets[i].reset();
    }
    triangleBuffer.reset();
    sphereBuffer.reset();
    materialBuffer.reset();
    materialArchetypeBuffer.reset();
    shadeMaterialBuffer.reset();
    lightBuffer.reset();
    lightTreeBuffer.reset();
    normalDepthImage.reset();
    prevNormalDepthImage.reset();
    motionVectorImage.reset();
    causticImage.reset();
    mlDiffuseImage.reset();
    mlSpecularImage.reset();
    mlAlbedoRoughnessImage.reset();
    mlSpecularMotionImage.reset();
    upscaler.reset();
}

MultiGpuManager::MultiGpuManager(const Config& config, VulkanContext* primaryContext, const SceneData& scene)
    : m_primaryContext(primaryContext), m_mode(config.mgpu_mode), m_config(config) {

    auto devices = VulkanContext::enumeratePhysicalDevices(primaryContext->getInstance());
    std::vector<VkPhysicalDevice> hwDevices;
    for (auto d : devices) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(d, &props);
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
            hwDevices.push_back(d);
            if (d != primaryContext->getPhysicalDevice() && m_secondaryDeviceName.empty()) {
                m_secondaryDeviceName = props.deviceName;
            }
        }
    }
    m_hwDeviceCount = std::max<uint32_t>(1u, static_cast<uint32_t>(hwDevices.size()));
    if (hwDevices.size() < 2) {
        Logger::info("Multi-GPU: only {} hardware GPU(s) found (ignoring CPU/software rasterizers). Multi-GPU unavailable.", hwDevices.size());
        m_mode = MultiGpuMode::Off;
        return;
    }

    if (m_mode != MultiGpuMode::Off) {
        Logger::info("Initializing Multi-GPU Manager across {} discrete GPUs (Initial State: Active)...", hwDevices.size());
        ensureSecondaryInitialized(config, scene);
    } else {
        Logger::info("Multi-GPU Manager in lazy standby across {} discrete GPUs ({} unallocated until Multi-GPU is enabled).",
                     hwDevices.size(), m_secondaryDeviceName.empty() ? "Secondary GPU" : m_secondaryDeviceName);
    }
}

bool MultiGpuManager::ensureSecondaryInitialized(const Config& config, const SceneData& scene) {
    if (m_hwDeviceCount < 2) {
        m_mode = MultiGpuMode::Off;
        return false;
    }
    if (isSecondaryInitialized()) {
        return true;
    }
    m_config = config;
    if (config.mgpu_mode != MultiGpuMode::Off) {
        m_mode = config.mgpu_mode;
    }
    initSecondaryDevice(config, scene);
    if (m_active && !m_workerThread.joinable()) {
        m_workerThread = std::thread(&MultiGpuManager::workerLoop, this);
    }
    return isSecondaryInitialized();
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
            if (m_devices[0]->primImportedTimelineSemaphore != VK_NULL_HANDLE) {
                vkDestroySemaphore(primDevice, m_devices[0]->primImportedTimelineSemaphore, nullptr);
                m_devices[0]->primImportedTimelineSemaphore = VK_NULL_HANDLE;
            }
        }
    }
    destroySharedHostBuffer();
    m_devices.clear();
}

void MultiGpuManager::collectSecondaryTimestamps(uint32_t bufferSlot) {
    if (!isMultiGpuActive() || m_devices.empty()) return;
    GpuDeviceNode* node = m_devices[0].get();
    uint32_t slot = bufferSlot % GpuDeviceNode::NUM_IN_FLIGHT;
    if (!node->slotHasExecuted[slot]) return;
    VkDevice device = node->context->getDevice();
    if (vkWaitForFences(device, 1, &node->renderFences[slot], VK_TRUE, 100'000'000ull /* 100 ms */) != VK_SUCCESS) {
        return;
    }
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

void MultiGpuManager::setWarmupSppOverride(uint32_t spp) {
    if (!m_devices.empty() && m_devices[0]->wavefrontPipeline) {
        m_devices[0]->wavefrontPipeline->setWarmupSppOverride(spp);
    }
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
    if (!m_devices.empty()) return m_devices[0]->deviceName;
    return m_secondaryDeviceName;
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

    std::vector<std::string> searchPaths;

    if (!exeDir.empty()) {
        searchPaths.push_back((exeDir / "shaders" / filename).string());
        searchPaths.push_back((exeDir / filename).string());
        searchPaths.push_back((exeDir / ".." / "share" / "pathways" / "shaders" / filename).string());
    }

    searchPaths.push_back(std::string(SHADER_DIR) + "/" + filename);
    searchPaths.push_back(std::string("shaders/") + filename);
    searchPaths.push_back(std::string("build/shaders/") + filename);
    searchPaths.push_back(std::string("../build/shaders/") + filename);
    searchPaths.push_back(std::string("/usr/share/pathways/shaders/") + filename);
    searchPaths.push_back(std::string("/usr/local/share/pathways/shaders/") + filename);
    searchPaths.push_back(filename);

    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (file.is_open()) {
                size_t fileSize = static_cast<size_t>(file.tellg());
                std::vector<char> buffer(fileSize);
                file.seekg(0);
                file.read(buffer.data(), fileSize);
                Logger::debug("MGPU loaded shader SPIR-V from: {}", path);
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

void MultiGpuManager::destroySharedP2PBuffer() {
    VkDevice dev0 = m_primaryContext ? m_primaryContext->getDevice() : VK_NULL_HANDLE;
    VkDevice dev1 = (!m_devices.empty() && m_devices[0]->context) ? m_devices[0]->context->getDevice() : VK_NULL_HANDLE;

    for (uint32_t slot = 0; slot < NUM_SHARED_BUFFERS; ++slot) {
        if (dev1 != VK_NULL_HANDLE) {
            if (m_p2pBufferSecondary[slot] != VK_NULL_HANDLE) {
                vkDestroyBuffer(dev1, m_p2pBufferSecondary[slot], nullptr);
                m_p2pBufferSecondary[slot] = VK_NULL_HANDLE;
            }
            if (m_p2pMemSecondary[slot] != VK_NULL_HANDLE) {
                vkFreeMemory(dev1, m_p2pMemSecondary[slot], nullptr);
                m_p2pMemSecondary[slot] = VK_NULL_HANDLE;
            }
        }
        if (dev0 != VK_NULL_HANDLE) {
            if (m_p2pBufferPrimary[slot] != VK_NULL_HANDLE) {
                vkDestroyBuffer(dev0, m_p2pBufferPrimary[slot], nullptr);
                m_p2pBufferPrimary[slot] = VK_NULL_HANDLE;
            }
            if (m_p2pMemPrimary[slot] != VK_NULL_HANDLE) {
                vkFreeMemory(dev0, m_p2pMemPrimary[slot], nullptr);
                m_p2pMemPrimary[slot] = VK_NULL_HANDLE;
            }
        }
    }
}

bool MultiGpuManager::initSharedP2PBuffer(VkDeviceSize bufferSize) {
    destroySharedP2PBuffer();

    if (!m_primaryContext || !m_primaryContext->hasExternalMemoryDmaBuf() || !m_primaryContext->hasExternalMemoryFd() ||
        m_devices.empty() || !m_devices[0]->context || !m_devices[0]->context->hasExternalMemoryDmaBuf() || !m_devices[0]->context->hasExternalMemoryFd()) {
        return false;
    }

    auto pfnGetFd0 = m_primaryContext->pfnGetMemoryFdKHR;
    auto pfnGetFdProps1 = m_devices[0]->context->pfnGetMemoryFdPropertiesKHR;
    if (!pfnGetFd0 || !pfnGetFdProps1) {
        return false;
    }

    VkDevice dev0 = m_primaryContext->getDevice();
    VkPhysicalDevice phys0 = m_primaryContext->getPhysicalDevice();
    VkDevice dev1 = m_devices[0]->context->getDevice();
    VkPhysicalDevice phys1 = m_devices[0]->context->getPhysicalDevice();

    VkPhysicalDeviceMemoryProperties memProps0, memProps1;
    vkGetPhysicalDeviceMemoryProperties(phys0, &memProps0);
    vkGetPhysicalDeviceMemoryProperties(phys1, &memProps1);

    // Find BAR-visible Device-Local memory type on Device 0 (DEVICE_LOCAL | HOST_VISIBLE, excluding uncached/device-coherent AMD heaps)
    constexpr VkMemoryPropertyFlags kDisallowedUncachedFlags =
        VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
    constexpr VkMemoryPropertyFlags kBarVisibleFlags =
        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    uint32_t devLocalIdx0 = UINT32_MAX;
    for (uint32_t i = 0; i < memProps0.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags flags = memProps0.memoryTypes[i].propertyFlags;
        if ((flags & kBarVisibleFlags) == kBarVisibleFlags && (flags & kDisallowedUncachedFlags) == 0) {
            devLocalIdx0 = i;
            break;
        }
    }
    if (devLocalIdx0 == UINT32_MAX) {
        for (uint32_t i = 0; i < memProps0.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags flags = memProps0.memoryTypes[i].propertyFlags;
            if ((flags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && (flags & kDisallowedUncachedFlags) == 0) {
                devLocalIdx0 = i;
                break;
            }
        }
    }
    if (devLocalIdx0 == UINT32_MAX) return false;

    // Detect BAR aperture size: inspect heaps mapped to HOST_VISIBLE | DEVICE_LOCAL
    VkDeviceSize barVisibleVramSize0 = 0;
    for (uint32_t i = 0; i < memProps0.memoryTypeCount; ++i) {
        const auto& type = memProps0.memoryTypes[i];
        if ((type.propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) ==
            (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
            barVisibleVramSize0 = std::max(barVisibleVramSize0, memProps0.memoryHeaps[type.heapIndex].size);
        }
    }

    // Align buffer size to 64KB (standard PCI page alignment)
    VkDeviceSize alignment = 65536;
    VkDeviceSize alignedSize = (bufferSize + alignment - 1) & ~(alignment - 1);
    auto handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    bool isSmallBar = (barVisibleVramSize0 > 0 && barVisibleVramSize0 <= 256 * 1024 * 1024);
    if (isSmallBar) {
        Logger::info("Inter-GPU P2P: Small-BAR topology detected ({:.0f} MB BAR0). Budgeted 2x {:.2f} MB in BAR-visible VRAM.",
                     static_cast<double>(barVisibleVramSize0) / (1024.0 * 1024.0),
                     static_cast<double>(alignedSize) / (1024.0 * 1024.0));
    } else if (barVisibleVramSize0 > 256 * 1024 * 1024) {
        Logger::info("Inter-GPU P2P: Resizable BAR detected ({:.2f} GB BAR0). Allocating 2x {:.2f} MB in VRAM.",
                     static_cast<double>(barVisibleVramSize0) / (1024.0 * 1024.0 * 1024.0),
                     static_cast<double>(alignedSize) / (1024.0 * 1024.0));
    }

    bool allSucceeded = true;
    for (uint32_t slot = 0; slot < NUM_SHARED_BUFFERS; ++slot) {
        // 1. Create buffer on Device 0 (GPU 0 VRAM)
        VkExternalMemoryBufferCreateInfo extBufInfo0{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
        extBufInfo0.handleTypes = handleType;

        std::vector<uint32_t> dev0QueueFamilies;
        dev0QueueFamilies.push_back(m_primaryContext->getGraphicsQueueFamily());
        if (m_primaryContext->hasDedicatedAsyncCompute() &&
            m_primaryContext->getAsyncComputeQueueFamily() != m_primaryContext->getGraphicsQueueFamily()) {
            dev0QueueFamilies.push_back(m_primaryContext->getAsyncComputeQueueFamily());
        }

        VkBufferCreateInfo bufInfo0{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufInfo0.pNext = &extBufInfo0;
        bufInfo0.size = alignedSize;
        bufInfo0.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (dev0QueueFamilies.size() > 1) {
            bufInfo0.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bufInfo0.queueFamilyIndexCount = static_cast<uint32_t>(dev0QueueFamilies.size());
            bufInfo0.pQueueFamilyIndices = dev0QueueFamilies.data();
        } else {
            bufInfo0.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

        VkResult res = vkCreateBuffer(dev0, &bufInfo0, nullptr, &m_p2pBufferPrimary[slot]);
        if (res != VK_SUCCESS) { allSucceeded = false; break; }

        VkBufferMemoryRequirementsInfo2 reqInfo0{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2 };
        reqInfo0.buffer = m_p2pBufferPrimary[slot];
        VkMemoryRequirements2 memReq2{ VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
        vkGetBufferMemoryRequirements2(dev0, &reqInfo0, &memReq2);
        VkMemoryRequirements memReq0 = memReq2.memoryRequirements;

        // 2. Allocate exportable Device-Local memory on Device 0
        VkExportMemoryAllocateInfo exportAllocInfo{ VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO };
        exportAllocInfo.handleTypes = handleType;

        VkMemoryAllocateInfo allocInfo0{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo0.pNext = &exportAllocInfo;
        allocInfo0.allocationSize = memReq0.size;
        allocInfo0.memoryTypeIndex = devLocalIdx0;

        res = vkAllocateMemory(dev0, &allocInfo0, nullptr, &m_p2pMemPrimary[slot]);
        if (res != VK_SUCCESS) { allSucceeded = false; break; }

        res = vkBindBufferMemory(dev0, m_p2pBufferPrimary[slot], m_p2pMemPrimary[slot], 0);
        if (res != VK_SUCCESS) { allSucceeded = false; break; }

        // 3. Export FD from Device 0
        VkMemoryGetFdInfoKHR getFdInfo{ VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR };
        getFdInfo.memory = m_p2pMemPrimary[slot];
        getFdInfo.handleType = handleType;

        int memFd = -1;
        res = pfnGetFd0(dev0, &getFdInfo, &memFd);
        if (res != VK_SUCCESS || memFd < 0) { allSucceeded = false; break; }

        // 4. Query FD memory type on Device 1
        VkMemoryFdPropertiesKHR fdProps{ VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
        res = pfnGetFdProps1(dev1, handleType, memFd, &fdProps);
        if (res != VK_SUCCESS) {
            closeFileDescriptor(memFd);
            allSucceeded = false;
            break;
        }

        uint32_t memIdx1 = UINT32_MAX;
        // Prefer external/peer aperture memory type (non-DEVICE_LOCAL on the importing peer device)
        // so the importing driver does not mark the foreign DMA-BUF BO's preferred domain as its own local VRAM.
        for (uint32_t i = 0; i < memProps1.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags flags1 = memProps1.memoryTypes[i].propertyFlags;
            if ((fdProps.memoryTypeBits & (1 << i)) &&
                (flags1 & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) == 0 &&
                (flags1 & kDisallowedUncachedFlags) == 0) {
                memIdx1 = i;
                break;
            }
        }
        if (memIdx1 == UINT32_MAX) {
            for (uint32_t i = 0; i < memProps1.memoryTypeCount; ++i) {
                if ((fdProps.memoryTypeBits & (1 << i)) &&
                    (memProps1.memoryTypes[i].propertyFlags & kDisallowedUncachedFlags) == 0) {
                    memIdx1 = i;
                    break;
                }
            }
        }
        if (memIdx1 == UINT32_MAX) {
            for (uint32_t i = 0; i < memProps1.memoryTypeCount; ++i) {
                if (fdProps.memoryTypeBits & (1 << i)) {
                    memIdx1 = i;
                    break;
                }
            }
        }
        if (memIdx1 == UINT32_MAX) {
            closeFileDescriptor(memFd);
            allSucceeded = false;
            break;
        }

        // 5. Import FD into Device 1
        VkImportMemoryFdInfoKHR importInfo{ VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR };
        importInfo.handleType = handleType;
        importInfo.fd = memFd;

        VkMemoryAllocateInfo allocInfo1{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        allocInfo1.pNext = &importInfo;
        allocInfo1.allocationSize = memReq0.size;
        allocInfo1.memoryTypeIndex = memIdx1;

        res = vkAllocateMemory(dev1, &allocInfo1, nullptr, &m_p2pMemSecondary[slot]);
        if (res != VK_SUCCESS) {
            closeFileDescriptor(memFd);
            allSucceeded = false;
            break;
        }

        // 6. Create buffer on Device 1 and bind imported memory
        VkExternalMemoryBufferCreateInfo extBufInfo1{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
        extBufInfo1.handleTypes = handleType;

        VkBufferCreateInfo bufInfo1{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufInfo1.pNext = &extBufInfo1;
        bufInfo1.size = alignedSize;
        bufInfo1.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufInfo1.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        res = vkCreateBuffer(dev1, &bufInfo1, nullptr, &m_p2pBufferSecondary[slot]);
        if (res != VK_SUCCESS) { allSucceeded = false; break; }

        res = vkBindBufferMemory(dev1, m_p2pBufferSecondary[slot], m_p2pMemSecondary[slot], 0);
        if (res != VK_SUCCESS) { allSucceeded = false; break; }
    }

    if (allSucceeded) {
        m_sharedBufferSize = alignedSize;
        m_transferMode = InterGpuTransferMode::P2P_Direct_BAR;
        m_useZeroCopyHost = false;
        ++m_sharedBufferGeneration;
        Logger::info("Inter-GPU: Activated P2P Direct BAR Transfer via Linux DMA-BUF (2x {:.2f} MB, zero host copies).",
                     static_cast<double>(alignedSize) / (1024.0 * 1024.0));
        return true;
    }

    destroySharedP2PBuffer();
    return false;
}

void MultiGpuManager::destroySharedHostBuffer() {
    destroySharedP2PBuffer();
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
#if defined(_WIN32)
            _aligned_free(m_sharedHostPtr[slot]);
#else
            free(m_sharedHostPtr[slot]);
#endif
            m_sharedHostPtr[slot] = nullptr;
        }
    }
    m_sharedBufferSize = 0;
    m_useZeroCopyHost = false;
    m_transferMode = InterGpuTransferMode::ZeroCopy_HostMemory;
}

void MultiGpuManager::initSharedHostBuffer(VkDeviceSize bufferSize) {
    if (m_p2pBufferPrimary[0] != VK_NULL_HANDLE || m_sharedBufferPrimary[0] != VK_NULL_HANDLE) {
        waitWorkerIdle();
        if (m_primaryContext && m_primaryContext->getDevice() != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(m_primaryContext->getDevice());
        }
        if (!m_devices.empty() && m_devices[0]->context && m_devices[0]->context->getDevice() != VK_NULL_HANDLE) {
            vkDeviceWaitIdle(m_devices[0]->context->getDevice());
        }
    }
    destroySharedHostBuffer();

    // Query BAR-visible VRAM size on Primary Device (Dev 0)
    VkPhysicalDevice phys0 = m_primaryContext ? m_primaryContext->getPhysicalDevice() : VK_NULL_HANDLE;
    VkDeviceSize barVisibleVramSize0 = 0;
    if (phys0 != VK_NULL_HANDLE) {
        VkPhysicalDeviceMemoryProperties memProps0;
        vkGetPhysicalDeviceMemoryProperties(phys0, &memProps0);
        for (uint32_t i = 0; i < memProps0.memoryTypeCount; ++i) {
            const auto& type = memProps0.memoryTypes[i];
            if ((type.propertyFlags & (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) ==
                (VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                barVisibleVramSize0 = std::max(barVisibleVramSize0, memProps0.memoryHeaps[type.heapIndex].size);
            }
        }
    }
    bool isSmallBar = (barVisibleVramSize0 > 0 && barVisibleVramSize0 <= 256 * 1024 * 1024);

    bool attemptP2P = false;
    if (m_config.mgpu_transfer_explicit) {
        attemptP2P = (m_config.mgpu_transfer_mode == Config::MgpuTransferMode::P2P);
    } else {
#if defined(__linux__)
        // On Linux amdgpu/radv, DMA-BUF exports share a single kernel dma_resv reservation object
        // (AMDGPU_SYNC_NE_OWNER) that serializes peer queue submissions and throttles SMU boost clocks.
        // Zero-Copy Host Memory (VK_EXT_external_memory_host) creates independent per-device BOs backed
        // by pinned host pages, achieving lock-free dual-PCIe 4.0 x16 concurrency and 1.85x-1.95x scaling.
        attemptP2P = false;
#else
        if (!isSmallBar) {
            attemptP2P = true;
        } else {
            Logger::info("Inter-GPU: Small-BAR topology detected ({:.0f} MB BAR0, Above 4G Decoding disabled). Auto-falling back to Zero-Copy Host Memory (2.0x scaling). Use --mgpu-transfer p2p to force direct BAR.",
                         static_cast<double>(barVisibleVramSize0) / (1024.0 * 1024.0));
            attemptP2P = false;
        }
#endif
    }

    if (attemptP2P) {
        if (initSharedP2PBuffer(bufferSize)) {
            Logger::info("Active Inter-GPU Transfer: P2P Direct BAR via Linux DMA-BUF (Device-local VRAM streaming).");
            return;
        }
        Logger::info("P2P Direct BAR not available or unsupported by platform/hardware topology. Falling back to Zero-Copy Host Memory.");
    }

    // Option 2 (Default): High-Performance Zero-Copy Host Memory via VK_EXT_external_memory_host
    if (!(m_primaryContext && m_primaryContext->hasExternalMemoryHost() &&
          !m_devices.empty() && m_devices[0]->context && m_devices[0]->context->hasExternalMemoryHost())) {
        throw std::runtime_error("Multi-GPU requires high-performance Zero-Copy Host Memory (VK_EXT_external_memory_host) or Direct P2P BAR. Slower CPU staging workarounds are not supported under Vulkan 1.4 baseline.");
    }

    VkDevice dev0 = m_primaryContext->getDevice();
    phys0 = m_primaryContext->getPhysicalDevice();
    VkDevice dev1 = m_devices[0]->context->getDevice();
    VkPhysicalDevice phys1 = m_devices[0]->context->getPhysicalDevice();

    // Query minImportedHostPointerAlignment from both physical devices (CRIT-08)
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostExtProps0{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props2_0{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2_0.pNext = &hostExtProps0;
    vkGetPhysicalDeviceProperties2(phys0, &props2_0);

    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hostExtProps1{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props2_1{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2_1.pNext = &hostExtProps1;
    vkGetPhysicalDeviceProperties2(phys1, &props2_1);

    // On Linux amdgpu/radv, every vkQueueSubmit2 validates userptr page mappings via hmm_range_fault.
    // Aligning host allocations to 2 MiB PMD boundaries and advising MADV_HUGEPAGE collapses 512x 4 KiB
    // PTE walks into contiguous Transparent Huge Pages, eliminating CPU submission bottlenecks.
    VkDeviceSize hostAlignment = std::max({
#if !defined(_WIN32)
        static_cast<VkDeviceSize>(2u * 1024u * 1024u),
#else
        static_cast<VkDeviceSize>(65536u),
#endif
        hostExtProps0.minImportedHostPointerAlignment,
        hostExtProps1.minImportedHostPointerAlignment
    });
    m_sharedBufferSize = (bufferSize + hostAlignment - 1) & ~(hostAlignment - 1);

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

    const uint32_t activeSlots = m_config.double_buffered_shared_mem ? NUM_SHARED_BUFFERS : 1u;
    bool allSucceeded = true;
    for (uint32_t slot = 0; slot < activeSlots; ++slot) {
#if defined(_WIN32)
        m_sharedHostPtr[slot] = _aligned_malloc(m_sharedBufferSize, hostAlignment);
        if (!m_sharedHostPtr[slot]) {
            Logger::warn("Failed to allocate page-aligned host memory for slot {}.", slot);
            allSucceeded = false;
            break;
        }
#else
        if (posix_memalign(&m_sharedHostPtr[slot], hostAlignment, m_sharedBufferSize) != 0 || !m_sharedHostPtr[slot]) {
            Logger::warn("Failed to allocate page-aligned host memory for slot {}.", slot);
            allSucceeded = false;
            break;
        }
#ifdef MADV_HUGEPAGE
        madvise(m_sharedHostPtr[slot], m_sharedBufferSize, MADV_HUGEPAGE);
#endif
#endif
        std::memset(m_sharedHostPtr[slot], 0, m_sharedBufferSize);

        VkMemoryHostPointerPropertiesEXT hostProps0{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        pfnGet0(dev0, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, m_sharedHostPtr[slot], &hostProps0);

        VkMemoryHostPointerPropertiesEXT hostProps1{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
        pfnGet1(dev1, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, m_sharedHostPtr[slot], &hostProps1);

        constexpr VkMemoryPropertyFlags kDisallowedUncachedFlags =
            VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD | VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD;
        uint32_t memIdx0 = UINT32_MAX, memIdx1 = UINT32_MAX;
        for (uint32_t i = 0; i < memProps0.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags flags = memProps0.memoryTypes[i].propertyFlags;
            if ((hostProps0.memoryTypeBits & (1 << i)) &&
                (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
                (flags & kDisallowedUncachedFlags) == 0) {
                memIdx0 = i;
                break;
            }
        }
        if (memIdx0 == UINT32_MAX) {
            for (uint32_t i = 0; i < memProps0.memoryTypeCount; ++i) {
                if ((hostProps0.memoryTypeBits & (1 << i)) &&
                    (memProps0.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                    memIdx0 = i;
                    break;
                }
            }
        }
        if (memIdx0 == UINT32_MAX) {
            for (uint32_t i = 0; i < memProps0.memoryTypeCount; ++i) {
                if (hostProps0.memoryTypeBits & (1 << i)) { memIdx0 = i; break; }
            }
        }

        for (uint32_t i = 0; i < memProps1.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags flags = memProps1.memoryTypes[i].propertyFlags;
            if ((hostProps1.memoryTypeBits & (1 << i)) &&
                (flags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) &&
                (flags & kDisallowedUncachedFlags) == 0) {
                memIdx1 = i;
                break;
            }
        }
        if (memIdx1 == UINT32_MAX) {
            for (uint32_t i = 0; i < memProps1.memoryTypeCount; ++i) {
                if ((hostProps1.memoryTypeBits & (1 << i)) &&
                    (memProps1.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
                    memIdx1 = i;
                    break;
                }
            }
        }
        if (memIdx1 == UINT32_MAX) {
            for (uint32_t i = 0; i < memProps1.memoryTypeCount; ++i) {
                if (hostProps1.memoryTypeBits & (1 << i)) { memIdx1 = i; break; }
            }
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

        std::vector<uint32_t> dev0QueueFamilies;
        dev0QueueFamilies.push_back(m_primaryContext->getGraphicsQueueFamily());
        if (m_primaryContext->hasDedicatedAsyncCompute() &&
            m_primaryContext->getAsyncComputeQueueFamily() != m_primaryContext->getGraphicsQueueFamily()) {
            dev0QueueFamilies.push_back(m_primaryContext->getAsyncComputeQueueFamily());
        }

        VkBufferCreateInfo bufInfo0{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bufInfo0.pNext = &extBufInfo0;
        bufInfo0.size = m_sharedBufferSize;
        bufInfo0.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (dev0QueueFamilies.size() > 1) {
            bufInfo0.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bufInfo0.queueFamilyIndexCount = static_cast<uint32_t>(dev0QueueFamilies.size());
            bufInfo0.pQueueFamilyIndices = dev0QueueFamilies.data();
        } else {
            bufInfo0.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }

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
        m_transferMode = InterGpuTransferMode::ZeroCopy_HostMemory;
        ++m_sharedBufferGeneration;
        Logger::info("{} Zero-Copy Inter-GPU Host Buffers initialized via VK_EXT_external_memory_host ({}x {:.2f} MB).",
                     m_config.double_buffered_shared_mem ? "Double-Buffered" : "Single-Buffered (Double-Buffering Disabled)",
                     activeSlots,
                     static_cast<double>(m_sharedBufferSize) / (1024.0 * 1024.0));
    } else {
        destroySharedHostBuffer();
        throw std::runtime_error("Failed to bind zero-copy host buffers on both GPUs. Slower CPU staging workarounds are not supported under Vulkan 1.4 baseline.");
    }
}

VkDeviceSize MultiGpuManager::calculateSharedBufferSize(uint32_t width, uint32_t height) const {
    auto align64k = [](VkDeviceSize sz) -> VkDeviceSize {
        return (sz + 65535) & ~static_cast<VkDeviceSize>(65535);
    };

    const bool isSampleBlendFsr =
        (m_config.upscaler_mode == UpscalerMode::FSR &&
         m_config.mgpu_upscale_mode == MgpuUpscaleMode::SampleBlend &&
         m_config.denoiser_mode != DenoiserMode::Svgf &&
         m_config.denoiser_mode != DenoiserMode::FfxRayRegeneration);
    if (isSampleBlendFsr) {
        return align64k(static_cast<VkDeviceSize>(width) * height * 8);
    }

    const uint32_t renderW = (m_config.upscaler_mode != UpscalerMode::None && m_config.render_scale < 1.0f)
        ? m_config.getRenderWidth() : width;
    const uint32_t renderH = (m_config.upscaler_mode != UpscalerMode::None && m_config.render_scale < 1.0f)
        ? m_config.getRenderHeight() : height;

    const uint32_t tileSize = (m_config.tile_size == 0u) ? 64u : m_config.tile_size;
    const uint32_t numTilesX = (renderW + tileSize - 1u) / tileSize;
    const uint32_t maxTilesPerGpuX = (numTilesX + 1u) / 2u;
    const uint32_t cbWidth = maxTilesPerGpuX * tileSize;

    const uint32_t bpp = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? 8 :
                         (m_config.accum_format == AccumFormat::R11G11B10_UFLOAT) ? 4 : 16;

    const bool needMl = needSecondaryMlGuides();
    const bool needNormals = needMl || m_config.enable_restir || m_config.enable_caustics;
    const bool needMvDepth = needNormals || (m_config.upscaler_mode == UpscalerMode::FSR);
    const bool copyMv = needNormals || (needMvDepth && m_config.accum_format == AccumFormat::R11G11B10_UFLOAT);

    // CheckerboardTile: radiance + only the active auxiliary G-buffer planes
    const VkDeviceSize cbRadAligned = align64k(static_cast<VkDeviceSize>(cbWidth) * renderH * bpp);
    const VkDeviceSize cbPlaneAligned = align64k(static_cast<VkDeviceSize>(cbWidth) * renderH * 8);
    VkDeviceSize cbTotalSize = cbRadAligned;
    if (needMl) {
        cbTotalSize += 6 * cbPlaneAligned;
    } else if (copyMv) {
        cbTotalSize += (needNormals ? 2 : 1) * cbPlaneAligned;
    }

    // SampleParallel: radiance (+ mlDiffuse + mlSpecular only when ML guides are active)
    const VkDeviceSize spRadAligned = align64k(static_cast<VkDeviceSize>(renderW) * renderH * bpp);
    const VkDeviceSize spPlaneAligned = align64k(static_cast<VkDeviceSize>(renderW) * renderH * 8);
    const VkDeviceSize spTotalSize = spRadAligned + (needMl ? 2 * spPlaneAligned : 0);

    const MultiGpuMode effectiveMode = (m_config.mgpu_mode != MultiGpuMode::Off) ? m_config.mgpu_mode : m_mode;
    if (effectiveMode == MultiGpuMode::CheckerboardTile || effectiveMode == MultiGpuMode::Off) {
        return cbTotalSize;
    }
    if (effectiveMode == MultiGpuMode::SampleParallel) {
        return spTotalSize;
    }
    if (m_config.adaptive_spp) {
        return std::max(cbTotalSize, spTotalSize);
    }
    return (m_config.spp > 1) ? spTotalSize : cbTotalSize;
}

void MultiGpuManager::initSecondaryDevice(const Config& config, const SceneData& scene) {
    auto devices = VulkanContext::enumeratePhysicalDevices(m_primaryContext->getInstance());
    uint32_t secondaryGpuIndex = UINT32_MAX;
    for (uint32_t i = 0; i < devices.size(); ++i) {
        if (devices[i] == m_primaryContext->getPhysicalDevice()) {
            continue;
        }
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU) {
            secondaryGpuIndex = i;
            break;
        }
    }

    if (secondaryGpuIndex == UINT32_MAX) {
        Logger::info("Multi-GPU: No secondary hardware GPU found (ignoring CPU/software rasterizers). Multi-GPU unavailable.");
        m_mode = MultiGpuMode::Off;
        return;
    }

    Config secConfig = config;
    secConfig.gpu_index = secondaryGpuIndex; // Explicit secondary hardware GPU
    m_boundsMin = scene.boundsMin;
    m_boundsMax = scene.boundsMax;
    m_hasDielectrics = scene.hasDielectrics;
    m_dielectricBoundsMin = scene.dielectricBoundsMin;
    m_dielectricBoundsMax = scene.dielectricBoundsMax;

    auto secNode = std::make_unique<GpuDeviceNode>();
    secNode->deviceIndex = secondaryGpuIndex;
    secNode->context = std::make_unique<VulkanContext>(secConfig, VK_NULL_HANDLE, "Secondary GPU / Peer Compute");
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

    // 2b. Cross-GPU Hardware Synchronization Semaphores
    VkDevice primDevice = m_primaryContext->getDevice();
#if defined(_WIN32)
    m_useCrossGpuSync = m_primaryContext->hasExternalSemaphoreWin32() && secNode->context->hasExternalSemaphoreWin32() &&
                        m_primaryContext->pfnImportSemaphoreWin32HandleKHR && secNode->context->pfnGetSemaphoreWin32HandleKHR;
    if (!m_useCrossGpuSync) {
        Logger::error("Multi-GPU requires cross-GPU hardware semaphore synchronization (VK_KHR_external_semaphore_win32), which is not supported by the Vulkan devices on this system. Disabling Multi-GPU.");
        m_mode = MultiGpuMode::Off;
        return;
    }
    for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
        VkExportSemaphoreCreateInfo exportInfo{ VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
        exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

        VkSemaphoreCreateInfo secSemInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        secSemInfo.pNext = &exportInfo;
        vkCreateSemaphore(secDevice, &secSemInfo, nullptr, &secNode->secSemaphores[i]);

        VkSemaphoreCreateInfo primSemInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        vkCreateSemaphore(primDevice, &primSemInfo, nullptr, &secNode->primImportedSemaphores[i]);
        secNode->exportedFd[i] = -1;
        secNode->slotFdReady[i] = false;
    }
    // Permanent Cross-GPU Hardware Timeline Semaphore (VK_KHR_external_semaphore_win32 + VK_SEMAPHORE_TYPE_TIMELINE)
    VkSemaphoreTypeCreateInfo timelineTypeSec{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    timelineTypeSec.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineTypeSec.initialValue = 0;

    VkExportSemaphoreCreateInfo exportTimelineInfo{ VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
    exportTimelineInfo.pNext = &timelineTypeSec;
    exportTimelineInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;

    VkSemaphoreCreateInfo secTimelineSemInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    secTimelineSemInfo.pNext = &exportTimelineInfo;
    vkCreateSemaphore(secDevice, &secTimelineSemInfo, nullptr, &secNode->secTimelineSemaphore);

    VkSemaphoreGetWin32HandleInfoKHR getTimelineHandleInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR };
    getTimelineHandleInfo.semaphore = secNode->secTimelineSemaphore;
    getTimelineHandleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
    HANDLE timelineHandle = NULL;
    secNode->context->pfnGetSemaphoreWin32HandleKHR(secDevice, &getTimelineHandleInfo, &timelineHandle);

    VkSemaphoreTypeCreateInfo timelineTypePrim{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    timelineTypePrim.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineTypePrim.initialValue = 0;

    VkSemaphoreCreateInfo primTimelineSemInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    primTimelineSemInfo.pNext = &timelineTypePrim;
    vkCreateSemaphore(primDevice, &primTimelineSemInfo, nullptr, &secNode->primImportedTimelineSemaphore);

    if (timelineHandle != NULL) {
        VkImportSemaphoreWin32HandleInfoKHR importTimelineHandleInfo{ VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_WIN32_HANDLE_INFO_KHR };
        importTimelineHandleInfo.semaphore = secNode->primImportedTimelineSemaphore;
        importTimelineHandleInfo.flags = 0; // Permanent import on primary device
        importTimelineHandleInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
        importTimelineHandleInfo.handle = timelineHandle;
        m_primaryContext->pfnImportSemaphoreWin32HandleKHR(primDevice, &importTimelineHandleInfo);
        CloseHandle(timelineHandle);
    }

    Logger::info("Cross-GPU Hardware Synchronization active via VK_KHR_external_semaphore_win32 & Timeline Semaphores (zero-wait GPU-to-GPU pipelining).");
#else
    m_useCrossGpuSync = m_primaryContext->hasExternalSemaphoreFd() && secNode->context->hasExternalSemaphoreFd() &&
                        m_primaryContext->pfnImportSemaphoreFdKHR && secNode->context->pfnGetSemaphoreFdKHR;
    if (!m_useCrossGpuSync) {
        Logger::error("Multi-GPU requires cross-GPU hardware semaphore synchronization (VK_KHR_external_semaphore_fd), which is not supported by the Vulkan devices on this system. Disabling Multi-GPU.");
        m_mode = MultiGpuMode::Off;
        return;
    }
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
    // Permanent Cross-GPU Hardware Timeline Semaphore (VK_KHR_external_semaphore_fd + VK_SEMAPHORE_TYPE_TIMELINE)
    VkSemaphoreTypeCreateInfo timelineTypeSec{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    timelineTypeSec.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineTypeSec.initialValue = 0;

    VkExportSemaphoreCreateInfo exportTimelineInfo{ VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
    exportTimelineInfo.pNext = &timelineTypeSec;
    exportTimelineInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo secTimelineSemInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    secTimelineSemInfo.pNext = &exportTimelineInfo;
    vkCreateSemaphore(secDevice, &secTimelineSemInfo, nullptr, &secNode->secTimelineSemaphore);

    VkSemaphoreGetFdInfoKHR getTimelineFdInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR };
    getTimelineFdInfo.semaphore = secNode->secTimelineSemaphore;
    getTimelineFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
    int timelineFd = -1;
    secNode->context->pfnGetSemaphoreFdKHR(secDevice, &getTimelineFdInfo, &timelineFd);

    VkSemaphoreTypeCreateInfo timelineTypePrim{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
    timelineTypePrim.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    timelineTypePrim.initialValue = 0;

    VkSemaphoreCreateInfo primTimelineSemInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    primTimelineSemInfo.pNext = &timelineTypePrim;
    vkCreateSemaphore(primDevice, &primTimelineSemInfo, nullptr, &secNode->primImportedTimelineSemaphore);

    if (timelineFd >= 0) {
        VkImportSemaphoreFdInfoKHR importTimelineFdInfo{ VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR };
        importTimelineFdInfo.semaphore = secNode->primImportedTimelineSemaphore;
        importTimelineFdInfo.flags = 0; // Permanent import on primary device
        importTimelineFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        importTimelineFdInfo.fd = timelineFd;
        m_primaryContext->pfnImportSemaphoreFdKHR(primDevice, &importTimelineFdInfo);
        closeFileDescriptor(timelineFd);
    }

    Logger::info("Cross-GPU Hardware Synchronization active via VK_KHR_external_semaphore_fd & Timeline Semaphores (zero-wait GPU-to-GPU pipelining).");
#endif

    // 3. Render Targets on secondary device (sized to active secondary dispatch dimensions)
    uint32_t initRenderW = (config.upscaler_mode != UpscalerMode::None && config.render_scale < 1.0f)
        ? config.getRenderWidth() : config.width;
    uint32_t initRenderH = (config.upscaler_mode != UpscalerMode::None && config.render_scale < 1.0f)
        ? config.getRenderHeight() : config.height;
    uint32_t initTileSize = (config.tile_size == 0u) ? 64u : config.tile_size;
    uint32_t initNumTilesX = (initRenderW + initTileSize - 1u) / initTileSize;
    uint32_t initMaxTilesPerGpuX = (initNumTilesX + 1u) / 2u;
    MultiGpuMode initEffectiveMode = config.mgpu_mode;
    if (initEffectiveMode == MultiGpuMode::Auto) {
        initEffectiveMode = (config.spp > 1) ? MultiGpuMode::SampleParallel : MultiGpuMode::CheckerboardTile;
    }
    bool initIsCheckerboard = (initEffectiveMode == MultiGpuMode::CheckerboardTile || initEffectiveMode == MultiGpuMode::Off);
    uint32_t secWidth = initIsCheckerboard ? (initMaxTilesPerGpuX * initTileSize) : std::max(initRenderW, initMaxTilesPerGpuX * initTileSize);
    uint32_t secHeight = initRenderH;
    VkFormat accumFormat = (config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT :
                           (config.accum_format == AccumFormat::R11G11B10_UFLOAT) ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 : VK_FORMAT_R32G32B32A32_SFLOAT;
    for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
        secNode->accumTargets[i] = std::make_unique<Image>(
            secDevice, secAlloc, secWidth, secHeight,
            accumFormat,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
    }

    // 4. Scene Buffers on secondary device (pure DEVICE_LOCAL VRAM with staging upload, 16-byte aligned)
    size_t numTris = scene.triangles.size();
    std::vector<glm::vec4> positions;
    positions.reserve(numTris * 3);
    std::vector<TriangleShadeGPU> shadeTriangles;
    shadeTriangles.reserve(numTris);

    for (const auto& tri : scene.triangles) {
        positions.push_back(glm::vec4(glm::vec3(tri.v0.position), 1.0f));
        positions.push_back(glm::vec4(glm::vec3(tri.v1.position), 1.0f));
        positions.push_back(glm::vec4(glm::vec3(tri.v2.position), 1.0f));

        shadeTriangles.push_back(createTriangleShadeGPU(tri));
    }

    VkDeviceSize posSize = std::max(sizeof(glm::vec4) * positions.size(), sizeof(glm::vec4) * 3);
    secNode->positionBuffer = std::make_unique<Buffer>(
        secAlloc, posSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (!positions.empty()) {
        uploadToDeviceBufferSec(*secNode, *secNode->positionBuffer, positions.data(), sizeof(glm::vec4) * positions.size());
    } else {
        glm::vec4 dummyVerts[3] = {
            glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
            glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
            glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
        };
        uploadToDeviceBufferSec(*secNode, *secNode->positionBuffer, dummyVerts, sizeof(dummyVerts));
    }

    VkDeviceSize triSize = std::max(sizeof(TriangleShadeGPU) * shadeTriangles.size(), sizeof(TriangleShadeGPU));
    secNode->triangleBuffer = std::make_unique<Buffer>(
        secAlloc, triSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (!shadeTriangles.empty()) {
        uploadToDeviceBufferSec(*secNode, *secNode->triangleBuffer, shadeTriangles.data(), sizeof(TriangleShadeGPU) * shadeTriangles.size());
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

    std::vector<uint32_t> matArchetypes(scene.materials.size());
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        matArchetypes[i] = computeMaterialArchetype(scene.materials[i]);
    }
    VkDeviceSize archSize = std::max(sizeof(uint32_t) * matArchetypes.size(), sizeof(uint32_t));
    secNode->materialArchetypeBuffer = std::make_unique<Buffer>(
        secAlloc, archSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!matArchetypes.empty()) {
        secNode->materialArchetypeBuffer->copyFrom(matArchetypes.data(), sizeof(uint32_t) * matArchetypes.size());
    }

    // Compact 64-byte Shading Material buffer (2 materials per 128B RDNA 4 vector cache line)
    std::vector<ShadeMaterialGPU> shadeMaterials(scene.materials.size());
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        shadeMaterials[i] = createShadeMaterial(scene.materials[i]);
    }
    VkDeviceSize shadeMatSize = std::max(sizeof(ShadeMaterialGPU) * shadeMaterials.size(), sizeof(ShadeMaterialGPU));
    secNode->shadeMaterialBuffer = std::make_unique<Buffer>(
        secAlloc, shadeMatSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!shadeMaterials.empty()) {
        secNode->shadeMaterialBuffer->copyFrom(shadeMaterials.data(), sizeof(ShadeMaterialGPU) * shadeMaterials.size());
    }

    VkDeviceSize lightSize = std::max(sizeof(LightGPU) * scene.lights.size(), sizeof(LightGPU));
    secNode->lightBuffer = std::make_unique<Buffer>(
        secAlloc, lightSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.lights.empty()) {
        secNode->lightBuffer->copyFrom(scene.lights.data(), sizeof(LightGPU) * scene.lights.size());
    }

    VkDeviceSize lightTreeSize = std::max(
        sizeof(LightTreeNodeGPU) * scene.lightTreeNodes.size(),
        sizeof(LightTreeNodeGPU) * std::max<size_t>(1, 2 * scene.lights.size())
    );
    secNode->lightTreeBuffer = std::make_unique<Buffer>(
        secAlloc, lightTreeSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.lightTreeNodes.empty()) {
        secNode->lightTreeBuffer->copyFrom(scene.lightTreeNodes.data(), sizeof(LightTreeNodeGPU) * scene.lightTreeNodes.size());
    }

    // Camera UBOs on secondary device
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
    auto updateTlasSpv = loadShaderSPIRV("update_tlas_instances.comp.spv");
    createSecondaryAccelerationStructures(*secNode, scene, updateTlasSpv);
    secNode->positionBuffer.reset();

    // Textures & HDRI Environment Map on secondary device
    VkQueue secQueue = secNode->context->getGraphicsQueue();
    VkCommandPool secPool = secNode->commandPool;

    secNode->dummyWhite = Texture::createDummyWhite(secDevice, secAlloc, secQueue, secPool);
    secNode->dummyNormal = Texture::createDummyNormal(secDevice, secAlloc, secQueue, secPool);
    secNode->blueNoiseTexture = Texture::createBlueNoise64(secDevice, secAlloc, secQueue, secPool);

    secNode->environmentMap = Texture::createSceneEnvironmentMap(
        secDevice, secAlloc, secQueue, secPool,
        config.hdri_path, config.scene_path, scene.domeLightHdriPath
    );

    secNode->sceneTextures.clear();
    for (const auto& texData : scene.textures) {
        if (texData.isHdr && !texData.hdrPixels.empty() && texData.width > 0 && texData.height > 0) {
            bool genMips = (texData.width >= 64 && texData.height >= 64);
            auto tex = Texture::createFromPixels(
                secDevice, secAlloc, secQueue, secPool,
                texData.width, texData.height,
                VK_FORMAT_R16G16B16A16_SFLOAT, texData.hdrPixels.data(),
                texData.hdrPixels.size() * sizeof(uint16_t), true, genMips
            );
            secNode->sceneTextures.push_back(std::move(tex));
        } else if (!texData.pixels.empty() && texData.width > 0 && texData.height > 0) {
            VkFormat fmt = texData.isSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
            bool genMips = (texData.width >= 64 && texData.height >= 64);
            auto tex = Texture::createFromPixels(
                secDevice, secAlloc, secQueue, secPool,
                texData.width, texData.height,
                fmt, texData.pixels.data(),
                texData.pixels.size(), false, genMips
            );
            secNode->sceneTextures.push_back(std::move(tex));
        } else {
            secNode->sceneTextures.push_back(Texture::createDummyWhite(secDevice, secAlloc, secQueue, secPool));
        }
    }

    // 5. Descriptor Pool & Sets on secondary device
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 64 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 4 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096 }
    };
    VkDescriptorPoolCreateInfo descPoolInfo{};
    descPoolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    descPoolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    descPoolInfo.pPoolSizes = poolSizes.data();
    descPoolInfo.maxSets = 32;
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
        { 12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 13, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, rtStages, nullptr },
        { 14, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 15, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr },
        { 26, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 27, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 28, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 29, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, rtStages, nullptr },
        { 30, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, rtStages, nullptr }
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

    // Allocate secondary G-Buffer & scratch images
    secNode->normalDepthImage = std::make_unique<Image>(secDevice, secAlloc, secWidth, secHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

    secNode->prevNormalDepthImage = std::make_unique<Image>(secDevice, secAlloc, secWidth, secHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    secNode->motionVectorImage = std::make_unique<Image>(secDevice, secAlloc, secWidth, secHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    secNode->causticImage = std::make_unique<Image>(secDevice, secAlloc, secWidth, secHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

    VkDescriptorImageInfo normDepthInfo{ VK_NULL_HANDLE, secNode->normalDepthImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo mvInfo{ VK_NULL_HANDLE, secNode->motionVectorImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo causticInfo{ VK_NULL_HANDLE, secNode->causticImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

    VkDescriptorBufferInfo triInfo{ secNode->triangleBuffer->getBuffer(), 0, secNode->triangleBuffer->getSize() };
    VkDescriptorBufferInfo sphereInfo{ secNode->sphereBuffer->getBuffer(), 0, secNode->sphereBuffer->getSize() };
    VkDescriptorBufferInfo matInfo{ secNode->materialBuffer->getBuffer(), 0, secNode->materialBuffer->getSize() };
    VkDescriptorBufferInfo lightInfo{ secNode->lightBuffer->getBuffer(), 0, secNode->lightBuffer->getSize() };
    Buffer* secLightTree = secNode->lightTreeBuffer ? secNode->lightTreeBuffer.get() : secNode->lightBuffer.get();
    Buffer* secInstances = secNode->instanceBuffer ? secNode->instanceBuffer.get() : secNode->triangleBuffer.get();
    VkDescriptorBufferInfo lightTreeInfo{ secLightTree->getBuffer(), 0, secLightTree->getSize() };
    VkDescriptorBufferInfo instanceInfo{ secInstances->getBuffer(), 0, secInstances->getSize() };

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    VkAccelerationStructureKHR tlasHandle = secNode->tlas ? secNode->tlas->getHandle() : VK_NULL_HANDLE;
    asInfo.pAccelerationStructures = &tlasHandle;

    VkDescriptorImageInfo envInfo = secNode->environmentMap ? secNode->environmentMap->getDescriptorInfo() : secNode->dummyWhite->getDescriptorInfo();
    VkDescriptorImageInfo blueNoiseInfo = secNode->blueNoiseTexture ? secNode->blueNoiseTexture->getDescriptorInfo() : secNode->dummyWhite->getDescriptorInfo();

    std::vector<VkDescriptorImageInfo> texInfos(MAX_SCENE_TEXTURES);
    for (size_t i = 0; i < MAX_SCENE_TEXTURES; ++i) {
        if (i < secNode->sceneTextures.size() && secNode->sceneTextures[i]) {
            texInfos[i] = secNode->sceneTextures[i]->getDescriptorInfo();
        } else {
            texInfos[i] = secNode->dummyWhite->getDescriptorInfo();
        }
    }

    for (uint32_t slot = 0; slot < GpuDeviceNode::NUM_IN_FLIGHT; ++slot) {
        VkDescriptorImageInfo accumImageInfo{};
        accumImageInfo.imageView = secNode->accumTargets[slot]->getImageView();
        accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

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
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normDepthInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 13, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &blueNoiseInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mvInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 15, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &causticInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 25, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightTreeInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 26, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normDepthInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 27, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normDepthInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 28, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normDepthInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 29, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normDepthInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 30, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &instanceInfo, nullptr }
        };
        vkUpdateDescriptorSets(secDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }

    // 6. Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline) on Secondary GPU
    VkPushConstantRange rtpPushConstant{};
    rtpPushConstant.stageFlags = rtStages;
    rtpPushConstant.offset = 0;
    rtpPushConstant.size = sizeof(uint32_t) * 16;

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
        rgenCode, rmissCode, shadowMissCode, rchitCode,
        secNode->context->hasRtSubgroupSizeControl()
    );
    Logger::info("Secondary GPU: Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline) initialized.");

    // Wavefront Path Tracing Pipeline on secondary device
    if (config.pipeline_type == PipelineType::Wavefront) {
        try {
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

            bool enableSecExecSet = secNode->context->supportsDgcExecutionSet() && Config::dgcExecSetRequested();

            secNode->wavefrontPipeline = std::make_unique<WavefrontPipeline>(
                secDevice, secAlloc,
                secWidth, secHeight,
                wfClassifyCode, wfIntersectCode, wfShadeCode, wfShadowCode,
                wfShadeDiffuseCode, wfShadeDielectricCode, wfShadeConductorCode, wfShadeComplexCode,
                wfShadeEmissiveCode, wfShadePassthroughCode,
                enableSecExecSet,
                wfShadeDiffuseSecCode, wfShadeComplexSecCode,
                config.dgc_preprocess && secNode->context->supportsDgcPreprocess(),
                secNode->context->hasSubgroupSizeControl(),
                0,
                wfTailMegakernelCode
            );
            secNode->wavefrontPipeline->initSpecializedVariants(
                loadShaderSPIRV("wavefront_intersect_alpha.comp.spv"),
                loadShaderSPIRV("wavefront_shade_diffuse_restir.comp.spv"),
                loadShaderSPIRV("wavefront_shade_diffuse_sec_restir.comp.spv"),
                loadShaderSPIRV("wavefront_shade_complex_restir.comp.spv"),
                loadShaderSPIRV("wavefront_shade_complex_sec_restir.comp.spv"),
                loadShaderSPIRV("wavefront_shade_dielectric_restir.comp.spv"),
                loadShaderSPIRV("wavefront_shade_emissive_restir.comp.spv")
            );
            secNode->restirManager = std::make_unique<ReSTIRManager>(secDevice, secAlloc);
            if (config.enable_restir && config.mgpu_mode != MultiGpuMode::Off) {
                secNode->restirManager->resize(std::max(initRenderW, secWidth), initRenderH);
            }
            GpuDeviceNode* rawNode = secNode.get();
            secNode->wavefrontPipeline->setPostClassifyCallback([this, rawNode](VkCommandBuffer cb, uint32_t cbSlot) {
                if (m_config.enable_caustics && rawNode->causticsPipeline && rawNode->normalDepthImage) {
                    rawNode->causticsPipeline->recordSplatAndFilter(
                        cb, cbSlot,
                        rawNode->normalDepthImage.get(),
                        rawNode->currentCameraFov,
                        rawNode->currentFrameIndex,
                        false,
                        rawNode->currentCameraMoved,
                        m_hasDielectrics,
                        m_dielectricBoundsMin,
                        m_dielectricBoundsMax,
                        rawNode->currentFullScreenWidth,
                        rawNode->currentCbTileSize
                    );
                }
            });
            updateSecondaryWavefrontDescriptors(secNode.get());
            Logger::info("Secondary GPU: Wavefront Path Tracing Pipeline (Ray Queues & DGC) initialized successfully.");
        } catch (const std::exception& e) {
            Logger::warn("Failed to initialize secondary GPU Wavefront pipeline: {}. Falling back to RTP.", e.what());
            secNode->wavefrontPipeline.reset();
        }
    }

    // 7b. Secondary GPU FSR Upscaler (Sample Parallel SampleBlend mode only)
    if (config.upscaler_mode == UpscalerMode::FSR && config.mgpu_upscale_mode == MgpuUpscaleMode::SampleBlend) {
        try {
            auto prepCode = loadShaderSPIRV("ffx_sdk_prep.comp.spv");
            uint32_t renderW = config.getRenderWidth();
            uint32_t renderH = config.getRenderHeight();
            secNode->upscaler = std::make_unique<FsrUpscaler>(
                secDevice,
                secNode->context->getPhysicalDevice(),
                secAlloc,
                renderW,
                renderH,
                config.width,
                config.height,
                prepCode,
                accumFormat,
                "[GPU 1 Secondary Peer]",
                config.fsr_version
            );
            Logger::info("Secondary GPU: FSR Upscaler initialized successfully [GPU 1 Secondary Peer].");
        } catch (const std::exception& e) {
            Logger::warn("Failed to initialize secondary GPU FSR upscaler: {}", e.what());
            secNode->upscaler.reset();
        }
    }

    // 8. Transition secondary images to GENERAL layout
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(secNode->commandBuffers[0], &beginInfo);

    secNode->motionVectorImage->transitionLayout(
        secNode->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );

    secNode->causticImage->transitionLayout(
        secNode->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );
    VkClearColorValue blackClr{};
    VkImageSubresourceRange clearRng{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    vkCmdClearColorImage(secNode->commandBuffers[0], secNode->causticImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &blackClr, 1, &clearRng);

    for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
        secNode->accumTargets[i]->transitionLayout(
            secNode->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }
    secNode->normalDepthImage->transitionLayout(
        secNode->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );
    secNode->prevNormalDepthImage->transitionLayout(
        secNode->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
    );
    vkCmdClearColorImage(secNode->commandBuffers[0], secNode->prevNormalDepthImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &blackClr, 1, &clearRng);

    VkClearColorValue clearColor = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
        vkCmdClearColorImage(secNode->commandBuffers[0], secNode->accumTargets[i]->getImage(),
                             VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &clearRange);
    }

    vkEndCommandBuffer(secNode->commandBuffers[0]);

    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = secNode->commandBuffers[0];

    VkSubmitInfo2 initSubmit{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    initSubmit.commandBufferInfoCount = 1;
    initSubmit.pCommandBufferInfos = &cmdSubmitInfo;
    vkQueueSubmit2(secNode->context->getGraphicsQueue(), 1, &initSubmit, VK_NULL_HANDLE);
    vkQueueWaitIdle(secNode->context->getGraphicsQueue());

    Logger::info("Secondary GPU Node fully initialized: {} ({})", secNode->deviceName, secNode->context->getPciLinkString());
    m_devices.push_back(std::move(secNode));
    m_active = true;

    // Initialize shared inter-GPU buffer across primary and secondary GPUs (Radiance + MV + Normal/Depth)
    m_config = config;
    VkDeviceSize bufferSize = calculateSharedBufferSize(config.width, config.height);
    initSharedHostBuffer(bufferSize);
}

bool MultiGpuManager::needSecondaryMlGuides() const {
    return (m_config.upscaler_mode == UpscalerMode::Upways ||
            m_config.denoiser_mode == DenoiserMode::Upways ||
            m_config.denoiser_mode == DenoiserMode::Svgf ||
            m_config.denoiser_mode == DenoiserMode::FfxRayRegeneration ||
            m_config.upways_superres ||
            !m_config.capture_training_data_dir.empty());
}

void MultiGpuManager::ensureSecondaryMlGuides(GpuDeviceNode* secNode) {
    if (!secNode || !secNode->context || !secNode->accumTargets[0] || !secNode->normalDepthImage) return;

    VkDevice secDevice = secNode->context->getDevice();
    VmaAllocator secAlloc = secNode->context->getAllocator();
    uint32_t secW = secNode->accumTargets[0]->getWidth();
    uint32_t secH = secNode->accumTargets[0]->getHeight();
    const bool needed = needSecondaryMlGuides();

    if (!needed) {
        if (secNode->mlDiffuseImage) {
            vkDeviceWaitIdle(secDevice);
            secNode->mlDiffuseImage.reset();
            secNode->mlSpecularImage.reset();
            secNode->mlAlbedoRoughnessImage.reset();
            secNode->mlSpecularMotionImage.reset();

            VkDescriptorImageInfo fallbackInfo{ VK_NULL_HANDLE, secNode->normalDepthImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            for (uint32_t slot = 0; slot < GpuDeviceNode::NUM_IN_FLIGHT; ++slot) {
                VkWriteDescriptorSet writes[4] = {
                    { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 26, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &fallbackInfo, nullptr, nullptr },
                    { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 27, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &fallbackInfo, nullptr, nullptr },
                    { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 28, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &fallbackInfo, nullptr, nullptr },
                    { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 29, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &fallbackInfo, nullptr, nullptr }
                };
                vkUpdateDescriptorSets(secDevice, 4, writes, 0, nullptr);
            }
            updateSecondaryWavefrontDescriptors(secNode);
        }
        return;
    }

    if (secNode->mlDiffuseImage &&
        secNode->mlDiffuseImage->getWidth() == secW &&
        secNode->mlDiffuseImage->getHeight() == secH) {
        return;
    }

    vkDeviceWaitIdle(secDevice);
    constexpr VkImageUsageFlags mlUsage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    secNode->mlDiffuseImage = std::make_unique<Image>(secDevice, secAlloc, secW, secH, VK_FORMAT_R16G16B16A16_SFLOAT, mlUsage);
    secNode->mlSpecularImage = std::make_unique<Image>(secDevice, secAlloc, secW, secH, VK_FORMAT_R16G16B16A16_SFLOAT, mlUsage);
    secNode->mlAlbedoRoughnessImage = std::make_unique<Image>(secDevice, secAlloc, secW, secH, VK_FORMAT_R16G16B16A16_SFLOAT, mlUsage);
    secNode->mlSpecularMotionImage = std::make_unique<Image>(secDevice, secAlloc, secW, secH, VK_FORMAT_R16G16B16A16_SFLOAT, mlUsage);

    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = secNode->commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer initCmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(secDevice, &allocInfo, &initCmd);

    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(initCmd, &beginInfo);

    VkClearColorValue zeroClr{};
    VkImageSubresourceRange clearRng{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    Image* mlImgs[4] = {
        secNode->mlDiffuseImage.get(),
        secNode->mlSpecularImage.get(),
        secNode->mlAlbedoRoughnessImage.get(),
        secNode->mlSpecularMotionImage.get()
    };
    for (Image* img : mlImgs) {
        img->transitionLayout(
            initCmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
        vkCmdClearColorImage(initCmd, img->getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClr, 1, &clearRng);
    }
    vkEndCommandBuffer(initCmd);

    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = initCmd;
    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
    vkQueueSubmit2(secNode->context->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(secNode->context->getGraphicsQueue());
    vkFreeCommandBuffers(secDevice, secNode->commandPool, 1, &initCmd);

    VkDescriptorImageInfo albInfo{ VK_NULL_HANDLE, secNode->mlAlbedoRoughnessImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo smvInfo{ VK_NULL_HANDLE, secNode->mlSpecularMotionImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diffInfo{ VK_NULL_HANDLE, secNode->mlDiffuseImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo specInfo{ VK_NULL_HANDLE, secNode->mlSpecularImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    for (uint32_t slot = 0; slot < GpuDeviceNode::NUM_IN_FLIGHT; ++slot) {
        VkWriteDescriptorSet writes[4] = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 26, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &albInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 27, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &smvInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 28, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &diffInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 29, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &specInfo, nullptr, nullptr }
        };
        vkUpdateDescriptorSets(secDevice, 4, writes, 0, nullptr);
    }
    updateSecondaryWavefrontDescriptors(secNode);
}

void MultiGpuManager::ensureSecondaryCaustics(GpuDeviceNode* secNode) {
    if (!secNode || !secNode->context || !secNode->accumTargets[0] || !secNode->causticImage) return;

    VkDevice secDevice = secNode->context->getDevice();
    VmaAllocator secAlloc = secNode->context->getAllocator();
    const bool needed = m_config.enable_caustics && m_hasDielectrics;

    if (!needed) {
        if (secNode->causticsPipeline) {
            vkDeviceWaitIdle(secDevice);
            secNode->causticsPipeline.reset();
            VkDescriptorImageInfo fallbackCaustic{ VK_NULL_HANDLE, secNode->causticImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            for (uint32_t slot = 0; slot < GpuDeviceNode::NUM_IN_FLIGHT; ++slot) {
                VkWriteDescriptorSet write{
                    VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot],
                    15, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &fallbackCaustic, nullptr, nullptr
                };
                vkUpdateDescriptorSets(secDevice, 1, &write, 0, nullptr);
            }
            updateSecondaryWavefrontDescriptors(secNode);
        }
        return;
    }

    uint32_t secW = secNode->accumTargets[0]->getWidth();
    uint32_t secH = secNode->accumTargets[0]->getHeight();
    if (secNode->causticsPipeline &&
        secNode->causticsPipeline->getFilteredCausticImage() &&
        secNode->causticsPipeline->getFilteredCausticImage()->getWidth() == secW &&
        secNode->causticsPipeline->getFilteredCausticImage()->getHeight() == secH) {
        return;
    }

    vkDeviceWaitIdle(secDevice);
    auto immediateSubmitSec = [secNode, secDevice](const std::function<void(VkCommandBuffer)>& recordFn) {
        VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
        allocInfo.commandPool = secNode->commandPool;
        allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        allocInfo.commandBufferCount = 1;
        VkCommandBuffer oneShotCmd = VK_NULL_HANDLE;
        vkAllocateCommandBuffers(secDevice, &allocInfo, &oneShotCmd);

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(oneShotCmd, &beginInfo);
        recordFn(oneShotCmd);
        vkEndCommandBuffer(oneShotCmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = oneShotCmd;
        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(secNode->context->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(secNode->context->getGraphicsQueue());
        vkFreeCommandBuffers(secDevice, secNode->commandPool, 1, &oneShotCmd);
    };

    try {
        if (!secNode->causticsPipeline) {
            auto traceSpv = loadShaderSPIRV("caustic_photon_trace.comp.spv");
            auto splatSpv = loadShaderSPIRV("caustic_splat.comp.spv");
            auto filterSpv = loadShaderSPIRV("caustic_filter.comp.spv");
            secNode->causticsPipeline = std::make_unique<CausticsPipeline>(
                secDevice, secAlloc,
                secNode->context->hasSubgroupSizeControl(),
                secW, secH, m_config.caustic_photons,
                traceSpv, splatSpv, filterSpv,
                immediateSubmitSec
            );
        } else {
            secNode->causticsPipeline->resize(secW, secH, immediateSubmitSec);
        }

        VkAccelerationStructureKHR tlasHandle = secNode->tlas ? secNode->tlas->getHandle() : VK_NULL_HANDLE;
        secNode->causticsPipeline->updateDescriptors(
            secNode->triangleBuffer.get(),
            secNode->sphereBuffer.get(),
            secNode->materialBuffer.get(),
            secNode->lightBuffer.get(),
            tlasHandle,
            secNode->sceneTextures,
            secNode->dummyWhite.get(),
            secNode->instanceBuffer.get(),
            secNode->cameraUBOs,
            secNode->normalDepthImage.get(),
            secNode->motionVectorImage.get(),
            secNode->prevNormalDepthImage.get()
        );

        Image* filteredCaustic = secNode->causticsPipeline->getFilteredCausticImage();
        VkDescriptorImageInfo causticInfo{
            VK_NULL_HANDLE,
            filteredCaustic ? filteredCaustic->getImageView() : secNode->causticImage->getImageView(),
            VK_IMAGE_LAYOUT_GENERAL
        };
        for (uint32_t slot = 0; slot < GpuDeviceNode::NUM_IN_FLIGHT; ++slot) {
            VkWriteDescriptorSet write{
                VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot],
                15, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &causticInfo, nullptr, nullptr
            };
            vkUpdateDescriptorSets(secDevice, 1, &write, 0, nullptr);
        }
        updateSecondaryWavefrontDescriptors(secNode);
    } catch (const std::exception& e) {
        Logger::warn("Failed to initialize secondary GPU Caustics pipeline: {}", e.what());
        secNode->causticsPipeline.reset();
    }
}

void MultiGpuManager::updateSecondaryWavefrontDescriptors(GpuDeviceNode* secNode) {
    if (!secNode || !secNode->wavefrontPipeline || !secNode->accumTargets[0] || !secNode->triangleBuffer) return;

    VkAccelerationStructureKHR tlasHandle = secNode->tlas ? secNode->tlas->getHandle() : VK_NULL_HANDLE;
    VkDescriptorImageInfo envInfo = secNode->environmentMap ? secNode->environmentMap->getDescriptorInfo() : secNode->dummyWhite->getDescriptorInfo();

    std::vector<VkDescriptorImageInfo> texInfos(MAX_SCENE_TEXTURES);
    for (size_t i = 0; i < MAX_SCENE_TEXTURES; ++i) {
        if (i < secNode->sceneTextures.size() && secNode->sceneTextures[i]) {
            texInfos[i] = secNode->sceneTextures[i]->getDescriptorInfo();
        } else {
            texInfos[i] = secNode->dummyWhite->getDescriptorInfo();
        }
    }

    VkImageView activeCausticView = (secNode->causticsPipeline && secNode->causticsPipeline->getFilteredCausticImage())
        ? secNode->causticsPipeline->getFilteredCausticImage()->getImageView()
        : (secNode->causticImage ? secNode->causticImage->getImageView() : VK_NULL_HANDLE);

    for (uint32_t slot = 0; slot < GpuDeviceNode::NUM_IN_FLIGHT; ++slot) {
        if (!secNode->cameraUBOs[slot]) continue;
        Buffer* restirCurrBuf = secNode->restirManager ? secNode->restirManager->getReservoirBuffer(slot) : nullptr;
        Buffer* restirPrevBuf = secNode->restirManager ? secNode->restirManager->getReservoirBuffer(1 - slot) : nullptr;
        VkBuffer restirReservoirBuf = restirCurrBuf ? restirCurrBuf->getBuffer() : VK_NULL_HANDLE;
        VkBuffer restirHistoryReservoirBuf = restirPrevBuf ? restirPrevBuf->getBuffer() : VK_NULL_HANDLE;
        Buffer* giCurrBuf = secNode->restirManager ? secNode->restirManager->getGiReservoirBuffer(slot) : nullptr;
        Buffer* giPrevBuf = secNode->restirManager ? secNode->restirManager->getGiReservoirBuffer(1 - slot) : nullptr;
        Buffer* x1CtxBuf = secNode->restirManager ? secNode->restirManager->getX1ContextBuffer() : nullptr;

        secNode->wavefrontPipeline->updateSceneDescriptors(
            slot,
            secNode->accumTargets[slot]->getImageView(),
            secNode->cameraUBOs[slot]->getBuffer(),
            secNode->triangleBuffer->getBuffer(), secNode->triangleBuffer->getSize(),
            secNode->sphereBuffer->getBuffer(), secNode->sphereBuffer->getSize(),
            secNode->materialBuffer->getBuffer(), secNode->materialBuffer->getSize(),
            secNode->lightBuffer->getBuffer(), secNode->lightBuffer->getSize(),
            tlasHandle,
            envInfo,
            texInfos,
            VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
            secNode->motionVectorImage ? secNode->motionVectorImage->getImageView() : VK_NULL_HANDLE,
            secNode->normalDepthImage ? secNode->normalDepthImage->getImageView() : VK_NULL_HANDLE,
            secNode->lightTreeBuffer ? secNode->lightTreeBuffer->getBuffer() : VK_NULL_HANDLE,
            secNode->lightTreeBuffer ? secNode->lightTreeBuffer->getSize() : 0,
            secNode->mlAlbedoRoughnessImage ? secNode->mlAlbedoRoughnessImage->getImageView() : VK_NULL_HANDLE,
            secNode->mlSpecularMotionImage ? secNode->mlSpecularMotionImage->getImageView() : VK_NULL_HANDLE,
            secNode->mlDiffuseImage ? secNode->mlDiffuseImage->getImageView() : VK_NULL_HANDLE,
            secNode->mlSpecularImage ? secNode->mlSpecularImage->getImageView() : VK_NULL_HANDLE,
            secNode->instanceBuffer ? secNode->instanceBuffer->getBuffer() : VK_NULL_HANDLE,
            secNode->instanceBuffer ? secNode->instanceBuffer->getSize() : 0,
            activeCausticView,
            restirReservoirBuf,
            secNode->materialArchetypeBuffer ? secNode->materialArchetypeBuffer->getBuffer() : VK_NULL_HANDLE,
            secNode->materialArchetypeBuffer ? secNode->materialArchetypeBuffer->getSize() : 0,
            secNode->shadeMaterialBuffer ? secNode->shadeMaterialBuffer->getBuffer() : VK_NULL_HANDLE,
            secNode->shadeMaterialBuffer ? secNode->shadeMaterialBuffer->getSize() : 0,
            restirHistoryReservoirBuf,
            x1CtxBuf ? x1CtxBuf->getBuffer() : VK_NULL_HANDLE,
            giCurrBuf ? giCurrBuf->getBuffer() : VK_NULL_HANDLE,
            giPrevBuf ? giPrevBuf->getBuffer() : VK_NULL_HANDLE,
            secNode->prevNormalDepthImage ? secNode->prevNormalDepthImage->getImageView() : VK_NULL_HANDLE
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
            m_workerBusy = true;
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
            m_workerBusy = false;
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

    // 1. Wait for previous execution using this slot to complete before recording.
    // Bounded wait: a wedged secondary GPU must not deadlock the worker thread forever.
    VkResult slotFenceRes = vkWaitForFences(device, 1, &node->renderFences[slot], VK_TRUE, 10'000'000'000ull /* 10 s */);
    if (slotFenceRes == VK_TIMEOUT) {
        Logger::error("Secondary GPU render fence timeout (10s) for slot {} — GPU may be hung. Aborting this frame's secondary work.", slot);
        return;
    } else if (slotFenceRes != VK_SUCCESS) {
        Logger::error("Secondary GPU vkWaitForFences failed for slot {} (VkResult: {})", slot, static_cast<int>(slotFenceRes));
        return;
    }

    // Read timestamp queries from completed execution of this slot (only if slot was previously executed)
    if (node->slotHasExecuted[slot]) {
        uint64_t slotTimestamps[4] = {0, 0, 0, 0};
        if (vkGetQueryPoolResults(device, node->queryPools[slot], 0, 4, sizeof(slotTimestamps), slotTimestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            if (slotTimestamps[1] > slotTimestamps[0]) {
                double rtMs = (slotTimestamps[1] - slotTimestamps[0]) * node->timestampPeriod * 1e-6;
                node->lastFrameTimeMs = rtMs;
            }
            if (slotTimestamps[3] > slotTimestamps[2]) {
                double copyMs = (slotTimestamps[3] - slotTimestamps[2]) * node->timestampPeriod * 1e-6;
                node->lastTransferTimeMs = copyMs;
            }
        }
        if (node->wavefrontPipeline && (getenv("PATHWAYS_PROFILE_WF") || m_config.benchmark)) {
            static int secProfCount = 0;
            if (++secProfCount == 10) {
                Logger::info("--- SECONDARY GPU (GPU 1) WAVEFRONT PROFILE ---");
                node->wavefrontPipeline->printProfilingBreakdown(slot, node->timestampPeriod, m_config.max_bounces);
            }
        }
    }
    vkResetFences(device, 1, &node->renderFences[slot]);

    if (m_config.enable_restir && node->restirManager && !node->restirManager->isInitialized()) {
        vkDeviceWaitIdle(device);
        uint32_t restirW = std::max(m_config.getRenderWidth(), node->accumTargets[0] ? node->accumTargets[0]->getWidth() : m_config.getRenderWidth());
        node->restirManager->resize(restirW, m_config.getRenderHeight());
        updateSecondaryWavefrontDescriptors(node);
    }

    ensureSecondaryMlGuides(node);
    ensureSecondaryCaustics(node);
    const bool needMl = needSecondaryMlGuides() &&
                        node->mlDiffuseImage && node->mlSpecularImage &&
                        node->mlAlbedoRoughnessImage && node->mlSpecularMotionImage;

    // Drain pending dynamic scene updates staged from the primary GPU before recording RT
    bool doInstanceSync = false;
    bool instanceRebuild = false;
    bool instanceUpdateMode = true;
    std::vector<InstanceGPU> localShaderInstances;
    std::vector<ASInstanceGPUData> localInputInstances;

    bool doLightSync = false;
    std::vector<LightGPU> localLights;
    std::vector<LightTreeNodeGPU> localLightTree;

    bool doVideoSync = false;
    std::vector<uint8_t> localVideoPixels;

    {
        std::lock_guard<std::mutex> syncLock(m_sceneSyncMutex);
        if (m_pendingInstanceSync) {
            doInstanceSync = true;
            instanceRebuild = m_pendingTlasRebuild;
            instanceUpdateMode = m_pendingTlasUpdateMode;
            localShaderInstances = std::move(m_stagedShaderInstances);
            localInputInstances = std::move(m_stagedInputInstances);
            m_pendingInstanceSync = false;
            m_pendingTlasRebuild = false;
        }
        if (m_pendingLightSync) {
            doLightSync = true;
            localLights = std::move(m_stagedLights);
            localLightTree = std::move(m_stagedLightTree);
            m_pendingLightSync = false;
        }
        if (m_pendingVideoFrame) {
            doVideoSync = true;
            localVideoPixels = std::move(m_stagedVideoPixels);
            m_pendingVideoFrame = false;
        }
    }

    // 2. Update Camera UBO for this slot
    CameraUniform uboSecCopy = packet.cameraUniform;
    uint32_t secSppLoop = packet.cameraUniform.spp;
    if (m_config.pipeline_type == PipelineType::Wavefront && node->wavefrontPipeline && node->wavefrontPipeline->getWarmupSppOverride() > 0u) {
        uboSecCopy.spp = node->wavefrontPipeline->getWarmupSppOverride();
    } else if (m_config.pipeline_type == PipelineType::Wavefront && packet.totalCompositeSpp > 0u) {
        uboSecCopy.spp = packet.totalCompositeSpp;
    }
    node->cameraUBOs[slot]->copyFrom(&uboSecCopy, sizeof(CameraUniform));

    // 3. Record secondary GPU commands
    VkCommandBuffer cmd = node->commandBuffers[slot];
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &beginInfo);

    vkCmdResetQueryPool(cmd, node->queryPools[slot], 0, 4);
    // Timestamp 0: RT Start
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, node->queryPools[slot], 0);

    // Apply GPU-timeline dynamic TLAS, dynamic light, and video billboard updates on Secondary GPU
    if (doInstanceSync && node->tlasUpdatePipeline && node->asManager && node->tlas) {
        node->tlasUpdatePipeline->syncFromCpuTransforms(localShaderInstances, localInputInstances, instanceRebuild);
        if (node->tlasUpdatePipeline->needsGpuUpdate()) {
            node->tlasUpdatePipeline->recordGpuTlasUpdate(cmd, slot, node->asManager.get(), node->tlas.get(), instanceUpdateMode);
        }
    }
    if (doLightSync && node->lightBuffer && !localLights.empty()) {
        VmaAllocator secAlloc = node->context->getAllocator();
        VkDeviceSize lightBytes = std::min<VkDeviceSize>(sizeof(LightGPU) * localLights.size(), node->lightBuffer->getSize());
        if (lightBytes > 0) {
            if (!node->lightStaging[slot] || node->lightStaging[slot]->getSize() < lightBytes) {
                node->lightStaging[slot] = std::make_unique<Buffer>(
                    secAlloc, std::max(lightBytes, node->lightBuffer->getSize()),
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
                );
            }
            node->lightStaging[slot]->copyFrom(localLights.data(), lightBytes);
            VkBufferCopy copyRegion{ 0, 0, lightBytes };
            vkCmdCopyBuffer(cmd, node->lightStaging[slot]->getBuffer(), node->lightBuffer->getBuffer(), 1, &copyRegion);
        }
        if (node->lightTreeBuffer && !localLightTree.empty()) {
            VkDeviceSize treeBytes = std::min<VkDeviceSize>(sizeof(LightTreeNodeGPU) * localLightTree.size(), node->lightTreeBuffer->getSize());
            if (treeBytes > 0) {
                if (!node->lightTreeStaging[slot] || node->lightTreeStaging[slot]->getSize() < treeBytes) {
                    node->lightTreeStaging[slot] = std::make_unique<Buffer>(
                        secAlloc, std::max(treeBytes, node->lightTreeBuffer->getSize()),
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
                    );
                }
                node->lightTreeStaging[slot]->copyFrom(localLightTree.data(), treeBytes);
                VkBufferCopy copyRegion{ 0, 0, treeBytes };
                vkCmdCopyBuffer(cmd, node->lightTreeStaging[slot]->getBuffer(), node->lightTreeBuffer->getBuffer(), 1, &copyRegion);
            }
        }
        VkMemoryBarrier2 lightBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        lightBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        lightBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        lightBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        lightBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo lightDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        lightDep.memoryBarrierCount = 1;
        lightDep.pMemoryBarriers = &lightBarrier;
        vkCmdPipelineBarrier2(cmd, &lightDep);
    }
    if (doVideoSync && !localVideoPixels.empty() && !node->sceneTextures.empty() && node->sceneTextures[0]) {
        VmaAllocator secAlloc = node->context->getAllocator();
        VkDeviceSize vidBytes = static_cast<VkDeviceSize>(localVideoPixels.size());
        if (!node->videoStagingBuffers[slot] || node->videoStagingBuffers[slot]->getSize() < vidBytes) {
            node->videoStagingBuffers[slot] = std::make_unique<Buffer>(
                secAlloc, vidBytes,
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
            );
        }
        node->sceneTextures[0]->updatePixelsAsync(cmd, *node->videoStagingBuffers[slot], localVideoPixels.data(), localVideoPixels.size());
    }

    uint32_t envBits = std::bit_cast<uint32_t>(packet.envMapIntensity);
    uint32_t fracBits = std::bit_cast<uint32_t>(packet.fractionalSpp);
    const uint32_t rtpAccumFlags = (packet.accumulateHistory & 1u) | (needMl ? 2u : 0u);
    uint32_t rtPushConstants[16] = {
        packet.numTriangles, packet.numSpheres, packet.numMaterials, packet.numLights,
        packet.tileOffsetX, packet.tileOffsetY, packet.tileWidth, packet.tileHeight,
        packet.useHardwareRT,
        packet.hasEnvMap,
        envBits,
        rtpAccumFlags,
        fracBits,
        packet.totalCompositeSpp,
        packet.numOpaqueTriangles, std::bit_cast<uint32_t>(m_config.indirect_clamp)
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
            uint32_t tileSize = (packet.tileOffsetY == 0u) ? 64u : packet.tileOffsetY;
            uint32_t numTilesX = (packet.tileWidth + tileSize - 1u) / tileSize;
            uint32_t maxTilesPerGpuX = (numTilesX + 1u) / 2u;
            dispatchWidth = maxTilesPerGpuX * tileSize;
            dispatchHeight = packet.tileHeight;
        }
    }

    const float projInv11 = std::abs(packet.cameraUniform.projInverse[1][1]);
    node->currentCameraFov = (projInv11 > 1e-6f) ? glm::degrees(2.0f * std::atan(projInv11)) : 45.0f;
    node->currentFrameIndex = packet.frameIndex;
    node->currentCameraMoved = packet.cameraMoved;
    node->currentFullScreenWidth = packet.tileWidth;
    node->currentCbTileSize = (packet.tileOffsetX == 2u && packet.tileOffsetY > 0u) ? packet.tileOffsetY : 0u;
    node->currentNumLights = packet.numLights;

    bool needNormals = (needMl ||
                        m_config.enable_restir ||
                        m_config.enable_caustics);
    bool needMvDepth = needNormals || (m_config.upscaler_mode == UpscalerMode::FSR);
    bool copyMvImage = (!packet.enableUpscaler && packet.tileOffsetX == 2u && node->motionVectorImage &&
                        (needNormals || (needMvDepth && m_config.accum_format == AccumFormat::R11G11B10_UFLOAT)));
    bool copyNdImage = (!packet.enableUpscaler && packet.tileOffsetX == 2u && node->normalDepthImage && needNormals);
    bool copyCbMlPlanes = (!packet.enableUpscaler && packet.tileOffsetX == 2u && needMl);
    bool copySpMlPlanes = (!packet.enableUpscaler && packet.tileOffsetX == 0u && needMl);
    bool packDepthInAlpha = (!packet.enableUpscaler && packet.tileOffsetX == 2u && needMvDepth && !copyMvImage);
    bool earlyGbufferCopy = false;

    if (m_config.pipeline_type == PipelineType::Wavefront && node->wavefrontPipeline) {
        VkClearColorValue clearVal = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        std::vector<VkImageMemoryBarrier2> clearBarriers;
        auto clearAndTrack = [&](Image* img) {
            if (!img) return;
            vkCmdClearColorImage(cmd, img->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearVal, 1, &clearRange);
            VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            b.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
            b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.image = img->getImage();
            b.subresourceRange = clearRange;
            clearBarriers.push_back(b);
        };
        if (packet.accumulateHistory == 0u) {
            clearAndTrack(node->accumTargets[slot].get());
        }
        if (needMl) {
            clearAndTrack(node->mlDiffuseImage.get());
            clearAndTrack(node->mlSpecularImage.get());
        }
        if (!clearBarriers.empty()) {
            VkDependencyInfo clearDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            clearDep.imageMemoryBarrierCount = static_cast<uint32_t>(clearBarriers.size());
            clearDep.pImageMemoryBarriers = clearBarriers.data();
            vkCmdPipelineBarrier2(cmd, &clearDep);
        }

        uint32_t reqCapacity = dispatchWidth * dispatchHeight;
        if (node->wavefrontPipeline->getMaxCapacity() < reqCapacity) {
            vkDeviceWaitIdle(device);
            node->wavefrontPipeline->resize(dispatchWidth, dispatchHeight, reqCapacity);
            updateSecondaryWavefrontDescriptors(node);
        }

        WavefrontSceneData wfSceneData{};
        wfSceneData.numTriangles = packet.numTriangles;
        wfSceneData.numSpheres = packet.numSpheres;
        wfSceneData.numMaterials = packet.numMaterials;
        wfSceneData.numLights = packet.numLights;
        wfSceneData.hasEnvMap = packet.hasEnvMap;
        wfSceneData.envMapIntensity = packet.envMapIntensity;
        wfSceneData.useHardwareRT = packet.useHardwareRT;
        wfSceneData.frameIndex = packet.cameraUniform.frameIndex;
        wfSceneData.useMorton = m_config.use_morton ? 1u : 0u;
        wfSceneData.accumulateHistory = packet.accumulateHistory;
        wfSceneData.sortMode = packet.sortMode;
        wfSceneData.numOpaqueTriangles = packet.numOpaqueTriangles;
        wfSceneData.secondarySortMode = static_cast<uint32_t>(m_config.secondary_sort_mode);
        wfSceneData.cameraFlags = packet.cameraUniform.flags;
        wfSceneData.enableNrc = false;
        wfSceneData.boundsMin = m_boundsMin;
        wfSceneData.boundsMax = m_boundsMax;
        wfSceneData.streamlineSecondaryShading = m_config.streamline_secondary_shading;
        wfSceneData.enableDistanceClamping = m_config.distance_clamping;
        wfSceneData.maxSecondaryRayDistance = m_config.max_secondary_distance;
        wfSceneData.indirectClamp = m_config.indirect_clamp;
        wfSceneData.enableTailMegakernel = m_config.enable_tail_megakernel;
        wfSceneData.tailMegakernelBounce = m_config.tail_megakernel_bounce;
        wfSceneData.deltaUnroll = m_config.delta_unroll;
        wfSceneData.inlineShadows = m_config.inline_primary_shadows;
        wfSceneData.earlyBounceCutoff = m_config.early_bounce_cutoff;
        wfSceneData.earlyBounceCutoffThreshold = m_config.early_bounce_cutoff_threshold;
        wfSceneData.restirMCap = m_config.restir_m_cap;
        wfSceneData.tileOffsetX = packet.tileOffsetX;
        wfSceneData.tileOffsetY = packet.tileOffsetY;
        wfSceneData.fullWidth = packet.tileWidth;
        wfSceneData.fullHeight = packet.tileHeight;
        wfSceneData.captureMlData = needMl ? 3u : 0u;
        wfSceneData.gbufferFlags = (1u << 31) |
                                   (needNormals ? 1u : 0u) |
                                   ((copyNdImage || packet.enableUpscaler) ? 2u : 0u) |
                                   ((copyMvImage || packet.enableUpscaler) ? 4u : 0u) |
                                   (packDepthInAlpha ? 8u : 0u);

        if (m_config.enable_caustics && node->causticsPipeline && m_hasDielectrics && packet.numLights > 0) {
            node->causticsPipeline->recordTrace(
                cmd, slot,
                packet.numTriangles, packet.numSpheres, packet.numMaterials, packet.numLights,
                packet.numOpaqueTriangles, packet.frameIndex,
                m_hasDielectrics, m_dielectricBoundsMin, m_dielectricBoundsMax
            );
        }

        if (m_config.enable_restir && node->restirManager) {
            node->restirManager->recordClearIfNeeded(cmd);
        }

        node->wavefrontPipeline->recordFrame(cmd, slot, dispatchWidth, dispatchHeight,
                                             secSppLoop, packet.cameraUniform.maxBounces, wfSceneData);
    } else {
        if (m_config.enable_caustics && node->causticsPipeline && m_hasDielectrics && packet.numLights > 0) {
            node->causticsPipeline->recordFullPass(
                cmd, slot,
                packet.numTriangles, packet.numSpheres, packet.numMaterials, packet.numLights,
                packet.numOpaqueTriangles, packet.frameIndex,
                node->normalDepthImage.get(),
                node->currentCameraFov,
                false,
                node->currentCameraMoved,
                m_hasDielectrics, m_dielectricBoundsMin, m_dielectricBoundsMax,
                node->currentFullScreenWidth,
                node->currentCbTileSize
            );
        }
        // Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline)
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, node->rtpKhrPipeline->getPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, node->rtpPipelineLayout, 0, 1, &node->rtDescSets[slot], 0, nullptr);
        VkShaderStageFlags rtpStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        vkCmdPushConstants(cmd, node->rtpPipelineLayout, rtpStages, 0, sizeof(rtPushConstants), rtPushConstants);

        node->rtpKhrPipeline->traceRays(cmd, dispatchWidth, dispatchHeight, 1);
    }



    if (packet.enableUpscaler && node->upscaler) {
        // Timestamp 1: RT End
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, node->queryPools[slot], 1);

        if (node->upscaler->getRenderWidth() != packet.renderWidth ||
            node->upscaler->getRenderHeight() != packet.renderHeight ||
            node->upscaler->getDisplayWidth() != packet.displayWidth ||
            node->upscaler->getDisplayHeight() != packet.displayHeight) {
            node->upscaler->resize(packet.renderWidth, packet.renderHeight, packet.displayWidth, packet.displayHeight);
        }

        UpscalerDispatchDesc descSec{};
        descSec.cmd = cmd;
        descSec.colorIn = node->accumTargets[slot]->getImageView();
        descSec.depthIn = node->normalDepthImage ? node->normalDepthImage->getImageView() : VK_NULL_HANDLE;
        descSec.motionVectorsIn = node->motionVectorImage ? node->motionVectorImage->getImageView() : VK_NULL_HANDLE;
        descSec.colorOut = node->upscaler->getOutputImage()->getImageView();
        descSec.renderWidth = packet.renderWidth;
        descSec.renderHeight = packet.renderHeight;
        descSec.displayWidth = packet.displayWidth;
        descSec.displayHeight = packet.displayHeight;
        descSec.jitterX = packet.jitterX;
        descSec.jitterY = packet.jitterY;
        descSec.enableSharpening = packet.enableSharpening;
        descSec.sharpness = packet.sharpness;
        descSec.resetHistory = packet.resetUpscalerHistory;
        descSec.cameraMoved = packet.cameraMoved;
        descSec.frameIndex = packet.frameIndex;
        descSec.totalSamples = packet.totalSamples;
        descSec.inputIsNormalized = false;

        node->upscaler->recordUpscale(descSec);

        // Transition upscaler output to TRANSFER_SRC_OPTIMAL
        node->upscaler->getOutputImage()->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
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
        copyRegion.imageExtent = { packet.displayWidth, packet.displayHeight, 1 };

        uint32_t sharedSlot = m_config.double_buffered_shared_mem ? slot : 0;
        VkBuffer targetBuffer = (m_transferMode == InterGpuTransferMode::P2P_Direct_BAR)
            ? m_p2pBufferSecondary[sharedSlot]
            : m_sharedBufferSecondary[sharedSlot];

        vkCmdCopyImageToBuffer(cmd, node->upscaler->getOutputImage()->getImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               targetBuffer, 1, &copyRegion);

        // Timestamp 3: Copy End
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, node->queryPools[slot], 3);

        node->upscaler->getOutputImage()->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    } else {
        // Timestamp 1: RT End
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, node->queryPools[slot], 1);

        // Batched transition of images to TRANSFER_SRC_OPTIMAL
        std::vector<VkImageMemoryBarrier2> toTransferBarriers;
        toTransferBarriers.reserve(7);

        VkImageSubresourceRange colorRange{};
        colorRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        colorRange.baseMipLevel = 0;
        colorRange.levelCount = 1;
        colorRange.baseArrayLayer = 0;
        colorRange.layerCount = 1;

        auto makeToTransferBarrier = [&](VkImage img) {
            VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = img;
            b.subresourceRange = colorRange;
            return b;
        };

        toTransferBarriers.push_back(makeToTransferBarrier(node->accumTargets[slot]->getImage()));
        if (copyMvImage && !earlyGbufferCopy) {
            toTransferBarriers.push_back(makeToTransferBarrier(node->motionVectorImage->getImage()));
        }
        if (copyNdImage && !earlyGbufferCopy) {
            toTransferBarriers.push_back(makeToTransferBarrier(node->normalDepthImage->getImage()));
        }
        if (copyCbMlPlanes) {
            toTransferBarriers.push_back(makeToTransferBarrier(node->mlDiffuseImage->getImage()));
            toTransferBarriers.push_back(makeToTransferBarrier(node->mlSpecularImage->getImage()));
            toTransferBarriers.push_back(makeToTransferBarrier(node->mlAlbedoRoughnessImage->getImage()));
            toTransferBarriers.push_back(makeToTransferBarrier(node->mlSpecularMotionImage->getImage()));
        } else if (copySpMlPlanes) {
            toTransferBarriers.push_back(makeToTransferBarrier(node->mlDiffuseImage->getImage()));
            toTransferBarriers.push_back(makeToTransferBarrier(node->mlSpecularImage->getImage()));
        }

        VkDependencyInfo toTransferDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toTransferDep.imageMemoryBarrierCount = static_cast<uint32_t>(toTransferBarriers.size());
        toTransferDep.pImageMemoryBarriers = toTransferBarriers.data();
        vkCmdPipelineBarrier2(cmd, &toTransferDep);

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
        VkBuffer targetBuffer = (m_transferMode == InterGpuTransferMode::P2P_Direct_BAR)
            ? m_p2pBufferSecondary[sharedSlot]
            : m_sharedBufferSecondary[sharedSlot];

        // Timestamp 2: Copy Start
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, node->queryPools[slot], 2);

        vkCmdCopyImageToBuffer(cmd, node->accumTargets[slot]->getImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               targetBuffer, 1, &copyRegion);

        uint32_t bpp = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? 8 :
                       (m_config.accum_format == AccumFormat::R11G11B10_UFLOAT) ? 4 : 16;
        VkDeviceSize radSize = static_cast<VkDeviceSize>(dispatchWidth) * dispatchHeight * bpp;
        VkDeviceSize radOffsetAligned = (radSize + 65535) & ~static_cast<VkDeviceSize>(65535);
        VkDeviceSize planeAligned = (static_cast<VkDeviceSize>(dispatchWidth) * dispatchHeight * 8 + 65535) & ~static_cast<VkDeviceSize>(65535);

        if (copyMvImage && !earlyGbufferCopy) {
            // Copy Motion Vectors (+ Primary Depth in .w)
            VkBufferImageCopy copyMv = copyRegion;
            copyMv.bufferOffset = radOffsetAligned;
            vkCmdCopyImageToBuffer(cmd, node->motionVectorImage->getImage(),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   targetBuffer, 1, &copyMv);

            if (copyNdImage) {
                // Copy Normal & Depth (only needed when normal-guided denoiser is active)
                VkBufferImageCopy copyNd = copyRegion;
                copyNd.bufferOffset = radOffsetAligned + planeAligned;
                vkCmdCopyImageToBuffer(cmd, node->normalDepthImage->getImage(),
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       targetBuffer, 1, &copyNd);
            }
        }

        if (copyCbMlPlanes) {
            Image* cbMlImgs[4] = {
                node->mlDiffuseImage.get(),
                node->mlSpecularImage.get(),
                node->mlAlbedoRoughnessImage.get(),
                node->mlSpecularMotionImage.get()
            };
            for (uint32_t p = 0; p < 4u; ++p) {
                VkBufferImageCopy copyMl = copyRegion;
                copyMl.bufferOffset = radOffsetAligned + static_cast<VkDeviceSize>(2u + p) * planeAligned;
                vkCmdCopyImageToBuffer(cmd, cbMlImgs[p]->getImage(),
                                       VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                       targetBuffer, 1, &copyMl);
            }
        } else if (copySpMlPlanes) {
            VkBufferImageCopy copyDiff = copyRegion;
            copyDiff.bufferOffset = radOffsetAligned;
            vkCmdCopyImageToBuffer(cmd, node->mlDiffuseImage->getImage(),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   targetBuffer, 1, &copyDiff);

            VkBufferImageCopy copySpec = copyRegion;
            copySpec.bufferOffset = radOffsetAligned + planeAligned;
            vkCmdCopyImageToBuffer(cmd, node->mlSpecularImage->getImage(),
                                   VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   targetBuffer, 1, &copySpec);
        }

        // Timestamp 3: Copy End
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TRANSFER_BIT, node->queryPools[slot], 3);

        // Batched transition of images back to GENERAL
        std::vector<VkImageMemoryBarrier2> toGeneralBarriers;
        toGeneralBarriers.reserve(7);

        auto makeToGeneralBarrier = [&](VkImage img) {
            VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            b.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            b.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = img;
            b.subresourceRange = colorRange;
            return b;
        };

        toGeneralBarriers.push_back(makeToGeneralBarrier(node->accumTargets[slot]->getImage()));
        if (copyMvImage) {
            toGeneralBarriers.push_back(makeToGeneralBarrier(node->motionVectorImage->getImage()));
        }
        if (copyNdImage) {
            toGeneralBarriers.push_back(makeToGeneralBarrier(node->normalDepthImage->getImage()));
        }
        if (copyCbMlPlanes) {
            toGeneralBarriers.push_back(makeToGeneralBarrier(node->mlDiffuseImage->getImage()));
            toGeneralBarriers.push_back(makeToGeneralBarrier(node->mlSpecularImage->getImage()));
            toGeneralBarriers.push_back(makeToGeneralBarrier(node->mlAlbedoRoughnessImage->getImage()));
            toGeneralBarriers.push_back(makeToGeneralBarrier(node->mlSpecularMotionImage->getImage()));
        } else if (copySpMlPlanes) {
            toGeneralBarriers.push_back(makeToGeneralBarrier(node->mlDiffuseImage->getImage()));
            toGeneralBarriers.push_back(makeToGeneralBarrier(node->mlSpecularImage->getImage()));
        }

        VkMemoryBarrier2 copyFlushBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        copyFlushBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
        copyFlushBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        copyFlushBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
        copyFlushBarrier.dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo toGeneralDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        toGeneralDep.memoryBarrierCount = 1;
        toGeneralDep.pMemoryBarriers = &copyFlushBarrier;
        toGeneralDep.imageMemoryBarrierCount = static_cast<uint32_t>(toGeneralBarriers.size());
        toGeneralDep.pImageMemoryBarriers = toGeneralBarriers.data();
        vkCmdPipelineBarrier2(cmd, &toGeneralDep);
    }

    if ((m_config.enable_restir || m_config.enable_caustics) && node->normalDepthImage && node->prevNormalDepthImage) {
        VkImageSubresourceRange ndRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        VkImageMemoryBarrier2 preCopyBarriers[2]{};
        preCopyBarriers[0] = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        preCopyBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        preCopyBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;
        preCopyBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        preCopyBarriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        preCopyBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        preCopyBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        preCopyBarriers[0].image = node->normalDepthImage->getImage();
        preCopyBarriers[0].subresourceRange = ndRange;

        preCopyBarriers[1] = { VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        preCopyBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        preCopyBarriers[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        preCopyBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        preCopyBarriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preCopyBarriers[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        preCopyBarriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        preCopyBarriers[1].image = node->prevNormalDepthImage->getImage();
        preCopyBarriers[1].subresourceRange = ndRange;

        VkDependencyInfo preCopyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        preCopyDep.imageMemoryBarrierCount = 2;
        preCopyDep.pImageMemoryBarriers = preCopyBarriers;
        vkCmdPipelineBarrier2(cmd, &preCopyDep);

        VkImageCopy ndCopy{};
        ndCopy.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        ndCopy.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        ndCopy.extent = { node->normalDepthImage->getWidth(), node->normalDepthImage->getHeight(), 1 };
        vkCmdCopyImage(cmd, node->normalDepthImage->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                       node->prevNormalDepthImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &ndCopy);

        VkImageMemoryBarrier2 postCopyBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        postCopyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        postCopyBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        postCopyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        postCopyBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        postCopyBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        postCopyBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        postCopyBarrier.image = node->prevNormalDepthImage->getImage();
        postCopyBarrier.subresourceRange = ndRange;

        VkDependencyInfo postCopyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        postCopyDep.imageMemoryBarrierCount = 1;
        postCopyDep.pImageMemoryBarriers = &postCopyBarrier;
        vkCmdPipelineBarrier2(cmd, &postCopyDep);
    }

    vkEndCommandBuffer(cmd);

    // 5. Submit secondary workload
    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = cmd;

    VkSemaphoreSubmitInfo sigInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    if (m_useCrossGpuSync) {
        sigInfo.semaphore = node->secTimelineSemaphore;
        sigInfo.value = packet.timelineValue;
        sigInfo.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
    }

    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
    if (m_useCrossGpuSync) {
        submitInfo.signalSemaphoreInfoCount = 1;
        submitInfo.pSignalSemaphoreInfos = &sigInfo;
    }
    vkQueueSubmit2(queue, 1, &submitInfo, node->renderFences[slot]);
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
                                         float fractionalSpp,
                                         void* dstHostPtr,
                                         size_t transferBytes,
                                         uint32_t totalCompositeSpp,
                                         uint32_t numOpaqueTriangles,
                                         bool enableUpscaler,
                                         uint32_t renderW,
                                         uint32_t renderH,
                                         uint32_t displayW,
                                         uint32_t displayH,
                                         glm::vec2 jitter,
                                         bool resetUpscaler,
                                         bool cameraMoved,
                                         uint32_t frameIndex,
                                         bool enableSharpening,
                                         float sharpness,
                                         uint32_t totalSamples,
                                         uint32_t sortMode) {
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
        m_pendingWork.fractionalSpp = fractionalSpp;
        m_pendingWork.dstHostPtr = dstHostPtr;
        m_pendingWork.transferBytes = transferBytes;
        m_pendingWork.totalCompositeSpp = totalCompositeSpp;
        m_pendingWork.numOpaqueTriangles = numOpaqueTriangles;
        m_pendingWork.enableUpscaler = enableUpscaler;
        m_pendingWork.renderWidth = renderW;
        m_pendingWork.renderHeight = renderH;
        m_pendingWork.displayWidth = displayW;
        m_pendingWork.displayHeight = displayH;
        m_pendingWork.jitterX = jitter.x;
        m_pendingWork.jitterY = jitter.y;
        m_pendingWork.resetUpscalerHistory = resetUpscaler;
        m_pendingWork.cameraMoved = cameraMoved;
        m_pendingWork.frameIndex = frameIndex;
        m_pendingWork.enableSharpening = enableSharpening;
        m_pendingWork.sharpness = sharpness;
        m_pendingWork.totalSamples = totalSamples;
        m_pendingWork.sortMode = sortMode;
        m_pendingWork.timelineValue = ++m_currentTimelineValue;
        m_pendingWork.valid = true;
        uint32_t slot = bufferSlot % GpuDeviceNode::NUM_IN_FLIGHT;
        m_waitingSlot = slot;
        m_workSubmitted = false;
        m_slotSubmitted[slot] = false;

        m_workCv.notify_one();
    }
}

void MultiGpuManager::syncAndTransfer(uint32_t slot, void* dstHostPtr, size_t byteSize) {
    if (!m_active || m_devices.empty()) return;

    if (!m_useCrossGpuSync) {
        throw std::runtime_error("MultiGpuManager::syncAndTransfer requires hardware cross-GPU semaphore synchronization under Vulkan 1.4 baseline.");
    }

    // Zero-wait GPU-to-GPU pipelining:
    // Synchronization is handled directly on GPU 0 via vkQueueSubmit2 wait on
    // primImportedTimelineSemaphore with value getCurrentTimelineValue().
    // No CPU blocking or per-frame FD export/import is required.
}

void MultiGpuManager::waitSecondarySlot(uint32_t slot) {
    if (!m_active || m_devices.empty()) return;
    uint32_t s = slot % GpuDeviceNode::NUM_IN_FLIGHT;

    // Wait until any pending work destined for this slot has been dequeued by the worker thread
    std::unique_lock<std::mutex> lock(m_workMutex);
    m_submitCv.wait(lock, [this, s]() {
        if (m_pendingWork.valid && (m_pendingWork.bufferSlot % GpuDeviceNode::NUM_IN_FLIGHT) == s) {
            return false;
        }
        return true;
    });
}

void MultiGpuManager::waitWorkerIdle() {
    if (!m_active || m_devices.empty()) return;
    std::unique_lock<std::mutex> lock(m_workMutex);
    m_submitCv.wait(lock, [this]() {
        return !m_pendingWork.valid && !m_workerBusy;
    });
}

void MultiGpuManager::resize(uint32_t width, uint32_t height) {
    if (!m_active || m_devices.empty()) return;

    waitWorkerIdle();

    m_config.width = width;
    m_config.height = height;

    for (auto& node : m_devices) {
        VkDevice secDevice = node->context->getDevice();
        VmaAllocator secAlloc = node->context->getAllocator();
        vkDeviceWaitIdle(secDevice);

        VkFormat accumFormat = (m_config.accum_format == AccumFormat::RGBA16_SFLOAT) ? VK_FORMAT_R16G16B16A16_SFLOAT :
                               (m_config.accum_format == AccumFormat::R11G11B10_UFLOAT) ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 : VK_FORMAT_R32G32B32A32_SFLOAT;
        uint32_t renderW = (m_config.upscaler_mode != UpscalerMode::None && m_config.render_scale < 1.0f)
            ? m_config.getRenderWidth() : width;
        uint32_t renderH = (m_config.upscaler_mode != UpscalerMode::None && m_config.render_scale < 1.0f)
            ? m_config.getRenderHeight() : height;
        uint32_t tileSize = (m_config.tile_size == 0u) ? 64u : m_config.tile_size;
        uint32_t numTilesX = (renderW + tileSize - 1u) / tileSize;
        uint32_t maxTilesPerGpuX = (numTilesX + 1u) / 2u;
        MultiGpuMode effectiveMode = m_mode;
        if (effectiveMode == MultiGpuMode::Auto) {
            effectiveMode = (m_config.spp > 1) ? MultiGpuMode::SampleParallel : MultiGpuMode::CheckerboardTile;
        }
        bool isCheckerboard = (effectiveMode == MultiGpuMode::CheckerboardTile || effectiveMode == MultiGpuMode::Off);
        uint32_t secWidth = isCheckerboard ? (maxTilesPerGpuX * tileSize) : std::max(renderW, maxTilesPerGpuX * tileSize);
        uint32_t secHeight = renderH;
        for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
            node->accumTargets[i] = std::make_unique<Image>(
                secDevice, secAlloc, secWidth, secHeight,
                accumFormat,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
            );
        }

        // Recreate secondary G-Buffer resources
        node->normalDepthImage = std::make_unique<Image>(secDevice, secAlloc, secWidth, secHeight,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);

        node->prevNormalDepthImage = std::make_unique<Image>(secDevice, secAlloc, secWidth, secHeight,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        node->motionVectorImage = std::make_unique<Image>(secDevice, secAlloc, secWidth, secHeight,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        node->causticImage = std::make_unique<Image>(secDevice, secAlloc, secWidth, secHeight,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);

        // Reset lazy ML guide images (reallocated on demand at new resolution by ensureSecondaryMlGuides)
        node->mlDiffuseImage.reset();
        node->mlSpecularImage.reset();
        node->mlAlbedoRoughnessImage.reset();
        node->mlSpecularMotionImage.reset();
        node->causticsPipeline.reset();

        // Transition secondary targets to GENERAL layout and clear accumulation
        vkResetCommandBuffer(node->commandBuffers[0], 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(node->commandBuffers[0], &beginInfo);

        for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
            node->accumTargets[i]->transitionLayout(
                node->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
            );
        }

        node->normalDepthImage->transitionLayout(
            node->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
        node->prevNormalDepthImage->transitionLayout(
            node->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        node->motionVectorImage->transitionLayout(
            node->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        VkClearColorValue clearColor = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
            vkCmdClearColorImage(node->commandBuffers[0], node->accumTargets[i]->getImage(),
                                 VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &clearRange);
        }
        vkCmdClearColorImage(node->commandBuffers[0], node->prevNormalDepthImage->getImage(),
                             VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &clearRange);

        node->causticImage->transitionLayout(
            node->commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
        vkCmdClearColorImage(node->commandBuffers[0], node->causticImage->getImage(),
                             VK_IMAGE_LAYOUT_GENERAL, &clearColor, 1, &clearRange);

        vkEndCommandBuffer(node->commandBuffers[0]);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = node->commandBuffers[0];

        VkSubmitInfo2 initSubmit{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        initSubmit.commandBufferInfoCount = 1;
        initSubmit.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(node->context->getGraphicsQueue(), 1, &initSubmit, VK_NULL_HANDLE);
        vkQueueWaitIdle(node->context->getGraphicsQueue());

        VkDeviceSize bufferSize = calculateSharedBufferSize(width, height);
        bool hasSharedBuf = (m_p2pBufferPrimary[0] != VK_NULL_HANDLE) || (m_sharedBufferPrimary[0] != VK_NULL_HANDLE);
        if (!hasSharedBuf || m_sharedBufferSize < bufferSize) {
            initSharedHostBuffer(bufferSize);
        }

        if (node->mlDiffuseImage &&
            (node->mlDiffuseImage->getWidth() != secWidth || node->mlDiffuseImage->getHeight() != secHeight)) {
            node->mlDiffuseImage.reset();
            node->mlSpecularImage.reset();
            node->mlAlbedoRoughnessImage.reset();
            node->mlSpecularMotionImage.reset();
        }

        VkDescriptorImageInfo normDepthInfo{ VK_NULL_HANDLE, node->normalDepthImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo mvInfo{ VK_NULL_HANDLE, node->motionVectorImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo causticInfo{ VK_NULL_HANDLE, node->causticImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        const bool hasValidMlGuides = (node->mlDiffuseImage && node->mlSpecularImage &&
                                       node->mlAlbedoRoughnessImage && node->mlSpecularMotionImage);
        VkDescriptorImageInfo albInfo{
            VK_NULL_HANDLE,
            hasValidMlGuides ? node->mlAlbedoRoughnessImage->getImageView() : node->normalDepthImage->getImageView(),
            VK_IMAGE_LAYOUT_GENERAL
        };
        VkDescriptorImageInfo smvInfo{
            VK_NULL_HANDLE,
            hasValidMlGuides ? node->mlSpecularMotionImage->getImageView() : node->normalDepthImage->getImageView(),
            VK_IMAGE_LAYOUT_GENERAL
        };
        VkDescriptorImageInfo diffInfo{
            VK_NULL_HANDLE,
            hasValidMlGuides ? node->mlDiffuseImage->getImageView() : node->normalDepthImage->getImageView(),
            VK_IMAGE_LAYOUT_GENERAL
        };
        VkDescriptorImageInfo specInfo{
            VK_NULL_HANDLE,
            hasValidMlGuides ? node->mlSpecularImage->getImageView() : node->normalDepthImage->getImageView(),
            VK_IMAGE_LAYOUT_GENERAL
        };

        for (uint32_t slot = 0; slot < GpuDeviceNode::NUM_IN_FLIGHT; ++slot) {
            VkDescriptorImageInfo accumImageInfo{};
            accumImageInfo.imageView = node->accumTargets[slot]->getImageView();
            accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

            std::vector<VkWriteDescriptorSet> writes = {
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normDepthInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mvInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 15, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &causticInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 26, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &albInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 27, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &smvInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 28, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &diffInfo, nullptr, nullptr },
                { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, node->rtDescSets[slot], 29, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &specInfo, nullptr, nullptr }
            };
            vkUpdateDescriptorSets(secDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);

            node->slotHasExecuted[slot] = false;
            node->slotFdReady[slot] = false;
            if (node->exportedFd[slot] >= 0) {
                closeFileDescriptor(node->exportedFd[slot]);
                node->exportedFd[slot] = -1;
            }
        }
        if (node->wavefrontPipeline) {
            node->wavefrontPipeline->resize(secWidth, secHeight);
            if (m_config.enable_restir && m_config.mgpu_mode != MultiGpuMode::Off) {
                if (!node->restirManager) {
                    node->restirManager = std::make_unique<ReSTIRManager>(secDevice, secAlloc);
                }
                node->restirManager->resize(std::max(renderW, secWidth), renderH);
            }
            updateSecondaryWavefrontDescriptors(node.get());
        }
        if (node->upscaler) {
            if (m_config.upscaler_mode == UpscalerMode::FSR && m_config.mgpu_upscale_mode == MgpuUpscaleMode::SampleBlend) {
                node->upscaler->setTargetVersion(m_config.fsr_version);
                node->upscaler->resize(renderW, renderH, width, height);
            } else {
                node->upscaler.reset();
            }
        }
        m_slotSubmitted = { false, false };
        m_workSubmitted = false;
    }

    Logger::info("MultiGpuManager resized secondary GPU targets (Display: {}x{})", width, height);
}

void MultiGpuManager::setConfig(const Config& config) {
    bool upscalerChanged = (config.upscaler_mode != m_config.upscaler_mode) ||
                           (config.render_scale != m_config.render_scale) ||
                           (config.fsr_version != m_config.fsr_version) ||
                           (config.mgpu_upscale_mode != m_config.mgpu_upscale_mode);
    bool formatChanged = (config.accum_format != m_config.accum_format);
    bool doubleBufChanged = (config.double_buffered_shared_mem != m_config.double_buffered_shared_mem);
    bool targetLayoutChanged = upscalerChanged || formatChanged ||
                               (config.mgpu_mode != m_config.mgpu_mode) ||
                               (config.tile_size != m_config.tile_size) ||
                               (config.spp != m_config.spp && m_mode == MultiGpuMode::Auto);
    m_config = config;
    if (!m_devices.empty()) {
        GpuDeviceNode* secNode = m_devices[0].get();
        if (targetLayoutChanged && secNode->accumTargets[0]) {
            uint32_t renderW = (m_config.upscaler_mode != UpscalerMode::None && m_config.render_scale < 1.0f)
                ? m_config.getRenderWidth() : m_config.width;
            uint32_t renderH = (m_config.upscaler_mode != UpscalerMode::None && m_config.render_scale < 1.0f)
                ? m_config.getRenderHeight() : m_config.height;
            uint32_t tileSize = (m_config.tile_size == 0u) ? 64u : m_config.tile_size;
            uint32_t numTilesX = (renderW + tileSize - 1u) / tileSize;
            uint32_t maxTilesPerGpuX = (numTilesX + 1u) / 2u;
            MultiGpuMode effectiveMode = m_mode;
            if (effectiveMode == MultiGpuMode::Auto) {
                effectiveMode = (m_config.spp > 1) ? MultiGpuMode::SampleParallel : MultiGpuMode::CheckerboardTile;
            }
            bool isCheckerboard = (effectiveMode == MultiGpuMode::CheckerboardTile || effectiveMode == MultiGpuMode::Off);
            uint32_t secWidth = isCheckerboard ? (maxTilesPerGpuX * tileSize) : std::max(renderW, maxTilesPerGpuX * tileSize);
            uint32_t secHeight = renderH;
            if (formatChanged ||
                secNode->accumTargets[0]->getWidth() != secWidth ||
                secNode->accumTargets[0]->getHeight() != secHeight) {
                resize(m_config.width, m_config.height);
            }
        }
        VkDeviceSize reqSharedSize = calculateSharedBufferSize(m_config.width, m_config.height);
        constexpr VkDeviceSize kShrinkAlign = 2u * 1024u * 1024u;
        VkDeviceSize alignedReq = (reqSharedSize + kShrinkAlign - 1u) & ~(kShrinkAlign - 1u);
        if (m_sharedBufferSize < reqSharedSize || m_sharedBufferSize > alignedReq * 2u || doubleBufChanged) {
            waitWorkerIdle();
            if (m_primaryContext && m_primaryContext->getDevice() != VK_NULL_HANDLE) {
                vkDeviceWaitIdle(m_primaryContext->getDevice());
            }
            vkDeviceWaitIdle(secNode->context->getDevice());
            initSharedHostBuffer(reqSharedSize);
        }
        if (m_config.enable_restir && m_config.mgpu_mode != MultiGpuMode::Off) {
            if (!secNode->restirManager) {
                secNode->restirManager = std::make_unique<ReSTIRManager>(secNode->context->getDevice(), secNode->context->getAllocator());
            }
            if (!secNode->restirManager->isInitialized()) {
                uint32_t secW = secNode->accumTargets[0] ? secNode->accumTargets[0]->getWidth() : m_config.getRenderWidth();
                secNode->restirManager->resize(std::max(m_config.getRenderWidth(), secW), m_config.getRenderHeight());
                updateSecondaryWavefrontDescriptors(secNode);
            }
        }
        if (upscalerChanged) {
            if (config.upscaler_mode == UpscalerMode::FSR && config.mgpu_upscale_mode == MgpuUpscaleMode::SampleBlend) {
                uint32_t renderW = config.getRenderWidth();
                uint32_t renderH = config.getRenderHeight();
                if (!secNode->upscaler) {
                    try {
                        auto prepCode = loadShaderSPIRV("ffx_sdk_prep.comp.spv");
                        VkFormat accumFormat = (config.accum_format == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT :
                                               (config.accum_format == AccumFormat::R11G11B10_UFLOAT) ? VK_FORMAT_B10G11R11_UFLOAT_PACK32 :
                                               VK_FORMAT_R16G16B16A16_SFLOAT;
                        secNode->upscaler = std::make_unique<FsrUpscaler>(
                            secNode->context->getDevice(),
                            secNode->context->getPhysicalDevice(),
                            secNode->context->getAllocator(),
                            renderW, renderH, config.width, config.height,
                            prepCode, accumFormat,
                            "[GPU 1 Secondary Peer]",
                            config.fsr_version
                        );
                        Logger::info("Secondary GPU: FSR Upscaler initialized on setConfig [GPU 1 Secondary Peer].");
                    } catch (const std::exception& e) {
                        Logger::warn("Failed to initialize secondary GPU FSR upscaler on setConfig: {}", e.what());
                        secNode->upscaler.reset();
                    }
                } else {
                    secNode->upscaler->setTargetVersion(config.fsr_version);
                    secNode->upscaler->resize(renderW, renderH, config.width, config.height);
                }
            } else {
                secNode->upscaler.reset();
            }
        }
    }
}

bool MultiGpuManager::loadScene(const SceneData& scene, const std::string& scenePath) {
    if (!scenePath.empty()) {
        m_config.scene_path = scenePath;
    }
    m_boundsMin = scene.boundsMin;
    m_boundsMax = scene.boundsMax;
    m_hasDielectrics = scene.hasDielectrics;
    m_dielectricBoundsMin = scene.dielectricBoundsMin;
    m_dielectricBoundsMax = scene.dielectricBoundsMax;
    {
        std::lock_guard<std::mutex> syncLock(m_sceneSyncMutex);
        m_pendingInstanceSync = false;
        m_pendingTlasRebuild = false;
        m_stagedShaderInstances.clear();
        m_stagedInputInstances.clear();
        m_pendingLightSync = false;
        m_stagedLights.clear();
        m_stagedLightTree.clear();
        m_pendingVideoFrame = false;
        m_stagedVideoPixels.clear();
    }
    if (m_devices.empty()) return false;
    auto& secNode = m_devices[0];
    if (!secNode || !secNode->context) return false;

    waitWorkerIdle();

    VkDevice secDevice = secNode->context->getDevice();
    VmaAllocator secAlloc = secNode->context->getAllocator();
    VkQueue secQueue = secNode->context->getGraphicsQueue();
    VkCommandPool secPool = secNode->commandPool;

    vkQueueWaitIdle(secQueue);

    // 1. Update scene buffers on secondary device (pure DEVICE_LOCAL VRAM with staging upload, 16-byte aligned)
    size_t numTris = scene.triangles.size();
    std::vector<glm::vec4> positions;
    positions.reserve(numTris * 3);
    std::vector<TriangleShadeGPU> shadeTriangles;
    shadeTriangles.reserve(numTris);

    for (const auto& tri : scene.triangles) {
        positions.push_back(glm::vec4(glm::vec3(tri.v0.position), 1.0f));
        positions.push_back(glm::vec4(glm::vec3(tri.v1.position), 1.0f));
        positions.push_back(glm::vec4(glm::vec3(tri.v2.position), 1.0f));

        shadeTriangles.push_back(createTriangleShadeGPU(tri));
    }

    VkDeviceSize posSize = std::max(sizeof(glm::vec4) * positions.size(), sizeof(glm::vec4) * 3);
    secNode->positionBuffer = std::make_unique<Buffer>(
        secAlloc, posSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (!positions.empty()) {
        uploadToDeviceBufferSec(*secNode, *secNode->positionBuffer, positions.data(), sizeof(glm::vec4) * positions.size());
    } else {
        glm::vec4 dummyVerts[3] = {
            glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
            glm::vec4(0.0f, 0.0f, 0.0f, 1.0f),
            glm::vec4(0.0f, 0.0f, 0.0f, 1.0f)
        };
        uploadToDeviceBufferSec(*secNode, *secNode->positionBuffer, dummyVerts, sizeof(dummyVerts));
    }

    VkDeviceSize triSize = std::max(sizeof(TriangleShadeGPU) * shadeTriangles.size(), sizeof(TriangleShadeGPU));
    secNode->triangleBuffer = std::make_unique<Buffer>(
        secAlloc, triSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (!shadeTriangles.empty()) {
        uploadToDeviceBufferSec(*secNode, *secNode->triangleBuffer, shadeTriangles.data(), sizeof(TriangleShadeGPU) * shadeTriangles.size());
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

    std::vector<uint32_t> matArchetypes(scene.materials.size());
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        matArchetypes[i] = computeMaterialArchetype(scene.materials[i]);
    }
    VkDeviceSize archSize = std::max(sizeof(uint32_t) * matArchetypes.size(), sizeof(uint32_t));
    secNode->materialArchetypeBuffer = std::make_unique<Buffer>(
        secAlloc, archSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!matArchetypes.empty()) {
        secNode->materialArchetypeBuffer->copyFrom(matArchetypes.data(), sizeof(uint32_t) * matArchetypes.size());
    }

    // Compact 64-byte Shading Material buffer (2 materials per 128B RDNA 4 vector cache line)
    std::vector<ShadeMaterialGPU> shadeMaterials(scene.materials.size());
    for (size_t i = 0; i < scene.materials.size(); ++i) {
        shadeMaterials[i] = createShadeMaterial(scene.materials[i]);
    }
    VkDeviceSize shadeMatSize = std::max(sizeof(ShadeMaterialGPU) * shadeMaterials.size(), sizeof(ShadeMaterialGPU));
    secNode->shadeMaterialBuffer = std::make_unique<Buffer>(
        secAlloc, shadeMatSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!shadeMaterials.empty()) {
        secNode->shadeMaterialBuffer->copyFrom(shadeMaterials.data(), sizeof(ShadeMaterialGPU) * shadeMaterials.size());
    }

    VkDeviceSize lightSize = std::max(sizeof(LightGPU) * scene.lights.size(), sizeof(LightGPU));
    secNode->lightBuffer = std::make_unique<Buffer>(
        secAlloc, lightSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.lights.empty()) {
        secNode->lightBuffer->copyFrom(scene.lights.data(), sizeof(LightGPU) * scene.lights.size());
    }

    VkDeviceSize lightTreeSize = std::max(
        sizeof(LightTreeNodeGPU) * scene.lightTreeNodes.size(),
        sizeof(LightTreeNodeGPU) * std::max<size_t>(1, 2 * scene.lights.size())
    );
    secNode->lightTreeBuffer = std::make_unique<Buffer>(
        secAlloc, lightTreeSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!scene.lightTreeNodes.empty()) {
        secNode->lightTreeBuffer->copyFrom(scene.lightTreeNodes.data(), sizeof(LightTreeNodeGPU) * scene.lightTreeNodes.size());
    }

    for (uint32_t i = 0; i < GpuDeviceNode::NUM_IN_FLIGHT; ++i) {
        secNode->lightStaging[i].reset();
        secNode->lightTreeStaging[i].reset();
        secNode->videoStagingBuffers[i].reset();
    }

    // 2. Rebuild secondary AS
    auto updateTlasSpv = loadShaderSPIRV("update_tlas_instances.comp.spv");
    createSecondaryAccelerationStructures(*secNode, scene, updateTlasSpv);
    secNode->positionBuffer.reset();

    // 3. Upload scene textures on secondary device
    secNode->sceneTextures.clear();
    for (const auto& texData : scene.textures) {
        if (texData.isHdr && !texData.hdrPixels.empty() && texData.width > 0 && texData.height > 0) {
            bool genMips = (texData.width >= 64 && texData.height >= 64);
            auto tex = Texture::createFromPixels(
                secDevice, secAlloc, secQueue, secPool,
                texData.width, texData.height,
                VK_FORMAT_R16G16B16A16_SFLOAT, texData.hdrPixels.data(),
                texData.hdrPixels.size() * sizeof(uint16_t), true, genMips
            );
            secNode->sceneTextures.push_back(std::move(tex));
        } else if (!texData.pixels.empty() && texData.width > 0 && texData.height > 0) {
            VkFormat fmt = texData.isSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
            bool genMips = (texData.width >= 64 && texData.height >= 64);
            auto tex = Texture::createFromPixels(
                secDevice, secAlloc, secQueue, secPool,
                texData.width, texData.height,
                fmt, texData.pixels.data(),
                texData.pixels.size(), false, genMips
            );
            secNode->sceneTextures.push_back(std::move(tex));
        } else {
            secNode->sceneTextures.push_back(Texture::createDummyWhite(secDevice, secAlloc, secQueue, secPool));
        }
    }

    // 3b. Unified HDRI sky dome synchronization on secondary GPU
    secNode->environmentMap = Texture::createSceneEnvironmentMap(
        secDevice, secAlloc, secQueue, secPool,
        m_config.hdri_path, m_config.scene_path, scene.domeLightHdriPath
    );

    // 4. Update secondary descriptors
    VkDescriptorBufferInfo triInfo{ secNode->triangleBuffer->getBuffer(), 0, secNode->triangleBuffer->getSize() };
    VkDescriptorBufferInfo sphereInfo{ secNode->sphereBuffer->getBuffer(), 0, secNode->sphereBuffer->getSize() };
    VkDescriptorBufferInfo matInfo{ secNode->materialBuffer->getBuffer(), 0, secNode->materialBuffer->getSize() };
    VkDescriptorBufferInfo lightInfo{ secNode->lightBuffer->getBuffer(), 0, secNode->lightBuffer->getSize() };
    Buffer* secLightTree = secNode->lightTreeBuffer ? secNode->lightTreeBuffer.get() : secNode->lightBuffer.get();
    Buffer* secInstances = secNode->instanceBuffer ? secNode->instanceBuffer.get() : secNode->triangleBuffer.get();
    VkDescriptorBufferInfo lightTreeInfo{ secLightTree->getBuffer(), 0, secLightTree->getSize() };
    VkDescriptorBufferInfo instanceInfo{ secInstances->getBuffer(), 0, secInstances->getSize() };

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

    VkDescriptorImageInfo normDepthInfo{};
    VkDescriptorImageInfo mvInfo{};
    if (secNode->normalDepthImage) {
        normDepthInfo = { VK_NULL_HANDLE, secNode->normalDepthImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    }
    if (secNode->motionVectorImage) {
        mvInfo = { VK_NULL_HANDLE, secNode->motionVectorImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    }
    VkDescriptorImageInfo blueNoiseInfo = secNode->blueNoiseTexture ? secNode->blueNoiseTexture->getDescriptorInfo() : secNode->dummyWhite->getDescriptorInfo();

    for (uint32_t slot = 0; slot < GpuDeviceNode::NUM_IN_FLIGHT; ++slot) {
        VkDescriptorImageInfo accumImageInfo{};
        accumImageInfo.imageView = secNode->accumTargets[slot]->getImageView();
        accumImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

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
        if (secNode->normalDepthImage) {
            writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normDepthInfo, nullptr, nullptr });
        }
        writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 13, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &blueNoiseInfo, nullptr, nullptr });
        if (secNode->motionVectorImage) {
            writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mvInfo, nullptr, nullptr });
        }
        VkDescriptorImageInfo causticInfo = (secNode->causticsPipeline && secNode->causticsPipeline->getFilteredCausticImage())
            ? VkDescriptorImageInfo{ VK_NULL_HANDLE, secNode->causticsPipeline->getFilteredCausticImage()->getImageView(), VK_IMAGE_LAYOUT_GENERAL }
            : (secNode->causticImage
                ? VkDescriptorImageInfo{ VK_NULL_HANDLE, secNode->causticImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL }
                : normDepthInfo);
        VkDescriptorImageInfo albInfo = secNode->mlAlbedoRoughnessImage
            ? VkDescriptorImageInfo{ VK_NULL_HANDLE, secNode->mlAlbedoRoughnessImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL }
            : normDepthInfo;
        VkDescriptorImageInfo smvInfo = secNode->mlSpecularMotionImage
            ? VkDescriptorImageInfo{ VK_NULL_HANDLE, secNode->mlSpecularMotionImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL }
            : normDepthInfo;
        VkDescriptorImageInfo diffInfo = secNode->mlDiffuseImage
            ? VkDescriptorImageInfo{ VK_NULL_HANDLE, secNode->mlDiffuseImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL }
            : normDepthInfo;
        VkDescriptorImageInfo specInfo = secNode->mlSpecularImage
            ? VkDescriptorImageInfo{ VK_NULL_HANDLE, secNode->mlSpecularImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL }
            : normDepthInfo;
        writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 15, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &causticInfo, nullptr, nullptr });
        writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 25, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightTreeInfo, nullptr });
        writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 26, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &albInfo, nullptr, nullptr });
        writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 27, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &smvInfo, nullptr, nullptr });
        writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 28, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &diffInfo, nullptr, nullptr });
        writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 29, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &specInfo, nullptr, nullptr });
        writes.push_back(VkWriteDescriptorSet{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, secNode->rtDescSets[slot], 30, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &instanceInfo, nullptr });
        vkUpdateDescriptorSets(secDevice, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
    if (secNode->causticsPipeline) {
        secNode->causticsPipeline->resetHistory();
        if (!m_hasDielectrics) {
            secNode->causticsPipeline.reset();
        } else {
            VkAccelerationStructureKHR secTlasHandle = secNode->tlas ? secNode->tlas->getHandle() : VK_NULL_HANDLE;
            secNode->causticsPipeline->updateDescriptors(
                secNode->triangleBuffer.get(),
                secNode->sphereBuffer.get(),
                secNode->materialBuffer.get(),
                secNode->lightBuffer.get(),
                secTlasHandle,
                secNode->sceneTextures,
                secNode->dummyWhite.get(),
                secNode->instanceBuffer.get(),
                secNode->cameraUBOs,
                secNode->normalDepthImage.get(),
                secNode->motionVectorImage.get(),
                secNode->prevNormalDepthImage.get()
            );
        }
    }
    if (secNode->wavefrontPipeline) {
        updateSecondaryWavefrontDescriptors(secNode.get());
    }
    Logger::info("Secondary GPU Node reloaded scene successfully ({} triangles, {} materials).", scene.triangles.size(), scene.materials.size());
    return true;
}

void MultiGpuManager::syncDynamicInstances(const std::vector<InstanceGPU>& shaderInstances,
                                           const std::vector<ASInstanceGPUData>& primaryInputInstances,
                                           bool triggerTlasRebuild,
                                           bool updateMode) {
    if (!m_active || m_devices.empty()) return;
    std::lock_guard<std::mutex> lock(m_sceneSyncMutex);
    m_stagedShaderInstances = shaderInstances;
    m_stagedInputInstances = primaryInputInstances;
    m_pendingTlasRebuild = m_pendingTlasRebuild || triggerTlasRebuild;
    m_pendingTlasUpdateMode = updateMode;
    m_pendingInstanceSync = true;
}

void MultiGpuManager::syncDynamicLights(const std::vector<LightGPU>& lights,
                                        const std::vector<LightTreeNodeGPU>& lightTree) {
    if (!m_active || m_devices.empty() || lights.empty()) return;
    std::lock_guard<std::mutex> lock(m_sceneSyncMutex);
    m_stagedLights = lights;
    m_stagedLightTree = lightTree;
    m_pendingLightSync = true;
}

void MultiGpuManager::stageVideoFrame(const uint8_t* rgbaPixels, size_t byteSize) {
    if (!m_active || m_devices.empty() || !rgbaPixels || byteSize == 0) return;
    std::lock_guard<std::mutex> lock(m_sceneSyncMutex);
    m_stagedVideoPixels.assign(rgbaPixels, rgbaPixels + byteSize);
    m_pendingVideoFrame = true;
}

} // namespace pathways

