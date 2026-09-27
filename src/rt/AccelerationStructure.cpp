#include "rt/AccelerationStructure.hpp"
#include "core/Logger.hpp"
#include <stdexcept>
#include <cstring>
#include <chrono>

namespace pathways {

AccelerationStructure::AccelerationStructure(VkDevice device, VmaAllocator allocator)
    : m_device(device), m_allocator(allocator) {}

AccelerationStructure::~AccelerationStructure() {
    release();
}

AccelerationStructure::AccelerationStructure(AccelerationStructure&& other) noexcept {
    m_device = other.m_device;
    m_allocator = other.m_allocator;
    m_handle = other.m_handle;
    m_deviceAddress = other.m_deviceAddress;
    m_buffer = std::move(other.m_buffer);

    other.m_device = VK_NULL_HANDLE;
    other.m_allocator = VK_NULL_HANDLE;
    other.m_handle = VK_NULL_HANDLE;
    other.m_deviceAddress = 0;
}

AccelerationStructure& AccelerationStructure::operator=(AccelerationStructure&& other) noexcept {
    if (this != &other) {
        release();
        m_device = other.m_device;
        m_allocator = other.m_allocator;
        m_handle = other.m_handle;
        m_deviceAddress = other.m_deviceAddress;
        m_buffer = std::move(other.m_buffer);

        other.m_device = VK_NULL_HANDLE;
        other.m_allocator = VK_NULL_HANDLE;
        other.m_handle = VK_NULL_HANDLE;
        other.m_deviceAddress = 0;
    }
    return *this;
}

void AccelerationStructure::setHandle(VkAccelerationStructureKHR handle, VkDeviceAddress addr, std::unique_ptr<Buffer> buffer) {
    release();
    m_handle = handle;
    m_deviceAddress = addr;
    m_buffer = std::move(buffer);
}

void AccelerationStructure::release() {
    if (m_handle && m_device) {
        auto pfn = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(m_device, "vkDestroyAccelerationStructureKHR");
        if (pfn) {
            pfn(m_device, m_handle, nullptr);
        }
        m_handle = VK_NULL_HANDLE;
    }
    m_buffer.reset();
}

AccelerationStructureManager::AccelerationStructureManager(VkDevice device, VmaAllocator allocator, VkQueue queue, uint32_t queueFamily)
    : m_device(device), m_allocator(allocator), m_queue(queue), m_queueFamily(queueFamily) {

    loadFunctionPointers();

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    poolInfo.queueFamilyIndex = m_queueFamily;
    vkCreateCommandPool(m_device, &poolInfo, nullptr, &m_commandPool);
}

AccelerationStructureManager::~AccelerationStructureManager() {
    if (m_commandPool) {
        vkDestroyCommandPool(m_device, m_commandPool, nullptr);
    }
}

void AccelerationStructureManager::loadFunctionPointers() {
    pfn_vkCreateAccelerationStructureKHR = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(m_device, "vkCreateAccelerationStructureKHR");
    pfn_vkDestroyAccelerationStructureKHR = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(m_device, "vkDestroyAccelerationStructureKHR");
    pfn_vkGetAccelerationStructureBuildSizesKHR = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureBuildSizesKHR");
    pfn_vkCmdBuildAccelerationStructuresKHR = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(m_device, "vkCmdBuildAccelerationStructuresKHR");
    pfn_vkGetAccelerationStructureDeviceAddressKHR = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(m_device, "vkGetAccelerationStructureDeviceAddressKHR");
    pfn_vkCmdWriteAccelerationStructuresPropertiesKHR = (PFN_vkCmdWriteAccelerationStructuresPropertiesKHR)vkGetDeviceProcAddr(m_device, "vkCmdWriteAccelerationStructuresPropertiesKHR");
    pfn_vkCmdCopyAccelerationStructureKHR = (PFN_vkCmdCopyAccelerationStructureKHR)vkGetDeviceProcAddr(m_device, "vkCmdCopyAccelerationStructureKHR");

    if (!pfn_vkCreateAccelerationStructureKHR || !pfn_vkCmdBuildAccelerationStructuresKHR) {
        throw std::runtime_error("Failed to load Vulkan Ray Tracing KHR function pointers!");
    }
}

void AccelerationStructureManager::submitCommandBuffer(VkCommandBuffer cmd) {
    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = cmd;

    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;

    vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_queue);
}

std::vector<std::unique_ptr<AccelerationStructure>> AccelerationStructureManager::buildBLASBatch(const std::vector<std::vector<ASGeometryInput>>& geometriesList) {
    if (geometriesList.empty()) {
        return {};
    }

    const uint32_t numBlases = static_cast<uint32_t>(geometriesList.size());
    std::vector<std::vector<VkAccelerationStructureGeometryKHR>> allAsGeometries(numBlases);
    std::vector<std::vector<VkAccelerationStructureBuildRangeInfoKHR>> allAsBuildRanges(numBlases);
    std::vector<std::vector<uint32_t>> allMaxPrimitiveCounts(numBlases);
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> buildInfos(numBlases);
    std::vector<VkAccelerationStructureBuildSizesInfoKHR> sizeInfos(numBlases);

    for (uint32_t i = 0; i < numBlases; ++i) {
        const auto& geometries = geometriesList[i];
        auto& asGeometries = allAsGeometries[i];
        auto& asBuildRanges = allAsBuildRanges[i];
        auto& maxPrimitiveCounts = allMaxPrimitiveCounts[i];

        for (const auto& g : geometries) {
            VkAccelerationStructureGeometryKHR asGeom{};
            asGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
            asGeom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            asGeom.flags = g.isOpaque ? VK_GEOMETRY_OPAQUE_BIT_KHR : 0;

            asGeom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            asGeom.geometry.triangles.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            asGeom.geometry.triangles.vertexData.deviceAddress = g.vertexBufferAddress;
            asGeom.geometry.triangles.vertexStride = g.vertexStride;
            asGeom.geometry.triangles.maxVertex = g.vertexCount;
            if (g.indexBufferAddress != 0) {
                asGeom.geometry.triangles.indexType = g.indexType;
                asGeom.geometry.triangles.indexData.deviceAddress = g.indexBufferAddress;
            } else {
                asGeom.geometry.triangles.indexType = VK_INDEX_TYPE_NONE_KHR;
                asGeom.geometry.triangles.indexData.deviceAddress = 0;
            }

            asGeometries.push_back(asGeom);

            VkAccelerationStructureBuildRangeInfoKHR range{};
            range.primitiveCount = g.triangleCount;
            range.primitiveOffset = 0;
            range.firstVertex = 0;
            range.transformOffset = 0;
            asBuildRanges.push_back(range);

            maxPrimitiveCounts.push_back(g.triangleCount);
        }

        auto& buildInfo = buildInfos[i];
        buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
        buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_COMPACTION_BIT_KHR;
        buildInfo.geometryCount = static_cast<uint32_t>(asGeometries.size());
        buildInfo.pGeometries = asGeometries.data();

        auto& sizeInfo = sizeInfos[i];
        sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
        pfn_vkGetAccelerationStructureBuildSizesKHR(
            m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
            &buildInfo, maxPrimitiveCounts.data(), &sizeInfo
        );
    }

    // Allocate single shared scratch buffer with non-overlapping 256-byte aligned regions
    // per Vulkan spec VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03698 & 03710
    constexpr VkDeviceSize scratchAlignment = 256;
    std::vector<VkDeviceSize> scratchOffsets(numBlases, 0);
    VkDeviceSize totalScratchSize = 0;
    for (uint32_t i = 0; i < numBlases; ++i) {
        scratchOffsets[i] = totalScratchSize;
        VkDeviceSize alignedScratch = (sizeInfos[i].buildScratchSize + (scratchAlignment - 1)) & ~(scratchAlignment - 1);
        totalScratchSize += alignedScratch;
    }
    if (totalScratchSize == 0) {
        totalScratchSize = scratchAlignment;
    }

    Buffer scratchBuffer(
        m_allocator, totalScratchSize + scratchAlignment,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, scratchAlignment
    );

    VkDeviceAddress rawScratch = scratchBuffer.getDeviceAddress(m_device);
    VkDeviceAddress alignedScratchBase = (rawScratch + (scratchAlignment - 1)) & ~(scratchAlignment - 1);

    // Allocate initial uncompacted BLAS buffers and handles
    std::vector<std::unique_ptr<Buffer>> initialBlasBuffers(numBlases);
    std::vector<VkAccelerationStructureKHR> initialBlasHandles(numBlases);

    for (uint32_t i = 0; i < numBlases; ++i) {
        buildInfos[i].scratchData.deviceAddress = alignedScratchBase + scratchOffsets[i];

        initialBlasBuffers[i] = std::make_unique<Buffer>(
            m_allocator, sizeInfos[i].accelerationStructureSize,
            VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 256
        );

        VkAccelerationStructureCreateInfoKHR createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
        createInfo.buffer = initialBlasBuffers[i]->getBuffer();
        createInfo.size = sizeInfos[i].accelerationStructureSize;
        createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

        VkResult res = pfn_vkCreateAccelerationStructureKHR(m_device, &createInfo, nullptr, &initialBlasHandles[i]);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("Failed to create BLAS handle in batch build!");
        }
        buildInfos[i].dstAccelerationStructure = initialBlasHandles[i];
    }

    // Create query pool for batched compaction sizing queries
    VkQueryPool queryPool = VK_NULL_HANDLE;
    if (pfn_vkCmdWriteAccelerationStructuresPropertiesKHR && pfn_vkCmdCopyAccelerationStructureKHR) {
        VkQueryPoolCreateInfo qpInfo{};
        qpInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        qpInfo.queryType = VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR;
        qpInfo.queryCount = numBlases;
        if (vkCreateQueryPool(m_device, &qpInfo, nullptr, &queryPool) != VK_SUCCESS) {
            queryPool = VK_NULL_HANDLE;
        }
    }

    // Build all BLASes concurrently in a single command buffer
    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = m_commandPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(m_device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    if (queryPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, queryPool, 0, numBlases);
    }

    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pRangeInfosList(numBlases);
    for (uint32_t i = 0; i < numBlases; ++i) {
        pRangeInfosList[i] = allAsBuildRanges[i].data();
    }
    pfn_vkCmdBuildAccelerationStructuresKHR(cmd, numBlases, buildInfos.data(), pRangeInfosList.data());

    // Memory barrier: wait for all BLAS builds to complete
    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    if (queryPool != VK_NULL_HANDLE) {
        pfn_vkCmdWriteAccelerationStructuresPropertiesKHR(
            cmd, numBlases, initialBlasHandles.data(),
            VK_QUERY_TYPE_ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
            queryPool, 0
        );
    }

    vkEndCommandBuffer(cmd);
    auto tStart = std::chrono::steady_clock::now();
    submitCommandBuffer(cmd);

    // Query compacted sizes
    std::vector<VkDeviceSize> compactedSizes(numBlases, 0);
    if (queryPool != VK_NULL_HANDLE) {
        VkResult qRes = vkGetQueryPoolResults(
            m_device, queryPool, 0, numBlases,
            numBlases * sizeof(VkDeviceSize), compactedSizes.data(), sizeof(VkDeviceSize),
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT
        );
        if (qRes != VK_SUCCESS) {
            std::fill(compactedSizes.begin(), compactedSizes.end(), 0);
        }
    }

    // Allocate tightly-fitted compact buffers and handles
    std::vector<std::unique_ptr<Buffer>> finalBlasBuffers(numBlases);
    std::vector<VkAccelerationStructureKHR> finalBlasHandles(numBlases);
    std::vector<bool> isCompacted(numBlases, false);
    uint32_t numCompacted = 0;

    for (uint32_t i = 0; i < numBlases; ++i) {
        if (compactedSizes[i] > 0 && compactedSizes[i] < sizeInfos[i].accelerationStructureSize) {
            auto compactBuffer = std::make_unique<Buffer>(
                m_allocator, compactedSizes[i],
                VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 256
            );

            VkAccelerationStructureCreateInfoKHR compactCreateInfo{};
            compactCreateInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
            compactCreateInfo.buffer = compactBuffer->getBuffer();
            compactCreateInfo.size = compactedSizes[i];
            compactCreateInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

            VkAccelerationStructureKHR compactHandle = VK_NULL_HANDLE;
            VkResult resCompact = pfn_vkCreateAccelerationStructureKHR(m_device, &compactCreateInfo, nullptr, &compactHandle);
            if (resCompact == VK_SUCCESS) {
                finalBlasBuffers[i] = std::move(compactBuffer);
                finalBlasHandles[i] = compactHandle;
                isCompacted[i] = true;
                numCompacted++;
            } else {
                finalBlasBuffers[i] = std::move(initialBlasBuffers[i]);
                finalBlasHandles[i] = initialBlasHandles[i];
            }
        } else {
            finalBlasBuffers[i] = std::move(initialBlasBuffers[i]);
            finalBlasHandles[i] = initialBlasHandles[i];
        }
    }

    // Batch copy compaction if any BLAS can be compacted
    if (numCompacted > 0) {
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &beginInfo);

        for (uint32_t i = 0; i < numBlases; ++i) {
            if (isCompacted[i]) {
                VkCopyAccelerationStructureInfoKHR copyInfo{};
                copyInfo.sType = VK_STRUCTURE_TYPE_COPY_ACCELERATION_STRUCTURE_INFO_KHR;
                copyInfo.src = initialBlasHandles[i];
                copyInfo.dst = finalBlasHandles[i];
                copyInfo.mode = VK_COPY_ACCELERATION_STRUCTURE_MODE_COMPACT_KHR;

                pfn_vkCmdCopyAccelerationStructureKHR(cmd, &copyInfo);
            }
        }

        VkMemoryBarrier2 copyBarrier{};
        copyBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        copyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
        copyBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        copyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        copyBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

        VkDependencyInfo copyDep{};
        copyDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        copyDep.memoryBarrierCount = 1;
        copyDep.pMemoryBarriers = &copyBarrier;
        vkCmdPipelineBarrier2(cmd, &copyDep);

        vkEndCommandBuffer(cmd);
        submitCommandBuffer(cmd);

        // Destroy uncompacted BLASes that were replaced by compacted versions
        for (uint32_t i = 0; i < numBlases; ++i) {
            if (isCompacted[i]) {
                pfn_vkDestroyAccelerationStructureKHR(m_device, initialBlasHandles[i], nullptr);
            }
        }
    }

    if (queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_device, queryPool, nullptr);
    }
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);

    auto tEnd = std::chrono::steady_clock::now();
    double buildDurationMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    m_lastBlasBuildTimeMs += buildDurationMs;

    double totalUncompactedKb = 0.0;
    double totalFinalKb = 0.0;
    uint32_t totalBatchTriangles = 0;

    std::vector<std::unique_ptr<AccelerationStructure>> results(numBlases);
    for (uint32_t i = 0; i < numBlases; ++i) {
        VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
        addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
        addressInfo.accelerationStructure = finalBlasHandles[i];
        VkDeviceAddress blasAddr = pfn_vkGetAccelerationStructureDeviceAddressKHR(m_device, &addressInfo);

        results[i] = std::make_unique<AccelerationStructure>(m_device, m_allocator);
        results[i]->setHandle(finalBlasHandles[i], blasAddr, std::move(finalBlasBuffers[i]));

        double uncompactedKb = sizeInfos[i].accelerationStructureSize / 1024.0;
        double finalKb = (isCompacted[i] ? compactedSizes[i] : sizeInfos[i].accelerationStructureSize) / 1024.0;
        totalUncompactedKb += uncompactedKb;
        totalFinalKb += finalKb;

        m_uncompactedBlasSizeKb += uncompactedKb;
        m_blasSizeKb += finalKb;
        m_blasCompacted = m_blasCompacted && isCompacted[i];

        for (const auto& g : geometriesList[i]) {
            m_blasTriangles += g.triangleCount;
            totalBatchTriangles += g.triangleCount;
        }
    }

    if (numCompacted > 0) {
        double ratio = (1.0 - (totalFinalKb / totalUncompactedKb)) * 100.0;
        Logger::info("Batch built & compacted {} BLASes successfully (uncompacted: {:.2f} KB -> compacted: {:.2f} KB, -{:.1f}%, time: {:.3f} ms, triangles: {}) [Total Scene BLAS: {:.2f} KB]",
                     numBlases, totalUncompactedKb, totalFinalKb, ratio, buildDurationMs, totalBatchTriangles, m_blasSizeKb);
    } else {
        Logger::info("Batch built {} BLASes successfully (size: {:.2f} KB, time: {:.3f} ms, triangles: {}) [Total Scene BLAS: {:.2f} KB]",
                     numBlases, totalFinalKb, buildDurationMs, totalBatchTriangles, m_blasSizeKb);
    }

    return results;
}

std::unique_ptr<AccelerationStructure> AccelerationStructureManager::buildBLAS(const std::vector<ASGeometryInput>& geometries) {
    auto results = buildBLASBatch({ geometries });
    if (results.empty()) {
        return nullptr;
    }
    return std::move(results[0]);
}

std::unique_ptr<AccelerationStructure> AccelerationStructureManager::buildTLAS(const std::vector<ASInstanceInput>& instances) {
    std::vector<VkAccelerationStructureInstanceKHR> vkInstances;
    for (const auto& inst : instances) {
        VkAccelerationStructureInstanceKHR vkInst{};
        VkTransformMatrixKHR vkTransform{};
        for (int r = 0; r < 3; ++r) {
            for (int c = 0; c < 4; ++c) {
                vkTransform.matrix[r][c] = inst.transform[c][r];
            }
        }
        vkInst.transform = vkTransform;

        vkInst.instanceCustomIndex = inst.customIndex;
        vkInst.mask = inst.mask;
        vkInst.instanceShaderBindingTableRecordOffset = inst.hitGroupId;
        vkInst.flags = inst.flags;
        vkInst.accelerationStructureReference = inst.blasAddress;

        vkInstances.push_back(vkInst);
    }

    // Upload instance buffer (64-byte alignment)
    VkDeviceSize instanceBufferSize = sizeof(VkAccelerationStructureInstanceKHR) * vkInstances.size();
    Buffer instanceBuffer(
        m_allocator, instanceBufferSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        64
    );
    instanceBuffer.copyFrom(vkInstances.data(), instanceBufferSize);

    VkAccelerationStructureGeometryKHR tlasGeom{};
    tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasGeom.geometry.instances.data.deviceAddress = instanceBuffer.getDeviceAddress(m_device);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &tlasGeom;

    uint32_t primitiveCount = static_cast<uint32_t>(vkInstances.size());
    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pfn_vkGetAccelerationStructureBuildSizesKHR(
        m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &primitiveCount, &sizeInfo
    );

    // Allocate TLAS buffer with 256-byte alignment
    auto tlasBuffer = std::make_unique<Buffer>(
        m_allocator, sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 256
    );

    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = tlasBuffer->getBuffer();
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    VkAccelerationStructureKHR tlasHandle;
    VkResult res = pfn_vkCreateAccelerationStructureKHR(m_device, &createInfo, nullptr, &tlasHandle);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS handle!");
    }

    // Allocate Scratch buffer with 256-byte alignment per Vulkan spec VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03710
    constexpr VkDeviceSize scratchAlignment = 256;
    Buffer scratchBuffer(
        m_allocator, sizeInfo.buildScratchSize + scratchAlignment,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, scratchAlignment
    );

    VkDeviceAddress rawScratch = scratchBuffer.getDeviceAddress(m_device);
    VkDeviceAddress alignedScratch = (rawScratch + (scratchAlignment - 1)) & ~(scratchAlignment - 1);
    buildInfo.dstAccelerationStructure = tlasHandle;
    buildInfo.scratchData.deviceAddress = alignedScratch;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = primitiveCount;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex = 0;
    rangeInfo.transformOffset = 0;

    VkCommandBufferAllocateInfo cmdAlloc{};
    cmdAlloc.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    cmdAlloc.commandPool = m_commandPool;
    cmdAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAlloc.commandBufferCount = 1;

    VkCommandBuffer cmd;
    vkAllocateCommandBuffers(m_device, &cmdAlloc, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;
    pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);

    VkMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    barrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo depInfo{};
    depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);

    vkEndCommandBuffer(cmd);
    auto tStart = std::chrono::steady_clock::now();
    submitCommandBuffer(cmd);
    auto tEnd = std::chrono::steady_clock::now();
    m_lastTlasBuildTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    m_tlasSizeKb = sizeInfo.accelerationStructureSize / 1024.0;
    m_tlasInstances = primitiveCount;
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = tlasHandle;
    VkDeviceAddress tlasAddr = pfn_vkGetAccelerationStructureDeviceAddressKHR(m_device, &addressInfo);

    auto result = std::make_unique<AccelerationStructure>(m_device, m_allocator);
    result->setHandle(tlasHandle, tlasAddr, std::move(tlasBuffer));

    Logger::info("Built TLAS successfully (size: {:.2f} KB, address: 0x{:x}, instances: {}, time: {:.3f} ms)",
                 m_tlasSizeKb, tlasAddr, primitiveCount, m_lastTlasBuildTimeMs);
    return result;
}

VkAccelerationStructureBuildSizesInfoKHR AccelerationStructureManager::getTLASBuildSizes(uint32_t instanceCount) {
    VkAccelerationStructureGeometryKHR tlasGeom{};
    tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasGeom.geometry.instances.data.deviceAddress = 0;

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &tlasGeom;

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pfn_vkGetAccelerationStructureBuildSizesKHR(
        m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, &instanceCount, &sizeInfo
    );
    return sizeInfo;
}

std::unique_ptr<AccelerationStructure> AccelerationStructureManager::createTLAS(uint32_t instanceCount) {
    auto sizeInfo = getTLASBuildSizes(instanceCount);

    auto tlasBuffer = std::make_unique<Buffer>(
        m_allocator, sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 256
    );

    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = tlasBuffer->getBuffer();
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;

    VkAccelerationStructureKHR tlasHandle;
    VkResult res = pfn_vkCreateAccelerationStructureKHR(m_device, &createInfo, nullptr, &tlasHandle);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create GPU TLAS handle!");
    }

    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = tlasHandle;
    VkDeviceAddress tlasAddr = pfn_vkGetAccelerationStructureDeviceAddressKHR(m_device, &addressInfo);

    auto result = std::make_unique<AccelerationStructure>(m_device, m_allocator);
    result->setHandle(tlasHandle, tlasAddr, std::move(tlasBuffer));
    return result;
}

void AccelerationStructureManager::recordBuildTLAS(VkCommandBuffer cmd,
                                                   Buffer* instanceBuffer,
                                                   uint32_t instanceCount,
                                                   Buffer* scratchBuffer,
                                                   AccelerationStructure* dstTlas,
                                                   bool updateMode) {
    if (!instanceBuffer || !scratchBuffer || !dstTlas || instanceCount == 0) return;

    // 1. Pipeline barrier: Wait for GPU instance updates (compute shader) to write out instances
    VkBufferMemoryBarrier2 instBarrier{};
    instBarrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2;
    instBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    instBarrier.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
    instBarrier.dstStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    instBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;
    instBarrier.buffer = instanceBuffer->getBuffer();
    instBarrier.offset = 0;
    instBarrier.size = VK_WHOLE_SIZE;

    VkDependencyInfo preDep{};
    preDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    preDep.bufferMemoryBarrierCount = 1;
    preDep.pBufferMemoryBarriers = &instBarrier;
    vkCmdPipelineBarrier2(cmd, &preDep);

    // 2. Geometry specification
    VkAccelerationStructureGeometryKHR tlasGeom{};
    tlasGeom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
    tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeom.geometry.instances.arrayOfPointers = VK_FALSE;
    tlasGeom.geometry.instances.data.deviceAddress = instanceBuffer->getDeviceAddress(m_device);

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
    buildInfo.mode = updateMode ? VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR : VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    buildInfo.srcAccelerationStructure = updateMode ? dstTlas->getHandle() : VK_NULL_HANDLE;
    buildInfo.dstAccelerationStructure = dstTlas->getHandle();
    buildInfo.geometryCount = 1;
    buildInfo.pGeometries = &tlasGeom;

    constexpr VkDeviceSize scratchAlignment = 256;
    VkDeviceAddress rawScratch = scratchBuffer->getDeviceAddress(m_device);
    VkDeviceAddress alignedScratch = (rawScratch + (scratchAlignment - 1)) & ~(scratchAlignment - 1);
    buildInfo.scratchData.deviceAddress = alignedScratch;

    VkAccelerationStructureBuildRangeInfoKHR rangeInfo{};
    rangeInfo.primitiveCount = instanceCount;
    rangeInfo.primitiveOffset = 0;
    rangeInfo.firstVertex = 0;
    rangeInfo.transformOffset = 0;
    const VkAccelerationStructureBuildRangeInfoKHR* pRange = &rangeInfo;

    pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRange);

    // 3. Post-barrier: Ensure TLAS build finishes before subsequent ray queries / compute
    VkMemoryBarrier2 postBarrier{};
    postBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
    postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR;
    postBarrier.srcAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
    postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR;

    VkDependencyInfo postDep{};
    postDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    postDep.memoryBarrierCount = 1;
    postDep.pMemoryBarriers = &postBarrier;
    vkCmdPipelineBarrier2(cmd, &postDep);
}

} // namespace pathways
