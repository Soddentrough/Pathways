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

    if (!pfn_vkCreateAccelerationStructureKHR || !pfn_vkCmdBuildAccelerationStructuresKHR) {
        throw std::runtime_error("Failed to load Vulkan Ray Tracing KHR function pointers!");
    }
}

void AccelerationStructureManager::submitCommandBuffer(VkCommandBuffer cmd) {
    VkSubmitInfo submitInfo{};
    submitInfo.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &cmd;

    vkQueueSubmit(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(m_queue);
}

std::unique_ptr<AccelerationStructure> AccelerationStructureManager::buildBLAS(const std::vector<ASGeometryInput>& geometries) {
    std::vector<VkAccelerationStructureGeometryKHR> asGeometries;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> asBuildRanges;
    std::vector<uint32_t> maxPrimitiveCounts;

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

    VkAccelerationStructureBuildGeometryInfoKHR buildInfo{};
    buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
    buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
    buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    buildInfo.geometryCount = static_cast<uint32_t>(asGeometries.size());
    buildInfo.pGeometries = asGeometries.data();

    VkAccelerationStructureBuildSizesInfoKHR sizeInfo{};
    sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
    pfn_vkGetAccelerationStructureBuildSizesKHR(
        m_device, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
        &buildInfo, maxPrimitiveCounts.data(), &sizeInfo
    );

    // Allocate BLAS buffer with 256-byte alignment
    auto blasBuffer = std::make_unique<Buffer>(
        m_allocator, sizeInfo.accelerationStructureSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, 256
    );

    // Create BLAS handle
    VkAccelerationStructureCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
    createInfo.buffer = blasBuffer->getBuffer();
    createInfo.size = sizeInfo.accelerationStructureSize;
    createInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;

    VkAccelerationStructureKHR blasHandle;
    VkResult res = pfn_vkCreateAccelerationStructureKHR(m_device, &createInfo, nullptr, &blasHandle);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create BLAS handle!");
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
    buildInfo.dstAccelerationStructure = blasHandle;
    buildInfo.scratchData.deviceAddress = alignedScratch;

    // Build BLAS on GPU
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

    const VkAccelerationStructureBuildRangeInfoKHR* pRangeInfos = asBuildRanges.data();
    pfn_vkCmdBuildAccelerationStructuresKHR(cmd, 1, &buildInfo, &pRangeInfos);

    // Memory barrier
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

    vkEndCommandBuffer(cmd);
    auto tStart = std::chrono::steady_clock::now();
    submitCommandBuffer(cmd);
    auto tEnd = std::chrono::steady_clock::now();
    m_lastBlasBuildTimeMs = std::chrono::duration<double, std::milli>(tEnd - tStart).count();
    m_blasSizeKb = sizeInfo.accelerationStructureSize / 1024.0;
    m_blasTriangles = 0;
    for (const auto& g : geometries) {
        m_blasTriangles += g.triangleCount;
    }
    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);

    // Query device address
    VkAccelerationStructureDeviceAddressInfoKHR addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
    addressInfo.accelerationStructure = blasHandle;
    VkDeviceAddress blasAddr = pfn_vkGetAccelerationStructureDeviceAddressKHR(m_device, &addressInfo);

    auto result = std::make_unique<AccelerationStructure>(m_device, m_allocator);
    result->setHandle(blasHandle, blasAddr, std::move(blasBuffer));

    Logger::info("Built BLAS successfully (size: {:.2f} KB, address: 0x{:x}, time: {:.3f} ms, triangles: {})",
                 m_blasSizeKb, blasAddr, m_lastBlasBuildTimeMs, m_blasTriangles);
    return result;
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
