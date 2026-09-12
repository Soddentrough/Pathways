#include "rt/WavefrontPipeline.hpp"
#include "core/Logger.hpp"
#include <array>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace pathways {

WavefrontPipeline::WavefrontPipeline(VkDevice device, VmaAllocator allocator,
                                     uint32_t width, uint32_t height,
                                     uint32_t tileSize,
                                     const std::vector<char>& classifyCode,
                                     const std::vector<char>& intersectCode,
                                     const std::vector<char>& shadeCode,
                                     const std::vector<char>& shadowCode,
                                     const std::vector<char>& shadeDiffuseCode,
                                     const std::vector<char>& shadeDielectricCode,
                                     const std::vector<char>& shadeConductorCode,
                                     const std::vector<char>& shadeComplexCode,
                                     const std::vector<char>& shadeEmissiveCode,
                                     const std::vector<char>& shadePassthroughCode,
                                     const std::vector<char>& raySortCode,
                                     bool supportsExecutionSet)
    : m_device(device), m_allocator(allocator), m_width(width), m_height(height), m_tileSize(tileSize),
      m_supportsExecutionSet(supportsExecutionSet) {

    if (m_tileSize > 0) {
        m_maxCapacity = std::min(m_tileSize * m_tileSize, m_width * m_height);
    } else {
        m_maxCapacity = m_width * m_height;
    }

    createDescriptorLayout();
    allocateDescriptorSets();
    allocateQueues(m_maxCapacity);
    updateQueueDescriptors();
    createPipelines(classifyCode, intersectCode, shadeCode, shadowCode,
                    shadeDiffuseCode, shadeDielectricCode, shadeConductorCode, shadeComplexCode,
                    shadeEmissiveCode, shadePassthroughCode, raySortCode);

    VkQueryPoolCreateInfo qpInfo{};
    qpInfo.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
    qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpInfo.queryCount = MAX_WAVEFRONT_TIMESTAMP_QUERIES;
    for (int i = 0; i < 2; ++i) {
        vkCreateQueryPool(m_device, &qpInfo, nullptr, &m_queryPools[i]);
    }

    Logger::info("Pure WavefrontPipeline created successfully (tileSize: {}, capacity: {} rays, Wave32 mode, DGC enabled, Material Pipelines: {}, RaySort: {}).",
                 m_tileSize, m_maxCapacity, (m_shadeDiffusePipeline != VK_NULL_HANDLE ? "enabled" : "disabled"),
                 (m_raySortPipeline != VK_NULL_HANDLE ? "enabled" : "disabled"));
}

WavefrontPipeline::~WavefrontPipeline() {
    m_dgcManager.reset();

    for (int i = 0; i < 2; ++i) {
        if (m_queryPools[i]) vkDestroyQueryPool(m_device, m_queryPools[i], nullptr);
    }

    if (m_shadeDiffusePipeline) vkDestroyPipeline(m_device, m_shadeDiffusePipeline, nullptr);
    if (m_shadeDielectricPipeline) vkDestroyPipeline(m_device, m_shadeDielectricPipeline, nullptr);
    if (m_shadeConductorPipeline) vkDestroyPipeline(m_device, m_shadeConductorPipeline, nullptr);
    if (m_shadeComplexPipeline) vkDestroyPipeline(m_device, m_shadeComplexPipeline, nullptr);
    if (m_shadeEmissivePipeline) vkDestroyPipeline(m_device, m_shadeEmissivePipeline, nullptr);
    if (m_shadePassthroughPipeline) vkDestroyPipeline(m_device, m_shadePassthroughPipeline, nullptr);
    if (m_raySortPipeline) vkDestroyPipeline(m_device, m_raySortPipeline, nullptr);

    if (m_classifyPipeline) vkDestroyPipeline(m_device, m_classifyPipeline, nullptr);
    if (m_intersectPipeline) vkDestroyPipeline(m_device, m_intersectPipeline, nullptr);
    if (m_shadePipeline) vkDestroyPipeline(m_device, m_shadePipeline, nullptr);
    if (m_shadowPipeline) vkDestroyPipeline(m_device, m_shadowPipeline, nullptr);
    if (m_pipelineLayout) vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);

    if (m_descriptorPool) vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    if (m_descSetLayout) vkDestroyDescriptorSetLayout(m_device, m_descSetLayout, nullptr);
}

VkShaderModule WavefrontPipeline::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("WavefrontPipeline: Failed to create shader module!");
    }
    return shaderModule;
}

void WavefrontPipeline::createDescriptorLayout() {
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },             // uAccumImage
        { 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },            // CameraUBO
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },            // TrianglesBuffer
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },            // SpheresBuffer
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },            // MaterialsBuffer
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },            // LightsBuffer
        { 6, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },// topLevelAS
        { 7, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },    // environmentMap
        { 8, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_SCENE_TEXTURES, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },   // sceneTextures[512]
        { 9, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },            // InRayStateQueue
        { 10, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // OutRayStateQueue
        { 11, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // QueueCounters
        { 12, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // ShadowRayQueue
        { 13, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // IndirectCommand
        { 14, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // DGCStream
        { 15, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // InRayGeomQueue
        { 16, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // OutRayGeomQueue
        { 17, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // RayHitQueue
        { 18, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // MaterialIndexQueue
        { 19, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // SecondaryIndexQueue
        { 20, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // NRCQueryQueue
        { 21, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // NRCTrainQueue
        { 22, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },           // NRCCounters
        { 23, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },            // uMotionVectorImage
        { 24, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }             // uNormalDepthImage
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descSetLayout) != VK_SUCCESS) {
        throw std::runtime_error("WavefrontPipeline: Failed to create descriptor set layout!");
    }
}

void WavefrontPipeline::allocateDescriptorSets() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64 },
        { VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 16 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 256 },
        { VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 16 },
        { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 4096 }
    };

    VkDescriptorPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    poolInfo.maxSets = 16;

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("WavefrontPipeline: Failed to create descriptor pool!");
    }

    std::array<VkDescriptorSetLayout, 4> layouts = {
        m_descSetLayout, m_descSetLayout, m_descSetLayout, m_descSetLayout
    };
    VkDescriptorSetAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO;
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 4;
    allocInfo.pSetLayouts = layouts.data();

    std::array<VkDescriptorSet, 4> sets = { VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE };
    if (vkAllocateDescriptorSets(m_device, &allocInfo, sets.data()) != VK_SUCCESS) {
        throw std::runtime_error("WavefrontPipeline: Failed to allocate descriptor sets!");
    }
    m_descSetsEven[0] = sets[0];
    m_descSetsOdd[0]  = sets[1];
    m_descSetsEven[1] = sets[2];
    m_descSetsOdd[1]  = sets[3];
}

void WavefrontPipeline::allocateQueues(uint32_t capacity) {
    m_maxCapacity = capacity;
    VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                               VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                               VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                               VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT |
                               VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    // RayGeometry = 16 bytes (packed originPackedDir)
    VkDeviceSize geomQueueSize = static_cast<VkDeviceSize>(m_maxCapacity) * 16;
    m_rayGeomQueueA = std::make_unique<Buffer>(m_allocator, geomQueueSize, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_rayGeomQueueB = std::make_unique<Buffer>(m_allocator, geomQueueSize, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // RayState = 32 bytes
    VkDeviceSize stateQueueSize = static_cast<VkDeviceSize>(m_maxCapacity) * 32;
    m_rayStateQueueA = std::make_unique<Buffer>(m_allocator, stateQueueSize, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
    m_rayStateQueueB = std::make_unique<Buffer>(m_allocator, stateQueueSize, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // RayHit = 16 bytes
    VkDeviceSize hitQueueSize = static_cast<VkDeviceSize>(m_maxCapacity) * 16;
    m_rayHitQueue = std::make_unique<Buffer>(m_allocator, hitQueueSize, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // PackedShadowRay = 32 bytes (packed originDist + dirPixelRad)
    VkDeviceSize shadowQueueSize = static_cast<VkDeviceSize>(m_maxCapacity) * 32;
    m_shadowQueue = std::make_unique<Buffer>(m_allocator, shadowQueueSize, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // MaterialIndexQueue = 6 archetypes * 4 bytes * maxCapacity (Index-Based Material Queues)
    VkDeviceSize matIndexQueueSize = static_cast<VkDeviceSize>(m_maxCapacity) * 6 * sizeof(uint32_t);
    m_materialIndexQueue = std::make_unique<Buffer>(m_allocator, matIndexQueueSize, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // SecondaryIndexQueue = 8 octants * 4 bytes * maxCapacity (for Mode 1 directional binning & Mode 2 spatial sort)
    VkDeviceSize secIndexQueueSize = static_cast<VkDeviceSize>(m_maxCapacity) * 8 * sizeof(uint32_t);
    m_secondaryIndexQueue = std::make_unique<Buffer>(m_allocator, secIndexQueueSize, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // QueueCounters = 256 bytes (supports unified QueueCountersBuffer struct)
    m_queueCounters = std::make_unique<Buffer>(m_allocator, 256, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
    void* pCounters = m_queueCounters->map();
    if (pCounters) {
        std::memset(pCounters, 0, 256);
        m_queueCounters->unmap();
    }

    // IndirectArgs = 65536 bytes (supports up to 256 bounces * 16 dispatches * 16 bytes) - double-buffered per frame slot
    for (uint32_t slot = 0; slot < 2; ++slot) {
        m_indirectArgs[slot] = std::make_unique<Buffer>(m_allocator, 65536, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        void* pIndirect = m_indirectArgs[slot]->map();
        if (pIndirect) {
            std::memset(pIndirect, 0, 65536);
            m_indirectArgs[slot]->unmap();
        }
    }

    // DGCStream = 65536 bytes
    m_dgcStream = std::make_unique<Buffer>(m_allocator, 65536, usage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
}

void WavefrontPipeline::updateQueueDescriptors() {
    VkDescriptorBufferInfo stateAInfo{ m_rayStateQueueA->getBuffer(), 0, m_rayStateQueueA->getSize() };
    VkDescriptorBufferInfo stateBInfo{ m_rayStateQueueB->getBuffer(), 0, m_rayStateQueueB->getSize() };
    VkDescriptorBufferInfo geomAInfo{ m_rayGeomQueueA->getBuffer(), 0, m_rayGeomQueueA->getSize() };
    VkDescriptorBufferInfo geomBInfo{ m_rayGeomQueueB->getBuffer(), 0, m_rayGeomQueueB->getSize() };
    VkDescriptorBufferInfo hitInfo{ m_rayHitQueue->getBuffer(), 0, m_rayHitQueue->getSize() };
    VkDescriptorBufferInfo matIndexInfo{ m_materialIndexQueue->getBuffer(), 0, m_materialIndexQueue->getSize() };
    VkDescriptorBufferInfo secIndexInfo{ m_secondaryIndexQueue->getBuffer(), 0, m_secondaryIndexQueue->getSize() };
    VkDescriptorBufferInfo countersInfo{ m_queueCounters->getBuffer(), 0, m_queueCounters->getSize() };
    VkDescriptorBufferInfo shadowQueueInfo{ m_shadowQueue->getBuffer(), 0, m_shadowQueue->getSize() };
    VkDescriptorBufferInfo dgcStreamInfo{ m_dgcStream->getBuffer(), 0, m_dgcStream->getSize() };

    for (uint32_t slot = 0; slot < 2; ++slot) {
        VkDescriptorBufferInfo indirectInfo{ m_indirectArgs[slot]->getBuffer(), 0, m_indirectArgs[slot]->getSize() };

        // Even sets: InState = A, OutState = B, InGeom = A, OutGeom = B, Hit = Hit, MatIdx = MatIdx, SecIdx = SecIdx
        std::vector<VkWriteDescriptorSet> writesEven = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsEven[slot], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &stateAInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsEven[slot], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &stateBInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsEven[slot], 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &countersInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsEven[slot], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &shadowQueueInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsEven[slot], 13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &indirectInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsEven[slot], 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dgcStreamInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsEven[slot], 15, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &geomAInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsEven[slot], 16, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &geomBInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsEven[slot], 17, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hitInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsEven[slot], 18, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matIndexInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsEven[slot], 19, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secIndexInfo, nullptr }
        };
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writesEven.size()), writesEven.data(), 0, nullptr);

        // Odd sets: InState = B, OutState = A, InGeom = B, OutGeom = A, Hit = Hit, MatIdx = MatIdx, SecIdx = SecIdx
        std::vector<VkWriteDescriptorSet> writesOdd = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsOdd[slot], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &stateBInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsOdd[slot], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &stateAInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsOdd[slot], 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &countersInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsOdd[slot], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &shadowQueueInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsOdd[slot], 13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &indirectInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsOdd[slot], 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &dgcStreamInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsOdd[slot], 15, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &geomBInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsOdd[slot], 16, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &geomAInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsOdd[slot], 17, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hitInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsOdd[slot], 18, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matIndexInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSetsOdd[slot], 19, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secIndexInfo, nullptr }
        };
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writesOdd.size()), writesOdd.data(), 0, nullptr);
    }
}

void WavefrontPipeline::updateSceneDescriptors(uint32_t frameSlot,
                                             VkImageView accumImageView,
                                             VkBuffer cameraUBO,
                                             VkBuffer triangleBuffer, VkDeviceSize triSize,
                                             VkBuffer sphereBuffer, VkDeviceSize sphereSize,
                                             VkBuffer matBuffer, VkDeviceSize matSize,
                                             VkBuffer lightBuffer, VkDeviceSize lightSize,
                                             VkAccelerationStructureKHR tlas,
                                              VkDescriptorImageInfo envMapInfo,
                                              const std::vector<VkDescriptorImageInfo>& sceneTexInfos,
                                               VkBuffer nrcQueryBuffer,
                                               VkBuffer nrcTrainBuffer,
                                               VkBuffer nrcCountersBuffer,
                                               VkImageView motionVectorImageView,
                                               VkImageView normalDepthImageView) {
    if (frameSlot >= 2) frameSlot = 0;

    VkDescriptorImageInfo accumImageInfo{ VK_NULL_HANDLE, accumImageView, VK_IMAGE_LAYOUT_GENERAL };
    VkImageView mvView = (motionVectorImageView != VK_NULL_HANDLE) ? motionVectorImageView : accumImageView;
    VkDescriptorImageInfo mvImageInfo{ VK_NULL_HANDLE, mvView, VK_IMAGE_LAYOUT_GENERAL };
    VkImageView ndView = (normalDepthImageView != VK_NULL_HANDLE) ? normalDepthImageView : accumImageView;
    VkDescriptorImageInfo ndImageInfo{ VK_NULL_HANDLE, ndView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo camInfo{ cameraUBO, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo triInfo{ triangleBuffer, 0, triSize };
    VkDescriptorBufferInfo sphereInfo{ sphereBuffer, 0, sphereSize };
    VkDescriptorBufferInfo matInfo{ matBuffer, 0, matSize };
    VkDescriptorBufferInfo lightInfo{ lightBuffer, 0, lightSize };

    VkDescriptorBufferInfo nrcQueryInfo{ nrcQueryBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo nrcTrainInfo{ nrcTrainBuffer, 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo nrcCountersInfo{ nrcCountersBuffer, 0, VK_WHOLE_SIZE };

    VkWriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
    asInfo.accelerationStructureCount = 1;
    asInfo.pAccelerationStructures = &tlas;

    std::array<VkDescriptorSet, 2> targetSets = { m_descSetsEven[frameSlot], m_descSetsOdd[frameSlot] };

    for (VkDescriptorSet dset : targetSets) {
        std::vector<VkWriteDescriptorSet> writes = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumImageInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 1, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nullptr, &camInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &triInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &sphereInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &matInfo, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &lightInfo, nullptr }
        };

        if (tlas != VK_NULL_HANDLE) {
            VkWriteDescriptorSet asWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, &asInfo, dset, 6, 0, 1, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, nullptr, nullptr, nullptr };
            writes.push_back(asWrite);
        }

        if (envMapInfo.imageView != VK_NULL_HANDLE) {
            VkWriteDescriptorSet envWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &envMapInfo, nullptr, nullptr };
            writes.push_back(envWrite);
        }

        if (!sceneTexInfos.empty()) {
            VkWriteDescriptorSet texWrite{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 8, 0, static_cast<uint32_t>(sceneTexInfos.size()), VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, sceneTexInfos.data(), nullptr, nullptr };
            writes.push_back(texWrite);
        }

        if (nrcQueryBuffer != VK_NULL_HANDLE) {
            writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 20, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nrcQueryInfo, nullptr });
        }
        if (nrcTrainBuffer != VK_NULL_HANDLE) {
            writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 21, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nrcTrainInfo, nullptr });
        }
        if (nrcCountersBuffer != VK_NULL_HANDLE) {
            writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 22, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &nrcCountersInfo, nullptr });
        }

        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 23, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mvImageInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dset, 24, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ndImageInfo, nullptr, nullptr });

        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}


void WavefrontPipeline::resize(uint32_t width, uint32_t height, uint32_t tileSize) {
    if (m_width == width && m_height == height && m_tileSize == tileSize) return;
    m_width = width;
    m_height = height;
    m_tileSize = tileSize;
    if (m_tileSize > 0) {
        m_maxCapacity = std::min(m_tileSize * m_tileSize, m_width * m_height);
    } else {
        m_maxCapacity = m_width * m_height;
    }
    allocateQueues(m_maxCapacity);
    updateQueueDescriptors();
    Logger::info("WavefrontPipeline resized to {}x{} (tileSize: {}, capacity: {} rays).",
                 m_width, m_height, m_tileSize, m_maxCapacity);
}

void WavefrontPipeline::setTileSize(uint32_t tileSize) {
    if (m_tileSize == tileSize) return;
    resize(m_width, m_height, tileSize);
}

void WavefrontPipeline::createPipelines(const std::vector<char>& classifyCode,
                                        const std::vector<char>& intersectCode,
                                        const std::vector<char>& shadeCode,
                                        const std::vector<char>& shadowCode,
                                        const std::vector<char>& shadeDiffuseCode,
                                        const std::vector<char>& shadeDielectricCode,
                                        const std::vector<char>& shadeConductorCode,
                                        const std::vector<char>& shadeComplexCode,
                                        const std::vector<char>& shadeEmissiveCode,
                                        const std::vector<char>& shadePassthroughCode,
                                        const std::vector<char>& raySortCode) {
    // Pipeline Layout (128 bytes push constants for all stages)
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = 128;

    VkPipelineLayoutCreateInfo layoutInfo{};
    layoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_descSetLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pcRange;

    if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("WavefrontPipeline: Failed to create pipeline layout!");
    }

    // Explicit Wave32 execution mode on RDNA 4
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    auto buildComputePipeline = [&](const std::vector<char>& code, const char* name) -> VkPipeline {
        VkShaderModule mod = createShaderModule(code);
        VkPipelineShaderStageCreateInfo stage{};
        stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stage.module = mod;
        stage.pName = "main";
        stage.pNext = &subgroupSize32;

        VkPipelineCreateFlags2CreateInfo flags2Info{};
        flags2Info.sType = VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO;
        flags2Info.pNext = nullptr;
        flags2Info.flags = VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT;

        VkComputePipelineCreateInfo compInfo{};
        compInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        compInfo.pNext = &flags2Info;
        compInfo.stage = stage;
        compInfo.layout = m_pipelineLayout;

        VkPipeline pipe = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &compInfo, nullptr, &pipe) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, mod, nullptr);
            throw std::runtime_error(std::string("WavefrontPipeline: Failed to create compute pipeline ") + name);
        }
        vkDestroyShaderModule(m_device, mod, nullptr);
        return pipe;
    };

    m_classifyPipeline  = buildComputePipeline(classifyCode, "classify");
    m_intersectPipeline = buildComputePipeline(intersectCode, "intersect");
    m_shadePipeline     = buildComputePipeline(shadeCode, "shade");
    m_shadowPipeline    = buildComputePipeline(shadowCode, "shadow");

    if (!shadeDiffuseCode.empty()) {
        m_shadeDiffusePipeline = buildComputePipeline(shadeDiffuseCode, "shade_diffuse");
    }
    if (!shadeDielectricCode.empty()) {
        m_shadeDielectricPipeline = buildComputePipeline(shadeDielectricCode, "shade_dielectric");
    }
    if (!shadeConductorCode.empty()) {
        m_shadeConductorPipeline = buildComputePipeline(shadeConductorCode, "shade_conductor");
    }
    if (!shadeComplexCode.empty()) {
        m_shadeComplexPipeline = buildComputePipeline(shadeComplexCode, "shade_complex");
    }
    if (!shadeEmissiveCode.empty()) {
        m_shadeEmissivePipeline = buildComputePipeline(shadeEmissiveCode, "shade_emissive");
    }
    if (!shadePassthroughCode.empty()) {
        m_shadePassthroughPipeline = buildComputePipeline(shadePassthroughCode, "shade_passthrough");
    }
    if (!raySortCode.empty()) {
        m_raySortPipeline = buildComputePipeline(raySortCode, "wavefront_raysort");
    }

    // Initialize DGCManager with Execution Set support flag:
    // [0] classify, [1] intersect, [2] shade, [3] shadow
    // Note: Mesa RADV currently requires multi-dispatch fallback for compute execution sets;
    // can be explicitly enabled via PATHWAYS_ENABLE_DGC_EXECSET=1 or --dgc-execset.
    if (getenv("PATHWAYS_ENABLE_DGC_EXECSET") != nullptr) {
        m_supportsExecutionSet = true;
    }
    m_dgcManager = std::make_unique<DGCManager>(m_device, m_allocator, m_pipelineLayout, m_supportsExecutionSet);
    m_dgcManager->initExecutionSet({
        m_classifyPipeline,
        m_intersectPipeline,
        m_shadePipeline,
        m_shadowPipeline
    });

    if (m_shadeDiffusePipeline && m_shadeDielectricPipeline && m_shadeConductorPipeline && m_shadeComplexPipeline) {
        std::vector<VkPipeline> matPipes = {
            m_shadeDiffusePipeline,
            m_shadeDielectricPipeline,
            m_shadeConductorPipeline,
            m_shadeComplexPipeline
        };
        if (m_shadeEmissivePipeline) matPipes.push_back(m_shadeEmissivePipeline);
        if (m_shadePassthroughPipeline) matPipes.push_back(m_shadePassthroughPipeline);
        m_dgcManager->initMaterialExecutionSet(matPipes);
    }
}

static inline VkBufferMemoryBarrier2 makeBufferBarrier2(
    VkBuffer buffer,
    VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
    VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess,
    VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE)
{
    VkBufferMemoryBarrier2 b{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    b.srcStageMask = srcStage;
    b.srcAccessMask = srcAccess;
    b.dstStageMask = dstStage;
    b.dstAccessMask = dstAccess;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.buffer = buffer;
    b.offset = offset;
    b.size = size;
    return b;
}

void WavefrontPipeline::recordFrame(VkCommandBuffer cmd, uint32_t frameSlot, uint32_t width, uint32_t height,
                                    uint32_t spp, uint32_t maxBounces,
                                    const WavefrontSceneData& sceneData) {
    if (frameSlot >= 2) frameSlot = 0;
    m_hasRecordedSlot[frameSlot] = true;
    m_slotBounces[frameSlot] = maxBounces;

    uint32_t endQuery = std::min(3u + maxBounces * 6u, MAX_WAVEFRONT_TIMESTAMP_QUERIES - 1u);
    vkCmdResetQueryPool(cmd, m_queryPools[frameSlot], 0, MAX_WAVEFRONT_TIMESTAMP_QUERIES);
    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPools[frameSlot], 0);

    uint32_t ts = m_tileSize;
    m_sortMode = sceneData.sortMode;
    m_secondarySortMode = sceneData.secondarySortMode;
    uint32_t numTilesX = (ts > 0) ? (width + ts - 1) / ts : 1;
    uint32_t numTilesY = (ts > 0) ? (height + ts - 1) / ts : 1;

    vkCmdFillBuffer(cmd, m_indirectArgs[frameSlot]->getBuffer(), 0, VK_WHOLE_SIZE, 0);
    vkCmdFillBuffer(cmd, m_dgcStream->getBuffer(), 0, VK_WHOLE_SIZE, 0);

    VkMemoryBarrier2 clearBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    clearBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    clearBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    clearBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    clearBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo clearDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    clearDep.memoryBarrierCount = 1;
    clearDep.pMemoryBarriers = &clearBarrier;
    vkCmdPipelineBarrier2(cmd, &clearDep);

    // 3. Tile Loop
    for (uint32_t ty = 0; ty < numTilesY; ++ty) {
        for (uint32_t tx = 0; tx < numTilesX; ++tx) {
            uint32_t tileOffsetX = (ts > 0) ? tx * ts : 0;
            uint32_t tileOffsetY = (ts > 0) ? ty * ts : 0;
            uint32_t curTileW = (ts > 0) ? std::min(ts, width - tileOffsetX) : width;
            uint32_t curTileH = (ts > 0) ? std::min(ts, height - tileOffsetY) : height;

            bool isFirstTile = (tx == 0 && ty == 0);

            for (uint32_t sampleIdx = 0; sampleIdx < spp; ++sampleIdx) {
                bool shouldProfile = (isFirstTile && sampleIdx == 0);

                // 3a. Classify & Primary Ray Generation
                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_classifyPipeline);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descSetsOdd[frameSlot], 0, nullptr);

                uint32_t classifyPC[18] = {
                    sceneData.numTriangles,
                    sceneData.numSpheres,
                    sceneData.numMaterials,
                    sceneData.numLights,
                    width,
                    height,
                    sceneData.useMorton,
                    sceneData.hasEnvMap,
                    std::bit_cast<uint32_t>(sceneData.envMapIntensity),
                    m_maxCapacity,
                    sampleIdx,
                    sceneData.useHardwareRT,
                    tileOffsetX,
                    tileOffsetY,
                    ts,
                    ts,
                    sceneData.sortMode,
                    sceneData.numOpaqueTriangles
                };
                vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(classifyPC), classifyPC);
                if (shouldProfile) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPools[frameSlot], 1);
                vkCmdDispatch(cmd, (curTileW + 7) / 8, (curTileH + 3) / 4, 1);
                if (shouldProfile) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPools[frameSlot], 2);

                // Barrier: Classify -> Bounce 0 Shade (Indirect / DGC dispatch, scoped buffer barriers)
                std::array<VkBufferMemoryBarrier2, 8> c2sBarriers = {
                    makeBufferBarrier2(m_indirectArgs[frameSlot]->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT,
                        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT),
                    makeBufferBarrier2(m_dgcStream->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT, VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT),
                    makeBufferBarrier2(m_queueCounters->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
                    makeBufferBarrier2(m_rayGeomQueueA->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT),
                    makeBufferBarrier2(m_rayStateQueueA->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT),
                    makeBufferBarrier2(m_rayHitQueue->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT),
                    makeBufferBarrier2(m_materialIndexQueue->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT),
                    makeBufferBarrier2(m_secondaryIndexQueue->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)
                };

                VkMemoryBarrier2 c2sImageBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                c2sImageBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                c2sImageBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                c2sImageBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                c2sImageBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

                VkDependencyInfo c2sDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                c2sDep.memoryBarrierCount = 1;
                c2sDep.pMemoryBarriers = &c2sImageBarrier;
                c2sDep.bufferMemoryBarrierCount = static_cast<uint32_t>(c2sBarriers.size());
                c2sDep.pBufferMemoryBarriers = c2sBarriers.data();
                vkCmdPipelineBarrier2(cmd, &c2sDep);

                bool useMaterialSort = (sceneData.sortMode != 0 && m_shadeDiffusePipeline != VK_NULL_HANDLE);
                std::vector<VkPipeline> matPipelines = {
                    m_shadeDiffusePipeline,
                    m_shadeDielectricPipeline,
                    m_shadeConductorPipeline,
                    m_shadeComplexPipeline
                };
                if (m_shadeEmissivePipeline) matPipelines.push_back(m_shadeEmissivePipeline);
                if (m_shadePassthroughPipeline) matPipelines.push_back(m_shadePassthroughPipeline);

                // 4. Multi-Bounce Loop for this Tile
                for (uint32_t b = 0; b < maxBounces; ++b) {
                    VkDescriptorSet shadeSet = (b % 2 == 0) ? m_descSetsEven[frameSlot] : m_descSetsOdd[frameSlot];
                    VkDescriptorSet intersectSet = (b % 2 == 0) ? m_descSetsOdd[frameSlot] : m_descSetsEven[frameSlot];

                    // 4a. Shading microkernel(s)
                    uint32_t shadePC[18] = {
                        sceneData.numTriangles,
                        sceneData.numSpheres,
                        sceneData.numMaterials,
                        sceneData.numLights,
                        width,
                        height,
                        b,
                        maxBounces,
                        m_maxCapacity,
                        sampleIdx,
                        sceneData.hasEnvMap,
                        std::bit_cast<uint32_t>(sceneData.envMapIntensity),
                        sceneData.useHardwareRT,
                        sceneData.secondarySortMode,
                        sceneData.enableNrc ? 1u : 0u,
                        sceneData.nrcBounce,
                        std::bit_cast<uint32_t>(sceneData.nrcTrainRatio),
                        sceneData.frameIndex
                    };
                    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shadePC), shadePC);

                    uint32_t qBase = 3 + b * 6;
                    bool canProfileBounce = shouldProfile && (qBase + 5 < MAX_WAVEFRONT_TIMESTAMP_QUERIES);
                    if (canProfileBounce) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPools[frameSlot], qBase + 0);

                    if (useMaterialSort) {
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &shadeSet, 0, nullptr);
                        VkDeviceSize shadeOffset = static_cast<VkDeviceSize>(b * 16 + 0) * 16;
                        uint32_t numMatPipes = static_cast<uint32_t>(matPipelines.size());
                        if (m_dgcManager->isSupported() && m_dgcManager->isMaterialDGCSupported()) {
                            m_dgcManager->recordMaterialPreprocess(cmd, matPipelines, m_dgcStream.get(), shadeOffset, 0, numMatPipes);
                            m_dgcManager->recordPreprocessBarrier(cmd);
                            m_dgcManager->recordMaterialExecute(cmd, matPipelines, m_dgcStream.get(), shadeOffset, 0, numMatPipes, true /* isPreprocessed */);
                        } else {
                            m_dgcManager->recordMaterialExecute(cmd, matPipelines, m_indirectArgs[frameSlot].get(), shadeOffset, 0, numMatPipes, false /* isPreprocessed */);
                        }
                    } else {
                        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_shadePipeline);
                        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &shadeSet, 0, nullptr);
                        VkDeviceSize shadeOffset = static_cast<VkDeviceSize>(b * 3 + 0) * 16;
                        if (m_dgcManager->isSupported()) {
                            m_dgcManager->recordPreprocess(cmd, m_shadePipeline, m_indirectArgs[frameSlot].get(), shadeOffset, 0 /* slice 0 */, 1);
                            m_dgcManager->recordPreprocessBarrier(cmd);
                            m_dgcManager->recordExecute(cmd, m_shadePipeline, m_indirectArgs[frameSlot].get(), shadeOffset, 0 /* slice 0 */, 1, true /* isPreprocessed */);
                        } else {
                            m_dgcManager->recordIndirectDispatch(cmd, m_indirectArgs[frameSlot].get(), shadeOffset);
                        }
                    }
                    if (canProfileBounce) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPools[frameSlot], qBase + 1);

                    // Barrier: Shade -> Downstream (Shadow & Intersect indirect dispatches, scoped buffer barriers)
                    Buffer* currentOutGeom = (b % 2 == 0) ? m_rayGeomQueueB.get() : m_rayGeomQueueA.get();
                    Buffer* currentOutState = (b % 2 == 0) ? m_rayStateQueueB.get() : m_rayStateQueueA.get();

                    std::vector<VkBufferMemoryBarrier2> s2dBarriers;
                    s2dBarriers.reserve(8);
                    s2dBarriers.push_back(makeBufferBarrier2(m_indirectArgs[frameSlot]->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT,
                        VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT));
                    s2dBarriers.push_back(makeBufferBarrier2(m_dgcStream->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT, VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT));
                    s2dBarriers.push_back(makeBufferBarrier2(m_queueCounters->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
                    s2dBarriers.push_back(makeBufferBarrier2(m_shadowQueue->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
                    s2dBarriers.push_back(makeBufferBarrier2(currentOutGeom->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
                    s2dBarriers.push_back(makeBufferBarrier2(currentOutState->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT));
                    s2dBarriers.push_back(makeBufferBarrier2(m_secondaryIndexQueue->getBuffer(),
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT));

                    VkDependencyInfo s2dDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                    s2dDep.bufferMemoryBarrierCount = static_cast<uint32_t>(s2dBarriers.size());
                    s2dDep.pBufferMemoryBarriers = s2dBarriers.data();
                    vkCmdPipelineBarrier2(cmd, &s2dDep);

                    // Tier 2 Optimization: Batched Preprocessing
                    // Shade has written arguments for both Shadow and Intersect.
                    // Preprocess both passes concurrently before executing Shadow (unless disabled via --no-dgc-batch-preprocess).
                    bool batchPreprocess = (getenv("PATHWAYS_DISABLE_DGC_BATCH_PREPROCESS") == nullptr);
                    VkDeviceSize shadowOffset = useMaterialSort ? static_cast<VkDeviceSize>(b * 16 + 6) * 16
                                                                : static_cast<VkDeviceSize>(b * 3 + 1) * 16;
                    VkDeviceSize intersectOffset = useMaterialSort ? static_cast<VkDeviceSize>(b * 16 + 7) * 16
                                                                   : static_cast<VkDeviceSize>(b * 3 + 2) * 16;

                    if (m_dgcManager->isSupported()) {
                        m_dgcManager->recordPreprocess(cmd, m_shadowPipeline, m_indirectArgs[frameSlot].get(), shadowOffset, 1 /* slice 1 */, 1);
                        if (batchPreprocess && (b + 1 < maxBounces)) {
                            if (sceneData.secondarySortMode == 1 && useMaterialSort) {
                                for (uint32_t oct = 0; oct < 8; ++oct) {
                                    VkDeviceSize octOffset = static_cast<VkDeviceSize>(b * 16 + 8 + oct) * 16;
                                    m_dgcManager->recordPreprocess(cmd, m_intersectPipeline, m_indirectArgs[frameSlot].get(), octOffset, 2 + oct, 1);
                                }
                            } else {
                                m_dgcManager->recordPreprocess(cmd, m_intersectPipeline, m_indirectArgs[frameSlot].get(), intersectOffset, 2 /* slice 2 */, 1);
                            }
                        }
                        m_dgcManager->recordPreprocessBarrier(cmd);
                    }

                    // 4b. Shadow microkernel (100% coherent hardware ray queries)
                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_shadowPipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &shadeSet, 0, nullptr);
                    uint32_t shadowPC[9] = {
                        sceneData.numTriangles,
                        sceneData.numSpheres,
                        sceneData.numMaterials,
                        sceneData.numLights,
                        width,
                        height,
                        sceneData.frameIndex,
                        sampleIdx,
                        sceneData.numOpaqueTriangles
                    };
                    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(shadowPC), shadowPC);
                    if (canProfileBounce) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPools[frameSlot], qBase + 2);
                    if (m_dgcManager->isSupported()) {
                        m_dgcManager->recordExecute(cmd, m_shadowPipeline, m_indirectArgs[frameSlot].get(), shadowOffset, 1 /* slice 1 */, 1, true /* isPreprocessed */);
                    } else {
                        m_dgcManager->recordIndirectDispatch(cmd, m_indirectArgs[frameSlot].get(), shadowOffset);
                    }
                    if (canProfileBounce) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPools[frameSlot], qBase + 3);

                    // 4c. Intersect microkernel (pure BVH traversal for secondary rays)
                    // Note: Preprocessing was already executed in the batch above; dispatches launch immediately
                    if (b + 1 < maxBounces) {
                        uint32_t intersectPC[17] = {
                            sceneData.numTriangles,
                            sceneData.numSpheres,
                            sceneData.numMaterials,
                            sceneData.numLights,
                            width,
                            height,
                            b,
                            maxBounces,
                            m_maxCapacity,
                            sampleIdx,
                            sceneData.hasEnvMap,
                            std::bit_cast<uint32_t>(sceneData.envMapIntensity),
                            sceneData.useHardwareRT,
                            sceneData.sortMode,
                            sceneData.numOpaqueTriangles,
                            sceneData.secondarySortMode,
                            0 // octantBin
                        };
                        if (!batchPreprocess && m_dgcManager->isSupported()) {
                            if (sceneData.secondarySortMode == 1 && useMaterialSort) {
                                for (uint32_t oct = 0; oct < 8; ++oct) {
                                    VkDeviceSize octOffset = static_cast<VkDeviceSize>(b * 16 + 8 + oct) * 16;
                                    m_dgcManager->recordPreprocess(cmd, m_intersectPipeline, m_indirectArgs[frameSlot].get(), octOffset, 2 + oct, 1);
                                }
                            } else {
                                m_dgcManager->recordPreprocess(cmd, m_intersectPipeline, m_indirectArgs[frameSlot].get(), intersectOffset, 2 /* slice 2 */, 1);
                            }
                            m_dgcManager->recordPreprocessBarrier(cmd);
                        }

                        if (sceneData.secondarySortMode == 1 && useMaterialSort) {
                            // Option 1: On-Chip Directional Multi-Queue Binning (8 directional octants)
                            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_intersectPipeline);
                            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &intersectSet, 0, nullptr);

                            if (canProfileBounce) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPools[frameSlot], qBase + 4);

                            for (uint32_t oct = 0; oct < 8; ++oct) {
                                intersectPC[16] = oct;
                                vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(intersectPC), intersectPC);
                                VkDeviceSize octOffset = static_cast<VkDeviceSize>(b * 16 + 8 + oct) * 16;
                                if (m_dgcManager->isSupported()) {
                                    m_dgcManager->recordExecute(cmd, m_intersectPipeline, m_indirectArgs[frameSlot].get(), octOffset, 2 + oct, 1, true /* isPreprocessed */);
                                } else {
                                    m_dgcManager->recordIndirectDispatch(cmd, m_indirectArgs[frameSlot].get(), octOffset);
                                }
                            }

                            if (canProfileBounce) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPools[frameSlot], qBase + 5);
                        } else if (sceneData.secondarySortMode == 2 && useMaterialSort) {
                            // Option 2: Fused In-Register Spatial-Morton Wave32 Reordering (0ms pass overhead, zero barrier)
                            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_intersectPipeline);
                            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &intersectSet, 0, nullptr);
                            vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(intersectPC), intersectPC);

                            if (canProfileBounce) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPools[frameSlot], qBase + 4);
                            if (m_dgcManager->isSupported()) {
                                m_dgcManager->recordExecute(cmd, m_intersectPipeline, m_indirectArgs[frameSlot].get(), intersectOffset, 2 /* slice 2 */, 1, true /* isPreprocessed */);
                            } else {
                                m_dgcManager->recordIndirectDispatch(cmd, m_indirectArgs[frameSlot].get(), intersectOffset);
                            }
                            if (canProfileBounce) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPools[frameSlot], qBase + 5);
                        } else {
                            // Secondary sort disabled / fallback
                            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_intersectPipeline);
                            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &intersectSet, 0, nullptr);
                            vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(intersectPC), intersectPC);

                            if (canProfileBounce) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, m_queryPools[frameSlot], qBase + 4);
                            if (m_dgcManager->isSupported()) {
                                m_dgcManager->recordExecute(cmd, m_intersectPipeline, m_indirectArgs[frameSlot].get(), intersectOffset, 2 /* slice 2 */, 1, true /* isPreprocessed */);
                            } else {
                                m_dgcManager->recordIndirectDispatch(cmd, m_indirectArgs[frameSlot].get(), intersectOffset);
                            }
                            if (canProfileBounce) vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPools[frameSlot], qBase + 5);
                        }

                        // Barrier: Shadow & Intersect -> Next Bounce Shade (scoped buffer barriers)
                        Buffer* nextInGeom = currentOutGeom;
                        Buffer* nextInState = currentOutState;
                        std::array<VkBufferMemoryBarrier2, 8> d2sBarriers = {
                            makeBufferBarrier2(m_indirectArgs[frameSlot]->getBuffer(),
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT,
                                VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT),
                            makeBufferBarrier2(m_dgcStream->getBuffer(),
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT, VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT),
                            makeBufferBarrier2(m_queueCounters->getBuffer(),
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT),
                            makeBufferBarrier2(m_rayHitQueue->getBuffer(),
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT),
                            makeBufferBarrier2(m_materialIndexQueue->getBuffer(),
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT),
                            makeBufferBarrier2(nextInGeom->getBuffer(),
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT),
                            makeBufferBarrier2(nextInState->getBuffer(),
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT),
                            makeBufferBarrier2(m_secondaryIndexQueue->getBuffer(),
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT)
                        };

                        VkDependencyInfo d2sDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                        d2sDep.bufferMemoryBarrierCount = static_cast<uint32_t>(d2sBarriers.size());
                        d2sDep.pBufferMemoryBarriers = d2sBarriers.data();
                        vkCmdPipelineBarrier2(cmd, &d2sDep);
                    } else {
                        if (canProfileBounce) {
                            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPools[frameSlot], qBase + 4);
                            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPools[frameSlot], qBase + 5);
                        }
                    }
                } // end multi-bounce loop
            } // end tx
        } // end ty
    } // end sampleIdx

    vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPools[frameSlot], endQuery);

    // 5. Final Barrier: Accumulation Image Writes -> Downstream Postprocessing & Tonemapping
    VkMemoryBarrier2 finalBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    finalBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    finalBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    finalBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    finalBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_READ_BIT;

    VkDependencyInfo finalDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    finalDep.memoryBarrierCount = 1;
    finalDep.pMemoryBarriers = &finalBarrier;
    vkCmdPipelineBarrier2(cmd, &finalDep);
}

double WavefrontPipeline::getQueueMemoryFootprintMb() const {
    // geomA(16B) + geomB(16B) + stateA(32B) + stateB(32B) + hit(16B) + shadow(32B) + matIndices(24B) + secIndices(32B) = 200B
    double bytes = static_cast<double>(m_maxCapacity) * 200.0 + 256.0 + 16384.0 + 16384.0;
    return bytes / (1024.0 * 1024.0);
}

WavefrontPipeline::WavefrontProfilingData WavefrontPipeline::getProfilingData(uint32_t frameSlot, double timestampPeriodNs, uint32_t maxBounces) {
    WavefrontProfilingData data{};
    if (frameSlot >= 2 || !m_queryPools[frameSlot] || !m_hasRecordedSlot[frameSlot]) return data;

    uint32_t effectiveBounces = (m_slotBounces[frameSlot] > 0) ? m_slotBounces[frameSlot] : maxBounces;
    std::vector<uint64_t> ts(MAX_WAVEFRONT_TIMESTAMP_QUERIES, 0);
    uint32_t endQuery = std::min(3u + effectiveBounces * 6u, MAX_WAVEFRONT_TIMESTAMP_QUERIES - 1u);
    uint32_t numQueries = endQuery + 1;

    VkResult res = vkGetQueryPoolResults(m_device, m_queryPools[frameSlot], 0, numQueries,
                                         numQueries * sizeof(uint64_t), ts.data(), sizeof(uint64_t),
                                         VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT);
    if (res != VK_SUCCESS) {
        return data;
    }

    auto toMs = [&](uint64_t t1, uint64_t t0) -> double {
        return (t1 > t0) ? (t1 - t0) * timestampPeriodNs * 1e-6 : 0.0;
    };

    data.valid = true;
    data.totalMs = toMs(ts[endQuery], ts[0]);
    data.classifyMs = toMs(ts[2], ts[1]);
    data.resolveMs = 0.0;
    data.sortMode = m_sortMode;
    data.secondarySortMode = m_secondarySortMode;
    data.queueMemoryFootprintMb = getQueueMemoryFootprintMb();

    struct BounceDispatchCPU {
        uint32_t shadeX, shadeY, shadeZ, activeCount;
        uint32_t shadowX, shadowY, shadowZ, shadowCount;
        uint32_t intersectX, intersectY, intersectZ, nextCount;
    };
    struct BounceMaterialDispatchCPU {
        uint32_t diffX, diffY, diffZ, diffCount;
        uint32_t dielX, dielY, dielZ, dielCount;
        uint32_t condX, condY, condZ, condCount;
        uint32_t compX, compY, compZ, compCount;
        uint32_t emisX, emisY, emisZ, emisCount;
        uint32_t passX, passY, passZ, passCount;
        uint32_t shadowX, shadowY, shadowZ, shadowCount;
        uint32_t intersectX, intersectY, intersectZ, nextCount;
        struct OctantDispatch {
            uint32_t x, y, z, count;
        } octants[8];
    };
    static_assert(sizeof(BounceMaterialDispatchCPU) == 256, "BounceMaterialDispatchCPU stride must be 256 bytes");

    const void* pMapped = m_indirectArgs[frameSlot]->map();
    const BounceDispatchCPU* dispatches = reinterpret_cast<const BounceDispatchCPU*>(pMapped);
    const BounceMaterialDispatchCPU* matDispatches = reinterpret_cast<const BounceMaterialDispatchCPU*>(pMapped);
    bool isMaterialMode = (m_shadeDiffusePipeline != VK_NULL_HANDLE && m_sortMode != 0);

    double totalTrafficBytes = 0.0;

    for (uint32_t b = 0; b < effectiveBounces; ++b) {
        uint32_t base = 3 + b * 6;
        if (base + 5 >= numQueries) break;
        BounceProfilingData bp{};
        bp.bounce = b;
        bp.shadeMs = toMs(ts[base + 1], ts[base + 0]);
        bp.shadowMs = toMs(ts[base + 3], ts[base + 2]);
        bp.intersectMs = (b + 1 < effectiveBounces) ? toMs(ts[base + 5], ts[base + 4]) : 0.0;

        if (isMaterialMode && matDispatches) {
            bp.diffCount = matDispatches[b].diffCount;
            bp.dielCount = matDispatches[b].dielCount;
            bp.condCount = matDispatches[b].condCount;
            bp.compCount = matDispatches[b].compCount;
            bp.emisCount = matDispatches[b].emisCount;
            bp.passCount = matDispatches[b].passCount;
            bp.activeCount = bp.diffCount + bp.dielCount + bp.condCount + bp.compCount + bp.emisCount + bp.passCount;
            bp.shadowCount = matDispatches[b].shadowCount;
            bp.nextCount = matDispatches[b].nextCount;
        } else if (dispatches) {
            bp.activeCount = dispatches[b].activeCount;
            bp.shadowCount = dispatches[b].shadowCount;
            bp.nextCount = dispatches[b].nextCount;
            bp.diffCount = bp.activeCount;
        }

        // Memory Traffic Estimation:
        // Shade reads Geom(16B) + State(32B) + Hit(16B) = 64B
        // Shade writes Shadow(32B) if shadow ray, NextGeom(16B) + NextState(32B) = 48B if active
        // Shadow reads Shadow(32B)
        // Intersect reads Geom(16B) and writes Hit(16B) = 32B
        double shadeRead = bp.activeCount * 64.0;
        double shadeWrite = (bp.shadowCount * 32.0) + (bp.nextCount * 48.0);
        double shadowRead = bp.shadowCount * 32.0;
        double intersectTraffic = bp.nextCount * 32.0;
        totalTrafficBytes += (shadeRead + shadeWrite + shadowRead + intersectTraffic);

        data.bounces.push_back(bp);
    }
    if (pMapped) m_indirectArgs[frameSlot]->unmap();

    // Primary Classify traffic: writes primary active rays (Geom 16B + State 32B + Hit 16B = 64B)
    uint32_t primaryRays = (m_width * m_height);
    totalTrafficBytes += primaryRays * 64.0;
    data.estimatedVramTrafficMb = totalTrafficBytes / (1024.0 * 1024.0);

    return data;
}

void WavefrontPipeline::printProfilingBreakdown(uint32_t frameSlot, double timestampPeriodNs, uint32_t maxBounces) {
    WavefrontProfilingData data = getProfilingData(frameSlot, timestampPeriodNs, maxBounces);
    if (!data.valid) return;

    const char* secSortStr = (data.secondarySortMode == 1) ? "Directional DGC (8-bin)" :
                             (data.secondarySortMode == 2) ? "Spatial Morton (Index)" : "None";
    Logger::info("    --- Wavefront Sub-Pass GPU Timing Breakdown (Frame Total: {:.3f} ms, Secondary Sort: {}) ---",
                 data.totalMs, secSortStr);
    Logger::info("      [Primary] Classify: {:.3f} ms | VRAM Traffic: {:.1f} MB",
                 data.classifyMs, data.estimatedVramTrafficMb);

    bool isMaterialMode = (m_shadeDiffusePipeline != VK_NULL_HANDLE && m_sortMode != 0);
    for (const auto& bp : data.bounces) {
        if (isMaterialMode) {
            Logger::info("      [Bounce {}] Shade: {:.3f} ms (diff: {}, diel: {}, cond: {}, comp: {}, emis: {}, pass: {}) | Shadow: {:.3f} ms ({} rays) | Intersect: {:.3f} ms ({} rays)",
                         bp.bounce, bp.shadeMs, bp.diffCount, bp.dielCount, bp.condCount, bp.compCount, bp.emisCount, bp.passCount,
                         bp.shadowMs, bp.shadowCount, bp.intersectMs, bp.nextCount);
        } else {
            Logger::info("      [Bounce {}] Shade: {:.3f} ms ({} rays) | Shadow: {:.3f} ms ({} rays) | Intersect: {:.3f} ms ({} rays)",
                         bp.bounce, bp.shadeMs, bp.activeCount, bp.shadowMs, bp.shadowCount, bp.intersectMs, bp.nextCount);
        }
    }
}

} // namespace pathways
