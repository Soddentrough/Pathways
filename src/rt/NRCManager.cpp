#include "NRCManager.hpp"
#include "core/Logger.hpp"

#include <stdexcept>
#include <random>
#include <cstring>
#include <algorithm>
#include <glm/gtc/packing.hpp>

namespace pathways {

NRCManager::NRCManager(VkDevice device, VmaAllocator allocator,
                       uint32_t width, uint32_t height,
                       const std::vector<char>& inferSpv,
                       const std::vector<char>& trainSpv)
    : m_device(device), m_allocator(allocator), m_width(width), m_height(height)
{
    initBuffers();
    initWeightsAndHashTable();
    createDescriptorSetLayouts();
    allocateDescriptorSets();
    createPipelines(inferSpv, trainSpv);
    m_initialized = true;
    Logger::info("NRCManager initialized: {}x{} cache resolution, Wave32 WMMA enabled.", width, height);
}

NRCManager::~NRCManager() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_inferPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_device, m_inferPipeline, nullptr);
        if (m_inferPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device, m_inferPipelineLayout, nullptr);
        if (m_inferDescLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_device, m_inferDescLayout, nullptr);

        if (m_trainPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_device, m_trainPipeline, nullptr);
        if (m_trainPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device, m_trainPipelineLayout, nullptr);
        if (m_trainDescLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_device, m_trainDescLayout, nullptr);

        if (m_descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    }
}

void NRCManager::initBuffers() {
    m_maxQueries = m_width * m_height;
    m_maxTrainRecords = std::max(16384u, m_maxQueries / 16u);

    VkBufferUsageFlags queueUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                    VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

    // 1. Query Queue: 80 bytes per query
    VkDeviceSize querySize = static_cast<VkDeviceSize>(m_maxQueries) * sizeof(NRCQuery);
    m_queryQueue = std::make_unique<Buffer>(m_allocator, querySize, queueUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // 2. Training Queue: 96 bytes per record
    VkDeviceSize trainSize = static_cast<VkDeviceSize>(m_maxTrainRecords) * sizeof(NRCTrainingRecord);
    m_trainQueue = std::make_unique<Buffer>(m_allocator, trainSize, queueUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // 3. Queue Counters: 256 bytes
    m_counters = std::make_unique<Buffer>(m_allocator, 256, queueUsage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                          VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    // 4. Hash Table: 12 levels * 262144 entries * 4 bytes = 12,582,912 bytes (~12.58 MB)
    VkDeviceSize hashTableSize = 12ull * 262144ull * sizeof(uint32_t);
    m_hashTable = std::make_unique<Buffer>(m_allocator, hashTableSize, queueUsage,
                                           VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    // 5. Weights buffer: 9360 * 2 bytes = 18,720 bytes (aligned to 19,200 bytes)
    VkDeviceSize weightsSize = 19200;
    m_weights = std::make_unique<Buffer>(m_allocator, weightsSize, queueUsage,
                                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

    // 6. Adam Momentum buffer: 2 * 9360 * sizeof(float) = 74,880 bytes
    VkDeviceSize momentumSize = 2ull * 9360ull * sizeof(float);
    m_weightMomentum = std::make_unique<Buffer>(m_allocator, momentumSize, queueUsage,
                                                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
}

void NRCManager::initWeightsAndHashTable() {
    std::mt19937 rng(42);
    std::normal_distribution<float> weightDist(0.0f, std::sqrt(2.0f / 64.0f)); // Kaiming He normal
    std::uniform_real_distribution<float> featDist(-0.01f, 0.01f);

    // Initialize Hash Table
    uint32_t* pHash = static_cast<uint32_t*>(m_hashTable->map());
    if (pHash) {
        size_t totalEntries = 12ull * 262144ull;
        for (size_t i = 0; i < totalEntries; ++i) {
            float f0 = featDist(rng);
            float f1 = featDist(rng);
            pHash[i] = glm::packHalf2x16(glm::vec2(f0, f1));
        }
        m_hashTable->flush();
        m_hashTable->unmap();
    }

    // Initialize Weights
    uint16_t* pWeights = static_cast<uint16_t*>(m_weights->map());
    if (pWeights) {
        // Layer 0: 64 x 64 weights
        for (size_t i = 0; i < 4096; ++i) {
            pWeights[i] = glm::packHalf1x16(weightDist(rng));
        }
        // Layer 0: 64 biases
        for (size_t i = 0; i < 64; ++i) {
            pWeights[4096 + i] = glm::packHalf1x16(0.0f);
        }
        // Layer 1: 64 x 64 weights
        for (size_t i = 0; i < 4096; ++i) {
            pWeights[4160 + i] = glm::packHalf1x16(weightDist(rng));
        }
        // Layer 1: 64 biases
        for (size_t i = 0; i < 64; ++i) {
            pWeights[8256 + i] = glm::packHalf1x16(0.0f);
        }
        // Layer 2: 64 x 16 weights
        for (size_t i = 0; i < 1024; ++i) {
            pWeights[8320 + i] = glm::packHalf1x16(weightDist(rng));
        }
        // Layer 2: 16 biases
        for (size_t i = 0; i < 16; ++i) {
            pWeights[9344 + i] = glm::packHalf1x16(0.0f);
        }
        m_weights->flush();
        m_weights->unmap();
    }

    // Zero out Momentum buffer
    void* pMom = m_weightMomentum->map();
    if (pMom) {
        std::memset(pMom, 0, 2ull * 9360ull * sizeof(float));
        m_weightMomentum->flush();
        m_weightMomentum->unmap();
    }

    // Zero out Counters
    void* pCount = m_counters->map();
    if (pCount) {
        std::memset(pCount, 0, 256);
        m_counters->flush();
        m_counters->unmap();
    }
}

void NRCManager::createDescriptorSetLayouts() {
    // Inference Layout: Image(0), QueryQueue(1), HashTable(2), Weights(3), Counters(4)
    std::vector<VkDescriptorSetLayoutBinding> inferBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo inferLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    inferLayoutInfo.bindingCount = static_cast<uint32_t>(inferBindings.size());
    inferLayoutInfo.pBindings = inferBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &inferLayoutInfo, nullptr, &m_inferDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to create inference descriptor set layout!");
    }

    // Training Layout: TrainQueue(0), HashTable(1), Weights(2), Momentum(3), Counters(4)
    std::vector<VkDescriptorSetLayoutBinding> trainBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo trainLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    trainLayoutInfo.bindingCount = static_cast<uint32_t>(trainBindings.size());
    trainLayoutInfo.pBindings = trainBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &trainLayoutInfo, nullptr, &m_trainDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to create training descriptor set layout!");
    }
}

void NRCManager::allocateDescriptorSets() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 4 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 }
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 4;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to create descriptor pool!");
    }

    VkDescriptorSetAllocateInfo allocInfer{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfer.descriptorPool = m_descriptorPool;
    allocInfer.descriptorSetCount = 1;
    allocInfer.pSetLayouts = &m_inferDescLayout;
    if (vkAllocateDescriptorSets(m_device, &allocInfer, &m_inferDescSet) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to allocate inference descriptor set!");
    }

    VkDescriptorSetAllocateInfo allocTrain{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocTrain.descriptorPool = m_descriptorPool;
    allocTrain.descriptorSetCount = 1;
    allocTrain.pSetLayouts = &m_trainDescLayout;
    if (vkAllocateDescriptorSets(m_device, &allocTrain, &m_trainDescSet) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to allocate training descriptor set!");
    }

    // Write persistent buffer bindings for training descriptor set
    VkDescriptorBufferInfo trainQueueInfo{ m_trainQueue->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo hashInfo{ m_hashTable->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo weightsInfo{ m_weights->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo momInfo{ m_weightMomentum->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo countersInfo{ m_counters->getBuffer(), 0, VK_WHOLE_SIZE };

    std::vector<VkWriteDescriptorSet> trainWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &trainQueueInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hashInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightsInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &momInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &countersInfo, nullptr }
    };
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(trainWrites.size()), trainWrites.data(), 0, nullptr);
}

void NRCManager::updateDescriptors(VkImageView accumImageView) {
    VkDescriptorImageInfo imageInfo{ VK_NULL_HANDLE, accumImageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo queryQueueInfo{ m_queryQueue->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo hashInfo{ m_hashTable->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo weightsInfo{ m_weights->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo countersInfo{ m_counters->getBuffer(), 0, VK_WHOLE_SIZE };

    std::vector<VkWriteDescriptorSet> writes = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &queryQueueInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &hashInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightsInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &countersInfo, nullptr }
    };
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

VkShaderModule NRCManager::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &mod) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to create shader module!");
    }
    return mod;
}

void NRCManager::createPipelines(const std::vector<char>& inferSpv, const std::vector<char>& trainSpv) {
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(uint32_t) * 4 + sizeof(float) * 8; // 48 bytes

    // Inference Pipeline
    VkPipelineLayoutCreateInfo inferLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    inferLayoutInfo.setLayoutCount = 1;
    inferLayoutInfo.pSetLayouts = &m_inferDescLayout;
    inferLayoutInfo.pushConstantRangeCount = 1;
    inferLayoutInfo.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(m_device, &inferLayoutInfo, nullptr, &m_inferPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to create inference pipeline layout!");
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    VkShaderModule inferMod = createShaderModule(inferSpv);
    VkComputePipelineCreateInfo inferPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    inferPipeInfo.layout = m_inferPipelineLayout;
    inferPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    inferPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    inferPipeInfo.stage.module = inferMod;
    inferPipeInfo.stage.pName = "main";
    inferPipeInfo.stage.pNext = &subgroupSize32;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &inferPipeInfo, nullptr, &m_inferPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, inferMod, nullptr);
        throw std::runtime_error("NRCManager: Failed to create inference compute pipeline!");
    }
    vkDestroyShaderModule(m_device, inferMod, nullptr);

    // Training Pipeline
    VkPipelineLayoutCreateInfo trainLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    trainLayoutInfo.setLayoutCount = 1;
    trainLayoutInfo.pSetLayouts = &m_trainDescLayout;
    trainLayoutInfo.pushConstantRangeCount = 1;
    trainLayoutInfo.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(m_device, &trainLayoutInfo, nullptr, &m_trainPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to create training pipeline layout!");
    }

    VkShaderModule trainMod = createShaderModule(trainSpv);
    VkComputePipelineCreateInfo trainPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    trainPipeInfo.layout = m_trainPipelineLayout;
    trainPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    trainPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    trainPipeInfo.stage.module = trainMod;
    trainPipeInfo.stage.pName = "main";
    trainPipeInfo.stage.pNext = &subgroupSize32;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &trainPipeInfo, nullptr, &m_trainPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, trainMod, nullptr);
        throw std::runtime_error("NRCManager: Failed to create training compute pipeline!");
    }
    vkDestroyShaderModule(m_device, trainMod, nullptr);
}

void NRCManager::resize(uint32_t width, uint32_t height) {
    if (m_width == width && m_height == height) return;
    m_width = width;
    m_height = height;
    initBuffers();
    initWeightsAndHashTable();
}

void NRCManager::resetCounters(VkCommandBuffer cmd) {
    vkCmdFillBuffer(cmd, m_counters->getBuffer(), 0, 256, 0);

    VkBufferMemoryBarrier2 fillBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
    fillBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    fillBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    fillBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    fillBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    fillBarrier.buffer = m_counters->getBuffer();
    fillBarrier.offset = 0;
    fillBarrier.size = 256;

    VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    depInfo.bufferMemoryBarrierCount = 1;
    depInfo.pBufferMemoryBarriers = &fillBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void NRCManager::recordInference(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                glm::vec3 worldMin, glm::vec3 worldMax, uint32_t queryCountOverride)
{
    if (!m_initialized || m_inferPipeline == VK_NULL_HANDLE) return;

    // Buffer memory barrier: Wavefront writes query queue -> Inference reads query queue
    std::array<VkBufferMemoryBarrier2, 2> inBarriers = {
        VkBufferMemoryBarrier2{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_queryQueue->getBuffer(), 0, VK_WHOLE_SIZE
        },
        VkBufferMemoryBarrier2{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_counters->getBuffer(), 0, 256
        }
    };
    VkDependencyInfo inDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    inDep.bufferMemoryBarrierCount = static_cast<uint32_t>(inBarriers.size());
    inDep.pBufferMemoryBarriers = inBarriers.data();
    vkCmdPipelineBarrier2(cmd, &inDep);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_inferPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_inferPipelineLayout, 0, 1, &m_inferDescSet, 0, nullptr);

    glm::vec3 extent = worldMax - worldMin;
    glm::vec3 invExtent = glm::vec3(
        extent.x > 1e-4f ? 1.0f / extent.x : 1.0f,
        extent.y > 1e-4f ? 1.0f / extent.y : 1.0f,
        extent.z > 1e-4f ? 1.0f / extent.z : 1.0f
    );

    uint32_t queryCount = queryCountOverride > 0 ? queryCountOverride : m_maxQueries;

    struct InferPushConstants {
        uint32_t width;
        uint32_t height;
        uint32_t queryCount;
        uint32_t pad;
        glm::vec4 worldMin;
        glm::vec4 invWorldExtent;
    } pc;
    pc.width = width;
    pc.height = height;
    pc.queryCount = queryCount;
    pc.pad = 0;
    pc.worldMin = glm::vec4(worldMin, 0.0f);
    pc.invWorldExtent = glm::vec4(invExtent, 0.0f);

    vkCmdPushConstants(cmd, m_inferPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // Each workgroup handles a batch of 16 queries
    uint32_t dispatchCount = (queryCount + 15u) / 16u;
    if (dispatchCount > 0) {
        vkCmdDispatch(cmd, dispatchCount, 1, 1);
    }

    // Barrier: Inference writes uAccumImage -> downstream tonemapping / denoiser reads
    VkMemoryBarrier2 outBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    outBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    outBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    outBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    outBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    VkDependencyInfo outDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    outDep.memoryBarrierCount = 1;
    outDep.pMemoryBarriers = &outBarrier;
    vkCmdPipelineBarrier2(cmd, &outDep);
}

void NRCManager::recordTraining(VkCommandBuffer cmd, uint32_t frameIndex,
                               glm::vec3 worldMin, glm::vec3 worldMax, float learningRate,
                               uint32_t batchSize)
{
    if (!m_initialized || m_trainPipeline == VK_NULL_HANDLE) return;

    // Buffer memory barrier: Wavefront writes train records -> Training reads train records
    std::array<VkBufferMemoryBarrier2, 2> inBarriers = {
        VkBufferMemoryBarrier2{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_trainQueue->getBuffer(), 0, VK_WHOLE_SIZE
        },
        VkBufferMemoryBarrier2{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_counters->getBuffer(), 0, 256
        }
    };
    VkDependencyInfo inDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    inDep.bufferMemoryBarrierCount = static_cast<uint32_t>(inBarriers.size());
    inDep.pBufferMemoryBarriers = inBarriers.data();
    vkCmdPipelineBarrier2(cmd, &inDep);

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_trainPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_trainPipelineLayout, 0, 1, &m_trainDescSet, 0, nullptr);

    glm::vec3 extent = worldMax - worldMin;
    glm::vec3 invExtent = glm::vec3(
        extent.x > 1e-4f ? 1.0f / extent.x : 1.0f,
        extent.y > 1e-4f ? 1.0f / extent.y : 1.0f,
        extent.z > 1e-4f ? 1.0f / extent.z : 1.0f
    );

    struct TrainPushConstants {
        uint32_t trainCount;
        uint32_t batchSize;
        float learningRate;
        uint32_t frameIndex;
        glm::vec4 worldMin;
        glm::vec4 invWorldExtent;
    } pc;
    pc.trainCount = m_maxTrainRecords;
    pc.batchSize = batchSize;
    pc.learningRate = learningRate;
    pc.frameIndex = frameIndex;
    pc.worldMin = glm::vec4(worldMin, 0.0f);
    pc.invWorldExtent = glm::vec4(invExtent, 0.0f);

    vkCmdPushConstants(cmd, m_trainPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t dispatchCount = (batchSize + 31u) / 32u;
    if (dispatchCount > 0) {
        vkCmdDispatch(cmd, dispatchCount, 1, 1);
    }

    // Barrier: Training updates weights and hash table -> next frame inference reads weights
    std::array<VkBufferMemoryBarrier2, 2> outBarriers = {
        VkBufferMemoryBarrier2{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_hashTable->getBuffer(), 0, VK_WHOLE_SIZE
        },
        VkBufferMemoryBarrier2{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_weights->getBuffer(), 0, VK_WHOLE_SIZE
        }
    };
    VkDependencyInfo outDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    outDep.bufferMemoryBarrierCount = static_cast<uint32_t>(outBarriers.size());
    outDep.pBufferMemoryBarriers = outBarriers.data();
    vkCmdPipelineBarrier2(cmd, &outDep);
}

} // namespace pathways
