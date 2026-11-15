#include "NRCManager.hpp"
#include "rt/FfxSdkRuntime.hpp"
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
                       const std::vector<char>& trainSpv,
                       const std::vector<char>& resolveSpv,
                       VkPhysicalDevice physDevice)
    : m_device(device), m_physDevice(physDevice), m_allocator(allocator), m_width(width), m_height(height)
{
    initBuffers();
    initWeights();
    createDescriptorSetLayouts();
    allocateDescriptorSets();
    createPipelines(inferSpv, trainSpv, resolveSpv);
    initFfxRadianceCacheContext();
    m_initialized = true;
    Logger::info("NRCManager initialized: {}x{} cache resolution ({}).", width, height, getActiveBackendName());
}

NRCManager::~NRCManager() {
    destroyFfxRadianceCacheContext();
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
        if (m_inferPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_device, m_inferPipeline, nullptr);
        if (m_inferPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device, m_inferPipelineLayout, nullptr);
        if (m_inferDescLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_device, m_inferDescLayout, nullptr);

        if (m_trainPipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_device, m_trainPipeline, nullptr);
        if (m_trainPipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device, m_trainPipelineLayout, nullptr);
        if (m_trainDescLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_device, m_trainDescLayout, nullptr);

        if (m_resolvePipeline != VK_NULL_HANDLE) vkDestroyPipeline(m_device, m_resolvePipeline, nullptr);
        if (m_resolvePipelineLayout != VK_NULL_HANDLE) vkDestroyPipelineLayout(m_device, m_resolvePipelineLayout, nullptr);
        if (m_resolveDescLayout != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(m_device, m_resolveDescLayout, nullptr);

        if (m_descriptorPool != VK_NULL_HANDLE) vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    }
}

void NRCManager::setBackend(NrcBackend backend) {
    if (m_backend == backend) return;
    m_backend = backend;
    initFfxRadianceCacheContext();
}

const char* NRCManager::getActiveBackendName() const {
    if (m_ffxRcContext != nullptr) {
        return "AMD FSR Radiance Cache 0.9.0 (SDK Provider)";
    }
    if (m_backend == NrcBackend::FfxRadianceCache) {
        return "Stub (No Native Vulkan Radiance Cache Provider)";
    }
    return "Native Wave32 WMMA Positional MLP";
}

void NRCManager::initFfxRadianceCacheContext() {
    destroyFfxRadianceCacheContext();
    if (m_backend == NrcBackend::NativeWave32) {
        return;
    }
#if defined(PATHWAYS_HAS_FFX_SDK)
    if (!FfxSdkRuntime::instance().hasNativeProvider(FfxEffectModule::RadianceCache) ||
        m_physDevice == VK_NULL_HANDLE) {
        return;
    }
    auto backendDesc = FfxSdkRuntime::makeBackendVkDesc(m_device, m_physDevice);
    ffxCreateContextDescRadianceCache createRc{};
    createRc.header.type = FFX_API_CREATE_CONTEXT_DESC_TYPE_RADIANCECACHE;
    createRc.header.pNext = &backendDesc.header;
    createRc.version = FFX_RADIANCECACHE_VERSION;
    createRc.maxInferenceSampleCount = m_maxQueries;
    createRc.maxTrainingSampleCount = m_maxTrainRecords;
    createRc.flags = FFX_RADIANCE_CACHE_CONTEXT_TRY_FORCE_WMMA;

    ffxContext ctx = nullptr;
    if (ffxCreateContext(&ctx, &createRc.header, nullptr) == FFX_API_RETURN_OK && ctx != nullptr) {
        m_ffxRcContext = ctx;
    }
#endif
}

void NRCManager::destroyFfxRadianceCacheContext() {
#if defined(PATHWAYS_HAS_FFX_SDK)
    if (m_ffxRcContext != nullptr) {
        ffxDestroyContext(&m_ffxRcContext, nullptr);
        m_ffxRcContext = nullptr;
    }
#else
    m_ffxRcContext = nullptr;
#endif
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

    // 2. Training Queue: 128 bytes per record (with self-training tail vertex)
    VkDeviceSize trainSize = static_cast<VkDeviceSize>(m_maxTrainRecords) * sizeof(NRCTrainingRecord);
    m_trainQueue = std::make_unique<Buffer>(m_allocator, trainSize, queueUsage, VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // 3. Queue Counters: 256 bytes (Device-Local VRAM for fast atomicAdd & Indirect Dispatch)
    m_counters = std::make_unique<Buffer>(m_allocator, 256, queueUsage | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT,
                                          VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // 4. Weights buffer: 9360 * 2 bytes = 18,720 bytes (aligned to 19,200 bytes in Device-Local VRAM)
    VkDeviceSize weightsSize = 19200;
    m_weights = std::make_unique<Buffer>(m_allocator, weightsSize, queueUsage,
                                         VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // 5. Adam Momentum + FP32 Master Weights buffer: 3 * 9360 * sizeof(float) = 112,320 bytes (Device-Local VRAM)
    VkDeviceSize momentumSize = 3ull * 9360ull * sizeof(float);
    m_weightMomentum = std::make_unique<Buffer>(m_allocator, momentumSize, queueUsage,
                                                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);

    // 6. Atomic Accumulation Buffer: 6 channels (diffuse RGB + specular RGB) * width * height * sizeof(uint32_t) bytes in Device-Local VRAM
    VkDeviceSize atomicAccumSize = static_cast<VkDeviceSize>(m_width) * m_height * 6ull * sizeof(uint32_t);
    m_atomicAccumBuffer = std::make_unique<Buffer>(m_allocator, atomicAccumSize, queueUsage,
                                                  VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
}

void NRCManager::initWeights() {
    m_adamStep = 0;
    std::mt19937 rng(42);
    std::normal_distribution<float> weightDist(0.0f, std::sqrt(2.0f / 64.0f)); // Kaiming He normal

    m_initialWeights.assign(19200 / sizeof(uint16_t), 0u);
    m_initialMasterWeights.assign(9360, 0.0f);
    uint16_t* pWeights = m_initialWeights.data();

    auto setParam = [&](size_t idx, float val) {
        m_initialMasterWeights[idx] = val;
        pWeights[idx] = glm::packHalf1x16(val);
    };

    // Layer 0: 64 x 64 weights
    for (size_t i = 0; i < 4096; ++i) {
        setParam(i, weightDist(rng));
    }
    // Layer 0: 64 biases
    for (size_t i = 0; i < 64; ++i) {
        setParam(4096 + i, 0.0f);
    }
    // Layer 1: 64 x 64 weights
    for (size_t i = 0; i < 4096; ++i) {
        setParam(4160 + i, weightDist(rng));
    }
    // Layer 1: 64 biases
    for (size_t i = 0; i < 64; ++i) {
        setParam(8256 + i, 0.0f);
    }
    // Layer 2: 64 x 16 weights (scaled by 0.02 so initial predictions start near 0 without frame-1 blotches)
    for (size_t i = 0; i < 1024; ++i) {
        setParam(8320 + i, weightDist(rng) * 0.02f);
    }
    // Layer 2: 16 biases
    for (size_t i = 0; i < 16; ++i) {
        setParam(9344 + i, 0.0f);
    }

    m_needsGpuInit = true;
}

void NRCManager::createDescriptorSetLayouts() {
    // Inference Layout: AtomicBuffer(0), QueryQueue(1), Weights(2), Counters(3)
    std::vector<VkDescriptorSetLayoutBinding> inferBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo inferLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    inferLayoutInfo.bindingCount = static_cast<uint32_t>(inferBindings.size());
    inferLayoutInfo.pBindings = inferBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &inferLayoutInfo, nullptr, &m_inferDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to create inference descriptor set layout!");
    }

    // Training Layout: TrainQueue(0), Weights(1), Momentum(2), Counters(3)
    std::vector<VkDescriptorSetLayoutBinding> trainBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo trainLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    trainLayoutInfo.bindingCount = static_cast<uint32_t>(trainBindings.size());
    trainLayoutInfo.pBindings = trainBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &trainLayoutInfo, nullptr, &m_trainDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to create training descriptor set layout!");
    }

    // Resolve Layout: uAccumImage(0), AtomicBuffer(1), uMlDiffuseImage(2), uMlSpecularImage(3)
    std::vector<VkDescriptorSetLayoutBinding> resolveBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo resolveLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    resolveLayoutInfo.bindingCount = static_cast<uint32_t>(resolveBindings.size());
    resolveLayoutInfo.pBindings = resolveBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &resolveLayoutInfo, nullptr, &m_resolveDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to create resolve descriptor set layout!");
    }
}

void NRCManager::allocateDescriptorSets() {
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 12 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 20 }
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

    std::array<VkDescriptorSetLayout, 2> resolveLayouts = { m_resolveDescLayout, m_resolveDescLayout };
    VkDescriptorSetAllocateInfo allocResolve{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocResolve.descriptorPool = m_descriptorPool;
    allocResolve.descriptorSetCount = 2;
    allocResolve.pSetLayouts = resolveLayouts.data();
    if (vkAllocateDescriptorSets(m_device, &allocResolve, m_resolveDescSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("NRCManager: Failed to allocate resolve descriptor sets!");
    }

    // Write persistent buffer bindings for inference descriptor set (0: AtomicBuffer)
    VkDescriptorBufferInfo atomicAccumInfo{ m_atomicAccumBuffer->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo queryQueueInfo{ m_queryQueue->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo weightsInfo{ m_weights->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo countersInfo{ m_counters->getBuffer(), 0, VK_WHOLE_SIZE };

    std::vector<VkWriteDescriptorSet> inferWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &atomicAccumInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &queryQueueInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightsInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &countersInfo, nullptr }
    };
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(inferWrites.size()), inferWrites.data(), 0, nullptr);

    // Write persistent buffer bindings for training descriptor set
    VkDescriptorBufferInfo trainQueueInfo{ m_trainQueue->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo momInfo{ m_weightMomentum->getBuffer(), 0, VK_WHOLE_SIZE };

    std::vector<VkWriteDescriptorSet> trainWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &trainQueueInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightsInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &momInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &countersInfo, nullptr }
    };
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(trainWrites.size()), trainWrites.data(), 0, nullptr);

    // Write persistent buffer binding for resolve descriptor sets (binding 1: AtomicBuffer)
    std::array<VkWriteDescriptorSet, 2> resolveBufWrites = {{
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_resolveDescSets[0], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &atomicAccumInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_resolveDescSets[1], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &atomicAccumInfo, nullptr }
    }};
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(resolveBufWrites.size()), resolveBufWrites.data(), 0, nullptr);
}

void NRCManager::updateDescriptors(VkImageView accumImageView, VkImageView mlDiffuseImageView, VkImageView mlSpecularImageView) {
    updateDescriptors(0, accumImageView, mlDiffuseImageView, mlSpecularImageView);
    updateDescriptors(1, accumImageView, mlDiffuseImageView, mlSpecularImageView);
}

void NRCManager::updateDescriptors(uint32_t slot, VkImageView accumImageView, VkImageView mlDiffuseImageView, VkImageView mlSpecularImageView) {
    if (accumImageView == VK_NULL_HANDLE || !m_atomicAccumBuffer || slot >= m_resolveDescSets.size()) return;
    VkImageView diffView = (mlDiffuseImageView != VK_NULL_HANDLE) ? mlDiffuseImageView : accumImageView;
    VkImageView specView = (mlSpecularImageView != VK_NULL_HANDLE) ? mlSpecularImageView : diffView;
    VkDescriptorImageInfo imageInfo{ VK_NULL_HANDLE, accumImageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo mlDiffInfo{ VK_NULL_HANDLE, diffView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo mlSpecInfo{ VK_NULL_HANDLE, specView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo atomicAccumInfo{ m_atomicAccumBuffer->getBuffer(), 0, VK_WHOLE_SIZE };

    std::vector<VkWriteDescriptorSet> writes = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_resolveDescSets[slot], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &imageInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_resolveDescSets[slot], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &atomicAccumInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_resolveDescSets[slot], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mlDiffInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_resolveDescSets[slot], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mlSpecInfo, nullptr, nullptr }
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

void NRCManager::createPipelines(const std::vector<char>& inferSpv, const std::vector<char>& trainSpv, const std::vector<char>& resolveSpv) {
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

    // Training Pipeline (64-byte push constants for multi-step bias-corrected Adam)
    VkPushConstantRange trainPcRange{};
    trainPcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    trainPcRange.offset = 0;
    trainPcRange.size = sizeof(uint32_t) * 8 + sizeof(float) * 8; // 64 bytes

    VkPipelineLayoutCreateInfo trainLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    trainLayoutInfo.setLayoutCount = 1;
    trainLayoutInfo.pSetLayouts = &m_trainDescLayout;
    trainLayoutInfo.pushConstantRangeCount = 1;
    trainLayoutInfo.pPushConstantRanges = &trainPcRange;
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

    // Resolve Pipeline (CRIT-05)
    if (!resolveSpv.empty()) {
        VkPushConstantRange resPcRange{};
        resPcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        resPcRange.offset = 0;
        resPcRange.size = sizeof(uint32_t) * 4; // width, height, captureMlData, pad (16 bytes)

        VkPipelineLayoutCreateInfo resLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        resLayoutInfo.setLayoutCount = 1;
        resLayoutInfo.pSetLayouts = &m_resolveDescLayout;
        resLayoutInfo.pushConstantRangeCount = 1;
        resLayoutInfo.pPushConstantRanges = &resPcRange;
        if (vkCreatePipelineLayout(m_device, &resLayoutInfo, nullptr, &m_resolvePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("NRCManager: Failed to create resolve pipeline layout!");
        }

        VkShaderModule resMod = createShaderModule(resolveSpv);
        VkComputePipelineCreateInfo resPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        resPipeInfo.layout = m_resolvePipelineLayout;
        resPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        resPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        resPipeInfo.stage.module = resMod;
        resPipeInfo.stage.pName = "main";
        resPipeInfo.stage.pNext = &subgroupSize32;
        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &resPipeInfo, nullptr, &m_resolvePipeline) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, resMod, nullptr);
            throw std::runtime_error("NRCManager: Failed to create resolve compute pipeline!");
        }
        vkDestroyShaderModule(m_device, resMod, nullptr);
    }
}

void NRCManager::resize(uint32_t width, uint32_t height) {
    if (m_width == width && m_height == height) return;
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }
    Logger::info("NRCManager resized: updating cache resolution to {}x{}", width, height);
    m_width = width;
    m_height = height;
    initBuffers();
    initWeights();

    // Re-bind updated buffer handles to persistent descriptor sets
    VkDescriptorBufferInfo atomicAccumInfo{ m_atomicAccumBuffer->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo queryQueueInfo{ m_queryQueue->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo weightsInfo{ m_weights->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo countersInfo{ m_counters->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo trainQueueInfo{ m_trainQueue->getBuffer(), 0, VK_WHOLE_SIZE };
    VkDescriptorBufferInfo momInfo{ m_weightMomentum->getBuffer(), 0, VK_WHOLE_SIZE };

    std::vector<VkWriteDescriptorSet> writes = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &atomicAccumInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &queryQueueInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightsInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_inferDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &countersInfo, nullptr },

        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &trainQueueInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightsInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &momInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_trainDescSet, 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &countersInfo, nullptr },

        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_resolveDescSets[0], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &atomicAccumInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_resolveDescSets[1], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &atomicAccumInfo, nullptr }
    };
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void NRCManager::resetCounters(VkCommandBuffer cmd) {
    if (m_needsGpuInit) {
        if (!m_initialWeights.empty()) {
            vkCmdUpdateBuffer(cmd, m_weights->getBuffer(), 0,
                              m_initialWeights.size() * sizeof(uint16_t),
                              m_initialWeights.data());
        }
        const VkDeviceSize adamStateBytes = 2ull * 9360ull * sizeof(float);
        vkCmdFillBuffer(cmd, m_weightMomentum->getBuffer(), 0, adamStateBytes, 0);
        if (!m_initialMasterWeights.empty()) {
            vkCmdUpdateBuffer(cmd, m_weightMomentum->getBuffer(), adamStateBytes,
                              m_initialMasterWeights.size() * sizeof(float),
                              m_initialMasterWeights.data());
        }
        vkCmdFillBuffer(cmd, m_atomicAccumBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0);
        m_needsGpuInit = false;
    }

    // Initialize NRCCountersBuffer + VkDispatchIndirectCommand at offset 8:
    // [0]=queryCount(0), [1]=trainCount(0), [2]=dispatchX(0), [3]=dispatchY(1), [4]=dispatchZ(1)
    const uint32_t initCounters[8] = { 0u, 0u, 0u, 1u, 1u, 0u, 0u, 0u };
    vkCmdUpdateBuffer(cmd, m_counters->getBuffer(), 0, sizeof(initCounters), initCounters);

    VkMemoryBarrier2 transferBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    transferBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    transferBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    transferBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
    transferBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
                                    VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
                                    VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

    VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &transferBarrier;
    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void NRCManager::recordInference(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                glm::vec3 worldMin, glm::vec3 worldMax, uint32_t queryCountOverride,
                                uint32_t frameSlot, uint32_t captureMlData, float maxRadiance)
{
    if (!m_initialized || m_inferPipeline == VK_NULL_HANDLE) return;

    // Buffer memory barrier: Wavefront writes query queue & indirect dispatch count -> Inference reads
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
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
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
        float maxRadiance;
        glm::vec4 worldMin;
        glm::vec4 invWorldExtent;
    } pc;
    pc.width = width;
    pc.height = height;
    pc.queryCount = queryCount;
    pc.maxRadiance = (maxRadiance > 0.0f) ? maxRadiance : 10.0f;
    pc.worldMin = glm::vec4(worldMin, 0.0f);
    pc.invWorldExtent = glm::vec4(invExtent, 0.0f);

    vkCmdPushConstants(cmd, m_inferPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // Each workgroup handles 4 Wave32 subgroups = 64 queries.
    // Use GPU-driven vkCmdDispatchIndirect (offset 8 = NRCCountersBuffer::dispatchX) so only active query batches run.
    if (queryCountOverride > 0) {
        uint32_t dispatchCount = (queryCountOverride + 63u) / 64u;
        vkCmdDispatch(cmd, dispatchCount, 1, 1);
    } else {
        vkCmdDispatchIndirect(cmd, m_counters->getBuffer(), 8);
    }

    // Barrier: Inference writes atomic buffer -> Resolve reads and resets atomic buffer (CRIT-05)
    VkBufferMemoryBarrier2 inferToResolveBarrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        m_atomicAccumBuffer->getBuffer(), 0, VK_WHOLE_SIZE
    };
    VkDependencyInfo inferToResolveDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    inferToResolveDep.bufferMemoryBarrierCount = 1;
    inferToResolveDep.pBufferMemoryBarriers = &inferToResolveBarrier;
    vkCmdPipelineBarrier2(cmd, &inferToResolveDep);

    // Resolve: Single-thread-per-pixel atomic buffer -> uAccumImage & uMlDiffuseImage (CRIT-05)
    if (m_resolvePipeline != VK_NULL_HANDLE) {
        uint32_t slot = frameSlot % 2u;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePipelineLayout, 0, 1, &m_resolveDescSets[slot], 0, nullptr);

        struct ResolvePushConstants {
            uint32_t width;
            uint32_t height;
            uint32_t captureMlData;
            uint32_t pad;
        } rpc{ width, height, captureMlData, 0u };
        vkCmdPushConstants(cmd, m_resolvePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rpc), &rpc);
        vkCmdDispatch(cmd, (width + 15u) / 16u, (height + 15u) / 16u, 1);
    }

    // Barrier: Resolve writes uAccumImage / uMlDiffuseImage -> downstream tonemapping / denoiser reads
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

    // Buffer memory barrier: Wavefront writes train records -> Training reads train records,
    // and prior Inference reads weights -> Training updates weights (WAR + WAW syncval).
    std::array<VkBufferMemoryBarrier2, 3> inBarriers = {
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
        },
        VkBufferMemoryBarrier2{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_weights->getBuffer(), 0, VK_WHOLE_SIZE
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

    const uint32_t numSteps = (batchSize > 0u) ? std::clamp((batchSize + 31u) / 32u, 1u, 16u) : 0u;

    struct TrainPushConstants {
        uint32_t trainCount;
        uint32_t batchSize;
        float learningRate;
        uint32_t frameIndex;
        uint32_t adamStepBase;
        uint32_t numSteps;
        uint32_t pad0;
        uint32_t pad1;
        glm::vec4 worldMin;
        glm::vec4 invWorldExtent;
    } pc;
    pc.trainCount = m_maxTrainRecords;
    pc.batchSize = batchSize;
    pc.learningRate = learningRate;
    pc.frameIndex = frameIndex;
    pc.adamStepBase = m_adamStep;
    pc.numSteps = numSteps;
    pc.pad0 = 0u;
    pc.pad1 = 0u;
    pc.worldMin = glm::vec4(worldMin, 0.0f);
    pc.invWorldExtent = glm::vec4(invExtent, 0.0f);

    vkCmdPushConstants(cmd, m_trainPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // nrc_train.comp executes numSteps x 32-sample self-training mini-batches in Workgroup 0.
    uint32_t dispatchCount = (numSteps > 0u && m_maxTrainRecords > 0u) ? 1u : 0u;
    if (dispatchCount > 0) {
        vkCmdDispatch(cmd, dispatchCount, 1, 1);
        m_adamStep += numSteps;
    }

    // Barrier: Training updates weights -> next frame inference reads weights
    VkBufferMemoryBarrier2 weightBarrier{
        VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
        VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
        m_weights->getBuffer(), 0, VK_WHOLE_SIZE
    };
    VkDependencyInfo outDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    outDep.bufferMemoryBarrierCount = 1;
    outDep.pBufferMemoryBarriers = &weightBarrier;
    vkCmdPipelineBarrier2(cmd, &outDep);
}

void NRCManager::recordInferenceAndTraining(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                            glm::vec3 worldMin, glm::vec3 worldMax,
                                            uint32_t queryCountOverride, uint32_t frameSlot,
                                            uint32_t captureMlData, float maxRadiance,
                                            uint32_t frameIndex, float learningRate,
                                            uint32_t batchSize)
{
    if (!m_initialized || m_inferPipeline == VK_NULL_HANDLE) return;

    std::array<VkBufferMemoryBarrier2, 2> preInferBarriers = {
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
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_counters->getBuffer(), 0, 256
        }
    };
    VkDependencyInfo preInferDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    preInferDep.bufferMemoryBarrierCount = static_cast<uint32_t>(preInferBarriers.size());
    preInferDep.pBufferMemoryBarriers = preInferBarriers.data();
    vkCmdPipelineBarrier2(cmd, &preInferDep);

    glm::vec3 extent = worldMax - worldMin;
    glm::vec3 invExtent = glm::vec3(
        extent.x > 1e-4f ? 1.0f / extent.x : 1.0f,
        extent.y > 1e-4f ? 1.0f / extent.y : 1.0f,
        extent.z > 1e-4f ? 1.0f / extent.z : 1.0f
    );

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_inferPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_inferPipelineLayout, 0, 1, &m_inferDescSet, 0, nullptr);

    struct InferPushConstants {
        uint32_t width;
        uint32_t height;
        uint32_t queryCount;
        float maxRadiance;
        glm::vec4 worldMin;
        glm::vec4 invWorldExtent;
    } ipc;
    ipc.width = width;
    ipc.height = height;
    ipc.queryCount = queryCountOverride > 0 ? queryCountOverride : m_maxQueries;
    ipc.maxRadiance = (maxRadiance > 0.0f) ? maxRadiance : 10.0f;
    ipc.worldMin = glm::vec4(worldMin, 0.0f);
    ipc.invWorldExtent = glm::vec4(invExtent, 0.0f);
    vkCmdPushConstants(cmd, m_inferPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(ipc), &ipc);

    if (queryCountOverride > 0) {
        uint32_t dispatchCount = (queryCountOverride + 63u) / 64u;
        vkCmdDispatch(cmd, dispatchCount, 1, 1);
    } else {
        vkCmdDispatchIndirect(cmd, m_counters->getBuffer(), 8);
    }

    // Combined post-infer barrier covering Resolve (atomicAccumBuffer) and Train (trainQueue, counters, weights WAR+WAW)
    std::array<VkBufferMemoryBarrier2, 4> midBarriers = {
        VkBufferMemoryBarrier2{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_atomicAccumBuffer->getBuffer(), 0, VK_WHOLE_SIZE
        },
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
        },
        VkBufferMemoryBarrier2{
            VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, nullptr,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED,
            m_weights->getBuffer(), 0, VK_WHOLE_SIZE
        }
    };
    VkDependencyInfo midDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    midDep.bufferMemoryBarrierCount = static_cast<uint32_t>(midBarriers.size());
    midDep.pBufferMemoryBarriers = midBarriers.data();
    vkCmdPipelineBarrier2(cmd, &midDep);

    // Dispatch 1-WG Training immediately before Resolve (disjoint resources -> no barrier between)
    const uint32_t numSteps = (batchSize > 0u) ? std::clamp((batchSize + 31u) / 32u, 1u, 16u) : 0u;
    if (m_trainPipeline != VK_NULL_HANDLE && numSteps > 0u && m_maxTrainRecords > 0u) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_trainPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_trainPipelineLayout, 0, 1, &m_trainDescSet, 0, nullptr);

        struct TrainPushConstants {
            uint32_t trainCount;
            uint32_t batchSize;
            float learningRate;
            uint32_t frameIndex;
            uint32_t adamStepBase;
            uint32_t numSteps;
            uint32_t pad0;
            uint32_t pad1;
            glm::vec4 worldMin;
            glm::vec4 invWorldExtent;
        } tpc;
        tpc.trainCount = m_maxTrainRecords;
        tpc.batchSize = batchSize;
        tpc.learningRate = learningRate;
        tpc.frameIndex = frameIndex;
        tpc.adamStepBase = m_adamStep;
        tpc.numSteps = numSteps;
        tpc.pad0 = 0u;
        tpc.pad1 = 0u;
        tpc.worldMin = glm::vec4(worldMin, 0.0f);
        tpc.invWorldExtent = glm::vec4(invExtent, 0.0f);
        vkCmdPushConstants(cmd, m_trainPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tpc), &tpc);

        vkCmdDispatch(cmd, 1, 1, 1);
        m_adamStep += numSteps;
    }

    if (m_resolvePipeline != VK_NULL_HANDLE) {
        uint32_t slot = frameSlot % 2u;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePipelineLayout, 0, 1, &m_resolveDescSets[slot], 0, nullptr);

        struct ResolvePushConstants {
            uint32_t width;
            uint32_t height;
            uint32_t captureMlData;
            uint32_t pad;
        } rpc{ width, height, captureMlData, 0u };
        vkCmdPushConstants(cmd, m_resolvePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rpc), &rpc);
        vkCmdDispatch(cmd, (width + 15u) / 16u, (height + 15u) / 16u, 1);
    }

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

} // namespace pathways
