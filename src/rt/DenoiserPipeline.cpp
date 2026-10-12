#include "rt/DenoiserPipeline.hpp"
#include "rt/PostProcessPipeline.hpp"
#include "core/Logger.hpp"

#include <stdexcept>
#include <vector>

namespace pathways {

DenoiserPipeline::DenoiserPipeline(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VmaAllocator allocator,
    VkQueue queue,
    ShaderLoaderFunc shaderLoader
) : m_device(device),
    m_physDevice(physicalDevice),
    m_allocator(allocator),
    m_queue(queue),
    m_shaderLoader(std::move(shaderLoader))
{
    createDescriptorSetLayouts();
    initPipelines();
    Logger::info("DenoiserPipeline (In-House ReLAX/A-SVGF) initialized successfully.");
}

DenoiserPipeline::~DenoiserPipeline() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_temporalPipeline) vkDestroyPipeline(m_device, m_temporalPipeline, nullptr);
        if (m_temporalPipelineLayout) vkDestroyPipelineLayout(m_device, m_temporalPipelineLayout, nullptr);
        if (m_temporalDescLayout) vkDestroyDescriptorSetLayout(m_device, m_temporalDescLayout, nullptr);

        if (m_variancePipeline) vkDestroyPipeline(m_device, m_variancePipeline, nullptr);
        if (m_variancePipelineLayout) vkDestroyPipelineLayout(m_device, m_variancePipelineLayout, nullptr);
        if (m_varianceDescLayout) vkDestroyDescriptorSetLayout(m_device, m_varianceDescLayout, nullptr);

        if (m_atrousPipeline) vkDestroyPipeline(m_device, m_atrousPipeline, nullptr);
        if (m_atrousPipelineLayout) vkDestroyPipelineLayout(m_device, m_atrousPipelineLayout, nullptr);
        if (m_atrousDescLayout) vkDestroyDescriptorSetLayout(m_device, m_atrousDescLayout, nullptr);

        if (m_remodulatePipeline) vkDestroyPipeline(m_device, m_remodulatePipeline, nullptr);
        if (m_remodulatePipelineLayout) vkDestroyPipelineLayout(m_device, m_remodulatePipelineLayout, nullptr);
        if (m_remodulateDescLayout) vkDestroyDescriptorSetLayout(m_device, m_remodulateDescLayout, nullptr);

        if (m_descPool) vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
    }
}

void DenoiserPipeline::createDescriptorSetLayouts() {
    auto makeBinding = [](uint32_t b) -> VkDescriptorSetLayoutBinding {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = b;
        binding.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        return binding;
    };

    // 1. Temporal descriptor set layout (13 storage images: 5 in, 4 hist in, 4 hist out)
    std::vector<VkDescriptorSetLayoutBinding> tempBindings;
    for (uint32_t b = 0; b < 13; ++b) tempBindings.push_back(makeBinding(b));
    VkDescriptorSetLayoutCreateInfo tempInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    tempInfo.bindingCount = static_cast<uint32_t>(tempBindings.size());
    tempInfo.pBindings = tempBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &tempInfo, nullptr, &m_temporalDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create temporal denoiser descriptor set layout");
    }

    // 2. Variance descriptor set layout (7 storage images: 4 hist in, 1 normalDepth in, 2 pingPong out)
    std::vector<VkDescriptorSetLayoutBinding> varBindings;
    for (uint32_t b = 0; b < 7; ++b) varBindings.push_back(makeBinding(b));
    VkDescriptorSetLayoutCreateInfo varInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    varInfo.bindingCount = static_cast<uint32_t>(varBindings.size());
    varInfo.pBindings = varBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &varInfo, nullptr, &m_varianceDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create variance denoiser descriptor set layout");
    }

    // 3. Atrous descriptor set layout (5 storage images: 2 in, 1 normalDepth in, 2 out)
    std::vector<VkDescriptorSetLayoutBinding> atrBindings;
    for (uint32_t b = 0; b < 5; ++b) atrBindings.push_back(makeBinding(b));
    VkDescriptorSetLayoutCreateInfo atrInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    atrInfo.bindingCount = static_cast<uint32_t>(atrBindings.size());
    atrInfo.pBindings = atrBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &atrInfo, nullptr, &m_atrousDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create atrous denoiser descriptor set layout");
    }

    // 4. Remodulate descriptor set layout (5 storage images: 2 in filtered, 1 albedo in, 1 accum in, 1 out radiance)
    std::vector<VkDescriptorSetLayoutBinding> remBindings;
    for (uint32_t b = 0; b < 5; ++b) remBindings.push_back(makeBinding(b));
    VkDescriptorSetLayoutCreateInfo remInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    remInfo.bindingCount = static_cast<uint32_t>(remBindings.size());
    remInfo.pBindings = remBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &remInfo, nullptr, &m_remodulateDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create remodulate denoiser descriptor set layout");
    }

    // Descriptor pool
    VkDescriptorPoolSize poolSize{};
    poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSize.descriptorCount = 128;

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 16;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create denoiser descriptor pool");
    }

    allocateDescriptorSets();
}

void DenoiserPipeline::allocateDescriptorSets() {
    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = m_descPool;

    // Temporal sets (2 slots)
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_temporalDescLayout;
        vkAllocateDescriptorSets(m_device, &allocInfo, &m_temporalDescSets[i]);
    }

    // Variance sets (2 slots)
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_varianceDescLayout;
        vkAllocateDescriptorSets(m_device, &allocInfo, &m_varianceDescSets[i]);
    }

    // Atrous sets (Ping-pong A and B)
    for (uint32_t i = 0; i < 2; ++i) {
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &m_atrousDescLayout;
        vkAllocateDescriptorSets(m_device, &allocInfo, &m_atrousDescSets[i]);
    }

    // Remodulate set
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_remodulateDescLayout;
    vkAllocateDescriptorSets(m_device, &allocInfo, &m_remodulateDescSet);
}

void DenoiserPipeline::initPipelines() {
    auto createComputePipeline = [this](
        const std::string& shaderName,
        VkDescriptorSetLayout descLayout,
        uint32_t pushConstantSize,
        VkPipelineLayout& outLayout,
        VkPipeline& outPipeline
    ) {
        auto code = m_shaderLoader(shaderName);
        VkShaderModuleCreateInfo smInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smInfo.codeSize = code.size();
        smInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());
        VkShaderModule module = VK_NULL_HANDLE;
        if (vkCreateShaderModule(m_device, &smInfo, nullptr, &module) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create denoiser shader module: " + shaderName);
        }

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = pushConstantSize;

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descLayout;
        layoutInfo.pushConstantRangeCount = (pushConstantSize > 0) ? 1 : 0;
        layoutInfo.pPushConstantRanges = (pushConstantSize > 0) ? &pcRange : nullptr;
        if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &outLayout) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, module, nullptr);
            throw std::runtime_error("Failed to create denoiser pipeline layout for: " + shaderName);
        }

        VkComputePipelineCreateInfo compInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        compInfo.layout = outLayout;
        compInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        compInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        compInfo.stage.module = module;
        compInfo.stage.pName = "main";

        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &compInfo, nullptr, &outPipeline) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, module, nullptr);
            throw std::runtime_error("Failed to create denoiser compute pipeline for: " + shaderName);
        }
        vkDestroyShaderModule(m_device, module, nullptr);
    };

    createComputePipeline("denoise_temporal.comp.spv", m_temporalDescLayout, sizeof(TemporalPushConstants), m_temporalPipelineLayout, m_temporalPipeline);
    createComputePipeline("denoise_variance.comp.spv", m_varianceDescLayout, sizeof(VariancePushConstants), m_variancePipelineLayout, m_variancePipeline);
    createComputePipeline("denoise_atrous.comp.spv", m_atrousDescLayout, sizeof(AtrousPushConstants), m_atrousPipelineLayout, m_atrousPipeline);
    createComputePipeline("denoise_remodulate.comp.spv", m_remodulateDescLayout, sizeof(RemodulatePushConstants), m_remodulatePipelineLayout, m_remodulatePipeline);
}

void DenoiserPipeline::resize(uint32_t width, uint32_t height, AccumFormat accumFormat, VkCommandBuffer cmd) {
    if (m_width == width && m_height == height && m_accumFormat == accumFormat && m_denoisedImage) {
        return;
    }

    m_width = width;
    m_height = height;
    m_accumFormat = accumFormat;
    m_descriptorsBound = false;

    VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_historyDiffuse[i] = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
        m_historySpecular[i] = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
        m_historyMoments[i] = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
        m_historyLength[i] = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16_SFLOAT, usage);
        m_historyNormalDepth[i] = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
    }

    for (uint32_t i = 0; i < 2; ++i) {
        m_pingPongDiffuse[i] = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
        m_pingPongSpecular[i] = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
    }

    VkFormat outFmt = (accumFormat == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
    m_denoisedImage = std::make_unique<Image>(m_device, m_allocator, width, height, outFmt, usage);

    if (cmd != VK_NULL_HANDLE) {
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &beginInfo);

        // Initial transition to VK_IMAGE_LAYOUT_GENERAL and clear history to zero
        VkClearColorValue clearZero = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            m_historyDiffuse[i]->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_historySpecular[i]->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_historyMoments[i]->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_historyLength[i]->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_historyNormalDepth[i]->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

            vkCmdClearColorImage(cmd, m_historyDiffuse[i]->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
            vkCmdClearColorImage(cmd, m_historySpecular[i]->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
            vkCmdClearColorImage(cmd, m_historyMoments[i]->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
            vkCmdClearColorImage(cmd, m_historyLength[i]->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
            vkCmdClearColorImage(cmd, m_historyNormalDepth[i]->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
        }
        for (uint32_t i = 0; i < 2; ++i) {
            m_pingPongDiffuse[i]->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            m_pingPongSpecular[i]->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        }
        m_denoisedImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = cmd;

        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_queue);
    }

    Logger::info("DenoiserPipeline allocated images ({}x{}, format {})", width, height, static_cast<int>(accumFormat));
}

void DenoiserPipeline::initTonemapDescriptors(PostProcessPipeline* postProcess, Image* outputImage) {
    if (!postProcess || !outputImage || !m_denoisedImage) return;

    if (m_tonemapDescSet == VK_NULL_HANDLE) {
        VkDescriptorSetLayout tmLayout = postProcess->getTonemapDescLayout();
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_descPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &tmLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_tonemapDescSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate denoiser tonemap descriptor set");
        }
    }

    postProcess->updateTonemapDescriptors(m_tonemapDescSet, m_denoisedImage.get(), outputImage);
}

void DenoiserPipeline::updateDescriptors(
    Image* currentDiffuse,
    Image* currentSpecular,
    Image* motionVectors,
    Image* normalDepth,
    Image* albedoRoughness,
    Image* accumImage
) {
    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> imgInfos;
    imgInfos.reserve(64);

    auto addImageWrite = [&](VkDescriptorSet dstSet, uint32_t dstBinding, VkImageView view) {
        imgInfos.push_back({ VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL });
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = dstSet;
        w.dstBinding = dstBinding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &imgInfos.back();
        writes.push_back(w);
    };

    // 1. Temporal sets (slot 0 reads hist 1, writes hist 0; slot 1 reads hist 0, writes hist 1)
    for (uint32_t slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
        uint32_t prevSlot = 1 - slot;
        VkDescriptorSet set = m_temporalDescSets[slot];
        addImageWrite(set, 0, currentDiffuse->getImageView());
        addImageWrite(set, 1, currentSpecular->getImageView());
        addImageWrite(set, 2, motionVectors->getImageView());
        addImageWrite(set, 3, normalDepth->getImageView());
        addImageWrite(set, 4, m_historyNormalDepth[prevSlot]->getImageView());

        addImageWrite(set, 5, m_historyDiffuse[prevSlot]->getImageView());
        addImageWrite(set, 6, m_historySpecular[prevSlot]->getImageView());
        addImageWrite(set, 7, m_historyMoments[prevSlot]->getImageView());
        addImageWrite(set, 8, m_historyLength[prevSlot]->getImageView());

        addImageWrite(set, 9, m_historyDiffuse[slot]->getImageView());
        addImageWrite(set, 10, m_historySpecular[slot]->getImageView());
        addImageWrite(set, 11, m_historyMoments[slot]->getImageView());
        addImageWrite(set, 12, m_historyLength[slot]->getImageView());
    }

    // 2. Variance sets (slot reads history[slot], writes to pingPong[0])
    for (uint32_t slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
        VkDescriptorSet set = m_varianceDescSets[slot];
        addImageWrite(set, 0, m_historyDiffuse[slot]->getImageView());
        addImageWrite(set, 1, m_historySpecular[slot]->getImageView());
        addImageWrite(set, 2, m_historyMoments[slot]->getImageView());
        addImageWrite(set, 3, m_historyLength[slot]->getImageView());
        addImageWrite(set, 4, normalDepth->getImageView());
        addImageWrite(set, 5, m_pingPongDiffuse[0]->getImageView());
        addImageWrite(set, 6, m_pingPongSpecular[0]->getImageView());
    }

    // 3. Atrous sets: Set 0 (in: pingPong 0, out: pingPong 1); Set 1 (in: pingPong 1, out: pingPong 0)
    for (uint32_t s = 0; s < 2; ++s) {
        uint32_t inIdx = s;
        uint32_t outIdx = 1 - s;
        VkDescriptorSet set = m_atrousDescSets[s];
        addImageWrite(set, 0, m_pingPongDiffuse[inIdx]->getImageView());
        addImageWrite(set, 1, m_pingPongSpecular[inIdx]->getImageView());
        addImageWrite(set, 2, normalDepth->getImageView());
        addImageWrite(set, 3, m_pingPongDiffuse[outIdx]->getImageView());
        addImageWrite(set, 4, m_pingPongSpecular[outIdx]->getImageView());
    }

    // 4. Remodulate set: reads pingPong[0], albedoRoughness, accumImage, writes m_denoisedImage
    addImageWrite(m_remodulateDescSet, 0, m_pingPongDiffuse[0]->getImageView());
    addImageWrite(m_remodulateDescSet, 1, m_pingPongSpecular[0]->getImageView());
    addImageWrite(m_remodulateDescSet, 2, albedoRoughness->getImageView());
    addImageWrite(m_remodulateDescSet, 3, accumImage->getImageView());
    addImageWrite(m_remodulateDescSet, 4, m_denoisedImage->getImageView());

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    m_descriptorsBound = true;
}

void DenoiserPipeline::recordDenoise(
    VkCommandBuffer cmd,
    uint32_t frameSlot,
    bool resetHistory,
    uint32_t accumulatedSamples,
    float crossfadeBlend,
    Image* currentDiffuse,
    Image* currentSpecular,
    Image* motionVectors,
    Image* normalDepth,
    Image* albedoRoughness,
    Image* accumImage
) {
    if (!isInitialized() || !currentDiffuse || !currentSpecular || !motionVectors ||
        !normalDepth || !albedoRoughness || !accumImage) {
        return;
    }

    if (!m_descriptorsBound) {
        updateDescriptors(currentDiffuse, currentSpecular, motionVectors, normalDepth,
                          albedoRoughness, accumImage);
    }

    uint32_t gx = (m_width + 7) / 8;
    uint32_t gy = (m_height + 7) / 8;

    auto insertComputeBarrier = [&cmd]() {
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    };

    // -------------------------------------------------------------------------
    // Pass 1: Temporal Reprojection & Accumulation
    // -------------------------------------------------------------------------
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipelineLayout,
                            0, 1, &m_temporalDescSets[frameSlot], 0, nullptr);

    TemporalPushConstants tempPC{};
    tempPC.width = m_width;
    tempPC.height = m_height;
    tempPC.resetHistory = resetHistory ? 1u : 0u;
    tempPC.maxHistoryLength = 32.0f;
    vkCmdPushConstants(cmd, m_temporalPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(TemporalPushConstants), &tempPC);

    vkCmdDispatch(cmd, gx, gy, 1);
    insertComputeBarrier();

    // -------------------------------------------------------------------------
    // Pass 2: Spatial Variance Estimation & Disocclusion History Fix
    // -------------------------------------------------------------------------
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_variancePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_variancePipelineLayout,
                            0, 1, &m_varianceDescSets[frameSlot], 0, nullptr);

    VariancePushConstants varPC{};
    varPC.width = m_width;
    varPC.height = m_height;
    vkCmdPushConstants(cmd, m_variancePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(VariancePushConstants), &varPC);

    vkCmdDispatch(cmd, gx, gy, 1);
    insertComputeBarrier();

    // -------------------------------------------------------------------------
    // Pass 3: Multi-Iteration À-Trous Wavelet Filter (4 iterations: 1, 2, 4, 8)
    // -------------------------------------------------------------------------
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousPipeline);

    const int32_t stepSizes[4] = { 1, 2, 4, 8 };
    for (uint32_t it = 0; it < 4; ++it) {
        uint32_t setIdx = it % 2; // Even it: read 0 -> write 1; Odd it: read 1 -> write 0
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousPipelineLayout,
                                0, 1, &m_atrousDescSets[setIdx], 0, nullptr);

        AtrousPushConstants atrPC{};
        atrPC.width = m_width;
        atrPC.height = m_height;
        atrPC.stepSize = stepSizes[it];
        atrPC.phiNormal = 32.0f;
        atrPC.phiDepth = 0.1f;
        atrPC.phiLuminance = 4.0f;
        atrPC.iteration = it;
        vkCmdPushConstants(cmd, m_atrousPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(AtrousPushConstants), &atrPC);

        vkCmdDispatch(cmd, gx, gy, 1);
        insertComputeBarrier();
    }
    // After 4 iterations (it=0..3), the final result is in pingPong[0]!

    // -------------------------------------------------------------------------
    // Pass 4: Albedo Remodulation & Unbiased Accumulation Crossfade
    // -------------------------------------------------------------------------
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_remodulatePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_remodulatePipelineLayout,
                            0, 1, &m_remodulateDescSet, 0, nullptr);

    RemodulatePushConstants remPC{};
    remPC.width = m_width;
    remPC.height = m_height;
    remPC.accumulatedSamples = accumulatedSamples;
    remPC.crossfadeBlend = crossfadeBlend;
    vkCmdPushConstants(cmd, m_remodulatePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                       0, sizeof(RemodulatePushConstants), &remPC);

    vkCmdDispatch(cmd, gx, gy, 1);

    // Final barrier to guarantee tonemapping reads the finished denoised image
    VkMemoryBarrier2 finalBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    finalBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    finalBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    finalBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    finalBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo finalDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    finalDep.memoryBarrierCount = 1;
    finalDep.pMemoryBarriers = &finalBarrier;
    vkCmdPipelineBarrier2(cmd, &finalDep);

    // Copy current normalDepth into history for next frame's reprojection
    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.extent = { m_width, m_height, 1 };

    VkImageMemoryBarrier2 imgBarriers[2]{};
    imgBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    imgBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    imgBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    imgBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    imgBarriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
    imgBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarriers[0].image = normalDepth->getImage();
    imgBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    imgBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    imgBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    imgBarriers[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    imgBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    imgBarriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    imgBarriers[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    imgBarriers[1].image = m_historyNormalDepth[frameSlot]->getImage();
    imgBarriers[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo copyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    copyDep.imageMemoryBarrierCount = 2;
    copyDep.pImageMemoryBarriers = imgBarriers;
    vkCmdPipelineBarrier2(cmd, &copyDep);

    vkCmdCopyImage(cmd,
                   normalDepth->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                   m_historyNormalDepth[frameSlot]->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                   1, &copyRegion);

    VkImageMemoryBarrier2 postBarrier{};
    postBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    postBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    postBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarrier.image = m_historyNormalDepth[frameSlot]->getImage();
    postBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    postDep.imageMemoryBarrierCount = 1;
    postDep.pImageMemoryBarriers = &postBarrier;
    vkCmdPipelineBarrier2(cmd, &postDep);
}

} // namespace pathways
