#include "UpwaysPipeline.hpp"
#include "core/Logger.hpp"
#include "upways_weights.hpp"

#include <fstream>
#include <filesystem>
#include <cstring>
#include <stdexcept>

namespace pathways {

UpwaysPipeline::UpwaysPipeline(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VmaAllocator allocator,
    uint32_t inputWidth,
    uint32_t inputHeight,
    uint32_t outputWidth,
    uint32_t outputHeight,
    const std::vector<char>& shaderSpv,
    const std::string& weightsPath,
    bool enableSuperRes,
    VkFormat imageFormat
) : m_device(device),
    m_physDevice(physicalDevice),
    m_allocator(allocator),
    m_inputWidth(inputWidth),
    m_inputHeight(inputHeight),
    m_outputWidth(outputWidth),
    m_outputHeight(outputHeight),
    m_superRes(enableSuperRes),
    m_format(imageFormat)
{
    Logger::info("UpwaysPipeline initializing: Input {}x{}, Output {}x{} (SuperRes: {}), Format: {}",
                 m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight,
                 m_superRes ? "Enabled" : "1.0x (Native)", static_cast<int>(m_format));

    initBuffers(weightsPath);
    initImages();
    createDescriptorSetLayout();
    allocateDescriptorSets();
    createPipeline(shaderSpv);

    m_initialized = true;
    Logger::info("UpwaysPipeline successfully initialized with Wave32 WMMA cooperative matrix support.");
}

UpwaysPipeline::~UpwaysPipeline() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_pipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_pipeline, nullptr);
            m_pipeline = VK_NULL_HANDLE;
        }
        if (m_pipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_pipelineLayout, nullptr);
            m_pipelineLayout = VK_NULL_HANDLE;
        }
        if (m_descLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_descLayout, nullptr);
            m_descLayout = VK_NULL_HANDLE;
        }
        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }
    }
}

void UpwaysPipeline::initBuffers(const std::string& weightsPath) {
    std::vector<std::string> candidates;
    if (!weightsPath.empty()) {
        candidates.push_back(weightsPath);
    }
    candidates.push_back("/home/naoki/Development/Upways/checkpoints/neural_reconstruct_run/upways_weights.bin");
    candidates.push_back("data/models/upways_weights.bin");
    candidates.push_back("../data/models/upways_weights.bin");
    candidates.push_back("../../data/models/upways_weights.bin");
    candidates.push_back("../../../data/models/upways_weights.bin");
    candidates.push_back("/home/naoki/Development/Upways/checkpoints/upways3_multiscale_kpn/vulkan_export/upways_v3_weights.bin");
    candidates.push_back("/home/naoki/Development/Upways/checkpoints/run_multiscene_superres/vulkan_export/upways_weights.bin");
    candidates.push_back("upways_weights.bin");

    std::string foundPath;
    for (const auto& p : candidates) {
        if (std::filesystem::exists(p)) {
            foundPath = p;
            break;
        }
    }

    std::vector<uint8_t> weightBytes;
    if (!foundPath.empty()) {
        std::ifstream file(foundPath, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            size_t size = static_cast<size_t>(file.tellg());
            file.seekg(0, std::ios::beg);
            weightBytes.resize(size);
            file.read(reinterpret_cast<char*>(weightBytes.data()), size);
            file.close();
            Logger::info("UpwaysPipeline loaded {} bytes of neural weights from '{}'", size, foundPath);
        }
    }

    if (weightBytes.empty()) {
        Logger::warn("UpwaysPipeline could not find weights file. Initializing default weight buffer ({} bytes)",
                     upways::TOTAL_WEIGHT_BUFFER_SIZE);
        weightBytes.resize(upways::TOTAL_WEIGHT_BUFFER_SIZE, 0);
    }

    // Ensure minimum size matches total weight buffer definition
    if (weightBytes.size() < upways::TOTAL_WEIGHT_BUFFER_SIZE) {
        weightBytes.resize(upways::TOTAL_WEIGHT_BUFFER_SIZE, 0);
    }

    VkDeviceSize bufferSize = weightBytes.size();
    m_weightBuffer = std::make_unique<Buffer>(
        m_allocator,
        bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_CPU_TO_GPU
    );

    void* mapped = m_weightBuffer->map();
    if (mapped) {
        std::memcpy(mapped, weightBytes.data(), bufferSize);
        m_weightBuffer->unmap();
    }
}

void UpwaysPipeline::initImages() {
    m_outputImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_outputWidth, m_outputHeight,
        m_format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    for (int i = 0; i < 2; ++i) {
        m_diffHistoryImages[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_outputWidth, m_outputHeight,
            m_format,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
        m_specHistoryImages[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_outputWidth, m_outputHeight,
            m_format,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
    }
    m_initialLayoutsTransitioned = false;
}

void UpwaysPipeline::transitionInitialLayouts(VkCommandBuffer cmd) {
    if (m_initialLayoutsTransitioned) {
        return;
    }

    if (m_outputImage) {
        m_outputImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }

    for (int i = 0; i < 2; ++i) {
        if (m_diffHistoryImages[i]) {
            m_diffHistoryImages[i]->transitionLayout(
                cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
            );
        }
        if (m_specHistoryImages[i]) {
            m_specHistoryImages[i]->transitionLayout(
                cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
            );
        }
    }

    m_initialLayoutsTransitioned = true;
}

void UpwaysPipeline::createDescriptorSetLayout() {
    // 13 Bindings:
    // 0: uAccumImage
    // 1: uNormalDepthImage
    // 2: uMotionVectorImage
    // 3: uAlbedoRoughnessImage
    // 4: uSpecularMotionImage
    // 5: uDiffuseImage
    // 6: uSpecularImage
    // 7: uDiffHistoryImage (read)
    // 8: uSpecHistoryImage (read)
    // 9: uOutputImage (write)
    // 10: uDiffHistoryOutputImage (write)
    // 11: uSpecHistoryOutputImage (write)
    // 12: uWeightBuffer
    std::vector<VkDescriptorSetLayoutBinding> bindings(13);
    for (uint32_t i = 0; i < 12; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    bindings[12].binding = 12;
    bindings[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[12].descriptorCount = 1;
    bindings[12].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayout) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create descriptor set layout!");
    }
}

void UpwaysPipeline::allocateDescriptorSets() {
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 24; // 12 per set * 2
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 2;

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create descriptor pool!");
    }

    VkDescriptorSetLayout layouts[2] = { m_descLayout, m_descLayout };
    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = m_descriptorPool;
    allocInfo.descriptorSetCount = 2;
    allocInfo.pSetLayouts = layouts;

    if (vkAllocateDescriptorSets(m_device, &allocInfo, m_descSets) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to allocate descriptor sets!");
    }
}

void UpwaysPipeline::updateDescriptors(
    VkImageView accumImageView,
    VkImageView normalDepthImageView,
    VkImageView motionVectorImageView,
    VkImageView albedoRoughnessImageView,
    VkImageView specularMotionImageView,
    VkImageView diffuseImageView,
    VkImageView specularImageView
) {
    if (accumImageView == VK_NULL_HANDLE || !m_outputImage ||
        !m_diffHistoryImages[0] || !m_diffHistoryImages[1] ||
        !m_specHistoryImages[0] || !m_specHistoryImages[1] ||
        !m_weightBuffer) {
        return;
    }

    VkDescriptorImageInfo accumInfo{ VK_NULL_HANDLE, accumImageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo ndInfo{ VK_NULL_HANDLE, normalDepthImageView ? normalDepthImageView : accumImageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo mvInfo{ VK_NULL_HANDLE, motionVectorImageView ? motionVectorImageView : accumImageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo arInfo{ VK_NULL_HANDLE, albedoRoughnessImageView ? albedoRoughnessImageView : accumImageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo smInfo{ VK_NULL_HANDLE, specularMotionImageView ? specularMotionImageView : accumImageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo diffInfo{ VK_NULL_HANDLE, diffuseImageView ? diffuseImageView : accumImageView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo specInfo{ VK_NULL_HANDLE, specularImageView ? specularImageView : accumImageView, VK_IMAGE_LAYOUT_GENERAL };

    VkDescriptorImageInfo outInfo{ VK_NULL_HANDLE, m_outputImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo weightBufInfo{ m_weightBuffer->getBuffer(), 0, m_weightBuffer->getSize() };

    for (int pingPong = 0; pingPong < 2; ++pingPong) {
        int readSlot = pingPong;
        int writeSlot = 1 - pingPong;

        VkDescriptorImageInfo diffHistReadInfo{ VK_NULL_HANDLE, m_diffHistoryImages[readSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo specHistReadInfo{ VK_NULL_HANDLE, m_specHistoryImages[readSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo diffHistWriteInfo{ VK_NULL_HANDLE, m_diffHistoryImages[writeSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo specHistWriteInfo{ VK_NULL_HANDLE, m_specHistoryImages[writeSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

        std::vector<VkWriteDescriptorSet> writes;
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &accumInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ndInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mvInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &arInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &smInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &diffInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &specInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &diffHistReadInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 8, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &specHistReadInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 9, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &diffHistWriteInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &specHistWriteInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufInfo, nullptr });

        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

VkShaderModule UpwaysPipeline::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create shader module!");
    }
    return shaderModule;
}

void UpwaysPipeline::createPipeline(const std::vector<char>& shaderSpv) {
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(UpwaysPushConstants);

    VkPipelineLayoutCreateInfo pipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &m_descLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayoutInfo.pPushConstantRanges = &pcRange;

    if (vkCreatePipelineLayout(m_device, &pipelineLayoutInfo, nullptr, &m_pipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create compute pipeline layout!");
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    VkShaderModule compModule = createShaderModule(shaderSpv);
    VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipelineInfo.layout = m_pipelineLayout;
    pipelineInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipelineInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipelineInfo.stage.module = compModule;
    pipelineInfo.stage.pName = "main";
    pipelineInfo.stage.pNext = &subgroupSize32;

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_pipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        throw std::runtime_error("UpwaysPipeline: Failed to create compute pipeline!");
    }
    vkDestroyShaderModule(m_device, compModule, nullptr);
}

void UpwaysPipeline::resize(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight, bool enableSuperRes) {
    m_inputWidth = inputWidth;
    m_inputHeight = inputHeight;
    m_outputWidth = outputWidth;
    m_outputHeight = outputHeight;
    m_superRes = enableSuperRes;

    initImages();
    // Note: Descriptors are updated by the host Engine via updateDescriptors() with freshly created image views.
    Logger::info("UpwaysPipeline resized: Input {}x{}, Output {}x{} (SuperRes: {}), Format: {}",
                 m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight,
                 m_superRes ? "Enabled" : "1.0x (Native)", static_cast<int>(m_format));
}

void UpwaysPipeline::recordFrame(
    VkCommandBuffer cmd,
    uint32_t frameIndex,
    bool resetHistory,
    bool cameraMoved,
    int32_t tileOffsetX,
    int32_t tileOffsetY,
    int32_t tileWidth,
    int32_t tileHeight,
    int32_t apronWidth,
    uint32_t totalSamples
) {
    if (!m_pipeline || !m_outputImage) return;

    if (!m_initialLayoutsTransitioned) {
        transitionInitialLayouts(cmd);
    }

    uint32_t activeSet = m_pingPongIndex;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_pipelineLayout, 0, 1, &m_descSets[activeSet], 0, nullptr);

    UpwaysPushConstants pc{};
    pc.inputWidth = static_cast<int32_t>(m_inputWidth);
    pc.inputHeight = static_cast<int32_t>(m_inputHeight);
    pc.outputWidth = static_cast<int32_t>(m_outputWidth);
    pc.outputHeight = static_cast<int32_t>(m_outputHeight);
    pc.invInputWidth = 1.0f / static_cast<float>(m_inputWidth);
    pc.invInputHeight = 1.0f / static_cast<float>(m_inputHeight);
    pc.invOutputWidth = 1.0f / static_cast<float>(m_outputWidth);
    pc.invOutputHeight = 1.0f / static_cast<float>(m_outputHeight);
    pc.tileOffsetX = tileOffsetX;
    pc.tileOffsetY = tileOffsetY;
    pc.tileWidth = (tileWidth > 0) ? tileWidth : static_cast<int32_t>(m_outputWidth);
    pc.tileHeight = (tileHeight > 0) ? tileHeight : static_cast<int32_t>(m_outputHeight);
    pc.apronWidth = apronWidth;
    pc.scaleFactorX = m_superRes ? (static_cast<float>(m_outputWidth) / static_cast<float>(m_inputWidth)) : 1.0f;
    pc.scaleFactorY = m_superRes ? (static_cast<float>(m_outputHeight) / static_cast<float>(m_inputHeight)) : 1.0f;
    pc.frameIndex = frameIndex;
    pc.resetHistory = resetHistory ? 1u : 0u;
    pc.cameraMoved = cameraMoved ? 1u : 0u;
    pc.superResMode = m_superRes ? 1u : 0u;
    pc.blendAlpha = 0.08f;
    pc.minTau = 0.08f;
    pc.learnedDemod = 1u;
    pc.totalSamples = std::max(totalSamples, 1u);
    pc.invTotalSamples = 1.0f; // Input accumulation buffer is already normalized running average

    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // Wave32 workgroups: each workgroup processes 16 output pixels
    uint32_t totalPixels = m_outputWidth * m_outputHeight;
    uint32_t groupsX = (totalPixels + 15) / 16;
    vkCmdDispatch(cmd, groupsX, 1, 1);

    // Memory barriers ensuring downstream compute (tonemapper) and subsequent history reads observe writes
    VkImageMemoryBarrier2 postBarriers[3]{};
    postBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    postBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    postBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    postBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarriers[0].image = m_outputImage->getImage();
    postBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    uint32_t outHistSlot = 1 - activeSet;
    postBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    postBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarriers[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    postBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarriers[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    postBarriers[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarriers[1].image = m_diffHistoryImages[outHistSlot]->getImage();
    postBarriers[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    postBarriers[2].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    postBarriers[2].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarriers[2].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    postBarriers[2].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarriers[2].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    postBarriers[2].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarriers[2].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarriers[2].image = m_specHistoryImages[outHistSlot]->getImage();
    postBarriers[2].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = 3;
    dep.pImageMemoryBarriers = postBarriers;
    vkCmdPipelineBarrier2(cmd, &dep);

    // Advance ping-pong slot
    m_pingPongIndex = 1 - m_pingPongIndex;
}

} // namespace pathways
