#include "UpwaysPipeline.hpp"
#include "core/Logger.hpp"
#include "upways_weights.hpp"

#include <fstream>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <algorithm>

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
        if (m_historySampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_device, m_historySampler, nullptr);
            m_historySampler = VK_NULL_HANDLE;
        }
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
    candidates.push_back("data/models/upways_weights.bin");
    candidates.push_back("../data/models/upways_weights.bin");
    candidates.push_back("../../data/models/upways_weights.bin");
    candidates.push_back("../../../data/models/upways_weights.bin");
    candidates.push_back("/home/naoki/Development/Pathways/data/models/upways_weights.bin");
    candidates.push_back("/home/naoki/Development/Upways/checkpoints/upways_kpn_production/vulkan_export/upways_kpn_weights.bin");
    candidates.push_back("/home/naoki/Development/Upways/checkpoints/neural_reconstruct_run/upways_weights.bin");
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

    m_confidenceImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    for (int i = 0; i < 2; ++i) {
        m_diffHistoryImages[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_outputWidth, m_outputHeight,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
        m_specHistoryImages[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_outputWidth, m_outputHeight,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
        m_normHistoryImages[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_outputWidth, m_outputHeight,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
    }

    if (m_historySampler == VK_NULL_HANDLE) {
        VkSamplerCreateInfo samplerInfo{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
        samplerInfo.magFilter = VK_FILTER_LINEAR;
        samplerInfo.minFilter = VK_FILTER_LINEAR;
        samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
        samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
        samplerInfo.minLod = 0.0f;
        samplerInfo.maxLod = 0.0f;
        if (vkCreateSampler(m_device, &samplerInfo, nullptr, &m_historySampler) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysPipeline: Failed to create history sampler!");
        }
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

    if (m_confidenceImage) {
        m_confidenceImage->transitionLayout(
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
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
            );
        }
        if (m_specHistoryImages[i]) {
            m_specHistoryImages[i]->transitionLayout(
                cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
            );
        }
        if (m_normHistoryImages[i]) {
            m_normHistoryImages[i]->transitionLayout(
                cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
            );
        }
    }

    m_initialLayoutsTransitioned = true;
}

void UpwaysPipeline::createDescriptorSetLayout() {
    // 17 Bindings matching neural_reconstruct.comp:
    // 0: uDemodDiffuse (image2D, readonly, rgba16f)
    // 1: uDemodSpecular (image2D, readonly, rgba16f)
    // 2: uNormalDepth (image2D, readonly, rgba16f)
    // 3: uAlbedoRoughness (image2D, readonly, rgba16f)
    // 4: uSurfaceMV (image2D, readonly, rg16f)
    // 5: uSpecularMV (image2D, readonly, rgba16f)
    // 6: uReStirMetadata (image2D, readonly, rgba16f)
    // 7: uHistoryLinearDiff (sampler2D)
    // 8: uHistoryLinearSpec (sampler2D)
    // 9: uHistoryWorldNormal (sampler2D)
    // 10: uDisplayAlbedo (image2D, readonly, rgba16f)
    // 11: uDisplayNormals (image2D, readonly, rgba16f)
    // 12: uFinalOutput (image2D, writeonly, rgba16f)
    // 13: uOutHistoryDiff (image2D, writeonly, rgba16f)
    // 14: uOutHistorySpec (image2D, writeonly, rgba16f)
    // 15: uOutHistoryNormal (image2D, writeonly, rgba16f)
    // 16: u_Weights (storage buffer, std430)
    std::vector<VkDescriptorSetLayoutBinding> bindings(17);
    for (uint32_t i = 0; i < 7; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    for (uint32_t i = 7; i < 10; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    for (uint32_t i = 10; i < 16; ++i) {
        bindings[i].binding = i;
        bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[i].descriptorCount = 1;
        bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    bindings[16].binding = 16;
    bindings[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    bindings[16].descriptorCount = 1;
    bindings[16].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_descLayout) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create descriptor set layout!");
    }
}

void UpwaysPipeline::allocateDescriptorSets() {
    VkDescriptorPoolSize poolSizes[3]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 13 * 2; // 13 storage images per set * 2 sets
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = 3 * 2;  // 3 samplers per set * 2 sets
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[2].descriptorCount = 1 * 2;  // 1 buffer per set * 2 sets

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 3;
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
    VkImageView demodDiffuseView,
    VkImageView demodSpecularView,
    VkImageView normalDepthView,
    VkImageView albedoRoughnessView,
    VkImageView surfaceMotionView,
    VkImageView specularMotionView,
    VkImageView restirMetadataView,
    VkImageView displayAlbedoView,
    VkImageView displayNormalsView
) {
    if (!m_outputImage || !m_weightBuffer) return;

    VkImageView fallbackView = m_outputImage->getImageView();
    VkImageView dDiffView = (demodDiffuseView != VK_NULL_HANDLE) ? demodDiffuseView : fallbackView;
    VkImageView dSpecView = (demodSpecularView != VK_NULL_HANDLE) ? demodSpecularView : fallbackView;
    VkImageView ndView    = (normalDepthView != VK_NULL_HANDLE) ? normalDepthView : fallbackView;
    VkImageView arView    = (albedoRoughnessView != VK_NULL_HANDLE) ? albedoRoughnessView : fallbackView;
    VkImageView smView    = (surfaceMotionView != VK_NULL_HANDLE) ? surfaceMotionView : fallbackView;
    VkImageView specMView = (specularMotionView != VK_NULL_HANDLE) ? specularMotionView : smView;
    VkImageView restirView = (restirMetadataView != VK_NULL_HANDLE) ? restirMetadataView : specMView;
    VkImageView dispAlbView = (displayAlbedoView != VK_NULL_HANDLE) ? displayAlbedoView : arView;
    VkImageView dispNormView = (displayNormalsView != VK_NULL_HANDLE) ? displayNormalsView : ndView;

    VkDescriptorImageInfo dDiffInfo{ VK_NULL_HANDLE, dDiffView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo dSpecInfo{ VK_NULL_HANDLE, dSpecView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo ndInfo{ VK_NULL_HANDLE, ndView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo arInfo{ VK_NULL_HANDLE, arView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo smInfo{ VK_NULL_HANDLE, smView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo specMInfo{ VK_NULL_HANDLE, specMView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo restirInfo{ VK_NULL_HANDLE, restirView, VK_IMAGE_LAYOUT_GENERAL };

    VkDescriptorImageInfo dispAlbInfo{ VK_NULL_HANDLE, dispAlbView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo dispNormInfo{ VK_NULL_HANDLE, dispNormView, VK_IMAGE_LAYOUT_GENERAL };

    VkDescriptorImageInfo outInfo{ VK_NULL_HANDLE, m_outputImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorBufferInfo weightBufInfo{ m_weightBuffer->getBuffer(), 0, m_weightBuffer->getSize() };

    for (int pingPong = 0; pingPong < 2; ++pingPong) {
        int readSlot = pingPong;
        int writeSlot = 1 - pingPong;

        VkDescriptorImageInfo diffHistReadInfo{ m_historySampler, m_diffHistoryImages[readSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo specHistReadInfo{ m_historySampler, m_specHistoryImages[readSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo normHistReadInfo{ m_historySampler, m_normHistoryImages[readSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

        VkDescriptorImageInfo diffHistWriteInfo{ VK_NULL_HANDLE, m_diffHistoryImages[writeSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo specHistWriteInfo{ VK_NULL_HANDLE, m_specHistoryImages[writeSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo normHistWriteInfo{ VK_NULL_HANDLE, m_normHistoryImages[writeSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

        std::vector<VkWriteDescriptorSet> writes;
        // 0..6: Render-res inputs
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dDiffInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dSpecInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &ndInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &arInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &smInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &specMInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &restirInfo, nullptr, nullptr });

        // 7..9: Display-res history samplers
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 7, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &diffHistReadInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 8, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &specHistReadInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 9, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &normHistReadInfo, nullptr, nullptr });

        // 10..11: Native display guides
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 10, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dispAlbInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 11, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dispNormInfo, nullptr, nullptr });

        // 12..15: Output targets
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 12, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 13, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &diffHistWriteInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 14, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &specHistWriteInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 15, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normHistWriteInfo, nullptr, nullptr });

        // 16: KPN Neural Weights SSBO
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_descSets[pingPong], 16, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufInfo, nullptr });

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
    Logger::info("UpwaysPipeline resized: Input {}x{}, Output {}x{} (SuperRes: {}), Format: {}",
                 m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight,
                 m_superRes ? "Enabled" : "1.0x (Native)", static_cast<int>(m_format));
}

void UpwaysPipeline::recordFrame(
    VkCommandBuffer cmd,
    uint32_t frameIndex,
    bool resetHistory,
    bool cameraMoved,
    const glm::mat4& currInvView,
    const glm::mat4& prevViewProj,
    const glm::mat4& invProj,
    const glm::mat4& prevView,
    const glm::vec2& jitterOffset,
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
    pc.currInvView = currInvView;
    pc.prevViewProj = prevViewProj;
    pc.invProj = invProj;
    // Row 2 of prevView for axial linear depth dot product
    pc.prevViewZ = glm::vec4(prevView[0][2], prevView[1][2], prevView[2][2], prevView[3][2]);
    pc.jitterOffset = jitterOffset;
    pc.renderRes = glm::uvec2(m_inputWidth, m_inputHeight);
    pc.displayRes = glm::uvec2(m_outputWidth, m_outputHeight);
    pc.scaleFactor = glm::vec2(
        static_cast<float>(m_outputWidth) / std::max(static_cast<float>(m_inputWidth), 1.0f),
        static_cast<float>(m_outputHeight) / std::max(static_cast<float>(m_inputHeight), 1.0f)
    );
    pc.resetHistory = resetHistory ? 1u : 0u;
    pc.frameIndex = frameIndex;
    pc.totalSamples = std::max(totalSamples, 1u);
    pc.invTotalSamples = 1.0f / static_cast<float>(pc.totalSamples);

    vkCmdPushConstants(cmd, m_pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // 2D Workgroup Dispatch: 16x16 tile for super-res (scale > 1.25), 8x8 tile for native 1x denoising
    bool isNative = (pc.scaleFactor.x <= 1.25f);
    uint32_t tileDim = isNative ? 8 : 16;
    uint32_t groupsX = (m_outputWidth + tileDim - 1) / tileDim;
    uint32_t groupsY = (m_outputHeight + tileDim - 1) / tileDim;
    vkCmdDispatch(cmd, groupsX, groupsY, 1);

    std::vector<VkImageMemoryBarrier2> postBarriers;
    auto addBarrier = [&](VkImage img) {
        if (!img) return;
        VkImageMemoryBarrier2 b{};
        b.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.image = img;
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        postBarriers.push_back(b);
    };

    addBarrier(m_outputImage->getImage());
    uint32_t outHistSlot = 1 - activeSet;
    addBarrier(m_diffHistoryImages[outHistSlot]->getImage());
    addBarrier(m_specHistoryImages[outHistSlot]->getImage());
    addBarrier(m_normHistoryImages[outHistSlot]->getImage());

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.imageMemoryBarrierCount = static_cast<uint32_t>(postBarriers.size());
    dep.pImageMemoryBarriers = postBarriers.data();
    vkCmdPipelineBarrier2(cmd, &dep);

    // Advance ping-pong slot
    m_pingPongIndex = 1 - m_pingPongIndex;
}

} // namespace pathways
