#include "UpwaysPipeline.hpp"
#include "core/Logger.hpp"
#include "upways_weights.hpp"
#include "upways_default_weights.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

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
    const std::vector<char>& temporalSpv,
    const std::vector<char>& reconstructSpv,
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
    Logger::info("UpwaysPipeline initializing two-pass pipeline: Input {}x{}, Output {}x{} (SuperRes: {}), Format: {}",
                 m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight,
                 m_superRes ? "Enabled" : "1.0x (Native)", static_cast<int>(m_format));

    initBuffers(weightsPath);
    initImages();
    createDescriptorSetLayouts();
    allocateDescriptorSets();
    createPipelines(temporalSpv, reconstructSpv);

    m_initialized = true;
    Logger::info("UpwaysPipeline successfully initialized with Decoupled Two-Pass Wave32 WMMA reconstructor.");
}

UpwaysPipeline::~UpwaysPipeline() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_historySampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_device, m_historySampler, nullptr);
            m_historySampler = VK_NULL_HANDLE;
        }
        if (m_temporalPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_temporalPipeline, nullptr);
            m_temporalPipeline = VK_NULL_HANDLE;
        }
        if (m_temporalPipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_temporalPipelineLayout, nullptr);
            m_temporalPipelineLayout = VK_NULL_HANDLE;
        }
        if (m_temporalDescLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_temporalDescLayout, nullptr);
            m_temporalDescLayout = VK_NULL_HANDLE;
        }
        if (m_reconstructPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_reconstructPipeline, nullptr);
            m_reconstructPipeline = VK_NULL_HANDLE;
        }
        if (m_reconstructPipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_reconstructPipelineLayout, nullptr);
            m_reconstructPipelineLayout = VK_NULL_HANDLE;
        }
        if (m_reconstructDescLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_reconstructDescLayout, nullptr);
            m_reconstructDescLayout = VK_NULL_HANDLE;
        }
        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }
    }
}

void UpwaysPipeline::initBuffers(const std::string& weightsPath) {
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

    std::vector<std::string> candidates;
    if (!weightsPath.empty()) {
        candidates.push_back(weightsPath);
    }
    if (!exeDir.empty()) {
        candidates.push_back((exeDir / "data" / "models" / "upways_weights.bin").string());
        candidates.push_back((exeDir / ".." / "data" / "models" / "upways_weights.bin").string());
        candidates.push_back((exeDir / "bin" / "data" / "models" / "upways_weights.bin").string());
    }
    candidates.push_back("data/models/upways_weights.bin");
    candidates.push_back("../data/models/upways_weights.bin");
    candidates.push_back("../../data/models/upways_weights.bin");
    candidates.push_back("/home/naoki/Development/Pathways/data/models/upways_weights.bin");
    candidates.push_back("/home/naoki/Development/Upways/data/models/upways_weights.bin");

    std::string foundPath;
    for (const auto& path : candidates) {
        std::error_code ecCheck;
        if (std::filesystem::exists(path, ecCheck) && !std::filesystem::is_directory(path, ecCheck)) {
            foundPath = path;
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
        Logger::info("UpwaysPipeline using embedded default neural reconstructor weights ({} bytes)",
                     sizeof(upways::DEFAULT_UPWAYS_WEIGHTS));
        weightBytes.assign(upways::DEFAULT_UPWAYS_WEIGHTS,
                           upways::DEFAULT_UPWAYS_WEIGHTS + sizeof(upways::DEFAULT_UPWAYS_WEIGHTS));
    }

    if (weightBytes.size() < upways::TOTAL_WEIGHT_BUFFER_SIZE) {
        weightBytes.resize(upways::TOTAL_WEIGHT_BUFFER_SIZE, 0);
    }

    VkDeviceSize bufferSize = weightBytes.size();
    m_weightBuffer = std::make_unique<Buffer>(
        m_allocator,
        bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    void* mapped = m_weightBuffer->map();
    if (mapped) {
        std::memcpy(mapped, weightBytes.data(), bufferSize);
        m_weightBuffer->unmap();
    }
}

void UpwaysPipeline::initImages() {
    // 1. Final Display Output (Display Resolution)
    m_outputImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_outputWidth, m_outputHeight,
        m_format,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // 2. Intermediate Render-Resolution Latents (Pass 1 -> Pass 2)
    m_temporalLatentDiff = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_temporalLatentSpec = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // 3. Render-Resolution Reliability Map (Pass 1 -> Raygen Feedback & Pass 2)
    m_confidenceImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_dummyBlackImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    // 4. Render-Resolution Recurrent Latent History (Ping-Pong)
    for (int i = 0; i < 2; ++i) {
        m_historyLatentDiff[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_inputWidth, m_inputHeight,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
        m_historyLatentSpec[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_inputWidth, m_inputHeight,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
        m_historyPosWorld[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_inputWidth, m_inputHeight,
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
    if (m_initialLayoutsTransitioned) return;

    auto transitionImg = [&](Image* img) {
        if (!img) return;
        VkImageMemoryBarrier b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER };
        b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = img->getImage();
        b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        b.srcAccessMask = 0;
        b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
    };

    transitionImg(m_outputImage.get());
    transitionImg(m_temporalLatentDiff.get());
    transitionImg(m_temporalLatentSpec.get());
    transitionImg(m_confidenceImage.get());
    transitionImg(m_dummyBlackImage.get());
    for (int i = 0; i < 2; ++i) {
        transitionImg(m_historyLatentDiff[i].get());
        transitionImg(m_historyLatentSpec[i].get());
        transitionImg(m_historyPosWorld[i].get());
    }

    m_initialLayoutsTransitioned = true;
}

void UpwaysPipeline::createDescriptorSetLayouts() {
    // -------------------------------------------------------------
    // Pass 1 Descriptor Set Layout (13 Bindings):
    // 0..3: Render-res G-Buffers (readonly storage images)
    // 4..6: History Latents & World Pos (combined image samplers)
    // 7..9: Intermediate Outputs: LatentDiff, LatentSpec, ReliabilityMap
    // 10..12: Next History Outputs: OutHistDiff, OutHistSpec, OutHistPos
    // -------------------------------------------------------------
    std::vector<VkDescriptorSetLayoutBinding> temporalBindings;
    for (uint32_t b = 0; b < 4; ++b) {
        temporalBindings.push_back({ b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr });
    }
    for (uint32_t b = 4; b < 7; ++b) {
        temporalBindings.push_back({ b, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr });
    }
    for (uint32_t b = 7; b < 13; ++b) {
        temporalBindings.push_back({ b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr });
    }

    VkDescriptorSetLayoutCreateInfo tempLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    tempLayoutInfo.bindingCount = static_cast<uint32_t>(temporalBindings.size());
    tempLayoutInfo.pBindings = temporalBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &tempLayoutInfo, nullptr, &m_temporalDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create temporal descriptor set layout!");
    }

    // -------------------------------------------------------------
    // Pass 2 Descriptor Set Layout (6 Bindings):
    // 0..1: LatentDiff, LatentSpec (combined image samplers)
    // 2: ReliabilityMap (readonly storage image, r16f)
    // 3: DisplayAlbedo (readonly storage image, rgba16f)
    // 4: OutputImage (writeonly storage image, rgba16f)
    // 5: Weights SSBO (storage buffer, std430)
    // -------------------------------------------------------------
    std::vector<VkDescriptorSetLayoutBinding> reconBindings = {
        { 0, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo reconLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    reconLayoutInfo.bindingCount = static_cast<uint32_t>(reconBindings.size());
    reconLayoutInfo.pBindings = reconBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &reconLayoutInfo, nullptr, &m_reconstructDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create reconstruct descriptor set layout!");
    }
}

void UpwaysPipeline::allocateDescriptorSets() {
    VkDescriptorPoolSize poolSizes[3]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 10 * 2 + 3 * 2; // Pass 1 + Pass 2
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = 3 * 2 + 2 * 2;  // Pass 1 + Pass 2
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[2].descriptorCount = 1 * 2;          // Pass 2

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 4; // 2 sets for Pass 1 + 2 sets for Pass 2
    poolInfo.poolSizeCount = 3;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create descriptor pool!");
    }

    VkDescriptorSetLayout tempLayouts[2] = { m_temporalDescLayout, m_temporalDescLayout };
    VkDescriptorSetAllocateInfo allocTempInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocTempInfo.descriptorPool = m_descriptorPool;
    allocTempInfo.descriptorSetCount = 2;
    allocTempInfo.pSetLayouts = tempLayouts;
    if (vkAllocateDescriptorSets(m_device, &allocTempInfo, m_temporalDescSets) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to allocate temporal descriptor sets!");
    }

    VkDescriptorSetLayout reconLayouts[2] = { m_reconstructDescLayout, m_reconstructDescLayout };
    VkDescriptorSetAllocateInfo allocReconInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocReconInfo.descriptorPool = m_descriptorPool;
    allocReconInfo.descriptorSetCount = 2;
    allocReconInfo.pSetLayouts = reconLayouts;
    if (vkAllocateDescriptorSets(m_device, &allocReconInfo, m_reconstructDescSets) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to allocate reconstruct descriptor sets!");
    }
}

VkShaderModule UpwaysPipeline::createShaderModule(const std::vector<char>& code) {
    if (code.empty()) {
        throw std::runtime_error("UpwaysPipeline: Shader bytecode is empty!");
    }
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create shader module!");
    }
    return shaderModule;
}

void UpwaysPipeline::createPipelines(const std::vector<char>& temporalSpv, const std::vector<char>& reconstructSpv) {
    // Exact 256-byte Push Constants struct matching UpwaysPushConstants
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(UpwaysPushConstants);

    // Pass 1: Temporal Pipeline
    VkPipelineLayoutCreateInfo tempPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    tempPipelineLayoutInfo.setLayoutCount = 1;
    tempPipelineLayoutInfo.pSetLayouts = &m_temporalDescLayout;
    tempPipelineLayoutInfo.pushConstantRangeCount = 1;
    tempPipelineLayoutInfo.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(m_device, &tempPipelineLayoutInfo, nullptr, &m_temporalPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create temporal pipeline layout!");
    }

    VkShaderModule tempModule = createShaderModule(temporalSpv);
    VkComputePipelineCreateInfo tempPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    tempPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    tempPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    tempPipeInfo.stage.module = tempModule;
    tempPipeInfo.stage.pName = "main";
    tempPipeInfo.layout = m_temporalPipelineLayout;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &tempPipeInfo, nullptr, &m_temporalPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, tempModule, nullptr);
        throw std::runtime_error("UpwaysPipeline: Failed to create temporal compute pipeline!");
    }
    vkDestroyShaderModule(m_device, tempModule, nullptr);

    // Pass 2: Reconstruct Pipeline
    VkPipelineLayoutCreateInfo reconPipelineLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    reconPipelineLayoutInfo.setLayoutCount = 1;
    reconPipelineLayoutInfo.pSetLayouts = &m_reconstructDescLayout;
    reconPipelineLayoutInfo.pushConstantRangeCount = 1;
    reconPipelineLayoutInfo.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(m_device, &reconPipelineLayoutInfo, nullptr, &m_reconstructPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create reconstruct pipeline layout!");
    }

    VkShaderModule reconModule = createShaderModule(reconstructSpv);
    VkComputePipelineCreateInfo reconPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    reconPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    reconPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    reconPipeInfo.stage.module = reconModule;
    reconPipeInfo.stage.pName = "main";
    reconPipeInfo.layout = m_reconstructPipelineLayout;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &reconPipeInfo, nullptr, &m_reconstructPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, reconModule, nullptr);
        throw std::runtime_error("UpwaysPipeline: Failed to create reconstruct compute pipeline!");
    }
    vkDestroyShaderModule(m_device, reconModule, nullptr);
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
    VkImageView defaultStorageView = m_outputImage ? m_outputImage->getImageView() : VK_NULL_HANDLE;
    VkImageView tex0 = (demodDiffuseView != VK_NULL_HANDLE) ? demodDiffuseView : defaultStorageView;
    VkImageView tex1 = (albedoRoughnessView != VK_NULL_HANDLE) ? albedoRoughnessView : defaultStorageView;
    VkImageView tex2 = (normalDepthView != VK_NULL_HANDLE) ? normalDepthView : defaultStorageView;
    VkImageView tex3 = (specularMotionView != VK_NULL_HANDLE) ? specularMotionView : (surfaceMotionView != VK_NULL_HANDLE ? surfaceMotionView : defaultStorageView);

    VkImageView dispAlbedo = (displayAlbedoView != VK_NULL_HANDLE) ? displayAlbedoView : tex1;

    for (uint32_t slot = 0; slot < 2; ++slot) {
        uint32_t readSlot = slot;
        uint32_t writeSlot = 1 - slot;

        // -------------------------------------------------------------
        // Update Pass 1 Descriptors (m_temporalDescSets[slot])
        // -------------------------------------------------------------
        VkDescriptorImageInfo gbufInfos[4] = {
            { VK_NULL_HANDLE, tex0, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, tex1, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, tex2, VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, tex3, VK_IMAGE_LAYOUT_GENERAL }
        };

        VkDescriptorImageInfo histReadInfos[3] = {
            { m_historySampler, m_historyLatentDiff[readSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL },
            { m_historySampler, m_historyLatentSpec[readSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL },
            { m_historySampler, m_historyPosWorld[readSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL }
        };

        VkDescriptorImageInfo interOutputInfos[3] = {
            { VK_NULL_HANDLE, m_temporalLatentDiff->getImageView(), VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, m_temporalLatentSpec->getImageView(), VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, m_confidenceImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL }
        };

        VkDescriptorImageInfo histWriteInfos[3] = {
            { VK_NULL_HANDLE, m_historyLatentDiff[writeSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, m_historyLatentSpec[writeSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL },
            { VK_NULL_HANDLE, m_historyPosWorld[writeSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL }
        };

        std::vector<VkWriteDescriptorSet> tempWrites;
        for (uint32_t b = 0; b < 4; ++b) {
            tempWrites.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_temporalDescSets[slot], b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &gbufInfos[b], nullptr, nullptr });
        }
        for (uint32_t b = 0; b < 3; ++b) {
            tempWrites.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_temporalDescSets[slot], 4 + b, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &histReadInfos[b], nullptr, nullptr });
        }
        for (uint32_t b = 0; b < 3; ++b) {
            tempWrites.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_temporalDescSets[slot], 7 + b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &interOutputInfos[b], nullptr, nullptr });
        }
        for (uint32_t b = 0; b < 3; ++b) {
            tempWrites.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_temporalDescSets[slot], 10 + b, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &histWriteInfos[b], nullptr, nullptr });
        }
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(tempWrites.size()), tempWrites.data(), 0, nullptr);

        // -------------------------------------------------------------
        // Update Pass 2 Descriptors (m_reconstructDescSets[slot])
        // -------------------------------------------------------------
        VkDescriptorImageInfo reconSamplerInfos[2] = {
            { m_historySampler, m_temporalLatentDiff->getImageView(), VK_IMAGE_LAYOUT_GENERAL },
            { m_historySampler, m_temporalLatentSpec->getImageView(), VK_IMAGE_LAYOUT_GENERAL }
        };
        VkDescriptorImageInfo reconRelInfo{ VK_NULL_HANDLE, m_confidenceImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo reconAlbedoInfo{ VK_NULL_HANDLE, dispAlbedo, VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorImageInfo reconOutInfo{ VK_NULL_HANDLE, m_outputImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        VkDescriptorBufferInfo weightBufInfo{ m_weightBuffer->getBuffer(), 0, VK_WHOLE_SIZE };

        std::vector<VkWriteDescriptorSet> reconWrites = {
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_reconstructDescSets[slot], 0, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &reconSamplerInfos[0], nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_reconstructDescSets[slot], 1, 0, 1, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, &reconSamplerInfos[1], nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_reconstructDescSets[slot], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &reconRelInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_reconstructDescSets[slot], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &reconAlbedoInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_reconstructDescSets[slot], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &reconOutInfo, nullptr, nullptr },
            { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_reconstructDescSets[slot], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &weightBufInfo, nullptr }
        };
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(reconWrites.size()), reconWrites.data(), 0, nullptr);
    }
}

void UpwaysPipeline::resize(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight, bool enableSuperRes) {
    if (m_inputWidth == inputWidth && m_inputHeight == inputHeight &&
        m_outputWidth == outputWidth && m_outputHeight == outputHeight &&
        m_superRes == enableSuperRes) {
        return;
    }

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
    if (!m_temporalPipeline || !m_reconstructPipeline || !m_outputImage) return;

    if (!m_initialLayoutsTransitioned) {
        transitionInitialLayouts(cmd);
    }

    uint32_t activeSet = m_pingPongIndex;

    UpwaysPushConstants pc{};
    pc.currInvView = currInvView;
    pc.prevViewProj = prevViewProj;
    pc.invProj = invProj;
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

    // =========================================================================
    // PASS 1: Temporal Kinematic Dispatch (Render Resolution 540p)
    // =========================================================================
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipelineLayout, 0, 1, &m_temporalDescSets[activeSet], 0, nullptr);
    vkCmdPushConstants(cmd, m_temporalPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t tempGroupsX = (m_inputWidth + 7) / 8;
    uint32_t tempGroupsY = (m_inputHeight + 7) / 8;
    vkCmdDispatch(cmd, tempGroupsX, tempGroupsY, 1);

    // =========================================================================
    // EXECUTION BARRIER: Pass 1 Compute Writes -> Pass 2 Compute Reads
    // =========================================================================
    VkImageMemoryBarrier2 latentBarriers[3]{};
    for (int i = 0; i < 3; ++i) {
        latentBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        latentBarriers[i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        latentBarriers[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        latentBarriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        latentBarriers[i].dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        latentBarriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        latentBarriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        latentBarriers[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    }
    latentBarriers[0].image = m_temporalLatentDiff->getImage();
    latentBarriers[1].image = m_temporalLatentSpec->getImage();
    latentBarriers[2].image = m_confidenceImage->getImage();

    VkDependencyInfo interPassDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    interPassDep.imageMemoryBarrierCount = 3;
    interPassDep.pImageMemoryBarriers = latentBarriers;
    vkCmdPipelineBarrier2(cmd, &interPassDep);

    // =========================================================================
    // PASS 2: Continuous Reconstruct Dispatch (Display Resolution 1080p / 4K)
    // =========================================================================
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_reconstructPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_reconstructPipelineLayout, 0, 1, &m_reconstructDescSets[activeSet], 0, nullptr);
    vkCmdPushConstants(cmd, m_reconstructPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    // Wave32 workgroup processes M=16 pixels (TILE_M = 16)
    uint32_t totalDisplayPixels = m_outputWidth * m_outputHeight;
    uint32_t reconGroupsX = (totalDisplayPixels + 15) / 16;
    vkCmdDispatch(cmd, reconGroupsX, 1, 1);

    // =========================================================================
    // POST-PASS 2 BARRIERS: Output & History State
    // =========================================================================
    VkImageMemoryBarrier2 postBarriers[4]{};
    for (int i = 0; i < 4; ++i) {
        postBarriers[i].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        postBarriers[i].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        postBarriers[i].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        postBarriers[i].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        postBarriers[i].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        postBarriers[i].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        postBarriers[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        postBarriers[i].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    }
    uint32_t outHistSlot = 1 - activeSet;
    postBarriers[0].image = m_outputImage->getImage();
    postBarriers[1].image = m_historyLatentDiff[outHistSlot]->getImage();
    postBarriers[2].image = m_historyLatentSpec[outHistSlot]->getImage();
    postBarriers[3].image = m_historyPosWorld[outHistSlot]->getImage();

    VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    postDep.imageMemoryBarrierCount = 4;
    postDep.pImageMemoryBarriers = postBarriers;
    vkCmdPipelineBarrier2(cmd, &postDep);

    // Advance ping-pong slot
    m_pingPongIndex = 1 - m_pingPongIndex;
}

} // namespace pathways
