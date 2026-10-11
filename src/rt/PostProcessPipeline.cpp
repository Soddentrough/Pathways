#include "PostProcessPipeline.hpp"
#include "core/Logger.hpp"

#include <stdexcept>
#include <algorithm>

namespace pathways {

PostProcessPipeline::PostProcessPipeline(
    VkDevice device,
    VmaAllocator allocator,
    bool hasSubgroupSizeControl,
    VkDescriptorPool globalPool,
    const std::vector<char>& tonemapSpv,
    const std::vector<char>& fusedAccumTonemapSpv,
    const std::vector<char>& runningAvgSpv,
    const std::vector<char>& blendSpv,
    const std::vector<char>& mergeSpv,
    const std::vector<char>& restirDebugSpv)
    : m_device(device),
      m_allocator(allocator),
      m_hasSubgroupSizeControl(hasSubgroupSizeControl),
      m_globalPool(globalPool)
{
    createDescriptorLayouts();
    allocateDescriptorSets();
    createPipelines(tonemapSpv, fusedAccumTonemapSpv, runningAvgSpv, blendSpv, mergeSpv);
    createDebugViewPipeline(restirDebugSpv);

    Logger::info("PostProcessPipeline (ACES Tonemap, Fused Accum/Tonemap, Running Avg, Blend4K, Multi-GPU Merge, ReSTIR Debug View) initialized.");
}

PostProcessPipeline::~PostProcessPipeline() {
    if (m_tonemapPipeline)          { vkDestroyPipeline(m_device, m_tonemapPipeline, nullptr); m_tonemapPipeline = VK_NULL_HANDLE; }
    if (m_accumRunningAvgPipeline)  { vkDestroyPipeline(m_device, m_accumRunningAvgPipeline, nullptr); m_accumRunningAvgPipeline = VK_NULL_HANDLE; }
    if (m_accumTonemapPipeline)     { vkDestroyPipeline(m_device, m_accumTonemapPipeline, nullptr); m_accumTonemapPipeline = VK_NULL_HANDLE; }
    if (m_fsr3BlendPipeline)        { vkDestroyPipeline(m_device, m_fsr3BlendPipeline, nullptr); m_fsr3BlendPipeline = VK_NULL_HANDLE; }
    if (m_mergePipeline)            { vkDestroyPipeline(m_device, m_mergePipeline, nullptr); m_mergePipeline = VK_NULL_HANDLE; }
    if (m_restirDebugPipeline)      { vkDestroyPipeline(m_device, m_restirDebugPipeline, nullptr); m_restirDebugPipeline = VK_NULL_HANDLE; }

    m_debugDummyBuffer.reset();

    if (m_tonemapPipelineLayout)          { vkDestroyPipelineLayout(m_device, m_tonemapPipelineLayout, nullptr); m_tonemapPipelineLayout = VK_NULL_HANDLE; }
    if (m_accumRunningAvgPipelineLayout)  { vkDestroyPipelineLayout(m_device, m_accumRunningAvgPipelineLayout, nullptr); m_accumRunningAvgPipelineLayout = VK_NULL_HANDLE; }
    if (m_accumTonemapPipelineLayout)     { vkDestroyPipelineLayout(m_device, m_accumTonemapPipelineLayout, nullptr); m_accumTonemapPipelineLayout = VK_NULL_HANDLE; }
    if (m_fsr3BlendPipelineLayout)        { vkDestroyPipelineLayout(m_device, m_fsr3BlendPipelineLayout, nullptr); m_fsr3BlendPipelineLayout = VK_NULL_HANDLE; }
    if (m_mergePipelineLayout)            { vkDestroyPipelineLayout(m_device, m_mergePipelineLayout, nullptr); m_mergePipelineLayout = VK_NULL_HANDLE; }

    if (m_fsr3BlendDescPool)        { vkDestroyDescriptorPool(m_device, m_fsr3BlendDescPool, nullptr); m_fsr3BlendDescPool = VK_NULL_HANDLE; }
    if (m_restirDebugDescPool)      { vkDestroyDescriptorPool(m_device, m_restirDebugDescPool, nullptr); m_restirDebugDescPool = VK_NULL_HANDLE; }

    if (m_tonemapDescLayout)          { vkDestroyDescriptorSetLayout(m_device, m_tonemapDescLayout, nullptr); m_tonemapDescLayout = VK_NULL_HANDLE; }
    if (m_accumRunningAvgDescLayout)  { vkDestroyDescriptorSetLayout(m_device, m_accumRunningAvgDescLayout, nullptr); m_accumRunningAvgDescLayout = VK_NULL_HANDLE; }
    if (m_accumTonemapDescLayout)     { vkDestroyDescriptorSetLayout(m_device, m_accumTonemapDescLayout, nullptr); m_accumTonemapDescLayout = VK_NULL_HANDLE; }
    if (m_fsr3BlendDescLayout)        { vkDestroyDescriptorSetLayout(m_device, m_fsr3BlendDescLayout, nullptr); m_fsr3BlendDescLayout = VK_NULL_HANDLE; }
    if (m_mergeDescLayout)            { vkDestroyDescriptorSetLayout(m_device, m_mergeDescLayout, nullptr); m_mergeDescLayout = VK_NULL_HANDLE; }
    if (m_restirDebugDescLayout)      { vkDestroyDescriptorSetLayout(m_device, m_restirDebugDescLayout, nullptr); m_restirDebugDescLayout = VK_NULL_HANDLE; }
    if (m_restirDebugPipelineLayout)  { vkDestroyPipelineLayout(m_device, m_restirDebugPipelineLayout, nullptr); m_restirDebugPipelineLayout = VK_NULL_HANDLE; }
}

VkShaderModule PostProcessPipeline::createShaderModule(const std::vector<char>& code) {
    if (code.empty()) return VK_NULL_HANDLE;
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule mod = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &mod) != VK_SUCCESS) {
        throw std::runtime_error("PostProcessPipeline: Failed to create shader module!");
    }
    return mod;
}

void PostProcessPipeline::createDescriptorLayouts() {
    // 1. Tonemap Layout: (0: inStorageImage, 1: outStorageImage)
    std::vector<VkDescriptorSetLayoutBinding> tonemapBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo tonemapLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    tonemapLayoutInfo.bindingCount = static_cast<uint32_t>(tonemapBindings.size());
    tonemapLayoutInfo.pBindings = tonemapBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &tonemapLayoutInfo, nullptr, &m_tonemapDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create tonemap descriptor set layout");
    }

    // 2. Running Avg Layout: (0: inFrame, 1: historyAccum)
    std::vector<VkDescriptorSetLayoutBinding> avgBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo avgLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    avgLayoutInfo.bindingCount = static_cast<uint32_t>(avgBindings.size());
    avgLayoutInfo.pBindings = avgBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &avgLayoutInfo, nullptr, &m_accumRunningAvgDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create running avg accumulation descriptor set layout");
    }

    // 3. Fused Accum/Tonemap Layout: (0: inFrame, 1: historyAccum, 2: outImage)
    std::vector<VkDescriptorSetLayoutBinding> fusedBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo fusedLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    fusedLayoutInfo.bindingCount = static_cast<uint32_t>(fusedBindings.size());
    fusedLayoutInfo.pBindings = fusedBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &fusedLayoutInfo, nullptr, &m_accumTonemapDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create fused accum tonemap descriptor set layout");
    }

    // 4. Blend 4K Layout: (0: dstStorageImage, 1: srcStorageImage)
    std::vector<VkDescriptorSetLayoutBinding> blendBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo blendLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    blendLayoutInfo.bindingCount = static_cast<uint32_t>(blendBindings.size());
    blendLayoutInfo.pBindings = blendBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &blendLayoutInfo, nullptr, &m_fsr3BlendDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create FSR3 blend descriptor set layout");
    }

    // 5. Multi-GPU Merge Layout: (0: img, 1..2: buf, 3..4: img, 5..7: buf)
    std::vector<VkDescriptorSetLayoutBinding> mergeBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 4, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 7, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo mergeLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    mergeLayoutInfo.bindingCount = static_cast<uint32_t>(mergeBindings.size());
    mergeLayoutInfo.pBindings = mergeBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &mergeLayoutInfo, nullptr, &m_mergeDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create merge descriptor set layout");
    }

    // 6. ReSTIR Debug View Layout: (0: outImage, 1: DI reservoirs, 2: GI reservoirs, 3: X1 contexts)
    std::vector<VkDescriptorSetLayoutBinding> debugBindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 3, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };
    VkDescriptorSetLayoutCreateInfo debugLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    debugLayoutInfo.bindingCount = static_cast<uint32_t>(debugBindings.size());
    debugLayoutInfo.pBindings = debugBindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &debugLayoutInfo, nullptr, &m_restirDebugDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ReSTIR debug view descriptor set layout");
    }
}

void PostProcessPipeline::allocateDescriptorSets() {
    // 1. Tonemap set
    VkDescriptorSetAllocateInfo tonemapAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    tonemapAllocInfo.descriptorPool = m_globalPool;
    tonemapAllocInfo.descriptorSetCount = 1;
    tonemapAllocInfo.pSetLayouts = &m_tonemapDescLayout;
    if (vkAllocateDescriptorSets(m_device, &tonemapAllocInfo, &m_tonemapDescSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate tonemap descriptor set");
    }

    // 2. Running Avg sets
    std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> avgLayouts = { m_accumRunningAvgDescLayout, m_accumRunningAvgDescLayout };
    VkDescriptorSetAllocateInfo avgAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    avgAllocInfo.descriptorPool = m_globalPool;
    avgAllocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    avgAllocInfo.pSetLayouts = avgLayouts.data();
    if (vkAllocateDescriptorSets(m_device, &avgAllocInfo, m_accumRunningAvgDescSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate running avg descriptor sets");
    }

    // 3. Fused Accum/Tonemap sets
    std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> fusedLayouts = { m_accumTonemapDescLayout, m_accumTonemapDescLayout };
    VkDescriptorSetAllocateInfo fusedAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    fusedAllocInfo.descriptorPool = m_globalPool;
    fusedAllocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    fusedAllocInfo.pSetLayouts = fusedLayouts.data();
    if (vkAllocateDescriptorSets(m_device, &fusedAllocInfo, m_accumTonemapDescSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate fused accum tonemap descriptor sets");
    }

    // 4. Blend 4K Pool & Set
    VkDescriptorPoolSize blendPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 };
    VkDescriptorPoolCreateInfo blendPoolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    blendPoolInfo.maxSets = 1;
    blendPoolInfo.poolSizeCount = 1;
    blendPoolInfo.pPoolSizes = &blendPoolSize;
    if (vkCreateDescriptorPool(m_device, &blendPoolInfo, nullptr, &m_fsr3BlendDescPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create FSR3 blend descriptor pool");
    }

    VkDescriptorSetAllocateInfo blendAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    blendAllocInfo.descriptorPool = m_fsr3BlendDescPool;
    blendAllocInfo.descriptorSetCount = 1;
    blendAllocInfo.pSetLayouts = &m_fsr3BlendDescLayout;
    if (vkAllocateDescriptorSets(m_device, &blendAllocInfo, &m_fsr3BlendDescSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate FSR3 blend descriptor set");
    }

    // 5. Merge sets
    std::array<VkDescriptorSetLayout, 2> mergeLayouts = { m_mergeDescLayout, m_mergeDescLayout };
    VkDescriptorSetAllocateInfo mergeAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    mergeAllocInfo.descriptorPool = m_globalPool;
    mergeAllocInfo.descriptorSetCount = 2;
    mergeAllocInfo.pSetLayouts = mergeLayouts.data();
    if (vkAllocateDescriptorSets(m_device, &mergeAllocInfo, m_mergeDescSets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate merge descriptor sets");
    }
}

void PostProcessPipeline::createPipelines(
    const std::vector<char>& tonemapSpv,
    const std::vector<char>& fusedAccumTonemapSpv,
    const std::vector<char>& runningAvgSpv,
    const std::vector<char>& blendSpv,
    const std::vector<char>& mergeSpv)
{
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
    };
    subgroupSize32.requiredSubgroupSize = 32;

    // 1. Tonemap Pipeline
    if (!tonemapSpv.empty()) {
        VkPushConstantRange tonemapPushConstant{};
        tonemapPushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        tonemapPushConstant.offset = 0;
        tonemapPushConstant.size = sizeof(TonemapPushConstants);

        VkPipelineLayoutCreateInfo tonemapPipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        tonemapPipeLayoutInfo.setLayoutCount = 1;
        tonemapPipeLayoutInfo.pSetLayouts = &m_tonemapDescLayout;
        tonemapPipeLayoutInfo.pushConstantRangeCount = 1;
        tonemapPipeLayoutInfo.pPushConstantRanges = &tonemapPushConstant;
        if (vkCreatePipelineLayout(m_device, &tonemapPipeLayoutInfo, nullptr, &m_tonemapPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create tonemap pipeline layout");
        }

        VkShaderModule tonemapModule = createShaderModule(tonemapSpv);
        VkComputePipelineCreateInfo tonemapPipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        tonemapPipelineInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, tonemapModule, "main", nullptr };
        if (m_hasSubgroupSizeControl) tonemapPipelineInfo.stage.pNext = &subgroupSize32;
        tonemapPipelineInfo.layout = m_tonemapPipelineLayout;
        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &tonemapPipelineInfo, nullptr, &m_tonemapPipeline) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, tonemapModule, nullptr);
            throw std::runtime_error("Failed to create tonemap pipeline");
        }
        vkDestroyShaderModule(m_device, tonemapModule, nullptr);
    }

    // 2. Running Avg Pipeline
    if (!runningAvgSpv.empty()) {
        VkPushConstantRange pushConstant{};
        pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstant.offset = 0;
        pushConstant.size = sizeof(RunningAvgPushConstants);

        VkPipelineLayoutCreateInfo pipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipeLayoutInfo.setLayoutCount = 1;
        pipeLayoutInfo.pSetLayouts = &m_accumRunningAvgDescLayout;
        pipeLayoutInfo.pushConstantRangeCount = 1;
        pipeLayoutInfo.pPushConstantRanges = &pushConstant;
        if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_accumRunningAvgPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create running avg pipeline layout");
        }

        VkShaderModule shaderModule = createShaderModule(runningAvgSpv);
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shaderModule, "main", nullptr };
        if (m_hasSubgroupSizeControl) pipelineInfo.stage.pNext = &subgroupSize32;
        pipelineInfo.layout = m_accumRunningAvgPipelineLayout;
        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_accumRunningAvgPipeline) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, shaderModule, nullptr);
            throw std::runtime_error("Failed to create running avg pipeline");
        }
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
    }

    // 3. Fused Accum/Tonemap Pipeline
    if (!fusedAccumTonemapSpv.empty()) {
        VkPushConstantRange pushConstant{};
        pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pushConstant.offset = 0;
        pushConstant.size = sizeof(FusedAccumTonemapPushConstants);

        VkPipelineLayoutCreateInfo pipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        pipeLayoutInfo.setLayoutCount = 1;
        pipeLayoutInfo.pSetLayouts = &m_accumTonemapDescLayout;
        pipeLayoutInfo.pushConstantRangeCount = 1;
        pipeLayoutInfo.pPushConstantRanges = &pushConstant;
        if (vkCreatePipelineLayout(m_device, &pipeLayoutInfo, nullptr, &m_accumTonemapPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create fused accum tonemap pipeline layout");
        }

        VkShaderModule shaderModule = createShaderModule(fusedAccumTonemapSpv);
        VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipelineInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, shaderModule, "main", nullptr };
        if (m_hasSubgroupSizeControl) pipelineInfo.stage.pNext = &subgroupSize32;
        pipelineInfo.layout = m_accumTonemapPipelineLayout;
        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_accumTonemapPipeline) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, shaderModule, nullptr);
            throw std::runtime_error("Failed to create fused accum tonemap pipeline");
        }
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
    }

    // 4. Blend 4K Pipeline
    if (!blendSpv.empty()) {
        VkPushConstantRange blendPcRange{};
        blendPcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        blendPcRange.offset = 0;
        blendPcRange.size = sizeof(Blend4KPushConstants);

        VkPipelineLayoutCreateInfo blendPipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        blendPipeLayoutInfo.setLayoutCount = 1;
        blendPipeLayoutInfo.pSetLayouts = &m_fsr3BlendDescLayout;
        blendPipeLayoutInfo.pushConstantRangeCount = 1;
        blendPipeLayoutInfo.pPushConstantRanges = &blendPcRange;
        if (vkCreatePipelineLayout(m_device, &blendPipeLayoutInfo, nullptr, &m_fsr3BlendPipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create FSR3 blend pipeline layout");
        }

        VkShaderModule blendModule = createShaderModule(blendSpv);
        VkComputePipelineCreateInfo blendPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        blendPipeInfo.layout = m_fsr3BlendPipelineLayout;
        blendPipeInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, blendModule, "main", nullptr };
        if (m_hasSubgroupSizeControl) blendPipeInfo.stage.pNext = &subgroupSize32;

        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &blendPipeInfo, nullptr, &m_fsr3BlendPipeline) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, blendModule, nullptr);
            throw std::runtime_error("Failed to create FSR3 blend compute pipeline");
        }
        vkDestroyShaderModule(m_device, blendModule, nullptr);
    }

    // 5. Merge Pipeline
    if (!mergeSpv.empty()) {
        VkPushConstantRange mergePushConstant{};
        mergePushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        mergePushConstant.offset = 0;
        mergePushConstant.size = sizeof(MergePushConstants);

        VkPipelineLayoutCreateInfo mergePipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        mergePipeLayoutInfo.setLayoutCount = 1;
        mergePipeLayoutInfo.pSetLayouts = &m_mergeDescLayout;
        mergePipeLayoutInfo.pushConstantRangeCount = 1;
        mergePipeLayoutInfo.pPushConstantRanges = &mergePushConstant;
        if (vkCreatePipelineLayout(m_device, &mergePipeLayoutInfo, nullptr, &m_mergePipelineLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create merge pipeline layout");
        }

        VkShaderModule mergeModule = createShaderModule(mergeSpv);
        VkComputePipelineCreateInfo mergePipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        mergePipelineInfo.layout = m_mergePipelineLayout;
        mergePipelineInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mergeModule, "main", nullptr };
        if (m_hasSubgroupSizeControl) mergePipelineInfo.stage.pNext = &subgroupSize32;

        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &mergePipelineInfo, nullptr, &m_mergePipeline) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, mergeModule, nullptr);
            throw std::runtime_error("Failed to create merge pipeline");
        }
        vkDestroyShaderModule(m_device, mergeModule, nullptr);
    }
}

void PostProcessPipeline::updateTonemapDescriptors(VkDescriptorSet descSet, Image* inImage, Image* outImage) {
    if (descSet == VK_NULL_HANDLE || !inImage || !outImage) return;

    VkDescriptorImageInfo inInfo{ VK_NULL_HANDLE, inImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo outInfo{ VK_NULL_HANDLE, outImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkWriteDescriptorSet, 2> writes{};
    writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &inInfo, nullptr, nullptr };
    writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outInfo, nullptr, nullptr };
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void PostProcessPipeline::updateDebugViewDescriptors(uint32_t slot, Buffer* diReservoirs, Buffer* giReservoirs, Buffer* x1Contexts, Image* outputImage) {
    if (slot >= MAX_FRAMES_IN_FLIGHT || m_restirDebugDescSets[slot] == VK_NULL_HANDLE || !outputImage) return;

    auto bufInfo = [this](Buffer* b, VkDescriptorBufferInfo& out) {
        // Null grids fall back to the small dummy buffer; the shader's
        // .length() bounds check then rejects (nearly) every pixel.
        out.buffer = b ? b->getBuffer() : m_debugDummyBuffer->getBuffer();
        out.offset = 0;
        out.range = b ? b->getSize() : m_debugDummyBuffer->getSize();
    };
    VkDescriptorBufferInfo diInfo{}, giInfo{}, x1Info{};
    bufInfo(diReservoirs, diInfo);
    bufInfo(giReservoirs, giInfo);
    bufInfo(x1Contexts, x1Info);

    VkDescriptorImageInfo outInfo{ VK_NULL_HANDLE, outputImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkWriteDescriptorSet, 4> writes{};
    writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_restirDebugDescSets[slot], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outInfo, nullptr, nullptr };
    writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_restirDebugDescSets[slot], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &diInfo, nullptr };
    writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_restirDebugDescSets[slot], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &giInfo, nullptr };
    writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_restirDebugDescSets[slot], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &x1Info, nullptr };
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

void PostProcessPipeline::recordDebugView(VkCommandBuffer cmd, uint32_t slot, const DebugViewPushConstants& pc) {
    if (!m_restirDebugPipeline || slot >= MAX_FRAMES_IN_FLIGHT || m_restirDebugDescSets[slot] == VK_NULL_HANDLE) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_restirDebugPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_restirDebugPipelineLayout, 0, 1, &m_restirDebugDescSets[slot], 0, nullptr);
    vkCmdPushConstants(cmd, m_restirDebugPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(DebugViewPushConstants), &pc);
    vkCmdDispatch(cmd, (pc.outWidth + 15) / 16, (pc.outHeight + 15) / 16, 1);
}

void PostProcessPipeline::createDebugViewPipeline(const std::vector<char>& spv) {
    if (spv.empty()) return;

    // Dedicated pool: the global pool budget is fully allocated elsewhere.
    VkDescriptorPoolSize poolSizes[] = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, MAX_FRAMES_IN_FLIGHT },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, MAX_FRAMES_IN_FLIGHT * 3 }
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_restirDebugDescPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ReSTIR debug view descriptor pool");
    }

    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> sets{};
    std::array<VkDescriptorSetLayout, MAX_FRAMES_IN_FLIGHT> layouts{ m_restirDebugDescLayout, m_restirDebugDescLayout };
    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = m_restirDebugDescPool;
    allocInfo.descriptorSetCount = MAX_FRAMES_IN_FLIGHT;
    allocInfo.pSetLayouts = layouts.data();
    if (vkAllocateDescriptorSets(m_device, &allocInfo, sets.data()) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate ReSTIR debug view descriptor sets");
    }
    m_restirDebugDescSets = sets;

    // Fallback binding for null reservoir grids (shader never reads past its
    // .length(), so a small buffer is safe).
    m_debugDummyBuffer = std::make_unique<Buffer>(
        m_allocator, 128,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
        VMA_MEMORY_USAGE_GPU_ONLY
    );

    VkPushConstantRange pushConstant{};
    pushConstant.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pushConstant.offset = 0;
    pushConstant.size = sizeof(DebugViewPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_restirDebugDescLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pushConstant;
    if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_restirDebugPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create ReSTIR debug view pipeline layout");
    }

    VkShaderModule mod = createShaderModule(spv);
    VkComputePipelineCreateInfo pipelineInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipelineInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr };
    pipelineInfo.layout = m_restirDebugPipelineLayout;
    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &m_restirDebugPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, mod, nullptr);
        throw std::runtime_error("Failed to create ReSTIR debug view pipeline");
    }
    vkDestroyShaderModule(m_device, mod, nullptr);
}

void PostProcessPipeline::updateRunningAvgDescriptors(
    const std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT>& frameImages,
    Image* accumImage)
{
    if (!accumImage) return;

    VkDescriptorImageInfo historyInfo{};
    historyInfo.imageView = accumImage->getImageView();
    historyInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkDescriptorImageInfo, MAX_FRAMES_IN_FLIGHT> frameInfos;
    std::vector<VkWriteDescriptorSet> writes;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (!frameImages[i] || m_accumRunningAvgDescSets[i] == VK_NULL_HANDLE) continue;

        frameInfos[i].sampler = VK_NULL_HANDLE;
        frameInfos[i].imageView = frameImages[i]->getImageView();
        frameInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_accumRunningAvgDescSets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &frameInfos[i], nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_accumRunningAvgDescSets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &historyInfo, nullptr, nullptr });
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void PostProcessPipeline::updateFusedAccumTonemapDescriptors(
    const std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT>& frameImages,
    Image* accumImage,
    Image* outputImage)
{
    if (!accumImage || !outputImage) return;

    VkDescriptorImageInfo historyInfo{};
    historyInfo.imageView = accumImage->getImageView();
    historyInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    VkDescriptorImageInfo outputInfo{};
    outputInfo.imageView = outputImage->getImageView();
    outputInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;

    std::array<VkDescriptorImageInfo, MAX_FRAMES_IN_FLIGHT> frameInfos;
    std::vector<VkWriteDescriptorSet> writes;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (!frameImages[i] || m_accumTonemapDescSets[i] == VK_NULL_HANDLE) continue;

        frameInfos[i].sampler = VK_NULL_HANDLE;
        frameInfos[i].imageView = frameImages[i]->getImageView();
        frameInfos[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;

        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_accumTonemapDescSets[i], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &frameInfos[i], nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_accumTonemapDescSets[i], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &historyInfo, nullptr, nullptr });
        writes.push_back({ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_accumTonemapDescSets[i], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &outputInfo, nullptr, nullptr });
    }

    if (!writes.empty()) {
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void PostProcessPipeline::updateBlendDescriptors(Image* dstImage, Image* srcImage) {
    if (m_fsr3BlendDescSet == VK_NULL_HANDLE || !dstImage || !srcImage) return;

    VkDescriptorImageInfo dstInfo{ VK_NULL_HANDLE, dstImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo srcInfo{ VK_NULL_HANDLE, srcImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

    std::array<VkWriteDescriptorSet, 2> blendWrites{};
    blendWrites[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_fsr3BlendDescSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dstInfo, nullptr, nullptr };
    blendWrites[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_fsr3BlendDescSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &srcInfo, nullptr, nullptr };
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(blendWrites.size()), blendWrites.data(), 0, nullptr);
}

void PostProcessPipeline::updateMergeDescriptors(
    uint32_t slot,
    VkBuffer secBuffer,
    VkDeviceSize curSize,
    Image* frameImage,
    Image* mvImage,
    Image* normDepthImage,
    uint32_t width,
    uint32_t height,
    AccumFormat format)
{
    if (slot >= 2 || m_mergeDescSets[slot] == VK_NULL_HANDLE || secBuffer == VK_NULL_HANDLE) return;

    uint32_t bpp = (format == AccumFormat::RGBA16_SFLOAT) ? 8 : (format == AccumFormat::R11G11B10_UFLOAT) ? 4 : 16;
    VkDeviceSize radSize = static_cast<VkDeviceSize>(width) * height * bpp;
    VkDeviceSize radOffsetAligned = (radSize + 65535) & ~static_cast<VkDeviceSize>(65535);
    VkDeviceSize mvSize = static_cast<VkDeviceSize>(width) * height * 4; // RG16F
    VkDeviceSize mvOffsetAligned = (radOffsetAligned + mvSize + 65535) & ~static_cast<VkDeviceSize>(65535);

    VkDescriptorBufferInfo secBufInfo{ secBuffer, 0, curSize };
    VkDeviceSize mvOffset = (curSize > radOffsetAligned) ? radOffsetAligned : 0;
    VkDeviceSize mvRange = (curSize > radOffsetAligned) ? (curSize - radOffsetAligned) : curSize;
    VkDescriptorBufferInfo secMvInfo{ secBuffer, mvOffset, mvRange };

    VkDeviceSize ndOffset = (curSize > mvOffsetAligned) ? mvOffsetAligned : 0;
    VkDeviceSize ndRange = (curSize > mvOffsetAligned) ? (curSize - mvOffsetAligned) : curSize;
    VkDescriptorBufferInfo secNdInfo{ secBuffer, ndOffset, ndRange };

    VkDescriptorImageInfo frameInfo{};
    if (frameImage) {
        frameInfo.sampler = VK_NULL_HANDLE;
        frameInfo.imageView = frameImage->getImageView();
        frameInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    VkDescriptorImageInfo mvImageInfo{};
    if (mvImage) {
        mvImageInfo.sampler = VK_NULL_HANDLE;
        mvImageInfo.imageView = mvImage->getImageView();
        mvImageInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    VkDescriptorImageInfo normDepthInfo{};
    if (normDepthImage) {
        normDepthInfo.sampler = VK_NULL_HANDLE;
        normDepthInfo.imageView = normDepthImage->getImageView();
        normDepthInfo.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
    }

    std::vector<VkWriteDescriptorSet> mergeWrites = {
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &frameInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secBufInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secBufInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mvImageInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &normDepthInfo, nullptr, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secMvInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 6, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secNdInfo, nullptr },
        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_mergeDescSets[slot], 7, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &secBufInfo, nullptr }
    };
    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(mergeWrites.size()), mergeWrites.data(), 0, nullptr);
}

void PostProcessPipeline::recordTonemap(VkCommandBuffer cmd, VkDescriptorSet descSet, uint32_t width, uint32_t height, const TonemapPushConstants& pc) {
    if (!m_tonemapPipeline) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_tonemapPipelineLayout, 0, 1, &descSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_tonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(TonemapPushConstants), &pc);
    vkCmdDispatch(cmd, (width + 15) / 16, (height + 15) / 16, 1);
}

void PostProcessPipeline::recordFusedAccumTonemap(VkCommandBuffer cmd, uint32_t frameSlot, const FusedAccumTonemapPushConstants& pc) {
    if (!m_accumTonemapPipeline) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_accumTonemapPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_accumTonemapPipelineLayout, 0, 1, &m_accumTonemapDescSets[frameSlot], 0, nullptr);
    vkCmdPushConstants(cmd, m_accumTonemapPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(FusedAccumTonemapPushConstants), &pc);
    vkCmdDispatch(cmd, (pc.width + 15) / 16, (pc.height + 15) / 16, 1);
}

void PostProcessPipeline::recordRunningAvg(VkCommandBuffer cmd, uint32_t frameSlot, const RunningAvgPushConstants& pc) {
    if (!m_accumRunningAvgPipeline) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_accumRunningAvgPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_accumRunningAvgPipelineLayout, 0, 1, &m_accumRunningAvgDescSets[frameSlot], 0, nullptr);
    vkCmdPushConstants(cmd, m_accumRunningAvgPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(RunningAvgPushConstants), &pc);
    vkCmdDispatch(cmd, (pc.width + 15) / 16, (pc.height + 15) / 16, 1);

    VkMemoryBarrier2 avgBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    avgBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    avgBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    avgBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    avgBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo avgDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    avgDep.memoryBarrierCount = 1;
    avgDep.pMemoryBarriers = &avgBarrier;
    vkCmdPipelineBarrier2(cmd, &avgDep);
}

void PostProcessPipeline::recordBlend4K(VkCommandBuffer cmd, const Blend4KPushConstants& pc) {
    if (!m_fsr3BlendPipeline || m_fsr3BlendDescSet == VK_NULL_HANDLE) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fsr3BlendPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fsr3BlendPipelineLayout, 0, 1, &m_fsr3BlendDescSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_fsr3BlendPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(Blend4KPushConstants), &pc);
    vkCmdDispatch(cmd, (pc.width + 15) / 16, (pc.height + 15) / 16, 1);

    VkMemoryBarrier2 blendToTmBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    blendToTmBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    blendToTmBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    blendToTmBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    blendToTmBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo blendDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    blendDep.memoryBarrierCount = 1;
    blendDep.pMemoryBarriers = &blendToTmBarrier;
    vkCmdPipelineBarrier2(cmd, &blendDep);
}

void PostProcessPipeline::recordMerge(VkCommandBuffer cmd, uint32_t slot, const MergePushConstants& pc, uint32_t groupCountX, uint32_t groupCountY) {
    if (!m_mergePipeline || slot >= 2 || m_mergeDescSets[slot] == VK_NULL_HANDLE) return;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_mergePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_mergePipelineLayout, 0, 1, &m_mergeDescSets[slot], 0, nullptr);
    vkCmdPushConstants(cmd, m_mergePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(MergePushConstants), &pc);
    vkCmdDispatch(cmd, groupCountX, groupCountY, 1);
}

} // namespace pathways
