#include "rt/DGCManager.hpp"
#include "core/Logger.hpp"
#include <cstring>
#include <stdexcept>

namespace pathways {

DGCManager::DGCManager(VkDevice device, VmaAllocator allocator, VkPipelineLayout pipelineLayout,
                       uint32_t pushConstantSize, VkShaderStageFlags pushConstantStages)
    : m_device(device), m_allocator(allocator), m_pipelineLayout(pipelineLayout),
      m_pushConstantSize(pushConstantSize) {

    loadFunctionPointers();
    if (!m_supported) {
        Logger::warn("DGCManager: VK_EXT_device_generated_commands not available or entry points failed to load.");
        return;
    }

    std::vector<VkIndirectCommandsLayoutTokenEXT> tokens;
    VkIndirectCommandsPushConstantTokenEXT pushToken{};

    if (pushConstantSize > 0) {
        pushToken.updateRange.stageFlags = pushConstantStages;
        pushToken.updateRange.offset = 0;
        pushToken.updateRange.size = pushConstantSize;

        VkIndirectCommandsLayoutTokenEXT t0{};
        t0.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
        t0.type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_PUSH_CONSTANT_EXT;
        t0.data.pPushConstant = &pushToken;
        t0.offset = 0;
        tokens.push_back(t0);

        VkIndirectCommandsLayoutTokenEXT t1{};
        t1.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
        t1.type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT;
        t1.offset = pushConstantSize;
        tokens.push_back(t1);
    } else {
        VkIndirectCommandsLayoutTokenEXT token{};
        token.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
        token.type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT;
        token.offset = 0;
        tokens.push_back(token);
    }

    VkIndirectCommandsLayoutCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT;
    createInfo.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
    createInfo.indirectStride = pushConstantSize + sizeof(VkDispatchIndirectCommand);
    createInfo.pipelineLayout = m_pipelineLayout;
    createInfo.tokenCount = static_cast<uint32_t>(tokens.size());
    createInfo.pTokens = tokens.data();

    VkResult res = pfn_vkCreateIndirectCommandsLayoutEXT(m_device, &createInfo, nullptr, &m_indirectLayout);
    if (res == VK_SUCCESS) {
        Logger::info("Created DGC indirect commands layout (tokens: {}, stride: {} bytes).",
                     createInfo.tokenCount, createInfo.indirectStride);
    } else if (pushConstantSize > 0) {
        Logger::warn("Multi-token DGC layout creation failed (code: {}). Falling back to single-token dispatch layout.", (int)res);
        m_pushConstantSize = 0;
        tokens.clear();
        VkIndirectCommandsLayoutTokenEXT token{};
        token.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
        token.type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT;
        token.offset = 0;
        tokens.push_back(token);
        createInfo.indirectStride = sizeof(VkDispatchIndirectCommand);
        createInfo.tokenCount = 1;
        createInfo.pTokens = tokens.data();
        res = pfn_vkCreateIndirectCommandsLayoutEXT(m_device, &createInfo, nullptr, &m_indirectLayout);
        if (res == VK_SUCCESS) {
            Logger::info("Created fallback single-token DGC layout (stride: {} bytes).", createInfo.indirectStride);
        } else {
            Logger::warn("Failed to create fallback DGC layout (code: {}). DGC disabled.", (int)res);
            m_supported = false;
        }
    } else {
        Logger::warn("Failed to create DGC indirect commands layout (code: {}). DGC disabled.", (int)res);
        m_supported = false;
    }
}

DGCManager::~DGCManager() {
    if (m_indirectLayout && pfn_vkDestroyIndirectCommandsLayoutEXT) {
        pfn_vkDestroyIndirectCommandsLayoutEXT(m_device, m_indirectLayout, nullptr);
    }
}

void DGCManager::loadFunctionPointers() {
    pfn_vkCreateIndirectCommandsLayoutEXT = (PFN_vkCreateIndirectCommandsLayoutEXT)vkGetDeviceProcAddr(m_device, "vkCreateIndirectCommandsLayoutEXT");
    pfn_vkDestroyIndirectCommandsLayoutEXT = (PFN_vkDestroyIndirectCommandsLayoutEXT)vkGetDeviceProcAddr(m_device, "vkDestroyIndirectCommandsLayoutEXT");
    pfn_vkCmdExecuteGeneratedCommandsEXT = (PFN_vkCmdExecuteGeneratedCommandsEXT)vkGetDeviceProcAddr(m_device, "vkCmdExecuteGeneratedCommandsEXT");
    pfn_vkGetGeneratedCommandsMemoryRequirementsEXT = (PFN_vkGetGeneratedCommandsMemoryRequirementsEXT)vkGetDeviceProcAddr(m_device, "vkGetGeneratedCommandsMemoryRequirementsEXT");

    m_supported = (pfn_vkCreateIndirectCommandsLayoutEXT != nullptr &&
                   pfn_vkDestroyIndirectCommandsLayoutEXT != nullptr &&
                   pfn_vkCmdExecuteGeneratedCommandsEXT != nullptr);
}

void DGCManager::recordExecute(VkCommandBuffer cmd, VkPipeline pipeline, Buffer* argumentBuffer,
                              VkDeviceSize argumentOffset, uint32_t maxSequenceCount,
                              Buffer* sequenceCountBuffer, VkDeviceSize sequenceCountOffset) {
    if (!argumentBuffer) return;

    if (!m_supported || !pfn_vkCmdExecuteGeneratedCommandsEXT) {
        // Fallback to standard Vulkan indirect compute dispatch
        vkCmdDispatchIndirect(cmd, argumentBuffer->getBuffer(), argumentOffset);
        return;
    }

    VkGeneratedCommandsPipelineInfoEXT pipelineInfo{};
    pipelineInfo.sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT;
    pipelineInfo.pipeline = pipeline;

    // Check memory requirements for preprocessing if not yet allocated
    if (!m_preprocessBuffer && pfn_vkGetGeneratedCommandsMemoryRequirementsEXT) {
        VkGeneratedCommandsMemoryRequirementsInfoEXT memInfo{};
        memInfo.sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT;
        memInfo.pNext = &pipelineInfo;
        memInfo.indirectCommandsLayout = m_indirectLayout;
        memInfo.maxSequenceCount = maxSequenceCount;
        memInfo.maxDrawCount = 0;

        VkMemoryRequirements2 memReqs{};
        memReqs.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
        pfn_vkGetGeneratedCommandsMemoryRequirementsEXT(m_device, &memInfo, &memReqs);

        if (memReqs.memoryRequirements.size > 0) {
            VkDeviceSize align = std::max<VkDeviceSize>(memReqs.memoryRequirements.alignment, 256);
            m_preprocessBuffer = std::make_unique<Buffer>(
                m_allocator, memReqs.memoryRequirements.size + align,
                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, align,
                VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT
            );
        }
    }

    VkDeviceAddress argAddress = argumentBuffer->getDeviceAddress(m_device) + argumentOffset;
    VkDeviceSize argSize = argumentBuffer->getSize() - argumentOffset;

    VkGeneratedCommandsInfoEXT genInfo{};
    genInfo.sType = VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT;
    genInfo.pNext = &pipelineInfo;
    genInfo.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
    genInfo.indirectCommandsLayout = m_indirectLayout;
    genInfo.indirectAddress = argAddress;
    genInfo.indirectAddressSize = argSize;
    genInfo.maxSequenceCount = maxSequenceCount;
    if (sequenceCountBuffer) {
        genInfo.sequenceCountAddress = sequenceCountBuffer->getDeviceAddress(m_device) + sequenceCountOffset;
    }
    if (m_preprocessBuffer) {
        genInfo.preprocessAddress = m_preprocessBuffer->getDeviceAddress(m_device);
        genInfo.preprocessSize = m_preprocessBuffer->getSize();
    }

    pfn_vkCmdExecuteGeneratedCommandsEXT(cmd, VK_FALSE, &genInfo);
}

void DGCManager::recordIndirectDispatch(VkCommandBuffer cmd, Buffer* argumentBuffer, VkDeviceSize argumentOffset) {
    if (!argumentBuffer) return;
    vkCmdDispatchIndirect(cmd, argumentBuffer->getBuffer(), argumentOffset);
}

} // namespace pathways
