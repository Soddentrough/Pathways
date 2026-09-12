#include "rt/DGCManager.hpp"
#include "core/Logger.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <array>

namespace pathways {

DGCManager::DGCManager(VkDevice device, VmaAllocator allocator, VkPipelineLayout pipelineLayout, bool supportsExecutionSet)
    : m_device(device), m_allocator(allocator), m_pipelineLayout(pipelineLayout), m_materialDGCSupported(supportsExecutionSet) {

    loadFunctionPointers();
    if (getenv("PATHWAYS_DISABLE_DGC")) {
        m_supported = false;
        Logger::info("DGC explicitly disabled via PATHWAYS_DISABLE_DGC environment variable. Using standard indirect dispatch.");
        return;
    }
    if (!m_supported) {
        Logger::warn("DGCManager: VK_EXT_device_generated_commands not available or entry points failed to load.");
        return;
    }

    m_explicitPreprocess = (getenv("PATHWAYS_DISABLE_DGC_TIER1") == nullptr &&
                            getenv("PATHWAYS_DISABLE_DGC_PREPROCESS") == nullptr);
    if (!m_explicitPreprocess) {
        Logger::info("DGC Tier 1 explicit preprocessing disabled. Running DGC baseline (implicit preprocessing, flags = 0).");
    } else {
        Logger::info("DGC Tier 1 optimizations enabled (explicit preprocessing + unordered sequences).");
    }

    // Modern DGC Token Layout:
    // Single Dispatch Token Limitation: Exactly one work-dispatching token strictly last
    VkIndirectCommandsLayoutTokenEXT token{};
    token.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
    token.type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT;
    token.offset = 0;

    VkIndirectCommandsLayoutCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT;
    createInfo.flags = m_explicitPreprocess ?
        (VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_EXT |
         VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT) : 0;
    createInfo.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
    createInfo.indirectStride = sizeof(VkDispatchIndirectCommand); // 12 bytes
    createInfo.pipelineLayout = m_pipelineLayout;
    createInfo.tokenCount = 1;
    createInfo.pTokens = &token;

    VkResult res = pfn_vkCreateIndirectCommandsLayoutEXT(m_device, &createInfo, nullptr, &m_indirectLayout);
    if (res == VK_SUCCESS) {
        Logger::info("Created DGC indirect commands layout (Single Dispatch Token, stride: {} bytes, flags: 0x{:x}).",
                     createInfo.indirectStride, createInfo.flags);
    } else {
        Logger::warn("Failed to create DGC indirect commands layout (code: {}). Falling back to standard indirect dispatch.", (int)res);
        m_supported = false;
    }

    if (m_supported && supportsExecutionSet) {
        VkIndirectCommandsExecutionSetTokenEXT execSetToken{};
        execSetToken.type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT;
        execSetToken.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;

        std::array<VkIndirectCommandsLayoutTokenEXT, 2> matTokens{};
        matTokens[0].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
        matTokens[0].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT;
        matTokens[0].data.pExecutionSet = &execSetToken;
        matTokens[0].offset = 0;

        matTokens[1].sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_TOKEN_EXT;
        matTokens[1].type = VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT;
        matTokens[1].offset = 4;

        VkIndirectCommandsLayoutCreateInfoEXT matCreateInfo{};
        matCreateInfo.sType = VK_STRUCTURE_TYPE_INDIRECT_COMMANDS_LAYOUT_CREATE_INFO_EXT;
        matCreateInfo.flags = m_explicitPreprocess ?
            (VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_EXT |
             VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT) : 0;
        matCreateInfo.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
        matCreateInfo.indirectStride = sizeof(DGCCommand); // 16 bytes
        matCreateInfo.pipelineLayout = m_pipelineLayout;
        matCreateInfo.tokenCount = 2;
        matCreateInfo.pTokens = matTokens.data();

        VkResult matRes = pfn_vkCreateIndirectCommandsLayoutEXT(m_device, &matCreateInfo, nullptr, &m_materialIndirectLayout);
        if (matRes == VK_SUCCESS) {
            Logger::info("Created DGC Material indirect commands layout (ExecutionSet + Dispatch, stride: {} bytes).",
                         matCreateInfo.indirectStride);
            m_materialDGCSupported = true;
        } else {
            Logger::warn("Note: DGC Material indirect commands layout not supported by driver (code: {}). Using multi-dispatch indirect.", (int)matRes);
            m_materialDGCSupported = false;
        }
    } else {
        m_materialDGCSupported = false;
        if (!supportsExecutionSet && m_supported) {
            Logger::info("DGC Execution Sets disabled. Using multi-dispatch indirect fallback.");
        }
    }
}

DGCManager::~DGCManager() {
    if (m_indirectLayout && pfn_vkDestroyIndirectCommandsLayoutEXT) {
        pfn_vkDestroyIndirectCommandsLayoutEXT(m_device, m_indirectLayout, nullptr);
        m_indirectLayout = VK_NULL_HANDLE;
    }
    if (m_materialIndirectLayout && pfn_vkDestroyIndirectCommandsLayoutEXT) {
        pfn_vkDestroyIndirectCommandsLayoutEXT(m_device, m_materialIndirectLayout, nullptr);
        m_materialIndirectLayout = VK_NULL_HANDLE;
    }
    if (m_executionSet && pfn_vkDestroyIndirectExecutionSetEXT) {
        pfn_vkDestroyIndirectExecutionSetEXT(m_device, m_executionSet, nullptr);
        m_executionSet = VK_NULL_HANDLE;
    }
    if (m_materialExecutionSet && pfn_vkDestroyIndirectExecutionSetEXT) {
        pfn_vkDestroyIndirectExecutionSetEXT(m_device, m_materialExecutionSet, nullptr);
        m_materialExecutionSet = VK_NULL_HANDLE;
    }
}

void DGCManager::loadFunctionPointers() {
    pfn_vkCreateIndirectCommandsLayoutEXT = (PFN_vkCreateIndirectCommandsLayoutEXT)vkGetDeviceProcAddr(m_device, "vkCreateIndirectCommandsLayoutEXT");
    pfn_vkDestroyIndirectCommandsLayoutEXT = (PFN_vkDestroyIndirectCommandsLayoutEXT)vkGetDeviceProcAddr(m_device, "vkDestroyIndirectCommandsLayoutEXT");
    pfn_vkCreateIndirectExecutionSetEXT = (PFN_vkCreateIndirectExecutionSetEXT)vkGetDeviceProcAddr(m_device, "vkCreateIndirectExecutionSetEXT");
    pfn_vkDestroyIndirectExecutionSetEXT = (PFN_vkDestroyIndirectExecutionSetEXT)vkGetDeviceProcAddr(m_device, "vkDestroyIndirectExecutionSetEXT");
    pfn_vkUpdateIndirectExecutionSetPipelineEXT = (PFN_vkUpdateIndirectExecutionSetPipelineEXT)vkGetDeviceProcAddr(m_device, "vkUpdateIndirectExecutionSetPipelineEXT");
    pfn_vkGetGeneratedCommandsMemoryRequirementsEXT = (PFN_vkGetGeneratedCommandsMemoryRequirementsEXT)vkGetDeviceProcAddr(m_device, "vkGetGeneratedCommandsMemoryRequirementsEXT");
    pfn_vkCmdPreprocessGeneratedCommandsEXT = (PFN_vkCmdPreprocessGeneratedCommandsEXT)vkGetDeviceProcAddr(m_device, "vkCmdPreprocessGeneratedCommandsEXT");
    pfn_vkCmdExecuteGeneratedCommandsEXT = (PFN_vkCmdExecuteGeneratedCommandsEXT)vkGetDeviceProcAddr(m_device, "vkCmdExecuteGeneratedCommandsEXT");

    m_supported = (pfn_vkCreateIndirectCommandsLayoutEXT != nullptr &&
                   pfn_vkDestroyIndirectCommandsLayoutEXT != nullptr &&
                   pfn_vkCreateIndirectExecutionSetEXT != nullptr &&
                   pfn_vkDestroyIndirectExecutionSetEXT != nullptr &&
                   pfn_vkUpdateIndirectExecutionSetPipelineEXT != nullptr &&
                   pfn_vkGetGeneratedCommandsMemoryRequirementsEXT != nullptr &&
                   pfn_vkCmdPreprocessGeneratedCommandsEXT != nullptr &&
                   pfn_vkCmdExecuteGeneratedCommandsEXT != nullptr);
}

void DGCManager::initExecutionSet(const std::vector<VkPipeline>& pipelines) {
    if (!m_supported || pipelines.empty()) return;

    if (m_executionSet && pfn_vkDestroyIndirectExecutionSetEXT) {
        pfn_vkDestroyIndirectExecutionSetEXT(m_device, m_executionSet, nullptr);
        m_executionSet = VK_NULL_HANDLE;
    }

    VkIndirectExecutionSetPipelineInfoEXT pipelineInfo{ VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT };
    pipelineInfo.initialPipeline = pipelines[0];
    pipelineInfo.maxPipelineCount = static_cast<uint32_t>(pipelines.size());

    VkIndirectExecutionSetCreateInfoEXT execSetCreateInfo{ VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT };
    execSetCreateInfo.type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT;
    execSetCreateInfo.info.pPipelineInfo = &pipelineInfo;

    VkResult res = pfn_vkCreateIndirectExecutionSetEXT(m_device, &execSetCreateInfo, nullptr, &m_executionSet);
    if (res != VK_SUCCESS) {
        Logger::warn("Failed to create VkIndirectExecutionSetEXT (code: {}). DGC disabled.", (int)res);
        m_supported = false;
        return;
    }

    std::vector<VkWriteIndirectExecutionSetPipelineEXT> writes(pipelines.size());
    for (size_t i = 0; i < pipelines.size(); ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT;
        writes[i].pNext = nullptr;
        writes[i].index = static_cast<uint32_t>(i);
        writes[i].pipeline = pipelines[i];
    }

    pfn_vkUpdateIndirectExecutionSetPipelineEXT(m_device, m_executionSet, static_cast<uint32_t>(writes.size()), writes.data());
    Logger::info("Initialized VkIndirectExecutionSetEXT with {} compute microkernel pipelines.", pipelines.size());

    // Pre-allocate preprocess buffer
    ensurePreprocessBuffer(pipelines[0], 1);
}

void DGCManager::ensurePreprocessBuffer(VkPipeline pipeline, uint32_t maxSequenceCount) {
    if (!m_supported || !m_indirectLayout) return;

    if (!m_preprocessBuffer) {
        VkGeneratedCommandsPipelineInfoEXT pipeInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT };
        pipeInfo.pipeline = pipeline;

        VkGeneratedCommandsMemoryRequirementsInfoEXT memInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT };
        memInfo.pNext = &pipeInfo;
        memInfo.indirectExecutionSet = VK_NULL_HANDLE;
        memInfo.indirectCommandsLayout = m_indirectLayout;
        memInfo.maxSequenceCount = maxSequenceCount;
        memInfo.maxDrawCount = 0;

        VkMemoryRequirements2 memReqs{ VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
        pfn_vkGetGeneratedCommandsMemoryRequirementsEXT(m_device, &memInfo, &memReqs);

        VkDeviceSize reqSize = std::max<VkDeviceSize>(memReqs.memoryRequirements.size, 1024);
        VkDeviceSize align = std::max<VkDeviceSize>(memReqs.memoryRequirements.alignment, 256);
        m_sliceSize = ((reqSize + align - 1) / align) * align;
        if (m_sliceSize < 4096) m_sliceSize = 4096;
        VkDeviceSize totalSize = m_sliceSize * 16;

        m_preprocessBuffer = std::make_unique<Buffer>(
            m_allocator, totalSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, align,
            VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT
        );
        Logger::info("Allocated DGC preprocess buffer (total: {} bytes, slice: {} bytes, align: {} bytes).",
                     totalSize, m_sliceSize, align);
    }
}

void DGCManager::recordPreprocess(VkCommandBuffer cmd, VkPipeline pipeline, Buffer* argumentBuffer,
                                  VkDeviceSize argumentOffset, uint32_t sliceIndex, uint32_t maxSequenceCount) {
    if (!m_supported || !m_explicitPreprocess || !argumentBuffer) return;

    ensurePreprocessBuffer(pipeline, maxSequenceCount);
    if (!m_preprocessBuffer) return;

    VkDeviceSize sliceOffset = static_cast<VkDeviceSize>(sliceIndex) * m_sliceSize;

    VkGeneratedCommandsPipelineInfoEXT pipeInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT };
    pipeInfo.pipeline = pipeline;

    VkGeneratedCommandsInfoEXT genInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT };
    genInfo.pNext = &pipeInfo;
    genInfo.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
    genInfo.indirectExecutionSet = VK_NULL_HANDLE;
    genInfo.indirectCommandsLayout = m_indirectLayout;
    genInfo.indirectAddress = argumentBuffer->getDeviceAddress(m_device) + argumentOffset;
    genInfo.indirectAddressSize = sizeof(VkDispatchIndirectCommand) * maxSequenceCount;
    genInfo.preprocessAddress = m_preprocessBuffer->getDeviceAddress(m_device) + sliceOffset;
    genInfo.preprocessSize = m_sliceSize;
    genInfo.maxSequenceCount = maxSequenceCount;

    pfn_vkCmdPreprocessGeneratedCommandsEXT(cmd, &genInfo, cmd);
}

void DGCManager::recordPreprocessBarrier(VkCommandBuffer cmd) {
    if (!m_supported || !m_explicitPreprocess || !m_preprocessBuffer) return;

    VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT;
    barrier.srcAccessMask = VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_EXT;
    barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT |
                           VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                           VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barrier.dstAccessMask = VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT |
                            VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;

    VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    depInfo.memoryBarrierCount = 1;
    depInfo.pMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(cmd, &depInfo);
}

void DGCManager::recordExecute(VkCommandBuffer cmd, VkPipeline pipeline, Buffer* argumentBuffer,
                              VkDeviceSize argumentOffset, uint32_t sliceIndex,
                              uint32_t maxSequenceCount, bool isPreprocessed) {
    if (!argumentBuffer) return;

    if (!m_supported || !m_indirectLayout) {
        // Fallback: Dispatch indirect
        vkCmdDispatchIndirect(cmd, argumentBuffer->getBuffer(), argumentOffset);
        return;
    }

    ensurePreprocessBuffer(pipeline, maxSequenceCount);

    VkDeviceSize sliceOffset = static_cast<VkDeviceSize>(sliceIndex) * m_sliceSize;

    VkGeneratedCommandsPipelineInfoEXT pipeInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT };
    pipeInfo.pipeline = pipeline;

    VkGeneratedCommandsInfoEXT genInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT };
    genInfo.pNext = &pipeInfo;
    genInfo.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
    genInfo.indirectExecutionSet = VK_NULL_HANDLE;
    genInfo.indirectCommandsLayout = m_indirectLayout;
    genInfo.indirectAddress = argumentBuffer->getDeviceAddress(m_device) + argumentOffset;
    genInfo.indirectAddressSize = sizeof(VkDispatchIndirectCommand) * maxSequenceCount;
    if (m_preprocessBuffer) {
        genInfo.preprocessAddress = m_preprocessBuffer->getDeviceAddress(m_device) + sliceOffset;
        genInfo.preprocessSize = m_sliceSize;
    }
    genInfo.maxSequenceCount = maxSequenceCount;

    bool executePreprocessed = m_explicitPreprocess && isPreprocessed;
    pfn_vkCmdExecuteGeneratedCommandsEXT(cmd, executePreprocessed ? VK_TRUE : VK_FALSE, &genInfo);
}

void DGCManager::recordIndirectDispatch(VkCommandBuffer cmd, Buffer* argumentBuffer, VkDeviceSize argumentOffset) {
    if (!argumentBuffer) return;
    vkCmdDispatchIndirect(cmd, argumentBuffer->getBuffer(), argumentOffset);
}

void DGCManager::initMaterialExecutionSet(const std::vector<VkPipeline>& materialPipelines) {
    if (!m_supported || !m_materialDGCSupported || materialPipelines.empty()) return;

    if (m_materialExecutionSet && pfn_vkDestroyIndirectExecutionSetEXT) {
        pfn_vkDestroyIndirectExecutionSetEXT(m_device, m_materialExecutionSet, nullptr);
        m_materialExecutionSet = VK_NULL_HANDLE;
    }

    VkIndirectExecutionSetPipelineInfoEXT pipelineInfo{ VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT };
    pipelineInfo.initialPipeline = materialPipelines[0];
    pipelineInfo.maxPipelineCount = static_cast<uint32_t>(materialPipelines.size());

    VkIndirectExecutionSetCreateInfoEXT execSetCreateInfo{ VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT };
    execSetCreateInfo.type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT;
    execSetCreateInfo.info.pPipelineInfo = &pipelineInfo;

    VkResult res = pfn_vkCreateIndirectExecutionSetEXT(m_device, &execSetCreateInfo, nullptr, &m_materialExecutionSet);
    if (res != VK_SUCCESS) {
        Logger::warn("Failed to create material VkIndirectExecutionSetEXT (code: {}). Falling back to multi-dispatch indirect.", (int)res);
        m_materialDGCSupported = false;
        return;
    }

    std::vector<VkWriteIndirectExecutionSetPipelineEXT> writes(materialPipelines.size());
    for (size_t i = 0; i < materialPipelines.size(); ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT;
        writes[i].pNext = nullptr;
        writes[i].index = static_cast<uint32_t>(i);
        writes[i].pipeline = materialPipelines[i];
    }

    pfn_vkUpdateIndirectExecutionSetPipelineEXT(m_device, m_materialExecutionSet, static_cast<uint32_t>(writes.size()), writes.data());
    Logger::info("Initialized material VkIndirectExecutionSetEXT with {} specialized material pipelines.", materialPipelines.size());
}

void DGCManager::recordMaterialExecute(VkCommandBuffer cmd, const std::vector<VkPipeline>& pipelines,
                                      Buffer* argumentBuffer, VkDeviceSize argumentOffset,
                                      uint32_t sliceIndex, uint32_t sequenceCount) {
    if (!argumentBuffer || pipelines.empty()) return;

    if (!m_materialDGCSupported || !m_materialIndirectLayout || !m_materialExecutionSet) {
        // Direct multi-dispatch indirect fallback (16 bytes stride per DispatchCommand)
        for (uint32_t k = 0; k < sequenceCount && k < pipelines.size(); ++k) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[k]);
            vkCmdDispatchIndirect(cmd, argumentBuffer->getBuffer(), argumentOffset + k * 16);
        }
        return;
    }

    // Bind initial pipeline before executing generated commands as required by VUID-vkCmdExecuteGeneratedCommandsEXT-indirectCommandsLayout-11053
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[0]);

    VkDeviceSize sliceOffset = static_cast<VkDeviceSize>(sliceIndex) * m_sliceSize;

    VkGeneratedCommandsInfoEXT genInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT };
    genInfo.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
    genInfo.indirectExecutionSet = m_materialExecutionSet;
    genInfo.indirectCommandsLayout = m_materialIndirectLayout;
    genInfo.indirectAddress = argumentBuffer->getDeviceAddress(m_device) + argumentOffset;
    genInfo.indirectAddressSize = sizeof(DGCCommand) * sequenceCount;
    if (m_preprocessBuffer) {
        genInfo.preprocessAddress = m_preprocessBuffer->getDeviceAddress(m_device) + sliceOffset;
        genInfo.preprocessSize = m_sliceSize;
    }
    genInfo.maxSequenceCount = sequenceCount;

    pfn_vkCmdExecuteGeneratedCommandsEXT(cmd, VK_FALSE, &genInfo);
}

} // namespace pathways
