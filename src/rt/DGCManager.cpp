#include "rt/DGCManager.hpp"
#include "core/Logger.hpp"
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <array>

namespace pathways {

DGCManager::DGCManager(VkDevice device, VmaAllocator allocator, VkPipelineLayout pipelineLayout, bool supportsExecutionSet, bool enableExplicitPreprocess)
    : m_device(device), m_allocator(allocator), m_pipelineLayout(pipelineLayout), m_materialDGCSupported(supportsExecutionSet), m_explicitPreprocess(enableExplicitPreprocess) {

    loadFunctionPointers();
    if (getenv("PATHWAYS_DISABLE_DGC")) {
        throw std::runtime_error("DGC explicitly disabled via PATHWAYS_DISABLE_DGC, but slower fallback workarounds are not supported under Vulkan 1.4 baseline.");
    }
    if (!m_supported) {
        throw std::runtime_error("DGCManager: VK_EXT_device_generated_commands not available or entry points failed to load.");
    }

    if (getenv("PATHWAYS_DISABLE_DGC_PREPROCESS")) {
        m_explicitPreprocess = false;
    } else if (getenv("PATHWAYS_ENABLE_DGC_PREPROCESS")) {
        m_explicitPreprocess = true;
    }

    if (!m_explicitPreprocess) {
        Logger::info("DGC baseline active (implicit preprocessing, flags = UNORDERED_SEQUENCES).");
    } else {
        Logger::info("DGC Tier 1 explicit preprocessing enabled.");
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
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create DGC indirect commands layout (code: " + std::to_string(res) + ")");
    }
    Logger::info("Created DGC indirect commands layout (Single Dispatch Token, stride: {} bytes, flags: 0x{:x}).",
                 createInfo.indirectStride, createInfo.flags);

    if (supportsExecutionSet) {
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
        } else {
            Logger::warn("Note: DGC Material indirect commands layout not supported by driver (code: {}). Using multi-dispatch indirect.", (int)matRes);
            m_materialDGCSupported = false;
        }
    } else {
        m_materialDGCSupported = false;
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
    if (m_materialExecutionSetSecondary && pfn_vkDestroyIndirectExecutionSetEXT) {
        pfn_vkDestroyIndirectExecutionSetEXT(m_device, m_materialExecutionSetSecondary, nullptr);
        m_materialExecutionSetSecondary = VK_NULL_HANDLE;
    }
    m_preprocessBuffer.reset();
    m_materialPreprocessBuffer.reset();
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

    // Single-dispatch DGC token layout does not require an Execution Set (uses VK_NULL_HANDLE).
    // Pre-allocate preprocess buffer with room for all sequences across the multi-slice ring buffer.
    ensurePreprocessBuffer(pipelines[0], 16);
}

void DGCManager::ensurePreprocessBuffer(VkPipeline pipeline, uint32_t maxSequenceCount) {
    if (!m_supported || !m_indirectLayout) return;

    if (!m_preprocessBuffer) {
        uint32_t targetSequenceCount = std::max(maxSequenceCount, 16u);

        VkGeneratedCommandsPipelineInfoEXT pipeInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_PIPELINE_INFO_EXT };
        pipeInfo.pipeline = pipeline;

        VkGeneratedCommandsMemoryRequirementsInfoEXT memInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT };
        memInfo.pNext = &pipeInfo;
        memInfo.indirectExecutionSet = VK_NULL_HANDLE;
        memInfo.indirectCommandsLayout = m_indirectLayout;
        memInfo.maxSequenceCount = targetSequenceCount;
        memInfo.maxDrawCount = 0;

        VkMemoryRequirements2 memReqs{ VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
        pfn_vkGetGeneratedCommandsMemoryRequirementsEXT(m_device, &memInfo, &memReqs);

        VkDeviceSize reqSize = std::max<VkDeviceSize>(memReqs.memoryRequirements.size, 1024);
        VkDeviceSize align = std::max<VkDeviceSize>(memReqs.memoryRequirements.alignment, 256);
        m_sliceSize = ((reqSize + align - 1) / align) * align;
        if (m_sliceSize < 4096) m_sliceSize = 4096;
        VkDeviceSize totalSize = m_sliceSize * NUM_SLICES;

        m_preprocessBuffer = std::make_unique<Buffer>(
            m_allocator, totalSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, align,
            VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT
        );
        Logger::info("Allocated DGC preprocess buffer (total: {} bytes, {} slices @ {} bytes, align: {} bytes).",
                     totalSize, NUM_SLICES, m_sliceSize, align);
    }
}

void DGCManager::recordPreprocess(VkCommandBuffer cmd, VkPipeline pipeline, Buffer* argumentBuffer,
                                  VkDeviceSize argumentOffset, uint32_t sliceIndex,
                                  uint32_t maxSequenceCount, VkDeviceAddress sequenceCountAddress) {
    if (!m_supported || !m_explicitPreprocess || !argumentBuffer || pipeline == VK_NULL_HANDLE) return;

    ensurePreprocessBuffer(pipeline, maxSequenceCount);
    if (!m_preprocessBuffer) return;

    // Bind initial pipeline before preprocessing
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    VkDeviceSize sliceOffset = static_cast<VkDeviceSize>(sliceIndex % NUM_SLICES) * m_sliceSize;

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

void DGCManager::recordPreprocessBarrier(VkCommandBuffer cmd, uint32_t sliceIndex, uint32_t sliceCount) {
    if (!m_supported || !m_explicitPreprocess) return;

    std::vector<VkBufferMemoryBarrier2> barriers;
    if (m_preprocessBuffer) {
        VkBufferMemoryBarrier2 bufferBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
        bufferBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT;
        bufferBarrier.srcAccessMask = VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_EXT;
        bufferBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT |
                                     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        bufferBarrier.dstAccessMask = VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT |
                                      VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        bufferBarrier.buffer = m_preprocessBuffer->getBuffer();

        if (sliceIndex != UINT32_MAX && (static_cast<VkDeviceSize>(sliceIndex % NUM_SLICES) * m_sliceSize < m_preprocessBuffer->getSize())) {
            bufferBarrier.offset = static_cast<VkDeviceSize>(sliceIndex % NUM_SLICES) * m_sliceSize;
            bufferBarrier.size = std::min(static_cast<VkDeviceSize>(sliceCount) * m_sliceSize, m_preprocessBuffer->getSize() - bufferBarrier.offset);
        } else {
            bufferBarrier.offset = 0;
            bufferBarrier.size = VK_WHOLE_SIZE;
        }
        barriers.push_back(bufferBarrier);
    }
    if (m_materialPreprocessBuffer) {
        VkBufferMemoryBarrier2 bufferBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
        bufferBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT;
        bufferBarrier.srcAccessMask = VK_ACCESS_2_COMMAND_PREPROCESS_WRITE_BIT_EXT;
        bufferBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT |
                                     VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT;
        bufferBarrier.dstAccessMask = VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT |
                                      VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT;
        bufferBarrier.buffer = m_materialPreprocessBuffer->getBuffer();

        if (sliceIndex != UINT32_MAX && (static_cast<VkDeviceSize>(sliceIndex % NUM_SLICES) * m_materialSliceSize < m_materialPreprocessBuffer->getSize())) {
            bufferBarrier.offset = static_cast<VkDeviceSize>(sliceIndex % NUM_SLICES) * m_materialSliceSize;
            bufferBarrier.size = std::min(static_cast<VkDeviceSize>(sliceCount) * m_materialSliceSize, m_materialPreprocessBuffer->getSize() - bufferBarrier.offset);
        } else {
            bufferBarrier.offset = 0;
            bufferBarrier.size = VK_WHOLE_SIZE;
        }
        barriers.push_back(bufferBarrier);
    }

    if (!barriers.empty()) {
        VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        depInfo.bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
        depInfo.pBufferMemoryBarriers = barriers.data();
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }
}

void DGCManager::recordExecute(VkCommandBuffer cmd, VkPipeline pipeline, Buffer* argumentBuffer,
                               VkDeviceSize argumentOffset, uint32_t sliceIndex,
                               uint32_t maxSequenceCount, bool isPreprocessed,
                               VkDeviceAddress sequenceCountAddress) {
    if (!argumentBuffer || pipeline == VK_NULL_HANDLE) return;

    if (!m_supported || !m_indirectLayout) {
        throw std::runtime_error("DGCManager::recordExecute: DGC indirect commands layout is not available.");
    }

    ensurePreprocessBuffer(pipeline, maxSequenceCount);

    // Bind initial pipeline before executing generated commands as required by VUID-vkCmdExecuteGeneratedCommandsEXT-indirectCommandsLayout-11053
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    VkDeviceSize sliceOffset = static_cast<VkDeviceSize>(sliceIndex % NUM_SLICES) * m_sliceSize;

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
    genInfo.sequenceCountAddress = sequenceCountAddress;

    bool executePreprocessed = m_explicitPreprocess && isPreprocessed;
    pfn_vkCmdExecuteGeneratedCommandsEXT(cmd, executePreprocessed ? VK_TRUE : VK_FALSE, &genInfo);
}

void DGCManager::initMaterialExecutionSet(const std::vector<VkPipeline>& materialPipelines) {
    initMaterialExecutionSets(materialPipelines, {});
}

void DGCManager::initMaterialExecutionSets(const std::vector<VkPipeline>& primaryPipelines,
                                         const std::vector<VkPipeline>& secondaryPipelines) {
    bool disabledViaEnv = (getenv("PATHWAYS_DISABLE_MATERIAL_DGC") != nullptr) ||
                          (getenv("PATHWAYS_DISABLE_DGC_EXECSET") != nullptr);
    bool enabledViaEnv = (getenv("PATHWAYS_ENABLE_MATERIAL_DGC") != nullptr) ||
                         (getenv("PATHWAYS_ENABLE_DGC_EXECSET") != nullptr);
    bool enableMaterialDGC = (!disabledViaEnv) && enabledViaEnv;
    if (!enableMaterialDGC || !m_supported || !m_materialIndirectLayout || primaryPipelines.empty()) {
        m_materialDGCSupported = false;
        Logger::info("Material microkernels active via GPU multi-dispatch indirect queues (6 specialized pipelines).");
        return;
    }

    auto createExecSet = [&](const std::vector<VkPipeline>& pipes, const char* name) -> VkIndirectExecutionSetEXT {
        if (pipes.empty()) return VK_NULL_HANDLE;

        VkIndirectExecutionSetPipelineInfoEXT pipelineInfo{ VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_PIPELINE_INFO_EXT };
        pipelineInfo.initialPipeline = pipes[0];
        pipelineInfo.maxPipelineCount = static_cast<uint32_t>(pipes.size());

        VkIndirectExecutionSetCreateInfoEXT execSetCreateInfo{ VK_STRUCTURE_TYPE_INDIRECT_EXECUTION_SET_CREATE_INFO_EXT };
        execSetCreateInfo.type = VK_INDIRECT_EXECUTION_SET_INFO_TYPE_PIPELINES_EXT;
        execSetCreateInfo.info.pPipelineInfo = &pipelineInfo;

        VkIndirectExecutionSetEXT execSet = VK_NULL_HANDLE;
        VkResult res = pfn_vkCreateIndirectExecutionSetEXT(m_device, &execSetCreateInfo, nullptr, &execSet);
        if (res != VK_SUCCESS || execSet == VK_NULL_HANDLE) {
            throw std::runtime_error(std::string("Failed to create ") + name + " VkIndirectExecutionSetEXT (code: " + std::to_string(res) + ")");
        }

        std::vector<VkWriteIndirectExecutionSetPipelineEXT> writes(pipes.size());
        for (size_t i = 0; i < pipes.size(); ++i) {
            writes[i].sType = VK_STRUCTURE_TYPE_WRITE_INDIRECT_EXECUTION_SET_PIPELINE_EXT;
            writes[i].pNext = nullptr;
            writes[i].index = static_cast<uint32_t>(i);
            writes[i].pipeline = pipes[i];
        }

        pfn_vkUpdateIndirectExecutionSetPipelineEXT(m_device, execSet, static_cast<uint32_t>(writes.size()), writes.data());
        return execSet;
    };

    m_materialPreprocessBuffer.reset();
    m_materialSliceSize = 0;
    if (m_materialExecutionSet && pfn_vkDestroyIndirectExecutionSetEXT) {
        pfn_vkDestroyIndirectExecutionSetEXT(m_device, m_materialExecutionSet, nullptr);
        m_materialExecutionSet = VK_NULL_HANDLE;
    }
    if (m_materialExecutionSetSecondary && pfn_vkDestroyIndirectExecutionSetEXT) {
        pfn_vkDestroyIndirectExecutionSetEXT(m_device, m_materialExecutionSetSecondary, nullptr);
        m_materialExecutionSetSecondary = VK_NULL_HANDLE;
    }

    m_materialExecutionSet = createExecSet(primaryPipelines, "primary material");

    if (!secondaryPipelines.empty()) {
        m_materialExecutionSetSecondary = createExecSet(secondaryPipelines, "secondary material");
        Logger::info("Initialized primary and secondary VkIndirectExecutionSetEXT ({} primary, {} secondary pipelines).",
                     primaryPipelines.size(), secondaryPipelines.size());
    } else {
        Logger::info("Initialized material VkIndirectExecutionSetEXT with {} specialized material pipelines.", primaryPipelines.size());
    }
    Logger::warn("Experimental DGC Material Execution Set active. Note: drivers such as Mesa RADV may fail to switch compute pipelines dynamically via indirect execution set tokens.");

    ensureMaterialPreprocessBuffer(static_cast<uint32_t>(primaryPipelines.size()));
}

void DGCManager::ensureMaterialPreprocessBuffer(uint32_t maxSequenceCount) {
    if (!m_supported || !m_materialDGCSupported || !m_materialIndirectLayout || !m_materialExecutionSet) return;
    if (m_materialPreprocessBuffer) return;

    uint32_t targetSequenceCount = std::max(maxSequenceCount, 6u);

    VkGeneratedCommandsMemoryRequirementsInfoEXT memInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT };
    memInfo.pNext = nullptr;
    memInfo.indirectExecutionSet = m_materialExecutionSet;
    memInfo.indirectCommandsLayout = m_materialIndirectLayout;
    memInfo.maxSequenceCount = targetSequenceCount;
    memInfo.maxDrawCount = 0;

    VkMemoryRequirements2 memReqs{ VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
    pfn_vkGetGeneratedCommandsMemoryRequirementsEXT(m_device, &memInfo, &memReqs);

    VkDeviceSize reqSize = std::max<VkDeviceSize>(memReqs.memoryRequirements.size, 1024);
    VkDeviceSize align = std::max<VkDeviceSize>(memReqs.memoryRequirements.alignment, 256);

    if (m_materialExecutionSetSecondary) {
        VkGeneratedCommandsMemoryRequirementsInfoEXT secMemInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_MEMORY_REQUIREMENTS_INFO_EXT };
        secMemInfo.pNext = nullptr;
        secMemInfo.indirectExecutionSet = m_materialExecutionSetSecondary;
        secMemInfo.indirectCommandsLayout = m_materialIndirectLayout;
        secMemInfo.maxSequenceCount = targetSequenceCount;
        secMemInfo.maxDrawCount = 0;

        VkMemoryRequirements2 secMemReqs{ VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
        pfn_vkGetGeneratedCommandsMemoryRequirementsEXT(m_device, &secMemInfo, &secMemReqs);
        reqSize = std::max<VkDeviceSize>(reqSize, secMemReqs.memoryRequirements.size);
        align = std::max<VkDeviceSize>(align, secMemReqs.memoryRequirements.alignment);
    }

    m_materialSliceSize = ((reqSize + align - 1) / align) * align;
    if (m_materialSliceSize < 4096) m_materialSliceSize = 4096;
    VkDeviceSize totalSize = m_materialSliceSize * NUM_SLICES;

    m_materialPreprocessBuffer = std::make_unique<Buffer>(
        m_allocator, totalSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE, 0, align,
        VK_BUFFER_USAGE_2_PREPROCESS_BUFFER_BIT_EXT
    );
    Logger::info("Allocated DGC material preprocess buffer (reqSize: {} bytes, align: {} bytes, slice: {} bytes, total: {} bytes).",
                 reqSize, align, m_materialSliceSize, totalSize);
}

void DGCManager::recordMaterialPreprocess(VkCommandBuffer cmd, const std::vector<VkPipeline>& pipelines,
                                        Buffer* argumentBuffer, VkDeviceSize argumentOffset,
                                        uint32_t sliceIndex, uint32_t sequenceCount,
                                        VkDeviceAddress sequenceCountAddress,
                                        bool isSecondary) {
    if (!m_supported || !m_explicitPreprocess || !argumentBuffer || pipelines.empty()) return;
    VkIndirectExecutionSetEXT targetSet = (isSecondary && m_materialExecutionSetSecondary) ? m_materialExecutionSetSecondary : m_materialExecutionSet;
    if (!m_materialDGCSupported || !m_materialIndirectLayout || !targetSet) return;

    ensureMaterialPreprocessBuffer(sequenceCount);
    if (!m_materialPreprocessBuffer) return;

    // Bind initial pipeline before preprocessing
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[0]);

    VkDeviceSize sliceOffset = static_cast<VkDeviceSize>(sliceIndex % NUM_SLICES) * m_materialSliceSize;

    VkGeneratedCommandsInfoEXT genInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT };
    genInfo.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
    genInfo.indirectExecutionSet = targetSet;
    genInfo.indirectCommandsLayout = m_materialIndirectLayout;
    genInfo.indirectAddress = argumentBuffer->getDeviceAddress(m_device) + argumentOffset;
    genInfo.indirectAddressSize = sizeof(DGCCommand) * sequenceCount;
    genInfo.preprocessAddress = m_materialPreprocessBuffer->getDeviceAddress(m_device) + sliceOffset;
    genInfo.preprocessSize = m_materialSliceSize;
    genInfo.maxSequenceCount = sequenceCount;
    genInfo.sequenceCountAddress = sequenceCountAddress;

    pfn_vkCmdPreprocessGeneratedCommandsEXT(cmd, &genInfo, cmd);
}

void DGCManager::recordMaterialExecute(VkCommandBuffer cmd, const std::vector<VkPipeline>& pipelines,
                                      Buffer* argumentBuffer, VkDeviceSize argumentOffset,
                                      uint32_t sliceIndex, uint32_t sequenceCount,
                                      bool isPreprocessed, VkDeviceAddress sequenceCountAddress,
                                      bool isSecondary) {
    if (!argumentBuffer || pipelines.empty()) return;

    VkIndirectExecutionSetEXT targetSet = (isSecondary && m_materialExecutionSetSecondary) ? m_materialExecutionSetSecondary : m_materialExecutionSet;

    if (!m_materialDGCSupported || !m_materialIndirectLayout || !targetSet || (m_explicitPreprocess && !isPreprocessed)) {
        // Direct multi-dispatch indirect (16 bytes stride per DispatchCommand)
        for (uint32_t k = 0; k < sequenceCount && k < pipelines.size(); ++k) {
            vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[k]);
            vkCmdDispatchIndirect(cmd, argumentBuffer->getBuffer(), argumentOffset + k * 16);
        }
        return;
    }

    ensureMaterialPreprocessBuffer(sequenceCount);

    // Bind initial pipeline before executing generated commands as required by VUID-vkCmdExecuteGeneratedCommandsEXT-indirectCommandsLayout-11053
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[0]);

    VkDeviceSize sliceOffset = static_cast<VkDeviceSize>(sliceIndex % NUM_SLICES) * m_materialSliceSize;

    VkGeneratedCommandsInfoEXT genInfo{ VK_STRUCTURE_TYPE_GENERATED_COMMANDS_INFO_EXT };
    genInfo.shaderStages = VK_SHADER_STAGE_COMPUTE_BIT;
    genInfo.indirectExecutionSet = targetSet;
    genInfo.indirectCommandsLayout = m_materialIndirectLayout;
    genInfo.indirectAddress = argumentBuffer->getDeviceAddress(m_device) + argumentOffset;
    genInfo.indirectAddressSize = sizeof(DGCCommand) * sequenceCount;
    if (m_materialPreprocessBuffer) {
        genInfo.preprocessAddress = m_materialPreprocessBuffer->getDeviceAddress(m_device) + sliceOffset;
        genInfo.preprocessSize = m_materialSliceSize;
    }
    genInfo.maxSequenceCount = sequenceCount;
    genInfo.sequenceCountAddress = sequenceCountAddress;

    bool executePreprocessed = m_explicitPreprocess && isPreprocessed;
    pfn_vkCmdExecuteGeneratedCommandsEXT(cmd, executePreprocessed ? VK_TRUE : VK_FALSE, &genInfo);
}

} // namespace pathways
