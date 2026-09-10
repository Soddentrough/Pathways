#pragma once

#include <vulkan/vulkan.h>
#include "vulkan/Buffer.hpp"
#include <memory>
#include <vector>

namespace pathways {

struct DGCCommand {
    uint32_t pipelineIndex; // Token 0: VK_INDIRECT_COMMANDS_TOKEN_TYPE_EXECUTION_SET_EXT
    uint32_t groupCountX;   // Token 1: VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT
    uint32_t groupCountY;
    uint32_t groupCountZ;
};

class DGCManager {
public:
    DGCManager(VkDevice device, VmaAllocator allocator, VkPipelineLayout pipelineLayout);
    ~DGCManager();

    bool isSupported() const { return m_supported; }
    VkIndirectCommandsLayoutEXT getLayout() const { return m_indirectLayout; }
    VkIndirectExecutionSetEXT getExecutionSet() const { return m_executionSet; }

    // Initialize execution set with the compute microkernel pipelines
    // [0] classify, [1] intersect, [2] shade, [3] shadow, [4] resolve
    void initExecutionSet(const std::vector<VkPipeline>& pipelines);

    // Asynchronous preprocessing of indirect commands
    void recordPreprocess(VkCommandBuffer cmd, VkPipeline pipeline, Buffer* argumentBuffer,
                          VkDeviceSize argumentOffset = 0, uint32_t sliceIndex = 0, uint32_t maxSequenceCount = 1);

    // Execute generated commands (with execution set + dispatch token)
    void recordExecute(VkCommandBuffer cmd, VkPipeline pipeline, Buffer* argumentBuffer,
                       VkDeviceSize argumentOffset = 0, uint32_t sliceIndex = 0,
                       uint32_t maxSequenceCount = 1, bool isPreprocessed = true);

    // Standard indirect dispatch fallback (for when DGC execution is disabled or bypassed)
    void recordIndirectDispatch(VkCommandBuffer cmd, Buffer* argumentBuffer, VkDeviceSize argumentOffset = 0);

    // Material execution set support (Techniques A, B, C)
    bool isMaterialDGCSupported() const { return m_materialDGCSupported; }
    VkIndirectExecutionSetEXT getMaterialExecutionSet() const { return m_materialExecutionSet; }
    void initMaterialExecutionSet(const std::vector<VkPipeline>& materialPipelines);
    void recordMaterialExecute(VkCommandBuffer cmd, const std::vector<VkPipeline>& pipelines,
                               Buffer* argumentBuffer, VkDeviceSize argumentOffset = 0,
                               uint32_t sliceIndex = 0, uint32_t sequenceCount = 4);

private:
    void loadFunctionPointers();
    void ensurePreprocessBuffer(VkPipeline pipeline, uint32_t maxSequenceCount);

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;

    VkIndirectCommandsLayoutEXT m_indirectLayout = VK_NULL_HANDLE;
    VkIndirectExecutionSetEXT m_executionSet = VK_NULL_HANDLE;
    VkIndirectCommandsLayoutEXT m_materialIndirectLayout = VK_NULL_HANDLE;
    VkIndirectExecutionSetEXT m_materialExecutionSet = VK_NULL_HANDLE;
    std::unique_ptr<Buffer> m_preprocessBuffer;
    VkDeviceSize m_sliceSize = 4096;
    bool m_supported = false;
    bool m_materialDGCSupported = false;

    PFN_vkCreateIndirectCommandsLayoutEXT pfn_vkCreateIndirectCommandsLayoutEXT = nullptr;
    PFN_vkDestroyIndirectCommandsLayoutEXT pfn_vkDestroyIndirectCommandsLayoutEXT = nullptr;
    PFN_vkCreateIndirectExecutionSetEXT pfn_vkCreateIndirectExecutionSetEXT = nullptr;
    PFN_vkDestroyIndirectExecutionSetEXT pfn_vkDestroyIndirectExecutionSetEXT = nullptr;
    PFN_vkUpdateIndirectExecutionSetPipelineEXT pfn_vkUpdateIndirectExecutionSetPipelineEXT = nullptr;
    PFN_vkGetGeneratedCommandsMemoryRequirementsEXT pfn_vkGetGeneratedCommandsMemoryRequirementsEXT = nullptr;
    PFN_vkCmdPreprocessGeneratedCommandsEXT pfn_vkCmdPreprocessGeneratedCommandsEXT = nullptr;
    PFN_vkCmdExecuteGeneratedCommandsEXT pfn_vkCmdExecuteGeneratedCommandsEXT = nullptr;
};

} // namespace pathways
