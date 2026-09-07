#pragma once

#include <vulkan/vulkan.h>
#include "vulkan/Buffer.hpp"
#include <memory>
#include <vector>

namespace pathways {

struct DGCDispatchCmd {
    uint32_t groupCountX;
    uint32_t groupCountY;
    uint32_t groupCountZ;
};

class DGCManager {
public:
    DGCManager(VkDevice device, VmaAllocator allocator, VkPipelineLayout pipelineLayout,
               uint32_t pushConstantSize = 0, VkShaderStageFlags pushConstantStages = VK_SHADER_STAGE_COMPUTE_BIT);
    ~DGCManager();

    bool isSupported() const { return m_supported; }
    bool isMultiToken() const { return m_pushConstantSize > 0; }
    uint32_t getPushConstantSize() const { return m_pushConstantSize; }
    VkIndirectCommandsLayoutEXT getLayout() const { return m_indirectLayout; }

    void recordExecute(VkCommandBuffer cmd, VkPipeline pipeline, Buffer* argumentBuffer,
                       VkDeviceSize argumentOffset = 0, uint32_t maxSequenceCount = 1,
                       Buffer* sequenceCountBuffer = nullptr, VkDeviceSize sequenceCountOffset = 0);
    void recordIndirectDispatch(VkCommandBuffer cmd, Buffer* argumentBuffer, VkDeviceSize argumentOffset = 0);

private:
    void loadFunctionPointers();

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkIndirectCommandsLayoutEXT m_indirectLayout = VK_NULL_HANDLE;
    std::unique_ptr<Buffer> m_preprocessBuffer;
    bool m_supported = false;
    uint32_t m_pushConstantSize = 0;

    PFN_vkCreateIndirectCommandsLayoutEXT pfn_vkCreateIndirectCommandsLayoutEXT = nullptr;
    PFN_vkDestroyIndirectCommandsLayoutEXT pfn_vkDestroyIndirectCommandsLayoutEXT = nullptr;
    PFN_vkCmdExecuteGeneratedCommandsEXT pfn_vkCmdExecuteGeneratedCommandsEXT = nullptr;
    PFN_vkGetGeneratedCommandsMemoryRequirementsEXT pfn_vkGetGeneratedCommandsMemoryRequirementsEXT = nullptr;
};

} // namespace pathways
