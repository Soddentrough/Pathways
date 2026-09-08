#pragma once

#include <vulkan/vulkan.h>
#include "vulkan/Buffer.hpp"
#include <vector>
#include <memory>

namespace pathways {

class RTPipeline {
public:
    RTPipeline(VkDevice device, VmaAllocator allocator,
               const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rtProps,
               VkPipelineLayout pipelineLayout,
               const std::vector<char>& rgenCode,
               const std::vector<char>& rmissCode,
               const std::vector<char>& shadowMissCode,
               const std::vector<char>& rchitCode);
    ~RTPipeline();

    RTPipeline(const RTPipeline&) = delete;
    RTPipeline& operator=(const RTPipeline&) = delete;

    void traceRays(VkCommandBuffer cmd, uint32_t width, uint32_t height, uint32_t depth = 1);
    void traceRaysIndirect(VkCommandBuffer cmd, VkDeviceAddress indirectDeviceAddress);

    VkPipeline getPipeline() const { return m_pipeline; }
    bool isSupported() const { return m_supported; }
    bool isIndirectSupported() const { return pfn_vkCmdTraceRaysIndirectKHR != nullptr; }

private:
    void loadFunctionPointers();
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void createPipeline(const std::vector<char>& rgenCode,
                        const std::vector<char>& rmissCode,
                        const std::vector<char>& shadowMissCode,
                        const std::vector<char>& rchitCode);
    void createShaderBindingTable();

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtProps{};
    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;

    std::unique_ptr<Buffer> m_sbtBuffer;
    VkStridedDeviceAddressRegionKHR m_rgenRegion{};
    VkStridedDeviceAddressRegionKHR m_missRegion{};
    VkStridedDeviceAddressRegionKHR m_hitRegion{};
    VkStridedDeviceAddressRegionKHR m_callableRegion{};

    bool m_supported = false;

    PFN_vkCreateRayTracingPipelinesKHR pfn_vkCreateRayTracingPipelinesKHR = nullptr;
    PFN_vkGetRayTracingShaderGroupHandlesKHR pfn_vkGetRayTracingShaderGroupHandlesKHR = nullptr;
    PFN_vkCmdTraceRaysKHR pfn_vkCmdTraceRaysKHR = nullptr;
    PFN_vkCmdTraceRaysIndirectKHR pfn_vkCmdTraceRaysIndirectKHR = nullptr;
};

} // namespace pathways
