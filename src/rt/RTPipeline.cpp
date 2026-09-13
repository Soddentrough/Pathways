#include "rt/RTPipeline.hpp"
#include "core/Logger.hpp"
#include <cstring>
#include <stdexcept>

namespace pathways {

static inline uint32_t alignUp(uint32_t val, uint32_t alignment) {
    if (alignment == 0) return val;
    return (val + alignment - 1) & ~(alignment - 1);
}

RTPipeline::RTPipeline(VkDevice device, VmaAllocator allocator,
                       const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& rtProps,
                       VkPipelineLayout pipelineLayout,
                       const std::vector<char>& rgenCode,
                       const std::vector<char>& rmissCode,
                       const std::vector<char>& shadowMissCode,
                       const std::vector<char>& rchitCode)
    : m_device(device), m_allocator(allocator), m_rtProps(rtProps), m_pipelineLayout(pipelineLayout) {
    loadFunctionPointers();
    createPipeline(rgenCode, rmissCode, shadowMissCode, rchitCode);
    createShaderBindingTable();
    m_supported = true;
    Logger::info("VK_KHR_ray_tracing_pipeline and Shader Binding Table initialized successfully.");
}

RTPipeline::~RTPipeline() {
    if (m_pipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_pipeline, nullptr);
        m_pipeline = VK_NULL_HANDLE;
    }
}

void RTPipeline::loadFunctionPointers() {
    pfn_vkCreateRayTracingPipelinesKHR = (PFN_vkCreateRayTracingPipelinesKHR)vkGetDeviceProcAddr(m_device, "vkCreateRayTracingPipelinesKHR");
    pfn_vkGetRayTracingShaderGroupHandlesKHR = (PFN_vkGetRayTracingShaderGroupHandlesKHR)vkGetDeviceProcAddr(m_device, "vkGetRayTracingShaderGroupHandlesKHR");
    pfn_vkCmdTraceRaysKHR = (PFN_vkCmdTraceRaysKHR)vkGetDeviceProcAddr(m_device, "vkCmdTraceRaysKHR");
    pfn_vkCmdTraceRaysIndirectKHR = (PFN_vkCmdTraceRaysIndirectKHR)vkGetDeviceProcAddr(m_device, "vkCmdTraceRaysIndirectKHR");

    if (!pfn_vkCreateRayTracingPipelinesKHR || !pfn_vkGetRayTracingShaderGroupHandlesKHR || !pfn_vkCmdTraceRaysKHR) {
        throw std::runtime_error("Failed to load VK_KHR_ray_tracing_pipeline extension function pointers!");
    }
    if (pfn_vkCmdTraceRaysIndirectKHR) {
        Logger::info("Hardware Indirect Ray Tracing (vkCmdTraceRaysIndirectKHR) available and loaded.");
    }
}

VkShaderModule RTPipeline::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule mod = VK_NULL_HANDLE;
    VkResult res = vkCreateShaderModule(m_device, &createInfo, nullptr, &mod);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create shader module for RT pipeline!");
    }
    return mod;
}

void RTPipeline::createPipeline(const std::vector<char>& rgenCode,
                                const std::vector<char>& rmissCode,
                                const std::vector<char>& shadowMissCode,
                                const std::vector<char>& rchitCode) {
    VkShaderModule rgenModule = createShaderModule(rgenCode);
    VkShaderModule rmissModule = createShaderModule(rmissCode);
    VkShaderModule shadowMissModule = createShaderModule(shadowMissCode);
    VkShaderModule rchitModule = createShaderModule(rchitCode);

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    std::vector<VkPipelineShaderStageCreateInfo> stages(4);
    // Stage 0: Raygen
    stages[0].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[0].stage = VK_SHADER_STAGE_RAYGEN_BIT_KHR;
    stages[0].module = rgenModule;
    stages[0].pName = "main";
    stages[0].pNext = &subgroupSize32;

    // Stage 1: Primary Miss
    stages[1].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[1].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[1].module = rmissModule;
    stages[1].pName = "main";
    stages[1].pNext = &subgroupSize32;

    // Stage 2: Shadow Miss
    stages[2].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[2].stage = VK_SHADER_STAGE_MISS_BIT_KHR;
    stages[2].module = shadowMissModule;
    stages[2].pName = "main";
    stages[2].pNext = &subgroupSize32;

    // Stage 3: Closest Hit
    stages[3].sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stages[3].stage = VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR;
    stages[3].module = rchitModule;
    stages[3].pName = "main";
    stages[3].pNext = &subgroupSize32;

    std::vector<VkRayTracingShaderGroupCreateInfoKHR> groups(4);
    // Group 0: Raygen
    groups[0].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[0].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[0].generalShader = 0;
    groups[0].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[0].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Group 1: Primary Miss
    groups[1].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[1].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[1].generalShader = 1;
    groups[1].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[1].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Group 2: Shadow Miss
    groups[2].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[2].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_GENERAL_KHR;
    groups[2].generalShader = 2;
    groups[2].closestHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[2].intersectionShader = VK_SHADER_UNUSED_KHR;

    // Group 3: Closest Hit
    groups[3].sType = VK_STRUCTURE_TYPE_RAY_TRACING_SHADER_GROUP_CREATE_INFO_KHR;
    groups[3].type = VK_RAY_TRACING_SHADER_GROUP_TYPE_TRIANGLES_HIT_GROUP_KHR;
    groups[3].generalShader = VK_SHADER_UNUSED_KHR;
    groups[3].closestHitShader = 3;
    groups[3].anyHitShader = VK_SHADER_UNUSED_KHR;
    groups[3].intersectionShader = VK_SHADER_UNUSED_KHR;

    VkRayTracingPipelineCreateInfoKHR pipeInfo{};
    pipeInfo.sType = VK_STRUCTURE_TYPE_RAY_TRACING_PIPELINE_CREATE_INFO_KHR;
    pipeInfo.stageCount = static_cast<uint32_t>(stages.size());
    pipeInfo.pStages = stages.data();
    pipeInfo.groupCount = static_cast<uint32_t>(groups.size());
    pipeInfo.pGroups = groups.data();
    pipeInfo.maxPipelineRayRecursionDepth = 1; // Iterative tracing in RayGen avoids recursion stack overhead
    pipeInfo.layout = m_pipelineLayout;

    VkResult res = pfn_vkCreateRayTracingPipelinesKHR(m_device, VK_NULL_HANDLE, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_pipeline);

    vkDestroyShaderModule(m_device, rgenModule, nullptr);
    vkDestroyShaderModule(m_device, rmissModule, nullptr);
    vkDestroyShaderModule(m_device, shadowMissModule, nullptr);
    vkDestroyShaderModule(m_device, rchitModule, nullptr);

    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan ray tracing pipeline! Result: " + std::to_string(res));
    }
}

void RTPipeline::createShaderBindingTable() {
    uint32_t handleSize = m_rtProps.shaderGroupHandleSize;
    uint32_t handleAlignment = m_rtProps.shaderGroupHandleAlignment;
    uint32_t baseAlignment = m_rtProps.shaderGroupBaseAlignment;

    uint32_t rgenStride = alignUp(handleSize, baseAlignment);
    uint32_t rgenSize   = rgenStride;

    uint32_t missStride = alignUp(handleSize, handleAlignment);
    uint32_t missSize   = alignUp(2 * missStride, baseAlignment);

    uint32_t hitStride  = alignUp(handleSize, handleAlignment);
    uint32_t hitSize    = alignUp(1 * hitStride, baseAlignment);

    VkDeviceSize sbtBufferSize = rgenSize + missSize + hitSize;

    m_sbtBuffer = std::make_unique<Buffer>(
        m_allocator, sbtBufferSize,
        VK_BUFFER_USAGE_SHADER_BINDING_TABLE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        baseAlignment
    );

    uint32_t groupCount = 4;
    size_t handleStorageSize = groupCount * handleSize;
    std::vector<uint8_t> handles(handleStorageSize);
    VkResult res = pfn_vkGetRayTracingShaderGroupHandlesKHR(m_device, m_pipeline, 0, groupCount, handleStorageSize, handles.data());
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to query ray tracing shader group handles!");
    }

    std::vector<uint8_t> sbtData(sbtBufferSize, 0);
    uint8_t* pRgen = sbtData.data();
    uint8_t* pMiss = sbtData.data() + rgenSize;
    uint8_t* pHit  = sbtData.data() + rgenSize + missSize;

    // Group 0: Raygen
    std::memcpy(pRgen, handles.data() + 0 * handleSize, handleSize);

    // Group 1 & 2: Miss (Primary, Shadow)
    std::memcpy(pMiss + 0 * missStride, handles.data() + 1 * handleSize, handleSize);
    std::memcpy(pMiss + 1 * missStride, handles.data() + 2 * handleSize, handleSize);

    // Group 3: Hit Group (Closest Hit)
    std::memcpy(pHit + 0 * hitStride, handles.data() + 3 * handleSize, handleSize);

    m_sbtBuffer->copyFrom(sbtData.data(), sbtBufferSize);

    VkDeviceAddress sbtAddr = m_sbtBuffer->getDeviceAddress(m_device);

    m_rgenRegion.deviceAddress = sbtAddr;
    m_rgenRegion.stride = rgenStride;
    m_rgenRegion.size = rgenSize;

    m_missRegion.deviceAddress = sbtAddr + rgenSize;
    m_missRegion.stride = missStride;
    m_missRegion.size = missSize;

    m_hitRegion.deviceAddress = sbtAddr + rgenSize + missSize;
    m_hitRegion.stride = hitStride;
    m_hitRegion.size = hitSize;

    m_callableRegion = {};
}

void RTPipeline::traceRays(VkCommandBuffer cmd, uint32_t width, uint32_t height, uint32_t depth) {
    pfn_vkCmdTraceRaysKHR(cmd, &m_rgenRegion, &m_missRegion, &m_hitRegion, &m_callableRegion, width, height, depth);
}

void RTPipeline::traceRaysIndirect(VkCommandBuffer cmd, VkDeviceAddress indirectDeviceAddress) {
    if (pfn_vkCmdTraceRaysIndirectKHR) {
        pfn_vkCmdTraceRaysIndirectKHR(cmd, &m_rgenRegion, &m_missRegion, &m_hitRegion, &m_callableRegion, indirectDeviceAddress);
    } else {
        Logger::warn("RTPipeline::traceRaysIndirect called but vkCmdTraceRaysIndirectKHR not supported!");
    }
}

} // namespace pathways
