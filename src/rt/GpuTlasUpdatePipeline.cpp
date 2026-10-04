#include "rt/GpuTlasUpdatePipeline.hpp"
#include "core/Logger.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <stdexcept>
#include <cstring>
#include <algorithm>
#include <array>

namespace pathways {

GpuTlasUpdatePipeline::GpuTlasUpdatePipeline(VkDevice device, VmaAllocator allocator, bool hasSubgroupSizeControl)
    : m_device(device), m_allocator(allocator), m_hasSubgroupSizeControl(hasSubgroupSizeControl) {
}

GpuTlasUpdatePipeline::~GpuTlasUpdatePipeline() {
    destroyPipelineResources();
}

void GpuTlasUpdatePipeline::destroyPipelineResources() {
    if (m_updateTlasPipeline) {
        vkDestroyPipeline(m_device, m_updateTlasPipeline, nullptr);
        m_updateTlasPipeline = VK_NULL_HANDLE;
    }
    if (m_updateTlasPipelineLayout) {
        vkDestroyPipelineLayout(m_device, m_updateTlasPipelineLayout, nullptr);
        m_updateTlasPipelineLayout = VK_NULL_HANDLE;
    }
    if (m_updateTlasDescLayout) {
        vkDestroyDescriptorSetLayout(m_device, m_updateTlasDescLayout, nullptr);
        m_updateTlasDescLayout = VK_NULL_HANDLE;
    }
    if (m_updateTlasDescPool) {
        vkDestroyDescriptorPool(m_device, m_updateTlasDescPool, nullptr);
        m_updateTlasDescPool = VK_NULL_HANDLE;
    }
    m_updateTlasDescSet = VK_NULL_HANDLE;
    m_tlasInstanceBuffer.reset();
    m_tlasInputInstancesBuffer.reset();
    m_tlasScratchBuffer.reset();
}

void GpuTlasUpdatePipeline::initBuffers(const std::vector<ASInstanceInput>& asInstances, AccelerationStructureManager* asManager) {
    if (!asManager || asInstances.empty()) return;
    uint32_t instanceCount = static_cast<uint32_t>(asInstances.size());
    m_tlasInstanceCount = instanceCount;

    // 1. Device-local TLAS Instance Buffer (64 bytes per VkAccelerationStructureInstanceKHR)
    VkDeviceSize instanceBufferSize = sizeof(VkAccelerationStructureInstanceKHR) * instanceCount;
    m_tlasInstanceBuffer = std::make_unique<Buffer>(
        m_allocator, instanceBufferSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT,
        64
    );

    // 2. Host-visible GPU Instance Data Buffer (96 bytes per ASInstanceGPUData)
    VkDeviceSize inputBufferSize = sizeof(ASInstanceGPUData) * instanceCount;
    m_tlasInputInstancesBuffer = std::make_unique<Buffer>(
        m_allocator, inputBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    // 3. Device-local Scratch Buffer (aligned to 256 bytes)
    auto sizeInfo = asManager->getTLASBuildSizes(instanceCount);
    VkDeviceSize scratchSize = std::max(sizeInfo.buildScratchSize, sizeInfo.updateScratchSize);
    if (scratchSize > 0) {
        scratchSize = (scratchSize + 255) & ~VkDeviceSize(255);
        m_tlasScratchBuffer = std::make_unique<Buffer>(
            m_allocator, scratchSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
            VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            0,
            256
        );
    }

    // 4. Initialize both instance buffers with their exact transforms, BLAS addresses, and metadata
    if (m_tlasInputInstancesBuffer) {
        std::vector<ASInstanceGPUData> initData(instanceCount);
        for (uint32_t i = 0; i < instanceCount; ++i) {
            initData[i].transform = asInstances[i].transform;
            initData[i].customIndex = asInstances[i].customIndex;
            initData[i].mask = asInstances[i].mask;
            initData[i].hitGroupId = asInstances[i].hitGroupId;
            initData[i].flags = asInstances[i].flags;
            initData[i].blasAddress = asInstances[i].blasAddress;
            initData[i].pad0 = 0;
            initData[i].pad1 = 0;
        }
        m_tlasInputInstancesBuffer->copyFrom(initData.data(), sizeof(ASInstanceGPUData) * instanceCount);
    }

    if (m_tlasInstanceBuffer) {
        std::vector<VkAccelerationStructureInstanceKHR> vkInstances(instanceCount);
        for (uint32_t i = 0; i < instanceCount; ++i) {
            VkTransformMatrixKHR vkTransform{};
            for (int r = 0; r < 3; ++r) {
                for (int c = 0; c < 4; ++c) {
                    vkTransform.matrix[r][c] = asInstances[i].transform[c][r];
                }
            }
            vkInstances[i].transform = vkTransform;
            vkInstances[i].instanceCustomIndex = asInstances[i].customIndex;
            vkInstances[i].mask = asInstances[i].mask;
            vkInstances[i].instanceShaderBindingTableRecordOffset = asInstances[i].hitGroupId;
            vkInstances[i].flags = asInstances[i].flags;
            vkInstances[i].accelerationStructureReference = asInstances[i].blasAddress;
        }
        m_tlasInstanceBuffer->copyFrom(vkInstances.data(), instanceBufferSize);
    }

    // 5. Update descriptor set if already created
    updateDescriptors();
}

void GpuTlasUpdatePipeline::initBuffers(uint32_t instanceCount, AccelerationStructureManager* asManager, VkDeviceAddress defaultBlasAddr) {
    std::vector<ASInstanceInput> dummy(instanceCount);
    for (uint32_t i = 0; i < instanceCount; ++i) {
        dummy[i].blasAddress = defaultBlasAddr;
        dummy[i].transform = glm::mat4(1.0f);
        dummy[i].customIndex = i;
    }
    initBuffers(dummy, asManager);
}

void GpuTlasUpdatePipeline::createPipeline(const std::vector<char>& compCode) {
    std::vector<VkDescriptorSetLayoutBinding> bindings = {
        { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
        { 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
    };

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_updateTlasDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS update descriptor set layout!");
    }

    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2 };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;
    poolInfo.maxSets = 1;
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_updateTlasDescPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS update descriptor pool!");
    }

    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = m_updateTlasDescPool;
    allocInfo.descriptorSetCount = 1;
    allocInfo.pSetLayouts = &m_updateTlasDescLayout;
    if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_updateTlasDescSet) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate TLAS update descriptor set!");
    }

    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(uint32_t) * 2;

    VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    plInfo.setLayoutCount = 1;
    plInfo.pSetLayouts = &m_updateTlasDescLayout;
    plInfo.pushConstantRangeCount = 1;
    plInfo.pPushConstantRanges = &pcRange;
    if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_updateTlasPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS update pipeline layout!");
    }

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
    subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
    subgroupSize32.requiredSubgroupSize = 32;

    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = compCode.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(compCode.data());
    VkShaderModule compModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &compModule) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create TLAS update shader module!");
    }

    VkComputePipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = compModule;
    pipeInfo.stage.pName = "main";
    if (m_hasSubgroupSizeControl) {
        pipeInfo.stage.pNext = &subgroupSize32;
    }
    pipeInfo.layout = m_updateTlasPipelineLayout;

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_updateTlasPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, compModule, nullptr);
        throw std::runtime_error("Failed to create TLAS update compute pipeline!");
    }
    vkDestroyShaderModule(m_device, compModule, nullptr);

    updateDescriptors();
    Logger::info("GPU TLAS Instance Writer & Refit Pipeline initialized successfully.");
}

void GpuTlasUpdatePipeline::updateDescriptors() {
    if (m_updateTlasDescSet != VK_NULL_HANDLE && m_tlasInputInstancesBuffer && m_tlasInstanceBuffer) {
        VkDescriptorBufferInfo inInfo{ m_tlasInputInstancesBuffer->getBuffer(), 0, VK_WHOLE_SIZE };
        VkDescriptorBufferInfo outInfo{ m_tlasInstanceBuffer->getBuffer(), 0, VK_WHOLE_SIZE };

        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[0].dstSet = m_updateTlasDescSet;
        writes[0].dstBinding = 0;
        writes[0].descriptorCount = 1;
        writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[0].pBufferInfo = &inInfo;

        writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[1].dstSet = m_updateTlasDescSet;
        writes[1].dstBinding = 1;
        writes[1].descriptorCount = 1;
        writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[1].pBufferInfo = &outInfo;

        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    }
}

void GpuTlasUpdatePipeline::recordGpuTlasUpdate(VkCommandBuffer cmd, AccelerationStructureManager* asManager, AccelerationStructure* tlas, bool updateMode) {
    if (!m_updateTlasPipeline || !m_tlasInstanceBuffer || !m_tlasScratchBuffer || !tlas || m_tlasInstanceCount == 0 || !asManager) {
        return;
    }

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updateTlasPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_updateTlasPipelineLayout, 0, 1, &m_updateTlasDescSet, 0, nullptr);

    struct {
        uint32_t instanceCount;
        uint32_t updateMode;
    } pc = { m_tlasInstanceCount, updateMode ? 1u : 0u };
    vkCmdPushConstants(cmd, m_updateTlasPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

    uint32_t groupCountX = (m_tlasInstanceCount + 63) / 64;
    vkCmdDispatch(cmd, groupCountX, 1, 1);

    asManager->recordBuildTLAS(cmd, m_tlasInstanceBuffer.get(), m_tlasInstanceCount, m_tlasScratchBuffer.get(), tlas, updateMode);
    m_tlasGpuUpdateCount++;
    m_needsGpuUpdate = false;
}

void GpuTlasUpdatePipeline::updateInstanceTransform(uint32_t index, const glm::mat4& transform, std::vector<SceneInstance>& sceneInstances) {
    if (index < sceneInstances.size()) {
        sceneInstances[index].transform = transform;
    }
    if (!m_tlasInputInstancesBuffer || index >= m_tlasInstanceCount) {
        return;
    }
    VkDeviceSize offset = index * sizeof(ASInstanceGPUData) + offsetof(ASInstanceGPUData, transform);
    void* mapped = m_tlasInputInstancesBuffer->map();
    if (mapped) {
        std::memcpy(static_cast<char*>(mapped) + offset, &transform, sizeof(glm::mat4));
        vmaFlushAllocation(m_allocator, m_tlasInputInstancesBuffer->getAllocation(), offset, sizeof(glm::mat4));
        m_needsGpuUpdate = true;
    }
}

void GpuTlasUpdatePipeline::updateAnimatedInstances(float frameDelta, bool animateObjects, float animSpeed,
                                                   const std::vector<AnimatedInstance>& animInstances,
                                                   std::vector<SceneInstance>& sceneInstances) {
    if (!animateObjects || animSpeed <= 0.0001f || animInstances.empty()) {
        return;
    }

    // Fixed timestep accumulator (Fix-Your-Timestep decoupled from display rate)
    float effectiveDelta = std::min(frameDelta, 0.1f) * animSpeed;
    m_simAccumulator += effectiveDelta;
    while (m_simAccumulator >= SIMULATION_FIXED_TIMESTEP) {
        m_simTime += SIMULATION_FIXED_TIMESTEP;
        m_simAccumulator -= SIMULATION_FIXED_TIMESTEP;
    }

    float alpha = m_simAccumulator / SIMULATION_FIXED_TIMESTEP;
    float renderTime = m_simTime + alpha * SIMULATION_FIXED_TIMESTEP;

    for (const auto& anim : animInstances) {
        if (anim.instanceIndex >= m_tlasInstanceCount) continue;
        float angle = renderTime * anim.rotationSpeed;
        glm::mat4 rotation = glm::rotate(glm::mat4(1.0f), angle, anim.rotationAxis);
        glm::mat4 transform = glm::translate(glm::mat4(1.0f), anim.basePosition) * rotation * anim.baseTransform;
        updateInstanceTransform(anim.instanceIndex, transform, sceneInstances);
    }
}

} // namespace pathways
