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
    m_stagingShaderInstances = {};
    m_stagingInputInstances = {};
}

void GpuTlasUpdatePipeline::initShaderInstances(Buffer* shaderInstBuf, const std::vector<InstanceGPU>& initialInstances) {
    m_shaderInstanceBuffer = shaderInstBuf;
    m_cpuShaderInstances = initialInstances;
    m_stagingShaderInstances = {};
    m_shaderInstancesDirty = false;
}

void GpuTlasUpdatePipeline::initBuffers(const std::vector<ASInstanceInput>& asInstances, AccelerationStructureManager* asManager) {
    if (!asManager || asInstances.empty()) return;
    uint32_t instanceCount = static_cast<uint32_t>(asInstances.size());
    m_tlasInstanceCount = instanceCount;

    // 1. Pure DEVICE_LOCAL TLAS Instance Buffer (64 bytes per VkAccelerationStructureInstanceKHR, written by compute shader)
    VkDeviceSize instanceBufferSize = sizeof(VkAccelerationStructureInstanceKHR) * instanceCount;
    m_tlasInstanceBuffer = std::make_unique<Buffer>(
        m_allocator, instanceBufferSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0,
        64
    );

    // 2. Device-local GPU Instance Data Buffer (96 bytes per ASInstanceGPUData, updated via frame-slot staging)
    VkDeviceSize inputBufferSize = sizeof(ASInstanceGPUData) * instanceCount;
    m_tlasInputInstancesBuffer = std::make_unique<Buffer>(
        m_allocator, inputBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
        VK_BUFFER_USAGE_TRANSFER_DST_BIT,
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

    // 4. Initialize CPU shadow & GPU input instance buffer
    m_cpuInputInstances.resize(instanceCount);
    for (uint32_t i = 0; i < instanceCount; ++i) {
        m_cpuInputInstances[i].transform = asInstances[i].transform;
        m_cpuInputInstances[i].customIndex = asInstances[i].customIndex;
        m_cpuInputInstances[i].mask = asInstances[i].mask;
        m_cpuInputInstances[i].hitGroupId = asInstances[i].hitGroupId;
        m_cpuInputInstances[i].flags = asInstances[i].flags;
        m_cpuInputInstances[i].blasAddress = asInstances[i].blasAddress;
        m_cpuInputInstances[i].pad0 = 0;
        m_cpuInputInstances[i].pad1 = 0;
    }
    if (m_tlasInputInstancesBuffer) {
        m_tlasInputInstancesBuffer->copyFrom(m_cpuInputInstances.data(), inputBufferSize);
    }
    m_stagingInputInstances = {};
    m_needsGpuUpdate = false;

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

void GpuTlasUpdatePipeline::recordGpuTlasUpdate(VkCommandBuffer cmd, uint32_t frameSlot, AccelerationStructureManager* asManager, AccelerationStructure* tlas, bool updateMode) {
    uint32_t slot = frameSlot % 2u;
    bool issuedTransfer = false;

    if (m_shaderInstancesDirty && m_shaderInstanceBuffer && !m_cpuShaderInstances.empty()) {
        VkDeviceSize shaderBytes = std::min<VkDeviceSize>(
            sizeof(InstanceGPU) * m_cpuShaderInstances.size(), m_shaderInstanceBuffer->getSize());
        if (shaderBytes > 0) {
            if (!m_stagingShaderInstances[slot] || m_stagingShaderInstances[slot]->getSize() < shaderBytes) {
                m_stagingShaderInstances[slot] = std::make_unique<Buffer>(
                    m_allocator, std::max(shaderBytes, m_shaderInstanceBuffer->getSize()),
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
                );
            }
            m_stagingShaderInstances[slot]->copyFrom(m_cpuShaderInstances.data(), shaderBytes);
            VkBufferCopy copyRegion{ 0, 0, shaderBytes };
            vkCmdCopyBuffer(cmd, m_stagingShaderInstances[slot]->getBuffer(), m_shaderInstanceBuffer->getBuffer(), 1, &copyRegion);
            issuedTransfer = true;
        }
        m_shaderInstancesDirty = false;
    }

    if (m_needsGpuUpdate && m_tlasInputInstancesBuffer && !m_cpuInputInstances.empty()) {
        VkDeviceSize inputBytes = std::min<VkDeviceSize>(
            sizeof(ASInstanceGPUData) * m_cpuInputInstances.size(), m_tlasInputInstancesBuffer->getSize());
        if (inputBytes > 0) {
            if (!m_stagingInputInstances[slot] || m_stagingInputInstances[slot]->getSize() < inputBytes) {
                m_stagingInputInstances[slot] = std::make_unique<Buffer>(
                    m_allocator, std::max(inputBytes, m_tlasInputInstancesBuffer->getSize()),
                    VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
                );
            }
            m_stagingInputInstances[slot]->copyFrom(m_cpuInputInstances.data(), inputBytes);
            VkBufferCopy copyRegion{ 0, 0, inputBytes };
            vkCmdCopyBuffer(cmd, m_stagingInputInstances[slot]->getBuffer(), m_tlasInputInstancesBuffer->getBuffer(), 1, &copyRegion);
            issuedTransfer = true;
        }
    }

    if (issuedTransfer) {
        VkMemoryBarrier2 xferBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        xferBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        xferBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        xferBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        xferBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        VkDependencyInfo xferDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        xferDep.memoryBarrierCount = 1;
        xferDep.pMemoryBarriers = &xferBarrier;
        vkCmdPipelineBarrier2(cmd, &xferDep);
    }

    if (!m_needsGpuUpdate) {
        return;
    }
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

void GpuTlasUpdatePipeline::syncFromCpuTransforms(const std::vector<InstanceGPU>& shaderInstances,
                                                  const std::vector<ASInstanceGPUData>& primaryInputInstances,
                                                  bool triggerTlasRebuild) {
    if (!shaderInstances.empty()) {
        m_cpuShaderInstances = shaderInstances;
        m_shaderInstancesDirty = true;
    }
    const size_t count = std::min(m_cpuInputInstances.size(), primaryInputInstances.size());
    for (size_t i = 0; i < count; ++i) {
        m_cpuInputInstances[i].transform = primaryInputInstances[i].transform;
    }
    if (triggerTlasRebuild && count > 0) {
        m_needsGpuUpdate = true;
    }
}

void GpuTlasUpdatePipeline::updateInstanceTransform(uint32_t index, const glm::mat4& transform, std::vector<SceneInstance>& sceneInstances) {
    glm::mat4 prevTransform = (index < sceneInstances.size()) ? sceneInstances[index].transform : transform;
    if (index < sceneInstances.size()) {
        sceneInstances[index].transform = transform;
    }
    if (index < m_cpuShaderInstances.size()) {
        m_cpuShaderInstances[index].transform = transform;
        m_cpuShaderInstances[index].invTransform = prevTransform;
        m_shaderInstancesDirty = true;
    }
    if (index >= m_tlasInstanceCount || index >= m_cpuInputInstances.size()) {
        return;
    }
    m_cpuInputInstances[index].transform = transform;
    m_needsGpuUpdate = true;
}

void GpuTlasUpdatePipeline::updateAnimatedInstances(float frameDelta, bool animateObjects, float animSpeed,
                                                   const std::vector<AnimatedInstance>& animInstances,
                                                   std::vector<SceneInstance>& sceneInstances) {
    if (!animateObjects || animSpeed <= 0.0001f || animInstances.empty()) {
        if (m_wasAnimatingLastFrame) {
            for (const auto& anim : animInstances) {
                uint32_t idx = anim.instanceIndex;
                if (idx < sceneInstances.size() && idx < m_cpuShaderInstances.size()) {
                    m_cpuShaderInstances[idx].invTransform = sceneInstances[idx].transform;
                    m_shaderInstancesDirty = true;
                }
            }
            m_wasAnimatingLastFrame = false;
        }
        return;
    }
    m_wasAnimatingLastFrame = true;

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
