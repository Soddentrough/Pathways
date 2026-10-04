#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan/Buffer.hpp"
#include "rt/AccelerationStructure.hpp"
#include "scene/ProceduralScene.hpp"

#include <glm/glm.hpp>
#include <vector>
#include <memory>
#include <cstdint>

namespace pathways {

class GpuTlasUpdatePipeline {
public:
    static constexpr float SIMULATION_FIXED_TIMESTEP = 1.0f / 60.0f; // 60 Hz fixed tick

    GpuTlasUpdatePipeline(VkDevice device, VmaAllocator allocator, bool hasSubgroupSizeControl);
    ~GpuTlasUpdatePipeline();

    GpuTlasUpdatePipeline(const GpuTlasUpdatePipeline&) = delete;
    GpuTlasUpdatePipeline& operator=(const GpuTlasUpdatePipeline&) = delete;

    void initBuffers(const std::vector<ASInstanceInput>& asInstances, AccelerationStructureManager* asManager);
    void initBuffers(uint32_t instanceCount, AccelerationStructureManager* asManager, VkDeviceAddress defaultBlasAddr);
    void createPipeline(const std::vector<char>& compCode);
    void updateDescriptors();

    void recordGpuTlasUpdate(VkCommandBuffer cmd, AccelerationStructureManager* asManager, AccelerationStructure* tlas, bool updateMode = true);
    void updateInstanceTransform(uint32_t index, const glm::mat4& transform, std::vector<SceneInstance>& sceneInstances);
    void updateAnimatedInstances(float frameDelta, bool animateObjects, float animSpeed,
                                const std::vector<AnimatedInstance>& animInstances,
                                std::vector<SceneInstance>& sceneInstances);

    [[nodiscard]] bool needsGpuUpdate() const noexcept { return m_needsGpuUpdate; }
    void markDirty() noexcept { m_needsGpuUpdate = true; }
    void clearDirty() noexcept { m_needsGpuUpdate = false; }
    [[nodiscard]] uint32_t getGpuUpdateCount() const noexcept { return m_tlasGpuUpdateCount; }
    [[nodiscard]] uint32_t getInstanceCount() const noexcept { return m_tlasInstanceCount; }

    [[nodiscard]] Buffer* getInstanceBuffer() const noexcept { return m_tlasInstanceBuffer.get(); }
    [[nodiscard]] Buffer* getInputInstancesBuffer() const noexcept { return m_tlasInputInstancesBuffer.get(); }
    [[nodiscard]] Buffer* getScratchBuffer() const noexcept { return m_tlasScratchBuffer.get(); }
    [[nodiscard]] VkPipeline getPipeline() const noexcept { return m_updateTlasPipeline; }
    [[nodiscard]] VkPipelineLayout getPipelineLayout() const noexcept { return m_updateTlasPipelineLayout; }
    [[nodiscard]] VkDescriptorSet getDescriptorSet() const noexcept { return m_updateTlasDescSet; }

private:
    void destroyPipelineResources();

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    bool m_hasSubgroupSizeControl = false;

    // GPU-Timeline TLAS Instance & Scratch Buffers (Tier 3)
    std::unique_ptr<Buffer> m_tlasInstanceBuffer;
    std::unique_ptr<Buffer> m_tlasInputInstancesBuffer;
    std::unique_ptr<Buffer> m_tlasScratchBuffer;
    uint32_t m_tlasInstanceCount = 0;
    bool m_needsGpuUpdate = false;
    uint32_t m_tlasGpuUpdateCount = 0;

    // Pipeline & Descriptors
    VkDescriptorSetLayout m_updateTlasDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_updateTlasDescPool = VK_NULL_HANDLE;
    VkDescriptorSet m_updateTlasDescSet = VK_NULL_HANDLE;
    VkPipelineLayout m_updateTlasPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_updateTlasPipeline = VK_NULL_HANDLE;

    // Decoupled Simulation & Animation Timing (Fix-Your-Timestep)
    float m_simAccumulator = 0.0f;
    float m_simTime = 0.0f;
};

} // namespace pathways
