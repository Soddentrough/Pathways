#pragma once

#include "rt/AccelerationStructure.hpp"
#include "vulkan/Buffer.hpp"
#include "scene/ProceduralScene.hpp"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <memory>
#include <vector>

namespace pathways {

class SceneGeometryPipeline;

class AccelerationStructurePipeline {
public:
    AccelerationStructurePipeline(VkDevice device, VmaAllocator allocator, VkQueue queue, uint32_t queueFamily);
    ~AccelerationStructurePipeline();

    AccelerationStructurePipeline(const AccelerationStructurePipeline&) = delete;
    AccelerationStructurePipeline& operator=(const AccelerationStructurePipeline&) = delete;

    /// Builds the index buffer, instance buffer, BLASes (monolithic or batch multi-BLAS), and TLAS.
    void build(
        const SceneData& sceneData,
        uint32_t numTriangles,
        uint32_t numOpaqueTriangles,
        Buffer* positionBuffer,
        SceneGeometryPipeline* geomPipeline
    );

    /// Resets all acceleration structures and buffers.
    void reset();

    [[nodiscard]] AccelerationStructure* getTlas() const noexcept { return m_tlas.get(); }
    [[nodiscard]] VkAccelerationStructureKHR getTlasHandle() const noexcept {
        return m_tlas ? m_tlas->getHandle() : VK_NULL_HANDLE;
    }
    [[nodiscard]] AccelerationStructureManager* getAsManager() const noexcept { return m_asManager.get(); }
    [[nodiscard]] Buffer* getInstanceBuffer() const noexcept { return m_instanceBuffer.get(); }
    [[nodiscard]] Buffer* getIndexBuffer() const noexcept { return m_asIndexBuffer.get(); }
    [[nodiscard]] const std::vector<std::unique_ptr<AccelerationStructure>>& getBlases() const noexcept { return m_blases; }
    [[nodiscard]] const std::vector<ASInstanceInput>& getAsInstances() const noexcept { return m_asInstances; }
    [[nodiscard]] VkDeviceAddress getDefaultBlasAddress() const noexcept {
        if (!m_blases.empty() && m_blases[0]) {
            return m_blases[0]->getDeviceAddress();
        }
        return 0;
    }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queueFamily = 0;

    std::unique_ptr<Buffer> m_asIndexBuffer;
    std::unique_ptr<Buffer> m_instanceBuffer;
    std::unique_ptr<AccelerationStructureManager> m_asManager;
    std::vector<std::unique_ptr<AccelerationStructure>> m_blases;
    std::unique_ptr<AccelerationStructure> m_tlas;
    std::vector<ASInstanceInput> m_asInstances;
};

} // namespace pathways
