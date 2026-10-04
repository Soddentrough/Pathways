#pragma once

#include "scene/ProceduralScene.hpp"
#include "scene/Material.hpp"
#include "core/Config.hpp"
#include "vulkan/Buffer.hpp"
#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"

#include <vector>
#include <cstdint>
#include <memory>

namespace pathways {

class SceneGeometryPipeline {
public:
    SceneGeometryPipeline(VkDevice device, VkQueue queue, VmaAllocator allocator, VkCommandPool commandPool);
    ~SceneGeometryPipeline() = default;

    SceneGeometryPipeline(const SceneGeometryPipeline&) = delete;
    SceneGeometryPipeline& operator=(const SceneGeometryPipeline&) = delete;

    /// Clusters prototype instances in scene into spatial Macro-BLASes to reduce TLAS instance count.
    void clusterInstancesToMacroBlas(SceneData& scene, const Config& config);

    /// Partitions scene triangles into opaque and non-opaque ranges for two-level ray traversal / any-hit bypassing.
    void partitionSceneGeometry(SceneData& sceneData, uint32_t& outNumOpaqueTriangles);

    /// Checks if any material in the scene possesses non-opaque properties (transmission, dielectrics, alpha mask).
    void updateSceneTransparency(const std::vector<MaterialGPU>& materials, bool& outHasNonOpaque, bool& outHasAlphaMask);

    /// Uploads arbitrary host data into a device-local Vulkan buffer in bounded 64 MB chunks using a staging buffer.
    void uploadToDeviceBuffer(Buffer& dstBuffer, const void* srcData, VkDeviceSize dataSize);

    /// Generates sequential 32-bit triangle indices (0,1,2, 3,4,5...) in 1M triangle chunks and uploads to device buffer.
    void uploadIndexBuffer(Buffer& dstBuffer, uint32_t triangleCount);

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;
};

} // namespace pathways
