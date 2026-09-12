#pragma once

#include <vulkan/vulkan.h>
#include "vulkan/Buffer.hpp"
#include <vector>
#include <memory>
#include <glm/glm.hpp>

namespace pathways {

struct ASGeometryInput {
    VkDeviceAddress vertexBufferAddress = 0;
    VkDeviceAddress indexBufferAddress = 0;
    uint32_t vertexCount = 0;
    uint32_t triangleCount = 0;
    VkDeviceSize vertexStride = sizeof(glm::vec4) * 2; // pos + normal
    VkIndexType indexType = VK_INDEX_TYPE_UINT32;
    bool isOpaque = true;
};

struct ASInstanceInput {
    VkDeviceAddress blasAddress = 0;
    glm::mat4 transform = glm::mat4(1.0f);
    uint32_t customIndex = 0;
    uint32_t mask = 0xFF;
    uint32_t hitGroupId = 0;
    VkGeometryInstanceFlagsKHR flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
};

// 96-byte GPU input struct matching compute shader update_tlas_instances.comp
struct ASInstanceGPUData {
    glm::mat4 transform = glm::mat4(1.0f); // 64 bytes (column-major transform)
    uint32_t customIndex = 0;              // 4 bytes
    uint32_t mask = 0xFF;                  // 4 bytes
    uint32_t hitGroupId = 0;               // 4 bytes
    uint32_t flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR; // 4 bytes
    uint64_t blasAddress = 0;              // 8 bytes
    uint32_t pad0 = 0;                     // 4 bytes
    uint32_t pad1 = 0;                     // 4 bytes
};
static_assert(sizeof(ASInstanceGPUData) == 96, "ASInstanceGPUData must be exactly 96 bytes (std430 aligned)");

class AccelerationStructure {
public:
    AccelerationStructure(VkDevice device, VmaAllocator allocator);
    ~AccelerationStructure();

    AccelerationStructure(const AccelerationStructure&) = delete;
    AccelerationStructure& operator=(const AccelerationStructure&) = delete;

    AccelerationStructure(AccelerationStructure&& other) noexcept;
    AccelerationStructure& operator=(AccelerationStructure&& other) noexcept;

    VkAccelerationStructureKHR getHandle() const { return m_handle; }
    VkDeviceAddress getDeviceAddress() const { return m_deviceAddress; }
    Buffer* getBuffer() { return m_buffer.get(); }

    void setHandle(VkAccelerationStructureKHR handle, VkDeviceAddress addr, std::unique_ptr<Buffer> buffer);

private:
    void release();

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkAccelerationStructureKHR m_handle = VK_NULL_HANDLE;
    VkDeviceAddress m_deviceAddress = 0;
    std::unique_ptr<Buffer> m_buffer;
};

class AccelerationStructureManager {
public:
    AccelerationStructureManager(VkDevice device, VmaAllocator allocator, VkQueue queue, uint32_t queueFamily);
    ~AccelerationStructureManager();

    std::unique_ptr<AccelerationStructure> buildBLAS(const std::vector<ASGeometryInput>& geometries);
    std::unique_ptr<AccelerationStructure> buildTLAS(const std::vector<ASInstanceInput>& instances);

    // GPU-timeline TLAS sizing, allocation, and build / update (refit)
    VkAccelerationStructureBuildSizesInfoKHR getTLASBuildSizes(uint32_t instanceCount);
    std::unique_ptr<AccelerationStructure> createTLAS(uint32_t instanceCount);
    void recordBuildTLAS(VkCommandBuffer cmd,
                         Buffer* instanceBuffer,
                         uint32_t instanceCount,
                         Buffer* scratchBuffer,
                         AccelerationStructure* dstTlas,
                         bool updateMode = false);

private:
    void loadFunctionPointers();
    void submitCommandBuffer(VkCommandBuffer cmd);

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    uint32_t m_queueFamily = 0;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;

    // Vulkan KHR function pointers
    PFN_vkCreateAccelerationStructureKHR pfn_vkCreateAccelerationStructureKHR = nullptr;
    PFN_vkDestroyAccelerationStructureKHR pfn_vkDestroyAccelerationStructureKHR = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR pfn_vkGetAccelerationStructureBuildSizesKHR = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR pfn_vkCmdBuildAccelerationStructuresKHR = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR pfn_vkGetAccelerationStructureDeviceAddressKHR = nullptr;
};

} // namespace pathways
