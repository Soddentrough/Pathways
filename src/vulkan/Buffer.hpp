#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include <cstdint>

namespace pathways {

class Buffer {
public:
    Buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
           VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags flags = 0, VkDeviceSize minAlignment = 0,
           VkBufferUsageFlags2KHR usage2 = 0);
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&& other) noexcept;
    Buffer& operator=(Buffer&& other) noexcept;

    VkBuffer getBuffer() const { return m_buffer; }
    VkDeviceSize getSize() const { return m_size; }
    VmaAllocation getAllocation() const { return m_allocation; }
    VkDeviceAddress getDeviceAddress(VkDevice device) const;

    void* map();
    void unmap();
    void copyFrom(const void* data, VkDeviceSize size);
    void invalidate(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);
    void flush(VkDeviceSize offset = 0, VkDeviceSize size = VK_WHOLE_SIZE);

private:
    void release();

    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkBuffer m_buffer = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VkDeviceSize m_size = 0;
    void* m_mappedData = nullptr;
};

} // namespace pathways
