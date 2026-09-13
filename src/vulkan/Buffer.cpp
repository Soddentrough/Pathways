#include "vulkan/Buffer.hpp"
#include "core/Logger.hpp"
#include <cstring>
#include <stdexcept>

namespace pathways {

Buffer::Buffer(VmaAllocator allocator, VkDeviceSize size, VkBufferUsageFlags usage,
               VmaMemoryUsage memoryUsage, VmaAllocationCreateFlags flags, VkDeviceSize minAlignment,
               VkBufferUsageFlags2KHR usage2)
    : m_allocator(allocator), m_size(size) {

    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBufferUsageFlags2CreateInfoKHR usage2Info{};
    if (usage2 != 0) {
        usage2Info.sType = VK_STRUCTURE_TYPE_BUFFER_USAGE_FLAGS_2_CREATE_INFO_KHR;
        usage2Info.usage = static_cast<VkBufferUsageFlags2KHR>(usage) | usage2;
        bufferInfo.pNext = &usage2Info;
    }

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = memoryUsage;
    allocInfo.flags = flags;
    allocInfo.minAlignment = minAlignment;

    VkResult res = vmaCreateBuffer(m_allocator, &bufferInfo, &allocInfo, &m_buffer, &m_allocation, nullptr);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer with VMA! Error: " + std::to_string(res));
    }
}

Buffer::~Buffer() {
    release();
}

Buffer::Buffer(Buffer&& other) noexcept {
    m_allocator = other.m_allocator;
    m_buffer = other.m_buffer;
    m_allocation = other.m_allocation;
    m_size = other.m_size;
    m_mappedData = other.m_mappedData;

    other.m_allocator = VK_NULL_HANDLE;
    other.m_buffer = VK_NULL_HANDLE;
    other.m_allocation = VK_NULL_HANDLE;
    other.m_size = 0;
    other.m_mappedData = nullptr;
}

Buffer& Buffer::operator=(Buffer&& other) noexcept {
    if (this != &other) {
        release();
        m_allocator = other.m_allocator;
        m_buffer = other.m_buffer;
        m_allocation = other.m_allocation;
        m_size = other.m_size;
        m_mappedData = other.m_mappedData;

        other.m_allocator = VK_NULL_HANDLE;
        other.m_buffer = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
        other.m_size = 0;
        other.m_mappedData = nullptr;
    }
    return *this;
}

void Buffer::release() {
    if (m_mappedData) {
        unmap();
    }
    if (m_buffer && m_allocator) {
        vmaDestroyBuffer(m_allocator, m_buffer, m_allocation);
        m_buffer = VK_NULL_HANDLE;
        m_allocation = VK_NULL_HANDLE;
    }
}

VkDeviceAddress Buffer::getDeviceAddress(VkDevice device) const {
    VkBufferDeviceAddressInfo addressInfo{};
    addressInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
    addressInfo.buffer = m_buffer;
    return vkGetBufferDeviceAddress(device, &addressInfo);
}

void* Buffer::map() {
    if (!m_mappedData && m_allocator && m_allocation) {
        VkResult res = vmaMapMemory(m_allocator, m_allocation, &m_mappedData);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("Failed to map buffer memory! Error: " + std::to_string(res));
        }
    }
    return m_mappedData;
}

void Buffer::unmap() {
    if (m_mappedData && m_allocator && m_allocation) {
        vmaUnmapMemory(m_allocator, m_allocation);
        m_mappedData = nullptr;
    }
}

void Buffer::copyFrom(const void* data, VkDeviceSize size) {
    if (size > m_size) {
        throw std::runtime_error("Buffer::copyFrom data size exceeds buffer size!");
    }
    void* mapped = map();
    std::memcpy(mapped, data, size);
    vmaFlushAllocation(m_allocator, m_allocation, 0, size);
}

void Buffer::invalidate(VkDeviceSize offset, VkDeviceSize size) {
    if (m_allocator && m_allocation) {
        vmaInvalidateAllocation(m_allocator, m_allocation, offset, size);
    }
}

void Buffer::flush(VkDeviceSize offset, VkDeviceSize size) {
    if (m_allocator && m_allocation) {
        vmaFlushAllocation(m_allocator, m_allocation, offset, size);
    }
}

} // namespace pathways
