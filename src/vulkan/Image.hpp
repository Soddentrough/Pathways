#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include <cstdint>

namespace pathways {

class Image {
public:
    Image(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height,
          VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT);
    ~Image();

    Image(const Image&) = delete;
    Image& operator=(const Image&) = delete;

    Image(Image&& other) noexcept;
    Image& operator=(Image&& other) noexcept;

    VkImage getImage() const { return m_image; }
    VkImageView getImageView() const { return m_imageView; }
    VkFormat getFormat() const { return m_format; }
    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }
    VkImageLayout getLayout() const { return m_layout; }

    void transitionLayout(VkCommandBuffer cmd, VkImageLayout newLayout,
                          VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                          VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess);

private:
    void release();

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkImage m_image = VK_NULL_HANDLE;
    VmaAllocation m_allocation = VK_NULL_HANDLE;
    VkImageView m_imageView = VK_NULL_HANDLE;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    VkFormat m_format = VK_FORMAT_UNDEFINED;
    VkImageLayout m_layout = VK_IMAGE_LAYOUT_UNDEFINED;
    VkImageAspectFlags m_aspectFlags = VK_IMAGE_ASPECT_COLOR_BIT;
};

} // namespace pathways
