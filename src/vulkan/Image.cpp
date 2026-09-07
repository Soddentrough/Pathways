#include "vulkan/Image.hpp"
#include "core/Logger.hpp"
#include <stdexcept>

namespace pathways {

Image::Image(VkDevice device, VmaAllocator allocator, uint32_t width, uint32_t height,
             VkFormat format, VkImageUsageFlags usage, VkImageAspectFlags aspectFlags)
    : m_device(device), m_allocator(allocator), m_width(width), m_height(height),
      m_format(format), m_aspectFlags(aspectFlags) {

    VkImageCreateInfo imageInfo{};
    imageInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.format = format;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = usage;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkResult res = vmaCreateImage(m_allocator, &imageInfo, &allocInfo, &m_image, &m_allocation, nullptr);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate image via VMA! Error: " + std::to_string(res));
    }

    VkImageViewCreateInfo viewInfo{};
    viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
    viewInfo.image = m_image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = format;
    viewInfo.subresourceRange.aspectMask = aspectFlags;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = 1;

    res = vkCreateImageView(m_device, &viewInfo, nullptr, &m_imageView);
    if (res != VK_SUCCESS) {
        vmaDestroyImage(m_allocator, m_image, m_allocation);
        throw std::runtime_error("Failed to create image view! Error: " + std::to_string(res));
    }
}

Image::~Image() {
    release();
}

Image::Image(Image&& other) noexcept {
    m_device = other.m_device;
    m_allocator = other.m_allocator;
    m_image = other.m_image;
    m_allocation = other.m_allocation;
    m_imageView = other.m_imageView;
    m_width = other.m_width;
    m_height = other.m_height;
    m_format = other.m_format;
    m_layout = other.m_layout;
    m_aspectFlags = other.m_aspectFlags;

    other.m_device = VK_NULL_HANDLE;
    other.m_allocator = VK_NULL_HANDLE;
    other.m_image = VK_NULL_HANDLE;
    other.m_allocation = VK_NULL_HANDLE;
    other.m_imageView = VK_NULL_HANDLE;
}

Image& Image::operator=(Image&& other) noexcept {
    if (this != &other) {
        release();
        m_device = other.m_device;
        m_allocator = other.m_allocator;
        m_image = other.m_image;
        m_allocation = other.m_allocation;
        m_imageView = other.m_imageView;
        m_width = other.m_width;
        m_height = other.m_height;
        m_format = other.m_format;
        m_layout = other.m_layout;
        m_aspectFlags = other.m_aspectFlags;

        other.m_device = VK_NULL_HANDLE;
        other.m_allocator = VK_NULL_HANDLE;
        other.m_image = VK_NULL_HANDLE;
        other.m_allocation = VK_NULL_HANDLE;
        other.m_imageView = VK_NULL_HANDLE;
    }
    return *this;
}

void Image::release() {
    if (m_imageView && m_device) {
        vkDestroyImageView(m_device, m_imageView, nullptr);
        m_imageView = VK_NULL_HANDLE;
    }
    if (m_image && m_allocator) {
        vmaDestroyImage(m_allocator, m_image, m_allocation);
        m_image = VK_NULL_HANDLE;
        m_allocation = VK_NULL_HANDLE;
    }
}

void Image::transitionLayout(VkCommandBuffer cmd, VkImageLayout newLayout,
                            VkPipelineStageFlags2 srcStage, VkAccessFlags2 srcAccess,
                            VkPipelineStageFlags2 dstStage, VkAccessFlags2 dstAccess) {

    VkImageMemoryBarrier2 barrier{};
    barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barrier.srcStageMask = srcStage;
    barrier.srcAccessMask = srcAccess;
    barrier.dstStageMask = dstStage;
    barrier.dstAccessMask = dstAccess;
    barrier.oldLayout = m_layout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = m_image;
    barrier.subresourceRange.aspectMask = m_aspectFlags;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    VkDependencyInfo dependencyInfo{};
    dependencyInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    vkCmdPipelineBarrier2(cmd, &dependencyInfo);
    m_layout = newLayout;
}

} // namespace pathways
