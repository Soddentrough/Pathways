#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

namespace pathways {

class Swapchain {
public:
    Swapchain(VkDevice device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
              uint32_t width, uint32_t height, uint32_t graphicsQueueFamily,
              VkFormat preferredFormat = VK_FORMAT_A2B10G10R10_UNORM_PACK32);
    ~Swapchain();

    VkSwapchainKHR getSwapchain() const { return m_swapchain; }
    VkFormat getFormat() const { return m_imageFormat; }
    VkExtent2D getExtent() const { return m_extent; }
    uint32_t getImageCount() const { return static_cast<uint32_t>(m_images.size()); }
    VkImage getImage(uint32_t index) const { return m_images[index]; }
    VkImageView getImageView(uint32_t index) const { return m_imageViews[index]; }

    VkResult acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t* imageIndex);
    VkResult queuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore);

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_imageViews;
    VkFormat m_preferredFormat = VK_FORMAT_A2B10G10R10_UNORM_PACK32;
    VkFormat m_imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR m_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    VkExtent2D m_extent{};
};

} // namespace pathways
