#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <cstdint>

namespace pathways {

enum class HdrDisplayMode : uint32_t {
    SDR = 0,   // Standard Dynamic Range (sRGB nonlinear ~2.2)
    scRGB = 1, // Extended Dynamic Range scRGB Linear (16-bit Float, 1.0 = 80 nits)
    HDR10 = 2  // High Dynamic Range HDR10 (10-bit Rec.2020 SMPTE ST 2084 PQ)
};

class Swapchain {
public:
    Swapchain(VkDevice device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
              uint32_t width, uint32_t height, uint32_t graphicsQueueFamily,
              bool enableHdr = true);
    ~Swapchain();

    VkSwapchainKHR getSwapchain() const { return m_swapchain; }
    VkFormat getFormat() const { return m_imageFormat; }
    VkExtent2D getExtent() const { return m_extent; }
    uint32_t getImageCount() const { return static_cast<uint32_t>(m_images.size()); }
    VkImage getImage(uint32_t index) const { return m_images[index]; }
    VkImageView getImageView(uint32_t index) const { return m_imageViews[index]; }

    VkResult acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t* imageIndex);
    VkResult queuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore);

    HdrDisplayMode getHdrMode() const { return m_hdrMode; }
    bool isHdr() const { return m_hdrMode != HdrDisplayMode::SDR; }
    VkColorSpaceKHR getColorSpace() const { return m_colorSpace; }
    const char* getColorSpaceName() const;
    const char* getFormatName() const;

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkSwapchainKHR m_swapchain = VK_NULL_HANDLE;
    std::vector<VkImage> m_images;
    std::vector<VkImageView> m_imageViews;
    VkFormat m_imageFormat = VK_FORMAT_B8G8R8A8_UNORM;
    VkColorSpaceKHR m_colorSpace = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    HdrDisplayMode m_hdrMode = HdrDisplayMode::SDR;
    VkExtent2D m_extent{};
};

} // namespace pathways
