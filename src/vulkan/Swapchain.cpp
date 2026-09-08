#include "vulkan/Swapchain.hpp"
#include "core/Logger.hpp"
#include <algorithm>
#include <stdexcept>

namespace pathways {

Swapchain::Swapchain(VkDevice device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                     uint32_t width, uint32_t height, uint32_t graphicsQueueFamily)
    : m_device(device) {

    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);

    uint32_t formatCount;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());

    // Prefer R8G8B8A8_UNORM to match engine storage output image, fallback to B8G8R8A8_UNORM
    m_imageFormat = formats[0].format;
    m_colorSpace = formats[0].colorSpace;
    bool foundFormat = false;
    for (const auto& f : formats) {
        if (f.format == VK_FORMAT_R8G8B8A8_UNORM) {
            m_imageFormat = f.format;
            m_colorSpace = f.colorSpace;
            foundFormat = true;
            break;
        }
    }
    if (!foundFormat) {
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_B8G8R8A8_UNORM) {
                m_imageFormat = f.format;
                m_colorSpace = f.colorSpace;
                break;
            }
        }
    }

    if (capabilities.currentExtent.width != UINT32_MAX) {
        m_extent = capabilities.currentExtent;
    } else {
        m_extent.width = std::clamp(width, capabilities.minImageExtent.width, capabilities.maxImageExtent.width);
        m_extent.height = std::clamp(height, capabilities.minImageExtent.height, capabilities.maxImageExtent.height);
    }

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount) {
        imageCount = capabilities.maxImageCount;
    }

    uint32_t presentModeCount = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, nullptr);
    std::vector<VkPresentModeKHR> presentModes(presentModeCount);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physicalDevice, surface, &presentModeCount, presentModes.data());

    VkPresentModeKHR chosenPresentMode = VK_PRESENT_MODE_FIFO_KHR;
    for (const auto& mode : presentModes) {
        if (mode == VK_PRESENT_MODE_MAILBOX_KHR) {
            chosenPresentMode = VK_PRESENT_MODE_MAILBOX_KHR;
            break;
        }
    }

    VkSwapchainCreateInfoKHR createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR;
    createInfo.surface = surface;
    createInfo.minImageCount = imageCount;
    createInfo.imageFormat = m_imageFormat;
    createInfo.imageColorSpace = m_colorSpace;
    createInfo.imageExtent = m_extent;
    createInfo.imageArrayLayers = 1;
    createInfo.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if (capabilities.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) {
        createInfo.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }
    createInfo.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    createInfo.preTransform = capabilities.currentTransform;
    createInfo.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    createInfo.presentMode = chosenPresentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

    Logger::info("Swapchain Present Mode: {}", chosenPresentMode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX (Low-Latency Triple Buffering)" : "FIFO (VSync)");

    VkResult res = vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan swapchain! Error: " + std::to_string(res));
    }

    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, nullptr);
    m_images.resize(imageCount);
    vkGetSwapchainImagesKHR(m_device, m_swapchain, &imageCount, m_images.data());

    m_imageViews.resize(imageCount);
    for (size_t i = 0; i < imageCount; ++i) {
        VkImageViewCreateInfo viewInfo{};
        viewInfo.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO;
        viewInfo.image = m_images[i];
        viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
        viewInfo.format = m_imageFormat;
        viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        viewInfo.subresourceRange.baseMipLevel = 0;
        viewInfo.subresourceRange.levelCount = 1;
        viewInfo.subresourceRange.baseArrayLayer = 0;
        viewInfo.subresourceRange.layerCount = 1;

        res = vkCreateImageView(m_device, &viewInfo, nullptr, &m_imageViews[i]);
        if (res != VK_SUCCESS) {
            throw std::runtime_error("Failed to create swapchain image view!");
        }
    }

    Logger::info("Created swapchain: {}x{}, {} images, format: {}", m_extent.width, m_extent.height, imageCount, (int)m_imageFormat);
}

Swapchain::~Swapchain() {
    for (auto imageView : m_imageViews) {
        if (imageView && m_device) {
            vkDestroyImageView(m_device, imageView, nullptr);
        }
    }
    if (m_swapchain && m_device) {
        vkDestroySwapchainKHR(m_device, m_swapchain, nullptr);
    }
}

VkResult Swapchain::acquireNextImage(VkSemaphore presentCompleteSemaphore, uint32_t* imageIndex) {
    return vkAcquireNextImageKHR(m_device, m_swapchain, UINT64_MAX, presentCompleteSemaphore, VK_NULL_HANDLE, imageIndex);
}

VkResult Swapchain::queuePresent(VkQueue queue, uint32_t imageIndex, VkSemaphore waitSemaphore) {
    VkPresentInfoKHR presentInfo{};
    presentInfo.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    presentInfo.waitSemaphoreCount = waitSemaphore != VK_NULL_HANDLE ? 1 : 0;
    presentInfo.pWaitSemaphores = &waitSemaphore;
    presentInfo.swapchainCount = 1;
    presentInfo.pSwapchains = &m_swapchain;
    presentInfo.pImageIndices = &imageIndex;

    return vkQueuePresentKHR(queue, &presentInfo);
}

} // namespace pathways
