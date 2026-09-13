#include "vulkan/Swapchain.hpp"
#include "vulkan/VulkanContext.hpp"
#include "core/Window.hpp"
#include "core/Logger.hpp"
#include <algorithm>
#include <stdexcept>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <vulkan/vulkan_win32.h>
#endif

namespace pathways {

Swapchain::Swapchain(VkDevice device, VkPhysicalDevice physicalDevice, VkSurfaceKHR surface,
                     uint32_t width, uint32_t height, uint32_t graphicsQueueFamily,
                     bool enableHdr, bool isFullscreen,
                     const VulkanContext* context,
                     const DisplayInfo* displayInfo,
                     float peakNits, float paperWhiteNits)
    : m_device(device), m_context(context) {

    VkSurfaceCapabilitiesKHR capabilities;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physicalDevice, surface, &capabilities);

    uint32_t formatCount = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(formatCount);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physicalDevice, surface, &formatCount, formats.data());

    Logger::debug("Surface formats supported by display ({} available):", formatCount);
    for (const auto& f : formats) {
        Logger::debug("  Format: {}, ColorSpace: {}", static_cast<int>(f.format), static_cast<int>(f.colorSpace));
    }

    bool chosen = false;
    m_hdrMode = HdrDisplayMode::SDR;

    bool allowHdr = enableHdr;
    if (enableHdr) {
        // Evaluate whether HDR output can be engaged:
        // 1. If in windowed mode while desktop is in SDR, DWM cannot output HDR -> fallback to SDR to prevent washed-out clipping.
        // 2. If in fullscreen mode, application can switch the display output color space to HDR10 via direct scanout.
        // 3. If desktop is already in HDR mode, both windowed and fullscreen can use HDR.
        bool isDesktopHdr = displayInfo ? displayInfo->isDesktopHdr : false;
        bool isDisplayCapable = displayInfo ? displayInfo->isDisplayHdrCapable : true;

        if (!isFullscreen && !isDesktopHdr && displayInfo) {
            Logger::info("Windowed mode on SDR desktop detected: Compositor / Desktop HDR unavailable. Using SDR sRGB. (Run in Fullscreen to engage display HDR10 mode).");
            allowHdr = false;
        } else if (!isDisplayCapable && displayInfo) {
            Logger::info("Connected display does not report HDR capabilities. Using SDR sRGB.");
            allowHdr = false;
        }
    }

    if (allowHdr) {
        // Priority 1: True HDR10 with native 10-bit A2B10G10R10_UNORM_PACK32 (Rec.2020 SMPTE ST 2084 PQ)
        // Explicitly prioritize A2B10G10R10 over A2R10G10B10 to match engine backbuffer layout
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
                f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
                m_imageFormat = f.format;
                m_colorSpace = f.colorSpace;
                m_hdrMode = HdrDisplayMode::HDR10;
                chosen = true;
                Logger::info("Selected HDR Display: A2B10G10R10_UNORM with HDR10_ST2084 (HDR10 PQ Rec.2020)");
                break;
            }
        }

        // Fallback for drivers/compositors exposing A2R10G10B10 instead of A2B10G10R10
        if (!chosen) {
            for (const auto& f : formats) {
                if (f.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32 &&
                    f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT) {
                    m_imageFormat = f.format;
                    m_colorSpace = f.colorSpace;
                    m_hdrMode = HdrDisplayMode::HDR10;
                    chosen = true;
                    Logger::info("Selected HDR Display: A2R10G10B10_UNORM with HDR10_ST2084 (HDR10 PQ Rec.2020)");
                    break;
                }
            }
        }

        // Priority 2: scRGB Linear (16-bit Float, 1.0 = 80 nits reference paper white)
        // Fallback for compositors that expose scRGB but not HDR10
        if (!chosen) {
            for (const auto& f : formats) {
                if (f.format == VK_FORMAT_R16G16B16A16_SFLOAT &&
                    f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT) {
                    m_imageFormat = f.format;
                    m_colorSpace = f.colorSpace;
                    m_hdrMode = HdrDisplayMode::scRGB;
                    chosen = true;
                    Logger::info("Selected HDR Display: R16G16B16A16_SFLOAT with EXTENDED_SRGB_LINEAR (scRGB Linear)");
                    break;
                }
            }
        }

        if (!chosen) {
            Logger::info("HDR display mode requested, but display compositor does not expose HDR10 or scRGB surface formats. Falling back to SDR.");
        }
    }

    // SDR Fallback: Prefer R8G8B8A8_UNORM, then B8G8R8A8_UNORM
    if (!chosen) {
        m_hdrMode = HdrDisplayMode::SDR;
        for (const auto& f : formats) {
            if (f.format == VK_FORMAT_R8G8B8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                m_imageFormat = f.format;
                m_colorSpace = f.colorSpace;
                chosen = true;
                break;
            }
        }
        if (!chosen) {
            for (const auto& f : formats) {
                if (f.format == VK_FORMAT_B8G8R8A8_UNORM && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                    m_imageFormat = f.format;
                    m_colorSpace = f.colorSpace;
                    chosen = true;
                    break;
                }
            }
        }
        if (!chosen && !formats.empty()) {
            m_imageFormat = formats[0].format;
            m_colorSpace = formats[0].colorSpace;
        }
        Logger::info("Selected SDR Display: format {} with color space {}", static_cast<int>(m_imageFormat), static_cast<int>(m_colorSpace));
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
    VkCompositeAlphaFlagBitsKHR compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) {
        if (capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR) {
            compositeAlpha = VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR;
        } else if (capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR) {
            compositeAlpha = VK_COMPOSITE_ALPHA_PRE_MULTIPLIED_BIT_KHR;
        } else if (capabilities.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR) {
            compositeAlpha = VK_COMPOSITE_ALPHA_POST_MULTIPLIED_BIT_KHR;
        }
    }
    createInfo.compositeAlpha = compositeAlpha;
    createInfo.presentMode = chosenPresentMode;
    createInfo.clipped = VK_TRUE;
    createInfo.oldSwapchain = VK_NULL_HANDLE;

#ifdef _WIN32
    VkSurfaceFullScreenExclusiveInfoEXT exclusiveInfo{};
    exclusiveInfo.sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_INFO_EXT;

    VkSurfaceFullScreenExclusiveWin32InfoEXT win32ExclusiveInfo{};
    win32ExclusiveInfo.sType = VK_STRUCTURE_TYPE_SURFACE_FULL_SCREEN_EXCLUSIVE_WIN32_INFO_EXT;

    if (m_context && m_context->hasFullScreenExclusive() && isFullscreen) {
        exclusiveInfo.fullScreenExclusive = VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT;

        HMONITOR targetHmon = nullptr;
        if (displayInfo && displayInfo->hmonitor) {
            targetHmon = static_cast<HMONITOR>(displayInfo->hmonitor);
        }
        if (!targetHmon) {
            targetHmon = MonitorFromWindow(GetActiveWindow(), MONITOR_DEFAULTTONEAREST);
        }
        win32ExclusiveInfo.hmonitor = targetHmon;

        exclusiveInfo.pNext = &win32ExclusiveInfo;
        win32ExclusiveInfo.pNext = const_cast<void*>(createInfo.pNext);
        createInfo.pNext = &exclusiveInfo;
    }
#endif

    Logger::info("Swapchain Present Mode: {}", chosenPresentMode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX (Low-Latency Triple Buffering)" : "FIFO (VSync)");

    VkResult res = vkCreateSwapchainKHR(m_device, &createInfo, nullptr, &m_swapchain);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan swapchain! Error: " + std::to_string(res));
    }

#ifdef _WIN32
    if (m_context && m_context->hasFullScreenExclusive() && isFullscreen &&
        exclusiveInfo.fullScreenExclusive == VK_FULL_SCREEN_EXCLUSIVE_APPLICATION_CONTROLLED_EXT) {
        auto pfnAcquire = m_context->getAcquireFullScreenExclusiveModeEXT();
        if (pfnAcquire) {
            VkResult acqRes = pfnAcquire(m_device, m_swapchain);
            if (acqRes == VK_SUCCESS) {
                m_exclusiveModeAcquired = true;
                Logger::info("Acquired FullScreen Exclusive Mode (VK_EXT_full_screen_exclusive). Direct display scanout active.");
            } else {
                Logger::warn("vkAcquireFullScreenExclusiveModeEXT returned: {}", static_cast<int>(acqRes));
            }
        }
    }
#endif

    if (m_hdrMode != HdrDisplayMode::SDR) {
        float minNits = displayInfo ? displayInfo->minLuminanceNits : 0.001f;
        setHdrMetadata(peakNits, paperWhiteNits, minNits, displayInfo);
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

    Logger::info("Created swapchain: {}x{}, {} images, format: {}, colorSpace: {}",
                 m_extent.width, m_extent.height, imageCount, getFormatName(), getColorSpaceName());
}

Swapchain::~Swapchain() {
#ifdef _WIN32
    if (m_exclusiveModeAcquired && m_context) {
        auto pfnRelease = m_context->getReleaseFullScreenExclusiveModeEXT();
        if (pfnRelease && m_device && m_swapchain) {
            pfnRelease(m_device, m_swapchain);
            m_exclusiveModeAcquired = false;
        }
    }
#endif
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

const char* Swapchain::getColorSpaceName() const {
    switch (m_colorSpace) {
        case VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT: return "EXTENDED_SRGB_LINEAR (scRGB Linear)";
        case VK_COLOR_SPACE_HDR10_ST2084_EXT:         return "HDR10_ST2084 (BT.2020 PQ)";
        case VK_COLOR_SPACE_SRGB_NONLINEAR_KHR:       return "SRGB_NONLINEAR (SDR standard)";
        case VK_COLOR_SPACE_DISPLAY_P3_NONLINEAR_EXT: return "DISPLAY_P3_NONLINEAR";
        case VK_COLOR_SPACE_BT2020_LINEAR_EXT:        return "BT2020_LINEAR";
        default:                                      return "Custom / Unspecified";
    }
}

const char* Swapchain::getFormatName() const {
    switch (m_imageFormat) {
        case VK_FORMAT_R16G16B16A16_SFLOAT:       return "R16G16B16A16_SFLOAT (64-bit Half)";
        case VK_FORMAT_A2B10G10R10_UNORM_PACK32:  return "A2B10G10R10_UNORM (10-bit)";
        case VK_FORMAT_A2R10G10B10_UNORM_PACK32:  return "A2R10G10B10_UNORM (10-bit)";
        case VK_FORMAT_R8G8B8A8_UNORM:            return "R8G8B8A8_UNORM (8-bit)";
        case VK_FORMAT_B8G8R8A8_UNORM:            return "B8G8R8A8_UNORM (8-bit)";
        default:                                  return "Unknown Format";
    }
}

void Swapchain::setHdrMetadata(float peakNits, float paperWhiteNits, float minNits, const DisplayInfo* displayInfo) {
    if (!m_context || !m_context->hasHdrMetadata()) return;
    auto pfnSetMeta = m_context->getSetHdrMetadataEXT();
    if (!pfnSetMeta || !m_swapchain) return;

    VkHdrMetadataEXT meta{};
    meta.sType = VK_STRUCTURE_TYPE_HDR_METADATA_EXT;
    if (displayInfo && displayInfo->isDisplayHdrCapable && displayInfo->maxLuminanceNits > 0.0f) {
        meta.displayPrimaryRed = { displayInfo->redPrimary[0], displayInfo->redPrimary[1] };
        meta.displayPrimaryGreen = { displayInfo->greenPrimary[0], displayInfo->greenPrimary[1] };
        meta.displayPrimaryBlue = { displayInfo->bluePrimary[0], displayInfo->bluePrimary[1] };
        meta.whitePoint = { displayInfo->whitePoint[0], displayInfo->whitePoint[1] };
    } else {
        // ITU-R BT.2020 reference primaries & D65 white point
        meta.displayPrimaryRed = { 0.708f, 0.292f };
        meta.displayPrimaryGreen = { 0.170f, 0.797f };
        meta.displayPrimaryBlue = { 0.131f, 0.046f };
        meta.whitePoint = { 0.3127f, 0.3290f };
    }
    meta.maxLuminance = peakNits;
    meta.minLuminance = minNits > 0.0f ? minNits : 0.001f;
    meta.maxContentLightLevel = peakNits;
    meta.maxFrameAverageLightLevel = paperWhiteNits;

    pfnSetMeta(m_device, 1, &m_swapchain, &meta);
    Logger::info("Applied HDR Metadata (CTA-861): Peak={:.1f} nits, PaperWhite={:.1f} nits, Min={:.4f} nits",
                 meta.maxLuminance, meta.maxFrameAverageLightLevel, meta.minLuminance);
}

} // namespace pathways
