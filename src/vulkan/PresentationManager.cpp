#include "vulkan/PresentationManager.hpp"
#include "vulkan/VulkanContext.hpp"
#include "core/Logger.hpp"
#include "core/QualityGovernor.hpp"

#include <algorithm>
#include <format>
#include <stdexcept>

namespace pathways {

PresentationManager::PresentationManager(
    Window* window,
    VulkanContext* context,
    Config& config,
    uint32_t maxFramesInFlight
) : m_window(window), m_context(context), m_maxFramesInFlight(maxFramesInFlight) {
    m_lastWallFrameStartTime = std::chrono::high_resolution_clock::now();
    m_currentFrameStartTime = m_lastWallFrameStartTime;

    if (!config.headless && m_window && m_context) {
        m_window->setTitle(std::format("Pathways - Vulkan 1.4 Path Tracer ({})", m_context->getShortArchName()));
        if (!config.custom_hdr_peak && m_window->getDisplayInfo().isDisplayHdrCapable &&
            m_window->getDisplayInfo().maxLuminanceNits > 0.0f) {
            config.hdr_peak_nits = m_window->getDisplayInfo().maxLuminanceNits;
        }

        m_surface = m_window->createSurface(m_context->getInstance());

        m_swapchain = std::make_unique<Swapchain>(
            m_context->getDevice(),
            m_context->getPhysicalDevice(),
            m_surface,
            m_window->getSurfaceWidth(),
            m_window->getSurfaceHeight(),
            m_context->getGraphicsQueueFamily(),
            config.enable_hdr,
            config.exclusive_fullscreen && m_window->isFullscreen(),
            m_context,
            &m_window->getDisplayInfo(),
            config.hdr_peak_nits,
            config.hdr_paper_white_nits
        );
        if (m_window->isComposedBorderless()) {
            config.width = std::min(m_window->getWidth(), m_swapchain->getExtent().width);
            config.height = std::min(m_window->getHeight(), m_swapchain->getExtent().height);
        } else {
            config.width = m_swapchain->getExtent().width;
            config.height = m_swapchain->getExtent().height;
        }
    }

    initSyncObjects();
}

PresentationManager::~PresentationManager() {
    if (!m_context) return;
    VkDevice device = m_context->getDevice();
    if (device != VK_NULL_HANDLE) {
        for (auto fence : m_inFlightFences) {
            if (fence) vkDestroyFence(device, fence, nullptr);
        }
        for (auto sem : m_imageAvailableSemaphores) {
            if (sem) vkDestroySemaphore(device, sem, nullptr);
        }
        for (auto sem : m_renderFinishedSemaphores) {
            if (sem) vkDestroySemaphore(device, sem, nullptr);
        }
    }
    m_inFlightFences.clear();
    m_imageAvailableSemaphores.clear();
    m_renderFinishedSemaphores.clear();

    m_uiDumpBuffer.reset();
    m_swapchain.reset();

    if (m_surface != VK_NULL_HANDLE && m_context->getInstance() != VK_NULL_HANDLE) {
        vkDestroySurfaceKHR(m_context->getInstance(), m_surface, nullptr);
        m_surface = VK_NULL_HANDLE;
    }
}

void PresentationManager::initSyncObjects() {
    if (!m_context) return;
    VkDevice device = m_context->getDevice();

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = VK_FENCE_CREATE_SIGNALED_BIT;

    m_inFlightFences.resize(m_maxFramesInFlight, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < m_maxFramesInFlight; ++i) {
        vkCreateFence(device, &fenceInfo, nullptr, &m_inFlightFences[i]);
    }

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;

    m_imageAvailableSemaphores.resize(m_maxFramesInFlight, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < m_maxFramesInFlight; ++i) {
        vkCreateSemaphore(device, &semInfo, nullptr, &m_imageAvailableSemaphores[i]);
    }

    uint32_t numSwapImages = m_swapchain ? m_swapchain->getImageCount() : m_maxFramesInFlight;
    m_renderFinishedSemaphores.resize(numSwapImages, VK_NULL_HANDLE);
    for (uint32_t i = 0; i < numSwapImages; ++i) {
        vkCreateSemaphore(device, &semInfo, nullptr, &m_renderFinishedSemaphores[i]);
    }
}

void PresentationManager::beginFrame(uint32_t frameSlot, uint64_t totalFramesRendered, const Config& config) {
    auto frameNow = std::chrono::high_resolution_clock::now();
    if (totalFramesRendered > 0) {
        double wallIntervalMs = std::chrono::duration<double, std::milli>(frameNow - m_lastWallFrameStartTime).count();
        if (wallIntervalMs > 0.01 && wallIntervalMs < 1000.0) {
            m_lastPresentationTimeMs = wallIntervalMs;
            m_presentationTimesMs.push_back(wallIntervalMs);
            if (!config.headless && m_presentationTimesMs.size() > 60) {
                m_presentationTimesMs.erase(m_presentationTimesMs.begin());
            }
        }
    }
    m_lastWallFrameStartTime = frameNow;
    m_currentFrameStartTime = frameNow;

    if (m_context && frameSlot < m_inFlightFences.size() && m_inFlightFences[frameSlot]) {
        // Bounded wait: a wedged GPU must not deadlock the render loop forever.
        VkResult fenceRes = vkWaitForFences(m_context->getDevice(), 1, &m_inFlightFences[frameSlot], VK_TRUE, 10'000'000'000ull /* 10 s */);
        if (fenceRes == VK_TIMEOUT) {
            Logger::error("In-flight fence timeout ({}s) for frame slot {} — GPU may be hung or validation-stalled; waiting for device idle before recycling command buffers.", 10, frameSlot);
            vkDeviceWaitIdle(m_context->getDevice());
        } else if (fenceRes != VK_SUCCESS) {
            Logger::error("vkWaitForFences failed for frame slot {} (VkResult: {})", frameSlot, static_cast<int>(fenceRes));
        }
    }
}

bool PresentationManager::acquireNextImage(
    uint32_t frameSlot,
    uint32_t& outImageIndex,
    Config& config,
    const std::function<void(uint32_t, uint32_t, bool)>& onResizeFn
) {
    outImageIndex = 0;
    if (config.headless || !m_swapchain || !m_window) {
        return true;
    }

    uint32_t targetViewW = 0, targetViewH = 0, targetSurfW = 0, targetSurfH = 0;
    m_window->queryViewportAndSurfacePixels(targetViewW, targetViewH, targetSurfW, targetSurfH);

    if (targetSurfW != m_swapchain->getExtent().width ||
        targetSurfH != m_swapchain->getExtent().height ||
        config.width != targetViewW ||
        config.height != targetViewH) {
        if (onResizeFn) {
            onResizeFn(targetViewW, targetViewH, true);
        }
    }

    VkResult res = m_swapchain->acquireNextImage(m_imageAvailableSemaphores[frameSlot], &outImageIndex);
    if (res == VK_ERROR_OUT_OF_DATE_KHR) {
        m_window->queryViewportAndSurfacePixels(targetViewW, targetViewH, targetSurfW, targetSurfH);
        if (onResizeFn) {
            onResizeFn(targetViewW, targetViewH, true);
        }
        return false;
    }
    if (res != VK_SUCCESS && res != VK_SUBOPTIMAL_KHR) {
        Logger::error("vkAcquireNextImageKHR failed with error: {}", static_cast<int>(res));
        return false;
    }
    return true;
}

bool PresentationManager::recordPresentation(
    VkCommandBuffer cmd,
    uint32_t imageIndex,
    Image* outputImage,
    GuiManager* gui,
    Config& config,
    const FrameStats& stats,
    bool isCameraActive,
    Camera* camera,
    GuiActions* outGuiActions,
    const std::vector<SceneEntry>& availableScenes,
    int currentSceneIndex
) {
    if (config.headless || !m_swapchain || !outputImage) {
        return false;
    }

    VkImage swapImage = m_swapchain->getImage(imageIndex);
    VkImageView swapView = m_swapchain->getImageView(imageIndex);

    // Transition outputImage to TRANSFER_SRC_OPTIMAL
    outputImage->transitionLayout(
        cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
    );

    // Transition swapImage from UNDEFINED to TRANSFER_DST_OPTIMAL
    VkImageMemoryBarrier2 toDst{};
    toDst.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    toDst.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
    toDst.srcAccessMask = VK_ACCESS_2_NONE;
    toDst.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
    toDst.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    toDst.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    toDst.newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
    toDst.image = swapImage;
    toDst.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo depToDst{};
    depToDst.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    depToDst.imageMemoryBarrierCount = 1;
    depToDst.pImageMemoryBarriers = &toDst;
    vkCmdPipelineBarrier2(cmd, &depToDst);

    // Copy or blit output image to swapchain image
    const uint32_t swapW = m_swapchain->getExtent().width;
    const uint32_t swapH = m_swapchain->getExtent().height;
    const bool extentsMatch = (config.width == swapW && config.height == swapH);
    const bool isPaddedBorderless = (!extentsMatch &&
                                     swapW >= config.width && swapH >= config.height &&
                                     (swapW - config.width) <= 2u && (swapH - config.height) <= 2u);
    const bool formatsMatch = (outputImage->getFormat() == m_swapchain->getFormat());

    if ((extentsMatch || isPaddedBorderless) && formatsMatch) {
        VkImageCopy copyRegion{};
        copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copyRegion.extent = { config.width, config.height, 1 };
        vkCmdCopyImage(cmd, outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        // Initialize the +1px off-screen composed-flip padding strip cleanly using separate
        // single-region copies so source regions never overlap within one vkCmdCopyImage call.
        if (swapH > config.height && config.height > 0) {
            for (uint32_t padY = config.height; padY < swapH; ++padY) {
                VkImageCopy padRow{};
                padRow.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                padRow.srcOffset = { 0, static_cast<int32_t>(config.height - 1), 0 };
                padRow.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                padRow.dstOffset = { 0, static_cast<int32_t>(padY), 0 };
                padRow.extent = { config.width, 1, 1 };
                vkCmdCopyImage(cmd, outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &padRow);
            }
        }
        if (swapW > config.width && config.width > 0) {
            for (uint32_t padX = config.width; padX < swapW; ++padX) {
                VkImageCopy padCol{};
                padCol.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                padCol.srcOffset = { static_cast<int32_t>(config.width - 1), 0, 0 };
                padCol.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                padCol.dstOffset = { static_cast<int32_t>(padX), 0, 0 };
                padCol.extent = { 1, config.height, 1 };
                vkCmdCopyImage(cmd, outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                               swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &padCol);
            }
        }
    } else {
        VkImageBlit blitRegion{};
        blitRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blitRegion.srcOffsets[0] = { 0, 0, 0 };
        blitRegion.srcOffsets[1] = { static_cast<int32_t>(config.width), static_cast<int32_t>(config.height), 1 };
        blitRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        blitRegion.dstOffsets[0] = { 0, 0, 0 };
        const uint32_t dstBlitW = isPaddedBorderless ? config.width : swapW;
        const uint32_t dstBlitH = isPaddedBorderless ? config.height : swapH;
        blitRegion.dstOffsets[1] = { static_cast<int32_t>(dstBlitW), static_cast<int32_t>(dstBlitH), 1 };
        VkFilter filter = (extentsMatch || isPaddedBorderless) ? VK_FILTER_NEAREST : VK_FILTER_LINEAR;
        vkCmdBlitImage(cmd, outputImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       swapImage, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blitRegion, filter);
    }

    // Transition outputImage back to GENERAL
    outputImage->transitionLayout(
        cmd, VK_IMAGE_LAYOUT_GENERAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    );

    bool resetAccum = false;
    const bool hasGui = (gui != nullptr && config.enable_gui);

    if (hasGui) {
        // Transition swapImage to COLOR_ATTACHMENT_OPTIMAL for ImGui
        VkImageMemoryBarrier2 toColor{};
        toColor.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toColor.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        toColor.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toColor.dstStageMask = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
        toColor.dstAccessMask = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT;
        toColor.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toColor.newLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        toColor.image = swapImage;
        toColor.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo depToColor{};
        depToColor.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depToColor.imageMemoryBarrierCount = 1;
        depToColor.pImageMemoryBarriers = &toColor;
        vkCmdPipelineBarrier2(cmd, &depToColor);

        // Render ImGui overlay
        gui->newFrame();
        bool camMode = isCameraActive;
        if (gui->render(cmd, swapView, m_swapchain->getExtent().width, m_swapchain->getExtent().height,
                        config, stats, camMode, camera,
                        m_window ? &m_window->getDisplayInfo() : nullptr,
                        m_window ? m_window->isFullscreen() : false,
                        outGuiActions,
                        availableScenes, currentSceneIndex)) {
            resetAccum = true;
        }
        if (camMode != isCameraActive && outGuiActions) {
            outGuiActions->cameraModeChanged = true;
            outGuiActions->newCameraMode = camMode;
        }
    }

    // Transition swapImage to PRESENT_SRC_KHR (or copy to staging if dumping UI)
    if (!config.dump_ui_path.empty()) {
        VkImageMemoryBarrier2 toSrc{};
        toSrc.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toSrc.srcStageMask = hasGui ? VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT : VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        toSrc.srcAccessMask = hasGui ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toSrc.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        toSrc.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        toSrc.oldLayout = hasGui ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toSrc.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toSrc.image = swapImage;
        toSrc.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo depToSrc{};
        depToSrc.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depToSrc.imageMemoryBarrierCount = 1;
        depToSrc.pImageMemoryBarriers = &toSrc;
        vkCmdPipelineBarrier2(cmd, &depToSrc);

        VkFormat swapFmt = m_swapchain->getFormat();
        size_t bpp = (swapFmt == VK_FORMAT_R16G16B16A16_SFLOAT) ? 8 : 4;
        if (!m_uiDumpBuffer) {
            VkDeviceSize size = static_cast<VkDeviceSize>(m_swapchain->getExtent().width) * m_swapchain->getExtent().height * bpp;
            m_uiDumpBuffer = std::make_unique<Buffer>(m_context->getAllocator(), size, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                                     VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);
        }

        VkBufferImageCopy copyRegion{};
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = { m_swapchain->getExtent().width, m_swapchain->getExtent().height, 1 };
        vkCmdCopyImageToBuffer(cmd, swapImage, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, m_uiDumpBuffer->getBuffer(), 1, &copyRegion);

        VkImageMemoryBarrier2 toPresent{};
        toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toPresent.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        toPresent.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        toPresent.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        toPresent.dstAccessMask = VK_ACCESS_2_NONE;
        toPresent.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
        toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toPresent.image = swapImage;
        toPresent.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo depToPresent{};
        depToPresent.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depToPresent.imageMemoryBarrierCount = 1;
        depToPresent.pImageMemoryBarriers = &toPresent;
        vkCmdPipelineBarrier2(cmd, &depToPresent);
    } else {
        VkImageMemoryBarrier2 toPresent{};
        toPresent.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        toPresent.srcStageMask = hasGui ? VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT : VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        toPresent.srcAccessMask = hasGui ? VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT : VK_ACCESS_2_TRANSFER_WRITE_BIT;
        toPresent.dstStageMask = VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT;
        toPresent.dstAccessMask = VK_ACCESS_2_NONE;
        toPresent.oldLayout = hasGui ? VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL : VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        toPresent.newLayout = VK_IMAGE_LAYOUT_PRESENT_SRC_KHR;
        toPresent.image = swapImage;
        toPresent.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo depToPresent{};
        depToPresent.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depToPresent.imageMemoryBarrierCount = 1;
        depToPresent.pImageMemoryBarriers = &toPresent;
        vkCmdPipelineBarrier2(cmd, &depToPresent);
    }

    return resetAccum;
}

void PresentationManager::present(
    VkQueue queue,
    uint32_t imageIndex,
    const std::function<void(uint32_t, uint32_t, bool)>& onResizeFn
) {
    if (!m_swapchain || !m_window || m_window->isHeadless()) return;
    VkResult res = m_swapchain->queuePresent(queue, imageIndex, m_renderFinishedSemaphores[imageIndex]);
    if (res == VK_ERROR_OUT_OF_DATE_KHR || res == VK_SUBOPTIMAL_KHR) {
        uint32_t targetViewW = 0, targetViewH = 0, targetSurfW = 0, targetSurfH = 0;
        m_window->queryViewportAndSurfacePixels(targetViewW, targetViewH, targetSurfW, targetSurfH);
        if (onResizeFn) {
            onResizeFn(targetViewW, targetViewH, true);
        }
    }
}

void PresentationManager::paceFrame(
    const Config& config,
    QualityGovernor* governor,
    bool accumReachedCutoff
) {
    if (governor && config.target_fps > 0) {
        governor->paceFrame(m_currentFrameStartTime);
    } else if (accumReachedCutoff && !config.headless) {
        // When progressive accumulation cutoff is reached and scene is static, pace at 60 Hz
        // to prevent runaway CPU/GPU utilization presenting identical frames
        SDL_Delay(16);
    }
}

void PresentationManager::onResize(
    uint32_t newWidth,
    uint32_t newHeight,
    Config& config,
    bool forceRecreate
) {
    if (config.headless || !m_swapchain) return;
    if (newWidth == 0 || newHeight == 0) return;

    newWidth = std::max(64u, newWidth);
    newHeight = std::max(64u, newHeight);

    const bool useComposedSurface = (m_window && m_window->isComposedBorderless() &&
                                     newWidth == m_window->getWidth() &&
                                     newHeight == m_window->getHeight());
    const uint32_t targetSurfW = useComposedSurface ? m_window->getSurfaceWidth() : newWidth;
    const uint32_t targetSurfH = useComposedSurface ? m_window->getSurfaceHeight() : newHeight;

    if (!forceRecreate &&
        newWidth == config.width && newHeight == config.height &&
        m_swapchain->getExtent().width == targetSurfW && m_swapchain->getExtent().height == targetSurfH) {
        return;
    }

    Logger::info("Handling window resize: updating viewport from {}x{} to {}x{} (surface: {}x{})",
                 config.width, config.height, newWidth, newHeight, targetSurfW, targetSurfH);

    VkDevice device = m_context->getDevice();
    vkDeviceWaitIdle(device);

    config.width = newWidth;
    config.height = newHeight;

    if (m_window && !config.custom_hdr_peak && m_window->getDisplayInfo().isDisplayHdrCapable &&
        m_window->getDisplayInfo().maxLuminanceNits > 0.0f) {
        config.hdr_peak_nits = m_window->getDisplayInfo().maxLuminanceNits;
    }

    VkSwapchainKHR oldSwapchainHandle = m_swapchain ? m_swapchain->getSwapchain() : VK_NULL_HANDLE;
    auto newSwapchain = std::make_unique<Swapchain>(
        device,
        m_context->getPhysicalDevice(),
        m_surface,
        targetSurfW,
        targetSurfH,
        m_context->getGraphicsQueueFamily(),
        config.enable_hdr,
        config.exclusive_fullscreen && (m_window ? m_window->isFullscreen() : false),
        m_context,
        m_window ? &m_window->getDisplayInfo() : nullptr,
        config.hdr_peak_nits,
        config.hdr_paper_white_nits,
        oldSwapchainHandle
    );
    m_swapchain = std::move(newSwapchain);
    if (useComposedSurface) {
        config.width = std::min(newWidth, m_swapchain->getExtent().width);
        config.height = std::min(newHeight, m_swapchain->getExtent().height);
    } else {
        config.width = m_swapchain->getExtent().width;
        config.height = m_swapchain->getExtent().height;
    }

    VkSemaphoreCreateInfo semInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    uint32_t numSwapImages = m_swapchain->getImageCount();
    while (m_renderFinishedSemaphores.size() < numSwapImages) {
        VkSemaphore sem = VK_NULL_HANDLE;
        vkCreateSemaphore(device, &semInfo, nullptr, &sem);
        m_renderFinishedSemaphores.push_back(sem);
    }

    if (m_uiDumpBuffer) {
        m_uiDumpBuffer.reset();
    }
}

} // namespace pathways
