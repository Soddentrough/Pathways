#pragma once

#include "vulkan/Swapchain.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "core/Config.hpp"
#include "core/Window.hpp"
#include "ui/GuiManager.hpp"
#include "scene/SceneRegistry.hpp"

#include <vulkan/vulkan.h>
#include <chrono>
#include <functional>
#include <memory>
#include <vector>

namespace pathways {

class VulkanContext;
class Camera;
class QualityGovernor;

class PresentationManager {
public:
    PresentationManager(
        Window* window,
        VulkanContext* context,
        Config& config,
        uint32_t maxFramesInFlight
    );
    ~PresentationManager();

    PresentationManager(const PresentationManager&) = delete;
    PresentationManager& operator=(const PresentationManager&) = delete;

    /// Waits for in-flight fence and measures wall-clock frame interval.
    void beginFrame(uint32_t frameSlot, uint64_t totalFramesRendered, const Config& config);

    /// Checks window extent and acquires next swapchain image index.
    /// Returns true on success, false if window was resized or acquisition failed.
    bool acquireNextImage(
        uint32_t frameSlot,
        uint32_t& outImageIndex,
        Config& config,
        const std::function<void(uint32_t, uint32_t, bool)>& onResizeFn
    );

    /// Records swapchain image transition, copy/blit from outputImage, ImGui overlay rendering,
    /// UI staging dump if configured, and transition to PRESENT_SRC_KHR.
    /// Returns true if accumulation should be reset (e.g. from UI interaction).
    bool recordPresentation(
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
    );

    /// Presents swapchain image to presentation queue.
    void present(
        VkQueue queue,
        uint32_t imageIndex,
        const std::function<void(uint32_t, uint32_t, bool)>& onResizeFn
    );

    /// Paces frame according to target FPS (governor or 60 Hz delay for static accumulation).
    void paceFrame(
        const Config& config,
        QualityGovernor* governor,
        bool accumReachedCutoff
    );

    /// Recreates swapchain and synchronizes semaphores on window resize.
    void onResize(
        uint32_t newWidth,
        uint32_t newHeight,
        Config& config,
        bool forceRecreate = false
    );

    // Getters
    [[nodiscard]] Swapchain* getSwapchain() const noexcept { return m_swapchain.get(); }
    [[nodiscard]] VkSurfaceKHR getSurface() const noexcept { return m_surface; }
    [[nodiscard]] VkFence getInFlightFence(uint32_t slot) const noexcept {
        return (slot < m_inFlightFences.size()) ? m_inFlightFences[slot] : VK_NULL_HANDLE;
    }
    [[nodiscard]] VkSemaphore getImageAvailableSemaphore(uint32_t slot) const noexcept {
        return (slot < m_imageAvailableSemaphores.size()) ? m_imageAvailableSemaphores[slot] : VK_NULL_HANDLE;
    }
    [[nodiscard]] VkSemaphore getRenderFinishedSemaphore(uint32_t imageIndex) const noexcept {
        return (imageIndex < m_renderFinishedSemaphores.size()) ? m_renderFinishedSemaphores[imageIndex] : VK_NULL_HANDLE;
    }
    [[nodiscard]] Buffer* getUiDumpBuffer() const noexcept { return m_uiDumpBuffer.get(); }
    void resetUiDumpBuffer() noexcept { m_uiDumpBuffer.reset(); }

    [[nodiscard]] double getLastPresentationTimeMs() const noexcept { return m_lastPresentationTimeMs; }
    [[nodiscard]] const std::vector<double>& getPresentationTimesMs() const noexcept { return m_presentationTimesMs; }
    void clearPresentationTimes() noexcept { m_presentationTimesMs.clear(); }
    [[nodiscard]] const std::chrono::high_resolution_clock::time_point& getCurrentFrameStartTime() const noexcept {
        return m_currentFrameStartTime;
    }

private:
    void initSyncObjects();

    Window* m_window = nullptr;
    VulkanContext* m_context = nullptr;
    uint32_t m_maxFramesInFlight = 2;

    VkSurfaceKHR m_surface = VK_NULL_HANDLE;
    std::unique_ptr<Swapchain> m_swapchain;

    std::vector<VkFence> m_inFlightFences;
    std::vector<VkSemaphore> m_imageAvailableSemaphores;
    std::vector<VkSemaphore> m_renderFinishedSemaphores;

    std::unique_ptr<Buffer> m_uiDumpBuffer;

    std::chrono::high_resolution_clock::time_point m_lastWallFrameStartTime;
    std::chrono::high_resolution_clock::time_point m_currentFrameStartTime;
    double m_lastPresentationTimeMs = 0.0;
    std::vector<double> m_presentationTimesMs;
};

} // namespace pathways
