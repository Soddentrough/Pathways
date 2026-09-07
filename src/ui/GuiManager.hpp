#pragma once

#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>
#include "core/Config.hpp"
#include "utils/ImageDumper.hpp"

namespace pathways {

class Camera;
struct DisplayInfo;

struct GuiActions {
    bool resetAccumulation = false;
    bool toggleFullscreen = false;
    uint32_t requestedWidth = 0;
    uint32_t requestedHeight = 0;
};

class GuiManager {
public:
    GuiManager(SDL_Window* window, VkInstance instance, VkPhysicalDevice physicalDevice,
               VkDevice device, uint32_t queueFamily, VkQueue queue,
               VkFormat colorFormat, uint32_t minImageCount, uint32_t imageCount);
    ~GuiManager();

    bool processEvent(const SDL_Event& event);
    void newFrame();
    bool render(VkCommandBuffer cmd, VkImageView targetView, uint32_t width, uint32_t height,
                Config& config, const FrameStats& stats, bool& cameraMode, Camera* camera = nullptr,
                const DisplayInfo* displayInfo = nullptr, bool isFullscreen = false,
                GuiActions* actions = nullptr);

    bool wantCaptureMouse() const;
    bool wantCaptureKeyboard() const;

private:
    VkDevice m_device = VK_NULL_HANDLE;
    bool m_initialized = false;
    bool m_layoutInitialized = false;
    bool m_lastWasPortrait = false;

    // Rolling latency history for live profiler HUD
    static constexpr size_t HISTORY_SIZE = 120;
    float m_frameTimeHistory[HISTORY_SIZE] = {0};
    int m_historyOffset = 0;
};

} // namespace pathways
