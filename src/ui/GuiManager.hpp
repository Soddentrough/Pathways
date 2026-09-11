#pragma once

#include <vulkan/vulkan.h>
#include <SDL3/SDL.h>
#include "core/Config.hpp"
#include "utils/ImageDumper.hpp"

#include "scene/SceneRegistry.hpp"

namespace pathways {

class Camera;
struct DisplayInfo;

struct GuiActions {
    bool resetAccumulation = false;
    bool toggleFullscreen = false;
    uint32_t requestedWidth = 0;
    uint32_t requestedHeight = 0;
    bool exportTelemetry = false;
    std::string exportTelemetryPath = "";
    bool refreshPciStatus = false;
    bool mgpuModeChanged = false;
    MultiGpuMode newMgpuMode = MultiGpuMode::Off;
    bool accumFormatChanged = false;
    AccumFormat newAccumFormat = AccumFormat::RGBA16_SFLOAT;
    bool doubleBufferChanged = false;
    bool newDoubleBuffer = true;
    bool tileSizeChanged = false;
    uint32_t newTileSize = 64;
    bool sceneChanged = false;
    std::string newScenePath = "";
    int newSceneIndex = -1;
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
                GuiActions* actions = nullptr,
                const std::vector<SceneEntry>& availableScenes = {}, int currentSceneIndex = -1);

    bool wantCaptureMouse() const;
    bool wantCaptureKeyboard() const;
    void resetHistory();

private:
    VkDevice m_device = VK_NULL_HANDLE;
    bool m_initialized = false;
    bool m_layoutInitialized = false;
    bool m_lastWasPortrait = false;
    uint32_t m_lastWidth = 0;
    uint32_t m_lastHeight = 0;
    float m_lastDispW = 0.0f;
    float m_lastDispH = 0.0f;

    // Rolling latency history for live profiler HUD (60-frame rolling window)
    static constexpr size_t HISTORY_SIZE = 60;
    float m_frameTimeHistory[HISTORY_SIZE] = {0};
    int m_historyOffset = 0;
    float m_smoothedFrameTime = 0.0f;
    float m_smoothedPresTime = 0.0f;

    // Telemetry export user notification banner
    std::string m_lastExportNotification;
    float m_exportNotificationTimer = 0.0f;
};

} // namespace pathways
