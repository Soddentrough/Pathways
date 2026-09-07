#pragma once

#include <SDL3/SDL.h>
#include <SDL3/SDL_vulkan.h>
#include <vulkan/vulkan.h>
#include "core/Config.hpp"
#include <string>
#include <functional>

namespace pathways {

struct DisplayInfo {
    std::string displayName = "Default Display";
    uint32_t nativeWidth = 1920;
    uint32_t nativeHeight = 1080;
    float refreshRate = 60.0f;
    float contentScale = 1.0f;
    uint32_t usableWidth = 1920;
    uint32_t usableHeight = 1080;
    float displayAspect = 16.0f / 9.0f;
    float windowAspect = 16.0f / 9.0f;
    bool isPortrait = false;
    bool isUltraWide = false;
};

class Window {
public:
    Window(const Config& config);
    ~Window();

    bool isHeadless() const { return m_headless; }
    bool shouldClose() const { return m_shouldClose; }
    void close() { m_shouldClose = true; }

    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }
    SDL_Window* getSDLWindow() const { return m_window; }
    const DisplayInfo& getDisplayInfo() const { return m_displayInfo; }

    VkSurfaceKHR createSurface(VkInstance instance);
    void pollEvents();

    void setWindowResolution(uint32_t width, uint32_t height);
    void setTitle(const std::string& title);
    void toggleFullscreen();
    bool isFullscreen() const { return m_isFullscreen; }

    // Event callbacks
    using KeyCallback = std::function<void(SDL_Keycode key, bool isDown)>;
    using MouseMoveCallback = std::function<void(float xrel, float yrel)>;
    using EventCallback = std::function<bool(const SDL_Event&)>;
    using ResizeCallback = std::function<void(uint32_t width, uint32_t height)>;

    void setKeyCallback(KeyCallback cb) { m_keyCallback = cb; }
    void setMouseMoveCallback(MouseMoveCallback cb) { m_mouseCallback = cb; }
    void setEventCallback(EventCallback cb) { m_eventCallback = cb; }
    void setResizeCallback(ResizeCallback cb) { m_resizeCallback = cb; }
    void setRelativeMouseMode(bool enabled);

private:
    bool m_headless = false;
    bool m_shouldClose = false;
    bool m_isFullscreen = false;
    uint32_t m_width = 3840;
    uint32_t m_height = 2160;
    DisplayInfo m_displayInfo;

    SDL_Window* m_window = nullptr;
    KeyCallback m_keyCallback;
    MouseMoveCallback m_mouseCallback;
    EventCallback m_eventCallback;
    ResizeCallback m_resizeCallback;
};

} // namespace pathways
