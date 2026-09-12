#include "core/Window.hpp"
#include "core/Logger.hpp"
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <vector>
#include "stb_image.h"

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

namespace pathways {

Window::Window(const Config& config)
    : m_headless(config.headless), m_width(config.width), m_height(config.height) {

    if (m_headless) {
        Logger::info("Running in HEADLESS mode (window creation bypassed). Resolution: {}x{}", m_width, m_height);
        m_displayInfo.nativeWidth = m_width;
        m_displayInfo.nativeHeight = m_height;
        m_displayInfo.usableWidth = m_width;
        m_displayInfo.usableHeight = m_height;
        m_displayInfo.displayAspect = static_cast<float>(m_width) / static_cast<float>(m_height);
        m_displayInfo.windowAspect = m_displayInfo.displayAspect;
        m_displayInfo.isPortrait = (m_displayInfo.displayAspect < 1.0f);
        m_displayInfo.isUltraWide = (m_displayInfo.displayAspect > 2.0f);
        return;
    }

#ifdef _WIN32
    // If launched in a virtual or non-interactive desktop (e.g. agent sandbox),
    // attach the window thread to "Default" desktop so the GUI window is visible on the user's monitor.
    HDESK hCurrentDesk = GetThreadDesktop(GetCurrentThreadId());
    char deskName[256] = {0};
    DWORD deskLen = 0;
    if (GetUserObjectInformationA(hCurrentDesk, UOI_NAME, deskName, sizeof(deskName), &deskLen)) {
        if (std::string(deskName) != "Default") {
            HDESK hUserDesk = OpenDesktopA("Default", 0, FALSE,
                DESKTOP_CREATEMENU | DESKTOP_CREATEWINDOW | DESKTOP_ENUMERATE |
                DESKTOP_HOOKCONTROL | DESKTOP_READOBJECTS | DESKTOP_SWITCHDESKTOP |
                DESKTOP_WRITEOBJECTS);
            if (hUserDesk) {
                if (SetThreadDesktop(hUserDesk)) {
                    Logger::info("Attached Window thread to interactive desktop 'Default'.");
                } else {
                    CloseDesktop(hUserDesk);
                }
            }
        }
    }
#endif

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS)) {
        throw std::runtime_error(std::string("Failed to initialize SDL3: ") + SDL_GetError());
    }

    SDL_SetAppMetadata("Pathways", "1.15.0", "pathways");

    // 1. Detect Physical Display & Resolution Characteristics
    SDL_DisplayID displayID = SDL_GetPrimaryDisplay();
    if (displayID != 0) {
        const char* name = SDL_GetDisplayName(displayID);
        if (name) {
            m_displayInfo.displayName = name;
        }

        const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(displayID);
        float density = 1.0f;
        if (mode) {
            density = (mode->pixel_density > 0.0f) ? mode->pixel_density : 1.0f;
            m_displayInfo.refreshRate = mode->refresh_rate;
        }

        float scale = SDL_GetDisplayContentScale(displayID);
        if (density <= 1.0f && scale > 1.0f) {
            density = scale;
        }
        m_displayInfo.contentScale = density;

        if (mode) {
            // mode->w and mode->h are in window coordinates/points in SDL3; multiply by pixel density for true physical pixels
            m_displayInfo.nativeWidth = static_cast<uint32_t>(std::round(mode->w * density));
            m_displayInfo.nativeHeight = static_cast<uint32_t>(std::round(mode->h * density));
        }

        SDL_Rect usable{};
        if (SDL_GetDisplayUsableBounds(displayID, &usable)) {
            m_displayInfo.usableWidth = static_cast<uint32_t>(std::round(usable.w * density));
            m_displayInfo.usableHeight = static_cast<uint32_t>(std::round(usable.h * density));
        } else {
            m_displayInfo.usableWidth = m_displayInfo.nativeWidth;
            m_displayInfo.usableHeight = m_displayInfo.nativeHeight;
        }

        if (m_displayInfo.nativeHeight > 0) {
            m_displayInfo.displayAspect = static_cast<float>(m_displayInfo.nativeWidth) / static_cast<float>(m_displayInfo.nativeHeight);
        }
        m_displayInfo.isPortrait = (m_displayInfo.displayAspect < 1.0f);
        m_displayInfo.isUltraWide = (m_displayInfo.displayAspect > 2.0f);
    }

    // 2. Select Sensible Window Bounds if user did not pass explicit --width or --height
    SDL_WindowFlags windowFlags = SDL_WINDOW_VULKAN | SDL_WINDOW_RESIZABLE | SDL_WINDOW_HIGH_PIXEL_DENSITY;

    m_isFullscreen = config.fullscreen;

    if (config.fullscreen) {
        windowFlags |= SDL_WINDOW_FULLSCREEN;
        if (!config.custom_resolution) {
            if (m_displayInfo.nativeWidth > 0 && m_displayInfo.nativeHeight > 0) {
                m_width = m_displayInfo.nativeWidth;
                m_height = m_displayInfo.nativeHeight;
            } else {
                m_width = 3840;
                m_height = 2160;
            }
        }
        Logger::info("Auto-detected display '{}' (Native: {}x{} px @ {:.2f}Hz, scale: {:.2f}, aspect: {:.3f}). Defaulting to Fullscreen: {}x{}",
                     m_displayInfo.displayName, m_displayInfo.nativeWidth, m_displayInfo.nativeHeight,
                     m_displayInfo.refreshRate, m_displayInfo.contentScale, m_displayInfo.displayAspect,
                     m_width, m_height);
    } else if (!config.custom_resolution) {
        uint32_t w = m_displayInfo.usableWidth;
        uint32_t h = m_displayInfo.usableHeight;
        m_width = w > 0 ? w : 1280;
        m_height = h > 0 ? h : 1080;
        windowFlags |= SDL_WINDOW_MAXIMIZED;

        Logger::info("Auto-detected display '{}' (Native: {}x{} px @ {:.2f}Hz, scale: {:.2f}, aspect: {:.3f}). Defaulting to full usable display (Maximized Windowed): {}x{}",
                     m_displayInfo.displayName, m_displayInfo.nativeWidth, m_displayInfo.nativeHeight,
                     m_displayInfo.refreshRate, m_displayInfo.contentScale, m_displayInfo.displayAspect,
                     m_width, m_height);
    } else {
        Logger::info("Using user-specified custom resolution: {}x{}", m_width, m_height);
    }

    m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);

    // SDL_CreateWindow takes window coordinates (points). Convert target physical pixels to points via density.
    float initDensity = (m_displayInfo.contentScale > 0.0f) ? m_displayInfo.contentScale : 1.0f;
    int initPointW = static_cast<int>(std::round(static_cast<float>(m_width) / initDensity));
    int initPointH = static_cast<int>(std::round(static_cast<float>(m_height) / initDensity));

    Logger::info("Initializing SDL3 window (pixel target: {}x{}, window points: {}x{})...", m_width, m_height, initPointW, initPointH);

    m_window = SDL_CreateWindow(
        "Pathways - Vulkan 1.4 Path Tracer",
        initPointW,
        initPointH,
        windowFlags
    );

    if (!m_window) {
        throw std::runtime_error(std::string("Failed to create SDL3 window: ") + SDL_GetError());
    }

    // Set application window icon
    const std::vector<std::string> iconSearchPaths = {
        "data/pathways.png",
        "../data/pathways.png",
        "/usr/share/pixmaps/pathways.png",
        "/usr/share/icons/hicolor/1024x1024/apps/pathways.png"
    };
    for (const auto& path : iconSearchPaths) {
        if (std::filesystem::exists(path)) {
            int iconW = 0, iconH = 0, channels = 0;
            unsigned char* iconData = stbi_load(path.c_str(), &iconW, &iconH, &channels, 4);
            if (iconData) {
                SDL_Surface* iconSurface = SDL_CreateSurfaceFrom(iconW, iconH, SDL_PIXELFORMAT_RGBA32, iconData, iconW * 4);
                if (iconSurface) {
                    SDL_SetWindowIcon(m_window, iconSurface);
                    SDL_DestroySurface(iconSurface);
                }
                stbi_image_free(iconData);
                break;
            }
        }
    }

    if (config.fullscreen) {
        SDL_SetWindowFullscreen(m_window, true);
    }

    SDL_ShowWindow(m_window);
    SDL_RaiseWindow(m_window);
    SDL_SyncWindow(m_window);
    SDL_PumpEvents();

    float actualDensity = SDL_GetWindowPixelDensity(m_window);
    if (actualDensity > 0.0f) {
        m_displayInfo.contentScale = actualDensity;
    }

    int actualW = 0, actualH = 0;
    SDL_GetWindowSizeInPixels(m_window, &actualW, &actualH);
    if (actualW > 0 && actualH > 0) {
        if (config.custom_resolution && !config.fullscreen) {
            m_width = config.width;
            m_height = config.height;
        } else {
            m_width = static_cast<uint32_t>(actualW);
            m_height = static_cast<uint32_t>(actualH);
        }
        m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);
    }

    Logger::info("SDL3 Window created and mapped successfully with pixel size {}x{} (density: {:.2f}).", m_width, m_height, m_displayInfo.contentScale);
}

Window::~Window() {
    if (!m_headless) {
        if (m_window) {
            SDL_DestroyWindow(m_window);
        }
        SDL_Quit();
        Logger::info("SDL3 Window destroyed.");
    }
}

VkSurfaceKHR Window::createSurface(VkInstance instance) {
    if (m_headless) {
        return VK_NULL_HANDLE;
    }

    VkSurfaceKHR surface = VK_NULL_HANDLE;
    if (!SDL_Vulkan_CreateSurface(m_window, instance, nullptr, &surface)) {
        throw std::runtime_error(std::string("Failed to create Vulkan surface: ") + SDL_GetError());
    }

    SDL_PumpEvents();
    int actualW = 0, actualH = 0;
    SDL_GetWindowSizeInPixels(m_window, &actualW, &actualH);
    if (actualW > 0 && actualH > 0) {
        m_width = static_cast<uint32_t>(actualW);
        m_height = static_cast<uint32_t>(actualH);
        m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);
    }

    Logger::info("Vulkan WSI Surface created successfully. Current window pixel size: {}x{}", m_width, m_height);
    return surface;
}

void Window::setRelativeMouseMode(bool enabled) {
    if (m_window) {
        SDL_SetWindowRelativeMouseMode(m_window, enabled);
    }
}

float Window::getPixelDensity() const {
    if (m_window) {
        float density = SDL_GetWindowPixelDensity(m_window);
        if (density > 0.0f) return density;
    }
    return (m_displayInfo.contentScale > 0.0f) ? m_displayInfo.contentScale : 1.0f;
}

void Window::getWindowSizeInPoints(int* w, int* h) const {
    if (m_window) {
        SDL_GetWindowSize(m_window, w, h);
    } else {
        float density = getPixelDensity();
        if (w) *w = static_cast<int>(std::round(static_cast<float>(m_width) / density));
        if (h) *h = static_cast<int>(std::round(static_cast<float>(m_height) / density));
    }
}

void Window::setWindowResolution(uint32_t width, uint32_t height) {
    if (m_headless || !m_window) return;

    if (m_isFullscreen || (SDL_GetWindowFlags(m_window) & SDL_WINDOW_FULLSCREEN)) {
        m_isFullscreen = false;
        SDL_SetWindowFullscreen(m_window, false);
        SDL_SyncWindow(m_window);
        SDL_PumpEvents();
    }

    width = std::max(64u, width);
    height = std::max(64u, height);

    float density = getPixelDensity();
    int pointW = static_cast<int>(std::round(static_cast<float>(width) / density));
    int pointH = static_cast<int>(std::round(static_cast<float>(height) / density));

    Logger::info("Setting window resolution: target {}x{} px (window coordinates: {}x{} pt, density: {:.2f})",
                 width, height, pointW, pointH, density);

    SDL_SetWindowSize(m_window, pointW, pointH);
    SDL_SyncWindow(m_window);
    SDL_PumpEvents();

    int actualW = 0, actualH = 0;
    SDL_GetWindowSizeInPixels(m_window, &actualW, &actualH);

    // Be accurate to the requested pixel dimensions rather than an inferred window size:
    // If within 2 pixels (fractional rounding tolerance), keep exact requested target dimensions.
    if (actualW > 0 && actualH > 0) {
        if (std::abs(actualW - static_cast<int>(width)) <= 2 &&
            std::abs(actualH - static_cast<int>(height)) <= 2) {
            m_width = width;
            m_height = height;
        } else {
            m_width = static_cast<uint32_t>(actualW);
            m_height = static_cast<uint32_t>(actualH);
        }
    } else {
        m_width = width;
        m_height = height;
    }
    m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);

    Logger::info("Applied window resolution: {}x{} px (framebuffer: {}x{} px)", m_width, m_height, actualW, actualH);

    if (m_resizeCallback) {
        m_resizeCallback(m_width, m_height);
    }
}

void Window::setTitle(const std::string& title) {
    if (m_window) {
        SDL_SetWindowTitle(m_window, title.c_str());
    }
}

void Window::toggleFullscreen() {
    if (m_headless || !m_window) return;

    bool currentlyFullscreen = (SDL_GetWindowFlags(m_window) & SDL_WINDOW_FULLSCREEN) != 0;
    bool targetFullscreen = !currentlyFullscreen;
    m_isFullscreen = targetFullscreen;

    Logger::info("Toggling fullscreen -> {}", targetFullscreen ? "Fullscreen" : "Windowed");
    if (!SDL_SetWindowFullscreen(m_window, targetFullscreen)) {
        Logger::error("Failed to set window fullscreen state: {}", SDL_GetError());
    }
    SDL_SyncWindow(m_window);
    SDL_PumpEvents();

    int actualW = 0, actualH = 0;
    SDL_GetWindowSizeInPixels(m_window, &actualW, &actualH);
    if (actualW > 0 && actualH > 0) {
        m_width = static_cast<uint32_t>(actualW);
        m_height = static_cast<uint32_t>(actualH);
        m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);
    }

    if (m_resizeCallback) {
        m_resizeCallback(m_width, m_height);
    }
}

void Window::pollEvents() {
    if (m_headless) {
        return;
    }

    SDL_Event event;
    while (SDL_PollEvent(&event)) {
        if (m_eventCallback && m_eventCallback(event)) {
            continue;
        }
        switch (event.type) {
            case SDL_EVENT_QUIT:
                Logger::info("Received SDL_EVENT_QUIT -> closing window.");
                m_shouldClose = true;
                break;
            case SDL_EVENT_WINDOW_CLOSE_REQUESTED:
                Logger::info("Received SDL_EVENT_WINDOW_CLOSE_REQUESTED -> closing window.");
                m_shouldClose = true;
                break;
            case SDL_EVENT_WINDOW_ENTER_FULLSCREEN:
                Logger::info("Window entered fullscreen mode.");
                m_isFullscreen = true;
                break;
            case SDL_EVENT_WINDOW_LEAVE_FULLSCREEN:
                Logger::info("Window left fullscreen mode.");
                m_isFullscreen = false;
                break;
            case SDL_EVENT_KEY_DOWN:
                if (event.key.key == SDLK_ESCAPE) {
                    Logger::info("Received SDLK_ESCAPE -> closing window.");
                    m_shouldClose = true;
                }
                if (event.key.key == SDLK_F11 && !event.key.repeat) {
                    toggleFullscreen();
                }
                if (m_keyCallback) {
                    m_keyCallback(event.key.key, true);
                }
                break;
            case SDL_EVENT_KEY_UP:
                if (m_keyCallback) {
                    m_keyCallback(event.key.key, false);
                }
                break;
            case SDL_EVENT_MOUSE_MOTION:
                if (m_mouseCallback) {
                    m_mouseCallback(event.motion.xrel, event.motion.yrel);
                }
                break;
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            case SDL_EVENT_WINDOW_RESIZED: {
                int actualW = 0, actualH = 0;
                SDL_GetWindowSizeInPixels(m_window, &actualW, &actualH);
                if (actualW > 0 && actualH > 0) {
                    uint32_t newW = static_cast<uint32_t>(actualW);
                    uint32_t newH = static_cast<uint32_t>(actualH);
                    if (newW != m_width || newH != m_height) {
                        Logger::info("Window pixel size changed: {}x{} -> {}x{}", m_width, m_height, newW, newH);
                        m_width = newW;
                        m_height = newH;
                        m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);
                        if (m_resizeCallback) {
                            m_resizeCallback(m_width, m_height);
                        }
                    }
                }
                break;
            }
            default:
                break;
        }
    }
}

} // namespace pathways
