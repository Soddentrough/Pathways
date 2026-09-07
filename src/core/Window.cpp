#include "core/Window.hpp"
#include "core/Logger.hpp"
#include <stdexcept>
#include <algorithm>

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

    // 1. Detect Physical Display & Resolution Characteristics
    SDL_DisplayID displayID = SDL_GetPrimaryDisplay();
    if (displayID != 0) {
        const char* name = SDL_GetDisplayName(displayID);
        if (name) {
            m_displayInfo.displayName = name;
        }

        const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(displayID);
        if (mode) {
            m_displayInfo.nativeWidth = static_cast<uint32_t>(mode->w);
            m_displayInfo.nativeHeight = static_cast<uint32_t>(mode->h);
            m_displayInfo.refreshRate = mode->refresh_rate;
        }

        m_displayInfo.contentScale = SDL_GetDisplayContentScale(displayID);

        SDL_Rect usable{};
        if (SDL_GetDisplayUsableBounds(displayID, &usable)) {
            m_displayInfo.usableWidth = static_cast<uint32_t>(usable.w);
            m_displayInfo.usableHeight = static_cast<uint32_t>(usable.h);
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

    if (!config.custom_resolution) {
        uint32_t w = (m_displayInfo.usableWidth / 16) * 16;
        uint32_t h = (m_displayInfo.usableHeight / 16) * 16;
        m_width = w > 0 ? w : 1280;
        m_height = h > 0 ? h : 1080;
        windowFlags |= SDL_WINDOW_MAXIMIZED;

        Logger::info("Auto-detected display '{}' ({}x{} @ {:.2f}Hz, scale: {:.2f}, aspect: {:.3f}). Defaulting to full usable display (Maximized): {}x{}",
                     m_displayInfo.displayName, m_displayInfo.nativeWidth, m_displayInfo.nativeHeight,
                     m_displayInfo.refreshRate, m_displayInfo.contentScale, m_displayInfo.displayAspect,
                     m_width, m_height);
    } else {
        Logger::info("Using user-specified custom resolution: {}x{}", m_width, m_height);
    }

    m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);

    Logger::info("Initializing SDL3 window ({}x{})...", m_width, m_height);

    m_window = SDL_CreateWindow(
        "Pathways - Vulkan 1.4 Path Tracer",
        static_cast<int>(m_width),
        static_cast<int>(m_height),
        windowFlags
    );

    if (!m_window) {
        throw std::runtime_error(std::string("Failed to create SDL3 window: ") + SDL_GetError());
    }

    SDL_ShowWindow(m_window);
    SDL_RaiseWindow(m_window);
    SDL_SyncWindow(m_window);
    SDL_PumpEvents();

    int actualW = 0, actualH = 0;
    SDL_GetWindowSizeInPixels(m_window, &actualW, &actualH);
    if (actualW > 0 && actualH > 0) {
        m_width = static_cast<uint32_t>(actualW);
        m_height = static_cast<uint32_t>(actualH);
        m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);
    }

    Logger::info("SDL3 Window created and mapped successfully with pixel size {}x{}.", m_width, m_height);
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

void Window::setWindowResolution(uint32_t width, uint32_t height) {
    if (m_headless || !m_window) return;

    if (m_isFullscreen || (SDL_GetWindowFlags(m_window) & SDL_WINDOW_FULLSCREEN)) {
        m_isFullscreen = false;
        SDL_SetWindowFullscreen(m_window, false);
        SDL_SyncWindow(m_window);
    }

    width = std::max(64u, (width / 16) * 16);
    height = std::max(64u, (height / 16) * 16);

    SDL_SetWindowSize(m_window, static_cast<int>(width), static_cast<int>(height));
    SDL_SyncWindow(m_window);

    int actualW = 0, actualH = 0;
    SDL_GetWindowSizeInPixels(m_window, &actualW, &actualH);
    if (actualW > 0 && actualH > 0) {
        m_width = static_cast<uint32_t>(actualW);
        m_height = static_cast<uint32_t>(actualH);
    } else {
        m_width = width;
        m_height = height;
    }
    m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);

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
