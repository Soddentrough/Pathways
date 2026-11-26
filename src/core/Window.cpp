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
    #include <dxgi1_6.h>
    #include <wrl/client.h>
#endif

namespace pathways {

#ifdef _WIN32
void Window::queryDxgiHdrInfo() {
    // Query DXGI for native HDR display capabilities and exact luminance limits
    Microsoft::WRL::ComPtr<IDXGIFactory1> dxgiFactory;
    if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&dxgiFactory)))) {
        Microsoft::WRL::ComPtr<IDXGIAdapter1> adapter;
        for (UINT a = 0; dxgiFactory->EnumAdapters1(a, &adapter) != DXGI_ERROR_NOT_FOUND; ++a) {
            Microsoft::WRL::ComPtr<IDXGIOutput> output;
            for (UINT o = 0; adapter->EnumOutputs(o, &output) != DXGI_ERROR_NOT_FOUND; ++o) {
                Microsoft::WRL::ComPtr<IDXGIOutput6> output6;
                if (SUCCEEDED(output.As(&output6))) {
                    DXGI_OUTPUT_DESC1 desc1{};
                    if (SUCCEEDED(output6->GetDesc1(&desc1)) && desc1.AttachedToDesktop) {
                        if (desc1.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020) {
                            m_displayInfo.isDesktopHdr = true;
                        }
                        if (desc1.BitsPerColor >= 10 || desc1.MaxLuminance > 300.0f) {
                            m_displayInfo.isDisplayHdrCapable = true;
                        }
                        if (desc1.MaxLuminance > 0.0f) {
                            m_displayInfo.maxLuminanceNits = desc1.MaxLuminance;
                            m_displayInfo.minLuminanceNits = desc1.MinLuminance;
                            m_displayInfo.maxFullFrameLuminanceNits = desc1.MaxFullFrameLuminance;
                            m_displayInfo.redPrimary[0] = desc1.RedPrimary[0];
                            m_displayInfo.redPrimary[1] = desc1.RedPrimary[1];
                            m_displayInfo.greenPrimary[0] = desc1.GreenPrimary[0];
                            m_displayInfo.greenPrimary[1] = desc1.GreenPrimary[1];
                            m_displayInfo.bluePrimary[0] = desc1.BluePrimary[0];
                            m_displayInfo.bluePrimary[1] = desc1.BluePrimary[1];
                            m_displayInfo.whitePoint[0] = desc1.WhitePoint[0];
                            m_displayInfo.whitePoint[1] = desc1.WhitePoint[1];
                        }
                        m_displayInfo.hmonitor = static_cast<void*>(desc1.Monitor);
                        Logger::info("DXGI Display Output: Desktop HDR: {}, HDR Capable: {}, Peak: {:.1f} nits, Min: {:.4f} nits (ColorSpace: {})",
                                     m_displayInfo.isDesktopHdr ? "ACTIVE" : "INACTIVE",
                                     m_displayInfo.isDisplayHdrCapable ? "YES" : "NO",
                                     m_displayInfo.maxLuminanceNits, m_displayInfo.minLuminanceNits,
                                     static_cast<int>(desc1.ColorSpace));
                        break;
                    }
                }
            }
            if (m_displayInfo.isDisplayHdrCapable) break;
        }
    }
}

namespace {
struct MonitorEdgeProbe {
    HMONITOR self = nullptr;
    RECT selfRect{};
    bool bottomBlocked = false;
    bool rightBlocked = false;
};

BOOL CALLBACK enumMonitorNeighborsProc(HMONITOR hMon, HDC, LPRECT, LPARAM lParam) {
    auto* probe = reinterpret_cast<MonitorEdgeProbe*>(lParam);
    if (!probe || hMon == probe->self) return TRUE;
    MONITORINFO mi{};
    mi.cbSize = sizeof(mi);
    if (!GetMonitorInfoW(hMon, &mi)) return TRUE;
    const RECT& r = mi.rcMonitor;
    const RECT& s = probe->selfRect;
    const bool horizOverlap = (r.left < s.right) && (r.right > s.left);
    const bool vertOverlap  = (r.top < s.bottom) && (r.bottom > s.top);
    if (horizOverlap && r.top == s.bottom) {
        probe->bottomBlocked = true;
    }
    if (vertOverlap && r.left == s.right) {
        probe->rightBlocked = true;
    }
    return TRUE;
}

HWND createComposedGuardWindow(HWND ownerHwnd, int x, int y) {
    static bool s_classRegistered = false;
    HINSTANCE hInst = GetModuleHandleW(nullptr);
    const wchar_t* kClassName = L"PathwaysDwmComposedGuard";
    if (!s_classRegistered) {
        WNDCLASSEXW wc{};
        wc.cbSize = sizeof(wc);
        wc.lpfnWndProc = DefWindowProcW;
        wc.hInstance = hInst;
        wc.lpszClassName = kClassName;
        if (RegisterClassExW(&wc) != 0 || GetLastError() == ERROR_CLASS_ALREADY_EXISTS) {
            s_classRegistered = true;
        }
    }
    if (!s_classRegistered) return nullptr;

    HWND guard = CreateWindowExW(
        WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE,
        kClassName,
        L"",
        WS_POPUP,
        x, y, 1, 1,
        ownerHwnd,
        nullptr,
        hInst,
        nullptr
    );
    if (guard) {
        SetLayeredWindowAttributes(guard, RGB(0, 0, 0), 1, LWA_ALPHA);
    }
    return guard;
}
} // namespace

void Window::applyWin32ComposedBorderless(bool showGuard) {
    if (!m_window || m_headless) return;
    HWND hwnd = static_cast<HWND>(SDL_GetPointerProperty(
        SDL_GetWindowProperties(m_window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, nullptr));
    if (!hwnd) return;

    HMONITOR hmon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    MONITORINFO mi{};
    mi.cbSize = sizeof(mi);
    if (!hmon || !GetMonitorInfoW(hmon, &mi)) return;

    m_displayInfo.hmonitor = static_cast<void*>(hmon);

    const int monX = mi.rcMonitor.left;
    const int monY = mi.rcMonitor.top;
    const int monW = static_cast<int>(mi.rcMonitor.right - mi.rcMonitor.left);
    const int monH = static_cast<int>(mi.rcMonitor.bottom - mi.rcMonitor.top);
    if (monW <= 0 || monH <= 0) return;

    MonitorEdgeProbe probe{};
    probe.self = hmon;
    probe.selfRect = mi.rcMonitor;
    EnumDisplayMonitors(nullptr, nullptr, enumMonitorNeighborsProc, reinterpret_cast<LPARAM>(&probe));

    // Extend swapchain surface by +1 physical pixel past an unshared monitor edge (prefer bottom, then right),
    // paired with an owned 1x1 click-through layered guard window.
    // Why: On Windows 11, an exact monitor-sized WS_POPUP surface (e.g. 3840x2160) causes DWM + dxgkrnl to
    // promote the flip swapchain out of Composed Flip into DirectFlip / Independent Flip, asserting HDMI 2.1
    // VRR (FreeSync) and ALLM InfoFrames that interrupt multiplexed HDMI audio on TVs/AVRs (e.g. LG OLED).
    // Disqualifying DirectFlip keeps presentation in native DWM Composed Flip (16-bit scRGB Linear HDR) with
    // full-screen coverage (taskbar hidden, 0 visible gaps, 1:1 native viewport pixels) and zero audio dropout.
    int padW = 0;
    int padH = 1;
    if (probe.bottomBlocked && !probe.rightBlocked) {
        padW = 1;
        padH = 0;
    }

    const int surfW = monW + padW;
    const int surfH = monH + padH;

    LONG_PTR style = GetWindowLongPtrW(hwnd, GWL_STYLE);
    style &= ~(WS_CAPTION | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX | WS_SYSMENU);
    style |= WS_POPUP;
    SetWindowLongPtrW(hwnd, GWL_STYLE, style);

    SetWindowPos(hwnd, HWND_TOP, monX, monY, surfW, surfH, SWP_NOACTIVATE | SWP_FRAMECHANGED);

    bool guardCreatedNow = false;
    if (showGuard) {
        if (!m_composedGuardHwnd) {
            m_composedGuardHwnd = static_cast<void*>(createComposedGuardWindow(hwnd, monX, monY));
            guardCreatedNow = (m_composedGuardHwnd != nullptr);
        }
        if (m_composedGuardHwnd) {
            HWND guard = static_cast<HWND>(m_composedGuardHwnd);
            SetWindowPos(guard, HWND_TOP, monX, monY, 1, 1, SWP_NOACTIVATE | SWP_SHOWWINDOW);
        }
    }

    const bool sizeChanged = (m_surfaceWidth != static_cast<uint32_t>(surfW) ||
                              m_surfaceHeight != static_cast<uint32_t>(surfH));
    m_width = static_cast<uint32_t>(monW);
    m_height = static_cast<uint32_t>(monH);
    m_surfaceWidth = static_cast<uint32_t>(surfW);
    m_surfaceHeight = static_cast<uint32_t>(surfH);
    m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);
    m_composedBorderlessActive = true;

    if (showGuard && (guardCreatedNow || sizeChanged)) {
        Logger::info("Win32 Borderless Composed Fullscreen active: viewport {}x{}, DWM surface {}x{} (DirectFlip/VRR/ALLM link-retrain suppressed).",
                     m_width, m_height, m_surfaceWidth, m_surfaceHeight);
    }
}

void Window::releaseWin32ComposedBorderless() {
    m_composedBorderlessActive = false;
    if (m_composedGuardHwnd) {
        DestroyWindow(static_cast<HWND>(m_composedGuardHwnd));
        m_composedGuardHwnd = nullptr;
    }
}
#endif

Window::Window(const Config& config)
    : m_headless(config.headless),
      m_exclusiveFullscreen(config.exclusive_fullscreen),
      m_width(config.width),
      m_height(config.height),
      m_surfaceWidth(config.width),
      m_surfaceHeight(config.height) {

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

    if (!SDL_Init(SDL_INIT_VIDEO | SDL_INIT_EVENTS | SDL_INIT_GAMEPAD)) {
        throw std::runtime_error(std::string("Failed to initialize SDL3: ") + SDL_GetError());
    }

    SDL_SetAppMetadata("Pathways", "1.18.0", "pathways");

    // 1. Detect Physical Display & Resolution Characteristics
    float initPixelDensity = 1.0f;
    SDL_DisplayID displayID = SDL_GetPrimaryDisplay();
    if (displayID != 0) {
        const char* name = SDL_GetDisplayName(displayID);
        if (name) {
            m_displayInfo.displayName = name;
        }

        const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(displayID);
        float pixelDensity = 1.0f;
        if (mode) {
            pixelDensity = (mode->pixel_density > 0.0f) ? mode->pixel_density : 1.0f;
            m_displayInfo.refreshRate = mode->refresh_rate;
        }
        initPixelDensity = pixelDensity;

        float scale = SDL_GetDisplayContentScale(displayID);
        m_displayInfo.contentScale = (scale > 0.0f) ? scale : pixelDensity;

        if (mode) {
            // mode->w and mode->h are in window coordinates/points in SDL3; multiply by mode->pixel_density for true physical pixels
            m_displayInfo.nativeWidth = static_cast<uint32_t>(std::round(mode->w * pixelDensity));
            m_displayInfo.nativeHeight = static_cast<uint32_t>(std::round(mode->h * pixelDensity));
        }

        SDL_Rect usable{};
        if (SDL_GetDisplayUsableBounds(displayID, &usable)) {
            m_displayInfo.usableWidth = static_cast<uint32_t>(std::round(usable.w * pixelDensity));
            m_displayInfo.usableHeight = static_cast<uint32_t>(std::round(usable.h * pixelDensity));
        } else {
            m_displayInfo.usableWidth = m_displayInfo.nativeWidth;
            m_displayInfo.usableHeight = m_displayInfo.nativeHeight;
        }

        if (m_displayInfo.nativeHeight > 0) {
            m_displayInfo.displayAspect = static_cast<float>(m_displayInfo.nativeWidth) / static_cast<float>(m_displayInfo.nativeHeight);
        }
        m_displayInfo.isPortrait = (m_displayInfo.displayAspect < 1.0f);
        m_displayInfo.isUltraWide = (m_displayInfo.displayAspect > 2.0f);

        // Cross-platform SDL3 display HDR query (Linux Wayland, KMS/DRM, Windows)
        SDL_PropertiesID dispProps = SDL_GetDisplayProperties(displayID);
        if (dispProps && SDL_GetBooleanProperty(dispProps, SDL_PROP_DISPLAY_HDR_ENABLED_BOOLEAN, false)) {
            m_displayInfo.isDisplayHdrCapable = true;
            m_displayInfo.isDesktopHdr = true;
        }

#ifdef _WIN32
        queryDxgiHdrInfo();
#endif
    }

    // 2. Select Sensible Window Bounds if user did not pass explicit --width or --height
    // Create hidden so the user does not stare at an un-presented black window while shaders compile.
    SDL_WindowFlags windowFlags = SDL_WINDOW_VULKAN | SDL_WINDOW_HIGH_PIXEL_DENSITY | SDL_WINDOW_HIDDEN;

    m_isFullscreen = config.fullscreen;

    SDL_Rect displayBounds{};
    bool hasDisplayBounds = (displayID != 0) && SDL_GetDisplayBounds(displayID, &displayBounds);

    if (config.fullscreen) {
        if (config.exclusive_fullscreen) {
            windowFlags |= SDL_WINDOW_FULLSCREEN | SDL_WINDOW_RESIZABLE;
        } else {
            // Borderless Composed Fullscreen Windowed mode (desktop bounds without mode-switch or DirectFlip link retrain)
            windowFlags |= SDL_WINDOW_BORDERLESS;
        }
        if (!config.custom_resolution) {
            if (m_displayInfo.nativeWidth > 0 && m_displayInfo.nativeHeight > 0) {
                m_width = m_displayInfo.nativeWidth;
                m_height = m_displayInfo.nativeHeight;
            } else {
                m_width = 3840;
                m_height = 2160;
            }
        }
        Logger::info("Auto-detected display '{}' (Native: {}x{} px @ {:.2f}Hz, scale: {:.2f}, aspect: {:.3f}). Defaulting to Fullscreen Windowed: {}x{}",
                     m_displayInfo.displayName, m_displayInfo.nativeWidth, m_displayInfo.nativeHeight,
                     m_displayInfo.refreshRate, m_displayInfo.contentScale, m_displayInfo.displayAspect,
                     m_width, m_height);
    } else if (!config.custom_resolution) {
        uint32_t w = m_displayInfo.usableWidth;
        uint32_t h = m_displayInfo.usableHeight;
        m_width = w > 0 ? w : 1280;
        m_height = h > 0 ? h : 1080;
        windowFlags |= SDL_WINDOW_RESIZABLE | SDL_WINDOW_MAXIMIZED;

        Logger::info("Auto-detected display '{}' (Native: {}x{} px @ {:.2f}Hz, scale: {:.2f}, aspect: {:.3f}). Defaulting to full usable display (Maximized Windowed): {}x{}",
                     m_displayInfo.displayName, m_displayInfo.nativeWidth, m_displayInfo.nativeHeight,
                     m_displayInfo.refreshRate, m_displayInfo.contentScale, m_displayInfo.displayAspect,
                     m_width, m_height);
    } else {
        windowFlags |= SDL_WINDOW_RESIZABLE;
        Logger::info("Using user-specified custom resolution: {}x{}", m_width, m_height);
    }

    m_surfaceWidth = m_width;
    m_surfaceHeight = m_height;
    m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);

    // SDL_CreateWindow takes window coordinates (points). Convert target physical pixels to points via pixel density.
    float initDensity = (initPixelDensity > 0.0f) ? initPixelDensity : 1.0f;
    int initPointW = static_cast<int>(std::round(static_cast<float>(m_width) / initDensity));
    int initPointH = static_cast<int>(std::round(static_cast<float>(m_height) / initDensity));

    if (config.fullscreen && !config.exclusive_fullscreen && hasDisplayBounds) {
        initPointW = displayBounds.w;
#ifdef _WIN32
        initPointH = displayBounds.h + 1;
#else
        initPointH = displayBounds.h;
#endif
    }

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

    if (config.fullscreen && !config.exclusive_fullscreen && hasDisplayBounds) {
        SDL_SetWindowPosition(m_window, displayBounds.x, displayBounds.y);
#ifdef _WIN32
        applyWin32ComposedBorderless(false);
#else
        SDL_SetWindowSize(m_window, displayBounds.w, displayBounds.h);
#endif
    }

#ifdef _WIN32
    HWND hwnd = (HWND)SDL_GetPointerProperty(SDL_GetWindowProperties(m_window), SDL_PROP_WINDOW_WIN32_HWND_POINTER, NULL);
    if (hwnd) {
        HMONITOR hmon = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
        if (hmon) {
            m_displayInfo.hmonitor = static_cast<void*>(hmon);
        }
    }
#endif

    // Query cross-platform SDL3 window HDR properties and dynamic headroom
    SDL_PropertiesID winProps = SDL_GetWindowProperties(m_window);
    if (winProps) {
        if (SDL_GetBooleanProperty(winProps, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false)) {
            m_displayInfo.isDisplayHdrCapable = true;
            m_displayInfo.isDesktopHdr = true;
        }
        float headroom = SDL_GetFloatProperty(winProps, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 1.0f);
        float sdrWhite = SDL_GetFloatProperty(winProps, SDL_PROP_WINDOW_SDR_WHITE_LEVEL_FLOAT, 1.0f);
        if (headroom > 1.0f && m_displayInfo.maxLuminanceNits <= 0.0f) {
            float baseWhite = (config.hdr_paper_white_nits > 0.0f) ? config.hdr_paper_white_nits : 200.0f;
            float calculatedPeak = baseWhite * headroom;
            m_displayInfo.maxLuminanceNits = std::clamp(calculatedPeak, 400.0f, 10000.0f);
        }
        Logger::info("SDL3 HDR Status: Display HDR: {}, Window HDR: {}, Headroom: {:.2f}x, SDR White: {:.2f} (Target Peak: {:.1f} nits)",
                     m_displayInfo.isDisplayHdrCapable ? "CAPABLE" : "SDR",
                     m_displayInfo.isDesktopHdr ? "ACTIVE" : "INACTIVE",
                     headroom, sdrWhite, m_displayInfo.maxLuminanceNits);
    }
    if (config.custom_hdr_peak && config.hdr_peak_nits > 0.0f) {
        m_displayInfo.maxLuminanceNits = config.hdr_peak_nits;
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

    if (config.fullscreen && config.exclusive_fullscreen) {
        SDL_SetWindowFullscreen(m_window, true);
    }

    float actualDensity = SDL_GetWindowPixelDensity(m_window);
    float winDisplayScale = SDL_GetWindowDisplayScale(m_window);
    if (winDisplayScale > 0.0f) {
        m_displayInfo.contentScale = winDisplayScale;
    } else if (actualDensity > 0.0f) {
        m_displayInfo.contentScale = actualDensity;
    }

    int actualW = 0, actualH = 0;
    SDL_GetWindowSizeInPixels(m_window, &actualW, &actualH);
    if (actualW > 0 && actualH > 0) {
        if (m_composedBorderlessActive) {
            m_surfaceWidth = static_cast<uint32_t>(actualW);
            m_surfaceHeight = static_cast<uint32_t>(actualH);
            if (config.custom_resolution) {
                m_width = std::min(config.width, m_surfaceWidth);
                m_height = std::min(config.height, m_surfaceHeight);
            }
        } else if (config.custom_resolution && !config.fullscreen) {
            m_width = config.width;
            m_height = config.height;
            m_surfaceWidth = m_width;
            m_surfaceHeight = m_height;
        } else {
            m_width = static_cast<uint32_t>(actualW);
            m_height = static_cast<uint32_t>(actualH);
            m_surfaceWidth = m_width;
            m_surfaceHeight = m_height;
        }
        m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);
    }

    Logger::info("SDL3 Window created (hidden until engine ready) with pixel size {}x{} (surface: {}x{}, pixel density: {:.2f}, UI scale: {:.2f}).",
                 m_width, m_height, getSurfaceWidth(), getSurfaceHeight(),
                 actualDensity > 0.0f ? actualDensity : 1.0f, m_displayInfo.contentScale);
}

void Window::show() {
    if (m_headless || !m_window || m_shown) {
        return;
    }
    SDL_ShowWindow(m_window);
    SDL_RaiseWindow(m_window);
#ifdef _WIN32
    if (m_isFullscreen && !m_exclusiveFullscreen) {
        applyWin32ComposedBorderless(true);
    }
#endif
    SDL_SyncWindow(m_window);
    SDL_PumpEvents();
    m_shown = true;
    Logger::info("SDL3 Window mapped and presented ({}x{}, surface: {}x{}).",
                 m_width, m_height, getSurfaceWidth(), getSurfaceHeight());
}

Window::~Window() {
    if (!m_headless) {
#ifdef _WIN32
        releaseWin32ComposedBorderless();
#endif
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

    SDL_SyncWindow(m_window);
    for (int p = 0; p < 8; ++p) {
        SDL_PumpEvents();
    }
    int actualW = 0, actualH = 0;
    SDL_GetWindowSizeInPixels(m_window, &actualW, &actualH);
    if (actualW > 0 && actualH > 0) {
        if (m_composedBorderlessActive) {
            m_surfaceWidth = static_cast<uint32_t>(actualW);
            m_surfaceHeight = static_cast<uint32_t>(actualH);
        } else {
            m_width = static_cast<uint32_t>(actualW);
            m_height = static_cast<uint32_t>(actualH);
            m_surfaceWidth = m_width;
            m_surfaceHeight = m_height;
            m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);
        }
    }

    Logger::info("Vulkan WSI Surface created successfully. Viewport pixel size: {}x{} (surface: {}x{})",
                 m_width, m_height, getSurfaceWidth(), getSurfaceHeight());
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
    return 1.0f;
}

float Window::getDisplayScale() const {
    if (m_window) {
        float scale = SDL_GetWindowDisplayScale(m_window);
        if (scale > 0.0f) return scale;
    }
    return (m_displayInfo.contentScale > 0.0f) ? m_displayInfo.contentScale : 1.0f;
}

void Window::queryViewportAndSurfacePixels(uint32_t& outViewW, uint32_t& outViewH,
                                           uint32_t& outSurfW, uint32_t& outSurfH) const {
    int curW = 0, curH = 0;
    if (m_window) {
        SDL_GetWindowSizeInPixels(m_window, &curW, &curH);
    }
    const uint32_t rawSurfW = (curW > 0) ? static_cast<uint32_t>(curW) : getSurfaceWidth();
    const uint32_t rawSurfH = (curH > 0) ? static_cast<uint32_t>(curH) : getSurfaceHeight();
    if (m_composedBorderlessActive && rawSurfW == m_surfaceWidth && rawSurfH == m_surfaceHeight) {
        outViewW = m_width;
        outViewH = m_height;
        outSurfW = m_surfaceWidth;
        outSurfH = m_surfaceHeight;
    } else {
        outViewW = (curW > 0) ? rawSurfW : m_width;
        outViewH = (curH > 0) ? rawSurfH : m_height;
        outSurfW = outViewW;
        outSurfH = outViewH;
    }
}

void Window::getWindowSizeInPoints(int* w, int* h) const {
    if (m_window && !m_composedBorderlessActive) {
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
#ifdef _WIN32
        releaseWin32ComposedBorderless();
#endif
        SDL_SetWindowFullscreen(m_window, false);
        SDL_SetWindowBordered(m_window, true);
        SDL_SetWindowResizable(m_window, true);
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
    SDL_SetWindowPosition(m_window, SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED);
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
    m_surfaceWidth = (actualW > 0) ? static_cast<uint32_t>(actualW) : m_width;
    m_surfaceHeight = (actualH > 0) ? static_cast<uint32_t>(actualH) : m_height;
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

    bool currentlyFullscreen = (SDL_GetWindowFlags(m_window) & SDL_WINDOW_FULLSCREEN) != 0 || m_isFullscreen;
    bool targetFullscreen = !currentlyFullscreen;
    m_isFullscreen = targetFullscreen;

    Logger::info("Toggling fullscreen -> {}", targetFullscreen ? "Fullscreen" : "Windowed");
    if (targetFullscreen) {
        if (m_exclusiveFullscreen) {
            SDL_SetWindowFullscreen(m_window, true);
        } else {
#ifdef _WIN32
            SDL_SetWindowFullscreen(m_window, false);
            SDL_SetWindowBordered(m_window, false);
            SDL_SetWindowResizable(m_window, false);
            applyWin32ComposedBorderless(true);
#else
            SDL_DisplayID disp = SDL_GetDisplayForWindow(m_window);
            SDL_Rect bounds{};
            if (disp != 0 && SDL_GetDisplayBounds(disp, &bounds)) {
                SDL_SetWindowBordered(m_window, false);
                SDL_SetWindowPosition(m_window, bounds.x, bounds.y);
                SDL_SetWindowSize(m_window, bounds.w, bounds.h);
            } else {
                SDL_SetWindowFullscreen(m_window, true);
            }
#endif
        }
    } else {
#ifdef _WIN32
        releaseWin32ComposedBorderless();
#endif
        SDL_SetWindowFullscreen(m_window, false);
        SDL_SetWindowBordered(m_window, true);
        SDL_SetWindowResizable(m_window, true);
        float density = getPixelDensity();
        uint32_t winPxW = (m_displayInfo.usableWidth > 0) ? m_displayInfo.usableWidth : m_width;
        uint32_t winPxH = (m_displayInfo.usableHeight > 0) ? m_displayInfo.usableHeight : m_height;
        int ptW = static_cast<int>(std::round(static_cast<float>(winPxW) / density));
        int ptH = static_cast<int>(std::round(static_cast<float>(winPxH) / density));
        SDL_SetWindowSize(m_window, ptW, ptH);
        SDL_MaximizeWindow(m_window);
    }
    SDL_SyncWindow(m_window);
    SDL_PumpEvents();

    int actualW = 0, actualH = 0;
    SDL_GetWindowSizeInPixels(m_window, &actualW, &actualH);
    if (actualW > 0 && actualH > 0) {
        if (m_composedBorderlessActive) {
            m_surfaceWidth = static_cast<uint32_t>(actualW);
            m_surfaceHeight = static_cast<uint32_t>(actualH);
        } else {
            m_width = static_cast<uint32_t>(actualW);
            m_height = static_cast<uint32_t>(actualH);
            m_surfaceWidth = m_width;
            m_surfaceHeight = m_height;
            m_displayInfo.windowAspect = static_cast<float>(m_width) / static_cast<float>(m_height);
        }
    }

    if (m_resizeCallback) {
        m_resizeCallback(m_width, m_height);
    }
}

bool Window::isMinimized() const {
    if (m_window) {
        Uint64 flags = SDL_GetWindowFlags(m_window);
        return (flags & SDL_WINDOW_MINIMIZED) != 0 || (flags & SDL_WINDOW_OCCLUDED) != 0;
    }
    return false;
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
            case SDL_EVENT_WINDOW_HDR_STATE_CHANGED: {
                SDL_PropertiesID wp = SDL_GetWindowProperties(m_window);
                if (wp) {
                    bool hdr = SDL_GetBooleanProperty(wp, SDL_PROP_WINDOW_HDR_ENABLED_BOOLEAN, false);
                    float headroom = SDL_GetFloatProperty(wp, SDL_PROP_WINDOW_HDR_HEADROOM_FLOAT, 1.0f);
                    m_displayInfo.isDisplayHdrCapable = hdr;
                    m_displayInfo.isDesktopHdr = hdr;
#ifdef _WIN32
                    queryDxgiHdrInfo();
#endif
                    if (headroom > 1.0f && m_displayInfo.maxLuminanceNits <= 0.0f) {
                        m_displayInfo.maxLuminanceNits = std::clamp(200.0f * headroom, 400.0f, 10000.0f);
                    }
                    Logger::info("SDL_EVENT_WINDOW_HDR_STATE_CHANGED: HDR active={}, headroom={:.2f}x (Peak: {:.1f} nits)",
                                 hdr, headroom, m_displayInfo.maxLuminanceNits);
                    if (m_resizeCallback) {
                        m_resizeCallback(m_width, m_height);
                    }
                }
                break;
            }
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
            case SDL_EVENT_WINDOW_DISPLAY_SCALE_CHANGED:
            case SDL_EVENT_WINDOW_DISPLAY_CHANGED: {
                float newScale = SDL_GetWindowDisplayScale(m_window);
                if (newScale > 0.0f && std::abs(newScale - m_displayInfo.contentScale) > 0.005f) {
                    Logger::info("Window display scale changed: {:.2f} -> {:.2f}", m_displayInfo.contentScale, newScale);
                    m_displayInfo.contentScale = newScale;
                }
                SDL_DisplayID dispId = SDL_GetDisplayForWindow(m_window);
                if (dispId != 0) {
                    const char* name = SDL_GetDisplayName(dispId);
                    if (name) m_displayInfo.displayName = name;
                    const SDL_DisplayMode* mode = SDL_GetCurrentDisplayMode(dispId);
                    float density = 1.0f;
                    if (mode) {
                        density = (mode->pixel_density > 0.0f) ? mode->pixel_density : 1.0f;
                        m_displayInfo.refreshRate = mode->refresh_rate;
                        m_displayInfo.nativeWidth = static_cast<uint32_t>(std::round(mode->w * density));
                        m_displayInfo.nativeHeight = static_cast<uint32_t>(std::round(mode->h * density));
                    }
                    SDL_Rect usable{};
                    if (SDL_GetDisplayUsableBounds(dispId, &usable)) {
                        m_displayInfo.usableWidth = static_cast<uint32_t>(std::round(usable.w * density));
                        m_displayInfo.usableHeight = static_cast<uint32_t>(std::round(usable.h * density));
                    }
                    if (m_displayInfo.nativeHeight > 0) {
                        m_displayInfo.displayAspect = static_cast<float>(m_displayInfo.nativeWidth) / static_cast<float>(m_displayInfo.nativeHeight);
                    }
                    m_displayInfo.isPortrait = (m_displayInfo.displayAspect < 1.0f);
                    m_displayInfo.isUltraWide = (m_displayInfo.displayAspect > 2.0f);
                }
#ifdef _WIN32
                if (m_composedBorderlessActive) {
                    uint32_t prevW = m_width;
                    uint32_t prevH = m_height;
                    applyWin32ComposedBorderless(true);
                    if ((m_width != prevW || m_height != prevH) && m_resizeCallback) {
                        m_resizeCallback(m_width, m_height);
                    }
                }
#endif
                break;
            }
            case SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED:
            case SDL_EVENT_WINDOW_RESIZED: {
                int actualW = 0, actualH = 0;
                SDL_GetWindowSizeInPixels(m_window, &actualW, &actualH);
                if (actualW > 0 && actualH > 0) {
                    uint32_t newW = static_cast<uint32_t>(actualW);
                    uint32_t newH = static_cast<uint32_t>(actualH);
                    if (m_composedBorderlessActive) {
                        if (newW == m_surfaceWidth && newH == m_surfaceHeight) {
                            break;
                        }
#ifdef _WIN32
                        applyWin32ComposedBorderless(true);
                        if (m_resizeCallback) {
                            m_resizeCallback(m_width, m_height);
                        }
                        break;
#endif
                    }
                    if (newW != m_width || newH != m_height) {
                        Logger::info("Window pixel size changed: {}x{} -> {}x{}", m_width, m_height, newW, newH);
                        m_width = newW;
                        m_height = newH;
                        m_surfaceWidth = newW;
                        m_surfaceHeight = newH;
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
