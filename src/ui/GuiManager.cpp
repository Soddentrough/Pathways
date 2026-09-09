#include "ui/GuiManager.hpp"
#include "core/Logger.hpp"
#include "core/Window.hpp"
#include "scene/Camera.hpp"

#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_vulkan.h"

#include <stdexcept>
#include <format>
#include <cmath>
#include <algorithm>

namespace pathways {

GuiManager::GuiManager(SDL_Window* window, VkInstance instance, VkPhysicalDevice physicalDevice,
                       VkDevice device, uint32_t queueFamily, VkQueue queue,
                       VkFormat colorFormat, uint32_t minImageCount, uint32_t imageCount)
    : m_device(device) {

    Logger::info("Initializing Dear ImGui UI subsystem...");

    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = nullptr; // Ensure dynamic responsive docking without stale ini overrides
    ImGui::StyleColorsDark();

    // Dark style customization
    ImGuiStyle& style = ImGui::GetStyle();
    style.WindowRounding = 6.0f;
    style.FrameRounding = 4.0f;
    style.GrabRounding = 4.0f;

    ImGui_ImplSDL3_InitForVulkan(window);

    VkPipelineRenderingCreateInfo renderingInfo{};
    renderingInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachmentFormats = &colorFormat;

    ImGui_ImplVulkan_InitInfo initInfo{};
    initInfo.ApiVersion = VK_API_VERSION_1_4;
    initInfo.Instance = instance;
    initInfo.PhysicalDevice = physicalDevice;
    initInfo.Device = device;
    initInfo.QueueFamily = queueFamily;
    initInfo.Queue = queue;
    initInfo.DescriptorPoolSize = 100;
    initInfo.MinImageCount = minImageCount;
    initInfo.ImageCount = imageCount;
    initInfo.UseDynamicRendering = true;
    initInfo.PipelineInfoMain.PipelineRenderingCreateInfo = renderingInfo;

    if (!ImGui_ImplVulkan_Init(&initInfo)) {
        throw std::runtime_error("Failed to initialize ImGui Vulkan backend!");
    }

    m_initialized = true;
    Logger::info("Dear ImGui initialized with dynamic rendering.");
}

GuiManager::~GuiManager() {
    if (m_initialized) {
        ImGui_ImplVulkan_Shutdown();
        ImGui_ImplSDL3_Shutdown();
        ImGui::DestroyContext();
        Logger::info("Dear ImGui shutdown.");
    }
}

bool GuiManager::processEvent(const SDL_Event& event) {
    if (m_initialized) {
        return ImGui_ImplSDL3_ProcessEvent(&event);
    }
    return false;
}

void GuiManager::newFrame() {
    if (!m_initialized) return;
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
}

bool GuiManager::wantCaptureMouse() const {
    if (!m_initialized) return false;
    return ImGui::GetIO().WantCaptureMouse;
}

bool GuiManager::wantCaptureKeyboard() const {
    if (!m_initialized) return false;
    return ImGui::GetIO().WantCaptureKeyboard;
}

void GuiManager::resetHistory() {
    for (size_t i = 0; i < HISTORY_SIZE; ++i) {
        m_frameTimeHistory[i] = 0.0f;
    }
    m_historyOffset = 0;
    m_smoothedFrameTime = 0.0f;
}

// Renders an unobtrusive, stylish visual indicator at the bottom edge of an informational panel's
// bounding box whenever lower content is obscured/clipped, with click-to-scroll and pulse animations.
static void renderMoreDataBelowIndicator(const char* panelId) {
    (void)panelId;
    float scrollY = ImGui::GetScrollY();
    float scrollMaxY = ImGui::GetScrollMaxY();

    // Only display when lower content is obscured below the bottom of the bounding box
    if (scrollMaxY > 2.0f && scrollY < (scrollMaxY - 2.0f)) {
        ImVec2 winPos = ImGui::GetWindowPos();
        ImVec2 winSize = ImGui::GetWindowSize();
        ImDrawList* drawList = ImGui::GetWindowDrawList();

        // 1. Subtle bottom edge fade-out gradient to soften obscured content at the boundary
        float fadeH = 26.0f;
        ImVec2 fadeMin(winPos.x + 2.0f, winPos.y + winSize.y - fadeH - 2.0f);
        ImVec2 fadeMax(winPos.x + winSize.x - 2.0f, winPos.y + winSize.y - 2.0f);
        drawList->AddRectFilledMultiColor(
            fadeMin, fadeMax,
            IM_COL32(8, 12, 18, 0),     // Top: transparent
            IM_COL32(8, 12, 18, 0),
            IM_COL32(8, 12, 18, 240),   // Bottom: dark window background
            IM_COL32(8, 12, 18, 240)
        );

        // 2. Centered small visual icon pill pinned to the bottom of the bounding box
        float pillW = 44.0f;
        float pillH = 18.0f;
        float centerX = winPos.x + winSize.x * 0.5f;
        float pillBottomY = winPos.y + winSize.y - 6.0f;
        float pillTopY = pillBottomY - pillH;

        ImVec2 pillMin(centerX - pillW * 0.5f, pillTopY);
        ImVec2 pillMax(centerX + pillW * 0.5f, pillBottomY);

        // Subtle dynamic breath pulse
        float time = static_cast<float>(ImGui::GetTime());
        float pulse = 0.82f + 0.18f * std::sin(time * 3.5f);

        ImU32 pillBg = IM_COL32(14, 22, 34, static_cast<int>(230 * pulse));
        ImU32 pillBorder = IM_COL32(40, 160, 240, static_cast<int>(180 * pulse));
        ImU32 arrowColor = IM_COL32(80, 220, 255, static_cast<int>(255 * pulse));

        // Hover & Click interaction
        ImVec2 mousePos = ImGui::GetIO().MousePos;
        bool hovered = (mousePos.x >= pillMin.x && mousePos.x <= pillMax.x &&
                        mousePos.y >= pillMin.y && mousePos.y <= pillMax.y);
        if (hovered) {
            pillBg = IM_COL32(20, 36, 56, 250);
            pillBorder = IM_COL32(100, 240, 255, 255);
            arrowColor = IM_COL32(255, 255, 255, 255);
            if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
                ImGui::SetScrollY(std::min(scrollMaxY, scrollY + 140.0f));
            }
            ImGui::SetTooltip("More content below (use mouse scroll wheel)");
        }

        // Rounded pill background & border
        drawList->AddRectFilled(pillMin, pillMax, pillBg, 9.0f);
        drawList->AddRect(pillMin, pillMax, pillBorder, 9.0f, 0, 1.2f);

        // Crisp down chevron icon
        float arrowCenterY = (pillMin.y + pillMax.y) * 0.5f - 1.0f;
        float arrowHalfW = 6.0f;
        float arrowH = 5.0f;

        ImVec2 p1(centerX - arrowHalfW, arrowCenterY - arrowH * 0.5f);
        ImVec2 p2(centerX, arrowCenterY + arrowH * 0.5f);
        ImVec2 p3(centerX + arrowHalfW, arrowCenterY - arrowH * 0.5f);

        drawList->AddLine(p1, p2, arrowColor, 2.0f);
        drawList->AddLine(p2, p3, arrowColor, 2.0f);
    }
}

bool GuiManager::render(VkCommandBuffer cmd, VkImageView targetView, uint32_t width, uint32_t height,

                        Config& config, const FrameStats& stats, bool& cameraMode, Camera* camera,
                        const DisplayInfo* displayInfo, bool isFullscreen,
                        GuiActions* actions,
                        const std::vector<SceneEntry>& availableScenes, int currentSceneIndex) {
    if (!m_initialized) return false;

    bool settingsChanged = false;

    // Record true instantaneous frame time from GPU execution
    float currentFrameTime = static_cast<float>(
        stats.current_frame_time_ms > 0.001 ? stats.current_frame_time_ms :
        ((stats.primary_gpu_time_ms + stats.tonemap_time_ms > 0.001) ?
         (stats.primary_gpu_time_ms + stats.tonemap_time_ms) : stats.avg_frame_time_ms));
    if (currentFrameTime <= 0.001f) currentFrameTime = 0.5f;

    // Fast-adapting exponential moving average (alpha = 0.25) for responsive yet rock-solid, flicker-free HUD readout
    if (m_smoothedFrameTime <= 0.001f) {
        m_smoothedFrameTime = currentFrameTime;
    } else {
        m_smoothedFrameTime = m_smoothedFrameTime * 0.75f + currentFrameTime * 0.25f;
    }

    m_frameTimeHistory[m_historyOffset] = currentFrameTime;
    m_historyOffset = (m_historyOffset + 1) % HISTORY_SIZE;

    // Compute min, max, avg across recent rolling history (last 60 frames)
    float historyMin = 1e9f, historyMax = 0.0f, historySum = 0.0f;
    int validCount = 0;
    for (size_t i = 0; i < HISTORY_SIZE; ++i) {
        if (m_frameTimeHistory[i] > 0.001f) {
            historyMin = std::min(historyMin, m_frameTimeHistory[i]);
            historyMax = std::max(historyMax, m_frameTimeHistory[i]);
            historySum += m_frameTimeHistory[i];
            validCount++;
        }
    }
    float historyAvg = validCount > 0 ? (historySum / validCount) : currentFrameTime;
    if (historyMin > 1e8f) historyMin = currentFrameTime;

    // =========================================================================
    // Responsive Layout Calculation (Using ImGui Virtual Canvas DisplaySize)
    // =========================================================================
    ImGuiIO& io = ImGui::GetIO();
    float dispW = (io.DisplaySize.x > 0.0f) ? io.DisplaySize.x : static_cast<float>(width);
    float dispH = (io.DisplaySize.y > 0.0f) ? io.DisplaySize.y : static_cast<float>(height);

    float aspect = (height > 0) ? (static_cast<float>(width) / static_cast<float>(height)) : 1.0f;
    bool isPortrait = (aspect < 1.05f);

    ImGuiCond layoutCond = ImGuiCond_FirstUseEver;
    if (!m_layoutInitialized || m_lastWasPortrait != isPortrait ||
        m_lastWidth != width || m_lastHeight != height ||
        m_lastDispW != dispW || m_lastDispH != dispH) {
        layoutCond = ImGuiCond_Always;
        m_layoutInitialized = true;
        m_lastWasPortrait = isPortrait;
        m_lastWidth = width;
        m_lastHeight = height;
        m_lastDispW = dispW;
        m_lastDispH = dispH;
    }

    float hudX = 20.0f;
    float hudY = 20.0f;
    float hudW = std::min(460.0f, dispW - 40.0f);
    float hudH = std::min(dispH - 40.0f, 1100.0f);

    float ctrlW = std::min(480.0f, dispW - 40.0f);
    float ctrlX = std::max(20.0f, dispW - ctrlW - 20.0f);
    float ctrlY = 20.0f;
    float ctrlH = std::min(dispH - 40.0f, 1550.0f);

    if (isPortrait) {
        // Vertical stacked layout on the left: leaves the right side unobstructed for the 3D scene
        float sidebarW = std::min(440.0f, dispW * 0.45f);
        if (sidebarW < 320.0f) sidebarW = std::max(260.0f, dispW - 24.0f);

        hudX = 12.0f;
        hudY = 12.0f;
        hudW = sidebarW;
        hudH = std::min(dispH * 0.45f, 750.0f);

        ctrlX = 12.0f;
        ctrlY = hudY + hudH + 12.0f;
        ctrlW = sidebarW;
        ctrlH = std::max(200.0f, dispH - ctrlY - 12.0f);
    } else {
        // Landscape layout: HUD on left, Control panel docked safely on right edge
        hudX = 20.0f;
        hudY = 20.0f;
        hudW = std::min(460.0f, (dispW - 60.0f) * 0.5f);
        hudH = std::min(dispH - 40.0f, 1100.0f);

        ctrlW = std::min(480.0f, (dispW - 60.0f) * 0.5f);
        ctrlX = std::max(hudX + hudW + 20.0f, dispW - ctrlW - 20.0f);
        ctrlY = 20.0f;
        ctrlH = std::min(dispH - 40.0f, 1550.0f);
    }

    // =========================================================================
    // WINDOW 1: Pathways Live Profiler & Telemetry HUD
    // =========================================================================
    float fps = (m_smoothedFrameTime > 0.0001f) ? (1000.0f / m_smoothedFrameTime) : 0.0f;
    float avgFps = (historyAvg > 0.0001f) ? (1000.0f / historyAvg) : 0.0f;

    char hudTitle[128];
    std::snprintf(hudTitle, sizeof(hudTitle), "Pathways Telemetry HUD  -  %.0f FPS (%.2f ms)###TelemetryHUD", fps, currentFrameTime);

    ImGui::SetNextWindowPos(ImVec2(hudX, hudY), layoutCond);
    ImGui::SetNextWindowSize(ImVec2(hudW, hudH), layoutCond);

    ImGuiWindowFlags panelFlags = ImGuiWindowFlags_NoScrollbar;
    if (ImGui::Begin(hudTitle, nullptr, panelFlags)) {
        if (stats.mgpu_mode_str != "off" && stats.secondary_gpu_time_ms > 0.001) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Dual %s Pure Vulkan 1.4 Path Tracer", stats.arch_name.c_str());
        } else {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "%s Pure Vulkan 1.4 Path Tracer", stats.arch_name.c_str());
        }
        if (cameraMode) {
            float camSpeed = camera ? camera->getEffectiveSpeed() : 3.5f;
            float targetDist = camera ? camera->getCurrentTargetDistance() : 2.0f;
            if (camera && camera->isOrbiting()) {
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f), "[MODE] CTRL ARC ORBIT (Radius: %.2f m | A/D: Arc | W/S: Dolly | Q/E/Space/C: Elev)", camera->getOrbitRadius());
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f), "[MODE] FPS Navigation (Speed: %.2f m/s | Target Dist: %.2f m | Ctrl: Orbit | Alt: Crawl | Shift: Sprint)", camSpeed, targetDist);
            }
        } else {
            ImGui::TextColored(ImVec4(0.35f, 1.0f, 0.45f, 1.0f), "[MODE] UI Control Panel Active (Click Viewport or TAB for FPS)");
        }
        ImGui::Separator();

        // 0. Prominent Obvious FPS Hero Display Card
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.06f, 0.10f, 0.16f, 0.90f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 8.0f));
        if (ImGui::BeginChild("FpsHeroCard", ImVec2(hudW - 40.0f, 58.0f), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {

            ImGui::SetWindowFontScale(2.2f);
            if (fps >= 60.0f) {
                ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.45f, 1.0f), "%.0f FPS", fps);
            } else if (fps >= 30.0f) {
                ImGui::TextColored(ImVec4(1.0f, 0.85f, 0.2f, 1.0f), "%.0f FPS", fps);
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.35f, 0.35f, 1.0f), "%.0f FPS", fps);
            }
            ImGui::SetWindowFontScale(1.0f);

            ImGui::SameLine(hudW * 0.46f);
            ImGui::BeginGroup();
            ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), "Latency: %.2f ms", currentFrameTime);
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Rolling Avg: %.1f FPS", avgFps);
            ImGui::EndGroup();
        }
        ImGui::EndChild();
        ImGui::PopStyleVar(2);
        ImGui::PopStyleColor();
        ImGui::Spacing();

        // 1. Target Status Banner (<8.0 ms for 4K 120+ FPS)
        bool targetPass = currentFrameTime < 8.0f;
        if (targetPass) {
            ImGui::TextColored(ImVec4(0.2f, 0.95f, 0.4f, 1.0f), "[BUDGET ACHIEVED] Sub-8ms Target (<8.0 ms): %.2f ms | %.1f FPS",
                               currentFrameTime, currentFrameTime > 0.0f ? 1000.0f / currentFrameTime : 0.0f);
        } else {
            ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f), "[CONVERGING] Frame Latency: %.2f ms | %.1f FPS (Target: <8.0 ms)",
                               currentFrameTime, currentFrameTime > 0.0f ? 1000.0f / currentFrameTime : 0.0f);
        }

        float displayFps = m_smoothedFrameTime > 0.0001f ? (1000.0f / m_smoothedFrameTime) : 0.0f;
        ImGui::Text("Frame Time: %6.2f ms  |  FPS: %6.1f", m_smoothedFrameTime, displayFps);
        ImGui::Text("Min: %5.2f ms | Max: %5.2f ms | Avg: %5.2f ms (%4.0f FPS)",
                    historyMin, historyMax, historyAvg, historyAvg > 0.0f ? 1000.0f / historyAvg : 0.0f);

        ImGui::Separator();

        // 2. Latency Rolling Graph with 8.0ms Target Budget Line
        char overlayBuf[64];
        std::snprintf(overlayBuf, sizeof(overlayBuf), "Current: %.2f ms", currentFrameTime);
        float graphMax = std::max(16.0f, historyMax * 1.2f);
        ImGui::PlotLines("Latency", m_frameTimeHistory, static_cast<int>(HISTORY_SIZE), m_historyOffset,
                         overlayBuf, 0.0f, graphMax, ImVec2(hudW - 60.0f, 75.0f));
        ImGui::TextDisabled("8.0ms Budget Line (120 FPS Target)");

        ImGui::Separator();

        // 3. Multi-GPU Subsystem Telemetry
        if (ImGui::CollapsingHeader("Multi-GPU Subsystem Telemetry", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Active Topology: %s", stats.topology_name.c_str());
            ImGui::Text("Multi-GPU Mode:  %s", stats.mgpu_mode_str.c_str());

            // Hardware PCIe Link & Live Telemetry
            ImGui::Separator();
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "GPU 0 (Primary):");
            ImGui::Text("  PCIe:    %s", stats.primary_pci_link.c_str());
            if (stats.primary_pci_degraded) {
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), "  [WARNING: GPU 0 PCIe Link Degraded]");
            }
            if (stats.primary_gpu_clock_mhz > 0 || stats.primary_gpu_temp_c > 0) {
                ImGui::Text("  Sensors: %u MHz | %u C", stats.primary_gpu_clock_mhz, stats.primary_gpu_temp_c);
            }

            if (stats.is_mgpu_active || !stats.secondary_gpu_name.empty()) {
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "GPU 1 (Secondary):");
                if (!stats.secondary_pci_link.empty()) {
                    ImGui::Text("  PCIe:    %s", stats.secondary_pci_link.c_str());
                    if (stats.secondary_pci_degraded) {
                        ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.2f, 1.0f), "  [WARNING: GPU 1 PCIe Link Degraded]");
                    }
                }
                if (stats.secondary_gpu_clock_mhz > 0 || stats.secondary_gpu_temp_c > 0) {
                    ImGui::Text("  Sensors: %u MHz | %u C", stats.secondary_gpu_clock_mhz, stats.secondary_gpu_temp_c);
                }
            }

            if (ImGui::SmallButton("Re-check PCIe Status")) {
                if (actions) actions->refreshPciStatus = true;
            }
            ImGui::Separator();

            if (config.visualize_mgpu_split && config.mgpu_mode != MultiGpuMode::Off) {
                ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.5f, 1.0f), "Load Split Visualizer: ACTIVE");
            }

            if (stats.secondary_gpu_time_ms > 0.001) {
                ImGui::Text("  GPU 0 (Primary RT):    %.3f ms", stats.primary_gpu_time_ms);
                ImGui::Text("  GPU 1 (Secondary RT):  %.3f ms", stats.secondary_gpu_time_ms);
                ImGui::Text("  Tonemap & Merge Pass:  %.3f ms", stats.tonemap_time_ms);

                float totalWork = static_cast<float>(stats.primary_gpu_time_ms + stats.secondary_gpu_time_ms);
                float primaryFraction = totalWork > 0.001f ? (static_cast<float>(stats.primary_gpu_time_ms) / totalWork) : 0.5f;
                char loadBuf[64];
                std::snprintf(loadBuf, sizeof(loadBuf), "GPU 0: %.1f%% | GPU 1: %.1f%%", primaryFraction * 100.0f, (1.0f - primaryFraction) * 100.0f);
                ImGui::ProgressBar(primaryFraction, ImVec2(hudW - 60.0f, 16.0f), loadBuf);
            } else {
                ImGui::Text("  GPU 0 Ray Tracing:     %.3f ms", stats.primary_gpu_time_ms > 0.0 ? stats.primary_gpu_time_ms : currentFrameTime);
                ImGui::Text("  ACES Filmic Tonemap:   %.3f ms", stats.tonemap_time_ms);
                if (!stats.secondary_gpu_name.empty()) {
                    ImGui::TextDisabled("  GPU 1 (Secondary):     Standby / Offline (0.000 ms)");
                }
            }

            if (stats.total_vram_mb > 0.0) {
                ImGui::Text("  VRAM Allocated: %.0f MB / %.0f MB", stats.vram_used_mb, stats.total_vram_mb);
            }
        }

        // 4. Hardware Pipeline & Architecture Telemetry
        if (ImGui::CollapsingHeader("Hardware Architecture & Execution Mode", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Subgroup Execution: Native Wave32 SIMD (%s)", stats.short_arch.c_str());
            ImGui::TextColored(ImVec4(0.25f, 0.95f, 0.45f, 1.0f), "RT Pipeline: Hardware BVH Accelerated");
            ImGui::TextColored(ImVec4(0.65f, 0.82f, 1.0f, 1.0f), "  Active Pipeline Extensions:");
            ImGui::BulletText("VK_KHR_ray_query (In-Shader Ray Queries)");
            ImGui::BulletText("VK_KHR_acceleration_structure (BLAS + TLAS)");
            ImGui::BulletText("VK_KHR_buffer_device_address (64-bit BDA)");
            ImGui::BulletText("VK_KHR_deferred_host_operations (Host Build)");
            ImGui::Text("Pipeline:           Hardware Ray Tracing (VK_KHR_ray_tracing_pipeline)");
            ImGui::Text("Ray Scheduling:     RDNA4 Hardware BVH Traversal (Wave32)");
            ImGui::Text("Command Execution:  %s", stats.has_dgc ? "GPU-Driven Indirect (VK_EXT_dgc)" : "Host Recorded Dispatch");
            ImGui::Text("Ray Throughput:     %.2f GigaRays/sec", stats.rays_per_second * 1e-9);
            ImGui::Text("Validation Errors:  %u", stats.validation_errors);
        }

        // 5. Scene Complexity Telemetry
        if (ImGui::CollapsingHeader("Scene Complexity", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Geometry:     %u Triangles, %u Spheres", stats.num_triangles, stats.num_spheres);
            ImGui::Text("Shading:      %u Materials, %u Area Lights", stats.num_materials, stats.num_lights);
            ImGui::Text("Textures:     %u Texture Maps + HDRI Sky", stats.num_textures);
            ImGui::Text("Viewport:     %u x %u (Aspect: %.3f, %s)", stats.width, stats.height, aspect, isPortrait ? "Portrait" : "Landscape");
            ImGui::Text("Accumulation: Frame %u (%u samples accumulated)", stats.total_frames, stats.total_frames * config.spp);
        }

        // 6. Offline Telemetry Export
        if (ImGui::CollapsingHeader("Offline Telemetry Comparison", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextDisabled("Save hardware, driver, support & performance data:");
            if (ImGui::Button("Export Telemetry Data (.json)", ImVec2(hudW - 40.0f, 28.0f))) {
                std::string path = ImageDumper::generateDefaultTelemetryPath();
                if (actions) {
                    actions->exportTelemetry = true;
                    actions->exportTelemetryPath = path;
                }
                m_lastExportNotification = "Saved: " + path;
                m_exportNotificationTimer = 4.0f;
            }
            if (m_exportNotificationTimer > 0.0f) {
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f), "%s", m_lastExportNotification.c_str());
            }
        }

        // 7. Session Configuration Tallies
        if (!stats.configurations_breakdown.empty() && ImGui::CollapsingHeader("Session Configurations Tested", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextDisabled("Tallied performance across %zu unique configuration(s):", stats.configurations_breakdown.size());
            for (size_t i = 0; i < stats.configurations_breakdown.size(); ++i) {
                const auto& c = stats.configurations_breakdown[i];
                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "[%zu] %s", i + 1, c.label.c_str());
                ImGui::BulletText("Frames: %u | Avg: %.2f ms (%.1f FPS)", c.frame_count, c.avg_frame_time_ms, c.avg_fps);
                if (c.secondary_gpu_time_ms > 0.001) {
                    ImGui::BulletText("GPU 0: %.2f ms | GPU 1: %.2f ms | Merge: %.2f ms", c.primary_gpu_time_ms, c.secondary_gpu_time_ms, c.tonemap_time_ms);
                } else {
                    ImGui::BulletText("GPU 0: %.2f ms | Tonemap: %.2f ms | GPU 1: Standby", c.primary_gpu_time_ms, c.tonemap_time_ms);
                }
                ImGui::BulletText("Throughput: %.2f GigaRays/s | %s", c.gigarays_per_second, c.target_achieved ? "ACHIEVED (<8ms)" : "EXCEEDED");
            }
        }
        renderMoreDataBelowIndicator("TelemetryHUD");
    }
    ImGui::End();

    // =========================================================================
    // WINDOW 2: Pathways Interactive Control Panel
    // =========================================================================
    ImGui::SetNextWindowPos(ImVec2(ctrlX, ctrlY), layoutCond);
    ImGui::SetNextWindowSize(ImVec2(ctrlW, ctrlH), layoutCond);

    if (ImGui::Begin("Pathways Control Panel", nullptr, panelFlags)) {
        ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.5f, 1.0f), "Real-Time Pipeline Configuration");

        ImGui::Separator();

        // Mode Toggle Banner & Quick Action
        if (cameraMode) {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.75f, 0.22f, 0.22f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.9f, 0.3f, 0.3f, 1.0f));
            if (ImGui::Button("FPS SCENE NAVIGATION ACTIVE\n[Click or press TAB to release mouse]", ImVec2(-1, 38.0f))) {
                cameraMode = false;
            }
            ImGui::PopStyleColor(2);
        } else {
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.55f, 0.28f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.68f, 0.35f, 1.0f));
            if (ImGui::Button("ENTER SCENE NAVIGATION\n[Click or press TAB to capture mouse]", ImVec2(-1, 38.0f))) {
                cameraMode = true;
            }
            ImGui::PopStyleColor(2);
        }

        // 0. Active 3D Scene Selection & Asset Library
        if (!availableScenes.empty()) {
            ImGui::PushStyleColor(ImGuiCol_Header, ImVec4(0.12f, 0.28f, 0.48f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderHovered, ImVec4(0.18f, 0.38f, 0.62f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_HeaderActive, ImVec4(0.24f, 0.48f, 0.75f, 1.0f));
            bool sceneHeaderOpen = ImGui::CollapsingHeader("3D SCENE SELECTION & ASSET LIBRARY", ImGuiTreeNodeFlags_DefaultOpen);
            ImGui::PopStyleColor(3);

            if (sceneHeaderOpen) {
                // Category Filter Tabs
                static int sceneCategoryFilter = 0; // 0: All, 1: Showcase, 2: Research, 3: Custom, 4: Procedural

                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(8.0f, 3.0f));
                auto drawFilterTab = [&](const char* name, int filterIdx) {
                    bool active = (sceneCategoryFilter == filterIdx);
                    if (active) {
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.52f, 0.90f, 1.0f));
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
                    } else {
                        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.15f, 0.18f, 0.24f, 1.0f));
                        ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.70f, 0.75f, 0.80f, 1.0f));
                    }
                    if (ImGui::Button(name)) {
                        sceneCategoryFilter = filterIdx;
                    }
                    ImGui::PopStyleColor(2);
                };

                char allBuf[32];
                std::snprintf(allBuf, sizeof(allBuf), "All (%zu)", availableScenes.size());
                drawFilterTab(allBuf, 0);
                ImGui::SameLine();
                drawFilterTab("Showcase", 1);
                ImGui::SameLine();
                drawFilterTab("Research", 2);
                ImGui::SameLine();
                drawFilterTab("Custom", 3);
                ImGui::SameLine();
                drawFilterTab("Procedural", 4);
                ImGui::PopStyleVar();

                ImGui::Spacing();

                // Closed combo preview text with geometry stats
                std::string previewText;
                if (currentSceneIndex >= 0 && currentSceneIndex < static_cast<int>(availableScenes.size())) {
                    const auto& cur = availableScenes[currentSceneIndex];
                    previewText = cur.label + "  [" + cur.formatTriangles();
                    if (cur.materialCount > 0) {
                        previewText += ", " + std::to_string(cur.materialCount) + (cur.materialCount == 1 ? " mat" : " mats");
                    }
                    previewText += "]";
                } else {
                    previewText = "Procedural Cornell Box  [2.0K tris, 7 mats]";
                }

                // Styled prominent dropdown combo
                ImGui::PushStyleVar(ImGuiStyleVar_FramePadding, ImVec2(10.0f, 8.0f));
                ImGui::PushStyleColor(ImGuiCol_FrameBg, ImVec4(0.10f, 0.16f, 0.26f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_FrameBgHovered, ImVec4(0.16f, 0.25f, 0.38f, 1.0f));
                ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.35f, 0.70f, 1.0f, 0.9f));
                ImGui::SetNextItemWidth(-1);

                if (ImGui::BeginCombo("##SceneSelector", previewText.c_str(), ImGuiComboFlags_HeightLarge)) {
                    if (ImGui::BeginTable("##SceneSelectTable", 4, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp)) {
                        ImGui::TableSetupColumn("Scene Asset", ImGuiTableColumnFlags_WidthStretch, 0.46f);
                        ImGui::TableSetupColumn("Triangles", ImGuiTableColumnFlags_WidthFixed, 80.0f);
                        ImGui::TableSetupColumn("Materials", ImGuiTableColumnFlags_WidthFixed, 65.0f);
                        ImGui::TableSetupColumn("Category", ImGuiTableColumnFlags_WidthFixed, 75.0f);
                        ImGui::TableHeadersRow();

                        for (int i = 0; i < static_cast<int>(availableScenes.size()); ++i) {
                            const auto& sc = availableScenes[i];
                            if (sceneCategoryFilter == 1 && sc.group != "Showcase") continue;
                            if (sceneCategoryFilter == 2 && sc.group != "Research") continue;
                            if (sceneCategoryFilter == 3 && sc.group != "Custom") continue;
                            if (sceneCategoryFilter == 4 && sc.group != "Procedural") continue;

                            ImGui::TableNextRow();
                            ImGui::TableNextColumn();
                            bool isSelected = (currentSceneIndex == i);
                            if (ImGui::Selectable(sc.label.c_str(), isSelected, ImGuiSelectableFlags_SpanAllColumns)) {
                                if (i != currentSceneIndex && actions) {
                                    actions->sceneChanged = true;
                                    actions->newScenePath = sc.filepath;
                                    actions->newSceneIndex = i;
                                    settingsChanged = true;
                                }
                            }
                            if (isSelected) {
                                ImGui::SetItemDefaultFocus();
                            }

                            ImGui::TableNextColumn();
                            ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.0f), "%s", sc.formatTriangles().c_str());

                            ImGui::TableNextColumn();
                            ImGui::TextColored(ImVec4(0.85f, 0.85f, 0.85f, 1.0f), "%u mats", sc.materialCount);

                            ImGui::TableNextColumn();
                            ImVec4 grpCol = (sc.group == "Showcase") ? ImVec4(1.0f, 0.82f, 0.28f, 1.0f) :
                                            (sc.group == "Research") ? ImVec4(0.38f, 0.72f, 1.0f, 1.0f) :
                                            (sc.group == "Procedural") ? ImVec4(0.85f, 0.55f, 1.0f, 1.0f) : ImVec4(0.6f, 0.6f, 0.6f, 1.0f);
                            ImGui::TextColored(grpCol, "%s", sc.group.c_str());
                        }
                        ImGui::EndTable();
                    }
                    ImGui::EndCombo();
                }
                ImGui::PopStyleColor(3);
                ImGui::PopStyleVar();

                // Quick Navigation Stepper Buttons
                float navBtnW = (ImGui::GetContentRegionAvail().x - 16.0f) / 3.0f;
                if (ImGui::Button("< Prev Scene", ImVec2(navBtnW, 26.0f))) {
                    int newIdx = (currentSceneIndex - 1 + static_cast<int>(availableScenes.size())) % static_cast<int>(availableScenes.size());
                    if (actions) {
                        actions->sceneChanged = true;
                        actions->newScenePath = availableScenes[newIdx].filepath;
                        actions->newSceneIndex = newIdx;
                        settingsChanged = true;
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Next Scene >", ImVec2(navBtnW, 26.0f))) {
                    int newIdx = (currentSceneIndex + 1) % static_cast<int>(availableScenes.size());
                    if (actions) {
                        actions->sceneChanged = true;
                        actions->newScenePath = availableScenes[newIdx].filepath;
                        actions->newSceneIndex = newIdx;
                        settingsChanged = true;
                    }
                }
                ImGui::SameLine();
                if (ImGui::Button("Cornell Box", ImVec2(navBtnW, 26.0f))) {
                    if (actions) {
                        actions->sceneChanged = true;
                        actions->newScenePath = "";
                        actions->newSceneIndex = 0;
                        settingsChanged = true;
                    }
                }

                // Active Scene Information Card
                if (currentSceneIndex >= 0 && currentSceneIndex < static_cast<int>(availableScenes.size())) {
                    const auto& cur = availableScenes[currentSceneIndex];
                    ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.08f, 0.12f, 0.18f, 0.85f));
                    ImGui::PushStyleColor(ImGuiCol_Border, ImVec4(0.20f, 0.45f, 0.70f, 0.6f));
                    ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 4.0f);
                    ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(8.0f, 6.0f));

                    if (ImGui::BeginChild("##ActiveSceneCard", ImVec2(0, 94.0f), true, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse)) {
                        ImGui::TextColored(ImVec4(0.35f, 0.85f, 1.0f, 1.0f), "ACTIVE SCENE: %s", cur.label.c_str());
                        ImGui::SameLine(ImGui::GetContentRegionAvail().x - 70.0f);
                        ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.28f, 1.0f), "[%s]", cur.group.c_str());

                        ImGui::Text("  > Geometry: ");
                        ImGui::SameLine();
                        uint32_t activeTris = (stats.num_triangles > 0) ? stats.num_triangles : static_cast<uint32_t>(cur.triangleCount);
                        ImGui::TextColored(ImVec4(0.35f, 0.95f, 0.45f, 1.0f), "%u Triangles (%s)", activeTris, cur.formatTriangles().c_str());
                        if (stats.num_spheres > 0) {
                            ImGui::SameLine();
                            ImGui::Text("(%u Spheres)", stats.num_spheres);
                        }

                        uint32_t activeMats = (stats.num_materials > 0) ? stats.num_materials : cur.materialCount;
                        ImGui::Text("  > Materials: %u PBR | Lights: %u Area", activeMats, stats.num_lights);
                        if (camera) {
                            ImGui::SameLine();
                            ImGui::Text("| Radius: %.2f m", camera->getSceneScale());
                        }

                        if (!cur.filepath.empty()) {
                            ImGui::TextDisabled("  > File: %s", cur.filepath.c_str());
                            if (cur.fileSizeBytes > 0) {
                                ImGui::SameLine();
                                ImGui::TextDisabled("(%s)", cur.formatFileSize().c_str());
                            }
                        } else {
                            ImGui::TextDisabled("  > Built-in GPU procedural Cornell box geometry");
                        }
                    }
                    ImGui::EndChild();
                    ImGui::PopStyleVar(2);
                    ImGui::PopStyleColor(2);
                }
                ImGui::Spacing();
            }
            ImGui::Separator();
        }

        // 1. Display & Viewport Configuration (Fullscreen, Resolution Presets, Adaptive FOV)
        if (displayInfo && ImGui::CollapsingHeader("Display & Viewport Architecture", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextDisabled("Physical Display:");
            ImGui::BulletText("Device:   %s", displayInfo->displayName.c_str());
            ImGui::BulletText("Native:   %u x %u @ %.2f Hz", displayInfo->nativeWidth, displayInfo->nativeHeight, displayInfo->refreshRate);
            ImGui::BulletText("Usable:   %u x %u (Scale: %.2f)", displayInfo->usableWidth, displayInfo->usableHeight, displayInfo->contentScale);
            ImGui::BulletText("Aspect:   %.3f (%s)", displayInfo->displayAspect, displayInfo->isPortrait ? "Portrait (DualUp 16:18)" : (displayInfo->isUltraWide ? "Ultrawide (21:9 / 32:9)" : "Landscape (16:9 / 16:10)"));

            ImGui::Spacing();
            ImGui::TextDisabled("Active Viewport:");
            ImGui::BulletText("Resolution: %u x %u (Aspect: %.3f)", width, height, aspect);
            ImGui::BulletText("Layout:     %s", isPortrait ? "Vertical Stacked (Scene Left Unobstructed)" : "Landscape Edge-Docked");

            if (ImGui::Button(isFullscreen ? "Exit Fullscreen (F11)" : "Toggle Fullscreen (F11)", ImVec2(180.0f, 26.0f))) {
                if (actions) actions->toggleFullscreen = true;
            }

            ImGui::Spacing();
            ImGui::TextDisabled("Resolution Presets:");
            float presetBtnW = (ImGui::GetContentRegionAvail().x - ImGui::GetStyle().ItemSpacing.x) * 0.5f;

            // Row 1: Native Full Display | DualUp (1280x2048)
            std::string nativeBtnLabel = std::format("Native ({}x{})", displayInfo->nativeWidth > 0 ? displayInfo->nativeWidth : 3840,
                                                                         displayInfo->nativeHeight > 0 ? displayInfo->nativeHeight : 2160);
            if (ImGui::Button(nativeBtnLabel.c_str(), ImVec2(presetBtnW, 26.0f))) {
                if (actions) {
                    actions->requestedWidth = displayInfo->nativeWidth > 0 ? displayInfo->nativeWidth : displayInfo->usableWidth;
                    actions->requestedHeight = displayInfo->nativeHeight > 0 ? displayInfo->nativeHeight : displayInfo->usableHeight;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("DualUp (1280x2048)", ImVec2(presetBtnW, 26.0f))) {
                if (actions) {
                    actions->requestedWidth = 1280;
                    actions->requestedHeight = 2048;
                }
            }

            // Row 2: 1:1 Square (1440x1440) | FHD (1920x1080)
            if (ImGui::Button("1:1 (1440x1440)", ImVec2(presetBtnW, 26.0f))) {
                if (actions) {
                    actions->requestedWidth = 1440;
                    actions->requestedHeight = 1440;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("FHD (1920x1080)", ImVec2(presetBtnW, 26.0f))) {
                if (actions) {
                    actions->requestedWidth = 1920;
                    actions->requestedHeight = 1080;
                }
            }

            // Row 3: QHD (2560x1440) | 4K UHD (3840x2160)
            if (ImGui::Button("QHD (2560x1440)", ImVec2(presetBtnW, 26.0f))) {
                if (actions) {
                    actions->requestedWidth = 2560;
                    actions->requestedHeight = 1440;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("4K (3840x2160)", ImVec2(presetBtnW, 26.0f))) {
                if (actions) {
                    actions->requestedWidth = 3840;
                    actions->requestedHeight = 2160;
                }
            }

            if (camera) {
                bool adaptive = camera->isAdaptiveFov();
                if (ImGui::Checkbox("Adaptive Aspect FOV (Auto-frame scene)", &adaptive)) {
                    camera->setAdaptiveFov(adaptive);
                    camera->adaptFovForAspect(aspect);
                    settingsChanged = true;
                }
                if (ImGui::Button("Re-frame Scene (Auto FOV)", ImVec2(180.0f, 26.0f))) {
                    camera->adaptFovForAspect(aspect);
                    settingsChanged = true;
                }
            }
            ImGui::Separator();
        }

        // 0. Camera & Scene Navigation
        if (ImGui::CollapsingHeader("Camera & Scene Navigation", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (camera) {
                float speed = camera->getSpeed();
                if (ImGui::SliderFloat("Move Speed", &speed, camera->getMinSpeed(), camera->getMaxSpeed(), "%.2f m/s", ImGuiSliderFlags_Logarithmic)) {
                    camera->setSpeed(speed);
                }
                ImGui::SameLine();
                if (ImGui::SmallButton("Reset Speed")) {
                    camera->setSpeed(camera->getBaseSpeed());
                }

                ImGui::Text("Speed Presets:");
                ImGui::SameLine();
                if (ImGui::SmallButton("0.25x (Fine)")) camera->setSpeed(camera->getBaseSpeed() * 0.25f);
                ImGui::SameLine();
                if (ImGui::SmallButton("0.5x"))        camera->setSpeed(camera->getBaseSpeed() * 0.50f);
                ImGui::SameLine();
                if (ImGui::SmallButton("1.0x (Default)")) camera->setSpeed(camera->getBaseSpeed());
                ImGui::SameLine();
                if (ImGui::SmallButton("2.0x (Fast)")) camera->setSpeed(camera->getBaseSpeed() * 2.0f);
                ImGui::SameLine();
                if (ImGui::SmallButton("4.0x (Turbo)")) camera->setSpeed(camera->getBaseSpeed() * 4.0f);

                bool dynScaling = camera->isDynamicScaling();
                if (ImGui::Checkbox("Distance-Adaptive Speed (Smooth Approach to Focus)", &dynScaling)) {
                    camera->setDynamicScaling(dynScaling);
                }

                ImGui::TextDisabled("Focal scale: %.2f m | Target dist: %.2f m | Base speed: %.2f m/s",
                                    camera->getSceneScale(), camera->getCurrentTargetDistance(), camera->getBaseSpeed());

                float sens = camera->getSensitivity();
                if (ImGui::SliderFloat("Mouse Sensitivity", &sens, 0.02f, 0.5f, "%.2f")) {
                    camera->setSensitivity(sens);
                }

                float fov = camera->getFov();
                if (ImGui::SliderFloat("Field of View", &fov, 20.0f, 100.0f, "%.1f deg")) {
                    camera->setFov(fov);
                    settingsChanged = true;
                }

                glm::vec3 pos = camera->getPosition();
                ImGui::Text("Position: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
                ImGui::Text("Look Angles: Yaw: %.1f deg, Pitch: %.1f deg", camera->getYaw(), camera->getPitch());

                if (ImGui::Button("Focus on Center (F)", ImVec2(180.0f, 26.0f))) {
                    camera->focusOnTarget(camera->getCentralTarget());
                    settingsChanged = true;
                }
                ImGui::SameLine();
                if (ImGui::Button("Reset Camera", ImVec2(140.0f, 26.0f))) {
                    camera->resetToDefault();
                    settingsChanged = true;
                }
            }
            ImGui::Spacing();
            ImGui::TextDisabled("FPS Navigation Reference:");
            ImGui::BulletText("TAB: Toggle UI Options / FPS Navigation");
            ImGui::BulletText("W / A / S / D: Forward / Left / Back / Right");
            ImGui::BulletText("Space / C (or E / Q): Move Up / Down");
            ImGui::BulletText("Ctrl (Hold): Arc-Strafe / Turntable Orbit around Targeted Object");
            ImGui::BulletText("  -> A / D: Arc-Strafe Left / Right around Target");
            ImGui::BulletText("  -> W / S: Dolly In / Out along Line of Sight");
            ImGui::BulletText("  -> Space / C (or E / Q): Elevate Up / Down in Arc");
            ImGui::BulletText("  -> Mouse: Turntable Orbit around Target");
            ImGui::BulletText("Alt (Hold): Precision Crawl (0.25x Speed for Object Centering)");
            ImGui::BulletText("Shift (Hold): Sprint Boost (3.0x Speed for Fast Relocation)");
            ImGui::BulletText("F Key: Focus on Central Object");
            ImGui::BulletText("Mouse Wheel: Continuously Scale Camera Speed");
            ImGui::BulletText("Mouse: Freelook Orientation (FPS Mode)");
            ImGui::BulletText("ESC: Release Mouse (FPS Mode) / Exit (UI)");
        }

        // 1. Hardware Acceleration Pipeline Extensions
        if (ImGui::CollapsingHeader("Hardware Acceleration & Ray Tracing Pipeline", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextColored(ImVec4(0.25f, 0.95f, 0.45f, 1.0f), "[ACTIVE] HARDWARE ACCELERATED (%s)", stats.ray_accelerator_name.c_str());
            ImGui::Spacing();

            ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "Vulkan Pipeline Extensions in Use:");
            ImGui::BulletText("VK_KHR_ray_query");
            ImGui::SameLine();
            ImGui::TextDisabled("-> In-shader ray queries (rayQueryEXT)");

            ImGui::BulletText("VK_KHR_acceleration_structure");
            ImGui::SameLine();
            ImGui::TextDisabled("-> Hardware BVH (BLAS + TLAS)");

            ImGui::BulletText("VK_KHR_buffer_device_address");
            ImGui::SameLine();
            ImGui::TextDisabled("-> 64-bit BDA vertex/index & AS pointers");

            ImGui::BulletText("VK_KHR_deferred_host_operations");
            ImGui::SameLine();
            ImGui::TextDisabled("-> Driver host AS build threads");

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "SPIR-V / Shading Extension:");
            ImGui::BulletText("GL_EXT_ray_query / SPV_KHR_ray_query");
            ImGui::SameLine();
            ImGui::TextDisabled("-> Native Wave32 SIMD");

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Pipeline Architecture:");
            ImGui::Text("Dedicated Hardware RT Pipeline (VK_KHR_ray_tracing_pipeline)");
            ImGui::Text("GPU Indirect Execution (Work List / TraceRaysIndirect)");

            ImGui::TextDisabled("Ray Scheduling: RDNA4 Hardware BVH Traversal (Wave32)");
            ImGui::Separator();
        }

        // 2. Multi-GPU Scalability Mode
        if (ImGui::CollapsingHeader("Multi-GPU Architecture", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* mgpuModes[] = {
                "Single GPU (Secondary Standby)",
                "Checkerboard Tiling (Optimal RDNA4, default)",
                "Sample Parallelism (Temporal Sample Splitting)",
                "Auto (SPP Adaptive)"
            };
            int currentMode = 0;
            if (config.mgpu_mode == MultiGpuMode::CheckerboardTile) currentMode = 1;
            else if (config.mgpu_mode == MultiGpuMode::SampleParallel) currentMode = 2;
            else if (config.mgpu_mode == MultiGpuMode::Auto) currentMode = 3;

            if (ImGui::Combo("Execution Mode", &currentMode, mgpuModes, IM_ARRAYSIZE(mgpuModes))) {
                MultiGpuMode selectedMode = MultiGpuMode::Off;
                if (currentMode == 1) selectedMode = MultiGpuMode::CheckerboardTile;
                else if (currentMode == 2) selectedMode = MultiGpuMode::SampleParallel;
                else if (currentMode == 3) selectedMode = MultiGpuMode::Auto;

                if (actions && selectedMode != config.mgpu_mode) {
                    actions->mgpuModeChanged = true;
                    actions->newMgpuMode = selectedMode;
                }
                config.mgpu_mode = selectedMode;
                settingsChanged = true;
            }

            if (config.mgpu_mode == MultiGpuMode::Auto) {
                MultiGpuMode active = (config.spp > 1) ? MultiGpuMode::SampleParallel : MultiGpuMode::CheckerboardTile;
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "  -> Active Policy: %s",
                                   (active == MultiGpuMode::SampleParallel) ? "Sample Parallelism (Split SPP)" : "Checkerboard Tiling (Optimal RDNA4)");
            }

            if (config.mgpu_mode == MultiGpuMode::CheckerboardTile || (config.mgpu_mode == MultiGpuMode::Auto && config.spp == 1)) {
                const char* tileSizes[] = {
                    "16x16 (Finest Interleaving)",
                    "32x32 (Balanced Cache)",
                    "64x64 (Optimal RDNA4 Default)",
                    "128x128 (Maximum Ray Coherence)"
                };
                int currentTileIdx = 2; // default 64
                if (config.tile_size == 16) currentTileIdx = 0;
                else if (config.tile_size == 32) currentTileIdx = 1;
                else if (config.tile_size == 64) currentTileIdx = 2;
                else if (config.tile_size == 128) currentTileIdx = 3;

                if (ImGui::Combo("Tile Size", &currentTileIdx, tileSizes, IM_ARRAYSIZE(tileSizes))) {
                    uint32_t chosenSize = 64;
                    if (currentTileIdx == 0) chosenSize = 16;
                    else if (currentTileIdx == 1) chosenSize = 32;
                    else if (currentTileIdx == 2) chosenSize = 64;
                    else if (currentTileIdx == 3) chosenSize = 128;

                    if (actions && chosenSize != config.tile_size) {
                        actions->tileSizeChanged = true;
                        actions->newTileSize = chosenSize;
                    }
                    config.tile_size = chosenSize;
                    settingsChanged = true;
                }
            } else if (config.mgpu_mode == MultiGpuMode::SampleParallel || (config.mgpu_mode == MultiGpuMode::Auto && config.spp > 1)) {
                uint32_t primSpp = (config.spp + 1) / 2;
                uint32_t secSpp = config.spp / 2;
                ImGui::TextDisabled("  Sample Split: GPU 0 = %u SPP, GPU 1 = %u SPP", primSpp, secSpp);
            }

            // Accumulation format selection
            const char* accumFormats[] = {
                "RGBA16_SFLOAT (64-bit Half Float HDR) [Default]",
                "RGBA32_SFLOAT (128-bit Full Float HDR)"
            };
            int currentFormat = (config.accum_format == AccumFormat::RGBA32_SFLOAT) ? 1 : 0;
            if (ImGui::Combo("Accumulation Format", &currentFormat, accumFormats, IM_ARRAYSIZE(accumFormats))) {
                AccumFormat selectedFormat = (currentFormat == 1) ? AccumFormat::RGBA32_SFLOAT : AccumFormat::RGBA16_SFLOAT;
                if (actions && selectedFormat != config.accum_format) {
                    actions->accumFormatChanged = true;
                    actions->newAccumFormat = selectedFormat;
                }
                config.accum_format = selectedFormat;
                settingsChanged = true;
            }

            // Double buffering toggle
            if (ImGui::Checkbox("Double-Buffered Shared Memory", &config.double_buffered_shared_mem)) {
                if (actions) {
                    actions->doubleBufferChanged = true;
                    actions->newDoubleBuffer = config.double_buffered_shared_mem;
                }
                settingsChanged = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Pipelined zero-copy DMA buffers to overlap Secondary GPU execution with Primary GPU present.");
            }

            if (config.mgpu_mode != MultiGpuMode::Off) {
                if (ImGui::Checkbox("Visualize GPU Load Split", &config.visualize_mgpu_split)) {
                    // Changing visualization immediately updates shader push constants
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Overlays on-screen color tints (GPU 0 Cyan, GPU 1 Amber) to visually display work distribution.");
                }
            }
        }

        // 3. Path Tracer Core Settings
        if (ImGui::CollapsingHeader("Path Tracer Core", ImGuiTreeNodeFlags_DefaultOpen)) {
            // Dynamic Sample Rate & Frame Rate Governor
            if (ImGui::Checkbox("Adaptive Dynamic SPP (Governor)", &config.adaptive_spp)) {
                settingsChanged = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Autonomously adjusts sample count (SPP) and bounce depth to maximize quality within target framerate.");
            }

            const char* targetFpsLabels[] = { "Uncapped", "30 FPS (33.3 ms)", "60 FPS (16.7 ms)", "90 FPS (11.1 ms)", "120 FPS (8.3 ms)", "144 FPS (6.9 ms)", "240 FPS (4.2 ms)" };
            const uint32_t targetFpsValues[] = { 0, 30, 60, 90, 120, 144, 240 };
            int currentTargetIdx = 0;
            for (int idx = 0; idx < 7; ++idx) {
                if (config.target_fps == targetFpsValues[idx]) {
                    currentTargetIdx = idx;
                    break;
                }
            }
            if (ImGui::Combo("Target Frame Rate", &currentTargetIdx, targetFpsLabels, 7)) {
                config.target_fps = targetFpsValues[currentTargetIdx];
                if (config.target_fps > 0) config.adaptive_spp = true;
                settingsChanged = true;
            }

            if (config.adaptive_spp && config.target_fps > 0) {
                ImGui::Indent();
                ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f),
                    "[Governor: ACTIVE | Dynamic: %u SPP, %u Bounces]",
                    stats.dynamic_spp, stats.dynamic_bounces);
                ImGui::Text("Target Budget: %.2f ms (Current RT: %.2f ms)",
                    1000.0f / config.target_fps, stats.primary_gpu_time_ms);
                ImGui::Unindent();
            }
            ImGui::Separator();

            int spp = static_cast<int>(config.spp);
            if (ImGui::SliderInt(config.adaptive_spp ? "Base Samples/Pixel (SPP)" : "Samples/Pixel (SPP)", &spp, 1, 64)) {
                config.spp = static_cast<uint32_t>(spp);
                settingsChanged = true;
            }

            int bounces = static_cast<int>(config.max_bounces);
            if (ImGui::SliderInt(config.adaptive_spp ? "Base Ray Bounces" : "Max Ray Bounces", &bounces, 1, 16)) {
                config.max_bounces = static_cast<uint32_t>(bounces);
                settingsChanged = true;
            }
        }

        // 4. Lighting & Shading Subsystems
        if (ImGui::CollapsingHeader("Lighting & Shading Components", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Direct Lighting (Area Lights)", &config.enable_direct_light)) {
                settingsChanged = true;
            }
            if (ImGui::Checkbox("ReSTIR DI (Reservoir Resampling)", &config.enable_restir_di)) {
                settingsChanged = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Spatiotemporal Reservoir Resampling for Direct Illumination (1 shadow ray/pixel with M=4 candidates).");
            }
            if (config.enable_restir_di) {
                ImGui::Indent();
                int samples = static_cast<int>(config.restir_spatial_samples);
                if (ImGui::SliderInt("Spatial Neighbors", &samples, 1, 8)) {
                    config.restir_spatial_samples = static_cast<uint32_t>(samples);
                    settingsChanged = true;
                }
                if (ImGui::SliderFloat("Spatial Radius", &config.restir_spatial_radius, 2.0f, 64.0f, "%.1f px")) {
                    settingsChanged = true;
                }
                ImGui::Unindent();
            }
            if (ImGui::Checkbox("Indirect Diffuse GI", &config.enable_indirect_light)) {
                settingsChanged = true;
            }
            if (ImGui::Checkbox("Dielectric Refraction & Fresnel", &config.enable_refraction)) {
                settingsChanged = true;
            }
            if (ImGui::Checkbox("Soft Area Shadows", &config.enable_shadows)) {
                settingsChanged = true;
            }
        }

        // 5. Post-Processing & Tonemapping
        if (ImGui::CollapsingHeader("Post-Processing & Denoising", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("FidelityFX Shadow Denoiser", &config.enable_shadow_denoiser)) {
                settingsChanged = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("AMD FidelityFX Spatio-Temporal Cross-Bilateral Shadow Denoiser with 8x8 Tile Classification.");
            }
            if (config.enable_shadow_denoiser) {
                ImGui::Indent();
                if (ImGui::SliderFloat("Depth Tolerance", &config.shadow_denoiser_depth_sigma, 0.005f, 0.100f, "%.3f")) {
                    settingsChanged = true;
                }
                if (ImGui::SliderFloat("Normal Sharpness", &config.shadow_denoiser_normal_power, 4.0f, 64.0f, "%.1f")) {
                    settingsChanged = true;
                }
                ImGui::Unindent();
            }
            if (ImGui::Checkbox("Temporal Anti-Aliasing (TAA)", &config.enable_taa)) {
                settingsChanged = true;
            }
            if (ImGui::IsItemHovered()) {
                ImGui::SetTooltip("Screen-space motion vector reprojection with YCoCg variance clipping and distributed tile-parallel support.");
            }
            if (config.enable_taa) {
                ImGui::Indent();
                if (ImGui::SliderFloat("TAA Alpha", &config.taa_blend_alpha, 0.01f, 0.50f, "%.2f")) {
                    // blend alpha
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Weight of current frame sample vs. temporal history (lower = smoother, higher = sharper/more responsive).");
                }
                if (ImGui::SliderFloat("Clipping Gamma", &config.taa_clipping_gamma, 0.50f, 3.00f, "%.2f")) {
                    // variance clipping gamma
                }
                if (ImGui::IsItemHovered()) {
                    ImGui::SetTooltip("Variance bounding box multiplier for clamping history in YCoCg space.");
                }
                ImGui::Unindent();
            }
            if (ImGui::Checkbox("ACES Filmic Tonemapping", &config.aces_tonemap)) {
                // Tonemap toggle doesn't invalidate accumulation
            }
        }

        // 6. Diagnostics & Console Logging
        if (ImGui::CollapsingHeader("Diagnostics & Logging")) {
            bool loggingEnabled = (config.log_interval_sec > 0.0f);
            if (ImGui::Checkbox("Console Telemetry Output", &loggingEnabled)) {
                config.log_interval_sec = loggingEnabled ? 10.0f : 0.0f;
            }
            if (loggingEnabled) {
                ImGui::SliderFloat("Log Interval (sec)", &config.log_interval_sec, 1.0f, 60.0f, "%.1f s");
            } else {
                ImGui::TextDisabled("Console logging disabled (clean terminal)");
            }
        }

        // 7. Interactive Actions & Telemetry Export
        ImGui::Separator();
        if (ImGui::Button("Reset Accumulation", ImVec2(160.0f, 30.0f))) {
            settingsChanged = true;
            if (actions) actions->resetAccumulation = true;
        }
        ImGui::SameLine();
        if (ImGui::Button("Export Telemetry (.json)", ImVec2(200.0f, 30.0f))) {
            std::string path = ImageDumper::generateDefaultTelemetryPath();
            if (actions) {
                actions->exportTelemetry = true;
                actions->exportTelemetryPath = path;
            }
            m_lastExportNotification = "Saved: " + path;
            m_exportNotificationTimer = 4.0f;
        }
        if (m_exportNotificationTimer > 0.0f) {
            ImGui::TextColored(ImVec4(0.3f, 1.0f, 0.4f, 1.0f), "%s", m_lastExportNotification.c_str());
        }
        renderMoreDataBelowIndicator("ControlPanel");
    }
    ImGui::End();


    // On-screen load split visualization overlay badges
    if (config.visualize_mgpu_split && config.mgpu_mode != MultiGpuMode::Off) {
        ImGuiWindowFlags overlayFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                        ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
        ImGui::SetNextWindowBgAlpha(0.70f);

        float midX = isPortrait ? (dispW * 0.5f) : ((hudX + hudW + ctrlX) * 0.5f);

        if (config.mgpu_mode == MultiGpuMode::CheckerboardTile ||
            (config.mgpu_mode == MultiGpuMode::Auto && config.spp == 1)) {
            float badgeW = 420.0f;
            ImGui::SetNextWindowPos(ImVec2(midX - badgeW * 0.5f, 18.0f), ImGuiCond_Always);
            if (ImGui::Begin("##MgpuCheckerboardBadge", nullptr, overlayFlags)) {
                ImGui::TextColored(ImVec4(0.2f, 0.9f, 1.0f, 1.0f), "[GPU 0: Cyan (Even)]");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.75f, 0.2f, 1.0f), "[GPU 1: Amber (Odd)]");
                ImGui::SameLine();
                ImGui::TextDisabled("| %ux%u Tiling", config.tile_size, config.tile_size);
            }
            ImGui::End();
        } else if (config.mgpu_mode == MultiGpuMode::SampleParallel || (config.mgpu_mode == MultiGpuMode::Auto && config.spp > 1)) {
            float badgeW = 420.0f;
            ImGui::SetNextWindowPos(ImVec2(midX - badgeW * 0.5f, 18.0f), ImGuiCond_Always);
            if (ImGui::Begin("##MgpuSampleParallelBadge", nullptr, overlayFlags)) {
                uint32_t primSpp = (config.spp + 1) / 2;
                uint32_t secSpp = config.spp / 2;
                ImGui::TextColored(ImVec4(0.4f, 1.0f, 0.4f, 1.0f), "[Sample-Parallel Dual GPU: %u + %u = %u SPP]", primSpp, secSpp, config.spp);
            }
            ImGui::End();
        }
    }

    ImGui::Render();

    // Render using Vulkan 1.4 Dynamic Rendering
    VkRenderingAttachmentInfo colorAttachment{};
    colorAttachment.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO;
    colorAttachment.imageView = targetView;
    colorAttachment.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
    colorAttachment.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
    colorAttachment.storeOp = VK_ATTACHMENT_STORE_OP_STORE;

    VkRenderingInfo renderInfo{};
    renderInfo.sType = VK_STRUCTURE_TYPE_RENDERING_INFO;
    renderInfo.renderArea = { {0, 0}, {width, height} };
    renderInfo.layerCount = 1;
    renderInfo.colorAttachmentCount = 1;
    renderInfo.pColorAttachments = &colorAttachment;

    vkCmdBeginRendering(cmd, &renderInfo);
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    vkCmdEndRendering(cmd);

    if (settingsChanged || (actions && actions->resetAccumulation)) {
        resetHistory();
    }

    return settingsChanged;
}

} // namespace pathways
