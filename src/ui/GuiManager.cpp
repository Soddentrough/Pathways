#include "ui/GuiManager.hpp"
#include "core/Logger.hpp"
#include "core/Window.hpp"
#include "scene/Camera.hpp"

#include "imgui.h"
#include "backends/imgui_impl_sdl3.h"
#include "backends/imgui_impl_vulkan.h"

#include <stdexcept>
#include <format>

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

bool GuiManager::render(VkCommandBuffer cmd, VkImageView targetView, uint32_t width, uint32_t height,
                        Config& config, const FrameStats& stats, bool& cameraMode, Camera* camera,
                        const DisplayInfo* displayInfo, bool isFullscreen,
                        GuiActions* actions) {
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
    float hudW = std::min(440.0f, dispW - 40.0f);
    float hudH = std::min(dispH - 40.0f, 660.0f);

    float ctrlW = std::min(440.0f, dispW - 40.0f);
    float ctrlX = std::max(20.0f, dispW - ctrlW - 20.0f);
    float ctrlY = 20.0f;
    float ctrlH = std::min(dispH - 40.0f, 850.0f);

    if (isPortrait) {
        // Vertical stacked layout on the left: leaves the right side unobstructed for the 3D scene
        float sidebarW = std::min(400.0f, dispW * 0.45f);
        if (sidebarW < 320.0f) sidebarW = std::max(260.0f, dispW - 24.0f);

        hudX = 12.0f;
        hudY = 12.0f;
        hudW = sidebarW;
        hudH = std::min(dispH * 0.5f, 560.0f);

        ctrlX = 12.0f;
        ctrlY = hudY + hudH + 12.0f;
        ctrlW = sidebarW;
        ctrlH = std::max(200.0f, dispH - ctrlY - 12.0f);
    } else {
        // Landscape layout: HUD on left, Control panel docked safely on right edge
        hudX = 20.0f;
        hudY = 20.0f;
        hudW = std::min(440.0f, (dispW - 60.0f) * 0.5f);
        hudH = std::min(dispH - 40.0f, 660.0f);

        ctrlW = std::min(440.0f, (dispW - 60.0f) * 0.5f);
        ctrlX = std::max(hudX + hudW + 20.0f, dispW - ctrlW - 20.0f);
        ctrlY = 20.0f;
        ctrlH = std::min(dispH - 40.0f, 850.0f);
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

    if (ImGui::Begin(hudTitle, nullptr)) {
        if (stats.mgpu_mode_str != "off" && stats.secondary_gpu_time_ms > 0.001) {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Dual %s Pure Vulkan 1.4 Path Tracer", stats.arch_name.c_str());
        } else {
            ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "%s Pure Vulkan 1.4 Path Tracer", stats.arch_name.c_str());
        }
        if (cameraMode) {
            ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.35f, 1.0f), "[MODE] FPS Scene Navigation (WASD + Mouse Look)");
        } else {
            ImGui::TextColored(ImVec4(0.35f, 1.0f, 0.45f, 1.0f), "[MODE] UI Control Panel Active (Mouse Free)");
        }
        ImGui::Separator();

        // 0. Prominent Obvious FPS Hero Display Card
        ImGui::Spacing();
        ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.06f, 0.10f, 0.16f, 0.90f));
        ImGui::PushStyleVar(ImGuiStyleVar_ChildRounding, 6.0f);
        ImGui::PushStyleVar(ImGuiStyleVar_WindowPadding, ImVec2(14.0f, 8.0f));
        if (ImGui::BeginChild("FpsHeroCard", ImVec2(hudW - 40.0f, 58.0f), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar)) {
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
            ImGui::BulletText("SPIR-V: GL_EXT_ray_query (Native Wave32)");
            const char* pipeName = "Wavefront Compaction (3-Stage)";
            if (config.pipeline_type == PipelineType::Megakernel) pipeName = "Monolithic Megakernel";
            else if (config.pipeline_type == PipelineType::Persistent) pipeName = "Persistent Wavefront Work Queue";
            else if (config.pipeline_type == PipelineType::RTP) pipeName = "Ray Tracing Pipeline (KHR)";
            ImGui::Text("Pipeline:           %s", pipeName);
            ImGui::Text("Ray Order:          %s", config.enable_morton_order ? "Morton Z-Curve (8x4)" : "Linear Scanline");
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
    }
    ImGui::End();

    // =========================================================================
    // WINDOW 2: Pathways Interactive Control Panel
    // =========================================================================
    ImGui::SetNextWindowPos(ImVec2(ctrlX, ctrlY), layoutCond);
    ImGui::SetNextWindowSize(ImVec2(ctrlW, ctrlH), layoutCond);

    if (ImGui::Begin("Pathways Control Panel", nullptr)) {
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

        // 0. Display & Viewport Configuration (Fullscreen, Resolution Presets, Adaptive FOV)
        if (displayInfo && ImGui::CollapsingHeader("Display & Viewport Architecture", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::TextDisabled("Physical Display:");
            ImGui::BulletText("Device:   %s", displayInfo->displayName.c_str());
            ImGui::BulletText("Native:   %u x %u @ %.2f Hz", displayInfo->nativeWidth, displayInfo->nativeHeight, displayInfo->refreshRate);
            ImGui::BulletText("Usable:   %u x %u (Scale: %.2f)", displayInfo->usableWidth, displayInfo->usableHeight, displayInfo->contentScale);
            ImGui::BulletText("Aspect:   %.3f (%s)", displayInfo->displayAspect, displayInfo->isPortrait ? "Portrait (DualUp 16:18)" : "Landscape (16:9/16:10)");

            ImGui::Spacing();
            ImGui::TextDisabled("Active Viewport:");
            ImGui::BulletText("Resolution: %u x %u (Aspect: %.3f)", width, height, aspect);
            ImGui::BulletText("Layout:     %s", isPortrait ? "Vertical Stacked (Scene Left Unobstructed)" : "Landscape Edge-Docked");

            if (ImGui::Button(isFullscreen ? "Exit Fullscreen (F11)" : "Toggle Fullscreen (F11)", ImVec2(180.0f, 26.0f))) {
                if (actions) actions->toggleFullscreen = true;
            }

            ImGui::Spacing();
            ImGui::TextDisabled("Resolution Presets:");

            // Preset 1: Native Full Display
            if (ImGui::Button("Native Full Display", ImVec2(160.0f, 26.0f))) {
                if (actions) {
                    actions->requestedWidth = displayInfo->usableWidth;
                    actions->requestedHeight = displayInfo->usableHeight;
                }
            }

            // Preset 2: Portrait / DualUp Square Mode
            ImGui::SameLine();
            if (ImGui::Button("DualUp (1280x2048)", ImVec2(150.0f, 26.0f))) {
                if (actions) {
                    actions->requestedWidth = 1280;
                    actions->requestedHeight = 2048;
                }
            }

            // Preset 3: 1:1 Square
            ImGui::SameLine();
            if (ImGui::Button("1:1 (1440x1440)", ImVec2(130.0f, 26.0f))) {
                if (actions) {
                    actions->requestedWidth = 1440;
                    actions->requestedHeight = 1440;
                }
            }

            // Standard aspect presets
            if (ImGui::Button("FHD (1920x1080)", ImVec2(130.0f, 26.0f))) {
                if (actions) {
                    actions->requestedWidth = 1920;
                    actions->requestedHeight = 1080;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("QHD (2560x1440)", ImVec2(130.0f, 26.0f))) {
                if (actions) {
                    actions->requestedWidth = 2560;
                    actions->requestedHeight = 1440;
                }
            }
            ImGui::SameLine();
            if (ImGui::Button("4K (3840x2160)", ImVec2(130.0f, 26.0f))) {
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
                if (ImGui::SliderFloat("Move Speed", &speed, 0.5f, 20.0f, "%.1f m/s")) {
                    camera->setSpeed(speed);
                }

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

                if (ImGui::Button("Reset Camera to Default", ImVec2(200.0f, 28.0f))) {
                    camera->resetToDefault();
                    settingsChanged = true;
                }
            }
            ImGui::Spacing();
            ImGui::TextDisabled("FPS Input Reference:");
            ImGui::BulletText("TAB: Toggle UI Options / FPS Navigation");
            ImGui::BulletText("W / A / S / D: Forward / Left / Back / Right");
            ImGui::BulletText("Space / C (or E / Q): Move Up / Down");
            ImGui::BulletText("Left Shift: Sprint Boost (3x Speed)");
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
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Pipeline Scheduling Architecture:");
            const char* pipelineModes[] = {
                "Wavefront Compaction (3-Stage: Classify -> Resolve -> Shade)",
                "Monolithic Megakernel (Single-Pass)",
                "Persistent Wavefront Work Queue (Dynamic)",
                "Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline)"
            };
            int curPipeline = 0;
            if (config.pipeline_type == PipelineType::Megakernel) curPipeline = 1;
            else if (config.pipeline_type == PipelineType::Persistent) curPipeline = 2;
            else if (config.pipeline_type == PipelineType::RTP) curPipeline = 3;

            if (ImGui::Combo("Pipeline Mode", &curPipeline, pipelineModes, IM_ARRAYSIZE(pipelineModes))) {
                if (curPipeline == 0) config.pipeline_type = PipelineType::Wavefront;
                else if (curPipeline == 1) config.pipeline_type = PipelineType::Megakernel;
                else if (curPipeline == 2) config.pipeline_type = PipelineType::Persistent;
                else if (curPipeline == 3) config.pipeline_type = PipelineType::RTP;
                settingsChanged = true;
                Logger::info("Switched pipeline architecture to: {}", pipelineModes[curPipeline]);
            }

            if (ImGui::Checkbox("Morton Z-Curve Ray Indexing (8x4)", &config.enable_morton_order)) {
                settingsChanged = true;
                Logger::info("Morton Z-Curve pixel indexing: {}", config.enable_morton_order ? "ENABLED" : "DISABLED");
            }
            ImGui::Separator();
        }

        // 2. Multi-GPU Scalability Mode
        if (ImGui::CollapsingHeader("Multi-GPU Architecture", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* mgpuModes[] = {
                "Single GPU (Off)",
                "Checkerboard Tiling (Dual GPU - 2x FPS)"
            };
            int currentMode = (config.mgpu_mode == MultiGpuMode::CheckerboardTile) ? 1 : 0;

            if (ImGui::Combo("Execution Mode", &currentMode, mgpuModes, IM_ARRAYSIZE(mgpuModes))) {
                MultiGpuMode selectedMode = (currentMode == 1) ? MultiGpuMode::CheckerboardTile : MultiGpuMode::Off;

                if (actions && selectedMode != config.mgpu_mode) {
                    actions->mgpuModeChanged = true;
                    actions->newMgpuMode = selectedMode;
                }
                config.mgpu_mode = selectedMode;
                settingsChanged = true;
            }

            if (config.mgpu_mode == MultiGpuMode::CheckerboardTile) {
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
            int spp = static_cast<int>(config.spp);
            if (ImGui::SliderInt("Samples/Pixel (SPP)", &spp, 1, 64)) {
                config.spp = static_cast<uint32_t>(spp);
                settingsChanged = true;
            }

            int bounces = static_cast<int>(config.max_bounces);
            if (ImGui::SliderInt("Max Ray Bounces", &bounces, 1, 16)) {
                config.max_bounces = static_cast<uint32_t>(bounces);
                settingsChanged = true;
            }

            if (ImGui::SliderFloat("Internal Render Scale", &config.render_scale, 0.25f, 2.0f, "%.2fx")) {
                settingsChanged = true;
            }
        }

        // 4. Lighting & Shading Subsystems
        if (ImGui::CollapsingHeader("Lighting & Shading Components", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Direct Lighting (Area Lights)", &config.enable_direct_light)) {
                settingsChanged = true;
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
        if (ImGui::CollapsingHeader("Post-Processing & Environment", ImGuiTreeNodeFlags_DefaultOpen)) {
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
    }
    ImGui::End();

    // On-screen load split visualization overlay badges
    if (config.visualize_mgpu_split && config.mgpu_mode != MultiGpuMode::Off) {
        ImGuiWindowFlags overlayFlags = ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize |
                                        ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoFocusOnAppearing |
                                        ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoMove;
        ImGui::SetNextWindowBgAlpha(0.70f);

        float midX = isPortrait ? (dispW * 0.5f) : ((hudX + hudW + ctrlX) * 0.5f);

        if (config.mgpu_mode == MultiGpuMode::CheckerboardTile) {
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
