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

bool GuiManager::render(VkCommandBuffer cmd, VkImageView targetView, uint32_t width, uint32_t height,
                        Config& config, const FrameStats& stats, bool& cameraMode, Camera* camera,
                        const DisplayInfo* displayInfo, bool isFullscreen,
                        GuiActions* actions) {
    if (!m_initialized) return false;

    bool settingsChanged = false;

    // Record frame time into rolling history
    float currentFrameTime = static_cast<float>(stats.avg_frame_time_ms > 0.01 ? stats.avg_frame_time_ms : (stats.primary_gpu_time_ms + stats.tonemap_time_ms));
    if (currentFrameTime <= 0.001f) currentFrameTime = 0.5f;

    m_frameTimeHistory[m_historyOffset] = currentFrameTime;
    m_historyOffset = (m_historyOffset + 1) % HISTORY_SIZE;

    // Compute min, max, avg across rolling history
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
    // Responsive Layout Calculation
    // =========================================================================
    float aspect = (height > 0) ? (static_cast<float>(width) / static_cast<float>(height)) : 1.0f;
    bool isPortrait = (aspect < 1.05f);

    ImGuiCond layoutCond = ImGuiCond_FirstUseEver;
    if (!m_layoutInitialized || m_lastWasPortrait != isPortrait) {
        layoutCond = ImGuiCond_Always;
        m_layoutInitialized = true;
        m_lastWasPortrait = isPortrait;
    }

    float hudX = 20.0f;
    float hudY = 20.0f;
    float hudW = 440.0f;
    float hudH = 540.0f;

    float ctrlX = 480.0f;
    float ctrlY = 20.0f;
    float ctrlW = 440.0f;
    float ctrlH = std::min(static_cast<float>(height) - 40.0f, 780.0f);

    if (isPortrait) {
        // Vertical stacked layout on the left: leaves the right side unobstructed for the 3D scene
        float sidebarW = std::min(400.0f, static_cast<float>(width) * 0.38f);
        if (sidebarW < 350.0f) sidebarW = std::max(280.0f, static_cast<float>(width) - 24.0f);

        hudX = 12.0f;
        hudY = 12.0f;
        hudW = sidebarW;
        hudH = 640.0f;

        ctrlX = 12.0f;
        ctrlY = hudY + hudH + 12.0f;
        ctrlW = sidebarW;
        ctrlH = std::max(220.0f, static_cast<float>(height) - ctrlY - 12.0f);
    } else {
        // Landscape layout: HUD on left, Control panel docked on right edge
        hudX = 20.0f;
        hudY = 20.0f;
        hudW = 440.0f;
        hudH = std::min(static_cast<float>(height) - 40.0f, 660.0f);

        ctrlW = 440.0f;
        ctrlX = static_cast<float>(width) - ctrlW - 20.0f;
        ctrlY = 20.0f;
        ctrlH = static_cast<float>(height) - 40.0f;
    }

    // =========================================================================
    // WINDOW 1: Pathways Live Profiler & Telemetry HUD
    // =========================================================================
    float fps = (currentFrameTime > 0.0001f) ? (1000.0f / currentFrameTime) : 0.0f;
    float avgFps = (historyAvg > 0.0001f) ? (1000.0f / historyAvg) : 0.0f;

    char hudTitle[128];
    std::snprintf(hudTitle, sizeof(hudTitle), "Pathways Telemetry HUD  -  %.0f FPS (%.2f ms)###TelemetryHUD", fps, currentFrameTime);

    ImGui::SetNextWindowPos(ImVec2(hudX, hudY), layoutCond);
    ImGui::SetNextWindowSize(ImVec2(hudW, hudH), layoutCond);

    if (ImGui::Begin(hudTitle, nullptr)) {
        ImGui::TextColored(ImVec4(0.2f, 0.8f, 1.0f, 1.0f), "Dual AMD RDNA4 (GFX1201) Pure Vulkan 1.4 Path Tracer");
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
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Average: %.1f FPS", avgFps);
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

        // 2. Real-Time Latency Histogram / Graph
        char overlayBuf[64];
        std::snprintf(overlayBuf, sizeof(overlayBuf), "Avg: %.2f ms (Min: %.2f, Max: %.2f)", historyAvg, historyMin, historyMax);
        float graphMax = std::max(16.0f, historyMax * 1.25f);
        ImGui::PlotLines("Latency", m_frameTimeHistory, static_cast<int>(HISTORY_SIZE), m_historyOffset,
                         overlayBuf, 0.0f, graphMax, ImVec2(hudW - 60.0f, 75.0f));
        ImGui::TextDisabled("8.0ms Budget Line (120 FPS Target)");

        ImGui::Separator();

        // 3. Multi-GPU Subsystem Telemetry
        if (ImGui::CollapsingHeader("Multi-GPU Subsystem Telemetry", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Active Topology: %s", stats.gpu_name.c_str());
            ImGui::Text("Interconnect: PCIe 5.0 x16 (32 GT/s / ~64 GB/s Full-Duplex)");
            ImGui::Text("Multi-GPU Mode: %s", stats.mgpu_mode_str.c_str());

            if (stats.secondary_gpu_time_ms > 0.001) {
                ImGui::Text("  GPU 0 (Primary RT):    %.3f ms", stats.primary_gpu_time_ms);
                ImGui::Text("  GPU 1 (Secondary RT):  %.3f ms", stats.secondary_gpu_time_ms);
                ImGui::Text("  Tonemap & Merge Pass:  %.3f ms", stats.tonemap_time_ms);

                float totalWork = static_cast<float>(stats.primary_gpu_time_ms + stats.secondary_gpu_time_ms);
                float primaryFraction = totalWork > 0.001f ? (static_cast<float>(stats.primary_gpu_time_ms) / totalWork) : 0.5f;
                ImGui::ProgressBar(primaryFraction, ImVec2(hudW - 60.0f, 14.0f), "Load Balance: GPU 0 vs GPU 1");
            } else {
                ImGui::Text("  GPU 0 Ray Tracing:     %.3f ms", stats.primary_gpu_time_ms > 0.0 ? stats.primary_gpu_time_ms : currentFrameTime);
                ImGui::Text("  ACES Filmic Tonemap:   %.3f ms", stats.tonemap_time_ms);
            }
        }

        // 4. Hardware Pipeline & Architecture Telemetry
        if (ImGui::CollapsingHeader("Hardware Architecture & Execution Mode", ImGuiTreeNodeFlags_DefaultOpen)) {
            ImGui::Text("Subgroup Execution: Native Wave32 SIMD (AMD RDNA4)");
            if (config.enable_hardware_rt) {
                ImGui::TextColored(ImVec4(0.25f, 0.95f, 0.45f, 1.0f), "RT Pipeline: Hardware BVH Accelerated");
                ImGui::TextColored(ImVec4(0.65f, 0.82f, 1.0f, 1.0f), "  Active Pipeline Extensions:");
                ImGui::BulletText("VK_KHR_ray_query (In-Shader Ray Queries)");
                ImGui::BulletText("VK_KHR_acceleration_structure (BLAS + TLAS)");
                ImGui::BulletText("VK_KHR_buffer_device_address (64-bit BDA)");
                ImGui::BulletText("VK_KHR_deferred_host_operations (Host Build)");
                ImGui::BulletText("SPIR-V: GL_EXT_ray_query (Native Wave32)");
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.2f, 1.0f), "RT Pipeline: Software ALU Loop (Compute Fallback)");
                ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.5f, 1.0f), "  HW RT Extensions: BYPASSED / INACTIVE");
                ImGui::BulletText("Active: ALU Möller-Trumbore + 32KB LDS Cache");
            }
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
            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.18f, 0.58f, 0.28f, 1.0f));
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered, ImVec4(0.24f, 0.72f, 0.36f, 1.0f));
            if (ImGui::Button("ENTER SCENE NAVIGATION\n[Click or press TAB to capture mouse]", ImVec2(-1, 38.0f))) {
                cameraMode = true;
            }
            ImGui::PopStyleColor(2);
        }

        // 0. Display & Viewport Architecture
        if (ImGui::CollapsingHeader("Display & Viewport Architecture", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (displayInfo) {
                ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Physical Display:");
                ImGui::Text("  Device:  %s", displayInfo->displayName.c_str());
                ImGui::Text("  Native:  %u x %u @ %.2f Hz", displayInfo->nativeWidth, displayInfo->nativeHeight, displayInfo->refreshRate);
                ImGui::Text("  Usable:  %u x %u (Scale: %.2f)", displayInfo->usableWidth, displayInfo->usableHeight, displayInfo->contentScale);
                ImGui::Text("  Aspect:  %.3f (%s)", displayInfo->displayAspect,
                            displayInfo->isPortrait ? "Portrait (DualUp 16:18)" : (displayInfo->isUltraWide ? "Ultra-Wide (>2:1)" : "Landscape (16:9)"));
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.4f, 0.8f, 1.0f, 1.0f), "Active Viewport:");
            ImGui::Text("  Resolution: %u x %u (Aspect: %.3f)", width, height, aspect);
            ImGui::Text("  Layout:     %s", isPortrait ? "Portrait Sidebar Stack" : "Landscape Edge-Docked");

            if (actions) {
                if (ImGui::Button(isFullscreen ? "Exit Fullscreen (F11)" : "Toggle Fullscreen (F11)", ImVec2(180.0f, 26.0f))) {
                    actions->toggleFullscreen = true;
                }

                ImGui::Spacing();
                ImGui::Text("Resolution Presets:");
                if (displayInfo && ImGui::Button("Native Full Display", ImVec2(160.0f, 24.0f))) {
                    actions->requestedWidth = displayInfo->usableWidth;
                    actions->requestedHeight = displayInfo->usableHeight;
                }
                if (ImGui::Button("DualUp (1280x2048)", ImVec2(160.0f, 24.0f))) {
                    actions->requestedWidth = 1280;
                    actions->requestedHeight = 2048;
                }
                ImGui::SameLine();
                if (ImGui::Button("1:1 (1440x1440)", ImVec2(140.0f, 24.0f))) {
                    actions->requestedWidth = 1440;
                    actions->requestedHeight = 1440;
                }

                if (ImGui::Button("FHD (1920x1080)", ImVec2(130.0f, 24.0f))) {
                    actions->requestedWidth = 1920;
                    actions->requestedHeight = 1080;
                }
                ImGui::SameLine();
                if (ImGui::Button("QHD (2560x1440)", ImVec2(130.0f, 24.0f))) {
                    actions->requestedWidth = 2560;
                    actions->requestedHeight = 1440;
                }
                ImGui::SameLine();
                if (ImGui::Button("4K (3840x2160)", ImVec2(130.0f, 24.0f))) {
                    actions->requestedWidth = 3840;
                    actions->requestedHeight = 2160;
                }
            }

            if (camera) {
                ImGui::Spacing();
                bool adaptive = camera->isAdaptiveFov();
                if (ImGui::Checkbox("Adaptive Aspect FOV (Auto-frame scene)", &adaptive)) {
                    camera->setAdaptiveFov(adaptive);
                    if (adaptive) {
                        camera->adaptFovForAspect(aspect);
                    }
                    settingsChanged = true;
                }
                if (ImGui::Button("Re-frame Scene (Auto FOV)", ImVec2(190.0f, 24.0f))) {
                    camera->setAdaptiveFov(true);
                    camera->adaptFovForAspect(aspect);
                    settingsChanged = true;
                }
            }
            ImGui::Separator();
        }

        // Camera & FPS Navigation Controls
        if (ImGui::CollapsingHeader("Camera & Scene Navigation", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (camera) {
                float speed = camera->getSpeed();
                if (ImGui::SliderFloat("Move Speed", &speed, 0.2f, 25.0f, "%.1f m/s")) {
                    camera->setSpeed(speed);
                }
                float sens = camera->getSensitivity();
                if (ImGui::SliderFloat("Mouse Sensitivity", &sens, 0.02f, 0.5f, "%.2f")) {
                    camera->setSensitivity(sens);
                }
                float fov = camera->getFov();
                if (ImGui::SliderFloat("Field of View", &fov, 20.0f, 120.0f, "%.1f deg")) {
                    camera->setAdaptiveFov(false);
                    camera->setFov(fov);
                    settingsChanged = true;
                }

                glm::vec3 pos = camera->getPosition();
                ImGui::Text("Position: (%.2f, %.2f, %.2f)", pos.x, pos.y, pos.z);
                ImGui::Text("Look Angles: Yaw: %.1f deg, Pitch: %.1f deg", camera->getYaw(), camera->getPitch());

                if (ImGui::Button("Reset Camera to Default", ImVec2(200.0f, 26.0f))) {
                    camera->resetToDefault();
                    camera->adaptFovForAspect(aspect);
                    settingsChanged = true;
                }
            }

            ImGui::Spacing();
            ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "FPS Input Reference:");
            ImGui::BulletText("TAB: Toggle UI Options / FPS Navigation");
            ImGui::BulletText("W / A / S / D: Forward / Left / Back / Right");
            ImGui::BulletText("Space / C (or E / Q): Move Up / Down");
            ImGui::BulletText("Left Shift: Sprint Boost (2.5x speed)");
            ImGui::BulletText("Mouse: Rotate Camera (Pitch & Yaw)");
            ImGui::BulletText("Mouse Wheel: Adjust Movement Speed");
            ImGui::BulletText("ESC: Release Mouse (FPS Mode) / Exit (UI)");
        }

        // 1. Hardware Acceleration Toggles & Pipeline Extensions
        if (ImGui::CollapsingHeader("Hardware Acceleration & Ray Tracing Pipeline", ImGuiTreeNodeFlags_DefaultOpen)) {
            if (ImGui::Checkbox("Enable Hardware Ray Tracing", &config.enable_hardware_rt)) {
                settingsChanged = true;
                if (config.enable_hardware_rt) {
                    Logger::info("Hardware Ray Tracing toggled: ENABLED");
                    Logger::info("  Active Vulkan Pipeline Extensions:");
                    Logger::info("    - VK_KHR_ray_query (in-shader ray queries / rayQueryEXT)");
                    Logger::info("    - VK_KHR_acceleration_structure (hardware two-level BVH: BLAS + TLAS)");
                    Logger::info("    - VK_KHR_buffer_device_address (64-bit GPU virtual addresses for geometry & AS)");
                    Logger::info("    - VK_KHR_deferred_host_operations (driver AS build host multi-threading)");
                    Logger::info("  Active SPIR-V Extension: GL_EXT_ray_query (Native Wave32 SIMD)");
                } else {
                    Logger::info("Hardware Ray Tracing toggled: DISABLED");
                    Logger::info("  HW RT Pipeline Extensions: BYPASSED / INACTIVE");
                    Logger::info("  Active Traversal: Compute shader ALU Möller-Trumbore loop with 32KB LDS cache");
                }
            }

            ImGui::Spacing();
            if (config.enable_hardware_rt) {
                ImGui::TextColored(ImVec4(0.25f, 0.95f, 0.45f, 1.0f), "[ACTIVE] HARDWARE ACCELERATED (AMD RDNA4 Ray Accelerators)");
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
            } else {
                ImGui::TextColored(ImVec4(1.0f, 0.72f, 0.2f, 1.0f), "[ACTIVE] SOFTWARE FALLBACK (Compute ALU Emulation)");
                ImGui::Spacing();

                ImGui::TextColored(ImVec4(1.0f, 0.82f, 0.5f, 1.0f), "Vulkan HW RT Extensions Status:");
                ImGui::BulletText("VK_KHR_ray_query:");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "BYPASSED / INACTIVE");
                ImGui::SameLine();
                ImGui::TextDisabled("(Zero RT core invocations)");

                ImGui::BulletText("VK_KHR_acceleration_structure:");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "BYPASSED");
                ImGui::SameLine();
                ImGui::TextDisabled("(TLAS/BLAS traversal skipped)");

                ImGui::BulletText("VK_KHR_deferred_host_operations:");
                ImGui::SameLine();
                ImGui::TextColored(ImVec4(1.0f, 0.4f, 0.4f, 1.0f), "INACTIVE");

                ImGui::Spacing();
                ImGui::TextColored(ImVec4(0.7f, 0.85f, 1.0f, 1.0f), "Active Compute Pipeline:");
                ImGui::BulletText("Moller-Trumbore ray-triangle intersection loop");
                ImGui::BulletText("32KB Local Data Share (LDS) tile cache per Dual-CU");
                ImGui::BulletText("Global memory SSBO primitive storage arrays");
            }
            ImGui::Separator();
        }

        // 2. Multi-GPU Scalability Mode
        if (ImGui::CollapsingHeader("Multi-GPU Architecture", ImGuiTreeNodeFlags_DefaultOpen)) {
            const char* mgpuModes[] = {
                "Single GPU (Off)",
                "Sample Parallelism (Dual GPU)",
                "Split-Frame Tiling (Dual GPU)",
                "Dynamic Work Queue (Dual GPU)"
            };
            int currentMode = 0;
            if (config.mgpu_mode == MultiGpuMode::SampleParallel) currentMode = 1;
            else if (config.mgpu_mode == MultiGpuMode::CheckerboardTile) currentMode = 2;
            else if (config.mgpu_mode == MultiGpuMode::DynamicWorkQueue) currentMode = 3;

            if (ImGui::Combo("Execution Mode", &currentMode, mgpuModes, IM_ARRAYSIZE(mgpuModes))) {
                if (currentMode == 0) config.mgpu_mode = MultiGpuMode::Off;
                else if (currentMode == 1) config.mgpu_mode = MultiGpuMode::SampleParallel;
                else if (currentMode == 2) config.mgpu_mode = MultiGpuMode::CheckerboardTile;
                else if (currentMode == 3) config.mgpu_mode = MultiGpuMode::DynamicWorkQueue;
                settingsChanged = true;
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

        // 6. Interactive Actions
        ImGui::Separator();
        if (ImGui::Button("Reset Accumulation", ImVec2(180.0f, 30.0f))) {
            settingsChanged = true;
            if (actions) actions->resetAccumulation = true;
        }
    }
    ImGui::End();

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

    return settingsChanged;
}

} // namespace pathways
