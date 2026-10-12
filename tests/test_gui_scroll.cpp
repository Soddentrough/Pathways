#include <iostream>
#include <cassert>
#include <cmath>
#include "imgui.h"

void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] Assertion failed: " << msg << std::endl;
        std::exit(1);
    }
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing GUI Mouse Wheel Scrolling & Indicator" << std::endl;
    std::cout << "==========================================================" << std::endl;

    IMGUI_CHECKVERSION();
    ImGuiContext* ctx = ImGui::CreateContext();
    check_true(ctx != nullptr, "ImGui context created");

    ImGuiIO& io = ImGui::GetIO();
    io.DisplaySize = ImVec2(1920.0f, 1080.0f);
    io.DeltaTime = 1.0f / 60.0f;
    // Enable keyboard & gamepad navigation matching GuiManager::init
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;

    // Build default font texture so metrics are valid
    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    ImGuiWindowFlags panelFlags = ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoNav;

    auto renderWindowContent = [&]() {
        // Top button (previously captured NavId focus and snapped scroll back)
        if (ImGui::Button("ENTER SCENE NAVIGATION", ImVec2(200.0f, 32.0f))) {}
        ImGui::SameLine();
        if (ImGui::Button("RESET ACCUMULATION", ImVec2(160.0f, 32.0f))) {}

        // Embedded child card without NoScrollWithMouse
        if (ImGui::BeginChild("FpsHeroCard", ImVec2(360.0f, 50.0f), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar)) {
            ImGui::Text("HERO METRIC: 120 FPS (8.33 ms)");
        }
        ImGui::EndChild();

        // 40 telemetry lines to ensure vertical content height > 1000px
        for (int i = 0; i < 40; ++i) {
            ImGui::Text("Informational Data Line #%d: Telemetry Metric = 123.45", i);
            if (i == 10) {
                static float sliderVal = 1.0f;
                ImGui::SliderFloat("##TestSlider", &sliderVal, 0.0f, 10.0f);
            }
        }
    };

    // Frame 1: Initial layout and measurement of tall content
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 1");
    renderWindowContent();
    float scrollY_f1 = ImGui::GetScrollY();
    float scrollMaxY_f1 = ImGui::GetScrollMaxY();
    ImGui::End();
    ImGui::Render();
    std::cout << "[PASS] Frame 1 rendered. ScrollY: " << scrollY_f1 << ", ScrollMaxY: " << scrollMaxY_f1 << std::endl;

    // Frame 2a: Hover over window at top button position and click down
    io.AddMousePosEvent(100.0f, 65.0f);
    io.AddMouseButtonEvent(0, true);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 2a (click)");
    renderWindowContent();
    ImGui::End();
    ImGui::Render();

    // Frame 2b: Release click
    io.AddMouseButtonEvent(0, false);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 2b (release)");
    renderWindowContent();
    float scrollY = ImGui::GetScrollY();
    float scrollMaxY = ImGui::GetScrollMaxY();
    check_true(scrollMaxY > 100.0f, "Content exceeds window height, scrollMaxY > 100px");
    check_true(scrollY == 0.0f, "Initial scroll position is at top (0px)");
    bool hasMoreBelow = (scrollMaxY > 2.0f) && (scrollY < (scrollMaxY - 2.0f));
    check_true(hasMoreBelow, "Indicator MUST be visible when content is obscured below");
    std::cout << "[PASS] Frame 2: Top button clicked, indicator ACTIVE (ScrollMax: " << scrollMaxY << " px)." << std::endl;
    ImGui::End();
    ImGui::Render();

    // Frame 3: Mouse wheel scroll down 2 notches (mouse already hovering window at 100, 65)
    io.AddMouseWheelEvent(0.0f, -2.0f);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 3");
    renderWindowContent();
    float scrollY_f3 = ImGui::GetScrollY();
    check_true(scrollY_f3 > 0.0f, "Mouse wheel successfully scrolled window down");
    std::cout << "[PASS] Frame 3: Scrolled down 2 notches! ScrollY: " << scrollY_f3 << " px." << std::endl;
    ImGui::End();
    ImGui::Render();

    // Frame 4: Next frame with NO mouse wheel input (CRITICAL: previously snapped back to top here!)
    io.AddMouseWheelEvent(0.0f, 0.0f);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 4");
    renderWindowContent();
    float scrollY_f4 = ImGui::GetScrollY();
    check_true(scrollY_f4 == scrollY_f3, "Scroll position MUST remain stable without snap-back on subsequent frame");
    std::cout << "[PASS] Frame 4: Multi-frame scroll retention verified (No snap-back)! ScrollY: " << scrollY_f4 << " px." << std::endl;
    ImGui::End();
    ImGui::Render();

    // Frame 5a: Move cursor over child card (FpsHeroCard) at (100.0f, 110.0f)
    io.AddMousePosEvent(100.0f, 110.0f);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 5a");
    renderWindowContent();
    ImGui::End();
    ImGui::Render();

    // Frame 5b: Mouse wheel scroll while hovering directly over child card (FpsHeroCard)
    // Verifies that child windows without NoScrollWithMouse forward scroll to parent
    io.AddMouseWheelEvent(0.0f, -2.0f);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 5b");
    renderWindowContent();
    float scrollY_f5b = ImGui::GetScrollY();
    check_true(scrollY_f5b > scrollY_f4, "Mouse wheel over child card forwarded to parent window");
    std::cout << "[PASS] Frame 5: Child card scroll bubbling verified! ScrollY: " << scrollY_f5b << " px." << std::endl;
    ImGui::End();
    ImGui::Render();

    // Frame 6: Click an item inside the scrolled content area
    io.AddMouseWheelEvent(0.0f, 0.0f);
    io.AddMouseButtonEvent(0, true);
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 6");
    renderWindowContent();
    float scrollY_f6 = ImGui::GetScrollY();
    check_true(scrollY_f6 == scrollY_f5b, "Clicking widget inside scrolled area maintains scroll position");
    std::cout << "[PASS] Frame 6: Widget interaction inside scrolled viewport verified! ScrollY: " << scrollY_f6 << " px." << std::endl;
    ImGui::End();
    ImGui::Render();
    io.AddMouseButtonEvent(0, false);

    // Frame 7: Scroll to bottom programmatically and verify bottom state
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 7a");
    renderWindowContent();
    ImGui::SetScrollY(scrollMaxY);
    ImGui::End();
    ImGui::Render();

    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 7b");
    renderWindowContent();
    float scrollY_bottom = ImGui::GetScrollY();
    float currentScrollMaxY = ImGui::GetScrollMaxY();
    bool hasMoreBelow_bottom = (currentScrollMaxY > 2.0f) && (scrollY_bottom < (currentScrollMaxY - 2.0f));
    check_true(!hasMoreBelow_bottom, "Indicator MUST disappear when scrolled to bottom");
    std::cout << "[PASS] Frame 7: Indicator correctly INACTIVE at bottom (ScrollY: " << scrollY_bottom << " px)." << std::endl;
    ImGui::End();
    ImGui::Render();

    // Frame 8: Mouse wheel scroll UP back towards the top
    io.AddMouseWheelEvent(0.0f, 4.0f); // Scroll up
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 8");
    renderWindowContent();
    float scrollY_f8 = ImGui::GetScrollY();
    check_true(scrollY_f8 < scrollY_bottom, "Mouse wheel successfully scrolled window back UP");
    bool hasMoreBelow_f8 = (scrollMaxY > 2.0f) && (scrollY_f8 < (scrollMaxY - 2.0f));
    check_true(hasMoreBelow_f8, "Indicator reappears when scrolled back up from bottom");
    std::cout << "[PASS] Frame 8: Upward scrolling and indicator reactivation verified! ScrollY: " << scrollY_f8 << " px." << std::endl;
    ImGui::End();
    ImGui::Render();

    ImGui::DestroyContext(ctx);
    std::cout << "==========================================================" << std::endl;
    std::cout << "[SUCCESS] ALL GUI MOUSE WHEEL & SCROLL TESTS PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
