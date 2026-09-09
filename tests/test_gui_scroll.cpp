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

    // Build default font texture so metrics are valid
    unsigned char* pixels;
    int width, height;
    io.Fonts->GetTexDataAsRGBA32(&pixels, &width, &height);

    // Frame 1: Initial layout and measurement of tall content
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));

    ImGuiWindowFlags panelFlags = ImGuiWindowFlags_NoScrollbar;
    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window");

    // Add many items to exceed 300px height
    for (int i = 0; i < 40; ++i) {
        ImGui::Text("Informational Data Line #%d: Telemetry Metric = 123.45", i);
    }

    float scrollY_f1 = ImGui::GetScrollY();
    float scrollMaxY_f1 = ImGui::GetScrollMaxY();

    ImGui::End();
    ImGui::Render();

    std::cout << "[PASS] Frame 1 rendered. ScrollY: " << scrollY_f1 << ", ScrollMaxY: " << scrollMaxY_f1 << std::endl;

    // Frame 2: Content size is known from frame 1, test indicator condition
    io.AddMousePosEvent(100.0f, 100.0f); // Hover over window
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));

    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 2");
    for (int i = 0; i < 40; ++i) {
        ImGui::Text("Informational Data Line #%d: Telemetry Metric = 123.45", i);
    }

    float scrollY = ImGui::GetScrollY();
    float scrollMaxY = ImGui::GetScrollMaxY();

    check_true(scrollMaxY > 100.0f, "Content exceeds window height, scrollMaxY > 100px");
    check_true(scrollY == 0.0f, "Initial scroll position is at top (0px)");

    bool hasMoreBelow = (scrollMaxY > 2.0f) && (scrollY < (scrollMaxY - 2.0f));
    check_true(hasMoreBelow, "Indicator MUST be visible when content is obscured below");
    std::cout << "[PASS] Frame 2: Indicator correctly ACTIVE at top (ScrollMax: " << scrollMaxY << " px)." << std::endl;

    ImGui::End();
    ImGui::Render();

    // Frame 3: Simulate mouse wheel scrolling down over the window
    io.AddMousePosEvent(100.0f, 100.0f); // Hover over window
    io.AddMouseWheelEvent(0.0f, -3.0f);   // Scroll down 3 wheel notches

    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));

    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 3");
    for (int i = 0; i < 40; ++i) {
        ImGui::Text("Informational Data Line #%d: Telemetry Metric = 123.45", i);
    }

    float scrollY_f3 = ImGui::GetScrollY();
    check_true(scrollY_f3 > 0.0f, "Mouse wheel successfully scrolled window down with ImGuiWindowFlags_NoScrollbar");
    std::cout << "[PASS] Frame 3: Mouse wheel down scroll verified! New ScrollY: " << scrollY_f3 << " px." << std::endl;

    ImGui::End();
    ImGui::Render();

    // Frame 4: Set scroll to bottom
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));

    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 4a");
    for (int i = 0; i < 40; ++i) {
        ImGui::Text("Informational Data Line #%d: Telemetry Metric = 123.45", i);
    }
    ImGui::SetScrollY(scrollMaxY); // Move to bottom for next frame
    ImGui::End();
    ImGui::Render();

    // Frame 4b: Verify at bottom
    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));

    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 4b");
    for (int i = 0; i < 40; ++i) {
        ImGui::Text("Informational Data Line #%d: Telemetry Metric = 123.45", i);
    }

    float scrollY_bottom = ImGui::GetScrollY();
    float currentScrollMaxY = ImGui::GetScrollMaxY();
    bool hasMoreBelow_bottom = (currentScrollMaxY > 2.0f) && (scrollY_bottom < (currentScrollMaxY - 2.0f));
    check_true(!hasMoreBelow_bottom, "Indicator MUST disappear when scrolled to bottom");
    std::cout << "[PASS] Frame 4: Indicator correctly INACTIVE when reaching bottom of content (ScrollY: " << scrollY_bottom << " px)." << std::endl;

    ImGui::End();
    ImGui::Render();


    // Frame 5: Simulate mouse wheel scrolling UP from bottom
    io.AddMousePosEvent(100.0f, 100.0f); // Hover over window
    io.AddMouseWheelEvent(0.0f, 2.0f);    // Scroll up 2 wheel notches

    ImGui::NewFrame();
    ImGui::SetNextWindowPos(ImVec2(20.0f, 20.0f));
    ImGui::SetNextWindowSize(ImVec2(400.0f, 300.0f));

    check_true(ImGui::Begin("TestInformationalPanel", nullptr, panelFlags), "Begin window frame 5");
    for (int i = 0; i < 40; ++i) {
        ImGui::Text("Informational Data Line #%d: Telemetry Metric = 123.45", i);
    }

    float scrollY_f5 = ImGui::GetScrollY();
    check_true(scrollY_f5 < scrollY_bottom, "Mouse wheel successfully scrolled window back UP");
    bool hasMoreBelow_f5 = (scrollMaxY > 2.0f) && (scrollY_f5 < (scrollMaxY - 2.0f));
    check_true(hasMoreBelow_f5, "Indicator reappears when scrolled back up from bottom");
    std::cout << "[PASS] Frame 5: Mouse wheel up scroll verified! New ScrollY: " << scrollY_f5 << " px." << std::endl;

    ImGui::End();
    ImGui::Render();

    ImGui::DestroyContext(ctx);
    std::cout << "==========================================================" << std::endl;
    std::cout << "[SUCCESS] ALL GUI MOUSE WHEEL & SCROLL INDICATOR TESTS PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
