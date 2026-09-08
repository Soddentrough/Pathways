#include "scene/Camera.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace pathways;

void assert_near(float a, float b, float eps = 0.01f, const char* msg = "") {
    if (std::abs(a - b) > eps) {
        std::cerr << "Assertion failed: " << a << " != " << b << " (eps: " << eps << ") " << msg << std::endl;
        std::exit(1);
    }
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing Camera & FPS Navigation Controls" << std::endl;
    std::cout << "==========================================================" << std::endl;

    Camera cam(glm::vec3(0.0f, 1.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 16.0f / 9.0f);

    // 1. Initial orientation
    assert_near(cam.getPosition().x, 0.0f, 0.001f, "Initial pos X");
    assert_near(cam.getPosition().y, 1.0f, 0.001f, "Initial pos Y");
    assert_near(cam.getPosition().z, 3.0f, 0.001f, "Initial pos Z");
    assert_near(cam.getYaw(), -90.0f, 0.01f, "Initial Yaw");
    assert_near(cam.getPitch(), 0.0f, 0.01f, "Initial Pitch");
    assert_near(cam.getFront().z, -1.0f, 0.01f, "Initial Front Z");
    assert_near(cam.getRight().x, 1.0f, 0.01f, "Initial Right X");
    assert_near(cam.getUp().y, 1.0f, 0.01f, "Initial Up Y");
    std::cout << "[PASS] Initial camera pose and orthogonal coordinate vectors verified." << std::endl;

    // 2. Forward movement (W)
    cam.resetMoved();
    cam.processFpsInput(1.0f, 0.0f, 0.0f, 0.1f, false); // forward 1.0, dt 0.1s, normal speed (3.0 m/s)
    assert_near(cam.getPosition().z, 3.0f - 0.3f, 0.01f, "Forward move Z");
    assert(cam.hasMoved());
    std::cout << "[PASS] Forward FPS movement (W) verified." << std::endl;

    // 3. Backward movement (S) with sprint (Shift -> 2.5x speed)
    cam.resetMoved();
    cam.processFpsInput(-1.0f, 0.0f, 0.0f, 0.1f, true); // back 1.0, dt 0.1s, sprint (3.0 * 2.5 = 7.5 m/s)
    assert_near(cam.getPosition().z, 2.7f + 0.75f, 0.01f, "Backward sprint move Z");
    assert(cam.hasMoved());
    std::cout << "[PASS] Backward FPS sprint movement (S + Shift) verified (2.5x speed multiplier)." << std::endl;

    // 4. Strafe movement (A and D)
    cam.resetMoved();
    cam.processFpsInput(0.0f, 1.0f, 0.0f, 0.1f, false); // strafe right
    assert_near(cam.getPosition().x, 0.3f, 0.01f, "Strafe right X");
    assert(cam.hasMoved());

    cam.resetMoved();
    cam.processFpsInput(0.0f, -1.0f, 0.0f, 0.1f, false); // strafe left
    assert_near(cam.getPosition().x, 0.0f, 0.01f, "Strafe left back to 0 X");
    assert(cam.hasMoved());
    std::cout << "[PASS] Lateral strafe movement (A / D) verified." << std::endl;

    // 5. Vertical elevation (Space / C)
    cam.resetMoved();
    cam.processFpsInput(0.0f, 0.0f, 1.0f, 0.1f, false); // Move up (Space)
    assert_near(cam.getPosition().y, 1.3f, 0.01f, "Vertical move up Y");

    cam.resetMoved();
    cam.processFpsInput(0.0f, 0.0f, -1.0f, 0.1f, false); // Move down (C)
    assert_near(cam.getPosition().y, 1.0f, 0.01f, "Vertical move down back to 1.0 Y");
    std::cout << "[PASS] Vertical elevation movement (Space / C) verified." << std::endl;

    // 6. Diagonal movement normalization
    cam.resetMoved();
    glm::vec3 posBefore = cam.getPosition();
    cam.processFpsInput(1.0f, 1.0f, 0.0f, 1.0f, false); // W + D for 1 second
    float distMoved = glm::length(cam.getPosition() - posBefore);
    assert_near(distMoved, 3.0f, 0.01f, "Diagonal move distance normalized");
    std::cout << "[PASS] Diagonal vector normalization verified (no sqrt(2) speed anomaly on W+D)." << std::endl;

    // 7. Mouse look rotation
    cam.resetMoved();
    cam.processMouseMovement(100.0f, 0.0f); // rotate yaw right (+10 deg at 0.1 sens)
    assert_near(cam.getYaw(), -80.0f, 0.01f, "Yaw rotation");
    assert(cam.hasMoved());

    cam.resetMoved();
    cam.processMouseMovement(0.0f, -50.0f); // rotate pitch up (+5 deg at 0.1 sens)
    assert_near(cam.getPitch(), 5.0f, 0.01f, "Pitch rotation up");
    assert(cam.hasMoved());
    std::cout << "[PASS] Mouse look pitch & yaw rotation verified." << std::endl;

    // 8. Pitch clamping (-89 deg to +89 deg)
    cam.processMouseMovement(0.0f, -2000.0f); // extreme look up
    assert_near(cam.getPitch(), 89.0f, 0.01f, "Max pitch clamp");

    cam.processMouseMovement(0.0f, 4000.0f); // extreme look down
    assert_near(cam.getPitch(), -89.0f, 0.01f, "Min pitch clamp");
    std::cout << "[PASS] Gimbal lock prevention (pitch clamped to [-89, +89] deg) verified." << std::endl;

    // 9. Reset to default
    cam.resetToDefault();
    assert_near(cam.getPosition().x, 0.0f, 0.01f, "Reset Pos X");
    assert_near(cam.getPosition().y, 1.0f, 0.01f, "Reset Pos Y");
    assert_near(cam.getPosition().z, 3.0f, 0.01f, "Reset Pos Z");
    assert_near(cam.getYaw(), -90.0f, 0.01f, "Reset Yaw");
    assert_near(cam.getPitch(), 0.0f, 0.01f, "Reset Pitch");
    // 10. Adaptive Aspect FOV for Portrait monitors (e.g. LG DualUp 1280x2160)
    cam.setAdaptiveFov(true);
    // At landscape 16:9 (1.777), FOV remains default 45 deg
    cam.adaptFovForAspect(16.0f / 9.0f);
    assert_near(cam.getFov(), 45.0f, 0.01f, "Landscape 16:9 FOV");

    // At portrait 1280x2160 (aspect 0.5926)
    cam.adaptFovForAspect(1280.0f / 2160.0f);
    // tan(28 deg) / 0.5926 ~= 0.8972 -> 2 * atan(0.8972) ~= 83.8 deg
    assert(cam.getFov() > 80.0f && cam.getFov() < 86.0f);
    std::cout << "[PASS] Portrait adaptive FOV calculation verified (" << cam.getFov() << " deg for 1280x2160)." << std::endl;

    // Disabled adaptive FOV test
    cam.setAdaptiveFov(false);
    cam.setFov(50.0f);
    cam.adaptFovForAspect(1280.0f / 2160.0f);
    // 11. Scale-Adaptive Camera Speeds
    // Small scene (e.g. coffee maker, radius 0.25m)
    cam.setSceneScale(0.25f);
    assert_near(cam.getSceneScale(), 0.25f, 0.001f, "Small scene radius");
    assert_near(cam.getSpeed(), 0.25f * 0.25f, 0.001f, "Small scene base speed");
    assert(cam.getSpeed() < 0.1f); // Ensures camera won't fly away in small scenes
    std::cout << "[PASS] Small scene scale adaptivity verified (radius: 0.25m -> speed: " << cam.getSpeed() << " m/s)." << std::endl;

    // Large scene (e.g. living room / house, radius 20.0m)
    cam.setSceneScale(20.0f);
    assert_near(cam.getSceneScale(), 20.0f, 0.001f, "Large scene radius");
    assert_near(cam.getSpeed(), 5.0f, 0.01f, "Large scene base speed"); // 20 * 0.25 = 5.0 m/s
    std::cout << "[PASS] Large scene scale adaptivity verified (radius: 20m -> speed: " << cam.getSpeed() << " m/s)." << std::endl;

    // 12. Mouse wheel speed adjustment (multiplicative)
    float speedBeforeWheel = cam.getSpeed();
    cam.adjustSpeedByWheel(1.0f); // Scroll up -> speed * 1.25
    assert_near(cam.getSpeed(), speedBeforeWheel * 1.25f, 0.01f, "Wheel scroll up speed");
    cam.adjustSpeedByWheel(-1.0f); // Scroll down -> speed / 1.25
    assert_near(cam.getSpeed(), speedBeforeWheel, 0.01f, "Wheel scroll down speed back");
    std::cout << "[PASS] Mouse wheel multiplicative speed scaling verified." << std::endl;

    // Speed clamping within [minSpeed, maxSpeed]
    cam.setSpeed(999999.0f);
    assert_near(cam.getSpeed(), cam.getMaxSpeed(), 0.01f, "Max speed clamping");
    cam.setSpeed(0.000001f);
    assert_near(cam.getSpeed(), cam.getMinSpeed(), 0.001f, "Min speed clamping");
    std::cout << "[PASS] Dynamic min/max speed clamping verified (" << cam.getMinSpeed() << " to " << cam.getMaxSpeed() << " m/s)." << std::endl;

    std::cout << "==========================================================" << std::endl;
    std::cout << "  All Camera & FPS Navigation Unit Tests PASSED Cleanly!" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
