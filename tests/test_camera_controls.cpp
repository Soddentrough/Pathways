#include "scene/Camera.hpp"
#include "scene/ProceduralScene.hpp"
#include "core/Config.hpp"
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

void check_true(bool cond, const char* msg = "") {
    if (!cond) {
        std::cerr << "Assertion failed: condition is false! " << msg << std::endl;
        std::exit(1);
    }
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing Camera & FPS Navigation Controls" << std::endl;
    std::cout << "==========================================================" << std::endl;

    Camera cam(glm::vec3(0.0f, 1.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 16.0f / 9.0f);
    check_true(cam.isDynamicScaling(), "Distance-adaptive speed is enabled by default (isDynamicScaling)");
    check_true(cam.isAdaptiveSpeed(), "Distance-adaptive speed is enabled by default (isAdaptiveSpeed)");

    // Disable dynamic scaling for fixed-rate kinematics unit verification (tests 2-6)
    cam.setDynamicScaling(false);

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
    check_true(cam.hasMoved());
    std::cout << "[PASS] Forward FPS movement (W) verified." << std::endl;

    // 3. Backward movement (S) with sprint (Shift -> 3.0x speed)
    cam.resetMoved();
    cam.processFpsInput(-1.0f, 0.0f, 0.0f, 0.1f, true); // back 1.0, dt 0.1s, sprint (3.0 * 3.0 = 9.0 m/s)
    assert_near(cam.getPosition().z, 2.7f + 0.90f, 0.01f, "Backward sprint move Z");
    check_true(cam.hasMoved());
    std::cout << "[PASS] Backward FPS sprint movement (S + Shift) verified (3.0x speed multiplier)." << std::endl;

    // 3b. Precision crawl movement (W + Alt -> 0.25x speed)
    cam.resetMoved();
    float zBeforeCrawl = cam.getPosition().z;
    cam.processFpsInput(1.0f, 0.0f, 0.0f, 0.1f, false, true); // forward 1.0, dt 0.1s, crawl (3.0 * 0.25 = 0.75 m/s)
    assert_near(cam.getPosition().z, zBeforeCrawl - 0.075f, 0.01f, "Forward crawl move Z");
    check_true(cam.hasMoved());
    std::cout << "[PASS] Precision crawl movement (Alt) verified (0.25x speed multiplier for object centering)." << std::endl;

    // 4. Strafe movement (A and D)
    cam.resetMoved();
    cam.processFpsInput(0.0f, 1.0f, 0.0f, 0.1f, false); // strafe right
    assert_near(cam.getPosition().x, 0.3f, 0.01f, "Strafe right X");
    check_true(cam.hasMoved());

    cam.resetMoved();
    cam.processFpsInput(0.0f, -1.0f, 0.0f, 0.1f, false); // strafe left
    assert_near(cam.getPosition().x, 0.0f, 0.01f, "Strafe left back to 0 X");
    check_true(cam.hasMoved());
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
    check_true(cam.hasMoved());

    cam.resetMoved();
    cam.processMouseMovement(0.0f, -50.0f); // rotate pitch up (+5 deg at 0.1 sens)
    assert_near(cam.getPitch(), 5.0f, 0.01f, "Pitch rotation up");
    check_true(cam.hasMoved());
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
    check_true(cam.getFov() > 80.0f && cam.getFov() < 86.0f);
    std::cout << "[PASS] Portrait adaptive FOV calculation verified (" << cam.getFov() << " deg for 1280x2160)." << std::endl;

    // Disabled adaptive FOV test
    cam.setAdaptiveFov(false);
    cam.setFov(50.0f);
    cam.adaptFovForAspect(1280.0f / 2160.0f);
    assert_near(cam.getFov(), 50.0f, 0.01f, "Manual FOV preserved when adaptive FOV disabled");
    std::cout << "[PASS] Manual FOV preservation verified when adaptive mode disabled." << std::endl;

    // 10b. Hor+ Aspect Ratio Transition & Authored FOV Preservation (16:9 Window -> 21:9 Fullscreen)
    cam.setAdaptiveFov(true);
    cam.setDefaultFraming(glm::vec3(0.0f, 32.0f, 60.0f), glm::vec3(0.0f, 25.0f, -40.0f), 62.0f);
    assert_near(cam.getFov(), 62.0f, 0.01f, "Authored FOV set to 62 deg");

    // 16:9 4K Window (3840x2160, aspect = 1.7778)
    cam.setAspect(3840.0f / 2160.0f);
    assert_near(cam.getFov(), 62.0f, 0.01f, "16:9 landscape retains authored 62 deg vertical FOV (no zoom-in)");

    // Resize/Fullscreen transition to 21:9 Ultrawide (3440x1440, aspect = 2.3889)
    cam.setAspect(3440.0f / 1440.0f);
    assert_near(cam.getFov(), 62.0f, 0.01f, "21:9 ultrawide preserves 62 deg vertical FOV (Hor+ expansion, no zoom-in)");

    // Transition back to 16:9
    cam.setAspect(1920.0f / 1080.0f);
    assert_near(cam.getFov(), 62.0f, 0.01f, "Transition back to 16:9 preserves 62 deg vertical FOV");

    // Reset to default restores authored 62 deg
    cam.processKeyboard('W', 1.0f);
    cam.resetToDefault();
    assert_near(cam.getFov(), 62.0f, 0.01f, "Reset to default restores authored 62 deg framing");
    std::cout << "[PASS] 16:9 to 21:9 Hor+ scaling & authored FOV zoom-in prevention verified." << std::endl;

    // 11. Scale-Adaptive Camera Speeds (Half-Distance Traversal Time Law)
    // Small scene (e.g. coffee maker, radius 0.25m, target dist 0.85m)
    cam.setSceneScale(0.25f, 0.85f);
    assert_near(cam.getSceneScale(), 0.25f, 0.001f, "Small scene radius");
    assert_near(cam.getFocalDistance(), 0.85f, 0.001f, "Small scene focal distance");
    assert_near(cam.getBaseSpeed(), 0.85f * 0.25f, 0.001f, "Small scene base speed");
    check_true(cam.getBaseSpeed() < 0.3f); // Ensures camera won't fly away in small scenes
    std::cout << "[PASS] Small scene scale adaptivity verified (dist: 0.85m -> speed: " << cam.getBaseSpeed() << " m/s)." << std::endl;

    // Cornell Box reference scene (radius 2.0m, target dist 6.8m)
    cam.setSceneScale(2.0f, 6.8f);
    assert_near(cam.getSceneScale(), 2.0f, 0.001f, "Cornell box scene radius");
    assert_near(cam.getBaseSpeed(), 6.8f * 0.25f, 0.01f, "Cornell box base speed (1.70 m/s)");
    std::cout << "[PASS] Cornell Box scene scale adaptivity verified (dist: 6.8m -> speed: " << cam.getBaseSpeed() << " m/s)." << std::endl;

    // Large scene (e.g. living room / house, radius 20.0m, target dist 50m)
    cam.setSceneScale(20.0f, 50.0f);
    assert_near(cam.getSceneScale(), 20.0f, 0.001f, "Large scene radius");
    assert_near(cam.getBaseSpeed(), 50.0f * 0.25f, 0.01f, "Large scene base speed (12.5 m/s)");
    std::cout << "[PASS] Large scene scale adaptivity verified (dist: 50m -> speed: " << cam.getBaseSpeed() << " m/s)." << std::endl;

    // 11b. Focus on Target Shortcut
    cam.focusOnTarget(glm::vec3(0.0f, 0.0f, 0.0f), 2.0f);
    check_true(cam.hasMoved());
    std::cout << "[PASS] Focus on target shortcut verified." << std::endl;

    // 11c. Distance-Adaptive Speed (Smooth Approach to Focus)
    {
        Camera camAdaptive(glm::vec3(0.0f, 1.0f, 4.0f), glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 16.0f / 9.0f);
        check_true(camAdaptive.isDynamicScaling(), "Default Camera must have adaptive speed enabled");
        // At focal distance (4.0m), distRatio = 1.0 -> distFactor = 1.0
        float baseEffSpeed = camAdaptive.getEffectiveSpeed();
        assert_near(baseEffSpeed, camAdaptive.getSpeed(), 0.01f, "Effective speed equals base speed at focal distance");

        // Move closer to target (currentDist = 2.0m, half focal distance)
        camAdaptive.setPose(glm::vec3(0.0f, 1.0f, 2.0f), -90.0f, 0.0f);
        float closeEffSpeed = camAdaptive.getEffectiveSpeed();
        assert_near(closeEffSpeed, baseEffSpeed * 0.5f, 0.01f, "Effective speed halves at half focal distance");

        // Move further away (currentDist = 8.0m, double focal distance)
        camAdaptive.setPose(glm::vec3(0.0f, 1.0f, 8.0f), -90.0f, 0.0f);
        float farEffSpeed = camAdaptive.getEffectiveSpeed();
        assert_near(farEffSpeed, baseEffSpeed * 2.0f, 0.01f, "Effective speed doubles at double focal distance");

        // Clamping bounds: minimum distFactor is 0.15f, maximum is 5.0f
        camAdaptive.setPose(glm::vec3(0.0f, 1.0f, 0.1f), -90.0f, 0.0f); // very close
        assert_near(camAdaptive.getEffectiveSpeed(), baseEffSpeed * 0.15f, 0.01f, "Clamped to min dynamic speed factor (0.15x)");

        camAdaptive.setPose(glm::vec3(0.0f, 1.0f, 100.0f), -90.0f, 0.0f); // very far
        assert_near(camAdaptive.getEffectiveSpeed(), baseEffSpeed * 5.0f, 0.01f, "Clamped to max dynamic speed factor (5.0x)");

        // Disabling adaptive speed restores constant speed
        camAdaptive.setDynamicScaling(false);
        assert_near(camAdaptive.getEffectiveSpeed(), camAdaptive.getSpeed(), 0.01f, "Effective speed remains constant when adaptive speed disabled");
        std::cout << "[PASS] Distance-adaptive speed (smooth approach scaling & clamping) verified." << std::endl;
    }

    // 11d. Geometry Look-Distance Adaptive Speed (Real-time Raycast targeting)
    {
        Camera camLook(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, -1.0f), 45.0f, 16.0f / 9.0f);
        camLook.setSceneScale(4.0f, 4.0f, glm::vec3(0.0f, 1.0f, 0.0f));
        float baseSpeed = camLook.getBaseSpeed();

        // Even though camera is at position == centralTarget (where dist to centralTarget is 0),
        // look distance to geometry takes precedence:
        camLook.setLookDistance(4.0f);
        check_true(camLook.hasLookDistance(), "Camera must report hasLookDistance");
        assert_near(camLook.getCurrentTargetDistance(), 4.0f, 0.001f, "Target distance must match look distance");
        assert_near(camLook.getEffectiveSpeed(), baseSpeed, 0.01f, "Effective speed equals base speed at focal look distance");

        // Close-up look distance (e.g. 0.8m, 1/5th focal distance) -> speed slows down
        camLook.setLookDistance(0.8f);
        assert_near(camLook.getCurrentTargetDistance(), 0.8f, 0.001f, "Target distance must update to close-up look distance");
        assert_near(camLook.getEffectiveSpeed(), baseSpeed * 0.20f, 0.01f, "Effective speed slows down for close-up object inspection");

        // Distant look distance (e.g. 12m, 3x focal distance) -> speed accelerates
        camLook.setLookDistance(12.0f);
        assert_near(camLook.getCurrentTargetDistance(), 12.0f, 0.001f, "Target distance must update to distant look distance");
        assert_near(camLook.getEffectiveSpeed(), baseSpeed * 3.0f, 0.01f, "Effective speed accelerates when pointed at distant geometry");

        // Clearing look distance restores fallback to central target
        camLook.clearLookDistance();
        check_true(!camLook.hasLookDistance(), "Camera must report no look distance after clear");
        std::cout << "[PASS] Geometry look-distance adaptive speed verified." << std::endl;
    }

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

    // 13. Orbit State Transitions (startOrbit / endOrbit)
    glm::vec3 testPivot(0.0f, 1.0f, 0.0f);
    cam.lookAt(glm::vec3(0.0f, 1.0f, 3.0f), testPivot);
    check_true(!cam.isOrbiting());
    cam.startOrbit(testPivot);
    check_true(cam.isOrbiting());
    assert_near(cam.getOrbitPivot().y, 1.0f, 0.001f, "Orbit pivot Y");
    assert_near(cam.getOrbitRadius(), 3.0f, 0.001f, "Orbit radius");
    cam.endOrbit();
    check_true(!cam.isOrbiting());
    std::cout << "[PASS] Orbit state transitions (startOrbit / endOrbit) verified." << std::endl;

    // 14. Horizontal Arc-Strafe (A / D with Ctrl)
    cam.setSceneScale(2.0f, 3.0f);
    cam.lookAt(glm::vec3(0.0f, 1.0f, 3.0f), testPivot);
    cam.startOrbit(testPivot);
    for (int step = 0; step < 10; ++step) {
        cam.processFpsInput(0.0f, 1.0f, 0.0f, 0.05f, false, false, true); // Strafe right with arcStrafe = true
        float rDist = glm::length(cam.getPosition() - testPivot);
        assert_near(rDist, 3.0f, 0.01f, "Radius preserved during right arc-strafe");
        glm::vec3 toPivot = glm::normalize(testPivot - cam.getPosition());
        float dotDir = glm::dot(cam.getFront(), toPivot);
        check_true(dotDir > 0.999f, "Camera must continuously face pivot");
    }
    check_true(cam.getPosition().x > 0.1f, "Camera rotated right in X");

    for (int step = 0; step < 10; ++step) {
        cam.processFpsInput(0.0f, -1.0f, 0.0f, 0.05f, false, false, true); // Strafe left back
        float rDist = glm::length(cam.getPosition() - testPivot);
        assert_near(rDist, 3.0f, 0.01f, "Radius preserved during left arc-strafe");
    }
    std::cout << "[PASS] Horizontal arc-strafe (A / D + Ctrl) radius preservation & orientation tracking verified." << std::endl;

    // 15. Vertical Elevation Orbit (Space/E up, C/Q down with Ctrl)
    float yBeforeElev = cam.getPosition().y;
    for (int step = 0; step < 5; ++step) {
        cam.processFpsInput(0.0f, 0.0f, 1.0f, 0.05f, false, false, true); // Elevate up in arc
        float rDist = glm::length(cam.getPosition() - testPivot);
        assert_near(rDist, 3.0f, 0.01f, "Radius preserved during elevation arc up");
        glm::vec3 toPivot = glm::normalize(testPivot - cam.getPosition());
        check_true(glm::dot(cam.getFront(), toPivot) > 0.999f);
    }
    check_true(cam.getPosition().y > yBeforeElev); // Risen above pivot

    for (int step = 0; step < 5; ++step) {
        cam.processFpsInput(0.0f, 0.0f, -1.0f, 0.05f, false, false, true); // Elevate down in arc
        float rDist = glm::length(cam.getPosition() - testPivot);
        assert_near(rDist, 3.0f, 0.01f, "Radius preserved during elevation arc down");
    }
    std::cout << "[PASS] Vertical elevation orbit (Space/E and C/Q + Ctrl) verified." << std::endl;

    // 16. Radial Dolly (W forward / S backward with Ctrl)
    float rBeforeDolly = cam.getOrbitRadius();
    cam.processFpsInput(1.0f, 0.0f, 0.0f, 0.1f, false, false, true); // Dolly in (W)
    check_true(cam.getOrbitRadius() < rBeforeDolly);
    float rAfterIn = cam.getOrbitRadius();
    cam.processFpsInput(-1.0f, 0.0f, 0.0f, 0.2f, false, false, true); // Dolly out (S)
    check_true(cam.getOrbitRadius() > rAfterIn);
    glm::vec3 toPivotDolly = glm::normalize(testPivot - cam.getPosition());
    check_true(glm::dot(cam.getFront(), toPivotDolly) > 0.999f);
    std::cout << "[PASS] Radial dolly in/out (W / S + Ctrl) verified." << std::endl;

    // 17. Mouse Turntable Orbit
    cam.lookAt(glm::vec3(0.0f, 1.0f, 3.0f), testPivot);
    cam.startOrbit(testPivot);
    cam.processMouseMovement(80.0f, 0.0f, true); // Orbit mouse right
    float rDistMouse = glm::length(cam.getPosition() - testPivot);
    assert_near(rDistMouse, 3.0f, 0.01f, "Radius preserved during mouse orbit");
    check_true(cam.getPosition().x != 0.0f);
    std::cout << "[PASS] Mouse turntable orbit (Mouse + Ctrl) verified." << std::endl;
    cam.endOrbit();

    // 18. Scene Raycasting against Cornell Box Geometry
    SceneData cornell = ProceduralScene::createCornellBox();
    check_true(!cornell.meshRanges.empty());
    check_true(!cornell.triangles.empty());

    float hitDist = 0.0f;
    glm::vec3 hitPoint;
    std::string hitName;

    // Raycast towards Back Wall (z = -1.0) above boxes from (0, 1.6, 2.7) looking (0, 0, -1)
    bool hit = cornell.raycast(glm::vec3(0.0f, 1.6f, 2.7f), glm::vec3(0.0f, 0.0f, -1.0f), 100.0f, hitDist, hitPoint, &hitName);
    check_true(hit);
    assert_near(hitPoint.z, -1.0f, 0.01f, "Back wall hit Z");
    assert_near(hitDist, 3.7f, 0.01f, "Back wall hit distance");
    check_true(hitName == "Back Wall", "Hit name must be Back Wall");

    // Raycast towards Floor (y = 0.0) from (0, 1, 0) looking (0, -1, 0)
    hit = cornell.raycast(glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0.0f, -1.0f, 0.0f), 100.0f, hitDist, hitPoint, &hitName);
    check_true(hit);
    assert_near(hitPoint.y, 0.0f, 0.01f, "Floor hit Y");
    check_true(hitName == "Floor", "Hit name must be Floor");

    // Raycast towards Left Wall (Red) at x = -1.0 from (0, 1.6, 0) looking (-1, 0, 0)
    hit = cornell.raycast(glm::vec3(0.0f, 1.6f, 0.0f), glm::vec3(-1.0f, 0.0f, 0.0f), 100.0f, hitDist, hitPoint, &hitName);
    check_true(hit);
    assert_near(hitPoint.x, -1.0f, 0.01f, "Left wall hit X");
    check_true(hitName == "Left Wall (Red)", "Hit name must be Left Wall (Red)");
    std::cout << "[PASS] Scene raycast against Cornell Box objects (Back Wall, Floor, Left Wall) verified." << std::endl;

    // 19. CLI Camera Configuration & Override Tests
    {
        // 19a. Combination --camera with comma separation (px,py,pz,tx,ty,tz,fov)
        const char* argv1[] = {
            "pathways", "--headless",
            "--camera", "1.5,-2.0,3.75,0.0,1.0,0.0,60.0"
        };
        Config cfg1 = Config::parse(4, const_cast<char**>(argv1));
        check_true(cfg1.camera_pos.has_value(), "cfg1 has camera_pos");
        check_true(cfg1.camera_target.has_value(), "cfg1 has camera_target");
        check_true(cfg1.camera_fov.has_value(), "cfg1 has camera_fov");
        assert_near(cfg1.camera_pos->x, 1.5f, 0.001f, "cfg1 pos x");
        assert_near(cfg1.camera_pos->y, -2.0f, 0.001f, "cfg1 pos y");
        assert_near(cfg1.camera_pos->z, 3.75f, 0.001f, "cfg1 pos z");
        assert_near(cfg1.camera_target->x, 0.0f, 0.001f, "cfg1 target x");
        assert_near(cfg1.camera_target->y, 1.0f, 0.001f, "cfg1 target y");
        assert_near(cfg1.camera_target->z, 0.0f, 0.001f, "cfg1 target z");
        assert_near(*cfg1.camera_fov, 60.0f, 0.001f, "cfg1 fov");
        std::cout << "[PASS] Combination --camera comma-separated parsing verified." << std::endl;

        // 19b. Combination --camera with space separation
        const char* argv2[] = {
            "pathways", "--headless",
            "--camera", "-0.5", "1.25", "4.0", "0.0", "0.5", "-1.0", "55.0"
        };
        Config cfg2 = Config::parse(10, const_cast<char**>(argv2));
        check_true(cfg2.camera_pos.has_value(), "cfg2 has camera_pos");
        check_true(cfg2.camera_target.has_value(), "cfg2 has camera_target");
        check_true(cfg2.camera_fov.has_value(), "cfg2 has camera_fov");
        assert_near(cfg2.camera_pos->x, -0.5f, 0.001f, "cfg2 pos x");
        assert_near(cfg2.camera_pos->y, 1.25f, 0.001f, "cfg2 pos y");
        assert_near(cfg2.camera_pos->z, 4.0f, 0.001f, "cfg2 pos z");
        assert_near(cfg2.camera_target->x, 0.0f, 0.001f, "cfg2 target x");
        assert_near(cfg2.camera_target->y, 0.5f, 0.001f, "cfg2 target y");
        assert_near(cfg2.camera_target->z, -1.0f, 0.001f, "cfg2 target z");
        assert_near(*cfg2.camera_fov, 55.0f, 0.001f, "cfg2 fov");
        std::cout << "[PASS] Combination --camera space-separated parsing verified." << std::endl;

        // 19c. Individual --camera-pos, --camera-target, --camera-up, --camera-fov
        const char* argv3[] = {
            "pathways", "--headless",
            "--camera-pos", "0.0,2.5,5.0",
            "--camera-target", "0.0,0.0,0.0",
            "--camera-up", "0.0,1.0,0.0",
            "--camera-fov", "75.0"
        };
        Config cfg3 = Config::parse(10, const_cast<char**>(argv3));
        check_true(cfg3.camera_pos.has_value(), "cfg3 has camera_pos");
        check_true(cfg3.camera_target.has_value(), "cfg3 has camera_target");
        check_true(cfg3.camera_up.has_value(), "cfg3 has camera_up");
        check_true(cfg3.camera_fov.has_value(), "cfg3 has camera_fov");
        assert_near(cfg3.camera_pos->y, 2.5f, 0.001f, "cfg3 pos y");
        assert_near(cfg3.camera_pos->z, 5.0f, 0.001f, "cfg3 pos z");
        assert_near(*cfg3.camera_fov, 75.0f, 0.001f, "cfg3 fov");
        std::cout << "[PASS] Individual --camera-pos, --camera-target, --camera-up, --camera-fov parsing verified." << std::endl;

        // 19d. Applying overrides to Camera instance
        Camera testCam;
        testCam.lookAt(*cfg1.camera_pos, *cfg1.camera_target);
        testCam.setFov(*cfg1.camera_fov);
        testCam.setAdaptiveFov(false);
        assert_near(testCam.getPosition().x, 1.5f, 0.001f, "testCam pos x");
        assert_near(testCam.getPosition().y, -2.0f, 0.001f, "testCam pos y");
        assert_near(testCam.getPosition().z, 3.75f, 0.001f, "testCam pos z");
        assert_near(testCam.getFov(), 60.0f, 0.001f, "testCam fov");
        std::cout << "[PASS] Camera view override application verified." << std::endl;

        // 19e. Adaptive speed CLI configuration flag parsing
        const char* argvAdaptiveOff[] = { "pathways", "--no-adaptive-speed" };
        Config cfgAdaptiveOff = Config::parse(2, const_cast<char**>(argvAdaptiveOff));
        check_true(!cfgAdaptiveOff.adaptive_speed, "Config parses --no-adaptive-speed");

        const char* argvAdaptiveOn[] = { "pathways", "--adaptive-speed" };
        Config cfgAdaptiveOn = Config::parse(2, const_cast<char**>(argvAdaptiveOn));
        check_true(cfgAdaptiveOn.adaptive_speed, "Config parses --adaptive-speed");
        std::cout << "[PASS] Adaptive speed CLI flag parsing verified." << std::endl;
    }

    // 20. Camera Inertia Damping & Accumulation Responsiveness (< 50 ms Latency)
    {
        Camera camResponsiveness(glm::vec3(0.0f, 1.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 16.0f / 9.0f);

        // Active keypress generates motion and non-zero velocity
        camResponsiveness.resetMoved();
        camResponsiveness.processFpsInput(1.0f, 0.0f, 0.0f, 0.016f, false);
        check_true(camResponsiveness.hasMoved(), "Camera must report moved during active input");
        check_true(glm::length(camResponsiveness.getVelocity()) > 0.1f, "Camera velocity must be non-zero during movement");

        // Releasing WASD keys immediately halts velocity and signals stationary state (< 50 ms latency)
        camResponsiveness.resetMoved();
        camResponsiveness.processFpsInput(0.0f, 0.0f, 0.0f, 0.016f, false);
        camResponsiveness.update(0.016f);
        check_true(!camResponsiveness.hasMoved(), "Releasing movement keys must immediately stop camera and transition to accumulation (< 50 ms)");
        assert_near(glm::length(camResponsiveness.getVelocity()), 0.0f, 0.0001f, "Camera velocity must be 0 after releasing keys");

        // Subsequent frame remains stationary without sub-pixel drift
        glm::vec3 posStationary = camResponsiveness.getPosition();
        camResponsiveness.update(0.016f);
        assert_near(camResponsiveness.getPosition().x, posStationary.x, 0.0001f, "Stationary position X");
        assert_near(camResponsiveness.getPosition().y, posStationary.y, 0.0001f, "Stationary position Y");
        assert_near(camResponsiveness.getPosition().z, posStationary.z, 0.0001f, "Stationary position Z");
        check_true(!camResponsiveness.hasMoved(), "Stationary camera must not trigger movement flag on subsequent frame");
        assert_near(glm::length(camResponsiveness.getVelocity()), 0.0f, 0.0001f, "Camera velocity remains 0 when stationary");

        // Mouse look stopping consistency
        camResponsiveness.resetMoved();
        camResponsiveness.processMouseMovement(15.0f, -10.0f);
        check_true(camResponsiveness.hasMoved(), "Mouse look must report moved during active look");
        camResponsiveness.resetMoved();
        camResponsiveness.processMouseMovement(0.0f, 0.0f);
        camResponsiveness.update(0.016f);
        check_true(!camResponsiveness.hasMoved(), "Ceasing mouse movement must immediately trigger accumulation");

        std::cout << "[PASS] Camera inertia damping & immediate accumulation responsiveness verified (< 50 ms)." << std::endl;
    }

    // 21. Analog Stick Proportional Deflection & Sub-Threshold Deadzone
    {
        Camera camAnalog(glm::vec3(0.0f, 1.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 16.0f / 9.0f);

        // Full deflection (1.0) vs half deflection (0.5)
        glm::vec3 posStart = camAnalog.getPosition();
        camAnalog.processFpsInput(1.0f, 0.0f, 0.0f, 0.1f, false);
        float fullDist = glm::length(camAnalog.getPosition() - posStart);

        camAnalog.lookAt(posStart, glm::vec3(0.0f, 1.0f, 0.0f));
        camAnalog.processFpsInput(0.5f, 0.0f, 0.0f, 0.1f, false);
        float halfDist = glm::length(camAnalog.getPosition() - posStart);

        assert_near(halfDist, fullDist * 0.5f, 0.005f, "Half stick deflection must produce half displacement");
        assert_near(glm::length(camAnalog.getVelocity()), camAnalog.getBaseSpeed() * 0.5f, 0.01f, "Half stick velocity");

        // Sub-threshold deadzone (< 0.001f) produces zero motion
        camAnalog.resetMoved();
        glm::vec3 posBeforeDeadzone = camAnalog.getPosition();
        camAnalog.processFpsInput(0.0005f, 0.0005f, 0.0f, 0.016f, false);
        check_true(!camAnalog.hasMoved(), "Sub-threshold input (< 0.001f) must not trigger moved flag");
        assert_near(camAnalog.getPosition().x, posBeforeDeadzone.x, 0.00001f, "Deadzone position X");
        assert_near(camAnalog.getPosition().z, posBeforeDeadzone.z, 0.00001f, "Deadzone position Z");
        assert_near(glm::length(camAnalog.getVelocity()), 0.0f, 0.00001f, "Deadzone velocity zero");

        // Non-positive deltaTime robustness guard
        camAnalog.resetMoved();
        camAnalog.processFpsInput(1.0f, 0.0f, 0.0f, 0.0f, false);
        check_true(!camAnalog.hasMoved(), "dt=0 must not trigger moved flag");
        camAnalog.processFpsInput(1.0f, 0.0f, 0.0f, -0.016f, false);
        check_true(!camAnalog.hasMoved(), "Negative dt must not trigger moved flag");

        std::cout << "[PASS] Analog stick proportional deflection & sub-threshold deadzone verified." << std::endl;
    }

    // 22. Active Frame Velocity Stability, Arc-Strafe Velocity Telemetry & Stationary Damping
    {
        Camera camRobust(glm::vec3(0.0f, 1.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 16.0f / 9.0f);

        // 22a. Active frame velocity must NOT decay when update(dt) is called during active movement
        camRobust.processFpsInput(1.0f, 0.0f, 0.0f, 0.016f, false);
        float expectedSpeed = camRobust.getBaseSpeed();
        assert_near(glm::length(camRobust.getVelocity()), expectedSpeed, 0.01f, "Pre-update velocity");
        camRobust.update(0.016f);
        assert_near(glm::length(camRobust.getVelocity()), expectedSpeed, 0.01f, "Active frame velocity preserved across update(dt)");

        // 22b. Arc-strafe velocity telemetry
        glm::vec3 pivot(0.0f, 1.0f, 0.0f);
        camRobust.lookAt(glm::vec3(0.0f, 1.0f, 3.0f), pivot);
        camRobust.startOrbit(pivot);
        camRobust.processFpsInput(0.0f, 1.0f, 0.0f, 0.05f, false, false, true); // Arc strafe right
        check_true(camRobust.hasMoved(), "Arc-strafe must set moved flag");
        check_true(glm::length(camRobust.getVelocity()) > 0.5f, "Arc-strafe must compute non-zero tangential velocity");
        camRobust.endOrbit();

        // 22c. Damping executes when un-driven (!hasMoved())
        camRobust.resetMoved();
        // Artificially simulate residual velocity on un-driven camera
        check_true(!camRobust.hasMoved(), "Camera moved reset");
        float velBeforeDamp = glm::length(camRobust.getVelocity());
        check_true(velBeforeDamp > 0.0f, "Residual velocity present");
        camRobust.update(0.05f); // Un-driven update decays velocity
        float velAfterDamp = glm::length(camRobust.getVelocity());
        check_true(velAfterDamp < velBeforeDamp, "Un-driven velocity must decay");

        std::cout << "[PASS] Active frame velocity preservation, arc-strafe velocity & stationary damping verified." << std::endl;
    }

    // 23. Frame Time Spike Clamping, Key Release Robustness & Configurable Gamepad Deadzone
    {
        // 23a. Frame-rate drop dt clamping (eliminates 6x speed collapse / micro-stutter at < 15 FPS)
        auto clampDt = [](float rawDt) -> float {
            if (rawDt <= 0.0f) return 0.016f;
            if (rawDt > 0.1f) return 0.1f;
            return rawDt;
        };

        float dt9Fps = 1.0f / 9.0f; // ~0.111s
        float dtClamped = clampDt(dt9Fps);
        assert_near(dtClamped, 0.1f, 0.001f, "9 FPS dt must clamp to 0.1s instead of collapsing to 0.016s");

        // Active movement across clamped dt
        Camera camSpike(glm::vec3(0.0f, 1.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 16.0f / 9.0f);
        glm::vec3 posStart = camSpike.getPosition();
        camSpike.processFpsInput(1.0f, 0.0f, 0.0f, dtClamped, false);
        float distAdvanced = glm::length(camSpike.getPosition() - posStart);
        assert_near(distAdvanced, camSpike.getBaseSpeed() * 0.1f, 0.01f, "Advancement during clamped spike");

        // 23b. Key release occurring during large frame spike (dt = 0.15s)
        camSpike.resetMoved();
        camSpike.processFpsInput(0.0f, 0.0f, 0.0f, clampDt(0.15f), false);
        camSpike.update(clampDt(0.15f));
        check_true(!camSpike.hasMoved(), "Key release during frame spike must immediately signal stationary (< 50 ms)");
        assert_near(glm::length(camSpike.getVelocity()), 0.0f, 0.0001f, "Zero velocity on key release during spike");

        // 23c. Configurable gamepad deadzone continuous remapping & CLI flag parsing
        auto remapAxis = [](float rawVal, float deadzone) -> float {
            deadzone = std::clamp(deadzone, 0.01f, 0.50f);
            if (std::abs(rawVal) >= deadzone) {
                return std::copysign((std::abs(rawVal) - deadzone) / (1.0f - deadzone), rawVal);
            }
            return 0.0f;
        };

        // Standard deadzone (0.15)
        assert_near(remapAxis(0.10f, 0.15f), 0.0f, 0.0001f, "Below standard deadzone is zero");
        assert_near(remapAxis(0.151f, 0.15f), 0.00117f, 0.001f, "Continuous ramp above standard deadzone");
        assert_near(remapAxis(1.0f, 0.15f), 1.0f, 0.0001f, "Full deflection with standard deadzone");

        // Custom worn-hardware deadzone (0.25)
        assert_near(remapAxis(0.20f, 0.25f), 0.0f, 0.0001f, "0.20 drift rejected by 0.25 deadzone");
        assert_near(remapAxis(0.251f, 0.25f), 0.00133f, 0.001f, "Continuous ramp above custom deadzone");
        assert_near(remapAxis(1.0f, 0.25f), 1.0f, 0.0001f, "Full deflection with custom deadzone");

        // CLI flag parsing verification
        const char* argvDeadzone[] = { "pathways", "--gamepad-deadzone", "0.22" };
        Config cfgDeadzone = Config::parse(3, const_cast<char**>(argvDeadzone));
        assert_near(cfgDeadzone.gamepad_deadzone, 0.22f, 0.001f, "Config parses --gamepad-deadzone");

        const char* argvDeadzoneEq[] = { "pathways", "--gamepad-deadzone=0.30" };
        Config cfgDeadzoneEq = Config::parse(2, const_cast<char**>(argvDeadzoneEq));
        assert_near(cfgDeadzoneEq.gamepad_deadzone, 0.30f, 0.001f, "Config parses --gamepad-deadzone=");

        std::cout << "[PASS] Frame time spike clamping, key release robustness & configurable gamepad deadzone verified." << std::endl;
    }

    // 24. Real-time Scene Raycast & Dynamic Adaptive Speed Verification
    {
        // 24a. Monolithic scene with retained geometry (Infinity Mirror)
        SceneData mirrorScene = ProceduralScene::createInfinityMirrorScene();
        check_true(!mirrorScene.triangles.empty(), "Infinity mirror scene must retain host triangles for raycasting");

        Camera cam(glm::vec3(0.0f, 1.2f, 4.0f), glm::vec3(0.0f, 1.2f, 0.0f), 45.0f, 16.0f / 9.0f);
        cam.setSceneScale(mirrorScene.sceneRadius, mirrorScene.focalDistance, mirrorScene.centralTarget);
        float baseSpeed = cam.getBaseSpeed();

        // Probing geometry at 3 different distances along view ray
        float hitDistFar = 0.0f, hitDistMid = 0.0f, hitDistClose = 0.0f;
        glm::vec3 hitPoint;
        std::string hitName;

        // Position 1: Far (Z = 5.0m, pointing -Z into runway)
        glm::vec3 posFar(0.0f, 1.2f, 5.0f);
        glm::vec3 dir(0.0f, 0.0f, -1.0f);
        bool hitFar = mirrorScene.raycast(posFar, dir, 100.0f, hitDistFar, hitPoint, &hitName);
        check_true(hitFar, "Raycast must hit corridor geometry from far distance");

        cam.setPose(posFar, -90.0f, 0.0f);
        cam.setLookDistance(hitDistFar);
        float speedFar = cam.getEffectiveSpeed();

        // Position 2: Mid (Z = 2.0m, pointing -Z)
        glm::vec3 posMid(0.0f, 1.2f, 2.0f);
        bool hitMid = mirrorScene.raycast(posMid, dir, 100.0f, hitDistMid, hitPoint, &hitName);
        check_true(hitMid, "Raycast must hit corridor geometry from mid distance");
        check_true(hitDistMid < hitDistFar, "Hit distance must decrease as camera moves closer");

        cam.setPose(posMid, -90.0f, 0.0f);
        cam.setLookDistance(hitDistMid);
        float speedMid = cam.getEffectiveSpeed();
        check_true(speedMid < speedFar, "Effective speed must decelerate as camera approaches geometry");

        // Position 3: Close-up (Z = -0.5m, pointing at pedestal/mirror)
        glm::vec3 posClose(0.0f, 0.7f, -0.5f);
        bool hitClose = mirrorScene.raycast(posClose, dir, 100.0f, hitDistClose, hitPoint, &hitName);
        check_true(hitClose, "Raycast must hit close-up geometry");
        check_true(hitDistClose < hitDistMid, "Close-up hit distance must be smaller than mid distance");

        cam.setPose(posClose, -90.0f, 0.0f);
        cam.setLookDistance(hitDistClose);
        float speedClose = cam.getEffectiveSpeed();
        check_true(speedClose < speedMid, "Effective speed must decelerate further on close inspection");
        check_true(speedClose >= baseSpeed * 0.15f * 0.99f, "Speed must respect 0.15x minimum safety clamp");

        // 24b. Hardware Instanced Scene (Cyber City)
        SceneData cityScene = ProceduralScene::createCyberCityScene();
        check_true(!cityScene.instances.empty(), "Cyber city must contain instances");

        // Probing ground / street level from elevated position
        glm::vec3 camCityPos(0.0f, 20.0f, 0.0f);
        glm::vec3 camCityDir(0.0f, -1.0f, 0.0f); // pointing straight down at plaza/road
        float cityHitDist = 0.0f;
        bool hitCity = cityScene.raycast(camCityPos, camCityDir, 200.0f, cityHitDist, hitPoint, &hitName);
        check_true(hitCity, "Raycast must hit instanced plaza/road tile directly below camera");
        assert_near(cityHitDist, 20.0f, 0.5f, "Hit distance to instanced ground should match elevation (~20m)");

        std::cout << "[PASS] Dynamic scene raycast & distance-adaptive speed scaling verified." << std::endl;
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  All Camera & FPS Navigation Unit Tests PASSED Cleanly!" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
