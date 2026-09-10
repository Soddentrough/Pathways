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
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  All Camera & FPS Navigation Unit Tests PASSED Cleanly!" << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
