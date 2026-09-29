#include "scene/CameraPath.hpp"
#include "scene/Camera.hpp"
#include "core/Logger.hpp"

#include <iostream>
#include <cassert>
#include <cmath>

using namespace pathways;

static void assert_near(float a, float b, float eps = 0.01f, const char* msg = "") {
    if (std::abs(a - b) > eps) {
        std::cerr << "Assertion failed: " << a << " != " << b << " (eps: " << eps << ") " << msg << std::endl;
        std::exit(1);
    }
}

static void assert_vec3_near(const glm::vec3& a, const glm::vec3& b, float eps = 0.01f, const char* msg = "") {
    float dist = glm::length(a - b);
    if (dist > eps) {
        std::cerr << "Assertion failed: (" << a.x << "," << a.y << "," << a.z << ") != ("
                  << b.x << "," << b.y << "," << b.z << ") dist=" << dist << " (eps: " << eps << ") " << msg << std::endl;
        std::exit(1);
    }
}

static void check_true(bool cond, const char* msg = "") {
    if (!cond) {
        std::cerr << "Assertion failed: condition is false! " << msg << std::endl;
        std::exit(1);
    }
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing Space/Time Camera Paths & Splines     " << std::endl;
    std::cout << "  Waypoints, Catmull-Rom, Bezier & Motion Vector Invariants" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // -------------------------------------------------------------------------
    // TEST 1: Centripetal Catmull-Rom Exact Waypoint Passing
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 1] Catmull-Rom Exact Waypoint Passing ($C^1$ Continuity)..." << std::endl;
        CameraPath path("CatmullTest", PathInterpolation::CatmullRom);

        CameraKeyframe kf0{ 0.0f, glm::vec3(0.0f, 1.0f, 5.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0,1,0), 45.0f };
        CameraKeyframe kf1{ 2.0f, glm::vec3(3.0f, 2.0f, 2.0f), glm::vec3(1.0f, 1.0f, 0.0f), glm::vec3(0,1,0), 50.0f };
        CameraKeyframe kf2{ 4.0f, glm::vec3(1.0f, 1.5f, -2.0f), glm::vec3(0.0f, 1.0f, -1.0f), glm::vec3(0,1,0), 55.0f };
        CameraKeyframe kf3{ 6.0f, glm::vec3(-2.0f, 1.0f, 0.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0,1,0), 45.0f };

        path.addKeyframe(kf0);
        path.addKeyframe(kf1);
        path.addKeyframe(kf2);
        path.addKeyframe(kf3);

        check_true(path.isValid(), "Path must be valid with 4 keyframes");
        assert_near(path.getDuration(), 6.0f, 0.001f, "Total path duration must be 6.0s");

        // Evaluated position at exact keyframe times must match keyframe positions exactly
        CameraSample s0 = path.evaluate(0.0f);
        assert_vec3_near(s0.position, kf0.position, 1e-4f, "Keyframe 0 position");
        assert_near(s0.fov, 45.0f, 1e-4f, "Keyframe 0 FOV");

        CameraSample s1 = path.evaluate(2.0f);
        assert_vec3_near(s1.position, kf1.position, 1e-4f, "Keyframe 1 position");
        assert_near(s1.fov, 50.0f, 1e-4f, "Keyframe 1 FOV");

        CameraSample s2 = path.evaluate(4.0f);
        assert_vec3_near(s2.position, kf2.position, 1e-4f, "Keyframe 2 position");

        CameraSample s3 = path.evaluate(6.0f);
        assert_vec3_near(s3.position, kf3.position, 1e-4f, "Keyframe 3 position");

        // Midway sample between kf0 and kf1 (t = 1.0s)
        CameraSample s_mid = path.evaluate(1.0f);
        check_true(s_mid.speed > 0.5f, "Midway speed must be non-zero");
        check_true(!s_mid.isStationary, "Midway sample is not stationary");
        assert_near(s_mid.fov, 47.5f, 0.1f, "Midway FOV must be linearly blended");

        std::cout << "  -> Catmull-Rom exact waypoint interpolation verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // TEST 2: Centripetal Parameterization Prevents Wild Overshoots
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 2] Centripetal Spline Corner Boundedness (No Cusps/Overshoots)..." << std::endl;
        CameraPath path("SharpCorner", PathInterpolation::CatmullRom);

        // Sharp 90-degree corner
        path.addKeyframe({ 0.0f, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0,0,1) });
        path.addKeyframe({ 1.0f, glm::vec3(10.0f, 0.0f, 0.0f), glm::vec3(10,0,1) });
        path.addKeyframe({ 2.0f, glm::vec3(10.0f, 10.0f, 0.0f), glm::vec3(10,10,1) });

        // Evaluate 50 samples along the corner
        for (int i = 0; i <= 50; ++i) {
            float t = (float)i / 25.0f; // 0.0 to 2.0
            CameraSample s = path.evaluate(t);
            // Centripetal Catmull-Rom must stay within reasonable bounds around [0..10, 0..10]
            check_true(s.position.x >= -1.0f && s.position.x <= 12.0f, "X bound");
            check_true(s.position.y >= -1.0f && s.position.y <= 12.0f, "Y bound");
        }
        std::cout << "  -> Centripetal Catmull-Rom corner stability verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // TEST 3: Cubic Bézier Curve Evaluation & Analytical Derivatives
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 3] Cubic Bézier Curve & Analytical Velocity..." << std::endl;
        CameraPath path("BezierTest", PathInterpolation::Bezier);

        CameraKeyframe kf0{};
        kf0.time = 0.0f;
        kf0.position = glm::vec3(0.0f, 0.0f, 0.0f);
        kf0.outTangent = glm::vec3(1.0f, 2.0f, 0.0f); // Control point 1: (1, 2, 0)
        kf0.target = glm::vec3(0.0f, 0.0f, 1.0f);

        CameraKeyframe kf1{};
        kf1.time = 2.0f;
        kf1.position = glm::vec3(4.0f, 0.0f, 0.0f);
        kf1.inTangent = glm::vec3(-1.0f, 2.0f, 0.0f); // Control point 2: (3, 2, 0)
        kf1.target = glm::vec3(4.0f, 0.0f, 1.0f);

        path.addKeyframe(kf0);
        path.addKeyframe(kf1);

        CameraSample s0 = path.evaluate(0.0f);
        assert_vec3_near(s0.position, glm::vec3(0.0f), 1e-4f, "Bezier start point");

        CameraSample s1 = path.evaluate(2.0f);
        assert_vec3_near(s1.position, glm::vec3(4.0f, 0.0f, 0.0f), 1e-4f, "Bezier end point");

        // At midpoint u = 0.5 (t = 1.0s), B(0.5) = 1/8(P0) + 3/8(C0) + 3/8(C1) + 1/8(P1)
        // C0 = (1, 2, 0), C1 = (3, 2, 0)
        // B(0.5) = 3/8*(1,2,0) + 3/8*(3,2,0) + 1/8*(4,0,0) = (3/8+9/8+4/8, 6/8+6/8, 0) = (2.0, 1.5, 0.0)
        CameraSample s_mid = path.evaluate(1.0f);
        assert_vec3_near(s_mid.position, glm::vec3(2.0f, 1.5f, 0.0f), 1e-3f, "Bezier midpoint");

        // Initial velocity B'(0) / dt: 3*(C0 - P0) / 2.0s = 3*(1, 2, 0) / 2.0 = (1.5, 3.0, 0.0)
        assert_vec3_near(s0.velocity, glm::vec3(1.5f, 3.0f, 0.0f), 1e-2f, "Bezier initial velocity");

        std::cout << "  -> Cubic Bézier position and velocity derivatives verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // TEST 4: Linear Interpolation Mode
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 4] Piecewise Linear Interpolation..." << std::endl;
        CameraPath path("LinearTest", PathInterpolation::Linear);
        path.addKeyframe({ 0.0f, glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0,0,1) });
        path.addKeyframe({ 4.0f, glm::vec3(8.0f, 4.0f, 0.0f), glm::vec3(8,4,1) });

        CameraSample s = path.evaluate(2.0f);
        assert_vec3_near(s.position, glm::vec3(4.0f, 2.0f, 0.0f), 1e-4f, "Linear midpoint");
        assert_vec3_near(s.velocity, glm::vec3(2.0f, 1.0f, 0.0f), 0.05f, "Linear constant velocity");
        std::cout << "  -> Linear mode verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // TEST 5: Looping vs Boundary Clamping
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 5] Looping Wrap-Around & Boundary Clamping..." << std::endl;
        CameraPath path("LoopTest", PathInterpolation::Linear);
        path.addKeyframe({ 0.0f, glm::vec3(1.0f, 0.0f, 0.0f), glm::vec3(0,0,0) });
        path.addKeyframe({ 10.0f, glm::vec3(5.0f, 0.0f, 0.0f), glm::vec3(0,0,0) });

        // Clamped (loop = false)
        CameraSample s_clamp = path.evaluate(15.0f, false);
        assert_vec3_near(s_clamp.position, glm::vec3(5.0f, 0.0f, 0.0f), 1e-4f, "Clamped at end");

        // Looped (loop = true): at t = 12.0s, equivalent to t = 2.0s
        // Pos at t = 2.0 is lerp(1, 5, 0.2) = 1 + 0.8 = 1.8
        CameraSample s_loop = path.evaluate(12.0f, true);
        assert_vec3_near(s_loop.position, glm::vec3(1.8f, 0.0f, 0.0f), 1e-3f, "Looped at t=12s (t=2s)");

        std::cout << "  -> Loop and clamp boundary conditions verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // TEST 6: Sprint & Sudden Halt (Stationary Hold Detection)
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 6] Stationary Hold & Accumulation Settling Detection..." << std::endl;
        CameraPath path("SprintHalt", PathInterpolation::Linear);

        // Sprinting forward from t=0 to t=2, then holding dead still from t=2 to t=5
        path.addKeyframe({ 0.0f, glm::vec3(0.0f, 1.0f, 10.0f), glm::vec3(0,1,0) });
        path.addKeyframe({ 2.0f, glm::vec3(0.0f, 1.0f, 2.0f), glm::vec3(0,1,0) });
        path.addKeyframe({ 5.0f, glm::vec3(0.0f, 1.0f, 2.0f), glm::vec3(0,1,0) }); // Stationary hold

        CameraSample s_sprint = path.evaluate(1.0f);
        check_true(!s_sprint.isStationary, "Moving during sprint phase");
        assert_near(s_sprint.speed, 4.0f, 0.1f, "Sprint speed must be 4 m/s");

        CameraSample s_halt = path.evaluate(3.5f);
        check_true(s_halt.isStationary, "Camera must be flagged stationary during hold phase");
        assert_near(s_halt.speed, 0.0f, 1e-4f, "Hold speed must be 0 m/s");
        assert_vec3_near(s_halt.position, glm::vec3(0.0f, 1.0f, 2.0f), 1e-4f, "Position locked");

        std::cout << "  -> Sprint and stationary hold transitions verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // TEST 7: Camera Pose & Motion Vector Invariant (m_hasPrevViewProj Preservation)
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 7] Non-Zero Motion Vector Invariant (m_hasPrevViewProj Preservation)..." << std::endl;
        Camera cam(glm::vec3(0.0f, 1.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 16.0f / 9.0f);

        // 1. Initial lookAt should reset m_hasPrevViewProj = false
        cam.lookAt(glm::vec3(0, 1, 3), glm::vec3(0, 1, 0));
        CameraUniform u0 = cam.getUniformData(0, 1, 4, 0, false, 1920, 1080, 0, true);
        // On initial frame with no prior view, prevViewProj equals current unjitteredViewProj
        assert_vec3_near(glm::vec3(u0.prevViewProj[3]), glm::vec3(u0.unjitteredViewProj[3]), 1e-4f, "Frame 0 has no velocity");

        // 2. Animate to new position via setAnimatedPose
        cam.setAnimatedPose(glm::vec3(1.0f, 1.0f, 3.0f), glm::vec3(0.0f, 1.0f, 0.0f), glm::vec3(0,1,0), 45.0f);
        check_true(cam.hasMoved(), "Camera marked moved");

        // Frame 1: getUniformData must preserve previous frame's matrix
        CameraUniform u1 = cam.getUniformData(1, 1, 4, 0, false, 1920, 1080, 0, true);
        // u1.prevViewProj must equal u0.unjitteredViewProj (from frame 0)!
        assert_vec3_near(glm::vec3(u1.prevViewProj[3]), glm::vec3(u0.unjitteredViewProj[3]), 1e-4f, "prevViewProj matches Frame 0 unjittered");

        // And u1.prevViewProj must NOT equal u1.unjitteredViewProj (meaning non-zero motion vector!)
        float deltaPos = glm::length(glm::vec3(u1.prevViewProj[3]) - glm::vec3(u1.unjitteredViewProj[3]));
        check_true(deltaPos > 0.01f, "Motion vector delta must be non-zero during camera animation");

        std::cout << "  -> Animated camera motion vector tracking verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // TEST 8: JSON Serialization & Round-Trip Deserialization
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 8] JSON Serialization & Round-Trip Deserialization..." << std::endl;
        CameraPath pathOriginal("LivingRoomTour", PathInterpolation::CatmullRom);
        pathOriginal.setLoop(true);

        pathOriginal.addKeyframe({ 0.0f, glm::vec3(5.105f, 0.731f, -2.318f), glm::vec3(-0.255f, 1.146f, -0.849f), glm::vec3(0,1,0), 58.7f });
        pathOriginal.addKeyframe({ 3.0f, glm::vec3(3.200f, 0.850f, -1.500f), glm::vec3(0.000f, 1.000f, -0.500f), glm::vec3(0,1,0), 58.7f });
        pathOriginal.addKeyframe({ 6.0f, glm::vec3(1.500f, 0.900f, -0.800f), glm::vec3(0.500f, 0.900f, -0.200f), glm::vec3(0,1,0), 55.0f });

        std::string jsonStr = pathOriginal.serializeToJSON();
        check_true(!jsonStr.empty(), "JSON output must not be empty");

        auto pathDeserialized = CameraPath::loadFromString(jsonStr);
        check_true(pathDeserialized != nullptr, "Deserialized path must not be null");
        check_true(pathDeserialized->isValid(), "Deserialized path must be valid");
        check_true(pathDeserialized->getName() == "LivingRoomTour", "Name must match");
        check_true(pathDeserialized->isLoop() == true, "Loop flag must match");
        check_true(pathDeserialized->getInterpolation() == PathInterpolation::CatmullRom, "Interpolation mode must match");
        check_true(pathDeserialized->getKeyframeCount() == 3, "Keyframe count must be 3");

        // Verify evaluation parity between original and deserialized across 10 evaluation points
        for (int i = 0; i <= 10; ++i) {
            float t = (float)i * 0.6f;
            CameraSample sOrig = pathOriginal.evaluate(t);
            CameraSample sDeser = pathDeserialized->evaluate(t);

            assert_vec3_near(sOrig.position, sDeser.position, 1e-4f, "Position round-trip match");
            assert_vec3_near(sOrig.target, sDeser.target, 1e-4f, "Target round-trip match");
            assert_near(sOrig.fov, sDeser.fov, 1e-4f, "FOV round-trip match");
            assert_near(sOrig.speed, sDeser.speed, 1e-3f, "Speed round-trip match");
        }

        std::cout << "  -> JSON serialization round-trip verified with 100% numerical parity." << std::endl;
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << " [SUCCESS] All Camera Path Spline Invariants Passed! (8/8)" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
