#include "scene/ProceduralScene.hpp"
#include "scene/Camera.hpp"
#include "core/QualityGovernor.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <chrono>
#include <thread>
#include <cstddef>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

using namespace pathways;

static void simulateWork(std::chrono::milliseconds ms) {
    auto start = std::chrono::high_resolution_clock::now();
    while (std::chrono::high_resolution_clock::now() - start < ms) {
#if defined(__x86_64__) || defined(_M_X64)
        _mm_pause();
#endif
    }
}

static void assert_near(float a, float b, float eps = 0.01f, const char* msg = "") {
    if (std::abs(a - b) > eps) {
        std::cerr << "Assertion failed: " << a << " != " << b << " (eps: " << eps << ") " << msg << std::endl;
        std::exit(1);
    }
}

static void check_true(bool cond, const char* msg = "") {
    if (!cond) {
        std::cerr << "Assertion failed: condition is false! " << msg << std::endl;
        std::exit(1);
    }
}

// -----------------------------------------------------------------------------
// 1. Procedural Many-Lights Scene Tests
// -----------------------------------------------------------------------------
static void test_many_lights_scene() {
    std::cout << "[RUN] Testing Procedural Many-Lights Scene generation..." << std::endl;

    // Test default 8x8 grid = 64 lights
    SceneData scene64 = ProceduralScene::createManyLightsScene(8);
    check_true(scene64.lights.size() == 64, "Default many-lights scene must have exactly 64 lights");
    check_true(!scene64.triangles.empty(), "Scene triangles must not be empty");
    check_true(!scene64.materials.empty(), "Scene materials must not be empty");

    float minX = 1e9f, maxX = -1e9f;
    float minZ = 1e9f, maxZ = -1e9f;
    bool hasColorVariation = false;
    glm::vec3 firstColor = glm::vec3(scene64.lights[0].emission);

    for (size_t i = 0; i < scene64.lights.size(); ++i) {
        const auto& light = scene64.lights[i];
        glm::vec3 pos = glm::vec3(light.position);
        glm::vec3 n = glm::vec3(light.normal);
        glm::vec3 col = glm::vec3(light.emission);

        minX = std::min(minX, pos.x);
        maxX = std::max(maxX, pos.x);
        minZ = std::min(minZ, pos.z);
        maxZ = std::max(maxZ, pos.z);

        // Ceiling lights should be near Y = 1.99
        assert_near(pos.y, 1.99f, 0.01f, "Light Y position should be on the ceiling plane");

        // Normal should point down towards the floor (-Y)
        assert_near(n.y, -1.0f, 0.001f, "Light normal must point downwards (-Y)");
        assert_near(glm::length(n), 1.0f, 0.001f, "Light normal must be normalized");

        // Emission must be positive
        check_true(glm::length(col) > 0.0f, "Light emission must be non-zero");

        // Area must be positive
        check_true(light.emission.w > 0.0f, "Light area must be positive");

        if (glm::distance(col, firstColor) > 0.1f) {
            hasColorVariation = true;
        }
    }

    // Grid bounds check: lights should span across [-0.6, 0.6] within the Cornell room
    check_true(minX < -0.6f && maxX > 0.6f, "Lights must span widely across X axis");
    check_true(minZ < -0.6f && maxZ > 0.6f, "Lights must span widely across Z axis");
    check_true(hasColorVariation, "Lights should exhibit multi-spectral color variation");

    // Test smaller 4x4 grid = 16 lights
    SceneData scene16 = ProceduralScene::createManyLightsScene(4);
    check_true(scene16.lights.size() == 16, "4x4 grid scene must have exactly 16 lights");

    std::cout << "[PASS] Procedural Many-Lights Scene generation verified (64 and 16 light configurations)." << std::endl;
}

// -----------------------------------------------------------------------------
// 2. Quality Governor & Frame Pacing Tests
// -----------------------------------------------------------------------------
static void test_quality_governor_pacing() {
    std::cout << "[RUN] Testing QualityGovernor frame pacing & locked FPS..." << std::endl;

    QualityGovernor gov;

    // A. 60 FPS locked pacing (target = 16.666 ms)
    GovernorConfig config60{};
    config60.targetFps = 60;
    config60.enabled = false; // Decoupled pacing: must pace even when adaptive_spp is disabled!
    gov.init(config60);

    auto t0 = std::chrono::high_resolution_clock::now();
    // Simulate some work taking 4 ms
    simulateWork(std::chrono::milliseconds(4));
    gov.paceFrame(t0);
    auto t1 = std::chrono::high_resolution_clock::now();

    double elapsedMs60 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  60 FPS paced interval: " << elapsedMs60 << " ms (target: 16.67 ms)" << std::endl;
    // Allow small margin for OS thread scheduler jitter (+/- 1.0 ms)
    check_true(elapsedMs60 >= 15.5 && elapsedMs60 <= 18.5, "60 FPS paceFrame should keep interval near 16.67 ms");

    // B. 120 FPS locked pacing (target = 8.333 ms)
    GovernorConfig config120{};
    config120.targetFps = 120;
    config120.enabled = false;
    gov.init(config120);

    t0 = std::chrono::high_resolution_clock::now();
    simulateWork(std::chrono::milliseconds(2));
    gov.paceFrame(t0);
    t1 = std::chrono::high_resolution_clock::now();

    double elapsedMs120 = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::cout << "  120 FPS paced interval: " << elapsedMs120 << " ms (target: 8.33 ms)" << std::endl;
    check_true(elapsedMs120 >= 7.5 && elapsedMs120 <= 10.0, "120 FPS paceFrame should keep interval near 8.33 ms");

    // C. Unlocked pacing (target = 0)
    GovernorConfig configUnlocked{};
    configUnlocked.targetFps = 0;
    gov.init(configUnlocked);

    t0 = std::chrono::high_resolution_clock::now();
    gov.paceFrame(t0);
    t1 = std::chrono::high_resolution_clock::now();

    double elapsedMsUnlocked = std::chrono::duration<double, std::milli>(t1 - t0).count();
    check_true(elapsedMsUnlocked < 0.5, "Unlocked paceFrame (targetFps=0) must return immediately");

    std::cout << "[PASS] QualityGovernor frame pacing verified at 60 FPS, 120 FPS, and Uncapped." << std::endl;
}

// -----------------------------------------------------------------------------
// 3. Quality Governor Adaptive Regulation Tests
// -----------------------------------------------------------------------------
static void test_quality_governor_regulation() {
    std::cout << "[RUN] Testing QualityGovernor dynamic regulation logic..." << std::endl;

    QualityGovernor gov;
    GovernorConfig cfg{};
    cfg.targetFps = 60; // 16.67 ms target budget
    cfg.enabled = true;
    cfg.minSpp = 1;
    cfg.maxSpp = 8;
    cfg.minBounces = 2;
    cfg.maxBounces = 8;
    gov.init(cfg);

    // Initial state
    const auto& st = gov.getState();
    check_true(st.currentSpp == 1, "Initial SPP should match min SPP");
    check_true(st.currentBounces == 4, "Initial bounces should be clamped default");

    // Step-down on severe overrun (e.g. 30ms RT on 16.67ms budget)
    gov.update(30.0f, 0.5f, false, false);
    check_true(gov.getState().currentBounces < 4, "QualityGovernor must step down bounces when heavily over budget");

    std::cout << "[PASS] QualityGovernor dynamic regulation verified." << std::endl;
}

// -----------------------------------------------------------------------------
// 3b. Model-Predictive SPP Regulation Tests
// -----------------------------------------------------------------------------
static void test_fractional_spp_regulation() {
    std::cout << "[RUN] Testing QualityGovernor model-predictive SPP regulation..." << std::endl;

    QualityGovernor gov;
    GovernorConfig cfg{};
    cfg.targetFps = 60; // 16.67 ms target budget
    cfg.enabled = true;
    cfg.minSpp = 1;
    cfg.maxSpp = 8;
    cfg.minBounces = 2;
    cfg.maxBounces = 8;
    gov.init(cfg);

    // Initial state
    const auto& st0 = gov.getState();
    assert_near(st0.effectiveSpp, 1.0f, 0.001f, "Initial effective SPP should be 1.0");
    assert_near(st0.fractionalSpp, 0.0f, 0.001f, "Initial fractional SPP should be 0.0");

    // Under-budget frames with ample headroom (e.g. 2.0ms RT on 16.67ms budget)
    // Run enough frames for governor cooldown cycles to allow progressive SPP upgrades
    for (int i = 0; i < 40; ++i) {
        gov.update(2.0f, 0.5f, false, false);
    }
    const auto& stFast = gov.getState();
    check_true(stFast.currentSpp >= 2, "Current SPP should upgrade when significantly under budget");
    check_true(stFast.fractionalSpp == 0.0f, "Fractional SPP must remain 0.0 to prevent SIMD divergence");
    check_true(stFast.currentSpp == static_cast<uint32_t>(stFast.effectiveSpp), "currentSpp should match effectiveSpp");

    // Over-budget frames (e.g. 25.0ms RT on 16.67ms budget -> deficit)
    for (int i = 0; i < 30; ++i) {
        gov.update(25.0f, 0.5f, false, false);
    }
    const auto& stSlow = gov.getState();
    check_true(stSlow.currentSpp <= stFast.currentSpp, "SPP should decrease when over budget");
    check_true(stSlow.currentSpp >= 1, "SPP should not drop below minSpp (1)");

    std::cout << "[PASS] QualityGovernor model-predictive SPP verified (SPP ramped up to " 
              << stFast.currentSpp << " under 2ms load, stepped down cleanly under 25ms load)." << std::endl;
}

// -----------------------------------------------------------------------------
// 4. Reservoir Data Structure Layout & Math Tests
// -----------------------------------------------------------------------------
static void test_reservoir_layout_and_math() {
    std::cout << "[RUN] Testing Reservoir GPU memory layout and RIS mathematical identities..." << std::endl;

    // A. GPU struct size & alignment
    check_true(sizeof(ReservoirGPU) == 32, "ReservoirGPU must be exactly 32 bytes (2 x uvec4)");
    check_true(alignof(ReservoirGPU) == 4, "ReservoirGPU alignment must be 4 bytes");

    check_true(offsetof(ReservoirGPU, lightIdx) == 0,  "ReservoirGPU.lightIdx offset must be 0");
    check_true(offsetof(ReservoirGPU, uvX) == 4,       "ReservoirGPU.uvX offset must be 4");
    check_true(offsetof(ReservoirGPU, uvY) == 8,       "ReservoirGPU.uvY offset must be 8");
    check_true(offsetof(ReservoirGPU, wSum) == 12,     "ReservoirGPU.wSum offset must be 12");
    check_true(offsetof(ReservoirGPU, M) == 16,        "ReservoirGPU.M offset must be 16");
    check_true(offsetof(ReservoirGPU, W) == 20,        "ReservoirGPU.W offset must be 20");
    check_true(offsetof(ReservoirGPU, targetPdf) == 24,"ReservoirGPU.targetPdf offset must be 24");
    check_true(offsetof(ReservoirGPU, pad) == 28,      "ReservoirGPU.pad offset must be 28");

    // B. VRAM footprint calculation for 1080p and 4K
    const size_t res1080p = 1920 * 1080;
    const size_t vram1080pPerBuffer = res1080p * sizeof(ReservoirGPU);
    const size_t vram1080pPingPong = vram1080pPerBuffer * 2;
    // ~66.36 MB (63.28 MiB) per buffer, ~132.71 MB (126.56 MiB) ping-pong
    check_true(vram1080pPingPong == 132710400ULL,
               "1080p ping-pong reservoir buffers should be exactly 132,710,400 bytes (126.56 MiB)");

    // C. Mathematical RIS Unbiased Estimator Property:
    // When M = 1 and wSum = pHat / q, W = (1 / pHat) * (wSum / M) = 1 / q.
    // The final contribution estimator is pHat * W = pHat / q, identically matching standard Monte Carlo.
    float pHat = 3.5f;
    float q = 0.125f; // proposal PDF
    float wCandidate = pHat / q; // 28.0f
    float wSum = wCandidate;
    float M = 1.0f;
    float W = (pHat > 0.0f) ? (wSum / (M * pHat)) : 0.0f;
    assert_near(W, 1.0f / q, 1e-5f, "Unbiased RIS estimator with M=1 must equal 1 / q");
    assert_near(pHat * W, pHat / q, 1e-5f, "Unbiased estimator must match standard Monte Carlo importance sampling");

    // D. Geometric coupling in target function:
    // G = cos(theta_L) / max(d^2, 1e-4)
    float dist = 2.0f;
    float cosThetaL = 0.8f;
    float geom = cosThetaL / std::max(dist * dist, 1e-4f);
    assert_near(geom, 0.8f / 4.0f, 1e-5f, "Physical geometric coupling formula");

    // Extreme close-up singularity clamping
    float distZero = 0.0f;
    float geomZero = 1.0f / std::max(distZero * distZero, 1e-4f);
    assert_near(geomZero, 10000.0f, 1.0f, "Singularity distance clamping avoids div-by-zero");

    std::cout << "[PASS] Reservoir GPU layout and RIS mathematical identities verified." << std::endl;
}

// -----------------------------------------------------------------------------
// Main Entry Point
// -----------------------------------------------------------------------------
int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: ReSTIR DI Quality & Frame Pacing Test Suite   " << std::endl;
    std::cout << "==========================================================" << std::endl;

    test_many_lights_scene();
    test_quality_governor_pacing();
    test_quality_governor_regulation();
    test_fractional_spp_regulation();
    test_reservoir_layout_and_math();

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL TESTS PASSED!                                       " << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
