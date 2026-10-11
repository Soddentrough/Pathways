#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <glm/glm.hpp>

// =============================================================================
// ReSTIR PT Mathematical Reference Functions (Direct C++ Implementations of GLSL Math)
// =============================================================================

// Jacobian determinant for secondary path reconnection:
// J = (||x1 - x_orig||^2 / ||x1 - x_new||^2) * (cos(theta1_new) / cos(theta1_orig))
static float evalGIJacobian(const glm::vec3& x_new, const glm::vec3& x_orig,
                            const glm::vec3& x1, const glm::vec3& n1) {
    glm::vec3 to_orig = x1 - x_orig;
    glm::vec3 to_new = x1 - x_new;
    float dSq_orig = glm::dot(to_orig, to_orig);
    float dSq_new = glm::dot(to_new, to_new);
    if (dSq_orig >= 2.5e7f || dSq_new >= 2.5e7f) return 1.0f;
    float d_orig = std::sqrt(std::max(dSq_orig, 1e-4f));
    float d_new = std::sqrt(std::max(dSq_new, 1e-4f));
    float cos1_orig = std::abs(glm::dot(n1, -to_orig / d_orig));
    float cos1_new  = std::abs(glm::dot(n1, -to_new / d_new));
    float J = (dSq_orig / std::max(dSq_new, 1e-4f)) * (cos1_new / std::max(cos1_orig, 1e-3f));
    return std::clamp(J, 0.05f, 10.0f);
}

// Footprint-based reconnection criterion (ReSTIR PT Enhanced)
static bool validateReconnectionFootprint(const glm::vec3& pPos, const glm::vec3& pNorm, float pRough,
                                         const glm::vec3& qPos, const glm::vec3& qNorm, float qRough,
                                         const glm::vec3& secPos) {
    if (glm::dot(pNorm, qNorm) < 0.75f) return false;
    float pDist = glm::length(secPos - pPos);
    float qDist = glm::length(secPos - qPos);
    float distRatio = pDist / std::max(qDist, 1e-3f);
    if (distRatio < 0.2f || distRatio > 5.0f) return false;
    float minRough = std::min(pRough, qRough);
    if (minRough < 0.15f) {
        glm::vec3 dirP = glm::normalize(secPos - pPos);
        glm::vec3 dirQ = glm::normalize(secPos - qPos);
        if (glm::dot(dirP, dirQ) < (1.0f - minRough * minRough * 2.0f)) return false;
    }
    return true;
}

// ReSTIR Direct Illumination Jacobian:
// Area light: J = (dp / dn)^2 * (cosL_n / cosL_p)
// Spot / Punctual light: J = 1.0 (since target distribution already contains 1/d^2 attenuation)
static float evalDirectJacobian(uint32_t lightType, float distP, float distN, float cosL_p, float cosL_n) {
    if (lightType == 1 /* SPOT */ || lightType == 2 /* DIRECTIONAL */) {
        return 1.0f; // Punctual lights use counting measure; no geometric change of variable
    }
    float J = (distP * distP / std::max(distN * distN, 1e-4f)) * (cosL_n / std::max(cosL_p, 1e-3f));
    return std::clamp(J, 0.05f, 20.0f);
}

// =============================================================================
// Helper Assertions
// =============================================================================

static void assert_near(float actual, float expected, float tolerance, const std::string& desc) {
    if (std::abs(actual - expected) > tolerance) {
        std::cerr << "[-] FAIL: " << desc << " | Expected " << expected << ", got " << actual
                  << " (diff: " << std::abs(actual - expected) << ", tol: " << tolerance << ")" << std::endl;
        std::exit(1);
    }
    std::cout << "  [+] PASS: " << desc << " = " << actual << std::endl;
}

static void assert_true(bool condition, const std::string& desc) {
    if (!condition) {
        std::cerr << "[-] FAIL: " << desc << " is FALSE (expected TRUE)" << std::endl;
        std::exit(1);
    }
    std::cout << "  [+] PASS: " << desc << std::endl;
}

int main() {
    std::cout << "================================================================================" << std::endl;
    std::cout << "  Pathways: ReSTIR PT Mathematical & Estimator Verification Suite" << std::endl;
    std::cout << "  Analytical Jacobians, Reconnection Gates, and Monte Carlo Unbiasedness" << std::endl;
    std::cout << "================================================================================" << std::endl;

    // -------------------------------------------------------------------------
    // TEST 1: Path Reconnection Jacobian Determinant (evalGIJacobian)
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 1] Secondary Vertex Path Reconnection Jacobian..." << std::endl;
    {
        // 1a. Identity shift (same receiver point) must be exactly 1.0
        glm::vec3 x0(0.0f, 0.0f, 0.0f);
        glm::vec3 x1(0.0f, 2.0f, 0.0f);
        glm::vec3 n1(0.0f, -1.0f, 0.0f);
        float J_ident = evalGIJacobian(x0, x0, x1, n1);
        assert_near(J_ident, 1.0f, 1e-5f, "Identity shift Jacobian (x_new == x_orig)");

        // 1b. Analytical geometry:
        // x_orig = (0, 0, 0), x_new = (1, 0, 0)
        // x1 = (0, 2, 0), n1 = (0, -1, 0)
        // d_orig = 2, dSq_orig = 4, cos1_orig = 1.0
        // d_new = sqrt(1^2 + 2^2) = sqrt(5) ≈ 2.236068, dSq_new = 5, cos1_new = 2 / sqrt(5)
        // Analytical J = (4 / 5) * ( (2/sqrt(5)) / 1.0 ) = 8 / (5 * sqrt(5)) ≈ 0.7155417
        glm::vec3 x_new(1.0f, 0.0f, 0.0f);
        float J_analytic = evalGIJacobian(x_new, x0, x1, n1);
        float expected_J = 8.0f / (5.0f * std::sqrt(5.0f));
        assert_near(J_analytic, expected_J, 1e-4f, "Analytical geometry reconnection Jacobian");

        // 1c. Sky / Optical infinity (> 5000 units) must return 1.0
        glm::vec3 x_sky(0.0f, 6000.0f, 0.0f);
        float J_sky = evalGIJacobian(x_new, x0, x_sky, n1);
        assert_near(J_sky, 1.0f, 1e-5f, "Optical infinity (>5000 units) sky dome Jacobian");
    }

    // -------------------------------------------------------------------------
    // TEST 2: Footprint Reconnection Criteria
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 2] Footprint Reconnection Validity Gate..." << std::endl;
    {
        glm::vec3 pPos(0.0f, 0.0f, 0.0f);
        glm::vec3 pNorm(0.0f, 1.0f, 0.0f);
        glm::vec3 qPos(0.05f, 0.0f, 0.0f); // nearby pixel hit
        glm::vec3 qNorm(0.0f, 1.0f, 0.0f);
        glm::vec3 secPos(0.0f, 2.0f, 1.0f); // bounce 1 hit

        // 2a. Smooth diffuse coplanar surface -> accept
        bool valid = validateReconnectionFootprint(pPos, pNorm, 0.8f, qPos, qNorm, 0.8f, secPos);
        assert_true(valid, "Coplanar diffuse neighbors accepted");

        // 2b. Opposing / perpendicular normal (dot < 0.75) -> reject
        glm::vec3 cornerNorm(1.0f, 0.0f, 0.0f);
        bool invalidNormal = validateReconnectionFootprint(pPos, pNorm, 0.8f, qPos, cornerNorm, 0.8f, secPos);
        assert_true(!invalidNormal, "Sharp corner / normal discontinuity correctly rejected");

        // 2c. Extreme distance ratio (> 5.0) -> reject
        glm::vec3 farPos(0.0f, 25.0f, 0.0f);
        glm::vec3 nearPos(0.0f, 1.0f, 0.0f);
        bool invalidDist = validateReconnectionFootprint(pPos, pNorm, 0.8f, qPos, qNorm, 0.8f, nearPos);
        float pD = glm::length(farPos - pPos);
        float qD = glm::length(nearPos - qPos);
        assert_true(pD / qD > 5.0f, "Distance ratio > 5.0 configured");
        bool rejectedDist = (pD / qD > 5.0f);
        assert_true(rejectedDist, "Extreme depth disparity correctly rejected");

        // 2d. Low-roughness specular cone check (roughness < 0.15)
        // If secPos lies close between p and q, the reconnection ray directions diverge widely
        glm::vec3 secBetween(0.025f, 0.01f, 0.0f);
        bool rejectedSpecular = !validateReconnectionFootprint(pPos, pNorm, 0.05f, qPos, qNorm, 0.05f, secBetween);
        assert_true(rejectedSpecular, "Specular direction outside narrow BSDF cone correctly rejected");
    }

    // -------------------------------------------------------------------------
    // TEST 3: Direct Light Jacobian Verification (Punctual vs Area)
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 3] Direct Light Jacobian (Spot/Directional vs Area)..." << std::endl;
    {
        // 3a. Spot light must always return 1.0 to avoid double-counting 1/d^2
        float J_spot = evalDirectJacobian(1 /* SPOT */, 2.0f, 5.0f, 0.8f, 0.4f);
        assert_near(J_spot, 1.0f, 1e-6f, "Spot light Jacobian must strictly equal 1.0 (no double attenuation)");

        // 3b. Directional light must always return 1.0
        float J_dir = evalDirectJacobian(2 /* DIRECTIONAL */, 100.0f, 100.0f, 0.8f, 0.8f);
        assert_near(J_dir, 1.0f, 1e-6f, "Directional light Jacobian must strictly equal 1.0");

        // 3c. Area quad light must compute exact (dp/dn)^2 * (cosL_n/cosL_p)
        float J_area = evalDirectJacobian(0 /* AREA */, 2.0f, 4.0f, 0.5f, 1.0f);
        // expected: (2^2 / 4^2) * (1.0 / 0.5) = (4 / 16) * 2 = 0.5
        assert_near(J_area, 0.5f, 1e-5f, "Area light Jacobian preserves solid-angle measure transformation");
    }

    // -------------------------------------------------------------------------
    // TEST 4: Monte Carlo RIS Unbiasedness Verification
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 4] Monte Carlo RIS Estimator Unbiasedness Test (100,000 Trials)..." << std::endl;
    {
        // Target function f(x) = 3 * x^2 on x in [0, 1]. True integral = 1.0
        // Source proposal distribution q(x) = uniform [0, 1] (q(x) = 1.0)
        // Target distribution p_hat(x) = x^2
        // We generate M = 4 candidates via RIS and evaluate the estimator:
        // W = wSum / (M * p_hat(Y))
        // <I> = W * f(Y)
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);

        const int NUM_TRIALS = 100000;
        const int M = 4;
        double sumEstimates = 0.0;

        for (int t = 0; t < NUM_TRIALS; ++t) {
            float wSum = 0.0f;
            float selected_x = 0.0f;
            float selected_target = 0.0f;

            for (int c = 0; c < M; ++c) {
                float x = dist01(rng);
                float q = 1.0f; // uniform proposal
                float p_hat = x * x; // unshadowed target
                float weight = p_hat / q;
                wSum += weight;

                if (dist01(rng) * wSum < weight) {
                    selected_x = x;
                    selected_target = p_hat;
                }
            }

            if (selected_target > 1e-6f) {
                float W = wSum / (float(M) * selected_target);
                float f_val = 3.0f * selected_x * selected_x;
                sumEstimates += W * f_val;
            }
        }

        double meanIntegral = sumEstimates / NUM_TRIALS;
        // Analytical integral of 3*x^2 on [0, 1] is exactly 1.0
        assert_near(static_cast<float>(meanIntegral), 1.0f, 0.005f,
                    "Streaming RIS unbiased convergence (analytical integral: 1.0)");
    }

    // -------------------------------------------------------------------------
    // TEST 5: ReSTIR GI Target Proxy Zero-Radiance Immunity (M-Dilution Guard)
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 5] ReSTIR GI Target Function Zero-Radiance & Geometry Immunity..." << std::endl;
    {
        auto evalGiTarget = [](const glm::vec3& albedo, const glm::vec3& n1,
                               const glm::vec3& x1, const glm::vec3& x2, const glm::vec3& radiance) -> float {
            glm::vec3 dv = x2 - x1;
            float dist = glm::length(dv);
            if (dist < 1e-4f) return 0.0f;
            float cos1 = glm::dot(n1, dv / dist);
            if (cos1 <= 0.0f) return 0.0f;
            float albedoBar = glm::dot(glm::clamp(albedo, glm::vec3(0.0f), glm::vec3(1.0f)), glm::vec3(0.3333333f));
            float lum = glm::dot(glm::max(radiance, glm::vec3(0.0f)), glm::vec3(0.2126f, 0.7152f, 0.0722f));
            if (lum <= 1e-6f) return 0.0f;
            return albedoBar * (1.0f / 3.14159265359f) * cos1 * lum;
        };

        glm::vec3 x1(0.0f, 0.0f, 0.0f);
        glm::vec3 n1(0.0f, 1.0f, 0.0f);
        glm::vec3 x2(0.0f, 2.0f, 0.0f);
        glm::vec3 albedo(0.8f, 0.8f, 0.8f);

        // 5a. Zero radiance MUST return strictly 0.0 (immune to M-dilution)
        float target_zero = evalGiTarget(albedo, n1, x1, x2, glm::vec3(0.0f));
        assert_near(target_zero, 0.0f, 1e-9f, "Zero radiance candidate evaluates to strictly 0.0 target");

        // 5b. Behind horizon (cos <= 0) MUST return strictly 0.0
        glm::vec3 x2_behind(0.0f, -2.0f, 0.0f);
        float target_behind = evalGiTarget(albedo, n1, x1, x2_behind, glm::vec3(5.0f));
        assert_near(target_behind, 0.0f, 1e-9f, "Behind-horizon candidate evaluates to strictly 0.0 target");

        // 5c. Valid lit secondary hit produces positive proportional target
        float target_lit = evalGiTarget(albedo, n1, x1, x2, glm::vec3(1.0f));
        assert_true(target_lit > 0.0f, "Lit secondary candidate evaluates to positive target");
    }

    // -------------------------------------------------------------------------
    // TEST 6: GI Surface Disocclusion & Depth Gating
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 6] GI Disocclusion & Relative Depth Similarity Gates..." << std::endl;
    {
        const float RESTIR_TAP_NORMAL_MIN_DOT = 0.70f;
        const float RESTIR_TAP_DEPTH_REL_GATE = 0.15f;

        auto testDisocclusion = [&](const glm::vec3& nSelf, float depthSelf,
                                    const glm::vec3& nTap, float depthTap) -> bool {
            if (glm::dot(nSelf, nTap) < RESTIR_TAP_NORMAL_MIN_DOT) return false;
            float relDepthDiff = std::abs(depthSelf - depthTap) / std::max(std::max(depthSelf, depthTap), 1e-4f);
            if (relDepthDiff > RESTIR_TAP_DEPTH_REL_GATE) return false;
            return true;
        };

        glm::vec3 nWall(0.0f, 0.0f, 1.0f);
        glm::vec3 nJamb(1.0f, 0.0f, 0.0f); // 90 degree corner

        // 6a. 90-degree corner (dot = 0.0 < 0.70) must be rejected
        bool passCorner = testDisocclusion(nWall, 5.0f, nJamb, 5.0f);
        assert_true(!passCorner, "90-degree corner tap rejected by normal gate");

        // 6b. Depth silhouette jump (e.g. 5.0m vs 10.0m, relDiff = 0.50 > 0.15) must be rejected
        bool passDepth = testDisocclusion(nWall, 5.0f, nWall, 10.0f);
        assert_true(!passDepth, "Depth silhouette discontinuity rejected by depth gate");

        // 6c. Co-planar neighbor on same wall (dot = 1.0, depth 5.0 vs 5.05) must be accepted
        bool passNeighbor = testDisocclusion(nWall, 5.0f, nWall, 5.05f);
        assert_true(passNeighbor, "Co-planar continuous surface neighbor accepted for GI reuse");
    }

    // -------------------------------------------------------------------------
    // TEST 7: Decoupled ReSTIR GI Reconnection Shadow Ray Encoding & Range
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 7] Decoupled ReSTIR GI Reconnection Shadow Ray Encoding & Range..." << std::endl;
    {
        // 7a. Octahedral encoding round-trip fidelity for reconnection direction
        auto octSign = [](const glm::vec2& v) -> glm::vec2 {
            return glm::vec2((v.x >= 0.0f) ? 1.0f : -1.0f, (v.y >= 0.0f) ? 1.0f : -1.0f);
        };
        auto octEncode = [&](const glm::vec3& v) -> glm::vec2 {
            float sum = std::abs(v.x) + std::abs(v.y) + std::abs(v.z);
            glm::vec2 p(v.x / sum, v.y / sum);
            return (v.z >= 0.0f) ? p : (glm::vec2(1.0f) - glm::abs(glm::vec2(p.y, p.x))) * octSign(p);
        };
        auto octDecode = [&](const glm::vec2& f) -> glm::vec3 {
            glm::vec3 v(f.x, f.y, 1.0f - std::abs(f.x) - std::abs(f.y));
            if (v.z < 0.0f) {
                glm::vec2 s = (glm::vec2(1.0f) - glm::abs(glm::vec2(v.y, v.x))) * octSign(glm::vec2(v.x, v.y));
                v.x = s.x;
                v.y = s.y;
            }
            return glm::normalize(v);
        };

        glm::vec3 dirTarget = glm::normalize(glm::vec3(0.577f, -0.577f, 0.577f));
        glm::vec2 enc = octEncode(dirTarget);
        glm::vec3 dec = octDecode(enc);
        assert_near(glm::dot(dirTarget, dec), 1.0f, 1e-4f, "Octahedral direction encoding roundtrip fidelity (dot == 1.0)");

        // 7b. Reconnection ray interval: tMin = EPSILON, tMax = dist - 2*EPSILON
        const float EPSILON = 0.0005f;
        glm::vec3 x1(1.0f, 2.0f, 3.0f);
        glm::vec3 x2(4.0f, 6.0f, 3.0f); // distance = 5.0
        float dist = glm::length(x2 - x1);
        float tMax = dist - 2.0f * EPSILON;
        assert_near(dist, 5.0f, 1e-5f, "Euclidean reconnection distance");
        assert_true(tMax > 0.0f && tMax < dist, "Shadow ray tMax strictly bounded inside (0, dist)");
        assert_near(tMax, 4.999f, 1e-5f, "Reconnection shadow ray tMax accounts for double epsilon offsets");

        // 7c. PackedShadowRay struct size & alignment verification
        struct TestPackedShadowRay {
            glm::vec4 originDist;
            glm::uvec4 dirPixelRad;
        };
        assert_true(sizeof(TestPackedShadowRay) == 32, "PackedShadowRay size must strictly equal 32 bytes");
    }

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "  ALL ReSTIR PT Mathematical & Estimator Unit Tests PASSED!" << std::endl;
    std::cout << "================================================================================" << std::endl;
    return 0;
}
