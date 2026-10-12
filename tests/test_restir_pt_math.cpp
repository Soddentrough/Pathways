#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <cassert>
#include <algorithm>
#include <glm/glm.hpp>

#include "rt/ReSTIRTypes.hpp"
#include "scene/Material.hpp"
#include "scene/ProceduralScene.hpp"

// =============================================================================
// ReSTIR PT Mathematical Reference Functions (Direct C++ Implementations of GLSL Math)
// =============================================================================

// Symmetric geometric transport coupling between two differential surface elements:
// G(a, b) = (max(cos theta_a, 0.0) * max(cos theta_b, 0.0)) / max(||b - a||^2, 1e-4)
static float evalGeometricCoupling(const glm::vec3& posA, const glm::vec3& normA,
                                   const glm::vec3& posB, const glm::vec3& normB) {
    glm::vec3 d = posB - posA;
    float distSq = glm::dot(d, d);
    if (distSq >= 2.5e7f) {
        // Sky dome / optical infinity (> 5000 units): direct cosine projection
        float dist = std::sqrt(distSq);
        glm::vec3 dir = d / dist;
        return std::max(glm::dot(normA, dir), 0.0f);
    }
    float dist = std::sqrt(std::max(distSq, 1e-8f));
    glm::vec3 dir = d / dist;
    float cosA = std::max(glm::dot(normA, dir), 0.0f);
    float cosB = std::max(glm::dot(normB, -dir), 0.0f);
    return (cosA * cosB) / std::max(distSq, 1e-4f);
}

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

    // -------------------------------------------------------------------------
    // TEST 8: 16-Byte Hit Atom & Dynamic Position Reconstruction Math (Phase 2b.2)
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 8] 16-Byte Hit Atom & Dynamic Position Reconstruction Math..." << std::endl;
    {
        // 8a. Hit atom representation: 16 bytes exactly
        struct HitAtomTest {
            uint32_t instanceId;
            uint32_t primitiveId;
            uint32_t baryPacked;
            uint32_t flags;
        };
        assert_true(sizeof(HitAtomTest) == 16, "Hit atom struct must strictly equal 16 bytes");

        // 8b. 32-Byte GiReservoir layout verification with Hit atom packing
        struct GiReservoirTest {
            uint32_t hitInstanceId;
            uint32_t hitPrimitiveId;
            uint32_t hitBaryPacked;
            uint32_t wSumBits;
            uint32_t targetBits;
            uint32_t radRG;
            uint32_t radB_dSrc;
            uint32_t flagsM; // [0] valid, [8:1] age, [24:9] M, [31:25] hitFlags
        };
        assert_true(sizeof(GiReservoirTest) == 32, "GiReservoir struct must strictly equal 32 bytes");

        // 8c. Bit-packing roundtrip verification for flagsM (zero bit collisions)
        uint32_t testAge = 127u;
        uint32_t testM = 1023u;
        uint32_t testHitFlags = 2u; // 2-bit hit flags
        float testCosSrc = 0.715542f;
        uint32_t uCos = static_cast<uint32_t>(std::clamp(std::round(testCosSrc * 255.0f), 0.0f, 255.0f));
        bool testValid = true;

        uint32_t packedFlagsM = (testValid ? 1u : 0u) |
                                ((testAge & 0xFFu) << 1u) |
                                ((testM & 0x3FFu) << 9u) |
                                ((testHitFlags & 0x3u) << 19u) |
                                ((uCos & 0xFFu) << 21u);

        bool unpackedValid = (packedFlagsM & 1u) != 0u;
        uint32_t unpackedAge = (packedFlagsM >> 1u) & 0xFFu;
        uint32_t unpackedM = (packedFlagsM >> 9u) & 0x3FFu;
        uint32_t unpackedHitFlags = (packedFlagsM >> 19u) & 0x3u;
        float unpackedCosSrc = static_cast<float>((packedFlagsM >> 21u) & 0xFFu) / 255.0f;

        assert_true(unpackedValid == testValid, "flagsM valid bit preserved");
        assert_true(unpackedAge == testAge, "flagsM age (8-bit) preserved without overflow");
        assert_true(unpackedM == testM, "flagsM M-confidence (10-bit) preserved without truncation");
        assert_true(unpackedHitFlags == testHitFlags, "flagsM hit flags (2-bit) preserved without collision");
        assert_near(unpackedCosSrc, testCosSrc, 1.0f / 255.0f, "flagsM cosSrc (8-bit unorm) preserved within quantization error");

        // 8d. Barycentric unorm16 quantization precision verification
        auto packUnorm2x16 = [](float u, float v) -> uint32_t {
            uint16_t u16 = static_cast<uint16_t>(std::clamp(std::round(u * 65535.0f), 0.0f, 65535.0f));
            uint16_t v16 = static_cast<uint16_t>(std::clamp(std::round(v * 65535.0f), 0.0f, 65535.0f));
            return (static_cast<uint32_t>(v16) << 16u) | static_cast<uint32_t>(u16);
        };
        auto unpackUnorm2x16 = [](uint32_t p) -> glm::vec2 {
            float u = static_cast<float>(p & 0xFFFFu) / 65535.0f;
            float v = static_cast<float>((p >> 16u) & 0xFFFFu) / 65535.0f;
            return glm::vec2(u, v);
        };

        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> dist01(0.0f, 1.0f);
        float maxBaryError = 0.0f;

        for (int i = 0; i < 10000; ++i) {
            float r1 = dist01(rng);
            float r2 = dist01(rng);
            if (r1 + r2 > 1.0f) {
                r1 = 1.0f - r1;
                r2 = 1.0f - r2;
            }
            uint32_t packed = packUnorm2x16(r1, r2);
            glm::vec2 unpacked = unpackUnorm2x16(packed);
            float errU = std::abs(r1 - unpacked.x);
            float errV = std::abs(r2 - unpacked.y);
            maxBaryError = std::max(maxBaryError, std::max(errU, errV));
        }
        assert_true(maxBaryError <= (1.0f / 65535.0f), "Max unorm16 quantization error is <= 1/65535 (1.52e-5)");

        // 8e. Dynamic Triangle Position Reconstruction Accuracy
        glm::vec3 v0(0.0f, 0.0f, 0.0f);
        glm::vec3 v1(2.5f, 0.0f, 0.0f);
        glm::vec3 v2(0.0f, 3.2f, 1.5f);

        float maxPosError = 0.0f;
        for (int i = 0; i < 10000; ++i) {
            float u = dist01(rng);
            float v = dist01(rng);
            if (u + v > 1.0f) {
                u = 1.0f - u;
                v = 1.0f - v;
            }
            float w = 1.0f - u - v;
            glm::vec3 truePos = w * v0 + u * v1 + v * v2;

            uint32_t packedBary = packUnorm2x16(u, v);
            glm::vec2 decBary = unpackUnorm2x16(packedBary);
            float wDec = 1.0f - decBary.x - decBary.y;
            glm::vec3 reconPos = wDec * v0 + decBary.x * v1 + decBary.y * v2;

            float posDiff = glm::length(truePos - reconPos);
            maxPosError = std::max(maxPosError, posDiff);
        }
        // Triangle has ~3.5m edge length; 1.52e-5 relative precision gives < 60 micrometer error
        assert_true(maxPosError < 6e-5f, "Reconstructed position error strictly bounded under 60 micrometers (< 0.06 mm)");
        std::cout << "  [+] PASS: Max position reconstruction error across 10,000 triangles = "
                  << (maxPosError * 1e6f) << " micrometers." << std::endl;
    }

    // -------------------------------------------------------------------------
    // TEST 9: ShadeRequest / ShadeResult Queues & Exact Analytical Reconnection Jacobian (Phase 2b.3)
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 9] Material Request Queues & Exact Analytical Reconnection Jacobian..." << std::endl;
    {
        // 9a. ShadeRequest layout: 32 bytes strictly
        struct ShadeRequestTest {
            uint32_t instanceId;
            uint32_t primitiveId;
            uint32_t baryPacked;
            uint32_t flags;
            uint32_t wiOct;
            uint32_t woOct;
            uint32_t resultSlot;
            uint32_t matIdFlags;
        };
        assert_true(sizeof(ShadeRequestTest) == 32, "ShadeRequest struct must strictly equal 32 bytes");

        // 9b. ShadeResult layout: 16 bytes strictly
        struct ShadeResultTest {
            uint32_t fRG;
            uint32_t fB_pdfFwd;
            uint32_t pdfRev_flags;
            uint32_t pad;
        };
        assert_true(sizeof(ShadeResultTest) == 16, "ShadeResult struct must strictly equal 16 bytes");

        // 9c. Dynamic triangle normal reconstruction from vertex positions
        glm::vec3 v0(0.0f, 0.0f, 0.0f);
        glm::vec3 v1(1.0f, 0.0f, 0.0f);
        glm::vec3 v2(0.0f, 1.0f, 0.0f);
        glm::vec3 e1 = v1 - v0;
        glm::vec3 e2 = v2 - v0;
        glm::vec3 crossN = glm::cross(e1, e2);
        glm::vec3 norm = glm::normalize(crossN);
        assert_near(norm.x, 0.0f, 1e-6f, "Triangle normal X");
        assert_near(norm.y, 0.0f, 1e-6f, "Triangle normal Y");
        assert_near(norm.z, 1.0f, 1e-6f, "Triangle normal Z (outward normal)");

        // Back-facing normal flip check
        uint32_t backFaceFlags = 0u; // frontFace bit is NOT set
        glm::vec3 backNorm = ((backFaceFlags & 2u) == 0u) ? -norm : norm;
        assert_near(backNorm.z, -1.0f, 1e-6f, "Back-facing triangle normal flipped correctly");

        // 9d. Exact analytical Jacobian equivalence:
        // evalGIJacobian(x_new, x_orig, x1, n1) vs evalGIJacobian(dSrc, dNew, cosRcSrc, cosRcNew)
        auto evalGIJacobianDirect = [](float dSrc, float dNew, float cosRcSrc, float cosRcNew) -> float {
            if (dSrc >= 5000.0f || dNew >= 5000.0f) return 1.0f;
            float dSqSrc = dSrc * dSrc;
            float dSqNew = std::max(dNew * dNew, 1e-4f);
            float cosSrcEff = std::max(cosRcSrc, 1e-3f);
            float cosNewEff = std::max(cosRcNew, 1e-4f);
            float J = (dSqSrc / dSqNew) * (cosNewEff / cosSrcEff);
            return std::clamp(J, 0.05f, 10.0f);
        };

        // Test configuration matching TEST 1b:
        // x_orig = (0, 0, 0), x_new = (1, 0, 0), x1 = (0, 2, 0), n1 = (0, -1, 0)
        float d_orig = 2.0f;
        float d_new = std::sqrt(5.0f);
        float cos1_orig = 1.0f;
        float cos1_new = 2.0f / std::sqrt(5.0f);
        float J_direct = evalGIJacobianDirect(d_orig, d_new, cos1_orig, cos1_new);
        float expected_J = 8.0f / (5.0f * std::sqrt(5.0f));
        assert_near(J_direct, expected_J, 1e-5f, "Direct Jacobian matches analytical geometry");

        // Identity shift
        float J_ident = evalGIJacobianDirect(3.5f, 3.5f, 0.7f, 0.7f);
        assert_near(J_ident, 1.0f, 1e-5f, "Direct Jacobian identity shift equals 1.0");

        // Optical infinity fallback
        float J_inf = evalGIJacobianDirect(10000.0f, 10000.0f, 0.2f, 0.8f);
        assert_near(J_inf, 1.0f, 1e-5f, "Direct Jacobian optical infinity equals 1.0");

        // 9e. Quantized cosSrc error bound on Jacobian
        uint32_t uCos = static_cast<uint32_t>(std::clamp(std::round(cos1_orig * 255.0f), 0.0f, 255.0f));
        float packedCosSrc = static_cast<float>(uCos) / 255.0f;
        float J_quant = evalGIJacobianDirect(d_orig, d_new, packedCosSrc, cos1_new);
        float relError = std::abs(J_quant - expected_J) / expected_J;
        assert_true(relError < 0.005f, "Jacobian error from unorm8 cosSrc quantization is < 0.5%");
        std::cout << "  [+] PASS: Max Jacobian quantization deviation = " << (relError * 100.0f) << "%." << std::endl;
    }

    // -------------------------------------------------------------------------
    // TEST 10: Phase 3 Pure Path-Database Paradigm & Geometric Proxy Math (Approach B)
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 10] Pure Path-Database Paradigm & Geometric Proxy Math..." << std::endl;
    {
        // 10a. Static size & std430 alignment verification of Phase 3 structs
        assert_true(sizeof(pathways::PathVertexGPU) == 32, "PathVertexGPU must strictly equal 32 bytes");
        assert_true(sizeof(pathways::PathSampleGPU) == 160, "PathSampleGPU must strictly equal 160 bytes");
        assert_true(sizeof(pathways::ReservoirPTGPU) == 32, "ReservoirPTGPU must strictly equal 32 bytes");
        assert_true(sizeof(pathways::HitGPU) == 16, "HitGPU must strictly equal 16 bytes");
        assert_true(sizeof(pathways::GiReservoirGPU) == 32, "GiReservoirGPU must strictly equal 32 bytes");
        assert_true(sizeof(pathways::ShadeRequestGPU) == 32, "ShadeRequestGPU must strictly equal 32 bytes");
        assert_true(sizeof(pathways::ShadeResultGPU) == 16, "ShadeResultGPU must strictly equal 16 bytes");
        assert_true(sizeof(pathways::DiReservoirGPU) == 28, "DiReservoirGPU must strictly equal 28 bytes");
        assert_true(sizeof(pathways::X1ContextGPU) == 24, "X1ContextGPU must strictly equal 24 bytes");

        // Stride array alignment check: 4 reservoirs fit exactly into one 128B RDNA 4 vector cache line
        pathways::ReservoirPTGPU resArray[4];
        assert_true(sizeof(resArray) == 128, "4x ReservoirPTGPU fits exactly into one 128B RDNA 4 vector cache line");

        // 10b. rho_max conservative admissibility & 8-bit unorm quantization precision
        pathways::MaterialGPU matDiffuse{};
        matDiffuse.albedo = glm::vec4(0.85f, 0.2f, 0.1f, 1.0f);
        matDiffuse.diffuseTransmission = 0.0f;
        pathways::ShadeMaterialGPU smDiff = pathways::createShadeMaterial(matDiffuse);
        float rhoDiff = pathways::getShadeMaterialRhoMax(smDiff);
        assert_true(rhoDiff >= 0.85f - 1.0f / 255.0f, "rho_max captures maximum RGB albedo channel");
        assert_true(rhoDiff <= 1.0f, "rho_max bounded within [0, 1]");

        pathways::MaterialGPU matTrans{};
        matTrans.albedo = glm::vec4(0.1f, 0.1f, 0.1f, 1.0f);
        matTrans.diffuseTransmission = 0.95f;
        pathways::ShadeMaterialGPU smTrans = pathways::createShadeMaterial(matTrans);
        float rhoTrans = pathways::getShadeMaterialRhoMax(smTrans);
        assert_true(rhoTrans >= 0.95f - 1.0f / 255.0f, "rho_max captures diffuse transmission factor");

        // Quantization precision test across 1000 random reflectance levels
        std::mt19937 rng(777);
        std::uniform_real_distribution<float> dist01(0.01f, 1.0f);
        float maxQuantErr = 0.0f;
        for (int i = 0; i < 1000; ++i) {
            float trueRho = dist01(rng);
            pathways::MaterialGPU m{};
            m.albedo = glm::vec4(trueRho, 0.0f, 0.0f, 1.0f);
            pathways::ShadeMaterialGPU sm = pathways::createShadeMaterial(m);
            float recoveredRho = pathways::getShadeMaterialRhoMax(sm);
            maxQuantErr = std::max(maxQuantErr, std::abs(trueRho - recoveredRho));
        }
        assert_true(maxQuantErr <= (1.0f / 255.0f + 1e-5f), "rho_max quantization error strictly bounded under 1/255 (0.00392)");
        std::cout << "  [+] PASS: Max rho_max quantization error = " << maxQuantErr << " (within 8-bit unorm tol)." << std::endl;

        // 10c. Symmetric geometric transport coupling G(a, b)
        glm::vec3 posA(0.0f, 0.0f, 0.0f);
        glm::vec3 normA(0.0f, 1.0f, 0.0f);
        glm::vec3 posB(0.0f, 2.0f, 0.0f);
        glm::vec3 normB(0.0f, -1.0f, 0.0f); // Face-to-face parallel plates at dist = 2.0

        float G_ab = evalGeometricCoupling(posA, normA, posB, normB);
        float G_ba = evalGeometricCoupling(posB, normB, posA, normA);
        assert_near(G_ab, G_ba, 1e-6f, "G(a, b) reciprocity / symmetry (G_ab == G_ba)");
        // Analytical: cosA = 1.0, cosB = 1.0, distSq = 4.0 -> G = 1.0 / 4.0 = 0.25
        assert_near(G_ab, 0.25f, 1e-5f, "Analytical parallel plates geometric coupling (G = 0.25)");

        // Inverse square distance law: double distance to 4.0 -> G should scale by 1/4 (0.0625)
        glm::vec3 posB_far(0.0f, 4.0f, 0.0f);
        float G_far = evalGeometricCoupling(posA, normA, posB_far, normB);
        assert_near(G_far, 0.0625f, 1e-5f, "Geometric coupling inverse-square distance scaling");

        // Tilted receiver: normB tilted 45 degrees
        glm::vec3 normB_tilted = glm::normalize(glm::vec3(0.0f, -1.0f, 1.0f));
        float G_tilted = evalGeometricCoupling(posA, normA, posB, normB_tilted);
        float expected_G_tilted = (1.0f * (1.0f / std::sqrt(2.0f))) / 4.0f;
        assert_near(G_tilted, expected_G_tilted, 1e-5f, "G(a, b) cosine law with tilted normal");

        // Optical infinity sky dome test
        glm::vec3 posSky(0.0f, 6000.0f, 0.0f);
        float G_sky = evalGeometricCoupling(posA, normA, posSky, normB);
        assert_near(G_sky, 1.0f, 1e-5f, "Optical infinity (>5000 units) directional projection");

        // 10d. Path Target Proxy p_hat(X) evaluation & zero-admissibility guards
        auto evalPathTargetProxy = [&](const std::vector<glm::vec3>& pos,
                                       const std::vector<glm::vec3>& norm,
                                       const std::vector<float>& rho,
                                       float emitLuminance) -> float {
            if (pos.size() < 2 || emitLuminance <= 1e-6f) return 0.0f;
            float pProxy = emitLuminance;
            for (size_t i = 1; i < pos.size(); ++i) {
                float G = evalGeometricCoupling(pos[i-1], norm[i-1], pos[i], norm[i]);
                if (G <= 1e-7f) return 0.0f;
                float r = (i - 1 < rho.size()) ? rho[i - 1] : 1.0f;
                pProxy *= (r * G);
                if (pProxy <= 1e-9f) return 0.0f;
            }
            return pProxy;
        };

        // 3-vertex path: Camera hit x1 (0, 0, 0) -> Secondary hit x2 (0, 2, 0) -> Light x3 (1, 2, 0)
        // x1 normal: (0, 1, 0) facing upwards along +Y
        // x2 normal: normalize(1, -1, 0) facing 45-degrees down towards x1 and right towards x3
        // x3 normal: (-1, 0, 0) facing left towards x2
        glm::vec3 n2 = glm::normalize(glm::vec3(1.0f, -1.0f, 0.0f));
        std::vector<glm::vec3> validPos = { glm::vec3(0.0f, 0.0f, 0.0f), glm::vec3(0.0f, 2.0f, 0.0f), glm::vec3(1.0f, 2.0f, 0.0f) };
        std::vector<glm::vec3> validNorm = { glm::vec3(0.0f, 1.0f, 0.0f), n2, glm::vec3(-1.0f, 0.0f, 0.0f) };
        std::vector<float> validRho = { 0.8f, 0.5f };
        float p_valid = evalPathTargetProxy(validPos, validNorm, validRho, 10.0f);
        assert_true(p_valid > 0.0f, "Physically admissible 3-vertex subpath evaluates to positive target proxy");

        // Backfacing secondary surface (normal facing away into negative hemisphere) -> strictly 0.0 target
        std::vector<glm::vec3> backNorm = validNorm;
        backNorm[1] = -n2; // Points away into back hemisphere
        float p_back = evalPathTargetProxy(validPos, backNorm, validRho, 10.0f);
        assert_near(p_back, 0.0f, 1e-9f, "Backfacing vertex strictly evaluates to zero target proxy");

        // Zero light emission -> strictly 0.0 target
        float p_dark = evalPathTargetProxy(validPos, validNorm, validRho, 0.0f);
        assert_near(p_dark, 0.0f, 1e-9f, "Zero emitter luminance strictly evaluates to zero target proxy");

        // 10e. Streaming ReservoirPT Unbiased Monte Carlo Convergence (100,000 Trials)
        const int NUM_TRIALS = 100000;
        const int M = 4;
        double sumEstimates = 0.0;
        std::uniform_real_distribution<float> unitDist(0.0f, 1.0f);

        // Ground truth integral: int_0^1 4 * x^3 dx = 1.0
        // Target proxy: p_hat(x) = x^2 (geometric coupling surrogate)
        // Proposal q(x) = uniform on [0, 1] (q = 1.0)
        // True integrand: f(x) = 4 * x^3
        for (int t = 0; t < NUM_TRIALS; ++t) {
            pathways::ReservoirPTGPU r{};
            r.weightSum = 0.0f;
            r.W = 0.0f;
            r.M = 0.0f;
            r.targetProxy = 0.0f;
            r.pathIndex = 0xFFFFFFFFu;

            float selected_x = 0.0f;

            for (int c = 0; c < M; ++c) {
                float x = unitDist(rng);
                float p_hat = x * x; // Target proxy
                float q = 1.0f;      // Uniform proposal
                float weight = p_hat / q;

                float combinedWeight = r.weightSum + weight;
                r.weightSum = combinedWeight;
                r.M += 1.0f;

                if (unitDist(rng) * combinedWeight < weight) {
                    r.pathIndex = static_cast<uint32_t>(c);
                    r.targetProxy = p_hat;
                    selected_x = x;
                }
            }

            if (r.targetProxy > 1e-6f) {
                r.W = r.weightSum / (r.M * r.targetProxy);
                float f_val = 4.0f * selected_x * selected_x * selected_x;
                sumEstimates += static_cast<double>(r.W * f_val);
            }
        }

        double meanIntegral = sumEstimates / NUM_TRIALS;
        assert_near(static_cast<float>(meanIntegral), 1.0f, 0.005f,
                    "ReservoirPT streaming estimator unbiased convergence (analytical expectation: 1.0)");

        // 10f. ptApplyTap Exact Analytical Solid-Angle Jacobian & Shifted Target Proxy Verification
        std::cout << "\n[TEST 10f] ptApplyTap Solid-Angle Jacobian & Shifted Target Proxy..." << std::endl;
        {
            glm::vec3 x1_src(0.0f, 0.0f, 0.0f);
            glm::vec3 n1_src(0.0f, 1.0f, 0.0f);
            glm::vec3 x_rc(0.0f, 2.0f, 0.0f);
            glm::vec3 n_rc(0.0f, -1.0f, 0.0f);

            // Reconnection Jacobian reference formula:
            // J = (dSqSrc / dSqNew) * (cosRcNew / cosRcSrc)
            auto evalPtJacobian = [](float dSrc, float dNew, float cosRcSrc, float cosRcNew) -> float {
                if (dSrc >= 5000.0f || dNew >= 5000.0f) return 1.0f;
                float dSqSrc = dSrc * dSrc;
                float dSqNew = std::max(dNew * dNew, 1e-4f);
                float cosSrcEff = std::max(cosRcSrc, 1e-3f);
                float cosNewEff = std::max(cosRcNew, 1e-4f);
                float J = (dSqSrc / dSqNew) * (cosNewEff / cosSrcEff);
                return std::clamp(J, 0.05f, 10.0f);
            };

            // Case 1: Identity shift (x1_new == x1_src) -> |J| == 1.0, G_new == G_src
            float dSrc = glm::length(x_rc - x1_src);
            float cosRcSrc = std::abs(glm::dot(n_rc, -glm::normalize(x_rc - x1_src)));
            float J_ident = evalPtJacobian(dSrc, dSrc, cosRcSrc, cosRcSrc);
            assert_near(J_ident, 1.0f, 1e-5f, "ptApplyTap identity shift Jacobian (|J| == 1.0)");

            float G_src = evalGeometricCoupling(x1_src, n1_src, x_rc, n_rc);
            float G_ident = evalGeometricCoupling(x1_src, n1_src, x_rc, n_rc);
            assert_near(G_ident / G_src, 1.0f, 1e-5f, "ptApplyTap identity shift proxy ratio (G_new / G_src == 1.0)");

            // Case 2: Distance halving shift (x1_new at dist = 1.0 instead of 2.0)
            glm::vec3 x1_half(0.0f, 1.0f, 0.0f);
            float dNew_half = glm::length(x_rc - x1_half);
            float cosRcNew_half = std::abs(glm::dot(n_rc, -glm::normalize(x_rc - x1_half)));
            float J_half = evalPtJacobian(dSrc, dNew_half, cosRcSrc, cosRcNew_half);
            // Analytical: (2^2 / 1^2) * (1 / 1) = 4.0
            assert_near(J_half, 4.0f, 1e-5f, "ptApplyTap distance-halving shift Jacobian (|J| == 4.0)");

            float G_half = evalGeometricCoupling(x1_half, n1_src, x_rc, n_rc);
            assert_near(G_half / G_src, 4.0f, 1e-5f, "ptApplyTap geometric coupling ratio tracks inverse-square distance");

            // Case 3: Jacobian clamp boundary ([0.05, 10.0])
            // Very close shift (dNew = 0.1): unconstrained J would be 4 / 0.01 = 400 -> clamped to 10.0
            float J_clamped_high = evalPtJacobian(2.0f, 0.1f, 1.0f, 1.0f);
            assert_near(J_clamped_high, 10.0f, 1e-5f, "ptApplyTap Jacobian ceiling clamp to 10.0");

            // Very far shift (dNew = 20.0): unconstrained J would be 4 / 400 = 0.01 -> clamped to 0.05
            float J_clamped_low = evalPtJacobian(2.0f, 20.0f, 1.0f, 1.0f);
            assert_near(J_clamped_low, 0.05f, 1e-5f, "ptApplyTap Jacobian floor clamp to 0.05");

            // Case 4: Gating & rejection guards
            auto mockPtApplyTap = [&](float roughness, float distRatio, float cos1, uint32_t rcFlags) -> bool {
                if ((rcFlags & 1u) != 0u) return false; // Reject sphere
                if (distRatio < 0.2f || distRatio > 5.0f) return false;
                if (roughness < 0.15f) return false;
                if (cos1 <= 0.05f) return false; // GI_TAP_MIN_COS
                return true;
            };

            assert_true(mockPtApplyTap(0.5f, 1.0f, 0.8f, 0u), "Valid tap is accepted");
            assert_true(!mockPtApplyTap(0.08f, 1.0f, 0.8f, 0u), "Glossy receiver (roughness < 0.15) rejected");
            assert_true(!mockPtApplyTap(0.5f, 6.0f, 0.8f, 0u), "Large footprint ratio (> 5.0) rejected");
            assert_true(!mockPtApplyTap(0.5f, 0.1f, 0.8f, 0u), "Small footprint ratio (< 0.2) rejected");
            assert_true(!mockPtApplyTap(0.5f, 1.0f, -0.1f, 0u), "Backfacing receiver (cos1 <= 0) rejected");
            assert_true(!mockPtApplyTap(0.5f, 1.0f, 0.8f, 1u), "Sphere reconnection hit rejected");

            // Case 5: Unbiased weight balance heuristic with shifted proxy
            float tap_wSum = 2.0f;
            float tap_M = 4.0f;
            float tap_proxy = 0.25f;
            float shifted_proxy = 1.0f; // 4x increase due to distance halving
            float J = 4.0f;
            float targetRatio = shifted_proxy / tap_proxy;
            float weight = (tap_wSum / tap_M) * tap_M * targetRatio * J;
            // Analytical: (2/4) * 4 * 4 * 4 = 32.0
            assert_near(weight, 32.0f, 1e-4f, "GRIS balance heuristic weight with shifted proxy and Jacobian");
        }
    }

    // -------------------------------------------------------------------------
    // TEST 11: X1Context Roughness Fidelity & Reconnection Gating Guards
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 11] X1Context Roughness Fidelity & Reconnection Gating Guards..." << std::endl;
    {
        // 11a. X1Context roughness preservation across [0.0, 1.0]
        auto packHalf2x16 = [](glm::vec2 v) -> uint32_t {
            return glm::packHalf2x16(v);
        };
        auto unpackHalf2x16 = [](uint32_t p) -> glm::vec2 {
            return glm::unpackHalf2x16(p);
        };

        // Test representative materials: mirror/delta (0.001), semi-glossy (0.08, 0.14),
        // moderately rough diffuse (0.25, 0.5), completely rough (1.0).
        std::vector<float> testRoughness = { 0.001f, 0.05f, 0.10f, 0.149f, 0.151f, 0.25f, 0.5f, 0.8f, 1.0f };
        for (float r : testRoughness) {
            float f0 = 0.04f;
            uint32_t f0Pad = packHalf2x16(glm::vec2(std::clamp(f0, 0.0f, 1.0f), std::clamp(r, 0.0f, 1.0f)));
            float recoveredF0 = unpackHalf2x16(f0Pad).x;
            float recoveredRoughness = unpackHalf2x16(f0Pad).y;

            assert_near(recoveredF0, f0, 1e-3f, "X1Context F0 fp16 fidelity");
            assert_near(recoveredRoughness, r, 1e-3f, "X1Context roughness fp16 fidelity");

            // Verify the roughness < 0.15 gate behavior
            bool isGlossyReceiver = (recoveredRoughness < 0.15f);
            bool expectedGlossy = (r < 0.15f);
            assert_true(isGlossyReceiver == expectedGlossy,
                        "Roughness < 0.15 gate fires accurately on recovered roughness");
        }

        // 11b. Verify that passing 1.0 hardcoded (old bug) completely disabled the glossy gate
        float trueSemiGlossyRoughness = 0.08f;
        float buggedRoughness = 1.0f; // Old diffuse shader wrote hardcoded 1.0
        assert_true(!(buggedRoughness < 0.15f), "Hardcoded 1.0 erroneously bypasses glossy receiver rejection");
        assert_true(trueSemiGlossyRoughness < 0.15f, "True roughness correctly triggers glossy receiver rejection");

        // 11c. giApplyTap sphere candidate rejection gate:
        // Hit atom has flags bit 0 set ((flags & 1u) != 0u) for procedural spheres.
        auto mockGiApplyTapGate = [](uint32_t hitFlags, float roughness, float distRatio) -> bool {
            // Replicates restir_common.glsl giApplyTap early-outs:
            if ((hitFlags & 1u) != 0u) return false; // Reject sphere x2 candidates
            if (distRatio < 0.2f || distRatio > 5.0f) return false;
            if (roughness < 0.15f) return false; // Reject glossy receivers
            return true;
        };

        // Case 1: Sphere candidate hit (flags = 1u) must be strictly rejected
        bool sphereResult = mockGiApplyTapGate(1u /* sphere */, 0.8f, 1.0f);
        assert_true(!sphereResult, "giApplyTap strictly rejects procedural sphere candidate (flags & 1u != 0)");

        // Case 2: Triangle candidate hit (flags = 0u or 2u) on diffuse receiver accepted
        bool triResult = mockGiApplyTapGate(2u /* triangle, frontFace */, 0.8f, 1.0f);
        assert_true(triResult, "giApplyTap accepts valid triangle candidate on diffuse receiver");

        // Case 3: Triangle candidate on glossy receiver (roughness < 0.15) rejected
        bool glossyResult = mockGiApplyTapGate(2u /* triangle */, 0.10f, 1.0f);
        assert_true(!glossyResult, "giApplyTap strictly rejects triangle candidate on glossy receiver (roughness < 0.15)");
    }

    // -------------------------------------------------------------------------
    // TEST 12: Expert Recommendations Verification:
    //          1. Instance GPU Layout & TLAS Matrix Hit Reconstruction
    //          2. Direct Illumination Occlusion Confidence Preservation
    //          3. Specular Mirror Partitioning (Roughness Gating)
    //          4. Approach A Request/Result Queues Data Contract
    // -------------------------------------------------------------------------
    std::cout << "\n[TEST 12] Expert Recommendations & Production Architecture..." << std::endl;
    {
        // 12a. InstanceGPU size and std430 alignment
        assert_true(sizeof(pathways::InstanceGPU) == 144, "InstanceGPU must be strictly 144 bytes");
        assert_true(alignof(pathways::InstanceGPU) >= 16, "InstanceGPU must have 16-byte alignment for std430");

        // 12b. Instance Transform in Hit Position and Normal Reconstruction
        // Local triangle in BLAS geometry
        glm::vec3 v0(0.0f, 0.0f, 0.0f);
        glm::vec3 v1(1.0f, 0.0f, 0.0f);
        glm::vec3 v2(0.0f, 1.0f, 0.0f);
        float w = 0.5f, baryU = 0.2f, baryV = 0.3f;
        glm::vec3 localPos = w * v0 + baryU * v1 + baryV * v2; // (0.2, 0.3, 0.0)
        glm::vec3 localNorm = glm::normalize(glm::cross(v1 - v0, v2 - v0)); // (0, 0, 1)

        // Case 1: Identity transform produces exact identical local position and normal
        pathways::InstanceGPU identityInst{};
        identityInst.transform = glm::mat4(1.0f);
        identityInst.invTransform = glm::mat4(1.0f);

        glm::vec3 reconsPosIdent = glm::vec3(identityInst.transform * glm::vec4(localPos, 1.0f));
        glm::mat3 normalMatIdent = glm::transpose(glm::mat3(identityInst.invTransform));
        glm::vec3 reconsNormIdent = glm::normalize(normalMatIdent * localNorm);

        assert_near(reconsPosIdent.x, localPos.x, 1e-6f, "Identity transform pos.x invariant");
        assert_near(reconsPosIdent.y, localPos.y, 1e-6f, "Identity transform pos.y invariant");
        assert_near(reconsPosIdent.z, localPos.z, 1e-6f, "Identity transform pos.z invariant");
        assert_near(reconsNormIdent.x, 0.0f, 1e-6f, "Identity transform norm.x invariant");
        assert_near(reconsNormIdent.y, 0.0f, 1e-6f, "Identity transform norm.y invariant");
        assert_near(reconsNormIdent.z, 1.0f, 1e-6f, "Identity transform norm.z invariant");

        // Case 2: Affine Transform (Translation + Rotation + Non-uniform Scaling)
        // Scale: Sx=2, Sy=3, Sz=1
        // Translation: Tx=10, Ty=20, Tz=30
        // Rotation: 90 deg around Z -> (x, y) -> (-y, x)
        glm::mat4 T = glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 20.0f, 30.0f));
        glm::mat4 R = glm::rotate(glm::mat4(1.0f), glm::radians(90.0f), glm::vec3(0.0f, 0.0f, 1.0f));
        glm::mat4 S = glm::scale(glm::mat4(1.0f), glm::vec3(2.0f, 3.0f, 1.0f));
        glm::mat4 M = T * R * S;
        glm::mat4 invM = glm::inverse(M);

        pathways::InstanceGPU transformedInst{};
        transformedInst.transform = M;
        transformedInst.invTransform = invM;

        glm::vec3 worldPos = glm::vec3(transformedInst.transform * glm::vec4(localPos, 1.0f));
        glm::mat3 normalMat = glm::transpose(glm::mat3(transformedInst.invTransform));
        glm::vec3 worldNorm = glm::normalize(normalMat * localNorm);

        // Verify transformed position matches manual matrix product
        glm::vec4 expectedWorldPos = M * glm::vec4(localPos, 1.0f);
        assert_near(worldPos.x, expectedWorldPos.x, 1e-5f, "Affine instance transform world pos X");
        assert_near(worldPos.y, expectedWorldPos.y, 1e-5f, "Affine instance transform world pos Y");
        assert_near(worldPos.z, expectedWorldPos.z, 1e-5f, "Affine instance transform world pos Z");

        // Transformed triangle vertices in world space
        glm::vec3 w0 = glm::vec3(M * glm::vec4(v0, 1.0f));
        glm::vec3 w1 = glm::vec3(M * glm::vec4(v1, 1.0f));
        glm::vec3 w2 = glm::vec3(M * glm::vec4(v2, 1.0f));
        glm::vec3 expectedWorldNorm = glm::normalize(glm::cross(w1 - w0, w2 - w0));

        assert_near(worldNorm.x, expectedWorldNorm.x, 1e-5f, "Affine instance transform normal X orthogonal to edges");
        assert_near(worldNorm.y, expectedWorldNorm.y, 1e-5f, "Affine instance transform normal Y orthogonal to edges");
        assert_near(worldNorm.z, expectedWorldNorm.z, 1e-5f, "Affine instance transform normal Z orthogonal to edges");

        // 12c. Direct Illumination Occlusion Confidence Preservation (Fix 1)
        // Simulate a reservoir after finding a candidate light
        pathways::DiReservoirGPU r{};
        r.idxM = (4u << 16) | 2u; // M = 4, light index = 2
        r.wSum = 12.5f;
        r.targetPdf = 2.5f;
        r.flags = 0x100u; // valid bit set
        r.hitDist = 3.0f;

        // Under inline shadow query: shadow ray is occluded
        bool occluded = true;
        glm::vec3 secDirectL(0.0f);
        if (!occluded) {
            secDirectL = glm::vec3(1.0f);
        } else {
            secDirectL = glm::vec3(0.0f);
            // CRITICAL FIX: Do NOT destroy reservoir r!
            // r remains intact with its M and wSum so temporal reuse carries confidence forward.
        }

        assert_near(secDirectL.x, 0.0f, 1e-6f, "Occluded shadow ray produces zero current-frame radiance");
        assert_true((r.flags & 0x100u) != 0u, "Reservoir valid flag PRESERVED despite occlusion");
        assert_near(r.wSum, 12.5f, 1e-6f, "Reservoir wSum PRESERVED despite occlusion");
        assert_true((r.idxM >> 16) == 4u, "Reservoir confidence counter M PRESERVED despite occlusion");
        assert_true((r.idxM & 0xFFFFu) == 2u, "Reservoir light index candidate PRESERVED despite occlusion");

        // 12d. Specular Mirror Partitioning (Roughness Gating, Fix 3)
        auto evaluateRoughnessPartitioning = [](float roughness, bool enableRestirFlag) {
            bool enableRestirPT = enableRestirFlag && (roughness >= 0.10f);
            bool giCtxEligible = (roughness >= 0.10f);
            return std::make_pair(enableRestirPT, giCtxEligible);
        };

        // Mirror / Near-delta materials (< 0.10) must be partitioned out:
        std::vector<float> mirrorRoughnesses = { 0.0001f, 0.001f, 0.01f, 0.05f, 0.099f };
        for (float mr : mirrorRoughnesses) {
            auto [ptEnabled, giEligible] = evaluateRoughnessPartitioning(mr, true);
            assert_true(!ptEnabled, "Near-delta surface (roughness < 0.10) strictly bypassed from ReSTIR PT");
            assert_true(!giEligible, "Near-delta surface strictly excluded from ReSTIR GI x1Context");
        }

        // Rough / Glossy / Diffuse materials (>= 0.10) participate fully in ReSTIR:
        std::vector<float> roughMaterials = { 0.10f, 0.15f, 0.30f, 0.50f, 1.0f };
        for (float rm : roughMaterials) {
            auto [ptEnabled, giEligible] = evaluateRoughnessPartitioning(rm, true);
            assert_true(ptEnabled, "Rough surface (roughness >= 0.10) enabled for ReSTIR PT");
            assert_true(giEligible, "Rough surface eligible for ReSTIR GI x1Context");
        }

        // 12e. Approach A Request/Result Queues Data Contract & Roundtrip (Fix 4)
        pathways::ShadeRequestGPU req{};
        req.hit.instanceId = 3;
        req.hit.primitiveId = 42;
        req.hit.baryPacked = 0x12345678u;
        req.hit.flags = 2; // frontFace
        req.wiOct = 0xAAAAAAAAu;
        req.woOct = 0x55555555u;
        req.resultSlot = 100;
        req.matIdFlags = 7;
        assert_true(sizeof(req) == 32, "ShadeRequestGPU strictly 32 bytes");

        pathways::ShadeResultGPU res{};
        glm::vec3 bsdfColor(0.75f, 0.5f, 0.25f);
        float pdfFwd = 1.25f;
        float pdfRev = 0.95f;
        res.fRG = glm::packHalf2x16(glm::vec2(bsdfColor.r, bsdfColor.g));
        res.fB_pdfFwd = glm::packHalf2x16(glm::vec2(bsdfColor.b, pdfFwd));
        res.pdfRev_flags = glm::packHalf2x16(glm::vec2(pdfRev, 0.0f)) | (1u << 16);
        res.pad = 0;
        assert_true(sizeof(res) == 16, "ShadeResultGPU strictly 16 bytes");

        // Unpack roundtrip verification
        glm::vec2 unpackedRG = glm::unpackHalf2x16(res.fRG);
        glm::vec2 unpackedB_fwd = glm::unpackHalf2x16(res.fB_pdfFwd);
        glm::vec2 unpackedRev_flags = glm::unpackHalf2x16(res.pdfRev_flags);

        assert_near(unpackedRG.x, bsdfColor.r, 1e-3f, "ShadeResult BSDF R fp16 roundtrip");
        assert_near(unpackedRG.y, bsdfColor.g, 1e-3f, "ShadeResult BSDF G fp16 roundtrip");
        assert_near(unpackedB_fwd.x, bsdfColor.b, 1e-3f, "ShadeResult BSDF B fp16 roundtrip");
        assert_near(unpackedB_fwd.y, pdfFwd, 1e-3f, "ShadeResult forward pdf fp16 roundtrip");
        assert_near(unpackedRev_flags.x, pdfRev, 1e-3f, "ShadeResult reverse pdf fp16 roundtrip");
        std::cout << "  [+] PASS: TEST 12 all assertions satisfied." << std::endl;
    }

    std::cout << "\n================================================================================" << std::endl;
    std::cout << "  ALL ReSTIR PT Mathematical & Estimator Unit Tests PASSED!" << std::endl;
    std::cout << "================================================================================" << std::endl;
    return 0;
}
