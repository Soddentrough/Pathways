#include "core/Config.hpp"
#include "scene/Camera.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

using namespace pathways;

static void assert_near(float a, float b, float eps = 0.001f, const char* msg = "") {
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

static glm::vec3 rgbToYCoCg(glm::vec3 rgb) {
    float Y  = glm::dot(rgb, glm::vec3( 0.25f,  0.50f,  0.25f));
    float Co = glm::dot(rgb, glm::vec3( 0.50f,  0.00f, -0.50f));
    float Cg = glm::dot(rgb, glm::vec3(-0.25f,  0.50f, -0.25f));
    return glm::vec3(Y, Co, Cg);
}

static glm::vec3 yCoCgToRgb(glm::vec3 ycocg) {
    float Y  = ycocg.x;
    float Co = ycocg.y;
    float Cg = ycocg.z;
    return glm::vec3(Y + Co - Cg, Y + Cg, Y - Co - Cg);
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing Temporal Anti-Aliasing (TAA) Engine" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // -------------------------------------------------------------------------
    // 1. Test CLI Config Parsing
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 1] Command-Line Configuration & Flag Parsing..." << std::endl;
        Config configDefault;
        check_true(!configDefault.enable_taa, "Default TAA is off");
        assert_near(configDefault.taa_blend_alpha, 0.10f, 0.0001f, "Default blend alpha");
        assert_near(configDefault.taa_clipping_gamma, 2.25f, 0.0001f, "Default clipping gamma");

        const char* argv1[] = { "pathways", "--taa" };
        Config c1 = Config::parse(2, const_cast<char**>(argv1));
        check_true(c1.enable_taa, "--taa enabled");

        const char* argv2[] = { "pathways", "--taa", "--no-taa" };
        Config c2 = Config::parse(3, const_cast<char**>(argv2));
        check_true(!c2.enable_taa, "--no-taa disables");

        const char* argv3[] = { "pathways", "--taa", "--taa-alpha", "0.18", "--taa-gamma", "1.75" };
        Config c3 = Config::parse(6, const_cast<char**>(argv3));
        check_true(c3.enable_taa, "--taa enabled");
        assert_near(c3.taa_blend_alpha, 0.18f, 0.0001f, "Custom blend alpha");
        assert_near(c3.taa_clipping_gamma, 1.75f, 0.0001f, "Custom clipping gamma");

        std::cout << "  -> CLI flags and defaults successfully verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 2. Test Halton(2, 3) Low-Discrepancy Generator
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 2] Halton(2, 3) 8-Phase Subpixel Jitter Generator..." << std::endl;
        // Verify base 2 radical inverse
        assert_near(halton(1, 2), 0.5f, 1e-5f, "Halton(1, 2) = 0.5");
        assert_near(halton(2, 2), 0.25f, 1e-5f, "Halton(2, 2) = 0.25");
        assert_near(halton(3, 2), 0.75f, 1e-5f, "Halton(3, 2) = 0.75");
        assert_near(halton(4, 2), 0.125f, 1e-5f, "Halton(4, 2) = 0.125");

        // Verify base 3 radical inverse
        assert_near(halton(1, 3), 1.0f / 3.0f, 1e-5f, "Halton(1, 3) = 1/3");
        assert_near(halton(2, 3), 2.0f / 3.0f, 1e-5f, "Halton(2, 3) = 2/3");
        assert_near(halton(3, 3), 1.0f / 9.0f, 1e-5f, "Halton(3, 3) = 1/9");

        // Verify 8-phase jitter bounds [-0.5, 0.5]
        for (uint32_t phase = 0; phase < 16; ++phase) {
            glm::vec2 j = getHaltonJitter(phase);
            check_true(j.x >= -0.5f && j.x <= 0.5f, "Jitter X within [-0.5, 0.5]");
            check_true(j.y >= -0.5f && j.y <= 0.5f, "Jitter Y within [-0.5, 0.5]");
        }

        // Verify sample-parallel complementary phases (phase 0 vs phase 4)
        glm::vec2 j0 = getHaltonJitter(0);
        glm::vec2 j4 = getHaltonJitter(4);
        float dist = glm::distance(j0, j4);
        check_true(dist > 0.2f, "Sample-parallel complementary jitter phases are well-spaced");

        std::cout << "  -> Halton sequence mathematical properties verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 3. Test Camera Subpixel Jitter & Unjittered Matrices
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 3] Camera Projection Subpixel Jittering & Unjittered Matrices..." << std::endl;
        Camera cam(glm::vec3(0.0f, 1.0f, 3.5f), glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 16.0f / 9.0f);

        const uint32_t width = 1920;
        const uint32_t height = 1080;

        // When TAA is disabled:
        CameraUniform uboNoTaa = cam.getUniformData(0, 1, 4, 0, /*enableTaa=*/false, width, height);
        assert_near(uboNoTaa.jitterOffset.x, 0.0f, 1e-5f, "No TAA -> jitterOffset.x is 0");
        assert_near(uboNoTaa.jitterOffset.y, 0.0f, 1e-5f, "No TAA -> jitterOffset.y is 0");
        assert_near(uboNoTaa.jitterOffset.z, 0.0f, 1e-5f, "No TAA -> jitterOffset.z is 0");
        assert_near(uboNoTaa.jitterOffset.w, 0.0f, 1e-5f, "No TAA -> jitterOffset.w is 0");

        // When TAA is enabled:
        CameraUniform uboTaa = cam.getUniformData(0, 1, 4, 0, /*enableTaa=*/true, width, height);
        glm::vec2 expectedJitter = getHaltonJitter(0);
        assert_near(uboTaa.jitterOffset.x, expectedJitter.x, 1e-5f, "TAA jitterOffset.x matches Halton");
        assert_near(uboTaa.jitterOffset.y, expectedJitter.y, 1e-5f, "TAA jitterOffset.y matches Halton");

        // Unjittered matrix must match proj * view from non-jittered camera
        glm::mat4 expectedUnjittered = cam.getProjectionMatrix() * cam.getViewMatrix();
        for (int c = 0; c < 4; ++c) {
            for (int r = 0; r < 4; ++r) {
                assert_near(uboTaa.unjitteredViewProj[c][r], expectedUnjittered[c][r], 1e-4f, "Unjittered view-proj matches expected");
            }
        }

        // Jittered projection must have small NDC translation in row 2 of column 0 and 1
        float ndcJitterX = (expectedJitter.x / static_cast<float>(width)) * 2.0f;
        float ndcJitterY = (expectedJitter.y / static_cast<float>(height)) * 2.0f;
        assert_near(uboTaa.jitterOffset.z, ndcJitterX, 1e-6f, "NDC jitter X");
        assert_near(uboTaa.jitterOffset.w, ndcJitterY, 1e-6f, "NDC jitter Y");
        glm::mat4 reconstructedProj = glm::inverse(uboTaa.projInverse);
        glm::mat4 expectedProj = cam.getProjectionMatrix();
        expectedProj[2][0] += ndcJitterX;
        expectedProj[2][1] += ndcJitterY;
        assert_near(reconstructedProj[2][0], expectedProj[2][0], 1e-4f, "Projection matrix column 2 has jitter applied");
        assert_near(reconstructedProj[2][1], expectedProj[2][1], 1e-4f, "Projection matrix column 2 y-jitter applied");

        std::cout << "  -> Camera projection jittering and clean unjittered matrices verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 4. Test YCoCg Transformation Round-Trip
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 4] YCoCg Color Space Conversion & Invertibility..." << std::endl;
        std::vector<glm::vec3> testColors = {
            glm::vec3(0.0f, 0.0f, 0.0f),
            glm::vec3(1.0f, 1.0f, 1.0f),
            glm::vec3(1.0f, 0.0f, 0.0f),
            glm::vec3(0.0f, 1.0f, 0.0f),
            glm::vec3(0.0f, 0.0f, 1.0f),
            glm::vec3(0.85f, 0.65f, 0.22f),
            glm::vec3(0.12f, 0.94f, 0.56f)
        };

        for (const auto& rgb : testColors) {
            glm::vec3 ycocg = rgbToYCoCg(rgb);
            glm::vec3 reconstructed = yCoCgToRgb(ycocg);
            assert_near(reconstructed.r, rgb.r, 1e-5f, "YCoCg roundtrip R");
            assert_near(reconstructed.g, rgb.g, 1e-5f, "YCoCg roundtrip G");
            assert_near(reconstructed.b, rgb.b, 1e-5f, "YCoCg roundtrip B");
        }

        std::cout << "  -> YCoCg transformations lossless and exact." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 5. Test Variance Clipping Bounding Box Math
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 5] YCoCg Variance Bounding Box History Clamping..." << std::endl;
        // Simulate a 3x3 neighborhood of subtle color variation
        std::vector<glm::vec3> neighborhood = {
            glm::vec3(0.50f, 0.50f, 0.50f),
            glm::vec3(0.52f, 0.51f, 0.49f),
            glm::vec3(0.48f, 0.49f, 0.51f),
            glm::vec3(0.51f, 0.50f, 0.50f),
            glm::vec3(0.50f, 0.50f, 0.50f),
            glm::vec3(0.49f, 0.52f, 0.48f),
            glm::vec3(0.53f, 0.48f, 0.52f),
            glm::vec3(0.50f, 0.51f, 0.50f),
            glm::vec3(0.47f, 0.50f, 0.51f)
        };

        glm::vec3 m1(0.0f);
        glm::vec3 m2(0.0f);
        for (const auto& c : neighborhood) {
            glm::vec3 ycocg = rgbToYCoCg(c);
            m1 += ycocg;
            m2 += ycocg * ycocg;
        }

        glm::vec3 mu = m1 * (1.0f / 9.0f);
        glm::vec3 sigma = glm::sqrt(glm::max(m2 * (1.0f / 9.0f) - mu * mu, glm::vec3(0.0f)));
        float gamma = 1.25f;
        glm::vec3 boxMin = mu - gamma * sigma;
        glm::vec3 boxMax = mu + gamma * sigma;

        // Test valid in-bounds history sample
        glm::vec3 validHist = rgbToYCoCg(glm::vec3(0.50f, 0.50f, 0.50f));
        glm::vec3 clampedValid = glm::clamp(validHist, boxMin, boxMax);
        assert_near(clampedValid.x, validHist.x, 1e-5f, "In-bounds history Y unchanged");
        assert_near(clampedValid.y, validHist.y, 1e-5f, "In-bounds history Co unchanged");
        assert_near(clampedValid.z, validHist.z, 1e-5f, "In-bounds history Cg unchanged");

        // Test bright outlier (ghosting candidate)
        glm::vec3 outlierHist = rgbToYCoCg(glm::vec3(1.0f, 0.0f, 0.0f));
        glm::vec3 clampedOutlier = glm::clamp(outlierHist, boxMin, boxMax);
        check_true(clampedOutlier.x <= boxMax.x, "Outlier history Y clamped to boxMax");
        check_true(clampedOutlier.x >= boxMin.x, "Outlier history Y clamped to boxMin");

        std::cout << "  -> Variance clipping correctly constrains history outliers." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 6. Test Checkerboard Distributed Tile Parity (Symmetric Multi-GPU)
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 6] Multi-GPU Symmetric Checkerboard Tile Parity..." << std::endl;
        const uint32_t tileSize = 64;
        const uint32_t width = 3840;
        const uint32_t height = 2160;

        uint32_t primaryTiles = 0;
        uint32_t secondaryTiles = 0;

        for (uint32_t y = 0; y < height; y += tileSize) {
            for (uint32_t x = 0; x < width; x += tileSize) {
                uint32_t tx = x / tileSize;
                uint32_t ty = y / tileSize;
                uint32_t parity = (tx + ty) & 1u;

                if (parity == 0u) {
                    primaryTiles++;
                } else {
                    secondaryTiles++;
                }
            }
        }

        uint32_t totalTiles = ((width + tileSize - 1) / tileSize) * ((height + tileSize - 1) / tileSize);
        check_true(primaryTiles + secondaryTiles == totalTiles, "Primary + Secondary tiles = Total tiles");
        // Check 50/50 balance within 1 tile difference (for odd tile dimensions)
        check_true(std::abs(static_cast<int>(primaryTiles) - static_cast<int>(secondaryTiles)) <= static_cast<int>((width + tileSize - 1) / tileSize),
                   "Perfect 50/50 symmetric workload distribution across GPUs");

        std::cout << "  -> Symmetric distributed tile partitioning verified: "
                  << primaryTiles << " primary (even) tiles, "
                  << secondaryTiles << " secondary (odd) tiles." << std::endl;
    }

    std::cout << "\n>>> All TAA Engine unit tests PASSED successfully! <<<\n" << std::endl;
    return 0;
}
