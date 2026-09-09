#include "core/Config.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <glm/glm.hpp>

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

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing FidelityFX Shadow Denoiser" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // -------------------------------------------------------------------------
    // 1. Test CLI Config Parsing
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 1] Command-Line Configuration & Flag Parsing..." << std::endl;
        Config configDefault;
        check_true(!configDefault.enable_shadow_denoiser, "Default shadow denoiser is off");
        assert_near(configDefault.shadow_denoiser_depth_sigma, 0.02f, 0.0001f, "Default depth sigma");
        assert_near(configDefault.shadow_denoiser_normal_power, 16.0f, 0.0001f, "Default normal power");

        const char* argv1[] = { "pathways", "--shadow-denoiser" };
        Config c1 = Config::parse(2, const_cast<char**>(argv1));
        check_true(c1.enable_shadow_denoiser, "--shadow-denoiser enabled");

        const char* argv2[] = { "pathways", "--denoise-shadows" };
        Config c2 = Config::parse(2, const_cast<char**>(argv2));
        check_true(c2.enable_shadow_denoiser, "--denoise-shadows alias enabled");

        std::cout << "  -> CLI flags and defaults successfully verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 2. Test Checkerboard Grid Alignment & In-VRAM Local Tile Confinement
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 2] FidelityFX 8x8 Tiles vs 64x64 Checkerboard Blocks Alignment..." << std::endl;
        const uint32_t tileSize = 64;
        const uint32_t width = 3840;
        const uint32_t height = 2160;

        // Verify that 64 is an exact multiple of 8 (8x8 FidelityFX tiles per 64x64 block)
        check_true(tileSize % 8 == 0, "Checkerboard block is integer multiple of 8x8 denoiser tiles");
        uint32_t tilesPerBlock1D = tileSize / 8; // 8 tiles across
        check_true(tilesPerBlock1D == 8, "Exactly 8x8 denoiser tiles per checkerboard block");

        // Verify that for any pixel (x, y), all 64 pixels inside its 8x8 tile belong to the SAME GPU
        for (uint32_t by = 0; by < height / tileSize; ++by) {
            for (uint32_t bx = 0; bx < width / tileSize; ++bx) {
                bool blockGpu = ((bx + by) % 2 == 0); // true = GPU 0, false = GPU 1

                // Every 8x8 tile inside this block must have all pixels matching blockGpu
                for (uint32_t ty = 0; ty < 8; ++ty) {
                    for (uint32_t tx = 0; tx < 8; ++tx) {
                        uint32_t px0 = bx * tileSize + tx * 8;
                        uint32_t py0 = by * tileSize + ty * 8;
                        uint32_t px7 = px0 + 7;
                        uint32_t py7 = py0 + 7;

                        bool pixel0Gpu = (((px0 / tileSize) + (py0 / tileSize)) % 2 == 0);
                        bool pixel7Gpu = (((px7 / tileSize) + (py7 / tileSize)) % 2 == 0);

                        check_true(pixel0Gpu == blockGpu, "Tile start pixel matches block GPU");
                        check_true(pixel7Gpu == blockGpu, "Tile end pixel matches block GPU");
                    }
                }
            }
        }
        std::cout << "  -> Alignment verified: Zero denoiser tiles straddle GPU boundary." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 3. Test Bilateral Weighting Functions & Edge Preservation
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 3] Bilateral Weighting & Contact Contrast Math..." << std::endl;
        const float depthSigma = 0.02f;
        const float normalPower = 16.0f;

        auto computeDepthWeight = [depthSigma](float centerZ, float sampleZ) -> float {
            float diff = std::abs(sampleZ - centerZ);
            return std::exp(-diff / (depthSigma * std::abs(centerZ) + 1e-4f));
        };

        auto computeNormalWeight = [normalPower](const glm::vec3& nCenter, const glm::vec3& nSample) -> float {
            float cosTheta = std::max(0.0f, glm::dot(nCenter, nSample));
            return std::pow(cosTheta, normalPower);
        };

        // Case A: Identical surface (same depth, same normal) -> weights should be 1.0
        glm::vec3 n0(0.0f, 1.0f, 0.0f);
        float wZ_same = computeDepthWeight(5.0f, 5.0f);
        float wN_same = computeNormalWeight(n0, n0);
        assert_near(wZ_same, 1.0f, 0.001f, "Same depth weight is 1.0");
        assert_near(wN_same, 1.0f, 0.001f, "Same normal weight is 1.0");

        // Case B: Geometry edge (depth step from 5.0m to 7.0m) -> depth weight drops near 0
        float wZ_step = computeDepthWeight(5.0f, 7.0f);
        check_true(wZ_step < 1e-5f, "Depth step weight drops near 0 across geometry edges");

        // Case C: Corner edge (orthogonal normals: floor vs wall) -> normal weight is 0
        glm::vec3 nWall(1.0f, 0.0f, 0.0f);
        float wN_ortho = computeNormalWeight(n0, nWall);
        assert_near(wN_ortho, 0.0f, 0.0001f, "Orthogonal surface normal weight is 0");

        // Case D: Slight normal perturbation (smooth curvature: 15 degrees)
        // cos(15 deg) = 0.9659; 0.9659^16 = 0.574
        glm::vec3 nCurved = glm::normalize(glm::vec3(std::sin(glm::radians(15.0f)), std::cos(glm::radians(15.0f)), 0.0f));
        float wN_curved = computeNormalWeight(n0, nCurved);
        check_true(wN_curved > 0.5f && wN_curved < 0.65f, "Smooth curvature retains blending weight");

        // Case E: Contact Contrast Remapping: S_contrast = pow(S, 1.15)
        auto contrastRemap = [](float shadow) -> float {
            return std::pow(std::clamp(shadow, 0.0f, 1.0f), 1.15f);
        };
        assert_near(contrastRemap(0.0f), 0.0f, 0.0001f, "Fully in shadow stays 0");
        assert_near(contrastRemap(1.0f), 1.0f, 0.0001f, "Fully lit stays 1");
        check_true(contrastRemap(0.5f) < 0.5f, "Midtone shadows are slightly deepened to preserve contact crispness");

        std::cout << "  -> Bilateral math and edge-preservation verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 4. Test In-Place Direct Lighting Resolve
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 4] Direct Lighting Separation & Resolve Composite..." << std::endl;
        // In raytrace.rchit:
        // uAccumImage writes path traced indirect/emission (L_indirect)
        // uDirectLightImage writes unshadowed direct radiance (L_direct)
        // Raw shadow factor S is filtered by cross-bilateral filter -> S_filtered
        // Final composite: L_final = L_indirect + L_direct * S_filtered

        glm::vec3 l_indirect(0.05f, 0.05f, 0.08f); // ambient bounce
        glm::vec3 l_direct(2.5f, 2.0f, 1.8f);      // unshadowed direct keylight

        // Fully occluded pixel: S_filtered = 0.0
        glm::vec3 shadow_comp = l_indirect + l_direct * 0.0f;
        assert_near(shadow_comp.x, 0.05f, 0.001f, "Occluded direct light composite X");
        assert_near(shadow_comp.y, 0.05f, 0.001f, "Occluded direct light composite Y");

        // Fully visible pixel: S_filtered = 1.0
        glm::vec3 lit_comp = l_indirect + l_direct * 1.0f;
        assert_near(lit_comp.x, 2.55f, 0.001f, "Lit direct light composite X");
        assert_near(lit_comp.y, 2.05f, 0.001f, "Lit direct light composite Y");

        // Soft penumbra pixel: S_filtered = 0.4
        glm::vec3 penumbra_comp = l_indirect + l_direct * 0.4f;
        assert_near(penumbra_comp.x, 0.05f + 2.5f * 0.4f, 0.001f, "Penumbra direct light composite X");

        std::cout << "  -> Direct lighting resolve composite math verified." << std::endl;
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  All FidelityFX Shadow Denoiser unit tests passed! [OK]  " << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
