#include "scene/Light.hpp"
#include "scene/LightMapping.hpp"
#include "scene/EmissiveGeometryLights.hpp"
#include "scene/LightTree.hpp"
#include "scene/ProceduralScene.hpp"

#include <glm/glm.hpp>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using namespace pathways;

namespace {

void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << std::endl;
        std::exit(1);
    }
}

void assert_near(float actual, float expected, float eps, const char* msg) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > eps) {
        std::cerr << "[FAIL] " << msg << ": actual=" << actual << " expected=" << expected
                  << " (eps=" << eps << ")" << std::endl;
        std::exit(1);
    }
}

TriangleGPU makeTri(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, uint32_t matId) {
    TriangleGPU t{};
    t.v0.position = glm::vec4(p0, 0.0f);
    t.v1.position = glm::vec4(p1, 0.0f);
    t.v2.position = glm::vec4(p2, 0.0f);
    glm::vec3 c = glm::cross(p1 - p0, p2 - p0);
    glm::vec3 n = (glm::length(c) > 1e-8f) ? glm::normalize(c) : glm::vec3(0.0f, 1.0f, 0.0f);
    t.v0.normal = glm::vec4(n, 0.0f);
    t.v1.normal = glm::vec4(n, 0.0f);
    t.v2.normal = glm::vec4(n, 0.0f);
    t.materialId = matId;
    return t;
}

LightGPU makeQuadLight(const glm::vec3& corner, const glm::vec3& u, const glm::vec3& v, const glm::vec3& emitRgb, float area = 0.0f) {
    LightGPU l{};
    glm::vec3 cr = glm::cross(u, v);
    glm::vec3 n = (glm::length(cr) > 1e-8f) ? glm::normalize(cr) : glm::vec3(0.0f, 1.0f, 0.0f);
    float computedArea = (area > 0.0f) ? area : glm::length(cr);
    l.position = glm::vec4(corner, static_cast<float>(LIGHT_AREA_QUAD));
    l.u = glm::vec4(u, 0.0f);
    l.v = glm::vec4(v, 0.0f);
    l.normal = glm::vec4(n, 0.0f);
    l.emission = glm::vec4(emitRgb, computedArea);
    return l;
}

} // namespace

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: LightMapping, EmissiveLights & Alias Tests    " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // -------------------------------------------------------------------------
    // TEST 1: Light Flux & Vose Alias Table Unbiasedness
    // -------------------------------------------------------------------------
    std::cout << "[TEST 1] calculateLightFlux & buildLightAliasTable..." << std::endl;
    {
        constexpr float kPi = 3.14159265359f;
        LightGPU quad = makeQuadLight(glm::vec3(0.0f), glm::vec3(2.0f, 0.0f, 0.0f), glm::vec3(0.0f, 0.0f, 3.0f), glm::vec3(1.0f, 1.0f, 1.0f), 0.0f);
        quad.emission.w = 0.0f; // force auto area calculation
        float qFlux = calculateLightFlux(quad);
        assert_near(qFlux, 6.0f * kPi, 1e-3f, "LIGHT_AREA_QUAD auto-area flux == 6 * pi");

        LightGPU tri = quad;
        tri.position.w = static_cast<float>(LIGHT_AREA_TRIANGLE);
        tri.emission.w = 0.0f; // auto area calculation = 0.5 * |u x v|
        float tFlux = calculateLightFlux(tri);
        assert_near(tFlux, 0.5f * qFlux, 1e-3f, "LIGHT_AREA_TRIANGLE auto-area flux is 0.5x quad flux");

        LightGPU spot{};
        spot.position = glm::vec4(0.0f, 5.0f, 0.0f, static_cast<float>(LIGHT_SPOT));
        spot.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
        spot.v = glm::vec4(0.0f, 0.0f, 0.0f, 0.5f); // cos(60 deg) -> solidAngle = 2*pi*(1 - 0.5) = pi
        spot.emission = glm::vec4(2.0f, 2.0f, 2.0f, 1.0f);
        assert_near(calculateLightFlux(spot), 2.0f * kPi, 1e-3f, "LIGHT_SPOT flux matches 2*pi*(1-outerCos)*lum");

        LightGPU dir{};
        dir.position = glm::vec4(0.0f, 0.0f, 0.0f, static_cast<float>(LIGHT_DIRECTIONAL));
        dir.emission = glm::vec4(3.0f, 3.0f, 3.0f, 1.0f);
        assert_near(calculateLightFlux(dir), 3.0f * kPi, 1e-3f, "LIGHT_DIRECTIONAL flux matches lum*pi");

        // Zero-flux alias table fallback -> uniform PDF 1/N
        std::vector<LightGPU> zeroLights(4, LightGPU{});
        buildLightAliasTable(zeroLights);
        for (size_t i = 0; i < zeroLights.size(); ++i) {
            assert_near(zeroLights[i].sampling.z, 0.25f, 1e-6f, "Zero-flux alias table yields uniform PDF");
            assert_near(zeroLights[i].sampling.w, 0.0f, 1e-6f, "Zero-flux alias table records 0 flux");
        }

        // Non-uniform multi-light alias table Monte Carlo verification
        std::vector<LightGPU> lights;
        lights.push_back(makeQuadLight(glm::vec3(0.0f), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(1.0f)));
        lights.push_back(makeQuadLight(glm::vec3(0.0f), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(4.0f)));
        lights.push_back(makeQuadLight(glm::vec3(0.0f), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(10.0f)));
        lights.push_back(makeQuadLight(glm::vec3(0.0f), glm::vec3(1, 0, 0), glm::vec3(0, 1, 0), glm::vec3(5.0f)));
        buildLightAliasTable(lights);

        double pdfSum = 0.0;
        for (const auto& l : lights) {
            check_true(l.sampling.x >= 0.0f && l.sampling.x <= 1.0f, "Alias threshold q in [0, 1]");
            pdfSum += l.sampling.z;
        }
        assert_near(static_cast<float>(pdfSum), 1.0f, 1e-5f, "Discrete alias PDFs sum to 1.0");

        std::mt19937 rng(12345u);
        std::uniform_real_distribution<float> u01(0.0f, 0.9999999f);
        std::vector<uint32_t> counts(lights.size(), 0u);
        constexpr uint32_t kDraws = 100000u;
        for (uint32_t d = 0; d < kDraws; ++d) {
            uint32_t col = std::min(static_cast<uint32_t>(u01(rng) * lights.size()), static_cast<uint32_t>(lights.size() - 1));
            float coin = u01(rng);
            uint32_t picked = (coin < lights[col].sampling.x) ? col : std::bit_cast<uint32_t>(lights[col].sampling.y);
            check_true(picked < lights.size(), "Alias sample index in range");
            counts[picked]++;
        }
        for (size_t i = 0; i < lights.size(); ++i) {
            float empProb = static_cast<float>(counts[i]) / static_cast<float>(kDraws);
            assert_near(empProb, lights[i].sampling.z, 0.01f, "Alias table empirical frequency matches discretePdf");
        }
    }
    std::cout << "[PASS] Light flux & Vose alias table tests passed." << std::endl;

    // -------------------------------------------------------------------------
    // TEST 2: EmissiveGeometryLights (Quad Pairing, Unpaired Tri, Skip Guards)
    // -------------------------------------------------------------------------
    std::cout << "[TEST 2] appendEmissiveGeometryLights pairing & guards..." << std::endl;
    {
        auto makeInstancedEmissiveScene = []() {
            SceneData sc;
            MaterialGPU emitMat{};
            emitMat.emissive = glm::vec4(8.0f, 6.0f, 4.0f, 1.0f);
            emitMat.type = MATERIAL_EMISSIVE;
            MaterialGPU texEmitMat = emitMat;
            texEmitMat.emissiveTex = 1u;
            sc.materials = { emitMat, texEmitMat };

            // Quad in XY plane [0..2] x [0..1] split into 2 triangles (paired into 1 quad)
            sc.triangles.push_back(makeTri({0, 0, 0}, {2, 0, 0}, {2, 1, 0}, 0u));
            sc.triangles.push_back(makeTri({0, 0, 0}, {2, 1, 0}, {0, 1, 0}, 0u));
            // Unpaired triangle -> becomes 1 LIGHT_AREA_TRIANGLE
            sc.triangles.push_back(makeTri({5, 0, 0}, {6, 0, 0}, {5, 1, 0}, 0u));
            // Textured emissive triangle -> skipped (BSDF-only)
            sc.triangles.push_back(makeTri({8, 0, 0}, {9, 0, 0}, {8, 1, 0}, 1u));

            sc.blasRanges.push_back({ 0u, 4u, 4u });

            SceneInstance i0{ 0u, glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 2.0f, 0.0f)), 0u };
            SceneInstance i1{ 0u, glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 2.0f, 0.0f)), 1u };
            sc.instances = { i0, i1 };
            sc.instanceData.resize(2);
            return sc;
        };

        // 2a. Normal instanced pairing: 2 proto emitters (1 quad + 1 triangle) x 2 instances = 4 lights
        {
            SceneData sc = makeInstancedEmissiveScene();
            EmissiveGeometryLightStats st = appendEmissiveGeometryLights(sc);
            check_true(st.skipReason == nullptr, "Valid instanced scene must not skip");
            check_true(st.protoQuads == 2u, "1 paired quad + 1 unpaired triangle = 2 proto emitters");
            check_true(st.textured == 1u, "Textured emissive triangle counted in stats.textured");
            check_true(st.addedLights == 4u && sc.lights.size() == 4u, "2 proto emitters x 2 instances = 4 lights");
            for (const auto& l : sc.lights) {
                check_true(isEmissiveGeometryLight(l), "Appended light carries emissive geometry marker");
            }
            check_true(static_cast<uint32_t>(sc.lights[0].position.w) == LIGHT_AREA_QUAD, "Paired triangles form LIGHT_AREA_QUAD");
            assert_near(sc.lights[0].emission.w, 2.0f, 1e-4f, "Paired quad area == 2.0");
            check_true(static_cast<uint32_t>(sc.lights[1].position.w) == LIGHT_AREA_TRIANGLE, "Unpaired triangle forms LIGHT_AREA_TRIANGLE");
            assert_near(sc.lights[1].emission.w, 0.5f, 1e-4f, "Unpaired triangle area == 0.5");
            check_true(sc.instanceData[0].flags == 1u && sc.instanceData[1].flags == 3u,
                       "Instance flags record 1-based first light index");
            check_true(sc.instancedEmissiveTriQuad.size() == 4u &&
                       sc.instancedEmissiveTriQuad[0] == 0u && sc.instancedEmissiveTriQuad[1] == 0u &&
                       sc.instancedEmissiveTriQuad[2] == 1u && sc.instancedEmissiveTriQuad[3] == 0xFFFFFFFFu,
                       "instancedEmissiveTriQuad maps triangles to prototype emitter index");
        }

        // 2b. Monolithic skip
        {
            SceneData mono;
            EmissiveGeometryLightStats st = appendEmissiveGeometryLights(mono);
            check_true(st.skipReason != nullptr && st.addedLights == 0u, "Monolithic scene skips cleanly");
        }

        // 2c. Animated instance skip
        {
            SceneData anim = makeInstancedEmissiveScene();
            anim.animatedInstances.push_back(AnimatedInstance{});
            EmissiveGeometryLightStats st = appendEmissiveGeometryLights(anim);
            check_true(st.skipReason != nullptr && st.addedLights == 0u, "Animated scene skips cleanly");
        }

        // 2d. Differing materialOffset skip
        {
            SceneData diffMo = makeInstancedEmissiveScene();
            diffMo.instanceData[1].materialOffset = 1u;
            EmissiveGeometryLightStats st = appendEmissiveGeometryLights(diffMo);
            check_true(st.skipReason != nullptr && st.addedLights == 0u, "Differing materialOffset skips cleanly");
        }

        // 2e. Max-light budget exceeded skip
        {
            SceneData budget = makeInstancedEmissiveScene();
            EmissiveGeometryLightStats st = appendEmissiveGeometryLights(budget, 2u);
            check_true(st.skipReason != nullptr && st.addedLights == 0u, "Light budget overflow skips cleanly");
        }

        // 2f. Duplicate explicit area light covering emissive geometry skip
        {
            SceneData dup = makeInstancedEmissiveScene();
            // Covers the translated quad of instance 0 at y = 2.0
            dup.lights.push_back(makeQuadLight({0.0f, 2.0f, 0.0f}, {2.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, {8.0f, 6.0f, 4.0f}));
            EmissiveGeometryLightStats st = appendEmissiveGeometryLights(dup);
            check_true(st.skipReason != nullptr && st.addedLights == 0u,
                       "Duplicate explicit area light skips to prevent double-counting");
        }

        // 2g. Degenerate instance transform -> zero-flux placeholder preserves indexing
        {
            SceneData deg = makeInstancedEmissiveScene();
            deg.instances[0].transform = glm::mat4(0.0f);
            EmissiveGeometryLightStats st = appendEmissiveGeometryLights(deg);
            check_true(st.skipReason == nullptr && st.addedLights == 4u, "Degenerate instance keeps dense quad indexing");
            assert_near(calculateLightFlux(deg.lights[0]), 0.0f, 1e-6f, "Degenerate quad placeholder has zero flux");
        }
    }
    std::cout << "[PASS] EmissiveGeometryLights tests passed." << std::endl;

    // -------------------------------------------------------------------------
    // TEST 3: LightMapping (Full, Partial, Mismatch, Uncovered, Ambiguous, Instanced)
    // -------------------------------------------------------------------------
    std::cout << "[TEST 3] buildEmissiveLightMapping MIS correspondence..." << std::endl;
    {
        // 3a. Full coverage -> LIGHT_FLAG_MIS + lightIdx + 1
        {
            SceneData sc;
            MaterialGPU emitMat{};
            emitMat.emissive = glm::vec4(12.0f, 10.0f, 8.0f, 1.0f);
            sc.materials = { emitMat };
            sc.triangles = {
                makeTri({0, 0, 0}, {2, 0, 0}, {2, 2, 0}, 0u),
                makeTri({0, 0, 0}, {2, 2, 0}, {0, 2, 0}, 0u)
            };
            sc.lights = { makeQuadLight({0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {12.0f, 10.0f, 8.0f}) };

            std::vector<uint32_t> triMap;
            LightMappingStats st = buildEmissiveLightMapping(sc, triMap);
            check_true(st.misLights == 1u && st.neeOnlyAreaLights == 0u, "Fully tiled quad gets LIGHT_FLAG_MIS");
            check_true((getLightFlags(sc.lights[0]) & LIGHT_FLAG_MIS) != 0u, "LIGHT_FLAG_MIS bit set");
            check_true(triMap.size() == 2u && triMap[0] == 1u && triMap[1] == 1u, "Matched triangles map to lightIdx + 1");
        }

        // 3b. Partial coverage (only 1 of 2 triangles) -> NEE-only
        {
            SceneData sc;
            MaterialGPU emitMat{};
            emitMat.emissive = glm::vec4(12.0f, 10.0f, 8.0f, 1.0f);
            sc.materials = { emitMat };
            sc.triangles = { makeTri({0, 0, 0}, {2, 0, 0}, {2, 2, 0}, 0u) };
            sc.lights = { makeQuadLight({0, 0, 0}, {2, 0, 0}, {0, 2, 0}, {12.0f, 10.0f, 8.0f}) };

            std::vector<uint32_t> triMap;
            LightMappingStats st = buildEmissiveLightMapping(sc, triMap);
            check_true(st.misLights == 0u && st.neeOnlyAreaLights == 1u, "Partially tiled quad stays NEE-only");
            check_true((getLightFlags(sc.lights[0]) & LIGHT_FLAG_MIS) == 0u, "LIGHT_FLAG_MIS cleared on partial quad");
            check_true(triMap[0] == TRI_LIGHT_NEE_ONLY, "Partially covered triangle marked TRI_LIGHT_NEE_ONLY");
        }

        // 3c. Emission mismatch -> NEE-only
        {
            SceneData sc;
            MaterialGPU emitMat{};
            emitMat.emissive = glm::vec4(5.0f, 5.0f, 5.0f, 1.0f);
            sc.materials = { emitMat };
            sc.triangles = {
                makeTri({0, 0, 0}, {1, 0, 0}, {1, 1, 0}, 0u),
                makeTri({0, 0, 0}, {1, 1, 0}, {0, 1, 0}, 0u)
            };
            sc.lights = { makeQuadLight({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {20.0f, 20.0f, 20.0f}) };

            std::vector<uint32_t> triMap;
            LightMappingStats st = buildEmissiveLightMapping(sc, triMap);
            check_true(st.misLights == 0u && st.neeOnlyAreaLights == 1u, "Emission mismatch stays NEE-only");
            check_true(triMap[0] == TRI_LIGHT_NEE_ONLY && triMap[1] == TRI_LIGHT_NEE_ONLY,
                       "Emission-mismatched triangles marked TRI_LIGHT_NEE_ONLY");
        }

        // 3d. Uncovered emissive triangle -> TRI_LIGHT_NONE (BSDF weight 1)
        {
            SceneData sc;
            MaterialGPU emitMat{};
            emitMat.emissive = glm::vec4(4.0f, 4.0f, 4.0f, 1.0f);
            sc.materials = { emitMat };
            sc.triangles = { makeTri({10, 10, 10}, {11, 10, 10}, {10, 11, 10}, 0u) };

            std::vector<uint32_t> triMap;
            LightMappingStats st = buildEmissiveLightMapping(sc, triMap);
            check_true(st.emissiveTriangles == 1u && st.matchedTriangles == 0u && st.neeOnlyTriangles == 0u,
                       "Uncovered emitter has 0 matched and 0 NEE-only");
            check_true(triMap[0] == TRI_LIGHT_NONE, "Uncovered emitter marked TRI_LIGHT_NONE");
        }

        // 3e. Ambiguous instancing (same prototype triangle matches 2 different explicit lights)
        {
            SceneData sc;
            MaterialGPU emitMat{};
            emitMat.emissive = glm::vec4(10.0f, 10.0f, 10.0f, 1.0f);
            sc.materials = { emitMat };
            sc.triangles = {
                makeTri({0, 0, 0}, {1, 0, 0}, {1, 1, 0}, 0u),
                makeTri({0, 0, 0}, {1, 1, 0}, {0, 1, 0}, 0u)
            };
            sc.blasRanges = { { 0u, 2u, 2u } };
            sc.instances = {
                { 0u, glm::mat4(1.0f), 0u },
                { 0u, glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, 0.0f, 0.0f)), 1u }
            };
            sc.instanceData.resize(2);
            sc.lights = {
                makeQuadLight({0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {10.0f, 10.0f, 10.0f}),
                makeQuadLight({5, 0, 0}, {1, 0, 0}, {0, 1, 0}, {10.0f, 10.0f, 10.0f})
            };

            std::vector<uint32_t> triMap;
            LightMappingStats st = buildEmissiveLightMapping(sc, triMap);
            check_true(st.misLights == 0u && st.neeOnlyAreaLights == 2u,
                       "Ambiguous multi-instance explicit lights stay NEE-only");
            check_true(triMap[0] == TRI_LIGHT_NEE_ONLY && triMap[1] == TRI_LIGHT_NEE_ONLY,
                       "Ambiguously mapped prototype triangles marked TRI_LIGHT_NEE_ONLY");
        }

        // 3f. Integration: appendEmissiveGeometryLights + buildEmissiveLightMapping
        {
            SceneData sc;
            MaterialGPU emitMat{};
            emitMat.emissive = glm::vec4(9.0f, 9.0f, 9.0f, 1.0f);
            sc.materials = { emitMat };
            sc.triangles = {
                makeTri({0, 0, 0}, {1, 0, 0}, {1, 1, 0}, 0u),
                makeTri({0, 0, 0}, {1, 1, 0}, {0, 1, 0}, 0u)
            };
            sc.blasRanges = { { 0u, 2u, 2u } };
            sc.instances = {
                { 0u, glm::mat4(1.0f), 0u },
                { 0u, glm::translate(glm::mat4(1.0f), glm::vec3(5.0f, 0.0f, 0.0f)), 1u }
            };
            sc.instanceData.resize(2);

            EmissiveGeometryLightStats egSt = appendEmissiveGeometryLights(sc);
            check_true(egSt.addedLights == 2u, "Added 2 instanced emissive geometry lights");
            std::vector<uint32_t> triMap;
            LightMappingStats lmSt = buildEmissiveLightMapping(sc, triMap);
            check_true(lmSt.misLights == 2u, "Both instanced emissive lights flagged LIGHT_FLAG_MIS");
            check_true(triMap[0] == (TRI_LIGHT_INSTANCED_BIT | 0u) && triMap[1] == (TRI_LIGHT_INSTANCED_BIT | 0u),
                       "Triangles tagged with TRI_LIGHT_INSTANCED_BIT | quadIdx");
        }
    }
    std::cout << "[PASS] LightMapping MIS correspondence tests passed." << std::endl;

    // -------------------------------------------------------------------------
    // TEST 4: LightTree Triangle Lights, Zero-Flux Stability & Path Replay
    // -------------------------------------------------------------------------
    std::cout << "[TEST 4] LightTree triangle bounds, zero-flux & bit-path replay..." << std::endl;
    {
        std::vector<LightGPU> lights;
        for (int i = 0; i < 12; ++i) {
            LightGPU l{};
            l.position = glm::vec4(static_cast<float>(i) * 2.0f, 1.0f, 0.0f, static_cast<float>(LIGHT_AREA_TRIANGLE));
            l.u = glm::vec4(1.0f, 0.0f, 0.0f, 0.0f);
            l.v = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
            l.normal = glm::vec4(0.0f, 1.0f, 0.0f, 0.0f);
            l.emission = (i % 4 == 0) ? glm::vec4(0.0f, 0.0f, 0.0f, 0.5f) : glm::vec4(6.0f + i, 4.0f, 2.0f, 0.5f);
            lights.push_back(l);
        }

        std::vector<LightTreeNodeGPU> nodes;
        buildLightTree(lights, nodes);
        check_true(computeLightTreePaths(nodes, lights), "All 12 triangle lights have valid root-to-leaf paths");

        // Replay each light's bit-path from root (0) and verify it lands on the exact leaf lightIndex
        for (uint32_t li = 0; li < lights.size(); ++li) {
            uint32_t bits = std::bit_cast<uint32_t>(lights[li].pad0.y);
            uint32_t depth = std::bit_cast<uint32_t>(lights[li].pad0.z);
            check_true(depth > 0u && depth <= 32u, "Valid encoded tree path depth");
            uint32_t cur = 0u;
            for (uint32_t d = 0; d < depth; ++d) {
                check_true(!nodes[cur].isLeaf(), "Internal node encountered along path");
                uint32_t childBase = nodes[cur].firstChild();
                cur = childBase + ((bits >> d) & 1u);
                check_true(cur < nodes.size(), "Path step stays within nodes bounds");
            }
            check_true(nodes[cur].isLeaf() && nodes[cur].lightIndex() == li,
                       "Bit-path replay reaches exact light leaf");
            check_true(std::isfinite(nodes[cur].flux()), "Leaf flux is finite (even for zero-flux lights)");
        }
    }
    std::cout << "[PASS] LightTree triangle & bit-path replay tests passed." << std::endl;

    std::cout << "\nALL LIGHT MAPPING & EMISSIVE GEOMETRY TESTS PASSED!" << std::endl;
    return 0;
}
