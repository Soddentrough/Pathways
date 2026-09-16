#include "scene/ProceduralScene.hpp"
#include "scene/LightTree.hpp"
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <unordered_set>

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
    std::cout << "  Pathways: Testing Procedural Cyber City Megastructure" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Scene Generation
    SceneData scene = ProceduralScene::createCyberCityScene();
    std::cout << "[PASS] ProceduralScene::createCyberCityScene() completed successfully." << std::endl;

    // 2. BLAS Prototypes
    check_true(scene.blasRanges.size() == 19, "Expected exactly 19 BLAS prototypes");
    check_true(scene.meshRanges.size() == 19, "Expected exactly 19 MeshRange records");
    check_true(scene.triangles.size() >= 8000, "Expected >= 8000 prototype triangles");
    std::cout << "[PASS] Verified " << scene.blasRanges.size() << " BLAS ranges and "
              << scene.triangles.size() << " prototype triangles." << std::endl;

    for (size_t b = 0; b < scene.blasRanges.size(); ++b) {
        const auto& range = scene.blasRanges[b];
        check_true(range.triangleCount > 0, "BLAS triangleCount must be positive");
        check_true(range.firstTriangle + range.triangleCount <= scene.triangles.size(),
                   "BLAS range must fit within scene.triangles");
    }
    std::cout << "[PASS] All BLAS ranges have valid triangle bounds." << std::endl;

    // 3. Materials Validation & Palette Usage
    check_true(scene.materials.size() == 48, "Expected 48 materials in palette");
    std::vector<bool> materialUsed(scene.materials.size(), false);
    bool hasMetallic = false;
    bool hasDielectric = false;
    bool hasEmissive = false;
    bool hasDiffuse = false;

    for (const auto& tri : scene.triangles) {
        check_true(tri.materialId < scene.materials.size(), "Triangle materialId within bounds");
        materialUsed[tri.materialId] = true;
    }

    for (size_t m = 0; m < scene.materials.size(); ++m) {
        const auto& mat = scene.materials[m];
        check_true(!std::isnan(mat.albedo.r) && !std::isnan(mat.albedo.g) && !std::isnan(mat.albedo.b), "Valid albedo");
        check_true(mat.roughness >= 0.0f && mat.roughness <= 1.0f, "Valid roughness");
        check_true(mat.metallic >= 0.0f && mat.metallic <= 1.0f, "Valid metallic");

        if (mat.type == MATERIAL_METALLIC) hasMetallic = true;
        if (mat.type == MATERIAL_DIELECTRIC) hasDielectric = true;
        if (mat.type == MATERIAL_EMISSIVE) hasEmissive = true;
        if (mat.type == MATERIAL_DIFFUSE) hasDiffuse = true;
        if (!materialUsed[m]) {
            std::cerr << "Material " << m << " was not used in any prototype triangle!" << std::endl;
        }
        check_true(materialUsed[m], "Every material in palette must be actively used in geometry");
    }

    check_true(hasMetallic, "Metallic conductors present");
    check_true(hasDielectric, "Transmissive dielectrics present");
    check_true(hasEmissive, "Spectral emissives present");
    check_true(hasDiffuse, "Diffuse composites present");
    std::cout << "[PASS] Verified all 48 materials are actively used across all physical shading classes." << std::endl;

    // 4. Hardware TLAS Instances
    check_true(scene.instances.size() >= 3500, "Instances count must be >= 3500");
    check_true(scene.instanceData.size() == scene.instances.size(), "instanceData size matches instances");

    uint64_t totalInstancedTriangles = 0;
    std::vector<uint32_t> instanceCounts(scene.blasRanges.size(), 0);
    for (size_t i = 0; i < scene.instances.size(); ++i) {
        check_true(scene.instances[i].blasIndex < scene.blasRanges.size(), "Valid instance blasIndex");
        check_true(scene.instances[i].customIndex == static_cast<uint32_t>(i), "Instance customIndex match");
        instanceCounts[scene.instances[i].blasIndex]++;
        totalInstancedTriangles += scene.blasRanges[scene.instances[i].blasIndex].triangleCount;
    }
    check_true(totalInstancedTriangles >= 3000000ULL, "Total instanced triangles must be >= 3,000,000 (millions)");
    std::cout << "[PASS] Verified " << scene.instances.size() << " TLAS hardware instances ("
              << totalInstancedTriangles << " total instanced triangles)." << std::endl;

    // 5. Physical Light Sources & Sampling Structures
    check_true(scene.lights.size() >= 1500, "Light source count must be >= 1500");
    std::cout << "[PASS] Active physical light count: " << scene.lights.size() << " (>= 1500)." << std::endl;

    for (size_t l = 0; l < scene.lights.size(); ++l) {
        const auto& light = scene.lights[l];
        uint32_t type = static_cast<uint32_t>(light.position.w);
        check_true(type == LIGHT_AREA_QUAD || type == LIGHT_SPOT, "Valid light type");

        float flux = calculateLightFlux(light);
        check_true(flux > 0.0f, "Light flux must be positive");
        check_true(!std::isnan(flux) && !std::isinf(flux), "Light flux must be finite");

        glm::vec3 n(light.normal);
        float lenN = glm::length(n);
        assert_near(lenN, 1.0f, 0.05f, "Light normal must be unit-length");
    }
    std::cout << "[PASS] All light sources have valid flux, positive emission, and outward unit normals." << std::endl;

    // 6. Light Alias Table & Light Tree Construction
    buildLightAliasTable(scene.lights);
    for (size_t l = 0; l < scene.lights.size(); ++l) {
        check_true(scene.lights[l].sampling.w > 0.0f, "Alias table flux must be positive");
        check_true(scene.lights[l].sampling.z > 0.0f, "Alias table PDF must be positive");
    }
    std::cout << "[PASS] Light alias table built cleanly." << std::endl;

    std::vector<LightTreeNodeGPU> treeNodes;
    buildLightTree(scene.lights, treeNodes);
    check_true(!treeNodes.empty(), "Light tree nodes must not be empty");
    std::cout << "[PASS] Hierarchical light tree built successfully (" << treeNodes.size() << " nodes)." << std::endl;

    // 7. Spatial Bounds & Dielectric Refraction Bounds
    check_true(scene.boundsMin.x < scene.boundsMax.x, "Valid X bounds");
    check_true(scene.boundsMin.y < scene.boundsMax.y, "Valid Y bounds");
    check_true(scene.boundsMin.z < scene.boundsMax.z, "Valid Z bounds");
    check_true(scene.sceneRadius > 50.0f, "Scene radius must cover urban district");
    check_true(scene.hasDielectrics, "Scene has dielectric flag set");
    check_true(scene.dielectricBoundsMin.x < scene.dielectricBoundsMax.x, "Valid dielectric X bounds");
    check_true(scene.dielectricBoundsMin.y < scene.dielectricBoundsMax.y, "Valid dielectric Y bounds");
    check_true(scene.dielectricBoundsMin.z < scene.dielectricBoundsMax.z, "Valid dielectric Z bounds");
    check_true(scene.dielectricBoundsMax.y >= 100.0f, "Dielectric bounds must enclose crown observation dome (>= 100m)");
    std::cout << "[PASS] Scene bounds: [" << scene.boundsMin.x << ", " << scene.boundsMin.y << ", " << scene.boundsMin.z
              << "] to [" << scene.boundsMax.x << ", " << scene.boundsMax.y << ", " << scene.boundsMax.z << "]." << std::endl;
    std::cout << "[PASS] Dielectric bounds: [" << scene.dielectricBoundsMin.x << ", " << scene.dielectricBoundsMin.y << ", " << scene.dielectricBoundsMin.z
              << "] to [" << scene.dielectricBoundsMax.x << ", " << scene.dielectricBoundsMax.y << ", " << scene.dielectricBoundsMax.z << "]." << std::endl;

    // 8. Default Camera Parameters
    check_true(scene.hasCamera, "Scene must have camera");
    check_true(scene.cameraFov >= 40.0f && scene.cameraFov <= 90.0f, "Camera FOV within reasonable range");
    check_true(scene.focalDistance > 10.0f, "Valid camera focal distance");
    std::cout << "[PASS] Default camera configured at: ("
              << scene.cameraPosition.x << ", " << scene.cameraPosition.y << ", " << scene.cameraPosition.z
              << ") looking towards ("
              << scene.cameraTarget.x << ", " << scene.cameraTarget.y << ", " << scene.cameraTarget.z << ")." << std::endl;

    std::cout << "==========================================================" << std::endl;
    std::cout << "  [SUCCESS] All Procedural Cyber City Tests PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
