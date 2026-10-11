#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#include "scene/SceneGeometryPipeline.hpp"
#include "scene/ProceduralScene.hpp"
#include "scene/GltfLoader.hpp"
#include "core/Logger.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <cstdlib>

using namespace pathways;

namespace {

void check_true(bool cond, const char* msg = "") {
    if (!cond) {
        std::cerr << "Assertion failed: condition is false! " << msg << std::endl;
        std::exit(1);
    }
}

void check_near_vec3(const glm::vec3& a, const glm::vec3& b, float eps = 0.01f, const char* msg = "") {
    glm::vec3 diff = glm::abs(a - b);
    if (diff.x > eps || diff.y > eps || diff.z > eps) {
        std::cerr << "Assertion failed: (" << a.x << ", " << a.y << ", " << a.z << ") != ("
                  << b.x << ", " << b.y << ", " << b.z << ") " << msg << std::endl;
        std::exit(1);
    }
}

TriangleGPU makeTriangle(const glm::vec3& p0, const glm::vec3& p1, const glm::vec3& p2, uint32_t matId) {
    TriangleGPU tri{};
    tri.v0.position = glm::vec4(p0, 1.0f);
    tri.v1.position = glm::vec4(p1, 1.0f);
    tri.v2.position = glm::vec4(p2, 1.0f);
    glm::vec3 n = glm::normalize(glm::cross(p1 - p0, p2 - p0));
    tri.v0.normal = glm::vec4(n, 0.0f);
    tri.v1.normal = glm::vec4(n, 0.0f);
    tri.v2.normal = glm::vec4(n, 0.0f);
    tri.materialId = matId;
    return tri;
}

} // anonymous namespace

int main() {
    Logger::setLogLevel(LogLevel::Info);

    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Caustics Dielectric Target Verification Suite " << std::endl;
    std::cout << "==========================================================" << std::endl;

    SceneGeometryPipeline geomPipeline(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE);

    // -------------------------------------------------------------------------
    // Test 1: Synthetic Triangle Geometry with MATERIAL_DIELECTRIC
    // -------------------------------------------------------------------------
    std::cout << "[Step 1] Validating MATERIAL_DIELECTRIC bounding box extraction..." << std::endl;
    {
        SceneData scene{};
        MaterialGPU mat{};
        mat.type = MATERIAL_DIELECTRIC;
        mat.ior = 1.5f;
        scene.materials.push_back(mat);

        scene.triangles.push_back(makeTriangle(glm::vec3(-0.5f, 0.0f, -0.5f),
                                               glm::vec3(0.5f, 0.0f, -0.5f),
                                               glm::vec3(0.0f, 1.0f, 0.5f), 0));

        uint32_t numOpaque = 0;
        geomPipeline.partitionSceneGeometry(scene, numOpaque);

        check_true(scene.hasDielectrics, "hasDielectrics must be true for MATERIAL_DIELECTRIC");
        check_near_vec3(scene.dielectricBoundsMin, glm::vec3(-0.5f, 0.0f, -0.5f), 0.01f, "dielectricBoundsMin mismatch");
        check_near_vec3(scene.dielectricBoundsMax, glm::vec3(0.5f, 1.0f, 0.5f), 0.01f, "dielectricBoundsMax mismatch");
        check_true(numOpaque == 0, "Dielectric triangle must be partitioned as non-opaque");
    }

    // -------------------------------------------------------------------------
    // Test 2: Synthetic Transmission Material without Thickness (Regression Guard)
    // -------------------------------------------------------------------------
    std::cout << "[Step 2] Validating Transmission material without KHR_materials_volume thickness..." << std::endl;
    {
        SceneData scene{};
        MaterialGPU mat{};
        mat.type = MATERIAL_DIFFUSE; // Standard glTF material with transmission extension
        mat.transmission = 1.0f;
        mat.thickness = 0.0f; // Thickness absent in basic transmission assets
        scene.materials.push_back(mat);

        scene.triangles.push_back(makeTriangle(glm::vec3(1.0f, 2.0f, 3.0f),
                                               glm::vec3(2.0f, 3.0f, 4.0f),
                                               glm::vec3(1.5f, 2.5f, 3.5f), 0));

        uint32_t numOpaque = 0;
        geomPipeline.partitionSceneGeometry(scene, numOpaque);

        check_true(scene.hasDielectrics, "Transmission > 0.05 must flag hasDielectrics = true regardless of thickness");
        check_near_vec3(scene.dielectricBoundsMin, glm::vec3(1.0f, 2.0f, 3.0f), 0.01f, "dielectricBoundsMin mismatch");
        check_near_vec3(scene.dielectricBoundsMax, glm::vec3(2.0f, 3.0f, 4.0f), 0.01f, "dielectricBoundsMax mismatch");
    }

    // -------------------------------------------------------------------------
    // Test 3: Multi-BLAS Instance Transformation
    // -------------------------------------------------------------------------
    std::cout << "[Step 3] Validating Multi-BLAS instanced dielectric world-space bounds..." << std::endl;
    {
        SceneData scene{};
        MaterialGPU mat{};
        mat.type = MATERIAL_DIELECTRIC;
        scene.materials.push_back(mat);

        // BLAS 0: local bounds [-1, -1, -1] to [1, 1, 1]
        scene.triangles.push_back(makeTriangle(glm::vec3(-1.0f, -1.0f, -1.0f),
                                               glm::vec3(1.0f, -1.0f, -1.0f),
                                               glm::vec3(0.0f, 1.0f, 1.0f), 0));
        BlasGeometryRange range{};
        range.firstTriangle = 0;
        range.triangleCount = 1;
        scene.blasRanges.push_back(range);

        // Instance 0: translated by (+10, +20, +30)
        SceneInstance inst{};
        inst.blasIndex = 0;
        inst.transform = glm::translate(glm::mat4(1.0f), glm::vec3(10.0f, 20.0f, 30.0f));
        scene.instances.push_back(inst);
        scene.instanceData.resize(1);

        uint32_t numOpaque = 0;
        geomPipeline.partitionSceneGeometry(scene, numOpaque);

        check_true(scene.hasDielectrics, "Instanced dielectric must flag hasDielectrics = true");
        check_near_vec3(scene.dielectricBoundsMin, glm::vec3(9.0f, 19.0f, 29.0f), 0.01f, "Instanced dielectricBoundsMin");
        check_near_vec3(scene.dielectricBoundsMax, glm::vec3(11.0f, 21.0f, 31.0f), 0.01f, "Instanced dielectricBoundsMax");
    }

    // -------------------------------------------------------------------------
    // Test 4: Procedural Sphere Dielectric Detection
    // -------------------------------------------------------------------------
    std::cout << "[Step 4] Validating procedural dielectric sphere bounds..." << std::endl;
    {
        SceneData scene{};
        MaterialGPU mat{};
        mat.type = MATERIAL_DIELECTRIC;
        scene.materials.push_back(mat);

        SphereGPU sph{};
        sph.centerRadius = glm::vec4(5.0f, 5.0f, 5.0f, 2.0f);
        sph.materialId = 0;
        scene.spheres.push_back(sph);

        uint32_t numOpaque = 0;
        geomPipeline.partitionSceneGeometry(scene, numOpaque);

        check_true(scene.hasDielectrics, "Procedural sphere must flag hasDielectrics = true");
        check_near_vec3(scene.dielectricBoundsMin, glm::vec3(3.0f, 3.0f, 3.0f), 0.01f, "Sphere dielectricBoundsMin");
        check_near_vec3(scene.dielectricBoundsMax, glm::vec3(7.0f, 7.0f, 7.0f), 0.01f, "Sphere dielectricBoundsMax");
    }

    // -------------------------------------------------------------------------
    // Test 5: Procedural Cornell Box
    // -------------------------------------------------------------------------
    std::cout << "[Step 5] Validating Procedural Cornell Box scene dielectric setup..." << std::endl;
    {
        SceneData scene = ProceduralScene::createCornellBox();
        uint32_t numOpaque = 0;
        geomPipeline.partitionSceneGeometry(scene, numOpaque);

        check_true(scene.hasDielectrics, "Procedural Cornell Box must have dielectric geometry");
        check_true(scene.dielectricBoundsMin.x < scene.dielectricBoundsMax.x, "Cornell Box dielectricBounds x valid");
        check_true(scene.dielectricBoundsMin.y < scene.dielectricBoundsMax.y, "Cornell Box dielectricBounds y valid");
        check_true(scene.dielectricBoundsMin.z < scene.dielectricBoundsMax.z, "Cornell Box dielectricBounds z valid");
    }

    // -------------------------------------------------------------------------
    // Test 6: glTF Asset Ingestion (cornell_caustic_extended.glb)
    // -------------------------------------------------------------------------
    std::cout << "[Step 6] Validating glTF asset ingestion (cornell_caustic_extended.glb)..." << std::endl;
    const std::string causticGlbPath = "scenes/cornell-caustic/cornell_caustic_extended.glb";
    if (std::filesystem::exists(causticGlbPath)) {
        SceneData scene = GltfLoader::loadSceneData(causticGlbPath);
        check_true(scene.hasDielectrics, "GltfLoader must detect dielectric materials in cornell_caustic_extended.glb");
        check_true(scene.dielectricBoundsMin.x < scene.dielectricBoundsMax.x, "glTF dielectricBounds x valid");
        check_true(scene.dielectricBoundsMin.y < scene.dielectricBoundsMax.y, "glTF dielectricBounds y valid");
        check_true(scene.dielectricBoundsMin.z < scene.dielectricBoundsMax.z, "glTF dielectricBounds z valid");
        std::cout << "  -> Detected dielectric bounds: [("
                  << scene.dielectricBoundsMin.x << ", " << scene.dielectricBoundsMin.y << ", " << scene.dielectricBoundsMin.z
                  << ") to ("
                  << scene.dielectricBoundsMax.x << ", " << scene.dielectricBoundsMax.y << ", " << scene.dielectricBoundsMax.z
                  << ")]" << std::endl;
    } else {
        std::cout << "  -> Note: " << causticGlbPath << " not found, skipping disk asset load." << std::endl;
    }

    std::cout << "\n[SUCCESS] All 6 Caustics Detection & Bounding Volume tests PASSED cleanly!" << std::endl;
    return 0;
}
