#include "scene/UsdLoader.hpp"
#include "core/Logger.hpp"

#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/prim.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/base/gf/camera.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>

#include <iostream>
#include <fstream>
#include <filesystem>
#include <cmath>
#include <chrono>
#include <cassert>

using namespace pathways;

namespace {

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

} // anonymous namespace

int main() {
    Logger::setLogLevel(LogLevel::Info);

    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: OpenUSD Loader Verification Suite             " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // -------------------------------------------------------------------------
    // Test 1: File Extension Recognition
    // -------------------------------------------------------------------------
    std::cout << "[Step 1] Validating USD file format recognition..." << std::endl;
    check_true(UsdLoader::isUsdFile("assets/model.usd"), "usd extension recognized");
    check_true(UsdLoader::isUsdFile("assets/model.usda"), "usda extension recognized");
    check_true(UsdLoader::isUsdFile("assets/model.usdc"), "usdc extension recognized");
    check_true(UsdLoader::isUsdFile("assets/model.usdz"), "usdz extension recognized");
    check_true(UsdLoader::isUsdFile("ASSETS/MODEL.USDA"), "case-insensitive extension recognized");
    check_true(!UsdLoader::isUsdFile("assets/model.glb"), "glb is not usd");
    check_true(!UsdLoader::isUsdFile("assets/model.gltf"), "gltf is not usd");
    check_true(!UsdLoader::isUsdFile("assets/model.obj"), "obj is not usd");
    std::cout << "[PASS] Extension identification verified." << std::endl;

    // -------------------------------------------------------------------------
    // Test 2: Ingestion of Synthetic OpenUSD Stage (.usda)
    // -------------------------------------------------------------------------
    std::cout << "[Step 2] Constructing synthetic OpenUSD stage (USDA ASCII)..." << std::endl;
    std::filesystem::path tempUsdaPath = std::filesystem::temp_directory_path() / "test_pathways_scene.usda";

    {
        // USDA scene containing:
        // - Stage metadata (upAxis = Z, metersPerUnit = 1.0)
        // - A Quad Mesh (faceVertexCounts = [4]) bound to a UsdPreviewSurface material
        // - A RectLight (UsdLuxRectLight)
        std::string usdaContent = R"(#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1.0
    upAxis = "Z"
)

def Xform "Root"
{
    def Scope "Materials"
    {
        def Material "GoldMaterial"
        {
            token outputs:surface.connect = </Root/Materials/GoldMaterial/PBRShader.outputs:surface>

            def Shader "PBRShader"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (1.0, 0.766, 0.336)
                float inputs:metallic = 1.0
                float inputs:roughness = 0.15
                float inputs:ior = 1.5
                token outputs:surface
            }
        }
    }

    def Mesh "QuadPlane" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        rel material:binding = </Root/Materials/GoldMaterial>
        int[] faceVertexCounts = [4]
        int[] faceVertexIndices = [0, 1, 2, 3]
        point3f[] points = [(-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "vertex"
        )
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "vertex"
        )
    }

    def RectLight "CeilingLight"
    {
        color3f inputs:color = (1.0, 0.95, 0.9)
        float inputs:intensity = 500.0
        float inputs:exposure = 2.0
        float inputs:width = 2.0
        float inputs:height = 2.0
        double3 xformOp:translate = (0, 0, 5)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }

    def Camera "TestCam"
    {
        float2 clippingRange = (0.1, 1000)
        float focalLength = 35.0
        float focusDistance = 15.0
        float horizontalAperture = 36.0
        float verticalAperture = 24.0
        double3 xformOp:translate = (0, -10, 2)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
}
)";
        std::ofstream out(tempUsdaPath);
        out << usdaContent;
        out.close();
    }

    std::cout << "[Step 3] Parsing stage via UsdLoader::loadSceneData..." << std::endl;
    SceneData scene = UsdLoader::loadSceneData(tempUsdaPath.string());

    // -------------------------------------------------------------------------
    // Test 3: Validate Geometry & Fan Triangulation
    // -------------------------------------------------------------------------
    std::cout << "[Step 4] Validating quad triangulation and coordinate transformation..." << std::endl;
    // A single quad (4 vertices) triangulates into 2 triangles:
    // Tri 0: (0, 1, 2), Tri 1: (0, 2, 3)
    check_true(scene.triangles.size() == 2, "Quad mesh must triangulate into exactly 2 triangles");

    // In USDA: upAxis is "Z". The Z-up to Y-up matrix maps:
    // (X, Y, Z) -> (X, Z, -Y).
    // Original points: (-1, -1, 0), (1, -1, 0), (1, 1, 0), (-1, 1, 0)
    // Z=0 maps to Y=0 in Pathways coordinate space!
    const auto& t0 = scene.triangles[0];
    assert_near(t0.v0.position.y, 0.0f, 0.01f, "Z-up converted to Y-up plane (Y = 0)");
    assert_near(t0.v1.position.y, 0.0f, 0.01f, "Z-up converted to Y-up plane (Y = 0)");
    assert_near(t0.v2.position.y, 0.0f, 0.01f, "Z-up converted to Y-up plane (Y = 0)");

    // -------------------------------------------------------------------------
    // Test 4: Validate Material Extraction (UsdPreviewSurface)
    // -------------------------------------------------------------------------
    std::cout << "[Step 5] Validating UsdPreviewSurface material binding..." << std::endl;
    check_true(!scene.materials.empty(), "Scene must contain parsed materials");
    uint32_t matId = t0.materialId;
    check_true(matId < scene.materials.size(), "Valid material ID referenced by triangle");

    const auto& mat = scene.materials[matId];
    assert_near(mat.albedo.r, 1.0f, 0.02f, "Gold albedo R matches");
    assert_near(mat.albedo.g, 0.766f, 0.02f, "Gold albedo G matches");
    assert_near(mat.albedo.b, 0.336f, 0.02f, "Gold albedo B matches");
    assert_near(mat.metallic, 1.0f, 0.01f, "Metallic factor matches");
    assert_near(mat.roughness, 0.15f, 0.01f, "Roughness factor matches");
    check_true(mat.type == MATERIAL_METALLIC, "Material flagged as MATERIAL_METALLIC");
    std::cout << "[PASS] Material parameters accurately extracted." << std::endl;

    // -------------------------------------------------------------------------
    // Test 5: Validate Physical Lighting (UsdLux)
    // -------------------------------------------------------------------------
    std::cout << "[Step 6] Validating UsdLuxRectLight extraction..." << std::endl;
    check_true(scene.lights.size() == 1, "Scene must contain 1 parsed UsdLux light");
    const auto& light = scene.lights[0];
    check_true(light.position.w == static_cast<float>(LIGHT_AREA_QUAD), "Light correctly mapped to LIGHT_AREA_QUAD");

    // Intensity: 500.0 * 2^2.0 = 2000.0 flux scale
    float expectedFluxScale = 500.0f * std::pow(2.0f, 2.0f);
    assert_near(light.emission.r, 1.0f * expectedFluxScale, 1.0f, "Light emission R matches flux scale");
    assert_near(light.emission.g, 0.95f * expectedFluxScale, 1.0f, "Light emission G matches flux scale");
    assert_near(light.emission.b, 0.90f * expectedFluxScale, 1.0f, "Light emission B matches flux scale");

    // -------------------------------------------------------------------------
    // Test 6: Validate UsdGeomCamera Extraction
    // -------------------------------------------------------------------------
    std::cout << "[Step 6b] Validating UsdGeomCamera extraction..." << std::endl;
    check_true(scene.hasCamera, "Scene must have authored camera extracted");
    assert_near(scene.cameraPosition.x, 0.0f, 0.01f, "Camera pos X");
    // USDA Z-up (0, -10, 2) transforms to Pathways Y-up: (X, Z, -Y) -> (0, 2, 10)
    assert_near(scene.cameraPosition.y, 2.0f, 0.01f, "Camera pos Y");
    assert_near(scene.cameraPosition.z, 10.0f, 0.01f, "Camera pos Z");
    check_true(scene.cameraFov > 10.0f && scene.cameraFov < 120.0f, "Camera FOV within normal range");
    std::cout << "[PASS] Authored camera position (" << scene.cameraPosition.x << ", "
              << scene.cameraPosition.y << ", " << scene.cameraPosition.z << "), FOV="
              << scene.cameraFov << " deg verified." << std::endl;

    // Clean up temporary USDA file
    std::error_code ec;
    std::filesystem::remove(tempUsdaPath, ec);

    // -------------------------------------------------------------------------
    // Test 7: PointInstancedMedCity.usd Validation & Hierarchy Inspection
    // -------------------------------------------------------------------------
    std::filesystem::path medCityPath = "scenes/PointInstancedMedCity/PointInstancedMedCity.usd";
    if (!std::filesystem::exists(medCityPath)) {
        medCityPath = "../scenes/PointInstancedMedCity/PointInstancedMedCity.usd";
    }
    if (std::filesystem::exists(medCityPath)) {
        std::cout << "[Step 7] Inspecting PointInstancedMedCity.usd..." << std::endl;
        pxr::UsdStageRefPtr cityStage = pxr::UsdStage::Open(medCityPath.string());
        check_true(cityStage != nullptr, "PointInstancedMedCity.usd opens successfully");
        size_t meshCount = 0;
        size_t instancerCount = 0;
        std::cout << "  Start TimeCode: " << cityStage->GetStartTimeCode()
                  << ", End TimeCode: " << cityStage->GetEndTimeCode() << std::endl;
        std::cout << "  metersPerUnit: " << pxr::UsdGeomGetStageMetersPerUnit(cityStage)
                  << " (authored: " << (cityStage->HasAuthoredMetadata(pxr::TfToken("metersPerUnit")) ? "yes" : "no") << ")"
                  << ", upAxis: " << pxr::UsdGeomGetStageUpAxis(cityStage).GetString() << std::endl;
        for (const auto& prim : cityStage->Traverse()) {
            if (prim.IsA<pxr::UsdGeomMesh>()) {
                meshCount++;
                pxr::UsdGeomMesh m(prim);
                pxr::VtArray<pxr::GfVec3f> pts;
                m.GetPointsAttr().Get(&pts, cityStage->GetStartTimeCode());
                if (!pts.empty()) {
                    pxr::GfVec3f bMin(1e30), bMax(-1e30);
                    for (const auto& p : pts) {
                        bMin[0] = std::min(bMin[0], p[0]); bMin[1] = std::min(bMin[1], p[1]); bMin[2] = std::min(bMin[2], p[2]);
                        bMax[0] = std::max(bMax[0], p[0]); bMax[1] = std::max(bMax[1], p[1]); bMax[2] = std::max(bMax[2], p[2]);
                    }
                    std::cout << "    Bounds for " << prim.GetPath().GetString() << ": min=("
                              << bMin[0] << ", " << bMin[1] << ", " << bMin[2] << ") max=("
                              << bMax[0] << ", " << bMax[1] << ", " << bMax[2] << ")" << std::endl;
                }
                pxr::VtArray<int> fvc;
                m.GetFaceVertexCountsAttr().Get(&fvc, cityStage->GetStartTimeCode());
                size_t numTris = 0;
                for (int count : fvc) {
                    if (count >= 3) numTris += (count - 2);
                }
                std::cout << "  Mesh " << meshCount << ": " << prim.GetPath().GetString()
                          << " (" << pts.size() << " vertices, " << fvc.size() << " faces, "
                          << numTris << " triangles)" << std::endl;
            }
            if (prim.IsA<pxr::UsdGeomCamera>()) {
                pxr::UsdGeomCamera c(prim);
                pxr::GfMatrix4d cMatDef = pxr::UsdGeomXformCache().GetLocalToWorldTransform(prim);
                pxr::GfMatrix4d cMatTime = pxr::UsdGeomXformCache(cityStage->GetStartTimeCode()).GetLocalToWorldTransform(prim);
                pxr::GfCamera gfCamDef = c.GetCamera(pxr::UsdTimeCode::Default());
                pxr::GfCamera gfCamTime = c.GetCamera(cityStage->GetStartTimeCode());
                std::cout << "  Camera found: " << prim.GetPath().GetString() << std::endl;
                std::cout << "    Pos at Default: (" << cMatDef[3][0] << ", " << cMatDef[3][1] << ", " << cMatDef[3][2] << ")" << std::endl;
                std::cout << "    Pos at StartTime: (" << cMatTime[3][0] << ", " << cMatTime[3][1] << ", " << cMatTime[3][2] << ")" << std::endl;
                std::cout << "    FOV at Default: " << gfCamDef.GetFieldOfView(pxr::GfCamera::FOVVertical)
                          << ", at StartTime: " << gfCamTime.GetFieldOfView(pxr::GfCamera::FOVVertical) << std::endl;
            }
            if (prim.IsA<pxr::UsdGeomPointInstancer>()) {
                instancerCount++;
                pxr::UsdGeomPointInstancer inst(prim);
                pxr::VtArray<pxr::GfVec3f> positions;
                inst.GetPositionsAttr().Get(&positions, pxr::UsdTimeCode::Default());
                std::cout << "  Positions count at Default: " << positions.size() << std::endl;
                if (positions.empty()) {
                    inst.GetPositionsAttr().Get(&positions, cityStage->GetStartTimeCode());
                    std::cout << "  Positions count at StartTimeCode: " << positions.size() << std::endl;
                }
                pxr::SdfPathVector targets;
                inst.GetPrototypesRel().GetTargets(&targets);
                for (size_t t = 0; t < targets.size(); ++t) {
                    pxr::UsdPrim pPrim = cityStage->GetPrimAtPath(targets[t]);
                    pxr::GfMatrix4d pMat = pxr::UsdGeomXformCache().GetLocalToWorldTransform(pPrim);
                    std::cout << "    Proto " << t << " (" << targets[t].GetString() << ") translation: ("
                              << pMat[3][0] << ", " << pMat[3][1] << ", " << pMat[3][2] << ")" << std::endl;
                }
                pxr::GfMatrix4d instWorld = pxr::UsdGeomXformCache().GetLocalToWorldTransform(prim);
                std::cout << "  Instancer localToWorld translation: ("
                          << instWorld[3][0] << ", " << instWorld[3][1] << ", " << instWorld[3][2] << ")" << std::endl;
                std::vector<double> timeSamples;
                inst.GetPositionsAttr().GetTimeSamples(&timeSamples);
                std::cout << "  Positions timeSamples count: " << timeSamples.size() << std::endl;
                for (size_t s = 0; s < std::min<size_t>(timeSamples.size(), 5); ++s) {
                    std::cout << "    Sample time: " << timeSamples[s] << std::endl;
                }
                pxr::VtIntArray protoIndices;
                inst.GetProtoIndicesAttr().Get(&protoIndices, cityStage->GetStartTimeCode());
                std::cout << "  ProtoIndices count at StartTimeCode: " << protoIndices.size() << std::endl;

                pxr::VtArray<pxr::GfMatrix4d> xforms;
                bool ok = inst.ComputeInstanceTransformsAtTime(&xforms, cityStage->GetStartTimeCode(), cityStage->GetStartTimeCode());
                std::cout << "  ComputeInstanceTransformsAtTime success: " << (ok ? "true" : "false")
                          << ", xforms count: " << xforms.size() << std::endl;
                if (!xforms.empty()) {
                    pxr::GfVec3d minBound(1e30), maxBound(-1e30);
                    for (const auto& xf : xforms) {
                        pxr::GfVec3d t(xf[3][0], xf[3][1], xf[3][2]);
                        minBound[0] = std::min(minBound[0], t[0]);
                        minBound[1] = std::min(minBound[1], t[1]);
                        minBound[2] = std::min(minBound[2], t[2]);
                        maxBound[0] = std::max(maxBound[0], t[0]);
                        maxBound[1] = std::max(maxBound[1], t[1]);
                        maxBound[2] = std::max(maxBound[2], t[2]);
                    }
                    std::cout << "  Bounds: min=(" << minBound[0] << ", " << minBound[1] << ", " << minBound[2]
                              << ") max=(" << maxBound[0] << ", " << maxBound[1] << ", " << maxBound[2] << ")" << std::endl;
                    for (size_t k = 0; k < std::min<size_t>(xforms.size(), 3); ++k) {
                        const auto& xf = xforms[k];
                        std::cout << "    xform[" << k << "] proto=" << protoIndices[k] << ":\n";
                        for (int r = 0; r < 4; ++r) {
                            std::cout << "      [" << xf[r][0] << ", " << xf[r][1] << ", " << xf[r][2] << ", " << xf[r][3] << "]\n";
                        }
                    }
                }

                auto tStart = std::chrono::high_resolution_clock::now();
                size_t totalExpectedTris = 0;
                // Pre-count prototype triangles
                std::vector<size_t> protoTriCounts(targets.size(), 0);
                for (size_t t = 0; t < targets.size(); ++t) {
                    pxr::UsdPrim pPrim = cityStage->GetPrimAtPath(targets[t]);
                    for (const auto& child : pxr::UsdPrimRange(pPrim)) {
                        if (child.IsA<pxr::UsdGeomMesh>()) {
                            pxr::UsdGeomMesh m(child);
                            pxr::VtArray<int> fvc;
                            m.GetFaceVertexCountsAttr().Get(&fvc, cityStage->GetStartTimeCode());
                            for (int count : fvc) {
                                if (count >= 3) protoTriCounts[t] += (count - 2);
                            }
                        }
                    }
                }
                for (int idx : protoIndices) {
                    if (idx >= 0 && static_cast<size_t>(idx) < protoTriCounts.size()) {
                        totalExpectedTris += protoTriCounts[idx];
                    }
                }
                auto tCount = std::chrono::high_resolution_clock::now();
                double countMs = std::chrono::duration<double, std::milli>(tCount - tStart).count();
                std::cout << "  Instance Expansion: " << xforms.size() << " instances, "
                          << totalExpectedTris << " total triangles ("
                          << (totalExpectedTris * sizeof(TriangleGPU)) / (1024 * 1024) << " MB buffer) counted in "
                          << countMs << " ms" << std::endl;
            }
        }
        std::cout << "  Total meshes: " << meshCount << ", Total PointInstancers: " << instancerCount << std::endl;

        // -------------------------------------------------------------------------
        // Test 7: Ingest PointInstancedMedCity via UsdLoader::loadSceneData
        // -------------------------------------------------------------------------
        std::cout << "[Step 8] Testing UsdLoader::loadSceneData on PointInstancedMedCity.usd..." << std::endl;
        // Test with 500 instances for fast unit-test verification
        setenv("PATHWAYS_USD_MAX_INSTANCES", "500", 1);
        SceneData cityScene = UsdLoader::loadSceneData(medCityPath.string());
        unsetenv("PATHWAYS_USD_MAX_INSTANCES");

        check_true(!cityScene.triangles.empty(), "PointInstancedMedCity must produce triangles");
        check_true(cityScene.triangles.size() == 27456, "PointInstancedMedCity must load exactly 27456 triangles (17624 non-instanced ground/sea + 9832 prototype triangles across 9 BLASes)");
        check_true(cityScene.materials.size() == 14, "PointInstancedMedCity must extract 14 authored materials (12 prototype materials + 2 terrain/sea materials)");
        check_true(cityScene.hasCamera, "PointInstancedMedCity must have camera configured");
        check_true(!cityScene.lights.empty(), "PointInstancedMedCity must have lighting configured");
        check_true(!cityScene.meshRanges.empty(), "PointInstancedMedCity must define mesh ranges");
        check_true(cityScene.sceneRadius > 5.0f, "PointInstancedMedCity scene radius must be non-trivial (>5m, not 0.05m)");
        std::cout << "  Scene Bounds: min=(" << cityScene.boundsMin.x << ", " << cityScene.boundsMin.y << ", " << cityScene.boundsMin.z
                  << ") max=(" << cityScene.boundsMax.x << ", " << cityScene.boundsMax.y << ", " << cityScene.boundsMax.z
                  << "), radius=" << cityScene.sceneRadius << " m" << std::endl;
        std::cout << "  Camera: pos=(" << cityScene.cameraPosition.x << ", " << cityScene.cameraPosition.y << ", " << cityScene.cameraPosition.z
                  << ") target=(" << cityScene.cameraTarget.x << ", " << cityScene.cameraTarget.y << ", " << cityScene.cameraTarget.z << ")" << std::endl;
        std::cout << "[PASS] PointInstancedMedCity successfully loaded: "
                  << cityScene.triangles.size() << " triangles, "
                  << cityScene.materials.size() << " materials, "
                  << cityScene.lights.size() << " lights" << std::endl;

        // Step 9: Verify UsdLoader::populateMetadata
        std::cout << "[Step 9] Testing UsdLoader::populateMetadata on PointInstancedMedCity.usd..." << std::endl;
        uint64_t metaTris = 0;
        uint32_t metaMats = 0;
        bool metaOk = UsdLoader::populateMetadata(medCityPath.string(), metaTris, metaMats);
        check_true(metaOk, "populateMetadata must succeed");
        check_true(metaTris > 0, "populateMetadata must report >0 triangles (not 0 tris)");
        check_true(metaMats == 14, "populateMetadata must report 14 materials");
        std::cout << "[PASS] populateMetadata returned: " << metaTris << " triangles, " << metaMats << " materials" << std::endl;

        // Step 10: Verify Kitchen_set.usd
        std::filesystem::path kitchenPath = "scenes/Kitchen_set/Kitchen_set.usd";
        if (!std::filesystem::exists(kitchenPath)) {
            kitchenPath = "../scenes/Kitchen_set/Kitchen_set.usd";
        }
        if (std::filesystem::exists(kitchenPath)) {
            std::cout << "[Step 10] Testing UsdLoader on Kitchen_set.usd..." << std::endl;
            uint64_t kTris = 0;
            uint32_t kMats = 0;
            bool kMetaOk = UsdLoader::populateMetadata(kitchenPath.string(), kTris, kMats);
            check_true(kMetaOk, "populateMetadata must succeed for Kitchen_set");
            check_true(kTris > 0, "Kitchen_set metadata must report >0 triangles");
            std::cout << "  Kitchen_set metadata: " << kTris << " triangles, " << kMats << " materials" << std::endl;
            SceneData kScene = UsdLoader::loadSceneData(kitchenPath.string());
            check_true(!kScene.triangles.empty(), "Kitchen_set must produce triangles");
            check_true(kScene.hasCamera, "Kitchen_set must have camera configured");
            std::cout << "  Kitchen_set Camera: pos=(" << kScene.cameraPosition.x << ", "
                      << kScene.cameraPosition.y << ", " << kScene.cameraPosition.z << "), target=("
                      << kScene.cameraTarget.x << ", " << kScene.cameraTarget.y << ", "
                      << kScene.cameraTarget.z << "), fov=" << kScene.cameraFov << " deg" << std::endl;
            std::cout << "[PASS] Kitchen_set successfully loaded: " << kScene.triangles.size()
                      << " triangles, " << kScene.materials.size() << " materials, "
                      << kScene.lights.size() << " lights" << std::endl;
        }
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL OpenUSD LOADER TESTS PASSED (100% SUCCESS)          " << std::endl;
    std::cout << "==========================================================" << std::endl;

    return 0;
}
