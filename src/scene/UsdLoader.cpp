#include "scene/UsdLoader.hpp"
#include "core/Logger.hpp"

#include <filesystem>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <thread>
#include <atomic>
#include <vector>
#include <cstdlib>

#include "stb_image.h"

#define TINYEXR_USE_MINIZ 0
#include <zlib.h>
#include "tinyexr.h"

#if defined(PATHWAYS_ENABLE_USD) && PATHWAYS_ENABLE_USD

#include <pxr/pxr.h>
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/pointInstancer.h>
#include <pxr/usd/usdGeom/camera.h>
#include <pxr/usd/usdGeom/xform.h>
#include <pxr/usd/usdGeom/xformCache.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/subset.h>
#include <pxr/usd/usdShade/material.h>
#include <pxr/usd/usdShade/materialBindingAPI.h>
#include <pxr/usd/usdShade/shader.h>
#include <pxr/usd/usdShade/connectableAPI.h>
#include <pxr/usd/usdLux/rectLight.h>
#include <pxr/usd/usdLux/diskLight.h>
#include <pxr/usd/usdLux/sphereLight.h>
#include <pxr/usd/usdLux/distantLight.h>
#include <pxr/usd/usdLux/domeLight.h>
#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/matrix4d.h>

PXR_NAMESPACE_USING_DIRECTIVE

static inline glm::mat4 gfToGlmMatrix(const pxr::GfMatrix4d& m) {
    const double* src = m.GetArray();
    float dst[16];
    for (int i = 0; i < 16; ++i) {
        dst[i] = static_cast<float>(src[i]);
    }
    return glm::make_mat4(dst);
}

#endif

namespace pathways {

bool UsdLoader::isUsdFile(const std::string& filepath) {
    std::filesystem::path p(filepath);
    std::string ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
    return (ext == ".usd" || ext == ".usda" || ext == ".usdc" || ext == ".usdz");
}

SceneData UsdLoader::loadSceneData(const std::string& filepath) {
#if defined(PATHWAYS_ENABLE_USD) && PATHWAYS_ENABLE_USD
    if (!std::filesystem::exists(filepath)) {
        Logger::error("UsdLoader: File not found: '{}'", filepath);
        return ProceduralScene::createCornellBox();
    }

    Logger::info("UsdLoader: Opening OpenUSD Stage: '{}'", filepath);
    UsdStageRefPtr stage = UsdStage::Open(filepath);
    if (!stage) {
        Logger::error("UsdLoader: Failed to open USD stage: '{}'", filepath);
        return ProceduralScene::createCornellBox();
    }

    SceneData data;

    // 1. Stage Metrics: Coordinate Axis & Units
    TfToken upAxis = UsdGeomGetStageUpAxis(stage);
    glm::mat4 upAxisMatrix(1.0f);
    if (upAxis == UsdGeomTokens->z) {
        upAxisMatrix = glm::rotate(glm::mat4(1.0f), glm::radians(-90.0f), glm::vec3(1.0f, 0.0f, 0.0f));
        Logger::info("UsdLoader: Detected Z-up stage orientation; converting to Pathways Y-up coordinate basis.");
    }

    double metersPerUnit = 1.0;
    if (stage->HasAuthoredMetadata(TfToken("metersPerUnit"))) {
        metersPerUnit = UsdGeomGetStageMetersPerUnit(stage);
    }
    float scaleFactor = static_cast<float>(metersPerUnit);
    glm::mat4 scaleMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(scaleFactor));
    glm::mat4 stageTransform = upAxisMatrix * scaleMatrix;

    // 2. Texture & Material Caches
    std::unordered_map<std::string, uint32_t> loadedTextures;
    std::unordered_map<std::string, uint32_t> packedMrTextures;

    auto resolveTexturePath = [&](const std::string& authoredPath) -> std::filesystem::path {
        if (authoredPath.empty()) return {};
        namespace fs = std::filesystem;
        fs::path authored(authoredPath);
        if (authored.is_absolute() && fs::exists(authored)) {
            return authored;
        }

        fs::path baseDir = fs::path(filepath).parent_path();

        std::string cleanStr = authored.generic_string();
        if (cleanStr.rfind("./", 0) == 0) {
            cleanStr = cleanStr.substr(2);
        }
        fs::path relPath(cleanStr);

        // 1. Relative to baseDir
        fs::path p1 = baseDir / relPath;
        if (fs::exists(p1)) return p1;

        // 2. In textures/, "textures and hdri"/, or usd/textures/ subdirectories
        fs::path p2 = baseDir / "textures" / relPath.filename();
        if (fs::exists(p2)) return p2;
        fs::path p3 = baseDir / "textures and hdri" / relPath.filename();
        if (fs::exists(p3)) return p3;
        fs::path p4 = baseDir / "usd" / "textures" / relPath.filename();
        if (fs::exists(p4)) return p4;

        // 3. Handle Blender export suffixes like foo.jpg.001.jpg or foo.001.png
        std::string stem = relPath.filename().stem().string();
        std::string ext = relPath.filename().extension().string();

        size_t dotPos = stem.rfind('.');
        if (dotPos != std::string::npos) {
            std::string stripped = stem.substr(0, dotPos);
            for (const auto& dir : {baseDir, baseDir / "textures", baseDir / "textures and hdri", baseDir / "usd" / "textures"}) {
                fs::path p = dir / (stripped + ext);
                if (fs::exists(p)) return p;
                fs::path pBase = dir / stripped;
                if (fs::exists(pBase)) return pBase;
            }
        }

        // 4. Try alternate image extensions (.png <-> .jpg <-> .jpeg <-> .exr)
        for (const auto& altExt : {".png", ".jpg", ".jpeg", ".exr"}) {
            for (const auto& dir : {baseDir, baseDir / "textures", baseDir / "textures and hdri", baseDir / "usd" / "textures"}) {
                fs::path p = dir / (stem + altExt);
                if (fs::exists(p)) return p;
                if (dotPos != std::string::npos) {
                    std::string stripped = stem.substr(0, dotPos);
                    fs::path pStrip = dir / (stripped + altExt);
                    if (fs::exists(pStrip)) return pStrip;
                }
            }
        }

        return {};
    };

    auto loadSingleImage = [](const std::filesystem::path& imgPath, TextureData& outTex, bool isSrgb) -> bool {
        outTex.isSrgb = isSrgb;
        std::string ext = imgPath.extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });

        if (ext == ".exr") {
            float* rgba = nullptr;
            int w = 0, h = 0;
            const char* err = nullptr;
            int ret = LoadEXR(&rgba, &w, &h, imgPath.string().c_str(), &err);
            if (ret == TINYEXR_SUCCESS && rgba && w > 0 && h > 0) {
                outTex.width = static_cast<uint32_t>(w);
                outTex.height = static_cast<uint32_t>(h);
                outTex.pixels.resize(w * h * 4);
                for (int i = 0; i < w * h * 4; ++i) {
                    outTex.pixels[i] = static_cast<uint8_t>(std::clamp(rgba[i] * 255.0f, 0.0f, 255.0f));
                }
                free(rgba);
                return true;
            } else {
                if (rgba) free(rgba);
                if (err) {
                    FreeEXRErrorMessage(err);
                }
                return false;
            }
        } else {
            int w = 0, h = 0, comp = 0;
            stbi_uc* rawPixels = stbi_load(imgPath.string().c_str(), &w, &h, &comp, 4);
            if (rawPixels && w > 0 && h > 0) {
                outTex.width = static_cast<uint32_t>(w);
                outTex.height = static_cast<uint32_t>(h);
                outTex.pixels.assign(rawPixels, rawPixels + (w * h * 4));
                stbi_image_free(rawPixels);
                return true;
            }
        }
        return false;
    };

    auto getOrCreateTexture = [&](const std::filesystem::path& imgPath, bool isSrgb) -> uint32_t {
        if (imgPath.empty()) return 0;
        std::string key = imgPath.string() + (isSrgb ? "_srgb" : "_linear");
        auto it = loadedTextures.find(key);
        if (it != loadedTextures.end()) return it->second;

        TextureData texData{};
        if (loadSingleImage(imgPath, texData, isSrgb)) {
            uint32_t idx = static_cast<uint32_t>(data.textures.size()) + 1; // 1-based index
            data.textures.push_back(std::move(texData));
            loadedTextures[key] = idx;
            return idx;
        }
        return 0;
    };

    auto getOrCreateMrTexture = [&](const std::filesystem::path& roughPath,
                                    const std::filesystem::path& metalPath,
                                    float defaultRoughness,
                                    float defaultMetallic) -> uint32_t {
        if (roughPath.empty() && metalPath.empty()) return 0;

        std::string key = roughPath.string() + "|" + metalPath.string();
        auto it = packedMrTextures.find(key);
        if (it != packedMrTextures.end()) return it->second;

        TextureData roughTex{};
        TextureData metalTex{};
        bool hasRough = !roughPath.empty() && loadSingleImage(roughPath, roughTex, false);
        bool hasMetal = !metalPath.empty() && loadSingleImage(metalPath, metalTex, false);

        if (!hasRough && !hasMetal) return 0;

        uint32_t w = hasRough ? roughTex.width : metalTex.width;
        uint32_t h = hasRough ? roughTex.height : metalTex.height;

        TextureData packedTex{};
        packedTex.width = w;
        packedTex.height = h;
        packedTex.isSrgb = false;
        packedTex.pixels.resize(w * h * 4);

        uint8_t defR = static_cast<uint8_t>(std::clamp(defaultRoughness * 255.0f, 0.0f, 255.0f));
        uint8_t defM = static_cast<uint8_t>(std::clamp(defaultMetallic * 255.0f, 0.0f, 255.0f));

        for (uint32_t y = 0; y < h; ++y) {
            for (uint32_t x = 0; x < w; ++x) {
                uint32_t dstIdx = (y * w + x) * 4;

                uint8_t rVal = defR;
                if (hasRough) {
                    uint32_t rx = (x * roughTex.width) / w;
                    uint32_t ry = (y * roughTex.height) / h;
                    uint32_t srcIdx = (ry * roughTex.width + rx) * 4;
                    rVal = roughTex.pixels[srcIdx];
                }

                uint8_t mVal = defM;
                if (hasMetal) {
                    uint32_t mx = (x * metalTex.width) / w;
                    uint32_t my = (y * metalTex.height) / h;
                    uint32_t srcIdx = (my * metalTex.width + mx) * 4;
                    mVal = metalTex.pixels[srcIdx];
                }

                packedTex.pixels[dstIdx + 0] = 255;  // R: Occlusion
                packedTex.pixels[dstIdx + 1] = rVal; // G: Roughness
                packedTex.pixels[dstIdx + 2] = mVal; // B: Metallic
                packedTex.pixels[dstIdx + 3] = 255;  // A: 1.0
            }
        }

        uint32_t idx = static_cast<uint32_t>(data.textures.size()) + 1;
        data.textures.push_back(std::move(packedTex));
        packedMrTextures[key] = idx;
        return idx;
    };

    auto getTextureFileFromInput = [](const UsdShadeInput& input) -> std::string {
        if (!input) return "";
        UsdShadeSourceInfoVector sources = input.GetConnectedSources();
        if (sources.empty()) return "";
        UsdPrim srcPrim = sources[0].source.GetPrim();
        if (!srcPrim || !srcPrim.IsA<UsdShadeShader>()) return "";
        UsdShadeShader texShader(srcPrim);
        UsdShadeInput fileIn = texShader.GetInput(TfToken("file"));
        if (!fileIn) return "";

        SdfAssetPath assetPath;
        if (fileIn.Get(&assetPath)) {
            std::string p = assetPath.GetResolvedPath();
            if (p.empty()) p = assetPath.GetAssetPath();
            return p;
        }
        std::string strPath;
        if (fileIn.Get(&strPath)) {
            return strPath;
        }
        return "";
    };

    struct TextureTransform2D {
        glm::vec2 scale{1.0f, 1.0f};
        glm::vec2 translation{0.0f, 0.0f};
        float rotation{0.0f};
    };
    std::vector<TextureTransform2D> materialTransforms;

    auto getTextureTransformFromInput = [](const UsdShadeInput& input) -> TextureTransform2D {
        TextureTransform2D xform{};
        if (!input) return xform;
        UsdShadeSourceInfoVector sources = input.GetConnectedSources();
        if (sources.empty()) return xform;
        UsdPrim texPrim = sources[0].source.GetPrim();
        if (!texPrim || !texPrim.IsA<UsdShadeShader>()) return xform;
        UsdShadeShader texShader(texPrim);

        UsdShadeInput stIn = texShader.GetInput(TfToken("st"));
        if (!stIn) stIn = texShader.GetInput(TfToken("uv"));
        if (!stIn) return xform;

        UsdShadeSourceInfoVector stSources = stIn.GetConnectedSources();
        if (stSources.empty()) return xform;
        UsdPrim xformPrim = stSources[0].source.GetPrim();
        if (!xformPrim || !xformPrim.IsA<UsdShadeShader>()) return xform;

        UsdShadeShader xformShader(xformPrim);
        TfToken shaderId;
        if (xformShader.GetIdAttr().Get(&shaderId) && shaderId == TfToken("UsdTransform2d")) {
            UsdShadeInput scaleIn = xformShader.GetInput(TfToken("scale"));
            if (scaleIn) {
                GfVec2f s(1.0f, 1.0f);
                if (scaleIn.Get(&s)) {
                    xform.scale = glm::vec2(s[0], s[1]);
                }
            }
            UsdShadeInput transIn = xformShader.GetInput(TfToken("translation"));
            if (transIn) {
                GfVec2f t(0.0f, 0.0f);
                if (transIn.Get(&t)) {
                    xform.translation = glm::vec2(t[0], t[1]);
                }
            }
            UsdShadeInput rotIn = xformShader.GetInput(TfToken("rotation"));
            if (rotIn) {
                float r = 0.0f;
                if (rotIn.Get(&r)) {
                    xform.rotation = r;
                }
            }
        }
        return xform;
    };

    std::unordered_map<std::string, uint32_t> materialCache;
    auto getOrCreateMaterial = [&](const UsdShadeMaterial& usdMat) -> uint32_t {
        if (!usdMat) return 0;
        std::string matPath = usdMat.GetPath().GetString();
        auto it = materialCache.find(matPath);
        if (it != materialCache.end()) {
            return it->second;
        }

        MaterialGPU gpuMat{};
        gpuMat.albedo = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
        gpuMat.roughness = 0.5f;
        gpuMat.metallic = 0.0f;
        gpuMat.ior = 1.5f;
        gpuMat.type = MATERIAL_DIFFUSE;

        std::string matName = usdMat.GetPath().GetName();
        std::string primName = usdMat.GetPrim().GetName().GetString();
        std::string lowerMatName = matName;
        std::transform(lowerMatName.begin(), lowerMatName.end(), lowerMatName.begin(), [](unsigned char c) { return std::tolower(c); });

        TextureTransform2D matXform{};

        UsdShadeShader surfaceShader = usdMat.ComputeSurfaceSource();
        if (surfaceShader) {
            // Check UsdPreviewSurface attributes
            UsdShadeInput diffuseInput = surfaceShader.GetInput(TfToken("diffuseColor"));
            if (!diffuseInput) diffuseInput = surfaceShader.GetInput(TfToken("baseColor"));
            if (diffuseInput) {
                GfVec3f color(0.8f, 0.8f, 0.8f);
                bool hasAuthoredColor = diffuseInput.Get(&color);
                gpuMat.albedo = glm::vec4(color[0], color[1], color[2], 1.0f);

                matXform = getTextureTransformFromInput(diffuseInput);

                std::string texFile = getTextureFileFromInput(diffuseInput);
                std::string lowerTex = texFile;
                std::transform(lowerTex.begin(), lowerTex.end(), lowerTex.begin(), [](unsigned char c) { return std::tolower(c); });

                if (lowerTex.find("anisotropy") != std::string::npos || lowerMatName.find("fiber") != std::string::npos || lowerMatName.find("carbon") != std::string::npos) {
                    // Anisotropy level masks are mistakenly connected as diffuseColor in some Blender USD exports.
                    // Carbon fiber roof has a dark charcoal base albedo.
                    gpuMat.albedo = glm::vec4(0.04f, 0.04f, 0.04f, 1.0f);
                    gpuMat.roughness = 0.35f;
                } else if (!texFile.empty()) {
                    auto resolved = resolveTexturePath(texFile);
                    if (!resolved.empty()) {
                        gpuMat.albedoTex = getOrCreateTexture(resolved, true);
                        if (!hasAuthoredColor) {
                            gpuMat.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
                        }
                    }
                }
            }

            UsdShadeInput metallicInput = surfaceShader.GetInput(TfToken("metallic"));
            if (metallicInput) {
                float metallic = 0.0f;
                metallicInput.Get(&metallic);
                gpuMat.metallic = metallic;
                if (gpuMat.metallic > 0.5f) {
                    gpuMat.type = MATERIAL_METALLIC;
                }
            }

            UsdShadeInput roughnessInput = surfaceShader.GetInput(TfToken("roughness"));
            if (roughnessInput) {
                float roughness = 0.5f;
                roughnessInput.Get(&roughness);
                gpuMat.roughness = roughness;
            }

            // Metallic / Roughness textures
            std::string roughFile = roughnessInput ? getTextureFileFromInput(roughnessInput) : "";
            std::string metalFile = metallicInput ? getTextureFileFromInput(metallicInput) : "";
            if (!roughFile.empty() || !metalFile.empty()) {
                auto resolvedRough = resolveTexturePath(roughFile);
                auto resolvedMetal = resolveTexturePath(metalFile);
                gpuMat.mrTex = getOrCreateMrTexture(resolvedRough, resolvedMetal, gpuMat.roughness, gpuMat.metallic);
            }

            UsdShadeInput normalInput = surfaceShader.GetInput(TfToken("normal"));
            if (normalInput) {
                if (matXform.scale == glm::vec2(1.0f, 1.0f)) {
                    matXform = getTextureTransformFromInput(normalInput);
                }
                std::string normFile = getTextureFileFromInput(normalInput);
                if (!normFile.empty()) {
                    auto resolved = resolveTexturePath(normFile);
                    if (!resolved.empty()) {
                        gpuMat.normalTex = getOrCreateTexture(resolved, false);
                    }
                }
            }

            UsdShadeInput iorInput = surfaceShader.GetInput(TfToken("ior"));
            if (iorInput) {
                float ior = 1.5f;
                iorInput.Get(&ior);
                gpuMat.ior = ior;
            }

            UsdShadeInput opacityInput = surfaceShader.GetInput(TfToken("opacity"));
            if (opacityInput) {
                float opacity = 1.0f;
                opacityInput.Get(&opacity);
                if (opacity < 0.95f) {
                    gpuMat.transmission = 1.0f - opacity;
                    gpuMat.type = MATERIAL_DIELECTRIC;
                }
            }

            // Glass material heuristic (name-based, tag-based, or low roughness dielectric)
            bool isGlass = (lowerMatName.find("glass") != std::string::npos ||
                            primName.find("glass") != std::string::npos ||
                            lowerMatName.find("window") != std::string::npos ||
                            lowerMatName.find("windshield") != std::string::npos ||
                            (lowerMatName.rfind("g4", 0) == 0) ||
                            (lowerMatName.size() == 2 && lowerMatName[0] == 'g' && std::isdigit(lowerMatName[1])) ||
                            (gpuMat.roughness <= 0.05f && gpuMat.metallic == 0.0f && gpuMat.ior >= 1.4f && (lowerMatName[0] == 'g' || lowerMatName[0] == 'w')));

            if (isGlass) {
                gpuMat.type = MATERIAL_DIELECTRIC;
                gpuMat.transmission = 1.0f;
                gpuMat.ior = 1.5f;
                gpuMat.roughness = 0.0f;
                gpuMat.thickness = 0.0f; // Explicitly thin-walled transmission
                // Preserve color tint for colored glass (e.g. taillights, turn signals), default clear to white
                if (std::abs(gpuMat.albedo.r - 0.8f) < 0.05f && std::abs(gpuMat.albedo.g - 0.8f) < 0.05f && std::abs(gpuMat.albedo.b - 0.8f) < 0.05f) {
                    gpuMat.albedo = glm::vec4(1.0f, 1.0f, 1.0f, 1.0f);
                }
            }

            UsdShadeInput clearcoatInput = surfaceShader.GetInput(TfToken("clearcoat"));
            if (clearcoatInput) {
                float cc = 0.0f;
                clearcoatInput.Get(&cc);
                gpuMat.clearcoat = std::clamp(cc, 0.0f, 1.0f);
            }

            UsdShadeInput clearcoatRoughnessInput = surfaceShader.GetInput(TfToken("clearcoatRoughness"));
            if (clearcoatRoughnessInput) {
                float ccr = 0.0f;
                clearcoatRoughnessInput.Get(&ccr);
                gpuMat.clearcoatRoughness = std::clamp(ccr, 0.04f, 1.0f);
            }

            UsdShadeInput emissiveInput = surfaceShader.GetInput(TfToken("emissiveColor"));
            if (emissiveInput) {
                GfVec3f emission(0.0f, 0.0f, 0.0f);
                emissiveInput.Get(&emission);
                gpuMat.emissive = glm::vec4(emission[0], emission[1], emission[2], 1.0f);
                if (glm::length(glm::vec3(gpuMat.emissive)) > 1e-3f) {
                    gpuMat.type = MATERIAL_EMISSIVE;
                }
                std::string emissiveFile = getTextureFileFromInput(emissiveInput);
                if (!emissiveFile.empty()) {
                    auto resolved = resolveTexturePath(emissiveFile);
                    if (!resolved.empty()) {
                        gpuMat.emissiveTex = getOrCreateTexture(resolved, true);
                    }
                }
            }
        }

        // Foliage and bark heuristics for untextured botanical materials (e.g. Botaniq assets without exported shader networks)
        if (gpuMat.albedoTex == 0 && gpuMat.type == MATERIAL_DIFFUSE) {
            if (lowerMatName.find("leaf") != std::string::npos || lowerMatName.find("leaves") != std::string::npos || lowerMatName.find("foliage") != std::string::npos) {
                if (lowerMatName.find("autumn") != std::string::npos || lowerMatName.find("orange") != std::string::npos || lowerMatName.find("yellow") != std::string::npos) {
                    gpuMat.albedo = glm::vec4(0.82f, 0.48f, 0.12f, 1.0f); // Warm autumn foliage
                } else {
                    gpuMat.albedo = glm::vec4(0.12f, 0.28f, 0.08f, 1.0f); // Lush summer green foliage
                }
                gpuMat.roughness = 0.65f;
            } else if (lowerMatName.find("bark") != std::string::npos || lowerMatName.find("trunk") != std::string::npos) {
                if (lowerMatName.find("populus") != std::string::npos || lowerMatName.find("birch") != std::string::npos) {
                    gpuMat.albedo = glm::vec4(0.48f, 0.45f, 0.40f, 1.0f); // Light birch/aspen bark
                } else {
                    gpuMat.albedo = glm::vec4(0.16f, 0.12f, 0.08f, 1.0f); // Dark oak bark
                }
                gpuMat.roughness = 0.85f;
            }
        }

        uint32_t newId = static_cast<uint32_t>(data.materials.size());
        data.materials.push_back(gpuMat);
        materialTransforms.push_back(matXform);
        materialCache[matPath] = newId;
        return newId;
    };

    std::unordered_map<uint32_t, uint32_t> colorMaterialCache;
    auto getOrCreateColorMaterial = [&](const GfVec3f& c) -> uint32_t {
        uint32_t r = static_cast<uint32_t>(std::clamp(c[0], 0.0f, 1.0f) * 255.0f);
        uint32_t g = static_cast<uint32_t>(std::clamp(c[1], 0.0f, 1.0f) * 255.0f);
        uint32_t b = static_cast<uint32_t>(std::clamp(c[2], 0.0f, 1.0f) * 255.0f);
        uint32_t key = (r << 16) | (g << 8) | b;
        auto it = colorMaterialCache.find(key);
        if (it != colorMaterialCache.end()) return it->second;

        MaterialGPU gpuMat{};
        gpuMat.albedo = glm::vec4(c[0], c[1], c[2], 1.0f);
        gpuMat.roughness = 0.5f;
        gpuMat.metallic = 0.0f;
        gpuMat.ior = 1.5f;
        gpuMat.type = MATERIAL_DIFFUSE;
        uint32_t newId = static_cast<uint32_t>(data.materials.size());
        data.materials.push_back(gpuMat);
        materialTransforms.push_back(TextureTransform2D{});
        colorMaterialCache[key] = newId;
        return newId;
    };

    if (data.materials.empty()) {
        MaterialGPU defaultMat{};
        defaultMat.albedo = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
        defaultMat.roughness = 0.5f;
        defaultMat.metallic = 0.0f;
        defaultMat.type = MATERIAL_DIFFUSE;
        data.materials.push_back(defaultMat);
        materialTransforms.push_back(TextureTransform2D{});
    }

    // Determine initial evaluation time code
    UsdTimeCode evalTime = UsdTimeCode::Default();
    if (stage->HasAuthoredTimeCodeRange()) {
        evalTime = UsdTimeCode(stage->GetStartTimeCode());
    }

    // 3. Scan for PointInstancers and their Prototype targets
    UsdGeomXformCache xformCache(evalTime);
    std::vector<UsdGeomPointInstancer> pointInstancers;
    std::vector<SdfPath> allPrototypeTargets;

    for (const UsdPrim& prim : stage->Traverse(UsdTraverseInstanceProxies())) {
        if (prim.IsA<UsdGeomPointInstancer>()) {
            UsdGeomPointInstancer inst(prim);
            pointInstancers.push_back(inst);
            SdfPathVector targets;
            inst.GetPrototypesRel().GetTargets(&targets);
            for (const auto& tgt : targets) {
                allPrototypeTargets.push_back(tgt);
            }
        }
    }

    // 4. Traverse Stage Prims: Regular Geometry & Lights
    bool hasDomeLight = false;
    for (const UsdPrim& prim : stage->Traverse(UsdTraverseInstanceProxies())) {
        // --- Process Meshes ---
        if (prim.IsA<UsdGeomMesh>()) {
            // If this mesh is part of any PointInstancer prototype hierarchy, skip it here
            bool isPrototypePrim = false;
            for (const auto& tgt : allPrototypeTargets) {
                if (prim.GetPath().HasPrefix(tgt)) {
                    isPrototypePrim = true;
                    break;
                }
            }
            if (isPrototypePrim) {
                continue;
            }

            UsdGeomMesh mesh(prim);
            GfMatrix4d usdWorldMat = xformCache.GetLocalToWorldTransform(prim);
            glm::mat4 primWorldMat = gfToGlmMatrix(usdWorldMat);
            glm::mat4 M = stageTransform * primWorldMat;
            glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(M)));

            // Extract geometry points
            VtArray<GfVec3f> points;
            mesh.GetPointsAttr().Get(&points, evalTime);
            if (points.empty() && evalTime != UsdTimeCode::Default()) {
                mesh.GetPointsAttr().Get(&points, UsdTimeCode::Default());
            }
            if (points.empty()) continue;

            VtArray<int> faceVertexCounts;
            mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts, evalTime);
            if (faceVertexCounts.empty() && evalTime != UsdTimeCode::Default()) {
                mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts, UsdTimeCode::Default());
            }
            VtArray<int> faceVertexIndices;
            mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices, evalTime);
            if (faceVertexIndices.empty() && evalTime != UsdTimeCode::Default()) {
                mesh.GetFaceVertexIndicesAttr().Get(&faceVertexIndices, UsdTimeCode::Default());
            }
            if (faceVertexCounts.empty() || faceVertexIndices.empty()) continue;

            // Extract normals
            VtArray<GfVec3f> normals;
            mesh.GetNormalsAttr().Get(&normals, evalTime);
            if (normals.empty() && evalTime != UsdTimeCode::Default()) {
                mesh.GetNormalsAttr().Get(&normals, UsdTimeCode::Default());
            }
            TfToken normInterp = mesh.GetNormalsInterpolation();
            TfToken orientation = UsdGeomTokens->rightHanded;
            mesh.GetOrientationAttr().Get(&orientation, evalTime);
            if (orientation.IsEmpty() && evalTime != UsdTimeCode::Default()) {
                mesh.GetOrientationAttr().Get(&orientation, UsdTimeCode::Default());
            }
            bool isLeftHanded = (orientation == UsdGeomTokens->leftHanded);

            // Extract UV coordinates
            VtArray<GfVec2f> uvs;
            TfToken uvInterp = UsdGeomTokens->constant;
            UsdGeomPrimvarsAPI primvarsAPI(mesh);
            UsdGeomPrimvar stPrimvar = primvarsAPI.GetPrimvar(TfToken("st"));
            if (!stPrimvar) stPrimvar = primvarsAPI.GetPrimvar(TfToken("uv"));
            if (!stPrimvar) stPrimvar = primvarsAPI.GetPrimvar(TfToken("UVMap"));
            if (stPrimvar) {
                stPrimvar.ComputeFlattened(&uvs, evalTime);
                if (uvs.empty() && evalTime != UsdTimeCode::Default()) {
                    stPrimvar.ComputeFlattened(&uvs, UsdTimeCode::Default());
                }
                uvInterp = stPrimvar.GetInterpolation();
            }

            // Extract Material Binding or displayColor
            uint32_t defaultMatId = 0;
            UsdShadeMaterialBindingAPI bindingAPI(prim);
            UsdShadeMaterial boundMat = bindingAPI.ComputeBoundMaterial();
            if (boundMat) {
                defaultMatId = getOrCreateMaterial(boundMat);
            } else {
                // If mesh has no bound material, check for a matching Material_001 on the stage
                UsdPrim mat001 = stage->GetPrimAtPath(SdfPath("/root/_materials/Material_001"));
                if (mat001 && mat001.IsA<UsdShadeMaterial>()) {
                    boundMat = UsdShadeMaterial(mat001);
                    defaultMatId = getOrCreateMaterial(boundMat);
                }
            }

            UsdGeomPrimvar dispColorPrim = primvarsAPI.GetPrimvar(TfToken("displayColor"));
            VtArray<GfVec3f> dispColors;
            TfToken dispColorInterp = UsdGeomTokens->constant;
            if (dispColorPrim) {
                dispColorPrim.Get(&dispColors, evalTime);
                if (dispColors.empty() && evalTime != UsdTimeCode::Default()) {
                    dispColorPrim.Get(&dispColors, UsdTimeCode::Default());
                }
                dispColorInterp = dispColorPrim.GetInterpolation();
            }
            if (!boundMat && !dispColors.empty()) {
                if (dispColorInterp == UsdGeomTokens->constant || dispColors.size() == 1) {
                    defaultMatId = getOrCreateColorMaterial(dispColors[0]);
                }
            }

            // Check for per-face GeomSubsets
            std::vector<uint32_t> faceMaterials(faceVertexCounts.size(), defaultMatId);
            std::vector<UsdGeomSubset> subsets = UsdGeomSubset::GetAllGeomSubsets(mesh);
            for (const auto& subset : subsets) {
                UsdShadeMaterialBindingAPI subBinding(subset.GetPrim());
                UsdShadeMaterial subMat = subBinding.ComputeBoundMaterial();
                if (subMat) {
                    uint32_t subMatId = getOrCreateMaterial(subMat);
                    VtIntArray subIndices;
                    subset.GetIndicesAttr().Get(&subIndices, evalTime);
                    if (subIndices.empty() && evalTime != UsdTimeCode::Default()) {
                        subset.GetIndicesAttr().Get(&subIndices, UsdTimeCode::Default());
                    }
                    for (int fIdx : subIndices) {
                        if (fIdx >= 0 && static_cast<size_t>(fIdx) < faceMaterials.size()) {
                            faceMaterials[fIdx] = subMatId;
                        }
                    }
                }
            }

            // Polygon Triangulation & Vertex Assembly
            size_t indexOffset = 0;
            for (size_t f = 0; f < faceVertexCounts.size(); ++f) {
                int count = faceVertexCounts[f];
                if (count < 3) {
                    indexOffset += count;
                    continue;
                }

                uint32_t faceMatId = faceMaterials[f];
                if (!boundMat && subsets.empty() && !dispColors.empty() && dispColorInterp == UsdGeomTokens->uniform && f < dispColors.size()) {
                    faceMatId = getOrCreateColorMaterial(dispColors[f]);
                }

                auto getVertex = [&](int localIdx) -> Vertex {
                    int globalIdx = faceVertexIndices[indexOffset + localIdx];
                    Vertex v{};

                    // Position
                    GfVec3f p = points[globalIdx];
                    v.position = glm::vec4(p[0], p[1], p[2], 0.0f);

                    // Normal
                    glm::vec3 n(0.0f, 0.0f, 0.0f);
                    if (!normals.empty()) {
                        if (normInterp == UsdGeomTokens->uniform && f < normals.size()) {
                            GfVec3f usdNorm = normals[f];
                            n = glm::vec3(usdNorm[0], usdNorm[1], usdNorm[2]);
                        } else if ((normInterp == UsdGeomTokens->vertex || normInterp == UsdGeomTokens->varying) && globalIdx < static_cast<int>(normals.size())) {
                            GfVec3f usdNorm = normals[globalIdx];
                            n = glm::vec3(usdNorm[0], usdNorm[1], usdNorm[2]);
                        } else if (normInterp == UsdGeomTokens->faceVarying && (indexOffset + localIdx) < normals.size()) {
                            GfVec3f usdNorm = normals[indexOffset + localIdx];
                            n = glm::vec3(usdNorm[0], usdNorm[1], usdNorm[2]);
                        }
                    }
                    v.normal = glm::vec4(n, 0.0f);

                    // UV
                    glm::vec2 uv(0.0f);
                    if (!uvs.empty()) {
                        if ((uvInterp == UsdGeomTokens->vertex || uvInterp == UsdGeomTokens->varying) && globalIdx < static_cast<int>(uvs.size())) {
                            uv = glm::vec2(uvs[globalIdx][0], uvs[globalIdx][1]);
                        } else if (uvInterp == UsdGeomTokens->faceVarying && (indexOffset + localIdx) < uvs.size()) {
                            uv = glm::vec2(uvs[indexOffset + localIdx][0], uvs[indexOffset + localIdx][1]);
                        }
                    }
                    if (faceMatId < materialTransforms.size()) {
                        const auto& xf = materialTransforms[faceMatId];
                        if (xf.rotation != 0.0f) {
                            float rad = glm::radians(xf.rotation);
                            float cosR = std::cos(rad);
                            float sinR = std::sin(rad);
                            uv = glm::vec2(uv.x * cosR - uv.y * sinR, uv.x * sinR + uv.y * cosR);
                        }
                        uv = uv * xf.scale + xf.translation;
                    }
                    v.position.w = uv.x;
                    v.normal.w = 1.0f - uv.y;
                    v.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);

                    return v;
                };

                // Fan triangulation: (0, i, i + 1) for CCW rightHanded, (0, i + 1, i) for CW leftHanded
                for (int i = 1; i < count - 1; ++i) {
                    int idx1 = isLeftHanded ? (i + 1) : i;
                    int idx2 = isLeftHanded ? i : (i + 1);
                    Vertex v0 = getVertex(0);
                    Vertex v1 = getVertex(idx1);
                    Vertex v2 = getVertex(idx2);

                    TriangleGPU tri{};
                    glm::vec4 p0 = M * glm::vec4(glm::vec3(v0.position), 1.0f);
                    glm::vec4 p1 = M * glm::vec4(glm::vec3(v1.position), 1.0f);
                    glm::vec4 p2 = M * glm::vec4(glm::vec3(v2.position), 1.0f);

                    tri.v0.position = glm::vec4(glm::vec3(p0), v0.position.w);
                    tri.v1.position = glm::vec4(glm::vec3(p1), v1.position.w);
                    tri.v2.position = glm::vec4(glm::vec3(p2), v2.position.w);

                    glm::vec3 p0_3 = glm::vec3(p0);
                    glm::vec3 p1_3 = glm::vec3(p1);
                    glm::vec3 p2_3 = glm::vec3(p2);
                    glm::vec3 geoNormal = glm::cross(p1_3 - p0_3, p2_3 - p0_3);
                    if (glm::length(geoNormal) > 1e-7f) {
                        geoNormal = glm::normalize(geoNormal);
                    } else {
                        geoNormal = glm::vec3(0.0f, 1.0f, 0.0f);
                    }

                    glm::vec3 n0 = glm::vec3(v0.normal);
                    glm::vec3 n1 = glm::vec3(v1.normal);
                    glm::vec3 n2 = glm::vec3(v2.normal);
                    tri.v0.normal = glm::vec4(glm::length(n0) < 1e-4f ? geoNormal : glm::normalize(normalMatrix * n0), v0.normal.w);
                    tri.v1.normal = glm::vec4(glm::length(n1) < 1e-4f ? geoNormal : glm::normalize(normalMatrix * n1), v1.normal.w);
                    tri.v2.normal = glm::vec4(glm::length(n2) < 1e-4f ? geoNormal : glm::normalize(normalMatrix * n2), v2.normal.w);

                    // Compute tangents
                    glm::vec3 e1 = p1_3 - p0_3;
                    glm::vec3 e2 = p2_3 - p0_3;
                    glm::vec2 uv0(v0.position.w, v0.normal.w);
                    glm::vec2 uv1(v1.position.w, v1.normal.w);
                    glm::vec2 uv2(v2.position.w, v2.normal.w);
                    glm::vec2 duv1 = uv1 - uv0;
                    glm::vec2 duv2 = uv2 - uv0;
                    float det = duv1.x * duv2.y - duv2.x * duv1.y;
                    glm::vec3 tanDir;
                    if (std::abs(det) > 1e-6f) {
                        tanDir = glm::normalize((e1 * duv2.y - e2 * duv1.y) / det);
                    } else {
                        glm::vec3 up = std::abs(geoNormal.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
                        tanDir = glm::normalize(glm::cross(up, geoNormal));
                    }
                    tri.v0.tangent = glm::vec4(tanDir, 1.0f);
                    tri.v1.tangent = glm::vec4(tanDir, 1.0f);
                    tri.v2.tangent = glm::vec4(tanDir, 1.0f);

                    tri.materialId = faceMatId;
                    data.triangles.push_back(tri);
                }

                indexOffset += count;
            }
        }
        // --- Process Lights (UsdLux) ---
        else if (prim.IsA<UsdLuxDomeLight>()) {
            hasDomeLight = true;
            Logger::info("UsdLoader: Detected UsdLuxDomeLight at '{}'", prim.GetPath().GetString());
        }
        else if (prim.IsA<UsdLuxRectLight>() || prim.IsA<UsdLuxDiskLight>() ||
                 prim.IsA<UsdLuxSphereLight>() || prim.IsA<UsdLuxDistantLight>()) {
            GfMatrix4d usdWorldMat = xformCache.GetLocalToWorldTransform(prim);
            glm::mat4 M = stageTransform * gfToGlmMatrix(usdWorldMat);
            glm::vec3 pos = glm::vec3(M[3]);

            LightGPU gpuLight{};
            GfVec3f color(1.0f, 1.0f, 1.0f);
            float intensity = 1.0f;
            float exposure = 0.0f;

            float fluxScale = intensity * std::pow(2.0f, exposure);

            if (prim.IsA<UsdLuxRectLight>()) {
                UsdLuxRectLight rl(prim);
                rl.GetColorAttr().Get(&color);
                rl.GetIntensityAttr().Get(&intensity);
                rl.GetExposureAttr().Get(&exposure);
                float width = 1.0f, height = 1.0f;
                rl.GetWidthAttr().Get(&width);
                rl.GetHeightAttr().Get(&height);

                glm::vec3 uVec = glm::vec3(M[0]) * width;
                glm::vec3 vVec = glm::vec3(M[1]) * height;
                glm::vec3 corner = pos - 0.5f * (uVec + vVec);
                float area = glm::length(glm::cross(uVec, vVec));

                gpuLight.position = glm::vec4(corner, LIGHT_AREA_QUAD);
                gpuLight.u = glm::vec4(uVec, 0.0f);
                gpuLight.v = glm::vec4(vVec, 0.0f);
                gpuLight.normal = glm::vec4(-glm::normalize(glm::vec3(M[2])), 0.0f);
                fluxScale = intensity * std::pow(2.0f, exposure);
                gpuLight.emission = glm::vec4(color[0] * fluxScale, color[1] * fluxScale, color[2] * fluxScale, area);
            } else if (prim.IsA<UsdLuxDistantLight>()) {
                UsdLuxDistantLight dl(prim);
                dl.GetColorAttr().Get(&color);
                dl.GetIntensityAttr().Get(&intensity);
                dl.GetExposureAttr().Get(&exposure);

                glm::vec3 dirToLight = glm::normalize(glm::vec3(M[2]));
                gpuLight.position = glm::vec4(pos, LIGHT_DIRECTIONAL);
                gpuLight.normal = glm::vec4(dirToLight, 0.0f);
                fluxScale = intensity * std::pow(2.0f, exposure);
                gpuLight.emission = glm::vec4(color[0] * fluxScale, color[1] * fluxScale, color[2] * fluxScale, 1.0f);
            } else {
                gpuLight.position = glm::vec4(pos, LIGHT_AREA_QUAD);
                fluxScale = intensity * std::pow(2.0f, exposure);
                gpuLight.emission = glm::vec4(color[0] * fluxScale, color[1] * fluxScale, color[2] * fluxScale, 1.0f);
            }

            data.lights.push_back(gpuLight);
        }
        // --- Process Cameras (UsdGeomCamera) ---
        else if (prim.IsA<UsdGeomCamera>()) {
            if (!data.hasCamera) {
                UsdGeomCamera usdCam(prim);
                GfMatrix4d usdWorldMat = xformCache.GetLocalToWorldTransform(prim);
                glm::mat4 M = stageTransform * gfToGlmMatrix(usdWorldMat);

                glm::vec3 pos = glm::vec3(M[3]);
                // In USD convention, camera looks along -Z with +Y up in local space
                glm::vec3 forward = -glm::normalize(glm::vec3(M[2]));
                glm::vec3 up = glm::normalize(glm::vec3(M[1]));

                GfCamera gfCam = usdCam.GetCamera(evalTime);
                float fov = gfCam.GetFieldOfView(GfCamera::FOVVertical);
                if (fov <= 0.0f || std::isnan(fov)) {
                    fov = 45.0f;
                }

                float focusDistance = gfCam.GetFocusDistance();
                float focalDist = (focusDistance > 0.01f) ? (focusDistance * scaleFactor) : 10.0f;

                data.hasCamera = true;
                data.cameraPosition = pos;
                data.cameraTarget = pos + forward * focalDist;
                data.cameraUp = up;
                data.cameraFov = fov;
                data.focalDistance = focalDist;
                Logger::info("UsdLoader: Authored camera '{}' loaded at pos ({:.2f}, {:.2f}, {:.2f}), forward ({:.2f}, {:.2f}, {:.2f}), fov {:.1f} deg",
                             prim.GetPath().GetString(), pos.x, pos.y, pos.z, forward.x, forward.y, forward.z, fov);
            }
        }
    }

    // 5. Ingest UsdGeomPointInstancer Prims
    struct LocalProtoTriangle {
        Vertex v0;
        Vertex v1;
        Vertex v2;
        uint32_t materialId = 0;
    };

    std::unordered_map<std::string, uint32_t> protoPathToBlas;

    if (!pointInstancers.empty() && !data.triangles.empty() && data.blasRanges.empty()) {
        BlasGeometryRange regRange{};
        regRange.firstTriangle = 0;
        regRange.triangleCount = static_cast<uint32_t>(data.triangles.size());
        regRange.numOpaqueTriangles = regRange.triangleCount;
        data.blasRanges.push_back(regRange);

        SceneInstance regInst{};
        regInst.blasIndex = 0;
        regInst.transform = glm::mat4(1.0f);
        regInst.customIndex = static_cast<uint32_t>(data.instanceData.size());
        data.instances.push_back(regInst);

        InstanceGPU regInstGPU{};
        regInstGPU.firstTriangle = 0;
        regInstGPU.numOpaqueTriangles = regRange.numOpaqueTriangles;
        regInstGPU.materialOffset = 0;
        regInstGPU.flags = 0;
        data.instanceData.push_back(regInstGPU);
    }

    for (const auto& inst : pointInstancers) {
        UsdPrim instPrim = inst.GetPrim();
        GfMatrix4d instWorldMat = xformCache.GetLocalToWorldTransform(instPrim);
        glm::mat4 instWorld = gfToGlmMatrix(instWorldMat);

        // Determine evaluation time for instancer
        UsdTimeCode instEvalTime = evalTime;
        VtVec3fArray testPos;
        inst.GetPositionsAttr().Get(&testPos, instEvalTime);
        if (testPos.empty()) {
            std::vector<double> samples;
            if (inst.GetPositionsAttr().GetTimeSamples(&samples) && !samples.empty()) {
                instEvalTime = UsdTimeCode(samples[0]);
            }
        }

        SdfPathVector targets;
        inst.GetPrototypesRel().GetTargets(&targets);
        if (targets.empty()) continue;

        VtIntArray protoIndices;
        inst.GetProtoIndicesAttr().Get(&protoIndices, instEvalTime);
        if (protoIndices.empty() && instEvalTime != UsdTimeCode::Default()) {
            inst.GetProtoIndicesAttr().Get(&protoIndices, UsdTimeCode::Default());
        }
        if (protoIndices.empty()) continue;

        VtArray<GfMatrix4d> xforms;
        bool ok = inst.ComputeInstanceTransformsAtTime(&xforms, instEvalTime, instEvalTime, UsdGeomPointInstancer::IncludeProtoXform);
        if (!ok || xforms.empty()) {
            Logger::warn("UsdLoader: Failed to compute instance transforms for PointInstancer '{}'", instPrim.GetPath().GetString());
            continue;
        }

        // Pre-triangulate all prototype geometries in prototype-root local space
        std::vector<std::vector<LocalProtoTriangle>> prototypeLocalTriangles(targets.size());
        for (size_t t = 0; t < targets.size(); ++t) {
            UsdPrim protoPrim = stage->GetPrimAtPath(targets[t]);
            if (!protoPrim) continue;
            GfMatrix4d protoRootWorld = xformCache.GetLocalToWorldTransform(protoPrim);
            GfMatrix4d protoRootWorldInv = protoRootWorld.GetInverse();

            for (const UsdPrim& childPrim : UsdPrimRange(protoPrim)) {
                if (!childPrim.IsA<UsdGeomMesh>()) continue;
                UsdGeomMesh protoMesh(childPrim);
                GfMatrix4d childWorld = xformCache.GetLocalToWorldTransform(childPrim);
                GfMatrix4d meshLocalToProtoGf = childWorld * protoRootWorldInv;
                glm::mat4 meshLocalToProto = gfToGlmMatrix(meshLocalToProtoGf);
                glm::mat3 protoNormalMatrix = glm::transpose(glm::inverse(glm::mat3(meshLocalToProto)));

                VtArray<GfVec3f> pts;
                protoMesh.GetPointsAttr().Get(&pts, instEvalTime);
                if (pts.empty() && instEvalTime != UsdTimeCode::Default()) {
                    protoMesh.GetPointsAttr().Get(&pts, UsdTimeCode::Default());
                }
                if (pts.empty()) continue;

                VtArray<int> fvc;
                protoMesh.GetFaceVertexCountsAttr().Get(&fvc, instEvalTime);
                if (fvc.empty() && instEvalTime != UsdTimeCode::Default()) {
                    protoMesh.GetFaceVertexCountsAttr().Get(&fvc, UsdTimeCode::Default());
                }
                VtArray<int> fvi;
                protoMesh.GetFaceVertexIndicesAttr().Get(&fvi, instEvalTime);
                if (fvi.empty() && instEvalTime != UsdTimeCode::Default()) {
                    protoMesh.GetFaceVertexIndicesAttr().Get(&fvi, UsdTimeCode::Default());
                }
                if (fvc.empty() || fvi.empty()) continue;

                VtArray<GfVec3f> normals;
                protoMesh.GetNormalsAttr().Get(&normals, instEvalTime);
                if (normals.empty() && instEvalTime != UsdTimeCode::Default()) {
                    protoMesh.GetNormalsAttr().Get(&normals, UsdTimeCode::Default());
                }
                TfToken normInterp = protoMesh.GetNormalsInterpolation();
                TfToken orientation = UsdGeomTokens->rightHanded;
                protoMesh.GetOrientationAttr().Get(&orientation, instEvalTime);
                if (orientation.IsEmpty() && instEvalTime != UsdTimeCode::Default()) {
                    protoMesh.GetOrientationAttr().Get(&orientation, UsdTimeCode::Default());
                }
                bool isLeftHanded = (orientation == UsdGeomTokens->leftHanded);

                VtArray<GfVec2f> uvs;
                TfToken uvInterp = UsdGeomTokens->constant;
                UsdGeomPrimvarsAPI pAPI(protoMesh);
                UsdGeomPrimvar stPrim = pAPI.GetPrimvar(TfToken("st"));
                if (!stPrim) stPrim = pAPI.GetPrimvar(TfToken("uv"));
                if (!stPrim) stPrim = pAPI.GetPrimvar(TfToken("UVMap"));
                if (stPrim) {
                    stPrim.ComputeFlattened(&uvs, instEvalTime);
                    if (uvs.empty() && instEvalTime != UsdTimeCode::Default()) {
                        stPrim.ComputeFlattened(&uvs, UsdTimeCode::Default());
                    }
                    uvInterp = stPrim.GetInterpolation();
                }

                uint32_t defaultMeshMatId = 0;
                UsdShadeMaterialBindingAPI bAPI(childPrim);
                UsdShadeMaterial boundMat = bAPI.ComputeBoundMaterial();
                if (boundMat) {
                    defaultMeshMatId = getOrCreateMaterial(boundMat);
                }

                UsdGeomPrimvar dispColorPrim = pAPI.GetPrimvar(TfToken("displayColor"));
                VtArray<GfVec3f> dispColors;
                TfToken dispColorInterp = UsdGeomTokens->constant;
                if (dispColorPrim) {
                    dispColorPrim.Get(&dispColors, instEvalTime);
                    if (dispColors.empty() && instEvalTime != UsdTimeCode::Default()) {
                        dispColorPrim.Get(&dispColors, UsdTimeCode::Default());
                    }
                    dispColorInterp = dispColorPrim.GetInterpolation();
                }
                if (!boundMat && !dispColors.empty()) {
                    if (dispColorInterp == UsdGeomTokens->constant || dispColors.size() == 1) {
                        defaultMeshMatId = getOrCreateColorMaterial(dispColors[0]);
                    }
                }

                // Check for per-face GeomSubsets in prototype mesh
                std::vector<uint32_t> protoFaceMaterials(fvc.size(), defaultMeshMatId);
                std::vector<UsdGeomSubset> pSubsets = UsdGeomSubset::GetAllGeomSubsets(protoMesh);
                for (const auto& subset : pSubsets) {
                    UsdShadeMaterialBindingAPI subBinding(subset.GetPrim());
                    UsdShadeMaterial subMat = subBinding.ComputeBoundMaterial();
                    if (subMat) {
                        uint32_t subMatId = getOrCreateMaterial(subMat);
                        VtIntArray subIndices;
                        subset.GetIndicesAttr().Get(&subIndices, instEvalTime);
                        if (subIndices.empty() && instEvalTime != UsdTimeCode::Default()) {
                            subset.GetIndicesAttr().Get(&subIndices, UsdTimeCode::Default());
                        }
                        for (int fIdx : subIndices) {
                            if (fIdx >= 0 && static_cast<size_t>(fIdx) < protoFaceMaterials.size()) {
                                protoFaceMaterials[fIdx] = subMatId;
                            }
                        }
                    }
                }

                size_t indexOffset = 0;
                for (size_t f = 0; f < fvc.size(); ++f) {
                    int count = fvc[f];
                    if (count < 3) {
                        indexOffset += count;
                        continue;
                    }

                    uint32_t faceMatId = protoFaceMaterials[f];
                    if (!boundMat && pSubsets.empty() && !dispColors.empty() && dispColorInterp == UsdGeomTokens->uniform && f < dispColors.size()) {
                        faceMatId = getOrCreateColorMaterial(dispColors[f]);
                    }

                    auto getLocalVertex = [&](int localIdx) -> Vertex {
                        int gIdx = fvi[indexOffset + localIdx];
                        Vertex v{};
                        GfVec3f p = pts[gIdx];
                        v.position = meshLocalToProto * glm::vec4(p[0], p[1], p[2], 1.0f);

                        glm::vec3 n(0.0f, 0.0f, 0.0f);
                        if (!normals.empty()) {
                            if (normInterp == UsdGeomTokens->uniform && f < normals.size()) {
                                GfVec3f usdNorm = normals[f];
                                n = glm::vec3(usdNorm[0], usdNorm[1], usdNorm[2]);
                            } else if ((normInterp == UsdGeomTokens->vertex || normInterp == UsdGeomTokens->varying) && gIdx < static_cast<int>(normals.size())) {
                                GfVec3f usdNorm = normals[gIdx];
                                n = glm::vec3(usdNorm[0], usdNorm[1], usdNorm[2]);
                            } else if (normInterp == UsdGeomTokens->faceVarying && (indexOffset + localIdx) < normals.size()) {
                                GfVec3f usdNorm = normals[indexOffset + localIdx];
                                n = glm::vec3(usdNorm[0], usdNorm[1], usdNorm[2]);
                            }
                        }
                        v.normal = glm::vec4(glm::length(n) > 1e-4f ? glm::normalize(protoNormalMatrix * n) : n, 0.0f);

                        glm::vec2 uv(0.0f);
                        if (!uvs.empty()) {
                            if ((uvInterp == UsdGeomTokens->vertex || uvInterp == UsdGeomTokens->varying) && gIdx < static_cast<int>(uvs.size())) {
                                uv = glm::vec2(uvs[gIdx][0], uvs[gIdx][1]);
                            } else if (uvInterp == UsdGeomTokens->faceVarying && (indexOffset + localIdx) < uvs.size()) {
                                uv = glm::vec2(uvs[indexOffset + localIdx][0], uvs[indexOffset + localIdx][1]);
                            }
                        }
                        v.position.w = uv.x;
                        v.normal.w = 1.0f - uv.y;
                        v.tangent = glm::vec4(1.0f, 0.0f, 0.0f, 1.0f);
                        return v;
                    };

                    // Fan triangulation: (0, i, i + 1) for CCW rightHanded, (0, i + 1, i) for CW leftHanded
                    for (int i = 1; i < count - 1; ++i) {
                        int idx1 = isLeftHanded ? (i + 1) : i;
                        int idx2 = isLeftHanded ? i : (i + 1);
                        Vertex v0 = getLocalVertex(0);
                        Vertex v1 = getLocalVertex(idx1);
                        Vertex v2 = getLocalVertex(idx2);

                        glm::vec3 p0_3 = glm::vec3(v0.position);
                        glm::vec3 p1_3 = glm::vec3(v1.position);
                        glm::vec3 p2_3 = glm::vec3(v2.position);
                        glm::vec3 geoNormal = glm::cross(p1_3 - p0_3, p2_3 - p0_3);
                        if (glm::length(geoNormal) > 1e-7f) {
                            geoNormal = glm::normalize(geoNormal);
                        } else {
                            geoNormal = glm::vec3(0.0f, 1.0f, 0.0f);
                        }

                        if (glm::length(glm::vec3(v0.normal)) < 1e-4f) v0.normal = glm::vec4(geoNormal, v0.normal.w);
                        if (glm::length(glm::vec3(v1.normal)) < 1e-4f) v1.normal = glm::vec4(geoNormal, v1.normal.w);
                        if (glm::length(glm::vec3(v2.normal)) < 1e-4f) v2.normal = glm::vec4(geoNormal, v2.normal.w);

                        glm::vec3 e1 = p1_3 - p0_3;
                        glm::vec3 e2 = p2_3 - p0_3;
                        glm::vec2 uv0(v0.position.w, v0.normal.w);
                        glm::vec2 uv1(v1.position.w, v1.normal.w);
                        glm::vec2 uv2(v2.position.w, v2.normal.w);
                        glm::vec2 duv1 = uv1 - uv0;
                        glm::vec2 duv2 = uv2 - uv0;
                        float det = duv1.x * duv2.y - duv2.x * duv1.y;
                        glm::vec3 tanDir;
                        if (std::abs(det) > 1e-6f) {
                            tanDir = glm::normalize((e1 * duv2.y - e2 * duv1.y) / det);
                        } else {
                            glm::vec3 up = std::abs(geoNormal.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
                            tanDir = glm::normalize(glm::cross(up, geoNormal));
                        }
                        v0.tangent = glm::vec4(tanDir, 1.0f);
                        v1.tangent = glm::vec4(tanDir, 1.0f);
                        v2.tangent = glm::vec4(tanDir, 1.0f);

                        LocalProtoTriangle ltri{};
                        ltri.v0 = v0;
                        ltri.v1 = v1;
                        ltri.v2 = v2;
                        ltri.materialId = faceMatId;
                        prototypeLocalTriangles[t].push_back(ltri);
                    }
                    indexOffset += count;
                }
            }
        }

        // Register prototype geometries as BLAS ranges (deduplicated across instancers)
        for (size_t t = 0; t < targets.size(); ++t) {
            std::string targetPath = targets[t].GetString();
            if (protoPathToBlas.find(targetPath) != protoPathToBlas.end()) {
                continue;
            }
            if (!prototypeLocalTriangles[t].empty()) {
                uint32_t startTri = static_cast<uint32_t>(data.triangles.size());
                for (const auto& ltri : prototypeLocalTriangles[t]) {
                    TriangleGPU tri{};
                    tri.v0 = ltri.v0;
                    tri.v1 = ltri.v1;
                    tri.v2 = ltri.v2;
                    tri.materialId = ltri.materialId;
                    data.triangles.push_back(tri);
                }
                uint32_t triCount = static_cast<uint32_t>(prototypeLocalTriangles[t].size());
                uint32_t protoBlasIdx = static_cast<uint32_t>(data.blasRanges.size());
                BlasGeometryRange range{};
                range.firstTriangle = startTri;
                range.triangleCount = triCount;
                range.numOpaqueTriangles = triCount; // Partitioned by Engine::partitionSceneGeometry
                data.blasRanges.push_back(range);
                protoPathToBlas[targetPath] = protoBlasIdx;
            } else {
                protoPathToBlas[targetPath] = UINT32_MAX;
            }
        }

        // Check optional instance limit via PATHWAYS_USD_MAX_INSTANCES
        size_t numInstances = xforms.size();
        if (protoIndices.size() < numInstances) numInstances = protoIndices.size();

        const char* maxInstEnv = std::getenv("PATHWAYS_USD_MAX_INSTANCES");
        if (maxInstEnv) {
            size_t maxLimit = std::strtoul(maxInstEnv, nullptr, 10);
            if (maxLimit > 0 && maxLimit < numInstances) {
                numInstances = maxLimit;
                Logger::info("UsdLoader: Capping PointInstancer to {} instances per PATHWAYS_USD_MAX_INSTANCES", numInstances);
            }
        }

        Logger::info("UsdLoader: Instancing PointInstancer '{}': {} instances referencing {} prototype BLASes (Total Unique Prototypes: {})",
                     instPrim.GetPath().GetString(), numInstances, targets.size(), protoPathToBlas.size());

        for (size_t i = 0; i < numInstances; ++i) {
            int pIdx = protoIndices[i];
            if (pIdx < 0 || static_cast<size_t>(pIdx) >= targets.size()) continue;
            std::string tPath = targets[pIdx].GetString();
            auto bit = protoPathToBlas.find(tPath);
            if (bit == protoPathToBlas.end() || bit->second == UINT32_MAX) continue;
            uint32_t bIdx = bit->second;

            GfMatrix4d xf = xforms[i];
            glm::mat4 instMat = gfToGlmMatrix(xf);
            glm::mat4 M = stageTransform * instWorld * instMat;

            uint32_t customIdx = static_cast<uint32_t>(data.instanceData.size());
            SceneInstance sInst{};
            sInst.blasIndex = bIdx;
            sInst.transform = M;
            sInst.customIndex = customIdx;
            data.instances.push_back(sInst);

            InstanceGPU instGPU{};
            instGPU.firstTriangle = data.blasRanges[bIdx].firstTriangle;
            instGPU.numOpaqueTriangles = data.blasRanges[bIdx].numOpaqueTriangles;
            instGPU.materialOffset = 0;
            instGPU.flags = 0;
            data.instanceData.push_back(instGPU);
        }
    }

    // 6. Compute Scene Spatial Bounds and Setup Default Camera & Lighting
    if (!data.triangles.empty() || !data.instances.empty()) {
        glm::vec3 bMin(1e30f);
        glm::vec3 bMax(-1e30f);
        if (!data.instances.empty() && !data.blasRanges.empty()) {
            std::vector<glm::vec3> blasMin(data.blasRanges.size(), glm::vec3(1e30f));
            std::vector<glm::vec3> blasMax(data.blasRanges.size(), glm::vec3(-1e30f));
            for (size_t b = 0; b < data.blasRanges.size(); ++b) {
                const auto& range = data.blasRanges[b];
                for (uint32_t t = 0; t < range.triangleCount; ++t) {
                    const auto& tri = data.triangles[range.firstTriangle + t];
                    blasMin[b] = glm::min(blasMin[b], glm::vec3(tri.v0.position));
                    blasMin[b] = glm::min(blasMin[b], glm::vec3(tri.v1.position));
                    blasMin[b] = glm::min(blasMin[b], glm::vec3(tri.v2.position));
                    blasMax[b] = glm::max(blasMax[b], glm::vec3(tri.v0.position));
                    blasMax[b] = glm::max(blasMax[b], glm::vec3(tri.v1.position));
                    blasMax[b] = glm::max(blasMax[b], glm::vec3(tri.v2.position));
                }
            }
            for (const auto& inst : data.instances) {
                if (inst.blasIndex >= data.blasRanges.size()) continue;
                const glm::vec3& lMin = blasMin[inst.blasIndex];
                const glm::vec3& lMax = blasMax[inst.blasIndex];
                for (int c = 0; c < 8; ++c) {
                    glm::vec3 corner(
                        (c & 1) ? lMax.x : lMin.x,
                        (c & 2) ? lMax.y : lMin.y,
                        (c & 4) ? lMax.z : lMin.z
                    );
                    glm::vec3 wCorner = glm::vec3(inst.transform * glm::vec4(corner, 1.0f));
                    bMin = glm::min(bMin, wCorner);
                    bMax = glm::max(bMax, wCorner);
                }
            }
        } else {
            for (const auto& tri : data.triangles) {
                bMin = glm::min(bMin, glm::vec3(tri.v0.position));
                bMin = glm::min(bMin, glm::vec3(tri.v1.position));
                bMin = glm::min(bMin, glm::vec3(tri.v2.position));
                bMax = glm::max(bMax, glm::vec3(tri.v0.position));
                bMax = glm::max(bMax, glm::vec3(tri.v1.position));
                bMax = glm::max(bMax, glm::vec3(tri.v2.position));
            }
        }
        data.boundsMin = bMin;
        data.boundsMax = bMax;
        data.focalBoundsMin = bMin;
        data.focalBoundsMax = bMax;
        glm::vec3 center = (bMin + bMax) * 0.5f;
        glm::vec3 extent = bMax - bMin;
        data.centralTarget = center;
        float maxDim = std::max({extent.x, extent.y, extent.z});
        if (maxDim < 1e-4f) maxDim = 2.0f;
        data.sceneRadius = maxDim * 0.5f;
        data.focalRadius = maxDim * 0.5f;

        if (data.hasCamera) {
            // If an authored camera had fallback focus distance, project towards scene central target
            if (data.focalDistance <= 0.01f || data.focalDistance == 10.0f) {
                glm::vec3 camFront = glm::normalize(data.cameraTarget - data.cameraPosition);
                glm::vec3 toCenter = center - data.cameraPosition;
                float projDist = glm::dot(toCenter, camFront);
                float focalDist = projDist > 0.2f ? projDist : glm::length(toCenter);
                if (focalDist < 0.1f) focalDist = data.focalRadius;
                data.focalDistance = focalDist;
                data.cameraTarget = data.cameraPosition + camFront * focalDist;
            }
        } else {
            // Robust aspect-ratio-aware bounding box frustum fitting
            data.hasCamera = true;
            data.cameraFov = 45.0f;
            float fovRad = glm::radians(data.cameraFov);
            float tanHalfFovV = std::tan(fovRad * 0.5f);
            float aspect = 16.0f / 9.0f;
            float tanHalfFovH = tanHalfFovV * aspect;

            // Compute viewing direction: elevated 3/4 vantage
            // Wide outdoor scenes benefit from an azimuth angle for 3/4 depth parallax
            float azimuthDeg = (extent.x > 50.0f) ? 18.0f : 28.0f;
            float pitchDeg = (extent.y < extent.x * 0.4f) ? 14.0f : 12.0f;

            float radAzimuth = glm::radians(azimuthDeg);
            float radPitch = glm::radians(pitchDeg);

            // Unit direction from scene center to camera
            glm::vec3 viewDir(
                std::sin(radAzimuth) * std::cos(radPitch),
                std::sin(radPitch),
                std::cos(radAzimuth) * std::cos(radPitch)
            );
            glm::vec3 fwd = -viewDir;
            glm::vec3 upWorld(0.0f, 1.0f, 0.0f);
            glm::vec3 right = glm::normalize(glm::cross(fwd, upWorld));
            glm::vec3 camUp = glm::normalize(glm::cross(right, fwd));

            // Test all 8 corners of the scene bounding box to compute exact required distance
            float requiredDist = 0.0f;
            glm::vec3 corners[8] = {
                {bMin.x, bMin.y, bMin.z}, {bMax.x, bMin.y, bMin.z},
                {bMin.x, bMax.y, bMin.z}, {bMax.x, bMax.y, bMin.z},
                {bMin.x, bMin.y, bMax.z}, {bMax.x, bMin.y, bMax.z},
                {bMin.x, bMax.y, bMax.z}, {bMax.x, bMax.y, bMax.z}
            };

            for (int i = 0; i < 8; ++i) {
                glm::vec3 delta = corners[i] - center;
                float projDir = glm::dot(delta, viewDir);
                float projRight = std::abs(glm::dot(delta, right));
                float projUp = std::abs(glm::dot(delta, camUp));

                float distH = projDir + projRight / tanHalfFovH;
                float distV = projDir + projUp / tanHalfFovV;
                requiredDist = std::max({requiredDist, distH, distV});
            }

            // Apply 10% framing margin for aesthetic breathing room
            float finalDist = requiredDist * 1.10f;
            if (finalDist < maxDim * 0.5f) {
                finalDist = maxDim * 0.5f;
            }

            data.cameraPosition = center + viewDir * finalDist;
            data.cameraTarget = center;
            data.cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
            data.focalDistance = finalDist;
            Logger::info("UsdLoader: Default camera framed at pos ({:.2f}, {:.2f}, {:.2f}), target ({:.2f}, {:.2f}, {:.2f}), dist {:.2f} m",
                         data.cameraPosition.x, data.cameraPosition.y, data.cameraPosition.z,
                         data.cameraTarget.x, data.cameraTarget.y, data.cameraTarget.z, finalDist);
        }

        if (data.lights.empty() && !hasDomeLight) {
            float lightSide = maxDim * 0.8f;
            float lightY = bMax.y + maxDim * 0.6f;
            LightGPU defaultLight{};
            defaultLight.position = glm::vec4(center.x - lightSide * 0.5f, lightY, center.z - lightSide * 0.5f, LIGHT_AREA_QUAD);
            defaultLight.u = glm::vec4(lightSide, 0.0f, 0.0f, 0.0f);
            defaultLight.v = glm::vec4(0.0f, 0.0f, lightSide, 0.0f);
            defaultLight.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
            float area = lightSide * lightSide;
            defaultLight.emission = glm::vec4(15.0f, 15.0f, 15.0f, area);
            data.lights.push_back(defaultLight);
        }

        MeshRange range{};
        range.name = "SceneGeometry";
        range.minBound = bMin;
        range.maxBound = bMax;
        range.firstTriangle = 0;
        range.triangleCount = static_cast<uint32_t>(data.triangles.size());
        data.meshRanges.push_back(range);
    }

    if (!data.blasRanges.empty()) {
        Logger::info("UsdLoader: Successfully ingested OpenUSD asset: {} Triangles ({} BLASes, {} Instances), {} Materials, {} Lights",
                     data.triangles.size(), data.blasRanges.size(), data.instances.size(), data.materials.size(), data.lights.size());
    } else {
        Logger::info("UsdLoader: Successfully ingested OpenUSD asset: {} Triangles, {} Materials, {} Lights",
                     data.triangles.size(), data.materials.size(), data.lights.size());
    }
    return data;
#else
    Logger::warn("UsdLoader: OpenUSD support is disabled in this build. Cannot load '{}'", filepath);
    return ProceduralScene::createCornellBox();
#endif
}

bool UsdLoader::populateMetadata(const std::string& filepath, uint64_t& outTriangles, uint32_t& outMaterials) {
#if defined(PATHWAYS_ENABLE_USD) && PATHWAYS_ENABLE_USD
    if (!std::filesystem::exists(filepath)) return false;
    UsdStageRefPtr stage = UsdStage::Open(filepath);
    if (!stage) return false;

    uint32_t totalTris = 0;
    uint32_t matCount = 0;
    std::unordered_map<std::string, uint32_t> protoTris;
    std::unordered_set<std::string> uniqueDisplayColors;

    UsdTimeCode evalTime = stage->HasAuthoredTimeCodeRange() ? UsdTimeCode(stage->GetStartTimeCode()) : UsdTimeCode::Default();
    std::vector<UsdGeomPointInstancer> pointInstancers;
    std::vector<SdfPath> allPrototypeTargets;

    for (const UsdPrim& prim : stage->Traverse(UsdTraverseInstanceProxies())) {
        if (prim.IsA<UsdGeomPointInstancer>()) {
            UsdGeomPointInstancer inst(prim);
            pointInstancers.push_back(inst);
            SdfPathVector targets;
            inst.GetPrototypesRel().GetTargets(&targets);
            for (const auto& tgt : targets) {
                allPrototypeTargets.push_back(tgt);
            }
        }
    }

    for (const UsdPrim& prim : stage->Traverse(UsdTraverseInstanceProxies())) {
        if (prim.IsA<UsdShadeMaterial>()) {
            matCount++;
        } else if (prim.IsA<UsdGeomMesh>()) {
            bool isPointInstancerProto = false;
            for (const auto& tgt : allPrototypeTargets) {
                if (prim.GetPath().HasPrefix(tgt)) {
                    isPointInstancerProto = true;
                    break;
                }
            }

            UsdGeomMesh mesh(prim);
            VtIntArray faceVertexCounts;
            mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts, evalTime);
            if (faceVertexCounts.empty() && evalTime != UsdTimeCode::Default()) {
                mesh.GetFaceVertexCountsAttr().Get(&faceVertexCounts, UsdTimeCode::Default());
            }
            uint32_t meshTris = 0;
            for (int count : faceVertexCounts) {
                if (count >= 3) meshTris += (count - 2);
            }
            std::string pathStr = prim.GetPath().GetString();
            protoTris[pathStr] = meshTris;
            if (!isPointInstancerProto &&
                pathStr.find("/Prototypes") == std::string::npos &&
                pathStr.find("/prototypes") == std::string::npos &&
                pathStr.find("prototype") == std::string::npos) {
                totalTris += meshTris;
            }

            UsdGeomPrimvarsAPI pAPI(mesh);
            UsdGeomPrimvar dispColorPrim = pAPI.GetPrimvar(TfToken("displayColor"));
            if (dispColorPrim) {
                VtArray<GfVec3f> dispColors;
                dispColorPrim.Get(&dispColors, evalTime);
                if (dispColors.empty() && evalTime != UsdTimeCode::Default()) {
                    dispColorPrim.Get(&dispColors, UsdTimeCode::Default());
                }
                for (const auto& c : dispColors) {
                    char colKey[64];
                    std::snprintf(colKey, sizeof(colKey), "%.3f_%.3f_%.3f", c[0], c[1], c[2]);
                    uniqueDisplayColors.insert(colKey);
                }
            }
        }
    }

    for (const auto& instancer : pointInstancers) {
        VtIntArray protoIndices;
        instancer.GetProtoIndicesAttr().Get(&protoIndices, evalTime);
        if (protoIndices.empty() && evalTime != UsdTimeCode::Default()) {
            instancer.GetProtoIndicesAttr().Get(&protoIndices, UsdTimeCode::Default());
        }
        if (protoIndices.empty()) {
            std::vector<double> samples;
            if (instancer.GetPositionsAttr().GetTimeSamples(&samples) && !samples.empty()) {
                instancer.GetProtoIndicesAttr().Get(&protoIndices, UsdTimeCode(samples[0]));
            }
        }
        SdfPathVector targets;
        instancer.GetPrototypesRel().GetTargets(&targets);

        size_t numInstances = protoIndices.size();
        const char* maxInstEnv = std::getenv("PATHWAYS_USD_MAX_INSTANCES");
        if (maxInstEnv) {
            size_t maxLimit = std::strtoul(maxInstEnv, nullptr, 10);
            if (maxLimit > 0 && maxLimit < numInstances) {
                numInstances = maxLimit;
            }
        }

        std::vector<uint32_t> targetTris(targets.size(), 0);
        for (size_t t = 0; t < targets.size(); ++t) {
            std::string pPath = targets[t].GetString();
            for (const auto& [meshPath, tris] : protoTris) {
                if (meshPath.rfind(pPath, 0) == 0) {
                    targetTris[t] += tris;
                }
            }
        }

        size_t projectedTriangles = 0;
        for (size_t i = 0; i < numInstances; ++i) {
            int pIdx = protoIndices[i];
            if (pIdx >= 0 && static_cast<size_t>(pIdx) < targetTris.size()) {
                projectedTriangles += targetTris[pIdx];
            }
        }
        totalTris += static_cast<uint32_t>(projectedTriangles);
    }

    outTriangles = totalTris;
    uint32_t totalMats = matCount + static_cast<uint32_t>(uniqueDisplayColors.size());
    if (!uniqueDisplayColors.empty()) {
        totalMats += 1; // Default fallback material at slot 0
    }
    outMaterials = (totalMats > 0) ? totalMats : 1;
    return (totalTris > 0);
#else
    (void)filepath;
    outTriangles = 0;
    outMaterials = 0;
    return false;
#endif
}

} // namespace pathways
