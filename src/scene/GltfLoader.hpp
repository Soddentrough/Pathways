#pragma once

#include <string>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include "scene/Material.hpp"
#include "scene/Light.hpp"
#include "scene/Camera.hpp"

#include "scene/ProceduralScene.hpp"

namespace pathways {

struct GltfVertex {
    glm::vec4 position; // xyz: pos, w: u
    glm::vec4 normal;   // xyz: norm, w: v
    glm::vec4 tangent;  // xyz: tangent, w: sign
};

struct GltfMeshPrimitive {
    std::vector<GltfVertex> vertices;
    std::vector<uint32_t> indices;
    uint32_t materialIndex = 0;
};

struct GltfMesh {
    std::string name;
    std::vector<GltfMeshPrimitive> primitives;
};

struct GltfNode {
    std::string name;
    int32_t meshIndex = -1;
    glm::mat4 worldTransform = glm::mat4(1.0f);
    std::vector<uint32_t> children;
};

struct GltfScene {
    std::vector<GltfMesh> meshes;
    std::vector<GltfNode> nodes;
    std::vector<MaterialGPU> materials;
    std::vector<LightGPU> lights;
    std::vector<Camera> cameras;
    std::vector<TextureData> textures;
    std::string assetName;
    std::vector<std::string> extensionsUsed;
    std::vector<std::string> extensionsRequired;
    bool hasMeshQuantization = false;

    // Scene Authoring Metadata & Fallback Tracking
    bool hasSceneMetadata = false;
    std::string sceneSourceFormat = "";
    uint32_t sceneDefaultSpp = 0;
    uint32_t sceneDefaultMaxBounces = 0;
    uint32_t sceneDefaultWidth = 0;
    uint32_t sceneDefaultHeight = 0;
    bool hasFallbackSun = false;
    std::vector<std::string> sceneWarnings;
};

class GltfLoader {
public:
    static bool load(const std::string& filepath, GltfScene& outScene);
    static SceneData loadSceneData(const std::string& filepath);
};

} // namespace pathways
