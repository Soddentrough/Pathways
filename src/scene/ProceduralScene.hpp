#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <cstdint>
#include "scene/Material.hpp"
#include "scene/Light.hpp"

namespace pathways {

struct Vertex {
    glm::vec4 position; // xyz: pos, w: u
    glm::vec4 normal;   // xyz: norm, w: v
};

struct TriangleGPU {
    Vertex v0;
    Vertex v1;
    Vertex v2;
    uint32_t materialId;
    uint32_t padding[3];
};

struct SphereGPU {
    glm::vec4 centerRadius; // xyz: center, w: radius
    uint32_t materialId;
    uint32_t padding[3];
};

struct TextureData {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels; // RGBA8
};

struct SceneData {
    std::vector<TriangleGPU> triangles;
    std::vector<SphereGPU> spheres;
    std::vector<MaterialGPU> materials;
    std::vector<LightGPU> lights;
    std::vector<TextureData> textures;

    bool hasCamera = false;
    glm::vec3 cameraPosition = glm::vec3(0.0f, 1.0f, 2.7f);
    glm::vec3 cameraTarget = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    float cameraFov = 45.0f;
};

class ProceduralScene {
public:
    static SceneData createCornellBox();
};

} // namespace pathways
