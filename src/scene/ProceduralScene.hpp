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
    glm::vec4 tangent;  // xyz: tangent, w: sign (+1 or -1)
};
static_assert(sizeof(Vertex) == 48, "Vertex must be exactly 48 bytes");

struct TriangleGPU {
    Vertex v0;
    Vertex v1;
    Vertex v2;
    uint32_t materialId;
    uint32_t padding[3];
};
static_assert(sizeof(TriangleGPU) == 160, "TriangleGPU must be exactly 160 bytes");

struct SphereGPU {
    glm::vec4 centerRadius; // xyz: center, w: radius
    uint32_t materialId;
    uint32_t padding[3];
};
static_assert(sizeof(SphereGPU) == 32, "SphereGPU must be exactly 32 bytes");

struct TextureData {
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> pixels; // RGBA8
    bool isSrgb = false;
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

    glm::vec3 boundsMin = glm::vec3(-1.0f, 0.0f, -1.0f);
    glm::vec3 boundsMax = glm::vec3(1.0f, 2.0f, 1.0f);
    float sceneRadius = 2.0f;
};

class ProceduralScene {
public:
    static SceneData createCornellBox();
};

} // namespace pathways
