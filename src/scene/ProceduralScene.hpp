#pragma once

#include <glm/glm.hpp>
#include <vector>
#include <string>
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

struct MeshRange {
    std::string name;
    glm::vec3 minBound{ 1e30f };
    glm::vec3 maxBound{ -1e30f };
    uint32_t firstTriangle = 0;
    uint32_t triangleCount = 0;
};

struct SceneData {
    std::vector<TriangleGPU> triangles;
    std::vector<SphereGPU> spheres;
    std::vector<MaterialGPU> materials;
    std::vector<LightGPU> lights;
    std::vector<TextureData> textures;
    std::vector<MeshRange> meshRanges;

    bool hasCamera = false;
    glm::vec3 cameraPosition = glm::vec3(0.0f, 1.0f, 2.7f);
    glm::vec3 cameraTarget = glm::vec3(0.0f, 1.0f, 0.0f);
    glm::vec3 cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
    float cameraFov = 45.0f;

    glm::vec3 boundsMin = glm::vec3(-1.0f, 0.0f, -1.0f);
    glm::vec3 boundsMax = glm::vec3(1.0f, 2.0f, 1.0f);
    float sceneRadius = 2.0f;

    glm::vec3 focalBoundsMin = glm::vec3(-1.0f, 0.0f, -1.0f);
    glm::vec3 focalBoundsMax = glm::vec3(1.0f, 2.0f, 1.0f);
    float focalRadius = 2.0f;
    float focalDistance = 2.7f;
    glm::vec3 centralTarget = glm::vec3(0.0f, 1.0f, 0.0f);

    // Fast CPU raycast against scene geometry for camera pivot targeting
    bool raycast(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float maxDist,
                 float& outHitDist, glm::vec3& outHitPoint, std::string* outHitName = nullptr) const;
};

class ProceduralScene {
public:
    static SceneData createCornellBox();
};

} // namespace pathways
