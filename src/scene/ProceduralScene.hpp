#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>
#include <vector>
#include <string>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include "scene/Material.hpp"
#include "scene/Light.hpp"
#include "scene/LightTree.hpp"

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

// 32-bit Octahedral normal/direction encoding (Cigolle et al.)
inline glm::vec2 octSign(glm::vec2 v) {
    return glm::vec2((v.x >= 0.0f) ? 1.0f : -1.0f, (v.y >= 0.0f) ? 1.0f : -1.0f);
}

inline glm::vec2 octEncode(glm::vec3 v) {
    float l1 = std::abs(v.x) + std::abs(v.y) + std::abs(v.z);
    if (l1 > 1e-6f) {
        v /= l1;
    } else {
        v = glm::vec3(0.0f, 0.0f, 1.0f);
    }
    glm::vec2 xy(v.x, v.y);
    if (v.z >= 0.0f) {
        return xy;
    }
    return (glm::vec2(1.0f) - glm::abs(glm::vec2(v.y, v.x))) * octSign(xy);
}

inline uint32_t packOct32(glm::vec3 v) {
    float len = glm::length(v);
    if (len > 1e-6f) {
        v /= len;
    } else {
        v = glm::vec3(0.0f, 0.0f, 1.0f);
    }
    return glm::packSnorm2x16(octEncode(v));
}

// 64-byte cache-line aligned shading triangle struct (2 triangles per 128B RDNA4 vector cache line)
struct alignas(16) TriangleShadeGPU {
    uint32_t octNormal0 = 0;
    uint32_t octNormal1 = 0;
    uint32_t octNormal2 = 0;
    uint32_t octTan0 = 0;

    uint32_t octTan1 = 0;
    uint32_t octTan2 = 0;
    uint32_t uv0 = 0;
    uint32_t uv1 = 0;

    uint32_t uv2 = 0;
    uint32_t tanSigns = 0;
    uint32_t materialId = 0;
    uint32_t padding0 = 0;

    uint32_t reserved[4] = {0, 0, 0, 0};
};
static_assert(sizeof(TriangleShadeGPU) == 64, "TriangleShadeGPU must be exactly 64 bytes (2 triangles per 128B cache line)");

inline TriangleShadeGPU createTriangleShadeGPU(const TriangleGPU& tri) {
    TriangleShadeGPU s{};
    s.octNormal0 = packOct32(glm::vec3(tri.v0.normal));
    s.octNormal1 = packOct32(glm::vec3(tri.v1.normal));
    s.octNormal2 = packOct32(glm::vec3(tri.v2.normal));
    s.octTan0    = packOct32(glm::vec3(tri.v0.tangent));

    s.octTan1    = packOct32(glm::vec3(tri.v1.tangent));
    s.octTan2    = packOct32(glm::vec3(tri.v2.tangent));
    s.uv0        = glm::packHalf2x16(glm::vec2(tri.v0.position.w, tri.v0.normal.w));
    s.uv1        = glm::packHalf2x16(glm::vec2(tri.v1.position.w, tri.v1.normal.w));

    s.uv2        = glm::packHalf2x16(glm::vec2(tri.v2.position.w, tri.v2.normal.w));

    uint32_t signs = 0;
    if (tri.v0.tangent.w >= 0.0f) signs |= (1u << 0);
    if (tri.v1.tangent.w >= 0.0f) signs |= (1u << 1);
    if (tri.v2.tangent.w >= 0.0f) signs |= (1u << 2);
    s.tanSigns   = signs;

    s.materialId = tri.materialId;
    s.padding0   = 0;
    s.reserved[0] = 0;
    s.reserved[1] = 0;
    s.reserved[2] = 0;
    s.reserved[3] = 0;

    return s;
}

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

struct InstanceGPU {
    uint32_t firstTriangle = 0;      // Start index in triangles[] buffer
    uint32_t numOpaqueTriangles = 0; // Number of opaque triangles in this prototype
    uint32_t materialOffset = 0;     // Optional material ID offset
    uint32_t flags = 0;              // Flags / metadata
};
static_assert(sizeof(InstanceGPU) == 16, "InstanceGPU must be 16 bytes (std430 aligned)");

struct BlasGeometryRange {
    uint32_t firstTriangle = 0;
    uint32_t triangleCount = 0;
    uint32_t numOpaqueTriangles = 0;
};

struct SceneInstance {
    uint32_t blasIndex = 0;
    glm::mat4 transform = glm::mat4(1.0f);
    uint32_t customIndex = 0;
};

struct AnimatedInstance {
    uint32_t instanceIndex = 0;
    glm::vec3 basePosition = glm::vec3(0.0f);
    glm::vec3 rotationAxis = glm::vec3(0.0f, 1.0f, 0.0f);
    float rotationSpeed = 0.8f; // radians per second
    glm::mat4 baseTransform = glm::mat4(1.0f);
};

struct SceneData {
    std::vector<TriangleGPU> triangles;
    std::vector<SphereGPU> spheres;
    std::vector<MaterialGPU> materials;
    std::vector<LightGPU> lights;
    std::vector<LightTreeNodeGPU> lightTreeNodes;
    std::vector<TextureData> textures;
    std::vector<MeshRange> meshRanges;
    uint32_t numOpaqueTriangles = 0;

    // Multi-BLAS & Hardware Ray Tracing Instancing
    std::vector<BlasGeometryRange> blasRanges; // If empty, monolithic single-BLAS is used
    std::vector<SceneInstance> instances;      // If empty, 1 identity instance is generated
    std::vector<InstanceGPU> instanceData;     // Uploaded to InstancesBuffer (binding 30)
    std::vector<AnimatedInstance> animatedInstances; // Dynamic kinematic animators

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

    // Dielectric Geometry Bounding Box for Caustic Photon Injection
    bool hasDielectrics = false;
    glm::vec3 dielectricBoundsMin = glm::vec3(1e9f);
    glm::vec3 dielectricBoundsMax = glm::vec3(-1e9f);

    // Environment Map & Dome Light Ingestion
    std::string domeLightHdriPath;
    float domeLightIntensity = 1.0f;

    // Fast CPU raycast against scene geometry for camera pivot targeting
    bool raycast(const glm::vec3& rayOrigin, const glm::vec3& rayDir, float maxDist,
                 float& outHitDist, glm::vec3& outHitPoint, std::string* outHitName = nullptr) const;
};

class ProceduralScene {
public:
    static SceneData createCornellBox();
    static SceneData createManyLightsScene(uint32_t gridDim = 8);
    static SceneData createCyberCityScene();
    static SceneData createInfinityMirrorScene();
};

} // namespace pathways
