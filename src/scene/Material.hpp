#pragma once

#include <glm/glm.hpp>
#include <cstdint>

namespace pathways {

enum MaterialType : uint32_t {
    MATERIAL_DIFFUSE = 0,
    MATERIAL_METALLIC = 1,
    MATERIAL_DIELECTRIC = 2, // Refraction / Glass
    MATERIAL_EMISSIVE = 3
};

struct MaterialGPU {
    glm::vec4 albedo;
    glm::vec4 emissive;
    float roughness;
    float metallic;
    float ior;          // Index of refraction (e.g. 1.5 for glass)
    float transmission; // 0.0 = opaque, 1.0 = full transmission / glass
    uint32_t type;      // MaterialType
    uint32_t albedoTex;
    uint32_t normalTex;
    uint32_t padding;
};

} // namespace pathways
