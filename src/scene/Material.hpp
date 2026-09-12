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

enum AlphaMode : uint32_t {
    ALPHA_MODE_OPAQUE = 0,
    ALPHA_MODE_MASK = 1,
    ALPHA_MODE_BLEND = 2
};

struct MaterialGPU {
    glm::vec4 albedo = glm::vec4(1.0f);        // 16 bytes (offset 0) - baseColorFactor (linear RGBA)
    glm::vec4 emissive = glm::vec4(0.0f);      // 16 bytes (offset 16) - emissiveFactor * emissiveStrength (linear RGB)
    float roughness = 1.0f;                    // 4 bytes (offset 32)
    float metallic = 0.0f;                     // 4 bytes (offset 36)
    float ior = 1.5f;                          // 4 bytes (offset 40)
    float transmission = 0.0f;                 // 4 bytes (offset 44)
    uint32_t type = MATERIAL_DIFFUSE;          // 4 bytes (offset 48) - MaterialType
    uint32_t albedoTex = 0;                    // 4 bytes (offset 52)
    uint32_t normalTex = 0;                    // 4 bytes (offset 56)
    uint32_t mrTex = 0;                        // 4 bytes (offset 60) - metallic-roughness texture (G = roughness, B = metallic)
    uint32_t emissiveTex = 0;                  // 4 bytes (offset 64)
    uint32_t occlusionTex = 0;                 // 4 bytes (offset 68) - ambient occlusion (R channel)
    uint32_t transmissionTex = 0;              // 4 bytes (offset 72)
    float alphaCutoff = 0.5f;                  // 4 bytes (offset 76) - default 0.5f
    uint32_t alphaMode = ALPHA_MODE_OPAQUE;    // 4 bytes (offset 80) - AlphaMode
    float occlusionStrength = 1.0f;            // 4 bytes (offset 84) - default 1.0f
    float normalScale = 1.0f;                  // 4 bytes (offset 88) - default 1.0f
    uint32_t thicknessTex = 0;                 // 4 bytes (offset 92) - volume thickness texture (G)

    // Extended glTF 2.0 PBR Properties (offsets 96-144)
    glm::vec4 attenuationColor = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f); // 16 bytes (offset 96) - rgb: attenuationColor, w: attenuationDistance (0 = infinite)
    float clearcoat = 0.0f;                    // 4 bytes (offset 112) - clearcoatFactor [0, 1]
    float clearcoatRoughness = 0.0f;           // 4 bytes (offset 116) - clearcoatRoughnessFactor [0, 1]
    uint32_t clearcoatTex = 0;                 // 4 bytes (offset 120) - clearcoat intensity texture (R)
    uint32_t clearcoatRoughnessTex = 0;        // 4 bytes (offset 124) - clearcoat roughness texture (G)
    uint32_t clearcoatNormalTex = 0;           // 4 bytes (offset 128) - independent clearcoat normal map
    float thickness = 0.0f;                    // 4 bytes (offset 132) - volume thicknessFactor
    float specularFactor = 1.0f;               // 4 bytes (offset 136) - KHR_materials_specular factor
    uint32_t specularTex = 0;                  // 4 bytes (offset 140) - KHR_materials_specular texture

    // Tier 2 glTF Extensions (offsets 144-192)
    float anisotropyStrength = 0.0f;           // 4 bytes (offset 144) - KHR_materials_anisotropy strength [0, 1]
    float anisotropyRotation = 0.0f;           // 4 bytes (offset 148) - KHR_materials_anisotropy rotation angle
    uint32_t anisotropyTex = 0;                // 4 bytes (offset 152) - anisotropy texture (RG = dir, B = strength)
    float dispersion = 0.0f;                   // 4 bytes (offset 156) - KHR_materials_dispersion (20/V_d)
    glm::vec3 sheenColor = glm::vec3(0.0f);    // 12 bytes (offset 160) - KHR_materials_sheen colorFactor
    float sheenRoughness = 0.0f;               // 4 bytes (offset 172) - KHR_materials_sheen roughnessFactor
    float iridescence = 0.0f;                  // 4 bytes (offset 176) - KHR_materials_iridescence factor
    float iridescenceIor = 1.3f;               // 4 bytes (offset 180) - KHR_materials_iridescence IOR
    float iridescenceThickness = 0.0f;         // 4 bytes (offset 184) - KHR_materials_iridescence thickness
    uint32_t sheenTex = 0;                     // 4 bytes (offset 188) - KHR_materials_sheen texture
};
static_assert(sizeof(MaterialGPU) == 192, "MaterialGPU must be exactly 192 bytes (std430 aligned)");

} // namespace pathways
