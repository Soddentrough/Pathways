#pragma once

#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>
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

enum MaterialFlags : uint32_t {
    MATERIAL_FLAG_NONE               = 0,
    MATERIAL_FLAG_PROCEDURAL_TERRAIN = (1u << 9),
    MATERIAL_FLAG_PROCEDURAL_WATER   = (1u << 11),
    MATERIAL_FLAG_PROCEDURAL_PUDDLE  = (1u << 12),
    MATERIAL_FLAG_PROCEDURAL_HOLO    = (1u << 13),
    MATERIAL_FLAG_HOLO_VIDEO         = (1u << 14)
};

inline constexpr uint32_t operator|(MaterialType a, MaterialFlags b) {
    return static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
}
inline constexpr uint32_t operator|(MaterialFlags a, MaterialType b) {
    return static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
}
inline constexpr uint32_t operator|(MaterialFlags a, MaterialFlags b) {
    return static_cast<uint32_t>(a) | static_cast<uint32_t>(b);
}

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

    // Thin-Walled Diffuse Transmission (offsets 192-208)
    float diffuseTransmission = 0.0f;          // 4 bytes (offset 192) - diffuse transmission factor [0, 1]
    uint32_t diffuseTransmissionTex = 0;       // 4 bytes (offset 196) - diffuse transmission texture (R)
    glm::vec2 diffuseTransPad = glm::vec2(0.0f); // 8 bytes (offsets 200, 204) - padding for 16B std430 alignment
};
static_assert(sizeof(MaterialGPU) == 208, "MaterialGPU must be exactly 208 bytes (std430 aligned)");

// Compact 64-byte cache-line aligned shading material (2 materials per 128B RDNA 4 cache line)
struct alignas(16) ShadeMaterialGPU {
    glm::vec4 albedo = glm::vec4(1.0f);        // 16 bytes: baseColorFactor (linear RGBA)
    glm::vec4 emissive_spec = glm::vec4(0.0f); // 16 bytes: xyz: emissive (linear RGB), w: specularFactor
    glm::vec4 pbrParams = glm::vec4(1.0f, 0.0f, 1.5f, 0.0f); // 16 bytes: x: roughness, y: metallic, z: ior, w: diffuseTransmission
    glm::uvec4 tex_flags = glm::uvec4(0);      // 16 bytes: x: albedoTex | (normalTex << 16)
                                               //           y: mrTex | (emissiveTex << 16)
                                               //           z: diffTransTex | (specularTex << 16)
                                               //           w: (type & 0xFFFF) | (packHalf16(normalScale) << 16)
};
static_assert(sizeof(ShadeMaterialGPU) == 64, "ShadeMaterialGPU must be exactly 64 bytes (std430 aligned)");

inline ShadeMaterialGPU createShadeMaterial(const MaterialGPU& mat) {
    ShadeMaterialGPU sm{};
    sm.albedo = mat.albedo;
    sm.emissive_spec = glm::vec4(glm::vec3(mat.emissive), mat.specularFactor);
    sm.pbrParams = glm::vec4(mat.roughness, mat.metallic, mat.ior, mat.diffuseTransmission);

    uint32_t alb = (mat.albedoTex <= 512u) ? mat.albedoTex : 0u;
    uint32_t nrm = (mat.normalTex <= 512u) ? mat.normalTex : 0u;
    uint32_t mr  = (mat.mrTex <= 512u) ? mat.mrTex : 0u;
    uint32_t em  = (mat.emissiveTex <= 512u) ? mat.emissiveTex : 0u;
    uint32_t dt  = (mat.diffuseTransmissionTex <= 512u) ? mat.diffuseTransmissionTex : 0u;
    uint32_t sp  = (mat.specularTex <= 512u) ? mat.specularTex : 0u;
    uint32_t tp  = mat.type & 0xFFFFu;
    uint32_t nScaleHalf = glm::packHalf2x16(glm::vec2(mat.normalScale, 0.0f)) & 0xFFFFu;

    sm.tex_flags.x = alb | (nrm << 16u);
    sm.tex_flags.y = mr | (em << 16u);
    sm.tex_flags.z = dt | (sp << 16u);
    sm.tex_flags.w = tp | (nScaleHalf << 16u);
    return sm;
}

enum MaterialArchetype : uint32_t {
    MATERIAL_ARCHETYPE_DIFFUSE    = 0,
    MATERIAL_ARCHETYPE_DIELECTRIC = 1,
    MATERIAL_ARCHETYPE_CONDUCTOR  = 2,
    MATERIAL_ARCHETYPE_COMPLEX    = 3,
    MATERIAL_ARCHETYPE_EMISSIVE   = 4,
    MATERIAL_ARCHETYPE_ALPHAMASK  = 5,
    NUM_MATERIAL_ARCHETYPES       = 6
};

inline uint32_t computeMaterialArchetype(const MaterialGPU& mat) {
    // 1. Alpha cutout passthrough: only true alpha-masked surfaces with textures
    if (mat.alphaMode == ALPHA_MODE_MASK && mat.albedoTex > 0u) {
        return MATERIAL_ARCHETYPE_ALPHAMASK;
    }
    // 2. Pure emissive mesh lights: pure emitters without scattering BSDF
    if ((mat.type & 0xFFu) == MATERIAL_EMISSIVE ||
        (glm::length(glm::vec3(mat.emissive)) > 0.1f && mat.albedoTex == 0u && glm::length(glm::vec3(mat.albedo)) < 0.05f && mat.metallic < 0.01f && mat.transmission < 0.01f)) {
        return MATERIAL_ARCHETYPE_EMISSIVE;
    }
    // 3. Procedural wet pavement & puddle surfaces (evaluated in diffuse microkernel)
    if ((mat.type & MATERIAL_FLAG_PROCEDURAL_PUDDLE) != 0u) {
        return MATERIAL_ARCHETYPE_DIFFUSE;
    }
    // 4. Multi-layer complex PBR (Clearcoat on top of substrate, or Sheen)
    if (mat.clearcoat > 0.001f || mat.clearcoatTex > 0u || glm::length(mat.sheenColor) > 0.001f || mat.sheenTex > 0u) {
        return MATERIAL_ARCHETYPE_COMPLEX;
    }
    // 5. Pure dielectric transmission / refraction / glass / dispersion / procedural water
    if (mat.transmission > 0.001f || (mat.type & 0xFFu) == MATERIAL_DIELECTRIC || (mat.type & MATERIAL_FLAG_PROCEDURAL_WATER) != 0u || mat.dispersion > 0.001f) {
        return MATERIAL_ARCHETYPE_DIELECTRIC;
    }
    // 6. Metallic conductors (GGX microfacet specular reflection, anisotropy, iridescence)
    if ((mat.type & 0xFFu) == MATERIAL_METALLIC || mat.metallic > 0.5f || mat.anisotropyStrength > 0.001f || mat.iridescence > 0.001f) {
        return MATERIAL_ARCHETYPE_CONDUCTOR;
    }
    // 7. Dielectric diffuse base + GGX specular dual-lobe PBR (plastics, wood, stone, cloth)
    return MATERIAL_ARCHETYPE_DIFFUSE;
}

} // namespace pathways
