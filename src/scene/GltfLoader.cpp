#include "scene/GltfLoader.hpp"
#include "core/Logger.hpp"

#define CGLTF_IMPLEMENTATION
#include "cgltf.h"
#include "stb_image.h"

#include <algorithm>
#include <glm/gtc/type_ptr.hpp>
#include <filesystem>
#include <cstring>

namespace pathways {

bool GltfLoader::load(const std::string& filepath, GltfScene& outScene) {
    if (!std::filesystem::exists(filepath)) {
        Logger::error("glTF file not found: {}", filepath);
        return false;
    }

    Logger::info("Loading glTF scene: {}", filepath);

    cgltf_options options{};
    cgltf_data* data = nullptr;
    cgltf_result result = cgltf_parse_file(&options, filepath.c_str(), &data);
    if (result != cgltf_result_success) {
        Logger::error("Failed to parse glTF file: {} (code: {})", filepath, (int)result);
        return false;
    }

    result = cgltf_load_buffers(&options, data, filepath.c_str());
    if (result != cgltf_result_success) {
        Logger::error("Failed to load glTF buffers: {} (code: {})", filepath, (int)result);
        cgltf_free(data);
        return false;
    }

    outScene.assetName = std::filesystem::path(filepath).stem().string();

    // Determine sRGB status for textures (baseColor and emissive textures are sRGB in glTF 2.0)
    std::vector<bool> textureIsSrgb(data->textures_count, false);
    for (size_t i = 0; i < data->materials_count; ++i) {
        const auto& mat = data->materials[i];
        if (mat.has_pbr_metallic_roughness && mat.pbr_metallic_roughness.base_color_texture.texture) {
            size_t idx = cgltf_texture_index(data, mat.pbr_metallic_roughness.base_color_texture.texture);
            if (idx < textureIsSrgb.size()) textureIsSrgb[idx] = true;
        }
        if (mat.has_pbr_specular_glossiness && mat.pbr_specular_glossiness.diffuse_texture.texture) {
            size_t idx = cgltf_texture_index(data, mat.pbr_specular_glossiness.diffuse_texture.texture);
            if (idx < textureIsSrgb.size()) textureIsSrgb[idx] = true;
        }
        if (mat.emissive_texture.texture) {
            size_t idx = cgltf_texture_index(data, mat.emissive_texture.texture);
            if (idx < textureIsSrgb.size()) textureIsSrgb[idx] = true;
        }
        if (mat.has_specular && mat.specular.specular_color_texture.texture) {
            size_t idx = cgltf_texture_index(data, mat.specular.specular_color_texture.texture);
            if (idx < textureIsSrgb.size()) textureIsSrgb[idx] = true;
        }
    }

    // 0. Parse Images and Textures
    std::filesystem::path sceneDir = std::filesystem::path(filepath).parent_path();
    for (size_t t = 0; t < data->textures_count; ++t) {
        const auto& tex = data->textures[t];
        TextureData texData{};
        texData.isSrgb = (t < textureIsSrgb.size()) ? textureIsSrgb[t] : false;
        if (tex.image) {
            int w = 0, h = 0, comp = 0;
            stbi_uc* rawPixels = nullptr;
            if (tex.image->buffer_view) {
                const uint8_t* bufPtr = cgltf_buffer_view_data(tex.image->buffer_view);
                if (bufPtr) {
                    rawPixels = stbi_load_from_memory(bufPtr, static_cast<int>(tex.image->buffer_view->size), &w, &h, &comp, 4);
                }
            } else if (tex.image->uri) {
                std::filesystem::path imagePath = sceneDir / tex.image->uri;
                rawPixels = stbi_load(imagePath.string().c_str(), &w, &h, &comp, 4);
            }

            if (rawPixels && w > 0 && h > 0) {
                texData.width = static_cast<uint32_t>(w);
                texData.height = static_cast<uint32_t>(h);
                texData.pixels.assign(rawPixels, rawPixels + (w * h * 4));
                stbi_image_free(rawPixels);
            }
        }
        outScene.textures.push_back(std::move(texData));
    }

    // 1. Parse Materials
    for (size_t i = 0; i < data->materials_count; ++i) {
        const auto& mat = data->materials[i];
        MaterialGPU gpuMat{};
        gpuMat.albedo = glm::vec4(1.0f);
        gpuMat.emissive = glm::vec4(0.0f);
        gpuMat.roughness = 1.0f;
        gpuMat.metallic = 0.0f;
        gpuMat.ior = 1.5f;
        gpuMat.transmission = 0.0f;
        gpuMat.type = MATERIAL_DIFFUSE;
        gpuMat.albedoTex = 0;
        gpuMat.normalTex = 0;
        gpuMat.mrTex = 0;
        gpuMat.emissiveTex = 0;
        gpuMat.occlusionTex = 0;
        gpuMat.transmissionTex = 0;
        gpuMat.alphaCutoff = mat.alpha_cutoff > 0.0f ? mat.alpha_cutoff : 0.5f;
        gpuMat.alphaMode = static_cast<uint32_t>(mat.alpha_mode);
        gpuMat.occlusionStrength = 1.0f;
        gpuMat.normalScale = 1.0f;
        gpuMat.thicknessTex = 0;

        if (mat.has_pbr_metallic_roughness) {
            const auto& pbr = mat.pbr_metallic_roughness;
            gpuMat.albedo = glm::make_vec4(pbr.base_color_factor);
            gpuMat.metallic = pbr.metallic_factor;
            gpuMat.roughness = pbr.roughness_factor;
            if (gpuMat.metallic > 0.5f) {
                gpuMat.type = MATERIAL_METALLIC;
            }
            if (pbr.base_color_texture.texture) {
                gpuMat.albedoTex = static_cast<uint32_t>(cgltf_texture_index(data, pbr.base_color_texture.texture)) + 1;
            }
            if (pbr.metallic_roughness_texture.texture) {
                gpuMat.mrTex = static_cast<uint32_t>(cgltf_texture_index(data, pbr.metallic_roughness_texture.texture)) + 1;
            }
        } else if (mat.has_pbr_specular_glossiness) {
            const auto& pbr = mat.pbr_specular_glossiness;
            gpuMat.albedo = glm::make_vec4(pbr.diffuse_factor);
            gpuMat.roughness = 1.0f - pbr.glossiness_factor;
            float specLum = 0.299f * pbr.specular_factor[0] + 0.587f * pbr.specular_factor[1] + 0.114f * pbr.specular_factor[2];
            gpuMat.metallic = specLum;
            if (gpuMat.metallic > 0.5f) {
                gpuMat.type = MATERIAL_METALLIC;
            }
            if (pbr.diffuse_texture.texture) {
                gpuMat.albedoTex = static_cast<uint32_t>(cgltf_texture_index(data, pbr.diffuse_texture.texture)) + 1;
            }
        }

        if (mat.normal_texture.texture) {
            gpuMat.normalTex = static_cast<uint32_t>(cgltf_texture_index(data, mat.normal_texture.texture)) + 1;
            if (mat.normal_texture.scale != 0.0f) {
                gpuMat.normalScale = mat.normal_texture.scale;
            }
        }

        if (mat.occlusion_texture.texture) {
            gpuMat.occlusionTex = static_cast<uint32_t>(cgltf_texture_index(data, mat.occlusion_texture.texture)) + 1;
            if (mat.occlusion_texture.scale != 0.0f) {
                gpuMat.occlusionStrength = mat.occlusion_texture.scale;
            }
        }

        if (mat.emissive_texture.texture) {
            gpuMat.emissiveTex = static_cast<uint32_t>(cgltf_texture_index(data, mat.emissive_texture.texture)) + 1;
        }

        if (mat.has_ior) {
            gpuMat.ior = mat.ior.ior;
        }

        if (mat.has_transmission) {
            gpuMat.transmission = mat.transmission.transmission_factor;
            if (mat.transmission.transmission_texture.texture) {
                gpuMat.transmissionTex = static_cast<uint32_t>(cgltf_texture_index(data, mat.transmission.transmission_texture.texture)) + 1;
            }
            if (gpuMat.transmission > 0.05f) {
                gpuMat.type = MATERIAL_DIELECTRIC;
            }
        }

        // Check PBRT extras if available from scene conversion
        if (mat.extras.data && mat.extras.data[0] != '\0') {
            std::string_view extrasStr(mat.extras.data);
            if (extrasStr.find("\"dielectric\"") != std::string_view::npos ||
                extrasStr.find("\"glass\"") != std::string_view::npos) {
                gpuMat.type = MATERIAL_DIELECTRIC;
                gpuMat.transmission = 1.0f;
                if (gpuMat.ior < 1.05f) gpuMat.ior = 1.5f;
            } else if (extrasStr.find("\"conductor\"") != std::string_view::npos) {
                gpuMat.type = MATERIAL_METALLIC;
                gpuMat.metallic = 1.0f;
            } else if (extrasStr.find("\"coateddiffuse\"") != std::string_view::npos) {
                gpuMat.clearcoat = 1.0f;
            }
        }

        if (mat.has_emissive_strength) {
            float strength = mat.emissive_strength.emissive_strength;
            gpuMat.emissive = glm::vec4(
                mat.emissive_factor[0] * strength,
                mat.emissive_factor[1] * strength,
                mat.emissive_factor[2] * strength,
                1.0f
            );
        } else {
            gpuMat.emissive = glm::vec4(
                mat.emissive_factor[0],
                mat.emissive_factor[1],
                mat.emissive_factor[2],
                1.0f
            );
        }

        if (mat.has_clearcoat) {
            gpuMat.clearcoat = mat.clearcoat.clearcoat_factor;
            gpuMat.clearcoatRoughness = mat.clearcoat.clearcoat_roughness_factor;
            if (mat.clearcoat.clearcoat_texture.texture) {
                gpuMat.clearcoatTex = static_cast<uint32_t>(cgltf_texture_index(data, mat.clearcoat.clearcoat_texture.texture)) + 1;
            }
            if (mat.clearcoat.clearcoat_roughness_texture.texture) {
                gpuMat.clearcoatRoughnessTex = static_cast<uint32_t>(cgltf_texture_index(data, mat.clearcoat.clearcoat_roughness_texture.texture)) + 1;
            }
            if (mat.clearcoat.clearcoat_normal_texture.texture) {
                gpuMat.clearcoatNormalTex = static_cast<uint32_t>(cgltf_texture_index(data, mat.clearcoat.clearcoat_normal_texture.texture)) + 1;
            }
        }

        if (mat.has_volume) {
            gpuMat.thickness = mat.volume.thickness_factor;
            gpuMat.attenuationColor = glm::vec4(
                mat.volume.attenuation_color[0],
                mat.volume.attenuation_color[1],
                mat.volume.attenuation_color[2],
                mat.volume.attenuation_distance
            );
            if (mat.volume.thickness_texture.texture) {
                gpuMat.thicknessTex = static_cast<uint32_t>(cgltf_texture_index(data, mat.volume.thickness_texture.texture)) + 1;
            }
        }

        if (mat.has_specular) {
            gpuMat.specularFactor = mat.specular.specular_factor;
            if (mat.specular.specular_texture.texture) {
                gpuMat.specularTex = static_cast<uint32_t>(cgltf_texture_index(data, mat.specular.specular_texture.texture)) + 1;
                // If scene lacked a separate metallicRoughnessTexture, specularTexture contains
                // Green = Roughness, Blue = Metallic (standard for Lumberyard / CryEngine specular export)
                if (gpuMat.mrTex == 0 && gpuMat.specularTex > 0) {
                    gpuMat.mrTex = gpuMat.specularTex;
                    size_t texIdx = gpuMat.specularTex - 1;
                    if (texIdx < outScene.textures.size() && !outScene.textures[texIdx].pixels.empty()) {
                        const auto& px = outScene.textures[texIdx].pixels;
                        uint8_t metalVal = (px.size() >= 3) ? px[2] : 0;
                        uint8_t roughVal = (px.size() >= 2) ? px[1] : 189;
                        gpuMat.roughness = static_cast<float>(roughVal) / 255.0f;
                        if (metalVal >= 128) {
                            gpuMat.metallic = static_cast<float>(metalVal) / 255.0f;
                            gpuMat.type = MATERIAL_METALLIC;
                        } else {
                            gpuMat.metallic = 0.0f;
                            if (gpuMat.transmission <= 0.05f) {
                                gpuMat.type = MATERIAL_DIFFUSE;
                            }
                        }
                    } else {
                        gpuMat.metallic = 0.0f;
                        if (gpuMat.transmission <= 0.05f) {
                            gpuMat.type = MATERIAL_DIFFUSE;
                        }
                    }
                }
            }
        }

        if (mat.has_anisotropy) {
            gpuMat.anisotropyStrength = mat.anisotropy.anisotropy_strength;
            gpuMat.anisotropyRotation = mat.anisotropy.anisotropy_rotation;
            if (mat.anisotropy.anisotropy_texture.texture) {
                gpuMat.anisotropyTex = static_cast<uint32_t>(cgltf_texture_index(data, mat.anisotropy.anisotropy_texture.texture)) + 1;
            }
        }

        if (mat.has_dispersion) {
            gpuMat.dispersion = mat.dispersion.dispersion;
        }

        if (mat.has_sheen) {
            gpuMat.sheenColor = glm::vec3(
                mat.sheen.sheen_color_factor[0],
                mat.sheen.sheen_color_factor[1],
                mat.sheen.sheen_color_factor[2]
            );
            gpuMat.sheenRoughness = mat.sheen.sheen_roughness_factor;
            if (mat.sheen.sheen_color_texture.texture) {
                gpuMat.sheenTex = static_cast<uint32_t>(cgltf_texture_index(data, mat.sheen.sheen_color_texture.texture)) + 1;
            }
        }

        if (mat.has_iridescence) {
            gpuMat.iridescence = mat.iridescence.iridescence_factor;
            gpuMat.iridescenceIor = mat.iridescence.iridescence_ior;
            gpuMat.iridescenceThickness = mat.iridescence.iridescence_thickness_max;
        }

        bool hasTextures = (gpuMat.albedoTex > 0 || gpuMat.mrTex > 0 || gpuMat.normalTex > 0 ||
                            gpuMat.occlusionTex > 0 || gpuMat.emissiveTex > 0 || gpuMat.transmissionTex > 0 ||
                            gpuMat.clearcoatTex > 0 || gpuMat.clearcoatRoughnessTex > 0 || gpuMat.clearcoatNormalTex > 0 ||
                            gpuMat.thicknessTex > 0 || gpuMat.specularTex > 0 ||
                            gpuMat.anisotropyTex > 0 || gpuMat.sheenTex > 0);
        if (!hasTextures && (gpuMat.emissive.r > 0.1f || gpuMat.emissive.g > 0.1f || gpuMat.emissive.b > 0.1f)) {
            gpuMat.type = MATERIAL_EMISSIVE;
        }

        outScene.materials.push_back(gpuMat);
    }

    if (outScene.materials.empty()) {
        // Default white diffuse material
        MaterialGPU defMat{};
        defMat.albedo = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
        defMat.roughness = 0.8f;
        defMat.metallic = 0.0f;
        defMat.type = MATERIAL_DIFFUSE;
        outScene.materials.push_back(defMat);
    }

    // 2. Parse Meshes
    for (size_t m = 0; m < data->meshes_count; ++m) {
        const auto& mesh = data->meshes[m];
        GltfMesh outMesh;
        outMesh.name = mesh.name ? mesh.name : ("Mesh_" + std::to_string(m));

        for (size_t p = 0; p < mesh.primitives_count; ++p) {
            const auto& prim = mesh.primitives[p];
            if (prim.type != cgltf_primitive_type_triangles) continue;

            GltfMeshPrimitive outPrim;
            if (prim.material) {
                outPrim.materialIndex = static_cast<uint32_t>(prim.material - data->materials);
            } else {
                outPrim.materialIndex = 0;
            }

            const cgltf_accessor* posAccessor = nullptr;
            const cgltf_accessor* normAccessor = nullptr;
            const cgltf_accessor* texAccessor = nullptr;
            const cgltf_accessor* tanAccessor = nullptr;

            for (size_t a = 0; a < prim.attributes_count; ++a) {
                const auto& attr = prim.attributes[a];
                if (attr.type == cgltf_attribute_type_position) posAccessor = attr.data;
                else if (attr.type == cgltf_attribute_type_normal) normAccessor = attr.data;
                else if (attr.type == cgltf_attribute_type_texcoord && attr.index == 0) texAccessor = attr.data;
                else if (attr.type == cgltf_attribute_type_tangent) tanAccessor = attr.data;
            }

            if (!posAccessor) continue;

            size_t vertexCount = posAccessor->count;
            outPrim.vertices.resize(vertexCount);

            for (size_t v = 0; v < vertexCount; ++v) {
                float pos[3] = {0, 0, 0};
                cgltf_accessor_read_float(posAccessor, v, pos, 3);
                outPrim.vertices[v].position = glm::vec4(pos[0], pos[1], pos[2], 0.0f);

                if (normAccessor) {
                    float norm[3] = {0, 1, 0};
                    cgltf_accessor_read_float(normAccessor, v, norm, 3);
                    outPrim.vertices[v].normal = glm::vec4(norm[0], norm[1], norm[2], 0.0f);
                } else {
                    outPrim.vertices[v].normal = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f);
                }

                if (texAccessor) {
                    float uv[2] = {0, 0};
                    cgltf_accessor_read_float(texAccessor, v, uv, 2);
                    outPrim.vertices[v].position.w = uv[0];
                    outPrim.vertices[v].normal.w = uv[1];
                }

                if (tanAccessor) {
                    float tan[4] = {1, 0, 0, 1};
                    cgltf_accessor_read_float(tanAccessor, v, tan, 4);
                    outPrim.vertices[v].tangent = glm::vec4(tan[0], tan[1], tan[2], tan[3]);
                }
            }

            // Indices
            if (prim.indices) {
                outPrim.indices.resize(prim.indices->count);
                for (size_t idx = 0; idx < prim.indices->count; ++idx) {
                    outPrim.indices[idx] = static_cast<uint32_t>(cgltf_accessor_read_index(prim.indices, idx));
                }
            } else {
                outPrim.indices.resize(vertexCount);
                for (size_t idx = 0; idx < vertexCount; ++idx) {
                    outPrim.indices[idx] = static_cast<uint32_t>(idx);
                }
            }

            outMesh.primitives.push_back(std::move(outPrim));
        }

        outScene.meshes.push_back(std::move(outMesh));
    }

    // 3. Parse Nodes & Hierarchy
    for (size_t n = 0; n < data->nodes_count; ++n) {
        const auto& node = data->nodes[n];
        GltfNode outNode;
        outNode.name = node.name ? node.name : ("Node_" + std::to_string(n));

        if (node.mesh) {
            outNode.meshIndex = static_cast<int32_t>(node.mesh - data->meshes);
        }

        float worldMat[16];
        cgltf_node_transform_world(&node, worldMat);
        outNode.worldTransform = glm::make_mat4(worldMat);

        for (size_t c = 0; c < node.children_count; ++c) {
            outNode.children.push_back(static_cast<uint32_t>(node.children[c] - data->nodes));
        }

        outScene.nodes.push_back(std::move(outNode));
    }

    // 4. Parse Punctual Lights (KHR_lights_punctual)
    for (size_t l = 0; l < data->lights_count; ++l) {
        const auto& light = data->lights[l];
        LightGPU gpuLight{};
        gpuLight.emission = glm::vec4(
            light.color[0] * light.intensity,
            light.color[1] * light.intensity,
            light.color[2] * light.intensity,
            1.0f
        );

        if (light.type == cgltf_light_type_spot) {
            gpuLight.position.w = LIGHT_SPOT;
            gpuLight.u.w = std::cos(light.spot_inner_cone_angle);
            gpuLight.v.w = std::cos(light.spot_outer_cone_angle);
        } else if (light.type == cgltf_light_type_directional) {
            gpuLight.position.w = LIGHT_DIRECTIONAL;
        } else {
            gpuLight.position.w = LIGHT_AREA_QUAD;
        }

        outScene.lights.push_back(gpuLight);
    }

    // 5. Parse Cameras
    for (size_t n = 0; n < data->nodes_count; ++n) {
        const auto& node = data->nodes[n];
        if (node.camera) {
            float worldMat[16];
            cgltf_node_transform_world(&node, worldMat);
            glm::mat4 M = glm::make_mat4(worldMat);
            glm::vec3 pos = glm::vec3(M[3]);
            glm::vec3 forward = -glm::normalize(glm::vec3(M[2]));
            float fov = 45.0f;
            if (node.camera->type == cgltf_camera_type_perspective && node.camera->data.perspective.yfov > 0.0f) {
                fov = glm::degrees(node.camera->data.perspective.yfov);
            }
            outScene.cameras.emplace_back(pos, pos + forward * 5.0f, fov);
        }
    }

    Logger::info("glTF loaded: {} meshes, {} nodes, {} materials, {} lights, {} cameras",
                 outScene.meshes.size(), outScene.nodes.size(), outScene.materials.size(),
                 outScene.lights.size(), outScene.cameras.size());

    cgltf_free(data);
    return true;
}

SceneData GltfLoader::loadSceneData(const std::string& filepath) {
    GltfScene gltfScene;
    if (!load(filepath, gltfScene)) {
        Logger::warn("GltfLoader: Failed to load '{}', falling back to Cornell Box.", filepath);
        return ProceduralScene::createCornellBox();
    }

    SceneData data;
    data.materials = gltfScene.materials;
    if (data.materials.empty()) {
        MaterialGPU defaultMat{};
        defaultMat.albedo = glm::vec4(0.8f, 0.8f, 0.8f, 1.0f);
        defaultMat.roughness = 0.5f;
        defaultMat.metallic = 0.0f;
        defaultMat.type = MATERIAL_DIFFUSE;
        data.materials.push_back(defaultMat);
    }

    data.lights = gltfScene.lights;
    data.textures = std::move(gltfScene.textures);

    struct MeshInfo {
        std::string name;
        glm::vec3 minBound{1e30f};
        glm::vec3 maxBound{-1e30f};
        uint32_t firstTriangle = 0;
        uint32_t triCount = 0;
    };
    std::vector<MeshInfo> meshInfos;

    for (const auto& node : gltfScene.nodes) {
        if (node.meshIndex < 0 || static_cast<size_t>(node.meshIndex) >= gltfScene.meshes.size()) {
            continue;
        }

        const auto& mesh = gltfScene.meshes[node.meshIndex];
        MeshInfo mInfo;
        mInfo.name = mesh.name;
        mInfo.firstTriangle = static_cast<uint32_t>(data.triangles.size());
        const glm::mat4& M = node.worldTransform;
        glm::mat3 normalMatrix = glm::transpose(glm::inverse(glm::mat3(M)));

        for (const auto& prim : mesh.primitives) {
            uint32_t matId = prim.materialIndex;
            if (matId >= data.materials.size()) matId = 0;

            auto computeTangents = [&](TriangleGPU& tri, const GltfVertex& v0_in, const GltfVertex& v1_in, const GltfVertex& v2_in, const glm::vec3& geoNormal) {
                glm::vec3 p0 = glm::vec3(tri.v0.position);
                glm::vec3 p1 = glm::vec3(tri.v1.position);
                glm::vec3 p2 = glm::vec3(tri.v2.position);
                glm::vec3 e1 = p1 - p0;
                glm::vec3 e2 = p2 - p0;
                glm::vec2 uv0 = glm::vec2(v0_in.position.w, v0_in.normal.w);
                glm::vec2 uv1 = glm::vec2(v1_in.position.w, v1_in.normal.w);
                glm::vec2 uv2 = glm::vec2(v2_in.position.w, v2_in.normal.w);
                glm::vec2 duv1 = uv1 - uv0;
                glm::vec2 duv2 = uv2 - uv0;
                float det = duv1.x * duv2.y - duv2.x * duv1.y;
                glm::vec3 defaultTan;
                if (std::abs(det) > 1e-6f) {
                    defaultTan = glm::normalize((e1 * duv2.y - e2 * duv1.y) / det);
                } else {
                    glm::vec3 up = std::abs(geoNormal.z) < 0.999f ? glm::vec3(0.0f, 0.0f, 1.0f) : glm::vec3(1.0f, 0.0f, 0.0f);
                    defaultTan = glm::normalize(glm::cross(up, geoNormal));
                }

                auto getVertTangent = [&](const GltfVertex& vert, const glm::vec3& norm) -> glm::vec4 {
                    glm::vec3 t = glm::vec3(vert.tangent);
                    if (glm::length(t) > 1e-4f) {
                        glm::vec3 worldT = glm::normalize(glm::mat3(M) * t);
                        worldT = glm::normalize(worldT - norm * glm::dot(worldT, norm));
                        float sign = vert.tangent.w != 0.0f ? vert.tangent.w : 1.0f;
                        return glm::vec4(worldT, sign);
                    } else {
                        glm::vec3 worldT = glm::normalize(defaultTan - norm * glm::dot(defaultTan, norm));
                        return glm::vec4(worldT, 1.0f);
                    }
                };

                tri.v0.tangent = getVertTangent(v0_in, glm::vec3(tri.v0.normal));
                tri.v1.tangent = getVertTangent(v1_in, glm::vec3(tri.v1.normal));
                tri.v2.tangent = getVertTangent(v2_in, glm::vec3(tri.v2.normal));
            };

            if (!prim.indices.empty()) {
                for (size_t i = 0; i + 2 < prim.indices.size(); i += 3) {
                    const auto& v0_in = prim.vertices[prim.indices[i]];
                    const auto& v1_in = prim.vertices[prim.indices[i + 1]];
                    const auto& v2_in = prim.vertices[prim.indices[i + 2]];

                    TriangleGPU tri{};
                    glm::vec4 p0 = M * glm::vec4(glm::vec3(v0_in.position), 1.0f);
                    glm::vec4 p1 = M * glm::vec4(glm::vec3(v1_in.position), 1.0f);
                    glm::vec4 p2 = M * glm::vec4(glm::vec3(v2_in.position), 1.0f);

                    tri.v0.position = glm::vec4(glm::vec3(p0), v0_in.position.w);
                    tri.v1.position = glm::vec4(glm::vec3(p1), v1_in.position.w);
                    tri.v2.position = glm::vec4(glm::vec3(p2), v2_in.position.w);

                    glm::vec3 p0_3 = glm::vec3(p0);
                    glm::vec3 p1_3 = glm::vec3(p1);
                    glm::vec3 p2_3 = glm::vec3(p2);
                    glm::vec3 geoNormal = glm::cross(p1_3 - p0_3, p2_3 - p0_3);
                    if (glm::length(geoNormal) > 1e-7f) {
                        geoNormal = glm::normalize(geoNormal);
                    } else {
                        geoNormal = glm::vec3(0.0f, 1.0f, 0.0f);
                    }

                    glm::vec3 n0 = glm::vec3(v0_in.normal);
                    glm::vec3 n1 = glm::vec3(v1_in.normal);
                    glm::vec3 n2 = glm::vec3(v2_in.normal);
                    tri.v0.normal = glm::vec4(glm::length(n0) < 1e-4f ? geoNormal : glm::normalize(normalMatrix * n0), v0_in.normal.w);
                    tri.v1.normal = glm::vec4(glm::length(n1) < 1e-4f ? geoNormal : glm::normalize(normalMatrix * n1), v1_in.normal.w);
                    tri.v2.normal = glm::vec4(glm::length(n2) < 1e-4f ? geoNormal : glm::normalize(normalMatrix * n2), v2_in.normal.w);

                    computeTangents(tri, v0_in, v1_in, v2_in, geoNormal);

                    tri.materialId = matId;
                    data.triangles.push_back(tri);

                    mInfo.minBound = glm::min(mInfo.minBound, p0_3);
                    mInfo.minBound = glm::min(mInfo.minBound, p1_3);
                    mInfo.minBound = glm::min(mInfo.minBound, p2_3);
                    mInfo.maxBound = glm::max(mInfo.maxBound, p0_3);
                    mInfo.maxBound = glm::max(mInfo.maxBound, p1_3);
                    mInfo.maxBound = glm::max(mInfo.maxBound, p2_3);
                    mInfo.triCount++;
                }
            } else {
                for (size_t i = 0; i + 2 < prim.vertices.size(); i += 3) {
                    const auto& v0_in = prim.vertices[i];
                    const auto& v1_in = prim.vertices[i + 1];
                    const auto& v2_in = prim.vertices[i + 2];

                    TriangleGPU tri{};
                    glm::vec4 p0 = M * glm::vec4(glm::vec3(v0_in.position), 1.0f);
                    glm::vec4 p1 = M * glm::vec4(glm::vec3(v1_in.position), 1.0f);
                    glm::vec4 p2 = M * glm::vec4(glm::vec3(v2_in.position), 1.0f);

                    tri.v0.position = glm::vec4(glm::vec3(p0), v0_in.position.w);
                    tri.v1.position = glm::vec4(glm::vec3(p1), v1_in.position.w);
                    tri.v2.position = glm::vec4(glm::vec3(p2), v2_in.position.w);

                    glm::vec3 p0_3 = glm::vec3(p0);
                    glm::vec3 p1_3 = glm::vec3(p1);
                    glm::vec3 p2_3 = glm::vec3(p2);
                    glm::vec3 geoNormal = glm::cross(p1_3 - p0_3, p2_3 - p0_3);
                    if (glm::length(geoNormal) > 1e-7f) {
                        geoNormal = glm::normalize(geoNormal);
                    } else {
                        geoNormal = glm::vec3(0.0f, 1.0f, 0.0f);
                    }

                    glm::vec3 n0 = glm::vec3(v0_in.normal);
                    glm::vec3 n1 = glm::vec3(v1_in.normal);
                    glm::vec3 n2 = glm::vec3(v2_in.normal);
                    tri.v0.normal = glm::vec4(glm::length(n0) < 1e-4f ? geoNormal : glm::normalize(normalMatrix * n0), v0_in.normal.w);
                    tri.v1.normal = glm::vec4(glm::length(n1) < 1e-4f ? geoNormal : glm::normalize(normalMatrix * n1), v1_in.normal.w);
                    tri.v2.normal = glm::vec4(glm::length(n2) < 1e-4f ? geoNormal : glm::normalize(normalMatrix * n2), v2_in.normal.w);

                    computeTangents(tri, v0_in, v1_in, v2_in, geoNormal);

                    tri.materialId = matId;
                    data.triangles.push_back(tri);

                    mInfo.minBound = glm::min(mInfo.minBound, p0_3);
                    mInfo.minBound = glm::min(mInfo.minBound, p1_3);
                    mInfo.minBound = glm::min(mInfo.minBound, p2_3);
                    mInfo.maxBound = glm::max(mInfo.maxBound, p0_3);
                    mInfo.maxBound = glm::max(mInfo.maxBound, p1_3);
                    mInfo.maxBound = glm::max(mInfo.maxBound, p2_3);
                    mInfo.triCount++;
                }
            }
        }

        if (mInfo.triCount > 0) {
            meshInfos.push_back(mInfo);
            MeshRange mr{};
            mr.name = mInfo.name.empty() ? ("Mesh_" + std::to_string(meshInfos.size())) : mInfo.name;
            mr.minBound = mInfo.minBound;
            mr.maxBound = mInfo.maxBound;
            mr.firstTriangle = mInfo.firstTriangle;
            mr.triangleCount = mInfo.triCount;
            data.meshRanges.push_back(mr);
        }
    }

    // Compute AABB for framing, lighting, and camera scaling
    glm::vec3 minBound(1e30f);
    glm::vec3 maxBound(-1e30f);
    for (const auto& mi : meshInfos) {
        minBound = glm::min(minBound, mi.minBound);
        maxBound = glm::max(maxBound, mi.maxBound);
    }
    if (meshInfos.empty()) {
        minBound = glm::vec3(-1.0f);
        maxBound = glm::vec3(1.0f);
    }

    glm::vec3 center = (minBound + maxBound) * 0.5f;
    glm::vec3 extent = maxBound - minBound;
    float fullDiag = glm::length(extent);
    float maxDim = std::max({extent.x, extent.y, extent.z});
    if (maxDim < 1e-4f) maxDim = 2.0f;

    // Detect outlier backdrop/ground plane meshes
    glm::vec3 focalMin(1e30f);
    glm::vec3 focalMax(-1e30f);
    size_t nonBackdropCount = 0;

    for (const auto& mi : meshInfos) {
        glm::vec3 mExt = mi.maxBound - mi.minBound;
        float mDiag = glm::length(mExt);
        bool isFlatY = mExt.y < 0.05f * std::max({mExt.x, mExt.z, 0.01f});
        bool isFlatZ = mExt.z < 0.05f * std::max({mExt.x, mExt.y, 0.01f});
        bool coversScene = (mExt.x > 0.65f * extent.x && mExt.z > 0.65f * extent.z) || (mDiag > 0.65f * fullDiag);

        std::string nameLower = mi.name;
        std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        bool isNamedGround = (nameLower.find("ground") != std::string::npos ||
                              nameLower.find("floor") != std::string::npos ||
                              nameLower.find("backdrop") != std::string::npos ||
                              nameLower.find("bounce") != std::string::npos ||
                              nameLower.find("mattefloor") != std::string::npos);

        if (meshInfos.size() > 1 && (isNamedGround || isFlatY || isFlatZ) && coversScene) {
            // Outlier studio ground / backdrop plane excluded from focal calculation
            continue;
        }

        focalMin = glm::min(focalMin, mi.minBound);
        focalMax = glm::max(focalMax, mi.maxBound);
        nonBackdropCount++;
    }

    if (nonBackdropCount == 0) {
        focalMin = minBound;
        focalMax = maxBound;
    }

    glm::vec3 focalCenter = (focalMin + focalMax) * 0.5f;
    glm::vec3 focalExtent = focalMax - focalMin;
    float focalRadius = std::max(glm::length(focalExtent) * 0.5f, 0.1f);

    // Identify central target point at (0, Y_center, 0) if within focal bounds, or focalCenter
    glm::vec3 centralTarget = focalCenter;
    if (focalMin.x <= 0.5f && focalMax.x >= -0.5f && focalMin.z <= 0.5f && focalMax.z >= -0.5f) {
        centralTarget = glm::vec3(0.0f, focalCenter.y, 0.0f);
    }

    data.boundsMin = minBound;
    data.boundsMax = maxBound;
    data.focalBoundsMin = focalMin;
    data.focalBoundsMax = focalMax;
    data.focalRadius = focalRadius;
    data.centralTarget = centralTarget;

    // Camera setup
    if (!gltfScene.cameras.empty()) {
        data.hasCamera = true;
        data.cameraPosition = gltfScene.cameras[0].getPosition();
        glm::vec3 camFront = gltfScene.cameras[0].getFront();
        glm::vec3 toCenter = centralTarget - data.cameraPosition;
        float projDist = glm::dot(toCenter, camFront);
        float focalDist = projDist > 0.2f ? projDist : glm::length(toCenter);
        if (focalDist < 0.1f) focalDist = focalRadius;
        data.focalDistance = focalDist;
        data.cameraTarget = data.cameraPosition + camFront * focalDist;
        data.cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
        data.cameraFov = gltfScene.cameras[0].getFov();
    } else if (!data.triangles.empty()) {
        float fMaxDim = std::max({focalExtent.x, focalExtent.y, focalExtent.z});
        if (fMaxDim < 1e-4f) fMaxDim = 2.0f;
        float dist = (fMaxDim * 0.5f) / std::tan(glm::radians(22.5f));
        data.hasCamera = true;
        data.cameraPosition = focalCenter + glm::vec3(0.0f, fMaxDim * 0.12f, dist * 1.3f);
        data.cameraTarget = focalCenter;
        data.focalDistance = dist * 1.3f;
        data.cameraUp = glm::vec3(0.0f, 1.0f, 0.0f);
        data.cameraFov = 45.0f;
    }

    float effectiveScale = std::clamp(focalRadius, 0.25f * data.focalDistance, 2.5f * data.focalDistance);
    data.sceneRadius = effectiveScale;

    // glTF 2.1 physical emissive mesh light extraction
    // Scan baked world-space triangles for emissive materials (KHR_materials_emissive_strength)
    // and convert them to physical area lights for direct MIS sampling
    std::vector<LightGPU> emissiveMeshLights;
    for (const auto& tri : data.triangles) {
        if (tri.materialId < data.materials.size()) {
            const auto& mat = data.materials[tri.materialId];
            glm::vec3 em = glm::vec3(mat.emissive);
            float emPower = glm::length(em);
            // Only extract as a physical area light if explicitly MATERIAL_EMISSIVE
            // or an untextured emissive source with high radiant flux
            bool isExplicitLight = (mat.type == MATERIAL_EMISSIVE) ||
                                   (emPower > 1.0f && mat.albedoTex == 0 && mat.mrTex == 0);
            if (isExplicitLight) {
                glm::vec3 p0 = glm::vec3(tri.v0.position);
                glm::vec3 p1 = glm::vec3(tri.v1.position);
                glm::vec3 p2 = glm::vec3(tri.v2.position);
                glm::vec3 u = p1 - p0;
                glm::vec3 v = p2 - p0;
                glm::vec3 n = glm::cross(u, v);
                float lenN = glm::length(n);
                if (lenN > 1e-6f) {
                    float triArea = 0.5f * lenN;
                    LightGPU light{};
                    light.position = glm::vec4(p0, LIGHT_AREA_QUAD);
                    light.u = glm::vec4(u, 0.0f);
                    light.v = glm::vec4(v, 0.0f);
                    light.normal = glm::vec4(n / lenN, 0.0f);
                    light.emission = glm::vec4(em, triArea);
                    emissiveMeshLights.push_back(light);
                }
            }
        }
    }

    if (!emissiveMeshLights.empty()) {
        Logger::info("Extracted {} physical emissive mesh lights from scene geometry", emissiveMeshLights.size());
        // If there are many emissive triangles, sort by total radiant flux and keep top 64
        if (emissiveMeshLights.size() > 64) {
            std::sort(emissiveMeshLights.begin(), emissiveMeshLights.end(), [](const LightGPU& a, const LightGPU& b) {
                float fluxA = (a.emission.r + a.emission.g + a.emission.b) * a.emission.w;
                float fluxB = (b.emission.r + b.emission.g + b.emission.b) * b.emission.w;
                return fluxA > fluxB;
            });
            emissiveMeshLights.resize(64);
        }
        for (const auto& l : emissiveMeshLights) {
            data.lights.push_back(l);
        }
    }

    // If the glTF had neither punctual lights nor physical emissive mesh lights,
    // add an overhead area light scaled to the model dimensions as fallback
    if (data.lights.empty()) {
        float lightSide = maxDim * 0.6f;
        float lightY = maxBound.y + maxDim * 0.5f;
        LightGPU defaultLight{};
        defaultLight.position = glm::vec4(center.x - lightSide * 0.5f, lightY, center.z - lightSide * 0.5f, LIGHT_AREA_QUAD);
        defaultLight.u = glm::vec4(lightSide, 0.0f, 0.0f, 0.0f);
        defaultLight.v = glm::vec4(0.0f, 0.0f, lightSide, 0.0f);
        defaultLight.normal = glm::vec4(0.0f, -1.0f, 0.0f, 0.0f);
        float area = lightSide * lightSide;
        defaultLight.emission = glm::vec4(25.0f, 25.0f, 25.0f, area);
        data.lights.push_back(defaultLight);
    }

    Logger::info("GltfLoader generated SceneData: {} Triangles, {} Spheres, {} Materials, {} Lights",
                 data.triangles.size(), data.spheres.size(), data.materials.size(), data.lights.size());

    return data;
}

} // namespace pathways
