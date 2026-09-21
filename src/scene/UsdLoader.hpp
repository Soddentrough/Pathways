#pragma once

#include <string>
#include <vector>
#include <memory>
#include <optional>
#include <glm/glm.hpp>
#include "scene/ProceduralScene.hpp"

namespace pathways {

struct UsdLoadOptions {
    float instanceDensity = 1.0f; // Scale factor for point instancing (0.0 to 1.0, default: 1.0)
    float cullDistance = 0.0f;    // Max distance in meters from camera to cull instances (0 = disabled)
    std::optional<glm::vec3> cameraPosOverride;
    float viewportAspect = 16.0f / 9.0f; // Viewport aspect ratio for fallback camera frustum fitting
};

class UsdLoader {
public:
    static bool isUsdFile(const std::string& filepath);
    static SceneData loadSceneData(const std::string& filepath, const UsdLoadOptions& options = {});
    static bool populateMetadata(const std::string& filepath, uint64_t& outTriangles, uint32_t& outMaterials);
};

} // namespace pathways
