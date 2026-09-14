#pragma once

#include <string>
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include "scene/ProceduralScene.hpp"

namespace pathways {

class UsdLoader {
public:
    static bool isUsdFile(const std::string& filepath);
    static SceneData loadSceneData(const std::string& filepath);
    static bool populateMetadata(const std::string& filepath, uint64_t& outTriangles, uint32_t& outMaterials);
};

} // namespace pathways
