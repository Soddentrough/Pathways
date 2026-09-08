#pragma once

#include <string>
#include <vector>

#include <cstdint>

namespace pathways {

struct SceneEntry {
    std::string label;
    std::string filepath;
    std::string group; // "Procedural", "Showcase", "Research", "Custom"
    uint64_t triangleCount = 0;
    uint32_t materialCount = 0;
    uint64_t fileSizeBytes = 0;

    std::string formatTriangles() const;
    std::string formatFileSize() const;
    std::string formatComboPreview() const;
};

class SceneRegistry {
public:
    static std::vector<SceneEntry> scan(const std::string& scenesDir = "scenes");
    static std::string formatSceneName(const std::string& rawName);
};

} // namespace pathways
