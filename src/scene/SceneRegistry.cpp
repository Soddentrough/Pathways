#include "scene/SceneRegistry.hpp"
#include "core/Logger.hpp"
#include "cgltf.h"

#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cstdio>

namespace pathways {

std::string SceneEntry::formatTriangles() const {
    if (triangleCount == 0) return "0 tris";
    if (triangleCount >= 1'000'000) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.2fM tris", static_cast<double>(triangleCount) / 1'000'000.0);
        return buf;
    }
    if (triangleCount >= 1'000) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1fK tris", static_cast<double>(triangleCount) / 1'000.0);
        return buf;
    }
    return std::to_string(triangleCount) + " tris";
}

std::string SceneEntry::formatFileSize() const {
    if (fileSizeBytes == 0) return "";
    if (fileSizeBytes >= 1024 * 1024) {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.1f MB", static_cast<double>(fileSizeBytes) / (1024.0 * 1024.0));
        return buf;
    }
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%.0f KB", static_cast<double>(fileSizeBytes) / 1024.0);
    return buf;
}

std::string SceneEntry::formatComboPreview() const {
    std::string out = label + "  [" + formatTriangles();
    if (materialCount > 0) {
        out += ", " + std::to_string(materialCount) + (materialCount == 1 ? " mat" : " mats");
    }
    out += "]";
    return out;
}

std::string SceneRegistry::formatSceneName(const std::string& rawName) {
    std::string out;
    bool capitalizeNext = true;
    for (char c : rawName) {
        if (c == '-' || c == '_') {
            out += ' ';
            capitalizeNext = true;
        } else if (capitalizeNext) {
            out += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            capitalizeNext = false;
        } else {
            out += static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        }
    }
    return out;
}

static void populateSceneMetadata(SceneEntry& entry) {
    if (entry.filepath.empty()) return;

    std::error_code ec;
    if (std::filesystem::exists(entry.filepath, ec)) {
        entry.fileSizeBytes = std::filesystem::file_size(entry.filepath, ec);
    }

    cgltf_options options{};
    cgltf_data* data = nullptr;
    cgltf_result res = cgltf_parse_file(&options, entry.filepath.c_str(), &data);
    if (res == cgltf_result_success && data) {
        uint64_t tris = 0;
        for (cgltf_size m = 0; m < data->meshes_count; ++m) {
            const auto& mesh = data->meshes[m];
            for (cgltf_size p = 0; p < mesh.primitives_count; ++p) {
                const auto& prim = mesh.primitives[p];
                if (prim.indices) {
                    tris += prim.indices->count / 3;
                } else {
                    for (cgltf_size a = 0; a < prim.attributes_count; ++a) {
                        if (prim.attributes[a].type == cgltf_attribute_type_position && prim.attributes[a].data) {
                            tris += prim.attributes[a].data->count / 3;
                            break;
                        }
                    }
                }
            }
        }
        entry.triangleCount = tris;
        entry.materialCount = static_cast<uint32_t>(data->materials_count);
        cgltf_free(data);
    }
}

std::vector<SceneEntry> SceneRegistry::scan(const std::string& scenesDir) {
    std::vector<SceneEntry> entries;
    namespace fs = std::filesystem;

    // 0. Always add Procedural Cornell Box as index 0
    SceneEntry cornellBox;
    cornellBox.label = "Procedural Cornell Box";
    cornellBox.filepath = ""; // Empty filepath represents built-in procedural scene
    cornellBox.group = "Procedural";
    cornellBox.triangleCount = 2048;
    cornellBox.materialCount = 7;
    cornellBox.fileSizeBytes = 0;
    entries.push_back(cornellBox);

    if (!fs::exists(scenesDir) || !fs::is_directory(scenesDir)) {
        Logger::warn("SceneRegistry: Directory '{}' does not exist.", scenesDir);
        return entries;
    }

    std::vector<SceneEntry> fileEntries;

    // 1. Scan standalone scenes directly under scenes/ (Showcase)
    for (const auto& item : fs::directory_iterator(scenesDir)) {
        if (item.is_regular_file()) {
            std::string ext = item.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
            if (ext == ".glb" || ext == ".gltf") {
                std::string stem = item.path().stem().string();
                std::string label = formatSceneName(stem);
                SceneEntry e{ label, item.path().string(), "Showcase" };
                populateSceneMetadata(e);
                fileEntries.push_back(e);
            }
        }
    }

    // 2. Scan subdirectories (scenes/<name>/)
    for (const auto& item : fs::directory_iterator(scenesDir)) {
        if (item.is_directory()) {
            std::string dirName = item.path().filename().string();
            std::string dirTitle = formatSceneName(dirName);

            // Look for extended and core models
            fs::path extGlb = item.path() / (dirName + "_extended.glb");
            fs::path coreGlb = item.path() / (dirName + "_core.glb");

            // Handle underscore variations (e.g. coffee-maker vs coffee_maker)
            std::string underscoreDir = dirName;
            std::replace(underscoreDir.begin(), underscoreDir.end(), '-', '_');
            if (!fs::exists(extGlb)) extGlb = item.path() / (underscoreDir + "_extended.glb");
            if (!fs::exists(coreGlb)) coreGlb = item.path() / (underscoreDir + "_core.glb");

            if (fs::exists(extGlb)) {
                SceneEntry e{ dirTitle + " (Extended)", extGlb.string(), "Research" };
                populateSceneMetadata(e);
                fileEntries.push_back(e);
            }
            if (fs::exists(coreGlb)) {
                SceneEntry e{ dirTitle + " (Core)", coreGlb.string(), "Research" };
                populateSceneMetadata(e);
                fileEntries.push_back(e);
            }

            // Also check for any other glb/gltf files in this directory
            for (const auto& subItem : fs::directory_iterator(item.path())) {
                if (subItem.is_regular_file()) {
                    std::string subExt = subItem.path().extension().string();
                    std::transform(subExt.begin(), subExt.end(), subExt.begin(), [](unsigned char c) { return std::tolower(c); });
                    if (subExt == ".glb" || subExt == ".gltf") {
                        if (subItem.path() != extGlb && subItem.path() != coreGlb) {
                            std::string subStem = subItem.path().stem().string();
                            SceneEntry e{ dirTitle + " - " + formatSceneName(subStem), subItem.path().string(), "Custom" };
                            populateSceneMetadata(e);
                            fileEntries.push_back(e);
                        }
                    }
                }
            }
        }
    }

    // Sort file entries: Showcase first, then Research, then Custom; within groups alphabetically
    std::sort(fileEntries.begin(), fileEntries.end(), [](const SceneEntry& a, const SceneEntry& b) {
        auto groupPriority = [](const std::string& g) {
            if (g == "Showcase") return 1;
            if (g == "Research") return 2;
            if (g == "Custom") return 3;
            return 4;
        };
        int pa = groupPriority(a.group);
        int pb = groupPriority(b.group);
        if (pa != pb) return pa < pb;
        return a.label < b.label;
    });

    entries.insert(entries.end(), fileEntries.begin(), fileEntries.end());

    Logger::info("SceneRegistry: Discovered and indexed {} scene(s) in '{}'.", entries.size(), scenesDir);
    return entries;
}

} // namespace pathways
