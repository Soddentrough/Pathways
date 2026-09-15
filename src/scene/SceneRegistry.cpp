#include "scene/SceneRegistry.hpp"
#include "scene/UsdLoader.hpp"
#include "core/Logger.hpp"
#include "cgltf.h"

#include <filesystem>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <cctype>
#include <cstdio>
#include <cstring>

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
    if (rawName == "ClassicCar" || rawName == "classic_car" || rawName == "classic-car" ||
        rawName == "BuickRiviera" || rawName == "buick_riviera" || rawName == "buick-riviera" ||
        rawName == "buick_rivera" || rawName == "buick-rivera") {
        return "Buick Riviera";
    }

    std::string out;
    bool capitalizeNext = true;
    for (size_t i = 0; i < rawName.size(); ++i) {
        char c = rawName[i];
        if (c == '-' || c == '_') {
            out += ' ';
            capitalizeNext = true;
        } else if (i > 0 && std::islower(static_cast<unsigned char>(rawName[i - 1])) &&
                   std::isupper(static_cast<unsigned char>(c))) {
            out += ' ';
            out += c;
            capitalizeNext = false;
        } else if (capitalizeNext) {
            out += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            capitalizeNext = false;
        } else {
            out += c;
        }
    }
    return out;
}

static bool populateSceneMetadata(SceneEntry& entry) {
    if (entry.filepath.empty()) return false;

    std::error_code ec;
    if (!std::filesystem::exists(entry.filepath, ec)) {
        return false;
    }
    entry.fileSizeBytes = std::filesystem::file_size(entry.filepath, ec);

    if (UsdLoader::isUsdFile(entry.filepath)) {
        // Attempt metadata extraction; if it fails (e.g. non-USD build or Windows resolver fallback),
        // retain the entry with 0 counts so USD scenes remain in the UI dropdown list by default.
        if (!UsdLoader::populateMetadata(entry.filepath, entry.triangleCount, entry.materialCount) || entry.triangleCount == 0) {
            entry.triangleCount = 0;
            entry.materialCount = 0;
        }
        return true;
    }

    cgltf_options options{};
    cgltf_data* data = nullptr;
    cgltf_result res = cgltf_parse_file(&options, entry.filepath.c_str(), &data);
    if (res != cgltf_result_success || !data) {
        Logger::warn("SceneRegistry: Failed to parse glTF manifest '{}'.", entry.filepath);
        return false;
    }

    // Validate that all declared external buffers actually exist on disk
    std::filesystem::path parentDir = std::filesystem::path(entry.filepath).parent_path();
    for (cgltf_size b = 0; b < data->buffers_count; ++b) {
        const char* uri = data->buffers[b].uri;
        if (uri && std::strncmp(uri, "data:", 5) != 0) {
            std::string rawUri(uri);
            std::string decodedUri;
            for (size_t i = 0; i < rawUri.length(); ++i) {
                if (rawUri[i] == '%' && i + 2 < rawUri.length()) {
                    std::string hex = rawUri.substr(i + 1, 2);
                    char ch = static_cast<char>(std::strtol(hex.c_str(), nullptr, 16));
                    decodedUri += ch;
                    i += 2;
                } else {
                    decodedUri += rawUri[i];
                }
            }
            if (!std::filesystem::exists(parentDir / rawUri) && !std::filesystem::exists(parentDir / decodedUri)) {
                Logger::warn("SceneRegistry: Skipping scene '{}' because external buffer '{}' does not exist.",
                             entry.filepath, uri);
                cgltf_free(data);
                return false;
            }
        }
    }

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
    return (tris > 0);
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

    // 0b. Add Procedural Many-Lights Cornell Box as index 1
    SceneEntry manyLights;
    manyLights.label = "Procedural Many-Lights (64 Lights)";
    manyLights.filepath = "procedural:many-lights";
    manyLights.group = "Procedural";
    manyLights.triangleCount = 2176;
    manyLights.materialCount = 7;
    manyLights.fileSizeBytes = 0;
    entries.push_back(manyLights);

    if (scenesDir.empty()) {
        return entries;
    }

    if (!fs::exists(scenesDir) || !fs::is_directory(scenesDir)) {
        Logger::warn("SceneRegistry: Directory '{}' does not exist.", scenesDir);
        return entries;
    }

    std::vector<SceneEntry> fileEntries;
    std::unordered_set<std::string> seenCanonicalPaths;
    std::unordered_set<std::string> seenLabels;

    auto addUniqueEntry = [&](const SceneEntry& e) {
        std::error_code ec;
        std::string canon = fs::canonical(e.filepath, ec).string();
        if (canon.empty()) canon = e.filepath;
        if (seenCanonicalPaths.insert(canon).second) {
            if (seenLabels.insert(e.label).second) {
                fileEntries.push_back(e);
            }
        }
    };

    // 1. Scan standalone scenes directly under scenes/ (Showcase & USD)
    for (const auto& item : fs::directory_iterator(scenesDir)) {
        if (item.is_regular_file()) {
            std::string ext = item.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) { return std::tolower(c); });
            if (ext == ".glb" || ext == ".gltf" || ext == ".usd" || ext == ".usda" || ext == ".usdc") {
                std::string stem = item.path().stem().string();
                std::string label = formatSceneName(stem);
                std::string group = UsdLoader::isUsdFile(item.path().string()) ? "USD Scenes" : "Showcase";
                SceneEntry e{ label, item.path().string(), group };
                if (populateSceneMetadata(e)) {
                    addUniqueEntry(e);
                }
            }
        }
    }

    // 2. Scan subdirectories (scenes/<name>/)
    for (const auto& item : fs::directory_iterator(scenesDir)) {
        if (item.is_directory()) {
            std::string dirName = item.path().filename().string();
            std::string dirTitle = formatSceneName(dirName);

            // Look for extended and core models or usd scenes
            fs::path extGlb = item.path() / (dirName + "_extended.glb");
            fs::path coreGlb = item.path() / (dirName + "_core.glb");
            fs::path usdScene;

            // Handle underscore variations (e.g. coffee-maker vs coffee_maker)
            std::string underscoreDir = dirName;
            std::replace(underscoreDir.begin(), underscoreDir.end(), '-', '_');
            if (!fs::exists(extGlb)) extGlb = item.path() / (underscoreDir + "_extended.glb");
            if (!fs::exists(coreGlb)) coreGlb = item.path() / (underscoreDir + "_core.glb");

            // Priority for primary USD: .usdc (binary crate), then .usd, then .usda
            for (const auto& ext : {".usdc", ".usd", ".usda"}) {
                fs::path p1 = item.path() / (dirName + ext);
                if (fs::exists(p1)) { usdScene = p1; break; }
                fs::path p2 = item.path() / (underscoreDir + ext);
                if (fs::exists(p2)) { usdScene = p2; break; }
            }

            bool hasExt = fs::exists(extGlb);
            bool hasCore = fs::exists(coreGlb);
            bool hasUsd = !usdScene.empty();

            if (hasUsd) {
                SceneEntry e{ dirTitle, usdScene.string(), "USD Scenes" };
                if (populateSceneMetadata(e)) {
                    addUniqueEntry(e);
                }
            }

            if (hasExt && hasCore) {
                SceneEntry e{ dirTitle + " (Extended)", extGlb.string(), "Research" };
                if (populateSceneMetadata(e)) {
                    addUniqueEntry(e);
                }

                SceneEntry c{ dirTitle + " (Core)", coreGlb.string(), "Research" };
                if (populateSceneMetadata(c)) {
                    addUniqueEntry(c);
                }
            } else if (hasExt) {
                SceneEntry e{ dirTitle, extGlb.string(), "Research" };
                if (populateSceneMetadata(e)) {
                    addUniqueEntry(e);
                }
            } else if (hasCore) {
                SceneEntry e{ dirTitle, coreGlb.string(), "Research" };
                if (populateSceneMetadata(e)) {
                    addUniqueEntry(e);
                }
            }

            // Also check for any other distinct glb/gltf/usd files in this directory (e.g. Bistro Interior vs Exterior)
            for (const auto& subItem : fs::directory_iterator(item.path())) {
                if (subItem.is_regular_file()) {
                    std::error_code ec;
                    if (fs::file_size(subItem.path(), ec) < 100) {
                        continue; // Skip Windows Git symlink text files (e.g. 17-byte ClassicCar.usdc)
                    }
                    std::string subExt = subItem.path().extension().string();
                    std::transform(subExt.begin(), subExt.end(), subExt.begin(), [](unsigned char c) { return std::tolower(c); });
                    if (subExt == ".glb" || subExt == ".gltf" || subExt == ".usd" || subExt == ".usda" || subExt == ".usdc") {
                        if (subItem.path() != extGlb && subItem.path() != coreGlb && subItem.path() != usdScene) {
                            std::string subStem = subItem.path().stem().string();

                            // Skip alternate file formats or instanced duplicates of already registered primary scenes
                            if (hasUsd && (subStem == usdScene.stem().string() || subStem == dirName || subStem == underscoreDir ||
                                           subStem.ends_with("_instanced") || subStem.ends_with("-instanced"))) {
                                continue;
                            }
                            if ((hasExt || hasCore) && (subStem == dirName || subStem == underscoreDir ||
                                                        subStem.ends_with("_core") || subStem.ends_with("_extended"))) {
                                continue;
                            }

                            std::string group = UsdLoader::isUsdFile(subItem.path().string()) ? "USD Scenes" : "Custom";
                            std::string labelName;
                            if (subStem.rfind(dirName, 0) == 0 || subStem.rfind(underscoreDir, 0) == 0) {
                                labelName = formatSceneName(subStem);
                            } else {
                                labelName = dirTitle + " - " + formatSceneName(subStem);
                            }
                            if (labelName == dirTitle || seenLabels.contains(labelName)) {
                                continue;
                            }
                            SceneEntry e{ labelName, subItem.path().string(), group };
                            if (populateSceneMetadata(e)) {
                                addUniqueEntry(e);
                            }
                        }
                    }
                }
            }
        }
    }

    // Sort file entries: Showcase first, then Research, then USD Scenes, then Custom; within groups alphabetically
    std::sort(fileEntries.begin(), fileEntries.end(), [](const SceneEntry& a, const SceneEntry& b) {
        auto groupPriority = [](const std::string& g) {
            if (g == "Showcase") return 1;
            if (g == "Research") return 2;
            if (g == "USD Scenes") return 3;
            if (g == "Custom") return 4;
            return 5;
        };
        int pa = groupPriority(a.group);
        int pb = groupPriority(b.group);
        if (pa != pb) return pa < pb;
        return a.label < b.label;
    });

    entries.insert(entries.end(), fileEntries.begin(), fileEntries.end());

    // Disambiguate any duplicate scene labels across groups (e.g. Cornell Box Showcase vs Research)
    std::unordered_map<std::string, int> labelCounts;
    for (const auto& e : entries) {
        labelCounts[e.label]++;
    }
    for (auto& e : entries) {
        if (labelCounts[e.label] > 1) {
            e.label += " (" + e.group + ")";
        }
    }

    Logger::info("SceneRegistry: Discovered and indexed {} scene(s) in '{}'.", entries.size(), scenesDir);
    return entries;
}

} // namespace pathways
