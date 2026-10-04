#include "scene/SceneManager.hpp"
#include "core/Logger.hpp"
#include "scene/SceneGeometryPipeline.hpp"

#include <algorithm>
#include <system_error>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif

namespace pathways {

SceneManager::SceneManager(VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool commandPool)
    : m_device(device), m_allocator(allocator), m_queue(queue), m_commandPool(commandPool) {
}

SceneManager::~SceneManager() = default;

std::filesystem::path SceneManager::discoverScenesDirectory() {
    std::filesystem::path scenesDir = "scenes";
    if (!std::filesystem::exists(scenesDir) || !std::filesystem::is_directory(scenesDir)) {
        std::filesystem::path exeDir;
#ifdef _WIN32
        char exePathBuf[MAX_PATH] = {0};
        if (GetModuleFileNameA(NULL, exePathBuf, MAX_PATH)) {
            exeDir = std::filesystem::path(exePathBuf).parent_path();
        }
#elif defined(__linux__) || defined(__unix__)
        std::error_code ec;
        auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
        if (!ec && !p.empty()) {
            exeDir = p.parent_path();
        } else {
            p = std::filesystem::canonical("/proc/self/exe", ec);
            if (!ec) exeDir = p.parent_path();
        }
#endif
        if (!exeDir.empty()) {
            auto adjacentScenes = exeDir / "scenes";
            auto parentScenes = exeDir / ".." / "scenes";
            auto relShare = exeDir / ".." / "share" / "pathways" / "scenes";
            auto devScenes = exeDir / ".." / ".." / "scenes";
            if (std::filesystem::exists(adjacentScenes) && std::filesystem::is_directory(adjacentScenes)) {
                scenesDir = adjacentScenes;
            } else if (std::filesystem::exists(parentScenes) && std::filesystem::is_directory(parentScenes)) {
                scenesDir = parentScenes;
            } else if (std::filesystem::exists(relShare) && std::filesystem::is_directory(relShare)) {
                scenesDir = relShare;
            } else if (std::filesystem::exists(devScenes) && std::filesystem::is_directory(devScenes)) {
                scenesDir = devScenes;
            }
        }
        if ((!std::filesystem::exists(scenesDir) || !std::filesystem::is_directory(scenesDir)) &&
            std::filesystem::exists("/usr/share/pathways/scenes")) {
            scenesDir = "/usr/share/pathways/scenes";
        }
    }
    return scenesDir;
}

void SceneManager::discoverScenes(const std::filesystem::path& scenesDir) {
    if (scenesDir.empty()) {
        m_scenesDir = discoverScenesDirectory();
    } else {
        m_scenesDir = scenesDir;
    }

    if (std::filesystem::exists(m_scenesDir) && std::filesystem::is_directory(m_scenesDir)) {
        m_availableScenes = SceneRegistry::scan(m_scenesDir.string());
    } else {
        m_availableScenes = SceneRegistry::scan("");
    }
    m_currentSceneIndex = 0;
}

bool SceneManager::isProceduralCornellBoxPath(const std::string& path) {
    return path.empty() || path == "__procedural_cornell_box__" ||
           path == "cornell-box" || path == "cornell_box" ||
           path == "procedural:cornell-box" || path == "procedural:cornell_box";
}

std::string SceneManager::resolveScenePath(const std::string& inputPath) const {
    if (inputPath.empty()) {
        return "";
    }
    if (isProceduralCornellBoxPath(inputPath) ||
        inputPath == "many-lights" || inputPath == "many_lights" || inputPath == "procedural:many-lights" ||
        inputPath == "cyber-city" || inputPath == "cyber_city" || inputPath == "procedural:cyber-city" ||
        inputPath == "procedural:cyber_city" || inputPath == "Procedural Cyber City" ||
        inputPath == "infinity-mirror" || inputPath == "procedural:infinity-mirror" ||
        inputPath == "infinity_mirror" || inputPath == "procedural:infinity_mirror" ||
        inputPath == "Infinity Mirror" || inputPath == "Procedural Infinity Mirror") {
        return inputPath;
    }

    std::string resolvedScene = inputPath;
    if (!std::filesystem::exists(resolvedScene)) {
        auto candidate = m_scenesDir / resolvedScene;
        if (std::filesystem::exists(candidate)) {
            resolvedScene = candidate.string();
        } else if (resolvedScene.rfind("scenes/", 0) == 0) {
            auto subCandidate = m_scenesDir / resolvedScene.substr(7);
            if (std::filesystem::exists(subCandidate)) {
                resolvedScene = subCandidate.string();
            }
        } else {
            std::string stem = std::filesystem::path(resolvedScene).stem().string();
            auto nested = m_scenesDir / stem / resolvedScene;
            if (std::filesystem::exists(nested)) {
                resolvedScene = nested.string();
            }
        }
    }

    // Map legacy ClassicCar aliases to BuickRiviera
    if (!std::filesystem::exists(resolvedScene) || (std::filesystem::is_regular_file(resolvedScene) && std::filesystem::file_size(resolvedScene) < 100)) {
        if (resolvedScene.find("ClassicCar") != std::string::npos || resolvedScene.find("classic_car") != std::string::npos) {
            auto buickCandidate = m_scenesDir / "BuickRiviera" / "BuickRiviera.usdc";
            if (std::filesystem::exists(buickCandidate)) {
                resolvedScene = buickCandidate.string();
            }
        }
    }

    // If resolvedScene is a directory, find primary file
    if (std::filesystem::exists(resolvedScene) && std::filesystem::is_directory(resolvedScene)) {
        std::filesystem::path dirPath(resolvedScene);
        std::string dirName = dirPath.filename().string();
        std::string underscoreDir = dirName;
        std::replace(underscoreDir.begin(), underscoreDir.end(), '-', '_');

        std::vector<std::filesystem::path> candidates = {
            dirPath / (dirName + ".usd"),
            dirPath / (underscoreDir + ".usd"),
            dirPath / (dirName + "_extended.glb"),
            dirPath / (underscoreDir + "_extended.glb"),
            dirPath / (dirName + ".glb"),
            dirPath / (underscoreDir + ".glb"),
            dirPath / (dirName + ".usda"),
            dirPath / (dirName + ".usdc")
        };
        for (const auto& c : candidates) {
            if (std::filesystem::exists(c)) {
                resolvedScene = c.string();
                break;
            }
        }
    }

    return resolvedScene;
}

std::string SceneManager::detectSceneHdri(const std::string& scenePath) const {
    if (scenePath.empty()) return "";
    std::filesystem::path sp(scenePath);
    std::filesystem::path sceneDir = sp.has_parent_path() ? sp.parent_path() : std::filesystem::current_path();
    std::vector<std::filesystem::path> hdriCandidates = {
        sceneDir / "textures and hdri" / "golden_gate_hills_2k.exr",
        sceneDir / "textures" / "studio_small_08_4k.exr",
        sceneDir / "textures" / "studio_small_08_4k.hdr",
        sceneDir / ".." / "textures" / "studio_small_08_4k.exr",
        sceneDir / "textures and hdri" / "golden_gate_hills_2k.hdr",
    };
    for (const auto& cand : hdriCandidates) {
        std::error_code ec;
        if (std::filesystem::exists(cand, ec)) {
            Logger::info("SceneManager: Auto-detected scene HDRI environment map: '{}'", cand.string());
            return cand.string();
        }
    }
    return "";
}

std::string SceneManager::getActiveSceneName(const std::string& fallbackPath) const {
    if (m_currentSceneIndex >= 0 && m_currentSceneIndex < static_cast<int>(m_availableScenes.size())) {
        if (!m_availableScenes[m_currentSceneIndex].label.empty()) {
            return m_availableScenes[m_currentSceneIndex].label;
        }
    }
    if (isProceduralCornellBoxPath(fallbackPath)) {
        return "Procedural Cornell Box";
    }
    std::filesystem::path p(fallbackPath);
    return SceneRegistry::formatSceneName(p.stem().string());
}

SceneData SceneManager::loadRawSceneData(const std::string& filepath, const UsdLoadOptions& usdOptions) {
    if (isProceduralCornellBoxPath(filepath)) {
        Logger::info("Loading Procedural Cornell Box...");
        return ProceduralScene::createCornellBox();
    } else if (filepath == "procedural:many-lights" || filepath == "many-lights" || filepath == "many_lights") {
        Logger::info("Loading Procedural Many-Lights Cornell Box (64 Lights)...");
        return ProceduralScene::createManyLightsScene();
    } else if (filepath == "procedural:cyber-city" || filepath == "procedural:cyber_city" || filepath == "cyber-city" || filepath == "cyber_city" || filepath == "Procedural Cyber City") {
        Logger::info("Loading Procedural Cyber City Megastructure...");
        return ProceduralScene::createCyberCityScene();
    } else if (filepath == "procedural:infinity-mirror" || filepath == "procedural:infinity_mirror" || filepath == "infinity-mirror" || filepath == "infinity_mirror" || filepath == "Procedural Infinity Mirror" || filepath == "Infinity Mirror") {
        Logger::info("Loading Infinity Mirror Corridor...");
        return ProceduralScene::createInfinityMirrorScene();
    } else {
        Logger::info("Loading user specified scene: {}", filepath);
        if (UsdLoader::isUsdFile(filepath)) {
            return UsdLoader::loadSceneData(filepath, usdOptions);
        } else {
            return GltfLoader::loadSceneData(filepath);
        }
    }
}

void SceneManager::requestSceneChange(const std::string& filepath, const Config& config) {
    if (m_isSceneLoading.load()) {
        Logger::warn("Scene loading already in progress; ignoring request for '{}'", filepath);
        return;
    }
    m_loadingScenePath = filepath;

    std::string prettyName;
    for (const auto& sc : m_availableScenes) {
        if (sc.filepath == filepath) {
            prettyName = sc.label;
            break;
        }
    }
    if (prettyName.empty()) {
        std::string stem = std::filesystem::path(filepath).stem().string();
        if (isProceduralCornellBoxPath(filepath) || stem.empty()) {
            prettyName = "Procedural Cornell Box";
        } else if (filepath == "procedural:many-lights" || filepath == "many-lights" || filepath == "many_lights") {
            prettyName = "Procedural Many-Lights";
        } else if (filepath == "procedural:cyber-city" || filepath == "procedural:cyber_city" || filepath == "cyber-city" || filepath == "cyber_city" || filepath == "Procedural Cyber City") {
            prettyName = "Cyber City";
        } else if (filepath == "procedural:infinity-mirror" || filepath == "procedural:infinity_mirror" || filepath == "infinity-mirror" || filepath == "infinity_mirror" || filepath == "Procedural Infinity Mirror" || filepath == "Infinity Mirror") {
            prettyName = "Infinity Mirror";
        } else {
            prettyName = SceneRegistry::formatSceneName(stem);
        }
    }
    m_loadingSceneName = prettyName;
    m_sceneLoadingStartTime = std::chrono::steady_clock::now();
    m_isSceneLoading.store(true);
    Logger::info("Initiating asynchronous scene load for '{}' ({})...", filepath, m_loadingSceneName);

    UsdLoadOptions usdOptions{};
    usdOptions.instanceDensity = config.instance_density;
    usdOptions.cullDistance = config.cull_distance;
    usdOptions.cameraPosOverride = config.camera_pos;
    usdOptions.viewportAspect = (config.height > 0) ? (static_cast<float>(config.width) / static_cast<float>(config.height)) : (16.0f / 9.0f);

    m_sceneLoadingFuture = std::async(std::launch::async, [filepath, usdOptions]() -> SceneData {
        return SceneManager::loadRawSceneData(filepath, usdOptions);
    });
}

bool SceneManager::pollAsyncLoading(SceneData& outScene, std::string& outPath) {
    if (m_isSceneLoading.load()) {
        if (m_sceneLoadingFuture.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
            outScene = m_sceneLoadingFuture.get();
            outPath = m_loadingScenePath;
            m_isSceneLoading.store(false);
            m_loadingScenePath.clear();
            m_loadingSceneName.clear();
            return true;
        }
    }
    return false;
}

bool SceneManager::ingestSceneData(SceneData newScene, const std::string& filepath, const Config& config,
                                   SceneGeometryPipeline* geomPipeline) {
    if (newScene.triangles.empty() && newScene.spheres.empty()) {
        Logger::warn("Loaded scene contains no renderable geometry! Falling back to procedural Cornell Box.");
        newScene = ProceduralScene::createCornellBox();
    }

    m_sceneData = std::move(newScene);
    m_cachedDivergentAreaRatio = -1.0f;

    if (config.enable_macro_blas && geomPipeline) {
        geomPipeline->clusterInstancesToMacroBlas(m_sceneData, config);
    }

    m_numTriangles = static_cast<uint32_t>(m_sceneData.triangles.size());
    m_numSpheres = static_cast<uint32_t>(m_sceneData.spheres.size());
    m_numMaterials = static_cast<uint32_t>(m_sceneData.materials.size());
    m_numLights = static_cast<uint32_t>(m_sceneData.lights.size());

    uint64_t totalInstTris = 0;
    if (!m_sceneData.instances.empty() && !m_sceneData.blasRanges.empty()) {
        for (const auto& inst : m_sceneData.instances) {
            if (inst.blasIndex < m_sceneData.blasRanges.size()) {
                totalInstTris += m_sceneData.blasRanges[inst.blasIndex].triangleCount;
            }
        }
        m_numInstances = static_cast<uint32_t>(m_sceneData.instances.size());
    } else {
        totalInstTris = m_numTriangles;
        m_numInstances = 1;
    }
    m_numInstancedTriangles = totalInstTris;

    if (geomPipeline) {
        geomPipeline->updateSceneTransparency(m_sceneData.materials, m_sceneHasNonOpaque, m_sceneHasAlphaMask);
        geomPipeline->partitionSceneGeometry(m_sceneData, m_numOpaqueTriangles);
    }

    if (m_numInstancedTriangles > m_numTriangles) {
        Logger::info("Active Scene: {} Base Triangles ({} Instanced across {} Instances), {} Spheres, {} Materials, {} Lights (Non-Opaque: {})",
                     m_numTriangles, m_numInstancedTriangles, m_numInstances, m_numSpheres, m_numMaterials, m_numLights, m_sceneHasNonOpaque ? "YES" : "NO");
    } else {
        Logger::info("Active Scene: {} Triangles, {} Spheres, {} Materials, {} Lights (Non-Opaque: {})",
                     m_numTriangles, m_numSpheres, m_numMaterials, m_numLights, m_sceneHasNonOpaque ? "YES" : "NO");
    }

    m_currentSceneIndex = -1;
    for (size_t i = 0; i < m_availableScenes.size(); ++i) {
        std::error_code ec;
        if (m_availableScenes[i].filepath == filepath ||
            (!m_availableScenes[i].filepath.empty() &&
             std::filesystem::exists(m_availableScenes[i].filepath) &&
             std::filesystem::exists(filepath) &&
             std::filesystem::equivalent(m_availableScenes[i].filepath, filepath, ec))) {
            m_currentSceneIndex = static_cast<int>(i);
            break;
        }
    }
    return true;
}

void SceneManager::uploadGpuBuffers(SceneGeometryPipeline* geomPipeline) {
    size_t numTris = m_sceneData.triangles.size();
    std::vector<glm::vec4> positions;
    positions.reserve(numTris * 3);
    std::vector<TriangleShadeGPU> shadeTriangles;
    shadeTriangles.reserve(numTris);

    for (const auto& tri : m_sceneData.triangles) {
        positions.push_back(glm::vec4(glm::vec3(tri.v0.position), 1.0f));
        positions.push_back(glm::vec4(glm::vec3(tri.v1.position), 1.0f));
        positions.push_back(glm::vec4(glm::vec3(tri.v2.position), 1.0f));
        shadeTriangles.push_back(createTriangleShadeGPU(tri));
    }

    VkDeviceSize posSize = std::max(sizeof(glm::vec4) * positions.size(), sizeof(glm::vec4) * 3);
    m_positionBuffer = std::make_unique<Buffer>(
        m_allocator, posSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (!positions.empty() && geomPipeline) {
        geomPipeline->uploadToDeviceBuffer(*m_positionBuffer, positions.data(), sizeof(glm::vec4) * positions.size());
    }

    VkDeviceSize triSize = std::max(sizeof(TriangleShadeGPU) * shadeTriangles.size(), sizeof(TriangleShadeGPU));
    m_triangleBuffer = std::make_unique<Buffer>(
        m_allocator, triSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (!shadeTriangles.empty() && geomPipeline) {
        geomPipeline->uploadToDeviceBuffer(*m_triangleBuffer, shadeTriangles.data(), sizeof(TriangleShadeGPU) * shadeTriangles.size());
    }

    VkDeviceSize sphereSize = std::max(sizeof(SphereGPU) * m_sceneData.spheres.size(), sizeof(SphereGPU));
    m_sphereBuffer = std::make_unique<Buffer>(
        m_allocator, sphereSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.spheres.empty()) {
        m_sphereBuffer->copyFrom(m_sceneData.spheres.data(), sizeof(SphereGPU) * m_sceneData.spheres.size());
    }

    VkDeviceSize matSize = std::max(sizeof(MaterialGPU) * m_sceneData.materials.size(), sizeof(MaterialGPU));
    m_materialBuffer = std::make_unique<Buffer>(
        m_allocator, matSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.materials.empty()) {
        m_materialBuffer->copyFrom(m_sceneData.materials.data(), sizeof(MaterialGPU) * m_sceneData.materials.size());
    }

    std::vector<uint32_t> matArchetypes(m_sceneData.materials.size());
    for (size_t i = 0; i < m_sceneData.materials.size(); ++i) {
        matArchetypes[i] = computeMaterialArchetype(m_sceneData.materials[i]);
    }
    VkDeviceSize archSize = std::max(sizeof(uint32_t) * matArchetypes.size(), sizeof(uint32_t));
    m_materialArchetypeBuffer = std::make_unique<Buffer>(
        m_allocator, archSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!matArchetypes.empty()) {
        m_materialArchetypeBuffer->copyFrom(matArchetypes.data(), sizeof(uint32_t) * matArchetypes.size());
    }

    std::vector<ShadeMaterialGPU> shadeMaterials(m_sceneData.materials.size());
    for (size_t i = 0; i < m_sceneData.materials.size(); ++i) {
        shadeMaterials[i] = createShadeMaterial(m_sceneData.materials[i]);
    }
    VkDeviceSize shadeMatSize = std::max(sizeof(ShadeMaterialGPU) * shadeMaterials.size(), sizeof(ShadeMaterialGPU));
    m_shadeMaterialBuffer = std::make_unique<Buffer>(
        m_allocator, shadeMatSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!shadeMaterials.empty()) {
        m_shadeMaterialBuffer->copyFrom(shadeMaterials.data(), sizeof(ShadeMaterialGPU) * shadeMaterials.size());
    }

    VkDeviceSize lightSize = std::max(sizeof(LightGPU) * m_sceneData.lights.size(), sizeof(LightGPU));
    m_lightBuffer = std::make_unique<Buffer>(
        m_allocator, lightSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.lights.empty()) {
        buildLightAliasTable(m_sceneData.lights);
        buildLightTree(m_sceneData.lights, m_sceneData.lightTreeNodes);
        m_lightBuffer->copyFrom(m_sceneData.lights.data(), sizeof(LightGPU) * m_sceneData.lights.size());
    }

    VkDeviceSize lightTreeSize = std::max(sizeof(LightTreeNodeGPU) * m_sceneData.lightTreeNodes.size(), sizeof(LightTreeNodeGPU));
    m_lightTreeBuffer = std::make_unique<Buffer>(
        m_allocator, lightTreeSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    if (!m_sceneData.lightTreeNodes.empty()) {
        m_lightTreeBuffer->copyFrom(m_sceneData.lightTreeNodes.data(), sizeof(LightTreeNodeGPU) * m_sceneData.lightTreeNodes.size());
    }
}

void SceneManager::loadTextures(const std::string& hdriPath, const std::string& scenePath) {
    if (!m_dummyWhite) {
        m_dummyWhite = Texture::createDummyWhite(m_device, m_allocator, m_queue, m_commandPool);
    }
    if (!m_dummyNormal) {
        m_dummyNormal = Texture::createDummyNormal(m_device, m_allocator, m_queue, m_commandPool);
    }
    if (!m_blueNoiseTexture) {
        m_blueNoiseTexture = Texture::createBlueNoise64(m_device, m_allocator, m_queue, m_commandPool);
    }

    m_environmentMap = Texture::createSceneEnvironmentMap(
        m_device, m_allocator, m_queue, m_commandPool,
        hdriPath, scenePath, m_sceneData.domeLightHdriPath
    );

    m_sceneTextures.clear();
    for (const auto& texData : m_sceneData.textures) {
        if (!texData.pixels.empty() && texData.width > 0 && texData.height > 0) {
            VkFormat fmt = texData.isSrgb ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM;
            auto tex = Texture::createFromPixels(
                m_device, m_allocator, m_queue, m_commandPool,
                texData.width, texData.height,
                fmt, texData.pixels.data(),
                texData.pixels.size(), false
            );
            m_sceneTextures.push_back(std::move(tex));
        } else {
            m_sceneTextures.push_back(Texture::createDummyWhite(m_device, m_allocator, m_queue, m_commandPool));
        }
    }
    Logger::info("Scene textures loaded: {} texture(s).", m_sceneTextures.size());
}

void SceneManager::updateCamera(Camera* camera, const Config& config) {
    if (!camera) return;

    camera->setSceneScale(m_sceneData.sceneRadius, m_sceneData.focalDistance, m_sceneData.centralTarget);

    glm::vec3 camPos = m_sceneData.cameraPosition;
    glm::vec3 camTarget = m_sceneData.cameraTarget;
    glm::vec3 camUp = m_sceneData.cameraUp;
    float camFov = m_sceneData.cameraFov;
    bool hasOverride = false;

    if (config.camera_pos.has_value()) {
        camPos = *config.camera_pos;
        hasOverride = true;
    }
    if (config.camera_target.has_value()) {
        camTarget = *config.camera_target;
        hasOverride = true;
    }
    if (config.camera_up.has_value()) {
        camUp = *config.camera_up;
        hasOverride = true;
    }
    if (config.camera_fov.has_value()) {
        camFov = *config.camera_fov;
        hasOverride = true;
    }

    if (m_sceneData.hasCamera || hasOverride) {
        camera->lookAt(camPos, camTarget, camUp);
        camera->setFov(camFov);
        if (config.camera_fov.has_value()) {
            camera->setAdaptiveFov(false);
        }
        camera->setDefaultFraming(camPos, camTarget, camFov);
        Logger::info("Active Camera: pos=({:.3f}, {:.3f}, {:.3f}), target=({:.3f}, {:.3f}, {:.3f}), up=({:.3f}, {:.3f}, {:.3f}), fov={:.1f}°{}",
                     camPos.x, camPos.y, camPos.z,
                     camTarget.x, camTarget.y, camTarget.z,
                     camUp.x, camUp.y, camUp.z,
                     camFov,
                     hasOverride ? " [CLI Override]" : "");
    }
}

} // namespace pathways
