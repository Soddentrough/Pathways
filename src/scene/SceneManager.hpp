#pragma once

#include "core/Config.hpp"
#include "scene/ProceduralScene.hpp"
#include "scene/SceneRegistry.hpp"
#include "scene/GltfLoader.hpp"
#include "scene/UsdLoader.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Texture.hpp"
#include "scene/Camera.hpp"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <memory>
#include <vector>
#include <string>
#include <future>
#include <atomic>
#include <filesystem>

namespace pathways {

class SceneGeometryPipeline;

class SceneManager {
public:
    SceneManager(VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool commandPool);
    ~SceneManager();

    SceneManager(const SceneManager&) = delete;
    SceneManager& operator=(const SceneManager&) = delete;

    // Discovery & Path Resolution
    static std::filesystem::path discoverScenesDirectory();
    void discoverScenes(const std::filesystem::path& scenesDir = "");
    static bool isProceduralCornellBoxPath(const std::string& path);
    std::string resolveScenePath(const std::string& inputPath) const;
    std::string detectSceneHdri(const std::string& scenePath) const;
    std::string getActiveSceneName(const std::string& fallbackPath = "") const;

    // Available scenes queries
    [[nodiscard]] const std::vector<SceneEntry>& getAvailableScenes() const noexcept { return m_availableScenes; }
    [[nodiscard]] int getCurrentSceneIndex() const noexcept { return m_currentSceneIndex; }
    void setCurrentSceneIndex(int idx) noexcept { m_currentSceneIndex = idx; }

    // Scene loading (synchronous)
    static SceneData loadRawSceneData(const std::string& filepath, const UsdLoadOptions& usdOptions);

    // Asynchronous scene loading
    void requestSceneChange(const std::string& filepath, const Config& config);
    [[nodiscard]] bool isSceneLoading() const noexcept { return m_isSceneLoading.load(); }
    [[nodiscard]] const std::string& getLoadingSceneName() const noexcept { return m_loadingSceneName; }
    [[nodiscard]] const std::string& getLoadingScenePath() const noexcept { return m_loadingScenePath; }
    [[nodiscard]] std::chrono::steady_clock::time_point getLoadingStartTime() const noexcept { return m_sceneLoadingStartTime; }
    [[nodiscard]] float getLoadingElapsedSec() const noexcept {
        if (!m_isSceneLoading.load()) return 0.0f;
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<float>(now - m_sceneLoadingStartTime).count();
    }
    bool pollAsyncLoading(SceneData& outScene, std::string& outPath);

    // Ingest & Apply Scene
    bool ingestSceneData(SceneData newScene, const std::string& filepath, const Config& config,
                         SceneGeometryPipeline* geomPipeline);

    void uploadGpuBuffers(SceneGeometryPipeline* geomPipeline);
    void loadTextures(const std::string& hdriPath, const std::string& scenePath);
    void updateCamera(Camera* camera, const Config& config);

    // Accessors for Scene Data
    [[nodiscard]] const SceneData& getSceneData() const noexcept { return m_sceneData; }
    [[nodiscard]] SceneData& getSceneData() noexcept { return m_sceneData; }
    [[nodiscard]] uint32_t getNumTriangles() const noexcept { return m_numTriangles; }
    [[nodiscard]] uint64_t getNumInstancedTriangles() const noexcept { return m_numInstancedTriangles; }
    [[nodiscard]] uint32_t getNumInstances() const noexcept { return m_numInstances; }
    [[nodiscard]] uint32_t getNumOpaqueTriangles() const noexcept { return m_numOpaqueTriangles; }
    [[nodiscard]] uint32_t getNumSpheres() const noexcept { return m_numSpheres; }
    [[nodiscard]] uint32_t getNumMaterials() const noexcept { return m_numMaterials; }
    [[nodiscard]] uint32_t getNumLights() const noexcept { return m_numLights; }
    [[nodiscard]] bool hasNonOpaque() const noexcept { return m_sceneHasNonOpaque; }
    [[nodiscard]] bool hasAlphaMask() const noexcept { return m_sceneHasAlphaMask; }
    [[nodiscard]] float getCachedDivergentAreaRatio() const noexcept { return m_cachedDivergentAreaRatio; }
    void setCachedDivergentAreaRatio(float r) noexcept { m_cachedDivergentAreaRatio = r; }

    // Accessors for Scene GPU Buffers
    [[nodiscard]] Buffer* getPositionBuffer() const noexcept { return m_positionBuffer.get(); }
    [[nodiscard]] Buffer* getTriangleBuffer() const noexcept { return m_triangleBuffer.get(); }
    [[nodiscard]] Buffer* getSphereBuffer() const noexcept { return m_sphereBuffer.get(); }
    [[nodiscard]] Buffer* getMaterialBuffer() const noexcept { return m_materialBuffer.get(); }
    [[nodiscard]] Buffer* getMaterialArchetypeBuffer() const noexcept { return m_materialArchetypeBuffer.get(); }
    [[nodiscard]] Buffer* getShadeMaterialBuffer() const noexcept { return m_shadeMaterialBuffer.get(); }
    [[nodiscard]] Buffer* getLightBuffer() const noexcept { return m_lightBuffer.get(); }
    [[nodiscard]] Buffer* getLightTreeBuffer() const noexcept { return m_lightTreeBuffer.get(); }

    // Accessors for Textures
    [[nodiscard]] Texture* getEnvironmentMap() const noexcept { return m_environmentMap.get(); }
    [[nodiscard]] Texture* getDummyWhite() const noexcept { return m_dummyWhite.get(); }
    [[nodiscard]] Texture* getDummyNormal() const noexcept { return m_dummyNormal.get(); }
    [[nodiscard]] Texture* getBlueNoiseTexture() const noexcept { return m_blueNoiseTexture.get(); }
    [[nodiscard]] const std::vector<std::unique_ptr<Texture>>& getSceneTextures() const noexcept { return m_sceneTextures; }

    // Reclaims host-side triangles after upload
    void reclaimHostTriangles() noexcept {
        m_sceneData.triangles.clear();
        m_sceneData.triangles.shrink_to_fit();
    }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    VkCommandPool m_commandPool = VK_NULL_HANDLE;

    std::filesystem::path m_scenesDir;
    std::vector<SceneEntry> m_availableScenes;
    int m_currentSceneIndex = -1;

    // Asynchronous loading state
    std::future<SceneData> m_sceneLoadingFuture;
    std::atomic<bool> m_isSceneLoading{false};
    std::string m_loadingScenePath;
    std::string m_loadingSceneName;
    std::chrono::steady_clock::time_point m_sceneLoadingStartTime;

    // Scene metadata
    SceneData m_sceneData;
    uint32_t m_numTriangles = 0;
    uint64_t m_numInstancedTriangles = 0;
    uint32_t m_numInstances = 1;
    uint32_t m_numOpaqueTriangles = 0;
    uint32_t m_numSpheres = 0;
    uint32_t m_numMaterials = 0;
    uint32_t m_numLights = 0;
    bool m_sceneHasNonOpaque = false;
    bool m_sceneHasAlphaMask = false;
    float m_cachedDivergentAreaRatio = -1.0f;

    // Scene GPU Buffers
    std::unique_ptr<Buffer> m_positionBuffer;
    std::unique_ptr<Buffer> m_triangleBuffer;
    std::unique_ptr<Buffer> m_sphereBuffer;
    std::unique_ptr<Buffer> m_materialBuffer;
    std::unique_ptr<Buffer> m_materialArchetypeBuffer;
    std::unique_ptr<Buffer> m_shadeMaterialBuffer;
    std::unique_ptr<Buffer> m_lightBuffer;
    std::unique_ptr<Buffer> m_lightTreeBuffer;

    // Scene Textures
    std::vector<std::unique_ptr<Texture>> m_sceneTextures;
    std::unique_ptr<Texture> m_environmentMap;
    std::unique_ptr<Texture> m_dummyWhite;
    std::unique_ptr<Texture> m_dummyNormal;
    std::unique_ptr<Texture> m_blueNoiseTexture;
};

} // namespace pathways
