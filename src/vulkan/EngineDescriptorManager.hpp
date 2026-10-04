#pragma once

#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "vulkan/Texture.hpp"
#include "scene/Camera.hpp"
#include "core/Config.hpp"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <array>
#include <vector>
#include <memory>

namespace pathways {

class WavefrontPipeline;
class NRCManager;
class ReSTIRManager;
class PostProcessPipeline;
class MultiGpuManager;

struct ImageDescriptorParams {
    Image* accumImage = nullptr;
    Image* outputImage = nullptr;
    Image* directLightImage = nullptr;
    Image* normalDepthImage = nullptr;
    Image* motionVectorImage = nullptr;
    Image* filteredCausticImage = nullptr;
    std::array<Image*, 2> frameImages = { nullptr, nullptr };
};

struct SceneDescriptorParams {
    Buffer* triangleBuffer = nullptr;
    Buffer* sphereBuffer = nullptr;
    Buffer* materialBuffer = nullptr;
    Buffer* lightBuffer = nullptr;
    VkAccelerationStructureKHR tlasHandle = VK_NULL_HANDLE;
    Texture* environmentMap = nullptr;
    Texture* dummyWhite = nullptr;
    Texture* blueNoiseTexture = nullptr;
    const std::vector<std::unique_ptr<Texture>>* sceneTextures = nullptr;
};

struct WavefrontDescriptorParams {
    WavefrontPipeline* wavefrontPipeline = nullptr;
    NRCManager* nrcManager = nullptr;
    ReSTIRManager* restirManager = nullptr;
    Image* accumImage = nullptr;
    std::array<Image*, 2> frameImages = { nullptr, nullptr };
    const std::array<std::unique_ptr<Buffer>, 2>* cameraUBOs = nullptr;
    Buffer* triangleBuffer = nullptr;
    Buffer* sphereBuffer = nullptr;
    Buffer* materialBuffer = nullptr;
    Buffer* lightBuffer = nullptr;
    Buffer* lightTreeBuffer = nullptr;
    Buffer* instanceBuffer = nullptr;
    Buffer* materialArchetypeBuffer = nullptr;
    Buffer* shadeMaterialBuffer = nullptr;
    VkAccelerationStructureKHR tlasHandle = VK_NULL_HANDLE;
    Texture* environmentMap = nullptr;
    Texture* dummyWhite = nullptr;
    const std::vector<std::unique_ptr<Texture>>* sceneTextures = nullptr;
    Image* motionVectorImage = nullptr;
    Image* normalDepthImage = nullptr;
    Image* mlAlbedoRoughnessImage = nullptr;
    Image* mlSpecularMotionImage = nullptr;
    Image* mlDiffuseImage = nullptr;
    Image* mlSpecularImage = nullptr;
    Image* filteredCausticImage = nullptr;
};

struct MergeDescriptorParams {
    PostProcessPipeline* postProcess = nullptr;
    MultiGpuManager* mgpu = nullptr;
    Image* motionVectorImage = nullptr;
    Image* normalDepthImage = nullptr;
    std::array<Image*, 2> frameImages = { nullptr, nullptr };
    uint32_t width = 0;
    uint32_t height = 0;
    AccumFormat accumFormat = AccumFormat::RGBA16_SFLOAT;
    std::vector<uint32_t> concurrentQueues;
};

class EngineDescriptorManager {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr uint32_t MAX_SCENE_TEXTURES = 512;

    EngineDescriptorManager(VkDevice device, VmaAllocator allocator);
    ~EngineDescriptorManager();

    EngineDescriptorManager(const EngineDescriptorManager&) = delete;
    EngineDescriptorManager& operator=(const EngineDescriptorManager&) = delete;

    /// Creates descriptor pool, RT descriptor set layout, allocates RT descriptor sets,
    /// and performs initial bindings for frame images and camera UBOs.
    void init(
        const std::array<Image*, MAX_FRAMES_IN_FLIGHT>& frameImages,
        const std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT>& cameraUBOs
    );

    /// Explicit destruction of Vulkan descriptor resources.
    void destroy();

    /// Updates image storage descriptors (bindings 0, 11, 12, 14, 15) in RT descriptor sets.
    void updatePrimaryImageDescriptors(const ImageDescriptorParams& params);

    /// Updates scene geometry, material, light buffers, TLAS, environment map, and textures (bindings 2..8, 13).
    void updatePrimarySceneDescriptors(const SceneDescriptorParams& params);

    /// Updates Wavefront compute pipeline scene descriptor sets across frames-in-flight and NRC descriptors.
    void updateWavefrontDescriptors(const WavefrontDescriptorParams& params);

    /// Ensures multi-GPU secondary transfer buffer is adequately sized and updates merge compute descriptors.
    void updateMergeDescriptors(const MergeDescriptorParams& params);

    [[nodiscard]] VkDescriptorPool getPool() const noexcept { return m_descriptorPool; }
    [[nodiscard]] VkDescriptorSetLayout getRtDescLayout() const noexcept { return m_rtDescLayout; }
    [[nodiscard]] VkDescriptorSet getRtDescSet(uint32_t slot) const noexcept { return m_rtDescSets[slot]; }
    [[nodiscard]] const std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT>& getRtDescSets() const noexcept { return m_rtDescSets; }
    [[nodiscard]] Buffer* getSecTransferBuffer() const noexcept { return m_secTransferBuffer.get(); }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_rtDescLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_rtDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    std::unique_ptr<Buffer> m_secTransferBuffer;
};

} // namespace pathways
