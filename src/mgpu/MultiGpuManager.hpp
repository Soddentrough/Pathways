#pragma once

#include "vulkan/VulkanContext.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "scene/ProceduralScene.hpp"
#include "scene/Camera.hpp"
#include "core/Config.hpp"
#include "rt/AccelerationStructure.hpp"
#include "vulkan/Texture.hpp"
#include <memory>
#include <vector>
#include <string>
#include <future>

namespace pathways {

struct GpuDeviceNode {
    uint32_t deviceIndex = 0;
    std::string deviceName;
    std::unique_ptr<VulkanContext> context;
    std::unique_ptr<Image> accumTarget;
    std::unique_ptr<Buffer> p2pStagingBuffer;

    // Commands & Sync
    VkCommandPool commandPool = VK_NULL_HANDLE;
    VkCommandBuffer commandBuffer = VK_NULL_HANDLE;
    VkFence renderFence = VK_NULL_HANDLE;
    VkQueryPool queryPool = VK_NULL_HANDLE;
    float timestampPeriod = 1.0f;
    double lastFrameTimeMs = 0.0;

    // Secondary scene buffers
    std::unique_ptr<Buffer> triangleBuffer;
    std::unique_ptr<Buffer> sphereBuffer;
    std::unique_ptr<Buffer> materialBuffer;
    std::unique_ptr<Buffer> lightBuffer;
    std::unique_ptr<Buffer> cameraUBO;

    // Secondary Hardware Acceleration Structures (VK_KHR_ray_query)
    std::unique_ptr<Buffer> asVertexBuffer;
    std::unique_ptr<AccelerationStructureManager> asManager;
    std::unique_ptr<AccelerationStructure> blas;
    std::unique_ptr<AccelerationStructure> tlas;

    // Secondary Textures & Environment Map (Bindings 7 & 8)
    std::unique_ptr<Texture> dummyWhite;
    std::unique_ptr<Texture> dummyNormal;
    std::unique_ptr<Texture> environmentMap;
    std::vector<std::unique_ptr<Texture>> sceneTextures;

    // Secondary pipeline & descriptors
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout rtDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet rtDescSet = VK_NULL_HANDLE;
    VkPipelineLayout rtPipelineLayout = VK_NULL_HANDLE;
    VkPipeline rtPipeline = VK_NULL_HANDLE;

    ~GpuDeviceNode();
};

class MultiGpuManager {
public:
    MultiGpuManager(const Config& config, VulkanContext* primaryContext, const SceneData& scene);
    ~MultiGpuManager();

    bool isMultiGpuActive() const { return m_active; }
    uint32_t getDeviceCount() const { return static_cast<uint32_t>(m_devices.size() + 1); }
    MultiGpuMode getMode() const { return m_mode; }
    void setMode(MultiGpuMode mode) { m_mode = mode; }
    double getSecondaryGpuTimeMs() const;
    const std::string& getSecondaryDeviceName() const;
    void resize(uint32_t width, uint32_t height);

    // Launch secondary GPU raytracing asynchronously in background thread
    void launchSecondaryWork(const CameraUniform& cameraUniform,
                             uint32_t frameIndex,
                             uint32_t tileOffsetX, uint32_t tileOffsetY,
                             uint32_t tileWidth, uint32_t tileHeight,
                             uint32_t numTriangles, uint32_t numSpheres,
                             uint32_t numMaterials, uint32_t numLights,
                             uint32_t useHardwareRT = 0,
                             uint32_t hasEnvMap = 0,
                             float envMapIntensity = 1.0f,
                             uint32_t accumulateHistory = 1);

    // Wait for secondary GPU completion and copy data to destination host buffer
    void syncAndTransfer(void* dstHostPtr, size_t byteSize);

private:
    void initSecondaryDevice(const Config& config, const SceneData& scene);
    std::vector<char> loadShaderSPIRV(const std::string& filename);
    VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);

    VulkanContext* m_primaryContext = nullptr;
    std::vector<std::unique_ptr<GpuDeviceNode>> m_devices;
    MultiGpuMode m_mode = MultiGpuMode::Off;
    bool m_active = false;
    std::future<void> m_asyncTask;
    Config m_config;
};

} // namespace pathways
