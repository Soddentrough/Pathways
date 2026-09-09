#pragma once

#include "vulkan/VulkanContext.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "scene/ProceduralScene.hpp"
#include "scene/Camera.hpp"
#include "core/Config.hpp"
#include "rt/AccelerationStructure.hpp"
#include "rt/RTPipeline.hpp"
#include "vulkan/Texture.hpp"
#include <memory>
#include <vector>
#include <string>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace pathways {

struct GpuDeviceNode {
    uint32_t deviceIndex = 0;
    std::string deviceName;
    std::unique_ptr<VulkanContext> context;
    std::unique_ptr<Image> accumTarget;
    std::unique_ptr<Buffer> p2pStagingBuffer;

    static constexpr uint32_t NUM_IN_FLIGHT = 2;

    // Commands & Sync (Double-buffered)
    VkCommandPool commandPool = VK_NULL_HANDLE;
    std::array<VkCommandBuffer, NUM_IN_FLIGHT> commandBuffers = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkFence, NUM_IN_FLIGHT> renderFences = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkQueryPool, NUM_IN_FLIGHT> queryPools = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    float timestampPeriod = 1.0f;
    double lastFrameTimeMs = 0.0;
    double lastTransferTimeMs = 0.0;

    // Cross-GPU Hardware Synchronization (VK_KHR_external_semaphore_fd)
    std::array<VkSemaphore, NUM_IN_FLIGHT> secSemaphores = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkSemaphore, NUM_IN_FLIGHT> primImportedSemaphores = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<int, NUM_IN_FLIGHT> exportedFd = { -1, -1 };
    std::array<bool, NUM_IN_FLIGHT> slotFdReady = { false, false };
    std::array<bool, NUM_IN_FLIGHT> slotHasExecuted = { false, false };

    // Secondary scene buffers
    std::unique_ptr<Buffer> triangleBuffer;
    std::unique_ptr<Buffer> sphereBuffer;
    std::unique_ptr<Buffer> materialBuffer;
    std::unique_ptr<Buffer> lightBuffer;
    std::array<std::unique_ptr<Buffer>, NUM_IN_FLIGHT> cameraUBOs;

    // Secondary ReSTIR DI Reservoir Buffers (Bindings 9 & 10)
    std::array<std::unique_ptr<Buffer>, 2> restirReservoirs;
    uint32_t restirPingPongIndex = 0;

    // Secondary Hardware Acceleration Structures (VK_KHR_ray_query)
    std::unique_ptr<Buffer> asVertexBuffer;
    std::unique_ptr<AccelerationStructureManager> asManager;
    std::unique_ptr<AccelerationStructure> blas;
    std::unique_ptr<AccelerationStructure> tlas;

    // Secondary Textures & Environment Map (Bindings 7 & 8)
    static constexpr uint32_t MAX_SCENE_TEXTURES = 64;
    std::unique_ptr<Texture> dummyWhite;
    std::unique_ptr<Texture> dummyNormal;
    std::unique_ptr<Texture> environmentMap;
    std::vector<std::unique_ptr<Texture>> sceneTextures;

    // Secondary hardware ray tracing pipeline & descriptors (VK_KHR_ray_tracing_pipeline)
    VkDescriptorPool descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout rtDescLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, NUM_IN_FLIGHT> rtDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout rtpPipelineLayout = VK_NULL_HANDLE;
    std::unique_ptr<RTPipeline> rtpKhrPipeline;

    ~GpuDeviceNode();
};

class MultiGpuManager {
public:
    static constexpr uint32_t MAX_SCENE_TEXTURES = 64;
    MultiGpuManager(const Config& config, VulkanContext* primaryContext, const SceneData& scene);
    ~MultiGpuManager();

    bool isMultiGpuActive() const { return m_active && m_mode != MultiGpuMode::Off; }
    bool isSecondaryInitialized() const { return m_active && !m_devices.empty(); }
    uint32_t getDeviceCount() const { return static_cast<uint32_t>(m_devices.size() + 1); }
    MultiGpuMode getMode() const { return m_mode; }
    void setMode(MultiGpuMode mode) { m_mode = mode; m_config.mgpu_mode = mode; }
    void setFormat(AccumFormat format) { m_config.accum_format = format; }
    double getSecondaryGpuTimeMs() const;
    double getSecondaryTransferTimeMs() const;
    const std::string& getSecondaryDeviceName() const;
    VulkanContext* getSecondaryContext() const { return m_devices.empty() ? nullptr : m_devices[0]->context.get(); }
    void resize(uint32_t width, uint32_t height);
    bool loadScene(const SceneData& scene);

    // Launch secondary GPU raytracing asynchronously via persistent dedicated worker thread
    void launchSecondaryWork(const CameraUniform& cameraUniform,
                             uint32_t bufferSlot,
                             uint32_t tileOffsetX, uint32_t tileOffsetY,
                             uint32_t tileWidth, uint32_t tileHeight,
                             uint32_t numTriangles, uint32_t numSpheres,
                             uint32_t numMaterials, uint32_t numLights,
                             uint32_t useHardwareRT = 0,
                             uint32_t hasEnvMap = 0,
                             float envMapIntensity = 1.0f,
                             uint32_t accumulateHistory = 1,
                             void* dstHostPtr = nullptr,
                             size_t transferBytes = 0);

    // Wait for secondary GPU completion and copy data to destination host buffer
    void syncAndTransfer(uint32_t slot = 0, void* dstHostPtr = nullptr, size_t byteSize = 0);

    // Wait for secondary GPU worker thread to become completely idle
    void waitWorkerIdle();

    // Wait for secondary GPU to finish execution of a specific slot (discarding old in-flight work)
    void waitSecondarySlot(uint32_t slot);

    bool isZeroCopyActive() const { return m_useZeroCopyHost; }
    bool isCrossGpuSyncActive() const { return m_useCrossGpuSync; }
    VkSemaphore getImportedSemaphore(uint32_t slot = 0) const {
        if (!m_useCrossGpuSync || m_devices.empty()) return VK_NULL_HANDLE;
        return m_devices[0]->primImportedSemaphores[slot % GpuDeviceNode::NUM_IN_FLIGHT];
    }
    static constexpr uint32_t NUM_SHARED_BUFFERS = 2;
    VkBuffer getPrimarySharedBuffer(uint32_t slot = 0) const { return m_sharedBufferPrimary[slot % NUM_SHARED_BUFFERS]; }
    VkBuffer getSecondarySharedBuffer(uint32_t slot = 0) const { return m_sharedBufferSecondary[slot % NUM_SHARED_BUFFERS]; }
    void* getSharedHostPointer(uint32_t slot = 0) const { return m_sharedHostPtr[slot % NUM_SHARED_BUFFERS]; }
    size_t getSharedBufferSize() const { return m_sharedBufferSize; }

private:
    struct SecondaryWorkPacket {
        CameraUniform cameraUniform{};
        uint32_t bufferSlot = 0;
        uint32_t tileOffsetX = 0;
        uint32_t tileOffsetY = 0;
        uint32_t tileWidth = 0;
        uint32_t tileHeight = 0;
        uint32_t numTriangles = 0;
        uint32_t numSpheres = 0;
        uint32_t numMaterials = 0;
        uint32_t numLights = 0;
        uint32_t useHardwareRT = 0;
        uint32_t hasEnvMap = 0;
        float envMapIntensity = 1.0f;
        uint32_t accumulateHistory = 1;
        void* dstHostPtr = nullptr;
        size_t transferBytes = 0;
        bool valid = false;
    };

    void workerLoop();
    void executeSecondaryWork(const SecondaryWorkPacket& packet);

    void initSecondaryDevice(const Config& config, const SceneData& scene);
    void initSharedHostBuffer(VkDeviceSize bufferSize);
    void destroySharedHostBuffer();
    std::vector<char> loadShaderSPIRV(const std::string& filename);
    VkShaderModule createShaderModule(VkDevice device, const std::vector<char>& code);

    VulkanContext* m_primaryContext = nullptr;
    std::vector<std::unique_ptr<GpuDeviceNode>> m_devices;
    MultiGpuMode m_mode = MultiGpuMode::Off;
    bool m_active = false;
    Config m_config;

    // Persistent dedicated worker thread for secondary GPU queue management
    std::thread m_workerThread;
    std::mutex m_workMutex;
    std::condition_variable m_workCv;
    std::condition_variable m_submitCv;
    bool m_stopWorker = false;
    SecondaryWorkPacket m_pendingWork;
    uint32_t m_waitingSlot = 0;
    bool m_workSubmitted = false;
    std::array<bool, 2> m_slotSubmitted = { false, false };
    bool m_workerBusy = false;

    // Zero-copy host allocation imported into both GPUs via VK_EXT_external_memory_host (Double-buffered)
    std::array<void*, NUM_SHARED_BUFFERS> m_sharedHostPtr = { nullptr, nullptr };
    size_t m_sharedBufferSize = 0;
    std::array<VkDeviceMemory, NUM_SHARED_BUFFERS> m_sharedMemPrimary = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkBuffer, NUM_SHARED_BUFFERS> m_sharedBufferPrimary = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDeviceMemory, NUM_SHARED_BUFFERS> m_sharedMemSecondary = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkBuffer, NUM_SHARED_BUFFERS> m_sharedBufferSecondary = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    bool m_useZeroCopyHost = false;
    bool m_useCrossGpuSync = false;
};

} // namespace pathways
