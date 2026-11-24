#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <array>

#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "core/Config.hpp"

namespace pathways {

struct NRCQuery {
    glm::vec4 pos_roughness;    // pos.xyz, roughness (w)
    glm::vec4 normal_flags;     // normal.xyz, flags (w)
    glm::vec4 dir_pixelIndex;   // dir.xyz, pixelIndex (w as float)
    glm::vec4 albedo_pad;       // albedo.rgb, pad (w)
    glm::vec4 throughput;       // throughput.rgb, pad (w)
};

struct NRCTrainingRecord {
    glm::vec4 pos_roughness;      // queryPos.xyz, queryRoughness (w)
    glm::vec4 normal_flags;       // queryNormal.xyz, flags (w)
    glm::vec4 dir_pixelIndex;     // queryDir.xyz, pixelIndex (w as float)
    glm::vec4 albedo_pad;         // queryAlbedo.rgb, pad (w)
    glm::vec4 throughput;         // tailBsdfWeight.rgb, hasTail (w: >0.5 to bootstrap MLP at tail vertex)
    glm::vec4 targetRadiance;     // localSuffixDirectRadiance.rgb, pad (w)
    glm::vec4 tailPos_roughness;  // tailPos.xyz, tailRoughness (w)
    glm::vec4 tailPacked;         // oct(tailNormal), oct(tailDir), unorm4x8(tailAlbedo), pad
};
static_assert(sizeof(NRCTrainingRecord) == 128, "NRCTrainingRecord must remain 128 bytes");

struct NRCCountersBuffer {
    uint32_t queryCount;
    uint32_t trainCount;
    uint32_t dispatchX;
    uint32_t pad;
};

class NRCManager {
public:
    NRCManager(VkDevice device, VmaAllocator allocator,
               uint32_t width, uint32_t height,
               const std::vector<char>& inferSpv,
               const std::vector<char>& trainSpv,
               const std::vector<char>& resolveSpv = {},
               VkPhysicalDevice physDevice = VK_NULL_HANDLE);
    ~NRCManager();

    NRCManager(const NRCManager&) = delete;
    NRCManager& operator=(const NRCManager&) = delete;

    void resize(uint32_t width, uint32_t height);
    void updateDescriptors(VkImageView accumImageView,
                           VkImageView mlDiffuseImageView = VK_NULL_HANDLE,
                           VkImageView mlSpecularImageView = VK_NULL_HANDLE,
                           VkImageView mlAlbedoRoughnessView = VK_NULL_HANDLE,
                           VkImageView mlSpecAlbedoView = VK_NULL_HANDLE,
                           VkImageView mlIndirectDiffuseView = VK_NULL_HANDLE,
                           VkImageView mlIndirectSpecularView = VK_NULL_HANDLE);
    void updateDescriptors(uint32_t slot, VkImageView accumImageView,
                           VkImageView mlDiffuseImageView = VK_NULL_HANDLE,
                           VkImageView mlSpecularImageView = VK_NULL_HANDLE,
                           VkImageView mlAlbedoRoughnessView = VK_NULL_HANDLE,
                           VkImageView mlSpecAlbedoView = VK_NULL_HANDLE,
                           VkImageView mlIndirectDiffuseView = VK_NULL_HANDLE,
                           VkImageView mlIndirectSpecularView = VK_NULL_HANDLE);

    void resetCounters(VkCommandBuffer cmd);
    void resetWeights() { initWeights(); }
    void recordInference(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                         glm::vec3 worldMin, glm::vec3 worldMax, uint32_t queryCountOverride = 0,
                         uint32_t frameSlot = 0, uint32_t captureMlData = 0, float maxRadiance = 10.0f);
    void recordTraining(VkCommandBuffer cmd, uint32_t frameIndex,
                        glm::vec3 worldMin, glm::vec3 worldMax, float learningRate = 1e-3f,
                        uint32_t batchSize = 8192);
    void recordInferenceAndTraining(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                                    glm::vec3 worldMin, glm::vec3 worldMax,
                                    uint32_t queryCountOverride = 0, uint32_t frameSlot = 0,
                                    uint32_t captureMlData = 0, float maxRadiance = 10.0f,
                                    uint32_t frameIndex = 0, float learningRate = 1e-3f,
                                    uint32_t batchSize = 8192,
                                    bool singleQueryPerPixel = false);

    Buffer* getQueryQueue() const { return m_queryQueue.get(); }
    Buffer* getTrainQueue() const { return m_trainQueue.get(); }
    Buffer* getCounters() const { return m_counters.get(); }
    Buffer* getWeights() const { return m_weights.get(); }
    Buffer* getAtomicAccumBuffer() const { return m_atomicAccumBuffer.get(); }

    uint32_t getMaxQueries() const { return m_maxQueries; }
    uint32_t getMaxTrainRecords() const { return m_maxTrainRecords; }
    uint32_t getAdamStep() const { return m_adamStep; }

    bool isInitialized() const { return m_initialized; }
    NrcBackend getBackend() const { return m_backend; }
    void setBackend(NrcBackend backend);
    bool isUsingSdkProvider() const { return m_ffxRcContext != nullptr; }
    const char* getActiveBackendName() const;

private:
    void initFfxRadianceCacheContext();
    void destroyFfxRadianceCacheContext();
    void initBuffers();
    void initWeights();
    void createDescriptorSetLayouts();
    void allocateDescriptorSets();
    void createPipelines(const std::vector<char>& inferSpv, const std::vector<char>& trainSpv, const std::vector<char>& resolveSpv);
    void syncInferenceSnapshot(VkCommandBuffer cmd);
    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_maxQueries = 0;
    uint32_t m_maxTrainRecords = 0;
    uint32_t m_adamStep = 0;
    bool m_initialized = false;
    bool m_needsGpuInit = true;
    bool m_atomicAccumNeedsClear = true;
    bool m_inferSnapshotDirty = false;
    NrcBackend m_backend = NrcBackend::Auto;
    void* m_ffxRcContext = nullptr;
    std::vector<uint16_t> m_initialWeights;
    std::vector<float> m_initialMasterWeights;

    // GPU Buffers
    std::unique_ptr<Buffer> m_weights;
    std::unique_ptr<Buffer> m_inferWeights;
    std::unique_ptr<Buffer> m_weightMomentum;
    std::unique_ptr<Buffer> m_hashGrid;
    std::unique_ptr<Buffer> m_inferHashGrid;
    std::unique_ptr<Buffer> m_hashGridState;
    std::unique_ptr<Buffer> m_queryQueue;
    std::unique_ptr<Buffer> m_trainQueue;
    std::unique_ptr<Buffer> m_counters;
    std::unique_ptr<Buffer> m_atomicAccumBuffer; // 32-bit fixed-point atomic accumulation buffer (CRIT-05)
    std::unique_ptr<Image> m_dummyStorageImage;  // 1x1 RGBA16F fallback for unbound resolve storage images

    // Descriptors & Pipelines
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_inferDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_inferDescSet = VK_NULL_HANDLE;
    VkPipelineLayout m_inferPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_inferPipeline = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_trainDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_trainDescSet = VK_NULL_HANDLE;
    VkPipelineLayout m_trainPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_trainPipeline = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_resolveDescLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> m_resolveDescSets{ VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkPipelineLayout m_resolvePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_resolvePipeline = VK_NULL_HANDLE;
};

} // namespace pathways
