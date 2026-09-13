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
    glm::vec4 pos_roughness;    // pos.xyz, roughness (w)
    glm::vec4 normal_flags;     // normal.xyz, flags (w)
    glm::vec4 dir_pixelIndex;   // dir.xyz, pixelIndex (w as float)
    glm::vec4 albedo_pad;       // albedo.rgb, pad (w)
    glm::vec4 throughput;       // throughput.rgb, pad (w)
    glm::vec4 targetRadiance;   // targetRadiance.rgb, pad (w)
};

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
               const std::vector<char>& resolveSpv = {});
    ~NRCManager();

    NRCManager(const NRCManager&) = delete;
    NRCManager& operator=(const NRCManager&) = delete;

    void resize(uint32_t width, uint32_t height);
    void updateDescriptors(VkImageView accumImageView);

    void resetCounters(VkCommandBuffer cmd);
    void recordInference(VkCommandBuffer cmd, uint32_t width, uint32_t height,
                         glm::vec3 worldMin, glm::vec3 worldMax, uint32_t queryCountOverride = 0);
    void recordTraining(VkCommandBuffer cmd, uint32_t frameIndex,
                        glm::vec3 worldMin, glm::vec3 worldMax, float learningRate = 1e-3f,
                        uint32_t batchSize = 8192);

    Buffer* getQueryQueue() const { return m_queryQueue.get(); }
    Buffer* getTrainQueue() const { return m_trainQueue.get(); }
    Buffer* getCounters() const { return m_counters.get(); }
    Buffer* getHashTable() const { return m_hashTable.get(); }
    Buffer* getWeights() const { return m_weights.get(); }
    Buffer* getAtomicAccumBuffer() const { return m_atomicAccumBuffer.get(); }

    uint32_t getMaxQueries() const { return m_maxQueries; }
    uint32_t getMaxTrainRecords() const { return m_maxTrainRecords; }

    bool isInitialized() const { return m_initialized; }

private:
    void initBuffers();
    void initWeightsAndHashTable();
    void createDescriptorSetLayouts();
    void allocateDescriptorSets();
    void createPipelines(const std::vector<char>& inferSpv, const std::vector<char>& trainSpv, const std::vector<char>& resolveSpv);
    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    uint32_t m_maxQueries = 0;
    uint32_t m_maxTrainRecords = 0;
    bool m_initialized = false;

    // GPU Buffers
    std::unique_ptr<Buffer> m_hashTable;
    std::unique_ptr<Buffer> m_weights;
    std::unique_ptr<Buffer> m_weightMomentum;
    std::unique_ptr<Buffer> m_queryQueue;
    std::unique_ptr<Buffer> m_trainQueue;
    std::unique_ptr<Buffer> m_counters;
    std::unique_ptr<Buffer> m_atomicAccumBuffer; // 32-bit fixed-point atomic accumulation buffer (CRIT-05)

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
    VkDescriptorSet m_resolveDescSet = VK_NULL_HANDLE;
    VkPipelineLayout m_resolvePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_resolvePipeline = VK_NULL_HANDLE;
};

} // namespace pathways
