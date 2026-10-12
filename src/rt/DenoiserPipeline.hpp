#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <memory>
#include <array>
#include <vector>
#include <functional>
#include <string>

#include "vulkan/Image.hpp"
#include "core/Config.hpp"

namespace pathways {

class PostProcessPipeline;

using ShaderLoaderFunc = std::function<std::vector<char>(const std::string&)>;

class DenoiserPipeline {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    struct TemporalPushConstants {
        uint32_t width;
        uint32_t height;
        uint32_t resetHistory;
        float maxHistoryLength;
    };
    static_assert(sizeof(TemporalPushConstants) == 16, "TemporalPushConstants must be 16 bytes");

    struct VariancePushConstants {
        uint32_t width;
        uint32_t height;
    };
    static_assert(sizeof(VariancePushConstants) == 8, "VariancePushConstants must be 8 bytes");

    struct AtrousPushConstants {
        uint32_t width;
        uint32_t height;
        int32_t stepSize;
        float phiNormal;
        float phiDepth;
        float phiLuminance;
        uint32_t iteration;
    };
    static_assert(sizeof(AtrousPushConstants) == 28, "AtrousPushConstants must be 28 bytes");

    struct RemodulatePushConstants {
        uint32_t width;
        uint32_t height;
        uint32_t accumulatedSamples;
        float crossfadeBlend;
    };
    static_assert(sizeof(RemodulatePushConstants) == 16, "RemodulatePushConstants must be 16 bytes");

    DenoiserPipeline(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VmaAllocator allocator,
        VkQueue queue,
        ShaderLoaderFunc shaderLoader
    );
    ~DenoiserPipeline();

    DenoiserPipeline(const DenoiserPipeline&) = delete;
    DenoiserPipeline& operator=(const DenoiserPipeline&) = delete;

    void resize(uint32_t width, uint32_t height, AccumFormat accumFormat, VkCommandBuffer cmd);

    void recordDenoise(
        VkCommandBuffer cmd,
        uint32_t frameSlot,
        bool resetHistory,
        uint32_t accumulatedSamples,
        float crossfadeBlend,
        Image* currentDiffuse,
        Image* currentSpecular,
        Image* motionVectors,
        Image* normalDepth,
        Image* albedoRoughness,
        Image* accumImage
    );

    void initTonemapDescriptors(PostProcessPipeline* postProcess, Image* outputImage);

    [[nodiscard]] Image* getDenoisedImage() const { return m_denoisedImage.get(); }
    [[nodiscard]] VkDescriptorSet getTonemapDescSet() const { return m_tonemapDescSet; }
    [[nodiscard]] bool isInitialized() const { return m_width > 0 && m_height > 0 && m_denoisedImage != nullptr; }

private:
    void initPipelines();
    void createDescriptorSetLayouts();
    void allocateDescriptorSets();
    void updateDescriptors(
        Image* currentDiffuse,
        Image* currentSpecular,
        Image* motionVectors,
        Image* normalDepth,
        Image* albedoRoughness,
        Image* accumImage
    );

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    ShaderLoaderFunc m_shaderLoader;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    AccumFormat m_accumFormat = AccumFormat::RGBA16_SFLOAT;

    // Descriptors & Pools
    VkDescriptorPool m_descPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_temporalDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_varianceDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_atrousDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_remodulateDescLayout = VK_NULL_HANDLE;

    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_temporalDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_varianceDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDescriptorSet, 2> m_atrousDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE }; // Ping-pong A and B
    VkDescriptorSet m_remodulateDescSet = VK_NULL_HANDLE;
    VkDescriptorSet m_tonemapDescSet = VK_NULL_HANDLE;

    // Pipeline Layouts & Pipelines
    VkPipelineLayout m_temporalPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_variancePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_atrousPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_remodulatePipelineLayout = VK_NULL_HANDLE;

    VkPipeline m_temporalPipeline = VK_NULL_HANDLE;
    VkPipeline m_variancePipeline = VK_NULL_HANDLE;
    VkPipeline m_atrousPipeline = VK_NULL_HANDLE;
    VkPipeline m_remodulatePipeline = VK_NULL_HANDLE;

    // Internal Images
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_historyDiffuse;
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_historySpecular;
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_historyMoments;
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_historyLength;
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_historyNormalDepth;

    std::array<std::unique_ptr<Image>, 2> m_pingPongDiffuse;
    std::array<std::unique_ptr<Image>, 2> m_pingPongSpecular;

    std::unique_ptr<Image> m_denoisedImage;

    bool m_descriptorsBound = false;
};

} // namespace pathways
