#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <string>
#include <array>

#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "core/Config.hpp"

namespace pathways {

struct UpwaysTemporalPushConstants {
    uint32_t renderWidth;
    uint32_t renderHeight;
    uint32_t resetHistory;
    float minAlphaDiffuse;
    float minAlphaSpecular;
};
static_assert(sizeof(UpwaysTemporalPushConstants) == 20, "UpwaysTemporalPushConstants must be 20 bytes");

struct UpwaysNeuralGuidePushConstants {
    uint32_t renderWidth;
    uint32_t renderHeight;
    uint32_t resetHistory;
    float minAlphaDiffuse;
    float minAlphaSpecular;
    float basePhiDepth;
    float basePhiNormal;
    float basePhiLumaDiffuse;
    float basePhiLumaSpecular;
};
static_assert(sizeof(UpwaysNeuralGuidePushConstants) == 36, "UpwaysNeuralGuidePushConstants must be 36 bytes");

struct UpwaysUpscaleResolvePushConstants {
    uint32_t renderWidth;
    uint32_t renderHeight;
    uint32_t displayWidth;
    uint32_t displayHeight;
    float jitterX;
    float jitterY;
    uint32_t enableDisplayHistory;
    float displayHistoryAlpha;
};
static_assert(sizeof(UpwaysUpscaleResolvePushConstants) == 32, "UpwaysUpscaleResolvePushConstants must be 32 bytes");

struct UpwaysPassTimings {
    double temporalMs = 0.0;       // Pass 1: upways_temporal.comp (Co-Designed Demodulation, Firefly Clamp, Reprojection & Quad Packing)
    double neuralGuideMs = 0.0;    // Pass 2: upways_neural_guide.comp (3-Scale Spatiotemporal Recurrent WMMA U-Net & Radiance Refinement)
    double upscaleResolveMs = 0.0; // Pass 3: upways_upscale_resolve.comp (Fused 2-Stage Coarse s=3 + Fine s=1 Neural KPN & Remodulation)
    double historyCopyMs = 0.0;    // History Roll: vkCmdCopyImage (normalDepth + specularMotion)
    double totalMs = 0.0;          // Total Upways GPU time across all passes + history copy
    bool valid = false;
};

class UpwaysPipeline {
public:
    static constexpr uint32_t MAX_QUERY_SLOTS = 2;
    static constexpr uint32_t QUERIES_PER_SLOT = 5;

    UpwaysPipeline(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VmaAllocator allocator,
        uint32_t inputWidth,
        uint32_t inputHeight,
        uint32_t outputWidth,
        uint32_t outputHeight,
        const std::vector<char>& temporalSpv,
        const std::vector<char>& neuralGuideSpv,
        const std::vector<char>& upscaleResolveSpv,
        const std::string& weightsPath = "",
        bool enableSuperRes = false,
        VkFormat imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT
    );
    ~UpwaysPipeline();

    UpwaysPipeline(const UpwaysPipeline&) = delete;
    UpwaysPipeline& operator=(const UpwaysPipeline&) = delete;

    void resize(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight, bool enableSuperRes);

    void setHistorySourceImages(VkImage normalDepthImage, VkImage specularMotionImage) {
        m_srcNormalDepthImage = normalDepthImage;
        m_srcSpecMotionImage = specularMotionImage;
    }

    void updateDescriptors(
        VkImageView demodDiffuseView,
        VkImageView demodSpecularView,
        VkImageView normalDepthView,
        VkImageView albedoRoughnessView,
        VkImageView surfaceMotionView,
        VkImageView specularMotionView,
        VkImageView restirMetadataView = VK_NULL_HANDLE,
        VkImageView displayAlbedoView = VK_NULL_HANDLE,
        VkImageView displayNormalsView = VK_NULL_HANDLE
    );

    void updateDescriptors(
        VkImageView accumImageView,
        VkImageView normalDepthImageView,
        VkImageView motionVectorImageView,
        VkImageView albedoRoughnessImageView,
        VkImageView specularMotionImageView,
        VkImageView diffuseImageView,
        VkImageView specularImageView,
        VkBuffer /*restirReservoirBuffer*/,
        VkImageView /*confidenceOutputImageView*/,
        VkBuffer /*restirReservoirBuffer1*/
    ) {
        updateDescriptors(
            diffuseImageView ? diffuseImageView : accumImageView,
            specularImageView ? specularImageView : accumImageView,
            normalDepthImageView,
            albedoRoughnessImageView,
            motionVectorImageView,
            specularMotionImageView ? specularMotionImageView : motionVectorImageView,
            VK_NULL_HANDLE,
            albedoRoughnessImageView,
            normalDepthImageView
        );
    }

    void recordFrame(
        VkCommandBuffer cmd,
        uint32_t frameIndex,
        bool resetHistory,
        bool cameraMoved,
        const glm::mat4& currInvView = glm::mat4(1.0f),
        const glm::mat4& prevViewProj = glm::mat4(1.0f),
        const glm::mat4& invProj = glm::mat4(1.0f),
        const glm::mat4& prevView = glm::mat4(1.0f),
        const glm::vec2& jitterOffset = glm::vec2(0.0f),
        uint32_t totalSamples = 1u
    );

    void recordFrame(
        VkCommandBuffer cmd,
        uint32_t frameIndex,
        bool resetHistory,
        bool cameraMoved,
        int32_t /*tileOffsetX*/,
        int32_t /*tileOffsetY*/,
        int32_t /*tileWidth*/,
        int32_t /*tileHeight*/,
        int32_t /*apronWidth*/,
        uint32_t totalSamples = 1u
    ) {
        recordFrame(cmd, frameIndex, resetHistory, cameraMoved, glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::vec2(0.0f), totalSamples);
    }

    void transitionInitialLayouts(VkCommandBuffer cmd);
    void collectQueryResults(int32_t targetSlot = -1);

    [[nodiscard]] const UpwaysPassTimings& getLastPassTimings() const { return m_lastTimings; }
    [[nodiscard]] std::array<float, 5> getPassTimingsMs() const {
        return {
            static_cast<float>(m_lastTimings.temporalMs),
            static_cast<float>(m_lastTimings.neuralGuideMs),
            static_cast<float>(m_lastTimings.upscaleResolveMs),
            static_cast<float>(m_lastTimings.historyCopyMs),
            0.0f
        };
    }

    Image* getOutputImage() const { return m_outputImage.get(); }
    Image* getConfidenceImage() const { return m_confidenceImage.get(); }
    uint32_t getInputWidth() const { return m_inputWidth; }
    uint32_t getInputHeight() const { return m_inputHeight; }
    uint32_t getOutputWidth() const { return m_outputWidth; }
    uint32_t getOutputHeight() const { return m_outputHeight; }
    bool isSuperResEnabled() const { return m_superRes; }
    bool isInitialized() const { return m_initialized; }

private:
    void initBuffers(const std::string& weightsPath);
    void initImages();
    void createDescriptorSetLayouts();
    void allocateDescriptorSets();
    void refreshDescriptors();
    void createPipelines(
        const std::vector<char>& temporalSpv,
        const std::vector<char>& neuralGuideSpv,
        const std::vector<char>& upscaleResolveSpv
    );
    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    double m_timestampPeriodNs = 1.0;
    uint32_t m_inputWidth = 0;
    uint32_t m_inputHeight = 0;
    uint32_t m_outputWidth = 0;
    uint32_t m_outputHeight = 0;
    uint32_t m_quadWidth = 0;
    uint32_t m_quadHeight = 0;
    bool m_superRes = false;
    VkFormat m_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    bool m_initialized = false;
    bool m_initialLayoutsTransitioned = false;
    uint32_t m_pingPongIndex = 0;

    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    bool m_slotQueriesRecorded[MAX_QUERY_SLOTS] = { false, false };
    UpwaysPassTimings m_lastTimings{};

    // External image handles & views cached for descriptor refresh across resizes
    VkImage m_srcNormalDepthImage = VK_NULL_HANDLE;
    VkImage m_srcSpecMotionImage = VK_NULL_HANDLE;
    VkImageView m_cachedDemodDiffuseView = VK_NULL_HANDLE;
    VkImageView m_cachedDemodSpecularView = VK_NULL_HANDLE;
    VkImageView m_cachedNormalDepthView = VK_NULL_HANDLE;
    VkImageView m_cachedAlbedoRoughnessView = VK_NULL_HANDLE;
    VkImageView m_cachedSurfaceMotionView = VK_NULL_HANDLE;
    VkImageView m_cachedSpecularMotionView = VK_NULL_HANDLE;

    // Buffers
    std::unique_ptr<Buffer> m_weightBuffer;
    std::unique_ptr<Buffer> m_featureBuffer;
    std::unique_ptr<Buffer> m_upscaleKernelBuffer;

    // Images
    std::unique_ptr<Image> m_outputImage;
    std::unique_ptr<Image> m_confidenceImage;
    std::unique_ptr<Image> m_dummyBlackImage;
    std::unique_ptr<Image> m_dummyMotionImage;
    std::unique_ptr<Image> m_diffHistoryImages[2];
    std::unique_ptr<Image> m_specHistoryImages[2];
    std::unique_ptr<Image> m_momentsHistoryImages[2];
    std::unique_ptr<Image> m_displayHistoryImages[2];
    std::unique_ptr<Image> m_tempDiffImage;
    std::unique_ptr<Image> m_tempSpecImage;
    std::unique_ptr<Image> m_reprojDiffImage;
    std::unique_ptr<Image> m_reprojSpecImage;
    std::unique_ptr<Image> m_prevNormalDepthImage;
    std::unique_ptr<Image> m_prevSpecMotionImage;

    // Descriptor sets & layouts
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_temporalDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_neuralGuideDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_upscaleDescLayout = VK_NULL_HANDLE;

    VkDescriptorSet m_temporalDescSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet m_neuralGuideDescSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet m_upscaleDescSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    // Pipelines & layouts
    VkPipelineLayout m_temporalPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_neuralGuidePipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_upscalePipeLayout = VK_NULL_HANDLE;

    VkPipeline m_temporalPipeline = VK_NULL_HANDLE;
    VkPipeline m_neuralGuidePipeline = VK_NULL_HANDLE;
    VkPipeline m_upscalePipeline = VK_NULL_HANDLE;
};

} // namespace pathways
