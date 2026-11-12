#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "core/Config.hpp"

namespace pathways {

// Unified 56-byte push constant block shared across all 3 passes of Upways 3.0 (FP8 RDNA 4)
struct UpwaysPushConstants {
    uint32_t renderWidth;
    uint32_t renderHeight;
    uint32_t displayWidth;
    uint32_t displayHeight;
    float jitterX;
    float jitterY;
    float preExposure;
    uint32_t resetHistory;
    uint32_t preserveJitterGuide;
    uint32_t stageId;
    uint32_t srcWidth;
    uint32_t srcHeight;
    uint32_t dstWidth;
    uint32_t dstHeight;
};
static_assert(sizeof(UpwaysPushConstants) == 56, "UpwaysPushConstants must be 56 bytes");

struct UpwaysPyramidDims {
    uint32_t w0 = 1;
    uint32_t h0 = 1;
    uint32_t w1 = 1;
    uint32_t h1 = 1;
    uint32_t w2 = 1;
    uint32_t h2 = 1;
    VkDeviceSize totalBytes = 256;
};

inline constexpr uint32_t upwaysAlign256(uint32_t bytes) noexcept {
    return (bytes + 255u) & ~255u;
}

inline constexpr UpwaysPyramidDims computeUpwaysPyramidDims(uint32_t w0In, uint32_t h0In) noexcept {
    UpwaysPyramidDims d{};
    d.w0 = w0In > 1u ? w0In : 1u;
    d.h0 = h0In > 1u ? h0In : 1u;
    d.w1 = (d.w0 + 1u) >> 1u;
    d.h1 = (d.h0 + 1u) >> 1u;
    d.w2 = (d.w1 + 1u) >> 1u;
    d.h2 = (d.h1 + 1u) >> 1u;

    const uint32_t n0 = d.w0 * d.h0;
    const uint32_t n1 = d.w1 * d.h1;
    const uint32_t n2 = d.w2 * d.h2;

    uint32_t total = 0u;
    total += upwaysAlign256(32u * n0);  // offX0
    total += upwaysAlign256(16u * n0);  // offHw0
    total += upwaysAlign256(32u * n0);  // offS0
    total += upwaysAlign256(32u * n1);  // offL1In
    total += upwaysAlign256(64u * n1);  // offS1
    total += upwaysAlign256(64u * n2);  // offL2In
    total += upwaysAlign256(128u * n2); // offB1
    total += upwaysAlign256(64u * n2);  // offB2
    total += upwaysAlign256(32u * n1);  // offD1
    d.totalBytes = total > 256u ? static_cast<VkDeviceSize>(total) : static_cast<VkDeviceSize>(256u);
    return d;
}

struct UpwaysPassTimings {
    double temporalMs = 0.0;       // Pass 1: upways_temporal.comp (Demodulation, 3x3 Closest-Depth MV Dilation & FP8 L0 Marshaling)
    double neuralGuideMs = 0.0;    // Pass 2: upways_neural_guide.comp (6-Stage L0/L1/L2 FP8 E4M3 WMMA Spatiotemporal Backbone)
    double upscaleResolveMs = 0.0; // Pass 3: upways_upscale_resolve.comp (Continuous Subpixel Ray-Locus Gather & Catmull-Rom Resolve)
    double historyCopyMs = 0.0;    // Zero-copy ping-pong swap (0.0 ms; zero vkCmdCopyImage bytes)
    double totalMs = 0.0;          // Total Upways GPU time across all passes
    bool valid = false;
};

class UpwaysPipeline {
public:
    static constexpr uint32_t MAX_QUERY_SLOTS = 2;
    static constexpr uint32_t QUERIES_PER_SLOT = 5;
    static constexpr uint32_t NUM_NEURAL_STAGES = 6;
    static constexpr uint32_t NEURAL_HIDDEN_LAYERS = 4;

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

    // Retained as no-op for caller compatibility (Upways 3.0 uses zero-copy ping-pong history).
    void setHistorySourceImages(VkImage /*normalDepthImage*/, VkImage /*specularMotionImage*/) {}

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
        uint32_t totalSamples = 1u,
        bool coupledWithExternalUpscaler = false,
        bool heuristicFallback = false
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
        uint32_t totalSamples = 1u,
        bool heuristicFallback = false
    ) {
        recordFrame(
            cmd, frameIndex, resetHistory, cameraMoved,
            glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f), glm::mat4(1.0f),
            glm::vec2(0.0f), totalSamples, false, heuristicFallback
        );
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
    uint32_t getInputWidth() const { return m_inputWidth; }
    uint32_t getInputHeight() const { return m_inputHeight; }
    uint32_t getOutputWidth() const { return m_outputWidth; }
    uint32_t getOutputHeight() const { return m_outputHeight; }
    bool isSuperResEnabled() const { return m_superRes; }
    bool isInitialized() const { return m_initialized; }
    bool isHistoryValid() const { return m_historyValid; }

private:
    struct ArrayImage2D {
        VkImage image = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VkImageView view = VK_NULL_HANDLE;
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t layers = 0;
    };

    void initBuffers(const std::string& weightsPath);
    void initImages();
    void initRenderResources(uint32_t slotMask);
    void ensureActiveSlotSized();
    void createNeuralHiddenArray(ArrayImage2D& outImg, uint32_t width, uint32_t height, uint32_t layers);
    void destroyNeuralHiddenArrays();
    void createLinearSampler();
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
    UpwaysPyramidDims m_pyramidDims{};
    bool m_superRes = false;
    VkFormat m_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    bool m_initialized = false;
    bool m_initialLayoutsTransitioned = false;
    bool m_hrTransitioned = false;
    bool m_transientsTransitioned = false;
    bool m_lrSlotTransitioned[2] = { false, false };
    uint32_t m_lrSlotWidth[2] = { 0, 0 };
    uint32_t m_lrSlotHeight[2] = { 0, 0 };
    bool m_historyValid = false;
    uint32_t m_pingPongIndex = 0;

    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    bool m_slotQueriesRecorded[MAX_QUERY_SLOTS] = { false, false };
    UpwaysPassTimings m_lastTimings{};

    // External render-resolution G-buffer views cached across resizes
    VkImageView m_cachedDemodDiffuseView = VK_NULL_HANDLE;
    VkImageView m_cachedDemodSpecularView = VK_NULL_HANDLE;
    VkImageView m_cachedNormalDepthView = VK_NULL_HANDLE;
    VkImageView m_cachedAlbedoRoughnessView = VK_NULL_HANDLE;
    VkImageView m_cachedSurfaceMotionView = VK_NULL_HANDLE;
    VkImageView m_cachedSpecularMotionView = VK_NULL_HANDLE;
    VkImageView m_cachedAuxMetadataView = VK_NULL_HANDLE;

    // Hardware bilinear / 5-tap Catmull-Rom linear clamp sampler
    VkSampler m_linearSampler = VK_NULL_HANDLE;

    // Device-local SSBOs
    std::unique_ptr<Buffer> m_weightBuffer;
    std::unique_ptr<Buffer> m_pyramidBuffer;
    std::unique_ptr<Buffer> m_subpixelParamsBuffer;

    // Render-resolution transient images (RGBA16F)
    std::unique_ptr<Image> m_tLogDiff;
    std::unique_ptr<Image> m_tLogSpec;
    std::unique_ptr<Image> m_tGuideDiff;
    std::unique_ptr<Image> m_tGuideSpec;
    std::unique_ptr<Image> m_tDilatedMv;
    std::unique_ptr<Image> m_tTemporalConf;

    // Ping-pong render-resolution geometry & 4-layer 16-ch neural hidden history
    std::unique_ptr<Image> m_geomNd[2];
    std::unique_ptr<Image> m_geomHitT[2];
    ArrayImage2D m_neuralHidden[2];

    // Ping-pong display-resolution hardware-sampled irradiance & BRDF guide history
    std::unique_ptr<Image> m_irrDiffHr[2];
    std::unique_ptr<Image> m_irrSpecHr[2];
    std::unique_ptr<Image> m_guideDiffHr[2];
    std::unique_ptr<Image> m_guideSpecHr[2];

    // Resolved HDR output & 1x1 zero fallback image
    std::unique_ptr<Image> m_outputImage;
    std::unique_ptr<Image> m_dummyBlackImage;

    // Descriptor pool, layouts & ping-pong sets
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_temporalDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_neuralGuideDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_upscaleDescLayout = VK_NULL_HANDLE;

    VkDescriptorSet m_temporalDescSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet m_neuralGuideDescSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet m_upscaleDescSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    // Compute pipeline cache, layouts & stage-specialized pipelines
    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;
    VkPipelineLayout m_temporalPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_neuralGuidePipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_upscalePipeLayout = VK_NULL_HANDLE;

    VkPipeline m_temporalPipeline = VK_NULL_HANDLE;
    std::array<VkPipeline, NUM_NEURAL_STAGES> m_neuralGuidePipelines = {
        VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE,
        VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE
    };
    VkPipeline m_upscalePipeline = VK_NULL_HANDLE;
};

} // namespace pathways
