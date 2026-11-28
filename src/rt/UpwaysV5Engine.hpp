#pragma once

#include "core/Config.hpp"
#include "rt/UpwaysPipeline.hpp"

#include <vk_mem_alloc.h>
#include <vulkan/vulkan.h>
#include <glm/glm.hpp>

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace pathways {

class Buffer;
class Image;

/**
 * Exact 128-byte push constant layout shared across all five Upways 5.0 compute shaders:
 *   upways_v5_prep.comp, upways_v5_unet_wmma.comp, upways_v5_affinity.comp,
 *   upways_v5_upscaler_wmma.comp, upways_v5_resolve.comp.
 */
struct UpwaysV5PushConstants {
    uint32_t  lrWidth;
    uint32_t  lrHeight;
    uint32_t  hrWidth;
    uint32_t  hrHeight;
    float     jitterX;
    float     jitterY;
    uint32_t  frameIndex;
    uint32_t  flags;
    uint32_t  inW;
    uint32_t  inH;
    uint32_t  outW;
    uint32_t  outH;
    uint32_t  off0;
    uint32_t  off1;
    uint32_t  bandLrY0;
    uint32_t  bandHrY0;
    uint32_t  fullLrHeight;
    uint32_t  fullHrHeight;
    uint32_t  keepHrYStart;
    uint32_t  keepHrYEnd;
    glm::vec4 prevToCurrViewQuat{ 0.0f, 0.0f, 0.0f, 1.0f };
    glm::vec4 currToPrevTransView{ 0.0f, 0.0f, 0.0f, 0.0f };
    glm::vec4 camMotionParams{ 1.0f, 0.0f, 0.0f, 0.0f };
};

static_assert(sizeof(UpwaysV5PushConstants) == 128, "UpwaysV5PushConstants must be exactly 128 bytes");

struct UpwaysV5Dims {
    uint32_t w0 = 1u, h0 = 1u;
    uint32_t w1 = 1u, h1 = 1u;
    uint32_t w2 = 1u, h2 = 1u;
    uint32_t w3 = 1u, h3 = 1u;
};

inline constexpr UpwaysV5Dims computeUpwaysV5Dims(uint32_t w0In, uint32_t h0In) noexcept {
    UpwaysV5Dims d{};
    d.w0 = (w0In > 0u) ? w0In : 1u;
    d.h0 = (h0In > 0u) ? h0In : 1u;
    d.w1 = (d.w0 + 1u) >> 1u;
    d.h1 = (d.h0 + 1u) >> 1u;
    d.w2 = (d.w1 + 1u) >> 1u;
    d.h2 = (d.h1 + 1u) >> 1u;
    d.w3 = (d.w2 + 1u) >> 1u;
    d.h3 = (d.h2 + 1u) >> 1u;
    return d;
}

/**
 * Upways 5.0 Pure-ML (158,400 params, 190,208 B FP8 E4M3fn / INT8 QAT) Vulkan 1.4 Wave32 WMMA Runtime.
 * Bit-faithful with UpwaysDenoiser + UpwaysUpscaler on Pathways' 6-image compact G-buffer (41 B/px)
 * with native fused mGPU Band-Parallel output and lossless DRS support.
 */
class UpwaysV5Engine {
public:
    static constexpr uint32_t MAX_QUERY_SLOTS     = 2u;
    static constexpr uint32_t QUERIES_PER_SLOT    = 5u;
    static constexpr uint32_t NUM_UNET_STAGES     = 12u;
    static constexpr uint32_t NUM_UPSCALER_STAGES = 3u;

    UpwaysV5Engine(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VmaAllocator allocator,
        uint32_t inputWidth,
        uint32_t inputHeight,
        uint32_t outputWidth,
        uint32_t outputHeight,
        const std::vector<char>& prepSpv,
        const std::vector<char>& unetWmmaSpv,
        const std::vector<char>& affinitySpv,
        const std::vector<char>& upscalerWmmaSpv,
        const std::vector<char>& resolveSpv,
        const std::string& weightsPath = "",
        bool enableSuperRes = false,
        VkFormat imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT,
        UpwaysComputePrecision precision = UpwaysComputePrecision::Fp8,
        uint32_t outputFullHeight = 0u,
        const std::vector<char>& stabilizeSpv = {}
    );
    ~UpwaysV5Engine();

    UpwaysV5Engine(const UpwaysV5Engine&) = delete;
    UpwaysV5Engine& operator=(const UpwaysV5Engine&) = delete;

    void transitionInitialLayouts(VkCommandBuffer cmd);

    void resize(
        uint32_t inputWidth,
        uint32_t inputHeight,
        uint32_t outputWidth,
        uint32_t outputHeight,
        bool enableSuperRes,
        uint32_t outputFullHeight = 0u
    );

    void updateDescriptors(
        VkImageView mlIrradianceView,
        VkImageView mlPrimaryGuideView,
        VkImageView mlSurfaceMvView,
        VkImageView mlDiffHitDistView = VK_NULL_HANDLE,
        VkImageView mlSpecVmvView = VK_NULL_HANDLE,
        VkImageView mlSkipCompactView = VK_NULL_HANDLE
    );

    void setBandPackTargets(
        VkBuffer slot0Buffer,
        VkDeviceSize slot0Size,
        VkBuffer slot1Buffer = VK_NULL_HANDLE,
        VkDeviceSize slot1Size = 0u
    );

    void recordFrame(
        VkCommandBuffer cmd,
        uint32_t frameIndex,
        bool resetHistory,
        glm::vec2 jitter,
        bool coupledWithExternalUpscaler = false,
        bool zeroResidualMode = false,
        float exposureScale = 1.0f,
        const UpwaysBandParams& band = {},
        const UpwaysCameraMotion& cameraMotion = {}
    );

    [[nodiscard]] Image* getOutputImage() const { return m_outputImage.get(); }
    [[nodiscard]] Image* getCleanRadLrImage() const { return m_cleanRadLr.get(); }
    [[nodiscard]] uint32_t getInputWidth() const { return m_inputWidth; }
    [[nodiscard]] uint32_t getInputHeight() const { return m_inputHeight; }
    [[nodiscard]] uint32_t getOutputWidth() const { return m_outputWidth; }
    [[nodiscard]] uint32_t getOutputHeight() const { return m_outputHeight; }
    [[nodiscard]] uint32_t getOutputFullHeight() const { return m_outputFullHeight; }
    [[nodiscard]] bool isSuperResEnabled() const { return m_enableSuperRes; }
    [[nodiscard]] bool isHistoryValid() const { return m_historyValid; }
    [[nodiscard]] UpwaysComputePrecision getPrecision() const { return m_precision; }
    [[nodiscard]] bool isInitialized() const {
        return m_prepPipeline != VK_NULL_HANDLE &&
               m_resolvePipeline != VK_NULL_HANDLE &&
               m_unetPipelines[0] != VK_NULL_HANDLE;
    }
    [[nodiscard]] const UpwaysPassTimings& getLastPassTimings() const { return m_lastTimings; }

private:
    void initWeights(const std::string& weightsPath);
    void initRenderResources(uint32_t slotMask);
    void initDisplayResources();
    void ensureActiveSlotSized();
    void refreshDescriptors();

    void createDescriptorSetLayouts();
    void allocateDescriptorSets();
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void createPipelines(
        const std::vector<char>& prepSpv,
        const std::vector<char>& unetWmmaSpv,
        const std::vector<char>& affinitySpv,
        const std::vector<char>& upscalerWmmaSpv,
        const std::vector<char>& resolveSpv,
        const std::vector<char>& stabilizeSpv
    );
    void collectQueryResults(uint32_t targetSlot, bool waitForResult);

    VkDevice         m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VmaAllocator     m_allocator = VK_NULL_HANDLE;

    uint32_t         m_inputWidth = 1u;
    uint32_t         m_inputHeight = 1u;
    uint32_t         m_outputWidth = 1u;
    uint32_t         m_outputHeight = 1u;
    uint32_t         m_outputFullHeight = 1u;
    bool             m_enableSuperRes = false;
    UpwaysComputePrecision m_precision = UpwaysComputePrecision::Fp8;
    VkFormat         m_imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT;
    UpwaysV5Dims     m_dims{};

    bool             m_hasSubgroupSizeControl = false;
    bool             m_hasComputeFullSubgroups = false;

    uint32_t               m_pingPongIndex = 0u;
    bool                   m_historyValid = false;
    bool                   m_hrTransitioned = false;
    bool                   m_transientsTransitioned = false;
    bool                   m_initialLayoutsTransitioned = false;
    std::array<bool, 2>    m_lrSlotTransitioned = { false, false };
    std::array<uint32_t, 2> m_slotWidth = { 0u, 0u };
    std::array<uint32_t, 2> m_slotHeight = { 0u, 0u };

    // Weights (190,208 B total split cleanly for descriptor SSBO bindings)
    std::unique_ptr<Buffer> m_weightBuffer;

    // LR & HR ping-pong history images
    std::array<std::unique_ptr<Image>, 2> m_diffHist;   // rgba16f
    std::array<std::unique_ptr<Image>, 2> m_specHist;   // rgba16f
    std::array<std::unique_ptr<Image>, 2> m_geomHist;   // rgba16f
    std::array<std::unique_ptr<Image>, 2> m_hiddenHist; // rg16f
    std::unique_ptr<Image> m_cleanRadLr;                // rgba16f
    std::array<std::unique_ptr<Image>, 2> m_colorHr;    // rgba16f (.rgb=hr, .a=conf)
    std::unique_ptr<Image> m_outputImage;               // rgba16f

    // Fallbacks
    std::unique_ptr<Image>  m_dummyRgba16fImage;
    std::unique_ptr<Image>  m_dummyRg16fImage;
    std::unique_ptr<Image>  m_dummyR32uiImage;
    std::unique_ptr<Image>  m_dummyRg32uiImage;
    std::unique_ptr<Image>  m_dummyRgba32uiImage;
    std::unique_ptr<Image>  m_dummyR8Image;
    std::unique_ptr<Buffer> m_dummySsboBuffer;
    std::unique_ptr<Buffer> m_dummyBandPackBuffer;
    std::array<VkBuffer, 2>     m_bandPackBuf = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDeviceSize, 2> m_bandPackSize = { 0u, 0u };

    // Intermediate SSBO buffers (per-stage, zero read-write hazard)
    std::unique_ptr<Buffer> m_x1Buf;
    std::unique_ptr<Buffer> m_s1Buf;
    std::unique_ptr<Buffer> m_s2Buf;
    std::unique_ptr<Buffer> m_unetABuf;
    std::unique_ptr<Buffer> m_unetBBuf;
    std::unique_ptr<Buffer> m_k3Buf;
    std::unique_ptr<Buffer> m_k2Buf;
    std::unique_ptr<Buffer> m_k1Buf;
    std::unique_ptr<Buffer> m_denL0Buf;
    std::unique_ptr<Buffer> m_pyr0Buf;
    std::unique_ptr<Buffer> m_pyr1Buf;
    std::unique_ptr<Buffer> m_pyr2Buf;
    std::unique_ptr<Buffer> m_pyr3Buf;
    std::unique_ptr<Buffer> m_hatABuf;
    std::unique_ptr<Buffer> m_hatBBuf;
    std::unique_ptr<Buffer> m_srInBuf;
    std::unique_ptr<Buffer> m_srFieldBuf;

    // Cached external G-buffer views
    VkImageView m_cachedMlIrradianceView = VK_NULL_HANDLE;
    VkImageView m_cachedMlPrimaryGuideView = VK_NULL_HANDLE;
    VkImageView m_cachedMlSurfaceMvView = VK_NULL_HANDLE;
    VkImageView m_cachedMlDiffHitDistView = VK_NULL_HANDLE;
    VkImageView m_cachedMlSpecVmvView = VK_NULL_HANDLE;
    VkImageView m_cachedMlSkipCompactView = VK_NULL_HANDLE;

    // Descriptors & pipelines
    VkSampler             m_linearSampler = VK_NULL_HANDLE;
    VkDescriptorPool      m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_prepDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_unetDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_affinityDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_upscalerDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_resolveDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_stabilizeDescLayout = VK_NULL_HANDLE;

    std::array<VkDescriptorSet, 2>               m_prepDescSets{};
    std::array<VkDescriptorSet, NUM_UNET_STAGES> m_unetDescSets{};
    std::array<VkDescriptorSet, 2>               m_affStage0DescSets{};
    std::array<VkDescriptorSet, 3>               m_affPoolDescSets{};   // L0->L1, L1->L2, L2->L3
    std::array<VkDescriptorSet, 3>               m_affReconDescSets{};  // L3, L2, L1
    std::array<std::array<VkDescriptorSet, 2>, 2> m_affL0DescSets{};    // [curr][pack]
    std::array<VkDescriptorSet, NUM_UPSCALER_STAGES> m_upscalerDescSets{};
    std::array<std::array<VkDescriptorSet, 2>, 2>    m_resolveDescSets{};   // [curr][pack]
    std::array<std::array<VkDescriptorSet, 2>, 2>    m_stabilizeDescSets{}; // [curr][pack]

    VkPipelineCache  m_pipelineCache = VK_NULL_HANDLE;
    VkPipelineLayout m_prepPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_unetPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_affinityPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_upscalerPipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_resolvePipeLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_stabilizePipeLayout = VK_NULL_HANDLE;

    VkPipeline m_prepPipeline = VK_NULL_HANDLE;
    std::array<VkPipeline, NUM_UNET_STAGES>     m_unetPipelines{};
    VkPipeline m_affStage0Pipeline = VK_NULL_HANDLE;
    VkPipeline m_affPoolPipeline = VK_NULL_HANDLE;
    VkPipeline m_affReconDil3Pipeline = VK_NULL_HANDLE;
    VkPipeline m_affReconDil1Pipeline = VK_NULL_HANDLE;
    VkPipeline m_affStage3Pipeline = VK_NULL_HANDLE;
    std::array<VkPipeline, NUM_UPSCALER_STAGES> m_upscalerPipelines{};
    VkPipeline m_resolvePipeline = VK_NULL_HANDLE;
    VkPipeline m_stabilizePipeline = VK_NULL_HANDLE;

    // Timestamps
    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    float       m_timestampPeriodNs = 1.0f;
    uint32_t    m_querySlot = 0u;
    std::array<bool, MAX_QUERY_SLOTS> m_querySlotSubmitted = { false, false };
    UpwaysPassTimings m_lastTimings{};
};

} // namespace pathways
