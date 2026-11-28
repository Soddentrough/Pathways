#pragma once

#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "core/Config.hpp"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace pathways {

class PostProcessPipeline;

/**
 * 6-pass NRD ReBLUR-grade analytical spatiotemporal compute denoiser:
 *   Pass 0: reblur_preblur.comp     -> safe albedo demodulation, AREA_3X3 HitT reconstruction,
 *                                      energy-preserving firefly diffusion, world-space normal/depth
 *   Pass 1: reblur_temporal.comp    -> world-normal + mvZ disocclusion, jittered-grid reprojection,
 *                                      dual surface + curvature-aware specular VMV, fast-history antilag
 *   Pass 2: reblur_history_fix.comp -> subgroupAll fast-path + multi-scale bilateral strided gather (N < 4)
 *   Pass 3-4: reblur_atrous.comp    -> 5x ping-pong HitT/roughness/variance-guided A-Trous (s = 1,2,4,8,16)
 *   Pass 5: reblur_stabilize.comp   -> albedo remodulation, 3x3 Blackman-Harris unjitter, 5-tap Catmull-Rom
 *                                      + YCoCg variance AABB stabilization (bypassed when upscalerCoupled)
 */
class ReblurDenoiserPipeline {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr uint32_t NUM_PASSES = 6; // preblur, temporal, historyFix, atrous(5x), stabilize, total

    struct CameraMatrices {
        glm::mat4 view{1.0f};
        glm::mat4 prevView{1.0f};
        glm::mat4 viewInverse{1.0f};
        glm::mat4 prevViewInverse{1.0f};
        glm::vec2 currJitterUv{0.0f};
        glm::vec2 prevJitterUv{0.0f};
        float tanHalfFovY = 0.41421356f;
        float aspectRatio = 16.0f / 9.0f;
        float prevTanHalfFovY = 0.41421356f;
        float prevAspectRatio = 16.0f / 9.0f;
        bool upscalerCoupled = false;
    };

    struct CameraUboData {
        glm::mat4 viewToWorld{1.0f};
        glm::mat4 prevViewToWorld{1.0f};
        glm::mat4 currViewToPrevView{1.0f};
        glm::vec2 currJitterUv{0.0f};
        glm::vec2 prevJitterUv{0.0f};
        float tanHalfFovY = 0.41421356f;
        float aspectRatio = 16.0f / 9.0f;
        float prevTanHalfFovY = 0.41421356f;
        float prevAspectRatio = 16.0f / 9.0f;
    };
    static_assert(sizeof(CameraUboData) == 224, "CameraUboData must match std140 224-byte ReblurCameraUBO");
    static_assert(offsetof(CameraUboData, prevViewToWorld) == 64, "prevViewToWorld offset mismatch");
    static_assert(offsetof(CameraUboData, currViewToPrevView) == 128, "currViewToPrevView offset mismatch");
    static_assert(offsetof(CameraUboData, currJitterUv) == 192, "currJitterUv offset mismatch");
    static_assert(offsetof(CameraUboData, tanHalfFovY) == 208, "tanHalfFovY offset mismatch");

    struct PreblurPushConstants {
        uint32_t width;
        uint32_t height;
        uint32_t pad0;
        uint32_t pad1;
    };
    static_assert(sizeof(PreblurPushConstants) == 16, "PreblurPushConstants must be 16 bytes");

    struct TemporalPushConstants {
        uint32_t width;
        uint32_t height;
        uint32_t resetHistory;
        uint32_t upscalerCoupled;
        float maxDiffHistory;
        float maxSpecHistory;
        float pad0;
        float pad1;
    };
    static_assert(sizeof(TemporalPushConstants) == 32, "TemporalPushConstants must be 32 bytes");

    struct HistoryFixPushConstants {
        uint32_t width;
        uint32_t height;
        uint32_t pad0;
        uint32_t pad1;
    };
    static_assert(sizeof(HistoryFixPushConstants) == 16, "HistoryFixPushConstants must be 16 bytes");

    struct AtrousPushConstants {
        uint32_t width;
        uint32_t height;
        int32_t stepSize;
        uint32_t passIndex;
        float phiNormal;
        float phiPlaneDepth;
        float phiLumDiff;
        float phiLumSpec;
    };
    static_assert(sizeof(AtrousPushConstants) == 32, "AtrousPushConstants must be 32 bytes");

    struct StabilizePushConstants {
        uint32_t width;
        uint32_t height;
        uint32_t resetHistory;
        uint32_t upscalerCoupled;
    };
    static_assert(sizeof(StabilizePushConstants) == 16, "StabilizePushConstants must be 16 bytes");

    using ShaderLoader = std::function<std::vector<char>(const std::string&)>;

    ReblurDenoiserPipeline(
        VkDevice device,
        VkPhysicalDevice physDevice,
        VmaAllocator allocator,
        VkQueue queue,
        ShaderLoader shaderLoader,
        bool forceWave32 = false,
        VkPipelineCache pipelineCache = VK_NULL_HANDLE
    );
    ~ReblurDenoiserPipeline();

    ReblurDenoiserPipeline(const ReblurDenoiserPipeline&) = delete;
    ReblurDenoiserPipeline& operator=(const ReblurDenoiserPipeline&) = delete;

    void resize(uint32_t width, uint32_t height, AccumFormat accumFormat, VkCommandBuffer cmd = VK_NULL_HANDLE);
    void initTonemapDescriptors(PostProcessPipeline* postProcess, Image* outputImage);
    void invalidateDescriptors();

    void recordDenoise(
        VkCommandBuffer cmd,
        uint32_t frameIndex,
        bool resetHistory,
        Image* normalDepth,
        Image* motionVectors,
        Image* specularMotion,
        Image* albedoRoughness,
        Image* specAlbedo,
        Image* skipImage,
        Image* directDiffuse,
        Image* indirectDiffuse,
        Image* directSpecular,
        Image* indirectSpecular,
        const CameraMatrices& cameraMatrices
    );

    void collectQueryResults();
    [[nodiscard]] const std::array<double, NUM_PASSES>& getPassTimingsMs() const { return m_lastPassTimingsMs; }
    [[nodiscard]] Image* getDenoisedImage() const { return m_denoisedImage.get(); }
    [[nodiscard]] VkDescriptorSet getTonemapDescSet() const { return m_tonemapDescSet; }
    [[nodiscard]] bool isInitialized() const {
        return m_denoisedImage != nullptr &&
               m_preblurPipeline != VK_NULL_HANDLE &&
               m_temporalPipeline != VK_NULL_HANDLE &&
               m_historyFixPipeline != VK_NULL_HANDLE &&
               m_atrousPipeline != VK_NULL_HANDLE &&
               m_stabilizePipeline != VK_NULL_HANDLE;
    }
    [[nodiscard]] uint32_t getWidth() const { return m_width; }
    [[nodiscard]] uint32_t getHeight() const { return m_height; }

private:
    static constexpr uint32_t NUM_DESC_VARIANTS = MAX_FRAMES_IN_FLIGHT * 2; // frameSlot * 2 + historyParity
    static constexpr uint32_t QUERIES_PER_SLOT = 10; // 5 pass pairs; total = t[9] - t[0]

    void createSampler();
    void createDescriptorSetLayouts();
    void allocateDescriptorSets();
    void initPipelines();
    void updateVariantDescriptors(uint32_t frameSlot, uint32_t currParity, const std::array<VkImageView, 10>& extViews);
    void ensureGeneralLayout(VkCommandBuffer cmd, Image* img);
    void clearAndTransitionInternalImages(VkCommandBuffer cmd);
    void collectQueryResultsForSlot(uint32_t slot);

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    ShaderLoader m_shaderLoader;
    bool m_forceWave32 = false;
    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    AccumFormat m_accumFormat = AccumFormat::RGBA16_SFLOAT;

    bool m_historyValid = false;
    bool m_needsInitialClear = true;
    uint32_t m_historyParity = 0;

    VkSampler m_linearSampler = VK_NULL_HANDLE;

    VkDescriptorPool m_descPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_preblurDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_temporalDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_historyFixDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_atrousDescLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_stabilizeDescLayout = VK_NULL_HANDLE;

    std::array<VkDescriptorSet, NUM_DESC_VARIANTS> m_preblurDescSets{};
    std::array<VkDescriptorSet, NUM_DESC_VARIANTS> m_temporalDescSets{};
    std::array<VkDescriptorSet, NUM_DESC_VARIANTS> m_historyFixDescSets{};
    std::array<std::array<VkDescriptorSet, 2>, NUM_DESC_VARIANTS> m_atrousDescSets{};
    std::array<VkDescriptorSet, NUM_DESC_VARIANTS> m_stabilizeDescSets{};
    VkDescriptorSet m_tonemapDescSet = VK_NULL_HANDLE;

    VkPipelineLayout m_preblurPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_temporalPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_historyFixPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_atrousPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_stabilizePipelineLayout = VK_NULL_HANDLE;

    VkPipeline m_preblurPipeline = VK_NULL_HANDLE;
    VkPipeline m_temporalPipeline = VK_NULL_HANDLE;
    VkPipeline m_historyFixPipeline = VK_NULL_HANDLE;
    VkPipeline m_atrousPipeline = VK_NULL_HANDLE;
    VkPipeline m_stabilizePipeline = VK_NULL_HANDLE;

    // Per-frame-in-flight Camera UBOs (224 bytes std140)
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_cameraUbos;

    // Pass 0 intermediate outputs
    std::unique_ptr<Image> m_preDiff;
    std::unique_ptr<Image> m_preSpec;
    std::array<std::unique_ptr<Image>, 2> m_currWorldND;

    // Pass 1 double-buffered temporal history
    std::array<std::unique_ptr<Image>, 2> m_histDiff;
    std::array<std::unique_ptr<Image>, 2> m_histSpec;
    std::array<std::unique_ptr<Image>, 2> m_histMoments;
    std::array<std::unique_ptr<Image>, 2> m_histMeta;
    std::array<std::unique_ptr<Image>, 2> m_histHitT;

    // Passes 2-4 spatial ping-pong targets
    std::array<std::unique_ptr<Image>, 2> m_pingPongDiff;
    std::array<std::unique_ptr<Image>, 2> m_pingPongSpec;

    // Pass 5 unjittered stabilization history + final denoised HDR output
    std::array<std::unique_ptr<Image>, 2> m_stabHistory;
    std::unique_ptr<Image> m_denoisedImage;

    // 1x1 zero-initialized fallback for optional null inputs
    std::unique_ptr<Image> m_fallbackBlack1x1;
    bool m_fallbackCleared = false;

    std::array<bool, NUM_DESC_VARIANTS> m_variantBound{};
    std::array<std::array<VkImageView, 10>, NUM_DESC_VARIANTS> m_cachedVariantViews{};

    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    float m_timestampPeriodNs = 0.0f;
    std::array<bool, MAX_FRAMES_IN_FLIGHT> m_querySlotUsed = { false, false };
    std::array<double, NUM_PASSES> m_lastPassTimingsMs = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
};

} // namespace pathways
