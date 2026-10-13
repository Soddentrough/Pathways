#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <memory>
#include <array>
#include <vector>
#include <functional>
#include <string>

#include "vulkan/Image.hpp"
#include "vulkan/Buffer.hpp"
#include "core/Config.hpp"

namespace pathways {

class PostProcessPipeline;

using ShaderLoaderFunc = std::function<std::vector<char>(const std::string&)>;

// =============================================================================
// Authoritative 5-Dispatch Dilated Edge-Aware SVGF (E-SVGF) Denoiser
// Reference: docs/design/adaptive_svgf_denoiser_proposal.md
//
//   Pass 1: denoise_fused_temporal.comp            (16x16 wg, 22x22 LDS apron,
//           in-LDS Sobel + tile thresholding + dual-model reprojection +
//           firefly soft-clamping + history + dual-channel edge density)
//   Pass 2: denoise_atrous_step1.comp              (s=1,  5 px, radiance+variance)
//   Pass 3: denoise_atrous_step2.comp              (s=2, 13 px, covers k=9)
//   Pass 4: denoise_atrous_step3.comp              (s=4, 29 px, covers k=17,
//           subgroupAll dense-edge bypass + smoothstep crossfade)
//   Pass 5: denoise_atrous_coarse_remodulate.comp  (s=8, 61 px, covers k=33,
//           subgroupAll moderate bypass + fused albedo remodulation)
// =============================================================================
class DenoiserPipeline {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr uint32_t NUM_PASSES = 5;

    // Camera matrices bound via CameraMatricesUBO at Binding 0 of Pass 1
    // (keeps push constants <= 80 bytes; std140 binary compatible, 160 bytes).
    struct alignas(16) CameraMatrices {
        float currViewToPrevView[16];   // 4x4 matrix, 64 bytes (offset 0)
        struct { float x, y, z, pad; } rotCols[3]; // 3x16B columns (mat3 std140), 48 bytes (offset 64)
        float currCameraJitter[2];      // UV units (offset 112)
        float prevCameraJitter[2];      // UV units (offset 120)
        float tanHalfFovY;              // offset 128
        float aspectRatio;              // offset 132
        float prevTanHalfFovY;          // offset 136
        float prevAspectRatio;          // offset 140
        float padTail[4];               // 16 bytes (offset 144)
    };
    static_assert(sizeof(CameraMatrices) == 160, "CameraMatricesUBO must be exactly 160 bytes");
    static_assert(offsetof(CameraMatrices, rotCols) == 64, "rotCols offset mismatch");
    static_assert(offsetof(CameraMatrices, currCameraJitter) == 112, "currCameraJitter offset mismatch");
    static_assert(offsetof(CameraMatrices, tanHalfFovY) == 128, "tanHalfFovY offset mismatch");

    // Pass 1 push constants (80 bytes, strictly <= 128 B Vulkan guarantee)
    struct FusedTemporalPushConstants {
        uint32_t width;
        uint32_t height;
        uint32_t resetHistory;
        float    maxHistoryLength;          // Default: 32.0f
        float    weightAlbedo;              // Default: 0.50f
        float    weightDepth;               // Default: 0.30f
        float    weightNormal;              // Default: 0.20f
        float    sigmaMultiplier;           // Default: 1.20f
        float    noiseFloor;                // Default: 0.05f
        float    contrastFloor;             // Default: 0.02f
        float    sigmaMin;                  // Default: 0.02f
        float    temporalAlpha;             // Default: 0.15f
        float    normalThreshold;           // Default: 0.85f (static geometry)
        float    normalThresholdDynamic;    // Default: 0.50f (dynamic objects)
        float    depthThreshold;            // Default: 0.08f (static geometry)
        float    dynamicDepthThreshold;     // Default: 0.15f (dynamic objects)
        float    mvDiscrepancyThreshold;    // Default: 2.00f
        float    _pad[3];
    };
    static_assert(sizeof(FusedTemporalPushConstants) == 80, "FusedTemporalPushConstants must be 80 bytes");

    // Passes 2-4 push constants (32 bytes)
    struct AtrousPushConstants {
        uint32_t width;
        uint32_t height;
        int32_t  stepSize;        // 1, 2 or 4
        float    phiNormal;       // Default: 32.0f
        float    phiPlaneDepth;   // Default: 2.00f (metric tangent-plane scale)
        float    phiLuminance;    // Default: 4.00f
        float    tanHalfFovY;
        float    aspectRatio;
    };
    static_assert(sizeof(AtrousPushConstants) == 32, "AtrousPushConstants must be 32 bytes");

    // Pass 5 push constants (48 bytes)
    struct AtrousCoarseRemodulatePushConstants {
        uint32_t width;
        uint32_t height;
        int32_t  stepSize;               // 8
        float    phiNormal;              // Default: 32.0f
        float    phiPlaneDepth;          // Default: 2.00f
        float    phiLuminance;           // Default: 4.00f
        float    tanHalfFovY;
        float    aspectRatio;
        float    edgeThresholdLow;       // Default: 0.35f (ramp top)
        uint32_t accumulatedSamples;     // Used only if ENABLE_STATIONARY_PREVIEW
        float    padding[2];
    };
    static_assert(sizeof(AtrousCoarseRemodulatePushConstants) == 48, "AtrousCoarseRemodulatePushConstants must be 48 bytes");

    DenoiserPipeline(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VmaAllocator allocator,
        VkQueue queue,
        ShaderLoaderFunc shaderLoader,
        bool forceWave32 = false
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
        Image* accumImage,
        Image* specularMotion,
        const CameraMatrices& cameraMatrices
    );

    void initTonemapDescriptors(PostProcessPipeline* postProcess, Image* outputImage);
    void invalidateDescriptors() { m_descriptorsBound = false; }

    [[nodiscard]] Image* getDenoisedImage() const { return m_denoisedImage.get(); }
    [[nodiscard]] VkDescriptorSet getTonemapDescSet() const { return m_tonemapDescSet; }
    [[nodiscard]] bool isInitialized() const { return m_width > 0 && m_height > 0 && m_denoisedImage != nullptr; }

    // Per-pass GPU timestamps (ms) collected asynchronously from the internal
    // 20-query pool (10 queries x 2 frames in flight, indexed frameSlot * 10).
    // Order: fusedTemporal, atrousStep1, atrousStep2, atrousStep3, coarseRemodulate.
    [[nodiscard]] const std::array<float, NUM_PASSES>& getPassTimingsMs() const { return m_passTimingsMs; }

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
        Image* accumImage,
        Image* specularMotion
    );
    void collectQueryResults();
    std::array<bool, MAX_FRAMES_IN_FLIGHT> m_slotQueriesRecorded = { false, false };

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    ShaderLoaderFunc m_shaderLoader;
    bool m_forceWave32 = false;
    double m_timestampPeriodNs = 1.0;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    AccumFormat m_accumFormat = AccumFormat::RGBA16_SFLOAT;

    // Descriptor set layouts (one per dispatch)
    VkDescriptorSetLayout m_fusedTemporalDescSetLayout = VK_NULL_HANDLE;   // 19 bindings (UBO + 18 images)
    VkDescriptorSetLayout m_atrousStep1DescSetLayout = VK_NULL_HANDLE;     // 6 image bindings
    VkDescriptorSetLayout m_atrousStep2DescSetLayout = VK_NULL_HANDLE;     // 7 image bindings
    VkDescriptorSetLayout m_atrousStep3DescSetLayout = VK_NULL_HANDLE;     // 7 image bindings
    VkDescriptorSetLayout m_atrousCoarseRemodulateDescSetLayout = VK_NULL_HANDLE; // 6 image bindings

    VkDescriptorPool m_descPool = VK_NULL_HANDLE;

    // Descriptor sets, double-buffered across frames in flight
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_fusedTemporalDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_atrousStep1DescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_atrousStep2DescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_atrousStep3DescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_coarseRemodulateDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet m_tonemapDescSet = VK_NULL_HANDLE;

    // Pipeline layouts & pipelines
    VkPipelineLayout m_fusedTemporalPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_atrousStep1PipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_atrousStep2PipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_atrousStep3PipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_coarseRemodulatePipelineLayout = VK_NULL_HANDLE;

    VkPipeline m_fusedTemporalPipeline = VK_NULL_HANDLE;
    VkPipeline m_atrousStep1Pipeline = VK_NULL_HANDLE;
    VkPipeline m_atrousStep2Pipeline = VK_NULL_HANDLE;
    VkPipeline m_atrousStep3Pipeline = VK_NULL_HANDLE;
    VkPipeline m_coarseRemodulatePipeline = VK_NULL_HANDLE;

    // Camera matrices UBO (per frame in flight)
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_cameraMatricesUBO;

    // Internal storage images (17 total, 124 B/px unpadded)
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_edgeDensity;          // R8G8: R raw rho, G rho_effective
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_historyDiffuse;       // RGBA16F: RGB + V^d
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_historySpecular;      // RGBA16F: RGB + V^s
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_historyMoments;       // RGBA16F: M1d, M2d, M1s, M2s
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_historyLength;        // RGBA16F: Ndiff, Nspec, d_refl, dyn
    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_historyNormalDepth;   // RGBA16F: normal + linear depth

    std::array<std::unique_ptr<Image>, 2> m_pingPongDiffuse;
    std::array<std::unique_ptr<Image>, 2> m_pingPongSpecular;

    std::unique_ptr<Image> m_denoisedImage;
    std::unique_ptr<Image> m_specMotionFallback; // 1x1 black fallback when VMV unavailable

    // Internal GPU profiling: 20 timestamp queries (10 x MAX_FRAMES_IN_FLIGHT)
    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    std::array<float, NUM_PASSES> m_passTimingsMs = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };

    bool m_descriptorsBound = false;
};

} // namespace pathways
