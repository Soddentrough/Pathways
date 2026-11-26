#pragma once

#include "vulkan/Image.hpp"
#include "core/Config.hpp"
#include "rt/FfxSdkRuntime.hpp"

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
 * 3-stage AMD FSR Ray Regeneration (FFX Denoiser 1.2.1) compute pipeline:
 *   Pass 0: ffx_rr_prep.comp    -> signed linear depth, motion vectors, octahedral normals + roughness,
 *                                  non-encoded diffuse/specular albedos, demodulated 4-signal inputs
 *   Pass 1: ffx::Dispatch       -> FFX Denoiser 1.2.1 4-signal chain (or vkCmdCopyImage passthrough)
 *   Pass 2: ffx_rr_compose.comp -> recomposes denoised 4 signals with albedos + primary skip radiance
 */
class DenoiserPipeline {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;
    static constexpr uint32_t NUM_PASSES = 5;

    struct CameraMatrices {
        glm::mat4 view{ 1.0f };
        glm::mat4 prevView{ 1.0f };
        glm::mat4 viewInverse{ 1.0f };
        glm::mat4 projection{ 1.0f }; // View-to-clip WITHOUT Vulkan Y-flip (proj[1][1] > 0) for FFX Denoiser
        float nearZ = 0.1f;
        float farZ = 10000.0f;
        float jitterOffsets[2] = { 0.0f, 0.0f };             // FFX subpixel jitter offset in pixels (-pixelJitter)
        float cameraPositionDelta[3] = { 0.0f, 0.0f, 0.0f }; // PreviousPosition - CurrentPosition (world space)
    };

    struct PrepPushConstants {
        uint32_t width;
        uint32_t height;
        float nearZ;
        float farZ;
        glm::mat4 viewInverse;
    };
    static_assert(sizeof(PrepPushConstants) == 80, "PrepPushConstants must be 80 bytes");
    static_assert(offsetof(PrepPushConstants, viewInverse) == 16, "PrepPushConstants::viewInverse offset mismatch");

    struct ComposePushConstants {
        uint32_t width;
        uint32_t height;
        uint32_t pad0;
        uint32_t pad1;
    };
    static_assert(sizeof(ComposePushConstants) == 16, "ComposePushConstants must be 16 bytes");

    using ShaderLoader = std::function<std::vector<char>(const std::string&)>;

    DenoiserPipeline(
        VkDevice device,
        VkPhysicalDevice physDevice,
        VmaAllocator allocator,
        VkQueue queue,
        ShaderLoader shaderLoader,
        bool forceWave32 = false,
        VkPipelineCache pipelineCache = VK_NULL_HANDLE
    );
    ~DenoiserPipeline();

    DenoiserPipeline(const DenoiserPipeline&) = delete;
    DenoiserPipeline& operator=(const DenoiserPipeline&) = delete;

    void resize(uint32_t width, uint32_t height, AccumFormat accumFormat, VkCommandBuffer cmd = VK_NULL_HANDLE);
    void initTonemapDescriptors(PostProcessPipeline* postProcess, Image* outputImage);
    void invalidateDescriptors();

    void recordDenoise(
        VkCommandBuffer cmd,
        uint32_t frameIndex,
        bool resetHistory,
        Image* normalDepth,
        Image* motionVectors,
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
    [[nodiscard]] bool isInitialized() const { return m_denoisedImage != nullptr && m_prepPipeline != VK_NULL_HANDLE; }
    [[nodiscard]] bool isUsingSdkProvider() const { return m_usingSdkProvider && m_ffxDenoiserContext != nullptr; }
    [[nodiscard]] uint32_t getWidth() const { return m_width; }
    [[nodiscard]] uint32_t getHeight() const { return m_height; }

private:
    void createDescriptorSetLayouts();
    void allocateDescriptorSets();
    void initPipelines();
    void initFfxDenoiserContext();
    void destroyFfxDenoiserContext();
    void updateSlotDescriptors(uint32_t slot, const std::array<VkImageView, 9>& extViews);
    void recordPassthroughCopy(VkCommandBuffer cmd);
    void ensureGeneralLayout(VkCommandBuffer cmd, Image* img);

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

    ffx::Context m_ffxDenoiserContext = nullptr;
    bool m_usingSdkProvider = false;
    bool m_historyValid = false;
    uint32_t m_ffxFrameIndex = 0;

    VkDescriptorPool m_descPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_prepDescSetLayout = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_composeDescSetLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_prepDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_composeDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet m_tonemapDescSet = VK_NULL_HANDLE;

    VkPipelineLayout m_prepPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_composePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_prepPipeline = VK_NULL_HANDLE;
    VkPipeline m_composePipeline = VK_NULL_HANDLE;

    // FFX RR intermediate guide & prepared 4-signal buffers
    std::unique_ptr<Image> m_rrLinearDepth;
    std::unique_ptr<Image> m_rrMotionVectors;
    std::unique_ptr<Image> m_rrNormals;
    std::unique_ptr<Image> m_rrDiffuseAlbedo;
    std::unique_ptr<Image> m_rrSpecularAlbedo;

    std::unique_ptr<Image> m_prepDirectDiffuse;
    std::unique_ptr<Image> m_prepIndirectDiffuse;
    std::unique_ptr<Image> m_prepDirectSpecular;
    std::unique_ptr<Image> m_prepIndirectSpecular;

    // FFX RR denoised outputs (also passthrough copy destinations)
    std::unique_ptr<Image> m_rrOutDirectDiffuse;
    std::unique_ptr<Image> m_rrOutIndirectDiffuse;
    std::unique_ptr<Image> m_rrOutDirectSpecular;
    std::unique_ptr<Image> m_rrOutIndirectSpecular;

    // Final composed linear HDR radiance output
    std::unique_ptr<Image> m_denoisedImage;

    // 1x1 zero-initialized fallback for optional null inputs
    std::unique_ptr<Image> m_fallbackBlack1x1;
    bool m_fallbackCleared = false;

    std::array<bool, MAX_FRAMES_IN_FLIGHT> m_slotBound = { false, false };
    std::array<std::array<VkImageView, 9>, MAX_FRAMES_IN_FLIGHT> m_cachedSlotViews{};

    VkQueryPool m_queryPool = VK_NULL_HANDLE;
    float m_timestampPeriodNs = 0.0f;
    std::array<bool, MAX_FRAMES_IN_FLIGHT> m_querySlotUsed = { false, false };
    std::array<double, NUM_PASSES> m_lastPassTimingsMs = { 0.0, 0.0, 0.0, 0.0, 0.0 };
};

} // namespace pathways
