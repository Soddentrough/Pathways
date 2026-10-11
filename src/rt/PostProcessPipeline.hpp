#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <array>
#include <string>

#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "core/Config.hpp"

namespace pathways {

class PostProcessPipeline {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    struct TonemapPushConstants {
        float exposure = 1.0f;
        uint32_t totalSamples = 1;
        uint32_t applyACES = 1;
        uint32_t visualizeSplit = 0;
        uint32_t tileSize = 64;
        uint32_t displayMode = 0;
        float peakNits = 1000.0f;
        float paperWhiteNits = 200.0f;
    };
    static_assert(sizeof(TonemapPushConstants) == 32, "TonemapPushConstants must be 32 bytes");

    struct FusedAccumTonemapPushConstants {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t sampleCount = 1;
        float invSpp = 1.0f;
        float exposure = 1.0f;
        uint32_t applyACES = 1;
        uint32_t displayMode = 0;
        float peakNits = 1000.0f;
        float paperWhiteNits = 200.0f;
        uint32_t pad = 0;
    };
    static_assert(sizeof(FusedAccumTonemapPushConstants) == 40, "FusedAccumTonemapPushConstants must be 40 bytes");

    struct RunningAvgPushConstants {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t sampleCount = 1;
        float invSpp = 1.0f;
    };
    static_assert(sizeof(RunningAvgPushConstants) == 16, "RunningAvgPushConstants must be 16 bytes");

    struct Blend4KPushConstants {
        uint32_t width = 0;
        uint32_t height = 0;
        float weightDst = 0.5f;
        float weightSrc = 0.5f;
    };
    static_assert(sizeof(Blend4KPushConstants) == 16, "Blend4KPushConstants must be 16 bytes");

    struct DebugViewPushConstants {
        uint32_t outWidth = 0;   // Output image (display) dimensions
        uint32_t outHeight = 0;
        uint32_t resWidth = 0;   // Reservoir grid (render resolution) dimensions
        uint32_t resHeight = 0;
        uint32_t mode = 0;       // 0 = off, 1..6 see restir_debug_view.comp
    };
    static_assert(sizeof(DebugViewPushConstants) == 20, "DebugViewPushConstants must be 20 bytes");

    struct MergePushConstants {
        uint32_t width = 0;
        uint32_t height = 0;
        uint32_t secondarySpp = 0;
        uint32_t tileSize = 64;
        uint32_t formatMode = 0;
        uint32_t mergeMode = 0;
        uint32_t primarySpp = 1;
        uint32_t secDispatchWidth = 0; // Secondary tile buffer pitch in pixels (0 = auto-derive from tile grid)
        uint32_t mergeGbuffers = 0;    // 1 = also merge secondary G-buffers (motion vectors, normal/depth)
    };
    static_assert(sizeof(MergePushConstants) == 36, "MergePushConstants must be 36 bytes (must match accum_merge.comp PC layout)");

    PostProcessPipeline(
        VkDevice device,
        VmaAllocator allocator,
        bool hasSubgroupSizeControl,
        VkDescriptorPool globalPool,
        const std::vector<char>& tonemapSpv,
        const std::vector<char>& fusedAccumTonemapSpv,
        const std::vector<char>& runningAvgSpv,
        const std::vector<char>& blendSpv,
        const std::vector<char>& mergeSpv,
        const std::vector<char>& restirDebugSpv
    );
    ~PostProcessPipeline();

    PostProcessPipeline(const PostProcessPipeline&) = delete;
    PostProcessPipeline& operator=(const PostProcessPipeline&) = delete;

    // Descriptors updates
    void updateTonemapDescriptors(VkDescriptorSet descSet, Image* inImage, Image* outImage);
    void updateRunningAvgDescriptors(const std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT>& frameImages, Image* accumImage);
    void updateFusedAccumTonemapDescriptors(const std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT>& frameImages, Image* accumImage, Image* outputImage);
    void updateBlendDescriptors(Image* dstImage, Image* srcImage);
    // ReSTIR debug view: binds the CURRENT frame's reservoir slot + X1 context
    // grid (null buffers fall back to a dummy — callers should skip the
    // dispatch when ReSTIR is inactive) and the output image to overwrite.
    void updateDebugViewDescriptors(uint32_t slot, Buffer* diReservoirs, Buffer* giReservoirs, Buffer* x1Contexts, Image* outputImage);
    void updateMergeDescriptors(uint32_t slot, VkBuffer secBuffer, VkDeviceSize curSize, Image* frameImage, Image* mvImage, Image* normDepthImage, uint32_t width, uint32_t height, AccumFormat format);

    // Recording commands
    void recordTonemap(VkCommandBuffer cmd, VkDescriptorSet descSet, uint32_t width, uint32_t height, const TonemapPushConstants& pc);
    void recordFusedAccumTonemap(VkCommandBuffer cmd, uint32_t frameSlot, const FusedAccumTonemapPushConstants& pc);
    void recordRunningAvg(VkCommandBuffer cmd, uint32_t frameSlot, const RunningAvgPushConstants& pc);
    void recordBlend4K(VkCommandBuffer cmd, const Blend4KPushConstants& pc);
    void recordDebugView(VkCommandBuffer cmd, uint32_t slot, const DebugViewPushConstants& pc);
    void recordMerge(VkCommandBuffer cmd, uint32_t slot, const MergePushConstants& pc, uint32_t groupCountX, uint32_t groupCountY);

    VkDescriptorSetLayout getTonemapDescLayout() const { return m_tonemapDescLayout; }
    VkDescriptorSet getTonemapDescSet() const { return m_tonemapDescSet; }
    VkDescriptorSet getBlendDescSet() const { return m_fsr3BlendDescSet; }
    VkDescriptorSet getMergeDescSet(uint32_t slot) const { return m_mergeDescSets[slot]; }
    VkPipelineLayout getTonemapPipelineLayout() const { return m_tonemapPipelineLayout; }
    VkPipeline getTonemapPipeline() const { return m_tonemapPipeline; }
    VkPipelineLayout getMergePipelineLayout() const { return m_mergePipelineLayout; }
    VkPipeline getMergePipeline() const { return m_mergePipeline; }
    VkPipelineLayout getAccumRunningAvgPipelineLayout() const { return m_accumRunningAvgPipelineLayout; }
    VkPipeline getAccumRunningAvgPipeline() const { return m_accumRunningAvgPipeline; }
    VkPipelineLayout getAccumTonemapPipelineLayout() const { return m_accumTonemapPipelineLayout; }
    VkPipeline getAccumTonemapPipeline() const { return m_accumTonemapPipeline; }
    VkDescriptorSet getAccumRunningAvgDescSet(uint32_t slot) const { return m_accumRunningAvgDescSets[slot]; }
    VkDescriptorSet getAccumTonemapDescSet(uint32_t slot) const { return m_accumTonemapDescSets[slot]; }
    VkDescriptorSetLayout getAccumRunningAvgDescLayout() const { return m_accumRunningAvgDescLayout; }
    VkDescriptorSetLayout getAccumTonemapDescLayout() const { return m_accumTonemapDescLayout; }
    VkDescriptorSetLayout getFsr3BlendDescLayout() const { return m_fsr3BlendDescLayout; }
    VkDescriptorSetLayout getMergeDescLayout() const { return m_mergeDescLayout; }
    VkDescriptorSet getDebugViewDescSet(uint32_t slot) const { return m_restirDebugDescSets[slot]; }

    bool hasFusedPipeline() const { return m_accumTonemapPipeline != VK_NULL_HANDLE; }
    bool hasRunningAvgPipeline() const { return m_accumRunningAvgPipeline != VK_NULL_HANDLE; }
    bool hasTonemapPipeline() const { return m_tonemapPipeline != VK_NULL_HANDLE; }
    bool hasBlendPipeline() const { return m_fsr3BlendPipeline != VK_NULL_HANDLE; }
    bool hasMergePipeline() const { return m_mergePipeline != VK_NULL_HANDLE; }
    bool hasDebugViewPipeline() const { return m_restirDebugPipeline != VK_NULL_HANDLE; }

private:
    void createDescriptorLayouts();
    void allocateDescriptorSets();
    void createPipelines(
        const std::vector<char>& tonemapSpv,
        const std::vector<char>& fusedAccumTonemapSpv,
        const std::vector<char>& runningAvgSpv,
        const std::vector<char>& blendSpv,
        const std::vector<char>& mergeSpv
    );
    VkShaderModule createShaderModule(const std::vector<char>& code);
    void createDebugViewPipeline(const std::vector<char>& spv);

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    bool m_hasSubgroupSizeControl = false;
    VkDescriptorPool m_globalPool = VK_NULL_HANDLE;

    // Descriptors
    VkDescriptorSetLayout m_tonemapDescLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_tonemapDescSet = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_accumRunningAvgDescLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_accumRunningAvgDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    VkDescriptorSetLayout m_accumTonemapDescLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_accumTonemapDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    VkDescriptorSetLayout m_fsr3BlendDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_fsr3BlendDescPool = VK_NULL_HANDLE;
    VkDescriptorSet m_fsr3BlendDescSet = VK_NULL_HANDLE;

    VkDescriptorSetLayout m_mergeDescLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> m_mergeDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    VkDescriptorSetLayout m_restirDebugDescLayout = VK_NULL_HANDLE;
    VkDescriptorPool m_restirDebugDescPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_restirDebugDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::unique_ptr<Buffer> m_debugDummyBuffer;

    // Pipeline Layouts
    VkPipelineLayout m_tonemapPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_accumRunningAvgPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_accumTonemapPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_fsr3BlendPipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_mergePipelineLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_restirDebugPipelineLayout = VK_NULL_HANDLE;

    // Pipelines
    VkPipeline m_tonemapPipeline = VK_NULL_HANDLE;
    VkPipeline m_accumRunningAvgPipeline = VK_NULL_HANDLE;
    VkPipeline m_accumTonemapPipeline = VK_NULL_HANDLE;
    VkPipeline m_fsr3BlendPipeline = VK_NULL_HANDLE;
    VkPipeline m_mergePipeline = VK_NULL_HANDLE;
    VkPipeline m_restirDebugPipeline = VK_NULL_HANDLE;
};

} // namespace pathways
