#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <string>

#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "core/Config.hpp"

namespace pathways {

struct UpwaysPushConstants {
    glm::mat4 currInvView;     // 0..63: View to World
    glm::mat4 prevViewProj;    // 64..127: World to Previous Clip
    glm::mat4 invProj;         // 128..191: Clip to View
    glm::vec4 prevViewZ;       // 192..207: Row 2 of PrevView for axial depth
    glm::vec2 jitterOffset;    // 208..215: Subpixel camera jitter in render pixels
    glm::uvec2 renderRes;      // 216..223
    glm::uvec2 displayRes;     // 224..231
    glm::vec2 scaleFactor;     // 232..239: DisplayRes / RenderRes (e.g. 2.0, 2.0)
    uint32_t resetHistory;     // 240..243
    uint32_t frameIndex;       // 244..247
    float invTotalSamples;     // 248..251
    uint32_t totalSamples;     // 252..255
};
static_assert(sizeof(UpwaysPushConstants) == 256, "UpwaysPushConstants must be exactly 256 bytes");

class UpwaysPipeline {
public:
    UpwaysPipeline(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VmaAllocator allocator,
        uint32_t inputWidth,
        uint32_t inputHeight,
        uint32_t outputWidth,
        uint32_t outputHeight,
        const std::vector<char>& shaderSpv,
        const std::string& weightsPath = "",
        bool enableSuperRes = false,
        VkFormat imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT
    );
    ~UpwaysPipeline();

    UpwaysPipeline(const UpwaysPipeline&) = delete;
    UpwaysPipeline& operator=(const UpwaysPipeline&) = delete;

    void resize(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight, bool enableSuperRes);

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

    // Overload for backwards compatibility with legacy 10-parameter signature
    void updateDescriptors(
        VkImageView accumImageView,
        VkImageView normalDepthImageView,
        VkImageView motionVectorImageView,
        VkImageView albedoRoughnessImageView,
        VkImageView specularMotionImageView,
        VkImageView diffuseImageView,
        VkImageView specularImageView,
        VkBuffer restirReservoirBuffer,
        VkImageView confidenceOutputImageView,
        VkBuffer restirReservoirBuffer1
    ) {
        updateDescriptors(
            diffuseImageView ? diffuseImageView : accumImageView,
            specularImageView ? specularImageView : accumImageView,
            normalDepthImageView,
            albedoRoughnessImageView,
            motionVectorImageView,
            specularMotionImageView ? specularMotionImageView : motionVectorImageView,
            specularMotionImageView ? specularMotionImageView : normalDepthImageView,
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

    // Overload for backwards compatibility with legacy tile-based dispatch signature
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
    void createDescriptorSetLayout();
    void allocateDescriptorSets();
    void createPipeline(const std::vector<char>& shaderSpv);
    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    uint32_t m_inputWidth = 0;
    uint32_t m_inputHeight = 0;
    uint32_t m_outputWidth = 0;
    uint32_t m_outputHeight = 0;
    bool m_superRes = false;
    VkFormat m_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    bool m_initialized = false;
    bool m_initialLayoutsTransitioned = false;
    uint32_t m_pingPongIndex = 0;

    std::unique_ptr<Buffer> m_weightBuffer;
    std::unique_ptr<Image> m_outputImage;
    std::unique_ptr<Image> m_confidenceImage;
    std::unique_ptr<Image> m_dummyBlackImage;
    std::unique_ptr<Image> m_diffHistoryImages[2];
    std::unique_ptr<Image> m_specHistoryImages[2];
    std::unique_ptr<Image> m_normHistoryImages[2];
    VkSampler m_historySampler = VK_NULL_HANDLE;

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_descSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

} // namespace pathways
