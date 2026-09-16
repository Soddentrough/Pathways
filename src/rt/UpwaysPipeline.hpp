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
    int32_t inputWidth;
    int32_t inputHeight;
    int32_t outputWidth;
    int32_t outputHeight;
    float invInputWidth;
    float invInputHeight;
    float invOutputWidth;
    float invOutputHeight;
    uint32_t frameIndex;
    uint32_t resetHistory;
    uint32_t cameraMoved;
    uint32_t superResMode;      // 0 = 1:1 Denoising, 1 = 2x Super-Resolution
    float depthThreshold;
    float normalThreshold;
    float blendAlpha;
    uint32_t pad;
};

class UpwaysPipeline {
public:
    UpwaysPipeline(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VmaAllocator allocator,
        uint32_t width,
        uint32_t height,
        const std::vector<char>& shaderSpv,
        const std::string& weightsPath = "",
        bool enableSuperRes = false,
        VkFormat imageFormat = VK_FORMAT_R16G16B16A16_SFLOAT
    );
    ~UpwaysPipeline();

    UpwaysPipeline(const UpwaysPipeline&) = delete;
    UpwaysPipeline& operator=(const UpwaysPipeline&) = delete;

    void resize(uint32_t width, uint32_t height);
    void updateDescriptors(
        VkImageView accumImageView,
        VkImageView normalDepthImageView,
        VkImageView motionVectorImageView,
        VkImageView albedoRoughnessImageView,
        VkImageView specularMotionImageView,
        VkImageView diffuseImageView,
        VkImageView specularImageView
    );

    void recordFrame(
        VkCommandBuffer cmd,
        uint32_t frameIndex,
        bool resetHistory,
        bool cameraMoved
    );

    void transitionInitialLayouts(VkCommandBuffer cmd);

    Image* getOutputImage() const { return m_outputImage.get(); }
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
    std::unique_ptr<Image> m_historyImages[2];

    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descLayout = VK_NULL_HANDLE;
    VkDescriptorSet m_descSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_pipeline = VK_NULL_HANDLE;
};

} // namespace pathways
