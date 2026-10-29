#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <string>

#include "rt/IUpscaler.hpp"
#include "vulkan/Image.hpp"
#include "core/Config.hpp"

namespace pathways {

struct FsrUpscalePushConstants {
    uint32_t renderWidth;
    uint32_t renderHeight;
    uint32_t displayWidth;
    uint32_t displayHeight;
    float invRenderWidth;
    float invRenderHeight;
    float invDisplayWidth;
    float invDisplayHeight;
    float invTotalSamples;
    float temporalWeight;
    float jitterX;
    float jitterY;
    uint32_t frameIndex;
    uint32_t resetHistory;
    uint32_t cameraMoved;
    uint32_t isInputNormalized;
    uint32_t fsrVersion;
};

struct FsrRcasPushConstants {
    uint32_t displayWidth;
    uint32_t displayHeight;
    float sharpness;
    uint32_t enableSharpening;
};

using Fsr3UpscalePushConstants = FsrUpscalePushConstants;
using Fsr3RcasPushConstants = FsrRcasPushConstants;

class FsrUpscaler : public IUpscaler {
public:
    FsrUpscaler(
        VkDevice device,
        VkPhysicalDevice physicalDevice,
        VmaAllocator allocator,
        uint32_t renderWidth,
        uint32_t renderHeight,
        uint32_t displayWidth,
        uint32_t displayHeight,
        const std::vector<char>& upscaleSpv,
        const std::vector<char>& rcasSpv,
        VkFormat format = VK_FORMAT_R16G16B16A16_SFLOAT,
        const std::string& instanceName = "GPU 0 Primary Viewport",
        FsrVersion targetVersion = FsrVersion::Auto
    );
    ~FsrUpscaler() override;

    FsrUpscaler(const FsrUpscaler&) = delete;
    FsrUpscaler& operator=(const FsrUpscaler&) = delete;

    void resize(uint32_t renderW, uint32_t renderH, uint32_t displayW, uint32_t displayH) override;
    void recordUpscale(const UpscalerDispatchDesc& desc) override;

    UpscalerMode getMode() const override { return UpscalerMode::FSR; }
    const char* getName() const override { return "AMD FidelityFX Super Resolution"; }
    bool isTemporal() const override { return true; }

    FsrVersion getTargetVersion() const { return m_targetVersion; }
    void setTargetVersion(FsrVersion version);
    FsrVersion getActiveVersion() const { return m_activeVersion; }
    bool isHardwareMlSupported() const { return m_hwMlSupported; }
    bool isFsr3Supported() const { return m_sdkFsr3Supported; }
    bool isFsr4Supported() const { return m_sdkFsr4Supported; }
    bool isUsingSdkProvider() const { return m_ffxContext != nullptr; }
    const char* getActiveVersionName() const;
    uint64_t getVramUsageBytes() const { return m_vramUsageBytes; }

    Image* getOutputImage() const { return m_sharpenedImage.get(); }
    Image* getUpscaledIntermediateImage() const { return m_upscaledImage.get(); }
    uint32_t getRenderWidth() const { return m_renderWidth; }
    uint32_t getRenderHeight() const { return m_renderHeight; }
    uint32_t getDisplayWidth() const { return m_displayWidth; }
    uint32_t getDisplayHeight() const { return m_displayHeight; }

private:
    void probeHardwareCapabilities();
    void resolveActiveVersion();
    void initFfxSdkContext();
    void destroyFfxSdkContext();
    void initImages();
    void createDescriptorSetLayouts();
    void allocateDescriptorSets();
    void updateDescriptorSets(uint32_t slot, VkImageView inputColorView, VkImageView motionVectorsView, VkImageView depthView);
    void createPipelines(const std::vector<char>& upscaleSpv, const std::vector<char>& rcasSpv);
    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    uint32_t m_renderWidth = 0;
    uint32_t m_renderHeight = 0;
    uint32_t m_displayWidth = 0;
    uint32_t m_displayHeight = 0;
    VkFormat m_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    FsrVersion m_targetVersion = FsrVersion::Auto;
    FsrVersion m_activeVersion = FsrVersion::Fsr4;
    bool m_hwMlSupported = false;
    void* m_ffxContext = nullptr;
    bool m_sdkFsr3Supported = false;
    bool m_sdkFsr4Supported = false;
    uint64_t m_providerVersionId = 0;
    std::string m_providerVersionName;
    std::string m_activeVersionLabel;
    uint64_t m_vramUsageBytes = 0;
    uint64_t m_requiredFfxResources = 0;

    std::unique_ptr<Image> m_upscaledImage;
    std::unique_ptr<Image> m_sharpenedImage;
    std::unique_ptr<Image> m_historyImages[2];
    std::unique_ptr<Image> m_sdkColorInput;
    std::unique_ptr<Image> m_sdkDepthInput;
    std::unique_ptr<Image> m_sdkMotionVectorsInput;
    uint32_t m_historyIndex = 0;
    bool m_historyValid = false;

    VkDescriptorPool m_descPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_upscaleDescLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_upscalePipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_upscalePipeline = VK_NULL_HANDLE;
    VkDescriptorSet m_upscaleDescSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkDescriptorSet m_sdkPrepDescSets[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    VkDescriptorSetLayout m_rcasDescLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_rcasPipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_rcasPipeline = VK_NULL_HANDLE;
    VkDescriptorSet m_rcasDescSet = VK_NULL_HANDLE;

    VkImageView m_lastBoundInputViews[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImageView m_lastBoundMvViews[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    VkImageView m_lastBoundDepthViews[2] = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    uint32_t m_nextPrepSlot = 0;
    std::string m_instanceName = "GPU 0 Primary Viewport";
};

using Fsr3Upscaler = FsrUpscaler;

} // namespace pathways
