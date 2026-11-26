#pragma once

#include "core/Config.hpp"
#include "rt/UpwaysPipeline.hpp"
#include "rt/FsrUpscaler.hpp"
#include "vulkan/Image.hpp"
#include "vulkan/Buffer.hpp"
#include "scene/Camera.hpp"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <memory>
#include <vector>
#include <functional>

namespace pathways {

class PostProcessPipeline;
class MultiGpuManager;

using ShaderLoaderFunc = std::function<std::vector<char>(const std::string&)>;

class SuperResolutionManager {
public:
    SuperResolutionManager(VkDevice device,
                           VkPhysicalDevice physicalDevice,
                           VmaAllocator allocator,
                           VkQueue queue,
                           ShaderLoaderFunc shaderLoader,
                           VkInstance instance = VK_NULL_HANDLE,
                           VkPipelineCache pipelineCache = VK_NULL_HANDLE);
    ~SuperResolutionManager();

    SuperResolutionManager(const SuperResolutionManager&) = delete;
    SuperResolutionManager& operator=(const SuperResolutionManager&) = delete;

    void init(const Config& config,
              PostProcessPipeline* postProcess,
              VkCommandBuffer cmd);

    void destroy();

    void createDescPool();
    void destroyDescPool();

    void resize(uint32_t width, uint32_t height, float renderScale,
                UpscalerMode upscalerMode, DenoiserMode denoiserMode,
                bool upwaysSuperres, AccumFormat accumFormat,
                PostProcessPipeline* postProcess, VkCommandBuffer cmd,
                FsrVersion fsrVersion = FsrVersion::Auto,
                UpwaysPrecision upwaysPrecision = UpwaysPrecision::Auto,
                UpwaysVersion upwaysVersion = UpwaysVersion::Auto,
                const std::string& upwaysWeightsPath = "");

    void updateDescriptors(PostProcessPipeline* postProcess,
                           Image* accumImage, Image* outputImage,
                           Image* normalDepthImage, Image* motionVectorImage,
                           Image* mlAlbedoRoughnessImage, Image* mlSpecularMotionImage,
                           Image* mlDiffuseImage, Image* mlSpecularImage,
                           Image* frameImage0,
                           Image* mlSkipImage = nullptr,
                           Image* mlSpecAlbedoImage = nullptr,
                           Image* mlIrradianceImage = nullptr,
                           Image* mlPrimaryGuideImage = nullptr,
                           Image* mlSurfaceMvImage = nullptr,
                           Image* mlDiffHitDistImage = nullptr,
                           Image* mlSpecVmvImage = nullptr,
                           Image* mlSkipCompactImage = nullptr);

    bool dispatchUpways(VkCommandBuffer cmd, bool resetHistory,
                        const Config& config, Camera* camera,
                        uint32_t frameIndex, bool cameraMovedLastFrame,
                        uint32_t accumulatedSamples,
                        Image* mlAlbedoRoughnessImage, Image* normalDepthImage,
                        PostProcessPipeline* postProcess, Image* outputImage,
                        const UpwaysBandSplit* bandSplit = nullptr);

    void dispatchUpwaysBandStitch(VkCommandBuffer cmd, uint32_t frameSlot,
                                  VkBuffer secPackedBuf, VkDeviceSize bufSize,
                                  uint32_t width, uint32_t dstYStart, uint32_t rowCount);

    void updateTonemapUpwaysBandDescriptor(uint32_t frameSlot,
                                           PostProcessPipeline* postProcess,
                                           Image* outputImage,
                                           VkBuffer stitchBuf,
                                           VkDeviceSize stitchSize);

    bool dispatchFsr(VkCommandBuffer cmd, bool resetHistory,
                     const Config& config, uint32_t frameIndex,
                     bool cameraMovedLastFrame, uint32_t accumulatedSamples,
                     Image* accumImage, Image* normalDepthImage, Image* motionVectorImage,
                     MultiGpuManager* mgpu, Buffer* secTransferBuffer, uint32_t currentFrame,
                     PostProcessPipeline* postProcess, Image* outputImage,
                     const std::function<void()>& onDescriptorsInvalidated,
                     Camera* camera = nullptr,
                     float frameTimeDeltaMs = 16.667f,
                     Image* rawFrameImage = nullptr,
                     bool inputIsDenoised = false);

    bool dispatchFsr3(VkCommandBuffer cmd, bool resetHistory,
                      const Config& config, uint32_t frameIndex,
                      bool cameraMovedLastFrame, uint32_t accumulatedSamples,
                      Image* accumImage, Image* normalDepthImage, Image* motionVectorImage,
                      MultiGpuManager* mgpu, Buffer* secTransferBuffer, uint32_t currentFrame,
                      PostProcessPipeline* postProcess, Image* outputImage,
                      const std::function<void()>& onDescriptorsInvalidated,
                      Camera* camera = nullptr,
                      float frameTimeDeltaMs = 16.667f,
                      Image* rawFrameImage = nullptr,
                      bool inputIsDenoised = false) {
        return dispatchFsr(cmd, resetHistory, config, frameIndex, cameraMovedLastFrame, accumulatedSamples,
                           accumImage, normalDepthImage, motionVectorImage, mgpu, secTransferBuffer, currentFrame,
                           postProcess, outputImage, onDescriptorsInvalidated,
                           camera, frameTimeDeltaMs, rawFrameImage, inputIsDenoised);
    }

    void releaseInactivePipelines(const Config& config);

    [[nodiscard]] UpwaysPipeline* getUpwaysPipeline() const { return m_upwaysPipeline.get(); }
    [[nodiscard]] FsrUpscaler* getFsrUpscaler() const { return m_fsrUpscaler.get(); }
    [[nodiscard]] FsrUpscaler* getFsr3Upscaler() const { return m_fsrUpscaler.get(); }
    [[nodiscard]] Image* getDisplayAlbedoImage() const { return m_displayAlbedoImage.get(); }
    [[nodiscard]] Image* getDisplayNormalsImage() const { return m_displayNormalsImage.get(); }
    [[nodiscard]] Image* getSecAccumImage() const { return m_secAccumImage.get(); }
    [[nodiscard]] VkDescriptorSet getTonemapUpwaysDescSet(uint32_t slot = 0) const { return m_tonemapUpwaysDescSets[slot % 2u]; }
    [[nodiscard]] VkDescriptorSet getTonemapFsrDescSet() const { return m_tonemapFsrDescSet; }
    [[nodiscard]] VkDescriptorSet getTonemapFsr3DescSet() const { return m_tonemapFsrDescSet; }
    [[nodiscard]] VkDescriptorPool getPostProcessDescPool() const { return m_postProcessDescPool; }

    void createUpwaysPipelines(const Config& config);
    void destroyUpwaysPipelines();
    void createUpwaysResources(const Config& config, PostProcessPipeline* postProcess, VkCommandBuffer cmd);
    void destroyUpwaysResources();

    void createFsrPipelines(const Config& config);
    void destroyFsrPipelines();
    void createFsrResources(const Config& config, PostProcessPipeline* postProcess);
    void destroyFsrResources();

    void createFsr3Pipelines(const Config& config) { createFsrPipelines(config); }
    void destroyFsr3Pipelines() { destroyFsrPipelines(); }
    void createFsr3Resources(const Config& config, PostProcessPipeline* postProcess) { createFsrResources(config, postProcess); }
    void destroyFsr3Resources() { destroyFsrResources(); }

private:
    void refreshCachedDescriptors();

    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    ShaderLoaderFunc m_shaderLoader;
    VkInstance m_instance = VK_NULL_HANDLE;
    VkPipelineCache m_pipelineCache = VK_NULL_HANDLE;

    VkDescriptorPool m_postProcessDescPool = VK_NULL_HANDLE;

    std::unique_ptr<UpwaysPipeline> m_upwaysPipeline;
    std::array<VkDescriptorSet, 2> m_tonemapUpwaysDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkImageView, 2> m_tonemapUpwaysBoundInView = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkImageView, 2> m_tonemapUpwaysBoundOutView = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkBuffer, 2> m_tonemapUpwaysBoundStitchBuf = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDeviceSize, 2> m_tonemapUpwaysBoundStitchSize = { 0, 0 };
    VkDescriptorSetLayout m_bandStitchDescLayout = VK_NULL_HANDLE;
    VkPipelineLayout m_bandStitchPipeLayout = VK_NULL_HANDLE;
    VkPipeline m_bandStitchPipeline = VK_NULL_HANDLE;
    VkDescriptorPool m_bandStitchDescPool = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, 2> m_bandStitchDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkBuffer, 2> m_bandStitchBoundBuf = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkImageView, 2> m_bandStitchBoundImg = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    std::array<VkDeviceSize, 2> m_bandStitchBoundSize = { 0, 0 };
    std::unique_ptr<Image> m_displayAlbedoImage;
    std::unique_ptr<Image> m_displayNormalsImage;

    std::unique_ptr<FsrUpscaler> m_fsrUpscaler;
    std::unique_ptr<Image> m_secAccumImage;
    VkDescriptorSet m_tonemapFsrDescSet = VK_NULL_HANDLE;
    glm::vec2 m_prevFsrJitter{ 0.0f, 0.0f };
    bool m_hasPrevFsrJitter = false;

    // Cached external frame targets for lazy on-demand pipeline instantiation
    PostProcessPipeline* m_cachedPostProcess = nullptr;
    Image* m_cachedAccumImage = nullptr;
    Image* m_cachedOutputImage = nullptr;
    Image* m_cachedNormalDepthImage = nullptr;
    Image* m_cachedMotionVectorImage = nullptr;
    Image* m_cachedMlAlbedoRoughnessImage = nullptr;
    Image* m_cachedMlSpecularMotionImage = nullptr;
    Image* m_cachedMlDiffuseImage = nullptr;
    Image* m_cachedMlSpecularImage = nullptr;
    Image* m_cachedFrameImage0 = nullptr;
    Image* m_cachedMlSkipImage = nullptr;
    Image* m_cachedMlSpecAlbedoImage = nullptr;
    Image* m_cachedMlIrradianceImage = nullptr;
    Image* m_cachedMlPrimaryGuideImage = nullptr;
    Image* m_cachedMlSurfaceMvImage = nullptr;
    Image* m_cachedMlDiffHitDistImage = nullptr;
    Image* m_cachedMlSpecVmvImage = nullptr;
    Image* m_cachedMlSkipCompactImage = nullptr;
};

} // namespace pathways
