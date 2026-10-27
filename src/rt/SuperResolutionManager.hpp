#pragma once

#include "core/Config.hpp"
#include "rt/UpwaysPipeline.hpp"
#include "rt/GatewayPipeline.hpp"
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
                           ShaderLoaderFunc shaderLoader);
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
                FsrVersion fsrVersion = FsrVersion::Auto);

    void updateDescriptors(PostProcessPipeline* postProcess,
                           Image* accumImage, Image* outputImage,
                           Image* normalDepthImage, Image* motionVectorImage,
                           Image* mlAlbedoRoughnessImage, Image* mlSpecularMotionImage,
                           Image* mlDiffuseImage, Image* mlSpecularImage,
                           Image* frameImage0);

    bool dispatchUpways(VkCommandBuffer cmd, bool resetHistory,
                        const Config& config, Camera* camera,
                        uint32_t frameIndex, bool cameraMovedLastFrame,
                        uint32_t accumulatedSamples,
                        Image* mlAlbedoRoughnessImage, Image* normalDepthImage,
                        PostProcessPipeline* postProcess, Image* outputImage);

    bool dispatchGateway(VkCommandBuffer cmd, bool resetHistory,
                         const Config& config, Camera* camera,
                         uint32_t frameIndex, bool cameraMovedLastFrame,
                         uint32_t accumulatedSamples,
                         Image* mlAlbedoRoughnessImage, Image* normalDepthImage,
                         PostProcessPipeline* postProcess, Image* outputImage);

    bool dispatchFsr(VkCommandBuffer cmd, bool resetHistory,
                     const Config& config, uint32_t frameIndex,
                     bool cameraMovedLastFrame, uint32_t accumulatedSamples,
                     Image* accumImage, Image* normalDepthImage, Image* motionVectorImage,
                     MultiGpuManager* mgpu, Buffer* secTransferBuffer, uint32_t currentFrame,
                     PostProcessPipeline* postProcess, Image* outputImage,
                     const std::function<void()>& onDescriptorsInvalidated);

    bool dispatchFsr3(VkCommandBuffer cmd, bool resetHistory,
                      const Config& config, uint32_t frameIndex,
                      bool cameraMovedLastFrame, uint32_t accumulatedSamples,
                      Image* accumImage, Image* normalDepthImage, Image* motionVectorImage,
                      MultiGpuManager* mgpu, Buffer* secTransferBuffer, uint32_t currentFrame,
                      PostProcessPipeline* postProcess, Image* outputImage,
                      const std::function<void()>& onDescriptorsInvalidated) {
        return dispatchFsr(cmd, resetHistory, config, frameIndex, cameraMovedLastFrame, accumulatedSamples,
                           accumImage, normalDepthImage, motionVectorImage, mgpu, secTransferBuffer, currentFrame,
                           postProcess, outputImage, onDescriptorsInvalidated);
    }

    [[nodiscard]] UpwaysPipeline* getUpwaysPipeline() const { return m_upwaysPipeline.get(); }
    [[nodiscard]] GatewayPipeline* getGatewayPipeline() const { return m_gatewayPipeline.get(); }
    [[nodiscard]] FsrUpscaler* getFsrUpscaler() const { return m_fsrUpscaler.get(); }
    [[nodiscard]] FsrUpscaler* getFsr3Upscaler() const { return m_fsrUpscaler.get(); }
    [[nodiscard]] Image* getDisplayAlbedoImage() const { return m_displayAlbedoImage.get(); }
    [[nodiscard]] Image* getDisplayNormalsImage() const { return m_displayNormalsImage.get(); }
    [[nodiscard]] Image* getSecAccumImage() const { return m_secAccumImage.get(); }
    [[nodiscard]] VkDescriptorSet getTonemapUpwaysDescSet() const { return m_tonemapUpwaysDescSet; }
    [[nodiscard]] VkDescriptorSet getTonemapGatewayDescSet() const { return m_tonemapGatewayDescSet; }
    [[nodiscard]] VkDescriptorSet getTonemapFsrDescSet() const { return m_tonemapFsrDescSet; }
    [[nodiscard]] VkDescriptorSet getTonemapFsr3DescSet() const { return m_tonemapFsrDescSet; }
    [[nodiscard]] VkDescriptorPool getPostProcessDescPool() const { return m_postProcessDescPool; }

    void createUpwaysPipelines(const Config& config);
    void destroyUpwaysPipelines();
    void createUpwaysResources(const Config& config, PostProcessPipeline* postProcess, VkCommandBuffer cmd);
    void destroyUpwaysResources();

    void createGatewayPipelines(const Config& config);
    void destroyGatewayPipelines();
    void createGatewayResources(const Config& config, PostProcessPipeline* postProcess, VkCommandBuffer cmd);
    void destroyGatewayResources();

    void createFsrPipelines(const Config& config);
    void destroyFsrPipelines();
    void createFsrResources(const Config& config, PostProcessPipeline* postProcess);
    void destroyFsrResources();

    void createFsr3Pipelines(const Config& config) { createFsrPipelines(config); }
    void destroyFsr3Pipelines() { destroyFsrPipelines(); }
    void createFsr3Resources(const Config& config, PostProcessPipeline* postProcess) { createFsrResources(config, postProcess); }
    void destroyFsr3Resources() { destroyFsrResources(); }

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    ShaderLoaderFunc m_shaderLoader;

    VkDescriptorPool m_postProcessDescPool = VK_NULL_HANDLE;

    std::unique_ptr<UpwaysPipeline> m_upwaysPipeline;
    VkDescriptorSet m_tonemapUpwaysDescSet = VK_NULL_HANDLE;
    std::unique_ptr<GatewayPipeline> m_gatewayPipeline;
    VkDescriptorSet m_tonemapGatewayDescSet = VK_NULL_HANDLE;
    std::unique_ptr<Image> m_displayAlbedoImage;
    std::unique_ptr<Image> m_displayNormalsImage;

    std::unique_ptr<FsrUpscaler> m_fsrUpscaler;
    std::unique_ptr<Image> m_secAccumImage;
    VkDescriptorSet m_tonemapFsrDescSet = VK_NULL_HANDLE;
};

} // namespace pathways
