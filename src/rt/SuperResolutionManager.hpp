#pragma once

#include "core/Config.hpp"
#include "rt/UpwaysPipeline.hpp"
#include "rt/Fsr3Upscaler.hpp"
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
                PostProcessPipeline* postProcess, VkCommandBuffer cmd);

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

    bool dispatchFsr3(VkCommandBuffer cmd, bool resetHistory,
                      const Config& config, uint32_t frameIndex,
                      bool cameraMovedLastFrame, uint32_t accumulatedSamples,
                      Image* accumImage, Image* normalDepthImage, Image* motionVectorImage,
                      MultiGpuManager* mgpu, Buffer* secTransferBuffer, uint32_t currentFrame,
                      PostProcessPipeline* postProcess, Image* outputImage,
                      const std::function<void()>& onDescriptorsInvalidated);

    [[nodiscard]] UpwaysPipeline* getUpwaysPipeline() const { return m_upwaysPipeline.get(); }
    [[nodiscard]] Fsr3Upscaler* getFsr3Upscaler() const { return m_fsr3Upscaler.get(); }
    [[nodiscard]] Image* getDisplayAlbedoImage() const { return m_displayAlbedoImage.get(); }
    [[nodiscard]] Image* getDisplayNormalsImage() const { return m_displayNormalsImage.get(); }
    [[nodiscard]] Image* getSecAccumImage() const { return m_secAccumImage.get(); }
    [[nodiscard]] VkDescriptorSet getTonemapUpwaysDescSet() const { return m_tonemapUpwaysDescSet; }
    [[nodiscard]] VkDescriptorSet getTonemapFsr3DescSet() const { return m_tonemapFsr3DescSet; }
    [[nodiscard]] VkDescriptorPool getPostProcessDescPool() const { return m_postProcessDescPool; }

    void createUpwaysPipelines(const Config& config);
    void destroyUpwaysPipelines();
    void createUpwaysResources(const Config& config, PostProcessPipeline* postProcess, VkCommandBuffer cmd);
    void destroyUpwaysResources();

    void createFsr3Pipelines(const Config& config);
    void destroyFsr3Pipelines();
    void createFsr3Resources(const Config& config, PostProcessPipeline* postProcess);
    void destroyFsr3Resources();

private:
    VkDevice m_device = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;
    ShaderLoaderFunc m_shaderLoader;

    VkDescriptorPool m_postProcessDescPool = VK_NULL_HANDLE;

    std::unique_ptr<UpwaysPipeline> m_upwaysPipeline;
    VkDescriptorSet m_tonemapUpwaysDescSet = VK_NULL_HANDLE;
    std::unique_ptr<Image> m_displayAlbedoImage;
    std::unique_ptr<Image> m_displayNormalsImage;

    std::unique_ptr<Fsr3Upscaler> m_fsr3Upscaler;
    std::unique_ptr<Image> m_secAccumImage;
    VkDescriptorSet m_tonemapFsr3DescSet = VK_NULL_HANDLE;
};

} // namespace pathways
