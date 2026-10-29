#include "rt/SuperResolutionManager.hpp"
#include "core/Logger.hpp"
#include "rt/PostProcessPipeline.hpp"
#include "mgpu/MultiGpuManager.hpp"

#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace pathways {

SuperResolutionManager::SuperResolutionManager(VkDevice device,
                                               VkPhysicalDevice physicalDevice,
                                               VmaAllocator allocator,
                                               VkQueue queue,
                                               ShaderLoaderFunc shaderLoader)
    : m_device(device),
      m_physicalDevice(physicalDevice),
      m_allocator(allocator),
      m_queue(queue),
      m_shaderLoader(std::move(shaderLoader)) {
}

SuperResolutionManager::~SuperResolutionManager() {
    destroy();
}

void SuperResolutionManager::destroy() {
    destroyFsrResources();
    destroyFsrPipelines();
    destroyGatewayResources();
    destroyGatewayPipelines();
    destroyUpwaysResources();
    destroyUpwaysPipelines();
    destroyDescPool();
}

void SuperResolutionManager::createDescPool() {
    if (m_postProcessDescPool != VK_NULL_HANDLE) {
        return;
    }
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64 }
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 16;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_postProcessDescPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create post-processing / super-resolution descriptor pool");
    }
}

void SuperResolutionManager::destroyDescPool() {
    if (m_postProcessDescPool) {
        vkDestroyDescriptorPool(m_device, m_postProcessDescPool, nullptr);
        m_postProcessDescPool = VK_NULL_HANDLE;
    }
    m_tonemapUpwaysDescSet = VK_NULL_HANDLE;
    m_tonemapGatewayDescSet = VK_NULL_HANDLE;
    m_tonemapFsrDescSet = VK_NULL_HANDLE;
}

void SuperResolutionManager::init(const Config& config,
                                  PostProcessPipeline* postProcess,
                                  VkCommandBuffer cmd) {
    createDescPool();
    createUpwaysPipelines(config);
    createUpwaysResources(config, postProcess, cmd);
    createGatewayPipelines(config);
    createGatewayResources(config, postProcess, cmd);
    createFsrPipelines(config);
    createFsrResources(config, postProcess);
}

void SuperResolutionManager::createUpwaysPipelines(const Config& config) {
    if (!m_shaderLoader) return;
    try {
        auto temporalCode = m_shaderLoader("upways_temporal.comp.spv");
        auto neuralGuideCode = m_shaderLoader("upways_neural_guide.comp.spv");
        auto upscaleResolveCode = m_shaderLoader("upways_upscale_resolve.comp.spv");

        VkFormat format = (config.accum_format == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
        uint32_t inW = config.getRenderWidth();
        uint32_t inH = config.getRenderHeight();
        uint32_t outW = (config.upscaler_mode == UpscalerMode::FSR) ? inW : config.width;
        uint32_t outH = (config.upscaler_mode == UpscalerMode::FSR) ? inH : config.height;
        bool isSuperRes = (inW < outW || inH < outH);

        m_upwaysPipeline = std::make_unique<UpwaysPipeline>(
            m_device,
            m_physicalDevice,
            m_allocator,
            inW,
            inH,
            outW,
            outH,
            temporalCode,
            neuralGuideCode,
            upscaleResolveCode,
            config.upways_weights_path,
            isSuperRes,
            format
        );
        Logger::info("Upways 3-Pass Neural Reconstruction & Multi-Scale KPN Pipeline initialized successfully.");
    } catch (const std::exception& e) {
        if (config.upscaler_mode == UpscalerMode::Upways || config.upways_superres || config.denoiser_mode == DenoiserMode::Upways) {
            throw std::runtime_error(std::string("Upways was explicitly requested but initialization failed: ") + e.what());
        }
        Logger::warn("UpwaysPipeline initialization failed: {}", e.what());
    }
}

void SuperResolutionManager::destroyUpwaysPipelines() {
    m_upwaysPipeline.reset();
    m_tonemapUpwaysDescSet = VK_NULL_HANDLE;
}

void SuperResolutionManager::createUpwaysResources(const Config& /*config*/, PostProcessPipeline* postProcess, VkCommandBuffer cmd) {
    if (m_postProcessDescPool && postProcess && m_tonemapUpwaysDescSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo tmAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        tmAllocInfo.descriptorPool = m_postProcessDescPool;
        tmAllocInfo.descriptorSetCount = 1;
        VkDescriptorSetLayout layout = postProcess->getTonemapDescLayout();
        tmAllocInfo.pSetLayouts = &layout;
        if (vkAllocateDescriptorSets(m_device, &tmAllocInfo, &m_tonemapUpwaysDescSet) != VK_SUCCESS) {
            Logger::warn("Failed to allocate Tonemap Upways descriptor set");
            m_tonemapUpwaysDescSet = VK_NULL_HANDLE;
        }
    }

    if (m_upwaysPipeline && cmd != VK_NULL_HANDLE) {
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &beginInfo);

        m_upwaysPipeline->transitionInitialLayouts(cmd);

        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = cmd;

        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_queue);
    }
}

void SuperResolutionManager::destroyUpwaysResources() {
    m_displayAlbedoImage.reset();
    m_displayNormalsImage.reset();
}

void SuperResolutionManager::createGatewayPipelines(const Config& config) {
    if (!m_shaderLoader) return;
    try {
        auto temporalCode = m_shaderLoader("gateway_temporal.comp.spv");
        auto neuralGuideCode = m_shaderLoader("gateway_neural_guide.comp.spv");
        auto atrousCode = m_shaderLoader("gateway_atrous.comp.spv");
        auto upscaleResolveCode = m_shaderLoader("gateway_upscale_resolve.comp.spv");

        VkFormat format = (config.accum_format == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
        uint32_t inW = config.getRenderWidth();
        uint32_t inH = config.getRenderHeight();
        uint32_t outW = (config.upscaler_mode == UpscalerMode::FSR) ? inW : config.width;
        uint32_t outH = (config.upscaler_mode == UpscalerMode::FSR) ? inH : config.height;
        bool isSuperRes = (inW < outW || inH < outH);

        m_gatewayPipeline = std::make_unique<GatewayPipeline>(
            m_device,
            m_physicalDevice,
            m_allocator,
            inW,
            inH,
            outW,
            outH,
            temporalCode,
            neuralGuideCode,
            atrousCode,
            upscaleResolveCode,
            config.gateway_weights_path,
            isSuperRes,
            format
        );
        Logger::info("Gateway 4-Pass Hybrid Denoiser & Super-Resolution Pipeline initialized successfully.");
    } catch (const std::exception& e) {
        if (config.upscaler_mode == UpscalerMode::Gateway || config.gateway_superres || config.denoiser_mode == DenoiserMode::Gateway) {
            throw std::runtime_error(std::string("Gateway was explicitly requested but initialization failed: ") + e.what());
        }
        Logger::warn("GatewayPipeline initialization failed: {}", e.what());
    }
}

void SuperResolutionManager::destroyGatewayPipelines() {
    m_gatewayPipeline.reset();
    m_tonemapGatewayDescSet = VK_NULL_HANDLE;
}

void SuperResolutionManager::createGatewayResources(const Config& /*config*/, PostProcessPipeline* postProcess, VkCommandBuffer cmd) {
    if (m_postProcessDescPool && postProcess && m_tonemapGatewayDescSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo tmAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        tmAllocInfo.descriptorPool = m_postProcessDescPool;
        tmAllocInfo.descriptorSetCount = 1;
        VkDescriptorSetLayout layout = postProcess->getTonemapDescLayout();
        tmAllocInfo.pSetLayouts = &layout;
        if (vkAllocateDescriptorSets(m_device, &tmAllocInfo, &m_tonemapGatewayDescSet) != VK_SUCCESS) {
            Logger::warn("Failed to allocate Tonemap Gateway descriptor set");
            m_tonemapGatewayDescSet = VK_NULL_HANDLE;
        }
    }

    if (m_gatewayPipeline && cmd != VK_NULL_HANDLE) {
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &beginInfo);

        m_gatewayPipeline->transitionInitialLayouts(cmd);

        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = cmd;

        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_queue);
    }
}

void SuperResolutionManager::destroyGatewayResources() {
}

void SuperResolutionManager::createFsrPipelines(const Config& config) {
    if (config.upscaler_mode != UpscalerMode::FSR || !m_shaderLoader) {
        return;
    }

    try {
        auto upscaleCode = m_shaderLoader("fsr3_upscale.comp.spv");
        auto rcasCode = m_shaderLoader("fsr3_rcas.comp.spv");
        VkFormat format = (config.accum_format == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
        uint32_t renderW = config.getRenderWidth();
        uint32_t renderH = config.getRenderHeight();

        m_fsrUpscaler = std::make_unique<FsrUpscaler>(
            m_device,
            m_physicalDevice,
            m_allocator,
            renderW,
            renderH,
            config.width,
            config.height,
            upscaleCode,
            rcasCode,
            format,
            "[GPU 0 Primary Viewport]",
            config.fsr_version
        );

        Logger::info("AMD FSR Super-Resolution Pipeline initialized successfully.");
    } catch (const std::exception& e) {
        Logger::warn("FsrUpscaler initialization failed: {}", e.what());
    }
}

void SuperResolutionManager::destroyFsrPipelines() {
    m_fsrUpscaler.reset();
    m_tonemapFsrDescSet = VK_NULL_HANDLE;
}

void SuperResolutionManager::createFsrResources(const Config& config, PostProcessPipeline* postProcess) {
    if (config.upscaler_mode != UpscalerMode::FSR) {
        return;
    }

    if (m_postProcessDescPool && postProcess && m_tonemapFsrDescSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo tmAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        tmAllocInfo.descriptorPool = m_postProcessDescPool;
        tmAllocInfo.descriptorSetCount = 1;
        VkDescriptorSetLayout layout = postProcess->getTonemapDescLayout();
        tmAllocInfo.pSetLayouts = &layout;
        if (vkAllocateDescriptorSets(m_device, &tmAllocInfo, &m_tonemapFsrDescSet) != VK_SUCCESS) {
            Logger::warn("Failed to allocate Tonemap FSR descriptor set");
            m_tonemapFsrDescSet = VK_NULL_HANDLE;
        }
    }

    VkFormat format = (config.accum_format == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;

    if (!m_secAccumImage || m_secAccumImage->getWidth() != config.width || m_secAccumImage->getHeight() != config.height) {
        m_secAccumImage = std::make_unique<Image>(
            m_device, m_allocator, config.width, config.height,
            format,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
            VK_IMAGE_ASPECT_COLOR_BIT
        );
    }
}

void SuperResolutionManager::destroyFsrResources() {
    m_secAccumImage.reset();
}

void SuperResolutionManager::resize(uint32_t width, uint32_t height, float renderScale,
                                    UpscalerMode upscalerMode, DenoiserMode denoiserMode,
                                    bool upwaysSuperres, AccumFormat accumFormat,
                                    PostProcessPipeline* postProcess, VkCommandBuffer cmd,
                                    FsrVersion fsrVersion) {
    Config cfgStub{};
    cfgStub.width = width;
    cfgStub.height = height;
    cfgStub.render_scale = renderScale;
    cfgStub.upscaler_mode = upscalerMode;
    cfgStub.denoiser_mode = denoiserMode;
    cfgStub.upways_superres = upwaysSuperres;
    cfgStub.accum_format = accumFormat;
    cfgStub.fsr_version = fsrVersion;

    const uint32_t inW = cfgStub.getRenderWidth();
    const uint32_t inH = cfgStub.getRenderHeight();
    const uint32_t outW = (upscalerMode == UpscalerMode::FSR) ? inW : width;
    const uint32_t outH = (upscalerMode == UpscalerMode::FSR) ? inH : height;
    const bool isSuperRes = (inW < outW || inH < outH);

    destroyUpwaysResources();
    if (m_upwaysPipeline) {
        m_upwaysPipeline->resize(inW, inH, outW, outH, isSuperRes);
    }

    destroyGatewayResources();
    if (m_gatewayPipeline) {
        m_gatewayPipeline->resize(inW, inH, outW, outH, isSuperRes);
    }

    createUpwaysResources(cfgStub, postProcess, cmd);
    createGatewayResources(cfgStub, postProcess, cmd);
    destroyFsrResources();
    if (upscalerMode == UpscalerMode::FSR && !m_fsrUpscaler) {
        createFsrPipelines(cfgStub);
    }
    if (m_fsrUpscaler) {
        m_fsrUpscaler->setTargetVersion(fsrVersion);
        m_fsrUpscaler->resize(inW, inH, width, height);
    }
    createFsrResources(cfgStub, postProcess);
}

void SuperResolutionManager::updateDescriptors(PostProcessPipeline* postProcess,
                                              Image* accumImage, Image* outputImage,
                                              Image* normalDepthImage, Image* motionVectorImage,
                                              Image* mlAlbedoRoughnessImage, Image* mlSpecularMotionImage,
                                              Image* mlDiffuseImage, Image* mlSpecularImage,
                                              Image* frameImage0) {
    if (m_upwaysPipeline && accumImage && outputImage) {
        VkImageView normDepthView = normalDepthImage ? normalDepthImage->getImageView() : VK_NULL_HANDLE;
        VkImageView motionView = motionVectorImage ? motionVectorImage->getImageView() : VK_NULL_HANDLE;
        VkImageView albedoView = mlAlbedoRoughnessImage ? mlAlbedoRoughnessImage->getImageView() : VK_NULL_HANDLE;
        VkImageView specMotionView = mlSpecularMotionImage ? mlSpecularMotionImage->getImageView() : VK_NULL_HANDLE;
        VkImageView diffView = mlDiffuseImage ? mlDiffuseImage->getImageView() : (frameImage0 ? frameImage0->getImageView() : VK_NULL_HANDLE);
        VkImageView specView = mlSpecularImage ? mlSpecularImage->getImageView() : VK_NULL_HANDLE;

        m_upwaysPipeline->setHistorySourceImages(
            normalDepthImage ? normalDepthImage->getImage() : VK_NULL_HANDLE,
            mlSpecularMotionImage ? mlSpecularMotionImage->getImage() : VK_NULL_HANDLE
        );

        m_upwaysPipeline->updateDescriptors(
            diffView,
            specView,
            normDepthView,
            albedoView,
            motionView,
            specMotionView,
            VK_NULL_HANDLE,
            albedoView,
            normDepthView
        );

        if (m_tonemapUpwaysDescSet != VK_NULL_HANDLE && m_upwaysPipeline->getOutputImage() && postProcess) {
            postProcess->updateTonemapDescriptors(m_tonemapUpwaysDescSet, m_upwaysPipeline->getOutputImage(), outputImage);
        }
    }

    if (m_gatewayPipeline && accumImage && outputImage) {
        VkImageView normDepthView = normalDepthImage ? normalDepthImage->getImageView() : VK_NULL_HANDLE;
        VkImageView motionView = motionVectorImage ? motionVectorImage->getImageView() : VK_NULL_HANDLE;
        VkImageView albedoView = mlAlbedoRoughnessImage ? mlAlbedoRoughnessImage->getImageView() : VK_NULL_HANDLE;
        VkImageView specMotionView = mlSpecularMotionImage ? mlSpecularMotionImage->getImageView() : VK_NULL_HANDLE;
        VkImageView diffView = mlDiffuseImage ? mlDiffuseImage->getImageView() : (frameImage0 ? frameImage0->getImageView() : VK_NULL_HANDLE);
        VkImageView specView = mlSpecularImage ? mlSpecularImage->getImageView() : VK_NULL_HANDLE;

        m_gatewayPipeline->setHistorySourceImages(
            normalDepthImage ? normalDepthImage->getImage() : VK_NULL_HANDLE,
            mlSpecularMotionImage ? mlSpecularMotionImage->getImage() : VK_NULL_HANDLE
        );

        m_gatewayPipeline->updateDescriptors(
            diffView,
            specView,
            normDepthView,
            albedoView,
            motionView,
            specMotionView,
            VK_NULL_HANDLE,
            albedoView,
            normDepthView
        );

        if (m_tonemapGatewayDescSet != VK_NULL_HANDLE && m_gatewayPipeline->getOutputImage() && postProcess) {
            postProcess->updateTonemapDescriptors(m_tonemapGatewayDescSet, m_gatewayPipeline->getOutputImage(), outputImage);
        }
    }

    if (m_fsrUpscaler && outputImage && postProcess) {
        if (m_tonemapFsrDescSet != VK_NULL_HANDLE && m_fsrUpscaler->getOutputImage()) {
            postProcess->updateTonemapDescriptors(m_tonemapFsrDescSet, m_fsrUpscaler->getOutputImage(), outputImage);
        }
        if (m_fsrUpscaler->getOutputImage() && m_secAccumImage) {
            postProcess->updateBlendDescriptors(m_fsrUpscaler->getOutputImage(), m_secAccumImage.get());
        }
    }
}

bool SuperResolutionManager::dispatchUpways(VkCommandBuffer cmd, bool resetHistory,
                                            const Config& config, Camera* camera,
                                            uint32_t frameIndex, bool cameraMovedLastFrame,
                                            uint32_t accumulatedSamples,
                                            Image* /*mlAlbedoRoughnessImage*/, Image* /*normalDepthImage*/,
                                            PostProcessPipeline* postProcess, Image* outputImage) {
    if ((config.denoiser_mode != DenoiserMode::Upways && config.upscaler_mode != UpscalerMode::Upways) || !m_upwaysPipeline) {
        return false;
    }

    uint32_t inW = config.getRenderWidth();
    uint32_t inH = config.getRenderHeight();
    uint32_t outW = (config.upscaler_mode == UpscalerMode::FSR) ? inW : config.width;
    uint32_t outH = (config.upscaler_mode == UpscalerMode::FSR) ? inH : config.height;
    bool isSuperRes = (inW < outW || inH < outH);

    if (m_upwaysPipeline->getInputWidth() != inW || m_upwaysPipeline->getInputHeight() != inH ||
        m_upwaysPipeline->getOutputWidth() != outW || m_upwaysPipeline->getOutputHeight() != outH ||
        m_upwaysPipeline->isSuperResEnabled() != isSuperRes) {
        vkDeviceWaitIdle(m_device);
        m_upwaysPipeline->resize(inW, inH, outW, outH, isSuperRes);
        if (m_tonemapUpwaysDescSet != VK_NULL_HANDLE && m_upwaysPipeline->getOutputImage() && postProcess && outputImage) {
            postProcess->updateTonemapDescriptors(m_tonemapUpwaysDescSet, m_upwaysPipeline->getOutputImage(), outputImage);
        }
    }

    glm::mat4 currInvView(1.0f);
    glm::mat4 prevViewProj(1.0f);
    glm::mat4 invProj(1.0f);
    glm::mat4 prevView(1.0f);
    glm::vec2 jitter(0.0f);

    if (camera) {
        if (resetHistory) {
            camera->resetPrevViewProj();
        }
        bool enableJitter = isSuperRes;
        CameraUniform ubo = camera->getUniformData(
            frameIndex, 1, config.max_bounces, 0,
            enableJitter, inW, inH, 0, false
        );
        currInvView = ubo.viewInverse;
        prevViewProj = camera->getPrevViewProjMatrix();
        invProj = ubo.projInverse;
        prevView = camera->getPrevViewMatrix();
        jitter = glm::vec2(ubo.jitterOffset.x, ubo.jitterOffset.y);
    }

    uint32_t totalSamples = (config.progressive_accumulation && accumulatedSamples > 0 && !isSuperRes) ? accumulatedSamples : 1u;

    m_upwaysPipeline->recordFrame(
        cmd, frameIndex, resetHistory, cameraMovedLastFrame,
        currInvView, prevViewProj, invProj, prevView, jitter, totalSamples
    );
    return true;
}

bool SuperResolutionManager::dispatchGateway(VkCommandBuffer cmd, bool resetHistory,
                                             const Config& config, Camera* camera,
                                             uint32_t frameIndex, bool cameraMovedLastFrame,
                                             uint32_t accumulatedSamples,
                                             Image* /*mlAlbedoRoughnessImage*/, Image* /*normalDepthImage*/,
                                             PostProcessPipeline* postProcess, Image* outputImage) {
    if ((config.denoiser_mode != DenoiserMode::Gateway && config.upscaler_mode != UpscalerMode::Gateway) || !m_gatewayPipeline) {
        return false;
    }

    uint32_t inW = config.getRenderWidth();
    uint32_t inH = config.getRenderHeight();
    uint32_t outW = (config.upscaler_mode == UpscalerMode::FSR) ? inW : config.width;
    uint32_t outH = (config.upscaler_mode == UpscalerMode::FSR) ? inH : config.height;
    bool isSuperRes = (inW < outW || inH < outH);

    if (m_gatewayPipeline->getInputWidth() != inW || m_gatewayPipeline->getInputHeight() != inH ||
        m_gatewayPipeline->getOutputWidth() != outW || m_gatewayPipeline->getOutputHeight() != outH ||
        m_gatewayPipeline->isSuperResEnabled() != isSuperRes) {
        vkDeviceWaitIdle(m_device);
        m_gatewayPipeline->resize(inW, inH, outW, outH, isSuperRes);
        if (m_tonemapGatewayDescSet != VK_NULL_HANDLE && m_gatewayPipeline->getOutputImage() && postProcess && outputImage) {
            postProcess->updateTonemapDescriptors(m_tonemapGatewayDescSet, m_gatewayPipeline->getOutputImage(), outputImage);
        }
    }

    glm::mat4 currInvView(1.0f);
    glm::mat4 prevViewProj(1.0f);
    glm::mat4 invProj(1.0f);
    glm::mat4 prevView(1.0f);
    glm::vec2 jitter(0.0f);

    if (camera) {
        if (resetHistory) {
            camera->resetPrevViewProj();
        }
        bool enableJitter = isSuperRes;
        CameraUniform ubo = camera->getUniformData(
            frameIndex, 1, config.max_bounces, 0,
            enableJitter, inW, inH, 0, false
        );
        currInvView = ubo.viewInverse;
        prevViewProj = camera->getPrevViewProjMatrix();
        invProj = ubo.projInverse;
        prevView = camera->getPrevViewMatrix();
        jitter = glm::vec2(ubo.jitterOffset.x, ubo.jitterOffset.y);
    }

    uint32_t totalSamples = (config.progressive_accumulation && accumulatedSamples > 0 && !isSuperRes) ? accumulatedSamples : 1u;

    m_gatewayPipeline->recordFrame(
        cmd, frameIndex, resetHistory, cameraMovedLastFrame,
        currInvView, prevViewProj, invProj, prevView, jitter, totalSamples
    );
    return true;
}

bool SuperResolutionManager::dispatchFsr(VkCommandBuffer cmd, bool resetHistory,
                                         const Config& config, uint32_t frameIndex,
                                         bool cameraMovedLastFrame, uint32_t accumulatedSamples,
                                         Image* accumImage, Image* normalDepthImage, Image* motionVectorImage,
                                         MultiGpuManager* mgpu, Buffer* secTransferBuffer, uint32_t currentFrame,
                                         PostProcessPipeline* postProcess, Image* outputImage,
                                         const std::function<void()>& onDescriptorsInvalidated) {
    if (config.upscaler_mode != UpscalerMode::FSR) {
        return false;
    }
    if (!m_fsrUpscaler) {
        createFsrPipelines(config);
        createFsrResources(config, postProcess);
        updateDescriptors(postProcess, accumImage, outputImage, normalDepthImage, motionVectorImage,
                          nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    if (!m_fsrUpscaler || !accumImage) {
        return false;
    }

    m_fsrUpscaler->setTargetVersion(config.fsr_version);

    uint32_t renderW = config.getRenderWidth();
    uint32_t renderH = config.getRenderHeight();

    if (m_fsrUpscaler->getRenderWidth() != renderW || m_fsrUpscaler->getRenderHeight() != renderH ||
        m_fsrUpscaler->getDisplayWidth() != config.width || m_fsrUpscaler->getDisplayHeight() != config.height) {
        vkDeviceWaitIdle(m_device);
        m_fsrUpscaler->resize(renderW, renderH, config.width, config.height);
        updateDescriptors(postProcess, accumImage, outputImage, normalDepthImage, motionVectorImage,
                          nullptr, nullptr, nullptr, nullptr, nullptr);
        if (onDescriptorsInvalidated) {
            onDescriptorsInvalidated();
        }
    }

    const uint32_t fsrPhaseCount = getFsrJitterPhaseCount(renderW, config.width);

    VkBuffer secBuffer = VK_NULL_HANDLE;
    if (mgpu && (mgpu->isZeroCopyActive() || mgpu->isP2PDirectBarActive())) {
        uint32_t slot = config.double_buffered_shared_mem ? (currentFrame % 2) : 0;
        secBuffer = mgpu->getPrimarySharedBuffer(slot);
    } else if (secTransferBuffer) {
        secBuffer = secTransferBuffer->getBuffer();
    }

    bool isSampleBlend = (mgpu && mgpu->isMultiGpuActive() && config.mgpu_mode != MultiGpuMode::Off &&
                          config.mgpu_upscale_mode == MgpuUpscaleMode::SampleBlend &&
                          m_secAccumImage && secBuffer != VK_NULL_HANDLE);

    if (isSampleBlend) {
        UpscalerDispatchDesc descPrim{};
        descPrim.cmd = cmd;
        descPrim.colorIn = accumImage->getImageView();
        descPrim.depthIn = normalDepthImage ? normalDepthImage->getImageView() : VK_NULL_HANDLE;
        descPrim.motionVectorsIn = motionVectorImage ? motionVectorImage->getImageView() : VK_NULL_HANDLE;
        descPrim.colorOut = m_fsrUpscaler->getOutputImage()->getImageView();
        descPrim.renderWidth = renderW;
        descPrim.renderHeight = renderH;
        descPrim.displayWidth = config.width;
        descPrim.displayHeight = config.height;
        glm::vec2 jitter(0.0f);
        const bool isDenoisingRenderRes = (config.denoiser_mode == DenoiserMode::Svgf ||
                                           config.denoiser_mode == DenoiserMode::FfxRayRegeneration);
        if (config.upscaler_mode == UpscalerMode::FSR && !isDenoisingRenderRes) {
            if (!config.progressive_accumulation || cameraMovedLastFrame || accumulatedSamples <= 1) {
                jitter = getHaltonJitter(frameIndex, fsrPhaseCount);
            }
        }
        descPrim.jitterX = jitter.x;
        descPrim.jitterY = jitter.y;
        descPrim.enableSharpening = config.upscaler_sharpening;
        descPrim.sharpness = config.upscaler_sharpening ? config.upscaler_sharpness : 0.0f;
        descPrim.resetHistory = resetHistory;
        descPrim.cameraMoved = cameraMovedLastFrame;
        descPrim.frameIndex = frameIndex;
        descPrim.inputIsNormalized = true;
        descPrim.totalSamples = (config.progressive_accumulation && accumulatedSamples > 0) ? accumulatedSamples : 1u;
        m_fsrUpscaler->recordUpscale(descPrim);

        m_secAccumImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
        );

        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copyRegion.imageOffset = { 0, 0, 0 };
        copyRegion.imageExtent = { config.width, config.height, 1 };

        vkCmdCopyBufferToImage(cmd, secBuffer, m_secAccumImage->getImage(),
                               VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copyRegion);

        m_secAccumImage->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );

        VkMemoryBarrier2 f2bBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        f2bBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        f2bBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        f2bBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        f2bBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        f2bBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo f2bDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        f2bDep.memoryBarrierCount = 1;
        f2bDep.pMemoryBarriers = &f2bBarrier;
        vkCmdPipelineBarrier2(cmd, &f2bDep);

        if (postProcess && postProcess->hasBlendPipeline()) {
            PostProcessPipeline::Blend4KPushConstants blendPC{};
            blendPC.width = config.width;
            blendPC.height = config.height;
            blendPC.weightDst = 0.5f;
            blendPC.weightSrc = 0.5f;
            postProcess->recordBlend4K(cmd, blendPC);
        }
    } else {
        UpscalerDispatchDesc desc{};
        desc.cmd = cmd;
        desc.colorIn = accumImage->getImageView();
        desc.depthIn = normalDepthImage ? normalDepthImage->getImageView() : VK_NULL_HANDLE;
        desc.motionVectorsIn = motionVectorImage ? motionVectorImage->getImageView() : VK_NULL_HANDLE;
        desc.colorOut = m_fsrUpscaler->getOutputImage()->getImageView();
        desc.renderWidth = renderW;
        desc.renderHeight = renderH;
        desc.displayWidth = config.width;
        desc.displayHeight = config.height;
        glm::vec2 jitter(0.0f);
        const bool isDenoisingRenderRes = (config.denoiser_mode == DenoiserMode::Svgf ||
                                           config.denoiser_mode == DenoiserMode::FfxRayRegeneration);
        if (config.upscaler_mode == UpscalerMode::FSR && !isDenoisingRenderRes) {
            if (!config.progressive_accumulation || cameraMovedLastFrame || accumulatedSamples <= 1) {
                glm::vec2 jitterPrim = getHaltonJitter(frameIndex, fsrPhaseCount);
                bool isMgpuSampleParallel = (mgpu && mgpu->isMultiGpuActive() && config.mgpu_mode != MultiGpuMode::Off &&
                    (config.mgpu_mode == MultiGpuMode::SampleParallel ||
                     (config.mgpu_mode == MultiGpuMode::Auto && (config.spp > 1))));
                if (isMgpuSampleParallel) {
                    glm::vec2 jitterSec = getHaltonJitter(frameIndex + (fsrPhaseCount / 2), fsrPhaseCount);
                    jitter = (jitterPrim + jitterSec) * 0.5f;
                } else {
                    jitter = jitterPrim;
                }
            }
        }
        desc.jitterX = jitter.x;
        desc.jitterY = jitter.y;
        desc.enableSharpening = config.upscaler_sharpening;
        desc.sharpness = config.upscaler_sharpening ? config.upscaler_sharpness : 0.0f;
        desc.resetHistory = resetHistory;
        desc.cameraMoved = cameraMovedLastFrame;
        desc.frameIndex = frameIndex;
        desc.inputIsNormalized = true;
        desc.totalSamples = (config.progressive_accumulation && accumulatedSamples > 0) ? accumulatedSamples : 1u;

        m_fsrUpscaler->recordUpscale(desc);
    }
    return true;
}

} // namespace pathways
