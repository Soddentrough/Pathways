#include "rt/SuperResolutionManager.hpp"
#include "core/Logger.hpp"
#include "rt/PostProcessPipeline.hpp"
#include "mgpu/MultiGpuManager.hpp"

#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace pathways {

namespace {
bool isUpwaysRequested(const Config& config) {
    return config.denoiser_mode == DenoiserMode::Upways ||
           config.upscaler_mode == UpscalerMode::Upways ||
           config.upways_superres;
}
} // namespace

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
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
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
    m_tonemapFsrDescSet = VK_NULL_HANDLE;
}

void SuperResolutionManager::init(const Config& config,
                                  PostProcessPipeline* postProcess,
                                  VkCommandBuffer cmd) {
    createDescPool();
    if (isUpwaysRequested(config)) {
        createUpwaysPipelines(config);
        createUpwaysResources(config, postProcess, cmd);
    }
    if (config.upscaler_mode == UpscalerMode::FSR) {
        createFsrPipelines(config);
        createFsrResources(config, postProcess);
    }
}

void SuperResolutionManager::releaseInactivePipelines(const Config& config) {
    if (!isUpwaysRequested(config) && m_upwaysPipeline) {
        destroyUpwaysResources();
        destroyUpwaysPipelines();
        Logger::info("UpwaysPipeline released (inactive, 0 MB idle VRAM).");
    }
    if (config.upscaler_mode != UpscalerMode::FSR && m_fsrUpscaler) {
        destroyFsrResources();
        destroyFsrPipelines();
        Logger::info("FsrUpscaler released (inactive, 0 MB idle VRAM).");
    }
}

void SuperResolutionManager::createUpwaysPipelines(const Config& config) {
    if (!isUpwaysRequested(config) || !m_shaderLoader) {
        return;
    }
    try {
        const bool useFp8 = UpwaysPipeline::supportsFp8CoopMatrix(m_physicalDevice);
        const char* ext = useFp8 ? ".comp.spv" : ".fp16.comp.spv";
        auto denoisePrepCode    = m_shaderLoader(std::string("upways_denoise_prep") + ext);
        auto denoiserWmmaCode   = m_shaderLoader(std::string("upways_denoiser_wmma") + ext);
        auto upscalerWmmaCode   = m_shaderLoader(std::string("upways_upscaler_wmma") + ext);
        auto upscaleResolveCode = m_shaderLoader(std::string("upways_upscale_resolve") + ext);

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
            denoisePrepCode,
            denoiserWmmaCode,
            upscalerWmmaCode,
            upscaleResolveCode,
            config.upways_weights_path,
            isSuperRes,
            format,
            useFp8
        );
        Logger::info("Upways 4.0 Neural Denoiser & Super-Resolution Pipeline initialized on demand.");
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Upways was requested but initialization failed: ") + e.what());
    }
}

void SuperResolutionManager::destroyUpwaysPipelines() {
    m_upwaysPipeline.reset();
    if (m_postProcessDescPool != VK_NULL_HANDLE && m_tonemapUpwaysDescSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(m_device, m_postProcessDescPool, 1, &m_tonemapUpwaysDescSet);
    }
    m_tonemapUpwaysDescSet = VK_NULL_HANDLE;
}

void SuperResolutionManager::createUpwaysResources(const Config& config, PostProcessPipeline* postProcess, VkCommandBuffer cmd) {
    if (!isUpwaysRequested(config)) {
        return;
    }
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

void SuperResolutionManager::createFsrPipelines(const Config& config) {
    if (config.upscaler_mode != UpscalerMode::FSR || !m_shaderLoader) {
        return;
    }

    try {
        auto sdkPrepCode = m_shaderLoader("ffx_sdk_prep.comp.spv");
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
            sdkPrepCode,
            format,
            "[GPU 0 Primary Viewport]",
            config.fsr_version
        );

        Logger::info("AMD FSR Super-Resolution Pipeline initialized on demand.");
    } catch (const std::exception& e) {
        Logger::warn("FsrUpscaler initialization failed: {}", e.what());
    }
}

void SuperResolutionManager::destroyFsrPipelines() {
    m_fsrUpscaler.reset();
    if (m_postProcessDescPool != VK_NULL_HANDLE && m_tonemapFsrDescSet != VK_NULL_HANDLE) {
        vkFreeDescriptorSets(m_device, m_postProcessDescPool, 1, &m_tonemapFsrDescSet);
    }
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

    if (config.mgpu_mode != MultiGpuMode::Off && config.mgpu_upscale_mode == MgpuUpscaleMode::SampleBlend) {
        VkFormat format = (config.accum_format == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
        if (!m_secAccumImage || m_secAccumImage->getWidth() != config.width || m_secAccumImage->getHeight() != config.height) {
            m_secAccumImage = std::make_unique<Image>(
                m_device, m_allocator, config.width, config.height,
                format,
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                VK_IMAGE_ASPECT_COLOR_BIT
            );
        }
    } else {
        m_secAccumImage.reset();
    }
}

void SuperResolutionManager::destroyFsrResources() {
    m_secAccumImage.reset();
    m_prevFsrJitter = glm::vec2(0.0f);
    m_hasPrevFsrJitter = false;
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

    releaseInactivePipelines(cfgStub);

    const uint32_t inW = cfgStub.getRenderWidth();
    const uint32_t inH = cfgStub.getRenderHeight();
    const uint32_t outW = (upscalerMode == UpscalerMode::FSR) ? inW : width;
    const uint32_t outH = (upscalerMode == UpscalerMode::FSR) ? inH : height;
    const bool isSuperRes = (inW < outW || inH < outH);

    destroyUpwaysResources();
    if (isUpwaysRequested(cfgStub)) {
        if (!m_upwaysPipeline) {
            createUpwaysPipelines(cfgStub);
        } else {
            m_upwaysPipeline->resize(inW, inH, outW, outH, isSuperRes);
        }
        createUpwaysResources(cfgStub, postProcess, cmd);
    }

    destroyFsrResources();
    m_prevFsrJitter = glm::vec2(0.0f);
    m_hasPrevFsrJitter = false;
    if (upscalerMode == UpscalerMode::FSR) {
        if (!m_fsrUpscaler) {
            createFsrPipelines(cfgStub);
        } else {
            m_fsrUpscaler->setTargetVersion(fsrVersion);
            m_fsrUpscaler->resize(inW, inH, width, height);
        }
        createFsrResources(cfgStub, postProcess);
    }
}

void SuperResolutionManager::updateDescriptors(PostProcessPipeline* postProcess,
                                              Image* accumImage, Image* outputImage,
                                              Image* normalDepthImage, Image* motionVectorImage,
                                              Image* mlAlbedoRoughnessImage, Image* mlSpecularMotionImage,
                                              Image* mlDiffuseImage, Image* mlSpecularImage,
                                              Image* frameImage0,
                                              Image* mlSkipImage,
                                              Image* mlSpecAlbedoImage) {
    m_cachedPostProcess = postProcess;
    m_cachedAccumImage = accumImage;
    m_cachedOutputImage = outputImage;
    m_cachedNormalDepthImage = normalDepthImage;
    m_cachedMotionVectorImage = motionVectorImage;
    m_cachedMlAlbedoRoughnessImage = mlAlbedoRoughnessImage;
    m_cachedMlSpecularMotionImage = mlSpecularMotionImage;
    m_cachedMlDiffuseImage = mlDiffuseImage;
    m_cachedMlSpecularImage = mlSpecularImage;
    m_cachedFrameImage0 = frameImage0;
    m_cachedMlSkipImage = mlSkipImage;
    m_cachedMlSpecAlbedoImage = mlSpecAlbedoImage;

    refreshCachedDescriptors();
}

void SuperResolutionManager::refreshCachedDescriptors() {
    if (m_upwaysPipeline && m_cachedAccumImage && m_cachedOutputImage) {
        VkImageView normDepthView = m_cachedNormalDepthImage ? m_cachedNormalDepthImage->getImageView() : VK_NULL_HANDLE;
        VkImageView motionView = m_cachedMotionVectorImage ? m_cachedMotionVectorImage->getImageView() : VK_NULL_HANDLE;
        VkImageView albedoView = m_cachedMlAlbedoRoughnessImage ? m_cachedMlAlbedoRoughnessImage->getImageView() : VK_NULL_HANDLE;
        VkImageView specMotionView = m_cachedMlSpecularMotionImage ? m_cachedMlSpecularMotionImage->getImageView() : VK_NULL_HANDLE;
        VkImageView diffView = m_cachedMlDiffuseImage ? m_cachedMlDiffuseImage->getImageView()
                             : (m_cachedFrameImage0 ? m_cachedFrameImage0->getImageView() : VK_NULL_HANDLE);
        VkImageView specView = m_cachedMlSpecularImage ? m_cachedMlSpecularImage->getImageView() : VK_NULL_HANDLE;
        VkImageView specAlbView = m_cachedMlSpecAlbedoImage ? m_cachedMlSpecAlbedoImage->getImageView() : VK_NULL_HANDLE;
        VkImageView skipView = m_cachedMlSkipImage ? m_cachedMlSkipImage->getImageView() : VK_NULL_HANDLE;

        m_upwaysPipeline->updateDescriptors(
            diffView,
            specView,
            albedoView,
            specAlbView,
            skipView,
            normDepthView,
            motionView,
            specMotionView
        );

        if (m_tonemapUpwaysDescSet != VK_NULL_HANDLE && m_upwaysPipeline->getOutputImage() && m_cachedPostProcess) {
            m_cachedPostProcess->updateTonemapDescriptors(m_tonemapUpwaysDescSet, m_upwaysPipeline->getOutputImage(), m_cachedOutputImage);
        }
    }

    if (m_fsrUpscaler && m_cachedOutputImage && m_cachedPostProcess) {
        if (m_tonemapFsrDescSet != VK_NULL_HANDLE && m_fsrUpscaler->getOutputImage()) {
            m_cachedPostProcess->updateTonemapDescriptors(m_tonemapFsrDescSet, m_fsrUpscaler->getOutputImage(), m_cachedOutputImage);
        }
        if (m_fsrUpscaler->getOutputImage() && m_secAccumImage) {
            m_cachedPostProcess->updateBlendDescriptors(m_fsrUpscaler->getOutputImage(), m_secAccumImage.get());
        }
    }
}

bool SuperResolutionManager::dispatchUpways(VkCommandBuffer cmd, bool resetHistory,
                                            const Config& config, Camera* camera,
                                            uint32_t frameIndex, bool cameraMovedLastFrame,
                                            uint32_t accumulatedSamples,
                                            Image* mlAlbedoRoughnessImage, Image* normalDepthImage,
                                            PostProcessPipeline* postProcess, Image* outputImage) {
    if (config.denoiser_mode != DenoiserMode::Upways && config.upscaler_mode != UpscalerMode::Upways) {
        return false;
    }

    if (!m_upwaysPipeline) {
        vkDeviceWaitIdle(m_device);
        createUpwaysPipelines(config);
        createUpwaysResources(config, postProcess ? postProcess : m_cachedPostProcess, VK_NULL_HANDLE);
        if (postProcess) m_cachedPostProcess = postProcess;
        if (outputImage) m_cachedOutputImage = outputImage;
        if (normalDepthImage) m_cachedNormalDepthImage = normalDepthImage;
        if (mlAlbedoRoughnessImage) m_cachedMlAlbedoRoughnessImage = mlAlbedoRoughnessImage;
        refreshCachedDescriptors();
    }
    if (!m_upwaysPipeline) {
        return false;
    }

    uint32_t inW = config.getRenderWidth();
    uint32_t inH = config.getRenderHeight();
    bool coupledWithExternalUpscaler = (config.upscaler_mode == UpscalerMode::FSR);
    uint32_t outW = coupledWithExternalUpscaler ? inW : config.width;
    uint32_t outH = coupledWithExternalUpscaler ? inH : config.height;
    bool isSuperRes = (inW < outW || inH < outH);

    if (m_upwaysPipeline->getInputWidth() != inW || m_upwaysPipeline->getInputHeight() != inH ||
        m_upwaysPipeline->getOutputWidth() != outW || m_upwaysPipeline->getOutputHeight() != outH ||
        m_upwaysPipeline->isSuperResEnabled() != isSuperRes) {
        vkDeviceWaitIdle(m_device);
        m_upwaysPipeline->resize(inW, inH, outW, outH, isSuperRes);
        refreshCachedDescriptors();
    }

    (void)cameraMovedLastFrame;
    (void)accumulatedSamples;

    glm::vec2 jitter(0.0f);

    if (camera) {
        if (resetHistory) {
            camera->resetPrevViewProj();
        }
        bool enableJitter = isSuperRes || coupledWithExternalUpscaler ||
                            (config.upscaler_mode == UpscalerMode::Upways && config.upways_superres);
        uint32_t jitterSeqLen = coupledWithExternalUpscaler ? getFsrJitterPhaseCount(inW, config.width) : 32u;
        CameraUniform ubo = camera->getUniformData(
            frameIndex, 1, config.max_bounces, 0,
            enableJitter, inW, inH, 0, false, jitterSeqLen
        );
        jitter = glm::vec2(ubo.jitterOffset.x, ubo.jitterOffset.y);
    }

    m_upwaysPipeline->recordFrame(
        cmd, frameIndex, resetHistory, jitter,
        coupledWithExternalUpscaler, config.upways_heuristic_fallback, 1.0f
    );
    return true;
}

bool SuperResolutionManager::dispatchFsr(VkCommandBuffer cmd, bool resetHistory,
                                         const Config& config, uint32_t frameIndex,
                                         bool cameraMovedLastFrame, uint32_t accumulatedSamples,
                                         Image* accumImage, Image* normalDepthImage, Image* motionVectorImage,
                                         MultiGpuManager* mgpu, Buffer* secTransferBuffer, uint32_t currentFrame,
                                         PostProcessPipeline* postProcess, Image* outputImage,
                                         const std::function<void()>& onDescriptorsInvalidated,
                                         Camera* camera,
                                         float frameTimeDeltaMs,
                                         Image* rawFrameImage,
                                         bool inputIsDenoised) {
    if (config.upscaler_mode != UpscalerMode::FSR) {
        return false;
    }
    if (!m_fsrUpscaler) {
        vkDeviceWaitIdle(m_device);
        createFsrPipelines(config);
        createFsrResources(config, postProcess ? postProcess : m_cachedPostProcess);
        if (postProcess) m_cachedPostProcess = postProcess;
        if (accumImage) m_cachedAccumImage = accumImage;
        if (outputImage) m_cachedOutputImage = outputImage;
        if (normalDepthImage) m_cachedNormalDepthImage = normalDepthImage;
        if (motionVectorImage) m_cachedMotionVectorImage = motionVectorImage;
        refreshCachedDescriptors();
    }
    if (!m_fsrUpscaler || !accumImage) {
        return false;
    }

    m_fsrUpscaler->setTargetVersion(config.fsr_version);
    if (!m_fsrUpscaler->isUsingSdkProvider()) {
        return false;
    }

    uint32_t renderW = config.getRenderWidth();
    uint32_t renderH = config.getRenderHeight();

    if (m_fsrUpscaler->getRenderWidth() != renderW || m_fsrUpscaler->getRenderHeight() != renderH ||
        m_fsrUpscaler->getDisplayWidth() != config.width || m_fsrUpscaler->getDisplayHeight() != config.height) {
        vkDeviceWaitIdle(m_device);
        m_fsrUpscaler->resize(renderW, renderH, config.width, config.height);
        refreshCachedDescriptors();
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
                          secBuffer != VK_NULL_HANDLE);
    if (isSampleBlend && (!m_secAccumImage || m_secAccumImage->getWidth() != config.width || m_secAccumImage->getHeight() != config.height)) {
        vkDeviceWaitIdle(m_device);
        createFsrResources(config, postProcess ? postProcess : m_cachedPostProcess);
        refreshCachedDescriptors();
    }

    const float camNear = camera ? std::max(camera->getNear(), 0.001f) : 0.05f;
    const float camFar  = camera ? std::max(camera->getFar(), camNear + 1.0f) : 10000.0f;
    const float camFovY = camera ? glm::radians(std::clamp(camera->getFov(), 5.0f, 170.0f)) : 0.785398f;
    const float effDtMs = (frameTimeDeltaMs > 0.1f && frameTimeDeltaMs < 1000.0f) ? frameTimeDeltaMs : 16.667f;
    const glm::vec2 prevJit = (!resetHistory && m_hasPrevFsrJitter) ? m_prevFsrJitter : glm::vec2(0.0f);

    if (isSampleBlend && m_secAccumImage) {
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

        const bool hasDenoiser = (config.denoiser_mode != DenoiserMode::None);
        const bool fsrUpscaling = (renderW < config.width || renderH < config.height);
        const bool isStationaryAccum = config.progressive_accumulation && !hasDenoiser && !fsrUpscaling && !cameraMovedLastFrame && (accumulatedSamples >= 1);
        if (!isStationaryAccum) {
            jitter = getHaltonJitter(frameIndex, fsrPhaseCount);
            if (!hasDenoiser && !cameraMovedLastFrame && config.progressive_accumulation && accumulatedSamples >= 2) {
                jitter *= (1.0f / static_cast<float>(accumulatedSamples));
            }
        }
        descPrim.jitterX = jitter.x;
        descPrim.jitterY = jitter.y;
        descPrim.cameraNear = camNear;
        descPrim.cameraFar = camFar;
        descPrim.cameraFovY = camFovY;
        descPrim.frameTimeDeltaMs = effDtMs;
        descPrim.enableSharpening = config.upscaler_sharpening;
        descPrim.sharpness = config.upscaler_sharpening ? config.upscaler_sharpness : 0.0f;
        descPrim.resetHistory = resetHistory;
        descPrim.cameraMoved = cameraMovedLastFrame;
        descPrim.frameIndex = frameIndex;
        descPrim.inputIsNormalized = true;
        descPrim.totalSamples = (config.progressive_accumulation && accumulatedSamples > 0) ? accumulatedSamples : 1u;
        descPrim.prevJitterX = (resetHistory || !m_hasPrevFsrJitter) ? jitter.x : prevJit.x;
        descPrim.prevJitterY = (resetHistory || !m_hasPrevFsrJitter) ? jitter.y : prevJit.y;
        descPrim.preExposure = std::clamp(config.exposure, 0.1f, 4.0f);
        descPrim.rawColorIn = (!inputIsDenoised && rawFrameImage) ? rawFrameImage->getImageView() : VK_NULL_HANDLE;
        descPrim.rawSampleScale = 1.0f;
        descPrim.inputIsDenoised = inputIsDenoised;
        m_fsrUpscaler->recordUpscale(descPrim);
        m_prevFsrJitter = jitter;
        m_hasPrevFsrJitter = true;

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

        const bool hasDenoiser = (config.denoiser_mode != DenoiserMode::None);
        const bool fsrUpscaling = (renderW < config.width || renderH < config.height);
        const bool isStationaryAccum = config.progressive_accumulation && !hasDenoiser && !fsrUpscaling && !cameraMovedLastFrame && (accumulatedSamples >= 1);
        if (!isStationaryAccum) {
            glm::vec2 jitterPrim = getHaltonJitter(frameIndex, fsrPhaseCount);
            bool isMgpuSampleParallel = (mgpu && mgpu->isMultiGpuActive() && config.mgpu_mode != MultiGpuMode::Off &&
                (config.mgpu_mode == MultiGpuMode::SampleParallel ||
                 (config.mgpu_mode == MultiGpuMode::Auto && (config.spp > 1))));
            if (isMgpuSampleParallel && config.mgpu_upscale_mode == MgpuUpscaleMode::SampleBlend) {
                glm::vec2 jitterSec = getHaltonJitter(frameIndex + (fsrPhaseCount / 2), fsrPhaseCount);
                jitter = (jitterPrim + jitterSec) * 0.5f;
            } else {
                jitter = jitterPrim;
            }
            if (!hasDenoiser && !cameraMovedLastFrame && config.progressive_accumulation && accumulatedSamples >= 2) {
                jitter *= (1.0f / static_cast<float>(accumulatedSamples));
            }
        }
        desc.jitterX = jitter.x;
        desc.jitterY = jitter.y;
        desc.cameraNear = camNear;
        desc.cameraFar = camFar;
        desc.cameraFovY = camFovY;
        desc.frameTimeDeltaMs = effDtMs;
        desc.enableSharpening = config.upscaler_sharpening;
        desc.sharpness = config.upscaler_sharpening ? config.upscaler_sharpness : 0.0f;
        desc.resetHistory = resetHistory;
        desc.cameraMoved = cameraMovedLastFrame;
        desc.frameIndex = frameIndex;
        desc.inputIsNormalized = true;
        desc.totalSamples = (config.progressive_accumulation && accumulatedSamples > 0) ? accumulatedSamples : 1u;
        desc.prevJitterX = (resetHistory || !m_hasPrevFsrJitter) ? jitter.x : prevJit.x;
        desc.prevJitterY = (resetHistory || !m_hasPrevFsrJitter) ? jitter.y : prevJit.y;
        desc.preExposure = std::clamp(config.exposure, 0.1f, 4.0f);
        desc.rawColorIn = (!inputIsDenoised && rawFrameImage) ? rawFrameImage->getImageView() : VK_NULL_HANDLE;
        desc.rawSampleScale = 1.0f;
        desc.inputIsDenoised = inputIsDenoised;

        m_fsrUpscaler->recordUpscale(desc);
        m_prevFsrJitter = jitter;
        m_hasPrevFsrJitter = true;
    }
    return true;
}

} // namespace pathways
