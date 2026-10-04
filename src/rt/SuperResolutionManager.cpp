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
    destroyFsr3Resources();
    destroyFsr3Pipelines();
    destroyUpwaysResources();
    destroyUpwaysPipelines();
    destroyDescPool();
}

void SuperResolutionManager::createDescPool() {
    if (m_postProcessDescPool != VK_NULL_HANDLE) {
        return;
    }
    std::vector<VkDescriptorPoolSize> poolSizes = {
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 32 }
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 8;
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
    m_tonemapFsr3DescSet = VK_NULL_HANDLE;
}

void SuperResolutionManager::init(const Config& config,
                                  PostProcessPipeline* postProcess,
                                  VkCommandBuffer cmd) {
    createDescPool();
    createUpwaysPipelines(config);
    createUpwaysResources(config, postProcess, cmd);
    createFsr3Pipelines(config);
    createFsr3Resources(config, postProcess);
}

void SuperResolutionManager::createUpwaysPipelines(const Config& config) {
    if (!m_shaderLoader) return;
    try {
        auto upwaysCode = m_shaderLoader("neural_reconstruct.comp.spv");
        if (upwaysCode.empty()) {
            upwaysCode = m_shaderLoader("upways_reconstruct.comp.spv");
        }
        VkFormat format = (config.accum_format == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
        bool isSuperRes = config.upways_superres || (config.upscaler_mode == UpscalerMode::Upways);
        uint32_t inW = (config.render_scale < 1.0f && (config.upscaler_mode != UpscalerMode::None || config.upways_superres)) ?
            static_cast<uint32_t>(config.width * config.render_scale) : config.width;
        uint32_t inH = (config.render_scale < 1.0f && (config.upscaler_mode != UpscalerMode::None || config.upways_superres)) ?
            static_cast<uint32_t>(config.height * config.render_scale) : config.height;
        uint32_t outW = config.width;
        uint32_t outH = config.height;
        isSuperRes = (inW < outW || inH < outH || isSuperRes);

        m_upwaysPipeline = std::make_unique<UpwaysPipeline>(
            m_device,
            m_physicalDevice,
            m_allocator,
            inW,
            inH,
            outW,
            outH,
            upwaysCode,
            config.upways_weights_path,
            isSuperRes,
            format
        );
        Logger::info("Upways Neural Reconstruction Pipeline initialized successfully.");
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

void SuperResolutionManager::createUpwaysResources(const Config& config, PostProcessPipeline* postProcess, VkCommandBuffer cmd) {
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

    uint32_t outW = config.width;
    uint32_t outH = config.height;
    m_displayAlbedoImage = std::make_unique<Image>(
        m_device, m_allocator, outW, outH,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    );
    m_displayNormalsImage = std::make_unique<Image>(
        m_device, m_allocator, outW, outH,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT
    );

    if (m_upwaysPipeline && cmd != VK_NULL_HANDLE) {
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &beginInfo);

        m_upwaysPipeline->transitionInitialLayouts(cmd);

        if (m_displayAlbedoImage) {
            m_displayAlbedoImage->transitionLayout(
                cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT
            );
        }
        if (m_displayNormalsImage) {
            m_displayNormalsImage->transitionLayout(
                cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_BLIT_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT
            );
        }

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

void SuperResolutionManager::createFsr3Pipelines(const Config& config) {
    if (config.upscaler_mode != UpscalerMode::FSR3 || !m_shaderLoader) {
        return;
    }

    try {
        auto upscaleCode = m_shaderLoader("fsr3_upscale.comp.spv");
        auto rcasCode = m_shaderLoader("fsr3_rcas.comp.spv");
        VkFormat format = (config.accum_format == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
        uint32_t renderW = (config.render_scale < 1.0f) ?
            static_cast<uint32_t>(config.width * config.render_scale) :
            config.width;
        uint32_t renderH = (config.render_scale < 1.0f) ?
            static_cast<uint32_t>(config.height * config.render_scale) :
            config.height;

        m_fsr3Upscaler = std::make_unique<Fsr3Upscaler>(
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
            "[GPU 0 Primary Viewport]"
        );

        Logger::info("AMD FSR 3.1 Super-Resolution Pipeline initialized successfully.");
    } catch (const std::exception& e) {
        Logger::warn("Fsr3Upscaler initialization failed: {}", e.what());
    }
}

void SuperResolutionManager::destroyFsr3Pipelines() {
    m_fsr3Upscaler.reset();
    m_tonemapFsr3DescSet = VK_NULL_HANDLE;
}

void SuperResolutionManager::createFsr3Resources(const Config& config, PostProcessPipeline* postProcess) {
    if (config.upscaler_mode != UpscalerMode::FSR3) {
        return;
    }

    if (m_postProcessDescPool && postProcess && m_tonemapFsr3DescSet == VK_NULL_HANDLE) {
        VkDescriptorSetAllocateInfo tmAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        tmAllocInfo.descriptorPool = m_postProcessDescPool;
        tmAllocInfo.descriptorSetCount = 1;
        VkDescriptorSetLayout layout = postProcess->getTonemapDescLayout();
        tmAllocInfo.pSetLayouts = &layout;
        if (vkAllocateDescriptorSets(m_device, &tmAllocInfo, &m_tonemapFsr3DescSet) != VK_SUCCESS) {
            Logger::warn("Failed to allocate Tonemap FSR3 descriptor set");
            m_tonemapFsr3DescSet = VK_NULL_HANDLE;
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

void SuperResolutionManager::destroyFsr3Resources() {
    m_secAccumImage.reset();
}

void SuperResolutionManager::resize(uint32_t width, uint32_t height, float renderScale,
                                    UpscalerMode upscalerMode, DenoiserMode denoiserMode,
                                    bool upwaysSuperres, AccumFormat accumFormat,
                                    PostProcessPipeline* postProcess, VkCommandBuffer cmd) {
    destroyUpwaysResources();
    if (m_upwaysPipeline) {
        bool isSuperRes = upwaysSuperres || (upscalerMode == UpscalerMode::Upways);
        uint32_t inW = (renderScale < 1.0f && (upscalerMode != UpscalerMode::None || upwaysSuperres)) ?
            static_cast<uint32_t>(width * renderScale) : width;
        uint32_t inH = (renderScale < 1.0f && (upscalerMode != UpscalerMode::None || upwaysSuperres)) ?
            static_cast<uint32_t>(height * renderScale) : height;
        uint32_t outW = width;
        uint32_t outH = height;
        isSuperRes = (inW < outW || inH < outH || isSuperRes);
        m_upwaysPipeline->resize(inW, inH, outW, outH, isSuperRes);
    }

    Config cfgStub{};
    cfgStub.width = width;
    cfgStub.height = height;
    cfgStub.render_scale = renderScale;
    cfgStub.upscaler_mode = upscalerMode;
    cfgStub.denoiser_mode = denoiserMode;
    cfgStub.upways_superres = upwaysSuperres;
    cfgStub.accum_format = accumFormat;

    createUpwaysResources(cfgStub, postProcess, cmd);
    destroyFsr3Resources();
    if (m_fsr3Upscaler) {
        uint32_t renderW = (renderScale < 1.0f) ?
            static_cast<uint32_t>(width * renderScale) :
            width;
        uint32_t renderH = (renderScale < 1.0f) ?
            static_cast<uint32_t>(height * renderScale) :
            height;
        m_fsr3Upscaler->resize(renderW, renderH, width, height);
    }
    createFsr3Resources(cfgStub, postProcess);
}

void SuperResolutionManager::updateDescriptors(PostProcessPipeline* postProcess,
                                              Image* accumImage, Image* outputImage,
                                              Image* normalDepthImage, Image* motionVectorImage,
                                              Image* mlAlbedoRoughnessImage, Image* mlSpecularMotionImage,
                                              Image* mlDiffuseImage, Image* mlSpecularImage,
                                              Image* frameImage0) {
    if (m_upwaysPipeline && accumImage && outputImage) {
        bool isSuperRes = m_upwaysPipeline->isSuperResEnabled();

        VkImageView normDepthView = normalDepthImage ? normalDepthImage->getImageView() : VK_NULL_HANDLE;
        VkImageView motionView = motionVectorImage ? motionVectorImage->getImageView() : VK_NULL_HANDLE;
        VkImageView albedoView = mlAlbedoRoughnessImage ? mlAlbedoRoughnessImage->getImageView() : VK_NULL_HANDLE;
        VkImageView specMotionView = mlSpecularMotionImage ? mlSpecularMotionImage->getImageView() : motionView;
        VkImageView diffView = mlDiffuseImage ? mlDiffuseImage->getImageView() : (frameImage0 ? frameImage0->getImageView() : VK_NULL_HANDLE);
        VkImageView specView = mlSpecularImage ? mlSpecularImage->getImageView() : VK_NULL_HANDLE;
        VkImageView displayAlbedoView = (isSuperRes && m_displayAlbedoImage) ? m_displayAlbedoImage->getImageView() : albedoView;
        VkImageView displayNormalsView = (isSuperRes && m_displayNormalsImage) ? m_displayNormalsImage->getImageView() : normDepthView;
        VkImageView restirMetadataView = VK_NULL_HANDLE;

        m_upwaysPipeline->updateDescriptors(
            diffView,
            specView,
            normDepthView,
            albedoView,
            motionView,
            specMotionView,
            restirMetadataView,
            displayAlbedoView,
            displayNormalsView
        );

        if (m_tonemapUpwaysDescSet != VK_NULL_HANDLE && m_upwaysPipeline->getOutputImage() && postProcess) {
            postProcess->updateTonemapDescriptors(m_tonemapUpwaysDescSet, m_upwaysPipeline->getOutputImage(), outputImage);
        }
    }

    if (m_fsr3Upscaler && outputImage && postProcess) {
        if (m_tonemapFsr3DescSet != VK_NULL_HANDLE && m_fsr3Upscaler->getOutputImage()) {
            postProcess->updateTonemapDescriptors(m_tonemapFsr3DescSet, m_fsr3Upscaler->getOutputImage(), outputImage);
        }
        if (m_fsr3Upscaler->getOutputImage() && m_secAccumImage) {
            postProcess->updateBlendDescriptors(m_fsr3Upscaler->getOutputImage(), m_secAccumImage.get());
        }
    }
}

bool SuperResolutionManager::dispatchUpways(VkCommandBuffer cmd, bool resetHistory,
                                            const Config& config, Camera* camera,
                                            uint32_t frameIndex, bool cameraMovedLastFrame,
                                            uint32_t accumulatedSamples,
                                            Image* mlAlbedoRoughnessImage, Image* normalDepthImage,
                                            PostProcessPipeline* postProcess, Image* outputImage) {
    if ((config.denoiser_mode != DenoiserMode::Upways && config.upscaler_mode != UpscalerMode::Upways) || !m_upwaysPipeline) {
        return false;
    }

    bool isSuperRes = config.upways_superres || (config.upscaler_mode == UpscalerMode::Upways);
    uint32_t inW = (config.render_scale < 1.0f && (config.upscaler_mode != UpscalerMode::None || config.upways_superres)) ?
        static_cast<uint32_t>(config.width * config.render_scale) : config.width;
    uint32_t inH = (config.render_scale < 1.0f && (config.upscaler_mode != UpscalerMode::None || config.upways_superres)) ?
        static_cast<uint32_t>(config.height * config.render_scale) : config.height;
    uint32_t outW = config.width;
    uint32_t outH = config.height;
    isSuperRes = (inW < outW || inH < outH || isSuperRes);

    if (isSuperRes) {
        if (!m_displayAlbedoImage || !m_displayNormalsImage ||
            m_displayAlbedoImage->getWidth() != outW || m_displayAlbedoImage->getHeight() != outH ||
            m_displayNormalsImage->getWidth() != outW || m_displayNormalsImage->getHeight() != outH) {
            vkDeviceWaitIdle(m_device);
            destroyUpwaysResources();
            createUpwaysResources(config, postProcess, cmd);
        }
    }

    if (m_upwaysPipeline->getInputWidth() != inW || m_upwaysPipeline->getInputHeight() != inH ||
        m_upwaysPipeline->getOutputWidth() != outW || m_upwaysPipeline->getOutputHeight() != outH ||
        m_upwaysPipeline->isSuperResEnabled() != isSuperRes) {
        vkDeviceWaitIdle(m_device);
        m_upwaysPipeline->resize(inW, inH, outW, outH, isSuperRes);
        updateDescriptors(postProcess, nullptr, outputImage, normalDepthImage, nullptr,
                          mlAlbedoRoughnessImage, nullptr, nullptr, nullptr, nullptr);
    }

    if (isSuperRes && m_displayAlbedoImage && m_displayNormalsImage) {
        Image* srcAlbedo = mlAlbedoRoughnessImage;
        Image* srcNormDepth = normalDepthImage;

        if (srcAlbedo && srcNormDepth) {
            std::vector<VkImageMemoryBarrier2> preBlitBarriers;
            auto addPreBlitBarrier = [&](Image* img, VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess) {
                VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
                b.srcAccessMask = srcAccess;
                b.dstStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
                b.dstAccessMask = dstAccess;
                b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = img->getImage();
                b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                preBlitBarriers.push_back(b);
            };

            addPreBlitBarrier(srcAlbedo, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            addPreBlitBarrier(srcNormDepth, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            addPreBlitBarrier(m_displayAlbedoImage.get(), VK_ACCESS_2_NONE, VK_ACCESS_2_TRANSFER_WRITE_BIT);
            addPreBlitBarrier(m_displayNormalsImage.get(), VK_ACCESS_2_NONE, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            VkDependencyInfo preBlitDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            preBlitDep.imageMemoryBarrierCount = static_cast<uint32_t>(preBlitBarriers.size());
            preBlitDep.pImageMemoryBarriers = preBlitBarriers.data();
            vkCmdPipelineBarrier2(cmd, &preBlitDep);

            VkImageBlit blitAlbedo{};
            blitAlbedo.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blitAlbedo.srcOffsets[0] = { 0, 0, 0 };
            blitAlbedo.srcOffsets[1] = { static_cast<int32_t>(inW), static_cast<int32_t>(inH), 1 };
            blitAlbedo.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blitAlbedo.dstOffsets[0] = { 0, 0, 0 };
            blitAlbedo.dstOffsets[1] = { static_cast<int32_t>(outW), static_cast<int32_t>(outH), 1 };
            vkCmdBlitImage(cmd, srcAlbedo->getImage(), VK_IMAGE_LAYOUT_GENERAL, m_displayAlbedoImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &blitAlbedo, VK_FILTER_LINEAR);

            VkImageBlit blitNormals{};
            blitNormals.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blitNormals.srcOffsets[0] = { 0, 0, 0 };
            blitNormals.srcOffsets[1] = { static_cast<int32_t>(inW), static_cast<int32_t>(inH), 1 };
            blitNormals.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            blitNormals.dstOffsets[0] = { 0, 0, 0 };
            blitNormals.dstOffsets[1] = { static_cast<int32_t>(outW), static_cast<int32_t>(outH), 1 };
            vkCmdBlitImage(cmd, srcNormDepth->getImage(), VK_IMAGE_LAYOUT_GENERAL, m_displayNormalsImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &blitNormals, VK_FILTER_NEAREST);

            std::vector<VkImageMemoryBarrier2> postBlitBarriers;
            auto addPostBlitBarrier = [&](Image* img, VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess) {
                VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
                b.srcStageMask = VK_PIPELINE_STAGE_2_BLIT_BIT;
                b.srcAccessMask = srcAccess;
                b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                b.dstAccessMask = dstAccess;
                b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
                b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
                b.image = img->getImage();
                b.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                postBlitBarriers.push_back(b);
            };

            addPostBlitBarrier(m_displayAlbedoImage.get(), VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            addPostBlitBarrier(m_displayNormalsImage.get(), VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            addPostBlitBarrier(srcAlbedo, VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            addPostBlitBarrier(srcNormDepth, VK_ACCESS_2_TRANSFER_READ_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

            VkDependencyInfo postBlitDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            postBlitDep.imageMemoryBarrierCount = static_cast<uint32_t>(postBlitBarriers.size());
            postBlitDep.pImageMemoryBarriers = postBlitBarriers.data();
            vkCmdPipelineBarrier2(cmd, &postBlitDep);
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
        bool enableJitter = (config.upscaler_mode == UpscalerMode::Upways || config.upways_superres);
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

bool SuperResolutionManager::dispatchFsr3(VkCommandBuffer cmd, bool resetHistory,
                                          const Config& config, uint32_t frameIndex,
                                          bool cameraMovedLastFrame, uint32_t accumulatedSamples,
                                          Image* accumImage, Image* normalDepthImage, Image* motionVectorImage,
                                          MultiGpuManager* mgpu, Buffer* secTransferBuffer, uint32_t currentFrame,
                                          PostProcessPipeline* postProcess, Image* outputImage,
                                          const std::function<void()>& onDescriptorsInvalidated) {
    if (config.upscaler_mode != UpscalerMode::FSR3) {
        return false;
    }
    if (!m_fsr3Upscaler) {
        createFsr3Pipelines(config);
        createFsr3Resources(config, postProcess);
        updateDescriptors(postProcess, accumImage, outputImage, normalDepthImage, motionVectorImage,
                          nullptr, nullptr, nullptr, nullptr, nullptr);
    }
    if (!m_fsr3Upscaler || !accumImage) {
        return false;
    }

    uint32_t renderW = (config.render_scale < 1.0f) ?
        static_cast<uint32_t>(config.width * config.render_scale) :
        config.width;
    uint32_t renderH = (config.render_scale < 1.0f) ?
        static_cast<uint32_t>(config.height * config.render_scale) :
        config.height;

    if (m_fsr3Upscaler->getRenderWidth() != renderW || m_fsr3Upscaler->getRenderHeight() != renderH ||
        m_fsr3Upscaler->getDisplayWidth() != config.width || m_fsr3Upscaler->getDisplayHeight() != config.height) {
        vkDeviceWaitIdle(m_device);
        m_fsr3Upscaler->resize(renderW, renderH, config.width, config.height);
        updateDescriptors(postProcess, accumImage, outputImage, normalDepthImage, motionVectorImage,
                          nullptr, nullptr, nullptr, nullptr, nullptr);
        if (onDescriptorsInvalidated) {
            onDescriptorsInvalidated();
        }
    }

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
        descPrim.colorOut = m_fsr3Upscaler->getOutputImage()->getImageView();
        descPrim.renderWidth = renderW;
        descPrim.renderHeight = renderH;
        descPrim.displayWidth = config.width;
        descPrim.displayHeight = config.height;
        glm::vec2 jitter(0.0f);
        if (config.upscaler_mode == UpscalerMode::FSR3) {
            if (!config.progressive_accumulation || cameraMovedLastFrame || accumulatedSamples <= 1) {
                jitter = getHaltonJitter(frameIndex);
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
        m_fsr3Upscaler->recordUpscale(descPrim);

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
        desc.colorOut = m_fsr3Upscaler->getOutputImage()->getImageView();
        desc.renderWidth = renderW;
        desc.renderHeight = renderH;
        desc.displayWidth = config.width;
        desc.displayHeight = config.height;
        glm::vec2 jitter(0.0f);
        if (config.upscaler_mode == UpscalerMode::FSR3) {
            if (!config.progressive_accumulation || cameraMovedLastFrame || accumulatedSamples <= 1) {
                glm::vec2 jitterPrim = getHaltonJitter(frameIndex);
                bool isMgpuSampleParallel = (mgpu && mgpu->isMultiGpuActive() && config.mgpu_mode != MultiGpuMode::Off &&
                    (config.mgpu_mode == MultiGpuMode::SampleParallel ||
                     (config.mgpu_mode == MultiGpuMode::Auto && (config.spp > 1))));
                if (isMgpuSampleParallel) {
                    glm::vec2 jitterSec = getHaltonJitter(frameIndex + 4);
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

        m_fsr3Upscaler->recordUpscale(desc);
    }
    return true;
}

} // namespace pathways
