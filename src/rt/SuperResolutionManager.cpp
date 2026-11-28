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

struct BandStitchPushConstants {
    uint32_t width;
    uint32_t dstYStart;
    uint32_t rowCount;
    uint32_t pad;
};
static_assert(sizeof(BandStitchPushConstants) == 16, "BandStitchPushConstants must be 16 bytes");
} // namespace

SuperResolutionManager::SuperResolutionManager(VkDevice device,
                                               VkPhysicalDevice physicalDevice,
                                               VmaAllocator allocator,
                                               VkQueue queue,
                                               ShaderLoaderFunc shaderLoader,
                                               VkInstance instance,
                                               VkPipelineCache pipelineCache)
    : m_device(device),
      m_physicalDevice(physicalDevice),
      m_allocator(allocator),
      m_queue(queue),
      m_shaderLoader(std::move(shaderLoader)),
      m_instance(instance),
      m_pipelineCache(pipelineCache) {
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
        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64 },
        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 16 }
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
    m_tonemapUpwaysDescSets.fill(VK_NULL_HANDLE);
    m_tonemapUpwaysBoundInView.fill(VK_NULL_HANDLE);
    m_tonemapUpwaysBoundOutView.fill(VK_NULL_HANDLE);
    m_tonemapUpwaysBoundStitchBuf.fill(VK_NULL_HANDLE);
    m_tonemapUpwaysBoundStitchSize.fill(0);
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
        const UpwaysRuntimeVersion ver = UpwaysPipeline::resolveVersion(
            config.upways_version, config.upways_weights_path
        );
        const UpwaysComputePrecision prec = UpwaysPipeline::selectPrecision(
            m_physicalDevice, config.upways_precision, m_instance, ver
        );
        const char* ext = ".comp.spv";
        if (prec == UpwaysComputePrecision::Int8) {
            ext = ".int8.comp.spv";
        } else if (prec == UpwaysComputePrecision::Fp16) {
            ext = ".fp16.comp.spv";
        }

        VkFormat format = (config.accum_format == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
        uint32_t inW = config.getRenderWidth();
        uint32_t inH = config.getRenderHeight();
        uint32_t outW = (config.upscaler_mode == UpscalerMode::FSR) ? inW : config.width;
        uint32_t outH = (config.upscaler_mode == UpscalerMode::FSR) ? inH : config.height;
        bool isSuperRes = (inW < outW || inH < outH);

        if (ver == UpwaysRuntimeVersion::V5) {
            auto prepCode      = m_shaderLoader(std::string("upways_v5_prep") + ext);
            auto unetCode      = m_shaderLoader(std::string("upways_v5_unet_wmma") + ext);
            auto affinityCode  = m_shaderLoader(std::string("upways_v5_affinity") + ext);
            auto upscalerCode  = m_shaderLoader(std::string("upways_v5_upscaler_wmma") + ext);
            auto resolveCode   = m_shaderLoader(std::string("upways_v5_resolve") + ext);
            auto stabilizeCode = m_shaderLoader("upways_v5_stabilize.comp.spv");

            m_upwaysPipeline = std::make_unique<UpwaysPipeline>(
                m_device,
                m_physicalDevice,
                m_allocator,
                inW,
                inH,
                outW,
                outH,
                prepCode,
                unetCode,
                affinityCode,
                upscalerCode,
                resolveCode,
                config.upways_weights_path,
                isSuperRes,
                format,
                prec,
                outH,
                stabilizeCode
            );
        } else {
            auto denoisePrepCode    = m_shaderLoader(std::string("upways_denoise_prep") + ext);
            auto denoiserWmmaCode   = m_shaderLoader(std::string("upways_denoiser_wmma") + ext);
            auto upscalerWmmaCode   = m_shaderLoader(std::string("upways_upscaler_wmma") + ext);
            auto upscaleResolveCode = m_shaderLoader(std::string("upways_upscale_resolve") + ext);

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
                prec,
                outH
            );
        }

        if (m_bandStitchPipeline == VK_NULL_HANDLE) {
            std::array<VkDescriptorSetLayoutBinding, 2> stitchBindings{};
            stitchBindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            stitchBindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
            VkDescriptorSetLayoutCreateInfo dslInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
            dslInfo.bindingCount = static_cast<uint32_t>(stitchBindings.size());
            dslInfo.pBindings = stitchBindings.data();
            if (vkCreateDescriptorSetLayout(m_device, &dslInfo, nullptr, &m_bandStitchDescLayout) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create Upways band stitch descriptor set layout");
            }

            std::array<VkDescriptorPoolSize, 2> poolSizes{};
            poolSizes[0] = { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2u };
            poolSizes[1] = { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2u };
            VkDescriptorPoolCreateInfo dpInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
            dpInfo.maxSets = 2u;
            dpInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
            dpInfo.pPoolSizes = poolSizes.data();
            if (vkCreateDescriptorPool(m_device, &dpInfo, nullptr, &m_bandStitchDescPool) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create Upways band stitch descriptor pool");
            }

            std::array<VkDescriptorSetLayout, 2> setLayouts{ m_bandStitchDescLayout, m_bandStitchDescLayout };
            VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            allocInfo.descriptorPool = m_bandStitchDescPool;
            allocInfo.descriptorSetCount = 2u;
            allocInfo.pSetLayouts = setLayouts.data();
            if (vkAllocateDescriptorSets(m_device, &allocInfo, m_bandStitchDescSets.data()) != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate Upways band stitch descriptor sets");
            }

            VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BandStitchPushConstants) };
            VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
            plInfo.setLayoutCount = 1u;
            plInfo.pSetLayouts = &m_bandStitchDescLayout;
            plInfo.pushConstantRangeCount = 1u;
            plInfo.pPushConstantRanges = &pcRange;
            if (vkCreatePipelineLayout(m_device, &plInfo, nullptr, &m_bandStitchPipeLayout) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create Upways band stitch pipeline layout");
            }

            auto stitchCode = m_shaderLoader("upways_band_stitch.comp.spv");
            VkShaderModuleCreateInfo smInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
            smInfo.codeSize = stitchCode.size();
            smInfo.pCode = reinterpret_cast<const uint32_t*>(stitchCode.data());
            VkShaderModule sm = VK_NULL_HANDLE;
            if (vkCreateShaderModule(m_device, &smInfo, nullptr, &sm) != VK_SUCCESS) {
                throw std::runtime_error("Failed to create Upways band stitch shader module");
            }

            VkComputePipelineCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
            cpInfo.layout = m_bandStitchPipeLayout;
            cpInfo.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, sm, "main", nullptr };
            if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1u, &cpInfo, nullptr, &m_bandStitchPipeline) != VK_SUCCESS) {
                vkDestroyShaderModule(m_device, sm, nullptr);
                throw std::runtime_error("Failed to create Upways band stitch compute pipeline");
            }
            vkDestroyShaderModule(m_device, sm, nullptr);
        }

        Logger::info(
            "{} ({}) Neural Denoiser & Super-Resolution Pipeline initialized on demand.",
            upwaysVersionDisplayName(ver),
            upwaysPrecisionDisplayName(prec)
        );
    } catch (const std::exception& e) {
        throw std::runtime_error(std::string("Upways was requested but initialization failed: ") + e.what());
    }
}

void SuperResolutionManager::destroyUpwaysPipelines() {
    if (m_bandStitchPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_bandStitchPipeline, nullptr);
        m_bandStitchPipeline = VK_NULL_HANDLE;
    }
    if (m_bandStitchPipeLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_bandStitchPipeLayout, nullptr);
        m_bandStitchPipeLayout = VK_NULL_HANDLE;
    }
    if (m_bandStitchDescPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_bandStitchDescPool, nullptr);
        m_bandStitchDescPool = VK_NULL_HANDLE;
    }
    m_bandStitchDescSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };
    m_bandStitchBoundBuf.fill(VK_NULL_HANDLE);
    m_bandStitchBoundImg.fill(VK_NULL_HANDLE);
    m_bandStitchBoundSize.fill(0);
    if (m_bandStitchDescLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_bandStitchDescLayout, nullptr);
        m_bandStitchDescLayout = VK_NULL_HANDLE;
    }
    m_upwaysPipeline.reset();
    if (m_postProcessDescPool != VK_NULL_HANDLE) {
        for (VkDescriptorSet& tmSet : m_tonemapUpwaysDescSets) {
            if (tmSet != VK_NULL_HANDLE) {
                vkFreeDescriptorSets(m_device, m_postProcessDescPool, 1, &tmSet);
            }
            tmSet = VK_NULL_HANDLE;
        }
    } else {
        m_tonemapUpwaysDescSets.fill(VK_NULL_HANDLE);
    }
    m_tonemapUpwaysBoundInView.fill(VK_NULL_HANDLE);
    m_tonemapUpwaysBoundOutView.fill(VK_NULL_HANDLE);
    m_tonemapUpwaysBoundStitchBuf.fill(VK_NULL_HANDLE);
    m_tonemapUpwaysBoundStitchSize.fill(0);
    m_prevUpwaysJitterPx = glm::vec2(0.0f);
    m_hasPrevUpwaysJitter = false;
}

void SuperResolutionManager::createUpwaysResources(const Config& config, PostProcessPipeline* postProcess, VkCommandBuffer cmd) {
    if (!isUpwaysRequested(config)) {
        return;
    }
    if (m_postProcessDescPool && postProcess) {
        VkDescriptorSetLayout layout = postProcess->getTonemapDescLayout();
        for (VkDescriptorSet& tmSet : m_tonemapUpwaysDescSets) {
            if (tmSet != VK_NULL_HANDLE) {
                continue;
            }
            VkDescriptorSetAllocateInfo tmAllocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
            tmAllocInfo.descriptorPool = m_postProcessDescPool;
            tmAllocInfo.descriptorSetCount = 1;
            tmAllocInfo.pSetLayouts = &layout;
            if (vkAllocateDescriptorSets(m_device, &tmAllocInfo, &tmSet) != VK_SUCCESS) {
                Logger::warn("Failed to allocate Tonemap Upways descriptor set");
                tmSet = VK_NULL_HANDLE;
            }
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
    m_prevUpwaysJitterPx = glm::vec2(0.0f);
    m_hasPrevUpwaysJitter = false;
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
            config.fsr_version,
            m_pipelineCache
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
                                    FsrVersion fsrVersion,
                                    UpwaysPrecision upwaysPrecision,
                                    UpwaysVersion upwaysVersion,
                                    const std::string& upwaysWeightsPath) {
    Config cfgStub{};
    cfgStub.width = width;
    cfgStub.height = height;
    cfgStub.render_scale = renderScale;
    cfgStub.upscaler_mode = upscalerMode;
    cfgStub.denoiser_mode = denoiserMode;
    cfgStub.upways_superres = upwaysSuperres;
    cfgStub.accum_format = accumFormat;
    cfgStub.fsr_version = fsrVersion;
    cfgStub.upways_precision = upwaysPrecision;
    cfgStub.upways_version = upwaysVersion;
    cfgStub.upways_weights_path = upwaysWeightsPath;

    releaseInactivePipelines(cfgStub);

    const uint32_t inW = cfgStub.getRenderWidth();
    const uint32_t inH = cfgStub.getRenderHeight();
    const uint32_t outW = (upscalerMode == UpscalerMode::FSR) ? inW : width;
    const uint32_t outH = (upscalerMode == UpscalerMode::FSR) ? inH : height;
    const bool isSuperRes = (inW < outW || inH < outH);

    destroyUpwaysResources();
    if (isUpwaysRequested(cfgStub)) {
        const UpwaysRuntimeVersion targetVer = UpwaysPipeline::resolveVersion(
            cfgStub.upways_version, cfgStub.upways_weights_path
        );
        const UpwaysComputePrecision targetPrec = UpwaysPipeline::selectPrecision(
            m_physicalDevice, cfgStub.upways_precision, m_instance, targetVer
        );
        bool recreated = false;
        if (m_upwaysPipeline &&
            (m_upwaysPipeline->getVersion() != targetVer || m_upwaysPipeline->getPrecision() != targetPrec)) {
            vkDeviceWaitIdle(m_device);
            destroyUpwaysPipelines();
            recreated = true;
        }
        if (!m_upwaysPipeline) {
            createUpwaysPipelines(cfgStub);
            recreated = true;
        } else {
            m_upwaysPipeline->resize(inW, inH, outW, outH, isSuperRes, outH);
        }
        createUpwaysResources(cfgStub, postProcess, cmd);
        if (recreated) {
            refreshCachedDescriptors();
        }
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
                                              Image* mlSpecAlbedoImage,
                                              Image* mlIrradianceImage,
                                              Image* mlPrimaryGuideImage,
                                              Image* mlSurfaceMvImage,
                                              Image* mlDiffHitDistImage,
                                              Image* mlSpecVmvImage,
                                              Image* mlSkipCompactImage) {
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
    m_cachedMlIrradianceImage = mlIrradianceImage;
    m_cachedMlPrimaryGuideImage = mlPrimaryGuideImage;
    m_cachedMlSurfaceMvImage = mlSurfaceMvImage;
    m_cachedMlDiffHitDistImage = mlDiffHitDistImage;
    m_cachedMlSpecVmvImage = mlSpecVmvImage;
    m_cachedMlSkipCompactImage = mlSkipCompactImage;

    refreshCachedDescriptors();
}

void SuperResolutionManager::refreshCachedDescriptors() {
    m_bandStitchBoundBuf.fill(VK_NULL_HANDLE);
    m_bandStitchBoundImg.fill(VK_NULL_HANDLE);
    m_bandStitchBoundSize.fill(0);

    if (m_upwaysPipeline && m_cachedAccumImage && m_cachedOutputImage) {
        VkImageView irrView = m_cachedMlIrradianceImage ? m_cachedMlIrradianceImage->getImageView() : VK_NULL_HANDLE;
        VkImageView primGuideView = m_cachedMlPrimaryGuideImage ? m_cachedMlPrimaryGuideImage->getImageView() : VK_NULL_HANDLE;
        VkImageView surfaceMvView = m_cachedMlSurfaceMvImage ? m_cachedMlSurfaceMvImage->getImageView() : VK_NULL_HANDLE;
        VkImageView diffHitDistView = m_cachedMlDiffHitDistImage ? m_cachedMlDiffHitDistImage->getImageView() : VK_NULL_HANDLE;
        VkImageView specVmvView = m_cachedMlSpecVmvImage ? m_cachedMlSpecVmvImage->getImageView() : VK_NULL_HANDLE;
        VkImageView skipCompactView = m_cachedMlSkipCompactImage ? m_cachedMlSkipCompactImage->getImageView() : VK_NULL_HANDLE;

        m_upwaysPipeline->updateDescriptors(
            irrView,
            primGuideView,
            surfaceMvView,
            diffHitDistView,
            specVmvView,
            skipCompactView
        );

        m_tonemapUpwaysBoundInView.fill(VK_NULL_HANDLE);
        m_tonemapUpwaysBoundOutView.fill(VK_NULL_HANDLE);
        m_tonemapUpwaysBoundStitchBuf.fill(VK_NULL_HANDLE);
        m_tonemapUpwaysBoundStitchSize.fill(0);
        if (m_upwaysPipeline->getOutputImage() && m_cachedPostProcess) {
            for (VkDescriptorSet tmSet : m_tonemapUpwaysDescSets) {
                if (tmSet != VK_NULL_HANDLE) {
                    m_cachedPostProcess->updateTonemapDescriptors(tmSet, m_upwaysPipeline->getOutputImage(), m_cachedOutputImage);
                }
            }
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
                                            PostProcessPipeline* postProcess, Image* outputImage,
                                            const UpwaysBandSplit* bandSplit) {
    if (config.denoiser_mode != DenoiserMode::Upways && config.upscaler_mode != UpscalerMode::Upways) {
        return false;
    }

    const UpwaysRuntimeVersion targetVer = UpwaysPipeline::resolveVersion(
        config.upways_version, config.upways_weights_path
    );
    const UpwaysComputePrecision targetPrec = UpwaysPipeline::selectPrecision(
        m_physicalDevice, config.upways_precision, m_instance, targetVer
    );
    if (m_upwaysPipeline &&
        (m_upwaysPipeline->getVersion() != targetVer || m_upwaysPipeline->getPrecision() != targetPrec)) {
        vkDeviceWaitIdle(m_device);
        destroyUpwaysPipelines();
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

    uint32_t fullInW = config.getRenderWidth();
    uint32_t fullInH = config.getRenderHeight();
    bool coupledWithExternalUpscaler = (config.upscaler_mode == UpscalerMode::FSR);
    uint32_t fullOutW = coupledWithExternalUpscaler ? fullInW : config.width;
    uint32_t fullOutH = coupledWithExternalUpscaler ? fullInH : config.height;
    bool isSuperRes = !coupledWithExternalUpscaler && ((fullInW < fullOutW || fullInH < fullOutH) || config.isUpwaysSuperResActive());

    uint32_t targetInH = bandSplit ? bandSplit->topLrH : fullInH;
    uint32_t targetOutH = bandSplit ? bandSplit->topHrH : fullOutH;

    if (m_upwaysPipeline->getInputWidth() != fullInW || m_upwaysPipeline->getInputHeight() != targetInH ||
        m_upwaysPipeline->getOutputWidth() != fullOutW || m_upwaysPipeline->getOutputHeight() != targetOutH ||
        m_upwaysPipeline->getOutputFullHeight() != fullOutH ||
        m_upwaysPipeline->isSuperResEnabled() != isSuperRes) {
        vkDeviceWaitIdle(m_device);
        m_upwaysPipeline->resize(fullInW, targetInH, fullOutW, targetOutH, isSuperRes, fullOutH);
        refreshCachedDescriptors();
    }

    (void)cameraMovedLastFrame;
    (void)accumulatedSamples;

    glm::vec2 jitter(0.0f);
    UpwaysCameraMotion cameraMotion{};

    if (camera) {
        if (resetHistory) {
            camera->resetPrevViewProj();
        }
        const bool upwaysV5Standalone1x =
            !isSuperRes &&
            !coupledWithExternalUpscaler &&
            !config.upways_heuristic_fallback &&
            (m_upwaysPipeline->getVersion() == UpwaysRuntimeVersion::V5);
        const bool enableJitter = isSuperRes || coupledWithExternalUpscaler || upwaysV5Standalone1x;
        const uint32_t jitterSeqLen = coupledWithExternalUpscaler
            ? getFsrJitterPhaseCount(fullInW, config.width)
            : (upwaysV5Standalone1x ? 16u : 32u);
        CameraUniform ubo = camera->getUniformData(
            frameIndex, 1, config.max_bounces, 0,
            enableJitter, fullInW, fullInH, 0, false, jitterSeqLen
        );
        jitter = glm::vec2(ubo.jitterOffset.x, ubo.jitterOffset.y);

        const glm::vec2 prevJitterPx =
            (resetHistory || !m_hasPrevUpwaysJitter || !m_upwaysPipeline->isHistoryValid())
                ? jitter
                : m_prevUpwaysJitterPx;

        const glm::mat4 currView = camera->getViewMatrix();
        const glm::mat4 prevView = resetHistory ? currView : camera->getPrevViewMatrix();
        const float tanHalfFovY = std::tan(0.5f * glm::radians(camera->getFov()));
        const float aspectRatio = (camera->getAspect() > 1e-4f)
            ? camera->getAspect()
            : ((fullInH > 0u) ? (static_cast<float>(fullInW) / static_cast<float>(fullInH)) : 1.0f);
        cameraMotion = makeUpwaysCameraMotion(currView, prevView, tanHalfFovY, aspectRatio, prevJitterPx);

        m_prevUpwaysJitterPx = jitter;
        m_hasPrevUpwaysJitter = true;
    } else {
        m_prevUpwaysJitterPx = glm::vec2(0.0f);
        m_hasPrevUpwaysJitter = false;
    }

    UpwaysBandParams bandParams{};
    if (bandSplit) {
        bandParams.bandLrY0 = 0u;
        bandParams.bandHrY0 = 0u;
        bandParams.fullLrHeight = fullInH;
        bandParams.fullHrHeight = fullOutH;
        bandParams.keepHrYStart = 0u;
        bandParams.keepHrYEnd = bandSplit->splitHrY;
    }

    m_upwaysPipeline->recordFrame(
        cmd, frameIndex, resetHistory, jitter,
        coupledWithExternalUpscaler, config.upways_heuristic_fallback, 1.0f, bandParams, cameraMotion
    );
    return true;
}

void SuperResolutionManager::dispatchUpwaysBandStitch(VkCommandBuffer cmd, uint32_t frameSlot,
                                                      VkBuffer secPackedBuf, VkDeviceSize bufSize,
                                                      uint32_t width, uint32_t dstYStart, uint32_t rowCount) {
    if (!m_upwaysPipeline || !m_upwaysPipeline->getOutputImage() || m_bandStitchPipeline == VK_NULL_HANDLE ||
        secPackedBuf == VK_NULL_HANDLE || width == 0u || rowCount == 0u) {
        return;
    }

    const uint32_t setIdx = frameSlot % 2u;
    VkDescriptorSet dstSet = m_bandStitchDescSets[setIdx];
    if (dstSet == VK_NULL_HANDLE) {
        return;
    }

    const VkImageView outView = m_upwaysPipeline->getOutputImage()->getImageView();
    const VkDeviceSize clampedSize = std::max<VkDeviceSize>(bufSize, 256ull);
    if (m_bandStitchBoundBuf[setIdx] != secPackedBuf ||
        m_bandStitchBoundImg[setIdx] != outView ||
        m_bandStitchBoundSize[setIdx] != clampedSize) {
        VkDescriptorBufferInfo srcBufInfo{
            secPackedBuf,
            0,
            clampedSize
        };
        VkDescriptorImageInfo dstImgInfo{
            VK_NULL_HANDLE,
            outView,
            VK_IMAGE_LAYOUT_GENERAL
        };
        std::array<VkWriteDescriptorSet, 2> writes{};
        writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dstSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &srcBufInfo, nullptr };
        writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, dstSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dstImgInfo, nullptr, nullptr };
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        m_bandStitchBoundBuf[setIdx] = secPackedBuf;
        m_bandStitchBoundImg[setIdx] = outView;
        m_bandStitchBoundSize[setIdx] = clampedSize;
    }

    VkMemoryBarrier2 preBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    preBarrier.srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    preBarrier.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    preBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    preBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo preDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    preDep.memoryBarrierCount = 1u;
    preDep.pMemoryBarriers = &preBarrier;
    vkCmdPipelineBarrier2(cmd, &preDep);

    BandStitchPushConstants stitchPC{};
    stitchPC.width = width;
    stitchPC.dstYStart = dstYStart;
    stitchPC.rowCount = rowCount;
    stitchPC.pad = 0u;

    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bandStitchPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_bandStitchPipeLayout, 0, 1u, &dstSet, 0, nullptr);
    vkCmdPushConstants(cmd, m_bandStitchPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(BandStitchPushConstants), &stitchPC);
    vkCmdDispatch(cmd, (width + 15u) / 16u, (rowCount + 15u) / 16u, 1u);

    VkMemoryBarrier2 postBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    postDep.memoryBarrierCount = 1u;
    postDep.pMemoryBarriers = &postBarrier;
    vkCmdPipelineBarrier2(cmd, &postDep);
}

void SuperResolutionManager::updateTonemapUpwaysBandDescriptor(uint32_t frameSlot,
                                                               PostProcessPipeline* postProcess,
                                                               Image* outputImage,
                                                               VkBuffer stitchBuf,
                                                               VkDeviceSize stitchSize) {
    if (!postProcess || !m_upwaysPipeline || !m_upwaysPipeline->getOutputImage()) {
        return;
    }
    const uint32_t s = frameSlot % 2u;
    VkDescriptorSet dstSet = m_tonemapUpwaysDescSets[s];
    Image* outImg = outputImage ? outputImage : m_cachedOutputImage;
    if (dstSet == VK_NULL_HANDLE || !outImg) {
        return;
    }
    Image* inImg = m_upwaysPipeline->getOutputImage();
    const VkImageView inView = inImg->getImageView();
    const VkImageView outView = outImg->getImageView();
    const VkDeviceSize clampedSize = (stitchBuf != VK_NULL_HANDLE) ? std::max<VkDeviceSize>(stitchSize, 256ull) : 0ull;
    if (m_tonemapUpwaysBoundInView[s] == inView &&
        m_tonemapUpwaysBoundOutView[s] == outView &&
        m_tonemapUpwaysBoundStitchBuf[s] == stitchBuf &&
        m_tonemapUpwaysBoundStitchSize[s] == clampedSize) {
        return;
    }
    postProcess->updateTonemapDescriptors(dstSet, inImg, outImg, stitchBuf, clampedSize);
    m_tonemapUpwaysBoundInView[s] = inView;
    m_tonemapUpwaysBoundOutView[s] = outView;
    m_tonemapUpwaysBoundStitchBuf[s] = stitchBuf;
    m_tonemapUpwaysBoundStitchSize[s] = clampedSize;
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
