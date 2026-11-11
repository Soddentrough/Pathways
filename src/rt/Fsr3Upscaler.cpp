#include "rt/FsrUpscaler.hpp"
#include "rt/FfxSdkRuntime.hpp"
#include "core/Logger.hpp"

#include <algorithm>
#include <cfloat>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace pathways {

FsrUpscaler::FsrUpscaler(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VmaAllocator allocator,
    uint32_t renderWidth,
    uint32_t renderHeight,
    uint32_t displayWidth,
    uint32_t displayHeight,
    const std::vector<char>& sdkPrepSpv,
    VkFormat format,
    const std::string& instanceName,
    FsrVersion targetVersion
) : m_device(device),
    m_physDevice(physicalDevice),
    m_allocator(allocator),
    m_renderWidth(renderWidth),
    m_renderHeight(renderHeight),
    m_displayWidth(displayWidth),
    m_displayHeight(displayHeight),
    m_format(format),
    m_targetVersion(targetVersion),
    m_instanceName(instanceName)
{
    probeHardwareCapabilities();
    resolveActiveVersion();

    initImages();
    createDescriptorSetLayouts();
    allocateDescriptorSets();
    createPipelines(sdkPrepSpv);
    initFfxSdkContext();

    Logger::info("FsrUpscaler [{}] initialized: Render {}x{}, Display {}x{}, Active: {} (VRAM: {:.2f} MB)",
                 m_instanceName, m_renderWidth, m_renderHeight, m_displayWidth, m_displayHeight,
                 getActiveVersionName(), static_cast<double>(m_vramUsageBytes) / (1024.0 * 1024.0));
}

FsrUpscaler::~FsrUpscaler() {
    destroyFfxSdkContext();
    if (m_device != VK_NULL_HANDLE) {
        if (m_sdkPrepPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_sdkPrepPipeline, nullptr);
            m_sdkPrepPipeline = VK_NULL_HANDLE;
        }
        if (m_sdkPrepPipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_sdkPrepPipelineLayout, nullptr);
            m_sdkPrepPipelineLayout = VK_NULL_HANDLE;
        }
        if (m_sdkPrepDescLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_sdkPrepDescLayout, nullptr);
            m_sdkPrepDescLayout = VK_NULL_HANDLE;
        }
        if (m_descPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
            m_descPool = VK_NULL_HANDLE;
        }
    }
}

void FsrUpscaler::probeHardwareCapabilities() {
    m_sdkFsr3Supported = FfxSdkRuntime::instance().isUpscalerVersionSupported(m_physDevice, m_device, 3u);
    m_sdkFsr4Supported = FfxSdkRuntime::instance().isUpscalerVersionSupported(m_physDevice, m_device, 4u);
    m_hwMlSupported = FfxSdkRuntime::supportsHardwareMl(m_physDevice);
}

void FsrUpscaler::resolveActiveVersion() {
    FsrVersion prevActive = m_activeVersion;
    if (m_targetVersion == FsrVersion::Fsr3) {
        m_activeVersion = FsrVersion::Fsr3;
    } else if (m_targetVersion == FsrVersion::Fsr4) {
        m_activeVersion = FsrVersion::Fsr4;
    } else {
        m_activeVersion = m_sdkFsr4Supported ? FsrVersion::Fsr4 : FsrVersion::Fsr3;
    }
    if (prevActive != m_activeVersion) {
        m_historyValid = false;
    }
}

void FsrUpscaler::setTargetVersion(FsrVersion version) {
    if (m_targetVersion == version) {
        return;
    }
    m_targetVersion = version;
    FsrVersion prevActive = m_activeVersion;
    resolveActiveVersion();
    if (prevActive != m_activeVersion || m_ffxContext == nullptr) {
        initFfxSdkContext();
        Logger::info("FsrUpscaler [{}] switched active version to: {}", m_instanceName, getActiveVersionName());
    }
}

const char* FsrUpscaler::getActiveVersionName() const {
    if (m_ffxContext != nullptr && !m_activeVersionLabel.empty()) {
        return m_activeVersionLabel.c_str();
    }
    if (m_ffxContext != nullptr) {
        if (m_activeVersion == FsrVersion::Fsr4) {
            return "AMD FSR 4.1.2 (SDK ML Super-Resolution)";
        }
        return "AMD FSR 3.1.6 (SDK Super-Resolution)";
    }
    if (m_activeVersion == FsrVersion::Fsr4) {
        return "Stub (No Native Vulkan FSR 4 Provider)";
    }
    return "Disabled (Unsupported)";
}

void FsrUpscaler::initFfxSdkContext() {
    destroyFfxSdkContext();

    // Baseline VRAM of Pathways-side output + SDK input prep targets
    const uint64_t outBytes = static_cast<uint64_t>(m_displayWidth) * static_cast<uint64_t>(m_displayHeight) * 8ull;
    const uint64_t prepBytes = static_cast<uint64_t>(m_renderWidth) * static_cast<uint64_t>(m_renderHeight) * (8ull + 4ull + 4ull);
    m_vramUsageBytes = outBytes + prepBytes;

#if defined(PATHWAYS_HAS_FFX_SDK)
    if (!FfxSdkRuntime::instance().hasNativeProvider(FfxEffectModule::Upscaler)) {
        return;
    }
    if (m_activeVersion == FsrVersion::Fsr4 && !m_sdkFsr4Supported) {
        Logger::error("FsrUpscaler [{}]: FSR 4 ML Upscaler has no native Vulkan provider on this runtime; context creation skipped.",
                      m_instanceName);
        return;
    }
    if (m_activeVersion == FsrVersion::Fsr3 && !m_sdkFsr3Supported) {
        Logger::error("FsrUpscaler [{}]: FSR 3.1 Upscaler is not supported on this device/OS; context creation aborted.",
                      m_instanceName);
        return;
    }

    const uint32_t desiredMajor = (m_activeVersion == FsrVersion::Fsr4) ? 4u : 3u;
    auto queriedVersions = FfxSdkRuntime::instance().queryEffectVersions(
        FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, m_physDevice, m_device);

    uint64_t selectedVersionId = 0;
    std::string selectedVersionName;
    for (const auto& v : queriedVersions) {
        if (FfxSdkRuntime::parseMajorVersion(v) == desiredMajor) {
            selectedVersionId = v.versionId;
            selectedVersionName = v.versionName;
            break;
        }
    }
    if (selectedVersionId == 0) {
        return;
    }

    auto backendDesc = FfxSdkRuntime::makeBackendVkDesc(m_device, m_physDevice);

    ffx::CreateContextDescOverrideVersion versionOpt{};
    versionOpt.versionId = selectedVersionId;

    ffx::QueryDescUpscaleGetResourceRequirements reqQuery{};
    if (ffx::Query(reqQuery, versionOpt, backendDesc) == ffx::ReturnCode::Ok) {
        m_requiredFfxResources = reqQuery.required_resources;
    }

    ffx::CreateContextDescUpscale createUpscale{};
    createUpscale.flags = FFX_UPSCALE_ENABLE_HIGH_DYNAMIC_RANGE |
                          FFX_UPSCALE_ENABLE_AUTO_EXPOSURE |
                          FFX_UPSCALE_ENABLE_DEPTH_INVERTED |
                          FFX_UPSCALE_ENABLE_DEPTH_INFINITE;
    createUpscale.maxRenderSize = { m_renderWidth, m_renderHeight };
    createUpscale.maxUpscaleSize = { m_displayWidth, m_displayHeight };
    createUpscale.fpMessage = nullptr;

    ffx::CreateContextDescUpscaleVersion headerVersion{};
    headerVersion.version = FFX_UPSCALER_VERSION;

    FfxApiEffectMemoryUsage gpuMem{};
    ffx::QueryDescUpscaleGetGPUMemoryUsageV2 memQuery{};
    memQuery.device = &backendDesc.deviceContext;
    memQuery.maxRenderSize = { m_renderWidth, m_renderHeight };
    memQuery.maxUpscaleSize = { m_displayWidth, m_displayHeight };
    memQuery.flags = createUpscale.flags;
    memQuery.gpuMemoryUsageUpscaler = &gpuMem;
    if (ffx::Query(memQuery, versionOpt) == ffx::ReturnCode::Ok && gpuMem.totalUsageInBytes > 0) {
        m_vramUsageBytes += gpuMem.totalUsageInBytes;
    }

    ffxContext ctx = nullptr;
    ffx::ReturnCode createRet = ffx::CreateContext(ctx, nullptr, createUpscale, backendDesc, headerVersion, versionOpt);
    if (createRet != ffx::ReturnCode::Ok || ctx == nullptr) {
        Logger::error("FsrUpscaler [{}]: ffx::CreateContext for {} returned {}; upscaler context could not be created.",
                      m_instanceName, selectedVersionName, static_cast<uint32_t>(createRet));
        return;
    }
    m_ffxContext = ctx;
    m_providerVersionId = selectedVersionId;
    m_providerVersionName = selectedVersionName;

    ffxQueryGetProviderVersion providerVerQuery{};
    providerVerQuery.header.type = FFX_API_QUERY_DESC_TYPE_GET_PROVIDER_VERSION;
    providerVerQuery.header.pNext = nullptr;
    if (ffxQuery(&m_ffxContext, &providerVerQuery.header) == FFX_API_RETURN_OK && providerVerQuery.versionName) {
        m_providerVersionId = providerVerQuery.versionId;
        m_providerVersionName = providerVerQuery.versionName;
    }

    if (m_activeVersion == FsrVersion::Fsr4) {
        m_activeVersionLabel = "AMD FSR " + (m_providerVersionName.empty() ? std::string("4.1.2") : m_providerVersionName) + " (SDK ML Super-Resolution)";
    } else {
        m_activeVersionLabel = "AMD FSR " + (m_providerVersionName.empty() ? std::string("3.1.6") : m_providerVersionName) + " (SDK Super-Resolution)";
    }
#endif
}

void FsrUpscaler::destroyFfxSdkContext() {
#if defined(PATHWAYS_HAS_FFX_SDK)
    if (m_ffxContext != nullptr) {
        ffx::DestroyContext(m_ffxContext);
        m_ffxContext = nullptr;
    }
#else
    m_ffxContext = nullptr;
#endif
    m_providerVersionId = 0;
    m_providerVersionName.clear();
    m_activeVersionLabel.clear();
}

void FsrUpscaler::initImages() {
    m_format = VK_FORMAT_R16G16B16A16_SFLOAT;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    m_sharpenedImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_displayWidth, m_displayHeight,
        m_format,
        usage
    );

    m_sdkColorInput = std::make_unique<Image>(
        m_device, m_allocator,
        m_renderWidth, m_renderHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        usage
    );

    m_sdkDepthInput = std::make_unique<Image>(
        m_device, m_allocator,
        m_renderWidth, m_renderHeight,
        VK_FORMAT_R32_SFLOAT,
        usage
    );

    m_sdkMotionVectorsInput = std::make_unique<Image>(
        m_device, m_allocator,
        m_renderWidth, m_renderHeight,
        VK_FORMAT_R16G16_SFLOAT,
        usage
    );

    m_historyValid = false;
    for (int i = 0; i < 2; ++i) {
        m_lastBoundInputViews[i] = VK_NULL_HANDLE;
        m_lastBoundMvViews[i] = VK_NULL_HANDLE;
        m_lastBoundDepthViews[i] = VK_NULL_HANDLE;
    }
    m_nextPrepSlot = 0;
}

void FsrUpscaler::createDescriptorSetLayouts() {
    VkDescriptorSetLayoutBinding bindings[6]{};
    for (uint32_t b = 0; b < 6; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = 6;
    layoutInfo.pBindings = bindings;

    if (vkCreateDescriptorSetLayout(m_device, &layoutInfo, nullptr, &m_sdkPrepDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to create SDK prep descriptor set layout!");
    }
}

void FsrUpscaler::allocateDescriptorSets() {
    VkDescriptorPoolSize poolSizes[1]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 16;

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 2;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to create descriptor pool!");
    }

    VkDescriptorSetLayout layouts[2] = { m_sdkPrepDescLayout, m_sdkPrepDescLayout };
    VkDescriptorSetAllocateInfo sdkPrepAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    sdkPrepAlloc.descriptorPool = m_descPool;
    sdkPrepAlloc.descriptorSetCount = 2;
    sdkPrepAlloc.pSetLayouts = layouts;
    if (vkAllocateDescriptorSets(m_device, &sdkPrepAlloc, m_sdkPrepDescSets) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to allocate SDK prep descriptor sets!");
    }
}

void FsrUpscaler::updateDescriptorSets(uint32_t slot, VkImageView inputColorView, VkImageView motionVectorsView, VkImageView depthView) {
    slot &= 1u;
    if (inputColorView == VK_NULL_HANDLE || !m_sharpenedImage ||
        !m_sdkColorInput || !m_sdkDepthInput || !m_sdkMotionVectorsInput) {
        return;
    }

    VkImageView mvView = (motionVectorsView != VK_NULL_HANDLE) ? motionVectorsView : inputColorView;
    VkImageView dView = (depthView != VK_NULL_HANDLE) ? depthView : inputColorView;

    VkDescriptorImageInfo inInfo{ VK_NULL_HANDLE, inputColorView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo mvInfo{ VK_NULL_HANDLE, mvView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo dInfo{ VK_NULL_HANDLE, dView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo sdkColorInfo{ VK_NULL_HANDLE, m_sdkColorInput->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo sdkDepthInfo{ VK_NULL_HANDLE, m_sdkDepthInput->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo sdkMvInfo{ VK_NULL_HANDLE, m_sdkMotionVectorsInput->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

    VkWriteDescriptorSet writes[6]{};
    writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_sdkPrepDescSets[slot], 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &inInfo, nullptr, nullptr };
    writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_sdkPrepDescSets[slot], 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &mvInfo, nullptr, nullptr };
    writes[2] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_sdkPrepDescSets[slot], 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &dInfo, nullptr, nullptr };
    writes[3] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_sdkPrepDescSets[slot], 3, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &sdkColorInfo, nullptr, nullptr };
    writes[4] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_sdkPrepDescSets[slot], 4, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &sdkDepthInfo, nullptr, nullptr };
    writes[5] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, m_sdkPrepDescSets[slot], 5, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &sdkMvInfo, nullptr, nullptr };

    vkUpdateDescriptorSets(m_device, 6, writes, 0, nullptr);
    m_lastBoundInputViews[slot] = inputColorView;
    m_lastBoundMvViews[slot] = mvView;
    m_lastBoundDepthViews[slot] = dView;
}

VkShaderModule FsrUpscaler::createShaderModule(const std::vector<char>& code) {
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to create shader module!");
    }
    return shaderModule;
}

void FsrUpscaler::createPipelines(const std::vector<char>& sdkPrepSpv) {
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(FfxSdkPrepPushConstants);

    VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    layoutInfo.setLayoutCount = 1;
    layoutInfo.pSetLayouts = &m_sdkPrepDescLayout;
    layoutInfo.pushConstantRangeCount = 1;
    layoutInfo.pPushConstantRanges = &pcRange;

    if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &m_sdkPrepPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to create SDK prep pipeline layout!");
    }

    VkShaderModule shaderModule = createShaderModule(sdkPrepSpv);
    VkComputePipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    pipeInfo.layout = m_sdkPrepPipelineLayout;
    pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    pipeInfo.stage.module = shaderModule;
    pipeInfo.stage.pName = "main";

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &m_sdkPrepPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, shaderModule, nullptr);
        throw std::runtime_error("FsrUpscaler: Failed to create SDK prep compute pipeline!");
    }
    vkDestroyShaderModule(m_device, shaderModule, nullptr);
}

void FsrUpscaler::resize(uint32_t renderW, uint32_t renderH, uint32_t displayW, uint32_t displayH) {
    if (m_renderWidth == renderW && m_renderHeight == renderH &&
        m_displayWidth == displayW && m_displayHeight == displayH &&
        m_sharpenedImage != nullptr && m_ffxContext != nullptr) {
        return;
    }

    m_renderWidth = renderW;
    m_renderHeight = renderH;
    m_displayWidth = displayW;
    m_displayHeight = displayH;

    initImages();
    initFfxSdkContext();
    Logger::info("FsrUpscaler resized: Render {}x{}, Display {}x{} ({})",
                 m_renderWidth, m_renderHeight, m_displayWidth, m_displayHeight, getActiveVersionName());
}

void FsrUpscaler::recordUpscale(const UpscalerDispatchDesc& desc) {
    if (!m_sdkPrepPipeline || !m_sharpenedImage) {
        return;
    }

    VkImageView effMvView = (desc.motionVectorsIn != VK_NULL_HANDLE) ? desc.motionVectorsIn : desc.colorIn;
    VkImageView effDepthView = (desc.depthIn != VK_NULL_HANDLE) ? desc.depthIn : desc.colorIn;

    uint32_t prepSlot = 0;
    if (desc.colorIn == m_lastBoundInputViews[0] &&
        effMvView == m_lastBoundMvViews[0] &&
        effDepthView == m_lastBoundDepthViews[0]) {
        prepSlot = 0;
    } else if (desc.colorIn == m_lastBoundInputViews[1] &&
               effMvView == m_lastBoundMvViews[1] &&
               effDepthView == m_lastBoundDepthViews[1]) {
        prepSlot = 1;
    } else {
        prepSlot = m_nextPrepSlot;
        m_nextPrepSlot = (m_nextPrepSlot + 1u) & 1u;
        updateDescriptorSets(prepSlot, desc.colorIn, desc.motionVectorsIn, desc.depthIn);
    }

    if (m_sharpenedImage->getLayout() != VK_IMAGE_LAYOUT_GENERAL) {
        m_sharpenedImage->transitionLayout(
            desc.cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );
    }

#if defined(PATHWAYS_HAS_FFX_SDK)
    if (m_ffxContext != nullptr && m_sdkColorInput && m_sdkDepthInput && m_sdkMotionVectorsInput && m_sdkPrepDescSets[prepSlot] != VK_NULL_HANDLE) {
        if (m_sdkColorInput->getLayout() != VK_IMAGE_LAYOUT_GENERAL) {
            m_sdkColorInput->transitionLayout(
                desc.cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
            );
        }
        if (m_sdkDepthInput->getLayout() != VK_IMAGE_LAYOUT_GENERAL) {
            m_sdkDepthInput->transitionLayout(
                desc.cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
            );
        }
        if (m_sdkMotionVectorsInput->getLayout() != VK_IMAGE_LAYOUT_GENERAL) {
            m_sdkMotionVectorsInput->transitionLayout(
                desc.cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
            );
        }

        vkCmdBindPipeline(desc.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sdkPrepPipeline);
        vkCmdBindDescriptorSets(desc.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_sdkPrepPipelineLayout, 0, 1, &m_sdkPrepDescSets[prepSlot], 0, nullptr);

        FfxSdkPrepPushConstants prepPC{};
        prepPC.renderWidth = m_renderWidth;
        prepPC.renderHeight = m_renderHeight;
        prepPC.sampleScale = desc.inputIsNormalized ? 1.0f : (1.0f / static_cast<float>(std::max(1u, desc.totalSamples)));
        prepPC.nearZ = std::max(desc.cameraNear, 0.001f);

        vkCmdPushConstants(desc.cmd, m_sdkPrepPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(prepPC), &prepPC);
        uint32_t renderGroupsX = (m_renderWidth + 15) / 16;
        uint32_t renderGroupsY = (m_renderHeight + 15) / 16;
        vkCmdDispatch(desc.cmd, renderGroupsX, renderGroupsY, 1);

        m_sdkColorInput->transitionLayout(
            desc.cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT
        );
        m_sdkDepthInput->transitionLayout(
            desc.cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT
        );
        m_sdkMotionVectorsInput->transitionLayout(
            desc.cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_READ_BIT
        );

        ffx::DispatchDescUpscale dispatchUpscale{};
        dispatchUpscale.commandList = reinterpret_cast<void*>(desc.cmd);
        dispatchUpscale.color = FfxSdkRuntime::toFfxResource(m_sdkColorInput.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatchUpscale.depth = FfxSdkRuntime::toFfxResource(m_sdkDepthInput.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatchUpscale.motionVectors = FfxSdkRuntime::toFfxResource(m_sdkMotionVectorsInput.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatchUpscale.exposure = FfxApiResource{};
        dispatchUpscale.reactive = FfxApiResource{};
        dispatchUpscale.transparencyAndComposition = FfxApiResource{};
        dispatchUpscale.output = FfxSdkRuntime::toFfxResource(m_sharpenedImage.get(), FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);
        dispatchUpscale.jitterOffset = { -desc.jitterX, -desc.jitterY };
        dispatchUpscale.motionVectorScale = { -static_cast<float>(m_renderWidth), -static_cast<float>(m_renderHeight) };
        dispatchUpscale.renderSize = { m_renderWidth, m_renderHeight };
        dispatchUpscale.upscaleSize = { m_displayWidth, m_displayHeight };
        dispatchUpscale.enableSharpening = desc.enableSharpening;
        dispatchUpscale.sharpness = desc.enableSharpening ? desc.sharpness : 0.0f;
        dispatchUpscale.frameTimeDelta = std::max(desc.frameTimeDeltaMs, 1.0f);
        dispatchUpscale.preExposure = 1.0f;
        dispatchUpscale.reset = desc.resetHistory || !m_historyValid;
        dispatchUpscale.cameraNear = std::max(desc.cameraNear, 0.001f);
        dispatchUpscale.cameraFar = FLT_MAX;
        dispatchUpscale.cameraFovAngleVertical = (desc.cameraFovY > 0.01f) ? desc.cameraFovY : 0.785398f;
        dispatchUpscale.viewSpaceToMetersFactor = 1.0f;
        dispatchUpscale.flags = 0;

        if (ffx::Dispatch(m_ffxContext, dispatchUpscale) == ffx::ReturnCode::Ok) {
            m_historyValid = true;

            VkImageMemoryBarrier2 sdkPostBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            sdkPostBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            sdkPostBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            sdkPostBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            sdkPostBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            sdkPostBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            sdkPostBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            sdkPostBarrier.image = m_sharpenedImage->getImage();
            sdkPostBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo sdkPostDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            sdkPostDep.imageMemoryBarrierCount = 1;
            sdkPostDep.pImageMemoryBarriers = &sdkPostBarrier;
            vkCmdPipelineBarrier2(desc.cmd, &sdkPostDep);
            return;
        }
        Logger::error("FsrUpscaler [{}]: ffx::Dispatch failed.", m_instanceName);
        return;
    }
#endif
    Logger::error("FsrUpscaler [{}]: FidelityFX SDK Upscaler context is not active; refusing silent fallback.", m_instanceName);
}

} // namespace pathways
