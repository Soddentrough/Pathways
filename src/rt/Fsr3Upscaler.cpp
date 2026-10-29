#include "rt/FsrUpscaler.hpp"
#include "rt/FfxSdkRuntime.hpp"
#include "core/Logger.hpp"

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
    const std::vector<char>& upscaleSpv,
    const std::vector<char>& rcasSpv,
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

    Logger::info("FsrUpscaler [{}] initializing: Render {}x{}, Display {}x{}, Format: {}, Target: {}, Active: {} (HW ML CoopMatrix: {})",
                 m_instanceName, m_renderWidth, m_renderHeight, m_displayWidth, m_displayHeight,
                 static_cast<int>(m_format), static_cast<uint32_t>(m_targetVersion),
                 getActiveVersionName(), m_hwMlSupported ? "YES" : "NO");

    initImages();
    createDescriptorSetLayouts();
    allocateDescriptorSets();
    createPipelines(upscaleSpv, rcasSpv);
    initFfxSdkContext();

    Logger::info("FsrUpscaler [{}] successfully initialized ({}, VRAM: {:.2f} MB).",
                 m_instanceName, getActiveVersionName(),
                 static_cast<double>(m_vramUsageBytes) / (1024.0 * 1024.0));
}

FsrUpscaler::~FsrUpscaler() {
    destroyFfxSdkContext();
    if (m_device != VK_NULL_HANDLE) {
        if (m_upscalePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_upscalePipeline, nullptr);
            m_upscalePipeline = VK_NULL_HANDLE;
        }
        if (m_upscalePipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_upscalePipelineLayout, nullptr);
            m_upscalePipelineLayout = VK_NULL_HANDLE;
        }
        if (m_upscaleDescLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_upscaleDescLayout, nullptr);
            m_upscaleDescLayout = VK_NULL_HANDLE;
        }
        if (m_rcasPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_rcasPipeline, nullptr);
            m_rcasPipeline = VK_NULL_HANDLE;
        }
        if (m_rcasPipelineLayout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, m_rcasPipelineLayout, nullptr);
            m_rcasPipelineLayout = VK_NULL_HANDLE;
        }
        if (m_rcasDescLayout != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, m_rcasDescLayout, nullptr);
            m_rcasDescLayout = VK_NULL_HANDLE;
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
    m_hwMlSupported = m_sdkFsr4Supported;
}

void FsrUpscaler::resolveActiveVersion() {
    FsrVersion prevActive = m_activeVersion;
    if (m_targetVersion == FsrVersion::Fsr3) {
        m_activeVersion = FsrVersion::Fsr3;
    } else if (m_targetVersion == FsrVersion::Auto ||
               m_targetVersion == FsrVersion::Fsr4 ||
               m_targetVersion == FsrVersion::Fsr5) {
        if (m_sdkFsr4Supported) {
            m_activeVersion = FsrVersion::Fsr4;
        } else {
            m_activeVersion = FsrVersion::Fsr3;
            if (m_targetVersion != FsrVersion::Auto) {
                Logger::warn("FsrUpscaler [{}]: FSR 4 ML Upscaler provider not available on this device; falling back to FSR 3.",
                             m_instanceName);
            }
        }
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
    if (prevActive != m_activeVersion) {
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
        return "AMD FSR 3.1.6 (SDK Analytical)";
    }
    return "In-House Lanczos-2 Temporal Upscaler (No SDK Context)";
}

void FsrUpscaler::initFfxSdkContext() {
    destroyFfxSdkContext();

    // Calculate baseline VRAM footprint from internal RGBA16F history + upscaled + sharpened targets (8 bytes/pixel * 4 targets)
    m_vramUsageBytes = static_cast<uint64_t>(m_displayWidth) * static_cast<uint64_t>(m_displayHeight) * 8ull * 4ull;

#if defined(PATHWAYS_HAS_FFX_SDK)
    if (!FfxSdkRuntime::instance().hasNativeProvider(FfxEffectModule::Upscaler)) {
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

    ffx::QueryDescUpscaleGetResourceRequirements reqQuery{};
    reqQuery.header.pNext = &backendDesc.header;
    if (ffx::Query(reqQuery) == ffx::ReturnCode::Ok) {
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

    ffx::CreateContextDescOverrideVersion versionOpt{};
    versionOpt.versionId = selectedVersionId;

    FfxApiEffectMemoryUsage gpuMem{};
    ffx::QueryDescUpscaleGetGPUMemoryUsageV2 memQuery{};
    memQuery.device = &backendDesc.deviceContext;
    memQuery.maxRenderSize = { m_renderWidth, m_renderHeight };
    memQuery.maxUpscaleSize = { m_displayWidth, m_displayHeight };
    memQuery.flags = createUpscale.flags;
    memQuery.gpuMemoryUsageUpscaler = &gpuMem;
    if (ffx::Query(memQuery, versionOpt) == ffx::ReturnCode::Ok && gpuMem.totalUsageInBytes > 0) {
        m_vramUsageBytes = gpuMem.totalUsageInBytes;
    }

    ffxContext ctx = nullptr;
    if (ffx::CreateContext(ctx, nullptr, createUpscale, backendDesc, headerVersion, versionOpt) == ffx::ReturnCode::Ok && ctx != nullptr) {
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
            m_activeVersionLabel = "AMD FSR " + (m_providerVersionName.empty() ? std::string("3.1.6") : m_providerVersionName) + " (SDK Analytical)";
        }
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

    m_upscaledImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_displayWidth, m_displayHeight,
        m_format,
        usage
    );

    m_sharpenedImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_displayWidth, m_displayHeight,
        m_format,
        usage
    );

    for (int i = 0; i < 2; ++i) {
        m_historyImages[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_displayWidth, m_displayHeight,
            m_format,
            usage
        );
    }

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

    m_historyIndex = 0;
    m_historyValid = false;
    m_lastBoundInputView = VK_NULL_HANDLE;
    m_lastBoundMvView = VK_NULL_HANDLE;
    m_lastBoundDepthView = VK_NULL_HANDLE;
}

void FsrUpscaler::createDescriptorSetLayouts() {
    // 1. Upscale & Temporal Accumulation Descriptor Layout:
    // Binding 0: Input Render-Resolution Radiance (image2D)
    // Binding 1: Input Render-Resolution Motion Vectors (image2D)
    // Binding 2: Input Render-Resolution Normal & Depth (image2D)
    // Binding 3: Input Display-Resolution Previous History (image2D)
    // Binding 4: Output Display-Resolution Next History (writeonly image2D)
    // Binding 5: Output Display-Resolution Upscaled Radiance for RCAS (writeonly image2D)
    VkDescriptorSetLayoutBinding upscaleBindings[6]{};
    for (uint32_t b = 0; b < 6; ++b) {
        upscaleBindings[b].binding = b;
        upscaleBindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        upscaleBindings[b].descriptorCount = 1;
        upscaleBindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo upscaleLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    upscaleLayoutInfo.bindingCount = 6;
    upscaleLayoutInfo.pBindings = upscaleBindings;

    if (vkCreateDescriptorSetLayout(m_device, &upscaleLayoutInfo, nullptr, &m_upscaleDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to create upscale descriptor set layout!");
    }

    // 2. RCAS Descriptor Layout:
    // Binding 0: Input Display-Resolution Upscaled Radiance (readonly image2D)
    // Binding 1: Output Display-Resolution Sharpened Radiance (writeonly image2D)
    VkDescriptorSetLayoutBinding rcasBindings[2]{};
    rcasBindings[0].binding = 0;
    rcasBindings[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    rcasBindings[0].descriptorCount = 1;
    rcasBindings[0].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    rcasBindings[1].binding = 1;
    rcasBindings[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    rcasBindings[1].descriptorCount = 1;
    rcasBindings[1].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;

    VkDescriptorSetLayoutCreateInfo rcasLayoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    rcasLayoutInfo.bindingCount = 2;
    rcasLayoutInfo.pBindings = rcasBindings;

    if (vkCreateDescriptorSetLayout(m_device, &rcasLayoutInfo, nullptr, &m_rcasDescLayout) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to create RCAS descriptor set layout!");
    }
}

void FsrUpscaler::allocateDescriptorSets() {
    VkDescriptorPoolSize poolSizes[1]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 32; // 3 sets * 6 bindings + 1 set * 2 bindings + headroom

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 6;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to create descriptor pool!");
    }

    VkDescriptorSetLayout upscaleLayouts[2] = { m_upscaleDescLayout, m_upscaleDescLayout };
    VkDescriptorSetAllocateInfo upscaleAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    upscaleAlloc.descriptorPool = m_descPool;
    upscaleAlloc.descriptorSetCount = 2;
    upscaleAlloc.pSetLayouts = upscaleLayouts;
    if (vkAllocateDescriptorSets(m_device, &upscaleAlloc, m_upscaleDescSets) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to allocate upscale descriptor sets!");
    }

    VkDescriptorSetAllocateInfo sdkPrepAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    sdkPrepAlloc.descriptorPool = m_descPool;
    sdkPrepAlloc.descriptorSetCount = 1;
    sdkPrepAlloc.pSetLayouts = &m_upscaleDescLayout;
    if (vkAllocateDescriptorSets(m_device, &sdkPrepAlloc, &m_sdkPrepDescSet) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to allocate SDK prep descriptor set!");
    }

    VkDescriptorSetAllocateInfo rcasAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    rcasAlloc.descriptorPool = m_descPool;
    rcasAlloc.descriptorSetCount = 1;
    rcasAlloc.pSetLayouts = &m_rcasDescLayout;
    if (vkAllocateDescriptorSets(m_device, &rcasAlloc, &m_rcasDescSet) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to allocate RCAS descriptor set!");
    }
}

void FsrUpscaler::updateDescriptorSets(VkImageView inputColorView, VkImageView motionVectorsView, VkImageView depthView) {
    if (inputColorView == VK_NULL_HANDLE || !m_upscaledImage || !m_sharpenedImage || !m_historyImages[0] || !m_historyImages[1] ||
        !m_sdkColorInput || !m_sdkDepthInput || !m_sdkMotionVectorsInput) {
        return;
    }

    VkImageView mvView = (motionVectorsView != VK_NULL_HANDLE) ? motionVectorsView : inputColorView;
    VkImageView dView = (depthView != VK_NULL_HANDLE) ? depthView : inputColorView;

    VkDescriptorImageInfo inInfo{ VK_NULL_HANDLE, inputColorView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo mvInfo{ VK_NULL_HANDLE, mvView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo dInfo{ VK_NULL_HANDLE, dView, VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo hist0Info{ VK_NULL_HANDLE, m_historyImages[0]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo hist1Info{ VK_NULL_HANDLE, m_historyImages[1]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo upscaledInfo{ VK_NULL_HANDLE, m_upscaledImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo sharpenedInfo{ VK_NULL_HANDLE, m_sharpenedImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo sdkColorInfo{ VK_NULL_HANDLE, m_sdkColorInput->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo sdkDepthInfo{ VK_NULL_HANDLE, m_sdkDepthInput->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
    VkDescriptorImageInfo sdkMvInfo{ VK_NULL_HANDLE, m_sdkMotionVectorsInput->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

    // 20 writes: 6 for Set 0, 6 for Set 1, 2 for RCAS, 6 for SDK Prep Set
    VkWriteDescriptorSet writes[20]{};

    // Set 0: Read history 0 -> Write history 1
    writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[0].dstSet = m_upscaleDescSets[0];
    writes[0].dstBinding = 0;
    writes[0].descriptorCount = 1;
    writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[0].pImageInfo = &inInfo;

    writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[1].dstSet = m_upscaleDescSets[0];
    writes[1].dstBinding = 1;
    writes[1].descriptorCount = 1;
    writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[1].pImageInfo = &mvInfo;

    writes[2].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[2].dstSet = m_upscaleDescSets[0];
    writes[2].dstBinding = 2;
    writes[2].descriptorCount = 1;
    writes[2].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[2].pImageInfo = &dInfo;

    writes[3].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[3].dstSet = m_upscaleDescSets[0];
    writes[3].dstBinding = 3;
    writes[3].descriptorCount = 1;
    writes[3].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[3].pImageInfo = &hist0Info;

    writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[4].dstSet = m_upscaleDescSets[0];
    writes[4].dstBinding = 4;
    writes[4].descriptorCount = 1;
    writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[4].pImageInfo = &hist1Info;

    writes[5].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[5].dstSet = m_upscaleDescSets[0];
    writes[5].dstBinding = 5;
    writes[5].descriptorCount = 1;
    writes[5].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[5].pImageInfo = &upscaledInfo;

    // Set 1: Read history 1 -> Write history 0
    writes[6].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[6].dstSet = m_upscaleDescSets[1];
    writes[6].dstBinding = 0;
    writes[6].descriptorCount = 1;
    writes[6].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[6].pImageInfo = &inInfo;

    writes[7].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[7].dstSet = m_upscaleDescSets[1];
    writes[7].dstBinding = 1;
    writes[7].descriptorCount = 1;
    writes[7].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[7].pImageInfo = &mvInfo;

    writes[8].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[8].dstSet = m_upscaleDescSets[1];
    writes[8].dstBinding = 2;
    writes[8].descriptorCount = 1;
    writes[8].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[8].pImageInfo = &dInfo;

    writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[9].dstSet = m_upscaleDescSets[1];
    writes[9].dstBinding = 3;
    writes[9].descriptorCount = 1;
    writes[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[9].pImageInfo = &hist1Info;

    writes[10].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[10].dstSet = m_upscaleDescSets[1];
    writes[10].dstBinding = 4;
    writes[10].descriptorCount = 1;
    writes[10].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[10].pImageInfo = &hist0Info;

    writes[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[11].dstSet = m_upscaleDescSets[1];
    writes[11].dstBinding = 5;
    writes[11].descriptorCount = 1;
    writes[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[11].pImageInfo = &upscaledInfo;

    // RCAS bindings
    writes[12].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[12].dstSet = m_rcasDescSet;
    writes[12].dstBinding = 0;
    writes[12].descriptorCount = 1;
    writes[12].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[12].pImageInfo = &upscaledInfo;

    writes[13].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[13].dstSet = m_rcasDescSet;
    writes[13].dstBinding = 1;
    writes[13].descriptorCount = 1;
    writes[13].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[13].pImageInfo = &sharpenedInfo;

    // SDK Prep bindings
    writes[14].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[14].dstSet = m_sdkPrepDescSet;
    writes[14].dstBinding = 0;
    writes[14].descriptorCount = 1;
    writes[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[14].pImageInfo = &inInfo;

    writes[15].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[15].dstSet = m_sdkPrepDescSet;
    writes[15].dstBinding = 1;
    writes[15].descriptorCount = 1;
    writes[15].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[15].pImageInfo = &mvInfo;

    writes[16].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[16].dstSet = m_sdkPrepDescSet;
    writes[16].dstBinding = 2;
    writes[16].descriptorCount = 1;
    writes[16].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[16].pImageInfo = &dInfo;

    writes[17].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[17].dstSet = m_sdkPrepDescSet;
    writes[17].dstBinding = 3;
    writes[17].descriptorCount = 1;
    writes[17].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[17].pImageInfo = &sdkColorInfo;

    writes[18].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[18].dstSet = m_sdkPrepDescSet;
    writes[18].dstBinding = 4;
    writes[18].descriptorCount = 1;
    writes[18].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[18].pImageInfo = &sdkDepthInfo;

    writes[19].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    writes[19].dstSet = m_sdkPrepDescSet;
    writes[19].dstBinding = 5;
    writes[19].descriptorCount = 1;
    writes[19].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    writes[19].pImageInfo = &sdkMvInfo;

    vkUpdateDescriptorSets(m_device, 20, writes, 0, nullptr);
    m_lastBoundInputView = inputColorView;
    m_lastBoundMvView = mvView;
    m_lastBoundDepthView = dView;
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

void FsrUpscaler::createPipelines(const std::vector<char>& upscaleSpv, const std::vector<char>& rcasSpv) {
    // 1. Upscale Pipeline Layout
    VkPushConstantRange upscalePcRange{};
    upscalePcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    upscalePcRange.offset = 0;
    upscalePcRange.size = sizeof(Fsr3UpscalePushConstants);

    VkPipelineLayoutCreateInfo upscaleLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    upscaleLayoutInfo.setLayoutCount = 1;
    upscaleLayoutInfo.pSetLayouts = &m_upscaleDescLayout;
    upscaleLayoutInfo.pushConstantRangeCount = 1;
    upscaleLayoutInfo.pPushConstantRanges = &upscalePcRange;

    if (vkCreatePipelineLayout(m_device, &upscaleLayoutInfo, nullptr, &m_upscalePipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to create upscale pipeline layout!");
    }

    VkShaderModule upscaleModule = createShaderModule(upscaleSpv);
    VkComputePipelineCreateInfo upscaleInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    upscaleInfo.layout = m_upscalePipelineLayout;
    upscaleInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    upscaleInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    upscaleInfo.stage.module = upscaleModule;
    upscaleInfo.stage.pName = "main";

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &upscaleInfo, nullptr, &m_upscalePipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, upscaleModule, nullptr);
        throw std::runtime_error("FsrUpscaler: Failed to create upscale compute pipeline!");
    }
    vkDestroyShaderModule(m_device, upscaleModule, nullptr);

    // 2. RCAS Pipeline Layout
    VkPushConstantRange rcasPcRange{};
    rcasPcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    rcasPcRange.offset = 0;
    rcasPcRange.size = sizeof(Fsr3RcasPushConstants);

    VkPipelineLayoutCreateInfo rcasLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    rcasLayoutInfo.setLayoutCount = 1;
    rcasLayoutInfo.pSetLayouts = &m_rcasDescLayout;
    rcasLayoutInfo.pushConstantRangeCount = 1;
    rcasLayoutInfo.pPushConstantRanges = &rcasPcRange;

    if (vkCreatePipelineLayout(m_device, &rcasLayoutInfo, nullptr, &m_rcasPipelineLayout) != VK_SUCCESS) {
        throw std::runtime_error("FsrUpscaler: Failed to create RCAS pipeline layout!");
    }

    VkShaderModule rcasModule = createShaderModule(rcasSpv);
    VkComputePipelineCreateInfo rcasInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    rcasInfo.layout = m_rcasPipelineLayout;
    rcasInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    rcasInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    rcasInfo.stage.module = rcasModule;
    rcasInfo.stage.pName = "main";

    if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &rcasInfo, nullptr, &m_rcasPipeline) != VK_SUCCESS) {
        vkDestroyShaderModule(m_device, rcasModule, nullptr);
        throw std::runtime_error("FsrUpscaler: Failed to create RCAS compute pipeline!");
    }
    vkDestroyShaderModule(m_device, rcasModule, nullptr);
}

void FsrUpscaler::resize(uint32_t renderW, uint32_t renderH, uint32_t displayW, uint32_t displayH) {
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
    if (!m_upscalePipeline || !m_rcasPipeline || !m_upscaledImage || !m_sharpenedImage ||
        !m_historyImages[0] || !m_historyImages[1]) {
        return;
    }

    if (desc.colorIn != m_lastBoundInputView ||
        desc.motionVectorsIn != m_lastBoundMvView ||
        desc.depthIn != m_lastBoundDepthView) {
        updateDescriptorSets(desc.colorIn, desc.motionVectorsIn, desc.depthIn);
    }

    // Transition initial layouts if needed
    if (m_upscaledImage->getLayout() != VK_IMAGE_LAYOUT_GENERAL) {
        m_upscaledImage->transitionLayout(
            desc.cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );
    }
    if (m_sharpenedImage->getLayout() != VK_IMAGE_LAYOUT_GENERAL) {
        m_sharpenedImage->transitionLayout(
            desc.cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );
    }

#if defined(PATHWAYS_HAS_FFX_SDK)
    if (m_ffxContext != nullptr && m_sdkColorInput && m_sdkDepthInput && m_sdkMotionVectorsInput && m_sdkPrepDescSet != VK_NULL_HANDLE) {
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

        // Prepare render-resolution SDK input textures (RGBA16F Color, R32F Inverted Device Depth, RG16F Motion Vectors)
        vkCmdBindPipeline(desc.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upscalePipeline);
        vkCmdBindDescriptorSets(desc.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upscalePipelineLayout, 0, 1, &m_sdkPrepDescSet, 0, nullptr);

        Fsr3UpscalePushConstants prepPC{};
        prepPC.renderWidth = m_renderWidth;
        prepPC.renderHeight = m_renderHeight;
        prepPC.displayWidth = m_displayWidth;
        prepPC.displayHeight = m_displayHeight;
        prepPC.invRenderWidth = 1.0f / static_cast<float>(m_renderWidth);
        prepPC.invRenderHeight = 1.0f / static_cast<float>(m_renderHeight);
        prepPC.invDisplayWidth = 1.0f / static_cast<float>(m_displayWidth);
        prepPC.invDisplayHeight = 1.0f / static_cast<float>(m_displayHeight);
        prepPC.invTotalSamples = desc.inputIsNormalized ? 1.0f : (1.0f / static_cast<float>(std::max(1u, desc.totalSamples)));
        prepPC.temporalWeight = std::max(desc.cameraNear, 0.001f); // Passed as cameraNear for inverted device depth conversion
        prepPC.jitterX = desc.jitterX;
        prepPC.jitterY = desc.jitterY;
        prepPC.frameIndex = desc.frameIndex;
        prepPC.resetHistory = (desc.resetHistory || !m_historyValid) ? 1u : 0u;
        prepPC.cameraMoved = desc.cameraMoved ? 1u : 0u;
        prepPC.isInputNormalized = desc.inputIsNormalized ? 1u : 0u;
        prepPC.fsrVersion = 0xFF00u;

        vkCmdPushConstants(desc.cmd, m_upscalePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(prepPC), &prepPC);
        uint32_t renderGroupsX = (m_renderWidth + 15) / 16;
        uint32_t renderGroupsY = (m_renderHeight + 15) / 16;
        vkCmdDispatch(desc.cmd, renderGroupsX, renderGroupsY, 1);

        // Transition SDK input textures to SHADER_READ_ONLY_OPTIMAL (FFX_API_RESOURCE_STATE_COMPUTE_READ)
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
        Logger::warn("FsrUpscaler [{}]: ffx::Dispatch failed; falling back to compute path.", m_instanceName);
    }
#endif
    for (int i = 0; i < 2; ++i) {
        if (m_historyImages[i]->getLayout() != VK_IMAGE_LAYOUT_GENERAL) {
            m_historyImages[i]->transitionLayout(
                desc.cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
            );
        }
    }

    // 1. Pass 1: Separable Lanczos-2 Super-Resolution with YCoCg AABB Clamping & Temporal Accumulation
    vkCmdBindPipeline(desc.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upscalePipeline);
    vkCmdBindDescriptorSets(desc.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upscalePipelineLayout, 0, 1, &m_upscaleDescSets[m_historyIndex], 0, nullptr);

    Fsr3UpscalePushConstants upscalePC{};
    upscalePC.renderWidth = m_renderWidth;
    upscalePC.renderHeight = m_renderHeight;
    upscalePC.displayWidth = m_displayWidth;
    upscalePC.displayHeight = m_displayHeight;
    upscalePC.invRenderWidth = 1.0f / static_cast<float>(m_renderWidth);
    upscalePC.invRenderHeight = 1.0f / static_cast<float>(m_renderHeight);
    upscalePC.invDisplayWidth = 1.0f / static_cast<float>(m_displayWidth);
    upscalePC.invDisplayHeight = 1.0f / static_cast<float>(m_displayHeight);
    upscalePC.invTotalSamples = desc.inputIsNormalized ? 1.0f : (1.0f / static_cast<float>(std::max(1u, desc.totalSamples)));
    upscalePC.temporalWeight = (desc.temporalWeight > 0.0f && desc.temporalWeight != 0.88f) ? desc.temporalWeight : 0.90f;
    upscalePC.jitterX = desc.jitterX;
    upscalePC.jitterY = desc.jitterY;
    upscalePC.frameIndex = desc.frameIndex;
    upscalePC.resetHistory = (desc.resetHistory || !m_historyValid) ? 1u : 0u;
    upscalePC.cameraMoved = desc.cameraMoved ? 1u : 0u;
    upscalePC.isInputNormalized = desc.inputIsNormalized ? 1u : 0u;
    upscalePC.fsrVersion = (m_activeVersion == FsrVersion::Fsr4) ? 4u : 3u;

    vkCmdPushConstants(desc.cmd, m_upscalePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(upscalePC), &upscalePC);

    uint32_t dispGroupsX = (m_displayWidth + 15) / 16;
    uint32_t dispGroupsY = (m_displayHeight + 15) / 16;
    vkCmdDispatch(desc.cmd, dispGroupsX, dispGroupsY, 1);

    // Written history image index (Set 0 writes to 1, Set 1 writes to 0)
    uint32_t writtenHistIdx = (m_historyIndex == 0) ? 1 : 0;

    // Barrier: Upscaled image write -> RCAS read, and Written history -> next frame read
    VkImageMemoryBarrier2 barriers[2]{};
    barriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    barriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[0].image = m_upscaledImage->getImage();
    barriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    barriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    barriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    barriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    barriers[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    barriers[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
    barriers[1].image = m_historyImages[writtenHistIdx]->getImage();
    barriers[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo u2rDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    u2rDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    u2rDep.imageMemoryBarrierCount = 2;
    u2rDep.pImageMemoryBarriers = barriers;
    vkCmdPipelineBarrier2(desc.cmd, &u2rDep);

    m_historyIndex = 1 - m_historyIndex;
    m_historyValid = true;

    // 2. Pass 2: Robust Contrast-Adaptive Sharpening (RCAS)
    vkCmdBindPipeline(desc.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rcasPipeline);
    vkCmdBindDescriptorSets(desc.cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_rcasPipelineLayout, 0, 1, &m_rcasDescSet, 0, nullptr);

    Fsr3RcasPushConstants rcasPC{};
    rcasPC.displayWidth = m_displayWidth;
    rcasPC.displayHeight = m_displayHeight;
    rcasPC.enableSharpening = desc.enableSharpening ? 1u : 0u;
    rcasPC.sharpness = desc.enableSharpening ? desc.sharpness : 0.0f;

    vkCmdPushConstants(desc.cmd, m_rcasPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(rcasPC), &rcasPC);
    vkCmdDispatch(desc.cmd, dispGroupsX, dispGroupsY, 1);

    // Barrier: Sharpened image write -> Downstream Tonemapper compute read
    VkImageMemoryBarrier2 postBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
    postBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
    postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
    postBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
    postBarrier.image = m_sharpenedImage->getImage();
    postBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    postDep.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
    postDep.imageMemoryBarrierCount = 1;
    postDep.pImageMemoryBarriers = &postBarrier;
    vkCmdPipelineBarrier2(desc.cmd, &postDep);
}

} // namespace pathways
