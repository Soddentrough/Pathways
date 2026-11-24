#include "rt/DenoiserPipeline.hpp"
#include "rt/PostProcessPipeline.hpp"
#include "rt/FfxSdkRuntime.hpp"
#include "core/Logger.hpp"

#include <ffx_denoiser.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace pathways {

namespace {

VkDescriptorSetLayoutBinding makeStorageBinding(uint32_t binding) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
}

} // namespace

DenoiserPipeline::DenoiserPipeline(
    VkDevice device,
    VkPhysicalDevice physDevice,
    VmaAllocator allocator,
    VkQueue queue,
    ShaderLoader shaderLoader,
    bool forceWave32
)
    : m_device(device)
    , m_physDevice(physDevice)
    , m_allocator(allocator)
    , m_queue(queue)
    , m_shaderLoader(std::move(shaderLoader))
    , m_forceWave32(forceWave32)
{
    createDescriptorSetLayouts();
    initPipelines();

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_physDevice, &props);
    m_timestampPeriodNs = props.limits.timestampPeriod;
    if (m_timestampPeriodNs > 0.0f) {
        VkQueryPoolCreateInfo qpInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = MAX_FRAMES_IN_FLIGHT * 6;
        if (vkCreateQueryPool(m_device, &qpInfo, nullptr, &m_queryPool) != VK_SUCCESS) {
            Logger::warn("DenoiserPipeline: failed to create timestamp query pool");
            m_queryPool = VK_NULL_HANDLE;
        }
    }

    Logger::info("DenoiserPipeline (3-stage FFX Ray Regeneration 1.2.1) initialized.");
}

DenoiserPipeline::~DenoiserPipeline() {
    destroyFfxDenoiserContext();

    if (m_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_device, m_queryPool, nullptr);
        m_queryPool = VK_NULL_HANDLE;
    }
    if (m_prepPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_prepPipeline, nullptr);
    }
    if (m_composePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_composePipeline, nullptr);
    }
    if (m_prepPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_prepPipelineLayout, nullptr);
    }
    if (m_composePipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_composePipelineLayout, nullptr);
    }
    if (m_descPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
    }
    if (m_prepDescSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_prepDescSetLayout, nullptr);
    }
    if (m_composeDescSetLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_composeDescSetLayout, nullptr);
    }
}

void DenoiserPipeline::createDescriptorSetLayouts() {
    auto createLayout = [this](uint32_t count, VkDescriptorSetLayout& outLayout, const char* name) {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.reserve(count);
        for (uint32_t b = 0; b < count; ++b) {
            bindings.push_back(makeStorageBinding(b));
        }
        VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        info.bindingCount = count;
        info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(m_device, &info, nullptr, &outLayout) != VK_SUCCESS) {
            throw std::runtime_error(std::string("Failed to create DenoiserPipeline layout: ") + name);
        }
    };

    createLayout(17, m_prepDescSetLayout, "ffx_rr_prep");
    createLayout(8, m_composeDescSetLayout, "ffx_rr_compose");

    std::array<VkDescriptorPoolSize, 2> poolSizes = {
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 64 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 8 },
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 16;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create DenoiserPipeline descriptor pool");
    }

    allocateDescriptorSets();
}

void DenoiserPipeline::allocateDescriptorSets() {
    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = m_descPool;
    allocInfo.descriptorSetCount = 1;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        allocInfo.pSetLayouts = &m_prepDescSetLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_prepDescSets[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate DenoiserPipeline prep descriptor set");
        }
        allocInfo.pSetLayouts = &m_composeDescSetLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_composeDescSets[i]) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate DenoiserPipeline compose descriptor set");
        }
    }
}

void DenoiserPipeline::initPipelines() {
    auto createCompute = [this](
        const std::string& spvName,
        VkDescriptorSetLayout descLayout,
        uint32_t pushSize,
        VkPipelineLayout& outLayout,
        VkPipeline& outPipeline
    ) {
        auto code = m_shaderLoader(spvName);

        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = pushSize;

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descLayout;
        layoutInfo.pushConstantRangeCount = pushSize > 0 ? 1u : 0u;
        layoutInfo.pPushConstantRanges = pushSize > 0 ? &pcRange : nullptr;
        if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &outLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create DenoiserPipeline layout for " + spvName);
        }

        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
        };
        subgroupInfo.requiredSubgroupSize = 32;

        VkShaderModuleCreateInfo smInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smInfo.pNext = m_forceWave32 ? &subgroupInfo : nullptr;
        smInfo.codeSize = code.size();
        smInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

        VkComputePipelineCreateInfo compInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        compInfo.layout = outLayout;
        compInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        compInfo.stage.pNext = &smInfo;
        compInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        compInfo.stage.module = VK_NULL_HANDLE;
        compInfo.stage.pName = "main";

        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &compInfo, nullptr, &outPipeline) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create DenoiserPipeline compute pipeline for " + spvName);
        }
    };

    createCompute("ffx_rr_prep.comp.spv", m_prepDescSetLayout, sizeof(PrepPushConstants),
                  m_prepPipelineLayout, m_prepPipeline);
    createCompute("ffx_rr_compose.comp.spv", m_composeDescSetLayout, sizeof(ComposePushConstants),
                  m_composePipelineLayout, m_composePipeline);
}

void DenoiserPipeline::initFfxDenoiserContext() {
    if (m_width == 0 || m_height == 0 || m_ffxDenoiserContext != nullptr) {
        return;
    }
    if (!FfxSdkRuntime::instance().hasNativeProvider(FfxEffectModule::Denoiser)) {
        m_usingSdkProvider = false;
        return;
    }

    ffx::CreateContextDescDenoiser desc{};
    desc.version = FFX_DENOISER_VERSION;
    desc.maxRenderSize = { m_width, m_height };
    desc.signalFlags = FFX_DENOISER_SIGNAL_DIRECT_DIFFUSE
                     | FFX_DENOISER_SIGNAL_DIRECT_SPECULAR
                     | FFX_DENOISER_SIGNAL_INDIRECT_DIFFUSE
                     | FFX_DENOISER_SIGNAL_INDIRECT_SPECULAR;
    desc.checkerboardSignalFlags = FFX_DENOISER_SIGNAL_NONE;
    desc.flags = 0;

    auto backendDesc = FfxSdkRuntime::makeBackendVkDesc(m_device, m_physDevice);
    ffx::ReturnCode rc = ffx::CreateContext(m_ffxDenoiserContext, nullptr, desc, backendDesc);
    if (rc == ffx::ReturnCode::Ok && m_ffxDenoiserContext != nullptr) {
        m_usingSdkProvider = true;
        m_historyValid = false;
        Logger::info("DenoiserPipeline: created AMD FFX Denoiser 1.2.1 context ({}x{}).", m_width, m_height);
    } else {
        m_ffxDenoiserContext = nullptr;
        m_usingSdkProvider = false;
        Logger::warn("DenoiserPipeline: ffx::CreateContext(Denoiser) returned code {}; using passthrough fallback.",
                     static_cast<int>(rc));
    }
}

void DenoiserPipeline::destroyFfxDenoiserContext() {
    if (m_ffxDenoiserContext != nullptr) {
        if (m_queue != VK_NULL_HANDLE) {
            vkQueueWaitIdle(m_queue);
        }
        ffx::DestroyContext(m_ffxDenoiserContext);
        m_ffxDenoiserContext = nullptr;
    }
    m_usingSdkProvider = false;
    m_historyValid = false;
}

void DenoiserPipeline::resize(uint32_t width, uint32_t height, AccumFormat accumFormat, VkCommandBuffer cmd) {
    (void)cmd;
    if (width == 0 || height == 0) {
        return;
    }
    if (m_width == width && m_height == height && m_accumFormat == accumFormat && m_denoisedImage) {
        return;
    }

    destroyFfxDenoiserContext();

    m_width = width;
    m_height = height;
    m_accumFormat = accumFormat;
    invalidateDescriptors();

    constexpr VkImageUsageFlags kUsage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    auto allocImg = [&](uint32_t w, uint32_t h, VkFormat fmt) {
        return std::make_unique<Image>(m_device, m_allocator, w, h, fmt, kUsage);
    };

    m_rrLinearDepth       = allocImg(width, height, VK_FORMAT_R32_SFLOAT);
    m_rrMotionVectors     = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_rrNormals           = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_rrDiffuseAlbedo     = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_rrSpecularAlbedo    = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);

    m_prepDirectDiffuse     = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_prepIndirectDiffuse   = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_prepDirectSpecular    = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_prepIndirectSpecular  = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);

    m_rrOutDirectDiffuse    = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_rrOutIndirectDiffuse  = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_rrOutDirectSpecular   = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_rrOutIndirectSpecular = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);

    const VkFormat outFmt = (accumFormat == AccumFormat::RGBA32_SFLOAT)
        ? VK_FORMAT_R32G32B32A32_SFLOAT
        : VK_FORMAT_R16G16B16A16_SFLOAT;
    m_denoisedImage = allocImg(width, height, outFmt);

    if (!m_fallbackBlack1x1) {
        m_fallbackBlack1x1 = allocImg(1, 1, VK_FORMAT_R16G16B16A16_SFLOAT);
        m_fallbackCleared = false;
    }

    initFfxDenoiserContext();
}

void DenoiserPipeline::initTonemapDescriptors(PostProcessPipeline* postProcess, Image* outputImage) {
    if (!postProcess || !m_denoisedImage || !outputImage) {
        return;
    }
    if (m_tonemapDescSet == VK_NULL_HANDLE) {
        VkDescriptorSetLayout tmLayout = postProcess->getTonemapDescLayout();
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_descPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &tmLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_tonemapDescSet) != VK_SUCCESS) {
            throw std::runtime_error("DenoiserPipeline: failed to allocate tonemap descriptor set");
        }
    }
    postProcess->updateTonemapDescriptors(m_tonemapDescSet, m_denoisedImage.get(), outputImage);
}

void DenoiserPipeline::invalidateDescriptors() {
    m_slotBound = { false, false };
    m_cachedSlotViews = {};
}

void DenoiserPipeline::updateSlotDescriptors(uint32_t slot, const std::array<VkImageView, 9>& extViews) {
    std::array<VkDescriptorImageInfo, 25> imgInfos{};
    std::array<VkWriteDescriptorSet, 25> writes{};
    uint32_t count = 0;

    auto addWrite = [&](VkDescriptorSet dstSet, uint32_t binding, VkImageView view) {
        imgInfos[count] = { VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL };
        VkWriteDescriptorSet& w = writes[count];
        w = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = dstSet;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &imgInfos[count];
        ++count;
    };

    // Pass 0 (ffx_rr_prep.comp): bindings 0..16 (extViews[4] is skipImage for compose binding 6)
    VkDescriptorSet prepSet = m_prepDescSets[slot];
    addWrite(prepSet, 0, extViews[0]); // uInNormalDepth
    addWrite(prepSet, 1, extViews[1]); // uInMotionVectors
    addWrite(prepSet, 2, extViews[2]); // uInAlbedoRoughness
    addWrite(prepSet, 3, extViews[3]); // uInSpecAlbedo
    addWrite(prepSet, 4, extViews[5]); // uInDirectDiffuse
    addWrite(prepSet, 5, extViews[6]); // uInIndirectDiffuse
    addWrite(prepSet, 6, extViews[7]); // uInDirectSpecular
    addWrite(prepSet, 7, extViews[8]); // uInIndirectSpecular

    addWrite(prepSet, 8,  m_rrLinearDepth->getImageView());
    addWrite(prepSet, 9,  m_rrMotionVectors->getImageView());
    addWrite(prepSet, 10, m_rrNormals->getImageView());
    addWrite(prepSet, 11, m_rrDiffuseAlbedo->getImageView());
    addWrite(prepSet, 12, m_rrSpecularAlbedo->getImageView());
    addWrite(prepSet, 13, m_prepDirectDiffuse->getImageView());
    addWrite(prepSet, 14, m_prepIndirectDiffuse->getImageView());
    addWrite(prepSet, 15, m_prepDirectSpecular->getImageView());
    addWrite(prepSet, 16, m_prepIndirectSpecular->getImageView());

    // Pass 2 (ffx_rr_compose.comp): bindings 0..7
    VkDescriptorSet composeSet = m_composeDescSets[slot];
    addWrite(composeSet, 0, m_rrOutDirectDiffuse->getImageView());
    addWrite(composeSet, 1, m_rrOutIndirectDiffuse->getImageView());
    addWrite(composeSet, 2, m_rrOutDirectSpecular->getImageView());
    addWrite(composeSet, 3, m_rrOutIndirectSpecular->getImageView());
    addWrite(composeSet, 4, m_rrDiffuseAlbedo->getImageView());
    addWrite(composeSet, 5, m_rrSpecularAlbedo->getImageView());
    addWrite(composeSet, 6, extViews[4]); // uInSkip
    addWrite(composeSet, 7, m_denoisedImage->getImageView());

    vkUpdateDescriptorSets(m_device, count, writes.data(), 0, nullptr);
    m_cachedSlotViews[slot] = extViews;
    m_slotBound[slot] = true;
}

void DenoiserPipeline::ensureGeneralLayout(VkCommandBuffer cmd, Image* img) {
    if (!img || img->getLayout() == VK_IMAGE_LAYOUT_GENERAL) {
        return;
    }
    const VkImageLayout cur = img->getLayout();
    const bool fromUndefined = (cur == VK_IMAGE_LAYOUT_UNDEFINED);
    img->transitionLayout(
        cmd,
        VK_IMAGE_LAYOUT_GENERAL,
        fromUndefined ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        fromUndefined ? VK_ACCESS_2_NONE : VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    );
}

void DenoiserPipeline::recordPassthroughCopy(VkCommandBuffer cmd) {
    const std::array<Image*, 4> srcs = {
        m_prepDirectDiffuse.get(),
        m_prepIndirectDiffuse.get(),
        m_prepDirectSpecular.get(),
        m_prepIndirectSpecular.get()
    };
    const std::array<Image*, 4> dsts = {
        m_rrOutDirectDiffuse.get(),
        m_rrOutIndirectDiffuse.get(),
        m_rrOutDirectSpecular.get(),
        m_rrOutIndirectSpecular.get()
    };

    for (size_t i = 0; i < 4; ++i) {
        const VkAccessFlags2 srcAccess = (srcs[i]->getLayout() == VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL)
            ? VK_ACCESS_2_SHADER_READ_BIT
            : VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        srcs[i]->transitionLayout(
            cmd,
            VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            srcAccess,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_READ_BIT
        );
        const bool dstUndef = (dsts[i]->getLayout() == VK_IMAGE_LAYOUT_UNDEFINED);
        dsts[i]->transitionLayout(
            cmd,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            dstUndef ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            dstUndef ? VK_ACCESS_2_NONE : VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT
        );
    }

    VkImageCopy copyRegion{};
    copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
    copyRegion.extent = { m_width, m_height, 1 };

    for (size_t i = 0; i < 4; ++i) {
        vkCmdCopyImage(
            cmd,
            srcs[i]->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            dsts[i]->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            1, &copyRegion
        );
    }

    for (Image* dst : dsts) {
        dst->transitionLayout(
            cmd,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    }
}

void DenoiserPipeline::collectQueryResults() {
    if (m_queryPool == VK_NULL_HANDLE || m_timestampPeriodNs <= 0.0f) {
        return;
    }
    for (uint32_t slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
        if (!m_querySlotUsed[slot]) {
            continue;
        }
        std::array<uint64_t, 12> results{};
        const VkResult res = vkGetQueryPoolResults(
            m_device,
            m_queryPool,
            slot * 6,
            6,
            sizeof(results),
            results.data(),
            sizeof(uint64_t) * 2,
            VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT
        );
        if (res == VK_SUCCESS) {
            for (uint32_t p = 0; p < 3; ++p) {
                const uint64_t t0 = results[(p * 2) * 2];
                const uint64_t a0 = results[(p * 2) * 2 + 1];
                const uint64_t t1 = results[(p * 2 + 1) * 2];
                const uint64_t a1 = results[(p * 2 + 1) * 2 + 1];
                if (a0 != 0 && a1 != 0 && t1 >= t0) {
                    m_lastPassTimingsMs[p] = static_cast<double>(t1 - t0) * static_cast<double>(m_timestampPeriodNs) * 1e-6;
                }
            }
            m_lastPassTimingsMs[3] = 0.0;
            m_lastPassTimingsMs[4] = 0.0;
            m_querySlotUsed[slot] = false;
        }
    }
}

void DenoiserPipeline::recordDenoise(
    VkCommandBuffer cmd,
    uint32_t frameIndex,
    bool resetHistory,
    Image* normalDepth,
    Image* motionVectors,
    Image* albedoRoughness,
    Image* specAlbedo,
    Image* skipImage,
    Image* directDiffuse,
    Image* indirectDiffuse,
    Image* directSpecular,
    Image* indirectSpecular,
    const CameraMatrices& cameraMatrices
) {
    if (!isInitialized() || cmd == VK_NULL_HANDLE || m_width == 0 || m_height == 0) {
        return;
    }

    const uint32_t frameSlot = frameIndex % MAX_FRAMES_IN_FLIGHT;

    // Lazy init 1x1 fallback zero image
    if (!m_fallbackCleared && m_fallbackBlack1x1) {
        m_fallbackBlack1x1->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
        );
        VkClearColorValue clearZero = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, m_fallbackBlack1x1->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clearZero, 1, &clearRange);
        m_fallbackBlack1x1->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
        m_fallbackCleared = true;
    }

    auto resolveView = [&](Image* img) -> VkImageView {
        if (img && img->getImageView() != VK_NULL_HANDLE) {
            return img->getImageView();
        }
        return m_fallbackBlack1x1->getImageView();
    };

    const std::array<VkImageView, 9> extViews = {
        resolveView(normalDepth),
        resolveView(motionVectors),
        resolveView(albedoRoughness),
        resolveView(specAlbedo),
        resolveView(skipImage),
        resolveView(directDiffuse),
        resolveView(indirectDiffuse),
        resolveView(directSpecular),
        resolveView(indirectSpecular),
    };

    if (!m_slotBound[frameSlot] || m_cachedSlotViews[frameSlot] != extViews) {
        updateSlotDescriptors(frameSlot, extViews);
    }

    // Ensure all internal buffers are in VK_IMAGE_LAYOUT_GENERAL before Stage 1
    for (Image* img : {
        m_rrLinearDepth.get(), m_rrMotionVectors.get(), m_rrNormals.get(),
        m_rrDiffuseAlbedo.get(), m_rrSpecularAlbedo.get(),
        m_prepDirectDiffuse.get(), m_prepIndirectDiffuse.get(),
        m_prepDirectSpecular.get(), m_prepIndirectSpecular.get(),
        m_rrOutDirectDiffuse.get(), m_rrOutIndirectDiffuse.get(),
        m_rrOutDirectSpecular.get(), m_rrOutIndirectSpecular.get(),
        m_denoisedImage.get()
    }) {
        ensureGeneralLayout(cmd, img);
    }

    auto insertComputeBarrier = [&cmd](VkAccessFlags2 srcAccess, VkAccessFlags2 dstAccess) {
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = srcAccess;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = dstAccess;

        VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    };

    // Synchronize upstream wavefront/NRC storage writes before Stage 1 reads them
    insertComputeBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                         VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);

    collectQueryResults();
    const uint32_t qBase = frameSlot * 6;
    if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, m_queryPool, qBase, 6);
        m_querySlotUsed[frameSlot] = true;
    }
    auto writeTs = [&](uint32_t idx) {
        if (m_queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + idx);
        }
    };

    const uint32_t groupsX = (m_width + 15) / 16;
    const uint32_t groupsY = (m_height + 15) / 16;

    // =========================================================================
    // Pass 0: ffx_rr_prep.comp
    // =========================================================================
    {
        PrepPushConstants pc{};
        pc.width = m_width;
        pc.height = m_height;
        pc.nearZ = std::max(cameraMatrices.nearZ, 0.001f);
        pc.farZ = std::max(cameraMatrices.farZ, pc.nearZ + 1.0f);
        pc.viewInverse = cameraMatrices.viewInverse;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_prepPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_prepPipelineLayout,
                                0, 1, &m_prepDescSets[frameSlot], 0, nullptr);
        vkCmdPushConstants(cmd, m_prepPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        writeTs(0);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        writeTs(1);
    }

    // =========================================================================
    // Pass 1: FFX Denoiser 1.2.1 4-signal dispatch (or vkCmdCopyImage passthrough)
    // =========================================================================
    writeTs(2);
    if (m_ffxDenoiserContext != nullptr) {
        for (Image* inImg : {
            m_rrLinearDepth.get(), m_rrMotionVectors.get(), m_rrNormals.get(),
            m_rrDiffuseAlbedo.get(), m_rrSpecularAlbedo.get(),
            m_prepDirectDiffuse.get(), m_prepIndirectDiffuse.get(),
            m_prepDirectSpecular.get(), m_prepIndirectSpecular.get()
        }) {
            inImg->transitionLayout(
                cmd,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT
            );
        }

        ffx::DispatchDescDenoiser dispatchDesc{};
        dispatchDesc.commandList = reinterpret_cast<void*>(cmd);
        dispatchDesc.linearDepth = FfxSdkRuntime::toFfxResource(m_rrLinearDepth.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatchDesc.motionVectors = FfxSdkRuntime::toFfxResource(m_rrMotionVectors.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatchDesc.normals = FfxSdkRuntime::toFfxResource(m_rrNormals.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatchDesc.specularAlbedo = FfxSdkRuntime::toFfxResource(m_rrSpecularAlbedo.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatchDesc.diffuseAlbedo = FfxSdkRuntime::toFfxResource(m_rrDiffuseAlbedo.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dispatchDesc.motionVectorScale = { 1.0f, 1.0f, 1.0f };
        dispatchDesc.jitterOffsets = { cameraMatrices.jitterOffsets[0], cameraMatrices.jitterOffsets[1] };
        dispatchDesc.cameraPositionDelta = {
            cameraMatrices.cameraPositionDelta[0],
            cameraMatrices.cameraPositionDelta[1],
            cameraMatrices.cameraPositionDelta[2]
        };
        std::memcpy(&dispatchDesc.view, &cameraMatrices.view[0][0], sizeof(dispatchDesc.view));
        std::memcpy(&dispatchDesc.projection, &cameraMatrices.projection[0][0], sizeof(dispatchDesc.projection));
        dispatchDesc.linearDepthBounds = {
            std::max(cameraMatrices.nearZ, 0.001f),
            std::max(cameraMatrices.farZ, cameraMatrices.nearZ + 1.0f)
        };
        dispatchDesc.renderSize = { m_width, m_height };
        dispatchDesc.frameIndex = m_ffxFrameIndex++;
        dispatchDesc.flags = static_cast<uint32_t>(FFX_DENOISER_DISPATCH_NON_ENCODED_ALBEDO)
                           | ((resetHistory || !m_historyValid) ? static_cast<uint32_t>(FFX_DENOISER_DISPATCH_RESET) : 0u);

        ffx::DispatchDescDenoiserDirectDiffuse ddDesc{};
        ddDesc.signal.input = FfxSdkRuntime::toFfxResource(m_prepDirectDiffuse.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        ddDesc.signal.output = FfxSdkRuntime::toFfxResource(m_rrOutDirectDiffuse.get(), FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

        ffx::DispatchDescDenoiserDirectSpecular dsDesc{};
        dsDesc.signal.input = FfxSdkRuntime::toFfxResource(m_prepDirectSpecular.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        dsDesc.signal.output = FfxSdkRuntime::toFfxResource(m_rrOutDirectSpecular.get(), FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

        ffx::DispatchDescDenoiserIndirectDiffuse idDesc{};
        idDesc.signal.input = FfxSdkRuntime::toFfxResource(m_prepIndirectDiffuse.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        idDesc.signal.output = FfxSdkRuntime::toFfxResource(m_rrOutIndirectDiffuse.get(), FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

        ffx::DispatchDescDenoiserIndirectSpecular isDesc{};
        isDesc.signal.input = FfxSdkRuntime::toFfxResource(m_prepIndirectSpecular.get(), FFX_API_RESOURCE_STATE_COMPUTE_READ);
        isDesc.signal.output = FfxSdkRuntime::toFfxResource(m_rrOutIndirectSpecular.get(), FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

        const ffx::ReturnCode rc = ffx::Dispatch(m_ffxDenoiserContext, dispatchDesc, ddDesc, dsDesc, idDesc, isDesc);
        if (rc == ffx::ReturnCode::Ok) {
            m_historyValid = true;
            insertComputeBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        } else {
            Logger::error("DenoiserPipeline: ffx::Dispatch failed with code {}; running passthrough fallback.",
                          static_cast<int>(rc));
            recordPassthroughCopy(cmd);
        }

        // Compose shader binds albedos as storage images (VK_IMAGE_LAYOUT_GENERAL)
        for (Image* albImg : { m_rrDiffuseAlbedo.get(), m_rrSpecularAlbedo.get() }) {
            albImg->transitionLayout(
                cmd,
                VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                VK_ACCESS_2_SHADER_STORAGE_READ_BIT
            );
        }
    } else {
        recordPassthroughCopy(cmd);
        insertComputeBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    }
    writeTs(3);

    // =========================================================================
    // Pass 2: ffx_rr_compose.comp
    // =========================================================================
    {
        ComposePushConstants pc{};
        pc.width = m_width;
        pc.height = m_height;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_composePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_composePipelineLayout,
                                0, 1, &m_composeDescSets[frameSlot], 0, nullptr);
        vkCmdPushConstants(cmd, m_composePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        writeTs(4);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        writeTs(5);
    }

    // Make m_denoisedImage visible to FSR upscaler, post-process tonemap, and staging copy
    {
        VkMemoryBarrier2 postBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        postBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                 | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                 | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        postBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                  | VK_ACCESS_2_SHADER_READ_BIT
                                  | VK_ACCESS_2_TRANSFER_READ_BIT;

        VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &postBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }
}

} // namespace pathways
