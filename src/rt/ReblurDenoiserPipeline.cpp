#include "rt/ReblurDenoiserPipeline.hpp"
#include "rt/PostProcessPipeline.hpp"
#include "core/Logger.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace pathways {

namespace {

VkDescriptorSetLayoutBinding makeBinding(uint32_t binding, VkDescriptorType type) {
    VkDescriptorSetLayoutBinding b{};
    b.binding = binding;
    b.descriptorType = type;
    b.descriptorCount = 1;
    b.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    return b;
}

} // namespace

ReblurDenoiserPipeline::ReblurDenoiserPipeline(
    VkDevice device,
    VkPhysicalDevice physDevice,
    VmaAllocator allocator,
    VkQueue queue,
    ShaderLoader shaderLoader,
    bool forceWave32,
    VkPipelineCache pipelineCache
)
    : m_device(device)
    , m_physDevice(physDevice)
    , m_allocator(allocator)
    , m_queue(queue)
    , m_shaderLoader(std::move(shaderLoader))
    , m_forceWave32(forceWave32)
    , m_pipelineCache(pipelineCache)
{
    createSampler();
    createDescriptorSetLayouts();
    initPipelines();

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_cameraUbos[i] = std::make_unique<Buffer>(
            m_allocator,
            sizeof(CameraUboData),
            VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
        );
    }

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_physDevice, &props);
    m_timestampPeriodNs = props.limits.timestampPeriod;
    if (m_timestampPeriodNs > 0.0f) {
        VkQueryPoolCreateInfo qpInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qpInfo.queryCount = MAX_FRAMES_IN_FLIGHT * QUERIES_PER_SLOT;
        if (vkCreateQueryPool(m_device, &qpInfo, nullptr, &m_queryPool) != VK_SUCCESS) {
            Logger::warn("ReblurDenoiserPipeline: failed to create timestamp query pool");
            m_queryPool = VK_NULL_HANDLE;
        }
    }

    Logger::info("ReblurDenoiserPipeline (6-pass NRD ReBLUR analytical denoiser) initialized.");
}

ReblurDenoiserPipeline::~ReblurDenoiserPipeline() {
    if (m_device == VK_NULL_HANDLE) {
        return;
    }

    if (m_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_device, m_queryPool, nullptr);
        m_queryPool = VK_NULL_HANDLE;
    }

    auto destroyPipe = [this](VkPipeline& pipe, VkPipelineLayout& layout, VkDescriptorSetLayout& desc) {
        if (pipe != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, pipe, nullptr);
            pipe = VK_NULL_HANDLE;
        }
        if (layout != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, layout, nullptr);
            layout = VK_NULL_HANDLE;
        }
        if (desc != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, desc, nullptr);
            desc = VK_NULL_HANDLE;
        }
    };

    destroyPipe(m_preblurPipeline,    m_preblurPipelineLayout,    m_preblurDescLayout);
    destroyPipe(m_temporalPipeline,   m_temporalPipelineLayout,   m_temporalDescLayout);
    destroyPipe(m_historyFixPipeline, m_historyFixPipelineLayout, m_historyFixDescLayout);
    destroyPipe(m_atrousPipeline,     m_atrousPipelineLayout,     m_atrousDescLayout);
    destroyPipe(m_stabilizePipeline,  m_stabilizePipelineLayout,  m_stabilizeDescLayout);

    if (m_descPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
        m_descPool = VK_NULL_HANDLE;
    }
    if (m_linearSampler != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_linearSampler, nullptr);
        m_linearSampler = VK_NULL_HANDLE;
    }
}

void ReblurDenoiserPipeline::createSampler() {
    VkSamplerCreateInfo info{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    info.magFilter = VK_FILTER_LINEAR;
    info.minFilter = VK_FILTER_LINEAR;
    info.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    info.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    info.anisotropyEnable = VK_FALSE;
    info.maxAnisotropy = 1.0f;
    info.compareEnable = VK_FALSE;
    info.minLod = 0.0f;
    info.maxLod = 0.0f;
    info.borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK;
    info.unnormalizedCoordinates = VK_FALSE;
    if (vkCreateSampler(m_device, &info, nullptr, &m_linearSampler) != VK_SUCCESS) {
        throw std::runtime_error("ReblurDenoiserPipeline: failed to create linear sampler");
    }
}

void ReblurDenoiserPipeline::createDescriptorSetLayouts() {
    auto createLayout = [this](const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                               VkDescriptorSetLayout& outLayout, const char* name) {
        VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        info.bindingCount = static_cast<uint32_t>(bindings.size());
        info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(m_device, &info, nullptr, &outLayout) != VK_SUCCESS) {
            throw std::runtime_error(std::string("ReblurDenoiserPipeline: failed to create layout ") + name);
        }
    };

    // Pass 0 (reblur_preblur.comp): binding 0 UBO + bindings 1..12 STORAGE_IMAGE
    {
        std::vector<VkDescriptorSetLayoutBinding> b;
        b.reserve(13);
        b.push_back(makeBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER));
        for (uint32_t i = 1; i <= 12; ++i) b.push_back(makeBinding(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        createLayout(b, m_preblurDescLayout, "reblur_preblur");
    }

    // Pass 1 (reblur_temporal.comp): binding 0 UBO + bindings 1..17 STORAGE_IMAGE
    {
        std::vector<VkDescriptorSetLayoutBinding> b;
        b.reserve(18);
        b.push_back(makeBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER));
        for (uint32_t i = 1; i <= 17; ++i) b.push_back(makeBinding(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        createLayout(b, m_temporalDescLayout, "reblur_temporal");
    }

    // Pass 2 (reblur_history_fix.comp): binding 0 UBO + bindings 1..8 STORAGE_IMAGE
    {
        std::vector<VkDescriptorSetLayoutBinding> b;
        b.reserve(9);
        b.push_back(makeBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER));
        for (uint32_t i = 1; i <= 8; ++i) b.push_back(makeBinding(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        createLayout(b, m_historyFixDescLayout, "reblur_history_fix");
    }

    // Passes 3-4 (reblur_atrous.comp): binding 0 UBO + bindings 1..8 STORAGE_IMAGE
    {
        std::vector<VkDescriptorSetLayoutBinding> b;
        b.reserve(9);
        b.push_back(makeBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER));
        for (uint32_t i = 1; i <= 8; ++i) b.push_back(makeBinding(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        createLayout(b, m_atrousDescLayout, "reblur_atrous");
    }

    // Pass 5 (reblur_stabilize.comp): 0 UBO + 1..8 STORAGE_IMAGE + 9 COMBINED_IMAGE_SAMPLER + 10..11 STORAGE_IMAGE
    {
        std::vector<VkDescriptorSetLayoutBinding> b;
        b.reserve(12);
        b.push_back(makeBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER));
        for (uint32_t i = 1; i <= 8; ++i) b.push_back(makeBinding(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        b.push_back(makeBinding(9, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER));
        b.push_back(makeBinding(10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        b.push_back(makeBinding(11, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        createLayout(b, m_stabilizeDescLayout, "reblur_stabilize");
    }

    std::array<VkDescriptorPoolSize, 4> poolSizes = {
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 32 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 320 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 16 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8 },
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 32;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool) != VK_SUCCESS) {
        throw std::runtime_error("ReblurDenoiserPipeline: failed to create descriptor pool");
    }

    allocateDescriptorSets();
}

void ReblurDenoiserPipeline::allocateDescriptorSets() {
    auto allocOne = [this](VkDescriptorSetLayout layout, VkDescriptorSet& outSet, const char* name) {
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_descPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &layout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &outSet) != VK_SUCCESS) {
            throw std::runtime_error(std::string("ReblurDenoiserPipeline: failed to allocate descriptor set ") + name);
        }
    };

    for (uint32_t v = 0; v < NUM_DESC_VARIANTS; ++v) {
        allocOne(m_preblurDescLayout,    m_preblurDescSets[v],    "preblur");
        allocOne(m_temporalDescLayout,   m_temporalDescSets[v],   "temporal");
        allocOne(m_historyFixDescLayout, m_historyFixDescSets[v], "history_fix");
        allocOne(m_atrousDescLayout,     m_atrousDescSets[v][0],  "atrous_0");
        allocOne(m_atrousDescLayout,     m_atrousDescSets[v][1],  "atrous_1");
        allocOne(m_stabilizeDescLayout,  m_stabilizeDescSets[v],  "stabilize");
    }
}

void ReblurDenoiserPipeline::initPipelines() {
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
            throw std::runtime_error("ReblurDenoiserPipeline: failed to create pipeline layout for " + spvName);
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

        if (vkCreateComputePipelines(m_device, m_pipelineCache, 1, &compInfo, nullptr, &outPipeline) != VK_SUCCESS) {
            throw std::runtime_error("ReblurDenoiserPipeline: failed to create compute pipeline for " + spvName);
        }
    };

    createCompute("reblur_preblur.comp.spv",     m_preblurDescLayout,    sizeof(PreblurPushConstants),
                  m_preblurPipelineLayout,       m_preblurPipeline);
    createCompute("reblur_temporal.comp.spv",    m_temporalDescLayout,   sizeof(TemporalPushConstants),
                  m_temporalPipelineLayout,      m_temporalPipeline);
    createCompute("reblur_history_fix.comp.spv", m_historyFixDescLayout, sizeof(HistoryFixPushConstants),
                  m_historyFixPipelineLayout,    m_historyFixPipeline);
    createCompute("reblur_atrous.comp.spv",      m_atrousDescLayout,     sizeof(AtrousPushConstants),
                  m_atrousPipelineLayout,        m_atrousPipeline);
    createCompute("reblur_stabilize.comp.spv",   m_stabilizeDescLayout,  sizeof(StabilizePushConstants),
                  m_stabilizePipelineLayout,     m_stabilizePipeline);
}

void ReblurDenoiserPipeline::resize(uint32_t width, uint32_t height, AccumFormat accumFormat, VkCommandBuffer cmd) {
    (void)cmd;
    if (width == 0 || height == 0) {
        return;
    }
    if (m_width == width && m_height == height && m_accumFormat == accumFormat && m_denoisedImage) {
        return;
    }

    m_width = width;
    m_height = height;
    m_accumFormat = accumFormat;
    m_historyValid = false;
    m_needsInitialClear = true;
    m_historyParity = 0;
    invalidateDescriptors();

    constexpr VkImageUsageFlags kUsage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    auto allocImg = [&](uint32_t w, uint32_t h, VkFormat fmt) {
        return std::make_unique<Image>(m_device, m_allocator, w, h, fmt, kUsage);
    };

    m_preDiff = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
    m_preSpec = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);

    for (uint32_t i = 0; i < 2; ++i) {
        m_currWorldND[i]  = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
        m_histDiff[i]     = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
        m_histSpec[i]     = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
        m_histMoments[i]  = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
        m_histMeta[i]     = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
        m_histHitT[i]     = allocImg(width, height, VK_FORMAT_R16G16_SFLOAT);
        m_pingPongDiff[i] = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
        m_pingPongSpec[i] = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
        m_stabHistory[i]  = allocImg(width, height, VK_FORMAT_R16G16B16A16_SFLOAT);
    }

    const VkFormat outFmt = (accumFormat == AccumFormat::RGBA32_SFLOAT)
        ? VK_FORMAT_R32G32B32A32_SFLOAT
        : VK_FORMAT_R16G16B16A16_SFLOAT;
    m_denoisedImage = allocImg(width, height, outFmt);

    if (!m_fallbackBlack1x1) {
        m_fallbackBlack1x1 = allocImg(1, 1, VK_FORMAT_R16G16B16A16_SFLOAT);
        m_fallbackCleared = false;
    }

    Logger::info("ReblurDenoiserPipeline: allocated internal buffers ({}x{}, accumFormat={}).",
                 width, height, static_cast<int>(accumFormat));
}

void ReblurDenoiserPipeline::initTonemapDescriptors(PostProcessPipeline* postProcess, Image* outputImage) {
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
            throw std::runtime_error("ReblurDenoiserPipeline: failed to allocate tonemap descriptor set");
        }
    }
    postProcess->updateTonemapDescriptors(m_tonemapDescSet, m_denoisedImage.get(), outputImage);
}

void ReblurDenoiserPipeline::invalidateDescriptors() {
    m_variantBound.fill(false);
    m_cachedVariantViews = {};
}

void ReblurDenoiserPipeline::updateVariantDescriptors(
    uint32_t frameSlot,
    uint32_t currParity,
    const std::array<VkImageView, 10>& extViews
) {
    const uint32_t variantIdx = frameSlot * 2u + currParity;
    const uint32_t prevParity = 1u - currParity;

    std::vector<VkDescriptorBufferInfo> bufInfos;
    std::vector<VkDescriptorImageInfo> imgInfos;
    std::vector<VkWriteDescriptorSet> writes;
    bufInfos.reserve(8);
    imgInfos.reserve(72);
    writes.reserve(80);

    auto addUbo = [&](VkDescriptorSet dstSet, uint32_t binding) {
        bufInfos.push_back({ m_cameraUbos[frameSlot]->getBuffer(), 0, sizeof(CameraUboData) });
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = dstSet;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo = &bufInfos.back();
        writes.push_back(w);
    };

    auto addImg = [&](VkDescriptorSet dstSet, uint32_t binding, VkImageView view) {
        imgInfos.push_back({ VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL });
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = dstSet;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &imgInfos.back();
        writes.push_back(w);
    };

    auto addSampledImg = [&](VkDescriptorSet dstSet, uint32_t binding, VkImageView view) {
        imgInfos.push_back({ m_linearSampler, view, VK_IMAGE_LAYOUT_GENERAL });
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = dstSet;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w.pImageInfo = &imgInfos.back();
        writes.push_back(w);
    };

    // extViews order:
    // 0: normalDepth, 1: motionVectors, 2: specularMotion, 3: albedoRoughness, 4: specAlbedo,
    // 5: skipImage, 6: directDiffuse, 7: indirectDiffuse, 8: directSpecular, 9: indirectSpecular

    // Pass 0: reblur_preblur.comp
    {
        VkDescriptorSet set = m_preblurDescSets[variantIdx];
        addUbo(set, 0);
        addImg(set, 1,  extViews[0]); // uInNormalDepth
        addImg(set, 2,  extViews[3]); // uInAlbedoRoughness
        addImg(set, 3,  extViews[4]); // uInSpecAlbedo
        addImg(set, 4,  extViews[5]); // uInSkip
        addImg(set, 5,  extViews[6]); // uInDirectDiffuse
        addImg(set, 6,  extViews[7]); // uInIndirectDiffuse
        addImg(set, 7,  extViews[8]); // uInDirectSpecular
        addImg(set, 8,  extViews[9]); // uInIndirectSpecular
        addImg(set, 9,  extViews[2]); // uInSpecularMotion
        addImg(set, 10, m_preDiff->getImageView());
        addImg(set, 11, m_preSpec->getImageView());
        addImg(set, 12, m_currWorldND[currParity]->getImageView());
    }

    // Pass 1: reblur_temporal.comp
    {
        VkDescriptorSet set = m_temporalDescSets[variantIdx];
        addUbo(set, 0);
        addImg(set, 1,  m_preDiff->getImageView());
        addImg(set, 2,  m_preSpec->getImageView());
        addImg(set, 3,  m_currWorldND[currParity]->getImageView());
        addImg(set, 4,  m_currWorldND[prevParity]->getImageView());
        addImg(set, 5,  extViews[1]); // uMotionVectors
        addImg(set, 6,  extViews[2]); // uSpecularMotion
        addImg(set, 7,  extViews[3]); // uAlbedoRoughness
        addImg(set, 8,  m_histDiff[prevParity]->getImageView());
        addImg(set, 9,  m_histSpec[prevParity]->getImageView());
        addImg(set, 10, m_histMoments[prevParity]->getImageView());
        addImg(set, 11, m_histMeta[prevParity]->getImageView());
        addImg(set, 12, m_histHitT[prevParity]->getImageView());
        addImg(set, 13, m_histDiff[currParity]->getImageView());
        addImg(set, 14, m_histSpec[currParity]->getImageView());
        addImg(set, 15, m_histMoments[currParity]->getImageView());
        addImg(set, 16, m_histMeta[currParity]->getImageView());
        addImg(set, 17, m_histHitT[currParity]->getImageView());
    }

    // Pass 2: reblur_history_fix.comp (+ fused LDS step=1 A-Trous) -> writes m_pingPong[0]
    {
        VkDescriptorSet set = m_historyFixDescSets[variantIdx];
        addUbo(set, 0);
        addImg(set, 1, m_histDiff[currParity]->getImageView());
        addImg(set, 2, m_histSpec[currParity]->getImageView());
        addImg(set, 3, m_histMeta[currParity]->getImageView());
        addImg(set, 4, m_currWorldND[currParity]->getImageView());
        addImg(set, 5, extViews[3]); // uAlbedoRoughness
        addImg(set, 6, m_histHitT[currParity]->getImageView());
        addImg(set, 7, m_pingPongDiff[0]->getImageView());
        addImg(set, 8, m_pingPongSpec[0]->getImageView());
    }

    // Passes 3-4: reblur_atrous.comp (dir 0: pingPong[0]->[1], dir 1: pingPong[1]->[0])
    for (uint32_t dir = 0; dir < 2; ++dir) {
        const uint32_t srcIdx = dir;
        const uint32_t dstIdx = 1u - dir;
        VkDescriptorSet set = m_atrousDescSets[variantIdx][dir];
        addUbo(set, 0);
        addImg(set, 1, m_pingPongDiff[srcIdx]->getImageView());
        addImg(set, 2, m_pingPongSpec[srcIdx]->getImageView());
        addImg(set, 3, m_currWorldND[currParity]->getImageView());
        addImg(set, 4, extViews[3]); // uAlbedoRoughness
        addImg(set, 5, m_histHitT[currParity]->getImageView());
        addImg(set, 6, m_histMeta[currParity]->getImageView());
        addImg(set, 7, m_pingPongDiff[dstIdx]->getImageView());
        addImg(set, 8, m_pingPongSpec[dstIdx]->getImageView());
    }

    // Pass 5: reblur_stabilize.comp (reads m_pingPong[1] after 5 atrous dispatches)
    {
        VkDescriptorSet set = m_stabilizeDescSets[variantIdx];
        addUbo(set, 0);
        addImg(set, 1,  m_pingPongDiff[1]->getImageView());
        addImg(set, 2,  m_pingPongSpec[1]->getImageView());
        addImg(set, 3,  extViews[3]); // uInAlbedoRoughness
        addImg(set, 4,  extViews[4]); // uInSpecAlbedo
        addImg(set, 5,  extViews[5]); // uInSkip
        addImg(set, 6,  m_currWorldND[currParity]->getImageView());
        addImg(set, 7,  extViews[1]); // uInMotionVectors
        addImg(set, 8,  m_histMeta[currParity]->getImageView());
        addSampledImg(set, 9, m_stabHistory[prevParity]->getImageView());
        addImg(set, 10, m_stabHistory[currParity]->getImageView());
        addImg(set, 11, m_denoisedImage->getImageView());
    }

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    m_cachedVariantViews[variantIdx] = extViews;
    m_variantBound[variantIdx] = true;
}

void ReblurDenoiserPipeline::ensureGeneralLayout(VkCommandBuffer cmd, Image* img) {
    if (!img || img->getLayout() == VK_IMAGE_LAYOUT_GENERAL) {
        return;
    }
    const bool fromUndefined = (img->getLayout() == VK_IMAGE_LAYOUT_UNDEFINED);
    img->transitionLayout(
        cmd,
        VK_IMAGE_LAYOUT_GENERAL,
        fromUndefined ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        fromUndefined ? VK_ACCESS_2_NONE : (VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT),
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
    );
}

void ReblurDenoiserPipeline::clearAndTransitionInternalImages(VkCommandBuffer cmd) {
    const std::array<Image*, 21> allImages = {
        m_preDiff.get(), m_preSpec.get(),
        m_currWorldND[0].get(),  m_currWorldND[1].get(),
        m_histDiff[0].get(),     m_histDiff[1].get(),
        m_histSpec[0].get(),     m_histSpec[1].get(),
        m_histMoments[0].get(),  m_histMoments[1].get(),
        m_histMeta[0].get(),     m_histMeta[1].get(),
        m_histHitT[0].get(),     m_histHitT[1].get(),
        m_pingPongDiff[0].get(), m_pingPongDiff[1].get(),
        m_pingPongSpec[0].get(), m_pingPongSpec[1].get(),
        m_stabHistory[0].get(),  m_stabHistory[1].get(),
        m_denoisedImage.get()
    };

    VkClearColorValue clearZero = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    for (Image* img : allImages) {
        if (!img) continue;
        const bool fromUndefined = (img->getLayout() == VK_IMAGE_LAYOUT_UNDEFINED);
        img->transitionLayout(
            cmd,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            fromUndefined ? VK_PIPELINE_STAGE_2_NONE : VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            fromUndefined ? VK_ACCESS_2_NONE : (VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT),
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT
        );
        vkCmdClearColorImage(cmd, img->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearZero, 1, &clearRange);
        img->transitionLayout(
            cmd,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT,
            VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
        );
    }
    m_needsInitialClear = false;
}

void ReblurDenoiserPipeline::collectQueryResultsForSlot(uint32_t slot) {
    if (m_queryPool == VK_NULL_HANDLE || m_timestampPeriodNs <= 0.0f ||
        slot >= MAX_FRAMES_IN_FLIGHT || !m_querySlotUsed[slot]) {
        return;
    }

    std::array<uint64_t, QUERIES_PER_SLOT * 2> results{};
    const VkResult res = vkGetQueryPoolResults(
        m_device,
        m_queryPool,
        slot * QUERIES_PER_SLOT,
        QUERIES_PER_SLOT,
        sizeof(results),
        results.data(),
        sizeof(uint64_t) * 2,
        VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT
    );

    if (res == VK_SUCCESS) {
        const double ticksToMs = static_cast<double>(m_timestampPeriodNs) * 1e-6;
        double sumMs = 0.0;
        for (uint32_t p = 0; p < 5; ++p) {
            const uint64_t t0 = results[(p * 2) * 2];
            const uint64_t a0 = results[(p * 2) * 2 + 1];
            const uint64_t t1 = results[(p * 2 + 1) * 2];
            const uint64_t a1 = results[(p * 2 + 1) * 2 + 1];
            if (a0 != 0 && a1 != 0 && t1 >= t0) {
                m_lastPassTimingsMs[p] = static_cast<double>(t1 - t0) * ticksToMs;
                sumMs += m_lastPassTimingsMs[p];
            }
        }
        const uint64_t firstT = results[0];
        const uint64_t firstA = results[1];
        const uint64_t lastT  = results[(QUERIES_PER_SLOT - 1) * 2];
        const uint64_t lastA  = results[(QUERIES_PER_SLOT - 1) * 2 + 1];
        if (firstA != 0 && lastA != 0 && lastT >= firstT) {
            m_lastPassTimingsMs[5] = static_cast<double>(lastT - firstT) * ticksToMs;
        } else {
            m_lastPassTimingsMs[5] = sumMs;
        }
        m_querySlotUsed[slot] = false;
    }
}

void ReblurDenoiserPipeline::collectQueryResults() {
    for (uint32_t slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
        collectQueryResultsForSlot(slot);
    }
}

void ReblurDenoiserPipeline::recordDenoise(
    VkCommandBuffer cmd,
    uint32_t frameIndex,
    bool resetHistory,
    Image* normalDepth,
    Image* motionVectors,
    Image* specularMotion,
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
    const uint32_t currParity = m_historyParity;
    const uint32_t variantIdx = frameSlot * 2u + currParity;
    const bool effectiveReset = resetHistory || !m_historyValid;

    // Lazy init 1x1 fallback zero image
    if (!m_fallbackCleared && m_fallbackBlack1x1) {
        m_fallbackBlack1x1->transitionLayout(
            cmd,
            VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
        );
        VkClearColorValue clearZero = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        vkCmdClearColorImage(cmd, m_fallbackBlack1x1->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                             &clearZero, 1, &clearRange);
        m_fallbackBlack1x1->transitionLayout(
            cmd,
            VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
        m_fallbackCleared = true;
    }

    // Initial clear & layout transition on first frame after resize
    if (m_needsInitialClear) {
        clearAndTransitionInternalImages(cmd);
    } else {
        for (Image* img : {
            m_preDiff.get(), m_preSpec.get(),
            m_currWorldND[0].get(),  m_currWorldND[1].get(),
            m_histDiff[0].get(),     m_histDiff[1].get(),
            m_histSpec[0].get(),     m_histSpec[1].get(),
            m_histMoments[0].get(),  m_histMoments[1].get(),
            m_histMeta[0].get(),     m_histMeta[1].get(),
            m_histHitT[0].get(),     m_histHitT[1].get(),
            m_pingPongDiff[0].get(), m_pingPongDiff[1].get(),
            m_pingPongSpec[0].get(), m_pingPongSpec[1].get(),
            m_stabHistory[0].get(),  m_stabHistory[1].get(),
            m_denoisedImage.get()
        }) {
            ensureGeneralLayout(cmd, img);
        }
    }

    auto resolveView = [&](Image* img) -> VkImageView {
        if (img && img->getImageView() != VK_NULL_HANDLE) {
            return img->getImageView();
        }
        return m_fallbackBlack1x1->getImageView();
    };

    const std::array<VkImageView, 10> extViews = {
        resolveView(normalDepth),
        resolveView(motionVectors),
        resolveView(specularMotion),
        resolveView(albedoRoughness),
        resolveView(specAlbedo),
        resolveView(skipImage),
        resolveView(directDiffuse),
        resolveView(indirectDiffuse),
        resolveView(directSpecular),
        resolveView(indirectSpecular),
    };

    if (!m_variantBound[variantIdx] || m_cachedVariantViews[variantIdx] != extViews) {
        updateVariantDescriptors(frameSlot, currParity, extViews);
    }

    // Upload per-frame-in-flight Camera UBO (std140, 224 bytes)
    {
        glm::mat4 S(1.0f);
        S[1][1] = -1.0f;
        S[2][2] = -1.0f;

        CameraUboData ubo{};
        ubo.viewToWorld        = cameraMatrices.viewInverse;
        ubo.prevViewToWorld    = cameraMatrices.prevViewInverse;
        ubo.currViewToPrevView = S * (cameraMatrices.prevView * cameraMatrices.viewInverse) * S;
        ubo.currJitterUv       = cameraMatrices.currJitterUv;
        ubo.prevJitterUv       = cameraMatrices.prevJitterUv;
        ubo.tanHalfFovY        = std::max(cameraMatrices.tanHalfFovY, 1e-4f);
        ubo.aspectRatio        = std::max(cameraMatrices.aspectRatio, 1e-4f);
        ubo.prevTanHalfFovY    = (cameraMatrices.prevTanHalfFovY > 1e-4f) ? cameraMatrices.prevTanHalfFovY : ubo.tanHalfFovY;
        ubo.prevAspectRatio    = (cameraMatrices.prevAspectRatio > 1e-4f) ? cameraMatrices.prevAspectRatio : ubo.aspectRatio;

        m_cameraUbos[frameSlot]->copyFrom(&ubo, sizeof(ubo));
        m_cameraUbos[frameSlot]->flush();
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

    // Synchronize upstream wavefront 4-signal & G-buffer storage writes
    insertComputeBarrier(
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
    );

    // Harvest previous timestamps for this slot before resetting
    collectQueryResultsForSlot(frameSlot);
    const uint32_t qBase = frameSlot * QUERIES_PER_SLOT;
    if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, m_queryPool, qBase, QUERIES_PER_SLOT);
        m_querySlotUsed[frameSlot] = true;
    }
    auto writeTs = [&](uint32_t idx) {
        if (m_queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + idx);
        }
    };

    const uint32_t groupsX = (m_width + 15u) / 16u;
    const uint32_t groupsY = (m_height + 15u) / 16u;

    // =========================================================================
    // Pass 0: reblur_preblur.comp
    // =========================================================================
    {
        PreblurPushConstants pc{};
        pc.width = m_width;
        pc.height = m_height;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_preblurPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_preblurPipelineLayout,
                                0, 1, &m_preblurDescSets[variantIdx], 0, nullptr);
        vkCmdPushConstants(cmd, m_preblurPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        writeTs(0);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        writeTs(1);
    }

    insertComputeBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    // =========================================================================
    // Pass 1: reblur_temporal.comp
    // =========================================================================
    {
        TemporalPushConstants pc{};
        pc.width = m_width;
        pc.height = m_height;
        pc.resetHistory = effectiveReset ? 1u : 0u;
        pc.upscalerCoupled = cameraMatrices.upscalerCoupled ? 1u : 0u;
        pc.maxDiffHistory = 32.0f;
        pc.maxSpecHistory = 16.0f;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipelineLayout,
                                0, 1, &m_temporalDescSets[variantIdx], 0, nullptr);
        vkCmdPushConstants(cmd, m_temporalPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        writeTs(2);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        writeTs(3);
    }

    insertComputeBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    // =========================================================================
    // Pass 2: reblur_history_fix.comp -> writes m_pingPong[0]
    // =========================================================================
    {
        HistoryFixPushConstants pc{};
        pc.width = m_width;
        pc.height = m_height;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_historyFixPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_historyFixPipelineLayout,
                                0, 1, &m_historyFixDescSets[variantIdx], 0, nullptr);
        vkCmdPushConstants(cmd, m_historyFixPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        writeTs(4);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        writeTs(5);
    }

    insertComputeBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);

    // =========================================================================
    // Passes 3-4: reblur_atrous.comp (3x alternating 5-tap ping-pong: s = 2, 4, 8;
    //             step=1 is fused into Pass 2 LDS for 4K throughput)
    //   iter 0 (s=2, +): pingPong[0] -> pingPong[1]
    //   iter 1 (s=4, x): pingPong[1] -> pingPong[0]
    //   iter 2 (s=8, +): pingPong[0] -> pingPong[1]
    // =========================================================================
    {
        constexpr std::array<int32_t, 3> kStepSizes = { 2, 4, 8 };
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousPipeline);

        writeTs(6);
        for (uint32_t iter = 0; iter < 3; ++iter) {
            const uint32_t dir = iter & 1u;
            AtrousPushConstants pc{};
            pc.width = m_width;
            pc.height = m_height;
            pc.stepSize = kStepSizes[iter];
            pc.passIndex = iter;
            pc.phiNormal = 32.0f;
            pc.phiPlaneDepth = 2.0f;
            pc.phiLumDiff = 4.0f;
            pc.phiLumSpec = 3.0f;

            vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousPipelineLayout,
                                    0, 1, &m_atrousDescSets[variantIdx][dir], 0, nullptr);
            vkCmdPushConstants(cmd, m_atrousPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
            vkCmdDispatch(cmd, groupsX, groupsY, 1);

            if (iter < 2) {
                insertComputeBarrier(VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            }
        }
        writeTs(7);
    }

    insertComputeBarrier(
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
    );

    // =========================================================================
    // Pass 5: reblur_stabilize.comp -> writes m_stabHistory[curr] & m_denoisedImage
    // =========================================================================
    {
        StabilizePushConstants pc{};
        pc.width = m_width;
        pc.height = m_height;
        pc.resetHistory = effectiveReset ? 1u : 0u;
        pc.upscalerCoupled = cameraMatrices.upscalerCoupled ? 1u : 0u;

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stabilizePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_stabilizePipelineLayout,
                                0, 1, &m_stabilizeDescSets[variantIdx], 0, nullptr);
        vkCmdPushConstants(cmd, m_stabilizePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);

        writeTs(8);
        vkCmdDispatch(cmd, groupsX, groupsY, 1);
        writeTs(9);
    }

    // Make m_denoisedImage visible to FSR upscaler, post-process tonemap, and staging readback
    {
        VkMemoryBarrier2 postBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        postBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT
                                 | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT
                                 | VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        postBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT
                                  | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
                                  | VK_ACCESS_2_TRANSFER_READ_BIT;

        VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &postBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    m_historyValid = true;
    m_historyParity = 1u - currParity;
}

} // namespace pathways
