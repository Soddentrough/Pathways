#include "rt/DenoiserPipeline.hpp"
#include "rt/PostProcessPipeline.hpp"
#include "rt/FfxSdkRuntime.hpp"
#include "core/Logger.hpp"

#include <stdexcept>
#include <vector>
#include <cstring>

namespace pathways {

DenoiserPipeline::DenoiserPipeline(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VmaAllocator allocator,
    VkQueue queue,
    ShaderLoaderFunc shaderLoader,
    bool forceWave32
) : m_device(device),
    m_physDevice(physicalDevice),
    m_allocator(allocator),
    m_queue(queue),
    m_shaderLoader(std::move(shaderLoader)),
    m_forceWave32(forceWave32)
{
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_physDevice, &props);
    m_timestampPeriodNs = props.limits.timestampPeriod;

    createDescriptorSetLayouts();
    initPipelines();

    // Internal 20-query timestamp pool (10 x MAX_FRAMES_IN_FLIGHT):
    // frame slot s uses queries [s*10, s*10+9], two per dispatch.
    VkQueryPoolCreateInfo queryPoolInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = NUM_PASSES * 2 * MAX_FRAMES_IN_FLIGHT;
    if (vkCreateQueryPool(m_device, &queryPoolInfo, nullptr, &m_queryPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create denoiser timestamp query pool");
    }

    Logger::info("DenoiserPipeline (Adaptive E-SVGF, 5-dispatch dilated) initialized successfully.");
}

DenoiserPipeline::~DenoiserPipeline() {
    destroyFfxDenoiserContext();
    if (m_device != VK_NULL_HANDLE) {
        auto destroyPipe = [this](VkPipelineLayout layout, VkPipeline pipe, VkDescriptorSetLayout desc) {
            if (pipe) vkDestroyPipeline(m_device, pipe, nullptr);
            if (layout) vkDestroyPipelineLayout(m_device, layout, nullptr);
            if (desc) vkDestroyDescriptorSetLayout(m_device, desc, nullptr);
        };
        destroyPipe(m_fusedTemporalPipelineLayout, m_fusedTemporalPipeline, m_fusedTemporalDescSetLayout);
        destroyPipe(m_atrousStep1PipelineLayout, m_atrousStep1Pipeline, m_atrousStep1DescSetLayout);
        destroyPipe(m_atrousStep2PipelineLayout, m_atrousStep2Pipeline, m_atrousStep2DescSetLayout);
        destroyPipe(m_atrousStep3PipelineLayout, m_atrousStep3Pipeline, m_atrousStep3DescSetLayout);
        destroyPipe(m_coarseRemodulatePipelineLayout, m_coarseRemodulatePipeline, m_atrousCoarseRemodulateDescSetLayout);

        if (m_queryPool) vkDestroyQueryPool(m_device, m_queryPool, nullptr);
        if (m_descPool) vkDestroyDescriptorPool(m_device, m_descPool, nullptr);
    }
}

void DenoiserPipeline::createDescriptorSetLayouts() {
    auto makeImageBinding = [](uint32_t b, VkDescriptorType type) -> VkDescriptorSetLayoutBinding {
        VkDescriptorSetLayoutBinding binding{};
        binding.binding = b;
        binding.descriptorType = type;
        binding.descriptorCount = 1;
        binding.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        return binding;
    };

    auto createLayout = [&](const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                            VkDescriptorSetLayout& out, const char* name) {
        VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        info.bindingCount = static_cast<uint32_t>(bindings.size());
        info.pBindings = bindings.data();
        if (vkCreateDescriptorSetLayout(m_device, &info, nullptr, &out) != VK_SUCCESS) {
            throw std::runtime_error(std::string("Failed to create denoiser descriptor set layout: ") + name);
        }
    };

    // Pass 1: Fused temporal (Binding 0 CameraMatricesUBO + Bindings 1-22 storage images)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        bindings.push_back(makeImageBinding(0, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER));
        for (uint32_t b = 1; b <= 22; ++b) bindings.push_back(makeImageBinding(b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        createLayout(bindings, m_fusedTemporalDescSetLayout, "fused_temporal");
    }
    // Pass 2: A-Trous s=1 (6 image bindings; uEdgeDensity intentionally omitted)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t b = 0; b < 6; ++b) bindings.push_back(makeImageBinding(b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        createLayout(bindings, m_atrousStep1DescSetLayout, "atrous_step1");
    }
    // Pass 3: A-Trous s=2 (7 image bindings incl. uEdgeDensity)
    // Pass 4: A-Trous s=4 (7 image bindings incl. uEdgeDensity)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t b = 0; b < 7; ++b) bindings.push_back(makeImageBinding(b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        createLayout(bindings, m_atrousStep2DescSetLayout, "atrous_step2");
        createLayout(bindings, m_atrousStep3DescSetLayout, "atrous_step3");
    }
    // Pass 5: Coarse s=8 + fused remodulate (6 image bindings in production)
    {
        std::vector<VkDescriptorSetLayoutBinding> bindings;
        for (uint32_t b = 0; b < 6; ++b) bindings.push_back(makeImageBinding(b, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE));
        createLayout(bindings, m_atrousCoarseRemodulateDescSetLayout, "atrous_coarse_remodulate");
    }

    // Descriptor pool: ~88 storage image descriptors + UBOs + tonemap set
    std::array<VkDescriptorPoolSize, 2> poolSizes = {
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 128 },
        VkDescriptorPoolSize{ VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8 },
    };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    poolInfo.maxSets = 64;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();
    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descPool) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create denoiser descriptor pool");
    }

    allocateDescriptorSets();
}

void DenoiserPipeline::allocateDescriptorSets() {
    VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocInfo.descriptorPool = m_descPool;

    struct Slot { VkDescriptorSetLayout layout; std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT>* sets; };
    std::array<Slot, 5> slots = {{
        { m_fusedTemporalDescSetLayout, &m_fusedTemporalDescSets },
        { m_atrousStep1DescSetLayout,   &m_atrousStep1DescSets },
        { m_atrousStep2DescSetLayout,   &m_atrousStep2DescSets },
        { m_atrousStep3DescSetLayout,   &m_atrousStep3DescSets },
        { m_atrousCoarseRemodulateDescSetLayout, &m_coarseRemodulateDescSets },
    }};

    for (auto& slot : slots) {
        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            allocInfo.descriptorSetCount = 1;
            allocInfo.pSetLayouts = &slot.layout;
            if (vkAllocateDescriptorSets(m_device, &allocInfo, &(*slot.sets)[i]) != VK_SUCCESS) {
                throw std::runtime_error("Failed to allocate denoiser descriptor sets");
            }
        }
    }
}

void DenoiserPipeline::initPipelines() {
    auto createComputePipeline = [this](
        const std::string& shaderName,
        VkDescriptorSetLayout descLayout,
        uint32_t pushConstantSize,
        VkPipelineLayout& outLayout,
        VkPipeline& outPipeline
    ) {
        auto code = m_shaderLoader(shaderName);
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = pushConstantSize;

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descLayout;
        layoutInfo.pushConstantRangeCount = (pushConstantSize > 0) ? 1 : 0;
        layoutInfo.pPushConstantRanges = (pushConstantSize > 0) ? &pcRange : nullptr;
        if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &outLayout) != VK_SUCCESS) {
            throw std::runtime_error("Failed to create denoiser pipeline layout for: " + shaderName);
        }

        // Wave32 enforcement + Vulkan 1.4 maintenance5 inline VkShaderModuleCreateInfo
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSizeInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
        };
        subgroupSizeInfo.requiredSubgroupSize = 32;

        VkShaderModuleCreateInfo smInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
        smInfo.pNext = m_forceWave32 ? &subgroupSizeInfo : nullptr;
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
            throw std::runtime_error("Failed to create denoiser compute pipeline for: " + shaderName);
        }
    };

    createComputePipeline("denoise_fused_temporal.comp.spv", m_fusedTemporalDescSetLayout,
                          sizeof(FusedTemporalPushConstants), m_fusedTemporalPipelineLayout, m_fusedTemporalPipeline);
    createComputePipeline("denoise_atrous_step1.comp.spv", m_atrousStep1DescSetLayout,
                          sizeof(AtrousPushConstants), m_atrousStep1PipelineLayout, m_atrousStep1Pipeline);
    createComputePipeline("denoise_atrous_step2.comp.spv", m_atrousStep2DescSetLayout,
                          sizeof(AtrousPushConstants), m_atrousStep2PipelineLayout, m_atrousStep2Pipeline);
    createComputePipeline("denoise_atrous_step3.comp.spv", m_atrousStep3DescSetLayout,
                          sizeof(AtrousPushConstants), m_atrousStep3PipelineLayout, m_atrousStep3Pipeline);
    createComputePipeline("denoise_atrous_coarse_remodulate.comp.spv", m_atrousCoarseRemodulateDescSetLayout,
                          sizeof(AtrousCoarseRemodulatePushConstants), m_coarseRemodulatePipelineLayout, m_coarseRemodulatePipeline);
}

void DenoiserPipeline::resize(uint32_t width, uint32_t height, AccumFormat accumFormat, VkCommandBuffer cmd) {
    if (m_width == width && m_height == height && m_accumFormat == accumFormat && m_denoisedImage) {
        return;
    }

    m_width = width;
    m_height = height;
    m_accumFormat = accumFormat;
    m_descriptorsBound = false;

    VkImageUsageFlags usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                              VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_edgeDensity[i]        = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R8G8_UNORM, usage);
        m_historyDiffuse[i]     = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
        m_historySpecular[i]    = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
        m_historyMoments[i]     = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
        m_historyLength[i]      = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
        m_historyNormalDepth[i] = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
    }

    for (uint32_t i = 0; i < 2; ++i) {
        m_pingPongDiffuse[i]  = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
        m_pingPongSpecular[i] = std::make_unique<Image>(m_device, m_allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage);
    }

    VkFormat outFmt = (accumFormat == AccumFormat::RGBA32_SFLOAT) ? VK_FORMAT_R32G32B32A32_SFLOAT : VK_FORMAT_R16G16B16A16_SFLOAT;
    m_denoisedImage = std::make_unique<Image>(m_device, m_allocator, width, height, outFmt, usage);

    // 1x1 zeroed fallback for the specular VMV binding (format-compatible RGBA16F)
    m_specMotionFallback = std::make_unique<Image>(m_device, m_allocator, 1, 1, VK_FORMAT_R16G16B16A16_SFLOAT, usage);

    // Camera matrices UBOs (per frame in flight, std140-compatible 160 bytes)
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (!m_cameraMatricesUBO[i]) {
            m_cameraMatricesUBO[i] = std::make_unique<Buffer>(
                m_allocator, sizeof(CameraMatrices),
                VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
                VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT | VMA_ALLOCATION_CREATE_MAPPED_BIT
            );
        }
    }

    if (cmd != VK_NULL_HANDLE) {
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkResetCommandBuffer(cmd, 0);
        vkBeginCommandBuffer(cmd, &beginInfo);

        // Initial transition to VK_IMAGE_LAYOUT_GENERAL and clear history to zero
        VkClearColorValue clearZero = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        auto transitionAndClear = [&](Image* img) {
            img->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0,
                                  VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            vkCmdClearColorImage(cmd, img->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
        };

        for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
            transitionAndClear(m_edgeDensity[i].get());
            transitionAndClear(m_historyDiffuse[i].get());
            transitionAndClear(m_historySpecular[i].get());
            transitionAndClear(m_historyMoments[i].get());
            transitionAndClear(m_historyLength[i].get());
            transitionAndClear(m_historyNormalDepth[i].get());
        }
        for (uint32_t i = 0; i < 2; ++i) {
            transitionAndClear(m_pingPongDiffuse[i].get());
            transitionAndClear(m_pingPongSpecular[i].get());
        }
        m_denoisedImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_NONE, 0,
                                          VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        transitionAndClear(m_specMotionFallback.get());

        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = cmd;

        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_queue);
    }

    Logger::info("DenoiserPipeline (Adaptive E-SVGF) allocated 17 internal storage images ({}x{}, format {})",
                 width, height, static_cast<int>(accumFormat));
    initFfxDenoiserContext();
}

void DenoiserPipeline::initFfxDenoiserContext() {
    destroyFfxDenoiserContext();
    m_vramUsageBytes = static_cast<uint64_t>(m_width) * static_cast<uint64_t>(m_height) * 124ull;

#if defined(PATHWAYS_HAS_FFX_SDK)
    if (m_width == 0 || m_height == 0) {
        return;
    }
    auto backendDesc = FfxSdkRuntime::makeBackendVkDesc(m_device, m_physDevice);
    constexpr uint32_t kAll4Signals = FFX_DENOISER_SIGNAL_DIRECT_DIFFUSE |
                                      FFX_DENOISER_SIGNAL_DIRECT_SPECULAR |
                                      FFX_DENOISER_SIGNAL_INDIRECT_DIFFUSE |
                                      FFX_DENOISER_SIGNAL_INDIRECT_SPECULAR;
    ffx::CreateContextDescDenoiser createDenoiser{};
    createDenoiser.version = FFX_DENOISER_VERSION;
    createDenoiser.maxRenderSize = { m_width, m_height };
    createDenoiser.signalFlags = kAll4Signals;
    createDenoiser.checkerboardSignalFlags = FFX_DENOISER_SIGNAL_NONE;
    createDenoiser.flags = 0;

    ffxContext ctx = nullptr;
    if (ffx::CreateContext(ctx, nullptr, createDenoiser, backendDesc) == ffx::ReturnCode::Ok && ctx != nullptr) {
        m_ffxDenoiserContext = ctx;
        FfxApiEffectMemoryUsage gpuMem{};
        ffx::QueryDescDenoiserGetGPUMemoryUsage memQuery{};
        memQuery.device = m_physDevice;
        memQuery.maxRenderSize = { m_width, m_height };
        memQuery.signalFlags = kAll4Signals;
        memQuery.checkerboardSignalFlags = FFX_DENOISER_SIGNAL_NONE;
        memQuery.flags = 0;
        memQuery.gpuMemoryUsage = &gpuMem;
        if (ffx::Query(m_ffxDenoiserContext, memQuery) == ffx::ReturnCode::Ok && gpuMem.totalUsageInBytes > 0) {
            m_vramUsageBytes = gpuMem.totalUsageInBytes;
        }
    }
#endif
}

void DenoiserPipeline::destroyFfxDenoiserContext() {
#if defined(PATHWAYS_HAS_FFX_SDK)
    if (m_ffxDenoiserContext != nullptr) {
        ffx::DestroyContext(m_ffxDenoiserContext);
        m_ffxDenoiserContext = nullptr;
    }
#else
    m_ffxDenoiserContext = nullptr;
#endif
}

void DenoiserPipeline::initTonemapDescriptors(PostProcessPipeline* postProcess, Image* outputImage) {
    if (!postProcess || !outputImage || !m_denoisedImage) return;

    if (m_tonemapDescSet == VK_NULL_HANDLE) {
        VkDescriptorSetLayout tmLayout = postProcess->getTonemapDescLayout();
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_descPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &tmLayout;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &m_tonemapDescSet) != VK_SUCCESS) {
            throw std::runtime_error("Failed to allocate denoiser tonemap descriptor set");
        }
    }

    postProcess->updateTonemapDescriptors(m_tonemapDescSet, m_denoisedImage.get(), outputImage);
}

void DenoiserPipeline::updateDescriptors(
    Image* currentDiffuse,
    Image* currentSpecular,
    Image* motionVectors,
    Image* normalDepth,
    Image* albedoRoughness,
    Image* accumImage,
    Image* specularMotion,
    Image* ffxDirectDiffuse,
    Image* ffxDirectSpecular,
    Image* ffxIndirectDiffuse,
    Image* ffxIndirectSpecular
) {
    // Specular VMV and optional FFX 4-signal images fall back to the 1x1 zeroed RGBA16F image when unavailable
    Image* specMotion = specularMotion ? specularMotion : m_specMotionFallback.get();
    Image* ffxDirDiff = ffxDirectDiffuse ? ffxDirectDiffuse : m_specMotionFallback.get();
    Image* ffxDirSpec = ffxDirectSpecular ? ffxDirectSpecular : m_specMotionFallback.get();
    Image* ffxIndDiff = ffxIndirectDiffuse ? ffxIndirectDiffuse : m_specMotionFallback.get();
    Image* ffxIndSpec = ffxIndirectSpecular ? ffxIndirectSpecular : m_specMotionFallback.get();

    std::vector<VkWriteDescriptorSet> writes;
    std::vector<VkDescriptorImageInfo> imgInfos;
    std::vector<VkDescriptorBufferInfo> bufInfos;
    imgInfos.reserve(112);
    bufInfos.reserve(MAX_FRAMES_IN_FLIGHT);

    auto addImageWrite = [&](VkDescriptorSet dstSet, uint32_t dstBinding, VkImageView view) {
        imgInfos.push_back({ VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL });
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = dstSet;
        w.dstBinding = dstBinding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &imgInfos.back();
        writes.push_back(w);
    };
    auto addBufferWrite = [&](VkDescriptorSet dstSet, uint32_t dstBinding, VkBuffer buffer, VkDeviceSize size) {
        bufInfos.push_back({ buffer, 0, size });
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = dstSet;
        w.dstBinding = dstBinding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w.pBufferInfo = &bufInfos.back();
        writes.push_back(w);
    };

    for (uint32_t slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
        uint32_t prevSlot = 1 - slot;

        // ---- Pass 1: Fused temporal (23 bindings) ----
        {
            VkDescriptorSet set = m_fusedTemporalDescSets[slot];
            addBufferWrite(set, 0, m_cameraMatricesUBO[slot]->getBuffer(), sizeof(CameraMatrices));
            addImageWrite(set, 1, currentDiffuse->getImageView());
            addImageWrite(set, 2, currentSpecular->getImageView());
            addImageWrite(set, 3, motionVectors->getImageView());
            addImageWrite(set, 4, specMotion->getImageView());
            addImageWrite(set, 5, normalDepth->getImageView());
            addImageWrite(set, 6, m_historyNormalDepth[prevSlot]->getImageView());
            addImageWrite(set, 7, albedoRoughness->getImageView());
            addImageWrite(set, 8, m_historyDiffuse[prevSlot]->getImageView());
            addImageWrite(set, 9, m_historySpecular[prevSlot]->getImageView());
            addImageWrite(set, 10, m_historyMoments[prevSlot]->getImageView());
            addImageWrite(set, 11, m_historyLength[prevSlot]->getImageView());
            addImageWrite(set, 12, m_edgeDensity[prevSlot]->getImageView());
            addImageWrite(set, 13, m_historyDiffuse[slot]->getImageView());
            addImageWrite(set, 14, m_historySpecular[slot]->getImageView());
            addImageWrite(set, 15, m_historyMoments[slot]->getImageView());
            addImageWrite(set, 16, m_historyLength[slot]->getImageView());
            addImageWrite(set, 17, m_edgeDensity[slot]->getImageView());
            addImageWrite(set, 18, m_historyNormalDepth[slot]->getImageView());
            addImageWrite(set, 19, ffxDirDiff->getImageView());
            addImageWrite(set, 20, ffxDirSpec->getImageView());
            addImageWrite(set, 21, ffxIndDiff->getImageView());
            addImageWrite(set, 22, ffxIndSpec->getImageView());
        }

        // ---- Pass 2: A-Trous s=1 (reads pre-spatial history of this slot) ----
        {
            VkDescriptorSet set = m_atrousStep1DescSets[slot];
            addImageWrite(set, 0, m_historyDiffuse[slot]->getImageView());
            addImageWrite(set, 1, m_historySpecular[slot]->getImageView());
            addImageWrite(set, 2, normalDepth->getImageView());
            addImageWrite(set, 3, albedoRoughness->getImageView());
            addImageWrite(set, 4, m_pingPongDiffuse[0]->getImageView());
            addImageWrite(set, 5, m_pingPongSpecular[0]->getImageView());
        }

        // ---- Pass 3: A-Trous s=2 (pingPong 0 -> 1) ----
        {
            VkDescriptorSet set = m_atrousStep2DescSets[slot];
            addImageWrite(set, 0, m_pingPongDiffuse[0]->getImageView());
            addImageWrite(set, 1, m_pingPongSpecular[0]->getImageView());
            addImageWrite(set, 2, normalDepth->getImageView());
            addImageWrite(set, 3, albedoRoughness->getImageView());
            addImageWrite(set, 4, m_edgeDensity[slot]->getImageView());
            addImageWrite(set, 5, m_pingPongDiffuse[1]->getImageView());
            addImageWrite(set, 6, m_pingPongSpecular[1]->getImageView());
        }

        // ---- Pass 4: A-Trous s=4 (pingPong 1 -> 0) ----
        {
            VkDescriptorSet set = m_atrousStep3DescSets[slot];
            addImageWrite(set, 0, m_pingPongDiffuse[1]->getImageView());
            addImageWrite(set, 1, m_pingPongSpecular[1]->getImageView());
            addImageWrite(set, 2, normalDepth->getImageView());
            addImageWrite(set, 3, albedoRoughness->getImageView());
            addImageWrite(set, 4, m_edgeDensity[slot]->getImageView());
            addImageWrite(set, 5, m_pingPongDiffuse[0]->getImageView());
            addImageWrite(set, 6, m_pingPongSpecular[0]->getImageView());
        }

        // ---- Pass 5: Coarse s=8 + fused remodulate (pingPong 0 -> final) ----
        {
            VkDescriptorSet set = m_coarseRemodulateDescSets[slot];
            addImageWrite(set, 0, m_pingPongDiffuse[0]->getImageView());
            addImageWrite(set, 1, m_pingPongSpecular[0]->getImageView());
            addImageWrite(set, 2, normalDepth->getImageView());
            addImageWrite(set, 3, albedoRoughness->getImageView());
            addImageWrite(set, 4, m_edgeDensity[slot]->getImageView());
            addImageWrite(set, 5, m_denoisedImage->getImageView());
        }
        (void)accumImage; // Stationary preview path is compile-time only (ENABLE_STATIONARY_PREVIEW)
    }

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
    m_descriptorsBound = true;
}

void DenoiserPipeline::collectQueryResults() {
    if (m_queryPool == VK_NULL_HANDLE) return;

    // Non-blocking collection of previously completed pass timestamps.
    // Each frame slot owns 10 queries; results are from 2 frames ago and
    // typically already retired. DONT_WAIT avoids any pipeline stall.
    // With VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT each
    // query yields two uint64_t entries: {timestamp, availability}.
    uint64_t results[10 * 2]{};
    for (uint32_t slot = 0; slot < MAX_FRAMES_IN_FLIGHT; ++slot) {
        if (!m_slotQueriesRecorded[slot]) continue; // Queries must be recorded by the GPU before reading
        VkResult r = vkGetQueryPoolResults(
            m_device, m_queryPool, slot * 10, 10,
            sizeof(results), results, sizeof(uint64_t) * 2,
            VK_QUERY_RESULT_WITH_AVAILABILITY_BIT | VK_QUERY_RESULT_64_BIT);
        if (r == VK_NOT_READY) continue;

        for (uint32_t pass = 0; pass < NUM_PASSES; ++pass) {
            uint64_t beginTicks = results[pass * 4 + 0];
            uint64_t beginAvail = results[pass * 4 + 1];
            uint64_t endTicks   = results[pass * 4 + 2];
            uint64_t endAvail   = results[pass * 4 + 3];
            if (beginAvail && endAvail && endTicks > beginTicks) {
                m_passTimingsMs[pass] = static_cast<float>(
                    static_cast<double>(endTicks - beginTicks) * m_timestampPeriodNs / 1.0e6);
            }
        }
    }
}

void DenoiserPipeline::recordDenoise(
    VkCommandBuffer cmd,
    uint32_t frameSlot,
    bool resetHistory,
    uint32_t accumulatedSamples,
    float crossfadeBlend,
    Image* currentDiffuse,
    Image* currentSpecular,
    Image* motionVectors,
    Image* normalDepth,
    Image* albedoRoughness,
    Image* accumImage,
    Image* specularMotion,
    const CameraMatrices& cameraMatrices,
    DenoiserMode denoiserMode,
    Image* ffxDirectDiffuse,
    Image* ffxDirectSpecular,
    Image* ffxIndirectDiffuse,
    Image* ffxIndirectSpecular
) {
    if (!isInitialized() || !currentDiffuse || !currentSpecular || !motionVectors ||
        !normalDepth || !albedoRoughness) {
        return;
    }
    (void)accumImage;
    (void)crossfadeBlend;

    if (!m_descriptorsBound) {
        updateDescriptors(currentDiffuse, currentSpecular, motionVectors, normalDepth,
                          albedoRoughness, accumImage, specularMotion,
                          ffxDirectDiffuse, ffxDirectSpecular, ffxIndirectDiffuse, ffxIndirectSpecular);
    }

    const uint32_t isRayRegen = (denoiserMode == DenoiserMode::FfxRayRegeneration && m_ffxDenoiserContext != nullptr) ? 1u : 0u;

    // Upload per-frame camera matrices for Pass 1 (std140 binary compatible)
    m_cameraMatricesUBO[frameSlot]->copyFrom(&cameraMatrices, sizeof(CameraMatrices));

    // Asynchronously harvest previous GPU pass timings, then reset this slot's queries
    collectQueryResults();
    uint32_t qBase = frameSlot * 10;
    vkCmdResetQueryPool(cmd, m_queryPool, qBase, 10);

    auto writeTimestamp = [&](uint32_t queryIndex) {
        if (m_queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, queryIndex);
        }
    };

    auto insertComputeBarrier = [&cmd]() {
        VkMemoryBarrier2 barrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        barrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        barrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &barrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    };

    const float phiNormal = 32.0f;
    const float phiPlaneDepth = 2.0f;
    const float phiLuminance = 4.0f;

    // =========================================================================
    // Pass 1: Fused Edge-Aware Spatiotemporal Accumulator (16x16 workgroups)
    // =========================================================================
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fusedTemporalPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_fusedTemporalPipelineLayout,
                                0, 1, &m_fusedTemporalDescSets[frameSlot], 0, nullptr);

        FusedTemporalPushConstants pc{};
        pc.width = m_width;
        pc.height = m_height;
        pc.resetHistory = resetHistory ? 1u : 0u;
        pc.maxHistoryLength = 32.0f;
        pc.weightAlbedo = 0.50f;
        pc.weightDepth = 0.30f;
        pc.weightNormal = 0.20f;
        pc.sigmaMultiplier = 1.20f;
        pc.noiseFloor = 0.05f;
        pc.contrastFloor = 0.02f;
        pc.sigmaMin = 0.02f;
        pc.temporalAlpha = 0.15f;
        pc.normalThreshold = 0.85f;
        pc.normalThresholdDynamic = 0.50f;
        pc.depthThreshold = 0.08f;
        pc.dynamicDepthThreshold = 0.15f;
        pc.mvDiscrepancyThreshold = 2.0f;
        pc.isRayRegeneration = isRayRegen;
        vkCmdPushConstants(cmd, m_fusedTemporalPipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(FusedTemporalPushConstants), &pc);

        writeTimestamp(qBase + 0);
        vkCmdDispatch(cmd, (m_width + 15) / 16, (m_height + 15) / 16, 1);
        writeTimestamp(qBase + 1);
        insertComputeBarrier();
    }

    // Passes 2-5 run on 8x8 workgroup grids
    const uint32_t gx = (m_width + 7) / 8;
    const uint32_t gy = (m_height + 7) / 8;

    // =========================================================================
    // Pass 2: Fine-Scale Wavelet Iteration 1 (s = 1, diameter 5)
    // =========================================================================
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousStep1Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousStep1PipelineLayout,
                                0, 1, &m_atrousStep1DescSets[frameSlot], 0, nullptr);

        AtrousPushConstants pc{};
        pc.width = m_width;
        pc.height = m_height;
        pc.stepSize = 1;
        pc.phiNormal = phiNormal;
        pc.phiPlaneDepth = phiPlaneDepth;
        pc.phiLuminance = phiLuminance;
        pc.tanHalfFovY = cameraMatrices.tanHalfFovY;
        pc.aspectRatio = cameraMatrices.aspectRatio;
        vkCmdPushConstants(cmd, m_atrousStep1PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(AtrousPushConstants), &pc);

        writeTimestamp(qBase + 2);
        vkCmdDispatch(cmd, gx, gy, 1);
        writeTimestamp(qBase + 3);
        insertComputeBarrier();
    }

    // =========================================================================
    // Pass 3: Fine-Scale Wavelet Iteration 2 (s = 2, diameter 13, covers k=9)
    // =========================================================================
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousStep2Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousStep2PipelineLayout,
                                0, 1, &m_atrousStep2DescSets[frameSlot], 0, nullptr);

        AtrousPushConstants pc{};
        pc.width = m_width;
        pc.height = m_height;
        pc.stepSize = 2;
        pc.phiNormal = phiNormal;
        pc.phiPlaneDepth = phiPlaneDepth;
        pc.phiLuminance = phiLuminance;
        pc.tanHalfFovY = cameraMatrices.tanHalfFovY;
        pc.aspectRatio = cameraMatrices.aspectRatio;
        vkCmdPushConstants(cmd, m_atrousStep2PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(AtrousPushConstants), &pc);

        writeTimestamp(qBase + 4);
        vkCmdDispatch(cmd, gx, gy, 1);
        writeTimestamp(qBase + 5);
        insertComputeBarrier();
    }

    // =========================================================================
    // Pass 4: Mid-Scale Wavelet Iteration 3 (s = 4, diameter 29, covers k=17)
    // =========================================================================
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousStep3Pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_atrousStep3PipelineLayout,
                                0, 1, &m_atrousStep3DescSets[frameSlot], 0, nullptr);

        AtrousPushConstants pc{};
        pc.width = m_width;
        pc.height = m_height;
        pc.stepSize = 4;
        pc.phiNormal = phiNormal;
        pc.phiPlaneDepth = phiPlaneDepth;
        pc.phiLuminance = phiLuminance;
        pc.tanHalfFovY = cameraMatrices.tanHalfFovY;
        pc.aspectRatio = cameraMatrices.aspectRatio;
        vkCmdPushConstants(cmd, m_atrousStep3PipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(AtrousPushConstants), &pc);

        writeTimestamp(qBase + 6);
        vkCmdDispatch(cmd, gx, gy, 1);
        writeTimestamp(qBase + 7);
        insertComputeBarrier();
    }

    // =========================================================================
    // Pass 5: Coarse Wavelet (s = 8, diameter 61, covers k=33) & Remodulation
    // =========================================================================
    {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_coarseRemodulatePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_coarseRemodulatePipelineLayout,
                                0, 1, &m_coarseRemodulateDescSets[frameSlot], 0, nullptr);

        AtrousCoarseRemodulatePushConstants pc{};
        pc.width = m_width;
        pc.height = m_height;
        pc.stepSize = 8;
        pc.phiNormal = phiNormal;
        pc.phiPlaneDepth = phiPlaneDepth;
        pc.phiLuminance = phiLuminance;
        pc.tanHalfFovY = cameraMatrices.tanHalfFovY;
        pc.aspectRatio = cameraMatrices.aspectRatio;
        pc.edgeThresholdLow = 0.35f;
        pc.accumulatedSamples = accumulatedSamples;
        pc.isRayRegeneration = isRayRegen;
        vkCmdPushConstants(cmd, m_coarseRemodulatePipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT,
                           0, sizeof(AtrousCoarseRemodulatePushConstants), &pc);

        writeTimestamp(qBase + 8);
        vkCmdDispatch(cmd, gx, gy, 1);
        writeTimestamp(qBase + 9);
        m_slotQueriesRecorded[frameSlot] = true;
    }

    // Output barrier: guarantee tonemapping reads the finished denoised image
    VkMemoryBarrier2 outputBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    outputBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    outputBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    outputBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    outputBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    VkDependencyInfo outputDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    outputDep.memoryBarrierCount = 1;
    outputDep.pMemoryBarriers = &outputBarrier;
    vkCmdPipelineBarrier2(cmd, &outputDep);
}

} // namespace pathways
