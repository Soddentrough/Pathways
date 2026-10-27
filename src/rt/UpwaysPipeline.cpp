#include "UpwaysPipeline.hpp"
#include "core/Logger.hpp"
#include "upways_weights.hpp"
#include "upways_default_weights.hpp"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

#include <fstream>
#include <filesystem>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <array>

namespace pathways {

UpwaysPipeline::UpwaysPipeline(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VmaAllocator allocator,
    uint32_t inputWidth,
    uint32_t inputHeight,
    uint32_t outputWidth,
    uint32_t outputHeight,
    const std::vector<char>& temporalSpv,
    const std::vector<char>& neuralGuideSpv,
    const std::vector<char>& upscaleResolveSpv,
    const std::string& weightsPath,
    bool enableSuperRes,
    VkFormat imageFormat
) : m_device(device),
    m_physDevice(physicalDevice),
    m_allocator(allocator),
    m_inputWidth(inputWidth),
    m_inputHeight(inputHeight),
    m_outputWidth(outputWidth),
    m_outputHeight(outputHeight),
    m_quadWidth((inputWidth + 1) / 2),
    m_quadHeight((inputHeight + 1) / 2),
    m_superRes(enableSuperRes),
    m_format(imageFormat)
{
    Logger::info("UpwaysPipeline (3-pass Combined Neural Spatiotemporal Reconstructor & Multi-Scale KPN Upscaler) initializing: Input {}x{}, Quad {}x{}, Output {}x{} (SuperRes: {})",
                 m_inputWidth, m_inputHeight, m_quadWidth, m_quadHeight, m_outputWidth, m_outputHeight,
                 m_superRes ? "Enabled" : "1.0x (Native)");

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_physDevice, &props);
    m_timestampPeriodNs = static_cast<double>(props.limits.timestampPeriod);

    VkQueryPoolCreateInfo queryPoolInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    queryPoolInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    queryPoolInfo.queryCount = MAX_QUERY_SLOTS * QUERIES_PER_SLOT;
    if (vkCreateQueryPool(m_device, &queryPoolInfo, nullptr, &m_queryPool) != VK_SUCCESS) {
        Logger::warn("UpwaysPipeline: Failed to create timestamp query pool; per-pass profiling disabled.");
        m_queryPool = VK_NULL_HANDLE;
    }

    initBuffers(weightsPath);
    initImages();
    createDescriptorSetLayouts();
    allocateDescriptorSets();
    createPipelines(temporalSpv, neuralGuideSpv, upscaleResolveSpv);
    refreshDescriptors();

    m_initialized = true;
    Logger::info("UpwaysPipeline successfully initialized (3-pass 81.6 KB 3-Scale 2D Halo-WMMA Recurrent U-Net + Fused 2-Stage KPN architecture).");
}

UpwaysPipeline::~UpwaysPipeline() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_queryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(m_device, m_queryPool, nullptr);
            m_queryPool = VK_NULL_HANDLE;
        }

        auto destroyPipe = [this](VkPipeline& pipe, VkPipelineLayout& layout, VkDescriptorSetLayout& descLayout) {
            if (pipe != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_device, pipe, nullptr);
                pipe = VK_NULL_HANDLE;
            }
            if (layout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(m_device, layout, nullptr);
                layout = VK_NULL_HANDLE;
            }
            if (descLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_device, descLayout, nullptr);
                descLayout = VK_NULL_HANDLE;
            }
        };

        destroyPipe(m_temporalPipeline, m_temporalPipeLayout, m_temporalDescLayout);
        destroyPipe(m_neuralGuidePipeline, m_neuralGuidePipeLayout, m_neuralGuideDescLayout);
        destroyPipe(m_upscalePipeline, m_upscalePipeLayout, m_upscaleDescLayout);

        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }
    }
}

void UpwaysPipeline::initBuffers(const std::string& weightsPath) {
    std::filesystem::path exeDir;
#ifdef _WIN32
    char exePathBuf[MAX_PATH] = {0};
    if (GetModuleFileNameA(NULL, exePathBuf, MAX_PATH)) {
        exeDir = std::filesystem::path(exePathBuf).parent_path();
    }
#elif defined(__linux__)
    std::error_code ec;
    auto p = std::filesystem::canonical("/proc/self/exe", ec);
    if (!ec) exeDir = p.parent_path();
#endif

    std::vector<std::string> candidates;
    if (!weightsPath.empty()) {
        candidates.push_back(weightsPath);
    }
    if (!exeDir.empty()) {
        candidates.push_back((exeDir / "data" / "models" / "upways_weights.bin").string());
        candidates.push_back((exeDir / "upways_weights.bin").string());
        candidates.push_back((exeDir / ".." / "share" / "pathways" / "data" / "models" / "upways_weights.bin").string());
    }
    candidates.push_back("data/models/upways_weights.bin");
    candidates.push_back("../data/models/upways_weights.bin");
    candidates.push_back("../../data/models/upways_weights.bin");
    candidates.push_back("../../../data/models/upways_weights.bin");
    candidates.push_back("/usr/share/pathways/data/models/upways_weights.bin");
    candidates.push_back("/usr/local/share/pathways/data/models/upways_weights.bin");
    candidates.push_back("upways_weights.bin");

    std::string foundPath;
    for (const auto& cand : candidates) {
        std::error_code ec2;
        if (std::filesystem::exists(cand, ec2) && !std::filesystem::is_directory(cand, ec2)) {
            foundPath = cand;
            break;
        }
    }

    std::vector<uint8_t> weightBytes;
    if (!foundPath.empty()) {
        std::ifstream file(foundPath, std::ios::binary | std::ios::ate);
        if (file.is_open()) {
            size_t size = static_cast<size_t>(file.tellg());
            if (size == upways::TOTAL_WEIGHT_BUFFER_SIZE) {
                file.seekg(0, std::ios::beg);
                weightBytes.resize(size);
                file.read(reinterpret_cast<char*>(weightBytes.data()), size);
                Logger::info("UpwaysPipeline loaded {} bytes of pre-tiled WMMA weights from '{}'", size, foundPath);
            } else {
                Logger::warn("UpwaysPipeline ignoring '{}' (size {} != expected {} bytes); using embedded weights.",
                             foundPath, size, upways::TOTAL_WEIGHT_BUFFER_SIZE);
            }
            file.close();
        }
    }

    if (weightBytes.empty()) {
        Logger::info("UpwaysPipeline using embedded neural guide weights ({} bytes)",
                     sizeof(upways::DEFAULT_UPWAYS_WEIGHTS));
        weightBytes.assign(upways::DEFAULT_UPWAYS_WEIGHTS,
                           upways::DEFAULT_UPWAYS_WEIGHTS + sizeof(upways::DEFAULT_UPWAYS_WEIGHTS));
    }

    VkDeviceSize bufferSize = upways::TOTAL_WEIGHT_BUFFER_SIZE;
    m_weightBuffer = std::make_unique<Buffer>(
        m_allocator,
        bufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    void* mapped = m_weightBuffer->map();
    if (mapped) {
        std::memcpy(mapped, weightBytes.data(), bufferSize);
        m_weightBuffer->unmap();
    }
}

void UpwaysPipeline::initImages() {
    m_quadWidth = std::max((m_inputWidth + 1) / 2, 1u);
    m_quadHeight = std::max((m_inputHeight + 1) / 2, 1u);

    size_t qGroupsX = (m_quadWidth + 15u) / 16u;
    size_t qGroupsY = (m_quadHeight + 15u) / 16u;
    size_t numQuadTilePixels = std::max(qGroupsX * qGroupsY * 256u, static_cast<size_t>(256u));
    size_t numQuadPixels = std::max(static_cast<size_t>(m_quadWidth) * m_quadHeight, static_cast<size_t>(1u));

    VkDeviceSize featBufSize = static_cast<VkDeviceSize>(numQuadTilePixels) * 16 * sizeof(uint16_t);
    // 3 * uvec4 = 24 uint16_t elements (48 bytes) per 2x2 quad
    VkDeviceSize kernBufSize = static_cast<VkDeviceSize>(numQuadPixels) * 24 * sizeof(uint16_t);

    m_featureBuffer = std::make_unique<Buffer>(
        m_allocator,
        featBufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );

    m_upscaleKernelBuffer = std::make_unique<Buffer>(
        m_allocator,
        kernBufSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );

    VkImageUsageFlags imgUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                                 VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    // Output image is RGBA16F matching upways_upscale_resolve.comp
    m_outputImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_outputWidth, m_outputHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        imgUsage
    );

    m_confidenceImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16_SFLOAT,
        imgUsage
    );

    m_dummyBlackImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    m_dummyMotionImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
    );

    for (int i = 0; i < 2; ++i) {
        m_diffHistoryImages[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_outputWidth, m_outputHeight,
            VK_FORMAT_R32G32B32A32_UINT,
            imgUsage
        );
        m_specHistoryImages[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_inputWidth, m_inputHeight,
            VK_FORMAT_R32G32_UINT,
            imgUsage
        );
        m_momentsHistoryImages[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_outputWidth, m_outputHeight,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            imgUsage
        );
        m_displayHistoryImages[i] = std::make_unique<Image>(
            m_device, m_allocator,
            m_outputWidth, m_outputHeight,
            VK_FORMAT_R32G32_UINT,
            imgUsage
        );
    }

    m_tempDiffImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R32G32B32A32_UINT,
        imgUsage
    );
    m_tempSpecImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R32G32B32A32_UINT,
        imgUsage
    );
    m_reprojDiffImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        imgUsage
    );
    m_reprojSpecImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        imgUsage
    );

    m_prevNormalDepthImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        imgUsage
    );

    m_prevSpecMotionImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_inputWidth, m_inputHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        imgUsage
    );

    m_initialLayoutsTransitioned = false;
}

void UpwaysPipeline::transitionInitialLayouts(VkCommandBuffer cmd) {
    if (m_initialLayoutsTransitioned) {
        return;
    }

    VkClearColorValue clearZero = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    auto transitionAndClear = [&](Image* img) {
        if (!img) return;
        img->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
        );
        vkCmdClearColorImage(cmd, img->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
        img->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT
        );
    };

    transitionAndClear(m_outputImage.get());
    transitionAndClear(m_confidenceImage.get());
    transitionAndClear(m_dummyBlackImage.get());
    transitionAndClear(m_dummyMotionImage.get());
    transitionAndClear(m_tempDiffImage.get());
    transitionAndClear(m_tempSpecImage.get());
    transitionAndClear(m_reprojDiffImage.get());
    transitionAndClear(m_reprojSpecImage.get());
    transitionAndClear(m_prevNormalDepthImage.get());
    transitionAndClear(m_prevSpecMotionImage.get());

    for (int i = 0; i < 2; ++i) {
        transitionAndClear(m_diffHistoryImages[i].get());
        transitionAndClear(m_specHistoryImages[i].get());
        transitionAndClear(m_momentsHistoryImages[i].get());
        transitionAndClear(m_displayHistoryImages[i].get());
    }

    if (m_featureBuffer) {
        vkCmdFillBuffer(cmd, m_featureBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0);
    }
    if (m_upscaleKernelBuffer) {
        vkCmdFillBuffer(cmd, m_upscaleKernelBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0);
    }

    VkMemoryBarrier2 bufBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    bufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
    bufBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    bufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    bufBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &bufBarrier;
    vkCmdPipelineBarrier2(cmd, &dep);

    m_initialLayoutsTransitioned = true;
}

void UpwaysPipeline::createDescriptorSetLayouts() {
    auto makeBindings = [](const std::vector<VkDescriptorType>& types) {
        std::vector<VkDescriptorSetLayoutBinding> bindings(types.size());
        for (size_t i = 0; i < types.size(); ++i) {
            bindings[i].binding = static_cast<uint32_t>(i);
            bindings[i].descriptorType = types[i];
            bindings[i].descriptorCount = 1;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        return bindings;
    };

    auto createLayout = [this](const std::vector<VkDescriptorSetLayoutBinding>& bindings) {
        VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        info.bindingCount = static_cast<uint32_t>(bindings.size());
        info.pBindings = bindings.data();
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(m_device, &info, nullptr, &layout) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysPipeline: Failed to create descriptor set layout!");
        }
        return layout;
    };

    // Pass 1: Temporal (0..15 storage images, 16 SSBO = 17 bindings)
    std::vector<VkDescriptorType> tTypes(16, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    tTypes.push_back(VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    m_temporalDescLayout = createLayout(makeBindings(tTypes));

    // Pass 2: Neural Guide (0..1: SSBO, 2..7: Image, 8: SSBO = 9 bindings)
    std::vector<VkDescriptorType> ngTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 0: uNeuralFeaturesIn
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 1: uWeights
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 2: uTemporalDiffuseInOut
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 3: uTemporalSpecularInOut
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 4: uReprojDiffuseIn
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 5: uReprojSpecularIn
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 6: uHistoryDiffuseOut
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 7: uHistorySpecularOut
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  // 8: uUpscaleKernelOut (3 * uvec4 per quad)
    };
    m_neuralGuideDescLayout = createLayout(makeBindings(ngTypes));

    // Pass 3: Fused 2-Stage Hierarchical Multi-Scale KPN Upscale & Resolve (0..4: Image, 5: SSBO, 6..16: Image, 17: SSBO = 18 bindings)
    std::vector<VkDescriptorType> upTypes = {
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 0: uDenoisedDiffuse
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 1: uDenoisedSpecular
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 2: uNormalDepth
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 3: uAlbedoRoughness
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 4: uMotionVectors (rgba16f)
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 5: uUpscaleKernelIn
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 6: uDisplayHistoryIn
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 7: uDisplayOut
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 8: uDisplayHistoryOut
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 9: uSpecularMetallic (.a = metallic)
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 10: uSpecularMotion
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 11: uHistoryDiffuseIn
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 12: uHistorySpecularIn
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 13: uHistoryMomentsIn
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 14: uHistoryDiffuseOut
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 15: uHistorySpecularOut
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,  // 16: uHistoryMomentsOut
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER  // 17: uWeights
    };
    m_upscaleDescLayout = createLayout(makeBindings(upTypes));
}

void UpwaysPipeline::allocateDescriptorSets() {
    VkDescriptorPoolSize poolSizes[2]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 128;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 32;

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 16;
    poolInfo.poolSizeCount = 2;
    poolInfo.pPoolSizes = poolSizes;

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create descriptor pool!");
    }

    auto allocSet = [this](VkDescriptorSetLayout layout) {
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1;
        allocInfo.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &set) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysPipeline: Failed to allocate descriptor set!");
        }
        return set;
    };

    for (int i = 0; i < 2; ++i) {
        m_temporalDescSets[i] = allocSet(m_temporalDescLayout);
        m_neuralGuideDescSets[i] = allocSet(m_neuralGuideDescLayout);
        m_upscaleDescSets[i] = allocSet(m_upscaleDescLayout);
    }
}

void UpwaysPipeline::updateDescriptors(
    VkImageView demodDiffuseView,
    VkImageView demodSpecularView,
    VkImageView normalDepthView,
    VkImageView albedoRoughnessView,
    VkImageView surfaceMotionView,
    VkImageView specularMotionView,
    VkImageView /*restirMetadataView*/,
    VkImageView /*displayAlbedoView*/,
    VkImageView /*displayNormalsView*/
) {
    m_cachedDemodDiffuseView = demodDiffuseView;
    m_cachedDemodSpecularView = demodSpecularView;
    m_cachedNormalDepthView = normalDepthView;
    m_cachedAlbedoRoughnessView = albedoRoughnessView;
    m_cachedSurfaceMotionView = surfaceMotionView;
    m_cachedSpecularMotionView = specularMotionView;

    refreshDescriptors();
}

void UpwaysPipeline::refreshDescriptors() {
    if (!m_outputImage || !m_weightBuffer || !m_featureBuffer || !m_upscaleKernelBuffer) {
        return;
    }

    VkImageView fallbackRgba = m_dummyBlackImage ? m_dummyBlackImage->getImageView() : m_outputImage->getImageView();
    VkImageView fallbackRg = m_dummyMotionImage ? m_dummyMotionImage->getImageView() : fallbackRgba;

    VkImageView dDiffView = (m_cachedDemodDiffuseView != VK_NULL_HANDLE) ? m_cachedDemodDiffuseView : fallbackRgba;
    VkImageView dSpecView = (m_cachedDemodSpecularView != VK_NULL_HANDLE) ? m_cachedDemodSpecularView : fallbackRgba;
    VkImageView ndView    = (m_cachedNormalDepthView != VK_NULL_HANDLE) ? m_cachedNormalDepthView : fallbackRgba;
    VkImageView arView    = (m_cachedAlbedoRoughnessView != VK_NULL_HANDLE) ? m_cachedAlbedoRoughnessView : fallbackRgba;
    VkImageView mvView    = (m_cachedSurfaceMotionView != VK_NULL_HANDLE) ? m_cachedSurfaceMotionView : fallbackRg;
    VkImageView smView    = (m_cachedSpecularMotionView != VK_NULL_HANDLE) ? m_cachedSpecularMotionView : fallbackRgba;

    VkImageView prevNdView = m_prevNormalDepthImage ? m_prevNormalDepthImage->getImageView() : ndView;
    VkImageView prevSmView = m_prevSpecMotionImage ? m_prevSpecMotionImage->getImageView() : smView;

    std::vector<VkDescriptorImageInfo> imgInfos;
    std::vector<VkDescriptorBufferInfo> bufInfos;
    std::vector<VkWriteDescriptorSet> writes;
    imgInfos.reserve(128);
    bufInfos.reserve(16);

    auto bindImg = [&](VkDescriptorSet set, uint32_t binding, VkImageView view) {
        imgInfos.push_back({ VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL });
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        w.pImageInfo = &imgInfos.back();
        writes.push_back(w);
    };

    auto bindBuf = [&](VkDescriptorSet set, uint32_t binding, Buffer* buf) {
        bufInfos.push_back({ buf->getBuffer(), 0, buf->getSize() });
        VkWriteDescriptorSet w{ VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET };
        w.dstSet = set;
        w.dstBinding = binding;
        w.descriptorCount = 1;
        w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w.pBufferInfo = &bufInfos.back();
        writes.push_back(w);
    };

    for (int pingPong = 0; pingPong < 2; ++pingPong) {
        int readSlot = pingPong;
        int writeSlot = 1 - pingPong;

        // Pass 1: Temporal Accumulation + 2x2 Quad Feature Extraction (17 bindings)
        VkDescriptorSet tSet = m_temporalDescSets[pingPong];
        bindImg(tSet, 0, dDiffView);
        bindImg(tSet, 1, dSpecView);
        bindImg(tSet, 2, ndView);
        bindImg(tSet, 3, prevNdView);
        bindImg(tSet, 4, mvView);
        bindImg(tSet, 5, arView);
        bindImg(tSet, 6, smView);
        bindImg(tSet, 7, prevSmView);
        bindImg(tSet, 8, m_diffHistoryImages[readSlot]->getImageView());
        bindImg(tSet, 9, m_specHistoryImages[readSlot]->getImageView());
        bindImg(tSet, 10, m_momentsHistoryImages[readSlot]->getImageView());
        bindImg(tSet, 11, m_tempDiffImage->getImageView());
        bindImg(tSet, 12, m_tempSpecImage->getImageView());
        bindImg(tSet, 13, m_diffHistoryImages[writeSlot]->getImageView());
        bindImg(tSet, 14, m_specHistoryImages[writeSlot]->getImageView());
        bindImg(tSet, 15, m_momentsHistoryImages[writeSlot]->getImageView());
        bindBuf(tSet, 16, m_featureBuffer.get());

        // Pass 2: 3-Scale Spatiotemporal Recurrent 2D Halo-WMMA U-Net (9 bindings)
        VkDescriptorSet ngSet = m_neuralGuideDescSets[pingPong];
        bindBuf(ngSet, 0, m_featureBuffer.get());
        bindBuf(ngSet, 1, m_weightBuffer.get());
        bindImg(ngSet, 2, m_tempDiffImage->getImageView());
        bindImg(ngSet, 3, m_tempSpecImage->getImageView());
        bindImg(ngSet, 4, m_reprojDiffImage->getImageView());
        bindImg(ngSet, 5, m_reprojSpecImage->getImageView());
        bindImg(ngSet, 6, m_diffHistoryImages[writeSlot]->getImageView());
        bindImg(ngSet, 7, m_specHistoryImages[writeSlot]->getImageView());
        bindBuf(ngSet, 8, m_upscaleKernelBuffer.get());

        // Pass 3: Fused 2-Stage Hierarchical Multi-Scale KPN Upscale & Physical Remodulation (18 bindings)
        VkDescriptorSet uSet = m_upscaleDescSets[pingPong];
        bindImg(uSet, 0, m_tempDiffImage->getImageView());
        bindImg(uSet, 1, m_tempSpecImage->getImageView());
        bindImg(uSet, 2, ndView);
        bindImg(uSet, 3, arView);
        bindImg(uSet, 4, mvView);
        bindBuf(uSet, 5, m_upscaleKernelBuffer.get());
        bindImg(uSet, 6, m_displayHistoryImages[readSlot]->getImageView());
        bindImg(uSet, 7, m_outputImage->getImageView());
        bindImg(uSet, 8, m_displayHistoryImages[writeSlot]->getImageView());
        bindImg(uSet, 9, dSpecView);
        bindImg(uSet, 10, smView);
        bindImg(uSet, 11, m_diffHistoryImages[readSlot]->getImageView());
        bindImg(uSet, 12, m_specHistoryImages[readSlot]->getImageView());
        bindImg(uSet, 13, m_momentsHistoryImages[readSlot]->getImageView());
        bindImg(uSet, 14, m_diffHistoryImages[writeSlot]->getImageView());
        bindImg(uSet, 15, m_specHistoryImages[writeSlot]->getImageView());
        bindImg(uSet, 16, m_momentsHistoryImages[writeSlot]->getImageView());
        bindBuf(uSet, 17, m_weightBuffer.get());
    }

    vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
}

VkShaderModule UpwaysPipeline::createShaderModule(const std::vector<char>& code) {
    if (code.empty()) {
        throw std::runtime_error("UpwaysPipeline: Empty SPIR-V code provided to createShaderModule!");
    }
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create shader module!");
    }
    return shaderModule;
}

void UpwaysPipeline::createPipelines(
    const std::vector<char>& temporalSpv,
    const std::vector<char>& neuralGuideSpv,
    const std::vector<char>& upscaleResolveSpv
) {
    auto buildPipe = [this](
        const std::vector<char>& spv,
        VkDescriptorSetLayout descLayout,
        uint32_t pcSize,
        bool requireWave32,
        VkPipelineLayout& outLayout,
        VkPipeline& outPipe
    ) {
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = pcSize;

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pcRange;

        if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &outLayout) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysPipeline: Failed to create compute pipeline layout!");
        }

        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
        subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
        subgroupSize32.requiredSubgroupSize = 32;

        VkShaderModule mod = createShaderModule(spv);
        VkComputePipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipeInfo.layout = outLayout;
        pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeInfo.stage.module = mod;
        pipeInfo.stage.pName = "main";
        if (requireWave32) {
            pipeInfo.stage.pNext = &subgroupSize32;
            pipeInfo.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        }

        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &outPipe) != VK_SUCCESS) {
            vkDestroyShaderModule(m_device, mod, nullptr);
            throw std::runtime_error("UpwaysPipeline: Failed to create compute pipeline!");
        }
        vkDestroyShaderModule(m_device, mod, nullptr);
    };

    buildPipe(temporalSpv, m_temporalDescLayout, sizeof(UpwaysTemporalPushConstants), false, m_temporalPipeLayout, m_temporalPipeline);
    buildPipe(neuralGuideSpv, m_neuralGuideDescLayout, sizeof(UpwaysNeuralGuidePushConstants), true, m_neuralGuidePipeLayout, m_neuralGuidePipeline);
    buildPipe(upscaleResolveSpv, m_upscaleDescLayout, sizeof(UpwaysUpscaleResolvePushConstants), true, m_upscalePipeLayout, m_upscalePipeline);
}

void UpwaysPipeline::resize(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight, bool enableSuperRes) {
    m_inputWidth = inputWidth;
    m_inputHeight = inputHeight;
    m_outputWidth = outputWidth;
    m_outputHeight = outputHeight;
    m_superRes = enableSuperRes;

    initImages();
    refreshDescriptors();
    Logger::info("UpwaysPipeline resized: Input {}x{}, Quad {}x{}, Output {}x{} (SuperRes: {})",
                 m_inputWidth, m_inputHeight, m_quadWidth, m_quadHeight, m_outputWidth, m_outputHeight,
                 m_superRes ? "Enabled" : "1.0x (Native)");
}

void UpwaysPipeline::collectQueryResults(int32_t targetSlot) {
    if (m_queryPool == VK_NULL_HANDLE) {
        return;
    }

    const uint32_t startSlot = (targetSlot >= 0) ? (static_cast<uint32_t>(targetSlot) % MAX_QUERY_SLOTS) : 0u;
    const uint32_t endSlot   = (targetSlot >= 0) ? (startSlot + 1u) : MAX_QUERY_SLOTS;

    for (uint32_t slot = startSlot; slot < endSlot; ++slot) {
        if (!m_slotQueriesRecorded[slot]) {
            continue;
        }
        uint64_t results[QUERIES_PER_SLOT * 2]{};
        VkResult r = vkGetQueryPoolResults(
            m_device,
            m_queryPool,
            slot * QUERIES_PER_SLOT,
            QUERIES_PER_SLOT,
            sizeof(results),
            results,
            sizeof(uint64_t) * 2,
            VK_QUERY_RESULT_WITH_AVAILABILITY_BIT | VK_QUERY_RESULT_64_BIT
        );
        if (r == VK_NOT_READY) {
            continue;
        }

        bool allAvailable = true;
        for (uint32_t q = 0; q < QUERIES_PER_SLOT; ++q) {
            if (results[q * 2 + 1] == 0) {
                allAvailable = false;
                break;
            }
        }
        if (!allAvailable) {
            continue;
        }

        const double toMs = m_timestampPeriodNs * 1.0e-6;
        const uint64_t t0 = results[0 * 2 + 0];
        const uint64_t t1 = results[1 * 2 + 0];
        const uint64_t t2 = results[2 * 2 + 0];
        const uint64_t t3 = results[3 * 2 + 0];
        const uint64_t t4 = results[4 * 2 + 0];

        m_lastTimings.temporalMs       = (t1 > t0) ? static_cast<double>(t1 - t0) * toMs : 0.0;
        m_lastTimings.neuralGuideMs    = (t2 > t1) ? static_cast<double>(t2 - t1) * toMs : 0.0;
        m_lastTimings.upscaleResolveMs = (t3 > t2) ? static_cast<double>(t3 - t2) * toMs : 0.0;
        m_lastTimings.historyCopyMs    = (t4 > t3) ? static_cast<double>(t4 - t3) * toMs : 0.0;
        m_lastTimings.totalMs          = (t4 > t0) ? static_cast<double>(t4 - t0) * toMs : 0.0;
        m_lastTimings.valid            = true;
        m_slotQueriesRecorded[slot]    = false;
    }
}

void UpwaysPipeline::recordFrame(
    VkCommandBuffer cmd,
    uint32_t /*frameIndex*/,
    bool resetHistory,
    bool cameraMoved,
    const glm::mat4& /*currInvView*/,
    const glm::mat4& /*prevViewProj*/,
    const glm::mat4& /*invProj*/,
    const glm::mat4& /*prevView*/,
    const glm::vec2& jitterOffset,
    uint32_t totalSamples
) {
    if (!m_temporalPipeline || !m_neuralGuidePipeline || !m_upscalePipeline || !m_outputImage) {
        return;
    }

    if (!m_initialLayoutsTransitioned) {
        transitionInitialLayouts(cmd);
    }

    const uint32_t activeSet = m_pingPongIndex;
    const uint32_t querySlot = activeSet % MAX_QUERY_SLOTS;
    // Harvest retired GPU pass timestamps for this completed slot before resetting it
    collectQueryResults(static_cast<int32_t>(querySlot));
    const uint32_t qBase = querySlot * QUERIES_PER_SLOT;
    if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, m_queryPool, qBase, QUERIES_PER_SLOT);
        m_slotQueriesRecorded[querySlot] = false;
    }

    auto writeTimestamp = [&](uint32_t queryOffset, VkPipelineStageFlags2 stage) {
        if (m_queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(cmd, stage, m_queryPool, qBase + queryOffset);
        }
    };

    auto computeBarrier = [&]() {
        VkMemoryBarrier2 b{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        b.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    };

    const bool isStationaryNative = (!cameraMoved && !resetHistory && !m_superRes && totalSamples > 2u);
    uint32_t resetFlag = (resetHistory ? 1u : 0u) | (isStationaryNative ? 4u : 0u);
    const float minAlphaDiff = (!cameraMoved && totalSamples > 1u)
        ? std::max(1.0f / static_cast<float>(totalSamples), 1.0f / 64.0f)
        : 0.04f;
    const float minAlphaSpec = (!cameraMoved && totalSamples > 1u)
        ? std::max(1.0f / static_cast<float>(totalSamples), 1.0f / 64.0f)
        : 0.06f;

    const uint32_t rGroupsX = (m_inputWidth + 15u) / 16u;
    const uint32_t rGroupsY = (m_inputHeight + 15u) / 16u;

    writeTimestamp(0, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Pass 1: Co-Designed Diffuse & Metallic-Fresnel Specular Demodulation, Temporal Reprojection & Quad Feature Packing
    UpwaysTemporalPushConstants pc1{ m_inputWidth, m_inputHeight, resetFlag, minAlphaDiff, minAlphaSpec };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeLayout, 0, 1, &m_temporalDescSets[activeSet], 0, nullptr);
    vkCmdPushConstants(cmd, m_temporalPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc1), &pc1);
    vkCmdDispatch(cmd, rGroupsX, rGroupsY, 1);
    computeBarrier();
    writeTimestamp(1, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Pass 2: 3-Scale Spatiotemporal Recurrent 2D Halo-WMMA U-Net (40,800 FP16 params = 81,600 bytes over 2x2 quads)
    UpwaysNeuralGuidePushConstants pc2{
        m_inputWidth, m_inputHeight, resetFlag,
        minAlphaDiff, minAlphaSpec, 0.02f, 32.0f, 4.0f, 3.0f
    };
    const uint32_t qGroupsX = (m_quadWidth + 15u) / 16u;
    const uint32_t qGroupsY = (m_quadHeight + 15u) / 16u;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_neuralGuidePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_neuralGuidePipeLayout, 0, 1, &m_neuralGuideDescSets[activeSet], 0, nullptr);
    vkCmdPushConstants(cmd, m_neuralGuidePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc2), &pc2);
    vkCmdDispatch(cmd, qGroupsX, qGroupsY, 1);
    computeBarrier();
    writeTimestamp(2, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Pass 3: Fused 2-Stage Hierarchical Multi-Scale Neural KPN Upscale & Physical Remodulation Resolve
    const uint32_t dGroupsX = (m_outputWidth + 15u) / 16u;
    const uint32_t dGroupsY = (m_outputHeight + 15u) / 16u;
    uint32_t enableDispHistory = (!resetHistory) ? 1u : 0u;
    float jitterScale = m_superRes ? 1.0f : 0.0f;
    float dispHistAlpha = m_superRes ? 0.15f : 0.20f;
    UpwaysUpscaleResolvePushConstants pc3{
        m_inputWidth, m_inputHeight, m_outputWidth, m_outputHeight,
        jitterOffset.x * jitterScale, jitterOffset.y * jitterScale, enableDispHistory, dispHistAlpha
    };
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upscalePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upscalePipeLayout, 0, 1, &m_upscaleDescSets[activeSet], 0, nullptr);
    vkCmdPushConstants(cmd, m_upscalePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc3), &pc3);
    vkCmdDispatch(cmd, dGroupsX, dGroupsY, 1);
    writeTimestamp(3, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Roll current frame normalDepth and specularMotion into m_prevNormalDepthImage and m_prevSpecMotionImage
    const bool needHistoryCopy = cameraMoved || resetHistory || m_superRes || (totalSamples <= 2u);
    if (needHistoryCopy && m_srcNormalDepthImage != VK_NULL_HANDLE && m_prevNormalDepthImage) {
        VkMemoryBarrier2 preCopy{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        preCopy.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        preCopy.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        preCopy.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        preCopy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        VkDependencyInfo preDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        preDep.memoryBarrierCount = 1;
        preDep.pMemoryBarriers = &preCopy;
        vkCmdPipelineBarrier2(cmd, &preDep);

        VkImageCopy copyRegion{};
        copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copyRegion.extent = { m_inputWidth, m_inputHeight, 1 };

        vkCmdCopyImage(cmd, m_srcNormalDepthImage, VK_IMAGE_LAYOUT_GENERAL,
                       m_prevNormalDepthImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);
        if (m_srcSpecMotionImage != VK_NULL_HANDLE && m_prevSpecMotionImage) {
            vkCmdCopyImage(cmd, m_srcSpecMotionImage, VK_IMAGE_LAYOUT_GENERAL,
                           m_prevSpecMotionImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);
        }
    }

    VkMemoryBarrier2 postBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_COPY_BIT;
    postBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
    postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT;
    postBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    postDep.memoryBarrierCount = 1;
    postDep.pMemoryBarriers = &postBarrier;
    vkCmdPipelineBarrier2(cmd, &postDep);

    writeTimestamp(4, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT);
    if (m_queryPool != VK_NULL_HANDLE) {
        m_slotQueriesRecorded[querySlot] = true;
    }

    m_pingPongIndex = 1 - m_pingPongIndex;
}

} // namespace pathways
