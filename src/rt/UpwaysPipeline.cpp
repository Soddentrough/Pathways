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

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <utility>

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
    m_inputWidth(std::max(inputWidth, 1u)),
    m_inputHeight(std::max(inputHeight, 1u)),
    m_outputWidth(std::max(outputWidth, 1u)),
    m_outputHeight(std::max(outputHeight, 1u)),
    m_pyramidDims(computeUpwaysPyramidDims(m_inputWidth, m_inputHeight)),
    m_superRes(enableSuperRes),
    m_format(imageFormat)
{
    Logger::info(
        "UpwaysPipeline 3.0 (Decoupled RDNA 4 FP8 WMMA Spatiotemporal Backbone + Continuous Subpixel Resolve) initializing: "
        "Render {}x{} (L1 {}x{}, L2 {}x{}), Display {}x{} (SuperRes: {})",
        m_inputWidth, m_inputHeight,
        m_pyramidDims.w1, m_pyramidDims.h1,
        m_pyramidDims.w2, m_pyramidDims.h2,
        m_outputWidth, m_outputHeight,
        m_superRes ? "Enabled" : "1.0x (Native)"
    );

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

    createLinearSampler();
    initBuffers(weightsPath);
    initImages();
    createDescriptorSetLayouts();
    allocateDescriptorSets();
    createPipelines(temporalSpv, neuralGuideSpv, upscaleResolveSpv);
    refreshDescriptors();

    m_initialized = true;
    Logger::info(
        "UpwaysPipeline 3.0 initialized ({} params, {} B FP8+FP32 weights, zero-copy ping-pong history).",
        upways::TOTAL_PARAMETER_COUNT,
        upways::TOTAL_WEIGHT_BUFFER_SIZE
    );
}

UpwaysPipeline::~UpwaysPipeline() {
    if (m_device != VK_NULL_HANDLE) {
        if (m_queryPool != VK_NULL_HANDLE) {
            vkDestroyQueryPool(m_device, m_queryPool, nullptr);
            m_queryPool = VK_NULL_HANDLE;
        }

        if (m_temporalPipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_temporalPipeline, nullptr);
            m_temporalPipeline = VK_NULL_HANDLE;
        }
        for (auto& p : m_neuralGuidePipelines) {
            if (p != VK_NULL_HANDLE) {
                vkDestroyPipeline(m_device, p, nullptr);
                p = VK_NULL_HANDLE;
            }
        }
        if (m_upscalePipeline != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, m_upscalePipeline, nullptr);
            m_upscalePipeline = VK_NULL_HANDLE;
        }

        auto destroyLayouts = [this](VkPipelineLayout& pipeLayout, VkDescriptorSetLayout& descLayout) {
            if (pipeLayout != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(m_device, pipeLayout, nullptr);
                pipeLayout = VK_NULL_HANDLE;
            }
            if (descLayout != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(m_device, descLayout, nullptr);
                descLayout = VK_NULL_HANDLE;
            }
        };

        destroyLayouts(m_temporalPipeLayout, m_temporalDescLayout);
        destroyLayouts(m_neuralGuidePipeLayout, m_neuralGuideDescLayout);
        destroyLayouts(m_upscalePipeLayout, m_upscaleDescLayout);

        if (m_descriptorPool != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
            m_descriptorPool = VK_NULL_HANDLE;
        }

        if (m_linearSampler != VK_NULL_HANDLE) {
            vkDestroySampler(m_device, m_linearSampler, nullptr);
            m_linearSampler = VK_NULL_HANDLE;
        }

        destroyNeuralHiddenArrays();
    }
}

void UpwaysPipeline::createLinearSampler() {
    VkSamplerCreateInfo sci{ VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO };
    sci.magFilter = VK_FILTER_LINEAR;
    sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST;
    sci.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.minLod = 0.0f;
    sci.maxLod = 0.0f;
    if (vkCreateSampler(m_device, &sci, nullptr, &m_linearSampler) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create linear clamp-to-edge sampler!");
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
            const size_t size = static_cast<size_t>(file.tellg());
            if (size == upways::TOTAL_WEIGHT_BUFFER_SIZE) {
                file.seekg(0, std::ios::beg);
                weightBytes.resize(size);
                file.read(reinterpret_cast<char*>(weightBytes.data()), static_cast<std::streamsize>(size));
                Logger::info("UpwaysPipeline loaded {} bytes of pre-tiled FP8+FP32 weights from '{}'", size, foundPath);
            } else {
                Logger::warn(
                    "UpwaysPipeline ignoring '{}' (size {} != expected {} bytes); using embedded weights.",
                    foundPath, size, upways::TOTAL_WEIGHT_BUFFER_SIZE
                );
            }
            file.close();
        }
    }

    if (weightBytes.empty()) {
        if (sizeof(upways::DEFAULT_UPWAYS_WEIGHTS) == upways::TOTAL_WEIGHT_BUFFER_SIZE) {
            Logger::info("UpwaysPipeline using embedded DEFAULT_UPWAYS_WEIGHTS ({} bytes)", sizeof(upways::DEFAULT_UPWAYS_WEIGHTS));
            weightBytes.assign(
                upways::DEFAULT_UPWAYS_WEIGHTS,
                upways::DEFAULT_UPWAYS_WEIGHTS + sizeof(upways::DEFAULT_UPWAYS_WEIGHTS)
            );
        } else {
            Logger::info("UpwaysPipeline using header kUpwaysWeightsFp8Blob ({} bytes)", sizeof(kUpwaysWeightsFp8Blob));
            weightBytes.assign(
                kUpwaysWeightsFp8Blob,
                kUpwaysWeightsFp8Blob + sizeof(kUpwaysWeightsFp8Blob)
            );
        }
    }

    const VkDeviceSize bufferSize = upways::TOTAL_WEIGHT_BUFFER_SIZE;
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

void UpwaysPipeline::createNeuralHiddenArray(ArrayImage2D& outImg, uint32_t width, uint32_t height, uint32_t layers) {
    if (outImg.view != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, outImg.view, nullptr);
        outImg.view = VK_NULL_HANDLE;
    }
    if (outImg.image != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, outImg.image, outImg.allocation);
        outImg.image = VK_NULL_HANDLE;
        outImg.allocation = VK_NULL_HANDLE;
    }

    VkImageCreateInfo imageInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.extent.width = width;
    imageInfo.extent.height = height;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = layers;
    imageInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    imageInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                      VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    VkResult res = vmaCreateImage(m_allocator, &imageInfo, &allocInfo, &outImg.image, &outImg.allocation, nullptr);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to allocate 2D array image via VMA!");
    }

    VkImageViewCreateInfo viewInfo{ VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO };
    viewInfo.image = outImg.image;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    viewInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.baseMipLevel = 0;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.baseArrayLayer = 0;
    viewInfo.subresourceRange.layerCount = layers;

    res = vkCreateImageView(m_device, &viewInfo, nullptr, &outImg.view);
    if (res != VK_SUCCESS) {
        vmaDestroyImage(m_allocator, outImg.image, outImg.allocation);
        outImg.image = VK_NULL_HANDLE;
        outImg.allocation = VK_NULL_HANDLE;
        throw std::runtime_error("UpwaysPipeline: Failed to create 2D array image view!");
    }

    outImg.width = width;
    outImg.height = height;
    outImg.layers = layers;
}

void UpwaysPipeline::destroyNeuralHiddenArrays() {
    for (auto& arr : m_neuralHidden) {
        if (arr.view != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
            vkDestroyImageView(m_device, arr.view, nullptr);
            arr.view = VK_NULL_HANDLE;
        }
        if (arr.image != VK_NULL_HANDLE && m_allocator != VK_NULL_HANDLE) {
            vmaDestroyImage(m_allocator, arr.image, arr.allocation);
            arr.image = VK_NULL_HANDLE;
            arr.allocation = VK_NULL_HANDLE;
        }
        arr.width = 0;
        arr.height = 0;
        arr.layers = 0;
    }
}

void UpwaysPipeline::initRenderResources(uint32_t slotMask) {
    m_inputWidth = std::max(m_inputWidth, 1u);
    m_inputHeight = std::max(m_inputHeight, 1u);
    m_pyramidDims = computeUpwaysPyramidDims(m_inputWidth, m_inputHeight);

    const VkDeviceSize subpixelBufBytes = static_cast<VkDeviceSize>(m_inputWidth) * m_inputHeight * 4u * 8u;

    m_pyramidBuffer = std::make_unique<Buffer>(
        m_allocator,
        m_pyramidDims.totalBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );

    m_subpixelParamsBuffer = std::make_unique<Buffer>(
        m_allocator,
        subpixelBufBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );

    constexpr VkImageUsageFlags imgUsage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    m_tLogDiff      = std::make_unique<Image>(m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
    m_tLogSpec      = std::make_unique<Image>(m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
    m_tGuideDiff    = std::make_unique<Image>(m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
    m_tGuideSpec    = std::make_unique<Image>(m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
    m_tDilatedMv    = std::make_unique<Image>(m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
    m_tTemporalConf = std::make_unique<Image>(m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
    m_transientsTransitioned = false;

    for (uint32_t i = 0; i < 2u; ++i) {
        if ((slotMask & (1u << i)) != 0u) {
            m_geomNd[i]   = std::make_unique<Image>(m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
            m_geomHitT[i] = std::make_unique<Image>(m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
            createNeuralHiddenArray(m_neuralHidden[i], m_inputWidth, m_inputHeight, NEURAL_HIDDEN_LAYERS);
            m_lrSlotWidth[i] = m_inputWidth;
            m_lrSlotHeight[i] = m_inputHeight;
            m_lrSlotTransitioned[i] = false;
        }
    }

    m_initialLayoutsTransitioned = false;
}

void UpwaysPipeline::ensureActiveSlotSized() {
    const uint32_t activeSet = m_pingPongIndex;
    if (m_lrSlotWidth[activeSet] == m_inputWidth && m_lrSlotHeight[activeSet] == m_inputHeight) {
        return;
    }

    vkDeviceWaitIdle(m_device);
    constexpr VkImageUsageFlags imgUsage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    m_geomNd[activeSet]   = std::make_unique<Image>(m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
    m_geomHitT[activeSet] = std::make_unique<Image>(m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
    createNeuralHiddenArray(m_neuralHidden[activeSet], m_inputWidth, m_inputHeight, NEURAL_HIDDEN_LAYERS);
    m_lrSlotWidth[activeSet] = m_inputWidth;
    m_lrSlotHeight[activeSet] = m_inputHeight;
    m_lrSlotTransitioned[activeSet] = false;
    m_initialLayoutsTransitioned = false;
    refreshDescriptors();
}

void UpwaysPipeline::initImages() {
    m_inputWidth = std::max(m_inputWidth, 1u);
    m_inputHeight = std::max(m_inputHeight, 1u);
    m_outputWidth = std::max(m_outputWidth, 1u);
    m_outputHeight = std::max(m_outputHeight, 1u);

    constexpr VkImageUsageFlags imgUsage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    m_outputImage = std::make_unique<Image>(
        m_device, m_allocator,
        m_outputWidth, m_outputHeight,
        VK_FORMAT_R16G16B16A16_SFLOAT,
        imgUsage
    );

    if (!m_dummyBlackImage) {
        m_dummyBlackImage = std::make_unique<Image>(
            m_device, m_allocator,
            1u, 1u,
            VK_FORMAT_R16G16B16A16_SFLOAT,
            imgUsage
        );
    }

    for (int i = 0; i < 2; ++i) {
        m_irrDiffHr[i]   = std::make_unique<Image>(m_device, m_allocator, m_outputWidth, m_outputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
        m_irrSpecHr[i]   = std::make_unique<Image>(m_device, m_allocator, m_outputWidth, m_outputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
        m_guideDiffHr[i] = std::make_unique<Image>(m_device, m_allocator, m_outputWidth, m_outputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
        m_guideSpecHr[i] = std::make_unique<Image>(m_device, m_allocator, m_outputWidth, m_outputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
    }

    m_hrTransitioned = false;
    initRenderResources(0x3u);
    m_historyValid = false;
}

void UpwaysPipeline::transitionInitialLayouts(VkCommandBuffer cmd) {
    if (m_initialLayoutsTransitioned) {
        return;
    }

    VkClearColorValue clearZero = { { 0.0f, 0.0f, 0.0f, 0.0f } };
    VkImageSubresourceRange clearRange1 = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    constexpr VkAccessFlags2 kPostClearComputeAccess =
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    auto transitionAndClear = [&](Image* img) {
        if (!img) return;
        img->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
        );
        vkCmdClearColorImage(cmd, img->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange1);
        img->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
            kPostClearComputeAccess
        );
    };

    auto transitionAndClearArray = [&](ArrayImage2D& arr) {
        if (arr.image == VK_NULL_HANDLE) return;
        VkImageSubresourceRange rangeArr = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, arr.layers };

        VkImageMemoryBarrier2 preClear{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        preClear.srcStageMask = VK_PIPELINE_STAGE_2_NONE;
        preClear.srcAccessMask = VK_ACCESS_2_NONE;
        preClear.dstStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        preClear.dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        preClear.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        preClear.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        preClear.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preClear.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        preClear.image = arr.image;
        preClear.subresourceRange = rangeArr;

        VkDependencyInfo dep1{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep1.imageMemoryBarrierCount = 1;
        dep1.pImageMemoryBarriers = &preClear;
        vkCmdPipelineBarrier2(cmd, &dep1);

        vkCmdClearColorImage(cmd, arr.image, VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &rangeArr);

        VkImageMemoryBarrier2 postClear{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        postClear.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        postClear.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        postClear.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        postClear.dstAccessMask = kPostClearComputeAccess;
        postClear.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        postClear.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        postClear.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        postClear.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        postClear.image = arr.image;
        postClear.subresourceRange = rangeArr;

        VkDependencyInfo dep2{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep2.imageMemoryBarrierCount = 1;
        dep2.pImageMemoryBarriers = &postClear;
        vkCmdPipelineBarrier2(cmd, &dep2);
    };

    if (!m_hrTransitioned) {
        transitionAndClear(m_outputImage.get());
        transitionAndClear(m_dummyBlackImage.get());
        for (int i = 0; i < 2; ++i) {
            transitionAndClear(m_irrDiffHr[i].get());
            transitionAndClear(m_irrSpecHr[i].get());
            transitionAndClear(m_guideDiffHr[i].get());
            transitionAndClear(m_guideSpecHr[i].get());
        }
        m_hrTransitioned = true;
    }

    bool anyBufCleared = false;
    if (!m_transientsTransitioned) {
        transitionAndClear(m_tLogDiff.get());
        transitionAndClear(m_tLogSpec.get());
        transitionAndClear(m_tGuideDiff.get());
        transitionAndClear(m_tGuideSpec.get());
        transitionAndClear(m_tDilatedMv.get());
        transitionAndClear(m_tTemporalConf.get());

        if (m_pyramidBuffer) {
            vkCmdFillBuffer(cmd, m_pyramidBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0);
            anyBufCleared = true;
        }
        if (m_subpixelParamsBuffer) {
            vkCmdFillBuffer(cmd, m_subpixelParamsBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0);
            anyBufCleared = true;
        }
        m_transientsTransitioned = true;
    }

    for (int i = 0; i < 2; ++i) {
        if (!m_lrSlotTransitioned[i]) {
            transitionAndClear(m_geomNd[i].get());
            transitionAndClear(m_geomHitT[i].get());
            transitionAndClearArray(m_neuralHidden[i]);
            m_lrSlotTransitioned[i] = true;
        }
    }

    if (anyBufCleared) {
        VkMemoryBarrier2 bufBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
        bufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        bufBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        bufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        bufBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

        VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &bufBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

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

    // Pass 1: Temporal Marshaling & Demodulation (21 bindings: 0..6 storage img, 7..11 sampled, 12..19 storage img, 20 SSBO)
    std::vector<VkDescriptorType> p1Types(21, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    for (uint32_t b = 7; b <= 11; ++b) {
        p1Types[b] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    p1Types[20] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    m_temporalDescLayout = createLayout(makeBindings(p1Types));

    // Pass 2: 6-Stage Global FP8 WMMA Multi-Scale Backbone (5 bindings: 0..3 SSBOs, 4 2D-Array storage image)
    std::vector<VkDescriptorType> p2Types = {
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 0: WeightFp8Buf
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 1: WeightFp32Buf
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 2: PyramidFp8Buf
        VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, // 3: SubpixelParamsBuf
        VK_DESCRIPTOR_TYPE_STORAGE_IMAGE   // 4: uCurrNeuralHidden (image2DArray)
    };
    m_neuralGuideDescLayout = createLayout(makeBindings(p2Types));

    // Pass 3: Continuous Subpixel Ray-Locus Gather & Catmull-Rom Resolve (18 bindings)
    std::vector<VkDescriptorType> p3Types(18, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    p3Types[7] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    p3Types[8] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    for (uint32_t b = 9; b <= 12; ++b) {
        p3Types[b] = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    }
    m_upscaleDescLayout = createLayout(makeBindings(p3Types));
}

void UpwaysPipeline::allocateDescriptorSets() {
    VkDescriptorPoolSize poolSizes[3]{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 128;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    poolSizes[1].descriptorCount = 64;
    poolSizes[2].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[2].descriptorCount = 64;

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 16;
    poolInfo.poolSizeCount = 3;
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
    VkImageView restirMetadataView,
    VkImageView /*displayAlbedoView*/,
    VkImageView /*displayNormalsView*/
) {
    m_cachedDemodDiffuseView = demodDiffuseView;
    m_cachedDemodSpecularView = demodSpecularView;
    m_cachedNormalDepthView = normalDepthView;
    m_cachedAlbedoRoughnessView = albedoRoughnessView;
    m_cachedSurfaceMotionView = surfaceMotionView;
    m_cachedSpecularMotionView = specularMotionView;
    m_cachedAuxMetadataView = restirMetadataView;

    refreshDescriptors();
}

void UpwaysPipeline::refreshDescriptors() {
    if (!m_outputImage || !m_weightBuffer || !m_pyramidBuffer || !m_subpixelParamsBuffer || m_linearSampler == VK_NULL_HANDLE) {
        return;
    }

    VkImageView fallbackZero = m_dummyBlackImage ? m_dummyBlackImage->getImageView() : m_outputImage->getImageView();

    VkImageView dDiffView = (m_cachedDemodDiffuseView != VK_NULL_HANDLE) ? m_cachedDemodDiffuseView : fallbackZero;
    VkImageView dSpecView = (m_cachedDemodSpecularView != VK_NULL_HANDLE) ? m_cachedDemodSpecularView : fallbackZero;
    VkImageView ndView    = (m_cachedNormalDepthView != VK_NULL_HANDLE) ? m_cachedNormalDepthView : fallbackZero;
    VkImageView arView    = (m_cachedAlbedoRoughnessView != VK_NULL_HANDLE) ? m_cachedAlbedoRoughnessView : fallbackZero;
    VkImageView mvView    = (m_cachedSurfaceMotionView != VK_NULL_HANDLE) ? m_cachedSurfaceMotionView : fallbackZero;
    VkImageView smView    = (m_cachedSpecularMotionView != VK_NULL_HANDLE) ? m_cachedSpecularMotionView : fallbackZero;
    VkImageView auxView   = (m_cachedAuxMetadataView != VK_NULL_HANDLE) ? m_cachedAuxMetadataView : fallbackZero;

    for (int curr = 0; curr < 2; ++curr) {
        const int prev = 1 - curr;

        // Pass 1 descriptor set (21 bindings)
        {
            std::array<VkDescriptorImageInfo, 20> ii{};
            auto setStorage = [&](uint32_t idx, VkImageView view) {
                ii[idx] = { VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL };
            };
            auto setSampled = [&](uint32_t idx, VkImageView view) {
                ii[idx] = { m_linearSampler, view, VK_IMAGE_LAYOUT_GENERAL };
            };

            setStorage(0, dDiffView);
            setStorage(1, dSpecView);
            setStorage(2, arView);
            setStorage(3, ndView);
            setStorage(4, mvView);
            setStorage(5, smView);
            setStorage(6, auxView);
            setSampled(7, m_irrDiffHr[prev]->getImageView());
            setSampled(8, m_irrSpecHr[prev]->getImageView());
            setSampled(9, m_geomNd[prev]->getImageView());
            setSampled(10, m_geomHitT[prev]->getImageView());
            setSampled(11, m_neuralHidden[prev].view);
            setStorage(12, m_geomNd[curr]->getImageView());
            setStorage(13, m_geomHitT[curr]->getImageView());
            setStorage(14, m_tLogDiff->getImageView());
            setStorage(15, m_tLogSpec->getImageView());
            setStorage(16, m_tGuideDiff->getImageView());
            setStorage(17, m_tGuideSpec->getImageView());
            setStorage(18, m_tDilatedMv->getImageView());
            setStorage(19, m_tTemporalConf->getImageView());

            VkDescriptorBufferInfo pyrBi{ m_pyramidBuffer->getBuffer(), 0, m_pyramidBuffer->getSize() };

            std::array<VkWriteDescriptorSet, 21> writes{};
            for (uint32_t b = 0; b < 20; ++b) {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = m_temporalDescSets[curr];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = (b >= 7u && b <= 11u)
                    ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER
                    : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[b].pImageInfo = &ii[b];
            }
            writes[20].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[20].dstSet = m_temporalDescSets[curr];
            writes[20].dstBinding = 20;
            writes[20].descriptorCount = 1;
            writes[20].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[20].pBufferInfo = &pyrBi;

            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        // Pass 2 descriptor set (5 bindings)
        {
            std::array<VkDescriptorBufferInfo, 4> bi = {{
                { m_weightBuffer->getBuffer(), upways::FP8_WEIGHT_SECTION_OFFSET, upways::FP8_WEIGHT_SECTION_BYTES },
                { m_weightBuffer->getBuffer(), upways::FP32_WEIGHT_SECTION_OFFSET, upways::FP32_WEIGHT_SECTION_BYTES },
                { m_pyramidBuffer->getBuffer(), 0, m_pyramidBuffer->getSize() },
                { m_subpixelParamsBuffer->getBuffer(), 0, m_subpixelParamsBuffer->getSize() },
            }};
            VkDescriptorImageInfo hidIi{ VK_NULL_HANDLE, m_neuralHidden[curr].view, VK_IMAGE_LAYOUT_GENERAL };

            std::array<VkWriteDescriptorSet, 5> writes{};
            for (uint32_t b = 0; b < 4; ++b) {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = m_neuralGuideDescSets[curr];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[b].pBufferInfo = &bi[b];
            }
            writes[4].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[4].dstSet = m_neuralGuideDescSets[curr];
            writes[4].dstBinding = 4;
            writes[4].descriptorCount = 1;
            writes[4].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[4].pImageInfo = &hidIi;

            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }

        // Pass 3 descriptor set (18 bindings)
        {
            std::array<VkDescriptorImageInfo, 18> ii{};
            auto setStorage = [&](uint32_t idx, VkImageView view) {
                ii[idx] = { VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL };
            };
            auto setSampled = [&](uint32_t idx, VkImageView view) {
                ii[idx] = { m_linearSampler, view, VK_IMAGE_LAYOUT_GENERAL };
            };

            setStorage(0, ndView);
            setStorage(1, m_tLogDiff->getImageView());
            setStorage(2, m_tLogSpec->getImageView());
            setStorage(3, m_tGuideDiff->getImageView());
            setStorage(4, m_tGuideSpec->getImageView());
            setStorage(5, m_tDilatedMv->getImageView());
            setStorage(6, m_tTemporalConf->getImageView());
            setSampled(9, m_irrDiffHr[prev]->getImageView());
            setSampled(10, m_irrSpecHr[prev]->getImageView());
            setSampled(11, m_guideDiffHr[prev]->getImageView());
            setSampled(12, m_guideSpecHr[prev]->getImageView());
            setStorage(13, m_outputImage->getImageView());
            setStorage(14, m_irrDiffHr[curr]->getImageView());
            setStorage(15, m_irrSpecHr[curr]->getImageView());
            setStorage(16, m_guideDiffHr[curr]->getImageView());
            setStorage(17, m_guideSpecHr[curr]->getImageView());

            VkDescriptorBufferInfo subBi{ m_subpixelParamsBuffer->getBuffer(), 0, m_subpixelParamsBuffer->getSize() };
            VkDescriptorBufferInfo w32Bi{
                m_weightBuffer->getBuffer(),
                upways::FP32_WEIGHT_SECTION_OFFSET,
                upways::FP32_WEIGHT_SECTION_BYTES
            };

            std::array<VkWriteDescriptorSet, 18> writes{};
            for (uint32_t b = 0; b < 18; ++b) {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = m_upscaleDescSets[curr];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1;
                if (b == 7u) {
                    writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[b].pBufferInfo = &subBi;
                } else if (b == 8u) {
                    writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[b].pBufferInfo = &w32Bi;
                } else if (b >= 9u && b <= 12u) {
                    writes[b].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
                    writes[b].pImageInfo = &ii[b];
                } else {
                    writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    writes[b].pImageInfo = &ii[b];
                }
            }

            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0, nullptr);
        }
    }
}

VkShaderModule UpwaysPipeline::createShaderModule(const std::vector<char>& code) {
    if (code.empty() || (code.size() % 4u) != 0u) {
        throw std::runtime_error("UpwaysPipeline: Invalid SPIR-V byte array provided to createShaderModule!");
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
    auto createPipeLayout = [this](VkDescriptorSetLayout descLayout) {
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0;
        pcRange.size = sizeof(UpwaysPushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1;
        layoutInfo.pSetLayouts = &descLayout;
        layoutInfo.pushConstantRangeCount = 1;
        layoutInfo.pPushConstantRanges = &pcRange;

        VkPipelineLayout outLayout = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &outLayout) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysPipeline: Failed to create compute pipeline layout!");
        }
        return outLayout;
    };

    auto buildComputePipe = [this](
        VkShaderModule mod,
        VkPipelineLayout layout,
        bool requireWave32,
        int32_t specStageId = -1
    ) {
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroupSize32{};
        subgroupSize32.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO;
        subgroupSize32.requiredSubgroupSize = 32;

        uint32_t stageVal = (specStageId >= 0) ? static_cast<uint32_t>(specStageId) : 0u;
        VkSpecializationMapEntry specEntry{};
        specEntry.constantID = 0;
        specEntry.offset = 0;
        specEntry.size = sizeof(uint32_t);

        VkSpecializationInfo specInfo{};
        specInfo.mapEntryCount = 1;
        specInfo.pMapEntries = &specEntry;
        specInfo.dataSize = sizeof(uint32_t);
        specInfo.pData = &stageVal;

        VkComputePipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipeInfo.layout = layout;
        pipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        pipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        pipeInfo.stage.module = mod;
        pipeInfo.stage.pName = "main";
        if (specStageId >= 0) {
            pipeInfo.stage.pSpecializationInfo = &specInfo;
        }
        if (requireWave32) {
            pipeInfo.stage.pNext = &subgroupSize32;
            pipeInfo.stage.flags = VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
        }

        VkPipeline outPipe = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(m_device, VK_NULL_HANDLE, 1, &pipeInfo, nullptr, &outPipe) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysPipeline: Failed to create compute pipeline!");
        }
        return outPipe;
    };

    m_temporalPipeLayout = createPipeLayout(m_temporalDescLayout);
    m_neuralGuidePipeLayout = createPipeLayout(m_neuralGuideDescLayout);
    m_upscalePipeLayout = createPipeLayout(m_upscaleDescLayout);

    VkShaderModule smP1 = createShaderModule(temporalSpv);
    VkShaderModule smP2 = createShaderModule(neuralGuideSpv);
    VkShaderModule smP3 = createShaderModule(upscaleResolveSpv);

    try {
        m_temporalPipeline = buildComputePipe(smP1, m_temporalPipeLayout, false);
        for (uint32_t st = 0; st < NUM_NEURAL_STAGES; ++st) {
            m_neuralGuidePipelines[st] = buildComputePipe(smP2, m_neuralGuidePipeLayout, true, static_cast<int32_t>(st));
        }
        m_upscalePipeline = buildComputePipe(smP3, m_upscalePipeLayout, false);
    } catch (...) {
        vkDestroyShaderModule(m_device, smP1, nullptr);
        vkDestroyShaderModule(m_device, smP2, nullptr);
        vkDestroyShaderModule(m_device, smP3, nullptr);
        throw;
    }

    vkDestroyShaderModule(m_device, smP1, nullptr);
    vkDestroyShaderModule(m_device, smP2, nullptr);
    vkDestroyShaderModule(m_device, smP3, nullptr);
}

void UpwaysPipeline::resize(uint32_t inputWidth, uint32_t inputHeight, uint32_t outputWidth, uint32_t outputHeight, bool enableSuperRes) {
    const uint32_t newInW  = std::max(inputWidth, 1u);
    const uint32_t newInH  = std::max(inputHeight, 1u);
    const uint32_t newOutW = std::max(outputWidth, 1u);
    const uint32_t newOutH = std::max(outputHeight, 1u);

    if (m_outputImage && newOutW == m_outputWidth && newOutH == m_outputHeight) {
        if (newInW == m_inputWidth && newInH == m_inputHeight) {
            m_superRes = enableSuperRes;
            return;
        }

        // Lossless Dynamic Resolution Scaling (DRS): display-resolution irradiance/guide
        // histories and the previous frame's LR recurrent state remain intact!
        m_inputWidth = newInW;
        m_inputHeight = newInH;
        m_superRes = enableSuperRes;
        initRenderResources(1u << m_pingPongIndex);
        refreshDescriptors();
        Logger::info(
            "UpwaysPipeline 3.0 lossless DRS resize: Render {}x{} (L1 {}x{}, L2 {}x{}), Display {}x{} preserved (SuperRes: {})",
            m_inputWidth, m_inputHeight,
            m_pyramidDims.w1, m_pyramidDims.h1,
            m_pyramidDims.w2, m_pyramidDims.h2,
            m_outputWidth, m_outputHeight,
            m_superRes ? "Enabled" : "1.0x (Native)"
        );
        return;
    }

    m_inputWidth = newInW;
    m_inputHeight = newInH;
    m_outputWidth = newOutW;
    m_outputHeight = newOutH;
    m_superRes = enableSuperRes;

    initImages();
    refreshDescriptors();
    Logger::info(
        "UpwaysPipeline 3.0 resized: Render {}x{} (L1 {}x{}, L2 {}x{}), Display {}x{} (SuperRes: {})",
        m_inputWidth, m_inputHeight,
        m_pyramidDims.w1, m_pyramidDims.h1,
        m_pyramidDims.w2, m_pyramidDims.h2,
        m_outputWidth, m_outputHeight,
        m_superRes ? "Enabled" : "1.0x (Native)"
    );
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
        m_lastTimings.historyCopyMs    = 0.0; // Zero-copy ping-pong history swap
        m_lastTimings.totalMs          = (t4 > t0) ? static_cast<double>(t4 - t0) * toMs : 0.0;
        m_lastTimings.valid            = true;
        m_slotQueriesRecorded[slot]    = false;
    }
}

void UpwaysPipeline::recordFrame(
    VkCommandBuffer cmd,
    uint32_t /*frameIndex*/,
    bool resetHistory,
    bool /*cameraMoved*/,
    const glm::mat4& /*currInvView*/,
    const glm::mat4& /*prevViewProj*/,
    const glm::mat4& /*invProj*/,
    const glm::mat4& /*prevView*/,
    const glm::vec2& jitterOffset,
    uint32_t /*totalSamples*/,
    bool coupledWithExternalUpscaler,
    bool /*heuristicFallback*/
) {
    if (!m_temporalPipeline || !m_neuralGuidePipelines[0] || !m_upscalePipeline || !m_outputImage) {
        return;
    }

    ensureActiveSlotSized();

    if (!m_initialLayoutsTransitioned) {
        transitionInitialLayouts(cmd);
    }

    const uint32_t activeSet = m_pingPongIndex;
    const uint32_t querySlot = activeSet % MAX_QUERY_SLOTS;
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
        b.dstAccessMask =
            VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
            VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
            VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        dep.memoryBarrierCount = 1;
        dep.pMemoryBarriers = &b;
        vkCmdPipelineBarrier2(cmd, &dep);
    };

    const bool effectiveReset = resetHistory || !m_historyValid;
    const bool preserveGuide = (!m_superRes) || coupledWithExternalUpscaler;
    const float jitterScale = (m_superRes && !coupledWithExternalUpscaler) ? 1.0f : 0.0f;

    UpwaysPushConstants pc{};
    pc.renderWidth = m_inputWidth;
    pc.renderHeight = m_inputHeight;
    pc.displayWidth = m_outputWidth;
    pc.displayHeight = m_outputHeight;
    pc.jitterX = jitterOffset.x * jitterScale;
    pc.jitterY = jitterOffset.y * jitterScale;
    pc.preExposure = 1.0f;
    pc.resetHistory = effectiveReset ? 1u : 0u;
    pc.preserveJitterGuide = preserveGuide ? 1u : 0u;
    pc.stageId = 0u;
    pc.srcWidth = m_inputWidth;
    pc.srcHeight = m_inputHeight;
    pc.dstWidth = m_inputWidth;
    pc.dstHeight = m_inputHeight;

    writeTimestamp(0, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT);

    // Pass 1: Point-wise Demodulation, 3x3 Closest-Depth MV Dilation & FP8 L0 Marshaling
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_temporalPipeLayout, 0, 1, &m_temporalDescSets[activeSet], 0, nullptr);
    vkCmdPushConstants(cmd, m_temporalPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UpwaysPushConstants), &pc);
    vkCmdDispatch(cmd, (m_inputWidth + 15u) / 16u, (m_inputHeight + 15u) / 16u, 1u);

    computeBarrier();
    writeTimestamp(1, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Pass 2: 6-Stage Global FP8 E4M3 WMMA Multi-Scale Backbone (L0 -> L1 -> L2 -> L2 -> L1 -> L0, 16x8 pixel tiles)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_neuralGuidePipeLayout, 0, 1, &m_neuralGuideDescSets[activeSet], 0, nullptr);

    const std::array<std::pair<uint32_t, uint32_t>, NUM_NEURAL_STAGES> stageDims = {{
        { m_pyramidDims.w0, m_pyramidDims.h0 },
        { m_pyramidDims.w1, m_pyramidDims.h1 },
        { m_pyramidDims.w2, m_pyramidDims.h2 },
        { m_pyramidDims.w2, m_pyramidDims.h2 },
        { m_pyramidDims.w1, m_pyramidDims.h1 },
        { m_pyramidDims.w0, m_pyramidDims.h0 },
    }};

    for (uint32_t st = 0; st < NUM_NEURAL_STAGES; ++st) {
        pc.stageId = st;
        pc.srcWidth = stageDims[st].first;
        pc.srcHeight = stageDims[st].second;
        pc.dstWidth = stageDims[st].first;
        pc.dstHeight = stageDims[st].second;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_neuralGuidePipelines[st]);
        vkCmdPushConstants(cmd, m_neuralGuidePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UpwaysPushConstants), &pc);
        vkCmdDispatch(cmd, (stageDims[st].first + 15u) / 16u, (stageDims[st].second + 7u) / 8u, 1u);
        computeBarrier();
    }

    writeTimestamp(2, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);

    // Pass 3: Continuous Subpixel Ray-Locus Gather & 5-Tap Catmull-Rom History Resolve
    pc.stageId = 0u;
    pc.srcWidth = m_inputWidth;
    pc.srcHeight = m_inputHeight;
    pc.dstWidth = m_outputWidth;
    pc.dstHeight = m_outputHeight;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upscalePipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upscalePipeLayout, 0, 1, &m_upscaleDescSets[activeSet], 0, nullptr);
    vkCmdPushConstants(cmd, m_upscalePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(UpwaysPushConstants), &pc);
    vkCmdDispatch(cmd, (m_outputWidth + 15u) / 16u, (m_outputHeight + 15u) / 16u, 1u);

    computeBarrier();
    writeTimestamp(3, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT);
    writeTimestamp(4, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT);

    if (m_queryPool != VK_NULL_HANDLE) {
        m_slotQueriesRecorded[querySlot] = true;
    }
    m_historyValid = true;
    m_pingPongIndex = 1u - activeSet;
}

} // namespace pathways
