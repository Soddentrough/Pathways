#include "UpwaysV5Engine.hpp"

#include "core/Logger.hpp"
#include "rt/upways_v5_default_int8_weights.hpp"
#include "rt/upways_v5_default_weights.hpp"
#include "rt/upways_v5_weights.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
#else
    #include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace pathways {

namespace {

constexpr uint32_t UPWAYS_V5_FLAG_RESET_HISTORY      = 1u;
constexpr uint32_t UPWAYS_V5_FLAG_ENABLE_SUPERRES    = 2u;
constexpr uint32_t UPWAYS_V5_FLAG_ZERO_RESIDUAL_MODE = 4u;
constexpr uint32_t UPWAYS_V5_FLAG_FUSED_BAND_PACK    = 8u;

constexpr uint32_t divCeil(uint32_t a, uint32_t b) noexcept {
    return (a + b - 1u) / b;
}

void emitComputeBarrier(
    VkCommandBuffer cmd,
    VkPipelineStageFlags2 dstStage = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
    VkAccessFlags2 dstAccess = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
) {
    VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mb.dstStageMask = dstStage;
    mb.dstAccessMask = dstAccess;

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.memoryBarrierCount = 1u;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
}

std::filesystem::path getExecutableDirectory() {
#ifdef _WIN32
    wchar_t buf[MAX_PATH]{};
    DWORD len = GetModuleFileNameW(nullptr, buf, MAX_PATH);
    if (len > 0 && len < MAX_PATH) {
        return std::filesystem::path(buf).parent_path();
    }
    return {};
#else
    std::error_code ec;
    std::filesystem::path exePath = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec && !exePath.empty()) {
        return exePath.parent_path();
    }
    return {};
#endif
}

std::unique_ptr<Buffer> makeDeviceSsbo(VmaAllocator allocator, VkDeviceSize bytes) {
    return std::make_unique<Buffer>(
        allocator,
        std::max<VkDeviceSize>(bytes, 256ULL),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );
}

VkDescriptorBufferInfo bufInfo(const Buffer* buf) noexcept {
    return { buf->getBuffer(), 0u, buf->getSize() };
}

} // namespace

UpwaysV5Engine::UpwaysV5Engine(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VmaAllocator allocator,
    uint32_t inputWidth,
    uint32_t inputHeight,
    uint32_t outputWidth,
    uint32_t outputHeight,
    const std::vector<char>& prepSpv,
    const std::vector<char>& unetWmmaSpv,
    const std::vector<char>& affinitySpv,
    const std::vector<char>& upscalerWmmaSpv,
    const std::vector<char>& resolveSpv,
    const std::string& weightsPath,
    bool enableSuperRes,
    VkFormat imageFormat,
    UpwaysComputePrecision precision,
    uint32_t outputFullHeight
)
    : m_device(device)
    , m_physDevice(physicalDevice)
    , m_allocator(allocator)
    , m_inputWidth(std::max(1u, inputWidth))
    , m_inputHeight(std::max(1u, inputHeight))
    , m_outputWidth(std::max(1u, outputWidth))
    , m_outputHeight(std::max(1u, outputHeight))
    , m_outputFullHeight((outputFullHeight > 0u) ? std::max(outputFullHeight, std::max(1u, outputHeight)) : std::max(1u, outputHeight))
    , m_enableSuperRes(enableSuperRes)
    , m_precision((precision == UpwaysComputePrecision::Int8) ? UpwaysComputePrecision::Int8 : UpwaysComputePrecision::Fp8)
    , m_imageFormat(VK_FORMAT_R16G16B16A16_SFLOAT)
    , m_dims(computeUpwaysV5Dims(m_inputWidth, m_inputHeight))
{
    if (precision == UpwaysComputePrecision::Fp16) {
        Logger::warn("UpwaysPipeline 5.0 supports FP8/INT8 only; normalizing requested FP16 precision to FP8.");
    }
    if (imageFormat != VK_FORMAT_UNDEFINED && imageFormat != VK_FORMAT_R16G16B16A16_SFLOAT) {
        Logger::info("UpwaysPipeline 5.0: normalizing output format to R16G16B16A16_SFLOAT to match rgba16f shader bindings.");
    }

    VkPhysicalDeviceVulkan13Features vk13Features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
    };
    VkPhysicalDeviceFeatures2 features2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    features2.pNext = &vk13Features;
    vkGetPhysicalDeviceFeatures2(m_physDevice, &features2);
    m_hasSubgroupSizeControl = (vk13Features.subgroupSizeControl == VK_TRUE);
    m_hasComputeFullSubgroups = (vk13Features.computeFullSubgroups == VK_TRUE);

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_physDevice, &props);
    m_timestampPeriodNs = (props.limits.timestampPeriod > 0.0f) ? props.limits.timestampPeriod : 1.0f;

    Logger::info(
        "UpwaysPipeline 5.0 ({} WMMA) initializing: LR {}x{} -> HR {}x{} (SuperRes: {}, L1 {}x{}, L2 {}x{}, L3 {}x{}).",
        upwaysPrecisionDisplayName(m_precision),
        m_inputWidth, m_inputHeight,
        m_outputWidth, m_outputHeight,
        m_enableSuperRes ? "ON" : "OFF",
        m_dims.w1, m_dims.h1,
        m_dims.w2, m_dims.h2,
        m_dims.w3, m_dims.h3
    );

    VkQueryPoolCreateInfo qpInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpInfo.queryCount = MAX_QUERY_SLOTS * QUERIES_PER_SLOT;
    if (vkCreateQueryPool(m_device, &qpInfo, nullptr, &m_queryPool) != VK_SUCCESS) {
        Logger::warn("UpwaysV5Engine: Failed to create timestamp query pool; per-pass profiling disabled.");
        m_queryPool = VK_NULL_HANDLE;
    }

    initWeights(weightsPath);
    initRenderResources(0x3u);
    initDisplayResources();
    createDescriptorSetLayouts();
    allocateDescriptorSets();
    refreshDescriptors();
    createPipelines(prepSpv, unetWmmaSpv, affinitySpv, upscalerWmmaSpv, resolveSpv);

    Logger::info(
        "UpwaysPipeline 5.0 initialized ({} active params, {} B uploaded {} weights).",
        upways_v5::ACTIVE_PARAMETER_COUNT,
        m_weightBuffer->getSize(),
        upwaysPrecisionDisplayName(m_precision)
    );
}

UpwaysV5Engine::~UpwaysV5Engine() {
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    auto destroyPipe = [this](VkPipeline& p) {
        if (p != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, p, nullptr);
            p = VK_NULL_HANDLE;
        }
    };
    auto destroyPipeLayout = [this](VkPipelineLayout& l) {
        if (l != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(m_device, l, nullptr);
            l = VK_NULL_HANDLE;
        }
    };
    auto destroyDescLayout = [this](VkDescriptorSetLayout& l) {
        if (l != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(m_device, l, nullptr);
            l = VK_NULL_HANDLE;
        }
    };

    destroyPipe(m_prepPipeline);
    for (VkPipeline& p : m_unetPipelines) {
        destroyPipe(p);
    }
    destroyPipe(m_affStage0Pipeline);
    destroyPipe(m_affPoolPipeline);
    destroyPipe(m_affReconDil3Pipeline);
    destroyPipe(m_affReconDil1Pipeline);
    destroyPipe(m_affStage3Pipeline);
    for (VkPipeline& p : m_upscalerPipelines) {
        destroyPipe(p);
    }
    destroyPipe(m_resolvePipeline);

    destroyPipeLayout(m_prepPipeLayout);
    destroyPipeLayout(m_unetPipeLayout);
    destroyPipeLayout(m_affinityPipeLayout);
    destroyPipeLayout(m_upscalerPipeLayout);
    destroyPipeLayout(m_resolvePipeLayout);

    if (m_pipelineCache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
    }
    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    }
    destroyDescLayout(m_prepDescLayout);
    destroyDescLayout(m_unetDescLayout);
    destroyDescLayout(m_affinityDescLayout);
    destroyDescLayout(m_upscalerDescLayout);
    destroyDescLayout(m_resolveDescLayout);

    if (m_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_device, m_queryPool, nullptr);
    }
}

void UpwaysV5Engine::initWeights(const std::string& weightsPath) {
    static_assert(sizeof(upways_v5::DEFAULT_UPWAYS_V5_WEIGHTS) == upways_v5::TOTAL_WEIGHT_BYTES,
                  "EMBEDDED DEFAULT_UPWAYS_V5_WEIGHTS size must equal 190,208 B");
    static_assert(sizeof(upways_v5::DEFAULT_UPWAYS_V5_INT8_WEIGHTS) == upways_v5::TOTAL_WEIGHT_BYTES,
                  "EMBEDDED DEFAULT_UPWAYS_V5_INT8_WEIGHTS size must equal 190,208 B");

    const bool isInt8 = (m_precision == UpwaysComputePrecision::Int8);
    const char* binName = isInt8 ? "upways_v5_weights_int8.bin" : "upways_v5_weights.bin";

    std::vector<uint8_t> rawWeights;
    std::vector<std::string> candidates;
    if (!weightsPath.empty()) {
        candidates.push_back(weightsPath);
    }
    if (std::filesystem::path exeDir = getExecutableDirectory(); !exeDir.empty()) {
        candidates.push_back((exeDir / "data" / "models" / binName).string());
        candidates.push_back((exeDir / binName).string());
        candidates.push_back((exeDir / ".." / "share" / "pathways" / "data" / "models" / binName).string());
    }
    candidates.push_back(std::string("data/models/") + binName);
    candidates.push_back(std::string("../data/models/") + binName);
    candidates.push_back(std::string("../../data/models/") + binName);
    candidates.push_back(std::string("../../../data/models/") + binName);
    candidates.push_back(std::string("/usr/share/pathways/data/models/") + binName);
    candidates.push_back(std::string("/usr/local/share/pathways/data/models/") + binName);
    candidates.push_back(binName);

    std::string loadedFrom;
    for (const auto& candidate : candidates) {
        if (candidate.empty()) {
            continue;
        }
        std::ifstream f(candidate, std::ios::binary | std::ios::ate);
        if (!f.is_open()) {
            continue;
        }
        const std::streamsize sz = f.tellg();
        if (sz == static_cast<std::streamsize>(upways_v5::TOTAL_WEIGHT_BYTES)) {
            rawWeights.resize(upways_v5::TOTAL_WEIGHT_BYTES);
            f.seekg(0, std::ios::beg);
            if (f.read(reinterpret_cast<char*>(rawWeights.data()), sz)) {
                loadedFrom = candidate;
                break;
            }
            rawWeights.clear();
        } else {
            Logger::warn(
                "UpwaysV5Engine: ignoring '{}' (size {} != expected {} B); using embedded weights.",
                candidate, static_cast<size_t>(sz), upways_v5::TOTAL_WEIGHT_BYTES
            );
        }
    }

    if (rawWeights.empty()) {
        const uint8_t* embedded = isInt8 ? upways_v5::DEFAULT_UPWAYS_V5_INT8_WEIGHTS : upways_v5::DEFAULT_UPWAYS_V5_WEIGHTS;
        rawWeights.assign(embedded, embedded + upways_v5::TOTAL_WEIGHT_BYTES);
        Logger::info(
            "UpwaysV5Engine using embedded {} ({} B).",
            isInt8 ? "DEFAULT_UPWAYS_V5_INT8_WEIGHTS" : "DEFAULT_UPWAYS_V5_WEIGHTS",
            rawWeights.size()
        );
    } else {
        Logger::info("UpwaysV5Engine loaded {} B {} weights from '{}'.",
                     rawWeights.size(), upwaysPrecisionDisplayName(m_precision), loadedFrom);
    }

    m_weightBuffer = std::make_unique<Buffer>(
        m_allocator,
        rawWeights.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    m_weightBuffer->copyFrom(rawWeights.data(), rawWeights.size());
}

void UpwaysV5Engine::initRenderResources(uint32_t slotMask) {
    m_dims = computeUpwaysV5Dims(m_inputWidth, m_inputHeight);

    constexpr VkImageUsageFlags kStorageUsage =
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    for (uint32_t s = 0; s < 2u; ++s) {
        if ((slotMask & (1u << s)) == 0u) {
            continue;
        }
        m_diffHist[s] = std::make_unique<Image>(
            m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, kStorageUsage
        );
        m_specHist[s] = std::make_unique<Image>(
            m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, kStorageUsage
        );
        m_geomHist[s] = std::make_unique<Image>(
            m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, kStorageUsage
        );
        m_hiddenHist[s] = std::make_unique<Image>(
            m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16_SFLOAT, kStorageUsage
        );
        m_slotWidth[s] = m_inputWidth;
        m_slotHeight[s] = m_inputHeight;
        m_lrSlotTransitioned[s] = false;
    }

    m_cleanRadLr = std::make_unique<Image>(
        m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, kStorageUsage
    );

    const VkDeviceSize n0 = static_cast<VkDeviceSize>(m_dims.w0) * m_dims.h0;
    const VkDeviceSize n1 = static_cast<VkDeviceSize>(m_dims.w1) * m_dims.h1;
    const VkDeviceSize n2 = static_cast<VkDeviceSize>(m_dims.w2) * m_dims.h2;
    const VkDeviceSize n3 = static_cast<VkDeviceSize>(m_dims.w3) * m_dims.h3;

    m_x1Buf   = makeDeviceSsbo(m_allocator, n1 * 128ULL);
    m_s1Buf   = makeDeviceSsbo(m_allocator, n1 * 96ULL);
    m_s2Buf   = makeDeviceSsbo(m_allocator, n2 * 160ULL);
    m_unetABuf = makeDeviceSsbo(m_allocator, std::max({ n1 * 128ULL, n2 * 192ULL, n3 * 256ULL }));
    m_unetBBuf = makeDeviceSsbo(m_allocator, std::max({ n1 * 128ULL, n2 * 160ULL, n3 * 256ULL }));
    m_k3Buf   = makeDeviceSsbo(m_allocator, n3 * 64ULL);
    m_k2Buf   = makeDeviceSsbo(m_allocator, n2 * 64ULL);
    m_k1Buf   = makeDeviceSsbo(m_allocator, n1 * 64ULL);
    m_denL0Buf = makeDeviceSsbo(m_allocator, n0 * 64ULL);

    m_pyr0Buf = makeDeviceSsbo(m_allocator, n0 * 32ULL);
    m_pyr1Buf = makeDeviceSsbo(m_allocator, n1 * 32ULL);
    m_pyr2Buf = makeDeviceSsbo(m_allocator, n2 * 32ULL);
    m_pyr3Buf = makeDeviceSsbo(m_allocator, n3 * 32ULL);
    m_hatABuf = makeDeviceSsbo(m_allocator, n1 * 32ULL);
    m_hatBBuf = makeDeviceSsbo(m_allocator, n0 * 32ULL);

    m_srInBuf    = makeDeviceSsbo(m_allocator, n1 * 64ULL);
    m_srFieldBuf = makeDeviceSsbo(m_allocator, n0 * 48ULL);

    m_transientsTransitioned = false;
    m_initialLayoutsTransitioned = false;
}

void UpwaysV5Engine::initDisplayResources() {
    constexpr VkImageUsageFlags kDisplayUsage =
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    for (uint32_t s = 0; s < 2u; ++s) {
        m_colorHr[s] = std::make_unique<Image>(
            m_device, m_allocator, m_outputWidth, m_outputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, kDisplayUsage
        );
    }

    m_outputImage = std::make_unique<Image>(
        m_device, m_allocator, m_outputWidth, m_outputFullHeight, m_imageFormat, kDisplayUsage
    );

    if (!m_dummySsboBuffer) {
        m_dummySsboBuffer = makeDeviceSsbo(m_allocator, 256ULL);
    }
    if (!m_dummyBandPackBuffer) {
        m_dummyBandPackBuffer = std::make_unique<Buffer>(
            m_allocator, 256ULL,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
        );
    }

    if (!m_dummyRgba16fImage) {
        m_dummyRgba16fImage = std::make_unique<Image>(
            m_device, m_allocator, 1u, 1u, VK_FORMAT_R16G16B16A16_SFLOAT, kDisplayUsage
        );
    }
    if (!m_dummyRg16fImage) {
        m_dummyRg16fImage = std::make_unique<Image>(
            m_device, m_allocator, 1u, 1u, VK_FORMAT_R16G16_SFLOAT, kDisplayUsage
        );
    }
    if (!m_dummyR32uiImage) {
        m_dummyR32uiImage = std::make_unique<Image>(
            m_device, m_allocator, 1u, 1u, VK_FORMAT_R32_UINT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
    }
    if (!m_dummyRg32uiImage) {
        m_dummyRg32uiImage = std::make_unique<Image>(
            m_device, m_allocator, 1u, 1u, VK_FORMAT_R32G32_UINT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
    }
    if (!m_dummyRgba32uiImage) {
        m_dummyRgba32uiImage = std::make_unique<Image>(
            m_device, m_allocator, 1u, 1u, VK_FORMAT_R32G32B32A32_UINT,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
    }
    if (!m_dummyR8Image) {
        m_dummyR8Image = std::make_unique<Image>(
            m_device, m_allocator, 1u, 1u, VK_FORMAT_R8_UNORM,
            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT
        );
    }

    m_hrTransitioned = false;
    m_initialLayoutsTransitioned = false;
    m_historyValid = false;
}

void UpwaysV5Engine::ensureActiveSlotSized() {
    const uint32_t curr = m_pingPongIndex & 1u;
    if (m_slotWidth[curr] == m_inputWidth && m_slotHeight[curr] == m_inputHeight) {
        return;
    }

    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    constexpr VkImageUsageFlags kStorageUsage =
        VK_IMAGE_USAGE_STORAGE_BIT |
        VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT;

    m_diffHist[curr] = std::make_unique<Image>(
        m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, kStorageUsage
    );
    m_specHist[curr] = std::make_unique<Image>(
        m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, kStorageUsage
    );
    m_geomHist[curr] = std::make_unique<Image>(
        m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, kStorageUsage
    );
    m_hiddenHist[curr] = std::make_unique<Image>(
        m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16_SFLOAT, kStorageUsage
    );
    m_slotWidth[curr] = m_inputWidth;
    m_slotHeight[curr] = m_inputHeight;
    m_lrSlotTransitioned[curr] = false;
    m_initialLayoutsTransitioned = false;
    refreshDescriptors();
}

void UpwaysV5Engine::transitionInitialLayouts(VkCommandBuffer cmd) {
    if (m_initialLayoutsTransitioned) {
        return;
    }

    VkClearColorValue clearZero{};
    VkImageSubresourceRange range1{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
    constexpr VkAccessFlags2 kPostClearAccess =
        VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT |
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT |
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;

    auto transitionAndClear = [&](Image* img) {
        if (!img) {
            return;
        }
        img->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
        );
        vkCmdClearColorImage(cmd, img->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &range1);
        img->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, kPostClearAccess
        );
    };

    if (!m_hrTransitioned) {
        transitionAndClear(m_outputImage.get());
        transitionAndClear(m_dummyRgba16fImage.get());
        transitionAndClear(m_dummyRg16fImage.get());
        transitionAndClear(m_dummyR32uiImage.get());
        transitionAndClear(m_dummyRg32uiImage.get());
        transitionAndClear(m_dummyRgba32uiImage.get());
        transitionAndClear(m_dummyR8Image.get());
        for (int i = 0; i < 2; ++i) {
            transitionAndClear(m_colorHr[i].get());
        }
        m_hrTransitioned = true;
    }

    bool anyBufCleared = false;
    auto clearSsbo = [&](Buffer* buf) {
        if (buf) {
            vkCmdFillBuffer(cmd, buf->getBuffer(), 0, VK_WHOLE_SIZE, 0u);
            anyBufCleared = true;
        }
    };

    if (!m_transientsTransitioned) {
        transitionAndClear(m_cleanRadLr.get());
        clearSsbo(m_dummySsboBuffer.get());
        clearSsbo(m_x1Buf.get());
        clearSsbo(m_s1Buf.get());
        clearSsbo(m_s2Buf.get());
        clearSsbo(m_unetABuf.get());
        clearSsbo(m_unetBBuf.get());
        clearSsbo(m_k3Buf.get());
        clearSsbo(m_k2Buf.get());
        clearSsbo(m_k1Buf.get());
        clearSsbo(m_denL0Buf.get());
        clearSsbo(m_pyr0Buf.get());
        clearSsbo(m_pyr1Buf.get());
        clearSsbo(m_pyr2Buf.get());
        clearSsbo(m_pyr3Buf.get());
        clearSsbo(m_hatABuf.get());
        clearSsbo(m_hatBBuf.get());
        clearSsbo(m_srInBuf.get());
        clearSsbo(m_srFieldBuf.get());
        m_transientsTransitioned = true;
    }

    for (int i = 0; i < 2; ++i) {
        if (!m_lrSlotTransitioned[i]) {
            transitionAndClear(m_diffHist[i].get());
            transitionAndClear(m_specHist[i].get());
            transitionAndClear(m_geomHist[i].get());
            transitionAndClear(m_hiddenHist[i].get());
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
        dep.memoryBarrierCount = 1u;
        dep.pMemoryBarriers = &bufBarrier;
        vkCmdPipelineBarrier2(cmd, &dep);
    }

    m_initialLayoutsTransitioned = true;
}

void UpwaysV5Engine::createDescriptorSetLayouts() {
    auto createLayout = [this](const std::vector<VkDescriptorType>& types) {
        std::vector<VkDescriptorSetLayoutBinding> bindings(types.size());
        for (size_t i = 0; i < types.size(); ++i) {
            bindings[i].binding = static_cast<uint32_t>(i);
            bindings[i].descriptorType = types[i];
            bindings[i].descriptorCount = 1u;
            bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo info{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
        info.bindingCount = static_cast<uint32_t>(bindings.size());
        info.pBindings = bindings.data();
        VkDescriptorSetLayout layout = VK_NULL_HANDLE;
        if (vkCreateDescriptorSetLayout(m_device, &info, nullptr, &layout) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysV5Engine: Failed to create descriptor set layout!");
        }
        return layout;
    };

    // Pass 1 (upways_v5_prep.comp): 0..10 storage images, 11 SSBO (OutX1Buf)
    std::vector<VkDescriptorType> prepTypes(12, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    prepTypes[11] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    m_prepDescLayout = createLayout(prepTypes);

    // Pass 2A (upways_v5_unet_wmma.comp): 0..5 SSBOs
    m_unetDescLayout = createLayout(std::vector<VkDescriptorType>(6, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));

    // Pass 2B (upways_v5_affinity.comp): 0..4 SSBOs, 5..17 storage images, 18 SSBO (BandPackOut)
    std::vector<VkDescriptorType> affinityTypes(19, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    for (uint32_t i = 0; i < 5u; ++i) {
        affinityTypes[i] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    }
    affinityTypes[18] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    m_affinityDescLayout = createLayout(affinityTypes);

    // Pass 3 (upways_v5_upscaler_wmma.comp): 0..3 SSBOs
    m_upscalerDescLayout = createLayout(std::vector<VkDescriptorType>(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER));

    // Pass 4 (upways_v5_resolve.comp): 0..2 SSBOs, 3..8 storage images, 9 SSBO (BandPackOut)
    std::vector<VkDescriptorType> resolveTypes(10, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    resolveTypes[0] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    resolveTypes[1] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    resolveTypes[2] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    resolveTypes[9] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    m_resolveDescLayout = createLayout(resolveTypes);
}

void UpwaysV5Engine::allocateDescriptorSets() {
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 260u;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 220u;

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 36u;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysV5Engine: Failed to create descriptor pool!");
    }

    auto allocSet = [this](VkDescriptorSetLayout layout) {
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1u;
        allocInfo.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &set) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysV5Engine: Failed to allocate descriptor set!");
        }
        return set;
    };

    for (uint32_t i = 0; i < 2u; ++i) {
        m_prepDescSets[i] = allocSet(m_prepDescLayout);
        m_affStage0DescSets[i] = allocSet(m_affinityDescLayout);
        for (uint32_t p = 0; p < 2u; ++p) {
            m_affL0DescSets[i][p] = allocSet(m_affinityDescLayout);
            m_resolveDescSets[i][p] = allocSet(m_resolveDescLayout);
        }
    }
    for (uint32_t s = 0; s < NUM_UNET_STAGES; ++s) {
        m_unetDescSets[s] = allocSet(m_unetDescLayout);
    }
    for (uint32_t i = 0; i < 3u; ++i) {
        m_affPoolDescSets[i] = allocSet(m_affinityDescLayout);
        m_affReconDescSets[i] = allocSet(m_affinityDescLayout);
    }
    for (uint32_t s = 0; s < NUM_UPSCALER_STAGES; ++s) {
        m_upscalerDescSets[s] = allocSet(m_upscalerDescLayout);
    }
}

void UpwaysV5Engine::updateDescriptors(
    VkImageView mlIrradianceView,
    VkImageView mlPrimaryGuideView,
    VkImageView mlSurfaceMvView,
    VkImageView mlDiffHitDistView,
    VkImageView mlSpecVmvView,
    VkImageView mlSkipCompactView
) {
    m_cachedMlIrradianceView = mlIrradianceView;
    m_cachedMlPrimaryGuideView = mlPrimaryGuideView;
    m_cachedMlSurfaceMvView = mlSurfaceMvView;
    m_cachedMlDiffHitDistView = mlDiffHitDistView;
    m_cachedMlSpecVmvView = mlSpecVmvView;
    m_cachedMlSkipCompactView = mlSkipCompactView;

    refreshDescriptors();
}

void UpwaysV5Engine::setBandPackTargets(
    VkBuffer slot0Buffer,
    VkDeviceSize slot0Size,
    VkBuffer slot1Buffer,
    VkDeviceSize slot1Size
) {
    const VkBuffer s1Buf = (slot1Buffer != VK_NULL_HANDLE) ? slot1Buffer : slot0Buffer;
    const VkDeviceSize s1Sz = (slot1Buffer != VK_NULL_HANDLE) ? slot1Size : slot0Size;
    if (m_bandPackBuf[0] == slot0Buffer &&
        m_bandPackSize[0] == slot0Size &&
        m_bandPackBuf[1] == s1Buf &&
        m_bandPackSize[1] == s1Sz) {
        return;
    }
    if (m_bandPackBuf[0] != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }
    m_bandPackBuf[0] = slot0Buffer;
    m_bandPackSize[0] = slot0Size;
    m_bandPackBuf[1] = s1Buf;
    m_bandPackSize[1] = s1Sz;
    refreshDescriptors();
}

void UpwaysV5Engine::refreshDescriptors() {
    if (!m_outputImage || !m_weightBuffer || !m_x1Buf || !m_srFieldBuf ||
        !m_dummySsboBuffer || !m_dummyBandPackBuffer) {
        return;
    }

    VkImageView fallbackRgba16f = m_dummyRgba16fImage ? m_dummyRgba16fImage->getImageView() : VK_NULL_HANDLE;
    VkImageView fallbackRg16f = m_dummyRg16fImage ? m_dummyRg16fImage->getImageView() : VK_NULL_HANDLE;
    VkImageView fallbackR32ui = m_dummyR32uiImage ? m_dummyR32uiImage->getImageView() : VK_NULL_HANDLE;
    VkImageView fallbackRg32ui = m_dummyRg32uiImage ? m_dummyRg32uiImage->getImageView() : VK_NULL_HANDLE;
    VkImageView fallbackRgba32ui = m_dummyRgba32uiImage ? m_dummyRgba32uiImage->getImageView() : VK_NULL_HANDLE;
    VkImageView fallbackR8 = m_dummyR8Image ? m_dummyR8Image->getImageView() : VK_NULL_HANDLE;

    const VkImageView irrView = (m_cachedMlIrradianceView != VK_NULL_HANDLE) ? m_cachedMlIrradianceView : fallbackRg32ui;
    const VkImageView primGuideView = (m_cachedMlPrimaryGuideView != VK_NULL_HANDLE) ? m_cachedMlPrimaryGuideView : fallbackRgba32ui;
    const VkImageView surfaceMvView = (m_cachedMlSurfaceMvView != VK_NULL_HANDLE) ? m_cachedMlSurfaceMvView : fallbackR32ui;
    const VkImageView diffHitDistView = (m_cachedMlDiffHitDistView != VK_NULL_HANDLE) ? m_cachedMlDiffHitDistView : fallbackR8;
    const VkImageView specVmvView = (m_cachedMlSpecVmvView != VK_NULL_HANDLE) ? m_cachedMlSpecVmvView : fallbackRg32ui;
    const VkImageView skipCompactView = (m_cachedMlSkipCompactView != VK_NULL_HANDLE) ? m_cachedMlSkipCompactView : fallbackR32ui;
    if (irrView == VK_NULL_HANDLE || primGuideView == VK_NULL_HANDLE ||
        surfaceMvView == VK_NULL_HANDLE || diffHitDistView == VK_NULL_HANDLE ||
        specVmvView == VK_NULL_HANDLE || skipCompactView == VK_NULL_HANDLE ||
        fallbackRgba16f == VK_NULL_HANDLE || fallbackRg16f == VK_NULL_HANDLE) {
        return;
    }

    const VkDescriptorBufferInfo w8Bi{
        m_weightBuffer->getBuffer(),
        0u,
        upways_v5::FP8_WEIGHT_SECTION_BYTES
    };
    const VkDescriptorBufferInfo w32Bi{
        m_weightBuffer->getBuffer(),
        upways_v5::FP8_WEIGHT_SECTION_BYTES,
        upways_v5::FP32_WEIGHT_SECTION_BYTES
    };
    const VkDescriptorBufferInfo dummyBi = bufInfo(m_dummySsboBuffer.get());
    std::array<VkDescriptorBufferInfo, 2> packBi{};
    for (int p = 0; p < 2; ++p) {
        if (m_bandPackBuf[p] != VK_NULL_HANDLE) {
            packBi[p] = { m_bandPackBuf[p], 0u, std::max<VkDeviceSize>(m_bandPackSize[p], 256ULL) };
        } else {
            packBi[p] = bufInfo(m_dummyBandPackBuffer.get());
        }
    }

    // 1. Pass 1: upways_v5_prep.comp (0..10 images, 11 SSBO m_x1Buf)
    const VkDescriptorBufferInfo x1Bi = bufInfo(m_x1Buf.get());
    for (int curr = 0; curr < 2; ++curr) {
        const int prev = 1 - curr;
        std::array<VkDescriptorImageInfo, 11> ii{};
        ii[0]  = { VK_NULL_HANDLE, irrView, VK_IMAGE_LAYOUT_GENERAL };
        ii[1]  = { VK_NULL_HANDLE, primGuideView, VK_IMAGE_LAYOUT_GENERAL };
        ii[2]  = { VK_NULL_HANDLE, surfaceMvView, VK_IMAGE_LAYOUT_GENERAL };
        ii[3]  = { VK_NULL_HANDLE, diffHitDistView, VK_IMAGE_LAYOUT_GENERAL };
        ii[4]  = { VK_NULL_HANDLE, specVmvView, VK_IMAGE_LAYOUT_GENERAL };
        ii[5]  = { VK_NULL_HANDLE, skipCompactView, VK_IMAGE_LAYOUT_GENERAL };
        ii[6]  = { VK_NULL_HANDLE, m_diffHist[prev]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        ii[7]  = { VK_NULL_HANDLE, m_specHist[prev]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        ii[8]  = { VK_NULL_HANDLE, m_geomHist[prev]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        ii[9]  = { VK_NULL_HANDLE, m_hiddenHist[prev]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        ii[10] = { VK_NULL_HANDLE, m_geomHist[curr]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

        std::array<VkWriteDescriptorSet, 12> writes{};
        for (uint32_t b = 0; b < 11u; ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = m_prepDescSets[curr];
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1u;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[b].pImageInfo = &ii[b];
        }
        writes[11].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[11].dstSet = m_prepDescSets[curr];
        writes[11].dstBinding = 11u;
        writes[11].descriptorCount = 1u;
        writes[11].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[11].pBufferInfo = &x1Bi;
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0u, nullptr);
    }

    // 2. Pass 2A: upways_v5_unet_wmma.comp (12 dedicated stage bindings, zero aliasing)
    const VkDescriptorBufferInfo s1Bi = bufInfo(m_s1Buf.get());
    const VkDescriptorBufferInfo s2Bi = bufInfo(m_s2Buf.get());
    const VkDescriptorBufferInfo unetABi = bufInfo(m_unetABuf.get());
    const VkDescriptorBufferInfo unetBBi = bufInfo(m_unetBBuf.get());
    const VkDescriptorBufferInfo k3Bi = bufInfo(m_k3Buf.get());
    const VkDescriptorBufferInfo k2Bi = bufInfo(m_k2Buf.get());
    const VkDescriptorBufferInfo k1Bi = bufInfo(m_k1Buf.get());
    const VkDescriptorBufferInfo denL0Bi = bufInfo(m_denL0Buf.get());

    const std::array<std::array<VkDescriptorBufferInfo, 4>, NUM_UNET_STAGES> unetBindings = {{
        { x1Bi,    dummyBi, unetABi, dummyBi }, // 0  ENC1_PW1
        { unetABi, dummyBi, s1Bi,    dummyBi }, // 1  ENC1_DW_PW2
        { s1Bi,    dummyBi, unetABi, dummyBi }, // 2  ENC2_POOL_PW1
        { unetABi, dummyBi, s2Bi,    dummyBi }, // 3  ENC2_DW_PW2
        { s2Bi,    dummyBi, unetABi, dummyBi }, // 4  BOT3_POOL_PW1
        { unetABi, dummyBi, unetBBi, dummyBi }, // 5  BOT3_DW1_PW2
        { unetBBi, dummyBi, unetABi, dummyBi }, // 6  BOT3_DW2_PW3
        { unetABi, dummyBi, unetBBi, k3Bi    }, // 7  BOT3_PW4_KHEAD
        { unetBBi, s2Bi,    unetABi, dummyBi }, // 8  DEC2_UP_CAT_PW1
        { unetABi, dummyBi, unetBBi, k2Bi    }, // 9  DEC2_DW_PW2_KH
        { unetBBi, s1Bi,    unetABi, dummyBi }, // 10 DEC1_UP_CAT_PW1
        { unetABi, dummyBi, k1Bi,    denL0Bi }, // 11 DEC1_DW_PW2_HD
    }};
    for (uint32_t s = 0; s < NUM_UNET_STAGES; ++s) {
        std::array<VkDescriptorBufferInfo, 6> bi = {
            w8Bi, w32Bi,
            unetBindings[s][0], unetBindings[s][1], unetBindings[s][2], unetBindings[s][3]
        };
        std::array<VkWriteDescriptorSet, 6> writes{};
        for (uint32_t b = 0; b < 6u; ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = m_unetDescSets[s];
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1u;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &bi[b];
        }
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0u, nullptr);
    }

    // 3. Pass 2B: upways_v5_affinity.comp helper (5 SSBOs + 13 images + 1 pack SSBO)
    const VkDescriptorBufferInfo pyr0Bi = bufInfo(m_pyr0Buf.get());
    const VkDescriptorBufferInfo pyr1Bi = bufInfo(m_pyr1Buf.get());
    const VkDescriptorBufferInfo pyr2Bi = bufInfo(m_pyr2Buf.get());
    const VkDescriptorBufferInfo pyr3Bi = bufInfo(m_pyr3Buf.get());
    const VkDescriptorBufferInfo hatABi = bufInfo(m_hatABuf.get());
    const VkDescriptorBufferInfo hatBBi = bufInfo(m_hatBBuf.get());
    const VkDescriptorBufferInfo srInBi = bufInfo(m_srInBuf.get());

    auto writeAffinitySet = [&](
        VkDescriptorSet dstSet,
        const VkDescriptorBufferInfo& in0Bi,
        const VkDescriptorBufferInfo& in1Bi,
        const VkDescriptorBufferInfo& khBi,
        const VkDescriptorBufferInfo& outBi,
        const VkDescriptorBufferInfo& srOutBi,
        int currSlot,
        bool bindLiveImages,
        const VkDescriptorBufferInfo& packTargetBi
    ) {
        std::array<VkDescriptorBufferInfo, 5> headBi = { in0Bi, in1Bi, khBi, outBi, srOutBi };
        const int prevSlot = 1 - currSlot;

        std::array<VkDescriptorImageInfo, 13> ii{};
        if (bindLiveImages) {
            ii[0]  = { VK_NULL_HANDLE, irrView, VK_IMAGE_LAYOUT_GENERAL };
            ii[1]  = { VK_NULL_HANDLE, primGuideView, VK_IMAGE_LAYOUT_GENERAL };
            ii[2]  = { VK_NULL_HANDLE, surfaceMvView, VK_IMAGE_LAYOUT_GENERAL };
            ii[3]  = { VK_NULL_HANDLE, specVmvView, VK_IMAGE_LAYOUT_GENERAL };
            ii[4]  = { VK_NULL_HANDLE, skipCompactView, VK_IMAGE_LAYOUT_GENERAL };
            ii[5]  = { VK_NULL_HANDLE, m_diffHist[prevSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            ii[6]  = { VK_NULL_HANDLE, m_specHist[prevSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            ii[7]  = { VK_NULL_HANDLE, m_hiddenHist[prevSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            ii[8]  = { VK_NULL_HANDLE, m_diffHist[currSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            ii[9]  = { VK_NULL_HANDLE, m_specHist[currSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            ii[10] = { VK_NULL_HANDLE, m_hiddenHist[currSlot]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            ii[11] = { VK_NULL_HANDLE, m_cleanRadLr->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            ii[12] = { VK_NULL_HANDLE, m_outputImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        } else {
            ii[0]  = { VK_NULL_HANDLE, fallbackRg32ui, VK_IMAGE_LAYOUT_GENERAL };
            ii[1]  = { VK_NULL_HANDLE, fallbackRgba32ui, VK_IMAGE_LAYOUT_GENERAL };
            ii[2]  = { VK_NULL_HANDLE, fallbackR32ui, VK_IMAGE_LAYOUT_GENERAL };
            ii[3]  = { VK_NULL_HANDLE, fallbackRg32ui, VK_IMAGE_LAYOUT_GENERAL };
            ii[4]  = { VK_NULL_HANDLE, fallbackR32ui, VK_IMAGE_LAYOUT_GENERAL };
            ii[5]  = { VK_NULL_HANDLE, fallbackRgba16f, VK_IMAGE_LAYOUT_GENERAL };
            ii[6]  = { VK_NULL_HANDLE, fallbackRgba16f, VK_IMAGE_LAYOUT_GENERAL };
            ii[7]  = { VK_NULL_HANDLE, fallbackRg16f, VK_IMAGE_LAYOUT_GENERAL };
            ii[8]  = { VK_NULL_HANDLE, fallbackRgba16f, VK_IMAGE_LAYOUT_GENERAL };
            ii[9]  = { VK_NULL_HANDLE, fallbackRgba16f, VK_IMAGE_LAYOUT_GENERAL };
            ii[10] = { VK_NULL_HANDLE, fallbackRg16f, VK_IMAGE_LAYOUT_GENERAL };
            ii[11] = { VK_NULL_HANDLE, fallbackRgba16f, VK_IMAGE_LAYOUT_GENERAL };
            ii[12] = { VK_NULL_HANDLE, fallbackRgba16f, VK_IMAGE_LAYOUT_GENERAL };
        }

        std::array<VkWriteDescriptorSet, 19> writes{};
        for (uint32_t b = 0; b < 5u; ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = dstSet;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1u;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &headBi[b];
        }
        for (uint32_t b = 0; b < 13u; ++b) {
            writes[5u + b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[5u + b].dstSet = dstSet;
            writes[5u + b].dstBinding = 5u + b;
            writes[5u + b].descriptorCount = 1u;
            writes[5u + b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            writes[5u + b].pImageInfo = &ii[b];
        }
        writes[18].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[18].dstSet = dstSet;
        writes[18].dstBinding = 18u;
        writes[18].descriptorCount = 1u;
        writes[18].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[18].pBufferInfo = &packTargetBi;
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0u, nullptr);
    };

    for (int curr = 0; curr < 2; ++curr) {
        writeAffinitySet(m_affStage0DescSets[curr], dummyBi, dummyBi, denL0Bi, pyr0Bi, dummyBi, curr, true, packBi[0]);
        for (int p = 0; p < 2; ++p) {
            writeAffinitySet(m_affL0DescSets[curr][p], pyr0Bi, hatABi, denL0Bi, hatBBi, srInBi, curr, true, packBi[p]);
        }
    }
    writeAffinitySet(m_affPoolDescSets[0], pyr0Bi, dummyBi, k1Bi, pyr1Bi, dummyBi, 0, false, packBi[0]);
    writeAffinitySet(m_affPoolDescSets[1], pyr1Bi, dummyBi, k2Bi, pyr2Bi, dummyBi, 0, false, packBi[0]);
    writeAffinitySet(m_affPoolDescSets[2], pyr2Bi, dummyBi, k3Bi, pyr3Bi, dummyBi, 0, false, packBi[0]);

    writeAffinitySet(m_affReconDescSets[0], pyr3Bi, dummyBi, k3Bi, hatABi, dummyBi, 0, false, packBi[0]);
    writeAffinitySet(m_affReconDescSets[1], pyr2Bi, hatABi,  k2Bi, hatBBi, dummyBi, 0, false, packBi[0]);
    writeAffinitySet(m_affReconDescSets[2], pyr1Bi, hatBBi,  k1Bi, hatABi, dummyBi, 0, false, packBi[0]);

    // 4. Pass 3: upways_v5_upscaler_wmma.comp (3 stages: srIn -> unetA -> unetB -> srField)
    const VkDescriptorBufferInfo srFieldBi = bufInfo(m_srFieldBuf.get());
    const std::array<std::array<VkDescriptorBufferInfo, 2>, NUM_UPSCALER_STAGES> upBindings = {{
        { srInBi,  unetABi }, // 0 SR_PW1
        { unetABi, unetBBi }, // 1 SR_DW1_PW2
        { unetBBi, srFieldBi }, // 2 SR_DW2_PW3_PW4_HEAD
    }};
    for (uint32_t s = 0; s < NUM_UPSCALER_STAGES; ++s) {
        std::array<VkDescriptorBufferInfo, 4> bi = { w8Bi, w32Bi, upBindings[s][0], upBindings[s][1] };
        std::array<VkWriteDescriptorSet, 4> writes{};
        for (uint32_t b = 0; b < 4u; ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = m_upscalerDescSets[s];
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1u;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &bi[b];
        }
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0u, nullptr);
    }

    // 5. Pass 4: upways_v5_resolve.comp (0..2 SSBOs, 3..8 storage images, 9 SSBO BandPackOut)
    for (int curr = 0; curr < 2; ++curr) {
        const int prev = 1 - curr;
        std::array<VkDescriptorImageInfo, 6> ii{};
        ii[0] = { VK_NULL_HANDLE, m_cleanRadLr->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        ii[1] = { VK_NULL_HANDLE, primGuideView, VK_IMAGE_LAYOUT_GENERAL };
        ii[2] = { VK_NULL_HANDLE, surfaceMvView, VK_IMAGE_LAYOUT_GENERAL };
        ii[3] = { VK_NULL_HANDLE, m_colorHr[prev]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        ii[4] = { VK_NULL_HANDLE, m_colorHr[curr]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
        ii[5] = { VK_NULL_HANDLE, m_outputImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

        for (int p = 0; p < 2; ++p) {
            std::array<VkWriteDescriptorSet, 10> writes{};
            std::array<VkDescriptorBufferInfo, 3> headBi = { w8Bi, w32Bi, srFieldBi };
            for (uint32_t b = 0; b < 3u; ++b) {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = m_resolveDescSets[curr][p];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1u;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[b].pBufferInfo = &headBi[b];
            }
            for (uint32_t b = 0; b < 6u; ++b) {
                writes[3u + b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[3u + b].dstSet = m_resolveDescSets[curr][p];
                writes[3u + b].dstBinding = 3u + b;
                writes[3u + b].descriptorCount = 1u;
                writes[3u + b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[3u + b].pImageInfo = &ii[b];
            }
            writes[9].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[9].dstSet = m_resolveDescSets[curr][p];
            writes[9].dstBinding = 9u;
            writes[9].descriptorCount = 1u;
            writes[9].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[9].pBufferInfo = &packBi[p];
            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0u, nullptr);
        }
    }
}

VkShaderModule UpwaysV5Engine::createShaderModule(const std::vector<char>& code) {
    if (code.empty() || (code.size() % 4u) != 0u) {
        throw std::runtime_error("UpwaysV5Engine: Invalid SPIR-V byte array provided to createShaderModule!");
    }
    VkShaderModuleCreateInfo createInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    createInfo.codeSize = code.size();
    createInfo.pCode = reinterpret_cast<const uint32_t*>(code.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(m_device, &createInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysV5Engine: Failed to create shader module!");
    }
    return shaderModule;
}

void UpwaysV5Engine::createPipelines(
    const std::vector<char>& prepSpv,
    const std::vector<char>& unetWmmaSpv,
    const std::vector<char>& affinitySpv,
    const std::vector<char>& upscalerWmmaSpv,
    const std::vector<char>& resolveSpv
) {
    auto createPipeLayout = [this](VkDescriptorSetLayout descLayout) {
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0u;
        pcRange.size = sizeof(UpwaysV5PushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1u;
        layoutInfo.pSetLayouts = &descLayout;
        layoutInfo.pushConstantRangeCount = 1u;
        layoutInfo.pPushConstantRanges = &pcRange;

        VkPipelineLayout outLayout = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &outLayout) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysV5Engine: Failed to create compute pipeline layout!");
        }
        return outLayout;
    };

    const auto tCompileStart = std::chrono::steady_clock::now();

    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(m_physDevice, &props);

    std::filesystem::path cacheDir;
    if (const char* envDir = std::getenv("PATHWAYS_PIPELINE_CACHE_DIR"); envDir && envDir[0] != '\0') {
        cacheDir = envDir;
    } else {
#ifdef _WIN32
        if (const char* localAppData = std::getenv("LOCALAPPDATA"); localAppData && localAppData[0] != '\0') {
            cacheDir = std::filesystem::path(localAppData) / "pathways";
        }
#else
        if (const char* xdgCache = std::getenv("XDG_CACHE_HOME"); xdgCache && xdgCache[0] != '\0') {
            cacheDir = std::filesystem::path(xdgCache) / "pathways";
        } else if (const char* home = std::getenv("HOME"); home && home[0] != '\0') {
            cacheDir = std::filesystem::path(home) / ".cache" / "pathways";
        }
#endif
    }

    std::filesystem::path cachePath;
    if (!cacheDir.empty()) {
        std::error_code ec;
        std::filesystem::create_directories(cacheDir, ec);
        if (!ec) {
            cachePath = cacheDir / std::format(
                "upways_v5_vk_pipeline_cache_{:04x}_{:04x}_{}.bin",
                props.vendorID,
                props.deviceID,
                upwaysPrecisionName(m_precision)
            );
        }
    }

    std::vector<uint8_t> initialCacheData;
    if (!cachePath.empty()) {
        std::ifstream f(cachePath, std::ios::binary | std::ios::ate);
        if (f.is_open()) {
            const std::streamsize sz = f.tellg();
            if (sz > 0) {
                initialCacheData.resize(static_cast<size_t>(sz));
                f.seekg(0, std::ios::beg);
                if (!f.read(reinterpret_cast<char*>(initialCacheData.data()), sz)) {
                    initialCacheData.clear();
                }
            }
        }
    }

    VkPipelineCacheCreateInfo cacheInfo{ VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO };
    cacheInfo.initialDataSize = initialCacheData.size();
    cacheInfo.pInitialData = initialCacheData.empty() ? nullptr : initialCacheData.data();
    if (vkCreatePipelineCache(m_device, &cacheInfo, nullptr, &m_pipelineCache) != VK_SUCCESS) {
        m_pipelineCache = VK_NULL_HANDLE;
        Logger::warn("UpwaysV5Engine: Failed to create VkPipelineCache; proceeding with uncached compilation.");
    }

    m_prepPipeLayout = createPipeLayout(m_prepDescLayout);
    m_unetPipeLayout = createPipeLayout(m_unetDescLayout);
    m_affinityPipeLayout = createPipeLayout(m_affinityDescLayout);
    m_upscalerPipeLayout = createPipeLayout(m_upscalerDescLayout);
    m_resolvePipeLayout = createPipeLayout(m_resolveDescLayout);

    VkShaderModule prepMod = createShaderModule(prepSpv);
    VkShaderModule unetMod = createShaderModule(unetWmmaSpv);
    VkShaderModule affMod  = createShaderModule(affinitySpv);
    VkShaderModule upMod   = createShaderModule(upscalerWmmaSpv);
    VkShaderModule resMod  = createShaderModule(resolveSpv);

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo subgroup32Info{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO
    };
    subgroup32Info.requiredSubgroupSize = 32u;

    auto createComputePipe = [this, &subgroup32Info](
        VkShaderModule mod,
        VkPipelineLayout layout,
        bool forceWave32,
        bool requireFullSubgroups,
        const VkSpecializationInfo* specInfo
    ) {
        VkPipelineShaderStageCreateInfo stageInfo{
            VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
        };
        stageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        stageInfo.module = mod;
        stageInfo.pName = "main";
        stageInfo.pSpecializationInfo = specInfo;

        if (forceWave32 && m_hasSubgroupSizeControl) {
            stageInfo.pNext = &subgroup32Info;
            if (requireFullSubgroups && m_hasComputeFullSubgroups) {
                stageInfo.flags |= VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
            }
        }

        VkComputePipelineCreateInfo pipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
        pipeInfo.stage = stageInfo;
        pipeInfo.layout = layout;

        VkPipeline outPipe = VK_NULL_HANDLE;
        if (vkCreateComputePipelines(m_device, m_pipelineCache, 1u, &pipeInfo, nullptr, &outPipe) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysV5Engine: Failed to create compute pipeline!");
        }
        return outPipe;
    };

    auto makeAffPipe = [&](uint32_t stageId, uint32_t dil) {
        const std::array<uint32_t, 2> vals = { stageId, dil };
        const std::array<VkSpecializationMapEntry, 2> entries = {{
            { 0u, 0u,               sizeof(uint32_t) },
            { 1u, sizeof(uint32_t), sizeof(uint32_t) }
        }};
        const VkSpecializationInfo specInfo{
            static_cast<uint32_t>(entries.size()), entries.data(),
            sizeof(vals), vals.data()
        };
        return createComputePipe(affMod, m_affinityPipeLayout, true, false, &specInfo);
    };

    try {
        m_prepPipeline = createComputePipe(prepMod, m_prepPipeLayout, true, false, nullptr);

        for (uint32_t s = 0; s < NUM_UNET_STAGES; ++s) {
            const uint32_t stageId = s;
            const VkSpecializationMapEntry entry{ 0u, 0u, sizeof(uint32_t) };
            const VkSpecializationInfo specInfo{ 1u, &entry, sizeof(uint32_t), &stageId };
            m_unetPipelines[s] = createComputePipe(unetMod, m_unetPipeLayout, true, true, &specInfo);
        }

        m_affStage0Pipeline    = makeAffPipe(0u, 1u);
        m_affPoolPipeline      = makeAffPipe(1u, 1u);
        m_affReconDil3Pipeline = makeAffPipe(2u, 3u);
        m_affReconDil1Pipeline = makeAffPipe(2u, 1u);
        m_affStage3Pipeline    = makeAffPipe(3u, 1u);

        for (uint32_t s = 0; s < NUM_UPSCALER_STAGES; ++s) {
            const uint32_t stageId = s;
            const VkSpecializationMapEntry entry{ 0u, 0u, sizeof(uint32_t) };
            const VkSpecializationInfo specInfo{ 1u, &entry, sizeof(uint32_t), &stageId };
            m_upscalerPipelines[s] = createComputePipe(upMod, m_upscalerPipeLayout, true, true, &specInfo);
        }

        m_resolvePipeline = createComputePipe(resMod, m_resolvePipeLayout, true, true, nullptr);
    } catch (...) {
        vkDestroyShaderModule(m_device, prepMod, nullptr);
        vkDestroyShaderModule(m_device, unetMod, nullptr);
        vkDestroyShaderModule(m_device, affMod, nullptr);
        vkDestroyShaderModule(m_device, upMod, nullptr);
        vkDestroyShaderModule(m_device, resMod, nullptr);
        throw;
    }

    vkDestroyShaderModule(m_device, prepMod, nullptr);
    vkDestroyShaderModule(m_device, unetMod, nullptr);
    vkDestroyShaderModule(m_device, affMod, nullptr);
    vkDestroyShaderModule(m_device, upMod, nullptr);
    vkDestroyShaderModule(m_device, resMod, nullptr);

    if (m_pipelineCache != VK_NULL_HANDLE && !cachePath.empty()) {
        size_t cacheBytes = 0u;
        if (vkGetPipelineCacheData(m_device, m_pipelineCache, &cacheBytes, nullptr) == VK_SUCCESS && cacheBytes > 0u) {
            std::vector<uint8_t> outCache(cacheBytes);
            if (vkGetPipelineCacheData(m_device, m_pipelineCache, &cacheBytes, outCache.data()) == VK_SUCCESS) {
                std::filesystem::path tmpPath = cacheDir / std::format(
                    "upways_v5_vk_pipeline_cache_{:04x}_{:04x}.tmp.{}",
                    props.vendorID,
                    props.deviceID,
                    static_cast<unsigned long>(std::chrono::steady_clock::now().time_since_epoch().count())
                );
                std::ofstream out(tmpPath, std::ios::binary | std::ios::trunc);
                if (out.is_open()) {
                    out.write(reinterpret_cast<const char*>(outCache.data()), static_cast<std::streamsize>(cacheBytes));
                    out.close();
                    std::error_code ec;
                    std::filesystem::rename(tmpPath, cachePath, ec);
                    if (ec) {
                        std::filesystem::remove(tmpPath, ec);
                    }
                }
            }
        }
    }

    const double compileMs = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - tCompileStart
    ).count();
    Logger::info(
        "UpwaysPipeline 5.0 compiled 22 compute pipelines in {:.1f} ms (disk cache: {}).",
        compileMs,
        cachePath.empty() ? "none" : cachePath.filename().string()
    );
}

void UpwaysV5Engine::resize(
    uint32_t inputWidth,
    uint32_t inputHeight,
    uint32_t outputWidth,
    uint32_t outputHeight,
    bool enableSuperRes,
    uint32_t outputFullHeight
) {
    const uint32_t newInW = std::max(1u, inputWidth);
    const uint32_t newInH = std::max(1u, inputHeight);
    const uint32_t newOutW = std::max(1u, outputWidth);
    const uint32_t newOutH = std::max(1u, outputHeight);
    const uint32_t newFullOutH = (outputFullHeight > 0u) ? std::max(outputFullHeight, newOutH) : newOutH;

    const bool renderChanged = (newInW != m_inputWidth || newInH != m_inputHeight);
    const bool displayChanged = (newOutW != m_outputWidth || newOutH != m_outputHeight || newFullOutH != m_outputFullHeight);
    m_enableSuperRes = enableSuperRes;

    if (!renderChanged && !displayChanged) {
        return;
    }

    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    if (renderChanged) {
        m_cachedMlIrradianceView = VK_NULL_HANDLE;
        m_cachedMlPrimaryGuideView = VK_NULL_HANDLE;
        m_cachedMlSurfaceMvView = VK_NULL_HANDLE;
        m_cachedMlDiffHitDistView = VK_NULL_HANDLE;
        m_cachedMlSpecVmvView = VK_NULL_HANDLE;
        m_cachedMlSkipCompactView = VK_NULL_HANDLE;
    }

    m_inputWidth = newInW;
    m_inputHeight = newInH;
    m_outputWidth = newOutW;
    m_outputHeight = newOutH;
    m_outputFullHeight = newFullOutH;

    if (!displayChanged && renderChanged && m_historyValid) {
        const uint32_t curr = m_pingPongIndex & 1u;
        initRenderResources(1u << curr);
        refreshDescriptors();
        Logger::info(
            "UpwaysPipeline 5.0 lossless DRS resize: LR {}x{} (L1 {}x{}, L2 {}x{}, L3 {}x{}), HR {}x{} preserved (SuperRes: {}).",
            m_inputWidth, m_inputHeight,
            m_dims.w1, m_dims.h1,
            m_dims.w2, m_dims.h2,
            m_dims.w3, m_dims.h3,
            m_outputWidth, m_outputHeight,
            m_enableSuperRes ? "ON" : "OFF"
        );
        return;
    }

    initRenderResources(0x3u);
    if (displayChanged) {
        initDisplayResources();
    } else {
        m_historyValid = false;
    }
    refreshDescriptors();

    Logger::info(
        "UpwaysPipeline 5.0 resized: LR {}x{} (L1 {}x{}, L2 {}x{}, L3 {}x{}), HR {}x{} (SuperRes: {}).",
        m_inputWidth, m_inputHeight,
        m_dims.w1, m_dims.h1,
        m_dims.w2, m_dims.h2,
        m_dims.w3, m_dims.h3,
        m_outputWidth, m_outputHeight,
        m_enableSuperRes ? "ON" : "OFF"
    );
}

void UpwaysV5Engine::collectQueryResults(uint32_t targetSlot, bool waitForResult) {
    if (m_queryPool == VK_NULL_HANDLE || targetSlot >= MAX_QUERY_SLOTS) {
        return;
    }
    if (!m_querySlotSubmitted[targetSlot]) {
        return;
    }

    std::array<uint64_t, QUERIES_PER_SLOT> timestamps{};
    const uint32_t firstQuery = targetSlot * QUERIES_PER_SLOT;
    const VkQueryResultFlags flags = waitForResult
        ? (VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT)
        : VK_QUERY_RESULT_64_BIT;

    const VkResult res = vkGetQueryPoolResults(
        m_device,
        m_queryPool,
        firstQuery,
        QUERIES_PER_SLOT,
        sizeof(timestamps),
        timestamps.data(),
        sizeof(uint64_t),
        flags
    );
    if (res != VK_SUCCESS) {
        return;
    }

    m_querySlotSubmitted[targetSlot] = false;
    const double toMs = static_cast<double>(m_timestampPeriodNs) * 1e-6;
    const uint64_t t0 = timestamps[0];
    const uint64_t t1 = timestamps[1];
    const uint64_t t2 = timestamps[2];
    const uint64_t t3 = timestamps[3];
    const uint64_t t4 = timestamps[4];

    m_lastTimings.prepMs     = (t1 > t0) ? static_cast<double>(t1 - t0) * toMs : 0.0;
    m_lastTimings.denoiserMs = (t2 > t1) ? static_cast<double>(t2 - t1) * toMs : 0.0;
    m_lastTimings.upscalerMs = (t3 > t2) ? static_cast<double>(t3 - t2) * toMs : 0.0;
    m_lastTimings.resolveMs  = (t4 > t3) ? static_cast<double>(t4 - t3) * toMs : 0.0;
    m_lastTimings.totalMs    = (t4 > t0) ? static_cast<double>(t4 - t0) * toMs : 0.0;
    m_lastTimings.valid      = true;
}

void UpwaysV5Engine::recordFrame(
    VkCommandBuffer cmd,
    uint32_t frameIndex,
    bool resetHistory,
    glm::vec2 jitter,
    bool coupledWithExternalUpscaler,
    bool zeroResidualMode,
    float exposureScale,
    const UpwaysBandParams& band
) {
    (void)exposureScale;
    if (!isInitialized()) {
        return;
    }

    collectQueryResults(m_querySlot, true);
    collectQueryResults(1u - m_querySlot, false);

    ensureActiveSlotSized();
    transitionInitialLayouts(cmd);

    const uint32_t curr = m_pingPongIndex & 1u;
    const bool doReset = resetHistory || !m_historyValid;
    const bool runSuperRes =
        (m_enableSuperRes || m_inputWidth != m_outputWidth || m_inputHeight != m_outputHeight) &&
        !coupledWithExternalUpscaler;

    UpwaysV5PushConstants pc{};
    pc.lrWidth = m_inputWidth;
    pc.lrHeight = m_inputHeight;
    pc.hrWidth = m_outputWidth;
    pc.hrHeight = m_outputHeight;
    pc.jitterX = jitter.x;
    pc.jitterY = jitter.y;
    pc.frameIndex = frameIndex;
    pc.flags = 0u;
    if (doReset) {
        pc.flags |= UPWAYS_V5_FLAG_RESET_HISTORY;
    }
    if (runSuperRes) {
        pc.flags |= UPWAYS_V5_FLAG_ENABLE_SUPERRES;
    }
    if (zeroResidualMode) {
        pc.flags |= UPWAYS_V5_FLAG_ZERO_RESIDUAL_MODE;
    }
    const bool useFusedPack = band.fusedBandPack;
    const uint32_t packSlot = useFusedPack ? (band.bandPackSlot & 1u) : 0u;
    if (useFusedPack) {
        pc.flags |= UPWAYS_V5_FLAG_FUSED_BAND_PACK;
    }
    pc.bandLrY0 = band.bandLrY0;
    pc.bandHrY0 = band.bandHrY0;
    pc.fullLrHeight = (band.fullLrHeight > 0u) ? band.fullLrHeight : m_inputHeight;
    pc.fullHrHeight = (band.fullHrHeight > 0u) ? band.fullHrHeight : m_outputHeight;
    pc.keepHrYStart = band.keepHrYStart;
    pc.keepHrYEnd = (band.keepHrYEnd > 0u) ? band.keepHrYEnd : m_outputHeight;

    const uint32_t qSlot = m_querySlot;
    const uint32_t qBase = qSlot * QUERIES_PER_SLOT;
    if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, m_queryPool, qBase, QUERIES_PER_SLOT);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 0u);
    }

    auto dispatchStage = [&](
        VkPipeline pipe,
        VkPipelineLayout layout,
        VkDescriptorSet set,
        uint32_t inW, uint32_t inH,
        uint32_t outW, uint32_t outH
    ) {
        pc.inW = inW;
        pc.inH = inH;
        pc.outW = outW;
        pc.outH = outH;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, layout, 0u, 1u, &set, 0u, nullptr);
        vkCmdPushConstants(cmd, layout, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(UpwaysV5PushConstants), &pc);
        vkCmdDispatch(cmd, divCeil(outW, 16u), divCeil(outH, 8u), 1u);
        emitComputeBarrier(cmd);
    };

    // Pass 1: upways_v5_prep.comp @ L0 (writes X1 @ L1)
    {
        pc.inW = m_dims.w0;
        pc.inH = m_dims.h0;
        pc.outW = m_dims.w1;
        pc.outH = m_dims.h1;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_prepPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_prepPipeLayout, 0u, 1u, &m_prepDescSets[curr], 0u, nullptr);
        vkCmdPushConstants(cmd, m_prepPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(UpwaysV5PushConstants), &pc);
        vkCmdDispatch(cmd, divCeil(m_dims.w0, 16u), divCeil(m_dims.h0, 8u), 1u);
        emitComputeBarrier(cmd);
    }

    if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 1u);
    }

    // Pass 2A: upways_v5_unet_wmma.comp (12 Wave32 WMMA stages)
    struct StageShape { uint32_t inW, inH, outW, outH; };
    const std::array<StageShape, NUM_UNET_STAGES> unetShapes = {{
        { m_dims.w1, m_dims.h1, m_dims.w1, m_dims.h1 }, // 0  ENC1_PW1
        { m_dims.w1, m_dims.h1, m_dims.w1, m_dims.h1 }, // 1  ENC1_DW_PW2
        { m_dims.w1, m_dims.h1, m_dims.w2, m_dims.h2 }, // 2  ENC2_POOL_PW1
        { m_dims.w2, m_dims.h2, m_dims.w2, m_dims.h2 }, // 3  ENC2_DW_PW2
        { m_dims.w2, m_dims.h2, m_dims.w3, m_dims.h3 }, // 4  BOT3_POOL_PW1
        { m_dims.w3, m_dims.h3, m_dims.w3, m_dims.h3 }, // 5  BOT3_DW1_PW2
        { m_dims.w3, m_dims.h3, m_dims.w3, m_dims.h3 }, // 6  BOT3_DW2_PW3
        { m_dims.w3, m_dims.h3, m_dims.w3, m_dims.h3 }, // 7  BOT3_PW4_KHEAD
        { m_dims.w3, m_dims.h3, m_dims.w2, m_dims.h2 }, // 8  DEC2_UP_CAT_PW1
        { m_dims.w2, m_dims.h2, m_dims.w2, m_dims.h2 }, // 9  DEC2_DW_PW2_KH
        { m_dims.w2, m_dims.h2, m_dims.w1, m_dims.h1 }, // 10 DEC1_UP_CAT_PW1
        { m_dims.w1, m_dims.h1, m_dims.w1, m_dims.h1 }, // 11 DEC1_DW_PW2_HD
    }};
    for (uint32_t s = 0; s < NUM_UNET_STAGES; ++s) {
        dispatchStage(
            m_unetPipelines[s], m_unetPipeLayout, m_unetDescSets[s],
            unetShapes[s].inW, unetShapes[s].inH, unetShapes[s].outW, unetShapes[s].outH
        );
    }

    // Pass 2B: upways_v5_affinity.comp (8 multi-scale affinity passes)
    dispatchStage(m_affStage0Pipeline,    m_affinityPipeLayout, m_affStage0DescSets[curr],        m_dims.w0, m_dims.h0, m_dims.w0, m_dims.h0);
    dispatchStage(m_affPoolPipeline,      m_affinityPipeLayout, m_affPoolDescSets[0],             m_dims.w0, m_dims.h0, m_dims.w1, m_dims.h1);
    dispatchStage(m_affPoolPipeline,      m_affinityPipeLayout, m_affPoolDescSets[1],             m_dims.w1, m_dims.h1, m_dims.w2, m_dims.h2);
    dispatchStage(m_affPoolPipeline,      m_affinityPipeLayout, m_affPoolDescSets[2],             m_dims.w2, m_dims.h2, m_dims.w3, m_dims.h3);
    dispatchStage(m_affReconDil3Pipeline, m_affinityPipeLayout, m_affReconDescSets[0],            m_dims.w3, m_dims.h3, m_dims.w3, m_dims.h3);
    dispatchStage(m_affReconDil1Pipeline, m_affinityPipeLayout, m_affReconDescSets[1],            m_dims.w3, m_dims.h3, m_dims.w2, m_dims.h2);
    dispatchStage(m_affReconDil1Pipeline, m_affinityPipeLayout, m_affReconDescSets[2],            m_dims.w2, m_dims.h2, m_dims.w1, m_dims.h1);
    dispatchStage(m_affStage3Pipeline,    m_affinityPipeLayout, m_affL0DescSets[curr][packSlot],  m_dims.w1, m_dims.h1, m_dims.w0, m_dims.h0);

    if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 2u);
    }

    if (runSuperRes) {
        // Pass 3: upways_v5_upscaler_wmma.comp (3 Wave32 stages @ L1)
        for (uint32_t s = 0; s < NUM_UPSCALER_STAGES; ++s) {
            dispatchStage(
                m_upscalerPipelines[s], m_upscalerPipeLayout, m_upscalerDescSets[s],
                m_dims.w1, m_dims.h1, m_dims.w1, m_dims.h1
            );
        }

        if (m_queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 3u);
        }

        // Pass 4: upways_v5_resolve.comp (16x8 HR tile, Wave32 cooperative MLP + warp + 9-tap recon)
        dispatchStage(
            m_resolvePipeline, m_resolvePipeLayout, m_resolveDescSets[curr][packSlot],
            m_dims.w0, m_dims.h0, m_outputWidth, m_outputHeight
        );

        if (m_queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 4u);
        }
    } else if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 3u);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 4u);
    }

    emitComputeBarrier(
        cmd,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
        VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
    );

    if (m_queryPool != VK_NULL_HANDLE) {
        m_querySlotSubmitted[qSlot] = true;
        m_querySlot = (qSlot + 1u) % MAX_QUERY_SLOTS;
    }

    m_pingPongIndex = 1u - curr;
    m_historyValid = true;
}

} // namespace pathways
