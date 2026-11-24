#include "UpwaysPipeline.hpp"

#include "core/Logger.hpp"
#include "rt/upways_default_int8_weights.hpp"
#include "rt/upways_default_weights.hpp"
#include "rt/upways_weights.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"

#include <glm/gtc/packing.hpp>

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
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace pathways {

namespace {

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
    dep.memoryBarrierCount = 1;
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

struct CoopMatrixSupport {
    bool fp8 = false;
    bool int8 = false;
    bool fp16 = false;
};

float decodeE4M3ToFloat(uint8_t b) noexcept {
    const uint32_t u = static_cast<uint32_t>(b);
    const float s = ((u & 0x80u) != 0u) ? -1.0f : 1.0f;
    const uint32_t e = (u >> 3u) & 0xFu;
    const uint32_t m = u & 0x7u;
    if (e == 0u) {
        return s * (static_cast<float>(m) / 512.0f);
    }
    if (e == 15u && m == 7u) {
        return 0.0f;
    }
    return s * std::exp2(static_cast<float>(static_cast<int32_t>(e) - 7))
             * (1.0f + static_cast<float>(m) / 8.0f);
}

std::vector<uint8_t> expandFp8WeightsToFp16(const std::vector<uint8_t>& blob24k) {
    if (blob24k.size() != upways::TOTAL_WEIGHT_BYTES) {
        throw std::runtime_error("UpwaysPipeline: Expected 24,064-byte weights blob for FP16 expansion");
    }
    constexpr size_t kFp16WeightBytes = upways::FP8_WEIGHT_SECTION_BYTES * 2u;
    std::vector<uint8_t> expanded(kFp16WeightBytes + upways::FP32_WEIGHT_SECTION_BYTES);
    for (size_t i = 0; i < upways::FP8_WEIGHT_SECTION_BYTES; ++i) {
        const uint16_t h = glm::packHalf1x16(decodeE4M3ToFloat(blob24k[i]));
        std::memcpy(expanded.data() + i * sizeof(uint16_t), &h, sizeof(uint16_t));
    }
    std::memcpy(
        expanded.data() + kFp16WeightBytes,
        blob24k.data() + upways::FP8_WEIGHT_SECTION_OFFSET + upways::FP8_WEIGHT_SECTION_BYTES,
        upways::FP32_WEIGHT_SECTION_BYTES
    );
    return expanded;
}

CoopMatrixSupport queryCoopMatrixSupport(VkPhysicalDevice physicalDevice, VkInstance instance) {
    CoopMatrixSupport sup{};
    if (physicalDevice == VK_NULL_HANDLE) {
        return sup;
    }

    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> exts(extCount);
    if (extCount > 0) {
        vkEnumerateDeviceExtensionProperties(physicalDevice, nullptr, &extCount, exts.data());
    }
    auto hasExt = [&](const char* name) {
        for (const auto& ext : exts) {
            if (std::strcmp(ext.extensionName, name) == 0) {
                return true;
            }
        }
        return false;
    };
    const bool hasCoopExt = hasExt(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
#ifdef VK_EXT_shader_float8
    const bool hasFloat8Ext = hasExt(VK_EXT_SHADER_FLOAT8_EXTENSION_NAME);
#else
    const bool hasFloat8Ext = false;
#endif
    if (!hasCoopExt) {
        return sup;
    }

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR
    };
    VkPhysicalDeviceVulkan13Features vk13Features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES
    };
    VkPhysicalDeviceVulkan12Features vk12Features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES
    };
    VkPhysicalDeviceVulkan11Features vk11Features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES
    };

    vk11Features.pNext = &vk12Features;
    vk12Features.pNext = &vk13Features;
    vk13Features.pNext = &coopFeatures;
#ifdef VK_EXT_shader_float8
    VkPhysicalDeviceShaderFloat8FeaturesEXT float8Features{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT
    };
    coopFeatures.pNext = &float8Features;
#endif

    VkPhysicalDeviceFeatures2 features2{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2 };
    features2.pNext = &vk11Features;
    vkGetPhysicalDeviceFeatures2(physicalDevice, &features2);

    if (vk13Features.subgroupSizeControl != VK_TRUE || coopFeatures.cooperativeMatrix != VK_TRUE) {
        return sup;
    }

#ifdef VK_EXT_shader_float8
    const bool featFp8 = hasFloat8Ext &&
                         (float8Features.shaderFloat8 == VK_TRUE) &&
                         (float8Features.shaderFloat8CooperativeMatrix == VK_TRUE);
#else
    const bool featFp8 = false;
    (void)hasFloat8Ext;
#endif
    const bool featInt8 = (vk12Features.shaderInt8 == VK_TRUE) &&
                          (vk12Features.storageBuffer8BitAccess == VK_TRUE) &&
                          (vk12Features.shaderFloat16 == VK_TRUE);
    const bool featFp16 = (vk12Features.shaderFloat16 == VK_TRUE) &&
                          (vk11Features.storageBuffer16BitAccess == VK_TRUE);

    bool propFp8 = featFp8;
    bool propInt8 = featInt8;
    bool propFp16 = featFp16;

    if (instance != VK_NULL_HANDLE) {
        auto pfnGetCoopProps = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
            vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
        if (pfnGetCoopProps != nullptr) {
            uint32_t propCount = 0;
            if (pfnGetCoopProps(physicalDevice, &propCount, nullptr) == VK_SUCCESS && propCount > 0) {
                std::vector<VkCooperativeMatrixPropertiesKHR> coopProps(propCount);
                for (auto& cp : coopProps) {
                    cp = {};
                    cp.sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
                }
                if (pfnGetCoopProps(physicalDevice, &propCount, coopProps.data()) == VK_SUCCESS) {
                    propFp8 = false;
                    propInt8 = false;
                    propFp16 = false;
                    for (const auto& cp : coopProps) {
                        if (cp.scope != VK_SCOPE_SUBGROUP_KHR ||
                            cp.MSize != 16u || cp.NSize != 16u || cp.KSize != 16u) {
                            continue;
                        }
#ifdef VK_EXT_shader_float8
                        if (cp.AType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
                            cp.BType == VK_COMPONENT_TYPE_FLOAT8_E4M3_EXT &&
                            cp.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                            propFp8 = true;
                        }
#endif
                        if (cp.AType == VK_COMPONENT_TYPE_SINT8_KHR &&
                            cp.BType == VK_COMPONENT_TYPE_SINT8_KHR &&
                            cp.ResultType == VK_COMPONENT_TYPE_SINT32_KHR) {
                            propInt8 = true;
                        }
                        if (cp.AType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            cp.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
                            cp.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR) {
                            propFp16 = true;
                        }
                    }
                }
            }
        }
    }

    sup.fp8 = featFp8 && propFp8;
    sup.int8 = featInt8 && propInt8;
    sup.fp16 = featFp16 && propFp16;
    return sup;
}

} // namespace

bool UpwaysPipeline::supportsFp8CoopMatrix(VkPhysicalDevice physicalDevice, VkInstance instance) {
    return queryCoopMatrixSupport(physicalDevice, instance).fp8;
}

bool UpwaysPipeline::supportsInt8CoopMatrix(VkPhysicalDevice physicalDevice, VkInstance instance) {
    return queryCoopMatrixSupport(physicalDevice, instance).int8;
}

bool UpwaysPipeline::supportsFp16CoopMatrix(VkPhysicalDevice physicalDevice, VkInstance instance) {
    return queryCoopMatrixSupport(physicalDevice, instance).fp16;
}

UpwaysComputePrecision UpwaysPipeline::selectPrecision(
    VkPhysicalDevice physicalDevice,
    UpwaysPrecision requested,
    VkInstance instance
) {
    const CoopMatrixSupport sup = queryCoopMatrixSupport(physicalDevice, instance);
    auto pickAuto = [&]() -> UpwaysComputePrecision {
        if (sup.fp8) return UpwaysComputePrecision::Fp8;
        if (sup.int8) return UpwaysComputePrecision::Int8;
        if (sup.fp16) return UpwaysComputePrecision::Fp16;
        throw std::runtime_error(
            "UpwaysPipeline: GPU lacks Wave32 16x16x16 VK_KHR_cooperative_matrix WMMA support for FP8, INT8, and FP16."
        );
    };

    switch (requested) {
        case UpwaysPrecision::Fp8:
            if (sup.fp8) return UpwaysComputePrecision::Fp8;
            Logger::warn("UpwaysPipeline: Requested precision 'fp8' is unsupported on this GPU; falling back automatically.");
            return pickAuto();
        case UpwaysPrecision::Int8:
            if (sup.int8) return UpwaysComputePrecision::Int8;
            Logger::warn("UpwaysPipeline: Requested precision 'int8' is unsupported on this GPU; falling back automatically.");
            return pickAuto();
        case UpwaysPrecision::Fp16:
            if (sup.fp16) return UpwaysComputePrecision::Fp16;
            Logger::warn("UpwaysPipeline: Requested precision 'fp16' is unsupported on this GPU; falling back automatically.");
            return pickAuto();
        case UpwaysPrecision::Auto:
        default:
            return pickAuto();
    }
}

UpwaysPipeline::UpwaysPipeline(
    VkDevice device,
    VkPhysicalDevice physicalDevice,
    VmaAllocator allocator,
    uint32_t inputWidth,
    uint32_t inputHeight,
    uint32_t outputWidth,
    uint32_t outputHeight,
    const std::vector<char>& denoisePrepSpv,
    const std::vector<char>& denoiserWmmaSpv,
    const std::vector<char>& upscalerWmmaSpv,
    const std::vector<char>& upscaleResolveSpv,
    const std::string& weightsPath,
    bool enableSuperRes,
    VkFormat imageFormat,
    UpwaysComputePrecision precision
)
    : m_device(device)
    , m_physDevice(physicalDevice)
    , m_allocator(allocator)
    , m_inputWidth(std::max(1u, inputWidth))
    , m_inputHeight(std::max(1u, inputHeight))
    , m_outputWidth(std::max(1u, outputWidth))
    , m_outputHeight(std::max(1u, outputHeight))
    , m_enableSuperRes(enableSuperRes)
    , m_precision(precision)
    , m_weightQuantBytes((precision == UpwaysComputePrecision::Fp16) ? (upways::FP8_WEIGHT_SECTION_BYTES * 2u) : upways::FP8_WEIGHT_SECTION_BYTES)
    , m_imageFormat(VK_FORMAT_R16G16B16A16_SFLOAT)
    , m_pyramidDims(computeUpwaysPyramidDims(m_inputWidth, m_inputHeight))
{
    if (imageFormat != VK_FORMAT_UNDEFINED && imageFormat != VK_FORMAT_R16G16B16A16_SFLOAT) {
        Logger::info("UpwaysPipeline 4.0: normalizing output format to R16G16B16A16_SFLOAT to match rgba16f shader bindings.");
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
        "UpwaysPipeline 4.0 ({} WMMA) initializing: LR {}x{} -> HR {}x{} (SuperRes: {}, L1 {}x{}, L2 {}x{}).",
        upwaysPrecisionDisplayName(m_precision),
        m_inputWidth, m_inputHeight,
        m_outputWidth, m_outputHeight,
        m_enableSuperRes ? "ON" : "OFF",
        m_pyramidDims.w1, m_pyramidDims.h1,
        m_pyramidDims.w2, m_pyramidDims.h2
    );

    VkQueryPoolCreateInfo qpInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
    qpInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qpInfo.queryCount = MAX_QUERY_SLOTS * QUERIES_PER_SLOT;
    if (vkCreateQueryPool(m_device, &qpInfo, nullptr, &m_queryPool) != VK_SUCCESS) {
        Logger::warn("UpwaysPipeline: Failed to create timestamp query pool; per-pass profiling disabled.");
        m_queryPool = VK_NULL_HANDLE;
    }

    initWeights(weightsPath);
    initRenderResources(0x3u);
    initDisplayResources();
    createDescriptorSetLayouts();
    allocateDescriptorSets();
    refreshDescriptors();
    createPipelines(denoisePrepSpv, denoiserWmmaSpv, upscalerWmmaSpv, upscaleResolveSpv);

    Logger::info(
        "UpwaysPipeline 4.0 initialized ({} active params, {} B uploaded {} weights).",
        upways::ACTIVE_PARAMETER_COUNT,
        m_weightBuffer->getSize(),
        upwaysPrecisionDisplayName(m_precision)
    );
}

UpwaysPipeline::~UpwaysPipeline() {
    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    destroyArrayImage(m_baseAnchorLr);

    if (m_prepPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_prepPipeline, nullptr);
    }
    for (VkPipeline& p : m_denoiserPipelines) {
        if (p != VK_NULL_HANDLE) {
            vkDestroyPipeline(m_device, p, nullptr);
            p = VK_NULL_HANDLE;
        }
    }
    if (m_upscalerPipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_upscalerPipeline, nullptr);
    }
    if (m_resolvePipeline != VK_NULL_HANDLE) {
        vkDestroyPipeline(m_device, m_resolvePipeline, nullptr);
    }

    if (m_prepPipeLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_prepPipeLayout, nullptr);
    }
    if (m_denoiserPipeLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_denoiserPipeLayout, nullptr);
    }
    if (m_upscalerPipeLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_upscalerPipeLayout, nullptr);
    }
    if (m_resolvePipeLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_resolvePipeLayout, nullptr);
    }

    if (m_pipelineCache != VK_NULL_HANDLE) {
        vkDestroyPipelineCache(m_device, m_pipelineCache, nullptr);
    }

    if (m_descriptorPool != VK_NULL_HANDLE) {
        vkDestroyDescriptorPool(m_device, m_descriptorPool, nullptr);
    }
    if (m_prepDescLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_prepDescLayout, nullptr);
    }
    if (m_denoiserDescLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_denoiserDescLayout, nullptr);
    }
    if (m_upscalerDescLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_upscalerDescLayout, nullptr);
    }
    if (m_resolveDescLayout != VK_NULL_HANDLE) {
        vkDestroyDescriptorSetLayout(m_device, m_resolveDescLayout, nullptr);
    }

    if (m_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(m_device, m_queryPool, nullptr);
    }
}

void UpwaysPipeline::initWeights(const std::string& weightsPath) {
    static_assert(sizeof(upways::DEFAULT_UPWAYS_WEIGHTS) == upways::TOTAL_WEIGHT_BYTES,
                  "EMBEDDED DEFAULT_UPWAYS_WEIGHTS size must equal 24,064 B");
    static_assert(sizeof(upways::DEFAULT_UPWAYS_INT8_WEIGHTS) == upways::TOTAL_WEIGHT_BYTES,
                  "EMBEDDED DEFAULT_UPWAYS_INT8_WEIGHTS size must equal 24,064 B");
    static_assert(sizeof(upways::DEFAULT_WEIGHTS) == upways::TOTAL_WEIGHT_BYTES,
                  "Header upways::DEFAULT_WEIGHTS size must equal 24,064 B");

    const bool isInt8 = (m_precision == UpwaysComputePrecision::Int8);
    const char* binName = isInt8 ? "upways_weights_int8.bin" : "upways_weights.bin";

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
        if (sz == static_cast<std::streamsize>(upways::TOTAL_WEIGHT_BYTES)) {
            rawWeights.resize(upways::TOTAL_WEIGHT_BYTES);
            f.seekg(0, std::ios::beg);
            if (f.read(reinterpret_cast<char*>(rawWeights.data()), sz)) {
                loadedFrom = candidate;
                break;
            }
            rawWeights.clear();
        } else {
            Logger::warn(
                "UpwaysPipeline: ignoring '{}' (size {} != expected {} B); using embedded weights.",
                candidate, static_cast<size_t>(sz), upways::TOTAL_WEIGHT_BYTES
            );
        }
    }

    if (rawWeights.empty()) {
        const uint8_t* embedded = isInt8 ? upways::DEFAULT_UPWAYS_INT8_WEIGHTS : upways::DEFAULT_UPWAYS_WEIGHTS;
        rawWeights.assign(embedded, embedded + upways::TOTAL_WEIGHT_BYTES);
        Logger::info(
            "UpwaysPipeline using embedded {} ({} B).",
            isInt8 ? "DEFAULT_UPWAYS_INT8_WEIGHTS" : "DEFAULT_UPWAYS_WEIGHTS",
            rawWeights.size()
        );
    } else {
        Logger::info("UpwaysPipeline loaded {} B {} weights from '{}'.",
                     rawWeights.size(), upwaysPrecisionDisplayName(m_precision), loadedFrom);
    }

    const std::vector<uint8_t> uploadWeights =
        (m_precision == UpwaysComputePrecision::Fp16)
            ? expandFp8WeightsToFp16(rawWeights)
            : std::move(rawWeights);
    m_weightQuantBytes = (m_precision == UpwaysComputePrecision::Fp16)
        ? (upways::FP8_WEIGHT_SECTION_BYTES * 2u)
        : upways::FP8_WEIGHT_SECTION_BYTES;

    m_weightBuffer = std::make_unique<Buffer>(
        m_allocator,
        uploadWeights.size(),
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    m_weightBuffer->copyFrom(uploadWeights.data(), uploadWeights.size());
}

void UpwaysPipeline::createArrayImage(ArrayImage2D& outImg, uint32_t width, uint32_t height, uint32_t layers) {
    destroyArrayImage(outImg);

    VkImageCreateInfo imgInfo{ VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO };
    imgInfo.imageType = VK_IMAGE_TYPE_2D;
    imgInfo.format = VK_FORMAT_R16G16B16A16_SFLOAT;
    imgInfo.extent = { width, height, 1u };
    imgInfo.mipLevels = 1u;
    imgInfo.arrayLayers = layers;
    imgInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imgInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imgInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT |
                    VK_IMAGE_USAGE_SAMPLED_BIT |
                    VK_IMAGE_USAGE_TRANSFER_SRC_BIT |
                    VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imgInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imgInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;

    VmaAllocationCreateInfo allocInfo{};
    allocInfo.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;

    if (vmaCreateImage(m_allocator, &imgInfo, &allocInfo, &outImg.image, &outImg.allocation, nullptr) != VK_SUCCESS) {
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

    if (vkCreateImageView(m_device, &viewInfo, nullptr, &outImg.view) != VK_SUCCESS) {
        vmaDestroyImage(m_allocator, outImg.image, outImg.allocation);
        outImg.image = VK_NULL_HANDLE;
        outImg.allocation = VK_NULL_HANDLE;
        throw std::runtime_error("UpwaysPipeline: Failed to create 2D array image view!");
    }

    outImg.width = width;
    outImg.height = height;
    outImg.layers = layers;
}

void UpwaysPipeline::destroyArrayImage(ArrayImage2D& img) {
    if (img.view != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
        vkDestroyImageView(m_device, img.view, nullptr);
        img.view = VK_NULL_HANDLE;
    }
    if (img.image != VK_NULL_HANDLE && m_allocator != VK_NULL_HANDLE) {
        vmaDestroyImage(m_allocator, img.image, img.allocation);
        img.image = VK_NULL_HANDLE;
        img.allocation = VK_NULL_HANDLE;
    }
    img.width = 0;
    img.height = 0;
    img.layers = 0;
}

void UpwaysPipeline::initRenderResources(uint32_t slotMask) {
    m_pyramidDims = computeUpwaysPyramidDims(m_inputWidth, m_inputHeight);

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
            m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R32_UINT, kStorageUsage
        );
        m_slotWidth[s] = m_inputWidth;
        m_slotHeight[s] = m_inputHeight;
        m_lrSlotTransitioned[s] = false;
    }

    createArrayImage(m_baseAnchorLr, m_inputWidth, m_inputHeight, BASE_ANCHOR_LAYERS);
    m_cleanRadLr = std::make_unique<Image>(
        m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R16G16B16A16_SFLOAT, kStorageUsage
    );

    const VkDeviceSize actBytes = (m_precision == UpwaysComputePrecision::Fp16)
        ? (m_pyramidDims.totalBytes * 2ULL)
        : m_pyramidDims.totalBytes;
    m_actBuffer = std::make_unique<Buffer>(
        m_allocator,
        actBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    const VkDeviceSize srSubBytes = std::max<VkDeviceSize>(
        static_cast<VkDeviceSize>(m_inputWidth) * m_inputHeight * 32ULL,
        256ULL
    );
    m_srSubpixelBuffer = std::make_unique<Buffer>(
        m_allocator,
        srSubBytes,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
    );

    m_transientsTransitioned = false;
    m_initialLayoutsTransitioned = false;
}

void UpwaysPipeline::initDisplayResources() {
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
        m_device, m_allocator, m_outputWidth, m_outputHeight, m_imageFormat, kDisplayUsage
    );

    if (!m_dummyBlackImage) {
        m_dummyBlackImage = std::make_unique<Image>(
            m_device, m_allocator, 1u, 1u, VK_FORMAT_R16G16B16A16_SFLOAT, kDisplayUsage
        );
    }

    m_hrTransitioned = false;
    m_initialLayoutsTransitioned = false;
    m_historyValid = false;
}

void UpwaysPipeline::ensureActiveSlotSized() {
    const uint32_t curr = m_pingPongIndex & 1u;
    if (m_slotWidth[curr] == m_inputWidth && m_slotHeight[curr] == m_inputHeight) {
        return;
    }

    // Ensure previous commands using curr before DRS finish before destroying curr images
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
        m_device, m_allocator, m_inputWidth, m_inputHeight, VK_FORMAT_R32_UINT, kStorageUsage
    );
    m_slotWidth[curr] = m_inputWidth;
    m_slotHeight[curr] = m_inputHeight;
    m_lrSlotTransitioned[curr] = false;
    m_initialLayoutsTransitioned = false;
    refreshDescriptors();
}

void UpwaysPipeline::transitionInitialLayouts(VkCommandBuffer cmd) {
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

    auto transitionAndClearArray = [&](ArrayImage2D& arr) {
        if (arr.image == VK_NULL_HANDLE) {
            return;
        }
        VkImageSubresourceRange rangeArr{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, arr.layers };

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
        postClear.dstAccessMask = kPostClearAccess;
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
            transitionAndClear(m_colorHr[i].get());
        }
        m_hrTransitioned = true;
    }

    bool anyBufCleared = false;
    if (!m_transientsTransitioned) {
        transitionAndClearArray(m_baseAnchorLr);
        transitionAndClear(m_cleanRadLr.get());
        if (m_actBuffer) {
            vkCmdFillBuffer(cmd, m_actBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0u);
            anyBufCleared = true;
        }
        if (m_srSubpixelBuffer) {
            vkCmdFillBuffer(cmd, m_srSubpixelBuffer->getBuffer(), 0, VK_WHOLE_SIZE, 0u);
            anyBufCleared = true;
        }
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

    // Pass 1 (upways_denoise_prep.comp): 0..13 storage images, 14 SSBO
    std::vector<VkDescriptorType> prepTypes(15, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    prepTypes[14] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    m_prepDescLayout = createLayout(makeBindings(prepTypes));

    // Pass 2 (upways_denoiser_wmma.comp): 0..2 SSBOs, 3..14 storage images
    std::vector<VkDescriptorType> denoiserTypes(15, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    denoiserTypes[0] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    denoiserTypes[1] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    denoiserTypes[2] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    m_denoiserDescLayout = createLayout(makeBindings(denoiserTypes));

    // Pass 3 (upways_upscaler_wmma.comp): 0..3 SSBOs
    std::vector<VkDescriptorType> upscalerTypes(4, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    m_upscalerDescLayout = createLayout(makeBindings(upscalerTypes));

    // Pass 4 (upways_upscale_resolve.comp): 0..4 storage images, 5 SSBO, 6..8 storage images
    std::vector<VkDescriptorType> resolveTypes(9, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    resolveTypes[5] = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    m_resolveDescLayout = createLayout(makeBindings(resolveTypes));
}

void UpwaysPipeline::allocateDescriptorSets() {
    std::array<VkDescriptorPoolSize, 2> poolSizes{};
    poolSizes[0].type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    poolSizes[0].descriptorCount = 128u;
    poolSizes[1].type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    poolSizes[1].descriptorCount = 32u;

    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 16u;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSizes.size());
    poolInfo.pPoolSizes = poolSizes.data();

    if (vkCreateDescriptorPool(m_device, &poolInfo, nullptr, &m_descriptorPool) != VK_SUCCESS) {
        throw std::runtime_error("UpwaysPipeline: Failed to create descriptor pool!");
    }

    auto allocSet = [this](VkDescriptorSetLayout layout) {
        VkDescriptorSetAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
        allocInfo.descriptorPool = m_descriptorPool;
        allocInfo.descriptorSetCount = 1u;
        allocInfo.pSetLayouts = &layout;
        VkDescriptorSet set = VK_NULL_HANDLE;
        if (vkAllocateDescriptorSets(m_device, &allocInfo, &set) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysPipeline: Failed to allocate descriptor set!");
        }
        return set;
    };

    m_upscalerDescSet = allocSet(m_upscalerDescLayout);
    for (int i = 0; i < 2; ++i) {
        m_prepDescSets[i] = allocSet(m_prepDescLayout);
        m_denoiserDescSets[i] = allocSet(m_denoiserDescLayout);
        m_resolveDescSets[i] = allocSet(m_resolveDescLayout);
    }
}

void UpwaysPipeline::updateDescriptors(
    VkImageView mlDiffuseView,
    VkImageView mlSpecularView,
    VkImageView albedoRoughnessView,
    VkImageView specAlbedoView,
    VkImageView skipView,
    VkImageView normalDepthView,
    VkImageView motionVectorView,
    VkImageView specularMotionView
) {
    m_cachedMlDiffView = mlDiffuseView;
    m_cachedMlSpecView = mlSpecularView;
    m_cachedAlbRoughView = albedoRoughnessView;
    m_cachedSpecAlbView = specAlbedoView;
    m_cachedSkipView = skipView;
    m_cachedNormalDepthView = normalDepthView;
    m_cachedMotionVectorView = motionVectorView;
    m_cachedSpecularMotionView = specularMotionView;

    refreshDescriptors();
}

void UpwaysPipeline::refreshDescriptors() {
    if (!m_outputImage || !m_weightBuffer || !m_actBuffer || !m_srSubpixelBuffer) {
        return;
    }

    VkImageView fallbackZero = m_dummyBlackImage
        ? m_dummyBlackImage->getImageView()
        : m_outputImage->getImageView();

    auto resolveView = [fallbackZero](VkImageView v) {
        return (v != VK_NULL_HANDLE) ? v : fallbackZero;
    };

    const VkImageView diffView = resolveView(m_cachedMlDiffView);
    const VkImageView specView = resolveView(m_cachedMlSpecView);
    const VkImageView albRoughView = resolveView(m_cachedAlbRoughView);
    const VkImageView specAlbView = resolveView(m_cachedSpecAlbView);
    const VkImageView skipView = resolveView(m_cachedSkipView);
    const VkImageView ndView = resolveView(m_cachedNormalDepthView);
    const VkImageView mvView = resolveView(m_cachedMotionVectorView);
    const VkImageView smvView = resolveView(m_cachedSpecularMotionView);

    const VkDescriptorBufferInfo w8Bi{
        m_weightBuffer->getBuffer(),
        0u,
        m_weightQuantBytes
    };
    const VkDescriptorBufferInfo w32Bi{
        m_weightBuffer->getBuffer(),
        m_weightQuantBytes,
        upways::FP32_WEIGHT_SECTION_BYTES
    };
    const VkDescriptorBufferInfo actBi{ m_actBuffer->getBuffer(), 0u, m_actBuffer->getSize() };
    const VkDescriptorBufferInfo srSubBi{ m_srSubpixelBuffer->getBuffer(), 0u, m_srSubpixelBuffer->getSize() };

    // Pass 3: upways_upscaler_wmma.comp (4 bindings)
    {
        std::array<VkDescriptorBufferInfo, 4> bi = { w8Bi, w32Bi, actBi, srSubBi };
        std::array<VkWriteDescriptorSet, 4> writes{};
        for (uint32_t b = 0; b < 4u; ++b) {
            writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[b].dstSet = m_upscalerDescSet;
            writes[b].dstBinding = b;
            writes[b].descriptorCount = 1u;
            writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[b].pBufferInfo = &bi[b];
        }
        vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0u, nullptr);
    }

    for (int curr = 0; curr < 2; ++curr) {
        const int prev = 1 - curr;

        // Pass 1: upways_denoise_prep.comp (15 bindings)
        {
            std::array<VkDescriptorImageInfo, 14> ii{};
            auto setImg = [&](uint32_t idx, VkImageView view) {
                ii[idx] = { VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL };
            };
            setImg(0u, diffView);
            setImg(1u, specView);
            setImg(2u, albRoughView);
            setImg(3u, specAlbView);
            setImg(4u, skipView);
            setImg(5u, ndView);
            setImg(6u, mvView);
            setImg(7u, smvView);
            setImg(8u, m_diffHist[prev]->getImageView());
            setImg(9u, m_specHist[prev]->getImageView());
            setImg(10u, m_geomHist[prev]->getImageView());
            setImg(11u, m_hiddenHist[prev]->getImageView());
            setImg(12u, m_baseAnchorLr.view);
            setImg(13u, m_geomHist[curr]->getImageView());

            std::array<VkWriteDescriptorSet, 15> writes{};
            for (uint32_t b = 0; b < 14u; ++b) {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = m_prepDescSets[curr];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1u;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[b].pImageInfo = &ii[b];
            }
            writes[14].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            writes[14].dstSet = m_prepDescSets[curr];
            writes[14].dstBinding = 14u;
            writes[14].descriptorCount = 1u;
            writes[14].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            writes[14].pBufferInfo = &actBi;

            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0u, nullptr);
        }

        // Pass 2: upways_denoiser_wmma.comp (15 bindings)
        {
            std::array<VkDescriptorBufferInfo, 3> bi = { w8Bi, w32Bi, actBi };
            std::array<VkDescriptorImageInfo, 12> ii{};
            auto setImg = [&](uint32_t bindingIdx, VkImageView view) {
                ii[bindingIdx - 3u] = { VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL };
            };
            setImg(3u, m_baseAnchorLr.view);
            setImg(4u, diffView);
            setImg(5u, specView);
            setImg(6u, albRoughView);
            setImg(7u, specAlbView);
            setImg(8u, skipView);
            setImg(9u, ndView);
            setImg(10u, m_diffHist[curr]->getImageView());
            setImg(11u, m_specHist[curr]->getImageView());
            setImg(12u, m_hiddenHist[curr]->getImageView());
            setImg(13u, m_cleanRadLr->getImageView());
            setImg(14u, m_outputImage->getImageView());

            std::array<VkWriteDescriptorSet, 15> writes{};
            for (uint32_t b = 0; b < 3u; ++b) {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = m_denoiserDescSets[curr];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1u;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                writes[b].pBufferInfo = &bi[b];
            }
            for (uint32_t b = 3u; b < 15u; ++b) {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = m_denoiserDescSets[curr];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1u;
                writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                writes[b].pImageInfo = &ii[b - 3u];
            }

            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0u, nullptr);
        }

        // Pass 4: upways_upscale_resolve.comp (9 bindings)
        {
            std::array<VkDescriptorImageInfo, 8> ii{};
            ii[0] = { VK_NULL_HANDLE, m_cleanRadLr->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            ii[1] = { VK_NULL_HANDLE, ndView, VK_IMAGE_LAYOUT_GENERAL };
            ii[2] = { VK_NULL_HANDLE, mvView, VK_IMAGE_LAYOUT_GENERAL };
            ii[3] = { VK_NULL_HANDLE, specView, VK_IMAGE_LAYOUT_GENERAL };
            ii[4] = { VK_NULL_HANDLE, skipView, VK_IMAGE_LAYOUT_GENERAL };
            ii[5] = { VK_NULL_HANDLE, m_colorHr[prev]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            ii[6] = { VK_NULL_HANDLE, m_colorHr[curr]->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
            ii[7] = { VK_NULL_HANDLE, m_outputImage->getImageView(), VK_IMAGE_LAYOUT_GENERAL };

            std::array<VkWriteDescriptorSet, 9> writes{};
            for (uint32_t b = 0; b < 9u; ++b) {
                writes[b].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
                writes[b].dstSet = m_resolveDescSets[curr];
                writes[b].dstBinding = b;
                writes[b].descriptorCount = 1u;
                if (b == 5u) {
                    writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                    writes[b].pBufferInfo = &srSubBi;
                } else {
                    const uint32_t imgIdx = (b < 5u) ? b : (b - 1u);
                    writes[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                    writes[b].pImageInfo = &ii[imgIdx];
                }
            }

            vkUpdateDescriptorSets(m_device, static_cast<uint32_t>(writes.size()), writes.data(), 0u, nullptr);
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
    const std::vector<char>& denoisePrepSpv,
    const std::vector<char>& denoiserWmmaSpv,
    const std::vector<char>& upscalerWmmaSpv,
    const std::vector<char>& upscaleResolveSpv
) {
    auto createPipeLayout = [this](VkDescriptorSetLayout descLayout) {
        VkPushConstantRange pcRange{};
        pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        pcRange.offset = 0u;
        pcRange.size = sizeof(UpwaysPushConstants);

        VkPipelineLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
        layoutInfo.setLayoutCount = 1u;
        layoutInfo.pSetLayouts = &descLayout;
        layoutInfo.pushConstantRangeCount = 1u;
        layoutInfo.pPushConstantRanges = &pcRange;

        VkPipelineLayout outLayout = VK_NULL_HANDLE;
        if (vkCreatePipelineLayout(m_device, &layoutInfo, nullptr, &outLayout) != VK_SUCCESS) {
            throw std::runtime_error("UpwaysPipeline: Failed to create compute pipeline layout!");
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
                "upways_v4_vk_pipeline_cache_{:04x}_{:04x}_{}.bin",
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
        Logger::warn("UpwaysPipeline: Failed to create VkPipelineCache; proceeding with uncached compilation.");
    }

    m_prepPipeLayout = createPipeLayout(m_prepDescLayout);
    m_denoiserPipeLayout = createPipeLayout(m_denoiserDescLayout);
    m_upscalerPipeLayout = createPipeLayout(m_upscalerDescLayout);
    m_resolvePipeLayout = createPipeLayout(m_resolveDescLayout);

    VkShaderModule prepMod = createShaderModule(denoisePrepSpv);
    VkShaderModule denoiserMod = createShaderModule(denoiserWmmaSpv);
    VkShaderModule upscalerMod = createShaderModule(upscalerWmmaSpv);
    VkShaderModule resolveMod = createShaderModule(upscaleResolveSpv);

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
            throw std::runtime_error("UpwaysPipeline: Failed to create compute pipeline!");
        }
        return outPipe;
    };

    try {
        m_prepPipeline = createComputePipe(prepMod, m_prepPipeLayout, true, false, nullptr);

        for (uint32_t s = 0; s < NUM_DENOISER_STAGES; ++s) {
            const uint32_t stageId = s;
            const VkSpecializationMapEntry entry{ 0u, 0u, sizeof(uint32_t) };
            const VkSpecializationInfo specInfo{ 1u, &entry, sizeof(uint32_t), &stageId };
            m_denoiserPipelines[s] = createComputePipe(denoiserMod, m_denoiserPipeLayout, true, true, &specInfo);
        }

        m_upscalerPipeline = createComputePipe(upscalerMod, m_upscalerPipeLayout, true, true, nullptr);
        m_resolvePipeline = createComputePipe(resolveMod, m_resolvePipeLayout, true, false, nullptr);
    } catch (...) {
        vkDestroyShaderModule(m_device, prepMod, nullptr);
        vkDestroyShaderModule(m_device, denoiserMod, nullptr);
        vkDestroyShaderModule(m_device, upscalerMod, nullptr);
        vkDestroyShaderModule(m_device, resolveMod, nullptr);
        throw;
    }

    vkDestroyShaderModule(m_device, prepMod, nullptr);
    vkDestroyShaderModule(m_device, denoiserMod, nullptr);
    vkDestroyShaderModule(m_device, upscalerMod, nullptr);
    vkDestroyShaderModule(m_device, resolveMod, nullptr);

    if (m_pipelineCache != VK_NULL_HANDLE && !cachePath.empty()) {
        size_t cacheBytes = 0u;
        if (vkGetPipelineCacheData(m_device, m_pipelineCache, &cacheBytes, nullptr) == VK_SUCCESS && cacheBytes > 0u) {
            std::vector<uint8_t> outCache(cacheBytes);
            if (vkGetPipelineCacheData(m_device, m_pipelineCache, &cacheBytes, outCache.data()) == VK_SUCCESS) {
                std::filesystem::path tmpPath = cacheDir / std::format(
                    "upways_v4_vk_pipeline_cache_{:04x}_{:04x}.tmp.{}",
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
        "UpwaysPipeline 4.0 compiled 7 compute pipelines in {:.1f} ms (disk cache: {}).",
        compileMs,
        cachePath.empty() ? "none" : cachePath.filename().string()
    );
}

void UpwaysPipeline::resize(
    uint32_t inputWidth,
    uint32_t inputHeight,
    uint32_t outputWidth,
    uint32_t outputHeight,
    bool enableSuperRes
) {
    const uint32_t newInW = std::max(1u, inputWidth);
    const uint32_t newInH = std::max(1u, inputHeight);
    const uint32_t newOutW = std::max(1u, outputWidth);
    const uint32_t newOutH = std::max(1u, outputHeight);

    const bool renderChanged = (newInW != m_inputWidth || newInH != m_inputHeight);
    const bool displayChanged = (newOutW != m_outputWidth || newOutH != m_outputHeight);
    m_enableSuperRes = enableSuperRes;

    if (!renderChanged && !displayChanged) {
        return;
    }

    if (m_device != VK_NULL_HANDLE) {
        vkDeviceWaitIdle(m_device);
    }

    if (renderChanged) {
        // External LR G-buffer views belong to the previous render dimensions and may
        // already be destroyed by RenderTargetManager before resize() runs; clear to
        // VK_NULL_HANDLE so refreshDescriptors() binds fallbackZero until updateDescriptors().
        m_cachedMlDiffView = VK_NULL_HANDLE;
        m_cachedMlSpecView = VK_NULL_HANDLE;
        m_cachedAlbRoughView = VK_NULL_HANDLE;
        m_cachedSpecAlbView = VK_NULL_HANDLE;
        m_cachedSkipView = VK_NULL_HANDLE;
        m_cachedNormalDepthView = VK_NULL_HANDLE;
        m_cachedMotionVectorView = VK_NULL_HANDLE;
        m_cachedSpecularMotionView = VK_NULL_HANDLE;
    }

    m_inputWidth = newInW;
    m_inputHeight = newInH;
    m_outputWidth = newOutW;
    m_outputHeight = newOutH;

    if (!displayChanged && renderChanged && m_historyValid) {
        // Lossless DRS: preserve m_colorHr ping-pong and the prev LR slot so temporal history survives
        const uint32_t curr = m_pingPongIndex & 1u;
        initRenderResources(1u << curr);
        refreshDescriptors();
        Logger::info(
            "UpwaysPipeline 4.0 lossless DRS resize: LR {}x{} (L1 {}x{}, L2 {}x{}), HR {}x{} preserved (SuperRes: {}).",
            m_inputWidth, m_inputHeight,
            m_pyramidDims.w1, m_pyramidDims.h1,
            m_pyramidDims.w2, m_pyramidDims.h2,
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
        "UpwaysPipeline 4.0 resized: LR {}x{} (L1 {}x{}, L2 {}x{}), HR {}x{} (SuperRes: {}).",
        m_inputWidth, m_inputHeight,
        m_pyramidDims.w1, m_pyramidDims.h1,
        m_pyramidDims.w2, m_pyramidDims.h2,
        m_outputWidth, m_outputHeight,
        m_enableSuperRes ? "ON" : "OFF"
    );
}

void UpwaysPipeline::collectQueryResults(uint32_t targetSlot, bool waitForResult) {
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

void UpwaysPipeline::recordFrame(
    VkCommandBuffer cmd,
    uint32_t frameIndex,
    bool resetHistory,
    glm::vec2 jitter,
    bool coupledWithExternalUpscaler,
    bool zeroResidualMode,
    float exposureScale
) {
    if (!isInitialized()) {
        return;
    }

    // Collect timestamps from the query slot about to be reused (wait if still in flight)
    // and opportunistic non-blocking read of the most recently completed slot.
    collectQueryResults(m_querySlot, true);
    collectQueryResults(1u - m_querySlot, false);

    // Ensure the target ping-pong LR slot matches the active render resolution (DRS 2nd frame)
    ensureActiveSlotSized();
    transitionInitialLayouts(cmd);

    const uint32_t curr = m_pingPongIndex & 1u;
    const bool doReset = resetHistory || !m_historyValid;
    const bool runSuperRes =
        (m_enableSuperRes || m_inputWidth != m_outputWidth || m_inputHeight != m_outputHeight) &&
        !coupledWithExternalUpscaler;

    UpwaysPushConstants pc{};
    pc.lrWidth = m_inputWidth;
    pc.lrHeight = m_inputHeight;
    pc.hrWidth = m_outputWidth;
    pc.hrHeight = m_outputHeight;
    pc.jitterX = jitter.x;
    pc.jitterY = jitter.y;
    pc.frameIndex = frameIndex;
    pc.flags = 0u;
    if (doReset) {
        pc.flags |= UPWAYS_FLAG_RESET_HISTORY;
    }
    if (runSuperRes) {
        pc.flags |= UPWAYS_FLAG_ENABLE_SUPERRES;
    }
    if (zeroResidualMode) {
        pc.flags |= UPWAYS_FLAG_ZERO_RESIDUAL_MODE;
    }
    pc.offX0 = m_pyramidDims.offX0;
    pc.offHw0 = m_pyramidDims.offHw0;
    pc.offS0 = m_pyramidDims.offS0;
    pc.offB1 = m_pyramidDims.offB1;
    pc.offB2 = m_pyramidDims.offB2;
    pc.exposureScale = (exposureScale > 0.0f) ? exposureScale : 1.0f;

    const uint32_t qSlot = m_querySlot;
    const uint32_t qBase = qSlot * QUERIES_PER_SLOT;
    if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, m_queryPool, qBase, QUERIES_PER_SLOT);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 0u);
    }

    // Pass 1: upways_denoise_prep.comp
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_prepPipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_prepPipeLayout, 0u, 1u, &m_prepDescSets[curr], 0u, nullptr);
    vkCmdPushConstants(cmd, m_prepPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(UpwaysPushConstants), &pc);
    vkCmdDispatch(cmd, divCeil(m_inputWidth, 16u), divCeil(m_inputHeight, 8u), 1u);

    emitComputeBarrier(cmd);
    if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 1u);
    }

    // Pass 2: upways_denoiser_wmma.comp (4 specialized Wave32 stages: 0, 1 @ W1xH1; 2 @ W2xH2; 3 @ W1xH1)
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_denoiserPipeLayout, 0u, 1u, &m_denoiserDescSets[curr], 0u, nullptr);
    vkCmdPushConstants(cmd, m_denoiserPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(UpwaysPushConstants), &pc);

    for (uint32_t stage = 0; stage < NUM_DENOISER_STAGES; ++stage) {
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_denoiserPipelines[stage]);
        const uint32_t sw = (stage == 2u) ? m_pyramidDims.w2 : m_pyramidDims.w1;
        const uint32_t sh = (stage == 2u) ? m_pyramidDims.h2 : m_pyramidDims.h1;
        vkCmdDispatch(cmd, divCeil(sw, 16u), divCeil(sh, 8u), 1u);
        emitComputeBarrier(cmd);
    }

    if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 2u);
    }

    if (runSuperRes) {
        // Pass 3: upways_upscaler_wmma.comp
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upscalerPipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_upscalerPipeLayout, 0u, 1u, &m_upscalerDescSet, 0u, nullptr);
        vkCmdPushConstants(cmd, m_upscalerPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(UpwaysPushConstants), &pc);
        vkCmdDispatch(cmd, divCeil(m_inputWidth, 16u), divCeil(m_inputHeight, 8u), 1u);

        emitComputeBarrier(cmd);
        if (m_queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 3u);
        }

        // Pass 4: upways_upscale_resolve.comp
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, m_resolvePipeLayout, 0u, 1u, &m_resolveDescSets[curr], 0u, nullptr);
        vkCmdPushConstants(cmd, m_resolvePipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0u, sizeof(UpwaysPushConstants), &pc);
        vkCmdDispatch(cmd, divCeil(m_outputWidth, 8u), divCeil(m_outputHeight, 8u), 1u);

        if (m_queryPool != VK_NULL_HANDLE) {
            vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, m_queryPool, qBase + 4u);
        }
    } else if (m_queryPool != VK_NULL_HANDLE) {
        // 1.0x native denoiser-only mode: Stage 3 wrote uOutputColor directly
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
