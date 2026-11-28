#include "rt/UpwaysPipeline.hpp"
#include "rt/PostProcessPipeline.hpp"
#include "rt/upways_weights.hpp"
#include "rt/upways_default_weights.hpp"
#include "rt/upways_default_int8_weights.hpp"
#include "rt/upways_v5_weights.hpp"
#include "rt/upways_v5_default_weights.hpp"
#include "rt/upways_v5_default_int8_weights.hpp"
#include "core/Config.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "vulkan/VulkanContext.hpp"

#include <glm/gtc/packing.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

using namespace pathways;

namespace {

int g_failures = 0;

void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << std::endl;
        ++g_failures;
    }
}

std::filesystem::path findSpv(const std::string& name) {
    namespace fs = std::filesystem;
    std::vector<fs::path> candidates = {
#ifdef SHADER_DIR
        fs::path(SHADER_DIR) / name,
#endif
        fs::current_path() / "shaders" / name,
        fs::current_path() / "build" / "shaders" / name,
        fs::current_path() / "build" / "linux-release" / "shaders" / name,
        fs::current_path() / "build" / "windows-clang-release" / "shaders" / name,
    };
    for (const auto& p : candidates) {
        if (fs::exists(p)) {
            return p;
        }
    }
    return {};
}

std::vector<char> loadSpv(const std::string& name) {
    auto p = findSpv(name);
    if (p.empty()) {
        return {};
    }
    std::ifstream f(p, std::ios::binary | std::ios::ate);
    if (!f.is_open()) {
        return {};
    }
    const size_t sz = static_cast<size_t>(f.tellg());
    std::vector<char> buf(sz);
    f.seekg(0);
    f.read(buf.data(), static_cast<std::streamsize>(sz));
    return buf;
}

bool validateSpirv(const std::vector<char>& bytes) {
    if (bytes.size() < 20) {
        return false;
    }
    uint32_t header[5]{};
    std::memcpy(header, bytes.data(), sizeof(header));
    return header[0] == 0x07230203u && header[3] > 5u;
}

uint32_t packRGB9E5Cpu(const glm::vec3& rgb) {
    auto sanitizeClamp = [](float x) -> float {
        if (!std::isfinite(x) || x <= 0.0f) return 0.0f;
        return std::min(x, 65408.0f);
    };
    const glm::vec3 v(sanitizeClamp(rgb.r), sanitizeClamp(rgb.g), sanitizeClamp(rgb.b));
    const float maxc = std::max(v.r, std::max(v.g, v.b));
    if (maxc <= 0.0f) return 0u;

    int exp_shared = std::max(-16, static_cast<int>(std::floor(std::log2(std::max(maxc, 1e-30f))))) + 16;
    float invDenom = std::exp2(static_cast<float>(24 - exp_shared));
    const int maxm = static_cast<int>(std::floor(maxc * invDenom + 0.5f));
    if (maxm >= 512) {
        exp_shared += 1;
        invDenom = std::exp2(static_cast<float>(24 - exp_shared));
    }
    const uint32_t rm = std::min(static_cast<uint32_t>(std::floor(v.r * invDenom + 0.5f)), 511u);
    const uint32_t gm = std::min(static_cast<uint32_t>(std::floor(v.g * invDenom + 0.5f)), 511u);
    const uint32_t bm = std::min(static_cast<uint32_t>(std::floor(v.b * invDenom + 0.5f)), 511u);
    if ((rm | gm | bm) == 0u) return 0u;
    const uint32_t e = static_cast<uint32_t>(std::clamp(exp_shared, 0, 31));
    return rm | (gm << 9u) | (bm << 18u) | (e << 27u);
}

glm::vec3 unpackRGB9E5Cpu(uint32_t p) {
    const int e = static_cast<int>((p >> 27u) & 31u) - 24;
    const float scale = std::exp2(static_cast<float>(e));
    return glm::vec3(
        static_cast<float>(p & 511u),
        static_cast<float>((p >> 9u) & 511u),
        static_cast<float>((p >> 18u) & 511u)
    ) * scale;
}

uint32_t packOct32Cpu(const glm::vec3& v) {
    const float l2 = glm::dot(v, v);
    const glm::vec3 n = (l2 > 1e-12f) ? (v / std::sqrt(l2)) : glm::vec3(0.0f, 0.0f, 1.0f);
    const float invL1 = 1.0f / std::max(std::abs(n.x) + std::abs(n.y) + std::abs(n.z), 1e-6f);
    glm::vec2 p(n.x * invL1, n.y * invL1);
    if (n.z < 0.0f) {
        p = glm::vec2(
            (1.0f - std::abs(p.y)) * (p.x >= 0.0f ? 1.0f : -1.0f),
            (1.0f - std::abs(p.x)) * (p.y >= 0.0f ? 1.0f : -1.0f)
        );
    }
    return glm::packSnorm2x16(p);
}

struct GpuGbufferSet {
    uint32_t width = 0;
    uint32_t height = 0;
    std::unique_ptr<Image> irr;
    std::unique_ptr<Image> primGuide;
    std::unique_ptr<Image> surfaceMv;
    std::unique_ptr<Image> diffHitDist;
    std::unique_ptr<Image> specVmv;
    std::unique_ptr<Image> skipCompact;
};

GpuGbufferSet createPopulatedGbufferSet(
    VkDevice device,
    VmaAllocator allocator,
    VkCommandBuffer cmd,
    uint32_t width,
    uint32_t height,
    std::vector<std::unique_ptr<Buffer>>& stagingKeepAlive,
    bool spatialPattern = false,
    uint32_t rowOffset = 0u,
    uint32_t fullH = 0u
) {
    GpuGbufferSet gbuf;
    gbuf.width = width;
    gbuf.height = height;
    const uint32_t refH = (fullH > 0u) ? fullH : height;

    const VkImageUsageFlags usage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    gbuf.irr = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R32G32_UINT, usage);
    gbuf.primGuide = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R32G32B32A32_UINT, usage);
    gbuf.surfaceMv = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R32_UINT, usage);
    gbuf.diffHitDist = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R8_UNORM, usage);
    gbuf.specVmv = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R32G32_UINT, usage);
    gbuf.skipCompact = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R32_UINT, usage);

    const size_t numPixels = static_cast<size_t>(width) * height;
    std::vector<uint32_t> irrData(numPixels * 2u, 0u);
    std::vector<uint32_t> primData(numPixels * 4u, 0u);
    std::vector<uint32_t> surfMvData(numPixels, 0u);
    std::vector<uint8_t> diffHitData(numPixels, 0u);
    std::vector<uint32_t> specVmvData(numPixels * 2u, 0u);
    std::vector<uint32_t> skipData(numPixels, 0u);

    for (uint32_t y = 0; y < height; ++y) {
        const uint32_t fy = rowOffset + y;
        const float v = static_cast<float>(fy) / static_cast<float>(std::max(refH, 1u));
        for (uint32_t x = 0; x < width; ++x) {
            const float u = static_cast<float>(x) / static_cast<float>(std::max(width, 1u));
            glm::vec3 aDiff(0.8f, 0.8f, 0.8f);
            glm::vec3 aSpec(0.06f, 0.06f, 0.06f);
            glm::vec3 diffIrr(1.0f, 1.0f, 1.0f);
            glm::vec3 specIrr(0.25f, 0.25f, 0.25f);
            glm::vec3 skip(0.05f, 0.05f, 0.05f);
            glm::vec3 nrm(0.0f, 0.0f, 1.0f);
            float linDepth = 3.0f;
            float diffHitDist = 0.5f;
            float react = 0.0f;
            float rough = 0.4f;
            float matClass = 0.5f;
            glm::vec2 surfMv(0.0f, 0.0f);
            glm::vec2 specMv(0.0f, 0.0f);
            float specHitDist = 0.5f;

            if (spatialPattern) {
                const float wave = 0.5f + 0.5f * std::sin(u * 12.0f + v * 16.0f);
                const float checker = (((x >> 3) + (fy >> 3)) & 1u) ? 0.85f : 0.45f;
                aDiff = glm::vec3(0.35f + 0.45f * u, 0.25f + 0.55f * v, 0.30f + 0.40f * wave) * checker;
                aSpec = glm::vec3(0.05f + 0.04f * v, 0.06f, 0.07f + 0.03f * u);
                diffIrr = glm::vec3(0.9f + 0.6f * wave, 0.8f + 0.4f * v, 0.7f + 0.5f * u);
                specIrr = glm::vec3(0.20f + 0.30f * v, 0.22f, 0.18f + 0.25f * u);
                skip = glm::vec3(0.02f + 0.03f * v);
                nrm = glm::normalize(glm::vec3(0.15f * std::sin(u * 6.0f), 0.15f * std::cos(v * 6.0f), 1.0f));
                linDepth = 2.0f + 3.5f * v;
                rough = 0.2f + 0.6f * u;
            }

            // Compact live format stores MODULATED radiance in irr (deferred demod in prep):
            const glm::vec3 modDiff = diffIrr * aDiff;
            const glm::vec3 modSpec = specIrr * aSpec;

            const size_t idx = static_cast<size_t>(y) * width + x;
            irrData[idx * 2u + 0u] = packRGB9E5Cpu(modDiff);
            irrData[idx * 2u + 1u] = packRGB9E5Cpu(modSpec);

            const uint32_t depthHalf = glm::packHalf2x16(glm::vec2(linDepth, 0.0f)) & 0xFFFFu;
            const uint32_t dist8 = static_cast<uint32_t>(std::clamp(std::round(diffHitDist * 255.0f), 0.0f, 255.0f));
            const uint32_t react8 = static_cast<uint32_t>(std::clamp(std::round(react * 255.0f), 0.0f, 255.0f));
            primData[idx * 4u + 0u] = packOct32Cpu(nrm);
            primData[idx * 4u + 1u] = depthHalf | (react8 << 24u);
            diffHitData[idx] = static_cast<uint8_t>(dist8);
            primData[idx * 4u + 2u] = glm::packUnorm4x8(glm::vec4(aDiff, rough));
            primData[idx * 4u + 3u] = glm::packUnorm4x8(glm::vec4(aSpec, matClass));

            surfMvData[idx] = glm::packHalf2x16(surfMv);
            specVmvData[idx * 2u + 0u] = glm::packHalf2x16(specMv);
            specVmvData[idx * 2u + 1u] = glm::packHalf2x16(glm::vec2(0.0f, specHitDist));
            skipData[idx] = packRGB9E5Cpu(skip);
        }
    }

    auto uploadImgData = [&](Image& img, const void* data, VkDeviceSize bytes) {
        auto stg = std::make_unique<Buffer>(
            allocator, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        );
        stg->copyFrom(data, bytes);

        img.transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
        );
        VkBufferImageCopy region{};
        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        region.imageExtent = { width, height, 1 };
        vkCmdCopyBufferToImage(cmd, stg->getBuffer(), img.getImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &region);
        stagingKeepAlive.push_back(std::move(stg));
    };

    uploadImgData(*gbuf.irr, irrData.data(), static_cast<VkDeviceSize>(irrData.size() * sizeof(uint32_t)));
    uploadImgData(*gbuf.primGuide, primData.data(), static_cast<VkDeviceSize>(primData.size() * sizeof(uint32_t)));
    uploadImgData(*gbuf.surfaceMv, surfMvData.data(), static_cast<VkDeviceSize>(surfMvData.size() * sizeof(uint32_t)));
    uploadImgData(*gbuf.diffHitDist, diffHitData.data(), static_cast<VkDeviceSize>(diffHitData.size()));
    uploadImgData(*gbuf.specVmv, specVmvData.data(), static_cast<VkDeviceSize>(specVmvData.size() * sizeof(uint32_t)));
    uploadImgData(*gbuf.skipCompact, skipData.data(), static_cast<VkDeviceSize>(skipData.size() * sizeof(uint32_t)));
    return gbuf;
}

void uploadGbufferRawBuffers(
    VkDevice device,
    VmaAllocator allocator,
    VkQueue queue,
    VkCommandBuffer cmd,
    const GpuGbufferSet& gbuf,
    const uint32_t* irrData,
    const uint32_t* primData,
    const uint32_t* surfMvData,
    const uint8_t* diffHitData,
    const uint32_t* specVmvData,
    const uint32_t* skipData
) {
    const size_t n = static_cast<size_t>(gbuf.width) * gbuf.height;
    std::vector<std::unique_ptr<Buffer>> stgKeep;
    auto stageImg = [&](Image& img, const void* ptr, VkDeviceSize bytes) {
        auto s = std::make_unique<Buffer>(
            allocator, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        );
        s->copyFrom(ptr, bytes);
        img.transitionLayout(
            cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_NONE, 0,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
        );
        VkBufferImageCopy r{};
        r.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        r.imageExtent = { gbuf.width, gbuf.height, 1 };
        vkCmdCopyBufferToImage(cmd, s->getBuffer(), img.getImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &r);
        stgKeep.push_back(std::move(s));
    };

    VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(cmd, 0);
    vkBeginCommandBuffer(cmd, &bi);
    stageImg(*gbuf.irr, irrData, n * 2u * sizeof(uint32_t));
    stageImg(*gbuf.primGuide, primData, n * 4u * sizeof(uint32_t));
    stageImg(*gbuf.surfaceMv, surfMvData, n * sizeof(uint32_t));
    stageImg(*gbuf.diffHitDist, diffHitData, n);
    stageImg(*gbuf.specVmv, specVmvData, n * 2u * sizeof(uint32_t));
    stageImg(*gbuf.skipCompact, skipData, n * sizeof(uint32_t));

    VkMemoryBarrier2 mb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    mb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
    mb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
    mb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.memoryBarrierCount = 1;
    dep.pMemoryBarriers = &mb;
    vkCmdPipelineBarrier2(cmd, &dep);
    vkEndCommandBuffer(cmd);

    VkSubmitInfo sub{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    sub.commandBufferCount = 1;
    sub.pCommandBuffers = &cmd;
    check_true(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE) == VK_SUCCESS, "Upload raw G-buffer set");
    vkQueueWaitIdle(queue);
    (void)device;
}

GpuGbufferSet makeEmptyCompactGbufferSet(
    VkDevice device,
    VmaAllocator allocator,
    uint32_t width,
    uint32_t height
) {
    constexpr VkImageUsageFlags usage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    GpuGbufferSet g;
    g.width = width;
    g.height = height;
    g.irr = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R32G32_UINT, usage);
    g.primGuide = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R32G32B32A32_UINT, usage);
    g.surfaceMv = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R32_UINT, usage);
    g.diffHitDist = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R8_UNORM, usage);
    g.specVmv = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R32G32_UINT, usage);
    g.skipCompact = std::make_unique<Image>(device, allocator, width, height, VK_FORMAT_R32_UINT, usage);
    return g;
}

GpuGbufferSet createSkyDielectricEdgeGbufferSet(
    VkDevice device,
    VmaAllocator allocator,
    VkQueue queue,
    VkCommandBuffer cmd,
    const glm::vec3& skyRgb
) {
    constexpr uint32_t W_E = 64u;
    constexpr uint32_t H_E = 64u;
    GpuGbufferSet gEdge = makeEmptyCompactGbufferSet(device, allocator, W_E, H_E);

    const size_t nEdge = static_cast<size_t>(W_E) * H_E;
    std::vector<uint32_t> irrE(nEdge * 2u, 0u);
    std::vector<uint32_t> primE(nEdge * 4u, 0u);
    std::vector<uint32_t> surfMvE(nEdge, 0u);
    std::vector<uint8_t> diffHdE(nEdge, 128u);
    std::vector<uint32_t> specVmvE(nEdge * 2u, 0u);
    std::vector<uint32_t> skipE(nEdge, 0u);

    const glm::vec3 dielDiffAlb(0.5f);
    const glm::vec3 dielSpecAlb(0.1f);
    const glm::vec3 dielSkip(0.1f);

    for (uint32_t y = 0; y < H_E; ++y) {
        const bool isSkyRow = (y < 32u);
        for (uint32_t x = 0; x < W_E; ++x) {
            const size_t idx = static_cast<size_t>(y) * W_E + x;
            primE[idx * 4u + 0u] = packOct32Cpu(glm::vec3(0.0f, 0.0f, 1.0f));
            if (isSkyRow) {
                primE[idx * 4u + 1u] = glm::packHalf2x16(glm::vec2(10000.0f, 0.0f)) & 0xFFFFu;
                primE[idx * 4u + 2u] = glm::packUnorm4x8(glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
                primE[idx * 4u + 3u] = 0u;
                skipE[idx] = packRGB9E5Cpu(skyRgb);
            } else {
                const uint32_t dHalf = glm::packHalf2x16(glm::vec2(2.5f, 0.0f)) & 0xFFFFu;
                primE[idx * 4u + 1u] = dHalf | (1u << 16u); // primaryTransmissive bit 16
                primE[idx * 4u + 2u] = glm::packUnorm4x8(glm::vec4(dielDiffAlb, 0.05f));
                primE[idx * 4u + 3u] = glm::packUnorm4x8(glm::vec4(dielSpecAlb, 2.0f / 3.0f));
                irrE[idx * 2u + 0u] = packRGB9E5Cpu(dielDiffAlb * 1.0f);
                irrE[idx * 2u + 1u] = packRGB9E5Cpu(dielSpecAlb * 0.2f);
                surfMvE[idx] = glm::packHalf2x16(glm::vec2(0.01f, -0.02f));
                specVmvE[idx * 2u + 0u] = glm::packHalf2x16(glm::vec2(0.0f));
                specVmvE[idx * 2u + 1u] = 0u; // force specVmv.y == 0 fallback
                skipE[idx] = packRGB9E5Cpu(dielSkip);
            }
        }
    }

    uploadGbufferRawBuffers(
        device, allocator, queue, cmd, gEdge,
        irrE.data(), primE.data(), surfMvE.data(), diffHdE.data(), specVmvE.data(), skipE.data()
    );
    return gEdge;
}

uint32_t countHardwareVulkan14Gpus() {
    uint32_t hwGpuCount = 0;
    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.apiVersion = VK_API_VERSION_1_4;
    VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instInfo.pApplicationInfo = &appInfo;
    VkInstance probeInst = VK_NULL_HANDLE;
    if (vkCreateInstance(&instInfo, nullptr, &probeInst) == VK_SUCCESS) {
        uint32_t devCount = 0;
        vkEnumeratePhysicalDevices(probeInst, &devCount, nullptr);
        std::vector<VkPhysicalDevice> devs(devCount);
        vkEnumeratePhysicalDevices(probeInst, &devCount, devs.data());
        for (VkPhysicalDevice pd : devs) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(pd, &props);
            if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU && props.apiVersion >= VK_API_VERSION_1_4) {
                ++hwGpuCount;
            }
        }
        vkDestroyInstance(probeInst, nullptr);
    }
    return hwGpuCount;
}

int runV5ParityCli(int argc, char** argv) {
    // --v5-parity <weights_path> <fp8|int8> <inW> <inH> <outW> <outH> <frames> <in_bin> <out_bin>
    if (argc < 11) {
        std::cerr << "Usage: --v5-parity <weights_path> <fp8|int8> <inW> <inH> <outW> <outH> <frames> <in_bin> <out_bin>\n";
        return 2;
    }
    const std::string weightsPath = argv[2];
    const std::string precStr = argv[3];
    const uint32_t inW = static_cast<uint32_t>(std::stoul(argv[4]));
    const uint32_t inH = static_cast<uint32_t>(std::stoul(argv[5]));
    const uint32_t outW = static_cast<uint32_t>(std::stoul(argv[6]));
    const uint32_t outH = static_cast<uint32_t>(std::stoul(argv[7]));
    const uint32_t frames = static_cast<uint32_t>(std::stoul(argv[8]));
    const std::string inBinPath = argv[9];
    const std::string outBinPath = argv[10];

    const UpwaysComputePrecision prec = (precStr == "int8")
        ? UpwaysComputePrecision::Int8
        : UpwaysComputePrecision::Fp8;
    const std::string ext = (prec == UpwaysComputePrecision::Int8) ? ".int8.comp.spv" : ".comp.spv";
    const auto prepSpv = loadSpv("upways_v5_prep" + ext);
    const auto unetSpv = loadSpv("upways_v5_unet_wmma" + ext);
    const auto affSpv  = loadSpv("upways_v5_affinity" + ext);
    const auto upSpv   = loadSpv("upways_v5_upscaler_wmma" + ext);
    const auto resSpv  = loadSpv("upways_v5_resolve" + ext);
    if (prepSpv.empty() || unetSpv.empty() || affSpv.empty() || upSpv.empty() || resSpv.empty()) {
        std::cerr << "Missing Upways 5.0 SPIR-Vs for precision " << precStr << "\n";
        return 2;
    }

    const uint32_t hwGpuCount = countHardwareVulkan14Gpus();
    if (hwGpuCount == 0u) {
        std::cerr << "No Vulkan 1.4 hardware GPU found\n";
        return 2;
    }

    Config cfg;
    cfg.headless = true;
    cfg.validation_layers = true;
    cfg.gpu_index = (hwGpuCount >= 2u) ? 1u : 0u;
    VulkanContext vkCtx(cfg, VK_NULL_HANDLE, "Upways 5.0 Parity CLI");
    VkDevice device = vkCtx.getDevice();
    VmaAllocator allocator = vkCtx.getAllocator();
    VkQueue queue = vkCtx.getGraphicsQueue();

    VkCommandPoolCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpInfo.queueFamilyIndex = vkCtx.getGraphicsQueueFamily();
    VkCommandPool cmdPool = VK_NULL_HANDLE;
    if (vkCreateCommandPool(device, &cpInfo, nullptr, &cmdPool) != VK_SUCCESS) return 2;

    VkCommandBufferAllocateInfo cbAlloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbAlloc.commandPool = cmdPool;
    cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbAlloc.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    if (vkAllocateCommandBuffers(device, &cbAlloc, &cmd) != VK_SUCCESS) return 2;

    std::ifstream inF(inBinPath, std::ios::binary);
    std::ofstream outF(outBinPath, std::ios::binary | std::ios::trunc);
    if (!inF.is_open() || !outF.is_open()) {
        std::cerr << "Failed to open input/output binary files\n";
        return 2;
    }

    const size_t nIn = static_cast<size_t>(inW) * inH;
    const size_t nOut = static_cast<size_t>(outW) * outH;
    std::vector<uint32_t> irr(nIn * 2u);
    std::vector<uint32_t> prim(nIn * 4u);
    std::vector<uint32_t> surfMv(nIn);
    std::vector<uint8_t>  diffHd(nIn);
    std::vector<uint32_t> specVmv(nIn * 2u);
    std::vector<uint32_t> skip(nIn);
    std::vector<float>    outRgb(nOut * 3u);

    {
        GpuGbufferSet g = makeEmptyCompactGbufferSet(device, allocator, inW, inH);
        const bool enableSr = (outW != inW || outH != inH);
        UpwaysPipeline pipe(
            device, vkCtx.getPhysicalDevice(), allocator,
            inW, inH, outW, outH,
            prepSpv, unetSpv, affSpv, upSpv, resSpv,
            weightsPath, enableSr, VK_FORMAT_R16G16B16A16_SFLOAT, prec
        );
        pipe.updateDescriptors(
            g.irr->getImageView(), g.primGuide->getImageView(), g.surfaceMv->getImageView(),
            g.diffHitDist->getImageView(), g.specVmv->getImageView(), g.skipCompact->getImageView()
        );
        Buffer readback(
            allocator, nOut * 4ULL * sizeof(uint16_t),
            VK_BUFFER_USAGE_TRANSFER_DST_BIT,
            VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
        );

        VkCommandBufferBeginInfo bi{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;

        for (uint32_t f = 0; f < frames; ++f) {
            float jit[2]{};
            inF.read(reinterpret_cast<char*>(jit), sizeof(jit));
            inF.read(reinterpret_cast<char*>(irr.data()), static_cast<std::streamsize>(irr.size() * sizeof(uint32_t)));
            inF.read(reinterpret_cast<char*>(prim.data()), static_cast<std::streamsize>(prim.size() * sizeof(uint32_t)));
            inF.read(reinterpret_cast<char*>(surfMv.data()), static_cast<std::streamsize>(surfMv.size() * sizeof(uint32_t)));
            inF.read(reinterpret_cast<char*>(diffHd.data()), static_cast<std::streamsize>(diffHd.size()));
            inF.read(reinterpret_cast<char*>(specVmv.data()), static_cast<std::streamsize>(specVmv.size() * sizeof(uint32_t)));
            inF.read(reinterpret_cast<char*>(skip.data()), static_cast<std::streamsize>(skip.size() * sizeof(uint32_t)));
            if (!inF) {
                std::cerr << "Truncated input binary at frame " << f << "\n";
                return 2;
            }

            uploadGbufferRawBuffers(
                device, allocator, queue, cmd, g,
                irr.data(), prim.data(), surfMv.data(), diffHd.data(), specVmv.data(), skip.data()
            );

            vkResetCommandBuffer(cmd, 0);
            vkBeginCommandBuffer(cmd, &bi);
            pipe.recordFrame(cmd, f, f == 0u, glm::vec2(jit[0], jit[1]), false, false, 1.0f);

            VkMemoryBarrier2 outBar{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
            outBar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            outBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            outBar.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            outBar.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            dep.memoryBarrierCount = 1;
            dep.pMemoryBarriers = &outBar;
            vkCmdPipelineBarrier2(cmd, &dep);

            VkBufferImageCopy reg{};
            reg.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            reg.imageExtent = { outW, outH, 1u };
            vkCmdCopyImageToBuffer(cmd, pipe.getOutputImage()->getImage(), VK_IMAGE_LAYOUT_GENERAL, readback.getBuffer(), 1, &reg);
            vkEndCommandBuffer(cmd);

            VkSubmitInfo sub{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            sub.commandBufferCount = 1;
            sub.pCommandBuffers = &cmd;
            if (vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE) != VK_SUCCESS) return 2;
            vkQueueWaitIdle(queue);

            readback.invalidate();
            const uint16_t* h = static_cast<const uint16_t*>(readback.map());
            if (!h) return 2;
            for (size_t i = 0; i < nOut; ++i) {
                outRgb[i * 3u + 0u] = glm::unpackHalf1x16(h[i * 4u + 0u]);
                outRgb[i * 3u + 1u] = glm::unpackHalf1x16(h[i * 4u + 1u]);
                outRgb[i * 3u + 2u] = glm::unpackHalf1x16(h[i * 4u + 2u]);
            }
            readback.unmap();
            outF.write(reinterpret_cast<const char*>(outRgb.data()), static_cast<std::streamsize>(outRgb.size() * sizeof(float)));
        }
        vkDeviceWaitIdle(device);
    }

    vkDestroyCommandPool(device, cmdPool, nullptr);
    if (vkCtx.getValidationErrors() != 0u) {
        std::cerr << "Vulkan validation errors in --v5-parity: " << vkCtx.getValidationErrors() << "\n";
        return 1;
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "--v5-parity") {
        return runV5ParityCli(argc, argv);
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Upways 4.0/5.0 Pipeline Unit & GPU Test       " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Push Constant Layout & Flags (80 B immutable ABI)
    std::cout << "[TEST 1] UpwaysPushConstants ABI (80 B) & Flags..." << std::endl;
    check_true(sizeof(UpwaysPushConstants) == 80, "UpwaysPushConstants must be 80 bytes");
    check_true(offsetof(UpwaysPushConstants, lrWidth) == 0, "lrWidth offset == 0");
    check_true(offsetof(UpwaysPushConstants, lrHeight) == 4, "lrHeight offset == 4");
    check_true(offsetof(UpwaysPushConstants, hrWidth) == 8, "hrWidth offset == 8");
    check_true(offsetof(UpwaysPushConstants, hrHeight) == 12, "hrHeight offset == 12");
    check_true(offsetof(UpwaysPushConstants, jitterX) == 16, "jitterX offset == 16");
    check_true(offsetof(UpwaysPushConstants, jitterY) == 20, "jitterY offset == 20");
    check_true(offsetof(UpwaysPushConstants, frameIndex) == 24, "frameIndex offset == 24");
    check_true(offsetof(UpwaysPushConstants, flags) == 28, "flags offset == 28");
    check_true(offsetof(UpwaysPushConstants, offX0) == 32, "offX0 offset == 32");
    check_true(offsetof(UpwaysPushConstants, offHw0) == 36, "offHw0 offset == 36");
    check_true(offsetof(UpwaysPushConstants, offS0) == 40, "offS0 offset == 40");
    check_true(offsetof(UpwaysPushConstants, offB1) == 44, "offB1 offset == 44");
    check_true(offsetof(UpwaysPushConstants, offB2) == 48, "offB2 offset == 48");
    check_true(offsetof(UpwaysPushConstants, exposureScale) == 52, "exposureScale offset == 52");
    check_true(offsetof(UpwaysPushConstants, bandLrY0) == 56, "bandLrY0 offset == 56");
    check_true(offsetof(UpwaysPushConstants, bandHrY0) == 60, "bandHrY0 offset == 60");
    check_true(offsetof(UpwaysPushConstants, fullLrHeight) == 64, "fullLrHeight offset == 64");
    check_true(offsetof(UpwaysPushConstants, fullHrHeight) == 68, "fullHrHeight offset == 68");
    check_true(offsetof(UpwaysPushConstants, keepHrYStart) == 72, "keepHrYStart offset == 72");
    check_true(offsetof(UpwaysPushConstants, keepHrYEnd) == 76, "keepHrYEnd offset == 76");
    check_true(UPWAYS_FLAG_RESET_HISTORY == 1u, "UPWAYS_FLAG_RESET_HISTORY == 1");
    check_true(UPWAYS_FLAG_ENABLE_SUPERRES == 2u, "UPWAYS_FLAG_ENABLE_SUPERRES == 2");
    check_true(UPWAYS_FLAG_ZERO_RESIDUAL_MODE == 4u, "UPWAYS_FLAG_ZERO_RESIDUAL_MODE == 4");
    check_true(UPWAYS_FLAG_FUSED_BAND_PACK == 8u, "UPWAYS_FLAG_FUSED_BAND_PACK == 8");
    check_true(UPWAYS_DEFAULT_BAND_HALO_LR == 24u, "UPWAYS_DEFAULT_BAND_HALO_LR == 24");

    // 2. Log1p/Expm1, Demodulation/Remodulation, RGB9E5 Roundtrip & BandSplit Invariants
    std::cout << "[TEST 2] safeLog1p / safeExpm1, RGB9E5 Roundtrip & BandSplit..." << std::endl;
    {
        for (float v : { 0.0f, 0.01f, 0.25f, 1.0f, 12.5f, 500.0f, 5000.0f }) {
            const float rt = upwaysSafeExpm1(upwaysSafeLog1p(v));
            check_true(std::abs(rt - v) <= 1e-3f * (1.0f + v), "safeLog1p/safeExpm1 roundtrip mismatch");
        }
        check_true(upwaysSafeLog1p(-5.0f) == 0.0f, "safeLog1p clamps negative inputs to 0");
        check_true(upwaysSafeExpm1(-5.0f) == 0.0f, "safeExpm1 clamps negative inputs to 0");
        check_true(std::abs(upwaysSafeLog1p(1e6f) - std::log1p(65504.0f)) < 1e-4f,
                   "safeLog1p upper clamp matches FP16 max (65504)");

        const glm::vec3 diffAlb(0.6f, 0.4f, 0.2f);
        const glm::vec3 specAlb(0.08f, 0.08f, 0.12f);
        const glm::vec3 diffIrr = upwaysDemodulate(glm::vec3(1.2f, 0.8f, 0.4f), diffAlb);
        const glm::vec3 specIrr(0.5f, 0.25f, 0.1f);
        const glm::vec3 skip(0.1f, 0.05f, 0.02f);
        const glm::vec3 remod = upwaysRemodulate(diffAlb, specAlb, diffIrr, specIrr, skip);
        const glm::vec3 expected(1.2f + 0.08f * 0.5f + 0.1f, 0.8f + 0.08f * 0.25f + 0.05f, 0.4f + 0.12f * 0.1f + 0.02f);
        check_true(glm::length(remod - expected) < 1e-5f, "upwaysDemodulate/upwaysRemodulate must conserve radiance");

        check_true(packRGB9E5Cpu(glm::vec3(0.0f)) == 0u, "packRGB9E5 zero -> 0");
        check_true(packRGB9E5Cpu(glm::vec3(-1.0f, std::numeric_limits<float>::quiet_NaN(), -0.5f)) == 0u,
                   "packRGB9E5 sanitizes NaN/negative to 0");
        for (const glm::vec3 c : {
            glm::vec3(0.001f, 0.002f, 0.003f),
            glm::vec3(0.8f, 0.5f, 0.25f),
            glm::vec3(12.3f, 45.6f, 0.4f),
            glm::vec3(65408.0f, 32704.0f, 16352.0f)
        }) {
            const glm::vec3 rt = unpackRGB9E5Cpu(packRGB9E5Cpu(c));
            const float tol = 0.003f * std::max({ c.r, c.g, c.b, 1e-4f });
            check_true(glm::length(rt - c) <= tol * 2.0f, "packRGB9E5 roundtrip within 9-bit mantissa precision");
        }
        const glm::vec3 clampedHi = unpackRGB9E5Cpu(packRGB9E5Cpu(glm::vec3(1e9f, 50.0f, 0.0f)));
        check_true(std::isfinite(clampedHi.r) && clampedHi.r <= 65408.0f + 128.0f, "packRGB9E5 clamps large HDR cleanly");

        const UpwaysBandSplit s4kQ = computeUpwaysBandSplit(1080u, 2160u, 32u);
        check_true(s4kQ.topLrH == s4kQ.splitLrY + 32u && s4kQ.botLrY0 == s4kQ.splitLrY - 32u,
                   "4K Quality band split maintains 32-row LR halo");
        check_true(s4kQ.botHrY0 + s4kQ.botHrH == 2160u && s4kQ.topHrH >= s4kQ.splitHrY && s4kQ.botHrY0 <= s4kQ.splitHrY,
                   "4K Quality HR bands cover splitHrY with overlap");

        const UpwaysBandSplit s4kDef = computeUpwaysBandSplit(1080u, 2160u);
        check_true(s4kDef.haloLr == 24u && (s4kDef.botLrY0 % 4u) == 0u,
                   "Default BandSplit uses 24-row L2-aligned LR halo");
        const UpwaysBandSplit s4k16 = computeUpwaysBandSplit(1080u, 2160u, 16u);
        check_true(s4k16.haloLr == 16u && s4k16.topLrH == s4k16.splitLrY + 16u && s4k16.botLrY0 == s4k16.splitLrY - 16u,
                   "Configurable 16-row LR halo applied cleanly");
        const UpwaysBandSplit s4kAlign = computeUpwaysBandSplit(1080u, 2160u, 18u, 0.90f);
        check_true(s4kAlign.haloLr == 20u && (s4kAlign.botLrY0 % 4u) == 0u,
                   "Unaligned haloLr=18 rounds to multiple-of-4 (20)");
        check_true(static_cast<double>(s4kAlign.splitLrY) / 1080.0 <= 0.76,
                   "topFraction clamps safely to [0.25, 0.75]");
    }

    // 3. Immutable 24,064-B Weight Contract (C++ Headers + On-Disk Binary)
    std::cout << "[TEST 3] Upways 4.0 Weight Contract (24,064 B / 17,392 params)..." << std::endl;
    check_true(upways::FP8_WEIGHT_BYTES == 15616u, "FP8_WEIGHT_BYTES == 15616");
    check_true(upways::FP32_WEIGHT_BYTES == 8448u, "FP32_WEIGHT_BYTES == 8448");
    check_true(upways::TOTAL_WEIGHT_BYTES == 24064u, "TOTAL_WEIGHT_BYTES == 24064");
    check_true(upways::TOTAL_WEIGHT_BUFFER_SIZE == 24064u, "TOTAL_WEIGHT_BUFFER_SIZE == 24064");
    check_true(upways::ACTIVE_PARAMETER_COUNT == 17392u, "ACTIVE_PARAMETER_COUNT == 17392");
    check_true(sizeof(upways::DEFAULT_WEIGHTS) == 24064u, "sizeof(DEFAULT_WEIGHTS) == 24064");
    check_true(upways::EMBEDDED_WEIGHTS_SIZE == 24064u, "EMBEDDED_WEIGHTS_SIZE == 24064");
    check_true(sizeof(upways::DEFAULT_UPWAYS_WEIGHTS) == 24064u, "sizeof(DEFAULT_UPWAYS_WEIGHTS) == 24064");
    check_true(upways::EMBEDDED_INT8_WEIGHTS_SIZE == 24064u, "EMBEDDED_INT8_WEIGHTS_SIZE == 24064");
    check_true(sizeof(upways::DEFAULT_UPWAYS_INT8_WEIGHTS) == 24064u, "sizeof(DEFAULT_UPWAYS_INT8_WEIGHTS) == 24064");
    {
        namespace fs = std::filesystem;
        auto checkWeightFile = [](const std::string& filename, const char* errMsg) {
            const std::vector<fs::path> weightCandidates = {
                fs::path("data/models") / filename,
                fs::path("../data/models") / filename,
                fs::path("../../data/models") / filename,
                fs::path("../../../data/models") / filename,
            };
            bool foundWeightFile = false;
            for (const auto& wp : weightCandidates) {
                if (fs::exists(wp)) {
                    foundWeightFile = true;
                    check_true(fs::file_size(wp) == upways::TOTAL_WEIGHT_BYTES, errMsg);
                    break;
                }
            }
            check_true(foundWeightFile, (filename + " found in expected data/models path").c_str());
        };
        checkWeightFile("upways_weights.bin", "On-disk upways_weights.bin must be exactly 24,064 bytes");
        checkWeightFile("upways_weights_int8.bin", "On-disk upways_weights_int8.bin must be exactly 24,064 bytes");
    }

    // 4. U-Net Pyramid Dimensions & 256-B Alignment
    std::cout << "[TEST 4] U-Net Pyramid Dimensions & 256-B Buffer Offsets..." << std::endl;
    {
        auto p540 = computeUpwaysPyramidDims(960u, 540u);
        check_true(p540.w0 == 960u && p540.h0 == 540u, "960x540 level 0");
        check_true(p540.w1 == 480u && p540.h1 == 270u, "960x540 level 1 == 480x270");
        check_true(p540.w2 == 240u && p540.h2 == 135u, "960x540 level 2 == 240x135");
        check_true((p540.offX0 % 64u) == 0u && (p540.offHw0 % 64u) == 0u &&
                   (p540.offS0 % 64u) == 0u && (p540.offB1 % 64u) == 0u &&
                   (p540.offB2 % 64u) == 0u && (p540.totalUints % 64u) == 0u,
                   "All pyramid activation offsets must be 256-B (64-uint) aligned");

        auto p1080 = computeUpwaysPyramidDims(1920u, 1080u);
        check_true(p1080.w1 == 960u && p1080.h1 == 540u && p1080.w2 == 480u && p1080.h2 == 270u,
                   "1920x1080 pyramid dims");

        auto pOdd = computeUpwaysPyramidDims(240u, 135u);
        check_true(pOdd.w1 == 120u && pOdd.h1 == 68u, "240x135 ceil half level 1 == 120x68");
        check_true(pOdd.w2 == 60u && pOdd.h2 == 34u, "240x135 ceil half level 2 == 60x34");
    }

    // 5. Pipeline Constants & LDS Budgets
    std::cout << "[TEST 5] Pipeline Constants & LDS Budgets..." << std::endl;
    static_assert(UpwaysPipeline::MAX_QUERY_SLOTS == 2, "MAX_QUERY_SLOTS == 2");
    static_assert(UpwaysPipeline::QUERIES_PER_SLOT == 5, "QUERIES_PER_SLOT == 5");
    static_assert(UpwaysPipeline::NUM_DENOISER_STAGES == 4, "NUM_DENOISER_STAGES == 4");
    static_assert(UpwaysPipeline::BASE_ANCHOR_LAYERS == 4, "BASE_ANCHOR_LAYERS == 4");
    {
        const uint32_t ldsPrep = 5u * 180u * 8u;                  // s_diff, s_spec, s_nrmDep, s_logDiff, s_logSpec
        const uint32_t ldsDenoiser8Bit = (3072u + 1024u) * 4u;    // s_buf + s_acc (FP8 / INT8)
        const uint32_t ldsDenoiserFp16 = (6144u + 1024u) * 4u;    // s_buf + s_acc (FP16)
        const uint32_t ldsUpscaler8Bit = 1024u * 8u + 1024u * 4u + 1024u * 4u;
        const uint32_t ldsUpscalerFp16 = 1024u * 8u + 2048u * 4u + 1024u * 4u;

        check_true(ldsPrep == 7200u && ldsPrep <= 32768u, "denoise_prep LDS == 7,200 B <= 32 KiB");
        check_true(ldsDenoiser8Bit == 16384u && ldsDenoiser8Bit <= 32768u, "denoiser_wmma 8-bit LDS == 16,384 B <= 32 KiB");
        check_true(ldsDenoiserFp16 == 28672u && ldsDenoiserFp16 <= 32768u, "denoiser_wmma FP16 LDS == 28,672 B <= 32 KiB");
        check_true(ldsUpscaler8Bit == 16384u && ldsUpscaler8Bit <= 32768u, "upscaler_wmma 8-bit LDS == 16,384 B <= 32 KiB");
        check_true(ldsUpscalerFp16 == 20480u && ldsUpscalerFp16 <= 32768u, "upscaler_wmma FP16 LDS == 20,480 B <= 32 KiB");
    }

    // 6. Config CLI Parsing
    std::cout << "[TEST 6] Config CLI Parsing (Upways + PTTD v3/v4 Defaults)..." << std::endl;
    {
        Config defCfg;
        check_true(defCfg.capture_channels == 24u, "Default Config::capture_channels == 24 preserved");
        check_true(defCfg.denoiser_mode == DenoiserMode::None, "Default DenoiserMode == None");
        check_true(defCfg.upways_precision == UpwaysPrecision::Auto, "Default UpwaysPrecision == Auto");
        check_true(defCfg.upways_band_halo == 24u, "Default upways_band_halo == 24");
        check_true(defCfg.upways_dynamic_band_balance, "Default upways_dynamic_band_balance == true");
        check_true(!defCfg.upways_superres && !defCfg.upways_heuristic_fallback, "Default Upways SR/heuristic disabled");

        const char* argv1[] = { "pathways", "--denoiser", "upways", "--capture-channels", "32", "--upways-precision", "auto", "--upways-band-halo", "16" };
        Config c1 = Config::parse(9, const_cast<char**>(argv1));
        check_true(c1.denoiser_mode == DenoiserMode::Upways, "--denoiser upways selects Upways");
        check_true(c1.upways_precision == UpwaysPrecision::Auto, "--upways-precision auto selects Auto");
        check_true(c1.upways_band_halo == 16u, "--upways-band-halo 16 sets 16");
        check_true(!c1.upways_superres, "--denoiser upways leaves upways_superres false");
        check_true(c1.capture_channels == 32u, "--capture-channels 32 accepted");

        const char* argv2[] = { "pathways", "--scaler", "upways", "--upways-heuristic", "--upways-precision", "fp8", "--upways-band-halo=18", "--no-upways-band-balance" };
        Config c2 = Config::parse(8, const_cast<char**>(argv2));
        check_true(c2.denoiser_mode == DenoiserMode::Upways, "--scaler upways enables Upways denoiser");
        check_true(c2.upscaler_mode == UpscalerMode::Upways, "--scaler upways selects UpscalerMode::Upways");
        check_true(c2.upways_superres, "--scaler upways sets upways_superres true");
        check_true(c2.upways_heuristic_fallback, "--upways-heuristic enables fallback flag");
        check_true(c2.upways_precision == UpwaysPrecision::Fp8, "--upways-precision fp8 selects Fp8");
        check_true(c2.upways_band_halo == 20u, "--upways-band-halo=18 aligns to 20");
        check_true(!c2.upways_dynamic_band_balance, "--no-upways-band-balance disables dynamic balance");

        const char* argv3[] = { "pathways", "--upways-precision", "int8" };
        Config c3 = Config::parse(3, const_cast<char**>(argv3));
        check_true(c3.upways_precision == UpwaysPrecision::Int8, "--upways-precision int8 selects Int8");

        const char* argv4[] = { "pathways", "--upways-precision", "fp16" };
        Config c4 = Config::parse(3, const_cast<char**>(argv4));
        check_true(c4.upways_precision == UpwaysPrecision::Fp16, "--upways-precision fp16 selects Fp16");

        const char* argv5[] = { "pathways", "--upways-precision=int8" };
        Config c5 = Config::parse(2, const_cast<char**>(argv5));
        check_true(c5.upways_precision == UpwaysPrecision::Int8, "--upways-precision=int8 selects Int8");

        check_true(!c1.isUpwaysSuperResActive(), "--denoiser upways leaves isUpwaysSuperResActive false");
        check_true(c2.isUpwaysSuperResActive(), "--scaler upways sets isUpwaysSuperResActive true");

        const char* argv6[] = { "pathways", "--scaler", "upways", "--render-scale", "1.0" };
        Config c6 = Config::parse(5, const_cast<char**>(argv6));
        check_true(c6.isUpwaysSuperResActive(), "--scaler upways --render-scale 1.0 sets isUpwaysSuperResActive true");

        const char* argv7[] = { "pathways", "--denoiser", "upways", "--scaler", "fsr", "quality" };
        Config c7 = Config::parse(6, const_cast<char**>(argv7));
        check_true(!c7.isUpwaysSuperResActive(), "--denoiser upways --scaler fsr quality sets isUpwaysSuperResActive false");
    }

    // 7. SPIR-V Shader Validation (FP8 + INT8 + FP16 + Band-Parallel / Fused Tonemap Variants)
    std::cout << "[TEST 7] SPIR-V Header Verification (15 Shaders)..." << std::endl;
    const char* shaderNames[] = {
        "upways_denoise_prep.comp.spv",
        "upways_denoiser_wmma.comp.spv",
        "upways_upscaler_wmma.comp.spv",
        "upways_upscale_resolve.comp.spv",
        "upways_denoise_prep.int8.comp.spv",
        "upways_denoiser_wmma.int8.comp.spv",
        "upways_upscaler_wmma.int8.comp.spv",
        "upways_upscale_resolve.int8.comp.spv",
        "upways_denoise_prep.fp16.comp.spv",
        "upways_denoiser_wmma.fp16.comp.spv",
        "upways_upscaler_wmma.fp16.comp.spv",
        "upways_upscale_resolve.fp16.comp.spv",
        "upways_band_pack.comp.spv",
        "upways_band_stitch.comp.spv",
        "tonemap_aces.comp.spv",
    };
    for (const char* sname : shaderNames) {
        const auto spv = loadSpv(sname);
        if (!validateSpirv(spv)) {
            std::cerr << "[FAIL] Invalid/missing SPIR-V: " << sname << std::endl;
            ++g_failures;
        }
    }

    // 9. Upways 5.0 Pure-ML Metadata, CLI, Version Resolution & 10 SPIR-Vs
    std::cout << "[TEST 9] Upways 5.0 Metadata, CLI, resolveVersion & 10 SPIR-Vs..." << std::endl;
    check_true(upways_v5::FP8_WEIGHT_BYTES == 150016u, "upways_v5::FP8_WEIGHT_BYTES == 150016");
    check_true(upways_v5::INT8_WEIGHT_BYTES == 150016u, "upways_v5::INT8_WEIGHT_BYTES == 150016");
    check_true(upways_v5::FP32_WEIGHT_BYTES == 40192u, "upways_v5::FP32_WEIGHT_BYTES == 40192");
    check_true(upways_v5::TOTAL_WEIGHT_BYTES == 190208u, "upways_v5::TOTAL_WEIGHT_BYTES == 190208");
    check_true(upways_v5::ACTIVE_PARAMETER_COUNT == 158400u, "upways_v5::ACTIVE_PARAMETER_COUNT == 158400");
    check_true(sizeof(upways_v5::DEFAULT_UPWAYS_V5_WEIGHTS) == 190208u, "sizeof(DEFAULT_UPWAYS_V5_WEIGHTS) == 190208");
    check_true(sizeof(upways_v5::DEFAULT_UPWAYS_V5_INT8_WEIGHTS) == 190208u, "sizeof(DEFAULT_UPWAYS_V5_INT8_WEIGHTS) == 190208");
    {
        namespace fs = std::filesystem;
        auto findDataFile = [](const std::string& filename) -> fs::path {
            const std::vector<fs::path> candidates = {
                fs::path("data/models") / filename,
                fs::path("../data/models") / filename,
                fs::path("../../data/models") / filename,
                fs::path("../../../data/models") / filename,
            };
            for (const auto& p : candidates) {
                if (fs::exists(p)) return p;
            }
            return {};
        };
        const fs::path v4Path = findDataFile("upways_weights.bin");
        const fs::path v5Fp8Path = findDataFile("upways_v5_weights.bin");
        const fs::path v5Int8Path = findDataFile("upways_v5_weights_int8.bin");
        check_true(!v5Fp8Path.empty() && fs::file_size(v5Fp8Path) == upways_v5::TOTAL_WEIGHT_BYTES,
                   "upways_v5_weights.bin exists with exact 190,208 B");
        check_true(!v5Int8Path.empty() && fs::file_size(v5Int8Path) == upways_v5::TOTAL_WEIGHT_BYTES,
                   "upways_v5_weights_int8.bin exists with exact 190,208 B");

        Config defV;
        check_true(defV.upways_version == UpwaysVersion::Auto, "Default Config::upways_version == Auto");
        const char* avAuto[] = { "pathways", "--upways-version", "auto" };
        const char* av4[]    = { "pathways", "--upways-version", "4" };
        const char* avV4[]   = { "pathways", "--upways-version=v4" };
        const char* av5[]    = { "pathways", "--upways-version", "5" };
        const char* avV5[]   = { "pathways", "--upways-version=v5" };
        check_true(Config::parse(3, const_cast<char**>(avAuto)).upways_version == UpwaysVersion::Auto, "--upways-version auto");
        check_true(Config::parse(3, const_cast<char**>(av4)).upways_version == UpwaysVersion::V4, "--upways-version 4");
        check_true(Config::parse(2, const_cast<char**>(avV4)).upways_version == UpwaysVersion::V4, "--upways-version=v4");
        check_true(Config::parse(3, const_cast<char**>(av5)).upways_version == UpwaysVersion::V5, "--upways-version 5");
        check_true(Config::parse(2, const_cast<char**>(avV5)).upways_version == UpwaysVersion::V5, "--upways-version=v5");

        check_true(UpwaysPipeline::resolveVersion(UpwaysVersion::V4, "") == UpwaysRuntimeVersion::V4, "resolveVersion(V4)");
        check_true(UpwaysPipeline::resolveVersion(UpwaysVersion::V5, "") == UpwaysRuntimeVersion::V5, "resolveVersion(V5)");
        check_true(UpwaysPipeline::resolveVersion(UpwaysVersion::Auto, "") == UpwaysRuntimeVersion::V5,
                   "resolveVersion(Auto, '') == V5 when upways_v5_weights*.bin is present");
        if (!v4Path.empty()) {
            check_true(UpwaysPipeline::resolveVersion(UpwaysVersion::Auto, v4Path.string()) == UpwaysRuntimeVersion::V4,
                       "resolveVersion(Auto, 24064 B V4 bin) == V4");
        }
        if (!v5Fp8Path.empty()) {
            check_true(UpwaysPipeline::resolveVersion(UpwaysVersion::Auto, v5Fp8Path.string()) == UpwaysRuntimeVersion::V5,
                       "resolveVersion(Auto, 190208 B V5 bin) == V5");
        }
    }
    {
        const char* v5Shaders[] = {
            "upways_v5_prep.comp.spv",
            "upways_v5_unet_wmma.comp.spv",
            "upways_v5_affinity.comp.spv",
            "upways_v5_upscaler_wmma.comp.spv",
            "upways_v5_resolve.comp.spv",
            "upways_v5_prep.int8.comp.spv",
            "upways_v5_unet_wmma.int8.comp.spv",
            "upways_v5_affinity.int8.comp.spv",
            "upways_v5_upscaler_wmma.int8.comp.spv",
            "upways_v5_resolve.int8.comp.spv",
        };
        for (const char* sname : v5Shaders) {
            const auto spv = loadSpv(sname);
            if (!validateSpirv(spv)) {
                std::cerr << "[FAIL] Invalid/missing V5 SPIR-V: " << sname << std::endl;
                ++g_failures;
            }
        }
    }

    // 8. Live Headless GPU Execution, Lossless DRS, Resize & Native 1.0x
    {
        const uint32_t hwGpuCount = countHardwareVulkan14Gpus();

        if (hwGpuCount > 0) {
            std::cout << "[TEST 8] Live GPU Upways 4.0 Pipeline, Lossless DRS & Native 1.0x..." << std::endl;
            Config cfg;
            cfg.headless = true;
            cfg.validation_layers = true;
            cfg.gpu_index = (hwGpuCount >= 2) ? 1u : 0u;

            VulkanContext vkCtx(cfg, VK_NULL_HANDLE, "Upways 4.0 GPU Test");
            VkDevice device = vkCtx.getDevice();
            VmaAllocator allocator = vkCtx.getAllocator();
            VkQueue queue = vkCtx.getGraphicsQueue();

            std::vector<UpwaysComputePrecision> supportedPrecisions;
            if (UpwaysPipeline::supportsFp8CoopMatrix(vkCtx.getPhysicalDevice(), vkCtx.getInstance())) {
                supportedPrecisions.push_back(UpwaysComputePrecision::Fp8);
            }
            if (UpwaysPipeline::supportsInt8CoopMatrix(vkCtx.getPhysicalDevice(), vkCtx.getInstance())) {
                supportedPrecisions.push_back(UpwaysComputePrecision::Int8);
            }
            if (UpwaysPipeline::supportsFp16CoopMatrix(vkCtx.getPhysicalDevice(), vkCtx.getInstance())) {
                supportedPrecisions.push_back(UpwaysComputePrecision::Fp16);
            }
            check_true(!supportedPrecisions.empty(),
                       "At least one Upways WMMA precision tier (FP8/INT8/FP16) must be supported on hardware GPU");
            check_true(vkCtx.supportsUpwaysWmma(), "VulkanContext::supportsUpwaysWmma() matches GPU WMMA capability");

            VkCommandPoolCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            cpInfo.queueFamilyIndex = vkCtx.getGraphicsQueueFamily();
            VkCommandPool cmdPool = VK_NULL_HANDLE;
            check_true(vkCreateCommandPool(device, &cpInfo, nullptr, &cmdPool) == VK_SUCCESS, "Create cmdPool");

            VkCommandBufferAllocateInfo cbAlloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cbAlloc.commandPool = cmdPool;
            cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbAlloc.commandBufferCount = 1;
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            check_true(vkAllocateCommandBuffers(device, &cbAlloc, &cmd) == VK_SUCCESS, "Allocate cmd");

            const VkDeviceSize maxReadBytes = 512ULL * 512ULL * 4ULL * sizeof(uint16_t);
            Buffer readbackBuf(
                allocator, maxReadBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
            );
            Buffer bandReadbackBuf(
                allocator, maxReadBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
            );

            {
                std::vector<std::unique_ptr<Buffer>> stagingKeepAlive;
                VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkResetCommandBuffer(cmd, 0);
                vkBeginCommandBuffer(cmd, &beginInfo);

                const UpwaysBandSplit bandSplit = computeUpwaysBandSplit(256u, 512u, 24u);
                const UpwaysBandSplit bandSplit1x = computeUpwaysBandSplit(256u, 256u, 24u);
                GpuGbufferSet g64 = createPopulatedGbufferSet(device, allocator, cmd, 64u, 64u, stagingKeepAlive);
                GpuGbufferSet g96 = createPopulatedGbufferSet(device, allocator, cmd, 96u, 96u, stagingKeepAlive);
                GpuGbufferSet g256Full = createPopulatedGbufferSet(device, allocator, cmd, 256u, 256u, stagingKeepAlive, true, 0u, 256u);
                GpuGbufferSet g256Top  = createPopulatedGbufferSet(device, allocator, cmd, 256u, bandSplit.topLrH, stagingKeepAlive, true, 0u, 256u);
                GpuGbufferSet g256Bot  = createPopulatedGbufferSet(device, allocator, cmd, 256u, bandSplit.botLrH, stagingKeepAlive, true, bandSplit.botLrY0, 256u);

                constexpr VkImageUsageFlags kTmUsage =
                    VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                Image tmFullOut(device, allocator, 512u, 512u, VK_FORMAT_R16G16B16A16_SFLOAT, kTmUsage);
                Image tmBandOut(device, allocator, 512u, 512u, VK_FORMAT_R16G16B16A16_SFLOAT, kTmUsage);
                tmFullOut.transitionLayout(
                    cmd, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                );
                tmBandOut.transitionLayout(
                    cmd, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
                );

                VkMemoryBarrier2 initBar{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                initBar.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                initBar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                initBar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                initBar.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
                VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers = &initBar;
                vkCmdPipelineBarrier2(cmd, &dep);
                vkEndCommandBuffer(cmd);

                VkSubmitInfo upSub{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                upSub.commandBufferCount = 1;
                upSub.pCommandBuffers = &cmd;
                check_true(vkQueueSubmit(queue, 1, &upSub, VK_NULL_HANDLE) == VK_SUCCESS, "Upload G-buffers");
                vkQueueWaitIdle(queue);
                stagingKeepAlive.clear();

                auto bindGbuf = [](UpwaysPipeline& pipe, const GpuGbufferSet& g) {
                    pipe.updateDescriptors(
                        g.irr->getImageView(),
                        g.primGuide->getImageView(),
                        g.surfaceMv->getImageView(),
                        g.diffHitDist->getImageView(),
                        g.specVmv->getImageView(),
                        g.skipCompact->getImageView()
                    );
                };

                auto submitFrame = [&](UpwaysPipeline& pipe, uint32_t frameIdx, bool resetHist,
                                       glm::vec2 jit, bool zeroResidual, bool readbackOut,
                                       const UpwaysBandParams& band = {}, Buffer* dstBuf = nullptr) {
                    vkResetCommandBuffer(cmd, 0);
                    vkBeginCommandBuffer(cmd, &beginInfo);
                    pipe.recordFrame(cmd, frameIdx, resetHist, jit, false, zeroResidual, 1.0f, band);
                    if (readbackOut) {
                        Buffer& outRb = dstBuf ? *dstBuf : readbackBuf;
                        VkMemoryBarrier2 outBar{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                        outBar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        outBar.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                        outBar.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                        outBar.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                        VkDependencyInfo rbDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                        rbDep.memoryBarrierCount = 1;
                        rbDep.pMemoryBarriers = &outBar;
                        vkCmdPipelineBarrier2(cmd, &rbDep);

                        VkBufferImageCopy region{};
                        region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                        region.imageExtent = { pipe.getOutputWidth(), pipe.getOutputFullHeight(), 1 };
                        vkCmdCopyImageToBuffer(
                            cmd, pipe.getOutputImage()->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                            outRb.getBuffer(), 1, &region
                        );
                    }
                    vkEndCommandBuffer(cmd);
                    VkSubmitInfo sub{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                    sub.commandBufferCount = 1;
                    sub.pCommandBuffers = &cmd;
                    check_true(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE) == VK_SUCCESS, "Submit Upways frame");
                    vkQueueWaitIdle(queue);
                };

                // Fused Tonemap Stitch helper pipeline for Band-Parallel parity & seam check (SR + 1x native)
                Buffer dummyStitchBuf(
                    allocator, 256ULL,
                    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                    VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
                );
                VkDescriptorSetLayout tmLayout = VK_NULL_HANDLE;
                VkPipelineLayout tmPipeLayout = VK_NULL_HANDLE;
                VkPipeline tmPipeline = VK_NULL_HANDLE;
                VkDescriptorPool tmPool = VK_NULL_HANDLE;
                VkDescriptorSet tmSetFull = VK_NULL_HANDLE;
                VkDescriptorSet tmSetBand = VK_NULL_HANDLE;

                auto mkShader = [&](const std::vector<char>& spv) {
                    VkShaderModuleCreateInfo smi{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                    smi.codeSize = spv.size();
                    smi.pCode = reinterpret_cast<const uint32_t*>(spv.data());
                    VkShaderModule sm = VK_NULL_HANDLE;
                    check_true(vkCreateShaderModule(device, &smi, nullptr, &sm) == VK_SUCCESS, "Create tonemap band test shader");
                    return sm;
                };
                {
                    VkDescriptorSetLayoutBinding tb[3] = {
                        { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
                        { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr },
                        { 2, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr }
                    };
                    VkDescriptorSetLayoutCreateInfo tdli{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
                    tdli.bindingCount = 3; tdli.pBindings = tb;
                    check_true(vkCreateDescriptorSetLayout(device, &tdli, nullptr, &tmLayout) == VK_SUCCESS, "tonemap DSL");

                    VkDescriptorPoolSize psz[2] = {
                        { VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8 },
                        { VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4 }
                    };
                    VkDescriptorPoolCreateInfo dpi{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
                    dpi.maxSets = 4; dpi.poolSizeCount = 2; dpi.pPoolSizes = psz;
                    check_true(vkCreateDescriptorPool(device, &dpi, nullptr, &tmPool) == VK_SUCCESS, "tonemap pool");

                    VkDescriptorSetAllocateInfo talloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
                    talloc.descriptorPool = tmPool; talloc.descriptorSetCount = 1; talloc.pSetLayouts = &tmLayout;
                    check_true(vkAllocateDescriptorSets(device, &talloc, &tmSetFull) == VK_SUCCESS, "alloc tmSetFull");
                    check_true(vkAllocateDescriptorSets(device, &talloc, &tmSetBand) == VK_SUCCESS, "alloc tmSetBand");

                    VkPushConstantRange pcr{ VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PostProcessPipeline::TonemapPushConstants) };
                    VkPipelineLayoutCreateInfo pli{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
                    pli.setLayoutCount = 1; pli.pSetLayouts = &tmLayout;
                    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
                    check_true(vkCreatePipelineLayout(device, &pli, nullptr, &tmPipeLayout) == VK_SUCCESS, "tonemap PL");

                    VkShaderModule tmSm = mkShader(loadSpv("tonemap_aces.comp.spv"));
                    VkComputePipelineCreateInfo cpi{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
                    cpi.layout = tmPipeLayout;
                    cpi.stage = { VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, tmSm, "main", nullptr };
                    check_true(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpi, nullptr, &tmPipeline) == VK_SUCCESS, "tonemap pipe");
                    vkDestroyShaderModule(device, tmSm, nullptr);
                }

                auto verifyFusedTonemapParity = [&](UpwaysPipeline& fullP, UpwaysPipeline& topP,
                                                    Buffer& packedBuf, VkDeviceSize packedBytes,
                                                    uint32_t outW, uint32_t outH, uint32_t splitY,
                                                    const char* modeLabel, float maxTol) {
                    VkDescriptorImageInfo fullIn{ VK_NULL_HANDLE, fullP.getOutputImage()->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
                    VkDescriptorImageInfo fullOut{ VK_NULL_HANDLE, tmFullOut.getImageView(), VK_IMAGE_LAYOUT_GENERAL };
                    VkDescriptorBufferInfo fullStitch{ dummyStitchBuf.getBuffer(), 0, dummyStitchBuf.getSize() };
                    VkWriteDescriptorSet fw[3] = {
                        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tmSetFull, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &fullIn, nullptr, nullptr },
                        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tmSetFull, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &fullOut, nullptr, nullptr },
                        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tmSetFull, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &fullStitch, nullptr }
                    };
                    vkUpdateDescriptorSets(device, 3, fw, 0, nullptr);

                    VkDescriptorImageInfo bandIn{ VK_NULL_HANDLE, topP.getOutputImage()->getImageView(), VK_IMAGE_LAYOUT_GENERAL };
                    VkDescriptorImageInfo bandOut{ VK_NULL_HANDLE, tmBandOut.getImageView(), VK_IMAGE_LAYOUT_GENERAL };
                    VkDescriptorBufferInfo bandStitch{ packedBuf.getBuffer(), 0, packedBytes };
                    VkWriteDescriptorSet bw[3] = {
                        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tmSetBand, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &bandIn, nullptr, nullptr },
                        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tmSetBand, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &bandOut, nullptr, nullptr },
                        { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, tmSetBand, 2, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, nullptr, &bandStitch, nullptr }
                    };
                    vkUpdateDescriptorSets(device, 3, bw, 0, nullptr);

                    vkResetCommandBuffer(cmd, 0);
                    vkBeginCommandBuffer(cmd, &beginInfo);

                    VkMemoryBarrier2 inBar{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                    inBar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    inBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    inBar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    inBar.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
                    VkDependencyInfo inDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                    inDep.memoryBarrierCount = 1; inDep.pMemoryBarriers = &inBar;
                    vkCmdPipelineBarrier2(cmd, &inDep);

                    PostProcessPipeline::TonemapPushConstants pcFull{};
                    pcFull.exposure = 1.0f;
                    pcFull.totalSamples = 1u;
                    pcFull.applyACES = 1u;
                    pcFull.stitchYStart = 0u;
                    pcFull.stitchRowCount = 0u;

                    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tmPipeline);
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tmPipeLayout, 0, 1, &tmSetFull, 0, nullptr);
                    vkCmdPushConstants(cmd, tmPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcFull), &pcFull);
                    vkCmdDispatch(cmd, (outW + 15u) / 16u, (outH + 15u) / 16u, 1u);

                    PostProcessPipeline::TonemapPushConstants pcBand = pcFull;
                    pcBand.stitchYStart = splitY;
                    pcBand.stitchRowCount = outH - splitY;
                    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, tmPipeLayout, 0, 1, &tmSetBand, 0, nullptr);
                    vkCmdPushConstants(cmd, tmPipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pcBand), &pcBand);
                    vkCmdDispatch(cmd, (outW + 15u) / 16u, (outH + 15u) / 16u, 1u);

                    VkMemoryBarrier2 copyBar{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                    copyBar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                    copyBar.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
                    copyBar.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                    copyBar.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                    VkDependencyInfo copyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                    copyDep.memoryBarrierCount = 1; copyDep.pMemoryBarriers = &copyBar;
                    vkCmdPipelineBarrier2(cmd, &copyDep);

                    VkBufferImageCopy region{};
                    region.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                    region.imageExtent = { outW, outH, 1u };
                    vkCmdCopyImageToBuffer(cmd, tmFullOut.getImage(), VK_IMAGE_LAYOUT_GENERAL, readbackBuf.getBuffer(), 1, &region);
                    vkCmdCopyImageToBuffer(cmd, tmBandOut.getImage(), VK_IMAGE_LAYOUT_GENERAL, bandReadbackBuf.getBuffer(), 1, &region);
                    vkEndCommandBuffer(cmd);

                    VkSubmitInfo bsub{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                    bsub.commandBufferCount = 1; bsub.pCommandBuffers = &cmd;
                    check_true(vkQueueSubmit(queue, 1, &bsub, VK_NULL_HANDLE) == VK_SUCCESS, "Submit fused tonemap stitch");
                    vkQueueWaitIdle(queue);

                    readbackBuf.invalidate();
                    bandReadbackBuf.invalidate();
                    const uint16_t* fullH = static_cast<const uint16_t*>(readbackBuf.map());
                    const uint16_t* bandH = static_cast<const uint16_t*>(bandReadbackBuf.map());
                    check_true(fullH != nullptr && bandH != nullptr, "Map full vs fused band tonemap readback");

                    if (fullH && bandH) {
                        float maxAbsDiffAll = 0.0f;
                        float maxAbsDiffSeam = 0.0f;
                        bool allFinite = true;
                        for (uint32_t y = 0; y < outH; ++y) {
                            for (uint32_t x = 0; x < outW; ++x) {
                                const size_t off = (static_cast<size_t>(y) * outW + x) * 4u;
                                for (int c = 0; c < 3; ++c) {
                                    const float fv = glm::unpackHalf1x16(fullH[off + c]);
                                    const float bv = glm::unpackHalf1x16(bandH[off + c]);
                                    if (!std::isfinite(fv) || !std::isfinite(bv)) {
                                        allFinite = false;
                                    }
                                    const float d = std::abs(fv - bv);
                                    maxAbsDiffAll = std::max(maxAbsDiffAll, d);
                                    if (y + 2u >= splitY && y <= splitY + 1u) {
                                        maxAbsDiffSeam = std::max(maxAbsDiffSeam, d);
                                    }
                                }
                            }
                        }
                        std::cout << "    -> Fused Band-Parallel vs Full [" << modeLabel << ", prec=" << upwaysPrecisionName(fullP.getPrecision())
                                  << "]: max_abs_diff_all=" << maxAbsDiffAll
                                  << ", max_abs_diff_seam=" << maxAbsDiffSeam << std::endl;
                        check_true(allFinite, "Fused Band-Parallel outputs are all finite");
                        check_true(maxAbsDiffAll < maxTol, "Fused Band-Parallel stitched tonemap matches full-screen output");
                        check_true(maxAbsDiffSeam < maxTol, "Fused Band-Parallel seam across splitHrY has no discontinuity");
                    }
                    if (fullH) readbackBuf.unmap();
                    if (bandH) bandReadbackBuf.unmap();
                    vkDeviceWaitIdle(device);
                };

            for (UpwaysComputePrecision prec : supportedPrecisions) {
                const std::string suffix = (prec == UpwaysComputePrecision::Fp8)
                    ? ".comp.spv"
                    : (prec == UpwaysComputePrecision::Int8 ? ".int8.comp.spv" : ".fp16.comp.spv");
                const auto prepSpv = loadSpv("upways_denoise_prep" + suffix);
                const auto denSpv = loadSpv("upways_denoiser_wmma" + suffix);
                const auto upSpv = loadSpv("upways_upscaler_wmma" + suffix);
                const auto resSpv = loadSpv("upways_upscale_resolve" + suffix);
                check_true(!prepSpv.empty() && !denSpv.empty() && !upSpv.empty() && !resSpv.empty(),
                           "Upways 4.0 SPIR-V binaries loaded for live GPU test");

                UpwaysPipeline upw(
                    device, vkCtx.getPhysicalDevice(), allocator,
                    64u, 64u, 128u, 128u,
                    prepSpv, denSpv, upSpv, resSpv,
                    "", true, VK_FORMAT_R16G16B16A16_SFLOAT, prec
                );
                check_true(upw.isInitialized(), "UpwaysPipeline 4.0 must initialize on GPU");
                check_true(upw.getPrecision() == prec, "UpwaysPipeline::getPrecision matches requested tier");
                check_true(upw.usesFp8CoopMatrix() == (prec == UpwaysComputePrecision::Fp8),
                           "usesFp8CoopMatrix reflects active precision tier");
                bindGbuf(upw, g64);

                // Phase A: 2x Super-Resolution (64x64 -> 128x128), zero-residual + neural passes
                submitFrame(upw, 0u, true, { 0.0f, 0.0f }, true, false);
                submitFrame(upw, 1u, false, { 0.25f, -0.25f }, false, false);
                submitFrame(upw, 2u, false, { -0.25f, 0.25f }, false, true);
                check_true(upw.isHistoryValid(), "History must be valid after initial SR frames");
                check_true(upw.getLastPassTimings().valid, "VkQueryPool timings valid after >= 3 frames");

                {
                    readbackBuf.invalidate();
                    const uint16_t* halfPtr = static_cast<const uint16_t*>(readbackBuf.map());
                    check_true(halfPtr != nullptr, "Map 128x128 SR readback");
                    if (halfPtr) {
                        bool allFinitePos = true;
                        const size_t nPix = 128ULL * 128ULL;
                        for (size_t i = 0; i < nPix; ++i) {
                            const float r = glm::unpackHalf1x16(halfPtr[i * 4 + 0]);
                            const float g = glm::unpackHalf1x16(halfPtr[i * 4 + 1]);
                            const float b = glm::unpackHalf1x16(halfPtr[i * 4 + 2]);
                            if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b) || r <= 0.0f) {
                                allFinitePos = false;
                                break;
                            }
                        }
                        readbackBuf.unmap();
                        check_true(allFinitePos, "All 128x128 SR output pixels must be finite and positive");
                    }
                }

                // Phase B: Lossless DRS resize (64x64 -> 96x96 @ 128x128)
                Image* preDrsOutput = upw.getOutputImage();
                upw.resize(96u, 96u, 128u, 128u, true);
                check_true(upw.getOutputImage() == preDrsOutput, "Lossless DRS preserves display output image");
                check_true(upw.isHistoryValid(), "Lossless DRS preserves temporal HR history validity");
                bindGbuf(upw, g96);
                submitFrame(upw, 3u, false, { 0.125f, 0.125f }, false, false);
                submitFrame(upw, 4u, false, { -0.125f, -0.125f }, false, false);
                check_true(upw.isHistoryValid(), "History remains valid across DRS dispatches");

                // Phase C: Display resize (96x96 -> 160x160) resets & recovers history
                upw.resize(96u, 96u, 160u, 160u, true);
                check_true(!upw.isHistoryValid(), "Display resize resets HR temporal history");
                check_true(upw.getOutputWidth() == 160u && upw.getOutputHeight() == 160u,
                           "Display resize updates output dimensions to 160x160");
                submitFrame(upw, 5u, true, { 0.0f, 0.0f }, false, false);
                check_true(upw.isHistoryValid(), "History recovers cleanly after display resize");

                // Phase D: Native 1.0x Denoiser-only mode (64x64 -> 64x64, zero-residual identity check)
                upw.resize(64u, 64u, 64u, 64u, false);
                check_true(!upw.isSuperResEnabled(), "Native 1.0x resize disables SuperRes");
                bindGbuf(upw, g64);
                submitFrame(upw, 6u, true, { 0.0f, 0.0f }, true, false);
                submitFrame(upw, 7u, false, { 0.0f, 0.0f }, true, true);

                {
                    readbackBuf.invalidate();
                    const uint16_t* halfPtr = static_cast<const uint16_t*>(readbackBuf.map());
                    check_true(halfPtr != nullptr, "Map 64x64 native zero-residual readback");
                    if (halfPtr) {
                        bool energyExact = true;
                        const size_t nPix = 64ULL * 64ULL;
                        for (size_t i = 0; i < nPix; ++i) {
                            const float r = glm::unpackHalf1x16(halfPtr[i * 4 + 0]);
                            const float g = glm::unpackHalf1x16(halfPtr[i * 4 + 1]);
                            const float b = glm::unpackHalf1x16(halfPtr[i * 4 + 2]);
                            if (!std::isfinite(r) || std::abs(r - 0.865f) > 0.02f ||
                                std::abs(g - 0.865f) > 0.02f || std::abs(b - 0.865f) > 0.02f) {
                                energyExact = false;
                                break;
                            }
                        }
                        readbackBuf.unmap();
                        check_true(energyExact, "Native 1.0x zero-residual output matches remodulated 0.865 radiance");
                    }
                }
                    vkDeviceWaitIdle(device);

                    // Phase E: Fused Band-Parallel (fused RGB9E5 pack + fused tonemap stitch) vs Full-Screen parity (2x SR + 1x)
                    {
                        const VkDeviceSize packedBytes = 512ULL * (512ULL - bandSplit.splitHrY) * 4ULL;
                        Buffer packedBuf(
                            allocator, packedBytes,
                            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                            VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
                        );

                        UpwaysPipeline fullPipe(
                            device, vkCtx.getPhysicalDevice(), allocator,
                            256u, 256u, 512u, 512u,
                            prepSpv, denSpv, upSpv, resSpv,
                            "", true, VK_FORMAT_R16G16B16A16_SFLOAT, prec, 512u
                        );
                        UpwaysPipeline topPipe(
                            device, vkCtx.getPhysicalDevice(), allocator,
                            256u, bandSplit.topLrH, 512u, bandSplit.topHrH,
                            prepSpv, denSpv, upSpv, resSpv,
                            "", true, VK_FORMAT_R16G16B16A16_SFLOAT, prec, 512u
                        );
                        UpwaysPipeline botPipe(
                            device, vkCtx.getPhysicalDevice(), allocator,
                            256u, bandSplit.botLrH, 512u, bandSplit.botHrH,
                            prepSpv, denSpv, upSpv, resSpv,
                            "", true, VK_FORMAT_R16G16B16A16_SFLOAT, prec, bandSplit.botHrH
                        );
                        bindGbuf(fullPipe, g256Full);
                        bindGbuf(topPipe, g256Top);
                        bindGbuf(botPipe, g256Bot);
                        botPipe.setBandPackTargets(packedBuf.getBuffer(), packedBytes);

                        // E.1: 2x Super-Resolution (256x256 -> 512x512, fused pack in upways_upscale_resolve)
                        const UpwaysBandParams topBandSr{ 0u, 0u, 256u, 512u, 0u, bandSplit.splitHrY, false, 0u };
                        const UpwaysBandParams botBandSr{
                            bandSplit.botLrY0, bandSplit.botHrY0, 256u, 512u,
                            bandSplit.splitHrY - bandSplit.botHrY0, bandSplit.botHrH, true, 0u
                        };
                        submitFrame(fullPipe, 0u, true,  {  0.0f,   0.0f }, false, false);
                        submitFrame(fullPipe, 1u, false, {  0.25f, -0.25f }, false, false);
                        submitFrame(fullPipe, 2u, false, { -0.25f,  0.25f }, false, false);

                        submitFrame(topPipe, 0u, true,  {  0.0f,   0.0f }, false, false, topBandSr);
                        submitFrame(topPipe, 1u, false, {  0.25f, -0.25f }, false, false, topBandSr);
                        submitFrame(topPipe, 2u, false, { -0.25f,  0.25f }, false, false, topBandSr);

                        submitFrame(botPipe, 0u, true,  {  0.0f,   0.0f }, false, false, botBandSr);
                        submitFrame(botPipe, 1u, false, {  0.25f, -0.25f }, false, false, botBandSr);
                        submitFrame(botPipe, 2u, false, { -0.25f,  0.25f }, false, false, botBandSr);
                        verifyFusedTonemapParity(fullPipe, topPipe, packedBuf, packedBytes, 512u, 512u, bandSplit.splitHrY, "V4 2x SR", 0.015f);

                        // E.2: 1x Native (256x256 -> 256x256, fused pack in upways_denoiser_wmma Stage 3)
                        fullPipe.resize(256u, 256u, 256u, 256u, false, 256u);
                        topPipe.resize(256u, bandSplit1x.topLrH, 256u, bandSplit1x.topHrH, false, 256u);
                        botPipe.resize(256u, bandSplit1x.botLrH, 256u, bandSplit1x.botHrH, false, bandSplit1x.botHrH);
                        bindGbuf(fullPipe, g256Full);
                        bindGbuf(topPipe, g256Top);
                        bindGbuf(botPipe, g256Bot);

                        const UpwaysBandParams topBand1x{ 0u, 0u, 256u, 256u, 0u, bandSplit1x.splitHrY, false, 0u };
                        const UpwaysBandParams botBand1x{
                            bandSplit1x.botLrY0, bandSplit1x.botHrY0, 256u, 256u,
                            bandSplit1x.splitHrY - bandSplit1x.botHrY0, bandSplit1x.botHrH, true, 0u
                        };
                        submitFrame(fullPipe, 3u, true,  { 0.0f, 0.0f }, false, false);
                        submitFrame(fullPipe, 4u, false, { 0.0f, 0.0f }, false, false);

                        submitFrame(topPipe, 3u, true,  { 0.0f, 0.0f }, false, false, topBand1x);
                        submitFrame(topPipe, 4u, false, { 0.0f, 0.0f }, false, false, topBand1x);

                        submitFrame(botPipe, 3u, true,  { 0.0f, 0.0f }, false, false, botBand1x);
                        submitFrame(botPipe, 4u, false, { 0.0f, 0.0f }, false, false, botBand1x);
                        verifyFusedTonemapParity(fullPipe, topPipe, packedBuf, packedBytes, 256u, 256u, bandSplit1x.splitHrY, "V4 1x Native", 0.015f);
                    }

                    // Phase F: Separated compact guide edge cases:
                    // (1) sky uMlSkipCompact still loads when isSky == true
                    // (2) specVmv.y == 0 fallback + primaryTransmissive bit 16 in pg.y produce finite exact radiance
                    {
                        const glm::vec3 skyRgb(0.2f, 0.4f, 0.8f);
                        GpuGbufferSet gEdge = createSkyDielectricEdgeGbufferSet(device, allocator, queue, cmd, skyRgb);

                        UpwaysPipeline edgePipe(
                            device, vkCtx.getPhysicalDevice(), allocator,
                            64u, 64u, 64u, 64u,
                            prepSpv, denSpv, upSpv, resSpv,
                            "", false, VK_FORMAT_R16G16B16A16_SFLOAT, prec
                        );
                        bindGbuf(edgePipe, gEdge);
                        submitFrame(edgePipe, 0u, true, { 0.0f, 0.0f }, true, false);
                        submitFrame(edgePipe, 1u, false, { 0.0f, 0.0f }, true, true);

                        readbackBuf.invalidate();
                        const uint16_t* edgeH = static_cast<const uint16_t*>(readbackBuf.map());
                        check_true(edgeH != nullptr, "Map Phase F edge readback");
                        if (edgeH) {
                            bool allEdgeFinite = true;
                            bool skyPassOk = true;
                            bool dielBit16Ok = true;
                            for (uint32_t y = 0; y < 64u; ++y) {
                                for (uint32_t x = 0; x < 64u; ++x) {
                                    const size_t idx = (static_cast<size_t>(y) * 64u + x) * 4u;
                                    const float r = glm::unpackHalf1x16(edgeH[idx + 0]);
                                    const float g = glm::unpackHalf1x16(edgeH[idx + 1]);
                                    const float b = glm::unpackHalf1x16(edgeH[idx + 2]);
                                    if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) {
                                        allEdgeFinite = false;
                                    }
                                    if (y >= 4u && y < 28u && x >= 4u && x < 60u) {
                                        if (std::abs(r - skyRgb.r) > 0.02f ||
                                            std::abs(g - skyRgb.g) > 0.02f ||
                                            std::abs(b - skyRgb.b) > 0.02f) {
                                            skyPassOk = false;
                                        }
                                    }
                                    if (y >= 36u && y < 60u && x >= 4u && x < 60u) {
                                        if (std::abs(r - 0.62f) > 0.025f ||
                                            std::abs(g - 0.62f) > 0.025f ||
                                            std::abs(b - 0.62f) > 0.025f) {
                                            dielBit16Ok = false;
                                        }
                                    }
                                }
                            }
                            readbackBuf.unmap();
                            check_true(allEdgeFinite, "Separated compact edge-case output has zero NaN/Inf");
                            check_true(skyPassOk, "Sky pixels still read uMlSkipCompact cleanly");
                            check_true(dielBit16Ok, "specVmv.y==0 fallback & primaryTrans bit 16 match exact dielectric radiance");
                        }
                        vkDeviceWaitIdle(device);
                    }
                    std::cout << "    -> Verified V4 precision=" << upwaysPrecisionName(prec) << " on live GPU." << std::endl;
                }

                // [TEST 10] Upways 5.0 Pure-ML Live GPU Phase G Suite (FP8 & INT8 QAT)
                std::cout << "[TEST 10] Live GPU Upways 5.0 Pure-ML (Phase G)..." << std::endl;
                {
                    const UpwaysComputePrecision v5Fp16Fallback = UpwaysPipeline::selectPrecision(
                        vkCtx.getPhysicalDevice(), UpwaysPrecision::Fp16, vkCtx.getInstance(), UpwaysRuntimeVersion::V5
                    );
                    check_true(v5Fp16Fallback != UpwaysComputePrecision::Fp16,
                               "Upways 5.0 selectPrecision(Fp16) must fall back to FP8/INT8");

                    std::vector<UpwaysComputePrecision> v5Precs;
                    for (UpwaysComputePrecision p : supportedPrecisions) {
                        if (p == UpwaysComputePrecision::Fp8 || p == UpwaysComputePrecision::Int8) {
                            v5Precs.push_back(p);
                        }
                    }
                    check_true(!v5Precs.empty(), "At least FP8 or INT8 WMMA must be supported for Upways 5.0");

                    namespace fs = std::filesystem;
                    struct ScopedTempCleanup {
                        fs::path path;
                        ~ScopedTempCleanup() {
                            if (!path.empty()) {
                                std::error_code ec;
                                fs::remove(path, ec);
                            }
                        }
                    };
                    const fs::path zeroV5Path = fs::temp_directory_path() / "pathways_test_zero_v5_weights.bin";
                    ScopedTempCleanup zeroV5Cleanup{ zeroV5Path };
                    {
                        std::vector<uint8_t> zeros(upways_v5::TOTAL_WEIGHT_BYTES, 0u);
                        std::ofstream zf(zeroV5Path, std::ios::binary | std::ios::trunc);
                        check_true(zf.is_open(), "Create pathways_test_zero_v5_weights.bin in temp directory");
                        zf.write(reinterpret_cast<const char*>(zeros.data()), static_cast<std::streamsize>(zeros.size()));
                    }
                    check_true(UpwaysPipeline::resolveVersion(UpwaysVersion::Auto, zeroV5Path.string()) == UpwaysRuntimeVersion::V5,
                               "resolveVersion(Auto, zero_v5_weights.bin[190208 B]) selects V5");

                    vkResetCommandBuffer(cmd, 0);
                    vkBeginCommandBuffer(cmd, &beginInfo);
                    GpuGbufferSet g64Tex = createPopulatedGbufferSet(device, allocator, cmd, 64u, 64u, stagingKeepAlive, true);
                    GpuGbufferSet g48Tex = createPopulatedGbufferSet(device, allocator, cmd, 48u, 48u, stagingKeepAlive, true);
                    GpuGbufferSet g65Tex = createPopulatedGbufferSet(device, allocator, cmd, 65u, 65u, stagingKeepAlive, true);
                    {
                        VkMemoryBarrier2 tb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                        tb.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                        tb.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                        tb.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                        tb.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
                        VkDependencyInfo tdep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                        tdep.memoryBarrierCount = 1; tdep.pMemoryBarriers = &tb;
                        vkCmdPipelineBarrier2(cmd, &tdep);
                    }
                    vkEndCommandBuffer(cmd);
                    VkSubmitInfo tsub{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                    tsub.commandBufferCount = 1; tsub.pCommandBuffers = &cmd;
                    check_true(vkQueueSubmit(queue, 1, &tsub, VK_NULL_HANDLE) == VK_SUCCESS, "Upload V5 Phase G test G-buffers");
                    vkQueueWaitIdle(queue);
                    stagingKeepAlive.clear();

                    const glm::vec3 skyRgbV5(0.2f, 0.4f, 0.8f);
                    GpuGbufferSet gEdgeV5 = createSkyDielectricEdgeGbufferSet(device, allocator, queue, cmd, skyRgbV5);

                    // Compute exact ground-truth radiance & std over g64Tex (64x64)
                    std::vector<glm::vec3> gt64(64u * 64u);
                    glm::dvec3 sumGt(0.0), sumSqGt(0.0);
                    for (uint32_t y = 0; y < 64u; ++y) {
                        const float v = static_cast<float>(y) / 64.0f;
                        for (uint32_t x = 0; x < 64u; ++x) {
                            const float u = static_cast<float>(x) / 64.0f;
                            const float wave = 0.5f + 0.5f * std::sin(u * 12.0f + v * 16.0f);
                            const float checker = (((x >> 3) + (y >> 3)) & 1u) ? 0.85f : 0.45f;
                            const glm::vec3 aDiff = glm::vec3(0.35f + 0.45f * u, 0.25f + 0.55f * v, 0.30f + 0.40f * wave) * checker;
                            const glm::vec3 aSpec(0.05f + 0.04f * v, 0.06f, 0.07f + 0.03f * u);
                            const glm::vec3 diffIrr(0.9f + 0.6f * wave, 0.8f + 0.4f * v, 0.7f + 0.5f * u);
                            const glm::vec3 specIrr(0.20f + 0.30f * v, 0.22f, 0.18f + 0.25f * u);
                            const glm::vec3 skip(0.02f + 0.03f * v);
                            const glm::vec3 rad = diffIrr * aDiff + specIrr * aSpec + skip;
                            gt64[y * 64u + x] = rad;
                            sumGt += glm::dvec3(rad);
                            sumSqGt += glm::dvec3(rad * rad);
                        }
                    }
                    const double n64 = 4096.0;
                    const glm::dvec3 meanGt = sumGt / n64;
                    const glm::dvec3 varGt = glm::max(sumSqGt / n64 - meanGt * meanGt, glm::dvec3(0.0));
                    const double rawStd = std::sqrt((varGt.r + varGt.g + varGt.b) / 3.0);

                    for (UpwaysComputePrecision prec : v5Precs) {
                        const std::string ext = (prec == UpwaysComputePrecision::Int8) ? ".int8.comp.spv" : ".comp.spv";
                        const auto p5 = loadSpv("upways_v5_prep" + ext);
                        const auto u5 = loadSpv("upways_v5_unet_wmma" + ext);
                        const auto a5 = loadSpv("upways_v5_affinity" + ext);
                        const auto s5 = loadSpv("upways_v5_upscaler_wmma" + ext);
                        const auto r5 = loadSpv("upways_v5_resolve" + ext);
                        check_true(!p5.empty() && !u5.empty() && !a5.empty() && !s5.empty() && !r5.empty(),
                                   "Upways 5.0 SPIR-V binaries loaded for live GPU Phase G");

                        // G.1: Gate M3 Zero-Weight Identity (1x < 2e-2 & 2x SR hr_std/raw_std >= 0.90)
                        {
                            UpwaysPipeline zeroPipe(
                                device, vkCtx.getPhysicalDevice(), allocator,
                                64u, 64u, 64u, 64u,
                                p5, u5, a5, s5, r5,
                                zeroV5Path.string(), false, VK_FORMAT_R16G16B16A16_SFLOAT, prec
                            );
                            check_true(zeroPipe.getVersion() == UpwaysRuntimeVersion::V5, "zeroPipe.getVersion() == V5");
                            check_true(zeroPipe.isInitialized(), "V5 zeroPipe initialized");
                            bindGbuf(zeroPipe, g64Tex);
                            submitFrame(zeroPipe, 0u, true, { 0.0f, 0.0f }, false, false);
                            submitFrame(zeroPipe, 1u, false, { 0.0f, 0.0f }, false, true);

                            readbackBuf.invalidate();
                            const uint16_t* zh1 = static_cast<const uint16_t*>(readbackBuf.map());
                            check_true(zh1 != nullptr, "Map V5 Gate M3 1x readback");
                            if (zh1) {
                                float maxAbsZero1x = 0.0f;
                                bool allFiniteZero1x = true;
                                for (size_t i = 0; i < 64ULL * 64ULL; ++i) {
                                    for (int c = 0; c < 3; ++c) {
                                        const float v = glm::unpackHalf1x16(zh1[i * 4u + c]);
                                        if (!std::isfinite(v)) allFiniteZero1x = false;
                                        maxAbsZero1x = std::max(maxAbsZero1x, std::abs(v - gt64[i][c]));
                                    }
                                }
                                readbackBuf.unmap();
                                std::cout << "    -> V5 Gate M3 [prec=" << upwaysPrecisionName(prec)
                                          << "] 1x max_abs=" << maxAbsZero1x << std::endl;
                                check_true(allFiniteZero1x && maxAbsZero1x < 0.02f,
                                           "V5 Gate M3 1x zero-weight identity max_abs < 2e-2");
                            }

                            // Gate M3 2x SR (64x64 -> 128x128)
                            zeroPipe.resize(64u, 64u, 128u, 128u, true);
                            bindGbuf(zeroPipe, g64Tex);
                            submitFrame(zeroPipe, 2u, true, { 0.0f, 0.0f }, false, true);

                            readbackBuf.invalidate();
                            const uint16_t* zh2 = static_cast<const uint16_t*>(readbackBuf.map());
                            check_true(zh2 != nullptr, "Map V5 Gate M3 2x SR readback");
                            if (zh2) {
                                glm::dvec3 sumHr(0.0), sumSqHr(0.0);
                                bool allFiniteZero2x = true;
                                for (size_t i = 0; i < 128ULL * 128ULL; ++i) {
                                    const glm::dvec3 rgb(
                                        glm::unpackHalf1x16(zh2[i * 4u + 0u]),
                                        glm::unpackHalf1x16(zh2[i * 4u + 1u]),
                                        glm::unpackHalf1x16(zh2[i * 4u + 2u])
                                    );
                                    if (!std::isfinite(rgb.r) || !std::isfinite(rgb.g) || !std::isfinite(rgb.b)) {
                                        allFiniteZero2x = false;
                                    }
                                    sumHr += rgb;
                                    sumSqHr += rgb * rgb;
                                }
                                readbackBuf.unmap();
                                const double n128 = 128.0 * 128.0;
                                const glm::dvec3 meanHr = sumHr / n128;
                                const glm::dvec3 varHr = glm::max(sumSqHr / n128 - meanHr * meanHr, glm::dvec3(0.0));
                                const double hrStd = std::sqrt((varHr.r + varHr.g + varHr.b) / 3.0);
                                const double stdRatio = hrStd / std::max(rawStd, 1e-6);
                                std::cout << "    -> V5 Gate M3 [prec=" << upwaysPrecisionName(prec)
                                          << "] 2x SR hr_std/raw_std=" << stdRatio << std::endl;
                                check_true(allFiniteZero2x && stdRatio >= 0.90,
                                           "V5 Gate M3 2x SR preserves detail (hr_std/raw_std >= 0.90)");
                            }
                            vkDeviceWaitIdle(device);
                        }

                        // G.2: Trained Multi-Frame (T=4), Lossless DRS (48x48 -> 128x128), Odd Resolution (65x65 -> 130x130)
                        {
                            UpwaysPipeline v5Pipe(
                                device, vkCtx.getPhysicalDevice(), allocator,
                                64u, 64u, 128u, 128u,
                                p5, u5, a5, s5, r5,
                                "", true, VK_FORMAT_R16G16B16A16_SFLOAT, prec
                            );
                            check_true(v5Pipe.getVersion() == UpwaysRuntimeVersion::V5 && v5Pipe.isInitialized(),
                                       "V5 trained pipeline initialized");
                            bindGbuf(v5Pipe, g64Tex);
                            submitFrame(v5Pipe, 0u, true,  {  0.0f,   0.0f  }, false, false);
                            submitFrame(v5Pipe, 1u, false, {  0.25f, -0.25f }, false, false);
                            submitFrame(v5Pipe, 2u, false, { -0.25f,  0.25f }, false, false);
                            submitFrame(v5Pipe, 3u, false, {  0.125f, 0.125f}, false, true);
                            check_true(v5Pipe.getLastPassTimings().valid && v5Pipe.getLastPassTimings().totalMs > 0.0,
                                       "V5 VkQueryPool pass timings valid after >= 3 frames");

                            auto checkFinitePos = [&](uint32_t w, uint32_t h, const char* msg) {
                                readbackBuf.invalidate();
                                const uint16_t* hPtr = static_cast<const uint16_t*>(readbackBuf.map());
                                check_true(hPtr != nullptr, "Map V5 trained readback");
                                if (hPtr) {
                                    bool ok = true;
                                    const size_t nP = static_cast<size_t>(w) * h;
                                    for (size_t i = 0; i < nP; ++i) {
                                        const float r = glm::unpackHalf1x16(hPtr[i * 4u + 0u]);
                                        const float g = glm::unpackHalf1x16(hPtr[i * 4u + 1u]);
                                        const float b = glm::unpackHalf1x16(hPtr[i * 4u + 2u]);
                                        if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b) || r < 0.0f || g < 0.0f || b < 0.0f) {
                                            ok = false;
                                            break;
                                        }
                                    }
                                    readbackBuf.unmap();
                                    check_true(ok, msg);
                                }
                            };
                            checkFinitePos(128u, 128u, "V5 trained T=4 128x128 SR output finite & non-negative");

                            // Lossless DRS (64x64 -> 48x48 @ 128x128)
                            Image* preOutV5 = v5Pipe.getOutputImage();
                            v5Pipe.resize(48u, 48u, 128u, 128u, true);
                            check_true(v5Pipe.getOutputImage() == preOutV5 && v5Pipe.isHistoryValid(),
                                       "V5 lossless DRS preserves display image & HR history");
                            bindGbuf(v5Pipe, g48Tex);
                            submitFrame(v5Pipe, 4u, false, { 0.0f, 0.0f }, false, false);
                            submitFrame(v5Pipe, 5u, false, { -0.25f, 0.25f }, false, true);
                            check_true(v5Pipe.isHistoryValid(), "V5 HR history valid across DRS 48x48->128x128");
                            checkFinitePos(128u, 128u, "V5 lossless DRS 48x48->128x128 output finite");

                            // Odd resolution (65x65 -> 130x130)
                            v5Pipe.resize(65u, 65u, 130u, 130u, true);
                            bindGbuf(v5Pipe, g65Tex);
                            submitFrame(v5Pipe, 6u, true, { 0.0f, 0.0f }, false, false);
                            submitFrame(v5Pipe, 7u, false, { 0.25f, -0.25f }, false, true);
                            checkFinitePos(130u, 130u, "V5 odd-resolution 65x65->130x130 output finite");

                            // Sky & Dielectric bit-16 (64x64 1x, zeroResidual=true & trained zeroResidual=false)
                            v5Pipe.resize(64u, 64u, 64u, 64u, false);
                            bindGbuf(v5Pipe, gEdgeV5);
                            submitFrame(v5Pipe, 8u, true, { 0.0f, 0.0f }, true, false);
                            submitFrame(v5Pipe, 9u, false, { 0.0f, 0.0f }, true, true);
                            {
                                readbackBuf.invalidate();
                                const uint16_t* edgeV5H = static_cast<const uint16_t*>(readbackBuf.map());
                                check_true(edgeV5H != nullptr, "Map V5 sky/dielectric readback");
                                if (edgeV5H) {
                                    bool finiteV5 = true, skyOkV5 = true, dielOkV5 = true;
                                    for (uint32_t y = 0; y < 64u; ++y) {
                                        for (uint32_t x = 0; x < 64u; ++x) {
                                            const size_t idx = (static_cast<size_t>(y) * 64u + x) * 4u;
                                            const float r = glm::unpackHalf1x16(edgeV5H[idx + 0]);
                                            const float g = glm::unpackHalf1x16(edgeV5H[idx + 1]);
                                            const float b = glm::unpackHalf1x16(edgeV5H[idx + 2]);
                                            if (!std::isfinite(r) || !std::isfinite(g) || !std::isfinite(b)) finiteV5 = false;
                                            if (y >= 4u && y < 28u && x >= 4u && x < 60u) {
                                                if (std::abs(r - skyRgbV5.r) > 0.02f || std::abs(g - skyRgbV5.g) > 0.02f || std::abs(b - skyRgbV5.b) > 0.02f) {
                                                    skyOkV5 = false;
                                                }
                                            }
                                            if (y >= 36u && y < 60u && x >= 4u && x < 60u) {
                                                if (std::abs(r - 0.62f) > 0.025f || std::abs(g - 0.62f) > 0.025f || std::abs(b - 0.62f) > 0.025f) {
                                                    dielOkV5 = false;
                                                }
                                            }
                                        }
                                    }
                                    readbackBuf.unmap();
                                    check_true(finiteV5 && skyOkV5 && dielOkV5, "V5 sky/dielectric bit-16 exact under zeroResidual=true");
                                }
                            }
                            submitFrame(v5Pipe, 10u, true, { 0.0f, 0.0f }, false, false);
                            submitFrame(v5Pipe, 11u, false, { 0.0f, 0.0f }, false, true);
                            checkFinitePos(64u, 64u, "V5 trained sky/dielectric output finite");
                            vkDeviceWaitIdle(device);
                        }

                        // G.3: V5 Fused mGPU Band-Parallel Parity (2x SR & 1x Native)
                        {
                            const VkDeviceSize packedBytes = 512ULL * (512ULL - bandSplit.splitHrY) * 4ULL;
                            Buffer packedBuf(
                                allocator, packedBytes,
                                VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                                VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE
                            );
                            UpwaysPipeline v5Full(device, vkCtx.getPhysicalDevice(), allocator, 256u, 256u, 512u, 512u, p5, u5, a5, s5, r5, "", true, VK_FORMAT_R16G16B16A16_SFLOAT, prec, 512u);
                            UpwaysPipeline v5Top(device, vkCtx.getPhysicalDevice(), allocator, 256u, bandSplit.topLrH, 512u, bandSplit.topHrH, p5, u5, a5, s5, r5, "", true, VK_FORMAT_R16G16B16A16_SFLOAT, prec, 512u);
                            UpwaysPipeline v5Bot(device, vkCtx.getPhysicalDevice(), allocator, 256u, bandSplit.botLrH, 512u, bandSplit.botHrH, p5, u5, a5, s5, r5, "", true, VK_FORMAT_R16G16B16A16_SFLOAT, prec, bandSplit.botHrH);
                            bindGbuf(v5Full, g256Full);
                            bindGbuf(v5Top, g256Top);
                            bindGbuf(v5Bot, g256Bot);
                            v5Bot.setBandPackTargets(packedBuf.getBuffer(), packedBytes);

                            const UpwaysBandParams topBandSr{ 0u, 0u, 256u, 512u, 0u, bandSplit.splitHrY, false, 0u };
                            const UpwaysBandParams botBandSr{
                                bandSplit.botLrY0, bandSplit.botHrY0, 256u, 512u,
                                bandSplit.splitHrY - bandSplit.botHrY0, bandSplit.botHrH, true, 0u
                            };
                            submitFrame(v5Full, 0u, true,  {  0.0f,   0.0f  }, false, false);
                            submitFrame(v5Full, 1u, false, {  0.25f, -0.25f }, false, false);
                            submitFrame(v5Full, 2u, false, { -0.25f,  0.25f }, false, false);

                            submitFrame(v5Top,  0u, true,  {  0.0f,   0.0f  }, false, false, topBandSr);
                            submitFrame(v5Top,  1u, false, {  0.25f, -0.25f }, false, false, topBandSr);
                            submitFrame(v5Top,  2u, false, { -0.25f,  0.25f }, false, false, topBandSr);

                            submitFrame(v5Bot,  0u, true,  {  0.0f,   0.0f  }, false, false, botBandSr);
                            submitFrame(v5Bot,  1u, false, {  0.25f, -0.25f }, false, false, botBandSr);
                            submitFrame(v5Bot,  2u, false, { -0.25f,  0.25f }, false, false, botBandSr);
                            verifyFusedTonemapParity(v5Full, v5Top, packedBuf, packedBytes, 512u, 512u, bandSplit.splitHrY, "V5 2x SR", 0.02f);

                            // 1x Native
                            v5Full.resize(256u, 256u, 256u, 256u, false, 256u);
                            v5Top.resize(256u, bandSplit1x.topLrH, 256u, bandSplit1x.topHrH, false, 256u);
                            v5Bot.resize(256u, bandSplit1x.botLrH, 256u, bandSplit1x.botHrH, false, bandSplit1x.botHrH);
                            bindGbuf(v5Full, g256Full);
                            bindGbuf(v5Top, g256Top);
                            bindGbuf(v5Bot, g256Bot);

                            const UpwaysBandParams topBand1x{ 0u, 0u, 256u, 256u, 0u, bandSplit1x.splitHrY, false, 0u };
                            const UpwaysBandParams botBand1x{
                                bandSplit1x.botLrY0, bandSplit1x.botHrY0, 256u, 256u,
                                bandSplit1x.splitHrY - bandSplit1x.botHrY0, bandSplit1x.botHrH, true, 0u
                            };
                            submitFrame(v5Full, 3u, true,  { 0.0f, 0.0f }, false, false);
                            submitFrame(v5Full, 4u, false, { 0.0f, 0.0f }, false, false);

                            submitFrame(v5Top,  3u, true,  { 0.0f, 0.0f }, false, false, topBand1x);
                            submitFrame(v5Top,  4u, false, { 0.0f, 0.0f }, false, false, topBand1x);

                            submitFrame(v5Bot,  3u, true,  { 0.0f, 0.0f }, false, false, botBand1x);
                            submitFrame(v5Bot,  4u, false, { 0.0f, 0.0f }, false, false, botBand1x);
                            verifyFusedTonemapParity(v5Full, v5Top, packedBuf, packedBytes, 256u, 256u, bandSplit1x.splitHrY, "V5 1x Native", 0.02f);
                        }

                        std::cout << "    -> Verified V5 Phase G precision=" << upwaysPrecisionName(prec) << " on live GPU." << std::endl;
                    }
                }
                if (tmPipeline) vkDestroyPipeline(device, tmPipeline, nullptr);
                if (tmPipeLayout) vkDestroyPipelineLayout(device, tmPipeLayout, nullptr);
                if (tmPool) vkDestroyDescriptorPool(device, tmPool, nullptr);
                if (tmLayout) vkDestroyDescriptorSetLayout(device, tmLayout, nullptr);
            }

            vkDestroyCommandPool(device, cmdPool, nullptr);
            check_true(vkCtx.getValidationErrors() == 0u, "Zero Vulkan validation errors during Upways 4.0/5.0 GPU tests");
            std::cout << "  -> Live GPU Upways 4.0 & 5.0 verified, 0 validation errors." << std::endl;
        } else {
            std::cout << "[TEST 8] Skipped (no Vulkan 1.4 hardware GPU available)." << std::endl;
        }
    }

    if (g_failures > 0) {
        std::cerr << "\n[RESULT] " << g_failures << " check(s) FAILED" << std::endl;
        return 1;
    }
    std::cout << "\n[RESULT] All Upways 4.0/5.0 pipeline checks PASSED" << std::endl;
    return 0;
}
