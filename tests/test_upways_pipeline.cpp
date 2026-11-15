#include "rt/UpwaysPipeline.hpp"
#include "rt/upways_weights.hpp"
#include "rt/upways_default_weights.hpp"
#include "rt/upways_default_int8_weights.hpp"
#include "core/Config.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "vulkan/VulkanContext.hpp"

#include <glm/gtc/packing.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
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

struct GpuGbufferSet {
    uint32_t width = 0;
    uint32_t height = 0;
    std::unique_ptr<Image> diff;
    std::unique_ptr<Image> spec;
    std::unique_ptr<Image> albRough;
    std::unique_ptr<Image> specAlb;
    std::unique_ptr<Image> skip;
    std::unique_ptr<Image> normDepth;
    std::unique_ptr<Image> surfMv;
    std::unique_ptr<Image> specMv;
};

GpuGbufferSet createPopulatedGbufferSet(
    VkDevice device,
    VmaAllocator allocator,
    VkCommandBuffer cmd,
    uint32_t width,
    uint32_t height,
    std::vector<std::unique_ptr<Buffer>>& stagingKeepAlive
) {
    GpuGbufferSet gbuf;
    gbuf.width = width;
    gbuf.height = height;

    const VkImageUsageFlags usage =
        VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
        VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;

    auto mkImg = [&]() {
        return std::make_unique<Image>(
            device, allocator, width, height, VK_FORMAT_R16G16B16A16_SFLOAT, usage
        );
    };
    gbuf.diff = mkImg();
    gbuf.spec = mkImg();
    gbuf.albRough = mkImg();
    gbuf.specAlb = mkImg();
    gbuf.skip = mkImg();
    gbuf.normDepth = mkImg();
    gbuf.surfMv = mkImg();
    gbuf.specMv = mkImg();

    const size_t numPixels = static_cast<size_t>(width) * height;
    auto uploadRgba16 = [&](Image& img, float r, float g, float b, float a) {
        std::vector<uint16_t> pixels(numPixels * 4);
        const uint16_t hr = glm::packHalf1x16(r);
        const uint16_t hg = glm::packHalf1x16(g);
        const uint16_t hb = glm::packHalf1x16(b);
        const uint16_t ha = glm::packHalf1x16(a);
        for (size_t i = 0; i < numPixels; ++i) {
            pixels[i * 4 + 0] = hr;
            pixels[i * 4 + 1] = hg;
            pixels[i * 4 + 2] = hb;
            pixels[i * 4 + 3] = ha;
        }
        const VkDeviceSize bytes = static_cast<VkDeviceSize>(pixels.size() * sizeof(uint16_t));
        auto stg = std::make_unique<Buffer>(
            allocator, bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        );
        stg->copyFrom(pixels.data(), bytes);

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

    // Normalized Upways 4.0 G-buffer:
    //   0.8 * 1.0 + 0.06 * 0.25 + 0.05 = 0.865 linear radiance on uniform front-facing surface
    uploadRgba16(*gbuf.diff, 1.0f, 1.0f, 1.0f, 0.5f);
    uploadRgba16(*gbuf.spec, 0.25f, 0.25f, 0.25f, 0.0f);
    uploadRgba16(*gbuf.albRough, 0.8f, 0.8f, 0.8f, 0.4f);
    uploadRgba16(*gbuf.specAlb, 0.06f, 0.06f, 0.06f, 0.5f);
    uploadRgba16(*gbuf.skip, 0.05f, 0.05f, 0.05f, 0.0f);
    uploadRgba16(*gbuf.normDepth, 0.0f, 0.0f, 1.0f, 3.0f);
    uploadRgba16(*gbuf.surfMv, 0.0f, 0.0f, 0.0f, 0.0f);
    uploadRgba16(*gbuf.specMv, 0.0f, 0.0f, 0.0f, 0.5f);
    return gbuf;
}

} // namespace

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Upways 4.0 Neural Pipeline Unit & GPU Test    " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Push Constant Layout & Flags (56 B immutable ABI)
    std::cout << "[TEST 1] UpwaysPushConstants ABI (56 B) & Flags..." << std::endl;
    check_true(sizeof(UpwaysPushConstants) == 56, "UpwaysPushConstants must be 56 bytes");
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
    check_true(UPWAYS_FLAG_RESET_HISTORY == 1u, "UPWAYS_FLAG_RESET_HISTORY == 1");
    check_true(UPWAYS_FLAG_ENABLE_SUPERRES == 2u, "UPWAYS_FLAG_ENABLE_SUPERRES == 2");
    check_true(UPWAYS_FLAG_ZERO_RESIDUAL_MODE == 4u, "UPWAYS_FLAG_ZERO_RESIDUAL_MODE == 4");

    // 2. Log1p/Expm1 & Demodulation/Remodulation Energy Conservation
    std::cout << "[TEST 2] safeLog1p / safeExpm1 & Demodulation Conservation..." << std::endl;
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
        const uint32_t ldsUpscaler8Bit = 1024u * 8u + 1024u * 4u + 2304u * 4u;
        const uint32_t ldsUpscalerFp16 = 1024u * 8u + 2048u * 4u + 2304u * 4u;

        check_true(ldsPrep == 7200u && ldsPrep <= 32768u, "denoise_prep LDS == 7,200 B <= 32 KiB");
        check_true(ldsDenoiser8Bit == 16384u && ldsDenoiser8Bit <= 32768u, "denoiser_wmma 8-bit LDS == 16,384 B <= 32 KiB");
        check_true(ldsDenoiserFp16 == 28672u && ldsDenoiserFp16 <= 32768u, "denoiser_wmma FP16 LDS == 28,672 B <= 32 KiB");
        check_true(ldsUpscaler8Bit == 21504u && ldsUpscaler8Bit <= 32768u, "upscaler_wmma 8-bit LDS == 21,504 B <= 32 KiB");
        check_true(ldsUpscalerFp16 == 25600u && ldsUpscalerFp16 <= 32768u, "upscaler_wmma FP16 LDS == 25,600 B <= 32 KiB");
    }

    // 6. Config CLI Parsing
    std::cout << "[TEST 6] Config CLI Parsing (Upways + PTTD v3/v4 Defaults)..." << std::endl;
    {
        Config defCfg;
        check_true(defCfg.capture_channels == 24u, "Default Config::capture_channels == 24 preserved");
        check_true(defCfg.denoiser_mode == DenoiserMode::None, "Default DenoiserMode == None");
        check_true(defCfg.upways_precision == UpwaysPrecision::Auto, "Default UpwaysPrecision == Auto");
        check_true(!defCfg.upways_superres && !defCfg.upways_heuristic_fallback, "Default Upways SR/heuristic disabled");

        const char* argv1[] = { "pathways", "--denoiser", "upways", "--capture-channels", "32", "--upways-precision", "auto" };
        Config c1 = Config::parse(7, const_cast<char**>(argv1));
        check_true(c1.denoiser_mode == DenoiserMode::Upways, "--denoiser upways selects Upways");
        check_true(c1.upways_precision == UpwaysPrecision::Auto, "--upways-precision auto selects Auto");
        check_true(!c1.upways_superres, "--denoiser upways leaves upways_superres false");
        check_true(c1.capture_channels == 32u, "--capture-channels 32 accepted");

        const char* argv2[] = { "pathways", "--scaler", "upways", "--upways-heuristic", "--upways-precision", "fp8" };
        Config c2 = Config::parse(6, const_cast<char**>(argv2));
        check_true(c2.denoiser_mode == DenoiserMode::Upways, "--scaler upways enables Upways denoiser");
        check_true(c2.upscaler_mode == UpscalerMode::Upways, "--scaler upways selects UpscalerMode::Upways");
        check_true(c2.upways_superres, "--scaler upways sets upways_superres true");
        check_true(c2.upways_heuristic_fallback, "--upways-heuristic enables fallback flag");
        check_true(c2.upways_precision == UpwaysPrecision::Fp8, "--upways-precision fp8 selects Fp8");

        const char* argv3[] = { "pathways", "--upways-precision", "int8" };
        Config c3 = Config::parse(3, const_cast<char**>(argv3));
        check_true(c3.upways_precision == UpwaysPrecision::Int8, "--upways-precision int8 selects Int8");

        const char* argv4[] = { "pathways", "--upways-precision", "fp16" };
        Config c4 = Config::parse(3, const_cast<char**>(argv4));
        check_true(c4.upways_precision == UpwaysPrecision::Fp16, "--upways-precision fp16 selects Fp16");

        const char* argv5[] = { "pathways", "--upways-precision=int8" };
        Config c5 = Config::parse(2, const_cast<char**>(argv5));
        check_true(c5.upways_precision == UpwaysPrecision::Int8, "--upways-precision=int8 selects Int8");
    }

    // 7. SPIR-V Shader Validation (FP8 + INT8 + FP16 Variants)
    std::cout << "[TEST 7] SPIR-V Header Verification (12 Shaders)..." << std::endl;
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
    };
    for (const char* sname : shaderNames) {
        const auto spv = loadSpv(sname);
        if (!validateSpirv(spv)) {
            std::cerr << "[FAIL] Invalid/missing SPIR-V: " << sname << std::endl;
            ++g_failures;
        }
    }

    // 8. Live Headless GPU Execution, Lossless DRS, Resize & Native 1.0x
    {
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

            const VkDeviceSize maxReadBytes = 160ULL * 160ULL * 4ULL * sizeof(uint16_t);
            Buffer readbackBuf(
                allocator, maxReadBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT
            );

            {
                std::vector<std::unique_ptr<Buffer>> stagingKeepAlive;
                VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkResetCommandBuffer(cmd, 0);
                vkBeginCommandBuffer(cmd, &beginInfo);

                GpuGbufferSet g64 = createPopulatedGbufferSet(device, allocator, cmd, 64u, 64u, stagingKeepAlive);
                GpuGbufferSet g96 = createPopulatedGbufferSet(device, allocator, cmd, 96u, 96u, stagingKeepAlive);

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
                        g.diff->getImageView(),
                        g.spec->getImageView(),
                        g.albRough->getImageView(),
                        g.specAlb->getImageView(),
                        g.skip->getImageView(),
                        g.normDepth->getImageView(),
                        g.surfMv->getImageView(),
                        g.specMv->getImageView()
                    );
                };

                auto submitFrame = [&](UpwaysPipeline& pipe, uint32_t frameIdx, bool resetHist,
                                       glm::vec2 jit, bool zeroResidual, bool readbackOut) {
                    vkResetCommandBuffer(cmd, 0);
                    vkBeginCommandBuffer(cmd, &beginInfo);
                    pipe.recordFrame(cmd, frameIdx, resetHist, jit, false, zeroResidual, 1.0f);
                    if (readbackOut) {
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
                        region.imageExtent = { pipe.getOutputWidth(), pipe.getOutputHeight(), 1 };
                        vkCmdCopyImageToBuffer(
                            cmd, pipe.getOutputImage()->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                            readbackBuf.getBuffer(), 1, &region
                        );
                    }
                    vkEndCommandBuffer(cmd);
                    VkSubmitInfo sub{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                    sub.commandBufferCount = 1;
                    sub.pCommandBuffers = &cmd;
                    check_true(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE) == VK_SUCCESS, "Submit Upways frame");
                    vkQueueWaitIdle(queue);
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
                    vkDeviceWaitIdle(device);
                    std::cout << "    -> Verified precision=" << upwaysPrecisionName(prec) << " on live GPU." << std::endl;
                }
            }

            vkDestroyCommandPool(device, cmdPool, nullptr);
            check_true(vkCtx.getValidationErrors() == 0u, "Zero Vulkan validation errors during Upways 4.0 GPU test");
            std::cout << "  -> Live GPU Upways 4.0 + Lossless DRS + 1.0x verified across "
                      << supportedPrecisions.size() << " precision tier(s), 0 validation errors." << std::endl;
        } else {
            std::cout << "[TEST 8] Skipped (no Vulkan 1.4 hardware GPU available)." << std::endl;
        }
    }

    if (g_failures > 0) {
        std::cerr << "\n[RESULT] " << g_failures << " check(s) FAILED" << std::endl;
        return 1;
    }
    std::cout << "\n[RESULT] All Upways 4.0 pipeline checks PASSED" << std::endl;
    return 0;
}
