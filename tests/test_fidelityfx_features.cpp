#include "rt/FfxSdkRuntime.hpp"
#include "core/Config.hpp"
#include "utils/ImageDumper.hpp"

#include <stb_image.h>
#include <vulkan/vulkan.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace pathways;

static void check_true(bool cond, const std::string& msg) {
    if (!cond) {
        std::cerr << "[FAIL] Assertion failed: " << msg << std::endl;
        std::exit(1);
    }
}

struct ImageMetrics {
    int width = 0;
    int height = 0;
    double meanLuminance = 0.0;
    double nonBlackRatio = 0.0;
    double laplacianNoiseVar = 0.0;
    double psnrVsRefDb = 0.0;
    double rmseVsRef = 0.0;
};

static ImageMetrics evaluateImageQuality(const std::string& pngPath, const std::vector<uint8_t>* refPixels = nullptr, int refW = 0, int refH = 0) {
    int w = 0, h = 0, ch = 0;
    stbi_uc* data = stbi_load(pngPath.c_str(), &w, &h, &ch, 4);
    check_true(data != nullptr && w > 0 && h > 0, "Failed to load dumped PNG image: " + pngPath);

    ImageMetrics m{};
    m.width = w;
    m.height = h;

    auto lumAt = [&](int x, int y) -> double {
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4u;
        const double r = static_cast<double>(data[idx + 0]) / 255.0;
        const double g = static_cast<double>(data[idx + 1]) / 255.0;
        const double b = static_cast<double>(data[idx + 2]) / 255.0;
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };

    double sumLum = 0.0;
    size_t nonBlack = 0;
    const size_t totalPixels = static_cast<size_t>(w) * static_cast<size_t>(h);
    for (size_t i = 0; i < totalPixels; ++i) {
        const double r = static_cast<double>(data[i * 4 + 0]) / 255.0;
        const double g = static_cast<double>(data[i * 4 + 1]) / 255.0;
        const double b = static_cast<double>(data[i * 4 + 2]) / 255.0;
        const double l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
        sumLum += l;
        if (l > 0.01) {
            ++nonBlack;
        }
    }
    m.meanLuminance = sumLum / static_cast<double>(totalPixels);
    m.nonBlackRatio = static_cast<double>(nonBlack) / static_cast<double>(totalPixels);

    // Evaluate high-frequency Laplacian noise variance across interior pixels,
    // excluding sharp geometric edges (|lap| > 0.25) so we measure stochastic MC noise.
    double sumLapSq = 0.0;
    size_t lapCount = 0;
    for (int y = h / 6; y < (h * 5) / 6; ++y) {
        for (int x = w / 6; x < (w * 5) / 6; ++x) {
            const double c = lumAt(x, y);
            const double l = lumAt(x - 1, y);
            const double r = lumAt(x + 1, y);
            const double u = lumAt(x, y - 1);
            const double d = lumAt(x, y + 1);
            const double lap = 4.0 * c - (l + r + u + d);
            if (std::abs(lap) < 0.25) {
                sumLapSq += lap * lap;
                ++lapCount;
            }
        }
    }
    m.laplacianNoiseVar = (lapCount > 0) ? (sumLapSq / static_cast<double>(lapCount)) : 0.0;

    if (refPixels && refW == w && refH == h && refPixels->size() == totalPixels * 4u) {
        double sumSqErr = 0.0;
        for (size_t i = 0; i < totalPixels; ++i) {
            for (size_t c = 0; c < 3; ++c) {
                const double diff = (static_cast<double>(data[i * 4 + c]) - static_cast<double>((*refPixels)[i * 4 + c])) / 255.0;
                sumSqErr += diff * diff;
            }
        }
        const double mse = sumSqErr / static_cast<double>(totalPixels * 3u);
        m.rmseVsRef = std::sqrt(mse);
        m.psnrVsRefDb = (mse > 1e-10) ? (10.0 * std::log10(1.0 / mse)) : 99.0;
    }

    stbi_image_free(data);
    return m;
}

static std::vector<uint8_t> loadRgbaPixels(const std::string& pngPath, int& outW, int& outH) {
    int ch = 0;
    stbi_uc* data = stbi_load(pngPath.c_str(), &outW, &outH, &ch, 4);
    check_true(data != nullptr && outW > 0 && outH > 0, "Failed to load reference PNG: " + pngPath);
    std::vector<uint8_t> vec(data, data + static_cast<size_t>(outW) * static_cast<size_t>(outH) * 4u);
    stbi_image_free(data);
    return vec;
}

static double extractJsonNumber(const std::string& json, const std::string& key, double fallback = -1.0) {
    const std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return fallback;
    pos += pattern.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    try {
        size_t idx = 0;
        double val = std::stod(json.substr(pos, 64), &idx);
        return idx > 0 ? val : fallback;
    } catch (...) {
        return fallback;
    }
}

static std::string extractJsonString(const std::string& json, const std::string& key) {
    const std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return "";
    pos = json.find('"', pos + pattern.size());
    if (pos == std::string::npos) return "";
    size_t endPos = json.find('"', pos + 1);
    if (endPos == std::string::npos) return "";
    return json.substr(pos + 1, endPos - pos - 1);
}

static bool extractJsonBool(const std::string& json, const std::string& key, bool fallback = false) {
    const std::string pattern = "\"" + key + "\":";
    size_t pos = json.find(pattern);
    if (pos == std::string::npos) return fallback;
    pos += pattern.size();
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t')) ++pos;
    if (json.compare(pos, 4, "true") == 0) return true;
    if (json.compare(pos, 5, "false") == 0) return false;
    return fallback;
}

int main(int argc, char** argv) {
    std::cout << "==================================================================" << std::endl;
    std::cout << "  Pathways: AMD FidelityFX SDK 2.4 Feature Verification Suite     " << std::endl;
    std::cout << "  API Capability Queries, Performance & Visual Quality Assessment " << std::endl;
    std::cout << "==================================================================" << std::endl;

    // -------------------------------------------------------------------------
    // TEST 1: API-Driven Hardware Capability Detection (No Product String Checks)
    // -------------------------------------------------------------------------
    std::cout << "[TEST 1] Vulkan & FidelityFX API-Driven Capability Detection..." << std::endl;
    check_true(!FfxSdkRuntime::supportsHardwareMl(VK_NULL_HANDLE),
               "supportsHardwareMl(VK_NULL_HANDLE) must return false");

    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.pApplicationName = "PathwaysFfxTest";
    appInfo.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instInfo.pApplicationInfo = &appInfo;

    VkInstance instance = VK_NULL_HANDLE;
    VkResult res = vkCreateInstance(&instInfo, nullptr, &instance);
    check_true(res == VK_SUCCESS && instance != VK_NULL_HANDLE, "Failed to create Vulkan 1.4 instance");

    uint32_t devCount = 0;
    vkEnumeratePhysicalDevices(instance, &devCount, nullptr);
    check_true(devCount > 0, "At least one Vulkan physical device must be enumerated");
    std::vector<VkPhysicalDevice> physDevices(devCount);
    vkEnumeratePhysicalDevices(instance, &devCount, physDevices.data());

    // Select discrete hardware GPU (prefer GPU 1 if multiple discrete GPUs are present)
    std::vector<uint32_t> discreteIndices;
    for (uint32_t i = 0; i < devCount; ++i) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(physDevices[i], &props);
        if (props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) {
            discreteIndices.push_back(i);
        }
    }
    uint32_t targetGpuIdx = (discreteIndices.size() >= 2) ? 1u : 0u;
    VkPhysicalDevice targetPhys = discreteIndices.empty() ? physDevices[0] : physDevices[discreteIndices[std::min<size_t>(targetGpuIdx, discreteIndices.size() - 1)]];

    bool hwMl = FfxSdkRuntime::supportsHardwareMl(targetPhys);
    std::cout << "  -> Selected GPU index " << targetGpuIdx << " | API Hardware ML Support (CoopMatrix + 8-bit + MutableDesc): "
              << (hwMl ? "YES" : "NO") << std::endl;

    // -------------------------------------------------------------------------
    // TEST 2: FidelityFX SDK 2.4.0 Effect Version Queries & Bitfield Extraction
    // -------------------------------------------------------------------------
    std::cout << "[TEST 2] FidelityFX SDK 2.4.0 Effect Version Enumeration..." << std::endl;
    FfxSdkRuntime::instance().initialize();
    check_true(FfxSdkRuntime::isCompiledWithSdkHeaders(), "Must be compiled with FidelityFX SDK 2.4+ headers");

#if defined(PATHWAYS_HAS_FFX_SDK)
    const bool hasUpscalerProvider = FfxSdkRuntime::instance().hasNativeProvider(FfxEffectModule::Upscaler);
    const bool expectsFsr4 = hasUpscalerProvider && FfxSdkRuntime::instance().isUpscalerVersionSupported(targetPhys, VK_NULL_HANDLE, 4u);
    auto upscaleVersions = FfxSdkRuntime::instance().queryEffectVersions(FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, targetPhys);
    if (hasUpscalerProvider) {
        check_true(!upscaleVersions.empty(), "Upscale effect versions must not be empty when Upscaler Vulkan provider is loaded");
        bool foundFsr4 = false;
        bool foundFsr3 = false;
        for (const auto& v : upscaleVersions) {
            const uint32_t majorVer = FfxSdkRuntime::parseMajorVersion(v);
            std::cout << "  -> Enumerated Upscaler Version: '" << v.versionName
                      << "' (ID: 0x" << std::hex << v.versionId << std::dec << ", major=" << majorVer << ")" << std::endl;
            if (majorVer >= 4u) foundFsr4 = true;
            if (majorVer == 3u) foundFsr3 = true;
        }
        check_true(foundFsr3, "FSR 3.1.6 analytical upscaler version must always be enumerated");
        check_true(expectsFsr4 == (foundFsr4 && hwMl),
                   "isUpscalerVersionSupported(4) must return true iff provider includes FSR 4 and hardware supports FP8 Cooperative Matrix");
        check_true(FfxSdkRuntime::instance().isUpscalerVersionSupported(targetPhys, VK_NULL_HANDLE, 3u),
                   "isUpscalerVersionSupported(3) must return true when amd_fidelityfx_upscaler_vk is present");
    } else {
        check_true(upscaleVersions.empty(), "Upscale versions must be empty when no Vulkan Upscaler provider binary is loaded (never fabricate versions)");
        check_true(!FfxSdkRuntime::instance().isUpscalerVersionSupported(targetPhys, VK_NULL_HANDLE, 3u),
                   "isUpscalerVersionSupported(3) must return false when no Vulkan Upscaler provider binary is loaded");
        check_true(!expectsFsr4,
                   "isUpscalerVersionSupported(4) must return false when no Vulkan Upscaler provider binary is loaded");
        std::cout << "  -> Vulkan Upscaler Provider (amd_fidelityfx_upscaler_vk): NOT PRESENT on this platform (honestly reported as unsupported)" << std::endl;
    }

    const bool hasDenoiserProvider = FfxSdkRuntime::instance().hasNativeProvider(FfxEffectModule::Denoiser);
    auto denoiserVersions = FfxSdkRuntime::instance().queryEffectVersions(FFX_API_CREATE_CONTEXT_DESC_TYPE_DENOISER, targetPhys);
    if (hasDenoiserProvider) {
        check_true(!denoiserVersions.empty(), "Denoiser versions must not be empty when Denoiser Vulkan provider is loaded");
        std::cout << "  -> Enumerated Denoiser Version: '" << denoiserVersions[0].versionName
                  << "' (ID: 0x" << std::hex << denoiserVersions[0].versionId << std::dec << ")" << std::endl;
    } else {
        check_true(denoiserVersions.empty(), "Denoiser versions must be empty when no Vulkan Denoiser provider DLL is loaded (never fabricate versions)");
        std::cout << "  -> Vulkan Denoiser Provider (amd_fidelityfx_denoiser_vk): NOT PRESENT in SDK 2.4.0 (honestly reported as unsupported)" << std::endl;
    }

    const bool hasRcProvider = FfxSdkRuntime::instance().hasNativeProvider(FfxEffectModule::RadianceCache);
    auto rcVersions = FfxSdkRuntime::instance().queryEffectVersions(FFX_API_CREATE_CONTEXT_DESC_TYPE_RADIANCECACHE, targetPhys);
    if (hasRcProvider) {
        check_true(!rcVersions.empty(), "Radiance Cache versions must not be empty when Radiance Cache Vulkan provider is loaded");
        std::cout << "  -> Enumerated Radiance Cache Version: '" << rcVersions[0].versionName
                  << "' (ID: 0x" << std::hex << rcVersions[0].versionId << std::dec << ")" << std::endl;
    } else {
        check_true(rcVersions.empty(), "Radiance Cache versions must be empty when no Vulkan Radiance Cache provider DLL is loaded (never fabricate versions)");
        std::cout << "  -> Vulkan Radiance Cache Provider (amd_fidelityfx_radiancecache_vk): NOT PRESENT in SDK 2.4.0 (honestly reported as unsupported)" << std::endl;
    }
#endif

    vkDestroyInstance(instance, nullptr);

    // -------------------------------------------------------------------------
    // TEST 3: CLI Configuration Parsing for All FidelityFX Features
    // -------------------------------------------------------------------------
    std::cout << "[TEST 3] CLI Flag Parsing for FSR 3.1, FSR 4.1 ML, Ray Regeneration & Radiance Caching..." << std::endl;
    {
        const char* argvFsr3[] = { "pathways", "--upscaler", "fsr3" };
        Config cFsr3 = Config::parse(3, const_cast<char**>(argvFsr3));
        check_true(cFsr3.upscaler_mode == UpscalerMode::FSR && cFsr3.fsr_version == FsrVersion::Fsr3,
                   "--upscaler fsr3 must select UpscalerMode::FSR and FsrVersion::Fsr3");

        const char* argvFsr4[] = { "pathways", "--upscaler", "fsr4" };
        Config cFsr4 = Config::parse(3, const_cast<char**>(argvFsr4));
        check_true(cFsr4.upscaler_mode == UpscalerMode::FSR && cFsr4.fsr_version == FsrVersion::Fsr4,
                   "--upscaler fsr4 must select UpscalerMode::FSR and FsrVersion::Fsr4");

        const char* argvRR[] = { "pathways", "--denoiser", "rayregen" };
        Config cRR = Config::parse(3, const_cast<char**>(argvRR));
        check_true(cRR.denoiser_mode == DenoiserMode::FfxRayRegeneration,
                   "--denoiser rayregen must select DenoiserMode::FfxRayRegeneration");

        const char* argvNrc[] = { "pathways", "--radiance-cache" };
        Config cNrc = Config::parse(2, const_cast<char**>(argvNrc));
        check_true(cNrc.enable_nrc, "--radiance-cache must enable NRC");
    }

    // -------------------------------------------------------------------------
    // TEST 4: End-to-End GPU Performance & Visual Assessment Suite
    // -------------------------------------------------------------------------
    std::filesystem::path exeDir = std::filesystem::current_path();
    if (argc > 0 && argv[0]) {
        std::error_code ec;
        auto selfPath = std::filesystem::canonical(argv[0], ec);
        if (!ec) exeDir = selfPath.parent_path();
    }

    std::vector<std::filesystem::path> candidateBins = {
        exeDir / "pathways",
        exeDir / "pathways.exe",
        std::filesystem::current_path() / "build/windows-clang-release/bin/pathways.exe",
        std::filesystem::current_path() / "build/bin/pathways",
        std::filesystem::current_path() / "bin/pathways"
    };
    std::filesystem::path pathwaysBin;
    for (const auto& c : candidateBins) {
        if (std::filesystem::exists(c)) {
            pathwaysBin = c;
            break;
        }
    }

    if (pathwaysBin.empty()) {
        std::cout << "[SKIP] pathways binary not found for headless GPU e2e benchmark." << std::endl;
        return 0;
    }

    std::cout << "[TEST 4] Running Dedicated End-to-End GPU Performance & Visual Assessments on GPU " << targetGpuIdx << "..." << std::endl;
    std::error_code ec;
    std::filesystem::path outDir = std::filesystem::current_path() / "output" / "ffx_feature_tests";
    std::filesystem::create_directories(outDir, ec);

    struct FeatureRunResult {
        std::string name;
        double avgFrameMs = 0.0;
        double avgFps = 0.0;
        double rtMs = 0.0;
        double denoiserSrMs = 0.0;
        std::string reportedUpscaler;
        std::string reportedDenoiser;
        bool reportedNrc = false;
        bool reportedSupportsFsr4 = false;
        ImageMetrics visual{};
    };

    auto runFeatureConfig = [&](const std::string& id, const std::string& label, const std::string& extraArgs,
                                const std::vector<uint8_t>* refPixels, int refW, int refH) -> FeatureRunResult {
        std::filesystem::path pngPath = outDir / (id + ".png");
        std::filesystem::path jsonPath = outDir / (id + ".json");
        std::filesystem::path logPath = outDir / (id + ".log");
        std::filesystem::remove(pngPath, ec);
        std::filesystem::remove(jsonPath, ec);

        std::string cmd = std::format(
            "\"{}\" --headless --gpu {} --width 1280 --height 720 --frames 12 --bounces 4 "
            "--dump-frame \"{}\" --dump-stats \"{}\" {} > \"{}\" 2>&1",
            pathwaysBin.string(), targetGpuIdx, pngPath.string(), jsonPath.string(), extraArgs, logPath.string());

#ifdef _WIN32
        std::string sysCmd = "\"" + cmd + "\"";
        int ret = std::system(sysCmd.c_str());
#else
        int ret = std::system(cmd.c_str());
#endif
        check_true(ret == 0, "Headless pathways execution failed for config: " + label + " (see " + logPath.string() + ")");
        check_true(std::filesystem::exists(pngPath), "Dumped PNG missing for config: " + label);
        check_true(std::filesystem::exists(jsonPath), "Dumped JSON missing for config: " + label);

        std::ifstream jf(jsonPath);
        std::string jContent((std::istreambuf_iterator<char>(jf)), std::istreambuf_iterator<char>());

        FeatureRunResult res{};
        res.name = label;
        res.avgFrameMs = extractJsonNumber(jContent, "avg_frame_time_ms");
        res.avgFps = extractJsonNumber(jContent, "avg_fps");
        res.rtMs = extractJsonNumber(jContent, "primary_gpu_time_ms");
        res.denoiserSrMs = extractJsonNumber(jContent, "denoiser_and_upscale_time_ms");
        res.reportedUpscaler = extractJsonString(jContent, "upscaler_mode");
        res.reportedDenoiser = extractJsonString(jContent, "denoiser_mode");
        res.reportedNrc = extractJsonBool(jContent, "enable_nrc");
        res.reportedSupportsFsr4 = extractJsonBool(jContent, "supports_fsr4_ml");
        res.visual = evaluateImageQuality(pngPath.string(), refPixels, refW, refH);
        return res;
    };

    // 1. Render Multi-Frame Reference Pure MC (4 SPP x 12 frames = 48 accumulated samples)
    FeatureRunResult refRun = runFeatureConfig(
        "00_ref_mc", "Reference MC (4 SPP)", "--upscaler none --denoiser none --spp 4", nullptr, 0, 0);
    int refW = 0, refH = 0;
    std::vector<uint8_t> refPixels = loadRgbaPixels((outDir / "00_ref_mc.png").string(), refW, refH);

    // 2. Render 1-SPP Pure MC (No accumulation across frames -> raw 1-SPP noise baseline via --no-accum)
    FeatureRunResult raw1SppRun = runFeatureConfig(
        "01_raw_1spp_mc", "Raw 1-SPP MC (No Accum)", "--upscaler none --denoiser none --spp 1 --no-accum", &refPixels, refW, refH);

    // 3. Render Progressive 1-SPP Pure MC (Standard native baseline)
    FeatureRunResult baseMcRun = runFeatureConfig(
        "02_base_1spp_mc", "Native 1-SPP MC", "--upscaler none --denoiser none --spp 1", &refPixels, refW, refH);

    // 4. Render AMD FSR 3.1.6 Analytical Upscaler (Quality 0.67x)
    FeatureRunResult fsr3Run = runFeatureConfig(
        "03_fsr3_quality", "FSR 3.1 (Quality 0.67x)", "--upscaler fsr3 --render-scale 0.67 --denoiser none --spp 1", &refPixels, refW, refH);

    // 5. Render AMD FSR 4.1.2 ML Upscaler (Quality 0.67x)
    FeatureRunResult fsr4Run = runFeatureConfig(
        "04_fsr4_ml_quality", "FSR 4.1 ML (Quality 0.67x)", "--upscaler fsr4 --render-scale 0.67 --denoiser none --spp 1", &refPixels, refW, refH);

    // 6. Render FFX Ray Regeneration (--denoiser rayregen: strict capability gating, disabled if amd_fidelityfx_denoiser_vk is absent)
    FeatureRunResult rayRegenRun = runFeatureConfig(
        "05_ray_regeneration",
        hasDenoiserProvider ? "FFX Ray Regeneration (1.2.1)" : "Ray Regen (Gated -> Off)",
        "--upscaler none --denoiser rayregen --spp 1", &refPixels, refW, refH);

    // 6b. Render In-House E-SVGF Denoiser (--denoiser svgf)
    FeatureRunResult svgfRun = runFeatureConfig(
        "05b_esvgf", "Denoiser: E-SVGF (Native 1.0x)", "--upscaler none --denoiser svgf --spp 1", &refPixels, refW, refH);

    // 7. Render Radiance Caching (--nrc)
    FeatureRunResult nrcRun = runFeatureConfig(
        "06_radiance_cache",
        hasRcProvider ? "FFX Radiance Caching (0.9.0)" : "NRC (Native Wave32 WMMA)",
        "--upscaler none --denoiser none --nrc --spp 1", &refPixels, refW, refH);

    // 8. Render Upways SR (Quality 0.67x) & Gateway SR (Quality 0.67x)
    FeatureRunResult upwaysSrRun = runFeatureConfig(
        "07a_upways_sr_quality", "Upways SR (Quality 0.67x)", "--upscaler upways --render-scale 0.67 --denoiser none --spp 1", &refPixels, refW, refH);
    FeatureRunResult gatewaySrRun = runFeatureConfig(
        "07b_gateway_sr_quality", "Gateway SR (Quality 0.67x)", "--upscaler gateway --render-scale 0.67 --denoiser none --spp 1", &refPixels, refW, refH);

    // 9. Render Upways Denoiser & Gateway Denoiser at Native 1.0x
    FeatureRunResult upwaysDenRun = runFeatureConfig(
        "07c_upways_denoiser", "Denoiser: Upways (Native 1.0x)", "--upscaler none --denoiser upways --spp 1", &refPixels, refW, refH);
    FeatureRunResult gatewayDenRun = runFeatureConfig(
        "07d_gateway_denoiser", "Denoiser: Gateway (Native 1.0x)", "--upscaler none --denoiser gateway --spp 1", &refPixels, refW, refH);

    // 10. Render ReSTIR PT & Real-Time Caustics
    FeatureRunResult restirRun = runFeatureConfig(
        "07e_restir_pt", "GI: ReSTIR PT (Native 1.0x)", "--upscaler none --denoiser none --restir --spp 1", &refPixels, refW, refH);
    FeatureRunResult causticsRun = runFeatureConfig(
        "07f_caustics", "GI: Caustics (Native 1.0x)", "--upscaler none --denoiser none --caustics --spp 1", &refPixels, refW, refH);

    // 11. Render Combined Stack: Supported Upscaler + Denoiser + Radiance Caching + ReSTIR + Caustics
    FeatureRunResult combinedRun = runFeatureConfig(
        "07_combined_ffx_stack",
        hasDenoiserProvider ? "Combined FSR + RayRegen + GI" : "Combined FSR + E-SVGF + GI",
        hasDenoiserProvider
            ? "--upscaler fsr --render-scale 0.67 --denoiser rayregen --nrc --restir --caustics --spp 1"
            : "--upscaler fsr --render-scale 0.67 --denoiser svgf --nrc --restir --caustics --spp 1",
        &refPixels, refW, refH);

    std::vector<FeatureRunResult> allRuns = {
        raw1SppRun, baseMcRun, fsr3Run, fsr4Run, upwaysSrRun, gatewaySrRun,
        rayRegenRun, svgfRun, upwaysDenRun, gatewayDenRun, nrcRun, restirRun, causticsRun, combinedRun
    };

    std::cout << "\n--------------------------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(32) << "Feature Configuration"
              << std::right << std::setw(10) << "Frame(ms)"
              << std::setw(9)  << "RT(ms)"
              << std::setw(11) << "Den/SR(ms)"
              << std::setw(10) << "FPS"
              << std::setw(11) << "MeanLum"
              << std::setw(12) << "NoiseVar"
              << std::setw(11) << "PSNR(dB)" << "\n";
    std::cout << "--------------------------------------------------------------------------------------------------------------------\n";
    for (const auto& r : allRuns) {
        std::cout << std::left << std::setw(32) << r.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << r.avgFrameMs
                  << std::setw(9)  << r.rtMs
                  << std::setw(11) << r.denoiserSrMs
                  << std::setprecision(1) << std::setw(10) << r.avgFps
                  << std::setprecision(4) << std::setw(11) << r.visual.meanLuminance
                  << std::setprecision(6) << std::setw(12) << r.visual.laplacianNoiseVar
                  << std::setprecision(2) << std::setw(11) << r.visual.psnrVsRefDb << "\n";
    }
    std::cout << "--------------------------------------------------------------------------------------------------------------------\n";

    // Assertions: Active Feature Verification (Strict Zero-Fallback Policy)
    if (hasUpscalerProvider) {
        check_true(fsr3Run.reportedUpscaler == "FSR 3.1",
                   "FSR 3.1 run must report upscaler_mode == 'FSR 3.1', got '" + fsr3Run.reportedUpscaler + "'");
    } else {
        check_true(fsr3Run.reportedUpscaler == "None",
                   "Without amd_fidelityfx_upscaler_vk, --upscaler fsr3 must be rejected and report 'None', got '" + fsr3Run.reportedUpscaler + "'");
    }
    if (expectsFsr4) {
        check_true(fsr4Run.reportedSupportsFsr4,
                   "Hardware ML capable GPU with FSR 4 Upscaler provider must report supports_fsr4_ml == true");
        check_true(fsr4Run.reportedUpscaler == "FSR 4.1 ML",
                   "FSR 4.1 ML run must report upscaler_mode == 'FSR 4.1 ML' without falling back to FSR 3.1, got '" + fsr4Run.reportedUpscaler + "'");
        check_true(combinedRun.reportedUpscaler == "FSR 4.1 ML",
                   "Combined run must report upscaler_mode == 'FSR 4.1 ML'");
    } else {
        check_true(!fsr4Run.reportedSupportsFsr4,
                   "When FSR 4 is not supported by the loaded provider or hardware, supports_fsr4_ml must be false");
        check_true(fsr4Run.reportedUpscaler == "None",
                   "When FSR 4 ML is unsupported or provider is absent, --upscaler fsr4 must be rejected (never fall back to FSR 3.1) and report 'None', got '" + fsr4Run.reportedUpscaler + "'");
        check_true(combinedRun.reportedUpscaler == (hasUpscalerProvider ? "FSR 3.1" : "None"),
                   "Combined --upscaler fsr run must select highest supported FSR version or None");
    }
    if (hasDenoiserProvider) {
        check_true(rayRegenRun.reportedDenoiser.find("Ray Regeneration") != std::string::npos,
                   "Ray Regeneration run must report FFX Ray Regeneration denoiser mode, got '" + rayRegenRun.reportedDenoiser + "'");
    } else {
        check_true(rayRegenRun.reportedDenoiser == "None",
                   "When Vulkan Denoiser provider is absent, --denoiser rayregen must be rejected (never fall back to SVGF) and report 'None', got '" + rayRegenRun.reportedDenoiser + "'");
    }
    check_true(svgfRun.reportedDenoiser == "SVGF",
               "E-SVGF run must report denoiser_mode == 'SVGF', got '" + svgfRun.reportedDenoiser + "'");
    check_true(nrcRun.reportedNrc, "Radiance Cache run must report enable_nrc == true");
    check_true(upwaysSrRun.reportedUpscaler == "Upways", "Upways SR run must report upscaler_mode == 'Upways'");
    check_true(gatewaySrRun.reportedUpscaler == "Gateway", "Gateway SR run must report upscaler_mode == 'Gateway'");
    check_true(upwaysDenRun.reportedDenoiser == "Upways", "Upways Denoiser run must report denoiser_mode == 'Upways'");
    check_true(gatewayDenRun.reportedDenoiser == "Gateway", "Gateway Denoiser run must report denoiser_mode == 'Gateway'");

    // Assertions: Visual Quality & Image Integrity
    for (const auto& r : allRuns) {
        check_true(r.visual.width == 1280 && r.visual.height == 720,
                   r.name + " output resolution must be 1280x720");
        check_true(r.visual.nonBlackRatio > 0.70,
                   r.name + " must not produce black/corrupted frames (nonBlackRatio=" + std::to_string(r.visual.nonBlackRatio) + ")");
        const double lumRatio = r.visual.meanLuminance / std::max(refRun.visual.meanLuminance, 1e-4);
        check_true(lumRatio >= 0.55 && lumRatio <= 1.55,
                   r.name + " mean luminance must preserve scene energy relative to Reference MC (ratio=" + std::to_string(lumRatio) + ")");
        const double minPsnrDb = (r.name.find("No Accum") != std::string::npos) ? 12.0 : 19.0;
        check_true(r.visual.psnrVsRefDb >= minPsnrDb,
                   r.name + " PSNR vs Reference MC must be >= " + std::to_string(minPsnrDb) + " dB (got " + std::to_string(r.visual.psnrVsRefDb) + " dB)");
    }

    // Assertions: Denoiser Visual Noise Reduction Assessment
    const FeatureRunResult& activeDenoiserRun = hasDenoiserProvider ? rayRegenRun : svgfRun;
    check_true(activeDenoiserRun.visual.laplacianNoiseVar < raw1SppRun.visual.laplacianNoiseVar * 0.55,
               "Active denoiser must reduce 1-SPP high-frequency noise variance by >45%");
    check_true(activeDenoiserRun.visual.psnrVsRefDb > raw1SppRun.visual.psnrVsRefDb + 2.0,
               "Active denoiser must improve PSNR over raw 1-SPP MC by at least +2.0 dB");

    // Assertions: Performance Assessment
    if (hasUpscalerProvider) {
        check_true(fsr3Run.rtMs < baseMcRun.rtMs && fsr3Run.denoiserSrMs > 0.0 && fsr3Run.avgFrameMs <= baseMcRun.avgFrameMs * 1.15,
                   "FSR 3.1 (0.67x) must reduce RT time and record valid upscale pass telemetry");
    }
    if (expectsFsr4) {
        check_true(fsr4Run.rtMs < baseMcRun.rtMs && fsr4Run.denoiserSrMs > 0.0 && fsr4Run.denoiserSrMs < 4.0 && fsr4Run.avgFrameMs < 8.33,
                   "FSR 4.1 ML (0.67x) must reduce RT time and record valid ML upscale pass telemetry within 120Hz frame budget");
    }
    check_true(activeDenoiserRun.denoiserSrMs < 3.5,
               "Active denoiser pass time at 720p must be < 3.5 ms (got " + std::to_string(activeDenoiserRun.denoiserSrMs) + " ms)");
    check_true(activeDenoiserRun.rtMs <= baseMcRun.rtMs * 1.25 + 0.35,
               "Enabling denoiser must not inflate Wavefront RT time by >25% (base RT=" +
               std::to_string(baseMcRun.rtMs) + " ms, denoiser RT=" + std::to_string(activeDenoiserRun.rtMs) + " ms)");

    // -------------------------------------------------------------------------
    // TEST 5: Dynamic Runtime Option Toggling (All Quick-Access GUI Options)
    // -------------------------------------------------------------------------
    std::cout << "[TEST 5] Dynamic Runtime Option Toggling (Upscalers, Presets, Denoisers, NRC, ReSTIR, Caustics)..." << std::endl;
    FeatureRunResult dynRun = runFeatureConfig(
        "08_dynamic_options", "Dynamic GUI Option Toggling", "--test-dynamic-options --spp 1", &refPixels, refW, refH);
    check_true(dynRun.visual.width == 1280 && dynRun.visual.height == 720 && dynRun.visual.nonBlackRatio > 0.70,
               "Dynamic GUI option toggling final frame must be valid 1280x720 non-black output");

    std::filesystem::remove_all(outDir, ec);

    std::cout << "==================================================================" << std::endl;
    std::cout << "  ALL FIDELITYFX SDK 2.4 FEATURE & VISUAL/PERF TESTS PASSED!      " << std::endl;
    std::cout << "==================================================================" << std::endl;
    return 0;
}
