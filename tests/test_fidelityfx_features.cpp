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
    double edgeRetention = 1.0;
    double psnrVsRefDb = 0.0;
    double rmseVsRef = 0.0;
};

static double computeTemporalFlickerRmse(const std::string& pngPathA, const std::string& pngPathB) {
    int wA = 0, hA = 0, chA = 0;
    int wB = 0, hB = 0, chB = 0;
    stbi_uc* dataA = stbi_load(pngPathA.c_str(), &wA, &hA, &chA, 4);
    stbi_uc* dataB = stbi_load(pngPathB.c_str(), &wB, &hB, &chB, 4);
    check_true(dataA != nullptr && dataB != nullptr && wA == wB && hA == hB && wA > 0 && hA > 0,
               "Failed to load consecutive PNG frames for temporal flicker check");
    const size_t totalPixels = static_cast<size_t>(wA) * static_cast<size_t>(hA);
    double sumSq = 0.0;
    for (size_t i = 0; i < totalPixels; ++i) {
        for (size_t c = 0; c < 3; ++c) {
            const double diff = (static_cast<double>(dataA[i * 4 + c]) - static_cast<double>(dataB[i * 4 + c])) / 255.0;
            sumSq += diff * diff;
        }
    }
    stbi_image_free(dataA);
    stbi_image_free(dataB);
    return std::sqrt(sumSq / static_cast<double>(totalPixels * 3u));
}

static double computeCheckerboardTileParity(const std::string& pngPath, const std::vector<uint8_t>& refPixels, int refW, int refH, double renderScale) {
    int w = 0, h = 0, ch = 0;
    stbi_uc* data = stbi_load(pngPath.c_str(), &w, &h, &ch, 4);
    check_true(data != nullptr && w == refW && h == refH && w > 0 && h > 0,
               "Failed to load PNG for checkerboard tile parity check: " + pngPath);
    const int internalW = std::max(1, static_cast<int>(std::round(static_cast<double>(w) * renderScale)));
    const int internalH = std::max(1, static_cast<int>(std::round(static_cast<double>(h) * renderScale)));

    double oddLum = 0.0, evenLum = 0.0;
    double oddRefLum = 0.0, evenRefLum = 0.0;
    for (int y = 0; y < h; ++y) {
        const int iy = std::clamp((y * internalH) / h, 0, internalH - 1);
        const int tileY = iy / 32;
        for (int x = 0; x < w; ++x) {
            const int ix = std::clamp((x * internalW) / w, 0, internalW - 1);
            const int tileX = ix / 32;
            const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4u;
            const double lum = (0.2126 * data[idx + 0] + 0.7152 * data[idx + 1] + 0.0722 * data[idx + 2]) / 255.0;
            const double rLum = (0.2126 * refPixels[idx + 0] + 0.7152 * refPixels[idx + 1] + 0.0722 * refPixels[idx + 2]) / 255.0;
            if (((tileX + tileY) & 1) != 0) {
                oddLum += lum;
                oddRefLum += rLum;
            } else {
                evenLum += lum;
                evenRefLum += rLum;
            }
        }
    }
    stbi_image_free(data);
    const double oddNorm = (oddRefLum > 1e-6) ? (oddLum / oddRefLum) : 0.0;
    const double evenNorm = (evenRefLum > 1e-6) ? (evenLum / evenRefLum) : 1.0;
    return (evenNorm > 1e-6) ? (oddNorm / evenNorm) : 0.0;
}

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

    const bool hasValidRef = (refPixels && refW == w && refH == h && refPixels->size() == totalPixels * 4u);
    auto refLumAt = [&](int x, int y) -> double {
        x = std::clamp(x, 0, w - 1);
        y = std::clamp(y, 0, h - 1);
        const size_t idx = (static_cast<size_t>(y) * static_cast<size_t>(w) + static_cast<size_t>(x)) * 4u;
        const double r = static_cast<double>((*refPixels)[idx + 0]) / 255.0;
        const double g = static_cast<double>((*refPixels)[idx + 1]) / 255.0;
        const double b = static_cast<double>((*refPixels)[idx + 2]) / 255.0;
        return 0.2126 * r + 0.7152 * g + 0.0722 * b;
    };

    // Evaluate high-frequency Laplacian noise variance across interior pixels.
    // Exclude sharp geometric edges using the converged reference image (|refLap| >= 0.15) when available
    // so high-amplitude 1-SPP Monte Carlo noise spikes on smooth surfaces are never falsely censored as edges.
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
            bool isNonEdge = true;
            if (hasValidRef) {
                const double rc = refLumAt(x, y);
                const double rl = refLumAt(x - 1, y);
                const double rr = refLumAt(x + 1, y);
                const double ru = refLumAt(x, y - 1);
                const double rd = refLumAt(x, y + 1);
                const double refLap = 4.0 * rc - (rl + rr + ru + rd);
                isNonEdge = (std::abs(refLap) < 0.15);
            } else {
                isNonEdge = (std::abs(lap) < 0.25);
            }
            if (isNonEdge) {
                sumLapSq += lap * lap;
                ++lapCount;
            }
        }
    }
    m.laplacianNoiseVar = (lapCount > 0) ? (sumLapSq / static_cast<double>(lapCount)) : 0.0;

    if (hasValidRef) {

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

        // Evaluate structural edge retention on coherent reference geometric edges, using a 3x3 smoothed
        // reference gradient for edge-pixel selection so residual MC grain is never mistaken for an edge.
        auto refSmoothAt = [&](int x, int y) -> double {
            double s = 0.0;
            for (int dy = -1; dy <= 1; ++dy) {
                for (int dx = -1; dx <= 1; ++dx) {
                    s += refLumAt(x + dx, y + dy);
                }
            }
            return s / 9.0;
        };

        double sumImgEdge = 0.0;
        double sumRefEdge = 0.0;
        for (int y = 3; y < h - 3; ++y) {
            for (int x = 3; x < w - 3; ++x) {
                const double sGx = -refSmoothAt(x - 1, y - 1) - 2.0 * refSmoothAt(x - 1, y) - refSmoothAt(x - 1, y + 1)
                                 + refSmoothAt(x + 1, y - 1) + 2.0 * refSmoothAt(x + 1, y) + refSmoothAt(x + 1, y + 1);
                const double sGy = -refSmoothAt(x - 1, y - 1) - 2.0 * refSmoothAt(x, y - 1) - refSmoothAt(x + 1, y - 1)
                                 + refSmoothAt(x - 1, y + 1) + 2.0 * refSmoothAt(x, y + 1) + refSmoothAt(x + 1, y + 1);
                const double smoothGrad = 0.25 * std::sqrt(sGx * sGx + sGy * sGy);
                if (smoothGrad > 0.18) {
                    const double refGx = -refLumAt(x - 1, y - 1) - 2.0 * refLumAt(x - 1, y) - refLumAt(x - 1, y + 1)
                                       + refLumAt(x + 1, y - 1) + 2.0 * refLumAt(x + 1, y) + refLumAt(x + 1, y + 1);
                    const double refGy = -refLumAt(x - 1, y - 1) - 2.0 * refLumAt(x, y - 1) - refLumAt(x + 1, y - 1)
                                       + refLumAt(x - 1, y + 1) + 2.0 * refLumAt(x, y + 1) + refLumAt(x + 1, y + 1);
                    const double refGrad = 0.25 * std::sqrt(refGx * refGx + refGy * refGy);

                    const double imgGx = -lumAt(x - 1, y - 1) - 2.0 * lumAt(x - 1, y) - lumAt(x - 1, y + 1)
                                       + lumAt(x + 1, y - 1) + 2.0 * lumAt(x + 1, y) + lumAt(x + 1, y + 1);
                    const double imgGy = -lumAt(x - 1, y - 1) - 2.0 * lumAt(x, y - 1) - lumAt(x + 1, y - 1)
                                       + lumAt(x - 1, y + 1) + 2.0 * lumAt(x, y + 1) + lumAt(x + 1, y + 1);
                    const double imgGrad = 0.25 * std::sqrt(imgGx * imgGx + imgGy * imgGy);
                    sumImgEdge += imgGrad;
                    sumRefEdge += refGrad;
                }
            }
        }
        m.edgeRetention = (sumRefEdge > 1e-6) ? (sumImgEdge / sumRefEdge) : 1.0;
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
                   "isUpscalerVersionSupported(4) must return true iff provider includes FSR 4 and hardware supports FP8 or INT8 Cooperative Matrix");
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
        vkDestroyInstance(instance, nullptr);
        std::cout << "[SKIP] pathways binary not found for headless GPU e2e benchmark." << std::endl;
        return 0;
    }

    VkPhysicalDeviceProperties targetProps{};
    vkGetPhysicalDeviceProperties(targetPhys, &targetProps);
    if (targetProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        vkDestroyInstance(instance, nullptr);
        std::cout << "[SKIP] CPU-only Vulkan device (" << targetProps.deviceName << "); skipping live hardware GPU e2e benchmark." << std::endl;
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
                                const std::vector<uint8_t>* refPixels, int refW, int refH, int frames = 12) -> FeatureRunResult {
        std::filesystem::path pngPath = outDir / (id + ".png");
        std::filesystem::path jsonPath = outDir / (id + ".json");
        std::filesystem::path logPath = outDir / (id + ".log");
        std::filesystem::remove(pngPath, ec);
        std::filesystem::remove(jsonPath, ec);

        std::string cmd = std::format(
            "\"{}\" --headless --gpu {} --width 1280 --height 720 --frames {} --bounces 4 "
            "--dump-frame \"{}\" --dump-stats \"{}\" {} > \"{}\" 2>&1",
            pathwaysBin.string(), targetGpuIdx, frames, pngPath.string(), jsonPath.string(), extraArgs, logPath.string());

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

    // 1. Render Converged High-Sample Reference Pure MC (16 SPP x 16 frames = 256 accumulated samples)
    FeatureRunResult refRun = runFeatureConfig(
        "00_ref_mc", "Reference MC (256 SPP)", "--upscaler none --denoiser none --spp 16", nullptr, 0, 0, 16);
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

    // 8. Render Upways SR (Quality 0.67x)
    FeatureRunResult upwaysSrRun = runFeatureConfig(
        "07a_upways_sr_quality", "Upways SR (Quality 0.67x)", "--upscaler upways --render-scale 0.67 --denoiser none --spp 1", &refPixels, refW, refH);

    // 9. Render Standalone Denoisers at Native 1.0x & Coupled with FSR 3.1 / FSR 4.1 ML (0.67x)
    FeatureRunResult upwaysDenRun = runFeatureConfig(
        "07c_upways_denoiser", "Denoiser: Upways (Native 1.0x)", "--upscaler none --denoiser upways --spp 1", &refPixels, refW, refH);
    FeatureRunResult fsr3UpwaysDenRun = runFeatureConfig(
        "07c1_fsr3_upways_denoiser", "FSR 3.1 + Upways Denoiser", "--upscaler fsr3 --render-scale 0.67 --denoiser upways --spp 1", &refPixels, refW, refH);
    FeatureRunResult fsr4UpwaysDenRun = runFeatureConfig(
        "07c2_fsr4_upways_denoiser", "FSR 4.1 ML + Upways Denoiser", "--upscaler fsr4 --render-scale 0.67 --denoiser upways --spp 1", &refPixels, refW, refH);
    FeatureRunResult fsr4SvgfDenRun = runFeatureConfig(
        "07c3_fsr4_esvgf_denoiser", "FSR 4.1 ML + E-SVGF Denoiser", "--upscaler fsr4 --render-scale 0.67 --denoiser svgf --spp 1", &refPixels, refW, refH);

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
        raw1SppRun, baseMcRun, fsr3Run, fsr4Run, upwaysSrRun,
        rayRegenRun, svgfRun, upwaysDenRun, fsr3UpwaysDenRun,
        fsr4UpwaysDenRun, fsr4SvgfDenRun,
        nrcRun, restirRun, causticsRun, combinedRun
    };

    std::cout << "\n-------------------------------------------------------------------------------------------------------------------------------\n";
    std::cout << std::left << std::setw(32) << "Feature Configuration"
              << std::right << std::setw(10) << "Frame(ms)"
              << std::setw(9)  << "RT(ms)"
              << std::setw(11) << "Den/SR(ms)"
              << std::setw(10) << "FPS"
              << std::setw(11) << "MeanLum"
              << std::setw(12) << "NoiseVar"
              << std::setw(10) << "EdgeRet"
              << std::setw(11) << "PSNR(dB)" << "\n";
    std::cout << "-------------------------------------------------------------------------------------------------------------------------------\n";
    for (const auto& r : allRuns) {
        std::cout << std::left << std::setw(32) << r.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(10) << r.avgFrameMs
                  << std::setw(9)  << r.rtMs
                  << std::setw(11) << r.denoiserSrMs
                  << std::setprecision(1) << std::setw(10) << r.avgFps
                  << std::setprecision(4) << std::setw(11) << r.visual.meanLuminance
                  << std::setprecision(6) << std::setw(12) << r.visual.laplacianNoiseVar
                  << std::setprecision(3) << std::setw(10) << r.visual.edgeRetention
                  << std::setprecision(2) << std::setw(11) << r.visual.psnrVsRefDb << "\n";
    }
    std::cout << "-------------------------------------------------------------------------------------------------------------------------------\n";

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
        check_true(fsr4UpwaysDenRun.reportedUpscaler == "FSR 4.1 ML" && fsr4UpwaysDenRun.reportedDenoiser == "Upways",
                   "FSR 4.1 ML + Upways Denoiser must report upscaler_mode == 'FSR 4.1 ML' and denoiser_mode == 'Upways'");
        check_true(fsr4SvgfDenRun.reportedUpscaler == "FSR 4.1 ML" && fsr4SvgfDenRun.reportedDenoiser == "SVGF",
                   "FSR 4.1 ML + E-SVGF Denoiser must report upscaler_mode == 'FSR 4.1 ML' and denoiser_mode == 'SVGF'");
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
    check_true(upwaysDenRun.reportedDenoiser == "Upways", "Upways Denoiser run must report denoiser_mode == 'Upways'");
    if (hasUpscalerProvider) {
        check_true(fsr3UpwaysDenRun.reportedUpscaler == "FSR 3.1" && fsr3UpwaysDenRun.reportedDenoiser == "Upways",
                   "FSR 3.1 + Upways Denoiser must report upscaler_mode == 'FSR 3.1' and denoiser_mode == 'Upways'");
    }

    // Assertions: Visual Quality, Energy Conservation, Edge Retention & Image Integrity
    for (const auto& r : allRuns) {
        check_true(r.visual.width == 1280 && r.visual.height == 720,
                   r.name + " output resolution must be 1280x720");
        check_true(r.visual.nonBlackRatio > 0.75,
                   r.name + " must not produce black/corrupted frames (nonBlackRatio=" + std::to_string(r.visual.nonBlackRatio) + ")");
        const double lumRatio = r.visual.meanLuminance / std::max(refRun.visual.meanLuminance, 1e-4);
        const bool isUnaccum1Spp = (r.name.find("No Accum") != std::string::npos);
        const double minLumRatio = isUnaccum1Spp ? 0.75 : 0.85;
        const double maxLumRatio = isUnaccum1Spp ? 1.25 : 1.15;
        check_true(lumRatio >= minLumRatio && lumRatio <= maxLumRatio,
                   r.name + " mean luminance must preserve scene energy within tight bounds relative to 256-SPP Reference MC (ratio=" + std::to_string(lumRatio) + ")");
        check_true(r.visual.edgeRetention >= 0.40,
                   r.name + " structural edge retention must be >= 0.40 (got " + std::to_string(r.visual.edgeRetention) + ")");
        const double minPsnrDb = isUnaccum1Spp ? 12.0 : 20.0;
        check_true(r.visual.psnrVsRefDb >= minPsnrDb,
                   r.name + " PSNR vs 256-SPP Reference MC must be >= " + std::to_string(minPsnrDb) + " dB (got " + std::to_string(r.visual.psnrVsRefDb) + " dB)");
    }

    // Assertions: Surface-Guided & ML Edge Retention
    check_true(upwaysDenRun.visual.edgeRetention >= 0.70 && svgfRun.visual.edgeRetention >= 0.70,
               "Surface-guided native denoisers (Upways, E-SVGF) must retain >= 70% of strong geometric edge gradients");
    check_true(upwaysSrRun.visual.edgeRetention >= 0.70,
               "Upways SR must retain >= 70% of strong geometric edge gradients (got " + std::to_string(upwaysSrRun.visual.edgeRetention) + ")");
    if (expectsFsr4) {
        check_true(fsr4Run.visual.edgeRetention > fsr3Run.visual.edgeRetention + 0.25,
                   "FSR 4.1 ML must preserve significantly sharper structural edges than analytical FSR 3.1");
        check_true(fsr4UpwaysDenRun.visual.edgeRetention >= 0.70 && fsr4SvgfDenRun.visual.edgeRetention >= 0.70,
                   "Coupled FSR 4.1 ML + Upways/E-SVGF Denoiser must retain >= 70% of strong geometric edge gradients");
    }

    // Assertions: Denoiser Visual Noise Reduction & Coupled Upscaler+Denoiser Assessment
    const FeatureRunResult& activeDenoiserRun = hasDenoiserProvider ? rayRegenRun : svgfRun;
    check_true(activeDenoiserRun.visual.laplacianNoiseVar < raw1SppRun.visual.laplacianNoiseVar * 0.20,
               "Active denoiser must reduce 1-SPP high-frequency noise variance by >80%");
    check_true(upwaysDenRun.visual.laplacianNoiseVar < raw1SppRun.visual.laplacianNoiseVar * 0.20,
               "Upways standalone denoiser must reduce 1-SPP high-frequency noise variance by >80%");
    check_true(upwaysSrRun.visual.laplacianNoiseVar < raw1SppRun.visual.laplacianNoiseVar * 0.20,
               "Upways SR coupled upscaler+denoiser must reduce 1-SPP high-frequency noise variance by >80%");
    if (expectsFsr4) {
        check_true(fsr4UpwaysDenRun.visual.laplacianNoiseVar < fsr4Run.visual.laplacianNoiseVar * 0.40,
                   "Coupling FSR 4.1 ML with Upways Denoiser must reduce residual 1-SPP noise variance by >60% vs standalone FSR 4.1 ML");
        check_true(fsr4SvgfDenRun.visual.laplacianNoiseVar < fsr4Run.visual.laplacianNoiseVar * 0.40,
                   "Coupling FSR 4.1 ML with E-SVGF Denoiser must reduce residual 1-SPP noise variance by >60% vs standalone FSR 4.1 ML");
    }

    // Assertions: Performance Assessment
    if (hasUpscalerProvider) {
        check_true(fsr3Run.rtMs < baseMcRun.rtMs && fsr3Run.denoiserSrMs > 0.0 && fsr3Run.avgFrameMs <= baseMcRun.avgFrameMs * 1.25 + 0.15,
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
    // TEST 5: Temporal Stability / Frame-to-Frame Flicker Assessment
    // -------------------------------------------------------------------------
    std::cout << "[TEST 5] Evaluating Frame-to-Frame Temporal Stability (Frame 11 vs Frame 12 Flicker RMSE)..." << std::endl;
    {
        runFeatureConfig("09a_raw_f11", "Raw 1-SPP F11", "--upscaler none --denoiser none --spp 1 --no-accum", nullptr, 0, 0, 11);
        const double rawFlicker = computeTemporalFlickerRmse(
            (outDir / "09a_raw_f11.png").string(), (outDir / "01_raw_1spp_mc.png").string());

        runFeatureConfig("09b_upways_sr_f11", "Upways SR F11", "--upscaler upways --render-scale 0.67 --denoiser none --spp 1", nullptr, 0, 0, 11);
        const double upwaysSrFlicker = computeTemporalFlickerRmse(
            (outDir / "09b_upways_sr_f11.png").string(), (outDir / "07a_upways_sr_quality.png").string());

        runFeatureConfig("09c_fsr4_upways_f11", "FSR4+Upways F11", "--upscaler fsr4 --render-scale 0.67 --denoiser upways --spp 1", nullptr, 0, 0, 11);
        const double fsr4UpwaysFlicker = computeTemporalFlickerRmse(
            (outDir / "09c_fsr4_upways_f11.png").string(), (outDir / "07c2_fsr4_upways_denoiser.png").string());

        std::cout << std::fixed << std::setprecision(5)
                  << "  -> Frame-to-Frame Temporal Flicker RMSE: Raw 1-SPP=" << rawFlicker
                  << " | Upways SR=" << upwaysSrFlicker
                  << " | FSR 4.1 ML + Upways Denoiser=" << fsr4UpwaysFlicker << std::endl;

        check_true(upwaysSrFlicker < rawFlicker * 0.30,
                   "Upways SR must reduce frame-to-frame temporal flicker RMSE by >70% vs raw 1-SPP MC");
        if (expectsFsr4) {
            check_true(fsr4UpwaysFlicker < rawFlicker * 0.30,
                       "FSR 4.1 ML + Upways Denoiser must reduce frame-to-frame temporal flicker RMSE by >70% vs raw 1-SPP MC");
        }
    }

    // -------------------------------------------------------------------------
    // TEST 6: Continuous Camera Motion (--camera-motion) User Experience Test
    // -------------------------------------------------------------------------
    std::cout << "[TEST 6] Evaluating Continuous Camera Motion (--camera-motion) Reprojection & Denoising..." << std::endl;
    {
        FeatureRunResult motionRefRun = runFeatureConfig(
            "10a_motion_ref", "Motion Reference MC (64 SPP)", "--camera-motion --upscaler none --denoiser none --spp 64", nullptr, 0, 0, 12);
        int mRefW = 0, mRefH = 0;
        std::vector<uint8_t> mRefPixels = loadRgbaPixels((outDir / "10a_motion_ref.png").string(), mRefW, mRefH);

        FeatureRunResult motionRaw1Spp = runFeatureConfig(
            "10b_motion_raw_1spp", "Motion: Raw 1-SPP MC", "--camera-motion --upscaler none --denoiser none --spp 1", &mRefPixels, mRefW, mRefH, 12);
        FeatureRunResult motionUpwaysSr = runFeatureConfig(
            "10c_motion_upways_sr", "Motion: Upways SR (0.67x)", "--camera-motion --upscaler upways --render-scale 0.67 --denoiser none --spp 1", &mRefPixels, mRefW, mRefH, 12);
        FeatureRunResult motionFsr4Upways = runFeatureConfig(
            "10d_motion_fsr4_upways", "Motion: FSR 4.1 ML + Upways", "--camera-motion --upscaler fsr4 --render-scale 0.67 --denoiser upways --spp 1", &mRefPixels, mRefW, mRefH, 12);
        FeatureRunResult motionFsr4Svgf = runFeatureConfig(
            "10e_motion_fsr4_svgf", "Motion: FSR 4.1 ML + E-SVGF", "--camera-motion --upscaler fsr4 --render-scale 0.67 --denoiser svgf --spp 1", &mRefPixels, mRefW, mRefH, 12);
        FeatureRunResult motionUpwaysDen = runFeatureConfig(
            "10f_motion_upways_den", "Motion: Upways Denoiser (1.0x)", "--camera-motion --upscaler none --denoiser upways --spp 1", &mRefPixels, mRefW, mRefH, 12);

        std::vector<FeatureRunResult> motionRuns = {
            motionRaw1Spp, motionUpwaysSr, motionFsr4Upways, motionFsr4Svgf, motionUpwaysDen
        };
        for (const auto& mr : motionRuns) {
            std::cout << "  -> " << std::left << std::setw(30) << mr.name
                      << " | FPS=" << std::fixed << std::setprecision(1) << std::setw(7) << mr.avgFps
                      << " | MeanLum=" << std::setprecision(4) << mr.visual.meanLuminance
                      << " | NoiseVar=" << std::setprecision(6) << mr.visual.laplacianNoiseVar
                      << " | EdgeRet=" << std::setprecision(3) << mr.visual.edgeRetention
                      << " | PSNR=" << std::setprecision(2) << mr.visual.psnrVsRefDb << " dB\n";
            check_true(mr.visual.nonBlackRatio > 0.75, mr.name + " must not produce black/corrupted frames under camera motion");
            const double lumRatio = mr.visual.meanLuminance / std::max(motionRefRun.visual.meanLuminance, 1e-4);
            check_true(lumRatio >= 0.80 && lumRatio <= 1.20,
                       mr.name + " must preserve scene energy under continuous camera motion (ratio=" + std::to_string(lumRatio) + ")");
            check_true(mr.visual.edgeRetention >= 0.65,
                       mr.name + " must retain structural edges without severe motion ghosting blur (EdgeRet=" + std::to_string(mr.visual.edgeRetention) + ")");
        }
        check_true(motionUpwaysSr.visual.laplacianNoiseVar < motionRaw1Spp.visual.laplacianNoiseVar * 0.45 &&
                   motionUpwaysSr.visual.psnrVsRefDb > motionRaw1Spp.visual.psnrVsRefDb + 2.5,
                   "Motion Upways SR must significantly reduce 1-SPP noise and boost PSNR under continuous camera motion");
        if (expectsFsr4) {
            check_true(motionFsr4Upways.visual.laplacianNoiseVar < motionRaw1Spp.visual.laplacianNoiseVar * 0.45 &&
                       motionFsr4Upways.visual.psnrVsRefDb > motionRaw1Spp.visual.psnrVsRefDb + 2.5,
                       "Motion FSR 4.1 ML + Upways Denoiser must significantly reduce 1-SPP noise and boost PSNR under continuous camera motion");
        }
    }

    // -------------------------------------------------------------------------
    // TEST 7: Multi-GPU Checkerboard (--mgpu tile) + Neural Upscaler/Denoiser Parity
    // -------------------------------------------------------------------------
    if (discreteIndices.size() >= 2) {
        std::cout << "[TEST 7] Evaluating Multi-GPU Checkerboard (--mgpu tile) with Upways SR & FSR 4.1 ML + Upways Denoiser..." << std::endl;
        FeatureRunResult mgpuUpwaysSr = runFeatureConfig(
            "11a_mgpu_upways_sr", "MGPU Tile: Upways SR (0.67x)", "--mgpu tile --upscaler upways --render-scale 0.67 --denoiser none --spp 1", &refPixels, refW, refH);
        FeatureRunResult mgpuFsr4Upways = runFeatureConfig(
            "11b_mgpu_fsr4_upways", "MGPU Tile: FSR4 + Upways Den", "--mgpu tile --upscaler fsr4 --render-scale 0.67 --denoiser upways --spp 1", &refPixels, refW, refH);
        FeatureRunResult mgpuUpwaysDen = runFeatureConfig(
            "11c_mgpu_upways_den", "MGPU Tile: Upways Den (1.0x)", "--mgpu tile --upscaler none --denoiser upways --spp 1", &refPixels, refW, refH);

        struct MgpuCheckEntry {
            const FeatureRunResult* mgpu;
            const FeatureRunResult* sgpu;
            std::string id;
            double renderScale;
        };
        for (const auto& entry : {
            MgpuCheckEntry{&mgpuUpwaysSr, &upwaysSrRun, "11a_mgpu_upways_sr", 0.67},
            MgpuCheckEntry{&mgpuFsr4Upways, &fsr4UpwaysDenRun, "11b_mgpu_fsr4_upways", 0.67},
            MgpuCheckEntry{&mgpuUpwaysDen, &upwaysDenRun, "11c_mgpu_upways_den", 1.00}
        }) {
            const auto& mgpuR = *entry.mgpu;
            const auto& sgpuR = *entry.sgpu;
            const double mgpuSgpuLumRatio = mgpuR.visual.meanLuminance / std::max(sgpuR.visual.meanLuminance, 1e-4);
            const double tileParity = computeCheckerboardTileParity(
                (outDir / (entry.id + ".png")).string(), refPixels, refW, refH, entry.renderScale);
            std::cout << "  -> " << std::left << std::setw(30) << mgpuR.name
                      << " | FPS=" << std::fixed << std::setprecision(1) << std::setw(7) << mgpuR.avgFps
                      << " | MeanLum=" << std::setprecision(4) << mgpuR.visual.meanLuminance
                      << " (vs 1-GPU ratio=" << std::setprecision(3) << mgpuSgpuLumRatio
                      << ", Odd/Even TileParity=" << std::setprecision(3) << tileParity << ")"
                      << " | NoiseVar=" << std::setprecision(6) << mgpuR.visual.laplacianNoiseVar
                      << " | PSNR=" << std::setprecision(2) << mgpuR.visual.psnrVsRefDb << " dB\n";
            check_true(mgpuSgpuLumRatio >= 0.92 && mgpuSgpuLumRatio <= 1.08,
                       mgpuR.name + " must match Single-GPU mean luminance within ±8% (no double-counted specular or missing odd-tile G-buffers; ratio=" + std::to_string(mgpuSgpuLumRatio) + ")");
            check_true(tileParity >= 0.92 && tileParity <= 1.08,
                       mgpuR.name + " odd vs even 32x32 checkerboard tiles must have matching luminance parity within ±8% (got " + std::to_string(tileParity) + ")");
            check_true(mgpuR.visual.psnrVsRefDb >= 21.5,
                       mgpuR.name + " must achieve PSNR >= 21.5 dB vs 256-SPP Reference MC (got " + std::to_string(mgpuR.visual.psnrVsRefDb) + " dB)");
        }
    }

    // -------------------------------------------------------------------------
    // TEST 8: Dynamic Runtime Option Toggling (All Quick-Access GUI Options)
    // -------------------------------------------------------------------------
    std::cout << "[TEST 8] Dynamic Runtime Option Toggling (Upscalers, Presets, Denoisers, NRC, ReSTIR, Caustics)..." << std::endl;
    FeatureRunResult dynRun = runFeatureConfig(
        "08_dynamic_options", "Dynamic GUI Option Toggling", "--test-dynamic-options --spp 1", &refPixels, refW, refH);
    check_true(dynRun.visual.width == 1280 && dynRun.visual.height == 720 && dynRun.visual.nonBlackRatio > 0.75,
               "Dynamic GUI option toggling final frame must be valid 1280x720 non-black output");
    check_true(dynRun.visual.psnrVsRefDb >= 21.0,
               "Dynamic GUI option toggling final frame must achieve PSNR >= 21.0 dB vs Reference MC (got " + std::to_string(dynRun.visual.psnrVsRefDb) + " dB)");

    if (const char* keepEnv = std::getenv("PATHWAYS_KEEP_TEST_OUTPUT"); !keepEnv || std::string(keepEnv) != "1") {
        std::filesystem::remove_all(outDir, ec);
    }

    std::cout << "==================================================================" << std::endl;
    std::cout << "  ALL FIDELITYFX SDK 2.4 FEATURE & VISUAL/PERF TESTS PASSED!      " << std::endl;
    std::cout << "==================================================================" << std::endl;
    return 0;
}
