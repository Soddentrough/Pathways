#include "rt/FfxSdkRuntime.hpp"
#include "vulkan/Image.hpp"
#include "vulkan/Buffer.hpp"
#include "core/Logger.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#else
    #include <dlfcn.h>
#endif

namespace pathways {

namespace {

float haltonBase(int32_t index, int32_t base) {
    float f = 1.0f;
    float result = 0.0f;
    int32_t current = index;
    while (current > 0) {
        f /= static_cast<float>(base);
        result += f * static_cast<float>(current % base);
        current /= base;
    }
    return result - 0.5f;
}

float qualityModeToRatio(uint32_t qualityMode) {
#if defined(PATHWAYS_HAS_FFX_SDK)
    switch (qualityMode) {
        case FFX_UPSCALE_QUALITY_MODE_NATIVEAA:          return 1.0f;
        case FFX_UPSCALE_QUALITY_MODE_QUALITY:           return 1.5f;
        case FFX_UPSCALE_QUALITY_MODE_BALANCED:          return 1.7f;
        case FFX_UPSCALE_QUALITY_MODE_PERFORMANCE:       return 2.0f;
        case FFX_UPSCALE_QUALITY_MODE_ULTRA_PERFORMANCE: return 3.0f;
        default:                                         return 1.5f;
    }
#else
    (void)qualityMode;
    return 1.5f;
#endif
}

} // namespace

FfxSdkRuntime& FfxSdkRuntime::instance() {
    static FfxSdkRuntime s_instance;
    return s_instance;
}

FfxSdkRuntime::FfxSdkRuntime() {
    initialize();
}

FfxSdkRuntime::~FfxSdkRuntime() {
    for (auto& mod : m_modules) {
        if (mod.handle) {
#ifdef _WIN32
            FreeLibrary(static_cast<HMODULE>(mod.handle));
#else
            dlclose(mod.handle);
#endif
            mod.handle = nullptr;
        }
    }
}

void FfxSdkRuntime::initialize(const std::string& sdkRootDir) {
    if (m_initialized && sdkRootDir.empty()) {
        return;
    }
    m_initialized = true;

    if (!sdkRootDir.empty()) {
        m_sdkRootDir = sdkRootDir;
    } else if (const char* envDir = std::getenv("FFX_SDK_PATH")) {
        m_sdkRootDir = envDir;
    } else if (const char* homeDir = std::getenv("HOME")) {
        std::string candidate = std::string(homeDir) + "/Development/FidelityFX-SDK";
        std::error_code ec;
        if (std::filesystem::exists(candidate, ec)) {
            m_sdkRootDir = candidate;
        }
    }
#if defined(PATHWAYS_FFX_SDK_ROOT_DIR)
    if (m_sdkRootDir.empty()) {
        m_sdkRootDir = PATHWAYS_FFX_SDK_ROOT_DIR;
    }
#endif

#ifdef _WIN32
    tryLoadModule(FfxEffectModule::Loader, {"amd_fidelityfx_loader_vk.dll", "amd_fidelityfx_loader_dx12.dll"});
    tryLoadModule(FfxEffectModule::Upscaler, {"amd_fidelityfx_upscaler_vk.dll", "amd_fidelityfx_vk.dll"});
    tryLoadModule(FfxEffectModule::Denoiser, {"amd_fidelityfx_denoiser_vk.dll", "amd_fidelityfx_denoiser_dx12.dll"});
    tryLoadModule(FfxEffectModule::RadianceCache, {"amd_fidelityfx_radiancecache_vk.dll", "amd_fidelityfx_radiancecache_dx12.dll"});
#else
    tryLoadModule(FfxEffectModule::Loader, {"libamd_fidelityfx_loader_vk.so", "amd_fidelityfx_loader_vk.so"});
    tryLoadModule(FfxEffectModule::Upscaler, {"libamd_fidelityfx_upscaler_vk.so", "amd_fidelityfx_upscaler_vk.so"});
    tryLoadModule(FfxEffectModule::Denoiser, {"libamd_fidelityfx_denoiser_vk.so", "amd_fidelityfx_denoiser_vk.so"});
    tryLoadModule(FfxEffectModule::RadianceCache, {"libamd_fidelityfx_radiancecache_vk.so", "amd_fidelityfx_radiancecache_vk.so"});
#endif
}

bool FfxSdkRuntime::tryLoadModule(FfxEffectModule module, const std::vector<std::string>& candidateNames) {
    auto& entry = m_modules[static_cast<uint32_t>(module)];
    if (entry.handle != nullptr) {
        return true;
    }

    std::vector<std::string> searchDirs;
    if (const char* dllEnv = std::getenv("FFX_DLL_PATH")) {
        searchDirs.emplace_back(dllEnv);
    }
    if (!m_sdkRootDir.empty()) {
        searchDirs.emplace_back(m_sdkRootDir + "/Kits/FidelityFX/signedbin");
        searchDirs.emplace_back(m_sdkRootDir + "/bin");
    }
    searchDirs.emplace_back(".");

    for (const auto& dir : searchDirs) {
        for (const auto& name : candidateNames) {
            std::filesystem::path fullPath = std::filesystem::path(dir) / name;
            std::error_code ec;
            if (!std::filesystem::exists(fullPath, ec)) {
                continue;
            }
#ifdef _WIN32
            HMODULE hMod = LoadLibraryA(fullPath.string().c_str());
            if (!hMod) continue;
            entry.handle = hMod;
    #if defined(PATHWAYS_HAS_FFX_SDK)
            entry.createContext  = reinterpret_cast<PfnFfxCreateContext>(GetProcAddress(hMod, "ffxCreateContext"));
            entry.destroyContext = reinterpret_cast<PfnFfxDestroyContext>(GetProcAddress(hMod, "ffxDestroyContext"));
            entry.configure      = reinterpret_cast<PfnFfxConfigure>(GetProcAddress(hMod, "ffxConfigure"));
            entry.query          = reinterpret_cast<PfnFfxQuery>(GetProcAddress(hMod, "ffxQuery"));
            entry.dispatch       = reinterpret_cast<PfnFfxDispatch>(GetProcAddress(hMod, "ffxDispatch"));
    #endif
#else
            void* hMod = dlopen(fullPath.string().c_str(), RTLD_NOW | RTLD_LOCAL);
            if (!hMod) continue;
            entry.handle = hMod;
    #if defined(PATHWAYS_HAS_FFX_SDK)
            entry.createContext  = reinterpret_cast<PfnFfxCreateContext>(dlsym(hMod, "ffxCreateContext"));
            entry.destroyContext = reinterpret_cast<PfnFfxDestroyContext>(dlsym(hMod, "ffxDestroyContext"));
            entry.configure      = reinterpret_cast<PfnFfxConfigure>(dlsym(hMod, "ffxConfigure"));
            entry.query          = reinterpret_cast<PfnFfxQuery>(dlsym(hMod, "ffxQuery"));
            entry.dispatch       = reinterpret_cast<PfnFfxDispatch>(dlsym(hMod, "ffxDispatch"));
    #endif
#endif
            entry.path = fullPath.string();
            Logger::info("[FfxSdkRuntime] Loaded FidelityFX provider module: {}", entry.path);
            return true;
        }
    }
    return false;
}

bool FfxSdkRuntime::hasNativeProvider(FfxEffectModule module) const {
    const auto& mod = m_modules[static_cast<uint32_t>(module)];
    const auto& loader = m_modules[static_cast<uint32_t>(FfxEffectModule::Loader)];
    return (mod.handle != nullptr) || (loader.handle != nullptr);
}

std::string FfxSdkRuntime::getSdkStatusString() const {
#if defined(PATHWAYS_HAS_FFX_SDK)
    bool hasProvider = hasNativeProvider(FfxEffectModule::Upscaler) ||
                       hasNativeProvider(FfxEffectModule::Denoiser) ||
                       hasNativeProvider(FfxEffectModule::RadianceCache);
    if (hasProvider) {
        return "FidelityFX SDK 2.4.0 (Vulkan Provider Loaded)";
    }
    return "FidelityFX SDK 2.4.0 (Headers + Native Vulkan 1.4 Compute)";
#else
    return "Native Vulkan 1.4 Compute (SDK Headers Not Found)";
#endif
}

bool FfxSdkRuntime::supportsHardwareMl(VkPhysicalDevice physDevice) {
    if (physDevice == VK_NULL_HANDLE) {
        return false;
    }

    uint32_t extCount = 0;
    if (vkEnumerateDeviceExtensionProperties(physDevice, nullptr, &extCount, nullptr) != VK_SUCCESS || extCount == 0) {
        return false;
    }
    std::vector<VkExtensionProperties> exts(extCount);
    if (vkEnumerateDeviceExtensionProperties(physDevice, nullptr, &extCount, exts.data()) != VK_SUCCESS) {
        return false;
    }

    bool hasCoopMatrixExt = false;
    bool hasFloat8Ext = false;
    for (const auto& ext : exts) {
        if (std::strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
            hasCoopMatrixExt = true;
        }
#ifdef VK_EXT_shader_float8
        if (std::strcmp(ext.extensionName, VK_EXT_SHADER_FLOAT8_EXTENSION_NAME) == 0) {
            hasFloat8Ext = true;
        }
#endif
    }
    if (!hasCoopMatrixExt) {
        return false;
    }

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeatures{};
    coopFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;

    VkPhysicalDeviceVulkan12Features vk12Features{};
    vk12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    vk12Features.pNext = &coopFeatures;

#ifdef VK_EXT_shader_float8
    VkPhysicalDeviceShaderFloat8FeaturesEXT float8Features{};
    float8Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT8_FEATURES_EXT;
    if (hasFloat8Ext) {
        coopFeatures.pNext = &float8Features;
    }
#endif

    VkPhysicalDeviceFeatures2 feat2{};
    feat2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    feat2.pNext = &vk12Features;
    vkGetPhysicalDeviceFeatures2(physDevice, &feat2);

    if (coopFeatures.cooperativeMatrix != VK_TRUE || vk12Features.shaderFloat16 != VK_TRUE) {
        return false;
    }

    bool has8BitSupport = (vk12Features.shaderInt8 == VK_TRUE && vk12Features.storageBuffer8BitAccess == VK_TRUE);
#ifdef VK_EXT_shader_float8
    if (hasFloat8Ext && float8Features.shaderFloat8CooperativeMatrix == VK_TRUE) {
        has8BitSupport = true;
    }
#endif
    return has8BitSupport;
}

std::vector<FfxSdkVersionInfo> FfxSdkRuntime::queryEffectVersions(uint64_t createDescType, VkPhysicalDevice physDevice) const {
    std::vector<FfxSdkVersionInfo> result;
#if defined(PATHWAYS_HAS_FFX_SDK)
    ffxQueryDescGetVersions versionQuery{};
    versionQuery.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versionQuery.header.pNext = nullptr;
    versionQuery.createDescType = createDescType;
    versionQuery.device = physDevice;
    uint64_t count = 0;
    versionQuery.outputCount = &count;
    versionQuery.versionIds = nullptr;
    versionQuery.versionNames = nullptr;

    if (ffxQuery(nullptr, &versionQuery.header) == FFX_API_RETURN_OK && count > 0) {
        std::vector<uint64_t> ids(count, 0);
        std::vector<const char*> names(count, nullptr);
        versionQuery.versionIds = ids.data();
        versionQuery.versionNames = names.data();
        if (ffxQuery(nullptr, &versionQuery.header) == FFX_API_RETURN_OK) {
            for (uint64_t i = 0; i < count; ++i) {
                FfxSdkVersionInfo info{};
                info.versionId = ids[i];
                info.versionName = names[i] ? names[i] : "";
                result.push_back(info);
            }
        }
    }
#else
    (void)createDescType;
    (void)physDevice;
#endif
    return result;
}

#if defined(PATHWAYS_HAS_FFX_SDK)

FfxApiResource FfxSdkRuntime::toFfxResource(const Image* image, uint32_t state, uint32_t additionalUsages) {
    if (!image || image->getImage() == VK_NULL_HANDLE) {
        FfxApiResource empty{};
        return empty;
    }
    VkImageCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO;
    createInfo.imageType = VK_IMAGE_TYPE_2D;
    createInfo.format = image->getFormat();
    createInfo.extent.width = image->getWidth();
    createInfo.extent.height = image->getHeight();
    createInfo.extent.depth = 1;
    createInfo.mipLevels = image->getMipLevels();
    createInfo.arrayLayers = 1;
    createInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    createInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    createInfo.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;

    FfxApiResourceDescription desc = ffxApiGetImageResourceDescriptionVK(image->getImage(), createInfo, additionalUsages);
    return ffxApiGetResourceVK(reinterpret_cast<void*>(image->getImage()), desc, state);
}

FfxApiResource FfxSdkRuntime::toFfxResource(const Buffer* buffer, uint32_t state) {
    if (!buffer || buffer->getBuffer() == VK_NULL_HANDLE) {
        FfxApiResource empty{};
        return empty;
    }
    VkBufferCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    createInfo.size = buffer->getSize();
    createInfo.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;

    FfxApiResourceDescription desc = ffxApiGetBufferResourceDescriptionVK(buffer->getBuffer(), createInfo, FFX_API_RESOURCE_USAGE_UAV);
    return ffxApiGetResourceVK(reinterpret_cast<void*>(buffer->getBuffer()), desc, state);
}

ffx::CreateBackendVKDesc FfxSdkRuntime::makeBackendVkDesc(VkDevice device, VkPhysicalDevice physDevice) {
    ffx::CreateBackendVKDesc backendDesc{};
    backendDesc.deviceContext.vkDevice = device;
    backendDesc.deviceContext.vkPhysicalDevice = physDevice;
    backendDesc.deviceContext.vkDeviceProcAddr = vkGetDeviceProcAddr;
    return backendDesc;
}

const FfxSdkRuntime::ModuleEntry* FfxSdkRuntime::selectModuleForHeader(uint64_t headerType) const {
    uint64_t effectMask = headerType & FFX_API_EFFECT_MASK;
    const ModuleEntry* chosen = nullptr;
    if (effectMask == FFX_API_EFFECT_ID_UPSCALE) {
        chosen = &m_modules[static_cast<uint32_t>(FfxEffectModule::Upscaler)];
    } else if (effectMask == FFX_API_EFFECT_ID_DENOISER) {
        chosen = &m_modules[static_cast<uint32_t>(FfxEffectModule::Denoiser)];
    } else if (effectMask == FFX_API_EFFECT_ID_RADIANCECACHE) {
        chosen = &m_modules[static_cast<uint32_t>(FfxEffectModule::RadianceCache)];
    }
    if (chosen && chosen->handle != nullptr) {
        return chosen;
    }
    const auto& loader = m_modules[static_cast<uint32_t>(FfxEffectModule::Loader)];
    if (loader.handle != nullptr) {
        return &loader;
    }
    return nullptr;
}

ffxReturnCode_t FfxSdkRuntime::dispatchCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb) {
    if (!context || !desc) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const ModuleEntry* mod = selectModuleForHeader(desc->type);
    if (mod && mod->createContext) {
        return mod->createContext(context, desc, memCb);
    }
    *context = nullptr;
    return FFX_API_RETURN_NO_PROVIDER;
}

ffxReturnCode_t FfxSdkRuntime::dispatchDestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb) {
    if (!context || !*context) {
        return FFX_API_RETURN_OK;
    }
    for (const auto& mod : m_modules) {
        if (mod.handle && mod.destroyContext) {
            return mod.destroyContext(context, memCb);
        }
    }
    *context = nullptr;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t FfxSdkRuntime::dispatchConfigure(ffxContext* context, const ffxConfigureDescHeader* desc) {
    if (!desc) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const ModuleEntry* mod = selectModuleForHeader(desc->type);
    if (mod && mod->configure) {
        return mod->configure(context, desc);
    }
    if (desc->type == FFX_API_CONFIGURE_DESC_TYPE_GLOBALDEBUG) {
        return FFX_API_RETURN_OK;
    }
    return FFX_API_RETURN_NO_PROVIDER;
}

ffxReturnCode_t FfxSdkRuntime::dispatchQuery(ffxContext* context, ffxQueryDescHeader* desc) {
    if (!desc) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const ModuleEntry* mod = selectModuleForHeader(desc->type);
    if (mod && mod->query) {
        return mod->query(context, desc);
    }

    // Built-in SDK 2.4.0 specification query responder when no external provider binary is loaded
    switch (desc->type) {
        case FFX_API_QUERY_DESC_TYPE_GET_VERSIONS: {
            auto* q = reinterpret_cast<ffxQueryDescGetVersions*>(desc);
            if (!q->outputCount) {
                return FFX_API_RETURN_ERROR_PARAMETER;
            }
            uint64_t effectMask = q->createDescType & FFX_API_EFFECT_MASK;
            if (effectMask == FFX_API_EFFECT_ID_UPSCALE) {
                const bool mlSupported = (q->device == nullptr) ||
                    supportsHardwareMl(reinterpret_cast<VkPhysicalDevice>(q->device));
                static constexpr uint64_t kUpscalerIdsAll[2] = {
                    0x000400010002ull, // FSR 4.1.2 (ML Super Resolution)
                    0x000300010006ull  // FSR 3.1.6 (Analytical Fallback)
                };
                static constexpr const char* kUpscalerNamesAll[2] = {
                    "4.1.2",
                    "3.1.6"
                };
                const uint64_t* idsPtr = mlSupported ? kUpscalerIdsAll : &kUpscalerIdsAll[1];
                const char* const* namesPtr = mlSupported ? kUpscalerNamesAll : &kUpscalerNamesAll[1];
                const uint64_t avail = mlSupported ? 2u : 1u;
                if (!q->versionIds && !q->versionNames) {
                    *q->outputCount = avail;
                    return FFX_API_RETURN_OK;
                }
                uint64_t toCopy = (*q->outputCount < avail) ? *q->outputCount : avail;
                for (uint64_t i = 0; i < toCopy; ++i) {
                    if (q->versionIds) q->versionIds[i] = idsPtr[i];
                    if (q->versionNames) q->versionNames[i] = namesPtr[i];
                }
                *q->outputCount = toCopy;
                return FFX_API_RETURN_OK;
            } else if (effectMask == FFX_API_EFFECT_ID_DENOISER) {
                static constexpr uint64_t kDenoiserId = 0x000100020001ull; // 1.2.1
                static constexpr const char* kDenoiserName = "1.2.1";
                if (!q->versionIds && !q->versionNames) {
                    *q->outputCount = 1;
                    return FFX_API_RETURN_OK;
                }
                if (*q->outputCount >= 1) {
                    if (q->versionIds) q->versionIds[0] = kDenoiserId;
                    if (q->versionNames) q->versionNames[0] = kDenoiserName;
                    *q->outputCount = 1;
                }
                return FFX_API_RETURN_OK;
            } else if (effectMask == FFX_API_EFFECT_ID_RADIANCECACHE) {
                static constexpr uint64_t kRcId = 0x000000090000ull; // 0.9.0
                static constexpr const char* kRcName = "0.9.0";
                if (!q->versionIds && !q->versionNames) {
                    *q->outputCount = 1;
                    return FFX_API_RETURN_OK;
                }
                if (*q->outputCount >= 1) {
                    if (q->versionIds) q->versionIds[0] = kRcId;
                    if (q->versionNames) q->versionNames[0] = kRcName;
                    *q->outputCount = 1;
                }
                return FFX_API_RETURN_OK;
            }
            *q->outputCount = 0;
            return FFX_API_RETURN_OK;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETUPSCALERATIOFROMQUALITYMODE: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetUpscaleRatioFromQualityMode*>(desc);
            if (!q->pOutUpscaleRatio) return FFX_API_RETURN_ERROR_PARAMETER;
            *q->pOutUpscaleRatio = qualityModeToRatio(q->qualityMode);
            return FFX_API_RETURN_OK;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETRENDERRESOLUTIONFROMQUALITYMODE: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetRenderResolutionFromQualityMode*>(desc);
            if (!q->pOutRenderWidth || !q->pOutRenderHeight) return FFX_API_RETURN_ERROR_PARAMETER;
            float ratio = qualityModeToRatio(q->qualityMode);
            *q->pOutRenderWidth  = static_cast<uint32_t>(std::max(1.0f, std::round(static_cast<float>(q->displayWidth) / ratio)));
            *q->pOutRenderHeight = static_cast<uint32_t>(std::max(1.0f, std::round(static_cast<float>(q->displayHeight) / ratio)));
            return FFX_API_RETURN_OK;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTERPHASECOUNT: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterPhaseCount*>(desc);
            if (!q->pOutPhaseCount || q->renderWidth == 0) return FFX_API_RETURN_ERROR_PARAMETER;
            float scale = static_cast<float>(q->displayWidth) / static_cast<float>(q->renderWidth);
            *q->pOutPhaseCount = std::max(8, static_cast<int32_t>(std::ceil(8.0f * scale * scale)));
            return FFX_API_RETURN_OK;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GETJITTEROFFSET: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetJitterOffset*>(desc);
            if (!q->pOutX || !q->pOutY || q->phaseCount <= 0) return FFX_API_RETURN_ERROR_PARAMETER;
            int32_t idx = (q->index % q->phaseCount) + 1;
            *q->pOutX = haltonBase(idx, 2);
            *q->pOutY = haltonBase(idx, 3);
            return FFX_API_RETURN_OK;
        }
        case FFX_API_QUERY_DESC_TYPE_UPSCALE_GET_RESOURCE_REQUIREMENTS: {
            auto* q = reinterpret_cast<ffxQueryDescUpscaleGetResourceRequirements*>(desc);
            q->required_resources = FFX_API_QUERY_RESOURCE_INPUT_COLOR |
                                    FFX_API_QUERY_RESOURCE_INPUT_DEPTH |
                                    FFX_API_QUERY_RESOURCE_INPUT_MV;
            q->optional_resources = FFX_API_QUERY_RESOURCE_INPUT_EXPOSURE |
                                    FFX_API_QUERY_RESOURCE_INPUT_REACTIVEMASK |
                                    FFX_API_QUERY_RESOURCE_INPUT_TRANSPARENCYCOMPOSITION;
            return FFX_API_RETURN_OK;
        }
        default:
            break;
    }

    return FFX_API_RETURN_NO_PROVIDER;
}

ffxReturnCode_t FfxSdkRuntime::dispatchExecute(ffxContext* context, const ffxDispatchDescHeader* desc) {
    if (!context || !*context || !desc) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const ModuleEntry* mod = selectModuleForHeader(desc->type);
    if (mod && mod->dispatch) {
        return mod->dispatch(context, desc);
    }
    return FFX_API_RETURN_NO_PROVIDER;
}

#endif // PATHWAYS_HAS_FFX_SDK

} // namespace pathways

#if defined(PATHWAYS_HAS_FFX_SDK)
extern "C" {

FFX_API_ENTRY ffxReturnCode_t ffxCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb) {
    return pathways::FfxSdkRuntime::instance().dispatchCreateContext(context, desc, memCb);
}

FFX_API_ENTRY ffxReturnCode_t ffxDestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb) {
    return pathways::FfxSdkRuntime::instance().dispatchDestroyContext(context, memCb);
}

FFX_API_ENTRY ffxReturnCode_t ffxConfigure(ffxContext* context, const ffxConfigureDescHeader* desc) {
    return pathways::FfxSdkRuntime::instance().dispatchConfigure(context, desc);
}

FFX_API_ENTRY ffxReturnCode_t ffxQuery(ffxContext* context, ffxQueryDescHeader* desc) {
    return pathways::FfxSdkRuntime::instance().dispatchQuery(context, desc);
}

FFX_API_ENTRY ffxReturnCode_t ffxDispatch(ffxContext* context, const ffxDispatchDescHeader* desc) {
    return pathways::FfxSdkRuntime::instance().dispatchExecute(context, desc);
}

} // extern "C"
#endif
