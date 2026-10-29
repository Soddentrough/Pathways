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
    } else {
        const char* homeDir = std::getenv("HOME");
#ifdef _WIN32
        if (!homeDir || !homeDir[0]) {
            homeDir = std::getenv("USERPROFILE");
        }
#endif
        if (homeDir && homeDir[0]) {
            std::filesystem::path candidate = std::filesystem::path(homeDir) / "Development" / "FidelityFX-SDK";
            std::error_code ec;
            if (std::filesystem::exists(candidate, ec)) {
                m_sdkRootDir = candidate.string();
            }
        }
    }
#if defined(PATHWAYS_FFX_SDK_ROOT_DIR)
    if (m_sdkRootDir.empty()) {
        m_sdkRootDir = PATHWAYS_FFX_SDK_ROOT_DIR;
    }
#endif

#ifdef _WIN32
    // Only load Vulkan provider DLLs (*_vk.dll); never load DirectX 12 (*_dx12.dll) providers into a Vulkan engine
    tryLoadModule(FfxEffectModule::Loader, {"amd_fidelityfx_loader_vk.dll"});
    tryLoadModule(FfxEffectModule::Upscaler, {"amd_fidelityfx_upscaler_vk.dll", "amd_fidelityfx_vk.dll"});
    tryLoadModule(FfxEffectModule::Denoiser, {"amd_fidelityfx_denoiser_vk.dll"});
    tryLoadModule(FfxEffectModule::RadianceCache, {"amd_fidelityfx_radiancecache_vk.dll"});
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
    if (mod.handle != nullptr) {
        return true;
    }
    if (module == FfxEffectModule::Upscaler) {
        const auto& loader = m_modules[static_cast<uint32_t>(FfxEffectModule::Loader)];
        return loader.handle != nullptr;
    }
    return false;
}

uint32_t FfxSdkRuntime::parseMajorVersion(const FfxSdkVersionInfo& info) {
    if (!info.versionName.empty()) {
        const char* s = info.versionName.c_str();
        while (*s && (*s < '0' || *s > '9')) {
            ++s;
        }
        if (*s >= '0' && *s <= '9') {
            uint32_t major = 0;
            while (*s >= '0' && *s <= '9') {
                major = major * 10u + static_cast<uint32_t>(*s - '0');
                ++s;
            }
            if (*s == '.') {
                return major;
            }
        }
    }
    const uint32_t major32 = static_cast<uint32_t>((info.versionId >> 32) & 0xFFFFu);
    if (major32 > 0 && major32 < 32) {
        return major32;
    }
    const uint32_t major48 = static_cast<uint32_t>((info.versionId >> 48) & 0xFFFFu);
    if (major48 > 0 && major48 < 32) {
        return major48;
    }
    const uint32_t major22 = static_cast<uint32_t>(((info.versionId & 0x7FFFFFFFull) >> 22) & 0x1FFu);
    return major22;
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

std::vector<FfxSdkVersionInfo> FfxSdkRuntime::queryEffectVersions(
    uint64_t createDescType,
    VkPhysicalDevice physDevice,
    VkDevice device) const
{
    std::vector<FfxSdkVersionInfo> result;
#if defined(PATHWAYS_HAS_FFX_SDK)
    VkDevice tempDevice = VK_NULL_HANDLE;
    if (physDevice != VK_NULL_HANDLE && device == VK_NULL_HANDLE) {
        float queuePriority = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = 0;
        qci.queueCount = 1;
        qci.pQueuePriorities = &queuePriority;

        VkDeviceCreateInfo dci{};
        dci.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount = 1;
        dci.pQueueCreateInfos = &qci;
        if (vkCreateDevice(physDevice, &dci, nullptr, &tempDevice) == VK_SUCCESS) {
            device = tempDevice;
        }
    }

    VkDeviceContext vkDevCtx{};
    vkDevCtx.vkDevice = device;
    vkDevCtx.vkPhysicalDevice = physDevice;
    vkDevCtx.vkDeviceProcAddr = vkGetDeviceProcAddr;

    ffxQueryDescGetVersions versionQuery{};
    versionQuery.header.type = FFX_API_QUERY_DESC_TYPE_GET_VERSIONS;
    versionQuery.header.pNext = nullptr;
    versionQuery.createDescType = createDescType;
    // Per FidelityFX SDK 2.4 Vulkan API spec, versionQuery.device on Vulkan must be VkDeviceContext*
    versionQuery.device = (physDevice != VK_NULL_HANDLE && device != VK_NULL_HANDLE) ? &vkDevCtx : nullptr;
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

    if (tempDevice != VK_NULL_HANDLE) {
        vkDestroyDevice(tempDevice, nullptr);
    }
#else
    (void)createDescType;
    (void)physDevice;
    (void)device;
#endif
    return result;
}

bool FfxSdkRuntime::isUpscalerVersionSupported(VkPhysicalDevice physDevice, VkDevice device, uint32_t majorVersion) const {
#if defined(PATHWAYS_HAS_FFX_SDK)
    if (!hasNativeProvider(FfxEffectModule::Upscaler)) {
        return false;
    }
    if (majorVersion >= 4u && physDevice != VK_NULL_HANDLE && !supportsHardwareMl(physDevice)) {
        return false;
    }
    auto versions = queryEffectVersions(FFX_API_CREATE_CONTEXT_DESC_TYPE_UPSCALE, physDevice, device);
    for (const auto& v : versions) {
        if (parseMajorVersion(v) == majorVersion) {
            return true;
        }
    }
#else
    (void)physDevice;
    (void)device;
    (void)majorVersion;
#endif
    return false;
}

bool FfxSdkRuntime::isRayRegenerationSupported(VkPhysicalDevice physDevice, VkDevice device) const {
#if defined(PATHWAYS_HAS_FFX_SDK)
    if (!hasNativeProvider(FfxEffectModule::Denoiser)) {
        return false;
    }
    auto versions = queryEffectVersions(FFX_API_CREATE_CONTEXT_DESC_TYPE_DENOISER, physDevice, device);
    return !versions.empty();
#else
    (void)physDevice;
    (void)device;
    return false;
#endif
}

bool FfxSdkRuntime::isRadianceCacheSupported(VkPhysicalDevice physDevice, VkDevice device) const {
#if defined(PATHWAYS_HAS_FFX_SDK)
    if (!hasNativeProvider(FfxEffectModule::RadianceCache)) {
        return false;
    }
    auto versions = queryEffectVersions(FFX_API_CREATE_CONTEXT_DESC_TYPE_RADIANCECACHE, physDevice, device);
    return !versions.empty();
#else
    (void)physDevice;
    (void)device;
    return false;
#endif
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
    if (effectMask == FFX_API_EFFECT_ID_DENOISER) {
        const auto& mod = m_modules[static_cast<uint32_t>(FfxEffectModule::Denoiser)];
        return (mod.handle != nullptr) ? &mod : nullptr;
    }
    if (effectMask == FFX_API_EFFECT_ID_RADIANCECACHE) {
        const auto& mod = m_modules[static_cast<uint32_t>(FfxEffectModule::RadianceCache)];
        return (mod.handle != nullptr) ? &mod : nullptr;
    }
    if (effectMask == FFX_API_EFFECT_ID_UPSCALE) {
        const auto& mod = m_modules[static_cast<uint32_t>(FfxEffectModule::Upscaler)];
        if (mod.handle != nullptr) {
            return &mod;
        }
    }
    const auto& loader = m_modules[static_cast<uint32_t>(FfxEffectModule::Loader)];
    if (loader.handle != nullptr) {
        return &loader;
    }
    return nullptr;
}

const FfxSdkRuntime::ModuleEntry* FfxSdkRuntime::findModuleForContext(const void* ctx) const {
    if (!ctx) return nullptr;
    for (const auto& kv : m_contextOwners) {
        if (kv.first == ctx) {
            return kv.second;
        }
    }
    return nullptr;
}

ffxReturnCode_t FfxSdkRuntime::dispatchCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb) {
    if (!context || !desc) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const ModuleEntry* mod = selectModuleForHeader(desc->type);
    if (mod && mod->createContext) {
        ffxReturnCode_t ret = mod->createContext(context, desc, memCb);
        if (ret == FFX_API_RETURN_OK && *context != nullptr) {
            m_contextOwners.emplace_back(*context, mod);
        }
        return ret;
    }
    *context = nullptr;
    return FFX_API_RETURN_NO_PROVIDER;
}

ffxReturnCode_t FfxSdkRuntime::dispatchDestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb) {
    if (!context || !*context) {
        return FFX_API_RETURN_OK;
    }
    const ModuleEntry* owner = findModuleForContext(*context);
    for (auto it = m_contextOwners.begin(); it != m_contextOwners.end(); ++it) {
        if (it->first == *context) {
            m_contextOwners.erase(it);
            break;
        }
    }
    if (owner && owner->destroyContext) {
        ffxReturnCode_t ret = owner->destroyContext(context, memCb);
        *context = nullptr;
        return ret;
    }
    const auto& upscalerMod = m_modules[static_cast<uint32_t>(FfxEffectModule::Upscaler)];
    if (upscalerMod.handle && upscalerMod.destroyContext) {
        ffxReturnCode_t ret = upscalerMod.destroyContext(context, memCb);
        *context = nullptr;
        return ret;
    }
    *context = nullptr;
    return FFX_API_RETURN_OK;
}

ffxReturnCode_t FfxSdkRuntime::dispatchConfigure(ffxContext* context, const ffxConfigureDescHeader* desc) {
    if (!desc) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const ModuleEntry* mod = (context && *context) ? findModuleForContext(*context) : nullptr;
    if (!mod) {
        mod = selectModuleForHeader(desc->type);
    }
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
    uint64_t lookupType = desc->type;
    if (desc->type == FFX_API_QUERY_DESC_TYPE_GET_VERSIONS) {
        const auto* q = reinterpret_cast<const ffxQueryDescGetVersions*>(desc);
        lookupType = q->createDescType;
    }
    const ModuleEntry* mod = (context && *context) ? findModuleForContext(*context) : nullptr;
    if (!mod) {
        mod = selectModuleForHeader(lookupType);
    }
    if (mod && mod->query) {
        return mod->query(context, desc);
    }

    if (desc->type == FFX_API_QUERY_DESC_TYPE_GET_VERSIONS) {
        auto* q = reinterpret_cast<ffxQueryDescGetVersions*>(desc);
        if (q->outputCount) {
            *q->outputCount = 0;
        }
        return FFX_API_RETURN_NO_PROVIDER;
    }

    return FFX_API_RETURN_NO_PROVIDER;
}

ffxReturnCode_t FfxSdkRuntime::dispatchExecute(ffxContext* context, const ffxDispatchDescHeader* desc) {
    if (!context || !*context || !desc) {
        return FFX_API_RETURN_ERROR_PARAMETER;
    }
    const ModuleEntry* mod = findModuleForContext(*context);
    if (!mod) {
        mod = selectModuleForHeader(desc->type);
    }
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
