#pragma once

#include <vulkan/vulkan.h>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#if defined(PATHWAYS_HAS_FFX_SDK)
#if !defined(_WIN32) && !defined(__declspec)
#define __declspec(x)
#endif
#include <ffx_api.h>
#include <ffx_api.hpp>
#include <vk/ffx_api_vk.h>
#include <vk/ffx_api_vk.hpp>
#include <ffx_upscale.h>
#include <ffx_upscale.hpp>
#include <ffx_denoiser.h>
#include <ffx_denoiser.hpp>
#include <ffx_radiancecache.h>
#endif

namespace pathways {

class Image;
class Buffer;
class VulkanContext;

enum class FfxEffectModule : uint32_t {
    Loader = 0,
    Upscaler = 1,
    Denoiser = 2,
    RadianceCache = 3,
    UpscalerFsr4 = 4
};

struct FfxSdkVersionInfo {
    uint64_t versionId = 0;
    std::string versionName;
};

class FfxSdkRuntime {
public:
    static FfxSdkRuntime& instance();

    // Initialize runtime discovery from SDK signedbin or FFX_DLL_PATH
    void initialize(const std::string& sdkRootDir = "");

    // Returns true if compiled against FidelityFX SDK 2.4+ headers
    static constexpr bool isCompiledWithSdkHeaders() {
#if defined(PATHWAYS_HAS_FFX_SDK)
        return true;
#else
        return false;
#endif
    }

    // Returns true if a native shared library (.so/.dll) provider is loaded for the given module
    bool hasNativeProvider(FfxEffectModule module) const;

    // Human-readable SDK status string for UI and diagnostics
    std::string getSdkStatusString() const;
    const std::string& getSdkRootPath() const { return m_sdkRootDir; }

    // Queries Vulkan API features on the given physical device (never product/architecture name strings):
    // verifies VK_KHR_cooperative_matrix + VkPhysicalDeviceCooperativeMatrixFeaturesKHR::cooperativeMatrix
    // and 8-bit/float8/int8 support required for ML Super-Resolution (FSR 4).
    static bool supportsHardwareMl(VkPhysicalDevice physDevice);

    // Query available effect versions via ffxQueryDescGetVersions
    std::vector<FfxSdkVersionInfo> queryEffectVersions(
        uint64_t createDescType,
        VkPhysicalDevice physDevice = VK_NULL_HANDLE,
        VkDevice device = VK_NULL_HANDLE) const;

    // Extract major version number from versionName ("4.1.2") or versionId bitfields
    static uint32_t parseMajorVersion(const FfxSdkVersionInfo& info);

    // Check whether a specific Upscaler major version (e.g. 3 or 4) is genuinely supported
    // by a loaded Vulkan SDK provider and the underlying Vulkan physical/logical device.
    bool isUpscalerVersionSupported(VkPhysicalDevice physDevice, VkDevice device, uint32_t majorVersion) const;

    // Check whether AMD FSR Ray Regeneration (Denoiser) is genuinely supported on Vulkan
    bool isRayRegenerationSupported(VkPhysicalDevice physDevice, VkDevice device) const;

    // Check whether AMD FSR Radiance Cache is genuinely supported on Vulkan
    bool isRadianceCacheSupported(VkPhysicalDevice physDevice, VkDevice device) const;

#if defined(PATHWAYS_HAS_FFX_SDK)
    // Helper to build FfxApiResource from a Pathways Image*
    static FfxApiResource toFfxResource(const Image* image, uint32_t state = FFX_API_RESOURCE_STATE_UNORDERED_ACCESS, uint32_t additionalUsages = 0);

    // Helper to build FfxApiResource from a Pathways Buffer*
    static FfxApiResource toFfxResource(const Buffer* buffer, uint32_t state = FFX_API_RESOURCE_STATE_UNORDERED_ACCESS);

    // Helper to populate ffx::CreateBackendVKDesc from VulkanContext or raw handles
    static ffx::CreateBackendVKDesc makeBackendVkDesc(VkDevice device, VkPhysicalDevice physDevice);

    // Route ffx API calls to loaded provider or built-in SDK 2.4.0 query handler
    ffxReturnCode_t dispatchCreateContext(ffxContext* context, ffxCreateContextDescHeader* desc, const ffxAllocationCallbacks* memCb);
    ffxReturnCode_t dispatchDestroyContext(ffxContext* context, const ffxAllocationCallbacks* memCb);
    ffxReturnCode_t dispatchConfigure(ffxContext* context, const ffxConfigureDescHeader* desc);
    ffxReturnCode_t dispatchQuery(ffxContext* context, ffxQueryDescHeader* desc);
    ffxReturnCode_t dispatchExecute(ffxContext* context, const ffxDispatchDescHeader* desc);
#endif

private:
    FfxSdkRuntime();
    ~FfxSdkRuntime();

    struct ModuleEntry {
        void* handle = nullptr;
        std::string path;
#if defined(PATHWAYS_HAS_FFX_SDK)
        PfnFfxCreateContext createContext = nullptr;
        PfnFfxDestroyContext destroyContext = nullptr;
        PfnFfxConfigure configure = nullptr;
        PfnFfxQuery query = nullptr;
        PfnFfxDispatch dispatch = nullptr;
#endif
    };

    bool tryLoadModule(FfxEffectModule module, const std::vector<std::string>& candidateNames);
    const ModuleEntry* selectModuleForHeader(uint64_t headerType) const;
    const ModuleEntry* selectModuleForDesc(const ffxApiHeader* desc) const;
    const ModuleEntry* findModuleForContext(const void* ctx) const;

    bool m_initialized = false;
    std::string m_sdkRootDir;
    ModuleEntry m_modules[5]{};
    std::vector<std::pair<void*, const ModuleEntry*>> m_contextOwners;
};

} // namespace pathways
