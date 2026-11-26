#include "core/Config.hpp"
#include "mgpu/MultiGpuManager.hpp"
#include "mgpu/MultiGpuCoordinator.hpp"
#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#include <cstdlib>

#if defined(_WIN32)
#include <malloc.h>
#else
#include <stdlib.h>
#include <unistd.h>
#endif

using namespace pathways;

static void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "Assertion failed: " << msg << std::endl;
        std::exit(1);
    }
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Multi-GPU Frame Pacing & Host Zero-Copy Test" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // -------------------------------------------------------------------------
    // 1. Test CLI Config Parsing for Multi-GPU Transfer Modes
    // -------------------------------------------------------------------------
    std::cout << "[TEST 1] Multi-GPU Transfer Mode Config Parsing..." << std::endl;
    {
        // Default configuration uses P2P Direct BAR with auto-fallback to Host Zero-Copy
        Config cfgDef;
        check_true(cfgDef.mgpu_transfer_mode == Config::MgpuTransferMode::P2P,
                   "Default mGPU transfer mode must be P2P (Direct BAR)");
        check_true(!cfgDef.mgpu_transfer_explicit,
                   "Default mgpu_transfer_explicit must be false");
        check_true(!cfgDef.camera_motion,
                   "Default camera_motion must be false");

        // Explicit --mgpu-transfer host
        const char* argvHost[] = { "pathways", "--mgpu", "--mgpu-transfer", "host" };
        Config cHost = Config::parse(4, const_cast<char**>(argvHost));
        check_true(cHost.mgpu_mode != MultiGpuMode::Off, "--mgpu enables mGPU");
        check_true(cHost.mgpu_transfer_mode == Config::MgpuTransferMode::Host,
                   "--mgpu-transfer host sets MgpuTransferMode::Host");
        check_true(cHost.mgpu_transfer_explicit,
                   "--mgpu-transfer host sets mgpu_transfer_explicit = true");

        // Explicit --mgpu-transfer p2p
        const char* argvP2P[] = { "pathways", "--mgpu", "--mgpu-transfer", "p2p" };
        Config cP2P = Config::parse(4, const_cast<char**>(argvP2P));
        check_true(cP2P.mgpu_transfer_mode == Config::MgpuTransferMode::P2P,
                   "--mgpu-transfer p2p sets MgpuTransferMode::P2P");
        check_true(cP2P.mgpu_transfer_explicit,
                   "--mgpu-transfer p2p sets mgpu_transfer_explicit = true");

        // Explicit --mgpu-transfer staging rejected under Vulkan 1.4 baseline
        bool caughtStagingError = false;
        try {
            const char* argvStaging[] = { "pathways", "--mgpu", "--mgpu-transfer", "staging" };
            Config::parse(4, const_cast<char**>(argvStaging));
        } catch (const std::runtime_error&) {
            caughtStagingError = true;
        }
        check_true(caughtStagingError,
                   "--mgpu-transfer staging is explicitly rejected under Vulkan 1.4 baseline");

        // Test --camera-motion flag
        const char* argvMotion[] = { "pathways", "--camera-motion" };
        Config cMotion = Config::parse(2, const_cast<char**>(argvMotion));
        check_true(cMotion.camera_motion, "--camera-motion sets camera_motion = true");

        // Test --mgpu tile (Final Frame Upscaling / PostMerge)
        const char* argvTile[] = { "pathways", "--mgpu", "tile", "--upscaler", "fsr3" };
        Config cTile = Config::parse(5, const_cast<char**>(argvTile));
        check_true(cTile.mgpu_mode == MultiGpuMode::CheckerboardTile,
                   "--mgpu tile sets CheckerboardTile mode");
        check_true(cTile.mgpu_upscale_mode == MgpuUpscaleMode::PostMerge,
                   "--mgpu tile sets MgpuUpscaleMode::PostMerge");

        // Test --mgpu sample (Merge Upscaled Frames / SampleBlend)
        const char* argvSample[] = { "pathways", "--mgpu", "sample", "--upscaler", "fsr3" };
        Config cSample = Config::parse(5, const_cast<char**>(argvSample));
        check_true(cSample.mgpu_mode == MultiGpuMode::SampleParallel,
                   "--mgpu sample sets SampleParallel mode");
        check_true(cSample.mgpu_upscale_mode == MgpuUpscaleMode::SampleBlend,
                   "--mgpu sample sets MgpuUpscaleMode::SampleBlend");

        std::cout << "  -> CLI config parsing passed successfully." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 2. Test Dual-GPU Vulkan Host Memory Zero-Copy Import (VK_EXT_external_memory_host)
    // -------------------------------------------------------------------------
    std::cout << "[TEST 2] Dual-GPU VK_EXT_external_memory_host Import..." << std::endl;
    bool hasDualHardwareGpus = false;
    {
        VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
        appInfo.apiVersion = VK_API_VERSION_1_4;

        VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instInfo.pApplicationInfo = &appInfo;

        VkInstance instance = VK_NULL_HANDLE;
        if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan instance." << std::endl;
            return 1;
        }

        uint32_t gpuCount = 0;
        vkEnumeratePhysicalDevices(instance, &gpuCount, nullptr);
        std::vector<VkPhysicalDevice> allDevs(gpuCount);
        if (gpuCount > 0) {
            vkEnumeratePhysicalDevices(instance, &gpuCount, allDevs.data());
        }
        std::vector<VkPhysicalDevice> gpus;
        for (VkPhysicalDevice pd : allDevs) {
            VkPhysicalDeviceProperties props{};
            vkGetPhysicalDeviceProperties(pd, &props);
            if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU && props.apiVersion >= VK_API_VERSION_1_4) {
                gpus.push_back(pd);
            }
        }
        hasDualHardwareGpus = (gpus.size() >= 2);
        if (!hasDualHardwareGpus) {
            std::cout << "  -> Less than 2 hardware Vulkan 1.4 GPUs found (" << gpus.size()
                      << "). Skipping dual-GPU hardware tests (Tests 2 & 3)." << std::endl;
            vkDestroyInstance(instance, nullptr);
        } else {

        // Check if both devices support VK_EXT_external_memory_host
        auto checkExt = [](VkPhysicalDevice physDev, const char* extName) -> bool {
            uint32_t count = 0;
            vkEnumerateDeviceExtensionProperties(physDev, nullptr, &count, nullptr);
            std::vector<VkExtensionProperties> exts(count);
            vkEnumerateDeviceExtensionProperties(physDev, nullptr, &count, exts.data());
            for (const auto& e : exts) {
                if (std::strcmp(e.extensionName, extName) == 0) return true;
            }
            return false;
        };

        bool ext0 = checkExt(gpus[0], "VK_EXT_external_memory_host");
        bool ext1 = checkExt(gpus[1], "VK_EXT_external_memory_host");
        check_true(ext0 && ext1, "Both discrete GPUs must support VK_EXT_external_memory_host");

        // Create logical devices with VK_EXT_external_memory_host enabled
        std::vector<const char*> devExts = { "VK_EXT_external_memory_host" };
        float priority = 1.0f;
        VkDeviceQueueCreateInfo qInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
        qInfo.queueFamilyIndex = 0;
        qInfo.queueCount = 1;
        qInfo.pQueuePriorities = &priority;

        VkDeviceCreateInfo dInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
        dInfo.queueCreateInfoCount = 1;
        dInfo.pQueueCreateInfos = &qInfo;
        dInfo.enabledExtensionCount = static_cast<uint32_t>(devExts.size());
        dInfo.ppEnabledExtensionNames = devExts.data();

        VkDevice dev0 = VK_NULL_HANDLE, dev1 = VK_NULL_HANDLE;
        if (vkCreateDevice(gpus[0], &dInfo, nullptr, &dev0) != VK_SUCCESS ||
            vkCreateDevice(gpus[1], &dInfo, nullptr, &dev1) != VK_SUCCESS) {
            std::cerr << "Failed to create Vulkan logical devices." << std::endl;
            if (dev0) vkDestroyDevice(dev0, nullptr);
            if (dev1) vkDestroyDevice(dev1, nullptr);
            vkDestroyInstance(instance, nullptr);
            return 1;
        }

        auto pfnGetHostPtrProps0 = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev0, "vkGetMemoryHostPointerPropertiesEXT");
        auto pfnGetHostPtrProps1 = (PFN_vkGetMemoryHostPointerPropertiesEXT)vkGetDeviceProcAddr(dev1, "vkGetMemoryHostPointerPropertiesEXT");
        check_true(pfnGetHostPtrProps0 != nullptr && pfnGetHostPtrProps1 != nullptr,
                   "vkGetMemoryHostPointerPropertiesEXT function pointers must be valid");

        VkPhysicalDeviceMemoryProperties memProps0, memProps1;
        vkGetPhysicalDeviceMemoryProperties(gpus[0], &memProps0);
        vkGetPhysicalDeviceMemoryProperties(gpus[1], &memProps1);

        // Allocate 16 MB 64KB page-aligned host memory
        const size_t testBufferSize = 16 * 1024 * 1024;
        const size_t hostAlignment = 65536;
        void* hostPtr = nullptr;
#if defined(_WIN32)
        hostPtr = _aligned_malloc(testBufferSize, hostAlignment);
#else
        if (posix_memalign(&hostPtr, hostAlignment, testBufferSize) != 0) {
            hostPtr = nullptr;
        }
#endif
        check_true(hostPtr != nullptr, "Failed to allocate 64KB-aligned host memory");
        std::memset(hostPtr, 0xAB, testBufferSize);

        // Query host pointer properties on both GPUs
        VkMemoryHostPointerPropertiesEXT hostProps0{ VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
        pfnGetHostPtrProps0(dev0, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, hostPtr, &hostProps0);

        VkMemoryHostPointerPropertiesEXT hostProps1{ VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT };
        pfnGetHostPtrProps1(dev1, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, hostPtr, &hostProps1);

        uint32_t memIdx0 = UINT32_MAX, memIdx1 = UINT32_MAX;
        for (uint32_t i = 0; i < memProps0.memoryTypeCount; ++i) {
            if (hostProps0.memoryTypeBits & (1 << i)) { memIdx0 = i; break; }
        }
        for (uint32_t i = 0; i < memProps1.memoryTypeCount; ++i) {
            if (hostProps1.memoryTypeBits & (1 << i)) { memIdx1 = i; break; }
        }
        check_true(memIdx0 != UINT32_MAX && memIdx1 != UINT32_MAX,
                   "Compatible host-visible memory type found on both GPUs");

        // Import host pointer into Dev 0
        VkImportMemoryHostPointerInfoEXT import0{ VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
        import0.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        import0.pHostPointer = hostPtr;

        VkMemoryAllocateInfo alloc0{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        alloc0.pNext = &import0;
        alloc0.allocationSize = testBufferSize;
        alloc0.memoryTypeIndex = memIdx0;

        VkDeviceMemory mem0 = VK_NULL_HANDLE;
        VkResult res = vkAllocateMemory(dev0, &alloc0, nullptr, &mem0);
        check_true(res == VK_SUCCESS, "Import host memory on Dev 0 must succeed");

        // Import host pointer into Dev 1
        VkImportMemoryHostPointerInfoEXT import1{ VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT };
        import1.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
        import1.pHostPointer = hostPtr;

        VkMemoryAllocateInfo alloc1{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
        alloc1.pNext = &import1;
        alloc1.allocationSize = testBufferSize;
        alloc1.memoryTypeIndex = memIdx1;

        VkDeviceMemory mem1 = VK_NULL_HANDLE;
        res = vkAllocateMemory(dev1, &alloc1, nullptr, &mem1);
        check_true(res == VK_SUCCESS, "Import host memory on Dev 1 must succeed");

        // Create buffer and bind on Dev 0
        VkExternalMemoryBufferCreateInfo extBufInfo0{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
        extBufInfo0.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

        VkBufferCreateInfo bufInfo0{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufInfo0.pNext = &extBufInfo0;
        bufInfo0.size = testBufferSize;
        bufInfo0.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufInfo0.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buf0 = VK_NULL_HANDLE;
        res = vkCreateBuffer(dev0, &bufInfo0, nullptr, &buf0);
        check_true(res == VK_SUCCESS, "vkCreateBuffer on Dev 0 must succeed");
        res = vkBindBufferMemory(dev0, buf0, mem0, 0);
        check_true(res == VK_SUCCESS, "vkBindBufferMemory on Dev 0 must succeed");

        // Create buffer and bind on Dev 1
        VkExternalMemoryBufferCreateInfo extBufInfo1{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
        extBufInfo1.handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;

        VkBufferCreateInfo bufInfo1{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
        bufInfo1.pNext = &extBufInfo1;
        bufInfo1.size = testBufferSize;
        bufInfo1.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bufInfo1.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VkBuffer buf1 = VK_NULL_HANDLE;
        res = vkCreateBuffer(dev1, &bufInfo1, nullptr, &buf1);
        check_true(res == VK_SUCCESS, "vkCreateBuffer on Dev 1 must succeed");
        res = vkBindBufferMemory(dev1, buf1, mem1, 0);
        check_true(res == VK_SUCCESS, "vkBindBufferMemory on Dev 1 must succeed");

        std::cout << "  -> Zero-Copy Host Memory successfully bound on both discrete GPUs!" << std::endl;

        // Cleanup
        vkDestroyBuffer(dev0, buf0, nullptr);
        vkFreeMemory(dev0, mem0, nullptr);
        vkDestroyBuffer(dev1, buf1, nullptr);
        vkFreeMemory(dev1, mem1, nullptr);

#if defined(_WIN32)
        _aligned_free(hostPtr);
#else
        free(hostPtr);
#endif

        vkDestroyDevice(dev0, nullptr);
        vkDestroyDevice(dev1, nullptr);
        vkDestroyInstance(instance, nullptr);
        }
    }

    // -------------------------------------------------------------------------
    // 3. Real MultiGpuManager Lazy Standby Init, Timeline Sync & 6-Plane Buffer Sizing
    // -------------------------------------------------------------------------
    std::cout << "[TEST 3] Real MultiGpuManager Lazy Standby Init & 6-Plane Shared Buffer..." << std::endl;
    if (hasDualHardwareGpus) {
        Config cfgOff;
        cfgOff.headless = true;
        cfgOff.validation_layers = true;
        cfgOff.width = 320;
        cfgOff.height = 180;
        cfgOff.mgpu_mode = MultiGpuMode::Off;

        VulkanContext primaryCtx(cfgOff, VK_NULL_HANDLE, "Primary GPU mGPU Test");
        SceneData cornellScene = ProceduralScene::createCornellBox();

        MultiGpuManager mgpu(cfgOff, &primaryCtx, cornellScene);
        check_true(mgpu.getDeviceCount() >= 2, "MultiGpuManager must detect >= 2 hardware GPUs");
        check_true(!mgpu.isSecondaryInitialized(),
                   "Secondary GPU must NOT initialize lazily when mgpu_mode == MultiGpuMode::Off");

        Config cfgOn = cfgOff;
        cfgOn.mgpu_mode = MultiGpuMode::CheckerboardTile;
        cfgOn.denoiser_mode = DenoiserMode::Upways;
        check_true(mgpu.ensureSecondaryInitialized(cfgOn, cornellScene),
                   "ensureSecondaryInitialized must initialize GPU 1 on demand");
        check_true(mgpu.isSecondaryInitialized() && mgpu.isMultiGpuActive(),
                   "MultiGpuManager must become active after on-demand init");
        check_true(mgpu.isCrossGpuSyncActive(),
                   "Cross-GPU hardware semaphore sync must be active");
        check_true(mgpu.getPrimaryImportedTimelineSemaphore() != VK_NULL_HANDLE,
                   "Primary imported timeline semaphore must be valid");

        // Verify shared buffer reserves space for worst-case dynamic bottom band (topFraction = 0.25)
        const UpwaysBandSplit maxBotSplit = computeUpwaysBandSplit(cfgOn.height, cfgOn.height, cfgOn.upways_band_halo, 0.25f);
        const VkDeviceSize minBandBytes = static_cast<VkDeviceSize>(cfgOn.width) * (cfgOn.height - maxBotSplit.splitHrY) * 4u;
        check_true(mgpu.getSharedBufferSize() >= minBandBytes,
                   "Shared inter-GPU buffer must fit worst-case dynamic bottom band 4 B/px RGB9E5 output");

        // Verify legacy 8-plane CheckerboardTile buffer sizing remains intact when training capture is enabled
        Config cfgLegacy = cfgOn;
        cfgLegacy.capture_training_data_dir = "tmp_cap";
        mgpu.setConfig(cfgLegacy);
        const uint32_t tileSize = (cfgLegacy.tile_size == 0u) ? 64u : cfgLegacy.tile_size;
        const uint32_t numTilesX = (cfgLegacy.width + tileSize - 1u) / tileSize;
        const uint32_t secTileCols = (numTilesX + 1u) / 2u;
        const VkDeviceSize secPixelCount = static_cast<VkDeviceSize>(secTileCols * tileSize) * cfgLegacy.height;
        const VkDeviceSize minExpectedBytes = secPixelCount * 8u + 8u * secPixelCount * 8u; // FP16 radiance + 8x RGBA16F planes
        check_true(mgpu.getSharedBufferSize() >= minExpectedBytes,
                   "Shared inter-GPU buffer must fit radiance + all 8 demodulated denoiser/ML guide planes in legacy capture mode");

        VulkanContext* secCtx = mgpu.getSecondaryContext();
        check_true(secCtx != nullptr && secCtx->getValidationErrors() == 0,
                   "Secondary GPU context must have zero Vulkan validation errors");
        check_true(primaryCtx.getValidationErrors() == 0,
                   "Primary GPU context must have zero Vulkan validation errors");
        std::cout << "  -> Real MultiGpuManager lazy init, timeline sync, and "
                  << (mgpu.getSharedBufferSize() / 1024) << " KiB Band-Parallel + legacy buffer verified (0 validation errors)." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 4. Dynamic Upways Band-Parallel Load Balancer (EMA + Deadband + Hysteresis + Cooldown)
    // -------------------------------------------------------------------------
    std::cout << "[TEST 4] Dynamic Upways Band-Parallel Load Balancer..." << std::endl;
    {
        MultiGpuCoordinator coord;
        constexpr uint32_t kLrH = 1080u;
        constexpr uint32_t kHrH = 2160u;
        constexpr uint32_t kHalo = 24u;

        coord.resetUpwaysBandBalance();
        check_true(std::abs(coord.getUpwaysBandTopFraction() - UPWAYS_DEFAULT_BAND_TOP_FRACTION) < 1e-5f,
                   "Default topFraction starts at 0.50");
        check_true(coord.getUpwaysBandAdjustmentCount() == 0u, "Initial adjustmentCount == 0");

        // Balanced timings remain inside deadband (0 adjustments)
        for (int f = 0; f < 16; ++f) {
            bool adj = coord.updateUpwaysBandBalance(5.0, 5.1, kLrH, kHrH, kHalo, true);
            check_true(!adj, "Balanced GPU timings must stay inside deadband");
        }
        check_true(coord.getUpwaysBandAdjustmentCount() == 0u, "No adjustments under balanced timings");

        // Single-frame spike must not trigger adjustment
        bool spikeAdj = coord.updateUpwaysBandBalance(2.0, 9.0, kLrH, kHrH, kHalo, true);
        check_true(!spikeAdj, "Single-frame spike rejected by hysteresis");
        for (int f = 0; f < 6; ++f) {
            coord.updateUpwaysBandBalance(5.0, 5.0, kLrH, kHrH, kHalo, true);
        }
        check_true(coord.getUpwaysBandAdjustmentCount() == 0u, "Recovery after spike leaves topFraction unchanged");

        // Sustained bottom-heavy imbalance (GPU 1 slower) shifts topFraction upward and respects cooldown
        bool shiftedUp = false;
        for (int f = 0; f < 16; ++f) {
            if (coord.updateUpwaysBandBalance(3.5, 6.5, kLrH, kHrH, kHalo, true)) {
                shiftedUp = true;
                break;
            }
        }
        check_true(shiftedUp, "Sustained bottom-heavy imbalance must increase topFraction");
        check_true(coord.getUpwaysBandTopFraction() > 0.50f && coord.getUpwaysBandTopFraction() <= 0.70f,
                   "Shifted topFraction moves above 0.50 within [0.30, 0.70]");
        const uint32_t adjAfterShift = coord.getUpwaysBandAdjustmentCount();

        // Next 20 frames are inside cooldown window -> 0 adjustments even with extreme imbalance
        for (int f = 0; f < 20; ++f) {
            bool adj = coord.updateUpwaysBandBalance(2.0, 8.0, kLrH, kHrH, kHalo, true);
            check_true(!adj, "Cooldown window must suppress back-to-back resizes");
        }
        check_true(coord.getUpwaysBandAdjustmentCount() == adjAfterShift, "No adjustment during 20-frame cooldown");

        // Top-heavy imbalance shifts topFraction downward and stays >= 0.30
        bool shiftedDown = false;
        for (int f = 0; f < 64; ++f) {
            if (coord.updateUpwaysBandBalance(7.0, 3.0, kLrH, kHrH, kHalo, true)) {
                shiftedDown = true;
            }
        }
        check_true(shiftedDown, "Sustained top-heavy imbalance shifts topFraction downward");
        check_true(coord.getUpwaysBandTopFraction() >= 0.30f && coord.getUpwaysBandTopFraction() <= 0.70f,
                   "topFraction stays clamped inside [0.30, 0.70]");

        // Disabled balancer never mutates state
        const float frozenFrac = coord.getUpwaysBandTopFraction();
        for (int f = 0; f < 12; ++f) {
            bool adj = coord.updateUpwaysBandBalance(1.0, 10.0, kLrH, kHrH, kHalo, false);
            check_true(!adj, "Disabled balancer returns false");
        }
        check_true(coord.getUpwaysBandTopFraction() == frozenFrac, "Disabled balancer never alters topFraction");

        // Reset restores 0.50
        coord.resetUpwaysBandBalance();
        check_true(std::abs(coord.getUpwaysBandTopFraction() - UPWAYS_DEFAULT_BAND_TOP_FRACTION) < 1e-5f,
                   "resetUpwaysBandBalance restores 0.50");
        std::cout << "  -> Dynamic Band-Parallel load balancer EMA/deadband/hysteresis/cooldown verified." << std::endl;
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL MULTI-GPU FRAME PACING TESTS PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
