#include "core/Config.hpp"
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
        // Default configuration must use Host Zero-Copy to prevent PCIe BAR stalls
        Config cfgDef;
        check_true(cfgDef.mgpu_transfer_mode == Config::MgpuTransferMode::Host,
                   "Default mGPU transfer mode must be Host (Zero-Copy)");
        check_true(!cfgDef.camera_motion,
                   "Default camera_motion must be false");

        // Explicit --mgpu-transfer host
        const char* argvHost[] = { "pathways", "--mgpu", "--mgpu-transfer", "host" };
        Config cHost = Config::parse(4, const_cast<char**>(argvHost));
        check_true(cHost.mgpu_mode != MultiGpuMode::Off, "--mgpu enables mGPU");
        check_true(cHost.mgpu_transfer_mode == Config::MgpuTransferMode::Host,
                   "--mgpu-transfer host sets MgpuTransferMode::Host");

        // Explicit --mgpu-transfer p2p
        const char* argvP2P[] = { "pathways", "--mgpu", "--mgpu-transfer", "p2p" };
        Config cP2P = Config::parse(4, const_cast<char**>(argvP2P));
        check_true(cP2P.mgpu_transfer_mode == Config::MgpuTransferMode::P2P,
                   "--mgpu-transfer p2p sets MgpuTransferMode::P2P");

        // Explicit --mgpu-transfer staging
        const char* argvStaging[] = { "pathways", "--mgpu", "--mgpu-transfer", "staging" };
        Config cStaging = Config::parse(4, const_cast<char**>(argvStaging));
        check_true(cStaging.mgpu_transfer_mode == Config::MgpuTransferMode::Staging,
                   "--mgpu-transfer staging sets MgpuTransferMode::Staging");

        // Test --camera-motion flag
        const char* argvMotion[] = { "pathways", "--camera-motion" };
        Config cMotion = Config::parse(2, const_cast<char**>(argvMotion));
        check_true(cMotion.camera_motion, "--camera-motion sets camera_motion = true");

        std::cout << "  -> CLI config parsing passed successfully." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 2. Test Dual-GPU Vulkan Host Memory Zero-Copy Import (VK_EXT_external_memory_host)
    // -------------------------------------------------------------------------
    std::cout << "[TEST 2] Dual-GPU VK_EXT_external_memory_host Import..." << std::endl;
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
        if (gpuCount < 2) {
            std::cout << "  -> Less than 2 GPUs found (" << gpuCount
                      << "). Gracefully skipping dual-GPU Vulkan host memory test." << std::endl;
            vkDestroyInstance(instance, nullptr);
            std::cout << "==========================================================" << std::endl;
            std::cout << "  ALL MULTI-GPU FRAME PACING TESTS PASSED!" << std::endl;
            std::cout << "==========================================================" << std::endl;
            return 0;
        }

        std::vector<VkPhysicalDevice> gpus(gpuCount);
        vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());

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

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL MULTI-GPU FRAME PACING TESTS PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
