#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <cassert>
#if defined(_WIN32)
#include <io.h>
#define close _close
#else
#include <unistd.h>
#endif

int main() {
    std::cout << "=== Running test_p2p_direct_bar ===" << std::endl;

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
        std::cout << "Less than 2 GPUs found (" << gpuCount << "). Skipping P2P test (requires dual GPUs)." << std::endl;
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());

    std::vector<const char*> devExts = {
        "VK_KHR_external_memory_fd",
        "VK_EXT_external_memory_dma_buf",
        "VK_KHR_external_semaphore_fd"
    };

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
        return 1;
    }

    auto pfnGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(dev1, "vkGetMemoryFdKHR");
    auto pfnGetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(dev0, "vkGetMemoryFdPropertiesKHR");

    if (!pfnGetMemoryFdKHR || !pfnGetMemoryFdPropertiesKHR) {
        std::cerr << "Failed to get memory FD function pointers." << std::endl;
        return 1;
    }

    VkPhysicalDeviceMemoryProperties memProps0, memProps1;
    vkGetPhysicalDeviceMemoryProperties(gpus[0], &memProps0);
    vkGetPhysicalDeviceMemoryProperties(gpus[1], &memProps1);

    uint32_t devLocalIdx1 = UINT32_MAX;
    for (uint32_t i = 0; i < memProps1.memoryTypeCount; ++i) {
        if (memProps1.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            devLocalIdx1 = i;
            break;
        }
    }
    assert(devLocalIdx1 != UINT32_MAX);

    const VkDeviceSize bufferSize = 4 * 1024 * 1024; // 4 MB test buffer
    auto handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_DMA_BUF_BIT_EXT;

    // 1. Create buffer on Dev 1 (GPU 1 VRAM)
    VkExternalMemoryBufferCreateInfo extBufInfo1{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
    extBufInfo1.handleTypes = handleType;

    VkBufferCreateInfo bufInfo1{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufInfo1.pNext = &extBufInfo1;
    bufInfo1.size = bufferSize;
    bufInfo1.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo1.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buf1 = VK_NULL_HANDLE;
    VkResult res = vkCreateBuffer(dev1, &bufInfo1, nullptr, &buf1);
    if (res != VK_SUCCESS) {
        std::cerr << "vkCreateBuffer on Dev 1 failed: " << res << std::endl;
        return 1;
    }

    VkBufferMemoryRequirementsInfo2 reqInfo{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2 };
    reqInfo.buffer = buf1;
    VkMemoryRequirements2 memReq2{ VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2 };
    vkGetBufferMemoryRequirements2(dev1, &reqInfo, &memReq2);
    VkMemoryRequirements memReq1 = memReq2.memoryRequirements;

    VkExportMemoryAllocateInfo exportAllocInfo{ VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO };
    exportAllocInfo.handleTypes = handleType;

    VkMemoryAllocateInfo allocInfo1{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo1.pNext = &exportAllocInfo;
    allocInfo1.allocationSize = memReq1.size;
    allocInfo1.memoryTypeIndex = devLocalIdx1;

    VkDeviceMemory mem1 = VK_NULL_HANDLE;
    res = vkAllocateMemory(dev1, &allocInfo1, nullptr, &mem1);
    if (res != VK_SUCCESS) {
        std::cerr << "vkAllocateMemory on Dev 1 failed: " << res << std::endl;
        return 1;
    }
    res = vkBindBufferMemory(dev1, buf1, mem1, 0);
    if (res != VK_SUCCESS) {
        std::cerr << "vkBindBufferMemory on Dev 1 failed: " << res << std::endl;
        return 1;
    }

    // 2. Export FD from Dev 1
    VkMemoryGetFdInfoKHR getFdInfo{ VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR };
    getFdInfo.memory = mem1;
    getFdInfo.handleType = handleType;

    int memFd = -1;
    res = pfnGetMemoryFdKHR(dev1, &getFdInfo, &memFd);
    if (res != VK_SUCCESS || memFd < 0) {
        std::cerr << "pfnGetMemoryFdKHR failed: " << res << std::endl;
        return 1;
    }

    // 3. Query properties on Dev 0
    VkMemoryFdPropertiesKHR fdProps{ VK_STRUCTURE_TYPE_MEMORY_FD_PROPERTIES_KHR };
    res = pfnGetMemoryFdPropertiesKHR(dev0, handleType, memFd, &fdProps);
    if (res != VK_SUCCESS) {
        std::cerr << "pfnGetMemoryFdPropertiesKHR failed: " << res << std::endl;
        close(memFd);
        return 1;
    }

    uint32_t memIdx0 = UINT32_MAX;
    for (uint32_t i = 0; i < memProps0.memoryTypeCount; ++i) {
        if (fdProps.memoryTypeBits & (1 << i)) {
            memIdx0 = i;
            break;
        }
    }
    if (memIdx0 == UINT32_MAX) {
        std::cerr << "No compatible memory type found on Dev 0." << std::endl;
        close(memFd);
        return 1;
    }

    // 4. Import FD into Dev 0
    VkImportMemoryFdInfoKHR importInfo{ VK_STRUCTURE_TYPE_IMPORT_MEMORY_FD_INFO_KHR };
    importInfo.handleType = handleType;
    importInfo.fd = memFd;

    VkMemoryAllocateInfo allocInfo0{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo0.pNext = &importInfo;
    allocInfo0.allocationSize = memReq1.size;
    allocInfo0.memoryTypeIndex = memIdx0;

    VkDeviceMemory mem0 = VK_NULL_HANDLE;
    res = vkAllocateMemory(dev0, &allocInfo0, nullptr, &mem0);
    if (res != VK_SUCCESS) {
        std::cerr << "vkAllocateMemory on Dev 0 failed: " << res << std::endl;
        close(memFd);
        return 1;
    }

    // 5. Create buffer on Dev 0
    VkExternalMemoryBufferCreateInfo extBufInfo0{ VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO };
    extBufInfo0.handleTypes = handleType;

    VkBufferCreateInfo bufInfo0{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufInfo0.pNext = &extBufInfo0;
    bufInfo0.size = bufferSize;
    bufInfo0.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufInfo0.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buf0 = VK_NULL_HANDLE;
    res = vkCreateBuffer(dev0, &bufInfo0, nullptr, &buf0);
    if (res != VK_SUCCESS) {
        std::cerr << "vkCreateBuffer on Dev 0 failed: " << res << std::endl;
        return 1;
    }

    res = vkBindBufferMemory(dev0, buf0, mem0, 0);
    if (res != VK_SUCCESS) {
        std::cerr << "vkBindBufferMemory on Dev 0 failed: " << res << std::endl;
        return 1;
    }

    std::cout << "[SUCCESS] P2P Direct BAR DMA-BUF memory bound across Dual GPUs successfully!" << std::endl;

    // Cleanup
    vkDestroyBuffer(dev0, buf0, nullptr);
    vkFreeMemory(dev0, mem0, nullptr);
    vkDestroyBuffer(dev1, buf1, nullptr);
    vkFreeMemory(dev1, mem1, nullptr);

    vkDestroyDevice(dev0, nullptr);
    vkDestroyDevice(dev1, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
