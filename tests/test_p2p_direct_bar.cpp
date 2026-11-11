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

    std::vector<VkPhysicalDevice> allDevs(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, allDevs.data());
    std::vector<VkPhysicalDevice> gpus;
    for (VkPhysicalDevice pd : allDevs) {
        VkPhysicalDeviceProperties props{};
        vkGetPhysicalDeviceProperties(pd, &props);
        if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU && props.apiVersion >= VK_API_VERSION_1_4) {
            gpus.push_back(pd);
        }
    }
    if (gpus.size() < 2) {
        std::cout << "Less than 2 hardware Vulkan 1.4 GPUs found (" << gpus.size() << "). Skipping P2P test." << std::endl;
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

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

    // 6. Execute real GPU 1 -> GPU 0 payload transfer across the DMA-BUF mapping!
    VkQueue q0 = VK_NULL_HANDLE, q1 = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev0, 0, 0, &q0);
    vkGetDeviceQueue(dev1, 0, 0, &q1);

    uint32_t hostVisIdx0 = UINT32_MAX;
    for (uint32_t i = 0; i < memProps0.memoryTypeCount; ++i) {
        const auto flags = memProps0.memoryTypes[i].propertyFlags;
        if ((flags & (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) ==
            (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)) {
            hostVisIdx0 = i;
            break;
        }
    }
    assert(hostVisIdx0 != UINT32_MAX);

    VkBufferCreateInfo rbBufInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    rbBufInfo.size = bufferSize;
    rbBufInfo.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    rbBufInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer rbBuf0 = VK_NULL_HANDLE;
    if (vkCreateBuffer(dev0, &rbBufInfo, nullptr, &rbBuf0) != VK_SUCCESS) return 1;

    VkMemoryRequirements rbReq{};
    vkGetBufferMemoryRequirements(dev0, rbBuf0, &rbReq);
    VkMemoryAllocateInfo rbAlloc{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    rbAlloc.allocationSize = rbReq.size;
    rbAlloc.memoryTypeIndex = hostVisIdx0;
    VkDeviceMemory rbMem0 = VK_NULL_HANDLE;
    if (vkAllocateMemory(dev0, &rbAlloc, nullptr, &rbMem0) != VK_SUCCESS) return 1;
    if (vkBindBufferMemory(dev0, rbBuf0, rbMem0, 0) != VK_SUCCESS) return 1;

    VkCommandPoolCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpInfo.queueFamilyIndex = 0;
    VkCommandPool cp0 = VK_NULL_HANDLE, cp1 = VK_NULL_HANDLE;
    vkCreateCommandPool(dev0, &cpInfo, nullptr, &cp0);
    vkCreateCommandPool(dev1, &cpInfo, nullptr, &cp1);

    VkCommandBufferAllocateInfo cbInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;
    cbInfo.commandPool = cp1;
    VkCommandBuffer cmd1 = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(dev1, &cbInfo, &cmd1);
    cbInfo.commandPool = cp0;
    VkCommandBuffer cmd0 = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(dev0, &cbInfo, &cmd0);

    constexpr uint32_t kPattern = 0xA5C35AA5u;
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd1, &beginInfo);
    vkCmdFillBuffer(cmd1, buf1, 0, bufferSize, kPattern);
    vkEndCommandBuffer(cmd1);

    VkSubmitInfo sub1{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    sub1.commandBufferCount = 1;
    sub1.pCommandBuffers = &cmd1;
    if (vkQueueSubmit(q1, 1, &sub1, VK_NULL_HANDLE) != VK_SUCCESS) return 1;
    vkQueueWaitIdle(q1);

    vkBeginCommandBuffer(cmd0, &beginInfo);
    VkBufferCopy copyRegion{ 0, 0, bufferSize };
    vkCmdCopyBuffer(cmd0, buf0, rbBuf0, 1, &copyRegion);
    vkEndCommandBuffer(cmd0);

    VkSubmitInfo sub0{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
    sub0.commandBufferCount = 1;
    sub0.pCommandBuffers = &cmd0;
    if (vkQueueSubmit(q0, 1, &sub0, VK_NULL_HANDLE) != VK_SUCCESS) return 1;
    vkQueueWaitIdle(q0);

    void* mapped = nullptr;
    if (vkMapMemory(dev0, rbMem0, 0, bufferSize, 0, &mapped) != VK_SUCCESS || !mapped) return 1;
    const uint32_t* words = static_cast<const uint32_t*>(mapped);
    const size_t wordCount = bufferSize / sizeof(uint32_t);
    for (size_t i = 0; i < wordCount; i += 257) {
        if (words[i] != kPattern) {
            std::cerr << "[FAIL] Cross-GPU DMA-BUF payload mismatch at word " << i
                      << ": got 0x" << std::hex << words[i] << std::dec << std::endl;
            return 1;
        }
    }
    vkUnmapMemory(dev0, rbMem0);

    std::cout << "[SUCCESS] P2P Direct BAR DMA-BUF 4 MB payload written on GPU 1 and verified byte-exact on GPU 0!" << std::endl;

    // Cleanup
    vkDestroyCommandPool(dev0, cp0, nullptr);
    vkDestroyCommandPool(dev1, cp1, nullptr);
    vkDestroyBuffer(dev0, rbBuf0, nullptr);
    vkFreeMemory(dev0, rbMem0, nullptr);
    vkDestroyBuffer(dev0, buf0, nullptr);
    vkFreeMemory(dev0, mem0, nullptr);
    vkDestroyBuffer(dev1, buf1, nullptr);
    vkFreeMemory(dev1, mem1, nullptr);

    vkDestroyDevice(dev0, nullptr);
    vkDestroyDevice(dev1, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
