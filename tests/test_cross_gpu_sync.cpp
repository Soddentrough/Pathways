#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <cstring>
#include <unistd.h>

int main() {
    std::cout << "Testing VK_KHR_external_semaphore_fd between GPU 0 and GPU 1..." << std::endl;

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
        std::cerr << "Less than 2 GPUs found: " << gpuCount << std::endl;
        return 1;
    }

    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());

    VkPhysicalDeviceProperties p0{}, p1{};
    vkGetPhysicalDeviceProperties(gpus[0], &p0);
    vkGetPhysicalDeviceProperties(gpus[1], &p1);
    std::cout << "GPU 0: " << p0.deviceName << std::endl;
    std::cout << "GPU 1: " << p1.deviceName << std::endl;

    // Create Device 0 and Device 1 with external_semaphore_fd
    const char* exts[] = { "VK_KHR_external_semaphore_fd" };
    float priority = 1.0f;

    VkDeviceQueueCreateInfo qInfo0{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qInfo0.queueFamilyIndex = 0;
    qInfo0.queueCount = 1;
    qInfo0.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan13Features features13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features13.synchronization2 = VK_TRUE;

    VkDeviceCreateInfo dInfo0{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    dInfo0.pNext = &features13;
    dInfo0.queueCreateInfoCount = 1;
    dInfo0.pQueueCreateInfos = &qInfo0;
    dInfo0.enabledExtensionCount = 1;
    dInfo0.ppEnabledExtensionNames = exts;

    VkDevice dev0 = VK_NULL_HANDLE;
    if (vkCreateDevice(gpus[0], &dInfo0, nullptr, &dev0) != VK_SUCCESS) {
        std::cerr << "Failed to create Device 0 with external_semaphore_fd." << std::endl;
        return 1;
    }

    VkDeviceQueueCreateInfo qInfo1 = qInfo0;
    VkDeviceCreateInfo dInfo1 = dInfo0;
    VkDevice dev1 = VK_NULL_HANDLE;
    if (vkCreateDevice(gpus[1], &dInfo1, nullptr, &dev1) != VK_SUCCESS) {
        std::cerr << "Failed to create Device 1 with external_semaphore_fd." << std::endl;
        return 1;
    }

    VkQueue q0 = VK_NULL_HANDLE;
    VkQueue q1 = VK_NULL_HANDLE;
    vkGetDeviceQueue(dev0, 0, 0, &q0);
    vkGetDeviceQueue(dev1, 0, 0, &q1);

    auto pfnGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(dev1, "vkGetSemaphoreFdKHR");
    auto pfnImportSemaphoreFdKHR = (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(dev0, "vkImportSemaphoreFdKHR");

    if (!pfnGetSemaphoreFdKHR || !pfnImportSemaphoreFdKHR) {
        std::cerr << "Failed to load vkGetSemaphoreFdKHR or vkImportSemaphoreFdKHR." << std::endl;
        return 1;
    }

    // Create exportable semaphore on Dev 1
    VkExportSemaphoreCreateInfo exportInfo{ VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
    exportInfo.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo semInfo1{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    semInfo1.pNext = &exportInfo;
    VkSemaphore sem1 = VK_NULL_HANDLE;
    if (vkCreateSemaphore(dev1, &semInfo1, nullptr, &sem1) != VK_SUCCESS) {
        std::cerr << "Failed to create exportable semaphore on Dev 1." << std::endl;
        return 1;
    }

    // Create importable semaphore on Dev 0
    VkSemaphoreCreateInfo semInfo0{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
    VkSemaphore sem0 = VK_NULL_HANDLE;
    if (vkCreateSemaphore(dev0, &semInfo0, nullptr, &sem0) != VK_SUCCESS) {
        std::cerr << "Failed to create semaphore on Dev 0." << std::endl;
        return 1;
    }

    // Command pools
    VkCommandPoolCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cpInfo.queueFamilyIndex = 0;
    VkCommandPool cp0 = VK_NULL_HANDLE, cp1 = VK_NULL_HANDLE;
    vkCreateCommandPool(dev0, &cpInfo, nullptr, &cp0);
    vkCreateCommandPool(dev1, &cpInfo, nullptr, &cp1);

    VkCommandBufferAllocateInfo cbInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cbInfo.commandBufferCount = 1;

    cbInfo.commandPool = cp0;
    VkCommandBuffer cmd0 = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(dev0, &cbInfo, &cmd0);

    cbInfo.commandPool = cp1;
    VkCommandBuffer cmd1 = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(dev1, &cbInfo, &cmd1);

    // Record empty cmd buffers
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    vkBeginCommandBuffer(cmd0, &beginInfo);
    vkEndCommandBuffer(cmd0);

    vkBeginCommandBuffer(cmd1, &beginInfo);
    vkEndCommandBuffer(cmd1);

    std::cout << "Testing 100 consecutive cross-device semaphore signal -> export -> import -> wait cycles..." << std::endl;
    for (int iter = 0; iter < 100; ++iter) {
        // 1. Submit on Dev 1 signaling sem1
        VkCommandBufferSubmitInfo cmdInfo1{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdInfo1.commandBuffer = cmd1;

        VkSemaphoreSubmitInfo sigInfo1{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        sigInfo1.semaphore = sem1;
        sigInfo1.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        VkSubmitInfo2 submit1{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submit1.commandBufferInfoCount = 1;
        submit1.pCommandBufferInfos = &cmdInfo1;
        submit1.signalSemaphoreInfoCount = 1;
        submit1.pSignalSemaphoreInfos = &sigInfo1;

        if (vkQueueSubmit2(q1, 1, &submit1, VK_NULL_HANDLE) != VK_SUCCESS) {
            std::cerr << "vkQueueSubmit2 on Dev 1 failed at iter " << iter << std::endl;
            return 1;
        }

        // 2. Export FD from Dev 1
        VkSemaphoreGetFdInfoKHR getFdInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR };
        getFdInfo.semaphore = sem1;
        getFdInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        int fd = -1;
        if (pfnGetSemaphoreFdKHR(dev1, &getFdInfo, &fd) != VK_SUCCESS || fd < 0) {
            std::cerr << "vkGetSemaphoreFdKHR failed at iter " << iter << std::endl;
            return 1;
        }

        // 3. Import FD into sem0 on Dev 0
        VkImportSemaphoreFdInfoKHR importInfo{ VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR };
        importInfo.semaphore = sem0;
        importInfo.flags = VK_SEMAPHORE_IMPORT_TEMPORARY_BIT;
        importInfo.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        importInfo.fd = fd;
        if (pfnImportSemaphoreFdKHR(dev0, &importInfo) != VK_SUCCESS) {
            std::cerr << "vkImportSemaphoreFdKHR failed at iter " << iter << std::endl;
            close(fd);
            return 1;
        }

        // 4. Submit on Dev 0 waiting on sem0
        VkCommandBufferSubmitInfo cmdInfo0{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdInfo0.commandBuffer = cmd0;

        VkSemaphoreSubmitInfo waitInfo0{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        waitInfo0.semaphore = sem0;
        waitInfo0.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;

        VkSubmitInfo2 submit0{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submit0.commandBufferInfoCount = 1;
        submit0.pCommandBufferInfos = &cmdInfo0;
        submit0.waitSemaphoreInfoCount = 1;
        submit0.pWaitSemaphoreInfos = &waitInfo0;

        VkFence fence0 = VK_NULL_HANDLE;
        VkFenceCreateInfo fInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
        vkCreateFence(dev0, &fInfo, nullptr, &fence0);

        if (vkQueueSubmit2(q0, 1, &submit0, fence0) != VK_SUCCESS) {
            std::cerr << "vkQueueSubmit2 on Dev 0 failed at iter " << iter << std::endl;
            return 1;
        }

        vkWaitForFences(dev0, 1, &fence0, VK_TRUE, UINT64_MAX);
        vkDestroyFence(dev0, fence0, nullptr);
    }

    std::cout << "[SUCCESS] 100 cross-GPU synchronization cycles passed with zero errors!" << std::endl;

    vkDestroySemaphore(dev0, sem0, nullptr);
    vkDestroySemaphore(dev1, sem1, nullptr);
    vkDestroyCommandPool(dev0, cp0, nullptr);
    vkDestroyCommandPool(dev1, cp1, nullptr);
    vkDestroyDevice(dev0, nullptr);
    vkDestroyDevice(dev1, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
