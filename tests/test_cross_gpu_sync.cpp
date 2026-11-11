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
        std::cout << "Less than 2 GPUs found (" << gpuCount << "). Skipping cross-GPU sync test (requires dual GPUs)." << std::endl;
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    std::vector<VkPhysicalDevice> gpus(gpuCount);
    vkEnumeratePhysicalDevices(instance, &gpuCount, gpus.data());

    VkPhysicalDeviceProperties p0{}, p1{};
    vkGetPhysicalDeviceProperties(gpus[0], &p0);
    vkGetPhysicalDeviceProperties(gpus[1], &p1);
    std::cout << "GPU 0: " << p0.deviceName << std::endl;
    std::cout << "GPU 1: " << p1.deviceName << std::endl;
    if (p0.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU || p1.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU) {
        std::cout << "One or more devices is a CPU/software renderer (" << (p1.deviceType == VK_PHYSICAL_DEVICE_TYPE_CPU ? p1.deviceName : p0.deviceName)
                  << "). Skipping cross-GPU sync test (requires dual hardware GPUs)." << std::endl;
        vkDestroyInstance(instance, nullptr);
        return 0;
    }

    // Create Device 0 and Device 1 with external_semaphore_fd
    const char* exts[] = { "VK_KHR_external_semaphore_fd" };
    float priority = 1.0f;

    VkDeviceQueueCreateInfo qInfo0{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    qInfo0.queueFamilyIndex = 0;
    qInfo0.queueCount = 1;
    qInfo0.pQueuePriorities = &priority;

    VkPhysicalDeviceVulkan12Features features12{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES };
    features12.timelineSemaphore = VK_TRUE;

    VkPhysicalDeviceVulkan13Features features13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    features13.pNext = &features12;
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
    dInfo1.pQueueCreateInfos = &qInfo1;
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

    // Phase 2: Permanent Cross-GPU Timeline Semaphore (VK_SEMAPHORE_TYPE_TIMELINE)
    // Zero per-frame FD export/import overhead, matching MultiGpuManager architecture
    std::cout << "Testing permanent cross-GPU VK_SEMAPHORE_TYPE_TIMELINE across 16 steps..." << std::endl;
    {
        VkSemaphoreTypeCreateInfo timelineType1{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        timelineType1.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineType1.initialValue = 0;

        VkExportSemaphoreCreateInfo tlExport{ VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO };
        tlExport.pNext = &timelineType1;
        tlExport.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

        VkSemaphoreCreateInfo tlInfo1{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        tlInfo1.pNext = &tlExport;
        VkSemaphore tlSem1 = VK_NULL_HANDLE;
        if (vkCreateSemaphore(dev1, &tlInfo1, nullptr, &tlSem1) != VK_SUCCESS) {
            std::cerr << "Failed to create exportable timeline semaphore on Dev 1." << std::endl;
            return 1;
        }

        VkSemaphoreTypeCreateInfo timelineType0{ VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO };
        timelineType0.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        timelineType0.initialValue = 0;

        VkSemaphoreCreateInfo tlInfo0{ VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO };
        tlInfo0.pNext = &timelineType0;
        VkSemaphore tlSem0 = VK_NULL_HANDLE;
        if (vkCreateSemaphore(dev0, &tlInfo0, nullptr, &tlSem0) != VK_SUCCESS) {
            std::cerr << "Failed to create importable timeline semaphore on Dev 0." << std::endl;
            return 1;
        }

        VkSemaphoreGetFdInfoKHR getTlFd{ VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR };
        getTlFd.semaphore = tlSem1;
        getTlFd.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        int tlFd = -1;
        if (pfnGetSemaphoreFdKHR(dev1, &getTlFd, &tlFd) != VK_SUCCESS || tlFd < 0) {
            std::cerr << "Failed to export timeline semaphore FD from Dev 1." << std::endl;
            return 1;
        }

        // Permanent import (flags = 0) — imported once and reused for all timeline steps
        VkImportSemaphoreFdInfoKHR impTl{ VK_STRUCTURE_TYPE_IMPORT_SEMAPHORE_FD_INFO_KHR };
        impTl.semaphore = tlSem0;
        impTl.flags = 0;
        impTl.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
        impTl.fd = tlFd;
        if (pfnImportSemaphoreFdKHR(dev0, &impTl) != VK_SUCCESS) {
            std::cerr << "Failed to permanently import timeline semaphore FD into Dev 0." << std::endl;
            close(tlFd);
            return 1;
        }

        for (uint64_t step = 1; step <= 16; ++step) {
            VkCommandBufferSubmitInfo cmdInfo1{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
            cmdInfo1.commandBuffer = cmd1;
            VkSemaphoreSubmitInfo sigTl{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            sigTl.semaphore = tlSem1;
            sigTl.value = step;
            sigTl.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            VkSubmitInfo2 sub1{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
            sub1.commandBufferInfoCount = 1;
            sub1.pCommandBufferInfos = &cmdInfo1;
            sub1.signalSemaphoreInfoCount = 1;
            sub1.pSignalSemaphoreInfos = &sigTl;
            if (vkQueueSubmit2(q1, 1, &sub1, VK_NULL_HANDLE) != VK_SUCCESS) return 1;

            VkCommandBufferSubmitInfo cmdInfo0{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
            cmdInfo0.commandBuffer = cmd0;
            VkSemaphoreSubmitInfo waitTl{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            waitTl.semaphore = tlSem0;
            waitTl.value = step;
            waitTl.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            VkSubmitInfo2 sub0{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
            sub0.commandBufferInfoCount = 1;
            sub0.pCommandBufferInfos = &cmdInfo0;
            sub0.waitSemaphoreInfoCount = 1;
            sub0.pWaitSemaphoreInfos = &waitTl;

            VkFence fence0 = VK_NULL_HANDLE;
            VkFenceCreateInfo fInfo{ VK_STRUCTURE_TYPE_FENCE_CREATE_INFO };
            vkCreateFence(dev0, &fInfo, nullptr, &fence0);
            if (vkQueueSubmit2(q0, 1, &sub0, fence0) != VK_SUCCESS) return 1;
            vkWaitForFences(dev0, 1, &fence0, VK_TRUE, UINT64_MAX);
            vkDestroyFence(dev0, fence0, nullptr);

            uint64_t counterVal0 = 0;
            if (vkGetSemaphoreCounterValue(dev0, tlSem0, &counterVal0) != VK_SUCCESS || counterVal0 < step) {
                std::cerr << "Timeline semaphore counter mismatch on Dev 0 at step " << step << std::endl;
                return 1;
            }
        }

        vkDestroySemaphore(dev0, tlSem0, nullptr);
        vkDestroySemaphore(dev1, tlSem1, nullptr);
        std::cout << "[SUCCESS] Permanent cross-GPU timeline semaphore verified across 16 steps!" << std::endl;
    }

    vkDestroySemaphore(dev0, sem0, nullptr);
    vkDestroySemaphore(dev1, sem1, nullptr);
    vkDestroyCommandPool(dev0, cp0, nullptr);
    vkDestroyCommandPool(dev1, cp1, nullptr);
    vkDestroyDevice(dev0, nullptr);
    vkDestroyDevice(dev1, nullptr);
    vkDestroyInstance(instance, nullptr);
    return 0;
}
