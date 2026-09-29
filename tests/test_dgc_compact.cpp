#include <vulkan/vulkan.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <cstring>
#include <cassert>
#include <random>

struct RayPayload {
    float origin[4];
    float direction[4];
    float throughput[4];
    float radiance[4];
};
static_assert(sizeof(RayPayload) == 64, "RayPayload must be 64 bytes");

struct IndirectDispatch {
    uint32_t x;
    uint32_t y;
    uint32_t z;
    uint32_t activeCount;
    uint32_t retiredWorkgroups;
};

static uint32_t findMemoryType(VkPhysicalDevice physDevice, uint32_t typeFilter, VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(physDevice, &memProperties);
    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((typeFilter & (1 << i)) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }
    throw std::runtime_error("Failed to find suitable memory type");
}

static void createBuffer(VkDevice device, VkPhysicalDevice physDevice, VkDeviceSize size, VkBufferUsageFlags usage,
                         VkMemoryPropertyFlags properties, VkBuffer& buffer, VkDeviceMemory& bufferMemory) {
    VkBufferCreateInfo bufferInfo{ VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO };
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (vkCreateBuffer(device, &bufferInfo, nullptr, &buffer) != VK_SUCCESS) {
        throw std::runtime_error("Failed to create buffer");
    }

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(device, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO };
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = findMemoryType(physDevice, memRequirements.memoryTypeBits, properties);
    if (vkAllocateMemory(device, &allocInfo, nullptr, &bufferMemory) != VK_SUCCESS) {
        throw std::runtime_error("Failed to allocate buffer memory");
    }
    vkBindBufferMemory(device, buffer, bufferMemory, 0);
}

static std::vector<char> loadShaderSPIRV(const std::string& filename) {
    std::filesystem::path exeDir;
#if defined(__linux__)
    std::error_code ec;
    auto p = std::filesystem::canonical("/proc/self/exe", ec);
    if (!ec) exeDir = p.parent_path();
#endif

    std::vector<std::string> searchPaths = {
        filename,
        "shaders/" + filename,
        "build/shaders/" + filename,
        (exeDir / "shaders" / filename).string(),
        (exeDir / filename).string(),
        (exeDir / ".." / "shaders" / filename).string()
    };

    for (const auto& path : searchPaths) {
        if (std::filesystem::exists(path)) {
            std::ifstream file(path, std::ios::ate | std::ios::binary);
            if (file.is_open()) {
                size_t fileSize = static_cast<size_t>(file.tellg());
                std::vector<char> buffer(fileSize);
                file.seekg(0);
                file.read(buffer.data(), fileSize);
                return buffer;
            }
        }
    }
    throw std::runtime_error("Could not find shader SPIR-V file: " + filename);
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing dgc_compact GPU Stream Compaction     " << std::endl;
    std::cout << "  Verifying Cross-Workgroup Atomic Retirement on Device   " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Initialize Vulkan 1.4+ Instance
    VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
    appInfo.apiVersion = VK_API_VERSION_1_4;

    VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
    instInfo.pApplicationInfo = &appInfo;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&instInfo, nullptr, &instance) != VK_SUCCESS) {
        std::cerr << "[FAIL] Failed to create Vulkan instance." << std::endl;
        return 1;
    }

    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    if (deviceCount == 0) {
        std::cerr << "[FAIL] No physical devices found." << std::endl;
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    VkPhysicalDevice physicalDevice = devices[0];

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    std::cout << "  -> Target GPU: " << props.deviceName << " (Driver: " << props.driverVersion << ")" << std::endl;

    // Find compute queue family
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(physicalDevice, &queueFamilyCount, queueFamilies.data());

    uint32_t computeFamily = UINT32_MAX;
    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            computeFamily = i;
            break;
        }
    }
    if (computeFamily == UINT32_MAX) {
        std::cerr << "[FAIL] No compute queue family found." << std::endl;
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    float queuePriority = 1.0f;
    VkDeviceQueueCreateInfo queueCreateInfo{ VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO };
    queueCreateInfo.queueFamilyIndex = computeFamily;
    queueCreateInfo.queueCount = 1;
    queueCreateInfo.pQueuePriorities = &queuePriority;

    VkPhysicalDeviceVulkan14Features feat14{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES };
    feat14.shaderSubgroupRotate = VK_TRUE;

    VkPhysicalDeviceVulkan13Features feat13{ VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES };
    feat13.pNext = &feat14;
    feat13.synchronization2 = VK_TRUE;

    VkDeviceCreateInfo deviceCreateInfo{ VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO };
    deviceCreateInfo.pNext = &feat13;
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &queueCreateInfo;

    VkDevice device = VK_NULL_HANDLE;
    if (vkCreateDevice(physicalDevice, &deviceCreateInfo, nullptr, &device) != VK_SUCCESS) {
        std::cerr << "[FAIL] Failed to create logical device." << std::endl;
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    VkQueue computeQueue = VK_NULL_HANDLE;
    vkGetDeviceQueue(device, computeFamily, 0, &computeQueue);

    // 2. Load SPIR-V and Create Compute Pipeline
    auto shaderCode = loadShaderSPIRV("dgc_compact.comp.spv");
    VkShaderModuleCreateInfo smInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
    smInfo.codeSize = shaderCode.size();
    smInfo.pCode = reinterpret_cast<const uint32_t*>(shaderCode.data());

    VkShaderModule shaderModule = VK_NULL_HANDLE;
    if (vkCreateShaderModule(device, &smInfo, nullptr, &shaderModule) != VK_SUCCESS) {
        std::cerr << "[FAIL] Failed to create shader module." << std::endl;
        vkDestroyDevice(device, nullptr);
        vkDestroyInstance(instance, nullptr);
        return 1;
    }

    // Descriptor set layout: 3 storage buffers
    VkDescriptorSetLayoutBinding bindings[3]{};
    for (uint32_t b = 0; b < 3; ++b) {
        bindings[b].binding = b;
        bindings[b].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[b].descriptorCount = 1;
        bindings[b].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }

    VkDescriptorSetLayoutCreateInfo layoutInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
    layoutInfo.bindingCount = 3;
    layoutInfo.pBindings = bindings;

    VkDescriptorSetLayout descLayout = VK_NULL_HANDLE;
    vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descLayout);

    // Push constant range: uint totalRays (4 bytes)
    VkPushConstantRange pcRange{};
    pcRange.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    pcRange.offset = 0;
    pcRange.size = sizeof(uint32_t);

    VkPipelineLayoutCreateInfo pipeLayoutInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
    pipeLayoutInfo.setLayoutCount = 1;
    pipeLayoutInfo.pSetLayouts = &descLayout;
    pipeLayoutInfo.pushConstantRangeCount = 1;
    pipeLayoutInfo.pPushConstantRanges = &pcRange;

    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    vkCreatePipelineLayout(device, &pipeLayoutInfo, nullptr, &pipelineLayout);

    VkComputePipelineCreateInfo compPipeInfo{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
    compPipeInfo.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    compPipeInfo.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    compPipeInfo.stage.module = shaderModule;
    compPipeInfo.stage.pName = "main";
    compPipeInfo.layout = pipelineLayout;

    VkPipeline pipeline = VK_NULL_HANDLE;
    if (vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &compPipeInfo, nullptr, &pipeline) != VK_SUCCESS) {
        std::cerr << "[FAIL] Failed to create compute pipeline." << std::endl;
        return 1;
    }

    // 3. Allocate Buffers
    const uint32_t numRays = 65536; // 1,024 workgroups @ local_size_x = 64
    VkDeviceSize rayBufferSize = numRays * sizeof(RayPayload);
    VkDeviceSize dispatchBufferSize = sizeof(IndirectDispatch);

    VkBuffer inRaysBuf = VK_NULL_HANDLE, outRaysBuf = VK_NULL_HANDLE, dispatchBuf = VK_NULL_HANDLE;
    VkDeviceMemory inRaysMem = VK_NULL_HANDLE, outRaysMem = VK_NULL_HANDLE, dispatchMem = VK_NULL_HANDLE;

    VkMemoryPropertyFlags hostMemFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    createBuffer(device, physicalDevice, rayBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemFlags, inRaysBuf, inRaysMem);
    createBuffer(device, physicalDevice, rayBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemFlags, outRaysBuf, outRaysMem);
    createBuffer(device, physicalDevice, dispatchBufferSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hostMemFlags, dispatchBuf, dispatchMem);

    // Descriptor pool & set
    VkDescriptorPoolSize poolSize{ VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 3 };
    VkDescriptorPoolCreateInfo poolInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
    poolInfo.maxSets = 1;
    poolInfo.poolSizeCount = 1;
    poolInfo.pPoolSizes = &poolSize;

    VkDescriptorPool descPool = VK_NULL_HANDLE;
    vkCreateDescriptorPool(device, &poolInfo, nullptr, &descPool);

    VkDescriptorSetAllocateInfo allocSetInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
    allocSetInfo.descriptorPool = descPool;
    allocSetInfo.descriptorSetCount = 1;
    allocSetInfo.pSetLayouts = &descLayout;

    VkDescriptorSet descSet = VK_NULL_HANDLE;
    vkAllocateDescriptorSets(device, &allocSetInfo, &descSet);

    VkDescriptorBufferInfo bInfos[3] = {
        { inRaysBuf, 0, rayBufferSize },
        { outRaysBuf, 0, rayBufferSize },
        { dispatchBuf, 0, dispatchBufferSize }
    };

    VkWriteDescriptorSet writes[3]{};
    for (uint32_t i = 0; i < 3; ++i) {
        writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = descSet;
        writes[i].dstBinding = i;
        writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        writes[i].pBufferInfo = &bInfos[i];
    }
    vkUpdateDescriptorSets(device, 3, writes, 0, nullptr);

    // Command Pool & Buffer
    VkCommandPoolCreateInfo cmdPoolInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
    cmdPoolInfo.queueFamilyIndex = computeFamily;
    cmdPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;

    VkCommandPool cmdPool = VK_NULL_HANDLE;
    vkCreateCommandPool(device, &cmdPoolInfo, nullptr, &cmdPool);

    VkCommandBufferAllocateInfo cmdAllocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    cmdAllocInfo.commandPool = cmdPool;
    cmdAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cmdAllocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &cmdAllocInfo, &cmd);

    void* pInRays = nullptr;
    void* pOutRays = nullptr;
    void* pDispatch = nullptr;
    vkMapMemory(device, inRaysMem, 0, rayBufferSize, 0, &pInRays);
    vkMapMemory(device, outRaysMem, 0, rayBufferSize, 0, &pOutRays);
    vkMapMemory(device, dispatchMem, 0, dispatchBufferSize, 0, &pDispatch);

    // 4. Stress Test Loop: Run 50 iterations with varying active patterns
    std::cout << "[TEST] Running 50 iterations of dgc_compact across 1,024 workgroups (65,536 rays)..." << std::endl;
    std::mt19937 rng(42);

    for (uint32_t iter = 0; iter < 50; ++iter) {
        // Vary active ratio between 10% and 80%
        float activeThreshold = 0.10f + (iter % 15) * 0.05f;
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        RayPayload* inRayPtr = reinterpret_cast<RayPayload*>(pInRays);
        uint32_t expectedActive = 0;

        for (uint32_t i = 0; i < numRays; ++i) {
            bool active = (dist(rng) < activeThreshold);
            inRayPtr[i].origin[0] = static_cast<float>(i);
            inRayPtr[i].origin[1] = 0.0f;
            inRayPtr[i].origin[2] = 0.0f;
            inRayPtr[i].origin[3] = active ? 1.0f : 0.0f; // Active flag in origin.w
            inRayPtr[i].direction[0] = 0.0f;
            inRayPtr[i].direction[1] = 1.0f;
            inRayPtr[i].direction[2] = 0.0f;
            inRayPtr[i].direction[3] = 0.0f;
            if (active) expectedActive++;
        }

        // Initialize dispatch buffer with zeros
        IndirectDispatch* dispatchPtr = reinterpret_cast<IndirectDispatch*>(pDispatch);
        dispatchPtr->x = 0;
        dispatchPtr->y = 0;
        dispatchPtr->z = 0;
        dispatchPtr->activeCount = 0;
        dispatchPtr->retiredWorkgroups = 0;

        // Zero out output buffer
        std::memset(pOutRays, 0, rayBufferSize);

        // Record & Dispatch
        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        vkBeginCommandBuffer(cmd, &beginInfo);

        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &descSet, 0, nullptr);
        vkCmdPushConstants(cmd, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint32_t), &numRays);

        uint32_t groupCountX = (numRays + 63) / 64; // 1,024 workgroups
        vkCmdDispatch(cmd, groupCountX, 1, 1);

        vkEndCommandBuffer(cmd);

        VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
        submitInfo.commandBufferCount = 1;
        submitInfo.pCommandBuffers = &cmd;

        if (vkQueueSubmit(computeQueue, 1, &submitInfo, VK_NULL_HANDLE) != VK_SUCCESS) {
            std::cerr << "[FAIL] vkQueueSubmit failed on iteration " << iter << std::endl;
            return 1;
        }
        vkQueueWaitIdle(computeQueue);

        // Verify Results
        uint32_t expectedGroups = (expectedActive + 63u) / 64u;

        if (dispatchPtr->activeCount != expectedActive) {
            std::cerr << "[FAIL] Iteration " << iter << ": activeCount mismatch! Expected "
                      << expectedActive << ", got " << dispatchPtr->activeCount << std::endl;
            return 1;
        }

        if (dispatchPtr->x != expectedGroups || dispatchPtr->y != 1 || dispatchPtr->z != 1) {
            std::cerr << "[FAIL] Iteration " << iter << ": dispatch dimensions mismatch! Expected ("
                      << expectedGroups << ", 1, 1), got ("
                      << dispatchPtr->x << ", " << dispatchPtr->y << ", " << dispatchPtr->z << ")" << std::endl;
            return 1;
        }

        if (dispatchPtr->retiredWorkgroups != 0) {
            std::cerr << "[FAIL] Iteration " << iter << ": retiredWorkgroups was not cleanly reset to 0! Got "
                      << dispatchPtr->retiredWorkgroups << std::endl;
            return 1;
        }

        // Verify compaction integrity: all expectedActive rays in outRays must be active
        const RayPayload* outRayPtr = reinterpret_cast<const RayPayload*>(pOutRays);
        for (uint32_t k = 0; k < expectedActive; ++k) {
            if (outRayPtr[k].origin[3] <= 0.5f) {
                std::cerr << "[FAIL] Iteration " << iter << ": compacted ray at index " << k
                          << " is inactive (hole in compacted stream)!" << std::endl;
                return 1;
            }
        }

        // Verify remaining elements in outRays are empty
        for (uint32_t k = expectedActive; k < expectedActive + 100 && k < numRays; ++k) {
            if (outRayPtr[k].origin[3] > 0.5f) {
                std::cerr << "[FAIL] Iteration " << iter << ": out-of-bounds ray written past activeCount at index "
                          << k << std::endl;
                return 1;
            }
        }
    }

    std::cout << "  -> 50/50 stress iterations PASSED! Atomic workgroup retirement is 100% race-free across all 1,024 workgroups." << std::endl;

    // Cleanup
    vkUnmapMemory(device, inRaysMem);
    vkUnmapMemory(device, outRaysMem);
    vkUnmapMemory(device, dispatchMem);

    vkDestroyBuffer(device, inRaysBuf, nullptr);
    vkDestroyBuffer(device, outRaysBuf, nullptr);
    vkDestroyBuffer(device, dispatchBuf, nullptr);
    vkFreeMemory(device, inRaysMem, nullptr);
    vkFreeMemory(device, outRaysMem, nullptr);
    vkFreeMemory(device, dispatchMem, nullptr);

    vkDestroyCommandPool(device, cmdPool, nullptr);
    vkDestroyDescriptorPool(device, descPool, nullptr);
    vkDestroyDescriptorSetLayout(device, descLayout, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyShaderModule(device, shaderModule, nullptr);

    vkDestroyDevice(device, nullptr);
    vkDestroyInstance(instance, nullptr);

    std::cout << "\n[SUCCESS] All dgc_compact GPU stream compaction tests passed!" << std::endl;
    return 0;
}
