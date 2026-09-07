#include "vulkan/VulkanContext.hpp"
#include "core/Logger.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <vector>
#include <cstring>
#include <stdexcept>
#include <algorithm>

namespace pathways {

uint32_t VulkanContext::s_validationErrors = 0;

static const std::vector<const char*> g_validationLayers = {
    "VK_LAYER_KHRONOS_validation"
};

VKAPI_ATTR VkBool32 VKAPI_CALL VulkanContext::debugCallback(
    VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    VkDebugUtilsMessageTypeFlagsEXT messageType,
    const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {

    if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) {
        Logger::error("[Vulkan Validation] {}", pCallbackData->pMessage);
        s_validationErrors++;
    } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) {
        Logger::warn("[Vulkan Validation] {}", pCallbackData->pMessage);
    } else if (messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT) {
        Logger::debug("[Vulkan Validation Info] {}", pCallbackData->pMessage);
    }
    return VK_FALSE;
}

VulkanContext::VulkanContext(const Config& config, VkSurfaceKHR surface) {
    Logger::info("Initializing Vulkan 1.4 Context...");
    createInstance(config);
    if (config.validation_layers) {
        setupDebugMessenger();
    }
    selectPhysicalDevice(config, surface);
    createLogicalDevice(config);
    initVMA();
    Logger::info("Vulkan 1.4 Context successfully initialized on: {}", m_deviceName);
}

VulkanContext::~VulkanContext() {
    Logger::info("Destroying Vulkan Context...");
    if (m_allocator) {
        vmaDestroyAllocator(m_allocator);
    }
    if (m_device) {
        vkDestroyDevice(m_device, nullptr);
    }
    if (m_debugMessenger) {
        auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkDestroyDebugUtilsMessengerEXT");
        if (func) {
            func(m_instance, m_debugMessenger, nullptr);
        }
    }
    if (m_instance) {
        vkDestroyInstance(m_instance, nullptr);
    }
}

std::vector<VkPhysicalDevice> VulkanContext::enumeratePhysicalDevices(VkInstance instance) {
    uint32_t deviceCount = 0;
    vkEnumeratePhysicalDevices(instance, &deviceCount, nullptr);
    std::vector<VkPhysicalDevice> devices(deviceCount);
    vkEnumeratePhysicalDevices(instance, &deviceCount, devices.data());
    return devices;
}

void VulkanContext::createInstance(const Config& config) {
    VkApplicationInfo appInfo{};
    appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    appInfo.pApplicationName = "Pathways";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "PathwaysEngine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = VK_API_VERSION_1_4;

    std::vector<const char*> instanceExtensions;
    if (!config.headless) {
        instanceExtensions.push_back("VK_KHR_surface");
        instanceExtensions.push_back("VK_KHR_wayland_surface");
        instanceExtensions.push_back("VK_KHR_xcb_surface");
        instanceExtensions.push_back("VK_KHR_xlib_surface");
    }
    if (config.validation_layers) {
        instanceExtensions.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }

    // Query supported instance extensions to filter out unsupported ones
    uint32_t supportedExtCount = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtCount, nullptr);
    std::vector<VkExtensionProperties> supportedExtensions(supportedExtCount);
    vkEnumerateInstanceExtensionProperties(nullptr, &supportedExtCount, supportedExtensions.data());

    std::vector<const char*> enabledExtensions;
    for (const char* ext : instanceExtensions) {
        bool found = false;
        for (const auto& sup : supportedExtensions) {
            if (std::strcmp(ext, sup.extensionName) == 0) {
                found = true;
                break;
            }
        }
        if (found) {
            enabledExtensions.push_back(ext);
        } else {
            Logger::debug("Instance extension not present: {}", ext);
        }
    }

    VkInstanceCreateInfo createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    createInfo.pApplicationInfo = &appInfo;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(enabledExtensions.size());
    createInfo.ppEnabledExtensionNames = enabledExtensions.data();

    if (config.validation_layers) {
        createInfo.enabledLayerCount = static_cast<uint32_t>(g_validationLayers.size());
        createInfo.ppEnabledLayerNames = g_validationLayers.data();
    } else {
        createInfo.enabledLayerCount = 0;
    }

    VkResult res = vkCreateInstance(&createInfo, nullptr, &m_instance);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan Instance! Result: " + std::to_string(res));
    }
}

void VulkanContext::setupDebugMessenger() {
    VkDebugUtilsMessengerCreateInfoEXT createInfo{};
    createInfo.sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT;
    createInfo.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT |
                                 VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    createInfo.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                             VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
    createInfo.pfnUserCallback = debugCallback;

    auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(m_instance, "vkCreateDebugUtilsMessengerEXT");
    if (func) {
        func(m_instance, &createInfo, nullptr, &m_debugMessenger);
    }
}

void VulkanContext::selectPhysicalDevice(const Config& config, VkSurfaceKHR surface) {
    auto devices = enumeratePhysicalDevices(m_instance);
    if (devices.empty()) {
        throw std::runtime_error("Failed to find GPUs with Vulkan support!");
    }

    Logger::info("Found {} Vulkan physical device(s):", devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        Logger::info("  [{}] {} (Vulkan {}.{}.{})", i, props.deviceName,
                     VK_VERSION_MAJOR(props.apiVersion),
                     VK_VERSION_MINOR(props.apiVersion),
                     VK_VERSION_PATCH(props.apiVersion));
    }

    // Pick GPU by index or prefer discrete RDNA4
    uint32_t selectedIdx = config.gpu_index;
    if (selectedIdx >= devices.size()) {
        Logger::warn("Selected GPU index {} out of range, defaulting to 0", selectedIdx);
        selectedIdx = 0;
    }

    m_physicalDevice = devices[selectedIdx];
    vkGetPhysicalDeviceProperties(m_physicalDevice, &m_deviceProperties);
    m_deviceName = m_deviceProperties.deviceName;

    if (m_deviceName.find("R9700") != std::string::npos || m_deviceName.find("GFX1201") != std::string::npos) {
        m_isRDNA4 = true;
        Logger::info("Identified target hardware: AMD RDNA4 Architecture (GFX1201)");
    }

    // Query Queue Families
    uint32_t queueFamilyCount = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, nullptr);
    std::vector<VkQueueFamilyProperties> queueFamilies(queueFamilyCount);
    vkGetPhysicalDeviceQueueFamilyProperties(m_physicalDevice, &queueFamilyCount, queueFamilies.data());

    for (uint32_t i = 0; i < queueFamilyCount; ++i) {
        if ((queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) &&
            (queueFamilies[i].queueFlags & VK_QUEUE_COMPUTE_BIT)) {
            m_queueIndices.graphicsComputeFamily = i;
        } else if ((queueFamilies[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                   !(queueFamilies[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            m_queueIndices.transferFamily = i;
        }
    }

    if (m_queueIndices.transferFamily == UINT32_MAX) {
        // Fallback to graphics family for transfer if no dedicated transfer queue
        m_queueIndices.transferFamily = m_queueIndices.graphicsComputeFamily;
    }

    // Query Device Extensions
    uint32_t extCount = 0;
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, nullptr);
    std::vector<VkExtensionProperties> availableExtensions(extCount);
    vkEnumerateDeviceExtensionProperties(m_physicalDevice, nullptr, &extCount, availableExtensions.data());

    for (const auto& ext : availableExtensions) {
        if (std::strcmp(ext.extensionName, VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME) == 0) {
            m_hasDGC = true;
        }
        if (std::strcmp(ext.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0) {
            m_hasRayTracing = true;
        }
    }

    // Check Subgroup Size Control (Wave32 support)
    VkPhysicalDeviceSubgroupSizeControlProperties subgroupProps{};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);

    if (subgroupProps.minSubgroupSize <= 32 && subgroupProps.maxSubgroupSize >= 32 &&
        (subgroupProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        m_hasSubgroupSizeControl = true;
    }

    Logger::info("Device Capabilities -> DGC: {}, Hardware RT: {}, SubgroupSizeControl (Wave32): {}",
                 m_hasDGC ? "SUPPORTED" : "NOT FOUND",
                 m_hasRayTracing ? "SUPPORTED" : "NOT FOUND",
                 m_hasSubgroupSizeControl ? "SUPPORTED" : "NOT FOUND");
}

void VulkanContext::createLogicalDevice(const Config& config) {
    std::vector<VkDeviceQueueCreateInfo> queueCreateInfos;
    float queuePriority = 1.0f;

    VkDeviceQueueCreateInfo graphicsQueueCreateInfo{};
    graphicsQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    graphicsQueueCreateInfo.queueFamilyIndex = m_queueIndices.graphicsComputeFamily;
    graphicsQueueCreateInfo.queueCount = 1;
    graphicsQueueCreateInfo.pQueuePriorities = &queuePriority;
    queueCreateInfos.push_back(graphicsQueueCreateInfo);

    if (m_queueIndices.transferFamily != m_queueIndices.graphicsComputeFamily) {
        VkDeviceQueueCreateInfo transferQueueCreateInfo{};
        transferQueueCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        transferQueueCreateInfo.queueFamilyIndex = m_queueIndices.transferFamily;
        transferQueueCreateInfo.queueCount = 1;
        transferQueueCreateInfo.pQueuePriorities = &queuePriority;
        queueCreateInfos.push_back(transferQueueCreateInfo);
    }

    // Device extensions
    std::vector<const char*> deviceExtensions;
    if (!config.headless) {
        deviceExtensions.push_back(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    }
    if (m_hasRayTracing) {
        deviceExtensions.push_back(VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_RAY_QUERY_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME);
    }
    if (m_hasDGC) {
        deviceExtensions.push_back(VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME);
    }
    deviceExtensions.push_back(VK_KHR_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME);
    deviceExtensions.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);

    // Vulkan 1.4 / 1.3 / 1.2 Features chaining
    VkPhysicalDeviceVulkan14Features features14{};
    features14.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_4_FEATURES;
    features14.shaderSubgroupRotate = VK_TRUE;
    features14.shaderSubgroupRotateClustered = VK_TRUE;
    features14.shaderFloatControls2 = VK_TRUE;
    features14.shaderExpectAssume = VK_TRUE;
    features14.dynamicRenderingLocalRead = VK_TRUE;
    features14.maintenance5 = VK_TRUE;
    features14.maintenance6 = VK_TRUE;
    features14.pushDescriptor = VK_TRUE;

    VkPhysicalDeviceVulkan13Features features13{};
    features13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    features13.dynamicRendering = VK_TRUE;
    features13.synchronization2 = VK_TRUE;
    if (m_hasSubgroupSizeControl) {
        features13.subgroupSizeControl = VK_TRUE;
        features13.computeFullSubgroups = VK_TRUE;
    }
    features13.pNext = &features14;

    VkPhysicalDeviceVulkan12Features features12{};
    features12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    features12.bufferDeviceAddress = VK_TRUE;
    features12.descriptorIndexing = VK_TRUE;
    features12.runtimeDescriptorArray = VK_TRUE;
    features12.descriptorBindingPartiallyBound = VK_TRUE;
    features12.descriptorBindingVariableDescriptorCount = VK_TRUE;
    features12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;
    features12.timelineSemaphore = VK_TRUE;
    features12.pNext = &features13;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{};
    rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeatures.rayTracingPipeline = VK_TRUE;

    VkPhysicalDeviceAccelerationStructureFeaturesKHR asFeatures{};
    asFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR;
    asFeatures.accelerationStructure = VK_TRUE;
    asFeatures.pNext = &rtPipelineFeatures;

    VkPhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
    rayQueryFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR;
    rayQueryFeatures.rayQuery = VK_TRUE;
    rayQueryFeatures.pNext = &asFeatures;

    VkPhysicalDeviceDeviceGeneratedCommandsFeaturesEXT dgcFeatures{};
    dgcFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_FEATURES_EXT;
    dgcFeatures.deviceGeneratedCommands = VK_TRUE;
    dgcFeatures.dynamicGeneratedPipelineLayout = VK_TRUE;
    dgcFeatures.pNext = &rayQueryFeatures;

    if (m_hasRayTracing && m_hasDGC) {
        features14.pNext = &dgcFeatures;
    } else if (m_hasRayTracing) {
        features14.pNext = &rayQueryFeatures;
    }

    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.features.samplerAnisotropy = VK_TRUE;
    deviceFeatures2.features.shaderInt64 = VK_TRUE;
    deviceFeatures2.pNext = &features12;

    VkDeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    deviceCreateInfo.queueCreateInfoCount = static_cast<uint32_t>(queueCreateInfos.size());
    deviceCreateInfo.pQueueCreateInfos = queueCreateInfos.data();
    deviceCreateInfo.enabledExtensionCount = static_cast<uint32_t>(deviceExtensions.size());
    deviceCreateInfo.ppEnabledExtensionNames = deviceExtensions.data();
    deviceCreateInfo.pNext = &deviceFeatures2;

    VkResult res = vkCreateDevice(m_physicalDevice, &deviceCreateInfo, nullptr, &m_device);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create Vulkan logical device! Result: " + std::to_string(res));
    }

    vkGetDeviceQueue(m_device, m_queueIndices.graphicsComputeFamily, 0, &m_graphicsQueue);
    vkGetDeviceQueue(m_device, m_queueIndices.transferFamily, 0, &m_transferQueue);
}

void VulkanContext::initVMA() {
    VmaVulkanFunctions vulkanFunctions{};
    vulkanFunctions.vkGetInstanceProcAddr = vkGetInstanceProcAddr;
    vulkanFunctions.vkGetDeviceProcAddr = vkGetDeviceProcAddr;

    VmaAllocatorCreateInfo allocatorCreateInfo{};
    allocatorCreateInfo.vulkanApiVersion = VK_API_VERSION_1_4;
    allocatorCreateInfo.physicalDevice = m_physicalDevice;
    allocatorCreateInfo.device = m_device;
    allocatorCreateInfo.instance = m_instance;
    allocatorCreateInfo.pVulkanFunctions = &vulkanFunctions;
    allocatorCreateInfo.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;

    VkResult res = vmaCreateAllocator(&allocatorCreateInfo, &m_allocator);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to initialize Vulkan Memory Allocator (VMA)!");
    }
    Logger::info("VMA Allocator initialized successfully.");
}

} // namespace pathways
