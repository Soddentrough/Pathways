#include "vulkan/VulkanContext.hpp"
#include "core/Logger.hpp"

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

#include <vector>
#include <cstring>
#include <stdexcept>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <format>

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
#ifdef _WIN32
        instanceExtensions.push_back("VK_KHR_win32_surface");
#else
        instanceExtensions.push_back("VK_KHR_wayland_surface");
        instanceExtensions.push_back("VK_KHR_xcb_surface");
        instanceExtensions.push_back("VK_KHR_xlib_surface");
#endif
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
        throw std::runtime_error("No Vulkan physical devices found!");
    }

    Logger::info("Enumerated {} physical Vulkan device(s):", devices.size());
    for (size_t i = 0; i < devices.size(); ++i) {
        VkPhysicalDeviceProperties props;
        vkGetPhysicalDeviceProperties(devices[i], &props);
        Logger::info("  [{}] {} (Driver: {}.{}.{})",
                     i, props.deviceName,
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

    std::string nameLower = m_deviceName;
    std::transform(nameLower.begin(), nameLower.end(), nameLower.begin(), ::tolower);

    if (m_deviceProperties.vendorID == 0x1002 || nameLower.find("amd") != std::string::npos || nameLower.find("radeon") != std::string::npos) {
        if (nameLower.find("rdna4") != std::string::npos || nameLower.find("gfx12") != std::string::npos ||
            nameLower.find("r9700") != std::string::npos || nameLower.find("9070") != std::string::npos ||
            nameLower.find("9080") != std::string::npos || nameLower.find("9060") != std::string::npos ||
            nameLower.find("rx 9") != std::string::npos || nameLower.find("rx9") != std::string::npos) {
            m_architecture = GpuArchitecture::AmdRDNA4;
            m_isRDNA4 = true;
        } else if (nameLower.find("rdna3.5") != std::string::npos || nameLower.find("gfx115") != std::string::npos ||
                   nameLower.find("890m") != std::string::npos || nameLower.find("880m") != std::string::npos) {
            m_architecture = GpuArchitecture::AmdRDNA3_5;
        } else if (nameLower.find("rdna3") != std::string::npos || nameLower.find("gfx11") != std::string::npos ||
                   nameLower.find("7900") != std::string::npos || nameLower.find("7800") != std::string::npos ||
                   nameLower.find("7700") != std::string::npos || nameLower.find("7600") != std::string::npos ||
                   nameLower.find("w7900") != std::string::npos || nameLower.find("w7800") != std::string::npos ||
                   nameLower.find("rx 7") != std::string::npos || nameLower.find("rx7") != std::string::npos) {
            m_architecture = GpuArchitecture::AmdRDNA3;
        } else if (nameLower.find("rdna2") != std::string::npos || nameLower.find("gfx103") != std::string::npos ||
                   nameLower.find("6950") != std::string::npos || nameLower.find("6900") != std::string::npos ||
                   nameLower.find("6800") != std::string::npos || nameLower.find("6700") != std::string::npos ||
                   nameLower.find("6600") != std::string::npos || nameLower.find("6500") != std::string::npos ||
                   nameLower.find("rx 6") != std::string::npos || nameLower.find("rx6") != std::string::npos) {
            m_architecture = GpuArchitecture::AmdRDNA2;
        } else if (nameLower.find("rdna1") != std::string::npos || nameLower.find("gfx101") != std::string::npos ||
                   nameLower.find("5700") != std::string::npos || nameLower.find("5600") != std::string::npos ||
                   nameLower.find("5500") != std::string::npos || nameLower.find("rx 57") != std::string::npos ||
                   nameLower.find("rx 56") != std::string::npos) {
            m_architecture = GpuArchitecture::AmdRDNA1;
        } else {
            m_architecture = GpuArchitecture::Generic;
        }
    } else if (m_deviceProperties.vendorID == 0x10DE || nameLower.find("nvidia") != std::string::npos || nameLower.find("geforce") != std::string::npos) {
        if (nameLower.find("blackwell") != std::string::npos || nameLower.find("5090") != std::string::npos ||
            nameLower.find("5080") != std::string::npos || nameLower.find("5070") != std::string::npos ||
            nameLower.find("5060") != std::string::npos || nameLower.find("rtx 50") != std::string::npos ||
            nameLower.find("rtx50") != std::string::npos) {
            m_architecture = GpuArchitecture::NvidiaBlackwell;
        } else if (nameLower.find("ada") != std::string::npos || nameLower.find("4090") != std::string::npos ||
                   nameLower.find("4080") != std::string::npos || nameLower.find("4070") != std::string::npos ||
                   nameLower.find("4060") != std::string::npos || nameLower.find("4050") != std::string::npos ||
                   nameLower.find("rtx 40") != std::string::npos || nameLower.find("rtx40") != std::string::npos) {
            m_architecture = GpuArchitecture::NvidiaAda;
        } else if (nameLower.find("ampere") != std::string::npos || nameLower.find("3090") != std::string::npos ||
                   nameLower.find("3080") != std::string::npos || nameLower.find("3070") != std::string::npos ||
                   nameLower.find("3060") != std::string::npos || nameLower.find("3050") != std::string::npos ||
                   nameLower.find("rtx 30") != std::string::npos || nameLower.find("rtx30") != std::string::npos) {
            m_architecture = GpuArchitecture::NvidiaAmpere;
        } else if (nameLower.find("turing") != std::string::npos || nameLower.find("2080") != std::string::npos ||
                   nameLower.find("2070") != std::string::npos || nameLower.find("2060") != std::string::npos ||
                   nameLower.find("rtx 20") != std::string::npos || nameLower.find("rtx20") != std::string::npos ||
                   nameLower.find("titan rtx") != std::string::npos) {
            m_architecture = GpuArchitecture::NvidiaTuring;
        } else {
            m_architecture = GpuArchitecture::Generic;
        }
    } else if (m_deviceProperties.vendorID == 0x8086 || nameLower.find("intel") != std::string::npos || nameLower.find("arc") != std::string::npos) {
        m_architecture = GpuArchitecture::IntelArc;
    } else {
        m_architecture = GpuArchitecture::Generic;
    }

    Logger::info("Identified target hardware: {} ({})", m_deviceName, getArchitectureName());

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

    bool hasPciBusInfo = false;
    for (const auto& ext : availableExtensions) {
        if (std::strcmp(ext.extensionName, VK_EXT_DEVICE_GENERATED_COMMANDS_EXTENSION_NAME) == 0) {
            m_hasDGC = true;
        }
        if (std::strcmp(ext.extensionName, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME) == 0) {
            m_hasRayTracing = true;
        }
        if (std::strcmp(ext.extensionName, VK_EXT_PCI_BUS_INFO_EXTENSION_NAME) == 0) {
            hasPciBusInfo = true;
        }
        if (std::strcmp(ext.extensionName, "VK_EXT_external_memory_host") == 0) {
            m_hasExternalMemoryHost = true;
        }
        if (std::strcmp(ext.extensionName, "VK_KHR_external_memory_fd") == 0) {
            m_hasExternalMemoryFd = true;
        }
        if (std::strcmp(ext.extensionName, "VK_EXT_external_memory_dma_buf") == 0) {
            m_hasExternalMemoryDmaBuf = true;
        }
        if (std::strcmp(ext.extensionName, "VK_KHR_external_semaphore_fd") == 0) {
            m_hasExternalSemaphoreFd = true;
        }
        if (std::strcmp(ext.extensionName, VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0) {
            m_hasCooperativeMatrix = true;
        }
    }

    // Check Subgroup Size Control (Wave32 support), DGC Properties, and Ray Tracing Pipeline Properties
    VkPhysicalDeviceSubgroupSizeControlProperties subgroupProps{};
    subgroupProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES;
    VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT dgcProps{};
    dgcProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DEVICE_GENERATED_COMMANDS_PROPERTIES_EXT;
    subgroupProps.pNext = &dgcProps;

    m_rtPipelineProperties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_PROPERTIES_KHR;
    dgcProps.pNext = &m_rtPipelineProperties;

    VkPhysicalDevicePCIBusInfoPropertiesEXT pciBusProps{};
    pciBusProps.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PCI_BUS_INFO_PROPERTIES_EXT;
    if (hasPciBusInfo) {
        m_rtPipelineProperties.pNext = &pciBusProps;
    }

    VkPhysicalDeviceProperties2 props2{};
    props2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2;
    props2.pNext = &subgroupProps;
    vkGetPhysicalDeviceProperties2(m_physicalDevice, &props2);

    if (hasPciBusInfo) {
        m_pciLinkInfo.domain = pciBusProps.pciDomain;
        m_pciLinkInfo.bus = pciBusProps.pciBus;
        m_pciLinkInfo.device = pciBusProps.pciDevice;
        m_pciLinkInfo.function = pciBusProps.pciFunction;
        m_pciLinkInfo.bdfString = std::format("{:04x}:{:02x}:{:02x}.{:x}",
                                              m_pciLinkInfo.domain,
                                              m_pciLinkInfo.bus,
                                              m_pciLinkInfo.device,
                                              m_pciLinkInfo.function);
        m_pciLinkInfo.valid = true;
        refreshPciLinkInfo();
        Logger::info("PCI Bus Info: {} | Link: {}", m_pciLinkInfo.bdfString, m_pciLinkInfo.formattedLink);
    }

    if (subgroupProps.minSubgroupSize <= 32 && subgroupProps.maxSubgroupSize >= 32 &&
        (subgroupProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT)) {
        m_hasSubgroupSizeControl = true;
    }

    Logger::info("Device Capabilities -> DGC: {}, Hardware RT: {}, SubgroupSizeControl (Wave32): {}",
                 m_hasDGC ? "SUPPORTED" : "NOT FOUND",
                 m_hasRayTracing ? "SUPPORTED" : "NOT FOUND",
                 m_hasSubgroupSizeControl ? "SUPPORTED" : "NOT FOUND");
    if (m_hasRayTracing) {
        Logger::info("RT Pipeline Properties -> HandleSize: {} B, BaseAlign: {} B, HandleAlign: {} B, MaxRecursion: {}",
                     m_rtPipelineProperties.shaderGroupHandleSize,
                     m_rtPipelineProperties.shaderGroupBaseAlignment,
                     m_rtPipelineProperties.shaderGroupHandleAlignment,
                     m_rtPipelineProperties.maxRayRecursionDepth);
    }
    if (m_hasDGC) {
        if ((dgcProps.supportedIndirectCommandsShaderStagesPipelineBinding & VK_SHADER_STAGE_COMPUTE_BIT) ||
            (dgcProps.supportedIndirectCommandsShaderStagesShaderBinding & VK_SHADER_STAGE_COMPUTE_BIT)) {
            m_hasDgcExecutionSet = true;
        }
        Logger::info("DGC Hardware Limits -> MaxTokens: {}, MaxSequences: {}, MaxStride: {} B, Stages: 0x{:x}, ComputeExecutionSet: {}",
                     dgcProps.maxIndirectCommandsTokenCount,
                     dgcProps.maxIndirectSequenceCount,
                     dgcProps.maxIndirectCommandsIndirectStride,
                     dgcProps.supportedIndirectCommandsShaderStages,
                     m_hasDgcExecutionSet ? "SUPPORTED" : "UNSUPPORTED");
    }
}

void VulkanContext::refreshPciLinkInfo() {
#ifdef __linux__
    if (!m_pciLinkInfo.valid || m_pciLinkInfo.bdfString.empty()) {
        // Fallback: enumerate /sys/class/drm to find device matching vendor/device ID
        std::error_code ec;
        for (const auto& entry : std::filesystem::directory_iterator("/sys/class/drm", ec)) {
            std::string fname = entry.path().filename().string();
            if (fname.rfind("card", 0) == 0 && fname.find('-') == std::string::npos) {
                std::string devPath = entry.path().string() + "/device";
                std::ifstream vFile(devPath + "/vendor");
                std::ifstream dFile(devPath + "/device");
                std::string vStr, dStr;
                if (vFile >> vStr && dFile >> dStr) {
                    try {
                        uint32_t venId = std::stoul(vStr, nullptr, 16);
                        uint32_t devId = std::stoul(dStr, nullptr, 16);
                        if (venId == m_deviceProperties.vendorID && devId == m_deviceProperties.deviceID) {
                            std::ifstream ueventFile(devPath + "/uevent");
                            std::string uline;
                            while (std::getline(ueventFile, uline)) {
                                if (uline.rfind("PCI_SLOT_NAME=", 0) == 0) {
                                    m_pciLinkInfo.bdfString = uline.substr(14);
                                    m_pciLinkInfo.valid = true;
                                    break;
                                }
                            }
                            if (m_pciLinkInfo.valid) break;
                        }
                    } catch (...) {}
                }
            }
        }
    }

    if (!m_pciLinkInfo.valid || m_pciLinkInfo.bdfString.empty()) {
        m_pciLinkInfo.formattedLink = "PCIe N/A";
        return;
    }

    std::string pciDir = "/sys/bus/pci/devices/" + m_pciLinkInfo.bdfString;
    std::error_code ec;
    if (!std::filesystem::exists(pciDir, ec)) {
        m_pciLinkInfo.formattedLink = "PCIe (Sysfs N/A)";
        return;
    }

    auto readFileTrim = [](const std::string& path) -> std::string {
        std::ifstream file(path);
        if (!file.is_open()) return "";
        std::string line;
        if (std::getline(file, line)) {
            while (!line.empty() && (line.back() == '\r' || line.back() == '\n' || line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            if (line.size() > 5 && line.substr(line.size() - 5) == " PCIe") {
                line.resize(line.size() - 5);
            }
            return line;
        }
        return "";
    };

    // 1. Check pp_dpm_pcie for active PHY link state on AMD GPUs
    std::string dpmPath = pciDir + "/pp_dpm_pcie";
    bool gotDpm = false;
    std::ifstream dpmFile(dpmPath);
    if (dpmFile.is_open()) {
        std::string line;
        while (std::getline(dpmFile, line)) {
            if (line.find('*') != std::string::npos) {
                // e.g. "1: 16.0GT/s, x16 616Mhz *" or "1: 16.0GT/s, x8 616Mhz *"
                size_t gtPos = line.find("GT/s");
                if (gtPos != std::string::npos) {
                    size_t start = gtPos;
                    while (start > 0 && (std::isdigit(line[start - 1]) || line[start - 1] == '.' || line[start - 1] == ' ')) {
                        start--;
                    }
                    while (start < gtPos && line[start] == ' ') start++;
                    std::string spd = line.substr(start, gtPos + 4 - start);
                    if (spd.find("GT/s") != std::string::npos && spd.find(" GT/s") == std::string::npos) {
                        size_t pos = spd.find("GT/s");
                        spd = spd.substr(0, pos) + " GT/s";
                    }
                    m_pciLinkInfo.currentSpeed = spd;
                }

                size_t xPos = line.find(", x");
                if (xPos != std::string::npos) {
                    size_t wStart = xPos + 3;
                    size_t wEnd = wStart;
                    while (wEnd < line.size() && std::isdigit(line[wEnd])) {
                        wEnd++;
                    }
                    if (wEnd > wStart) {
                        try {
                            m_pciLinkInfo.currentWidth = std::stoul(line.substr(wStart, wEnd - wStart));
                            gotDpm = true;
                        } catch (...) {}
                    }
                }
                break;
            }
        }
    }

    // 2. Fallback to current_link_speed and current_link_width if pp_dpm_pcie was not present
    if (!gotDpm) {
        m_pciLinkInfo.currentSpeed = readFileTrim(pciDir + "/current_link_speed");
        std::string curWidthStr = readFileTrim(pciDir + "/current_link_width");
        try {
            if (!curWidthStr.empty()) m_pciLinkInfo.currentWidth = std::stoul(curWidthStr);
        } catch (...) {}
    }

    // 3. Read max capabilities
    m_pciLinkInfo.maxSpeed = readFileTrim(pciDir + "/max_link_speed");
    std::string maxWidthStr = readFileTrim(pciDir + "/max_link_width");
    try {
        if (!maxWidthStr.empty()) m_pciLinkInfo.maxWidth = std::stoul(maxWidthStr);
    } catch (...) {}

    auto getGenName = [](const std::string& speed) -> std::string {
        if (speed.find("32.0 GT/s") != std::string::npos || speed.find("32.0GT/s") != std::string::npos) return "PCIe 5.0";
        if (speed.find("16.0 GT/s") != std::string::npos || speed.find("16.0GT/s") != std::string::npos) return "PCIe 4.0";
        if (speed.find("8.0 GT/s") != std::string::npos || speed.find("8.0GT/s") != std::string::npos) return "PCIe 3.0";
        if (speed.find("5.0 GT/s") != std::string::npos || speed.find("5.0GT/s") != std::string::npos) return "PCIe 2.0";
        if (speed.find("2.5 GT/s") != std::string::npos || speed.find("2.5GT/s") != std::string::npos) return "PCIe 1.0";
        return "PCIe";
    };

    m_pciLinkInfo.generationName = getGenName(m_pciLinkInfo.currentSpeed);
    m_pciLinkInfo.maxGenerationName = getGenName(m_pciLinkInfo.maxSpeed);

    // 4. Identify link configuration (bifurcated slot width vs device maximum capability)
    std::string reason;
    if (m_pciLinkInfo.maxWidth > 0 && m_pciLinkInfo.currentWidth > 0 && m_pciLinkInfo.currentWidth < m_pciLinkInfo.maxWidth) {
        reason = std::format("Slot allocated x{} lanes via platform bifurcation/storage (Device supports up to x{})", m_pciLinkInfo.currentWidth, m_pciLinkInfo.maxWidth);
    }
    if (!m_pciLinkInfo.maxGenerationName.empty() && !m_pciLinkInfo.generationName.empty() &&
        m_pciLinkInfo.generationName != "PCIe" && m_pciLinkInfo.maxGenerationName != "PCIe" &&
        m_pciLinkInfo.generationName != m_pciLinkInfo.maxGenerationName) {
        std::string speedMsg = std::format("Platform link negotiated at {} (Device supports {})",
                                           m_pciLinkInfo.generationName, m_pciLinkInfo.maxGenerationName);
        if (reason.empty()) {
            reason = speedMsg;
        } else {
            reason += "; " + speedMsg;
        }
    }
    m_pciLinkInfo.isDegraded = false; // Link width is intentional platform bifurcation / lane allocation
    m_pciLinkInfo.degradationReason = reason;

    if (m_pciLinkInfo.currentWidth > 0 && !m_pciLinkInfo.currentSpeed.empty()) {
        if (m_pciLinkInfo.maxWidth > 0 && !m_pciLinkInfo.maxGenerationName.empty()) {
            m_pciLinkInfo.formattedLink = std::format("{} x{} (Max: {} x{})",
                m_pciLinkInfo.generationName,
                m_pciLinkInfo.currentWidth,
                m_pciLinkInfo.maxGenerationName,
                m_pciLinkInfo.maxWidth);
        } else {
            m_pciLinkInfo.formattedLink = std::format("{} x{}",
                m_pciLinkInfo.generationName,
                m_pciLinkInfo.currentWidth);
        }
    } else {
        m_pciLinkInfo.formattedLink = "PCIe Unknown";
    }

    // Locate hwmon directory under pciDir/hwmon
    std::string hwmonBase = pciDir + "/hwmon";
    if (std::filesystem::exists(hwmonBase, ec)) {
        for (const auto& entry : std::filesystem::directory_iterator(hwmonBase, ec)) {
            if (entry.is_directory() && entry.path().filename().string().rfind("hwmon", 0) == 0) {
                m_pciLinkInfo.hwmonPath = entry.path().string();
                break;
            }
        }
    }
#else
    m_pciLinkInfo.formattedLink = "PCIe (Non-Linux)";
#endif
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
    if (m_hasExternalMemoryHost || m_hasExternalMemoryFd) {
        deviceExtensions.push_back("VK_KHR_external_memory");
    }
    if (m_hasExternalMemoryHost) {
        deviceExtensions.push_back("VK_EXT_external_memory_host");
    }
    if (m_hasExternalMemoryFd) {
        deviceExtensions.push_back("VK_KHR_external_memory_fd");
    }
    if (m_hasExternalMemoryDmaBuf) {
        deviceExtensions.push_back("VK_EXT_external_memory_dma_buf");
    }
    if (m_hasExternalSemaphoreFd) {
        deviceExtensions.push_back("VK_KHR_external_semaphore_fd");
    }
    if (m_hasCooperativeMatrix) {
        deviceExtensions.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
        deviceExtensions.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    }

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
    features12.vulkanMemoryModel = VK_TRUE;
    features12.vulkanMemoryModelDeviceScope = VK_TRUE;
    features12.shaderFloat16 = VK_TRUE;
    features12.pNext = &features13;

    VkPhysicalDeviceVulkan11Features features11{};
    features11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    features11.storageBuffer16BitAccess = VK_TRUE;
    features11.uniformAndStorageBuffer16BitAccess = VK_TRUE;
    features11.pNext = &features12;

    VkPhysicalDeviceRayTracingPipelineFeaturesKHR rtPipelineFeatures{};
    rtPipelineFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_TRACING_PIPELINE_FEATURES_KHR;
    rtPipelineFeatures.rayTracingPipeline = VK_TRUE;
    rtPipelineFeatures.rayTracingPipelineTraceRaysIndirect = VK_TRUE;

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

    void* tailFeature = nullptr;
    if (m_hasRayTracing && m_hasDGC) {
        features14.pNext = &dgcFeatures;
        tailFeature = &rtPipelineFeatures;
    } else if (m_hasRayTracing) {
        features14.pNext = &rayQueryFeatures;
        tailFeature = &rtPipelineFeatures;
    } else {
        tailFeature = &features14;
    }

    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopMatFeatures{};
    coopMatFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    coopMatFeatures.cooperativeMatrix = VK_TRUE;
    coopMatFeatures.pNext = nullptr;

    if (m_hasCooperativeMatrix) {
        if (tailFeature == &rtPipelineFeatures) {
            rtPipelineFeatures.pNext = &coopMatFeatures;
        } else if (tailFeature == &features14) {
            features14.pNext = &coopMatFeatures;
        }
    }

    VkPhysicalDeviceFeatures2 deviceFeatures2{};
    deviceFeatures2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    deviceFeatures2.features.samplerAnisotropy = VK_TRUE;
    deviceFeatures2.features.shaderInt64 = VK_TRUE;
    deviceFeatures2.features.shaderStorageImageWriteWithoutFormat = VK_TRUE;
    deviceFeatures2.features.shaderStorageImageReadWithoutFormat = VK_TRUE;
    deviceFeatures2.pNext = &features11;

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

    if (m_hasExternalSemaphoreFd) {
        pfnGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(m_device, "vkGetSemaphoreFdKHR");
        pfnImportSemaphoreFdKHR = (PFN_vkImportSemaphoreFdKHR)vkGetDeviceProcAddr(m_device, "vkImportSemaphoreFdKHR");
    }
    if (m_hasExternalMemoryFd) {
        pfnGetMemoryFdKHR = (PFN_vkGetMemoryFdKHR)vkGetDeviceProcAddr(m_device, "vkGetMemoryFdKHR");
        pfnGetMemoryFdPropertiesKHR = (PFN_vkGetMemoryFdPropertiesKHR)vkGetDeviceProcAddr(m_device, "vkGetMemoryFdPropertiesKHR");
    }
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

std::string VulkanContext::getArchitectureName() const {
    switch (m_architecture) {
        case GpuArchitecture::AmdRDNA4:        return "AMD RDNA4 (GFX1201)";
        case GpuArchitecture::AmdRDNA3_5:      return "AMD RDNA3.5 (GFX1150)";
        case GpuArchitecture::AmdRDNA3:        return "AMD RDNA3 (Navi 3x)";
        case GpuArchitecture::AmdRDNA2:        return "AMD RDNA2 (Navi 2x)";
        case GpuArchitecture::AmdRDNA1:        return "AMD RDNA1 (Navi 1x)";
        case GpuArchitecture::NvidiaBlackwell: return "NVIDIA Blackwell";
        case GpuArchitecture::NvidiaAda:       return "NVIDIA Ada Lovelace";
        case GpuArchitecture::NvidiaAmpere:    return "NVIDIA Ampere";
        case GpuArchitecture::NvidiaTuring:    return "NVIDIA Turing";
        case GpuArchitecture::IntelArc:        return "Intel Arc Xe-HPG";
        default:                               return "Vulkan 1.4 Native GPU";
    }
}

std::string VulkanContext::getShortArchName() const {
    switch (m_architecture) {
        case GpuArchitecture::AmdRDNA4:        return "RDNA4";
        case GpuArchitecture::AmdRDNA3_5:      return "RDNA3.5";
        case GpuArchitecture::AmdRDNA3:        return "RDNA3";
        case GpuArchitecture::AmdRDNA2:        return "RDNA2";
        case GpuArchitecture::AmdRDNA1:        return "RDNA1";
        case GpuArchitecture::NvidiaBlackwell: return "Blackwell";
        case GpuArchitecture::NvidiaAda:       return "Ada";
        case GpuArchitecture::NvidiaAmpere:    return "Ampere";
        case GpuArchitecture::NvidiaTuring:    return "Turing";
        case GpuArchitecture::IntelArc:        return "Intel Arc";
        default:                               return "Vulkan";
    }
}

std::string VulkanContext::getRayAcceleratorName() const {
    switch (m_architecture) {
        case GpuArchitecture::AmdRDNA4:        return "AMD RDNA4 3rd Gen Ray Accelerators";
        case GpuArchitecture::AmdRDNA3_5:
        case GpuArchitecture::AmdRDNA3:        return "AMD RDNA3 2nd Gen Ray Accelerators";
        case GpuArchitecture::AmdRDNA2:        return "AMD RDNA2 1st Gen Ray Accelerators";
        case GpuArchitecture::NvidiaBlackwell: return "NVIDIA 5th Gen RT Cores";
        case GpuArchitecture::NvidiaAda:       return "NVIDIA 4th Gen RT Cores";
        case GpuArchitecture::NvidiaAmpere:    return "NVIDIA 3rd Gen RT Cores";
        case GpuArchitecture::NvidiaTuring:    return "NVIDIA 2nd Gen RT Cores";
        case GpuArchitecture::IntelArc:        return "Intel Xe Ray Tracing Units";
        default:                               return "Hardware Ray Queries (VK_KHR_ray_query)";
    }
}

bool VulkanContext::isRDNA() const {
    return m_architecture == GpuArchitecture::AmdRDNA1 ||
           m_architecture == GpuArchitecture::AmdRDNA2 ||
           m_architecture == GpuArchitecture::AmdRDNA3 ||
           m_architecture == GpuArchitecture::AmdRDNA3_5 ||
           m_architecture == GpuArchitecture::AmdRDNA4;
}

uint64_t VulkanContext::getTotalVramBytes() const {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);
    uint64_t totalDeviceLocal = 0;
    for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i) {
        if (memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            totalDeviceLocal += memProperties.memoryHeaps[i].size;
        }
    }
    return totalDeviceLocal;
}

uint64_t VulkanContext::getAllocatedVramBytes() const {
    if (!m_allocator) return 0;
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProperties);
    std::vector<VmaBudget> budgets(memProperties.memoryHeapCount);
    vmaGetHeapBudgets(m_allocator, budgets.data());
    uint64_t allocated = 0;
    for (uint32_t i = 0; i < memProperties.memoryHeapCount; ++i) {
        if (memProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            allocated += budgets[i].usage;
        }
    }
    return allocated;
}

} // namespace pathways
