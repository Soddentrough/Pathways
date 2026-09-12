#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "core/Config.hpp"
#include <vector>
#include <string>
#include <memory>

namespace pathways {

struct QueueFamilyIndices {
    uint32_t graphicsComputeFamily = UINT32_MAX;
    uint32_t transferFamily = UINT32_MAX;

    bool isComplete() const {
        return graphicsComputeFamily != UINT32_MAX;
    }
};

enum class GpuArchitecture {
    Generic,
    AmdRDNA1,
    AmdRDNA2,
    AmdRDNA3,
    AmdRDNA3_5,
    AmdRDNA4,
    NvidiaTuring,
    NvidiaAmpere,
    NvidiaAda,
    NvidiaBlackwell,
    IntelArc
};

struct PciLinkInfo {
    bool valid = false;
    uint32_t domain = 0;
    uint32_t bus = 0;
    uint32_t device = 0;
    uint32_t function = 0;
    std::string bdfString = "";
    std::string currentSpeed = "";
    uint32_t currentWidth = 0;
    std::string maxSpeed = "";
    uint32_t maxWidth = 0;
    std::string generationName = "";
    std::string maxGenerationName = "";
    std::string formattedLink = "PCIe N/A";
    std::string hwmonPath = "";
    bool isDegraded = false;
    std::string degradationReason = "";
};

class VulkanContext {
public:
    VulkanContext(const Config& config, VkSurfaceKHR surface = VK_NULL_HANDLE);
    ~VulkanContext();

    VkInstance getInstance() const { return m_instance; }
    VkPhysicalDevice getPhysicalDevice() const { return m_physicalDevice; }
    VkDevice getDevice() const { return m_device; }
    VmaAllocator getAllocator() const { return m_allocator; }
    VkQueue getGraphicsQueue() const { return m_graphicsQueue; }
    VkQueue getTransferQueue() const { return m_transferQueue; }
    uint32_t getGraphicsQueueFamily() const { return m_queueIndices.graphicsComputeFamily; }
    uint32_t getTransferQueueFamily() const { return m_queueIndices.transferFamily; }

    const VkPhysicalDeviceProperties& getDeviceProperties() const { return m_deviceProperties; }
    const std::string& getDeviceName() const { return m_deviceName; }
    uint32_t getVendorID() const { return m_deviceProperties.vendorID; }
    uint32_t getDeviceID() const { return m_deviceProperties.deviceID; }
    uint32_t getDriverVersion() const { return m_deviceProperties.driverVersion; }
    uint32_t getApiVersion() const { return m_deviceProperties.apiVersion; }
    VkPhysicalDeviceType getDeviceType() const { return m_deviceProperties.deviceType; }
    uint64_t getTotalVramBytes() const;
    uint64_t getAllocatedVramBytes() const;
    GpuArchitecture getArchitecture() const { return m_architecture; }
    std::string getArchitectureName() const;
    std::string getShortArchName() const;
    std::string getRayAcceleratorName() const;
    bool isRDNA() const;
    bool isRDNA3() const { return m_architecture == GpuArchitecture::AmdRDNA3; }
    bool isRDNA4() const { return m_architecture == GpuArchitecture::AmdRDNA4; }
    bool hasDGC() const { return m_hasDGC; }
    bool hasRayTracing() const { return m_hasRayTracing; }
    bool hasSubgroupSizeControl() const { return m_hasSubgroupSizeControl; }
    bool hasExternalMemoryHost() const { return m_hasExternalMemoryHost; }
    bool hasExternalMemoryFd() const { return m_hasExternalMemoryFd; }
    bool hasExternalMemoryDmaBuf() const { return m_hasExternalMemoryDmaBuf; }
    bool hasExternalSemaphoreFd() const { return m_hasExternalSemaphoreFd; }
    bool hasCooperativeMatrix() const { return m_hasCooperativeMatrix; }
    PFN_vkGetSemaphoreFdKHR pfnGetSemaphoreFdKHR = nullptr;
    PFN_vkImportSemaphoreFdKHR pfnImportSemaphoreFdKHR = nullptr;
    PFN_vkGetMemoryFdKHR pfnGetMemoryFdKHR = nullptr;
    PFN_vkGetMemoryFdPropertiesKHR pfnGetMemoryFdPropertiesKHR = nullptr;
    const VkPhysicalDeviceRayTracingPipelinePropertiesKHR& getRayTracingPipelineProperties() const { return m_rtPipelineProperties; }
    uint32_t getValidationErrors() const { return s_validationErrors; }
    const PciLinkInfo& getPciLinkInfo() const { return m_pciLinkInfo; }
    const std::string& getPciLinkString() const { return m_pciLinkInfo.formattedLink; }
    bool isPciLinkDegraded() const { return m_pciLinkInfo.isDegraded; }
    void refreshPciLinkInfo();

    // Physical devices enumeration (for multi-GPU)
    static std::vector<VkPhysicalDevice> enumeratePhysicalDevices(VkInstance instance);

private:
    void createInstance(const Config& config);
    void setupDebugMessenger();
    void selectPhysicalDevice(const Config& config, VkSurfaceKHR surface);
    void createLogicalDevice(const Config& config);
    void initVMA();

    VkInstance m_instance = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT m_debugMessenger = VK_NULL_HANDLE;
    VkPhysicalDevice m_physicalDevice = VK_NULL_HANDLE;
    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;

    VkQueue m_graphicsQueue = VK_NULL_HANDLE;
    VkQueue m_transferQueue = VK_NULL_HANDLE;
    QueueFamilyIndices m_queueIndices;

    VkPhysicalDeviceProperties m_deviceProperties{};
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR m_rtPipelineProperties{};
    std::string m_deviceName;
    GpuArchitecture m_architecture = GpuArchitecture::Generic;
    bool m_isRDNA4 = false;
    bool m_hasDGC = false;
    bool m_hasRayTracing = false;
    bool m_hasSubgroupSizeControl = false;
    bool m_hasExternalMemoryHost = false;
    bool m_hasExternalMemoryFd = false;
    bool m_hasExternalMemoryDmaBuf = false;
    bool m_hasExternalSemaphoreFd = false;
    bool m_hasCooperativeMatrix = false;
    PciLinkInfo m_pciLinkInfo;

    static uint32_t s_validationErrors;
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
};

} // namespace pathways
