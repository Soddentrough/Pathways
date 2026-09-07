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
    uint32_t getValidationErrors() const { return s_validationErrors; }

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
    std::string m_deviceName;
    GpuArchitecture m_architecture = GpuArchitecture::Generic;
    bool m_isRDNA4 = false;
    bool m_hasDGC = false;
    bool m_hasRayTracing = false;
    bool m_hasSubgroupSizeControl = false;

    static uint32_t s_validationErrors;
    static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(
        VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
        VkDebugUtilsMessageTypeFlagsEXT messageType,
        const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
        void* pUserData);
};

} // namespace pathways
