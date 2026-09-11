# Pathways Vulkan API Call Analysis & Specification Report

*Generated on 2026-09-10 12:44:45 UTC | Workload Filter: **all** | Platform: **All Devices (Global)** (2376 devices recorded) | Auditor: `scripts/audit_vulkan_api.py`*

## 1. Executive Summary & Repository Metrics

| Metric | Pathways Engine (`src/`, `tests/`) | Third-Party (`imgui`, `vma`) | Entire Repository Total |
|---|:---:|:---:|:---:|
| **Total API Invocations** | **699** | **320** | **1019** |
| **Unique Functions Called** | **106** | **92** | **134** |
| **Target Platform / Device Scope** | colspan=2 | **All Devices (Global)** (2376 devices recorded) |

## 2. Vulkan API Call Tally & Multi-Platform Specification Breakdown

| # | Vulkan API Call | Pathways Calls | Third-Party | Total | Specification & Date Added | Desktop Support | Mobile Support | Global Support | Architectural Domain |
|---|---|:---:|:---:|:---:|---|:---:|:---:|:---:|---|
| 1 | `vkCmdPipelineBarrier2` | **35** | 0 | 35 | Vulkan 1.3 Core (VK_KHR_synchronization2)<br>*(Added: 2020-08-31 (Ext) / 2022-01-25 (Core 1.3))* | 🟢 **92.5%** | 🟡 **68.4%** | 🟡 **77.4%** | Synchronization & Barriers |
| 2 | `vkGetDeviceProcAddr` | **30** | 3 | 33 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 3 | `vkBeginCommandBuffer` | **25** | 7 | 32 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 4 | `vkCmdBindDescriptorSets` | **25** | 5 | 30 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 5 | `vkEndCommandBuffer` | **24** | 7 | 31 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 6 | `vkCmdBindPipeline` | **24** | 1 | 25 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 7 | `vkCmdWriteTimestamp2` | **24** | 0 | 24 | Vulkan 1.3 Core (VK_KHR_synchronization2)<br>*(Added: 2020-08-31 (Ext) / 2022-01-25 (Core 1.3))* | 🟢 **92.5%** | 🟡 **68.4%** | 🟡 **77.4%** | Queries & Telemetry |
| 8 | `vkQueueSubmit` | **23** | 7 | 30 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 9 | `vkCmdPushConstants` | **22** | 1 | 23 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Compute Dispatch & Shaders |
| 10 | `vkDestroyPipeline` | **19** | 4 | 23 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 11 | `vkQueueWaitIdle` | **19** | 1 | 20 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 12 | `vkUpdateDescriptorSets` | **17** | 2 | 19 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 13 | `vkResetCommandBuffer` | **17** | 0 | 17 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 14 | `vkDestroyShaderModule` | **15** | 2 | 17 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 15 | `vkCmdDispatch` | **14** | 0 | 14 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Compute Dispatch & Shaders |
| 16 | `vkDeviceWaitIdle` | **13** | 8 | 21 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 17 | `vkDestroyDescriptorSetLayout` | **13** | 2 | 15 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 18 | `vkAllocateDescriptorSets` | **13** | 2 | 15 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 19 | `vkDestroyPipelineLayout` | **13** | 1 | 14 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 20 | `vkCreatePipelineLayout` | **12** | 1 | 13 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 21 | `vkCreateDescriptorSetLayout` | **12** | 1 | 13 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 22 | `vkGetPhysicalDeviceMemoryProperties` | **10** | 1 | 11 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 23 | `vkCreateComputePipelines` | **10** | 0 | 10 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Compute Dispatch & Shaders |
| 24 | `vkDestroyDevice` | **9** | 5 | 14 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 25 | `vkDestroyBuffer` | **8** | 9 | 17 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 26 | `vkFreeMemory` | **8** | 6 | 14 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 27 | `vkBindBufferMemory` | **8** | 4 | 12 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 28 | `vkAllocateMemory` | **8** | 4 | 12 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 29 | `vkCreateBuffer` | **8** | 3 | 11 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 30 | `vkAllocateCommandBuffers` | **8** | 3 | 11 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 31 | `vkEnumeratePhysicalDevices` | **8** | 2 | 10 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 32 | `vkDestroySemaphore` | **8** | 2 | 10 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 33 | `vkCreateSemaphore` | **8** | 2 | 10 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 34 | `vkCmdFillBuffer` | **8** | 0 | 8 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 35 | `vkCmdClearColorImage` | **8** | 0 | 8 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 36 | `vkDestroyInstance` | **7** | 5 | 12 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 37 | `vkCreateDevice` | **7** | 5 | 12 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 38 | `vkDestroyCommandPool` | **5** | 3 | 8 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 39 | `vkCreateCommandPool` | **5** | 3 | 8 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 40 | `vkGetQueryPoolResults` | **5** | 0 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Queries & Telemetry |
| 41 | `vkCmdCopyImageToBuffer` | **5** | 0 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 42 | `vkEnumerateDeviceExtensionProperties` | **4** | 11 | 15 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 43 | `vkWaitForFences` | **4** | 7 | 11 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 44 | `vkGetDeviceQueue` | **4** | 6 | 10 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 45 | `vkCreateInstance` | **4** | 5 | 9 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 46 | `vkGetPhysicalDeviceProperties` | **4** | 2 | 6 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 47 | `vkDestroySampler` | **4** | 2 | 6 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 48 | `vkCreateShaderModule` | **4** | 2 | 6 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 49 | `vkCmdCopyImage` | **4** | 2 | 6 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 50 | `vkGetMemoryHostPointerPropertiesEXT` | **4** | 0 | 4 | VK_EXT_external_memory_host<br>*(Added: 2018-01-17)* | 🟢 **84.9%** | 🔴 **6.5%** | 🔴 **36.1%** | Multi-GPU & External Interop |
| 51 | `vkDestroyIndirectExecutionSetEXT` | **4** | 0 | 4 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 52 | `vkCmdResetQueryPool` | **4** | 0 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Queries & Telemetry |
| 53 | `vkDestroyDescriptorPool` | **3** | 6 | 9 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 54 | `vkCreateDescriptorPool` | **3** | 6 | 9 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 55 | `vkFreeCommandBuffers` | **3** | 3 | 6 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 56 | `vkDestroyFence` | **3** | 2 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 57 | `vkCreateSampler` | **3** | 2 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 58 | `vkCreateFence` | **3** | 2 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 59 | `vkDestroyQueryPool` | **3** | 0 | 3 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Queries & Telemetry |
| 60 | `vkCreateQueryPool` | **3** | 0 | 3 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Queries & Telemetry |
| 61 | `vkCmdDispatchIndirect` | **3** | 0 | 3 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Compute Dispatch & Shaders |
| 62 | `vkGetInstanceProcAddr` | **2** | 11 | 13 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 63 | `vkEnumerateInstanceExtensionProperties` | **2** | 10 | 12 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 64 | `vkGetBufferMemoryRequirements` | **2** | 8 | 10 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 65 | `vkResetFences` | **2** | 6 | 8 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 66 | `vkGetSwapchainImagesKHR` | **2** | 2 | 4 | VK_KHR_swapchain<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **99.9%** | 🟢 **99.9%** | Window System & Presentation |
| 67 | `vkGetPhysicalDeviceSurfacePresentModesKHR` | **2** | 2 | 4 | VK_KHR_surface<br>*(Added: 2016-02-16)* | 🟢 **91.7%** | 🟢 **97.9%** | 🟢 **96.0%** | Window System & Presentation |
| 68 | `vkGetPhysicalDeviceSurfaceFormatsKHR` | **2** | 2 | 4 | VK_KHR_surface<br>*(Added: 2016-02-16)* | 🟢 **91.7%** | 🟢 **97.9%** | 🟢 **96.0%** | Window System & Presentation |
| 69 | `vkGetPhysicalDeviceQueueFamilyProperties` | **2** | 2 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 70 | `vkDestroyImageView` | **2** | 2 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 71 | `vkCreateImageView` | **2** | 2 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 72 | `vkUpdateIndirectExecutionSetPipelineEXT` | **2** | 0 | 2 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 73 | `vkImportSemaphoreFdKHR` | **2** | 0 | 2 | VK_KHR_external_semaphore_fd<br>*(Added: 2017-04-24)* | 🟡 **42.7%** | 🟢 **99.7%** | 🟡 **79.8%** | Multi-GPU & External Interop |
| 74 | `vkGetSemaphoreFdKHR` | **2** | 0 | 2 | VK_KHR_external_semaphore_fd<br>*(Added: 2017-04-24)* | 🟡 **42.7%** | 🟢 **99.7%** | 🟡 **79.8%** | Multi-GPU & External Interop |
| 75 | `vkGetMemoryFdPropertiesKHR` | **2** | 0 | 2 | VK_KHR_external_memory_fd<br>*(Added: 2017-04-24)* | 🟡 **44.2%** | 🟢 **93.1%** | 🟡 **76.0%** | Multi-GPU & External Interop |
| 76 | `vkGetMemoryFdKHR` | **2** | 0 | 2 | VK_KHR_external_memory_fd<br>*(Added: 2017-04-24)* | 🟡 **44.2%** | 🟢 **93.1%** | 🟡 **76.0%** | Multi-GPU & External Interop |
| 77 | `vkGetAccelerationStructureDeviceAddressKHR` | **2** | 0 | 2 | VK_KHR_acceleration_structure<br>*(Added: 2020-11-20)* | 🟡 **52.9%** | 🔴 **26.6%** | 🔴 **34.1%** | Hardware Ray Tracing |
| 78 | `vkGetAccelerationStructureBuildSizesKHR` | **2** | 0 | 2 | VK_KHR_acceleration_structure<br>*(Added: 2020-11-20)* | 🟡 **52.9%** | 🔴 **26.6%** | 🔴 **34.1%** | Hardware Ray Tracing |
| 79 | `vkDestroyIndirectCommandsLayoutEXT` | **2** | 0 | 2 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 80 | `vkCreateIndirectExecutionSetEXT` | **2** | 0 | 2 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 81 | `vkCreateIndirectCommandsLayoutEXT` | **2** | 0 | 2 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 82 | `vkCreateAccelerationStructureKHR` | **2** | 0 | 2 | VK_KHR_acceleration_structure<br>*(Added: 2020-11-20)* | 🟡 **52.9%** | 🔴 **26.6%** | 🔴 **34.1%** | Hardware Ray Tracing |
| 83 | `vkCmdExecuteGeneratedCommandsEXT` | **2** | 0 | 2 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 84 | `vkCmdBuildAccelerationStructuresKHR` | **2** | 0 | 2 | VK_KHR_acceleration_structure<br>*(Added: 2020-11-20)* | 🟡 **52.9%** | 🔴 **26.6%** | 🔴 **34.1%** | Hardware Ray Tracing |
| 85 | `vkQueuePresentKHR` | **1** | 5 | 6 | VK_KHR_swapchain<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **99.9%** | 🟢 **99.9%** | Window System & Presentation |
| 86 | `vkDestroySurfaceKHR` | **1** | 5 | 6 | VK_KHR_surface<br>*(Added: 2016-02-16)* | 🟢 **91.7%** | 🟢 **97.9%** | 🟢 **96.0%** | Window System & Presentation |
| 87 | `vkAcquireNextImageKHR` | **1** | 5 | 6 | VK_KHR_swapchain<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **99.9%** | 🟢 **99.9%** | Window System & Presentation |
| 88 | `vkDestroySwapchainKHR` | **1** | 2 | 3 | VK_KHR_swapchain<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **99.9%** | 🟢 **99.9%** | Window System & Presentation |
| 89 | `vkCmdCopyBuffer` | **1** | 2 | 3 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 90 | `vkGetPhysicalDeviceSurfaceCapabilitiesKHR` | **1** | 1 | 2 | VK_KHR_surface<br>*(Added: 2016-02-16)* | 🟢 **91.7%** | 🟢 **97.9%** | 🟢 **96.0%** | Window System & Presentation |
| 91 | `vkCreateSwapchainKHR` | **1** | 1 | 2 | VK_KHR_swapchain<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **99.9%** | 🟢 **99.9%** | Window System & Presentation |
| 92 | `vkCmdCopyBufferToImage` | **1** | 1 | 2 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 93 | `vkGetRayTracingShaderGroupHandlesKHR` | **1** | 0 | 1 | VK_KHR_ray_tracing_pipeline<br>*(Added: 2020-11-20)* | 🟡 **51.3%** | 🔴 **7.6%** | 🔴 **21.8%** | Hardware Ray Tracing |
| 94 | `vkGetPhysicalDeviceProperties2` | **1** | 0 | 1 | Vulkan 1.1 Core (VK_KHR_get_physical_device_properties2)<br>*(Added: 2017-03-13 (Ext) / 2018-03-07 (Core 1.1))* | 🟢 **90.5%** | 🟢 **97.0%** | 🟢 **95.0%** | Core Device & Instance |
| 95 | `vkGetGeneratedCommandsMemoryRequirementsEXT` | **1** | 0 | 1 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 96 | `vkGetBufferDeviceAddress` | **1** | 0 | 1 | Vulkan 1.2 Core (VK_KHR_buffer_device_address)<br>*(Added: 2019-06-24 (Ext) / 2020-01-15 (Core 1.2))* | 🟢 **92.1%** | 🟢 **91.4%** | 🟢 **91.0%** | Memory Management |
| 97 | `vkDestroyDebugUtilsMessengerEXT` | **1** | 0 | 1 | VK_EXT_debug_utils<br>*(Added: 2017-10-05 (Rev 1) / 2018-05-18)* | 🟢 **89.7%** | 🔴 **36.5%** | 🟡 **57.7%** | Core Device & Instance |
| 98 | `vkDestroyAccelerationStructureKHR` | **1** | 0 | 1 | VK_KHR_acceleration_structure<br>*(Added: 2020-11-20)* | 🟡 **52.9%** | 🔴 **26.6%** | 🔴 **34.1%** | Hardware Ray Tracing |
| 99 | `vkCreateRayTracingPipelinesKHR` | **1** | 0 | 1 | VK_KHR_ray_tracing_pipeline<br>*(Added: 2020-11-20)* | 🟡 **51.3%** | 🔴 **7.6%** | 🔴 **21.8%** | Hardware Ray Tracing |
| 100 | `vkCreateDebugUtilsMessengerEXT` | **1** | 0 | 1 | VK_EXT_debug_utils<br>*(Added: 2017-10-05 (Rev 1) / 2018-05-18)* | 🟢 **89.7%** | 🔴 **36.5%** | 🟡 **57.7%** | Core Device & Instance |
| 101 | `vkCmdTraceRaysKHR` | **1** | 0 | 1 | VK_KHR_ray_tracing_pipeline<br>*(Added: 2020-11-20)* | 🟡 **51.3%** | 🔴 **7.6%** | 🔴 **21.8%** | Hardware Ray Tracing |
| 102 | `vkCmdTraceRaysIndirectKHR` | **1** | 0 | 1 | VK_KHR_ray_tracing_pipeline<br>*(Added: 2020-11-20)* | 🟡 **51.3%** | 🔴 **7.6%** | 🔴 **21.8%** | Hardware Ray Tracing |
| 103 | `vkCmdPreprocessGeneratedCommandsEXT` | **1** | 0 | 1 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 104 | `vkCmdEndRendering` | **1** | 0 | 1 | Vulkan 1.3 Core (VK_KHR_dynamic_rendering)<br>*(Added: 2021-10-19 (Ext) / 2022-01-25 (Core 1.3))* | 🟢 **88.1%** | 🟡 **55.6%** | 🟡 **67.9%** | Raster & Dynamic Rendering |
| 105 | `vkCmdBlitImage` | **1** | 0 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 106 | `vkCmdBeginRendering` | **1** | 0 | 1 | Vulkan 1.3 Core (VK_KHR_dynamic_rendering)<br>*(Added: 2021-10-19 (Ext) / 2022-01-25 (Core 1.3))* | 🟢 **88.1%** | 🟡 **55.6%** | 🟡 **67.9%** | Raster & Dynamic Rendering |
| 107 | `vkResetCommandPool` | 0 | 7 | 7 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 108 | `vkMapMemory` | 0 | 7 | 7 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 109 | `vkDestroyImage` | 0 | 6 | 6 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 110 | `vkCmdPipelineBarrier` | 0 | 6 | 6 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 111 | `vkGetPhysicalDeviceSurfaceSupportKHR` | 0 | 5 | 5 | VK_KHR_surface<br>*(Added: 2016-02-16)* | 🟢 **91.7%** | 🟢 **97.9%** | 🟢 **96.0%** | Window System & Presentation |
| 112 | `vkFlushMappedMemoryRanges` | 0 | 5 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 113 | `vkCreateImage` | 0 | 5 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 114 | `vkCmdEndRenderPass` | 0 | 5 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 115 | `vkCmdBeginRenderPass` | 0 | 5 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 116 | `vkUnmapMemory` | 0 | 4 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 117 | `vkGetImageMemoryRequirements` | 0 | 4 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 118 | `vkInvalidateMappedMemoryRanges` | 0 | 3 | 3 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 119 | `vkDestroyRenderPass` | 0 | 2 | 2 | Vulkan Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 120 | `vkCmdSetScissor` | 0 | 2 | 2 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 121 | `vkBindImageMemory` | 0 | 2 | 2 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 122 | `vkGetPhysicalDeviceMemoryProperties2KHR` | 0 | 1 | 1 | VK_KHR_get_physical_device_properties2 / Core 1.1<br>*(Added: 2017-03-13 (Ext) / 2018-03-07 (Core 1.1))* | 🟢 **90.5%** | 🟢 **97.0%** | 🟢 **95.0%** | Memory Management |
| 123 | `vkGetBufferMemoryRequirements2KHR` | 0 | 1 | 1 | VK_KHR_get_memory_requirements2 / Core 1.1<br>*(Added: 2017-09-05 (Ext) / 2018-03-07 (Core 1.1))* | 🟢 **98.9%** | 🟢 **99.7%** | 🟢 **99.4%** | Memory Management |
| 124 | `vkFreeDescriptorSets` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 125 | `vkDestroyFramebuffer` | 0 | 1 | 1 | Vulkan Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 126 | `vkCreateWin32SurfaceKHR` | 0 | 1 | 1 | VK_KHR_win32_surface<br>*(Added: 2016-02-16)* | 🟡 **53.5%** | 🔴 **0.6%** | 🔴 **19.8%** | Window System & Presentation |
| 127 | `vkCreateScreenSurfaceQNX` | 0 | 1 | 1 | VK_QNX_screen_surface<br>*(Added: 2021-01-13)* | 🔴 **0.0%** | 🔴 **0.0%** | 🔴 **0.0%** | Window System & Presentation |
| 128 | `vkCreateRenderPass` | 0 | 1 | 1 | Vulkan Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 129 | `vkCreateGraphicsPipelines` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 130 | `vkCreateFramebuffer` | 0 | 1 | 1 | Vulkan Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 131 | `vkCmdSetViewport` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 132 | `vkCmdDrawIndexed` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 133 | `vkCmdBindVertexBuffers` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 134 | `vkCmdBindIndexBuffer` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
