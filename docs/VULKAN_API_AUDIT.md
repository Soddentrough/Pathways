# Pathways Vulkan API Call Analysis & Specification Report

*Generated on 2026-09-21 01:39:41 UTC | Workload Filter: **all** | Platform: **All Devices (Global)** (2376 devices recorded) | Auditor: `scripts/audit_vulkan_api.py`*

## 1. Executive Summary & Repository Metrics

| Metric | Pathways Engine (`src/`, `tests/`) | Third-Party (`imgui`, `vma`) | Entire Repository Total |
|---|:---:|:---:|:---:|
| **Total API Invocations** | **888** | **320** | **1208** |
| **Unique Functions Called** | **107** | **92** | **136** |
| **Target Platform / Device Scope** | colspan=2 | **All Devices (Global)** (2376 devices recorded) |

## 2. Vulkan API Call Tally & Multi-Platform Specification Breakdown

| # | Vulkan API Call | Pathways Calls | Third-Party | Total | Specification & Date Added | Desktop Support | Mobile Support | Global Support | Architectural Domain |
|---|---|:---:|:---:|:---:|---|:---:|:---:|:---:|---|
| 1 | `vkCmdPipelineBarrier2` | **52** | 0 | 52 | Vulkan 1.3 Core (VK_KHR_synchronization2)<br>*(Added: 2020-08-31 (Ext) / 2022-01-25 (Core 1.3))* | 🟢 **92.5%** | 🟡 **68.4%** | 🟡 **77.4%** | Synchronization & Barriers |
| 2 | `vkGetDeviceProcAddr` | **35** | 3 | 38 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 3 | `vkDestroyShaderModule` | **33** | 2 | 35 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 4 | `vkCmdBindPipeline` | **31** | 1 | 32 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 5 | `vkCmdBindDescriptorSets` | **30** | 5 | 35 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 6 | `vkBeginCommandBuffer` | **29** | 7 | 36 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 7 | `vkDestroyPipeline` | **29** | 4 | 33 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 8 | `vkCmdWriteTimestamp2` | **29** | 0 | 29 | Vulkan 1.3 Core (VK_KHR_synchronization2)<br>*(Added: 2020-08-31 (Ext) / 2022-01-25 (Core 1.3))* | 🟢 **92.5%** | 🟡 **68.4%** | 🟡 **77.4%** | Queries & Telemetry |
| 9 | `vkEndCommandBuffer` | **28** | 7 | 35 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 10 | `vkCmdPushConstants` | **26** | 1 | 27 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Compute Dispatch & Shaders |
| 11 | `vkQueueSubmit2` | **26** | 0 | 26 | Vulkan Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 12 | `vkUpdateDescriptorSets` | **25** | 2 | 27 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 13 | `vkQueueWaitIdle` | **22** | 1 | 23 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 14 | `vkAllocateDescriptorSets` | **20** | 2 | 22 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 15 | `vkCmdDispatch` | **19** | 0 | 19 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Compute Dispatch & Shaders |
| 16 | `vkDestroyDescriptorSetLayout` | **18** | 2 | 20 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 17 | `vkDestroyPipelineLayout` | **18** | 1 | 19 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 18 | `vkCreatePipelineLayout` | **18** | 1 | 19 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 19 | `vkCreateDescriptorSetLayout` | **18** | 1 | 19 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 20 | `vkResetCommandBuffer` | **17** | 0 | 17 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 21 | `vkCmdClearColorImage` | **17** | 0 | 17 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 22 | `vkCreateComputePipelines` | **16** | 0 | 16 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Compute Dispatch & Shaders |
| 23 | `vkDeviceWaitIdle` | **15** | 8 | 23 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 24 | `vkAllocateCommandBuffers` | **12** | 3 | 15 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 25 | `vkDestroyDescriptorPool` | **11** | 6 | 17 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 26 | `vkCreateDescriptorPool` | **11** | 6 | 17 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 27 | `vkGetPhysicalDeviceMemoryProperties` | **10** | 1 | 11 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 28 | `vkDestroyDevice` | **9** | 5 | 14 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 29 | `vkCmdCopyImageToBuffer` | **9** | 0 | 9 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 30 | `vkDestroyBuffer` | **8** | 9 | 17 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 31 | `vkFreeMemory` | **8** | 6 | 14 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 32 | `vkDestroyInstance` | **8** | 5 | 13 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 33 | `vkBindBufferMemory` | **8** | 4 | 12 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 34 | `vkAllocateMemory` | **8** | 4 | 12 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 35 | `vkCreateBuffer` | **8** | 3 | 11 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 36 | `vkEnumeratePhysicalDevices` | **8** | 2 | 10 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 37 | `vkCreateShaderModule` | **8** | 2 | 10 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 38 | `vkCreateSemaphore` | **8** | 2 | 10 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 39 | `vkCreateDevice` | **7** | 5 | 12 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 40 | `vkFreeCommandBuffers` | **7** | 3 | 10 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 41 | `vkDestroySemaphore` | **7** | 2 | 9 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 42 | `vkCmdFillBuffer` | **7** | 0 | 7 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 43 | `vkGetQueryPoolResults` | **6** | 0 | 6 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Queries & Telemetry |
| 44 | `vkGetDeviceQueue` | **5** | 6 | 11 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 45 | `vkDestroyCommandPool` | **5** | 3 | 8 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 46 | `vkCreateCommandPool` | **5** | 3 | 8 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 47 | `vkCmdCopyBuffer` | **5** | 2 | 7 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 48 | `vkDestroyIndirectExecutionSetEXT` | **5** | 0 | 5 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 49 | `vkCmdResetQueryPool` | **5** | 0 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Queries & Telemetry |
| 50 | `vkEnumerateDeviceExtensionProperties` | **4** | 11 | 15 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 51 | `vkWaitForFences` | **4** | 7 | 11 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 52 | `vkCreateInstance` | **4** | 5 | 9 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 53 | `vkGetPhysicalDeviceProperties` | **4** | 2 | 6 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 54 | `vkGetMemoryHostPointerPropertiesEXT` | **4** | 0 | 4 | VK_EXT_external_memory_host<br>*(Added: 2018-01-17)* | 🟢 **84.9%** | 🔴 **6.5%** | 🔴 **36.1%** | Multi-GPU & External Interop |
| 55 | `vkDestroyQueryPool` | **4** | 0 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Queries & Telemetry |
| 56 | `vkCreateQueryPool` | **4** | 0 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Queries & Telemetry |
| 57 | `vkCreateAccelerationStructureKHR` | **4** | 0 | 4 | VK_KHR_acceleration_structure<br>*(Added: 2020-11-20)* | 🟡 **52.9%** | 🔴 **26.6%** | 🔴 **34.1%** | Hardware Ray Tracing |
| 58 | `vkDestroyFence` | **3** | 2 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 59 | `vkCreateFence` | **3** | 2 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 60 | `vkCmdCopyImage` | **3** | 2 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 61 | `vkGetPhysicalDeviceProperties2` | **3** | 0 | 3 | Vulkan 1.1 Core (VK_KHR_get_physical_device_properties2)<br>*(Added: 2017-03-13 (Ext) / 2018-03-07 (Core 1.1))* | 🟢 **90.5%** | 🟢 **97.0%** | 🟢 **95.0%** | Core Device & Instance |
| 62 | `vkGetAccelerationStructureDeviceAddressKHR` | **3** | 0 | 3 | VK_KHR_acceleration_structure<br>*(Added: 2020-11-20)* | 🟡 **52.9%** | 🔴 **26.6%** | 🔴 **34.1%** | Hardware Ray Tracing |
| 63 | `vkGetAccelerationStructureBuildSizesKHR` | **3** | 0 | 3 | VK_KHR_acceleration_structure<br>*(Added: 2020-11-20)* | 🟡 **52.9%** | 🔴 **26.6%** | 🔴 **34.1%** | Hardware Ray Tracing |
| 64 | `vkCmdDispatchIndirect` | **3** | 0 | 3 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Compute Dispatch & Shaders |
| 65 | `vkCmdBuildAccelerationStructuresKHR` | **3** | 0 | 3 | VK_KHR_acceleration_structure<br>*(Added: 2020-11-20)* | 🟡 **52.9%** | 🔴 **26.6%** | 🔴 **34.1%** | Hardware Ray Tracing |
| 66 | `vkGetInstanceProcAddr` | **2** | 11 | 13 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 67 | `vkEnumerateInstanceExtensionProperties` | **2** | 10 | 12 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 68 | `vkResetFences` | **2** | 6 | 8 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 69 | `vkGetPhysicalDeviceSurfaceSupportKHR` | **2** | 5 | 7 | VK_KHR_surface<br>*(Added: 2016-02-16)* | 🟢 **91.7%** | 🟢 **97.9%** | 🟢 **96.0%** | Window System & Presentation |
| 70 | `vkGetSwapchainImagesKHR` | **2** | 2 | 4 | VK_KHR_swapchain<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **99.9%** | 🟢 **99.9%** | Window System & Presentation |
| 71 | `vkGetPhysicalDeviceSurfacePresentModesKHR` | **2** | 2 | 4 | VK_KHR_surface<br>*(Added: 2016-02-16)* | 🟢 **91.7%** | 🟢 **97.9%** | 🟢 **96.0%** | Window System & Presentation |
| 72 | `vkGetPhysicalDeviceSurfaceFormatsKHR` | **2** | 2 | 4 | VK_KHR_surface<br>*(Added: 2016-02-16)* | 🟢 **91.7%** | 🟢 **97.9%** | 🟢 **96.0%** | Window System & Presentation |
| 73 | `vkGetPhysicalDeviceQueueFamilyProperties` | **2** | 2 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 74 | `vkDestroyImageView` | **2** | 2 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 75 | `vkCreateImageView` | **2** | 2 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 76 | `vkCmdCopyBufferToImage` | **2** | 1 | 3 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 77 | `vkImportSemaphoreFdKHR` | **2** | 0 | 2 | VK_KHR_external_semaphore_fd<br>*(Added: 2017-04-24)* | 🟡 **42.7%** | 🟢 **99.7%** | 🟡 **79.8%** | Multi-GPU & External Interop |
| 78 | `vkGetSemaphoreFdKHR` | **2** | 0 | 2 | VK_KHR_external_semaphore_fd<br>*(Added: 2017-04-24)* | 🟡 **42.7%** | 🟢 **99.7%** | 🟡 **79.8%** | Multi-GPU & External Interop |
| 79 | `vkGetMemoryFdPropertiesKHR` | **2** | 0 | 2 | VK_KHR_external_memory_fd<br>*(Added: 2017-04-24)* | 🟡 **44.2%** | 🟢 **93.1%** | 🟡 **76.0%** | Multi-GPU & External Interop |
| 80 | `vkGetMemoryFdKHR` | **2** | 0 | 2 | VK_KHR_external_memory_fd<br>*(Added: 2017-04-24)* | 🟡 **44.2%** | 🟢 **93.1%** | 🟡 **76.0%** | Multi-GPU & External Interop |
| 81 | `vkGetBufferMemoryRequirements2` | **2** | 0 | 2 | Vulkan Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 82 | `vkDestroyIndirectCommandsLayoutEXT` | **2** | 0 | 2 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 83 | `vkDestroyAccelerationStructureKHR` | **2** | 0 | 2 | VK_KHR_acceleration_structure<br>*(Added: 2020-11-20)* | 🟡 **52.9%** | 🔴 **26.6%** | 🔴 **34.1%** | Hardware Ray Tracing |
| 84 | `vkCreateIndirectCommandsLayoutEXT` | **2** | 0 | 2 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 85 | `vkCmdPreprocessGeneratedCommandsEXT` | **2** | 0 | 2 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 86 | `vkCmdExecuteGeneratedCommandsEXT` | **2** | 0 | 2 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 87 | `vkQueuePresentKHR` | **1** | 5 | 6 | VK_KHR_swapchain<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **99.9%** | 🟢 **99.9%** | Window System & Presentation |
| 88 | `vkDestroySurfaceKHR` | **1** | 5 | 6 | VK_KHR_surface<br>*(Added: 2016-02-16)* | 🟢 **91.7%** | 🟢 **97.9%** | 🟢 **96.0%** | Window System & Presentation |
| 89 | `vkAcquireNextImageKHR` | **1** | 5 | 6 | VK_KHR_swapchain<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **99.9%** | 🟢 **99.9%** | Window System & Presentation |
| 90 | `vkDestroySwapchainKHR` | **1** | 2 | 3 | VK_KHR_swapchain<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **99.9%** | 🟢 **99.9%** | Window System & Presentation |
| 91 | `vkDestroySampler` | **1** | 2 | 3 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 92 | `vkCreateSampler` | **1** | 2 | 3 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 93 | `vkGetPhysicalDeviceSurfaceCapabilitiesKHR` | **1** | 1 | 2 | VK_KHR_surface<br>*(Added: 2016-02-16)* | 🟢 **91.7%** | 🟢 **97.9%** | 🟢 **96.0%** | Window System & Presentation |
| 94 | `vkCreateSwapchainKHR` | **1** | 1 | 2 | VK_KHR_swapchain<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **99.9%** | 🟢 **99.9%** | Window System & Presentation |
| 95 | `vkUpdateIndirectExecutionSetPipelineEXT` | **1** | 0 | 1 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 96 | `vkGetRayTracingShaderGroupHandlesKHR` | **1** | 0 | 1 | VK_KHR_ray_tracing_pipeline<br>*(Added: 2020-11-20)* | 🟡 **51.3%** | 🔴 **7.6%** | 🔴 **21.8%** | Hardware Ray Tracing |
| 97 | `vkGetGeneratedCommandsMemoryRequirementsEXT` | **1** | 0 | 1 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 98 | `vkGetBufferDeviceAddress` | **1** | 0 | 1 | Vulkan 1.2 Core (VK_KHR_buffer_device_address)<br>*(Added: 2019-06-24 (Ext) / 2020-01-15 (Core 1.2))* | 🟢 **92.1%** | 🟢 **91.4%** | 🟢 **91.0%** | Memory Management |
| 99 | `vkDestroyDebugUtilsMessengerEXT` | **1** | 0 | 1 | VK_EXT_debug_utils<br>*(Added: 2017-10-05 (Rev 1) / 2018-05-18)* | 🟢 **89.7%** | 🔴 **36.5%** | 🟡 **57.7%** | Core Device & Instance |
| 100 | `vkCreateRayTracingPipelinesKHR` | **1** | 0 | 1 | VK_KHR_ray_tracing_pipeline<br>*(Added: 2020-11-20)* | 🟡 **51.3%** | 🔴 **7.6%** | 🔴 **21.8%** | Hardware Ray Tracing |
| 101 | `vkCreateIndirectExecutionSetEXT` | **1** | 0 | 1 | VK_EXT_device_generated_commands<br>*(Added: 2023-12-14)* | 🟡 **49.3%** | 🔴 **0.1%** | 🔴 **16.8%** | Device Generated Commands |
| 102 | `vkCreateDebugUtilsMessengerEXT` | **1** | 0 | 1 | VK_EXT_debug_utils<br>*(Added: 2017-10-05 (Rev 1) / 2018-05-18)* | 🟢 **89.7%** | 🔴 **36.5%** | 🟡 **57.7%** | Core Device & Instance |
| 103 | `vkCmdTraceRaysKHR` | **1** | 0 | 1 | VK_KHR_ray_tracing_pipeline<br>*(Added: 2020-11-20)* | 🟡 **51.3%** | 🔴 **7.6%** | 🔴 **21.8%** | Hardware Ray Tracing |
| 104 | `vkCmdTraceRaysIndirectKHR` | **1** | 0 | 1 | VK_KHR_ray_tracing_pipeline<br>*(Added: 2020-11-20)* | 🟡 **51.3%** | 🔴 **7.6%** | 🔴 **21.8%** | Hardware Ray Tracing |
| 105 | `vkCmdEndRendering` | **1** | 0 | 1 | Vulkan 1.3 Core (VK_KHR_dynamic_rendering)<br>*(Added: 2021-10-19 (Ext) / 2022-01-25 (Core 1.3))* | 🟢 **88.1%** | 🟡 **55.6%** | 🟡 **67.9%** | Raster & Dynamic Rendering |
| 106 | `vkCmdBlitImage` | **1** | 0 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Transfer & Copies |
| 107 | `vkCmdBeginRendering` | **1** | 0 | 1 | Vulkan 1.3 Core (VK_KHR_dynamic_rendering)<br>*(Added: 2021-10-19 (Ext) / 2022-01-25 (Core 1.3))* | 🟢 **88.1%** | 🟡 **55.6%** | 🟡 **67.9%** | Raster & Dynamic Rendering |
| 108 | `vkGetBufferMemoryRequirements` | 0 | 8 | 8 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 109 | `vkResetCommandPool` | 0 | 7 | 7 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Core Device & Instance |
| 110 | `vkQueueSubmit` | 0 | 7 | 7 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 111 | `vkMapMemory` | 0 | 7 | 7 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 112 | `vkDestroyImage` | 0 | 6 | 6 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 113 | `vkCmdPipelineBarrier` | 0 | 6 | 6 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Synchronization & Barriers |
| 114 | `vkFlushMappedMemoryRanges` | 0 | 5 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 115 | `vkCreateImage` | 0 | 5 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 116 | `vkCmdEndRenderPass` | 0 | 5 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 117 | `vkCmdBeginRenderPass` | 0 | 5 | 5 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 118 | `vkUnmapMemory` | 0 | 4 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 119 | `vkGetImageMemoryRequirements` | 0 | 4 | 4 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 120 | `vkInvalidateMappedMemoryRanges` | 0 | 3 | 3 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 121 | `vkDestroyRenderPass` | 0 | 2 | 2 | Vulkan Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 122 | `vkCmdSetScissor` | 0 | 2 | 2 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 123 | `vkBindImageMemory` | 0 | 2 | 2 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Memory Management |
| 124 | `vkGetPhysicalDeviceMemoryProperties2KHR` | 0 | 1 | 1 | VK_KHR_get_physical_device_properties2 / Core 1.1<br>*(Added: 2017-03-13 (Ext) / 2018-03-07 (Core 1.1))* | 🟢 **90.5%** | 🟢 **97.0%** | 🟢 **95.0%** | Memory Management |
| 125 | `vkGetBufferMemoryRequirements2KHR` | 0 | 1 | 1 | VK_KHR_get_memory_requirements2 / Core 1.1<br>*(Added: 2017-09-05 (Ext) / 2018-03-07 (Core 1.1))* | 🟢 **98.9%** | 🟢 **99.7%** | 🟢 **99.4%** | Memory Management |
| 126 | `vkFreeDescriptorSets` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Descriptors & Resource Binding |
| 127 | `vkDestroyFramebuffer` | 0 | 1 | 1 | Vulkan Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 128 | `vkCreateWin32SurfaceKHR` | 0 | 1 | 1 | VK_KHR_win32_surface<br>*(Added: 2016-02-16)* | 🟡 **53.5%** | 🔴 **0.6%** | 🔴 **19.8%** | Window System & Presentation |
| 129 | `vkCreateScreenSurfaceQNX` | 0 | 1 | 1 | VK_QNX_screen_surface<br>*(Added: 2021-01-13)* | 🔴 **0.0%** | 🔴 **0.0%** | 🔴 **0.0%** | Window System & Presentation |
| 130 | `vkCreateRenderPass` | 0 | 1 | 1 | Vulkan Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 131 | `vkCreateGraphicsPipelines` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 132 | `vkCreateFramebuffer` | 0 | 1 | 1 | Vulkan Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 133 | `vkCmdSetViewport` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 134 | `vkCmdDrawIndexed` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 135 | `vkCmdBindVertexBuffers` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
| 136 | `vkCmdBindIndexBuffer` | 0 | 1 | 1 | Vulkan 1.0 Core<br>*(Added: 2016-02-16)* | 🟢 **100.0%** | 🟢 **100.0%** | 🟢 **100.0%** | Raster & Dynamic Rendering |
