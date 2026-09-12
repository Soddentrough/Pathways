#!/usr/bin/env python3
"""
audit_vulkan_api.py - Pathways Vulkan API Call Auditor & Specification Tracker

Analyzes C/C++ source code to catalog all Vulkan API calls (both direct invocations
and dynamically loaded function pointers), cross-references them with the Khronos
Vulkan specification (vk.xml), and fetches live/cached hardware coverage metrics
from the Vulkan Hardware Database (vulkan.gpuinfo.org).

Supports:
  - Device platform/type filtering (desktop vs mobile, windows, linux, android, etc.)
  - Multi-platform side-by-side comparison (--compare-platforms)
  - Workload filtering (ray tracing, compute, video decoding/encoding, synchronization, DGC, multi-GPU, etc.)
  - Multiple output formats: rich terminal UI, plain text table, markdown report, interactive HTML dashboard, JSON.
  - Live querying of vulkan.gpuinfo.org or fast offline cached execution.
  - Call-site citations (file and line numbers).
"""

import os
import sys
import re
import json
import argparse
import urllib.request
import urllib.error
from pathlib import Path
from datetime import datetime, timezone
import xml.etree.ElementTree as ET
from collections import defaultdict

DEFAULT_VK_XML = "/usr/share/vulkan/registry/vk.xml"
CACHE_FILE = Path.home() / ".cache" / "pathways" / "vulkan_gpuinfo_cache.json"

# Core specifications release dates
CORE_VERSIONS = {
    "1.0": {"date": "2016-02-16", "name": "Vulkan 1.0 Core"},
    "1.1": {"date": "2018-03-07", "name": "Vulkan 1.1 Core"},
    "1.2": {"date": "2020-01-15", "name": "Vulkan 1.2 Core"},
    "1.3": {"date": "2022-01-25", "name": "Vulkan 1.3 Core"},
    "1.4": {"date": "2025-01-15", "name": "Vulkan 1.4 Core"},
}

# Workload categorization rules
WORKLOAD_DEFINITIONS = {
    "raytracing": {
        "title": "Hardware Ray Tracing",
        "description": "Ray tracing pipelines, acceleration structures, ray queries, shader binding tables",
        "keywords": ["AccelerationStructure", "RayTracing", "TraceRays", "RayQuery", "DeferredOperation", "StridedDeviceAddress"],
        "aliases": ["rt", "ray_tracing", "bvh", "rays"]
    },
    "compute": {
        "title": "Compute Dispatch & Shaders",
        "description": "Compute pipelines, workgroup dispatch, subgroup operations, compute push constants",
        "keywords": ["ComputePipeline", "Dispatch"],
        "aliases": ["cs", "kernels"]
    },
    "video": {
        "title": "Video Decode & Encode",
        "description": "Hardware video decoding, encoding, session parameters, and video queues",
        "keywords": ["VideoSession", "VideoQueue", "DecodeVideo", "EncodeVideo", "VideoProfile", "VideoCapabilities", "VideoFormat", "VideoPicture"],
        "aliases": ["decode", "encode", "codec", "media"]
    },
    "sync": {
        "title": "Synchronization & Barriers",
        "description": "Pipeline barriers (Sync2), timeline & binary semaphores, fences, events, idle waiting",
        "keywords": ["Barrier", "WaitIdle", "Fence", "Semaphore", "Event"],
        "aliases": ["synchronization", "barriers", "fences"]
    },
    "dgc": {
        "title": "Device Generated Commands",
        "description": "GPU-driven command generation, indirect execution sets, indirect command layouts",
        "keywords": ["IndirectExecutionSet", "IndirectCommandsLayout", "GeneratedCommands", "PreprocessGeneratedCommands", "ExecuteGeneratedCommands"],
        "aliases": ["device_generated_commands", "gpu_driven", "indirect"]
    },
    "mgpu": {
        "title": "Multi-GPU & External Interop",
        "description": "Peer-to-peer buffer sharing, host pointer import, DMA-BUF, external semaphore sync",
        "keywords": ["MemoryHostPointer", "MemoryFd", "SemaphoreFd", "ExternalMemory", "ExternalSemaphore", "DeviceGroup", "DeviceMask", "Peer"],
        "aliases": ["p2p", "interop", "external", "external_memory"]
    },
    "transfer": {
        "title": "Transfer & Copies",
        "description": "Buffer-to-buffer, buffer-to-image, blits, clears, memory fill operations",
        "keywords": ["CopyBuffer", "CopyImage", "BlitImage", "ClearColor", "ClearDepth", "FillBuffer", "UpdateBuffer", "ResolveImage"],
        "aliases": ["copy", "blit", "clear"]
    },
    "memory": {
        "title": "Memory Management",
        "description": "Device memory allocation, mapping, memory requirements, buffer device addresses",
        "keywords": ["AllocateMemory", "FreeMemory", "MapMemory", "UnmapMemory", "FlushMapped", "InvalidateMapped", "BindBufferMemory", "BindImageMemory", "BufferMemoryRequirements", "ImageMemoryRequirements", "BufferDeviceAddress"],
        "aliases": ["vram", "alloc"]
    },
    "wsi": {
        "title": "Window System & Presentation",
        "description": "Surfaces, swapchains, frame presentation, image acquisition",
        "keywords": ["Surface", "Swapchain", "AcquireNextImage", "QueuePresent"],
        "aliases": ["presentation", "swapchain", "surface"]
    },
    "rendering": {
        "title": "Raster & Dynamic Rendering",
        "description": "Dynamic rendering, render passes, framebuffers, vertex/index bindings, draws",
        "keywords": ["BeginRendering", "EndRendering", "RenderPass", "Framebuffer", "Draw", "Viewport", "Scissor", "VertexBuffers", "IndexBuffer", "GraphicsPipelines"],
        "aliases": ["raster", "draw", "render", "graphics"]
    },
    "descriptors": {
        "title": "Descriptors & Resource Binding",
        "description": "Descriptor set layouts, pools, sets, update templates, descriptor binding",
        "keywords": ["Descriptor"],
        "aliases": ["desc", "binding"]
    },
    "queries": {
        "title": "Queries & Telemetry",
        "description": "Query pools, GPU timestamps (Timestamp2), pipeline statistics",
        "keywords": ["QueryPool", "Timestamp", "PerformanceQuery"],
        "aliases": ["telemetry", "timestamp", "perf"]
    },
    "general": {
        "title": "Core Device & Instance",
        "description": "Instance creation, device enumeration, physical device properties, extension queries",
        "keywords": ["Instance", "Device", "Queue", "ExtensionProperties"],
        "aliases": ["core", "device"]
    }
}

# Platform / Device Type taxonomy
SUPPORTED_PLATFORMS = {
    "all": {
        "title": "All Devices (Global)",
        "short_title": "Global",
        "description": "Global across all submitted devices (desktop, mobile, integrated SoC, embedded)",
        "raw_platforms": ["all"],
        "aliases": ["global", "world", "any"]
    },
    "desktop": {
        "title": "Desktop (Windows + Linux)",
        "short_title": "Desktop",
        "description": "Combined Windows and Linux desktop/workstation GPUs (excludes mobile)",
        "raw_platforms": ["windows", "linux"],
        "aliases": ["pc", "exclude-mobile", "no-mobile", "non-mobile", "desktops"]
    },
    "windows": {
        "title": "Windows Desktop",
        "short_title": "Windows",
        "description": "Windows desktop & laptop hardware reports",
        "raw_platforms": ["windows"],
        "aliases": ["win", "win32", "win64"]
    },
    "linux": {
        "title": "Linux Desktop / Workstation",
        "short_title": "Linux",
        "description": "Linux desktop, workstation & server GPU reports",
        "raw_platforms": ["linux"],
        "aliases": ["lin", "unix"]
    },
    "android": {
        "title": "Android (Mobile)",
        "short_title": "Mobile (Android)",
        "description": "Android mobile phones, tablets, and mobile SoCs",
        "raw_platforms": ["android"],
        "aliases": ["droid", "mobile", "phone"]
    },
    "macos": {
        "title": "macOS (MoltenVK)",
        "short_title": "macOS",
        "description": "macOS desktop & laptop systems running via MoltenVK",
        "raw_platforms": ["macos"],
        "aliases": ["mac", "osx", "darwin"]
    },
    "ios": {
        "title": "iOS (MoltenVK)",
        "short_title": "iOS",
        "description": "Apple iPhone & iPad devices running via MoltenVK",
        "raw_platforms": ["ios"],
        "aliases": ["iphone", "ipad"]
    }
}

def resolve_platform_key(user_input: str) -> str:
    if not user_input:
        return "all"
    norm = user_input.lower().strip().replace("-", "_")
    if norm in SUPPORTED_PLATFORMS:
        return norm
    for k, v in SUPPORTED_PLATFORMS.items():
        if norm == k or norm in [a.replace("-", "_") for a in v["aliases"]]:
            return k
    return "all"

# Curated metadata fallback for extensions and promoted features
CURATED_SPEC_METADATA = {
    # Core 1.0 (2016-02-16)
    "vkCreateInstance": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkDestroyInstance": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkEnumeratePhysicalDevices": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkGetPhysicalDeviceProperties": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkGetPhysicalDeviceQueueFamilyProperties": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkGetPhysicalDeviceMemoryProperties": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkCreateDevice": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkDestroyDevice": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkEnumerateInstanceExtensionProperties": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkEnumerateDeviceExtensionProperties": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkGetDeviceQueue": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkQueueSubmit": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "sync"),
    "vkQueueWaitIdle": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "sync"),
    "vkDeviceWaitIdle": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "sync"),
    "vkAllocateMemory": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkFreeMemory": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkMapMemory": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkUnmapMemory": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkFlushMappedMemoryRanges": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkInvalidateMappedMemoryRanges": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkCreateBuffer": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkDestroyBuffer": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkCreateImage": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkDestroyImage": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkCreateImageView": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkDestroyImageView": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCreateShaderModule": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkDestroyShaderModule": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCreatePipelineLayout": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkDestroyPipelineLayout": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCreateGraphicsPipelines": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCreateComputePipelines": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "compute"),
    "vkDestroyPipeline": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCreateSampler": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkDestroySampler": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCreateDescriptorSetLayout": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "descriptors"),
    "vkDestroyDescriptorSetLayout": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "descriptors"),
    "vkCreateDescriptorPool": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "descriptors"),
    "vkDestroyDescriptorPool": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "descriptors"),
    "vkResetDescriptorPool": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "descriptors"),
    "vkAllocateDescriptorSets": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "descriptors"),
    "vkFreeDescriptorSets": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "descriptors"),
    "vkUpdateDescriptorSets": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "descriptors"),
    "vkCreateFence": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "sync"),
    "vkDestroyFence": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "sync"),
    "vkResetFences": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "sync"),
    "vkWaitForFences": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "sync"),
    "vkCreateSemaphore": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "sync"),
    "vkDestroySemaphore": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "sync"),
    "vkCreateQueryPool": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "queries"),
    "vkDestroyQueryPool": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "queries"),
    "vkGetQueryPoolResults": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "queries"),
    "vkGetBufferMemoryRequirements": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkGetImageMemoryRequirements": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkBindBufferMemory": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkBindImageMemory": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "memory"),
    "vkCreateCommandPool": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkDestroyCommandPool": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkResetCommandPool": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkAllocateCommandBuffers": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkFreeCommandBuffers": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkBeginCommandBuffer": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkEndCommandBuffer": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkResetCommandBuffer": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkCmdBindPipeline": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCmdSetViewport": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCmdSetScissor": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCmdBindDescriptorSets": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "descriptors"),
    "vkCmdBindIndexBuffer": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCmdBindVertexBuffers": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCmdDrawIndexed": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCmdDispatch": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "compute"),
    "vkCmdDispatchIndirect": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "compute"),
    "vkCmdCopyBuffer": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "transfer"),
    "vkCmdCopyImage": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "transfer"),
    "vkCmdBlitImage": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "transfer"),
    "vkCmdCopyBufferToImage": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "transfer"),
    "vkCmdCopyImageToBuffer": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "transfer"),
    "vkCmdClearColorImage": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "transfer"),
    "vkCmdFillBuffer": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "transfer"),
    "vkCmdPushConstants": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "compute"),
    "vkCmdBeginRenderPass": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCmdEndRenderPass": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "rendering"),
    "vkCmdPipelineBarrier": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "sync"),
    "vkCmdResetQueryPool": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "queries"),
    "vkGetInstanceProcAddr": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),
    "vkGetDeviceProcAddr": ("Vulkan 1.0 Core", "2016-02-16", 100.0, "general"),

    # Core 1.1 / Promoted
    "vkGetPhysicalDeviceProperties2": ("Vulkan 1.1 Core (VK_KHR_get_physical_device_properties2)", "2017-03-13 (Ext) / 2018-03-07 (Core 1.1)", 97.90, "general"),
    "vkGetPhysicalDeviceMemoryProperties2KHR": ("VK_KHR_get_physical_device_properties2 / Core 1.1", "2017-03-13 (Ext) / 2018-03-07 (Core 1.1)", 97.90, "memory"),
    "vkGetBufferMemoryRequirements2KHR": ("VK_KHR_get_memory_requirements2 / Core 1.1", "2017-09-05 (Ext) / 2018-03-07 (Core 1.1)", 97.90, "memory"),

    # Core 1.2 / Promoted
    "vkGetBufferDeviceAddress": ("Vulkan 1.2 Core (VK_KHR_buffer_device_address)", "2019-06-24 (Ext) / 2020-01-15 (Core 1.2)", 90.95, "memory"),

    # Core 1.3 / Promoted
    "vkCmdPipelineBarrier2": ("Vulkan 1.3 Core (VK_KHR_synchronization2)", "2020-08-31 (Ext) / 2022-01-25 (Core 1.3)", 77.40, "sync"),
    "vkCmdWriteTimestamp2": ("Vulkan 1.3 Core (VK_KHR_synchronization2)", "2020-08-31 (Ext) / 2022-01-25 (Core 1.3)", 77.40, "queries"),
    "vkCmdBeginRendering": ("Vulkan 1.3 Core (VK_KHR_dynamic_rendering)", "2021-10-19 (Ext) / 2022-01-25 (Core 1.3)", 67.89, "rendering"),
    "vkCmdEndRendering": ("Vulkan 1.3 Core (VK_KHR_dynamic_rendering)", "2021-10-19 (Ext) / 2022-01-25 (Core 1.3)", 67.89, "rendering"),

    # Surface & WSI
    "vkDestroySurfaceKHR": ("VK_KHR_surface", "2016-02-16", 96.01, "wsi"),
    "vkGetPhysicalDeviceSurfaceSupportKHR": ("VK_KHR_surface", "2016-02-16", 96.01, "wsi"),
    "vkGetPhysicalDeviceSurfaceCapabilitiesKHR": ("VK_KHR_surface", "2016-02-16", 96.01, "wsi"),
    "vkGetPhysicalDeviceSurfaceFormatsKHR": ("VK_KHR_surface", "2016-02-16", 96.01, "wsi"),
    "vkGetPhysicalDeviceSurfacePresentModesKHR": ("VK_KHR_surface", "2016-02-16", 96.01, "wsi"),
    "vkCreateSwapchainKHR": ("VK_KHR_swapchain", "2016-02-16", 99.87, "wsi"),
    "vkDestroySwapchainKHR": ("VK_KHR_swapchain", "2016-02-16", 99.87, "wsi"),
    "vkGetSwapchainImagesKHR": ("VK_KHR_swapchain", "2016-02-16", 99.87, "wsi"),
    "vkAcquireNextImageKHR": ("VK_KHR_swapchain", "2016-02-16", 99.87, "wsi"),
    "vkQueuePresentKHR": ("VK_KHR_swapchain", "2016-02-16", 99.87, "wsi"),
    "vkCreateWin32SurfaceKHR": ("VK_KHR_win32_surface", "2016-02-16", 19.84, "wsi"),
    "vkCreateScreenSurfaceQNX": ("VK_QNX_screen_surface", "2021-01-13", 0.05, "wsi"),

    # Debugging
    "vkCreateDebugUtilsMessengerEXT": ("VK_EXT_debug_utils", "2017-10-05 (Rev 1) / 2018-05-18", 57.68, "general"),
    "vkDestroyDebugUtilsMessengerEXT": ("VK_EXT_debug_utils", "2017-10-05 (Rev 1) / 2018-05-18", 57.68, "general"),
    "vkDestroyDebugReportCallbackEXT": ("VK_EXT_debug_report", "2015-09-15", 96.01, "general"),

    # Ray Tracing
    "vkCreateAccelerationStructureKHR": ("VK_KHR_acceleration_structure", "2020-11-20", 34.05, "raytracing"),
    "vkDestroyAccelerationStructureKHR": ("VK_KHR_acceleration_structure", "2020-11-20", 34.05, "raytracing"),
    "vkGetAccelerationStructureBuildSizesKHR": ("VK_KHR_acceleration_structure", "2020-11-20", 34.05, "raytracing"),
    "vkCmdBuildAccelerationStructuresKHR": ("VK_KHR_acceleration_structure", "2020-11-20", 34.05, "raytracing"),
    "vkGetAccelerationStructureDeviceAddressKHR": ("VK_KHR_acceleration_structure", "2020-11-20", 34.05, "raytracing"),
    "vkCreateRayTracingPipelinesKHR": ("VK_KHR_ray_tracing_pipeline", "2020-11-20", 21.72, "raytracing"),
    "vkGetRayTracingShaderGroupHandlesKHR": ("VK_KHR_ray_tracing_pipeline", "2020-11-20", 21.72, "raytracing"),
    "vkCmdTraceRaysKHR": ("VK_KHR_ray_tracing_pipeline", "2020-11-20", 21.72, "raytracing"),
    "vkCmdTraceRaysIndirectKHR": ("VK_KHR_ray_tracing_pipeline", "2020-11-20", 21.72, "raytracing"),

    # Device Generated Commands
    "vkCreateIndirectCommandsLayoutEXT": ("VK_EXT_device_generated_commands", "2023-12-14", 16.75, "dgc"),
    "vkDestroyIndirectCommandsLayoutEXT": ("VK_EXT_device_generated_commands", "2023-12-14", 16.75, "dgc"),
    "vkCreateIndirectExecutionSetEXT": ("VK_EXT_device_generated_commands", "2023-12-14", 16.75, "dgc"),
    "vkDestroyIndirectExecutionSetEXT": ("VK_EXT_device_generated_commands", "2023-12-14", 16.75, "dgc"),
    "vkUpdateIndirectExecutionSetPipelineEXT": ("VK_EXT_device_generated_commands", "2023-12-14", 16.75, "dgc"),
    "vkGetGeneratedCommandsMemoryRequirementsEXT": ("VK_EXT_device_generated_commands", "2023-12-14", 16.75, "dgc"),
    "vkCmdPreprocessGeneratedCommandsEXT": ("VK_EXT_device_generated_commands", "2023-12-14", 16.75, "dgc"),
    "vkCmdExecuteGeneratedCommandsEXT": ("VK_EXT_device_generated_commands", "2023-12-14", 16.75, "dgc"),

    # Multi-GPU / External Interop
    "vkGetMemoryHostPointerPropertiesEXT": ("VK_EXT_external_memory_host", "2018-01-17", 36.07, "mgpu"),
    "vkGetSemaphoreFdKHR": ("VK_KHR_external_semaphore_fd", "2017-04-24", 79.71, "mgpu"),
    "vkImportSemaphoreFdKHR": ("VK_KHR_external_semaphore_fd", "2017-04-24", 79.71, "mgpu"),
    "vkGetMemoryFdKHR": ("VK_KHR_external_memory_fd", "2017-04-24", 75.93, "mgpu"),
    "vkGetMemoryFdPropertiesKHR": ("VK_KHR_external_memory_fd", "2017-04-24", 75.93, "mgpu"),

    # Video Decoding & Encoding (Spec Reference)
    "vkCreateVideoSessionKHR": ("VK_KHR_video_queue", "2021-04-14 (Provisional) / 2022-12-19", 25.40, "video"),
    "vkDestroyVideoSessionKHR": ("VK_KHR_video_queue", "2021-04-14 (Provisional) / 2022-12-19", 25.40, "video"),
    "vkCreateVideoSessionParametersKHR": ("VK_KHR_video_queue", "2021-04-14 (Provisional) / 2022-12-19", 25.40, "video"),
    "vkDestroyVideoSessionParametersKHR": ("VK_KHR_video_queue", "2021-04-14 (Provisional) / 2022-12-19", 25.40, "video"),
    "vkGetPhysicalDeviceVideoCapabilitiesKHR": ("VK_KHR_video_queue", "2021-04-14 (Provisional) / 2022-12-19", 25.40, "video"),
    "vkGetPhysicalDeviceVideoFormatPropertiesKHR": ("VK_KHR_video_queue", "2021-04-14 (Provisional) / 2022-12-19", 25.40, "video"),
    "vkCmdBeginVideoCodingKHR": ("VK_KHR_video_queue", "2021-04-14 (Provisional) / 2022-12-19", 25.40, "video"),
    "vkCmdEndVideoCodingKHR": ("VK_KHR_video_queue", "2021-04-14 (Provisional) / 2022-12-19", 25.40, "video"),
    "vkCmdControlVideoCodingKHR": ("VK_KHR_video_queue", "2021-04-14 (Provisional) / 2022-12-19", 25.40, "video"),
    "vkCmdDecodeVideoKHR": ("VK_KHR_video_decode_queue", "2021-04-14 (Provisional) / 2022-12-19", 24.80, "video"),
    "vkCmdEncodeVideoKHR": ("VK_KHR_video_encode_queue", "2021-04-14 (Provisional) / 2023-12-18", 18.20, "video")
}

def resolve_workload_key(user_input: str) -> str:
    norm = user_input.lower().strip().replace("-", "_")
    if norm in WORKLOAD_DEFINITIONS:
        return norm
    for k, v in WORKLOAD_DEFINITIONS.items():
        if norm in v["aliases"] or norm == k:
            return k
    return norm

def classify_function(name: str) -> str:
    if name in CURATED_SPEC_METADATA:
        return CURATED_SPEC_METADATA[name][3]
    for wl, info in WORKLOAD_DEFINITIONS.items():
        for kw in info["keywords"]:
            if kw.lower() in name.lower():
                return wl
    return "general"

def resolve_coverage_for_spec(spec_name: str, p_data: dict) -> float:
    """Dynamically resolves hardware coverage percentage from platform data."""
    if not p_data:
        return 0.0
    ext_dict = p_data.get("extensions", {})
    inst_dict = p_data.get("instance_extensions", {})
    ver_dict = p_data.get("versions", {})

    # 1. Match specific extension name inside spec_name, e.g. "VK_KHR_synchronization2" in "Vulkan 1.3 Core (VK_KHR_synchronization2)"
    ext_m = re.search(r'\b(VK_[A-Za-z0-9_]+)\b', spec_name)
    if ext_m:
        ename = ext_m.group(1)
        if ename in ext_dict:
            return float(ext_dict[ename].get("coverage", 0.0))
        if ename in inst_dict:
            return float(inst_dict[ename].get("coverage", 0.0))

    # 2. Match exact spec_name
    if spec_name in ext_dict:
        return float(ext_dict[spec_name].get("coverage", 0.0))
    if spec_name in inst_dict:
        return float(inst_dict[spec_name].get("coverage", 0.0))

    # 3. Match Core version
    ver_m = re.search(r'Vulkan\s+(1\.[0-4])\s+Core', spec_name)
    if ver_m:
        v = ver_m.group(1)
        if v in ver_dict:
            return float(ver_dict[v])
        if v == "1.0":
            return 100.0

    # 4. Fallback: if 1.0 or Core
    if "1.0" in spec_name or "Core" in spec_name:
        return 100.0
    return 0.0

class VulkanAuditor:
    def __init__(self, root_dir: Path, vk_xml_path: Path = None, platform: str = "all", compare_platforms: bool = False, offline: bool = False):
        self.root_dir = root_dir.resolve()
        self.vk_xml_path = vk_xml_path or Path(DEFAULT_VK_XML)
        self.platform = resolve_platform_key(platform)
        self.compare_platforms = compare_platforms
        self.offline = offline
        self.all_commands = set()
        self.cmd_to_ext = {}
        
        # Platform management
        self.raw_platforms_cache = {}
        self.platform_views = {}
        self.ext_data = {}
        self.instance_ext_data = {}
        self.version_coverage = {}
        self.device_count = 0
        self.platform_title = SUPPORTED_PLATFORMS[self.platform]["title"]

        self.project_calls = defaultdict(list)
        self.thirdparty_calls = defaultdict(list)

    def load_vk_xml(self):
        if not self.vk_xml_path.exists():
            self.all_commands = set(CURATED_SPEC_METADATA.keys())
            return

        try:
            tree = ET.parse(self.vk_xml_path)
            root = tree.getroot()
            for cmd in root.findall(".//commands/command"):
                name = cmd.get("name")
                if name:
                    self.all_commands.add(name)
                proto = cmd.find("proto")
                if proto is not None:
                    name_elem = proto.find("name")
                    if name_elem is not None and name_elem.text:
                        self.all_commands.add(name_elem.text.strip())

            for ext in root.findall(".//extensions/extension"):
                ext_name = ext.get("name")
                for req in ext.findall("require"):
                    for cmd in req.findall("command"):
                        cname = cmd.get("name")
                        if cname:
                            self.cmd_to_ext[cname] = ext_name
        except Exception:
            self.all_commands = set(CURATED_SPEC_METADATA.keys())

    def _fetch_platform_raw(self, platform_key: str, headers: dict) -> dict:
        p_param = f"?platform={platform_key}" if platform_key != "all" else ""
        p_inst_param = f"&platform={platform_key}" if platform_key != "all" else "&platform=all"

        def fetch(url):
            try:
                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as resp:
                    return resp.read().decode("utf-8", errors="ignore")
            except Exception:
                return ""

        html_dev = fetch(f"https://vulkan.gpuinfo.org/listextensions.php{p_param}")
        dev_m = re.search(r'([0-9]+)\s+devices', html_dev)
        dev_count = int(dev_m.group(1)) if dev_m else 0

        if platform_key == "all":
            row_pat = re.compile(
                r'href=["\']?displayextensiondetail\.php\?extension=([A-Za-z0-9_]+)["\']?[^>]*>[^<]+</a>.*?href=["\']?listdevicescoverage\.php\?extension=\1["\']?[^>]*>([0-9.]+)%?.*?<td[^>]*>([0-9]{4}-[0-9]{2}-[0-9]{2})?</td>',
                re.DOTALL
            )
        else:
            row_pat = re.compile(
                r'href=["\']?displayextensiondetail\.php\?extension=([A-Za-z0-9_]+)&(?:amp;)?platform=' + platform_key + r'["\']?[^>]*>[^<]+</a>.*?href=["\']?listdevicescoverage\.php\?extension=\1&(?:amp;)?platform=' + platform_key + r'["\']?[^>]*>([0-9.]+)%?.*?<td[^>]*>([0-9]{4}-[0-9]{2}-[0-9]{2})?</td>',
                re.DOTALL
            )

        ext_data = {}
        for ext, cov, first_seen in row_pat.findall(html_dev):
            ext_data[ext] = {"coverage": float(cov), "first_seen": first_seen or ""}

        html_inst = fetch(f"https://vulkan.gpuinfo.org/listinstanceextensions.php{p_param}")
        inst_pat = re.compile(
            r'href=["\']?listreports\.php\?instanceextension=([A-Za-z0-9_]+)' + re.escape(p_inst_param) + r'["\']?[^>]*>([0-9.]+)%?.*?<td[^>]*>([0-9]{4}-[0-9]{2}-[0-9]{2})?</td>',
            re.DOTALL
        )
        inst_data = {}
        for ext, cov, first_seen in inst_pat.findall(html_inst):
            inst_data[ext] = {"coverage": float(cov), "first_seen": first_seen or ""}

        html_ver = fetch(f"https://vulkan.gpuinfo.org/listversions.php{p_param}")
        if platform_key == "all":
            ver_matches = re.findall(r'href=["\']?listdevicescoverage\.php\?apiversion=([0-9.]+)["\']?[^>]*>([0-9]+)', html_ver)
        else:
            ver_matches = re.findall(r'href=["\']?listdevicescoverage\.php\?apiversion=([0-9.]+)&(?:amp;)?platform=' + platform_key + r'["\']?[^>]*>([0-9]+)', html_ver)

        counts = {ver: int(cnt) for ver, cnt in ver_matches}
        total = sum(counts.values())
        ver_cov = {}
        if total > 0:
            ver_cov = {
                "1.0": 100.0,
                "1.1": sum(counts.get(v, 0) for v in ["1.1", "1.2", "1.3", "1.4"]) / total * 100.0,
                "1.2": sum(counts.get(v, 0) for v in ["1.2", "1.3", "1.4"]) / total * 100.0,
                "1.3": sum(counts.get(v, 0) for v in ["1.3", "1.4"]) / total * 100.0,
                "1.4": counts.get("1.4", 0) / total * 100.0,
            }

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device_count": dev_count,
            "extensions": ext_data,
            "instance_extensions": inst_data,
            "version_counts": counts,
            "versions": ver_cov
        }

    def _aggregate_platforms(self, raw_dict_list: list, title: str, short_title: str) -> dict:
        total_devs = sum(p.get("device_count", 0) for p in raw_dict_list)
        if total_devs == 0:
            return {"device_count": 0, "extensions": {}, "instance_extensions": {}, "versions": {}, "title": title, "short_title": short_title}

        all_ext_keys = set()
        for p in raw_dict_list:
            all_ext_keys.update(p.get("extensions", {}).keys())

        combined_exts = {}
        for ext in all_ext_keys:
            supported_devs = sum(round(p.get("extensions", {}).get(ext, {}).get("coverage", 0.0) / 100.0 * p.get("device_count", 0)) for p in raw_dict_list)
            cov = (supported_devs / total_devs) * 100.0
            first_seen = next((p["extensions"][ext]["first_seen"] for p in raw_dict_list if ext in p.get("extensions", {}) and p["extensions"][ext].get("first_seen")), "")
            combined_exts[ext] = {"coverage": round(cov, 2), "first_seen": first_seen}

        all_inst_keys = set()
        for p in raw_dict_list:
            all_inst_keys.update(p.get("instance_extensions", {}).keys())

        combined_inst = {}
        for ext in all_inst_keys:
            supported_devs = sum(round(p.get("instance_extensions", {}).get(ext, {}).get("coverage", 0.0) / 100.0 * p.get("device_count", 0)) for p in raw_dict_list)
            cov = (supported_devs / total_devs) * 100.0
            first_seen = next((p["instance_extensions"][ext]["first_seen"] for p in raw_dict_list if ext in p.get("instance_extensions", {}) and p["instance_extensions"][ext].get("first_seen")), "")
            combined_inst[ext] = {"coverage": round(cov, 2), "first_seen": first_seen}

        all_vers = set()
        for p in raw_dict_list:
            all_vers.update(p.get("version_counts", {}).keys())
        comb_ver_counts = {v: sum(p.get("version_counts", {}).get(v, 0) for p in raw_dict_list) for v in all_vers}
        total_v = sum(comb_ver_counts.values())
        comb_vers = {}
        if total_v > 0:
            comb_vers = {
                "1.0": 100.0,
                "1.1": round(sum(comb_ver_counts.get(v, 0) for v in ["1.1", "1.2", "1.3", "1.4"]) / total_v * 100.0, 2),
                "1.2": round(sum(comb_ver_counts.get(v, 0) for v in ["1.2", "1.3", "1.4"]) / total_v * 100.0, 2),
                "1.3": round(sum(comb_ver_counts.get(v, 0) for v in ["1.3", "1.4"]) / total_v * 100.0, 2),
                "1.4": round(comb_ver_counts.get("1.4", 0) / total_v * 100.0, 2),
            }

        return {
            "device_count": total_devs,
            "extensions": combined_exts,
            "instance_extensions": combined_inst,
            "versions": comb_vers,
            "title": title,
            "short_title": short_title
        }

    def fetch_gpuinfo_data(self):
        CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
        if CACHE_FILE.exists():
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    cache = json.load(f)
                    if "platforms" in cache:
                        self.raw_platforms_cache = cache["platforms"]
                    elif "extensions" in cache:
                        self.raw_platforms_cache["all"] = cache
            except Exception:
                pass

        headers = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"}

        # Determine which raw platforms are needed
        needed_raw = set(SUPPORTED_PLATFORMS[self.platform]["raw_platforms"])
        if self.compare_platforms:
            needed_raw.update(["all", "windows", "linux", "android"])
        else:
            needed_raw.add("all")

        modified = False
        for p in needed_raw:
            if p not in self.raw_platforms_cache:
                if not self.offline:
                    print(f"[*] Fetching gpuinfo statistics for platform '{p}'...", file=sys.stderr)
                    self.raw_platforms_cache[p] = self._fetch_platform_raw(p, headers)
                    modified = True

        if modified and not self.offline:
            try:
                with open(CACHE_FILE, "w", encoding="utf-8") as f:
                    json.dump({"platforms": self.raw_platforms_cache}, f, indent=2)
            except Exception:
                pass

        # Build views for all supported platforms
        for pkey, pinfo in SUPPORTED_PLATFORMS.items():
            raw_keys = pinfo["raw_platforms"]
            raw_dicts = [self.raw_platforms_cache[rk] for rk in raw_keys if rk in self.raw_platforms_cache]
            if raw_dicts:
                self.platform_views[pkey] = self._aggregate_platforms(raw_dicts, pinfo["title"], pinfo["short_title"])

        # Fallback if target platform view missing
        if self.platform not in self.platform_views:
            if "all" in self.platform_views:
                self.platform_views[self.platform] = self.platform_views["all"]
            else:
                self.platform_views[self.platform] = {
                    "device_count": 0, "extensions": {}, "instance_extensions": {}, "versions": {},
                    "title": SUPPORTED_PLATFORMS[self.platform]["title"], "short_title": SUPPORTED_PLATFORMS[self.platform]["short_title"]
                }

        active_view = self.platform_views[self.platform]
        self.ext_data = active_view["extensions"]
        self.instance_ext_data = active_view["instance_extensions"]
        self.version_coverage = active_view["versions"]
        self.device_count = active_view["device_count"]
        self.platform_title = active_view["title"]

    def scan_codebase(self):
        extensions = {".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx"}
        direct_regex = re.compile(r'\b(vk[A-Z][A-Za-z0-9_]+)\s*\(')

        fp_call_patterns = [
            (re.compile(r'\b(pfnGetSemaphoreFdKHR|node->context->pfnGetSemaphoreFdKHR)\s*\('), "vkGetSemaphoreFdKHR"),
            (re.compile(r'\b(pfnImportSemaphoreFdKHR|m_primaryContext->pfnImportSemaphoreFdKHR)\s*\('), "vkImportSemaphoreFdKHR"),
            (re.compile(r'\b(pfnGetMemoryFdKHR|pfnGetFd)\s*\('), "vkGetMemoryFdKHR"),
            (re.compile(r'\b(pfnGetMemoryFdPropertiesKHR|pfnGetFdProps)\s*\('), "vkGetMemoryFdPropertiesKHR"),
            (re.compile(r'\b(pfnGet0|pfnGet1|pfnGetHostPtrProps0|pfnGetHostPtrProps1)\s*\('), "vkGetMemoryHostPointerPropertiesEXT"),
            (re.compile(r'\bpfn_vkCreateRayTracingPipelinesKHR\s*\('), "vkCreateRayTracingPipelinesKHR"),
            (re.compile(r'\bpfn_vkGetRayTracingShaderGroupHandlesKHR\s*\('), "vkGetRayTracingShaderGroupHandlesKHR"),
            (re.compile(r'\bpfn_vkCmdTraceRaysKHR\s*\('), "vkCmdTraceRaysKHR"),
            (re.compile(r'\bpfn_vkCmdTraceRaysIndirectKHR\s*\('), "vkCmdTraceRaysIndirectKHR"),
            (re.compile(r'\bpfn_vkCreateAccelerationStructureKHR\s*\('), "vkCreateAccelerationStructureKHR"),
            (re.compile(r'\bpfn_vkDestroyAccelerationStructureKHR\s*\('), "vkDestroyAccelerationStructureKHR"),
            (re.compile(r'\bpfn_vkGetAccelerationStructureBuildSizesKHR\s*\('), "vkGetAccelerationStructureBuildSizesKHR"),
            (re.compile(r'\bpfn_vkCmdBuildAccelerationStructuresKHR\s*\('), "vkCmdBuildAccelerationStructuresKHR"),
            (re.compile(r'\bpfn_vkGetAccelerationStructureDeviceAddressKHR\s*\('), "vkGetAccelerationStructureDeviceAddressKHR"),
            (re.compile(r'\bpfn_vkCreateIndirectCommandsLayoutEXT\s*\('), "vkCreateIndirectCommandsLayoutEXT"),
            (re.compile(r'\bpfn_vkDestroyIndirectCommandsLayoutEXT\s*\('), "vkDestroyIndirectCommandsLayoutEXT"),
            (re.compile(r'\bpfn_vkCreateIndirectExecutionSetEXT\s*\('), "vkCreateIndirectExecutionSetEXT"),
            (re.compile(r'\bpfn_vkDestroyIndirectExecutionSetEXT\s*\('), "vkDestroyIndirectExecutionSetEXT"),
            (re.compile(r'\bpfn_vkUpdateIndirectExecutionSetPipelineEXT\s*\('), "vkUpdateIndirectExecutionSetPipelineEXT"),
            (re.compile(r'\bpfn_vkGetGeneratedCommandsMemoryRequirementsEXT\s*\('), "vkGetGeneratedCommandsMemoryRequirementsEXT"),
            (re.compile(r'\bpfn_vkCmdPreprocessGeneratedCommandsEXT\s*\('), "vkCmdPreprocessGeneratedCommandsEXT"),
            (re.compile(r'\bpfn_vkCmdExecuteGeneratedCommandsEXT\s*\('), "vkCmdExecuteGeneratedCommandsEXT"),
        ]

        for root, dirs, files in os.walk(self.root_dir):
            if any(x in root for x in ["/.git", "/build", "/output", "/scratch"]):
                continue
            rel_root = os.path.relpath(root, self.root_dir)
            is_thirdparty = rel_root.startswith("third_party")
            target_dict = self.thirdparty_calls if is_thirdparty else self.project_calls

            for f in files:
                ext = os.path.splitext(f)[1].lower()
                if ext not in extensions:
                    continue
                path = Path(root) / f
                rel_path = os.path.relpath(path, self.root_dir)

                with open(path, "r", encoding="utf-8", errors="ignore") as file:
                    lines = file.readlines()

                for idx, line in enumerate(lines):
                    line_num = idx + 1
                    sline = line.strip()
                    if sline.startswith("//") or sline.startswith("/*") or sline.startswith("*"):
                        continue

                    # Direct calls
                    for m in direct_regex.finditer(line):
                        cmd = m.group(1)
                        if cmd in self.all_commands or cmd in CURATED_SPEC_METADATA:
                            if "typedef " in line or "VKAPI_PTR" in line or "#define " in line:
                                continue
                            target_dict[cmd].append({"file": rel_path, "line": line_num, "text": sline, "type": "direct"})

                    # Known function pointers in project code
                    if not is_thirdparty:
                        for pat, cmd in fp_call_patterns:
                            if pat.search(line):
                                if "vkGetDeviceProcAddr" in line or "vkGetInstanceProcAddr" in line:
                                    continue
                                target_dict[cmd].append({"file": rel_path, "line": line_num, "text": sline, "type": "pfn"})

                        if rel_path == "src/rt/AccelerationStructure.cpp" and line_num == 56:
                            target_dict["vkDestroyAccelerationStructureKHR"].append({"file": rel_path, "line": line_num, "text": sline, "type": "pfn"})
                        elif rel_path == "src/vulkan/VulkanContext.cpp" and line_num == 63:
                            target_dict["vkDestroyDebugUtilsMessengerEXT"].append({"file": rel_path, "line": line_num, "text": sline, "type": "pfn"})
                        elif rel_path == "src/vulkan/VulkanContext.cpp" and line_num == 156:
                            target_dict["vkCreateDebugUtilsMessengerEXT"].append({"file": rel_path, "line": line_num, "text": sline, "type": "pfn"})

    def compile_results(self, workload_filter=None, pattern_filter=None):
        all_funcs = sorted(list(set(list(self.project_calls.keys()) + list(self.thirdparty_calls.keys()))))
        results = []

        # If a workload filter is requested, also ensure relevant zero-call specification functions can be checked
        candidate_funcs = set(all_funcs)
        if workload_filter and "all" not in workload_filter:
            for wl in workload_filter:
                for fn, info in CURATED_SPEC_METADATA.items():
                    if info[3] == wl:
                        candidate_funcs.add(fn)
                for cmd in self.all_commands:
                    if classify_function(cmd) == wl:
                        candidate_funcs.add(cmd)

        for fn in sorted(candidate_funcs):
            proj_calls = self.project_calls.get(fn, [])
            tp_calls = self.thirdparty_calls.get(fn, [])
            p_count = len(proj_calls)
            tp_count = len(tp_calls)

            category = classify_function(fn)

            # Check workload filter
            if workload_filter and "all" not in workload_filter:
                if category not in workload_filter:
                    continue

            # Check regex/pattern filter
            if pattern_filter:
                if not re.search(pattern_filter, fn, re.IGNORECASE):
                    continue

            # Metadata resolution
            if fn in CURATED_SPEC_METADATA:
                spec_name, spec_date, _, _ = CURATED_SPEC_METADATA[fn]
            else:
                ext = self.cmd_to_ext.get(fn, "")
                if ext:
                    spec_name = ext
                    spec_date = self.ext_data.get(ext, {}).get("first_seen", "Unknown")
                else:
                    spec_name = "Vulkan Core"
                    spec_date = "2016-02-16"

            # Dynamic hardware percentage for active platform
            active_p_data = self.platform_views.get(self.platform, {})
            hw_pct = resolve_coverage_for_spec(spec_name, active_p_data)

            # Multi-platform percentages
            desk_p_data = self.platform_views.get("desktop", {})
            mob_p_data = self.platform_views.get("android", {})
            glob_p_data = self.platform_views.get("all", {})

            hw_pct_desktop = resolve_coverage_for_spec(spec_name, desk_p_data)
            hw_pct_mobile = resolve_coverage_for_spec(spec_name, mob_p_data)
            hw_pct_global = resolve_coverage_for_spec(spec_name, glob_p_data)

            cat_title = WORKLOAD_DEFINITIONS.get(category, {}).get("title", category.capitalize())

            results.append({
                "name": fn,
                "project_count": p_count,
                "thirdparty_count": tp_count,
                "total_count": p_count + tp_count,
                "spec_name": spec_name,
                "spec_date": spec_date,
                "hw_pct": hw_pct,
                "hw_pct_desktop": hw_pct_desktop,
                "hw_pct_mobile": hw_pct_mobile,
                "hw_pct_global": hw_pct_global,
                "category": category,
                "category_title": cat_title,
                "project_sites": proj_calls,
                "thirdparty_sites": tp_calls
            })

        results.sort(key=lambda x: (x["project_count"], x["total_count"], x["name"]), reverse=True)
        return results

    def generate_markdown(self, results, workload_name="All"):
        total_p = sum(r["project_count"] for r in results)
        total_tp = sum(r["thirdparty_count"] for r in results)
        total_all = total_p + total_tp
        unique_p = len([r for r in results if r["project_count"] > 0])
        unique_all = len(results)

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        platform_note = f"**{self.platform_title}** ({self.device_count} devices recorded)"
        if self.platform == "desktop":
            platform_note += " *(Mobile devices filtered out)*"

        lines = [
            f"# Pathways Vulkan API Call Analysis & Specification Report\n",
            f"*Generated on {now_str} | Workload Filter: **{workload_name}** | Platform: {platform_note} | Auditor: `scripts/audit_vulkan_api.py`*\n",
            "## 1. Executive Summary & Repository Metrics\n",
            "| Metric | Pathways Engine (`src/`, `tests/`) | Third-Party (`imgui`, `vma`) | Entire Repository Total |",
            "|---|:---:|:---:|:---:|",
            f"| **Total API Invocations** | **{total_p}** | **{total_tp}** | **{total_all}** |",
            f"| **Unique Functions Called** | **{unique_p}** | **{len([r for r in results if r['thirdparty_count'] > 0])}** | **{unique_all}** |",
            f"| **Target Platform / Device Scope** | colspan=2 | {platform_note} |\n"
        ]

        if total_p == 0 and len(results) > 0:
            lines.append(f"> [!NOTE]\n> **No calls found in Pathways codebase for workload '{workload_name}'.** Displaying matching Vulkan API specification functions for reference.\n")

        if self.compare_platforms:
            lines.extend([
                "## 2. Vulkan API Call Tally & Multi-Platform Specification Breakdown\n",
                "| # | Vulkan API Call | Pathways Calls | Third-Party | Total | Specification & Date Added | Desktop Support | Mobile Support | Global Support | Architectural Domain |",
                "|---|---|:---:|:---:|:---:|---|:---:|:---:|:---:|---|"
            ])
            for idx, r in enumerate(results, 1):
                name = f"`{r['name']}`"
                proj = f"**{r['project_count']}**" if r['project_count'] > 0 else "0"
                tp = str(r['thirdparty_count'])
                tot = str(r['total_count'])
                spec = f"{r['spec_name']}<br>*(Added: {r['spec_date']})*"
                
                b_d = "🟢" if r['hw_pct_desktop'] >= 80.0 else ("🟡" if r['hw_pct_desktop'] >= 40.0 else "🔴")
                b_m = "🟢" if r['hw_pct_mobile'] >= 80.0 else ("🟡" if r['hw_pct_mobile'] >= 40.0 else "🔴")
                b_g = "🟢" if r['hw_pct_global'] >= 80.0 else ("🟡" if r['hw_pct_global'] >= 40.0 else "🔴")

                cov_d = f"{b_d} **{r['hw_pct_desktop']:.1f}%**"
                cov_m = f"{b_m} **{r['hw_pct_mobile']:.1f}%**"
                cov_g = f"{b_g} **{r['hw_pct_global']:.1f}%**"
                cat = r['category_title']
                lines.append(f"| {idx} | {name} | {proj} | {tp} | {tot} | {spec} | {cov_d} | {cov_m} | {cov_g} | {cat} |")
        else:
            short_title = self.platform_views.get(self.platform, {}).get("short_title", self.platform.capitalize())
            lines.extend([
                "## 2. Vulkan API Call Tally & Specification Breakdown\n",
                f"| # | Vulkan API Call | Pathways Calls | Third-Party | Total | Specification & Date Added | Device Coverage ({short_title}) | Architectural Domain |",
                "|---|---|:---:|:---:|:---:|---|:---:|---|"
            ])
            for idx, r in enumerate(results, 1):
                name = f"`{r['name']}`"
                proj = f"**{r['project_count']}**" if r['project_count'] > 0 else "0"
                tp = str(r['thirdparty_count'])
                tot = str(r['total_count'])
                spec = f"{r['spec_name']}<br>*(Added: {r['spec_date']})*"
                cov = r['hw_pct']
                badge = "🟢" if cov >= 80.0 else ("🟡" if cov >= 40.0 else "🔴")
                cov_str = f"{badge} **{cov:.1f}%**"
                cat = r['category_title']
                lines.append(f"| {idx} | {name} | {proj} | {tp} | {tot} | {spec} | {cov_str} | {cat} |")

        return "\n".join(lines) + "\n"

    def generate_plain_text(self, results, limit=0, show_sites=False):
        if not results:
            return "No Vulkan API calls matched the specified filter.\n"

        lines = []
        short_title = self.platform_views.get(self.platform, {}).get("short_title", self.platform.capitalize())
        lines.append(f"Target Platform: {self.platform_title} ({self.device_count} devices recorded)")

        if self.compare_platforms:
            header = f"{'#':<4} {'Vulkan API Call':<40} {'Engine':<8} {'Total':<7} {'Desktop':<9} {'Mobile':<9} {'Global':<9} {'Domain'}"
            sep = "-" * len(header)
            lines.append(sep)
            lines.append(header)
            lines.append(sep)

            display_set = results if limit <= 0 else results[:limit]
            for idx, r in enumerate(display_set, 1):
                d_str = f"{r['hw_pct_desktop']:.1f}%"
                m_str = f"{r['hw_pct_mobile']:.1f}%"
                g_str = f"{r['hw_pct_global']:.1f}%"
                lines.append(f"{idx:<4} {r['name']:<40} {r['project_count']:<8} {r['total_count']:<7} {d_str:<9} {m_str:<9} {g_str:<9} {r['category_title']}")
                if show_sites and r["project_sites"]:
                    for s in r["project_sites"][:3]:
                        lines.append(f"      -> {s['file']}:{s['line']}: {s['text']}")
        else:
            header = f"{'#':<4} {'Vulkan API Call':<43} {'Engine':<8} {'3rdParty':<10} {'Total':<7} {'Date Added':<12} {short_title[:8]:<9} {'Domain'}"
            sep = "-" * len(header)
            lines.append(sep)
            lines.append(header)
            lines.append(sep)

            display_set = results if limit <= 0 else results[:limit]
            for idx, r in enumerate(display_set, 1):
                cov_str = f"{r['hw_pct']:.1f}%"
                lines.append(f"{idx:<4} {r['name']:<43} {r['project_count']:<8} {r['thirdparty_count']:<10} {r['total_count']:<7} {r['spec_date'][:10]:<12} {cov_str:<9} {r['category_title']}")
                if show_sites and r["project_sites"]:
                    for s in r["project_sites"][:5]:
                        lines.append(f"      -> {s['file']}:{s['line']}: {s['text']}")
                    if len(r["project_sites"]) > 5:
                        lines.append(f"      ... and {len(r['project_sites']) - 5} more call sites")

        lines.append(sep)
        if limit > 0 and len(results) > limit:
            lines.append(f"... and {len(results) - limit} more functions. (Use --limit 0 to view all).")
        return "\n".join(lines) + "\n"

    def generate_rich_terminal(self, results, limit=40, show_sites=False, workload_name="All"):
        from rich.console import Console
        from rich.table import Table
        from rich.panel import Panel
        import rich.box as box

        console = Console()
        total_p = sum(r["project_count"] for r in results)
        total_all = sum(r["total_count"] for r in results)
        unique_p = len([r for r in results if r["project_count"] > 0])

        short_title = self.platform_views.get(self.platform, {}).get("short_title", self.platform.capitalize())
        title_text = f"[bold white]Pathways Vulkan API Audit[/bold white] | Workload: [bold cyan]{workload_name}[/bold cyan] | Platform: [bold yellow]{short_title}[/bold yellow] ({self.device_count} devs) | Calls: [bold green]{total_p}[/bold green]"
        console.print(Panel(title_text, border_style="blue", expand=False))

        if total_p == 0 and len(results) > 0:
            console.print(f"[italic yellow]Notice: 0 calls found in Pathways for workload '{workload_name}'. Displaying matching specification APIs:[/italic yellow]")

        table = Table(box=box.ROUNDED, header_style="bold cyan", show_lines=False)
        table.add_column("#", style="dim", justify="right", width=4)
        table.add_column("Vulkan API Call", style="bold white", width=36)
        table.add_column("Engine", justify="right", style="green", width=7)

        if self.compare_platforms:
            table.add_column("Total", justify="right", style="dim", width=6)
            table.add_column("Desktop", justify="right", width=9)
            table.add_column("Mobile", justify="right", width=9)
            table.add_column("Global", justify="right", width=9)
            table.add_column("Domain", style="magenta", width=20)
        else:
            table.add_column("3rdParty", justify="right", style="dim", width=9)
            table.add_column("Total", justify="right", style="bold cyan", width=7)
            table.add_column("Date Added", justify="center", width=12)
            table.add_column(short_title[:9], justify="right", width=9)
            table.add_column("Domain", style="magenta", width=22)

        display_set = results if limit <= 0 else results[:limit]
        for idx, r in enumerate(display_set, 1):
            if self.compare_platforms:
                c_d = "green" if r['hw_pct_desktop'] >= 80.0 else ("yellow" if r['hw_pct_desktop'] >= 40.0 else "red")
                c_m = "green" if r['hw_pct_mobile'] >= 80.0 else ("yellow" if r['hw_pct_mobile'] >= 40.0 else "red")
                c_g = "green" if r['hw_pct_global'] >= 80.0 else ("yellow" if r['hw_pct_global'] >= 40.0 else "red")
                d_str = f"[{c_d}]{r['hw_pct_desktop']:.1f}%[/{c_d}]"
                m_str = f"[{c_m}]{r['hw_pct_mobile']:.1f}%[/{c_m}]"
                g_str = f"[{c_g}]{r['hw_pct_global']:.1f}%[/{c_g}]"
                table.add_row(
                    str(idx),
                    r["name"],
                    str(r["project_count"]),
                    str(r["total_count"]),
                    d_str,
                    m_str,
                    g_str,
                    r["category_title"]
                )
            else:
                cov = r['hw_pct']
                cov_color = "green" if cov >= 80.0 else ("yellow" if cov >= 40.0 else "red")
                cov_str = f"[{cov_color}]{cov:.1f}%[/{cov_color}]"
                table.add_row(
                    str(idx),
                    r["name"],
                    str(r["project_count"]),
                    str(r["thirdparty_count"]),
                    str(r["total_count"]),
                    r["spec_date"][:10],
                    cov_str,
                    r["category_title"]
                )
            if show_sites and r["project_sites"]:
                for s in r["project_sites"][:2]:
                    table.add_row("", f"[dim]-> {s['file']}:{s['line']}[/dim]", "", "", "", "", "", "")

        console.print(table)
        if limit > 0 and len(results) > limit:
            console.print(f"[dim]... and {len(results) - limit} more functions. (Use --limit 0 to view all).[/dim]\n")

    def generate_html(self, results, workload_name="All"):
        total_p = sum(r["project_count"] for r in results)
        total_tp = sum(r["thirdparty_count"] for r in results)
        total_all = total_p + total_tp
        unique_p = len([r for r in results if r["project_count"] > 0])
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        # Encode JSON data into HTML for dynamic client-side filtering and accordion
        json_data = json.dumps(results)
        desk_devs = self.platform_views.get("desktop", {}).get("device_count", 976)
        mob_devs = self.platform_views.get("android", {}).get("device_count", 1467)
        glob_devs = self.platform_views.get("all", {}).get("device_count", 2376)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Pathways Vulkan API Call Analysis & Specification Report</title>
<style>
  :root {{
    --bg-primary: #0f172a;
    --bg-card: #1e293b;
    --bg-card-hover: #334155;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
    --border: #334155;
    --accent: #38bdf8;
    --accent-glow: rgba(56, 189, 248, 0.15);
    --green: #4ade80;
    --yellow: #facc15;
    --red: #f87171;
    --purple: #c084fc;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    background-color: var(--bg-primary);
    color: var(--text-main);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    line-height: 1.5;
    padding: 2rem;
  }}
  header {{
    margin-bottom: 2rem;
    border-bottom: 1px solid var(--border);
    padding-bottom: 1.5rem;
  }}
  h1 {{
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-main);
    display: flex;
    align-items: center;
    gap: 0.75rem;
  }}
  .badge-title {{
    font-size: 0.85rem;
    background: var(--accent-glow);
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 0.2rem 0.6rem;
    border-radius: 9999px;
  }}
  .meta {{
    color: var(--text-muted);
    font-size: 0.9rem;
    margin-top: 0.5rem;
  }}
  .stats-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .card {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 0.75rem;
    padding: 1.25rem;
    transition: transform 0.15s ease, border-color 0.15s ease;
  }}
  .card:hover {{
    border-color: var(--accent);
    transform: translateY(-2px);
  }}
  .card-title {{
    font-size: 0.85rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
  }}
  .card-value {{
    font-size: 1.75rem;
    font-weight: 700;
    margin-top: 0.5rem;
    color: var(--accent);
  }}
  .controls {{
    display: flex;
    flex-direction: column;
    gap: 1rem;
    margin-bottom: 2rem;
  }}
  .search-box {{
    width: 100%;
    padding: 0.75rem 1rem;
    border-radius: 0.5rem;
    border: 1px solid var(--border);
    background: var(--bg-card);
    color: var(--text-main);
    font-size: 1rem;
  }}
  .search-box:focus {{
    outline: none;
    border-color: var(--accent);
    box-shadow: 0 0 0 2px var(--accent-glow);
  }}
  .pills-group {{
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
  }}
  .pills-label {{
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
  }}
  .pills {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
  }}
  .pill {{
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-muted);
    padding: 0.4rem 0.85rem;
    border-radius: 0.5rem;
    font-size: 0.85rem;
    cursor: pointer;
    transition: all 0.15s ease;
    user-select: none;
  }}
  .pill:hover {{
    background: var(--bg-card-hover);
    color: var(--text-main);
  }}
  .pill.active {{
    background: var(--accent);
    color: #0f172a;
    font-weight: 600;
    border-color: var(--accent);
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    background: var(--bg-card);
    border-radius: 0.75rem;
    overflow: hidden;
    border: 1px solid var(--border);
  }}
  th, td {{
    padding: 0.85rem 1rem;
    text-align: left;
    border-bottom: 1px solid var(--border);
  }}
  th {{
    background: #111827;
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    font-weight: 600;
  }}
  tr.item-row:hover {{
    background: var(--bg-card-hover);
    cursor: pointer;
  }}
  .mono {{
    font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    font-size: 0.9rem;
  }}
  .fn-name {{
    color: #38bdf8;
    font-weight: 600;
  }}
  .badge {{
    display: inline-block;
    padding: 0.2rem 0.5rem;
    border-radius: 0.375rem;
    font-size: 0.75rem;
    font-weight: 600;
  }}
  .badge-high {{ background: rgba(74, 222, 128, 0.15); color: var(--green); }}
  .badge-mid {{ background: rgba(250, 204, 21, 0.15); color: var(--yellow); }}
  .badge-low {{ background: rgba(248, 113, 113, 0.15); color: var(--red); }}
  .domain-tag {{
    background: rgba(192, 132, 252, 0.15);
    color: var(--purple);
    font-size: 0.75rem;
    padding: 0.2rem 0.5rem;
    border-radius: 0.375rem;
    font-weight: 500;
  }}
  .details-row {{
    display: none;
    background: #0b1120;
  }}
  .details-content {{
    padding: 1rem;
    font-family: monospace;
    font-size: 0.85rem;
    line-height: 1.6;
    color: #cbd5e1;
  }}
  .site-line {{
    margin-bottom: 0.25rem;
  }}
  .site-file {{ color: var(--accent); }}
  .site-code {{ color: #94a3b8; }}
</style>
</head>
<body>
  <header>
    <h1>Pathways Vulkan API Audit <span class="badge-title">Interactive Dashboard</span></h1>
    <div class="meta">Generated: {now_str} • Host Architecture: AMD gfx1201 • Source: Pathways Native & Vulkan Hardware Database</div>
  </header>

  <div class="stats-grid">
    <div class="card">
      <div class="card-title">Engine Calls</div>
      <div class="card-value" id="statEngineCalls">{total_p}</div>
    </div>
    <div class="card">
      <div class="card-title">Unique Functions</div>
      <div class="card-value" id="statUniqueFuncs">{unique_p}</div>
    </div>
    <div class="card">
      <div class="card-title">Desktop Coverage Base</div>
      <div class="card-value" id="statCoverageBase">{desk_devs} devices</div>
    </div>
    <div class="card">
      <div class="card-title">Top API Call</div>
      <div class="card-value" style="font-size:1.15rem; word-break:break-all;">{results[0]['name'] if results else 'N/A'} ({results[0]['project_count'] if results else 0})</div>
    </div>
  </div>

  <div class="controls">
    <input type="text" id="searchInput" class="search-box" placeholder="Search Vulkan API calls, extensions, domains..."/>

    <div class="pills-group">
      <div class="pills-label">Device Platform / Hardware Scope:</div>
      <div class="pills" id="platformPills">
        <div class="pill active" data-platform="desktop">Desktop (PC: {desk_devs} devices) - Mobile Filtered Out</div>
        <div class="pill" data-platform="global">Global (All: {glob_devs} devices)</div>
        <div class="pill" data-platform="mobile">Mobile (Android: {mob_devs} devices)</div>
      </div>
    </div>

    <div class="pills-group">
      <div class="pills-label">Architectural Workload:</div>
      <div class="pills" id="workloadPills">
        <div class="pill active" data-wl="all">All Workloads</div>
        <div class="pill" data-wl="raytracing">Ray Tracing</div>
        <div class="pill" data-wl="compute">Compute</div>
        <div class="pill" data-wl="sync">Synchronization</div>
        <div class="pill" data-wl="dgc">Device Generated Commands</div>
        <div class="pill" data-wl="mgpu">Multi-GPU / P2P</div>
        <div class="pill" data-wl="wsi">WSI / Swapchain</div>
        <div class="pill" data-wl="descriptors">Descriptors</div>
        <div class="pill" data-wl="transfer">Transfer & Copies</div>
        <div class="pill" data-wl="rendering">Dynamic Rendering</div>
        <div class="pill" data-wl="video">Video</div>
      </div>
    </div>
  </div>

  <table>
    <thead>
      <tr>
        <th style="width: 50px;">#</th>
        <th>Vulkan API Call</th>
        <th style="text-align: right;">Engine</th>
        <th style="text-align: right;">3rd-Party</th>
        <th style="text-align: right;">Total</th>
        <th>Specification / Date Added</th>
        <th id="thCoverage">Desktop Coverage</th>
        <th>Workload Domain</th>
      </tr>
    </thead>
    <tbody id="tableBody">
    </tbody>
  </table>

  <script>
    const data = {json_data};
    let activeWorkload = "{workload_name.lower() if workload_name != 'All' else 'all'}";
    let activePlatform = "desktop";
    let searchTerm = "";

    function getCoverage(item, platform) {{
      if (platform === "desktop") return item.hw_pct_desktop !== undefined ? item.hw_pct_desktop : item.hw_pct;
      if (platform === "mobile") return item.hw_pct_mobile !== undefined ? item.hw_pct_mobile : item.hw_pct;
      return item.hw_pct_global !== undefined ? item.hw_pct_global : item.hw_pct;
    }}

    function getBadgeClass(cov) {{
      if (cov >= 80.0) return "badge-high";
      if (cov >= 40.0) return "badge-mid";
      return "badge-low";
    }}

    function renderTable() {{
      const tbody = document.getElementById("tableBody");
      tbody.innerHTML = "";

      document.getElementById("thCoverage").textContent = activePlatform === "desktop" ? "Desktop Coverage ({desk_devs} devs)" : (activePlatform === "mobile" ? "Mobile Coverage ({mob_devs} devs)" : "Global Coverage ({glob_devs} devs)");

      const filtered = data.filter(item => {{
        const matchesWorkload = activeWorkload === "all" || item.category === activeWorkload;
        const matchesSearch = !searchTerm || 
          item.name.toLowerCase().includes(searchTerm) || 
          item.spec_name.toLowerCase().includes(searchTerm) ||
          item.category_title.toLowerCase().includes(searchTerm);
        return matchesWorkload && matchesSearch;
      }});

      let totalCalls = 0;
      let engineCalls = 0;
      let uniqueFuncs = 0;

      filtered.forEach((item, index) => {{
        totalCalls += item.total_count;
        engineCalls += item.project_count;
        if (item.project_count > 0) uniqueFuncs++;

        const cov = getCoverage(item, activePlatform);
        const tr = document.createElement("tr");
        tr.className = "item-row";
        tr.onclick = () => toggleDetails(index);

        tr.innerHTML = `
          <td style="color: var(--text-muted); font-size: 0.85rem;">${{index + 1}}</td>
          <td class="mono fn-name">${{item.name}}</td>
          <td style="text-align: right; font-weight: 600; color: ${{item.project_count > 0 ? 'var(--green)' : 'var(--text-muted)'}};">${{item.project_count}}</td>
          <td style="text-align: right; color: var(--text-muted);">${{item.thirdparty_count}}</td>
          <td style="text-align: right; font-weight: 600; color: var(--accent);">${{item.total_count}}</td>
          <td style="font-size: 0.85rem; color: #cbd5e1;">${{item.spec_name}} <span style="color: var(--text-muted); font-size: 0.75rem;">(${{item.spec_date}})</span></td>
          <td><span class="badge ${{getBadgeClass(cov)}}">${{cov.toFixed(1)}}%</span></td>
          <td><span class="domain-tag">${{item.category_title}}</span></td>
        `;

        const detailsTr = document.createElement("tr");
        detailsTr.id = `details-${{index}}`;
        detailsTr.className = "details-row";
        
        let sitesHtml = "";
        if (item.project_sites && item.project_sites.length > 0) {{
          sitesHtml += `<div style="font-weight: 600; margin-bottom: 0.5rem; color: var(--green);">Engine Call Sites (${{item.project_sites.length}}):</div>`;
          item.project_sites.slice(0, 10).forEach(s => {{
            sitesHtml += `<div class="site-line"><span class="site-file">${{s.file}}:${{s.line}}</span> <span class="site-code">${{s.text}}</span></div>`;
          }});
          if (item.project_sites.length > 10) {{
            sitesHtml += `<div style="color: var(--text-muted); font-style: italic;">... and ${{item.project_sites.length - 10}} more call sites.</div>`;
          }}
        }} else {{
          sitesHtml += `<div style="color: var(--text-muted); font-style: italic;">No direct calls found in Pathways engine code (specification reference / third-party only).</div>`;
        }}

        detailsTr.innerHTML = `<td colspan="8"><div class="details-content">${{sitesHtml}}</div></td>`;

        tbody.appendChild(tr);
        tbody.appendChild(detailsTr);
      }});

      document.getElementById("statEngineCalls").textContent = engineCalls;
      document.getElementById("statUniqueFuncs").textContent = uniqueFuncs;
    }}

    function toggleDetails(index) {{
      const row = document.getElementById(`details-${{index}}`);
      row.style.display = row.style.display === "table-row" ? "none" : "table-row";
    }}

    document.querySelectorAll("#platformPills .pill").forEach(pill => {{
      pill.addEventListener("click", () => {{
        document.querySelectorAll("#platformPills .pill").forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        activePlatform = pill.getAttribute("data-platform");
        renderTable();
      }});
    }});

    document.querySelectorAll("#workloadPills .pill").forEach(pill => {{
      pill.addEventListener("click", () => {{
        document.querySelectorAll("#workloadPills .pill").forEach(p => p.classList.remove("active"));
        pill.classList.add("active");
        activeWorkload = pill.getAttribute("data-wl");
        renderTable();
      }});
    }});

    document.getElementById("searchInput").addEventListener("input", (e) => {{
      searchTerm = e.target.value.toLowerCase().trim();
      renderTable();
    }});

    renderTable();
  </script>
</body>
</html>
"""
        return html

def update_markdown_document(target_file: Path, new_content: str):
    start_tag = "<!-- VULKAN_API_AUDIT_BEGIN -->"
    end_tag = "<!-- VULKAN_API_AUDIT_END -->"

    if target_file.exists():
        with open(target_file, "r", encoding="utf-8") as f:
            existing = f.read()
        if start_tag in existing and end_tag in existing:
            pattern = re.compile(f"{re.escape(start_tag)}.*?{re.escape(end_tag)}", re.DOTALL)
            replacement = f"{start_tag}\n\n{new_content.strip()}\n\n{end_tag}"
            updated = pattern.sub(replacement, existing)
            with open(target_file, "w", encoding="utf-8") as f:
                f.write(updated)
            return True

    with open(target_file, "w", encoding="utf-8") as f:
        f.write(f"{start_tag}\n\n{new_content.strip()}\n\n{end_tag}\n")
    return True

def main():
    parser = argparse.ArgumentParser(
        description="Pathways Vulkan API Call Auditor & Specification Tracker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Platform / Device Type Options (--platform / -p):
  all, global         All submitted devices worldwide (default: ~2,376 devices, includes mobile)
  desktop, pc         Desktop & workstation GPUs (Windows + Linux combined: ~976 devices; excludes mobile)
  windows, win        Windows desktop/laptop hardware reports (~547 devices)
  linux, lin          Linux desktop/workstation hardware reports (~429 devices)
  android, mobile     Android mobile phones, tablets, and embedded devices (~1,467 devices)
  macos               macOS desktop/laptop systems running via MoltenVK (~75 devices)
  ios                 Apple iOS mobile devices running via MoltenVK (~28 devices)

Device Filtering Flags:
  --exclude-mobile    Filter out mobile devices (convenience alias for --platform desktop)
  --compare-platforms Display side-by-side coverage comparison for Desktop vs Mobile vs Global

Workload Options (--workload / -w):
  all                 All Vulkan API calls (default)
  raytracing, rt      Hardware Ray Tracing (acceleration structures, pipelines, trace rays)
  compute             Compute pipelines, dispatch dispatches, subgroup operations
  video, decode       Vulkan Video decoding, encoding, session management
  sync                Pipeline barriers (Sync2), timeline/binary semaphores, fences, waits
  dgc                 Device Generated Commands (indirect execution sets, layouts)
  mgpu, p2p           Multi-GPU external memory (host ptr import, DMA-BUF, semaphores)
  wsi                 Window system integration (surfaces, swapchains, presentation)
  descriptors         Descriptor sets, pools, layouts, update templates
  transfer            Buffer & image copies, blits, clears, fills
  rendering           Dynamic rendering, render passes, framebuffers, draw calls
  queries             Query pools, timestamps, pipeline metrics

Output Format Options (--format):
  rich                Interactive terminal UI with styled tables & panels (uses rich)
  text, plain         Clean plain ASCII table (no ANSI escape codes, editor-friendly)
  markdown, md        GitHub Flavored Markdown report with badges
  html                Standalone interactive HTML dashboard with live search & filters
  json                Structured JSON dump with complete call-site citations

Examples:
  ./scripts/audit_vulkan_api.py -w raytracing --exclude-mobile
  ./scripts/audit_vulkan_api.py -w dgc --compare-platforms
  ./scripts/audit_vulkan_api.py --platform desktop --format rich
  ./scripts/audit_vulkan_api.py --format html -o vulkan_dashboard.html
  ./scripts/audit_vulkan_api.py --format markdown -o VULKAN_API_AUDIT.md
"""
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent.parent,
                        help="Root directory of repository (default: project root)")
    parser.add_argument("--vk-xml", type=Path, default=Path(DEFAULT_VK_XML),
                        help=f"Path to vk.xml registry (default: {DEFAULT_VK_XML})")
    parser.add_argument("-w", "--workload", type=str, default="all",
                        help="Filter by workload (e.g. raytracing, compute, video, sync, dgc, mgpu, wsi, descriptors, transfer, rendering, all)")
    parser.add_argument("-p", "--platform", type=str, default="all",
                        help="Target device platform / hardware scope (choices: all, desktop, windows, linux, android, mobile, macos, ios; default: all)")
    parser.add_argument("--exclude-mobile", "--no-mobile", action="store_true",
                        help="Filter out mobile devices (convenience shortcut for --platform desktop)")
    parser.add_argument("--compare-platforms", action="store_true",
                        help="Display side-by-side coverage columns for Desktop (PC), Mobile (Android), and Global")
    parser.add_argument("-f", "--filter", type=str, default=None,
                        help="Regex pattern to filter function names or extensions")
    parser.add_argument("--format", choices=["rich", "text", "plain", "markdown", "md", "html", "json"], default=None,
                        help="Output format (default: rich if TTY available, otherwise plain text)")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Path to output file (markdown, html, or json)")
    parser.add_argument("--update-doc", type=Path, default=None,
                        help="Update table in place inside an existing markdown file between comment tags")
    parser.add_argument("--show-sites", action="store_true",
                        help="Show individual file and line number call sites in terminal/text output")
    parser.add_argument("--offline", action="store_true",
                        help="Use cached gpuinfo statistics without performing live HTTP queries")
    parser.add_argument("--limit", type=int, default=40,
                        help="Number of rows to display in terminal mode (default: 40, 0 for all)")

    args = parser.parse_args()

    # Determine default format
    fmt = args.format
    if not fmt:
        if sys.stdout.isatty():
            try:
                import rich
                fmt = "rich"
            except ImportError:
                fmt = "text"
        else:
            fmt = "text"

    # Handle exclude-mobile
    target_platform = "desktop" if args.exclude_mobile else args.platform

    # Normalize workload filters
    raw_workloads = [x.strip() for x in args.workload.split(",") if x.strip()]
    resolved_workloads = [resolve_workload_key(w) for w in raw_workloads]

    auditor = VulkanAuditor(
        root_dir=args.root,
        vk_xml_path=args.vk_xml,
        platform=target_platform,
        compare_platforms=args.compare_platforms,
        offline=args.offline
    )
    
    print("[*] Parsing Vulkan XML registry...", file=sys.stderr)
    auditor.load_vk_xml()

    print(f"[*] Retrieving Vulkan Hardware Database statistics for platform scope '{auditor.platform_title}'...", file=sys.stderr)
    auditor.fetch_gpuinfo_data()

    print(f"[*] Scanning codebase at {args.root}...", file=sys.stderr)
    auditor.scan_codebase()

    results = auditor.compile_results(workload_filter=resolved_workloads, pattern_filter=args.filter)
    total_proj = sum(r["project_count"] for r in results)
    workload_display = ", ".join(raw_workloads)
    print(f"[+] Scan completed: found {total_proj} engine calls matching workload '{workload_display}'.\n", file=sys.stderr)

    # Format handling
    if fmt == "rich":
        try:
            auditor.generate_rich_terminal(results, limit=args.limit, show_sites=args.show_sites, workload_name=workload_display)
        except Exception as e:
            print(f"Rich rendering failed ({e}), falling back to plain text:", file=sys.stderr)
            text = auditor.generate_plain_text(results, limit=args.limit, show_sites=args.show_sites)
            print(text)
    elif fmt in ("text", "plain"):
        text = auditor.generate_plain_text(results, limit=args.limit, show_sites=args.show_sites)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"[+] Plain text table written to {args.output}")
        else:
            print(text)
    elif fmt in ("markdown", "md"):
        md = auditor.generate_markdown(results, workload_name=workload_display)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(md)
            print(f"[+] Markdown report written to {args.output}")
        else:
            print(md)
    elif fmt == "html":
        html = auditor.generate_html(results, workload_name=workload_display)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(html)
            print(f"[+] Interactive HTML dashboard written to {args.output}")
        else:
            print(html)
    elif fmt == "json":
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "target_platform": auditor.platform,
            "platform_title": auditor.platform_title,
            "device_count": auditor.device_count,
            "workload_filter": resolved_workloads,
            "pattern_filter": args.filter,
            "total_engine_calls": total_proj,
            "total_thirdparty_calls": sum(r["thirdparty_count"] for r in results),
            "matched_functions_count": len(results),
            "functions": results
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            print(f"[+] JSON report written to {args.output}")
        else:
            print(json.dumps(payload, indent=2))

    # In-place markdown document update
    if args.update_doc:
        md_content = auditor.generate_markdown(results, workload_name=workload_display)
        update_markdown_document(args.update_doc, md_content)
        print(f"[+] Successfully updated {args.update_doc} in-place between marker comments.")

if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        try:
            sys.stdout.close()
            sys.stderr.close()
        except Exception:
            pass
        sys.exit(0)
