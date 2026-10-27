#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <string>
#include "core/Config.hpp"

namespace pathways {

struct UpscalerDispatchDesc {
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkImageView colorIn = VK_NULL_HANDLE;         // Internal render resolution HDR radiance (or denoised output)
    VkImageView depthIn = VK_NULL_HANDLE;         // Internal render resolution linear/device depth
    VkImageView motionVectorsIn = VK_NULL_HANDLE; // Internal render resolution motion vectors (RGBA16F 2.5D: .rg = 2D MV, .b = depth delta)
    VkImageView reactiveMaskIn = VK_NULL_HANDLE;  // Optional R8_UNORM reactive mask (particles, alpha-blended / specular changes)
    VkImageView transparencyMaskIn = VK_NULL_HANDLE; // Optional R8_UNORM transparency & composition mask
    VkImageView colorOut = VK_NULL_HANDLE;        // Display resolution reconstructed radiance
    uint32_t renderWidth = 0;
    uint32_t renderHeight = 0;
    uint32_t displayWidth = 0;
    uint32_t displayHeight = 0;
    float jitterX = 0.0f;
    float jitterY = 0.0f;
    float cameraNear = 0.05f;                     // Camera near plane distance (meters) for FfxApiDispatchDescUpscale
    float cameraFar = 10000.0f;                   // Camera far plane distance (meters) for FfxApiDispatchDescUpscale
    float cameraFovY = 0.785398f;                 // Camera vertical FOV in radians for FfxApiDispatchDescUpscale
    float frameTimeDeltaMs = 16.667f;             // Frame delta time in milliseconds for FfxApiDispatchDescUpscale
    bool enableSharpening = false;                // Enable an additional sharpening pass [default: false]
    float sharpness = 0.0f;                       // Contrast-adaptive sharpening factor [0.0 - 1.0] [default: 0.0]
    float temporalWeight = 0.88f;                 // History blend weight [0.0 - 1.0]
    bool resetHistory = false;
    bool cameraMoved = false;
    uint32_t frameIndex = 0;
    uint32_t totalSamples = 1;
    bool inputIsNormalized = false;
};

class IUpscaler {
public:
    virtual ~IUpscaler() = default;

    virtual void resize(uint32_t renderW, uint32_t renderH, uint32_t displayW, uint32_t displayH) = 0;
    virtual void recordUpscale(const UpscalerDispatchDesc& desc) = 0;
    virtual UpscalerMode getMode() const = 0;
    virtual const char* getName() const = 0;
    virtual bool isTemporal() const = 0;
};

} // namespace pathways
