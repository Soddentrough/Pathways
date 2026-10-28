#include "vulkan/Texture.hpp"
#include "core/Logger.hpp"
#include "assets/BlueNoise64.hpp"

#include "stb_image.h"

#define TINYEXR_USE_MINIZ 0
#include <zlib.h>
#include "tinyexr.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <filesystem>
#include <glm/glm.hpp>

namespace pathways {

Texture::Texture(VkDevice device, VmaAllocator allocator, std::unique_ptr<Image> image, VkSampler sampler)
    : m_device(device), m_allocator(allocator), m_image(std::move(image)), m_sampler(sampler) {
}

Texture::~Texture() {
    release();
}

Texture::Texture(Texture&& other) noexcept {
    m_device = other.m_device;
    m_allocator = other.m_allocator;
    m_image = std::move(other.m_image);
    m_sampler = other.m_sampler;

    other.m_device = VK_NULL_HANDLE;
    other.m_allocator = VK_NULL_HANDLE;
    other.m_sampler = VK_NULL_HANDLE;
}

Texture& Texture::operator=(Texture&& other) noexcept {
    if (this != &other) {
        release();
        m_device = other.m_device;
        m_allocator = other.m_allocator;
        m_image = std::move(other.m_image);
        m_sampler = other.m_sampler;

        other.m_device = VK_NULL_HANDLE;
        other.m_allocator = VK_NULL_HANDLE;
        other.m_sampler = VK_NULL_HANDLE;
    }
    return *this;
}

void Texture::release() {
    if (m_sampler != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
        vkDestroySampler(m_device, m_sampler, nullptr);
        m_sampler = VK_NULL_HANDLE;
    }
    m_image.reset();
}

VkDescriptorImageInfo Texture::getDescriptorInfo() const {
    VkDescriptorImageInfo info{};
    info.sampler = m_sampler;
    info.imageView = m_image ? m_image->getImageView() : VK_NULL_HANDLE;
    info.imageLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
    return info;
}

void Texture::updatePixelsAsync(
    VkCommandBuffer cmd,
    Buffer& stagingBuffer,
    const void* pixels,
    size_t dataSize
) {
    if (!pixels || dataSize == 0 || !m_image) return;

    stagingBuffer.copyFrom(pixels, dataSize);

    uint32_t width = m_image->getWidth();
    uint32_t height = m_image->getHeight();

    // Transition image: SHADER_READ_ONLY_OPTIMAL -> TRANSFER_DST_OPTIMAL
    m_image->transitionLayout(
        cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
    );

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(cmd, stagingBuffer.getBuffer(), m_image->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    // Transition image: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
    m_image->transitionLayout(
        cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
        VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR,
        VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
    );
}

std::unique_ptr<Texture> Texture::createFromPixels(
    VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool,
    uint32_t width, uint32_t height, VkFormat format, const void* pixels, size_t dataSize,
    bool isHdr, bool generateMips
) {
    if (!pixels || width == 0 || height == 0 || dataSize == 0) {
        throw std::runtime_error("Texture::createFromPixels: invalid pixel parameters!");
    }

    uint32_t mipLevels = (generateMips && (width > 1 || height > 1))
        ? (static_cast<uint32_t>(std::floor(std::log2(std::max(width, height)))) + 1u)
        : 1u;
    VkImageUsageFlags usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
    if (mipLevels > 1u) {
        usage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    }

    auto image = std::make_unique<Image>(
        device, allocator, width, height, format,
        usage, VK_IMAGE_ASPECT_COLOR_BIT, std::vector<uint32_t>{}, mipLevels
    );

    // Staging buffer
    Buffer staging(
        allocator, dataSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );
    staging.copyFrom(pixels, dataSize);

    // Command buffer for transfer
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = pool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(device, &allocInfo, &cmd);

    VkCommandBufferBeginInfo beginInfo{};
    beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    // Transition image (all mip levels): UNDEFINED -> TRANSFER_DST_OPTIMAL
    image->transitionLayout(
        cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
        VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
        VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT
    );

    VkBufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = {0, 0, 0};
    region.imageExtent = {width, height, 1};

    vkCmdCopyBufferToImage(cmd, staging.getBuffer(), image->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);

    if (mipLevels > 1u) {
        int32_t mipWidth = static_cast<int32_t>(width);
        int32_t mipHeight = static_cast<int32_t>(height);

        for (uint32_t i = 1; i < mipLevels; ++i) {
            VkImageMemoryBarrier2 barrier{};
            barrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            barrier.image = image->getImage();
            barrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            barrier.subresourceRange.baseMipLevel = i - 1;
            barrier.subresourceRange.levelCount = 1;
            barrier.subresourceRange.baseArrayLayer = 0;
            barrier.subresourceRange.layerCount = 1;

            VkDependencyInfo depInfo{};
            depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
            depInfo.imageMemoryBarrierCount = 1;
            depInfo.pImageMemoryBarriers = &barrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);

            int32_t nextWidth = mipWidth > 1 ? mipWidth / 2 : 1;
            int32_t nextHeight = mipHeight > 1 ? mipHeight / 2 : 1;

            VkImageBlit blit{};
            blit.srcOffsets[0] = {0, 0, 0};
            blit.srcOffsets[1] = {mipWidth, mipHeight, 1};
            blit.srcSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.srcSubresource.mipLevel = i - 1;
            blit.srcSubresource.baseArrayLayer = 0;
            blit.srcSubresource.layerCount = 1;
            blit.dstOffsets[0] = {0, 0, 0};
            blit.dstOffsets[1] = {nextWidth, nextHeight, 1};
            blit.dstSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            blit.dstSubresource.mipLevel = i;
            blit.dstSubresource.baseArrayLayer = 0;
            blit.dstSubresource.layerCount = 1;

            vkCmdBlitImage(
                cmd,
                image->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                image->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1, &blit, VK_FILTER_LINEAR
            );

            barrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            barrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            barrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            barrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
            barrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL;
            barrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
            vkCmdPipelineBarrier2(cmd, &depInfo);

            mipWidth = nextWidth;
            mipHeight = nextHeight;
        }

        VkImageMemoryBarrier2 lastBarrier{};
        lastBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        lastBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
        lastBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        lastBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        lastBarrier.dstAccessMask = VK_ACCESS_2_SHADER_SAMPLED_READ_BIT;
        lastBarrier.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        lastBarrier.newLayout = VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL;
        lastBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        lastBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        lastBarrier.image = image->getImage();
        lastBarrier.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        lastBarrier.subresourceRange.baseMipLevel = mipLevels - 1;
        lastBarrier.subresourceRange.levelCount = 1;
        lastBarrier.subresourceRange.baseArrayLayer = 0;
        lastBarrier.subresourceRange.layerCount = 1;

        VkDependencyInfo lastDepInfo{};
        lastDepInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        lastDepInfo.imageMemoryBarrierCount = 1;
        lastDepInfo.pImageMemoryBarriers = &lastBarrier;
        vkCmdPipelineBarrier2(cmd, &lastDepInfo);
        image->setLayout(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    } else {
        // Transition image: TRANSFER_DST_OPTIMAL -> SHADER_READ_ONLY_OPTIMAL
        image->transitionLayout(
            cmd, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_SAMPLED_READ_BIT
        );
    }

    vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = cmd;

    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submitInfo.commandBufferInfoCount = 1;
    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;

    vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(device, pool, 1, &cmd);

    // Create Sampler
    VkSamplerCreateInfo samplerInfo{};
    samplerInfo.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO;
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeV = isHdr ? VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE : VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    samplerInfo.anisotropyEnable = VK_FALSE;
    samplerInfo.maxAnisotropy = 1.0f;
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = static_cast<float>(mipLevels);
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_BLACK;
    samplerInfo.unnormalizedCoordinates = VK_FALSE;
    samplerInfo.compareEnable = VK_FALSE;

    VkSampler sampler = VK_NULL_HANDLE;
    VkResult res = vkCreateSampler(device, &samplerInfo, nullptr, &sampler);
    if (res != VK_SUCCESS) {
        throw std::runtime_error("Failed to create texture sampler!");
    }

    return std::make_unique<Texture>(device, allocator, std::move(image), sampler);
}

std::unique_ptr<Texture> Texture::createDummyWhite(
    VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool
) {
    uint8_t white[4] = {255, 255, 255, 255};
    return createFromPixels(device, allocator, queue, pool, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, white, sizeof(white), false);
}

std::unique_ptr<Texture> Texture::createDummyNormal(
    VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool
) {
    // Normal vector [0, 0, 1] encoded into [128, 128, 255, 255]
    uint8_t flatNormal[4] = {128, 128, 255, 255};
    return createFromPixels(device, allocator, queue, pool, 1, 1, VK_FORMAT_R8G8B8A8_UNORM, flatNormal, sizeof(flatNormal), false);
}

std::unique_ptr<Texture> Texture::createBlueNoise64(
    VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool
) {
    return createFromPixels(device, allocator, queue, pool,
                            BLUE_NOISE_WIDTH, BLUE_NOISE_HEIGHT,
                            VK_FORMAT_R8G8B8A8_UNORM,
                            BLUE_NOISE_64X64_RGBA8.data(),
                            BLUE_NOISE_BYTES, false);
}

std::unique_ptr<Texture> Texture::createProceduralHdrSky(
    VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool,
    uint32_t width, uint32_t height
) {
    std::vector<glm::vec4> pixels(width * height);
    const float PI = 3.14159265358979323846f;
    glm::vec3 sunDir = glm::normalize(glm::vec3(0.5f, 0.7f, 0.5f));
    glm::vec3 sunColor = glm::vec3(160.0f, 140.0f, 115.0f); // High dynamic range radiance

    for (uint32_t y = 0; y < height; ++y) {
        float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
        float theta = v * PI;
        float cosTheta = std::cos(theta);
        float sinTheta = std::sin(theta);

        for (uint32_t x = 0; x < width; ++x) {
            float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
            float phi = (u * 2.0f - 1.0f) * PI;

            glm::vec3 dir = glm::normalize(glm::vec3(
                sinTheta * std::cos(phi),
                cosTheta,
                sinTheta * std::sin(phi)
            ));

            glm::vec3 radiance(0.0f);
            float cosSun = glm::dot(dir, sunDir);

            if (dir.y > 0.0f) {
                // Atmospheric Rayleigh scattering gradient
                float skyFactor = std::pow(dir.y, 0.45f);
                glm::vec3 zenithColor(0.08f, 0.18f, 0.45f);
                glm::vec3 horizonColor(0.65f, 0.55f, 0.45f);
                radiance = glm::mix(horizonColor, zenithColor, skyFactor);

                // Mie forward scattering corona
                float g = 0.85f;
                float mie = 0.12f * (1.0f - g * g) / std::pow(1.0f + g * g - 2.0f * g * cosSun, 1.5f);
                radiance += sunColor * mie * 0.015f;

                // Sun disk (approx 0.53 degrees angular diameter -> cos ~ 0.99989)
                if (cosSun > 0.99985f) {
                    radiance += sunColor;
                }
            } else {
                // Atmospheric aerial perspective haze smoothly blending horizon into earth ground reflection
                float groundFactor = std::clamp(-dir.y, 0.0f, 1.0f);
                glm::vec3 deepGround = glm::mix(glm::vec3(0.08f, 0.07f, 0.06f), glm::vec3(0.02f, 0.02f, 0.025f), groundFactor);
                glm::vec3 horizonHaze(0.65f, 0.55f, 0.45f);
                float haze = std::exp(-groundFactor * 14.0f);
                radiance = glm::mix(deepGround, horizonHaze, haze);
            }

            pixels[y * width + x] = glm::vec4(radiance, 1.0f);
        }
    }

    size_t byteSize = pixels.size() * sizeof(glm::vec4);
    return createFromPixels(
        device, allocator, queue, pool,
        width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
        pixels.data(), byteSize, true
    );
}

std::unique_ptr<Texture> Texture::createProceduralNightHdrSky(
    VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool,
    uint32_t width, uint32_t height
) {
    std::vector<glm::vec4> pixels(width * height);
    const float PI = 3.14159265358979323846f;
    glm::vec3 moonDir = glm::normalize(glm::vec3(-0.45f, 0.65f, -0.60f));

    for (uint32_t y = 0; y < height; ++y) {
        float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
        float theta = v * PI;
        float cosTheta = std::cos(theta);
        float sinTheta = std::sin(theta);

        for (uint32_t x = 0; x < width; ++x) {
            float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
            float phi = (u * 2.0f - 1.0f) * PI;

            glm::vec3 dir = glm::normalize(glm::vec3(
                sinTheta * std::cos(phi),
                cosTheta,
                sinTheta * std::sin(phi)
            ));

            glm::vec3 radiance(0.0f);

            if (dir.y > 0.0f) {
                // 1. Base cosmic midnight sky gradient (zenith is obsidian navy)
                float skyFactor = std::pow(dir.y, 0.55f);
                glm::vec3 zenithColor(0.0004f, 0.0008f, 0.0022f);
                glm::vec3 horizonColor(0.0080f, 0.0050f, 0.0150f);
                radiance = glm::mix(horizonColor, zenithColor, skyFactor);

                // 2. Urban horizon light pollution (cyber cyan, magenta, and amber glows from the megacity)
                float haze = std::exp(-dir.y * 11.0f);
                float cyanGlow = std::max(0.0f, std::sin(phi * 2.0f + 0.8f));
                float magentaGlow = std::max(0.0f, std::cos(phi * 3.0f - 1.2f));
                float amberGlow = std::max(0.0f, std::sin(phi * 1.5f + 2.5f));
                glm::vec3 cityLight = (glm::vec3(0.006f, 0.022f, 0.032f) * cyanGlow +
                                       glm::vec3(0.028f, 0.004f, 0.018f) * magentaGlow +
                                       glm::vec3(0.022f, 0.014f, 0.003f) * amberGlow) * haze;
                radiance += cityLight;

                // 3. Cyber Lunar Body (HDR Moon) with limb-darkening and atmospheric Mie halo
                float cosMoon = glm::dot(dir, moonDir);
                if (cosMoon > 0.9996f) {
                    float limb = std::pow((cosMoon - 0.9996f) / (1.0f - 0.9996f), 0.35f);
                    radiance += glm::vec3(14.0f, 15.5f, 18.0f) * (0.4f + 0.6f * limb);
                }
                float g = 0.94f;
                float corona = 0.06f * (1.0f - g * g) / std::pow(1.0f + g * g - 2.0f * g * cosMoon, 1.5f);
                radiance += glm::vec3(0.04f, 0.07f, 0.12f) * corona;
            } else {
                // Ground reflection / dark asphalt & foundation albedo
                float groundFactor = std::clamp(-dir.y, 0.0f, 1.0f);
                radiance = glm::mix(glm::vec3(0.005f, 0.006f, 0.009f), glm::vec3(0.001f, 0.001f, 0.002f), groundFactor);
            }

            pixels[y * width + x] = glm::vec4(radiance, 1.0f);
        }
    }

    // 4. Procedural Starfield: 2,400 deterministic stars with continuous sub-pixel Gaussian falloff
    // Completely eliminates blocky/square pixel artifacts and cell boundary clipping
    uint32_t rngState = 0x853c49e6u;
    auto lcg = [&rngState]() -> float {
        rngState = rngState * 1664525u + 1013904223u;
        return static_cast<float>(rngState & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
    };

    for (int s = 0; s < 2400; ++s) {
        float u = lcg();
        float elev = lcg();
        if (elev < 0.04f) continue; // Keep above immediate horizon haze
        float v = (1.0f - elev) * 0.5f; // Upper hemisphere: v in [0, 0.48]

        float phi = (u * 2.0f - 1.0f) * PI;
        float theta = v * PI;
        glm::vec3 sDir(
            std::sin(theta) * std::cos(phi),
            std::cos(theta),
            std::sin(theta) * std::sin(phi)
        );

        if (glm::dot(sDir, moonDir) > 0.995f) continue;

        float magRnd = lcg();
        float brightness = 0.25f + 1.25f * std::pow(magRnd, 4.0f);

        float colorRnd = lcg();
        glm::vec3 starColor;
        if (colorRnd < 0.45f) {
            starColor = glm::vec3(0.85f, 0.95f, 1.4f); // B-class blue-white
        } else if (colorRnd < 0.75f) {
            starColor = glm::vec3(1.0f, 1.0f, 1.0f);   // A/F-class pure white
        } else if (colorRnd < 0.92f) {
            starColor = glm::vec3(1.3f, 1.1f, 0.8f);   // G/K-class warm gold
        } else {
            starColor = glm::vec3(1.5f, 0.7f, 0.5f);   // M-class crimson giant
        }

        float cx = u * static_cast<float>(width);
        float cy = v * static_cast<float>(height);
        float sigma = 0.50f + 0.15f * (brightness > 1.0f ? 1.0f : 0.0f);
        float twoSigmaSq = 2.0f * sigma * sigma;
        float maxR = 2.4f * sigma;

        int cyi = static_cast<int>(cy);
        int cxi = static_cast<int>(cx);
        int ky = std::min(static_cast<int>(std::ceil(maxR)), 3);

        for (int dy = -ky; dy <= ky; ++dy) {
            int py = cyi + dy;
            if (py < 0 || py >= static_cast<int>(height)) continue;
            float latFactor = std::max(0.15f, std::sin(static_cast<float>(py + 0.5f) / static_cast<float>(height) * PI));
            float dY = static_cast<float>(py) + 0.5f - cy;

            int curKx = std::min(static_cast<int>(std::ceil(maxR / latFactor)), 6);
            for (int dx = -curKx; dx <= curKx; ++dx) {
                int px = (cxi + dx + static_cast<int>(width)) % static_cast<int>(width);
                float dX = (static_cast<float>(cxi + dx) + 0.5f - cx) * latFactor;
                float dist = std::sqrt(dX * dX + dY * dY);
                if (dist >= maxR) continue;

                // Hann-windowed Gaussian taper: continuously and smoothly vanishes to 0 at maxR
                float window = 0.5f * (1.0f + std::cos(PI * dist / maxR));
                float w = std::exp(-dist * dist / twoSigmaSq) * window;
                pixels[py * width + px] += glm::vec4(starColor * (brightness * w), 0.0f);
            }
        }
    }

    size_t byteSize = pixels.size() * sizeof(glm::vec4);
    return createFromPixels(
        device, allocator, queue, pool,
        width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
        pixels.data(), byteSize, true
    );
}

std::unique_ptr<Texture> Texture::createProceduralStudioHdrSky(
    VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool,
    uint32_t width, uint32_t height
) {
    std::vector<glm::vec4> pixels(width * height);
    const float PI = 3.14159265358979323846f;

    for (uint32_t y = 0; y < height; ++y) {
        float v = (static_cast<float>(y) + 0.5f) / static_cast<float>(height);
        float theta = v * PI;
        float cosTheta = std::cos(theta);
        float sinTheta = std::sin(theta);

        for (uint32_t x = 0; x < width; ++x) {
            float u = (static_cast<float>(x) + 0.5f) / static_cast<float>(width);
            float phi = (u * 2.0f - 1.0f) * PI;

            glm::vec3 dir = glm::normalize(glm::vec3(
                sinTheta * std::cos(phi),
                cosTheta,
                sinTheta * std::sin(phi)
            ));

            glm::vec3 radiance(0.0f);

            if (dir.y > 0.0f) {
                // Subtle dark studio ambient gradient with deep midnight slate ceiling
                float skyFactor = std::pow(dir.y, 0.6f);
                glm::vec3 zenithColor(0.0018f, 0.0022f, 0.0040f);
                glm::vec3 horizonColor(0.0055f, 0.0065f, 0.0090f);
                radiance = glm::mix(horizonColor, zenithColor, skyFactor);
            } else {
                // Dark matte ground / studio floor reflection
                float groundFactor = std::clamp(-dir.y, 0.0f, 1.0f);
                glm::vec3 deepGround(0.0008f, 0.0008f, 0.0012f);
                glm::vec3 horizonHaze(0.0055f, 0.0065f, 0.0090f);
                radiance = glm::mix(horizonHaze, deepGround, groundFactor);
            }

            pixels[y * width + x] = glm::vec4(radiance, 1.0f);
        }
    }

    // Procedural Starry Void: 1,600 subtle deterministic stars in upper hemisphere
    // Provides an authentic, deep cosmic starry backdrop outside the mirror gallery void
    uint32_t rngState = 0x9e3779b9u;
    auto lcg = [&rngState]() -> float {
        rngState = rngState * 1664525u + 1013904223u;
        return static_cast<float>(rngState & 0x00FFFFFFu) / static_cast<float>(0x01000000u);
    };

    for (int s = 0; s < 1600; ++s) {
        float u = lcg();
        float elev = lcg();
        if (elev < 0.03f) continue;
        float v = (1.0f - elev) * 0.5f; // Upper hemisphere: v in [0, 0.485]

        float magRnd = lcg();
        // Subtle star brightness between 0.15 and 0.85 (delicate twinkling without caustic fireflies)
        float brightness = 0.15f + 0.65f * std::pow(magRnd, 3.0f);

        float colorRnd = lcg();
        glm::vec3 starColor;
        if (colorRnd < 0.50f) {
            starColor = glm::vec3(0.85f, 0.95f, 1.30f); // Blue-white
        } else if (colorRnd < 0.80f) {
            starColor = glm::vec3(1.0f, 1.0f, 1.0f);   // Pure white
        } else {
            starColor = glm::vec3(1.20f, 1.10f, 0.85f); // Warm starlight
        }

        float cx = u * static_cast<float>(width);
        float cy = v * static_cast<float>(height);
        float sigma = 0.45f + 0.10f * (brightness > 0.5f ? 1.0f : 0.0f);
        float twoSigmaSq = 2.0f * sigma * sigma;
        float maxR = 2.2f * sigma;

        int cyi = static_cast<int>(cy);
        int cxi = static_cast<int>(cx);
        int ky = std::min(static_cast<int>(std::ceil(maxR)), 3);

        for (int dy = -ky; dy <= ky; ++dy) {
            int py = cyi + dy;
            if (py < 0 || py >= static_cast<int>(height)) continue;
            float latFactor = std::max(0.15f, std::sin(static_cast<float>(py + 0.5f) / static_cast<float>(height) * PI));
            float dY = static_cast<float>(py) + 0.5f - cy;

            int curKx = std::min(static_cast<int>(std::ceil(maxR / latFactor)), 5);
            for (int dx = -curKx; dx <= curKx; ++dx) {
                int px = (cxi + dx + static_cast<int>(width)) % static_cast<int>(width);
                float dX = (static_cast<float>(cxi + dx) + 0.5f - cx) * latFactor;
                float dist = std::sqrt(dX * dX + dY * dY);
                if (dist >= maxR) continue;

                // Hann-windowed Gaussian taper: continuously and smoothly vanishes to 0 at maxR
                float window = 0.5f * (1.0f + std::cos(PI * dist / maxR));
                float w = std::exp(-dist * dist / twoSigmaSq) * window;
                pixels[py * width + px] += glm::vec4(starColor * (brightness * w), 0.0f);
            }
        }
    }

    size_t byteSize = pixels.size() * sizeof(glm::vec4);
    return createFromPixels(
        device, allocator, queue, pool,
        width, height, VK_FORMAT_R32G32B32A32_SFLOAT,
        pixels.data(), byteSize, true
    );
}

std::unique_ptr<Texture> Texture::loadFromFile(
    VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool,
    const std::string& filepath
) {
    if (!std::filesystem::exists(filepath)) {
        Logger::warn("Texture file not found: {}", filepath);
        return nullptr;
    }

    std::string ext = std::filesystem::path(filepath).extension().string();
    for (auto& c : ext) c = std::tolower(c);

    int w = 0, h = 0, comp = 0;
    if (ext == ".hdr") {
        float* data = stbi_loadf(filepath.c_str(), &w, &h, &comp, 4);
        if (!data) {
            Logger::error("Failed to load HDR image: {} ({})", filepath, stbi_failure_reason());
            return nullptr;
        }
        size_t byteSize = static_cast<size_t>(w) * h * 4 * sizeof(float);
        auto tex = createFromPixels(
            device, allocator, queue, pool,
            static_cast<uint32_t>(w), static_cast<uint32_t>(h),
            VK_FORMAT_R32G32B32A32_SFLOAT, data, byteSize, true
        );
        stbi_image_free(data);
        Logger::info("Loaded HDRI environment map: {} ({}x{}, 32-bit Float)", filepath, w, h);
        return tex;
    } else if (ext == ".exr") {
        float* rgba = nullptr;
        const char* err = nullptr;
        int ret = LoadEXR(&rgba, &w, &h, filepath.c_str(), &err);
        if (ret != TINYEXR_SUCCESS || !rgba || w <= 0 || h <= 0) {
            Logger::error("Failed to load EXR image: {} ({})", filepath, err ? err : "unknown error");
            if (err) FreeEXRErrorMessage(err);
            return nullptr;
        }
        size_t byteSize = static_cast<size_t>(w) * h * 4 * sizeof(float);
        auto tex = createFromPixels(
            device, allocator, queue, pool,
            static_cast<uint32_t>(w), static_cast<uint32_t>(h),
            VK_FORMAT_R32G32B32A32_SFLOAT, rgba, byteSize, true
        );
        free(rgba);
        Logger::info("Loaded EXR environment map: {} ({}x{}, 32-bit Float)", filepath, w, h);
        return tex;
    } else {
        stbi_uc* data = stbi_load(filepath.c_str(), &w, &h, &comp, 4);
        if (!data) {
            Logger::error("Failed to load LDR texture image: {} ({})", filepath, stbi_failure_reason());
            return nullptr;
        }
        size_t byteSize = static_cast<size_t>(w) * h * 4;
        auto tex = createFromPixels(
            device, allocator, queue, pool,
            static_cast<uint32_t>(w), static_cast<uint32_t>(h),
            VK_FORMAT_R8G8B8A8_UNORM, data, byteSize, false, (w >= 64 && h >= 64)
        );
        stbi_image_free(data);
        Logger::info("Loaded texture image: {} ({}x{}, RGBA8)", filepath, w, h);
        return tex;
    }
}

std::unique_ptr<Texture> Texture::createSceneEnvironmentMap(
    VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool,
    const std::string& hdriPath, const std::string& scenePath,
    const std::string& domeLightHdriPath
) {
    std::unique_ptr<Texture> envMap = nullptr;

    // 1. Explicit user-provided or scene-associated HDRI path
    if (!hdriPath.empty() && hdriPath != "night" && hdriPath != "night-sky") {
        envMap = Texture::loadFromFile(device, allocator, queue, pool, hdriPath);
    }

    // 2. USD DomeLight HDRI
    if (!envMap && !domeLightHdriPath.empty()) {
        envMap = Texture::loadFromFile(device, allocator, queue, pool, domeLightHdriPath);
        if (envMap) {
            Logger::info("Loaded USD DomeLight HDRI from UsdLuxDomeLight schema: {}", domeLightHdriPath);
        }
    }

    // 3. Procedural Sky Dome selection based on scene type
    if (!envMap) {
        bool isCyber = (hdriPath == "night" || hdriPath == "night-sky" ||
                        scenePath == "cyber-city" || scenePath == "procedural:cyber-city" ||
                        scenePath == "procedural:cyber_city" || scenePath == "cyber_city" ||
                        scenePath == "Procedural Cyber City" || scenePath == "Cyber City");
        bool isInfinityMirror = (hdriPath == "studio" || hdriPath == "dark" ||
                                 scenePath == "infinity-mirror" || scenePath == "procedural:infinity-mirror" ||
                                 scenePath == "procedural:infinity_mirror" || scenePath == "infinity_mirror" ||
                                 scenePath == "Procedural Infinity Mirror" || scenePath == "Infinity Mirror");
        if (isCyber) {
            envMap = Texture::createProceduralNightHdrSky(device, allocator, queue, pool);
            Logger::info("Initialized procedural Cyber Night HDRI sky dome (1024x512, 32-bit Float).");
        } else if (isInfinityMirror) {
            envMap = Texture::createProceduralStudioHdrSky(device, allocator, queue, pool);
            Logger::info("Initialized procedural Starry Void HDRI environment (1024x512, 32-bit Float, Subtle Stars).");
        } else {
            envMap = Texture::createProceduralHdrSky(device, allocator, queue, pool);
            Logger::info("Initialized procedural Day HDRI sky dome (512x256, 32-bit Float).");
        }
    }

    return envMap;
}

} // namespace pathways
