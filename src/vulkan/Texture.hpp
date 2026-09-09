#pragma once

#include <vulkan/vulkan.h>
#include "vk_mem_alloc.h"
#include "vulkan/Image.hpp"
#include "vulkan/Buffer.hpp"
#include <memory>
#include <string>
#include <vector>

namespace pathways {

class Texture {
public:
    Texture(VkDevice device, VmaAllocator allocator, std::unique_ptr<Image> image, VkSampler sampler);
    ~Texture();

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    VkImage getImage() const { return m_image ? m_image->getImage() : VK_NULL_HANDLE; }
    VkImageView getImageView() const { return m_image ? m_image->getImageView() : VK_NULL_HANDLE; }
    VkSampler getSampler() const { return m_sampler; }
    VkDescriptorImageInfo getDescriptorInfo() const;

    uint32_t getWidth() const { return m_image ? m_image->getWidth() : 0; }
    uint32_t getHeight() const { return m_image ? m_image->getHeight() : 0; }

    static std::unique_ptr<Texture> createFromPixels(
        VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool,
        uint32_t width, uint32_t height, VkFormat format, const void* pixels, size_t dataSize,
        bool isHdr = false
    );

    static std::unique_ptr<Texture> createDummyWhite(
        VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool
    );

    static std::unique_ptr<Texture> createDummyNormal(
        VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool
    );

    static std::unique_ptr<Texture> createBlueNoise64(
        VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool
    );

    static std::unique_ptr<Texture> createProceduralHdrSky(
        VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool,
        uint32_t width = 512, uint32_t height = 256
    );

    static std::unique_ptr<Texture> loadFromFile(
        VkDevice device, VmaAllocator allocator, VkQueue queue, VkCommandPool pool,
        const std::string& filepath
    );

private:
    void release();

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    std::unique_ptr<Image> m_image;
    VkSampler m_sampler = VK_NULL_HANDLE;
};

} // namespace pathways
