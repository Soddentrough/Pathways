#pragma once

#include "vulkan/Image.hpp"
#include "core/Config.hpp"

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <memory>
#include <array>
#include <vector>

namespace pathways {

class RenderTargetManager {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    RenderTargetManager(VkDevice device, VmaAllocator allocator, VkQueue queue);
    ~RenderTargetManager();

    RenderTargetManager(const RenderTargetManager&) = delete;
    RenderTargetManager& operator=(const RenderTargetManager&) = delete;

    void createRenderTargets(uint32_t width, uint32_t height, AccumFormat accumFormat,
                             bool isHdr, VkFormat swapchainFormat,
                             const std::vector<uint32_t>& concurrentQueues,
                             VkCommandBuffer cmd);

    void recreateAccumImages(uint32_t width, uint32_t height, AccumFormat accumFormat,
                             const std::vector<uint32_t>& concurrentQueues,
                             VkCommandBuffer cmd);

    void createGBufferResources(uint32_t width, uint32_t height,
                                const std::vector<uint32_t>& concurrentQueues,
                                VkCommandBuffer cmd);
    void destroyGBufferResources();

    void resize(uint32_t width, uint32_t height, AccumFormat accumFormat,
                bool isHdr, VkFormat swapchainFormat,
                const std::vector<uint32_t>& concurrentQueues,
                VkCommandBuffer cmd);

    void destroy();

    [[nodiscard]] Image* getFrameImage(uint32_t slot) const { return m_frameImages[slot].get(); }
    [[nodiscard]] const std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT>& getFrameImages() const { return m_frameImages; }
    [[nodiscard]] std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT>& getFrameImages() { return m_frameImages; }

    [[nodiscard]] Image* getAccumImage() const { return m_accumImage.get(); }
    [[nodiscard]] Image* getOutputImage() const { return m_outputImage.get(); }
    [[nodiscard]] Image* getMotionVectorImage() const { return m_motionVectorImage.get(); }

    [[nodiscard]] Image* getDirectLightImage() const { return m_directLightImage.get(); }
    [[nodiscard]] Image* getNormalDepthImage() const { return m_normalDepthImage.get(); }
    [[nodiscard]] Image* getPrevNormalDepthImage() const { return m_prevNormalDepthImage.get(); }

    [[nodiscard]] Image* getMlAlbedoRoughnessImage() const { return m_mlAlbedoRoughnessImage.get(); }
    [[nodiscard]] Image* getMlSpecularMotionImage() const { return m_mlSpecularMotionImage.get(); }
    [[nodiscard]] Image* getMlDiffuseImage() const { return m_mlDiffuseImage.get(); }
    [[nodiscard]] Image* getMlSpecularImage() const { return m_mlSpecularImage.get(); }

private:
    void transitionAllToGeneral(VkCommandBuffer cmd);

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    VkQueue m_queue = VK_NULL_HANDLE;

    std::array<std::unique_ptr<Image>, MAX_FRAMES_IN_FLIGHT> m_frameImages;
    std::unique_ptr<Image> m_accumImage;
    std::unique_ptr<Image> m_outputImage;
    std::unique_ptr<Image> m_motionVectorImage;

    std::unique_ptr<Image> m_directLightImage;
    std::unique_ptr<Image> m_normalDepthImage;
    std::unique_ptr<Image> m_prevNormalDepthImage;

    std::unique_ptr<Image> m_mlAlbedoRoughnessImage;
    std::unique_ptr<Image> m_mlSpecularMotionImage;
    std::unique_ptr<Image> m_mlDiffuseImage;
    std::unique_ptr<Image> m_mlSpecularImage;
};

} // namespace pathways
