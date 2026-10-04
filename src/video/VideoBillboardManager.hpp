#pragma once

#include "video/VideoDecoder.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Texture.hpp"
#include "scene/ProceduralScene.hpp"

#include <memory>
#include <array>
#include <string>
#include <vector>
#include <vulkan/vulkan.h>

namespace pathways {

class VideoBillboardManager {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    VideoBillboardManager() = default;
    ~VideoBillboardManager() = default;

    VideoBillboardManager(const VideoBillboardManager&) = delete;
    VideoBillboardManager& operator=(const VideoBillboardManager&) = delete;

    /// Initializes video decoder if the scene contains holographic video billboards.
    /// Allocates per-frame staging buffers for asynchronous GPU uploads.
    void init(const std::string& scenePath, const SceneData& sceneData, VmaAllocator allocator);

    /// Advances video playback by dt (in seconds) if video is active.
    void update(float dt);

    /// Checks if a new video frame has been decoded and is ready for GPU transfer.
    bool hasNewFrame() const;

    /// Checks if video decoder is open and ready.
    bool isOpen() const;

    /// Uploads decoded frame pixels to target texture asynchronously using the per-frame staging buffer.
    void uploadFrame(VkCommandBuffer cmd, uint32_t frameSlot, Texture* targetTexture);

    /// Cleans up resources.
    void reset();

    VideoDecoder* getDecoder() const { return m_videoDecoder.get(); }

private:
    std::unique_ptr<VideoDecoder> m_videoDecoder;
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_videoStagingBuffers;
};

} // namespace pathways
