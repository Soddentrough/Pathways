#include "video/VideoBillboardManager.hpp"
#include "core/Logger.hpp"
#include "scene/Material.hpp"

#include <filesystem>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#endif

namespace pathways {

void VideoBillboardManager::init(const std::string& scenePath, const SceneData& sceneData, VmaAllocator allocator) {
    bool isCyberCity = (scenePath == "cyber-city" || scenePath == "cyber_city" ||
                        scenePath == "procedural:cyber-city" || scenePath == "procedural:cyber_city" ||
                        scenePath == "Procedural Cyber City");
    if (!isCyberCity) {
        for (const auto& mat : sceneData.materials) {
            if ((mat.type & MATERIAL_FLAG_HOLO_VIDEO) != 0) {
                isCyberCity = true;
                break;
            }
        }
    }

    if (!isCyberCity) {
        reset();
        return;
    }

    std::filesystem::path exeDir;
#ifdef _WIN32
    char exePathBuf[MAX_PATH] = {0};
    if (GetModuleFileNameA(NULL, exePathBuf, MAX_PATH)) {
        exeDir = std::filesystem::path(exePathBuf).parent_path();
    }
#elif defined(__linux__) || defined(__unix__)
    std::error_code ec;
    auto p = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec && !p.empty()) {
        exeDir = p.parent_path();
    } else {
        p = std::filesystem::canonical("/proc/self/exe", ec);
        if (!ec) exeDir = p.parent_path();
    }
#endif

    std::vector<std::filesystem::path> candidates = {
        "scenes/cyber_city/cyber_city_ad_1.mp4",
        "scenes/cyber_city_ad_1.mp4",
        "../scenes/cyber_city/cyber_city_ad_1.mp4",
        "../../scenes/cyber_city/cyber_city_ad_1.mp4",
    };
    if (!exeDir.empty()) {
        candidates.push_back(exeDir / "scenes" / "cyber_city" / "cyber_city_ad_1.mp4");
        candidates.push_back(exeDir / ".." / "scenes" / "cyber_city" / "cyber_city_ad_1.mp4");
        candidates.push_back(exeDir / ".." / ".." / "scenes" / "cyber_city" / "cyber_city_ad_1.mp4");
    }

    std::filesystem::path foundPath;
    for (const auto& c : candidates) {
        std::error_code err;
        if (std::filesystem::exists(c, err)) {
            foundPath = c;
            break;
        }
    }

    if (foundPath.empty()) {
        Logger::warn("VideoBillboardManager: Video billboard file 'cyber_city_ad_1.mp4' not found in candidate paths.");
        return;
    }

    m_videoDecoder = std::make_unique<VideoDecoder>();
    if (!m_videoDecoder->open(foundPath.string())) {
        Logger::error("VideoBillboardManager: Failed to open video billboard stream from '{}'", foundPath.string());
        m_videoDecoder.reset();
        return;
    }

    // Allocate per-frame-in-flight staging buffers for lock-free asynchronous GPU uploads
    size_t rgbaSize = m_videoDecoder->getRgbaSize();
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_videoStagingBuffers[i] = std::make_unique<Buffer>(
            allocator, rgbaSize,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
            VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
        );
    }

    Logger::info("VideoBillboardManager: Video billboard initialized ({}x{} @ {:.2f} fps) using '{}'",
                 m_videoDecoder->getWidth(), m_videoDecoder->getHeight(), m_videoDecoder->getFps(), foundPath.string());
}

void VideoBillboardManager::update(float dt) {
    if (m_videoDecoder && m_videoDecoder->isOpen()) {
        m_videoDecoder->update(static_cast<double>(dt));
    }
}

bool VideoBillboardManager::hasNewFrame() const {
    return m_videoDecoder && m_videoDecoder->hasNewFrame();
}

bool VideoBillboardManager::isOpen() const {
    return m_videoDecoder && m_videoDecoder->isOpen();
}

void VideoBillboardManager::uploadFrame(VkCommandBuffer cmd, uint32_t frameSlot, Texture* targetTexture) {
    if (!m_videoDecoder || !m_videoDecoder->isOpen()) {
        return;
    }
    if (!targetTexture) {
        return;
    }
    if (!m_videoDecoder->hasNewFrame()) {
        return;
    }
    if (frameSlot >= MAX_FRAMES_IN_FLIGHT || !m_videoStagingBuffers[frameSlot]) {
        return;
    }

    const uint8_t* pixels = m_videoDecoder->getRgbaPixels();
    size_t size = m_videoDecoder->getRgbaSize();
    if (pixels && size > 0) {
        targetTexture->updatePixelsAsync(cmd, *m_videoStagingBuffers[frameSlot], pixels, size);
        m_videoDecoder->clearNewFrameFlag();
    }
}

void VideoBillboardManager::reset() {
    m_videoDecoder.reset();
    for (auto& sb : m_videoStagingBuffers) {
        sb.reset();
    }
}

} // namespace pathways
