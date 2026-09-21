#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <memory>

struct AVFormatContext;
struct AVCodecContext;
struct SwsContext;
struct AVFrame;
struct AVPacket;

namespace pathways {

class VideoDecoder {
public:
    VideoDecoder();
    ~VideoDecoder();

    VideoDecoder(const VideoDecoder&) = delete;
    VideoDecoder& operator=(const VideoDecoder&) = delete;

    VideoDecoder(VideoDecoder&& other) noexcept;
    VideoDecoder& operator=(VideoDecoder&& other) noexcept;

    bool open(const std::string& filepath);
    void close();
    bool isOpen() const { return m_isOpen; }

    /// Advances playback timer by dtSeconds and decodes a new frame if due.
    /// Returns true if a new frame has been decoded and is ready for GPU transfer.
    bool update(double dtSeconds);

    /// Forces decoding the next video frame immediately.
    bool decodeNextFrame();

    /// Rewinds video stream back to frame 0 and flushes codec.
    void rewind();

    void setLooping(bool loop) { m_loop = loop; }
    bool isLooping() const { return m_loop; }

    uint32_t getWidth() const { return m_width; }
    uint32_t getHeight() const { return m_height; }
    double getFps() const { return m_fps; }
    double getDurationSeconds() const { return m_durationSeconds; }

    const uint8_t* getRgbaPixels() const { return m_rgbaBuffer.data(); }
    size_t getRgbaSize() const { return m_rgbaBuffer.size(); }

    bool hasNewFrame() const { return m_hasNewFrame; }
    void clearNewFrameFlag() { m_hasNewFrame = false; }

private:
    void release();

    bool m_isOpen = false;
    bool m_loop = true;
    bool m_hasNewFrame = false;

    uint32_t m_width = 0;
    uint32_t m_height = 0;
    double m_fps = 24.0;
    double m_frameDuration = 1.0 / 24.0;
    double m_durationSeconds = 0.0;
    double m_accumTime = 0.0;

    int m_videoStreamIdx = -1;
    AVFormatContext* m_formatCtx = nullptr;
    AVCodecContext* m_codecCtx = nullptr;
    SwsContext* m_swsCtx = nullptr;
    AVFrame* m_frame = nullptr;
    AVFrame* m_frameRgba = nullptr;
    AVPacket* m_packet = nullptr;

    std::vector<uint8_t> m_rgbaBuffer;
};

} // namespace pathways
