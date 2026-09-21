#include "video/VideoDecoder.hpp"
#include "core/Logger.hpp"

#include <cmath>
#include <cstring>
#include <stdexcept>

#ifdef PATHWAYS_HAS_FFMPEG
extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/avutil.h>
}
#endif

namespace pathways {

#ifdef PATHWAYS_HAS_FFMPEG

VideoDecoder::VideoDecoder() {
}

VideoDecoder::~VideoDecoder() {
    release();
}

VideoDecoder::VideoDecoder(VideoDecoder&& other) noexcept {
    *this = std::move(other);
}

VideoDecoder& VideoDecoder::operator=(VideoDecoder&& other) noexcept {
    if (this != &other) {
        release();

        m_isOpen = other.m_isOpen;
        m_loop = other.m_loop;
        m_hasNewFrame = other.m_hasNewFrame;
        m_width = other.m_width;
        m_height = other.m_height;
        m_fps = other.m_fps;
        m_frameDuration = other.m_frameDuration;
        m_durationSeconds = other.m_durationSeconds;
        m_accumTime = other.m_accumTime;
        m_videoStreamIdx = other.m_videoStreamIdx;
        m_formatCtx = other.m_formatCtx;
        m_codecCtx = other.m_codecCtx;
        m_swsCtx = other.m_swsCtx;
        m_frame = other.m_frame;
        m_frameRgba = other.m_frameRgba;
        m_packet = other.m_packet;
        m_rgbaBuffer = std::move(other.m_rgbaBuffer);

        other.m_isOpen = false;
        other.m_formatCtx = nullptr;
        other.m_codecCtx = nullptr;
        other.m_swsCtx = nullptr;
        other.m_frame = nullptr;
        other.m_frameRgba = nullptr;
        other.m_packet = nullptr;
    }
    return *this;
}

void VideoDecoder::release() {
    if (m_swsCtx) {
        sws_freeContext(m_swsCtx);
        m_swsCtx = nullptr;
    }
    if (m_frameRgba) {
        av_frame_free(&m_frameRgba);
        m_frameRgba = nullptr;
    }
    if (m_frame) {
        av_frame_free(&m_frame);
        m_frame = nullptr;
    }
    if (m_packet) {
        av_packet_free(&m_packet);
        m_packet = nullptr;
    }
    if (m_codecCtx) {
        avcodec_free_context(&m_codecCtx);
        m_codecCtx = nullptr;
    }
    if (m_formatCtx) {
        avformat_close_input(&m_formatCtx);
        m_formatCtx = nullptr;
    }

    m_isOpen = false;
    m_hasNewFrame = false;
    m_width = 0;
    m_height = 0;
    m_videoStreamIdx = -1;
    m_rgbaBuffer.clear();
}

void VideoDecoder::close() {
    release();
}

bool VideoDecoder::open(const std::string& filepath) {
    release();

    int ret = avformat_open_input(&m_formatCtx, filepath.c_str(), nullptr, nullptr);
    if (ret < 0) {
        char errBuf[AV_ERROR_MAX_STRING_SIZE] = {0};
        av_strerror(ret, errBuf, sizeof(errBuf));
        Logger::error("VideoDecoder: Failed to open video file '{}': {}", filepath, errBuf);
        return false;
    }

    ret = avformat_find_stream_info(m_formatCtx, nullptr);
    if (ret < 0) {
        Logger::error("VideoDecoder: Could not retrieve stream info from '{}'", filepath);
        release();
        return false;
    }

    m_videoStreamIdx = -1;
    for (unsigned int i = 0; i < m_formatCtx->nb_streams; ++i) {
        if (m_formatCtx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
            m_videoStreamIdx = static_cast<int>(i);
            break;
        }
    }

    if (m_videoStreamIdx < 0) {
        Logger::error("VideoDecoder: No video stream found in '{}'", filepath);
        release();
        return false;
    }

    AVStream* stream = m_formatCtx->streams[m_videoStreamIdx];
    const AVCodec* codec = avcodec_find_decoder(stream->codecpar->codec_id);
    if (!codec) {
        Logger::error("VideoDecoder: Unsupported codec for video in '{}'", filepath);
        release();
        return false;
    }

    m_codecCtx = avcodec_alloc_context3(codec);
    if (!m_codecCtx) {
        Logger::error("VideoDecoder: Failed to allocate codec context");
        release();
        return false;
    }

    ret = avcodec_parameters_to_context(m_codecCtx, stream->codecpar);
    if (ret < 0) {
        Logger::error("VideoDecoder: Failed to copy codec parameters to context");
        release();
        return false;
    }

    ret = avcodec_open2(m_codecCtx, codec, nullptr);
    if (ret < 0) {
        Logger::error("VideoDecoder: Failed to open codec");
        release();
        return false;
    }

    m_width = static_cast<uint32_t>(m_codecCtx->width);
    m_height = static_cast<uint32_t>(m_codecCtx->height);

    AVRational rFps = stream->r_frame_rate;
    if (rFps.den > 0 && rFps.num > 0) {
        m_fps = static_cast<double>(rFps.num) / static_cast<double>(rFps.den);
    } else {
        m_fps = 24.0;
    }
    m_frameDuration = 1.0 / m_fps;

    if (stream->duration > 0 && stream->time_base.den > 0) {
        m_durationSeconds = static_cast<double>(stream->duration) * av_q2d(stream->time_base);
    } else if (m_formatCtx->duration > 0) {
        m_durationSeconds = static_cast<double>(m_formatCtx->duration) / AV_TIME_BASE;
    } else {
        m_durationSeconds = 0.0;
    }

    // Allocate frames and packet
    m_frame = av_frame_alloc();
    m_frameRgba = av_frame_alloc();
    m_packet = av_packet_alloc();

    if (!m_frame || !m_frameRgba || !m_packet) {
        Logger::error("VideoDecoder: Failed to allocate AVFrame/AVPacket");
        release();
        return false;
    }

    // Allocate RGBA pixel buffer
    size_t numBytes = static_cast<size_t>(m_width) * static_cast<size_t>(m_height) * 4;
    m_rgbaBuffer.resize(numBytes);

    ret = av_image_fill_arrays(
        m_frameRgba->data, m_frameRgba->linesize,
        m_rgbaBuffer.data(), AV_PIX_FMT_RGBA,
        m_width, m_height, 1
    );
    if (ret < 0) {
        Logger::error("VideoDecoder: Failed to fill image arrays");
        release();
        return false;
    }

    // SwsContext for converting incoming format (e.g. YUV420P) to RGBA
    m_swsCtx = sws_getContext(
        m_width, m_height, m_codecCtx->pix_fmt,
        m_width, m_height, AV_PIX_FMT_RGBA,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr
    );
    if (!m_swsCtx) {
        Logger::error("VideoDecoder: Failed to initialize SwsContext");
        release();
        return false;
    }

    m_isOpen = true;
    m_accumTime = 0.0;
    m_hasNewFrame = false;

    Logger::info("VideoDecoder: Successfully opened '{}' ({}x{} @ {:.2f} fps, {:.2f}s)",
                 filepath, m_width, m_height, m_fps, m_durationSeconds);

    // Decode first frame immediately so buffer is initialized
    decodeNextFrame();

    return true;
}

void VideoDecoder::rewind() {
    if (m_formatCtx && m_videoStreamIdx >= 0) {
        av_seek_frame(m_formatCtx, m_videoStreamIdx, 0, AVSEEK_FLAG_BACKWARD);
        if (m_codecCtx) {
            avcodec_flush_buffers(m_codecCtx);
        }
    }
}

bool VideoDecoder::decodeNextFrame() {
    if (!m_isOpen || !m_formatCtx || !m_codecCtx) {
        return false;
    }

    while (true) {
        int ret = av_read_frame(m_formatCtx, m_packet);
        if (ret < 0) {
            // EOF reached
            if (m_loop) {
                rewind();
                ret = av_read_frame(m_formatCtx, m_packet);
                if (ret < 0) {
                    return false;
                }
            } else {
                return false;
            }
        }

        if (m_packet->stream_index == m_videoStreamIdx) {
            ret = avcodec_send_packet(m_codecCtx, m_packet);
            av_packet_unref(m_packet);

            if (ret < 0) {
                continue;
            }

            ret = avcodec_receive_frame(m_codecCtx, m_frame);
            if (ret == 0) {
                // Successfully received decoded video frame -> convert to RGBA
                sws_scale(
                    m_swsCtx,
                    m_frame->data, m_frame->linesize,
                    0, m_height,
                    m_frameRgba->data, m_frameRgba->linesize
                );
                m_hasNewFrame = true;
                return true;
            }
        } else {
            av_packet_unref(m_packet);
        }
    }

    return false;
}

bool VideoDecoder::update(double dtSeconds) {
    if (!m_isOpen) {
        return false;
    }

    m_accumTime += dtSeconds;
    if (m_accumTime >= m_frameDuration) {
        // Prevent accumulating unbounded backlog if render stalls
        if (m_accumTime > m_frameDuration * 4.0) {
            m_accumTime = m_frameDuration;
        } else {
            m_accumTime -= m_frameDuration;
        }
        return decodeNextFrame();
    }

    return false;
}

#else

VideoDecoder::VideoDecoder() {}
VideoDecoder::~VideoDecoder() {}
VideoDecoder::VideoDecoder(VideoDecoder&& other) noexcept = default;
VideoDecoder& VideoDecoder::operator=(VideoDecoder&& other) noexcept = default;

void VideoDecoder::release() {}
void VideoDecoder::close() {}

bool VideoDecoder::open(const std::string& filepath) {
    Logger::info("VideoDecoder: Built without FFmpeg support. Video playback disabled for '{}'", filepath);
    return false;
}

void VideoDecoder::rewind() {}
bool VideoDecoder::decodeNextFrame() { return false; }
bool VideoDecoder::update(double) { return false; }

#endif

} // namespace pathways
