#include "TrainingDataWriter.hpp"
#include "core/Logger.hpp"
#include <filesystem>
#include <fstream>
#include <cmath>
#include <cstring>
#include <algorithm>
#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>

#if defined(__linux__) || defined(__unix__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace pathways {

void TrainingDataWriter::normalizeReferenceBuffer(void* data, uint32_t width, uint32_t height, uint32_t dataType) {
    if (!data || width == 0 || height == 0) return;

    size_t numPixels = static_cast<size_t>(width) * height;

    if (dataType == 1) {
        // Float32 (RGBA32F)
        float* pixels = static_cast<float*>(data);
        for (size_t i = 0; i < numPixels; ++i) {
            float a = pixels[i * 4 + 3];
            if (a > 1.0001f) {
                float invA = 1.0f / a;
                pixels[i * 4 + 0] = std::isfinite(pixels[i * 4 + 0]) ? std::max(0.0f, pixels[i * 4 + 0] * invA) : 0.0f;
                pixels[i * 4 + 1] = std::isfinite(pixels[i * 4 + 1]) ? std::max(0.0f, pixels[i * 4 + 1] * invA) : 0.0f;
                pixels[i * 4 + 2] = std::isfinite(pixels[i * 4 + 2]) ? std::max(0.0f, pixels[i * 4 + 2] * invA) : 0.0f;
                pixels[i * 4 + 3] = 1.0f;
            }
        }
    } else if (dataType == 0) {
        // Float16 (RGBA16F)
        uint16_t* pixels = static_cast<uint16_t*>(data);
        for (size_t i = 0; i < numPixels; ++i) {
            float a = glm::unpackHalf1x16(pixels[i * 4 + 3]);
            if (a > 1.0001f) {
                float invA = 1.0f / a;
                float r = glm::unpackHalf1x16(pixels[i * 4 + 0]);
                float g = glm::unpackHalf1x16(pixels[i * 4 + 1]);
                float b = glm::unpackHalf1x16(pixels[i * 4 + 2]);

                r = std::isfinite(r) ? std::max(0.0f, r * invA) : 0.0f;
                g = std::isfinite(g) ? std::max(0.0f, g * invA) : 0.0f;
                b = std::isfinite(b) ? std::max(0.0f, b * invA) : 0.0f;

                pixels[i * 4 + 0] = glm::packHalf1x16(r);
                pixels[i * 4 + 1] = glm::packHalf1x16(g);
                pixels[i * 4 + 2] = glm::packHalf1x16(b);
                pixels[i * 4 + 3] = glm::packHalf1x16(1.0f);
            }
        }
    }
}

bool TrainingDataWriter::writeTensor(const std::string& filepath,
                                    uint32_t width, uint32_t height,
                                    uint32_t channels, uint32_t dataType,
                                    uint32_t frameIndex, uint32_t spp,
                                    const void* data, size_t byteSize) {
    if (!data || byteSize == 0 || width == 0 || height == 0) {
        Logger::error("Invalid tensor buffer passed to writeTensor");
        return false;
    }

    std::filesystem::path p(filepath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    TrainingTensorHeader header{};
    header.magic[0] = 'P';
    header.magic[1] = 'T';
    header.magic[2] = 'T';
    header.magic[3] = 'D';
    header.version = 1;
    header.width = width;
    header.height = height;
    header.channels = channels;
    header.dataType = dataType;
    header.frameIndex = frameIndex;
    header.spp = spp;
    header.payloadByteSize = static_cast<uint64_t>(byteSize);

    std::vector<uint8_t> normalizedCopy;
    const void* dataToWrite = data;

    if (channels == 4) {
        bool needsNorm = false;
        if (dataType == 1 && byteSize >= sizeof(float) * 4) {
            float a0 = static_cast<const float*>(data)[3];
            if (a0 > 1.0001f) needsNorm = true;
        } else if (dataType == 0 && byteSize >= sizeof(uint16_t) * 4) {
            float a0 = glm::unpackHalf1x16(static_cast<const uint16_t*>(data)[3]);
            if (a0 > 1.0001f) needsNorm = true;
        }

        if (needsNorm) {
            normalizedCopy.resize(byteSize);
            std::memcpy(normalizedCopy.data(), data, byteSize);
            normalizeReferenceBuffer(normalizedCopy.data(), width, height, dataType);
            dataToWrite = normalizedCopy.data();
        }
    }

#if defined(__linux__) || defined(__unix__)
    int fd = open(filepath.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) {
        Logger::error("Failed to open binary tensor file for writing: {}", filepath);
        return false;
    }

    ssize_t hdrWritten = write(fd, &header, sizeof(header));
    if (hdrWritten != sizeof(header)) {
        Logger::error("Failed to write binary tensor header to: {}", filepath);
        close(fd);
        return false;
    }

    const uint8_t* ptr = static_cast<const uint8_t*>(dataToWrite);
    size_t remaining = byteSize;
    while (remaining > 0) {
        ssize_t written = write(fd, ptr, remaining);
        if (written <= 0) {
            Logger::error("Failed writing tensor payload to: {}", filepath);
            close(fd);
            return false;
        }
        ptr += written;
        remaining -= static_cast<size_t>(written);
    }
    close(fd);
#else
    std::ofstream ofs(filepath, std::ios::binary);
    if (!ofs) {
        Logger::error("Failed to open binary tensor file for writing: {}", filepath);
        return false;
    }
    ofs.write(reinterpret_cast<const char*>(&header), sizeof(header));
    ofs.write(reinterpret_cast<const char*>(dataToWrite), byteSize);
    if (!ofs) {
        Logger::error("Failed writing tensor payload to: {}", filepath);
        return false;
    }
#endif

    Logger::info("Saved training tensor [{}x{}x{} dtype={}] to: {} ({:.2f} MB)",
                 width, height, channels, dataType, filepath, (byteSize + sizeof(header)) / (1024.0 * 1024.0));
    return true;
}

} // namespace pathways
