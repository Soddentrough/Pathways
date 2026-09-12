#include "TrainingDataWriter.hpp"
#include "core/Logger.hpp"
#include <filesystem>
#include <fstream>

#if defined(__linux__) || defined(__unix__)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace pathways {

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

    const uint8_t* ptr = static_cast<const uint8_t*>(data);
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
    ofs.write(reinterpret_cast<const char*>(data), byteSize);
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
