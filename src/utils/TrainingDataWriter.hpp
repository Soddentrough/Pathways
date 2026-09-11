#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <cstddef>

namespace pathways {

#pragma pack(push, 1)
struct TrainingTensorHeader {
    char magic[4] = {'P', 'T', 'T', 'D'}; // Pathways Training Tensor Data
    uint32_t version = 1;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t channels = 0;       // 16 for input, 4 for reference
    uint32_t dataType = 0;       // 0 = Float16, 1 = Float32
    uint32_t frameIndex = 0;
    uint32_t spp = 1;
    uint64_t payloadByteSize = 0;
    uint8_t padding[24] = {0};   // 64 bytes total
};
static_assert(sizeof(TrainingTensorHeader) == 64, "TrainingTensorHeader must be exactly 64 bytes");
#pragma pack(pop)

class TrainingDataWriter {
public:
    static void normalizeReferenceBuffer(void* data, uint32_t width, uint32_t height, uint32_t dataType);

    static bool writeTensor(const std::string& filepath,
                            uint32_t width, uint32_t height,
                            uint32_t channels, uint32_t dataType,
                            uint32_t frameIndex, uint32_t spp,
                            const void* data, size_t byteSize);
};

} // namespace pathways
