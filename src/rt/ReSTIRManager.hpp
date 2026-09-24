#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <glm/glm.hpp>
#include <memory>
#include <vector>
#include <array>

#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "core/Config.hpp"

namespace pathways {

// Unified 16-byte ReSTIR DI Reservoir matching GLSL layout
struct alignas(16) UnifiedReservoirPT {
    uint32_t lightIndex_M;             // lower 16 bits: lightIndex / candidate, upper 16 bits: M
    float    wSum;                     // sum of weights / final evaluation weight W
    uint32_t flags_uv_age;             // [0..7] age, [8] valid, [9..10] pathLength (1=DI), [11] lobe, [16..31] packed UV
    float    targetPdf;                // scalar unshadowed target distribution p_hat
};
static_assert(sizeof(UnifiedReservoirPT) == 16, "UnifiedReservoirPT must be exactly 16 bytes for cache-aligned memory footprint");

using ReservoirDI = UnifiedReservoirPT;

struct ReSTIRPushConstants {
    uint32_t width;
    uint32_t height;
    uint32_t numLights;
    uint32_t frameIndex;
    uint32_t mCap;
    uint32_t numTriangles;
    uint32_t hasLightTree;
    uint32_t passIndex;
};
static_assert(sizeof(ReSTIRPushConstants) == 32, "ReSTIRPushConstants must be exactly 32 bytes");

class ReSTIRManager {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    ReSTIRManager(VkDevice device, VmaAllocator allocator,
                  uint32_t width, uint32_t height,
                  const std::vector<char>& temporalSpv,
                  const std::vector<char>& spatialSpv);
    ~ReSTIRManager();

    ReSTIRManager(const ReSTIRManager&) = delete;
    ReSTIRManager& operator=(const ReSTIRManager&) = delete;

    void resize(uint32_t width, uint32_t height);

    void recordFrame(VkCommandBuffer cmd, uint32_t frameSlot, uint32_t width, uint32_t height,
                     uint32_t numLights, uint32_t numTriangles, bool hasLightTree,
                     uint32_t frameIndex, uint32_t mCap,
                     Buffer* rayGeomQueue, Buffer* rayHitQueue, Buffer* pixelToRayQueue,
                     Buffer* lightsBuffer, Buffer* materialsBuffer,
                     Buffer* cameraUBO, Buffer* lightTreeBuffer,
                     VkImageView motionVectorView, VkImageView normalDepthView, VkImageView prevNormalDepthView,
                     VkImageView reconstructConfidenceView = VK_NULL_HANDLE);

    Buffer* getSpatialReservoirBuffer(uint32_t frameSlot) const {
        return m_spatialBuffers[frameSlot].get();
    }

    bool isInitialized() const { return m_initialized; }

private:
    void initBuffers();
    void createDescriptorLayout();
    void allocateDescriptorSets();
    void createPipelines(const std::vector<char>& temporalSpv,
                         const std::vector<char>& spatialSpv);
    VkShaderModule createShaderModule(const std::vector<char>& code);

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_initialized = false;
    uint32_t m_temporalIndex = 0; // Ping-pong index for temporal history (0 or 1)

    // Reservoirs (16 bytes each):
    // Temporal ping-pong buffers across frames (0: prev/curr, 1: curr/prev)
    std::array<std::unique_ptr<Buffer>, 2> m_temporalBuffers;
    // Final spatial output buffers (double-buffered per in-flight frame slot)
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_spatialBuffers;
    std::unique_ptr<Image> m_dummyConfidenceImage;

    // Descriptors & Pipelines
    VkDescriptorPool m_descriptorPool = VK_NULL_HANDLE;
    VkDescriptorSetLayout m_descSetLayout = VK_NULL_HANDLE;
    std::array<VkDescriptorSet, MAX_FRAMES_IN_FLIGHT> m_descSets = { VK_NULL_HANDLE, VK_NULL_HANDLE };

    VkPipelineLayout m_pipelineLayout = VK_NULL_HANDLE;
    VkPipeline m_temporalPipeline = VK_NULL_HANDLE;
    VkPipeline m_spatialPipeline = VK_NULL_HANDLE;
};

} // namespace pathways
