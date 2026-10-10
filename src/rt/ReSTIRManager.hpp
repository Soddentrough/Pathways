#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <memory>
#include <array>

#include "vulkan/Buffer.hpp"

namespace pathways {

// 28-byte ReSTIR DI reservoir. Must match `DiReservoir` in
// shaders/compute/restir_common.glsl (scalar-only std430 layout).
struct DiReservoirGPU {
    uint32_t idxM;        // [15:0] light index, [31:16] M
    float    wSum;        // RIS weight sum
    float    targetPdf;   // p-hat of the selected sample at the storing pixel
    uint32_t flags;       // [7:0] age, [8] valid, [9] trans lobe, [25:10] oct16(x1 normal)
    float    hitDist;     // x1 -> light-sample distance
    uint32_t lightPacked; // [15:0] oct16(light normal), [23:16] u8, [31:24] v8
    uint32_t x1Misc;      // [15:0] fp16 cos(theta at light), [31:16] fp16 primary hit distance
};
static_assert(sizeof(DiReservoirGPU) == 28, "DiReservoirGPU must be exactly 28 bytes (GLSL std430 scalar layout)");

// Owns the double-buffered per-pixel ReSTIR DI reservoir grid consumed by the
// wavefront shade passes (binding 32 = current frame slot, binding 36 = the
// previous frame slot read as history taps).
//
// All resampling itself happens inline in the wavefront shade shaders; this
// class deliberately contains no pipelines or dispatches (the former "fused
// ReSTIR pass" pipeline was dead code and has been removed — see
// docs/reports/restir_review_2026_10_10.md §6).
class ReSTIRManager {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    ReSTIRManager(VkDevice device, VmaAllocator allocator);
    ~ReSTIRManager();

    ReSTIRManager(const ReSTIRManager&) = delete;
    ReSTIRManager& operator=(const ReSTIRManager&) = delete;

    // (Re)allocate the reservoir grid. Buffers are marked for zero-fill; call
    // recordClearIfNeeded() on the frame command buffer before the shade
    // dispatches read them (uninitialized GPU memory must never be interpreted
    // as a valid reservoir).
    void resize(uint32_t width, uint32_t height);

    // Emits vkCmdFillBuffer zero-clears (plus barriers) if a (re)allocation is
    // pending. Cheap no-op once cleared.
    void recordClearIfNeeded(VkCommandBuffer cmd);

    // Returns nullptr until the first resize() — callers must tolerate that
    // (descriptor updates fall back to a dummy buffer).
    Buffer* getReservoirBuffer(uint32_t frameSlot) const {
        return (frameSlot < MAX_FRAMES_IN_FLIGHT) ? m_reservoirBuffers[frameSlot].get() : nullptr;
    }

    bool isInitialized() const { return m_width > 0 && m_height > 0; }

private:
    void initBuffers();

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_clearPending[MAX_FRAMES_IN_FLIGHT] = { true, true };

    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_reservoirBuffers;
};

} // namespace pathways
