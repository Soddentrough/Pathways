#pragma once

#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <memory>
#include <array>

#include "vulkan/Buffer.hpp"
#include "rt/ReSTIRTypes.hpp"

namespace pathways {

// Owns the per-pixel ReSTIR state grids consumed by the wavefront shade
// passes:
//   binding 32/36: double-buffered DI reservoirs (current / history)
//   binding 37:    single-buffered X1 context (same-frame producer/consumer)
//   binding 38/39: double-buffered GI reservoirs (current / history)
//   binding 41/42: double-buffered PT reservoirs (current / history)
//   binding 43:    single-buffered PathSample pool (Approach B)
//
// All resampling happens inline in the wavefront shade shaders; this class
// deliberately contains no pipelines or dispatches (the former "fused ReSTIR
// pass" pipeline was dead code — see docs/reports/restir_review_2026_10_10.md
// §6). Buffers are allocated lazily on the first resize() and zero-filled
// before first use so uninitialized memory can never alias a valid reservoir.
class ReSTIRManager {
public:
    static constexpr uint32_t MAX_FRAMES_IN_FLIGHT = 2;

    ReSTIRManager(VkDevice device, VmaAllocator allocator);
    ~ReSTIRManager();

    ReSTIRManager(const ReSTIRManager&) = delete;
    ReSTIRManager& operator=(const ReSTIRManager&) = delete;

    // (Re)allocate all state grids. Buffers are marked for zero-fill; call
    // recordClearIfNeeded() on the frame command buffer before the shade
    // dispatches read them.
    void resize(uint32_t width, uint32_t height);

    // Emits vkCmdFillBuffer zero-clears (plus barriers) if a (re)allocation is
    // pending. Cheap no-op once cleared.
    void recordClearIfNeeded(VkCommandBuffer cmd);

    // Return nullptr until the first resize() — callers must tolerate that
    // (descriptor updates fall back to a dummy buffer).
    Buffer* getReservoirBuffer(uint32_t frameSlot) const {
        return (frameSlot < MAX_FRAMES_IN_FLIGHT) ? m_reservoirBuffers[frameSlot].get() : nullptr;
    }
    Buffer* getGiReservoirBuffer(uint32_t frameSlot) const {
        return (frameSlot < MAX_FRAMES_IN_FLIGHT) ? m_giReservoirBuffers[frameSlot].get() : nullptr;
    }
    Buffer* getReservoirPTBuffer(uint32_t frameSlot) const {
        return (frameSlot < MAX_FRAMES_IN_FLIGHT) ? m_reservoirPTBuffers[frameSlot].get() : nullptr;
    }
    Buffer* getPathSampleBuffer() const { return m_pathSampleBuffer.get(); }
    Buffer* getX1ContextBuffer() const { return m_x1Context.get(); }
    Buffer* getShadeRequestBuffer() const { return m_shadeRequestBuffer.get(); }
    Buffer* getShadeResultBuffer() const { return m_shadeResultBuffer.get(); }

    bool isInitialized() const { return m_width > 0 && m_height > 0; }

    bool isPhase3Enabled() const { return m_enablePhase3; }
    void setPhase3Enabled(bool enabled) { m_enablePhase3 = enabled; }

private:
    void initBuffers();

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_enablePhase3 = true;
    bool m_clearPending[MAX_FRAMES_IN_FLIGHT] = { true, true };
    bool m_giClearPending[MAX_FRAMES_IN_FLIGHT] = { true, true };
    bool m_ptClearPending[MAX_FRAMES_IN_FLIGHT] = { true, true };
    bool m_pathClearPending = true;
    bool m_ctxClearPending = true;
    bool m_reqClearPending = true;
    bool m_resClearPending = true;

    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_reservoirBuffers;
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_giReservoirBuffers;
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_reservoirPTBuffers;
    std::unique_ptr<Buffer> m_pathSampleBuffer;
    std::unique_ptr<Buffer> m_x1Context;
    std::unique_ptr<Buffer> m_shadeRequestBuffer;
    std::unique_ptr<Buffer> m_shadeResultBuffer;
};

} // namespace pathways
