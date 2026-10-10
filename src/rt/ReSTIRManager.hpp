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

// 32-byte ReSTIR GI reservoir (path-space, docs/RESTIR_PT_DESIGN.md).
// Must match `GiReservoir` in shaders/compute/restir_common.glsl.
struct GiReservoirGPU {
    uint32_t x2x;      // fp32 bits of secondary vertex world position
    uint32_t x2y;
    uint32_t x2z;
    uint32_t wSum;     // fp32 bits
    uint32_t target;   // fp32 bits, p-hat at the recording pixel
    uint32_t radRG;    // fp16 (L2.r, L2.g)
    uint32_t radB_dSrc;// fp16 L2.b, fp16 |x2 - x1| at recording time
    uint32_t flagsM;   // [0] valid, [8:1] age, [24:9] M
};
static_assert(sizeof(GiReservoirGPU) == 32, "GiReservoirGPU must be exactly 32 bytes");

// 24-byte per-pixel BRDF context at the primary hit x1, written at bounce 0
// and consumed at bounce 1 of the same frame. Must match `X1Context` in GLSL.
struct X1ContextGPU {
    uint32_t n1Oct;     // oct32(x1 normal)
    uint32_t albedoRG;  // fp16 (albedo.r, albedo.g)
    uint32_t albedoB;   // fp16 albedo.b
    uint32_t flags;     // [0] valid (diffuse cosine reflection lobe)
    uint32_t woOct;     // oct32(x1 -> camera direction)
    uint32_t f0Pad;     // fp16 scalar F0, fp16 reserved
};
static_assert(sizeof(X1ContextGPU) == 24, "X1ContextGPU must be exactly 24 bytes");

// Owns the per-pixel ReSTIR state grids consumed by the wavefront shade
// passes:
//   binding 32/36: double-buffered DI reservoirs (current / history)
//   binding 37:    single-buffered X1 context (same-frame producer/consumer)
//   binding 38/39: double-buffered GI reservoirs (current / history)
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
    Buffer* getX1ContextBuffer() const { return m_x1Context.get(); }

    bool isInitialized() const { return m_width > 0 && m_height > 0; }

private:
    void initBuffers();

    VkDevice m_device = VK_NULL_HANDLE;
    VmaAllocator m_allocator = VK_NULL_HANDLE;
    uint32_t m_width = 0;
    uint32_t m_height = 0;
    bool m_clearPending[MAX_FRAMES_IN_FLIGHT] = { true, true };
    bool m_giClearPending[MAX_FRAMES_IN_FLIGHT] = { true, true };
    bool m_ctxClearPending = true;

    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_reservoirBuffers;
    std::array<std::unique_ptr<Buffer>, MAX_FRAMES_IN_FLIGHT> m_giReservoirBuffers;
    std::unique_ptr<Buffer> m_x1Context;
};

} // namespace pathways
