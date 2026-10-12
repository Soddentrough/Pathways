#include "ReSTIRManager.hpp"
#include "core/Logger.hpp"

#include <vector>

namespace pathways {

ReSTIRManager::ReSTIRManager(VkDevice device, VmaAllocator allocator)
    : m_device(device), m_allocator(allocator) {
    Logger::info("ReSTIRManager (ReSTIR PT path-space reservoir subsystem) constructed.");
}

ReSTIRManager::~ReSTIRManager() {
    for (auto& b : m_reservoirBuffers) b.reset();
    for (auto& b : m_giReservoirBuffers) b.reset();
    for (auto& b : m_reservoirPTBuffers) b.reset();
    m_pathSampleBuffer.reset();
    m_x1Context.reset();
    m_shadeRequestBuffer.reset();
    m_shadeResultBuffer.reset();
}

void ReSTIRManager::initBuffers() {
    VkDeviceSize pixels = static_cast<VkDeviceSize>(m_width) * m_height;
    if (pixels == 0) return;

    const VkBufferUsageFlags usage =
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_reservoirBuffers[i] = std::make_unique<Buffer>(
            m_allocator, pixels * sizeof(DiReservoirGPU), usage, VMA_MEMORY_USAGE_GPU_ONLY);
        m_clearPending[i] = true;

        m_giReservoirBuffers[i] = std::make_unique<Buffer>(
            m_allocator, pixels * sizeof(GiReservoirGPU), usage, VMA_MEMORY_USAGE_GPU_ONLY);
        m_giClearPending[i] = true;

        if (m_enablePhase3) {
            m_reservoirPTBuffers[i] = std::make_unique<Buffer>(
                m_allocator, pixels * sizeof(ReservoirPTGPU), usage, VMA_MEMORY_USAGE_GPU_ONLY);
            m_ptClearPending[i] = true;
        } else {
            m_reservoirPTBuffers[i].reset();
            m_ptClearPending[i] = false;
        }
    }

    if (m_enablePhase3) {
        m_pathSampleBuffer = std::make_unique<Buffer>(
            m_allocator, pixels * sizeof(PathSampleGPU), usage, VMA_MEMORY_USAGE_GPU_ONLY);
        m_pathClearPending = true;
    } else {
        m_pathSampleBuffer.reset();
        m_pathClearPending = false;
    }

    m_x1Context = std::make_unique<Buffer>(
        m_allocator, pixels * sizeof(X1ContextGPU), usage, VMA_MEMORY_USAGE_GPU_ONLY);
    m_ctxClearPending = true;

    if (m_enablePhase3) {
        m_shadeRequestBuffer = std::make_unique<Buffer>(
            m_allocator, pixels * sizeof(ShadeRequestGPU), usage, VMA_MEMORY_USAGE_GPU_ONLY);
        m_reqClearPending = true;

        m_shadeResultBuffer = std::make_unique<Buffer>(
            m_allocator, pixels * sizeof(ShadeResultGPU), usage, VMA_MEMORY_USAGE_GPU_ONLY);
        m_resClearPending = true;
    } else {
        m_shadeRequestBuffer.reset();
        m_reqClearPending = false;
        m_shadeResultBuffer.reset();
        m_resClearPending = false;
    }
}

void ReSTIRManager::resize(uint32_t width, uint32_t height) {
    if (m_width == width && m_height == height) return;
    m_width = width;
    m_height = height;
    initBuffers();
    uint32_t activeBytesPerPixel = static_cast<uint32_t>(
        2 * sizeof(DiReservoirGPU) + 2 * sizeof(GiReservoirGPU) + sizeof(X1ContextGPU)
        + (m_enablePhase3 ? (2 * sizeof(ReservoirPTGPU) + sizeof(PathSampleGPU) + sizeof(ShadeRequestGPU) + sizeof(ShadeResultGPU)) : 0));
    double totalMB = (static_cast<double>(m_width) * m_height * activeBytesPerPixel) / (1024.0 * 1024.0);
    Logger::info("ReSTIRManager grids (re)allocated: {}x{} ({} B/pixel, {:.2f} MB) — Direct 2x{}B, Indirect 2x{}B, X1 ctx {}B{}",
                 m_width, m_height,
                 activeBytesPerPixel,
                 totalMB,
                 static_cast<uint32_t>(sizeof(DiReservoirGPU)),
                 static_cast<uint32_t>(sizeof(GiReservoirGPU)),
                 static_cast<uint32_t>(sizeof(X1ContextGPU)),
                 m_enablePhase3 ? " [Phase 3 PT buffers active]" : "");
}

void ReSTIRManager::recordClearIfNeeded(VkCommandBuffer cmd) {
    struct Pending {
        Buffer* buffer;
        bool* flag;
    };
    std::vector<Pending> pending;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_reservoirBuffers[i] && m_clearPending[i]) pending.push_back({ m_reservoirBuffers[i].get(), &m_clearPending[i] });
        if (m_giReservoirBuffers[i] && m_giClearPending[i]) pending.push_back({ m_giReservoirBuffers[i].get(), &m_giClearPending[i] });
        if (m_reservoirPTBuffers[i] && m_ptClearPending[i]) pending.push_back({ m_reservoirPTBuffers[i].get(), &m_ptClearPending[i] });
    }
    if (m_pathSampleBuffer && m_pathClearPending) pending.push_back({ m_pathSampleBuffer.get(), &m_pathClearPending });
    if (m_x1Context && m_ctxClearPending) pending.push_back({ m_x1Context.get(), &m_ctxClearPending });
    if (m_shadeRequestBuffer && m_reqClearPending) pending.push_back({ m_shadeRequestBuffer.get(), &m_reqClearPending });
    if (m_shadeResultBuffer && m_resClearPending) pending.push_back({ m_shadeResultBuffer.get(), &m_resClearPending });
    if (pending.empty()) return;

    std::vector<VkBufferMemoryBarrier2> barriers;
    barriers.reserve(pending.size());
    for (auto& p : pending) {
        vkCmdFillBuffer(cmd, p.buffer->getBuffer(), 0, VK_WHOLE_SIZE, 0u);
        *p.flag = false;

        VkBufferMemoryBarrier2& b = barriers.emplace_back(VkBufferMemoryBarrier2{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 });
        b.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = p.buffer->getBuffer();
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
    }

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.bufferMemoryBarrierCount = static_cast<uint32_t>(barriers.size());
    dep.pBufferMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace pathways
