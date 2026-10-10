#include "ReSTIRManager.hpp"
#include "core/Logger.hpp"

namespace pathways {

ReSTIRManager::ReSTIRManager(VkDevice device, VmaAllocator allocator)
    : m_device(device), m_allocator(allocator) {
    Logger::info("ReSTIRManager (inline ReSTIR DI reservoir owner) constructed.");
}

ReSTIRManager::~ReSTIRManager() {
    for (auto& b : m_reservoirBuffers) b.reset();
}

void ReSTIRManager::initBuffers() {
    VkDeviceSize resSize = static_cast<VkDeviceSize>(m_width) * m_height * sizeof(DiReservoirGPU);
    if (resSize == 0) return;

    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        m_reservoirBuffers[i] = std::make_unique<Buffer>(
            m_allocator, resSize,
            VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            VMA_MEMORY_USAGE_GPU_ONLY);
        m_clearPending[i] = true;
    }
}

void ReSTIRManager::resize(uint32_t width, uint32_t height) {
    if (m_width == width && m_height == height) return;
    m_width = width;
    m_height = height;
    initBuffers();
    Logger::info("ReSTIRManager reservoir grid (re)allocated: {}x{} x 2 slots x {} B.",
                 m_width, m_height, static_cast<uint32_t>(sizeof(DiReservoirGPU)));
}

void ReSTIRManager::recordClearIfNeeded(VkCommandBuffer cmd) {
    bool anyPending = false;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_reservoirBuffers[i] && m_clearPending[i]) anyPending = true;
    }
    if (!anyPending) return;

    std::array<VkBufferMemoryBarrier2, MAX_FRAMES_IN_FLIGHT> barriers{};
    uint32_t barrierCount = 0;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        Buffer* buf = m_reservoirBuffers[i].get();
        if (!buf) continue;
        if (m_clearPending[i]) {
            vkCmdFillBuffer(cmd, buf->getBuffer(), 0, VK_WHOLE_SIZE, 0u);
            m_clearPending[i] = false;
        }
        VkBufferMemoryBarrier2& b = barriers[barrierCount++];
        b = VkBufferMemoryBarrier2{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
        b.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
        b.srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT;
        b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.buffer = buf->getBuffer();
        b.offset = 0;
        b.size = VK_WHOLE_SIZE;
    }

    VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    dep.bufferMemoryBarrierCount = barrierCount;
    dep.pBufferMemoryBarriers = barriers.data();
    vkCmdPipelineBarrier2(cmd, &dep);
}

} // namespace pathways
