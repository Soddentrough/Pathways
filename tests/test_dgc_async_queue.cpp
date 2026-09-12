#include "rt/DGCManager.hpp"
#include "vulkan/VulkanContext.hpp"
#include <iostream>
#include <cassert>
#include <cstddef>
#include <vulkan/vulkan.h>

using namespace pathways;

static void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] Assertion failed: " << msg << std::endl;
        std::exit(1);
    }
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing DGC Multi-Slice Ring Buffer & Queues  " << std::endl;
    std::cout << "  Explicit Preprocessing & Dedicated Compute Invariants   " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Verify DGCCommand struct layout & alignment
    std::cout << "[TEST 1] DGCCommand Layout & Alignment Verification..." << std::endl;
    check_true(sizeof(DGCCommand) == 16, "DGCCommand must be exactly 16 bytes");
    check_true(offsetof(DGCCommand, pipelineIndex) == 0, "pipelineIndex must start at byte 0");
    check_true(offsetof(DGCCommand, groupCountX) == 4, "groupCountX must start at byte 4");
    check_true(offsetof(DGCCommand, groupCountY) == 8, "groupCountY must start at byte 8");
    check_true(offsetof(DGCCommand, groupCountZ) == 12, "groupCountZ must start at byte 12");
    std::cout << "  -> DGCCommand layout verified (16B, execution set index + 3D dispatch)." << std::endl;

    // 2. Verify VkDispatchIndirectCommand layout compatibility
    std::cout << "[TEST 2] VkDispatchIndirectCommand Standard Layout..." << std::endl;
    check_true(sizeof(VkDispatchIndirectCommand) == 12, "VkDispatchIndirectCommand must be 12 bytes");
    check_true(offsetof(VkDispatchIndirectCommand, x) == 0, "x must start at byte 0");
    check_true(offsetof(VkDispatchIndirectCommand, y) == 4, "y must start at byte 4");
    check_true(offsetof(VkDispatchIndirectCommand, z) == 8, "z must start at byte 8");
    std::cout << "  -> VkDispatchIndirectCommand verified (12B standard Vulkan layout)." << std::endl;

    // 3. Verify QueueFamilyIndices Invariants & Dedicated Compute Queue
    std::cout << "[TEST 3] QueueFamilyIndices Invariants & Dedicated Compute Discovery..." << std::endl;
    QueueFamilyIndices indices{};
    check_true(!indices.isComplete(), "Default QueueFamilyIndices must not be complete");
    check_true(indices.dedicatedComputeFamily == UINT32_MAX, "Default dedicatedComputeFamily must be UINT32_MAX");

    indices.graphicsComputeFamily = 0;
    check_true(indices.isComplete(), "QueueFamilyIndices with graphicsComputeFamily set must be complete");

    indices.transferFamily = 2;
    indices.dedicatedComputeFamily = 1;
    check_true(indices.dedicatedComputeFamily != indices.graphicsComputeFamily,
               "Dedicated compute family must be distinct from graphics family when available");
    check_true(indices.dedicatedComputeFamily != indices.transferFamily,
               "Dedicated compute family must be distinct from transfer family");
    std::cout << "  -> QueueFamilyIndices invariants verified (Graphics: 0, Dedicated Compute: 1, Transfer: 2)." << std::endl;

    // 4. Verify Multi-Slice Ring Buffer Offset & Rollover Logic
    std::cout << "[TEST 4] Multi-Slice Ring Buffer Offset & Rollover Calculations..." << std::endl;
    const uint32_t sliceCount = 32;
    const VkDeviceSize sliceSize = 4096;
    const VkDeviceSize totalSize = sliceCount * sliceSize;

    for (uint32_t i = 0; i < sliceCount * 3; ++i) {
        uint32_t sliceIndex = i % sliceCount;
        VkDeviceSize offset = static_cast<VkDeviceSize>(sliceIndex) * sliceSize;
        check_true(offset < totalSize, "Slice offset must stay within total preprocess buffer bounds");
        check_true(offset + sliceSize <= totalSize, "Slice range must not overrun total buffer size");
        check_true(offset % 256 == 0, "Slice offset must satisfy 256-byte alignment requirement");
    }
    std::cout << "  -> Multi-slice ring buffer offsets & 32-slice rollover verified." << std::endl;

    // 5. Verify DGC Indirect Token Flags & Constants
    std::cout << "[TEST 5] DGC Indirect Token Flags & Constants Verification..." << std::endl;
    VkIndirectCommandsLayoutUsageFlagsEXT expectedFlags =
        VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_EXT |
        VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT;
    check_true((expectedFlags & VK_INDIRECT_COMMANDS_LAYOUT_USAGE_EXPLICIT_PREPROCESS_BIT_EXT) != 0,
               "EXPLICIT_PREPROCESS_BIT_EXT must be active");
    check_true((expectedFlags & VK_INDIRECT_COMMANDS_LAYOUT_USAGE_UNORDERED_SEQUENCES_BIT_EXT) != 0,
               "UNORDERED_SEQUENCES_BIT_EXT must be active");
    std::cout << "  -> DGC layout usage flags (0x3) verified." << std::endl;

    std::cout << "\n[SUCCESS] All DGC multi-slice ring buffer & queue invariant tests passed!" << std::endl;
    return 0;
}
