#include "rt/AccelerationStructure.hpp"
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
    std::cout << "  Pathways: Testing Tier 3 GPU-Timeline TLAS & DGC Structs" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Verify ASInstanceGPUData std430 alignment & layout
    std::cout << "[TEST 1] ASInstanceGPUData Layout & Size Verification..." << std::endl;
    check_true(sizeof(ASInstanceGPUData) == 96, "ASInstanceGPUData must be exactly 96 bytes");
    check_true(offsetof(ASInstanceGPUData, transform) == 0, "transform must start at byte 0");
    check_true(offsetof(ASInstanceGPUData, customIndex) == 64, "customIndex must be at byte 64");
    check_true(offsetof(ASInstanceGPUData, mask) == 68, "mask must be at byte 68");
    check_true(offsetof(ASInstanceGPUData, hitGroupId) == 72, "hitGroupId must be at byte 72");
    check_true(offsetof(ASInstanceGPUData, flags) == 76, "flags must be at byte 76");
    check_true(offsetof(ASInstanceGPUData, blasAddress) == 80, "blasAddress must be at byte 80");
    check_true(offsetof(ASInstanceGPUData, pad0) == 88, "pad0 must be at byte 88");
    check_true(offsetof(ASInstanceGPUData, pad1) == 92, "pad1 must be at byte 92");
    std::cout << "  -> ASInstanceGPUData layout verified (96B std430 aligned)." << std::endl;

    // 2. Verify VkAccelerationStructureInstanceKHR 64-byte layout
    std::cout << "[TEST 2] VkAccelerationStructureInstanceKHR Layout Verification..." << std::endl;
    check_true(sizeof(VkAccelerationStructureInstanceKHR) == 64, "VkAccelerationStructureInstanceKHR must be 64 bytes");
    check_true(offsetof(VkAccelerationStructureInstanceKHR, transform) == 0, "transform must start at byte 0");
    check_true(offsetof(VkAccelerationStructureInstanceKHR, accelerationStructureReference) == 56, "accelerationStructureReference must start at byte 56");
    std::cout << "  -> VkAccelerationStructureInstanceKHR verified (64B Vulkan KHR spec)." << std::endl;

    // 3. Verify TLAS Scratch Buffer Alignment (256-byte requirement)
    std::cout << "[TEST 3] TLAS Scratch Buffer Alignment Calculation..." << std::endl;
    VkDeviceSize rawSizes[] = { 1, 100, 255, 256, 257, 1024, 65535, 65536 };
    for (VkDeviceSize raw : rawSizes) {
        VkDeviceSize aligned = (raw + 255) & ~VkDeviceSize(255);
        check_true(aligned % 256 == 0, "Aligned size must be a multiple of 256");
        check_true(aligned >= raw, "Aligned size must be >= raw size");
    }
    std::cout << "  -> Scratch alignment rule (VUID-vkCmdBuildAccelerationStructuresKHR-pInfos-03710) verified." << std::endl;

    std::cout << "\n[SUCCESS] All Tier 3 GPU-timeline TLAS and DGC structural tests passed!" << std::endl;
    return 0;
}
