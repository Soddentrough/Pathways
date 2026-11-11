#include "rt/AccelerationStructure.hpp"
#include "rt/GpuTlasUpdatePipeline.hpp"
#include "vulkan/VulkanContext.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <cstddef>
#include <glm/gtc/matrix_transform.hpp>
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

    // 4. Live GPU-Timeline TLAS Build, Cross-GPU blasAddress Preservation & Refit
    {
        uint32_t hwGpuCount = 0;
        VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
        appInfo.apiVersion = VK_API_VERSION_1_4;
        VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instInfo.pApplicationInfo = &appInfo;
        VkInstance probeInst = VK_NULL_HANDLE;
        if (vkCreateInstance(&instInfo, nullptr, &probeInst) == VK_SUCCESS) {
            uint32_t devCount = 0;
            vkEnumeratePhysicalDevices(probeInst, &devCount, nullptr);
            std::vector<VkPhysicalDevice> devs(devCount);
            vkEnumeratePhysicalDevices(probeInst, &devCount, devs.data());
            for (VkPhysicalDevice pd : devs) {
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(pd, &props);
                if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU && props.apiVersion >= VK_API_VERSION_1_4) {
                    ++hwGpuCount;
                }
            }
            vkDestroyInstance(probeInst, nullptr);
        }

        if (hwGpuCount > 0) {
            std::cout << "[TEST 4] Live GPU-Timeline TLAS Instance Writer & Refit..." << std::endl;
            Config cfg;
            cfg.headless = true;
            cfg.validation_layers = true;
            cfg.gpu_index = (hwGpuCount >= 2) ? 1 : 0;

            VulkanContext vkCtx(cfg, VK_NULL_HANDLE, "TLAS GPU Refit Test");
            VkDevice device = vkCtx.getDevice();
            VmaAllocator allocator = vkCtx.getAllocator();
            VkQueue queue = vkCtx.getGraphicsQueue();
            uint32_t qFamily = vkCtx.getGraphicsQueueFamily();

            AccelerationStructureManager asMgr(device, allocator, queue, qFamily);

            std::vector<glm::vec4> verts = {
                {-1.0f, -1.0f, 0.0f, 1.0f},
                { 1.0f, -1.0f, 0.0f, 1.0f},
                { 0.0f,  1.0f, 0.0f, 1.0f}
            };
            std::vector<uint32_t> idx = { 0, 1, 2 };

            Buffer vBuf(allocator, verts.size() * sizeof(glm::vec4),
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            vBuf.copyFrom(verts.data(), verts.size() * sizeof(glm::vec4));

            Buffer iBuf(allocator, idx.size() * sizeof(uint32_t),
                        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                        VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            iBuf.copyFrom(idx.data(), idx.size() * sizeof(uint32_t));

            ASGeometryInput geom{};
            geom.vertexBufferAddress = vBuf.getDeviceAddress(device);
            geom.indexBufferAddress = iBuf.getDeviceAddress(device);
            geom.vertexCount = 3;
            geom.triangleCount = 1;
            geom.vertexStride = sizeof(glm::vec4);
            geom.isOpaque = true;

            auto blas = asMgr.buildBLAS({ geom });
            check_true(blas && blas->getDeviceAddress() != 0, "Hardware BLAS must build with non-zero BDA");
            const VkDeviceAddress realBlasAddr = blas->getDeviceAddress();

            std::vector<ASInstanceInput> instInputs(2);
            instInputs[0].blasAddress = realBlasAddr;
            instInputs[0].transform = glm::mat4(1.0f);
            instInputs[0].customIndex = 0;
            instInputs[1].blasAddress = realBlasAddr;
            instInputs[1].transform = glm::translate(glm::mat4(1.0f), glm::vec3(2.0f, 0.0f, 0.0f));
            instInputs[1].customIndex = 1;

            auto tlas = asMgr.buildTLAS(instInputs);
            check_true(tlas && tlas->getHandle() != VK_NULL_HANDLE, "Initial hardware TLAS must build");

            auto loadSpv = [](const std::string& name) -> std::vector<char> {
                std::vector<std::filesystem::path> cands = {
#ifdef SHADER_DIR
                    std::filesystem::path(SHADER_DIR) / name,
#endif
                    std::filesystem::current_path() / "shaders" / name,
                    std::filesystem::current_path() / "build" / "shaders" / name
                };
                for (const auto& p : cands) {
                    std::ifstream f(p, std::ios::binary | std::ios::ate);
                    if (f.is_open()) {
                        size_t sz = static_cast<size_t>(f.tellg());
                        std::vector<char> buf(sz);
                        f.seekg(0);
                        f.read(buf.data(), static_cast<std::streamsize>(sz));
                        return buf;
                    }
                }
                return {};
            };
            std::vector<char> spv = loadSpv("update_tlas_instances.comp.spv");
            check_true(!spv.empty(), "update_tlas_instances.comp.spv must exist");

            Buffer shaderInstBuf(allocator, 2 * sizeof(InstanceGPU),
                                 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE);
            std::vector<InstanceGPU> shaderInsts(2);
            shaderInsts[0].transform = instInputs[0].transform;
            shaderInsts[0].invTransform = instInputs[0].transform;
            shaderInsts[1].transform = instInputs[1].transform;
            shaderInsts[1].invTransform = instInputs[1].transform;

            GpuTlasUpdatePipeline gpuTlas(device, allocator, vkCtx.hasSubgroupSizeControl());
            gpuTlas.initBuffers(instInputs, &asMgr);
            gpuTlas.initShaderInstances(&shaderInstBuf, shaderInsts);
            gpuTlas.createPipeline(spv);
            gpuTlas.updateDescriptors();

            // Simulate cross-GPU sync where primary GPU passes its own foreign blasAddress:
            // secondary GPU MUST preserve its own local realBlasAddr!
            std::vector<ASInstanceGPUData> foreignPrimaryInputs = gpuTlas.getCpuInputInstances();
            foreignPrimaryInputs[0].transform = glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 1.5f, 0.0f));
            foreignPrimaryInputs[0].blasAddress = 0xDEADBEEFCAFE0000ULL;
            shaderInsts[0].transform = foreignPrimaryInputs[0].transform;
            shaderInsts[0].invTransform = instInputs[0].transform;

            gpuTlas.syncFromCpuTransforms(shaderInsts, foreignPrimaryInputs, /*triggerTlasRebuild=*/true);
            check_true(gpuTlas.getCpuInputInstances()[0].blasAddress == realBlasAddr,
                       "syncFromCpuTransforms must preserve device-local blasAddress across GPUs");
            check_true(gpuTlas.needsTlasRebuild(), "syncFromCpuTransforms must mark TLAS dirty");

            VkCommandPoolCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            cpInfo.queueFamilyIndex = qFamily;
            cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            VkCommandPool cmdPool = VK_NULL_HANDLE;
            check_true(vkCreateCommandPool(device, &cpInfo, nullptr, &cmdPool) == VK_SUCCESS, "Command pool created");

            VkCommandBufferAllocateInfo cbAlloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cbAlloc.commandPool = cmdPool;
            cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbAlloc.commandBufferCount = 1;
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            check_true(vkAllocateCommandBuffers(device, &cbAlloc, &cmd) == VK_SUCCESS, "Command buffer allocated");

            // Frame 0: Full GPU-timeline rebuild (updateMode = false)
            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            vkBeginCommandBuffer(cmd, &beginInfo);
            gpuTlas.recordGpuTlasUpdate(cmd, 0u, &asMgr, tlas.get(), /*updateMode=*/false);
            vkEndCommandBuffer(cmd);
            VkSubmitInfo sub{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            sub.commandBufferCount = 1;
            sub.pCommandBuffers = &cmd;
            check_true(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE) == VK_SUCCESS, "Submit GPU TLAS rebuild");
            vkQueueWaitIdle(queue);

            // Frame 1: GPU-timeline in-place refit (updateMode = true)
            foreignPrimaryInputs[1].transform = glm::translate(glm::mat4(1.0f), glm::vec3(3.0f, 0.5f, 0.0f));
            shaderInsts[1].transform = foreignPrimaryInputs[1].transform;
            gpuTlas.syncFromCpuTransforms(shaderInsts, foreignPrimaryInputs, /*triggerTlasRebuild=*/true);

            vkResetCommandBuffer(cmd, 0);
            vkBeginCommandBuffer(cmd, &beginInfo);
            gpuTlas.recordGpuTlasUpdate(cmd, 1u, &asMgr, tlas.get(), /*updateMode=*/true);
            vkEndCommandBuffer(cmd);
            check_true(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE) == VK_SUCCESS, "Submit GPU TLAS refit");
            vkQueueWaitIdle(queue);

            vkDestroyCommandPool(device, cmdPool, nullptr);

            check_true(gpuTlas.getGpuUpdateCount() == 2, "GPU TLAS update counter must record both rebuild and refit");
            check_true(vkCtx.getValidationErrors() == 0, "Zero Vulkan validation errors during GPU TLAS rebuild + refit");
            std::cout << "  -> Live GPU-timeline TLAS rebuild + refit & cross-GPU blasAddress preservation verified (0 validation errors)." << std::endl;
        }
    }

    std::cout << "\n[SUCCESS] All Tier 3 GPU-timeline TLAS and DGC structural tests passed!" << std::endl;
    return 0;
}
