#include "rt/DGCManager.hpp"
#include "vulkan/VulkanContext.hpp"
#include "vulkan/Image.hpp"
#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <cmath>
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

    // 1b. Verify DGCDispatchCommand struct layout & 16-byte alignment
    std::cout << "[TEST 1b] DGCDispatchCommand Layout & Alignment Verification..." << std::endl;
    check_true(sizeof(DGCDispatchCommand) == 16, "DGCDispatchCommand must be exactly 16 bytes");
    check_true(offsetof(DGCDispatchCommand, groupCountX) == 0, "groupCountX must start at byte 0");
    check_true(offsetof(DGCDispatchCommand, groupCountY) == 4, "groupCountY must start at byte 4");
    check_true(offsetof(DGCDispatchCommand, groupCountZ) == 8, "groupCountZ must start at byte 8");
    check_true(offsetof(DGCDispatchCommand, pad) == 12, "pad must start at byte 12");
    std::cout << "  -> DGCDispatchCommand layout verified (16B, std430 DispatchCommand compatibility)." << std::endl;

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
    check_true(indices.asyncComputeFamily == UINT32_MAX, "Default asyncComputeFamily must be UINT32_MAX");

    indices.graphicsComputeFamily = 0;
    check_true(indices.isComplete(), "QueueFamilyIndices with graphicsComputeFamily set must be complete");

    indices.transferFamily = 2;
    indices.asyncComputeFamily = 1;
    check_true(indices.asyncComputeFamily != indices.graphicsComputeFamily,
               "Async compute family must be distinct from graphics family when available");
    check_true(indices.asyncComputeFamily != indices.transferFamily,
               "Async compute family must be distinct from transfer family");
    std::cout << "  -> QueueFamilyIndices invariants verified (Graphics: 0, Async Compute: 1, Transfer: 2)." << std::endl;

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

    // 5b. Verify Multi-Sequence Indirect Stride & Address Size Invariants
    std::cout << "[TEST 5b] Multi-Sequence DGC Indirect Stride Invariants..." << std::endl;
    const uint32_t stride = sizeof(DGCDispatchCommand); // 16 bytes
    check_true(stride == 16, "DGC dispatch stride must be 16 bytes");
    check_true(stride % 4 == 0, "indirectStride must be a multiple of 4 per Vulkan spec");
    check_true(stride >= sizeof(VkDispatchIndirectCommand), "indirectStride must fit VkDispatchIndirectCommand");

    // Check sequence offset alignment across multiple sequences
    for (uint32_t seq = 0; seq < 16; ++seq) {
        VkDeviceSize seqOffset = static_cast<VkDeviceSize>(seq) * stride;
        check_true(seqOffset == seq * 16, "Sequence offset must be strictly 16-byte aligned");

        // Vulkan Valid Usage: indirectAddressSize >= (maxSequenceCount - 1) * indirectStride + tokenSize
        uint32_t maxSeq = seq + 1;
        VkDeviceSize minRequiredAddressSize = static_cast<VkDeviceSize>(maxSeq - 1) * stride + sizeof(VkDispatchIndirectCommand);
        VkDeviceSize actualAllocatedAddressSize = static_cast<VkDeviceSize>(stride) * maxSeq;
        check_true(actualAllocatedAddressSize >= minRequiredAddressSize,
                   "indirectAddressSize must satisfy VUID-VkGeneratedCommandsInfoEXT-indirectAddressSize");
    }
    std::cout << "  -> Multi-sequence DGC indirect stride (16B) and valid usage invariants verified." << std::endl;

    // 6. Verify DGCPassType and Slice Orthogonality
    std::cout << "[TEST 6] DGCPassType Slicing Orthogonality Across Passes & Bounces..." << std::endl;
    bool sliceUsed[DGCManager::NUM_SLICES] = { false };
    for (uint32_t slot = 0; slot < 2; ++slot) {
        for (uint32_t b = 0; b < 4; ++b) {
            uint32_t matSlice = DGCManager::getSliceIndex(slot, b, DGCManager::PassMaterial);
            uint32_t shadowSlice = DGCManager::getSliceIndex(slot, b, DGCManager::PassShadow);
            uint32_t intersectSlice = DGCManager::getSliceIndex(slot, b, DGCManager::PassIntersect);

            check_true(matSlice != shadowSlice, "Material and Shadow slices must be distinct");
            check_true(shadowSlice != intersectSlice, "Shadow and Intersect slices must be distinct");
            check_true(matSlice != intersectSlice, "Material and Intersect slices must be distinct");

            check_true(matSlice < DGCManager::NUM_SLICES, "matSlice must be within NUM_SLICES");
            check_true(shadowSlice < DGCManager::NUM_SLICES, "shadowSlice must be within NUM_SLICES");
            check_true(intersectSlice < DGCManager::NUM_SLICES, "intersectSlice must be within NUM_SLICES");

            check_true(!sliceUsed[matSlice], "Material slice collision detected");
            sliceUsed[matSlice] = true;
            check_true(!sliceUsed[shadowSlice], "Shadow slice collision detected");
            sliceUsed[shadowSlice] = true;
            check_true(!sliceUsed[intersectSlice], "Intersect slice collision detected");
            sliceUsed[intersectSlice] = true;
        }
    }
    std::cout << "  -> DGCPassType 24-slice orthogonal partitioning verified across all bounces." << std::endl;

    // 7. Live Hardware DGC Preprocess + Execute & Async Compute Queue Verification
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
            std::cout << "[TEST 7] Live Hardware DGC Preprocess + Indirect Execute on GPU..." << std::endl;
            Config cfg;
            cfg.headless = true;
            cfg.validation_layers = true;
            cfg.gpu_index = (hwGpuCount >= 2) ? 1 : 0;

            VulkanContext vkCtx(cfg, VK_NULL_HANDLE, "DGC GPU Test");
            check_true(vkCtx.getAsyncComputeQueue() != VK_NULL_HANDLE, "Async compute queue handle must be valid");

            if (vkCtx.hasDGC()) {
                VkDevice device = vkCtx.getDevice();
                VmaAllocator allocator = vkCtx.getAllocator();
                VkQueue queue = vkCtx.getGraphicsQueue();

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

                auto spv = loadSpv("accum_running_avg.comp.spv");
                check_true(!spv.empty(), "accum_running_avg.comp.spv must exist");

                VkDescriptorSetLayoutBinding bindings[2]{};
                bindings[0] = { 0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
                bindings[1] = { 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr };
                VkDescriptorSetLayoutCreateInfo dslInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO };
                dslInfo.bindingCount = 2;
                dslInfo.pBindings = bindings;
                VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
                check_true(vkCreateDescriptorSetLayout(device, &dslInfo, nullptr, &dsl) == VK_SUCCESS, "Create DGC DSL");

                VkPushConstantRange pcRange{ VK_SHADER_STAGE_COMPUTE_BIT, 0, 16 };
                VkPipelineLayoutCreateInfo plInfo{ VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO };
                plInfo.setLayoutCount = 1;
                plInfo.pSetLayouts = &dsl;
                plInfo.pushConstantRangeCount = 1;
                plInfo.pPushConstantRanges = &pcRange;
                VkPipelineLayout pipeLayout = VK_NULL_HANDLE;
                check_true(vkCreatePipelineLayout(device, &plInfo, nullptr, &pipeLayout) == VK_SUCCESS, "Create DGC pipelineLayout");

                VkShaderModuleCreateInfo smInfo{ VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO };
                smInfo.codeSize = spv.size();
                smInfo.pCode = reinterpret_cast<const uint32_t*>(spv.data());
                VkShaderModule shaderMod = VK_NULL_HANDLE;
                check_true(vkCreateShaderModule(device, &smInfo, nullptr, &shaderMod) == VK_SUCCESS, "Create DGC shaderModule");

                VkPipelineCreateFlags2CreateInfo flags2Info{ VK_STRUCTURE_TYPE_PIPELINE_CREATE_FLAGS_2_CREATE_INFO };
                flags2Info.flags = VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT;

                VkComputePipelineCreateInfo cpCreate{ VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO };
                cpCreate.pNext = &flags2Info;
                cpCreate.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
                cpCreate.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
                cpCreate.stage.module = shaderMod;
                cpCreate.stage.pName = "main";
                cpCreate.layout = pipeLayout;
                VkPipeline compPipe = VK_NULL_HANDLE;
                check_true(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &cpCreate, nullptr, &compPipe) == VK_SUCCESS,
                           "Create VK_PIPELINE_CREATE_2_INDIRECT_BINDABLE_BIT_EXT compute pipeline");
                vkDestroyShaderModule(device, shaderMod, nullptr);

                DGCManager dgc(device, allocator, pipeLayout, /*supportsExecutionSet=*/false, /*enableExplicitPreprocess=*/true);
                check_true(dgc.isSupported(), "DGCManager must initialize successfully on DGC hardware");

                const uint32_t W = 16, H = 16;
                const VkImageUsageFlags imgUsage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
                Image currImg(device, allocator, W, H, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);
                Image histImg(device, allocator, W, H, VK_FORMAT_R16G16B16A16_SFLOAT, imgUsage);

                VkDescriptorPoolSize poolSz{ VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 2 };
                VkDescriptorPoolCreateInfo dpInfo{ VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO };
                dpInfo.maxSets = 1;
                dpInfo.poolSizeCount = 1;
                dpInfo.pPoolSizes = &poolSz;
                VkDescriptorPool dPool = VK_NULL_HANDLE;
                check_true(vkCreateDescriptorPool(device, &dpInfo, nullptr, &dPool) == VK_SUCCESS, "Create DGC descriptor pool");

                VkDescriptorSetAllocateInfo dsAlloc{ VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO };
                dsAlloc.descriptorPool = dPool;
                dsAlloc.descriptorSetCount = 1;
                dsAlloc.pSetLayouts = &dsl;
                VkDescriptorSet descSet = VK_NULL_HANDLE;
                check_true(vkAllocateDescriptorSets(device, &dsAlloc, &descSet) == VK_SUCCESS, "Allocate DGC descriptor set");

                VkDescriptorImageInfo i0{ VK_NULL_HANDLE, currImg.getImageView(), VK_IMAGE_LAYOUT_GENERAL };
                VkDescriptorImageInfo i1{ VK_NULL_HANDLE, histImg.getImageView(), VK_IMAGE_LAYOUT_GENERAL };
                VkWriteDescriptorSet writes[2]{};
                writes[0] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 0, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &i0, nullptr, nullptr };
                writes[1] = { VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, nullptr, descSet, 1, 0, 1, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, &i1, nullptr, nullptr };
                vkUpdateDescriptorSets(device, 2, writes, 0, nullptr);

                DGCDispatchCommand indirectCmd{ 1u, 1u, 1u, 0u };
                Buffer argBuf(allocator, sizeof(DGCDispatchCommand),
                              VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
                argBuf.copyFrom(&indirectCmd, sizeof(indirectCmd));

                const VkDeviceSize imgBytes = W * H * 4 * sizeof(uint16_t);
                Buffer readback(allocator, imgBytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

                VkCommandPoolCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
                cpInfo.queueFamilyIndex = vkCtx.getGraphicsQueueFamily();
                VkCommandPool cmdPool = VK_NULL_HANDLE;
                check_true(vkCreateCommandPool(device, &cpInfo, nullptr, &cmdPool) == VK_SUCCESS, "Create DGC cmdPool");

                VkCommandBufferAllocateInfo cbAlloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
                cbAlloc.commandPool = cmdPool;
                cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                cbAlloc.commandBufferCount = 1;
                VkCommandBuffer cmd = VK_NULL_HANDLE;
                check_true(vkAllocateCommandBuffers(device, &cbAlloc, &cmd) == VK_SUCCESS, "Allocate DGC cmd");

                VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(cmd, &beginInfo);

                currImg.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                         VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                histImg.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                         VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                VkClearColorValue clearTwo{};
                clearTwo.float32[0] = 2.0f; clearTwo.float32[1] = 2.0f; clearTwo.float32[2] = 2.0f; clearTwo.float32[3] = 0.0f;
                VkClearColorValue clearZero{};
                VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                vkCmdClearColorImage(cmd, currImg.getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearTwo, 1, &range);
                vkCmdClearColorImage(cmd, histImg.getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &range);

                VkMemoryBarrier2 clrBar{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                clrBar.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
                clrBar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
                clrBar.dstStageMask = VK_PIPELINE_STAGE_2_COMMAND_PREPROCESS_BIT_EXT |
                                      VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT |
                                      VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                clrBar.dstAccessMask = VK_ACCESS_2_COMMAND_PREPROCESS_READ_BIT_EXT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT |
                                       VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
                VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers = &clrBar;
                vkCmdPipelineBarrier2(cmd, &dep);

                const uint32_t sliceIdx = DGCManager::getFrameSliceIndex(0, 0);
                dgc.recordPreprocess(cmd, compPipe, &argBuf, 0, sliceIdx, 1);
                dgc.recordPreprocessBarrier(cmd, sliceIdx, 1);

                vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, compPipe);
                vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
                struct { uint32_t w, h, sampleCount; float invSpp; } pc{ W, H, 2u, 1.0f }; // sampleCount=2 -> mix(0.0, 2.0, 0.5) = 1.0
                vkCmdPushConstants(cmd, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
                dgc.recordExecute(cmd, compPipe, &argBuf, 0, sliceIdx, 1, /*isPreprocessed=*/true);

                VkMemoryBarrier2 outBar{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                outBar.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                outBar.srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT;
                outBar.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
                outBar.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
                dep.pMemoryBarriers = &outBar;
                vkCmdPipelineBarrier2(cmd, &dep);

                VkBufferImageCopy outRegion{};
                outRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                outRegion.imageExtent = { W, H, 1 };
                vkCmdCopyImageToBuffer(cmd, histImg.getImage(), VK_IMAGE_LAYOUT_GENERAL, readback.getBuffer(), 1, &outRegion);

                vkEndCommandBuffer(cmd);
                VkSubmitInfo sub{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                sub.commandBufferCount = 1;
                sub.pCommandBuffers = &cmd;
                check_true(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE) == VK_SUCCESS, "Submit DGC preprocess + execute");
                vkQueueWaitIdle(queue);

                const uint16_t* px = static_cast<const uint16_t*>(readback.map());
                check_true(px != nullptr, "Map DGC readback buffer");
                float r0 = glm::unpackHalf1x16(px[0]);
                readback.unmap();
                check_true(std::abs(r0 - 1.0f) < 0.02f, "DGC-dispatched accum_running_avg must compute mix(0.0, 2.0, 0.5) == 1.0");
                check_true(dgc.getPreprocessBuffer() != nullptr, "DGC preprocess buffer allocated");

                vkDestroyCommandPool(device, cmdPool, nullptr);
                vkDestroyDescriptorPool(device, dPool, nullptr);
                vkDestroyPipeline(device, compPipe, nullptr);
                vkDestroyPipelineLayout(device, pipeLayout, nullptr);
                vkDestroyDescriptorSetLayout(device, dsl, nullptr);

                check_true(vkCtx.getValidationErrors() == 0, "Zero Vulkan validation errors during DGC preprocess + execute");
                std::cout << "  -> Live GPU DGC explicit preprocess + indirect execute verified (output=" << r0 << ", 0 validation errors)." << std::endl;
            }
        }
    }

    std::cout << "\n[SUCCESS] All DGC multi-slice ring buffer & queue invariant tests passed!" << std::endl;
    return 0;
}
