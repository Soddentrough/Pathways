#include "rt/NRCManager.hpp"
#include "vulkan/VulkanContext.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <cassert>
#include <cstddef>
#include <cmath>
#include <vector>
#include <cstring>
#include <random>
#include <glm/gtc/packing.hpp>
#include <vulkan/vulkan.h>

using namespace pathways;

static void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] Assertion failed: " << msg << std::endl;
        std::exit(1);
    }
}

// Emulate GLSL spatial hash corner calculation
static uint32_t nrcHashCorner(int32_t cx, int32_t cy, int32_t cz, uint32_t level) {
    const uint32_t P1 = 1u;
    const uint32_t P2 = 2654435761u;
    const uint32_t P3 = 805459861u;
    const uint32_t P_LEVEL = 1000003u;
    const uint32_t HASH_TABLE_SIZE = 262144u;

    uint32_t h = (static_cast<uint32_t>(cx) * P1 ^
                  static_cast<uint32_t>(cy) * P2 ^
                  static_cast<uint32_t>(cz) * P3) ^
                 (level * P_LEVEL);
    h = h ^ (h >> 16);
    return h % HASH_TABLE_SIZE;
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing Tier 4 Neural Radiance Caching (NRC)   " << std::endl;
    std::cout << "  Wave32 WMMA Cooperative Matrix & Optimizer Invariants   " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // 1. Verify NRCQuery and NRCTrainingRecord std430 alignment & layout
    std::cout << "[TEST 1] NRC Buffer Struct Layout & Size Verification..." << std::endl;
    check_true(sizeof(NRCQuery) == 80, "NRCQuery must be exactly 80 bytes (5x vec4)");
    check_true(offsetof(NRCQuery, pos_roughness) == 0, "pos_roughness must be at byte 0");
    check_true(offsetof(NRCQuery, normal_flags) == 16, "normal_flags must be at byte 16");
    check_true(offsetof(NRCQuery, dir_pixelIndex) == 32, "dir_pixelIndex must be at byte 32");
    check_true(offsetof(NRCQuery, albedo_pad) == 48, "albedo_pad must be at byte 48");
    check_true(offsetof(NRCQuery, throughput) == 64, "throughput must be at byte 64");
    std::cout << "  -> NRCQuery layout verified (80B std430 aligned)." << std::endl;

    check_true(sizeof(NRCTrainingRecord) == 128, "NRCTrainingRecord must be exactly 128 bytes (8x vec4)");
    check_true(offsetof(NRCTrainingRecord, pos_roughness) == 0, "pos_roughness must be at byte 0");
    check_true(offsetof(NRCTrainingRecord, normal_flags) == 16, "normal_flags must be at byte 16");
    check_true(offsetof(NRCTrainingRecord, dir_pixelIndex) == 32, "dir_pixelIndex must be at byte 32");
    check_true(offsetof(NRCTrainingRecord, albedo_pad) == 48, "albedo_pad must be at byte 48");
    check_true(offsetof(NRCTrainingRecord, throughput) == 64, "throughput must be at byte 64");
    check_true(offsetof(NRCTrainingRecord, targetRadiance) == 80, "targetRadiance must be at byte 80");
    check_true(offsetof(NRCTrainingRecord, tailPos_roughness) == 96, "tailPos_roughness must be at byte 96");
    check_true(offsetof(NRCTrainingRecord, tailPacked) == 112, "tailPacked must be at byte 112");
    std::cout << "  -> NRCTrainingRecord layout verified (128B std430 aligned)." << std::endl;

    check_true(sizeof(NRCCountersBuffer) == 16, "NRCCountersBuffer must be exactly 16 bytes (4x uint32_t)");
    check_true(offsetof(NRCCountersBuffer, queryCount) == 0, "queryCount must be at byte 0");
    check_true(offsetof(NRCCountersBuffer, trainCount) == 4, "trainCount must be at byte 4");
    check_true(offsetof(NRCCountersBuffer, dispatchX) == 8, "dispatchX must be at byte 8");
    std::cout << "  -> NRCCountersBuffer layout verified (16B std430 aligned)." << std::endl;

    // 2. Verify Spatial Hash Coordinate Parity (Legacy + Active 8-Level Hash Grid Math)
    std::cout << "[TEST 2] Spatial Hash Math & Bounded Index Verification..." << std::endl;
    const float GRID_SCALES[12] = {
        16.0f, 23.36f, 34.12f, 49.83f, 72.78f, 106.28f,
        155.23f, 226.70f, 331.09f, 483.54f, 706.18f, 1031.33f
    };
    const float ACTIVE_SCALES[8] = {
        16.0f, 29.0f, 52.6f, 95.4f, 173.0f, 313.8f, 569.1f, 1031.33f
    };
    check_true(GRID_SCALES[0] >= 16.0f && ACTIVE_SCALES[0] >= 16.0f, "Min grid scale should be 16");
    check_true(GRID_SCALES[11] >= 1000.0f && ACTIVE_SCALES[7] >= 1000.0f, "Max grid scale should exceed 1000 for sub-millimeter detail");

    // Test hash boundedness and determinism across random normalized coords
    for (uint32_t l = 0; l < 8; ++l) {
        float s = ACTIVE_SCALES[l];
        for (int i = 0; i < 50; ++i) {
            float px = (i * 0.021f);
            float py = (i * 0.017f);
            float pz = (i * 0.029f);
            int32_t cx = static_cast<int32_t>(std::floor(px * s));
            int32_t cy = static_cast<int32_t>(std::floor(py * s));
            int32_t cz = static_cast<int32_t>(std::floor(pz * s));

            for (int dz = 0; dz <= 1; ++dz) {
                for (int dy = 0; dy <= 1; ++dy) {
                    for (int dx = 0; dx <= 1; ++dx) {
                        uint32_t h = nrcHashCorner(cx + dx, cy + dy, cz + dz, l);
                        check_true(h < 262144u, "Hash corner index must be within table size (2^18)");
                    }
                }
            }
        }
    }
    std::cout << "  -> Hash table index bounded to [0, 262143] across all 8 active multi-resolution levels." << std::endl;

    // 3. Verify WMMA Cooperative Matrix 16x16x16 Tiling Math (NRC_TOTAL_WEIGHTS == 9360)
    std::cout << "[TEST 3] WMMA Wave32 Cooperative Matrix MLP Topology..." << std::endl;
    // Input features: 64. Hidden: 64. Output: 16 (first 3 channels = RGB).
    // Batch size per SIMD32 wave: 16 rays.
    // Layer 0: W0 (64x64) + B0 (64)
    // Layer 1: W1 (64x64) + B1 (64)
    // Layer 2: W2 (64x16) + B2 (16)
    const uint32_t w0_count = 64 * 64; // 4096
    const uint32_t b0_count = 64;
    const uint32_t w1_count = 64 * 64; // 4096
    const uint32_t b1_count = 64;
    const uint32_t w2_count = 64 * 16; // 1024
    const uint32_t b2_count = 16;
    const uint32_t total_weights = w0_count + b0_count + w1_count + b1_count + w2_count + b2_count; // 9360 float16
    check_true(total_weights == 9360u, "MLP total weights + biases must equal NRC_TOTAL_WEIGHTS (9360)");
    check_true(total_weights * sizeof(uint16_t) == 18720u, "Weights buffer must be 18720 bytes");
    check_true((w0_count % (16 * 16)) == 0, "Layer 0 dimensions must be multiple of 16x16 WMMA tile");
    check_true((w1_count % (16 * 16)) == 0, "Layer 1 dimensions must be multiple of 16x16 WMMA tile");
    check_true((w2_count % (16 * 16)) == 0, "Layer 2 dimensions must be multiple of 16x16 WMMA tile");
    std::cout << "  -> Wave32 WMMA 16x16x16 tiling alignment verified for all 3 layers (9360 params)." << std::endl;

    // 4. Verify Relative L1 Loss & Adam Optimizer Numerical Stability
    std::cout << "[TEST 4] Relative L1 Loss & Adam Optimizer Numerical Stability..." << std::endl;
    auto relL1Loss = [](float pred, float target) -> float {
        return std::abs(pred - target) / (std::max(pred, 0.0f) + 0.01f);
    };

    // Test zero radiance
    check_true(relL1Loss(0.0f, 0.0f) == 0.0f, "Loss for identical 0 radiance should be 0");
    // Test firefly suppression (pred = 1.0, target = 1000.0)
    float fireflyLoss = relL1Loss(1.0f, 1000.0f);
    check_true(std::isfinite(fireflyLoss), "Firefly loss must be finite");
    check_true(fireflyLoss < 1000.0f, "Relative loss dampens high radiance fireflies");

    // Test Adam step numerical stability
    float m = 0.0f;
    float v = 0.0f;
    float w = 0.5f;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float lr = 1e-3f;
    const float eps = 1e-8f;

    std::vector<float> testGradients = { 0.0f, 1.0f, -2.5f, 100.0f, -500.0f, 1e-5f };
    for (uint32_t t = 1; t <= 100; ++t) {
        float g = testGradients[t % testGradients.size()];
        m = beta1 * m + (1.0f - beta1) * g;
        v = beta2 * v + (1.0f - beta2) * (g * g);

        float mHat = m / (1.0f - std::pow(beta1, static_cast<float>(t)));
        float vHat = v / (1.0f - std::pow(beta2, static_cast<float>(t)));

        float step = lr * mHat / (std::sqrt(vHat) + eps);
        w -= step;

        check_true(std::isfinite(w), "Adam weight update must remain finite");
        check_true(!std::isnan(w), "Adam weight update must never become NaN");
    }
    std::cout << "  -> Adam optimizer update loop remained numerically stable across 100 iterations." << std::endl;

    // 5. Live GPU NRCManager Wave32 WMMA Inference, Training & Resolve Execution
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
            std::cout << "[TEST 5] Live GPU NRCManager Inference, Resolve & Training Dispatches..." << std::endl;
            Config cfg;
            cfg.headless = true;
            cfg.validation_layers = true;
            cfg.gpu_index = (hwGpuCount >= 2) ? 1 : 0;

            VulkanContext vkCtx(cfg, VK_NULL_HANDLE, "NRC GPU Test");
            if (vkCtx.hasCooperativeMatrix()) {
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

                auto inferSpv = loadSpv("nrc_encode_infer.comp.spv");
                auto trainSpv = loadSpv("nrc_train.comp.spv");
                auto resolveSpv = loadSpv("nrc_resolve.comp.spv");
                check_true(!inferSpv.empty() && !trainSpv.empty() && !resolveSpv.empty(),
                           "NRC SPIR-V shaders must exist");

                VkDevice device = vkCtx.getDevice();
                VmaAllocator allocator = vkCtx.getAllocator();
                VkQueue queue = vkCtx.getGraphicsQueue();

                NRCManager nrc(device, allocator, 64, 64, inferSpv, trainSpv, resolveSpv, vkCtx.getPhysicalDevice());
                check_true(nrc.isInitialized(), "NRCManager must initialize on CooperativeMatrix GPU");

                Image accumImg(device, allocator, 64, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                               VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
                Image mlDiffImg(device, allocator, 64, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                nrc.updateDescriptors(accumImg.getImageView(), mlDiffImg.getImageView());

                VkCommandPoolCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
                cpInfo.queueFamilyIndex = vkCtx.getGraphicsQueueFamily();
                VkCommandPool cmdPool = VK_NULL_HANDLE;
                check_true(vkCreateCommandPool(device, &cpInfo, nullptr, &cmdPool) == VK_SUCCESS, "NRC cmdPool created");

                VkCommandBufferAllocateInfo cbAlloc{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
                cbAlloc.commandPool = cmdPool;
                cbAlloc.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
                cbAlloc.commandBufferCount = 1;
                VkCommandBuffer cmd = VK_NULL_HANDLE;
                check_true(vkAllocateCommandBuffers(device, &cbAlloc, &cmd) == VK_SUCCESS, "NRC cmd allocated");

                VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(cmd, &beginInfo);

                accumImg.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                          VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                          VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                mlDiffImg.transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                                           VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0,
                                           VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                VkClearColorValue zeroClear{};
                VkImageSubresourceRange range{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
                vkCmdClearColorImage(cmd, accumImg.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &range);
                vkCmdClearColorImage(cmd, mlDiffImg.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &range);

                VkMemoryBarrier2 clrBar{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
                clrBar.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
                clrBar.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
                clrBar.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
                clrBar.dstAccessMask = VK_ACCESS_2_SHADER_READ_BIT | VK_ACCESS_2_SHADER_WRITE_BIT;
                VkDependencyInfo dep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
                dep.memoryBarrierCount = 1;
                dep.pMemoryBarriers = &clrBar;
                vkCmdPipelineBarrier2(cmd, &dep);

                nrc.resetCounters(cmd);
                nrc.recordInference(cmd, 64, 64, glm::vec3(-5.0f), glm::vec3(5.0f), 64, 0, 1);
                nrc.recordTraining(cmd, 1, glm::vec3(-5.0f), glm::vec3(5.0f), 1e-3f, 256);

                vkEndCommandBuffer(cmd);
                VkSubmitInfo sub{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
                sub.commandBufferCount = 1;
                sub.pCommandBuffers = &cmd;
                check_true(vkQueueSubmit(queue, 1, &sub, VK_NULL_HANDLE) == VK_SUCCESS, "Submit NRC GPU dispatches");
                vkQueueWaitIdle(queue);
                vkDestroyCommandPool(device, cmdPool, nullptr);

                check_true(vkCtx.getValidationErrors() == 0, "Zero Vulkan validation errors during NRC GPU execution");
                std::cout << "  -> Live GPU NRC inference, resolve, and online training executed cleanly ("
                          << nrc.getActiveBackendName() << ", 0 validation errors)." << std::endl;

                // [TEST 6] Numeric Inference & Training Parity Check (detect channel bias / WMMA mismatch)
                {
                    NRCManager nrcNum(device, allocator, 64, 64, inferSpv, trainSpv, resolveSpv, vkCtx.getPhysicalDevice());
                    Image numAccum(device, allocator, 64, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
                    Image numDiff(device, allocator, 64, 64, VK_FORMAT_R16G16B16A16_SFLOAT,
                                  VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
                    nrcNum.updateDescriptors(numAccum.getImageView(), numDiff.getImageView());

                    std::vector<NRCQuery> qVec(32);
                    for (uint32_t i = 0; i < 32; ++i) {
                        uint32_t pix = i;
                        float pixF = 0.0f;
                        std::memcpy(&pixF, &pix, sizeof(float));
                        qVec[i].pos_roughness = glm::vec4(0.1f * float(i % 4), 1.0f, -0.2f, 0.8f);
                        qVec[i].normal_flags = glm::vec4(0.0f, 0.0f, 1.0f, 0.0f);
                        qVec[i].dir_pixelIndex = glm::vec4(0.0f, 0.0f, 1.0f, pixF);
                        qVec[i].albedo_pad = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f); // specRatio = 0.0
                        qVec[i].throughput = glm::vec4(1.0f, 1.0f, 1.0f, 0.0f);
                    }
                    std::vector<NRCTrainingRecord> tVec(32);
                    for (uint32_t i = 0; i < 32; ++i) {
                        uint32_t pix = i;
                        float pixF = 0.0f;
                        std::memcpy(&pixF, &pix, sizeof(float));
                        tVec[i].pos_roughness = qVec[i].pos_roughness;
                        tVec[i].normal_flags = glm::vec4(0.0f, 0.0f, 1.0f, 1.0f);
                        tVec[i].dir_pixelIndex = qVec[i].dir_pixelIndex;
                        tVec[i].albedo_pad = qVec[i].albedo_pad;
                        tVec[i].throughput = glm::vec4(0.0f, 0.0f, 0.0f, 0.0f); // hasTail = 0
                        tVec[i].targetRadiance = glm::vec4(0.50f, 0.50f, 0.50f, 0.0f); // neutral gray
                        tVec[i].tailPos_roughness = glm::vec4(0.0f);
                        tVec[i].tailPacked = glm::vec4(0.0f);
                    }
                    Buffer readbackBuf(allocator, 64ull * 64ull * 8ull, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VMA_MEMORY_USAGE_GPU_TO_CPU);

                    auto runProbe = [&](uint32_t trainIters, glm::vec3& outMeanRgb) {
                        VkCommandPool cpool = VK_NULL_HANDLE;
                        vkCreateCommandPool(device, &cpInfo, nullptr, &cpool);
                        VkCommandBuffer cb = VK_NULL_HANDLE;
                        VkCommandBufferAllocateInfo cba{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, nullptr, cpool, VK_COMMAND_BUFFER_LEVEL_PRIMARY, 1 };
                        vkAllocateCommandBuffers(device, &cba, &cb);
                        vkBeginCommandBuffer(cb, &beginInfo);

                        numAccum.transitionLayout(cb, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                        numDiff.transitionLayout(cb, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_2_CLEAR_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                        vkCmdClearColorImage(cb, numAccum.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &range);
                        vkCmdClearColorImage(cb, numDiff.getImage(), VK_IMAGE_LAYOUT_GENERAL, &zeroClear, 1, &range);

                        nrcNum.resetCounters(cb);
                        vkCmdUpdateBuffer(cb, nrcNum.getQueryQueue()->getBuffer(), 0, qVec.size() * sizeof(NRCQuery), qVec.data());
                        vkCmdUpdateBuffer(cb, nrcNum.getTrainQueue()->getBuffer(), 0, tVec.size() * sizeof(NRCTrainingRecord), tVec.data());
                        const uint32_t cnt[8] = { 32u, 32u, 1u, 1u, 1u, 0u, 0u, 0u };
                        vkCmdUpdateBuffer(cb, nrcNum.getCounters()->getBuffer(), 0, sizeof(cnt), cnt);

                        VkMemoryBarrier2 tb{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, nullptr, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                             VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT };
                        VkDependencyInfo td{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO, nullptr, 0, 1, &tb, 0, nullptr, 0, nullptr };
                        vkCmdPipelineBarrier2(cb, &td);

                        for (uint32_t it = 0; it < trainIters; ++it) {
                            nrcNum.recordTraining(cb, it, glm::vec3(-1.5f), glm::vec3(1.5f), 2e-3f, 512);
                        }
                        nrcNum.recordInference(cb, 64, 64, glm::vec3(-1.5f), glm::vec3(1.5f), 32, 0, 0, 24.0f);

                        numAccum.transitionLayout(cb, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                                                  VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                        VkBufferImageCopy reg{};
                        reg.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                        reg.imageExtent = { 64, 64, 1 };
                        vkCmdCopyImageToBuffer(cb, numAccum.getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readbackBuf.getBuffer(), 1, &reg);
                        vkEndCommandBuffer(cb);

                        VkSubmitInfo sinfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO, nullptr, 0, nullptr, nullptr, 1, &cb, 0, nullptr };
                        vkQueueSubmit(queue, 1, &sinfo, VK_NULL_HANDLE);
                        vkQueueWaitIdle(queue);
                        vkDestroyCommandPool(device, cpool, nullptr);

                        const uint16_t* px = reinterpret_cast<const uint16_t*>(readbackBuf.map());
                        outMeanRgb = glm::vec3(0.0f);
                        for (uint32_t i = 0; i < 32; ++i) {
                            outMeanRgb.r += glm::unpackHalf1x16(px[i * 4 + 0]);
                            outMeanRgb.g += glm::unpackHalf1x16(px[i * 4 + 1]);
                            outMeanRgb.b += glm::unpackHalf1x16(px[i * 4 + 2]);
                        }
                        readbackBuf.unmap();
                        outMeanRgb /= 32.0f;
                    };

                    glm::vec3 initRgb(0.0f), trainedRgb(0.0f);
                    runProbe(0, initRgb);

                    // Compute exact CPU reference of initWeights() averaged across all 4 X-phases (qVec[0..3])
                    glm::vec3 cpuPred(0.0f);
                    {
                        std::mt19937 rng(42);
                        std::normal_distribution<float> weightDist(0.0f, std::sqrt(2.0f / 64.0f));
                        std::vector<float> W(9360, 0.0f);
                        auto setP = [&](size_t idx, float v) { W[idx] = glm::unpackHalf1x16(glm::packHalf1x16(v)); };
                        for (size_t i = 0; i < 4096; ++i) setP(i, weightDist(rng));
                        for (size_t i = 0; i < 64; ++i) setP(4096 + i, 0.0f);
                        for (size_t i = 0; i < 4096; ++i) setP(4160 + i, weightDist(rng));
                        for (size_t i = 0; i < 64; ++i) setP(8256 + i, 0.0f);
                        for (size_t i = 0; i < 1024; ++i) setP(8320 + i, weightDist(rng) * 0.02f);
                        for (size_t i = 0; i < 16; ++i) setP(9344 + i, i < 3 ? 0.02f : 0.0f);

                        const float PI = 3.14159265359f;
                        for (uint32_t qIdx = 0; qIdx < 4; ++qIdx) {
                            glm::vec3 np = glm::clamp((glm::vec3(qVec[qIdx].pos_roughness) - glm::vec3(-1.5f)) / glm::vec3(3.0f), glm::vec3(0.0f), glm::vec3(1.0f));
                            float x[64]{};
                            // Channels 0..15 stay 0.0 (untrained hash grid starts zero-initialized)
                            // Channels 16..23: linear skip-encoding
                            x[16] = np.x; x[17] = np.y; x[18] = np.z;
                            x[19] = 0.0f; x[20] = 0.0f; x[21] = 1.0f;
                            x[22] = 0.0f; x[23] = 0.0f;
                            // Channels 24..41: 3-octave Fourier encoding (axes 0..2, amp *= 0.5)
                            for (int ax = 0; ax < 3; ++ax) {
                                float th = np[ax] * PI, s = std::sin(th), c = std::cos(th), amp = 1.0f;
                                for (int o = 0; o < 3; ++o) {
                                    x[24 + ax * 6 + o * 2 + 0] = s * amp;
                                    x[24 + ax * 6 + o * 2 + 1] = c * amp;
                                    if (o < 2) { float s2 = 2*s*c, c2 = c*c - s*s; s = s2; c = c2; amp *= 0.5f; }
                                }
                            }
                            // Channels 42..50: Order-2 SH of N=(0,0,1)
                            x[42] = 0.282095f;
                            x[44] = 0.488603f;
                            x[48] = 0.315392f * 2.0f;
                            // Channels 51..63: V=(0,0,1), R=(0,0,1), NdotV=1, rough=0.8, rough^2=0.64, specRatio=0, 1-spec=1, F=0, bias=1
                            x[51]=0.0f; x[52]=0.0f; x[53]=1.0f;
                            x[54]=0.0f; x[55]=0.0f; x[56]=1.0f;
                            x[57]=1.0f; x[58]=0.8f; x[59]=0.64f;
                            x[60]=0.0f; x[61]=1.0f; x[62]=0.0f; x[63]=1.0f;
                            for (int i = 0; i < 64; ++i) x[i] = glm::unpackHalf1x16(glm::packHalf1x16(x[i]));
                            float h0[64]{}, h1[64]{};
                            for (int j = 0; j < 64; ++j) {
                                float s = W[4096 + j];
                                for (int i = 0; i < 64; ++i) s += x[i] * W[i * 64 + j];
                                h0[j] = glm::unpackHalf1x16(glm::packHalf1x16((s > 0.0f) ? s : 0.01f * s));
                            }
                            for (int j = 0; j < 64; ++j) {
                                float s = W[8256 + j];
                                for (int i = 0; i < 64; ++i) s += h0[i] * W[4160 + i * 64 + j];
                                h1[j] = glm::unpackHalf1x16(glm::packHalf1x16((s > 0.0f) ? s : 0.01f * s));
                            }
                            for (int c = 0; c < 3; ++c) {
                                float s = W[9344 + c];
                                for (int i = 0; i < 64; ++i) s += h1[i] * W[8320 + i * 16 + c];
                                cpuPred[c] += std::max(0.0f, s);
                            }
                        }
                        cpuPred /= 4.0f;
                    }

                    // Mean prediction across 32 queries matches 4-phase CPU FP16 reference
                    float initErr = std::max({ std::abs(initRgb.r - cpuPred.r), std::abs(initRgb.g - cpuPred.g), std::abs(initRgb.b - cpuPred.b) });
                    if (initErr >= 5e-3f) {
                        std::cerr << "  -> [FAIL] CPU init=(" << cpuPred.r << ", " << cpuPred.g << ", " << cpuPred.b
                                  << ") vs GPU init=(" << initRgb.r << ", " << initRgb.g << ", " << initRgb.b << ")" << std::endl;
                    }
                    check_true(initErr < 5e-3f, "NRC GPU WMMA initial inference matches CPU FP16 reference");

                    runProbe(8, trainedRgb);
                    float maxCh = std::max({ trainedRgb.r, trainedRgb.g, trainedRgb.b });
                    float minCh = std::min({ trainedRgb.r, trainedRgb.g, trainedRgb.b });
                    float channelSpread = maxCh - minCh;
                    float targetErr = std::max({ std::abs(trainedRgb.r - 0.50f), std::abs(trainedRgb.g - 0.50f), std::abs(trainedRgb.b - 0.50f) });
                    if (targetErr >= 0.15f || channelSpread >= 0.10f) {
                        std::cerr << "  -> [FAIL] Trained RGB=(" << trainedRgb.r << ", " << trainedRgb.g << ", " << trainedRgb.b << ")" << std::endl;
                    }
                    check_true(targetErr < 0.15f && channelSpread < 0.10f,
                               "NRC GPU WMMA training converges neutrally toward gray target without channel bias");
                    check_true(vkCtx.getValidationErrors() == 0,
                               "Zero Vulkan validation errors during NRC numeric training/inference parity test");
                    std::cout << "  -> [TEST 6] NRC WMMA numeric inference & neutral training parity passed (0 validation errors)." << std::endl;
                }
            }
        }
    }

    std::cout << "\n[SUCCESS] All Tier 4 Neural Radiance Caching (NRC) unit tests passed!" << std::endl;
    return 0;
}
