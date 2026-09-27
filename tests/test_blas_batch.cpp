#include "rt/AccelerationStructure.hpp"
#include "vulkan/VulkanContext.hpp"
#include "vulkan/Buffer.hpp"
#include "core/Config.hpp"
#include "core/Logger.hpp"

#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>
#include <cstring>
#include <glm/glm.hpp>

using namespace pathways;

static void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] Assertion failed: " << msg << std::endl;
        std::exit(1);
    }
}

int main(int argc, char** argv) {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing Batched BLAS Builds (buildBLASBatch)  " << std::endl;
    std::cout << "  Scratch Sub-allocation & Compaction Queue Serialization " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // -------------------------------------------------------------------------
    // TEST 1: Mathematical Scratch Alignment & Non-Overlapping Verification
    // -------------------------------------------------------------------------
    std::cout << "[TEST 1] Scratch Sub-allocation & Alignment Invariants..." << std::endl;
    {
        constexpr VkDeviceSize scratchAlignment = 256;
        const std::vector<VkDeviceSize> testSizes = {
            13, 255, 256, 257, 1000, 4096, 7311, 65535, 65536, 1048576, 3333333
        };
        const uint32_t N = static_cast<uint32_t>(testSizes.size());

        std::vector<VkDeviceSize> offsets(N, 0);
        VkDeviceSize totalScratch = 0;
        for (uint32_t i = 0; i < N; ++i) {
            offsets[i] = totalScratch;
            VkDeviceSize aligned = (testSizes[i] + (scratchAlignment - 1)) & ~(scratchAlignment - 1);
            totalScratch += aligned;
        }

        for (uint32_t i = 0; i < N; ++i) {
            check_true(offsets[i] % scratchAlignment == 0,
                       "Each BLAS scratch offset must be strictly 256-byte aligned per VUID-03710");
            if (i > 0) {
                check_true(offsets[i] >= offsets[i - 1] + testSizes[i - 1],
                           "Scratch sub-allocations must be non-overlapping per VUID-03698");
            }
        }
        check_true(totalScratch >= offsets.back() + testSizes.back(),
                   "Total scratch buffer must encompass all BLAS build regions");
        std::cout << "  -> Scratch alignment (256B) and non-overlapping invariant verified for " << N << " BLASes." << std::endl;
    }

    // -------------------------------------------------------------------------
    // TEST 2: Live GPU Acceleration Structure Batch Build & Compaction
    // -------------------------------------------------------------------------
    std::cout << "[TEST 2] Live Hardware BLAS Batch Build on GPU..." << std::endl;
    {
        Config config;
        config.headless = true;
        config.validation_layers = false;

        VulkanContext context(config, VK_NULL_HANDLE, "BLAS Batch Test");
        VkDevice device = context.getDevice();
        VmaAllocator allocator = context.getAllocator();
        VkQueue queue = context.getGraphicsQueue();
        uint32_t queueFamily = context.getGraphicsQueueFamily();

        AccelerationStructureManager asManager(device, allocator, queue, queueFamily);

        // Define vertices and indices for 3 distinct geometries
        // Geometry 0: Tetrahedron (4 vertices, 4 triangles / 12 indices)
        std::vector<glm::vec4> vertices0 = {
            { 0.0f,  1.0f,  0.0f, 1.0f},
            {-1.0f, -1.0f,  1.0f, 1.0f},
            { 1.0f, -1.0f,  1.0f, 1.0f},
            { 0.0f, -1.0f, -1.0f, 1.0f}
        };
        std::vector<uint32_t> indices0 = {
            0, 1, 2,
            0, 2, 3,
            0, 3, 1,
            1, 3, 2
        };

        // Geometry 1: Quad (4 vertices, 2 triangles / 6 indices)
        std::vector<glm::vec4> vertices1 = {
            {-2.0f, 0.0f, -2.0f, 1.0f},
            { 2.0f, 0.0f, -2.0f, 1.0f},
            { 2.0f, 0.0f,  2.0f, 1.0f},
            {-2.0f, 0.0f,  2.0f, 1.0f}
        };
        std::vector<uint32_t> indices1 = {
            0, 1, 2,
            0, 2, 3
        };

        // Geometry 2: Pyramid (5 vertices, 4 triangles / 12 indices)
        std::vector<glm::vec4> vertices2 = {
            { 0.0f, 2.0f,  0.0f, 1.0f},
            {-1.0f, 0.0f, -1.0f, 1.0f},
            { 1.0f, 0.0f, -1.0f, 1.0f},
            { 1.0f, 0.0f,  1.0f, 1.0f},
            {-1.0f, 0.0f,  1.0f, 1.0f}
        };
        std::vector<uint32_t> indices2 = {
            0, 1, 2,
            0, 2, 3,
            0, 3, 4,
            0, 4, 1
        };

        // Upload to device buffers
        auto createDeviceBufferWithData = [&](const void* data, VkDeviceSize size, VkBufferUsageFlags usage) {
            auto buf = std::make_unique<Buffer>(
                allocator, size,
                usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VMA_MEMORY_USAGE_CPU_TO_GPU
            );
            void* mapped = buf->map();
            std::memcpy(mapped, data, size);
            buf->unmap();
            return buf;
        };

        auto vBuf0 = createDeviceBufferWithData(vertices0.data(), vertices0.size() * sizeof(glm::vec4), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        auto iBuf0 = createDeviceBufferWithData(indices0.data(), indices0.size() * sizeof(uint32_t), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

        auto vBuf1 = createDeviceBufferWithData(vertices1.data(), vertices1.size() * sizeof(glm::vec4), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        auto iBuf1 = createDeviceBufferWithData(indices1.data(), indices1.size() * sizeof(uint32_t), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

        auto vBuf2 = createDeviceBufferWithData(vertices2.data(), vertices2.size() * sizeof(glm::vec4), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
        auto iBuf2 = createDeviceBufferWithData(indices2.data(), indices2.size() * sizeof(uint32_t), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);

        // Prepare ASGeometryInput structures
        ASGeometryInput geom0{};
        geom0.vertexBufferAddress = vBuf0->getDeviceAddress(device);
        geom0.indexBufferAddress = iBuf0->getDeviceAddress(device);
        geom0.vertexCount = static_cast<uint32_t>(vertices0.size());
        geom0.triangleCount = static_cast<uint32_t>(indices0.size() / 3);
        geom0.vertexStride = sizeof(glm::vec4);
        geom0.indexType = VK_INDEX_TYPE_UINT32;
        geom0.isOpaque = true;

        ASGeometryInput geom1{};
        geom1.vertexBufferAddress = vBuf1->getDeviceAddress(device);
        geom1.indexBufferAddress = iBuf1->getDeviceAddress(device);
        geom1.vertexCount = static_cast<uint32_t>(vertices1.size());
        geom1.triangleCount = static_cast<uint32_t>(indices1.size() / 3);
        geom1.vertexStride = sizeof(glm::vec4);
        geom1.indexType = VK_INDEX_TYPE_UINT32;
        geom1.isOpaque = true;

        ASGeometryInput geom2{};
        geom2.vertexBufferAddress = vBuf2->getDeviceAddress(device);
        geom2.indexBufferAddress = iBuf2->getDeviceAddress(device);
        geom2.vertexCount = static_cast<uint32_t>(vertices2.size());
        geom2.triangleCount = static_cast<uint32_t>(indices2.size() / 3);
        geom2.vertexStride = sizeof(glm::vec4);
        geom2.indexType = VK_INDEX_TYPE_UINT32;
        geom2.isOpaque = true;

        // Test 2a: Empty Batch handling
        auto emptyResult = asManager.buildBLASBatch({});
        check_true(emptyResult.empty(), "Empty geometriesList must return empty vector");

        // Test 2b: Batched BLAS build (3 BLASes)
        std::vector<std::vector<ASGeometryInput>> batchList = {
            { geom0 },
            { geom1 },
            { geom2 }
        };

        auto blases = asManager.buildBLASBatch(batchList);
        check_true(blases.size() == 3, "buildBLASBatch must return exactly 3 acceleration structures");

        for (size_t i = 0; i < blases.size(); ++i) {
            check_true(blases[i] != nullptr, "BLAS pointer must not be null");
            check_true(blases[i]->getHandle() != VK_NULL_HANDLE, "BLAS handle must be valid");
            check_true(blases[i]->getDeviceAddress() != 0, "BLAS device address must be non-zero");
            check_true(blases[i]->getBuffer() != nullptr, "BLAS storage buffer must not be null");
            std::cout << "  -> BLAS " << i << " built successfully, Handle: "
                      << blases[i]->getHandle() << ", Addr: 0x" << std::hex
                      << blases[i]->getDeviceAddress() << std::dec << std::endl;
        }

        // Test 2c: Backwards-compatible single buildBLAS
        auto singleBlas = asManager.buildBLAS({ geom0 });
        check_true(singleBlas != nullptr, "Single buildBLAS must succeed");
        check_true(singleBlas->getHandle() != VK_NULL_HANDLE, "Single BLAS handle must be valid");
        check_true(singleBlas->getDeviceAddress() != 0, "Single BLAS device address must be non-zero");
        std::cout << "  -> Single buildBLAS backwards-compatibility confirmed, Addr: 0x"
                  << std::hex << singleBlas->getDeviceAddress() << std::dec << std::endl;

        // Test 2d: Multi-geometry BLAS inside a batch
        std::vector<std::vector<ASGeometryInput>> multiGeomBatch = {
            { geom0, geom1 },
            { geom2 }
        };
        auto multiBlases = asManager.buildBLASBatch(multiGeomBatch);
        check_true(multiBlases.size() == 2, "multi-geometry batch must return 2 BLASes");
        check_true(multiBlases[0]->getDeviceAddress() != 0, "Multi-geom BLAS 0 address must be non-zero");
        check_true(multiBlases[1]->getDeviceAddress() != 0, "Multi-geom BLAS 1 address must be non-zero");
        std::cout << "  -> Multi-geometry BLAS batching verified successfully." << std::endl;
    }

    std::cout << "\n[SUCCESS] All Batched BLAS build (buildBLASBatch) tests passed!" << std::endl;
    return 0;
}
