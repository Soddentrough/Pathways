#include "rt/AccelerationStructurePipeline.hpp"
#include "core/Logger.hpp"
#include "scene/SceneGeometryPipeline.hpp"

#include <stdexcept>
#include <algorithm>

namespace pathways {

AccelerationStructurePipeline::AccelerationStructurePipeline(VkDevice device, VmaAllocator allocator, VkQueue queue, uint32_t queueFamily)
    : m_device(device), m_allocator(allocator), m_queue(queue), m_queueFamily(queueFamily) {
}

AccelerationStructurePipeline::~AccelerationStructurePipeline() {
    reset();
}

void AccelerationStructurePipeline::reset() {
    m_tlas.reset();
    m_blases.clear();
    m_instanceBuffer.reset();
    m_asManager.reset();
    m_asInstances.clear();
    m_initialShaderInstances.clear();
}

void AccelerationStructurePipeline::build(
    const SceneData& sceneData,
    uint32_t numTriangles,
    uint32_t numOpaqueTriangles,
    Buffer* positionBuffer,
    SceneGeometryPipeline* geomPipeline
) {
    reset();

    m_asManager = std::make_unique<AccelerationStructureManager>(
        m_device, m_allocator, m_queue, m_queueFamily
    );

    struct MonoBucket {
        uint32_t firstTri = 0;
        uint32_t triCount = 0;
        uint32_t numOpaqueField = 0;
        bool isGeomOpaque = true;
        uint8_t mask = 0x01;
    };

    auto buildMonolithicBuckets = [&]() -> std::vector<MonoBucket> {
        std::vector<MonoBucket> buckets;
        const uint32_t numNonOpaque = (numTriangles > numOpaqueTriangles) ? (numTriangles - numOpaqueTriangles) : 0;
        uint32_t numSolidNonOpaque = 0;
        if (numNonOpaque > 0) {
            if (sceneData.numSolidNonOpaqueTriangles > 0) {
                numSolidNonOpaque = std::min(sceneData.numSolidNonOpaqueTriangles, numNonOpaque);
            } else if (!sceneData.triangles.empty()) {
                const uint32_t endTri = std::min<uint32_t>(numTriangles, static_cast<uint32_t>(sceneData.triangles.size()));
                for (uint32_t t = numOpaqueTriangles; t < endTri; ++t) {
                    const uint32_t matId = sceneData.triangles[t].materialId;
                    if (matId < sceneData.materials.size() &&
                        sceneData.materials[matId].alphaMode != ALPHA_MODE_OPAQUE) {
                        break;
                    }
                    ++numSolidNonOpaque;
                }
            }
        }
        const uint32_t numAlphaNonOpaque = numNonOpaque - numSolidNonOpaque;

        if (numTriangles > 0 && numOpaqueTriangles > 0) {
            buckets.push_back({ 0, numOpaqueTriangles, numOpaqueTriangles, true, 0x01 });
        }
        if (numSolidNonOpaque > 0) {
            buckets.push_back({ numOpaqueTriangles, numSolidNonOpaque, 0, true, 0x02 });
        }
        if (numAlphaNonOpaque > 0) {
            buckets.push_back({ numOpaqueTriangles + numSolidNonOpaque, numAlphaNonOpaque, 0, false, 0x02 });
        }
        if (buckets.empty()) {
            buckets.push_back({ 0, 0, 0, true, 0x02 });
        }
        return buckets;
    };

    const std::vector<MonoBucket> monoBuckets =
        (sceneData.instanceData.empty() || sceneData.blasRanges.empty()) ? buildMonolithicBuckets() : std::vector<MonoBucket>{};

    // 1. Instance Buffer (std430, binding 30) — pure DEVICE_LOCAL
    std::vector<InstanceGPU>& instanceUpload = m_initialShaderInstances;
    if (!sceneData.instanceData.empty()) {
        instanceUpload = sceneData.instanceData;
        for (size_t i = 0; i < instanceUpload.size(); ++i) {
            if (i < sceneData.instances.size()) {
                instanceUpload[i].transform = sceneData.instances[i].transform;
                instanceUpload[i].invTransform = sceneData.instances[i].transform;
            }
        }
    } else {
        instanceUpload.reserve(monoBuckets.size());
        for (const auto& b : monoBuckets) {
            InstanceGPU inst{};
            inst.firstTriangle = b.firstTri;
            inst.numOpaqueTriangles = b.numOpaqueField;
            inst.materialOffset = 0;
            inst.flags = 0;
            instanceUpload.push_back(inst);
        }
    }

    VkDeviceSize instanceBufferSize = sizeof(InstanceGPU) * instanceUpload.size();
    m_instanceBuffer = std::make_unique<Buffer>(
        m_allocator, instanceBufferSize,
        VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (geomPipeline) {
        geomPipeline->uploadToDeviceBuffer(*m_instanceBuffer, instanceUpload.data(), instanceBufferSize);
    }

    // 2. Acceleration Structures (Indexless VK_INDEX_TYPE_NONE_KHR + VK_KHR_ray_tracing_position_fetch)
    VkDeviceAddress vertexBaseAddr = positionBuffer ? positionBuffer->getDeviceAddress(m_device) : 0;
    constexpr VkDeviceSize kTriVertexBytes = 3 * sizeof(glm::vec4);

    auto subrangeHasAlphaCutout = [&sceneData](uint32_t firstTri, uint32_t count, uint32_t matOffset) -> bool {
        const uint32_t endTri = std::min<uint32_t>(firstTri + count, static_cast<uint32_t>(sceneData.triangles.size()));
        for (uint32_t t = firstTri; t < endTri; ++t) {
            const uint32_t matId = sceneData.triangles[t].materialId + matOffset;
            if (matId < sceneData.materials.size() &&
                sceneData.materials[matId].alphaMode != ALPHA_MODE_OPAQUE) {
                return true;
            }
        }
        return false;
    };

    if (!sceneData.blasRanges.empty()) {
        // Multi-BLAS path: Batch all BLAS builds into a single GPU dispatch & compaction pass
        m_asManager->resetStats();
        std::vector<bool> blasHasAlphaCutout(sceneData.blasRanges.size(), false);
        for (size_t b = 0; b < sceneData.blasRanges.size(); ++b) {
            const auto& range = sceneData.blasRanges[b];
            const uint32_t numNonOpaque = (range.triangleCount > range.numOpaqueTriangles) ? (range.triangleCount - range.numOpaqueTriangles) : 0;
            if (numNonOpaque > 0 && subrangeHasAlphaCutout(range.firstTriangle + range.numOpaqueTriangles, numNonOpaque, 0)) {
                blasHasAlphaCutout[b] = true;
            }
        }
        for (size_t i = 0; i < sceneData.instances.size(); ++i) {
            const uint32_t bIdx = sceneData.instances[i].blasIndex;
            const uint32_t matOffset = (i < sceneData.instanceData.size()) ? sceneData.instanceData[i].materialOffset : 0;
            if (matOffset != 0 && bIdx < sceneData.blasRanges.size() && !blasHasAlphaCutout[bIdx]) {
                const auto& range = sceneData.blasRanges[bIdx];
                const uint32_t numNonOpaque = (range.triangleCount > range.numOpaqueTriangles) ? (range.triangleCount - range.numOpaqueTriangles) : 0;
                if (numNonOpaque > 0 && subrangeHasAlphaCutout(range.firstTriangle + range.numOpaqueTriangles, numNonOpaque, matOffset)) {
                    blasHasAlphaCutout[bIdx] = true;
                }
            }
        }

        std::vector<std::vector<ASGeometryInput>> allBlasGeoms;
        allBlasGeoms.reserve(sceneData.blasRanges.size());
        for (size_t b = 0; b < sceneData.blasRanges.size(); ++b) {
            const auto& range = sceneData.blasRanges[b];
            std::vector<ASGeometryInput> geoms;
            if (range.numOpaqueTriangles > 0) {
                ASGeometryInput geomOpaque{};
                geomOpaque.vertexBufferAddress = vertexBaseAddr + static_cast<VkDeviceSize>(range.firstTriangle) * kTriVertexBytes;
                geomOpaque.indexBufferAddress = 0;
                geomOpaque.vertexCount = 3 * range.numOpaqueTriangles - 1;
                geomOpaque.triangleCount = range.numOpaqueTriangles;
                geomOpaque.vertexStride = sizeof(glm::vec4);
                geomOpaque.indexType = VK_INDEX_TYPE_NONE_KHR;
                geomOpaque.isOpaque = true;
                geoms.push_back(geomOpaque);
            }
            uint32_t numNonOpaque = (range.triangleCount > range.numOpaqueTriangles) ? (range.triangleCount - range.numOpaqueTriangles) : 0;
            if (numNonOpaque > 0) {
                ASGeometryInput geomNonOpaque{};
                geomNonOpaque.vertexBufferAddress = vertexBaseAddr + static_cast<VkDeviceSize>(range.firstTriangle + range.numOpaqueTriangles) * kTriVertexBytes;
                geomNonOpaque.indexBufferAddress = 0;
                geomNonOpaque.vertexCount = 3 * numNonOpaque - 1;
                geomNonOpaque.triangleCount = numNonOpaque;
                geomNonOpaque.vertexStride = sizeof(glm::vec4);
                geomNonOpaque.indexType = VK_INDEX_TYPE_NONE_KHR;
                // Keep VK_GEOMETRY_OPAQUE_BIT_KHR on solid dielectrics; only disable hardware
                // opacity when the subrange genuinely contains alpha-cutout triangles.
                geomNonOpaque.isOpaque = !blasHasAlphaCutout[b];
                geoms.push_back(geomNonOpaque);
            }
            if (geoms.empty()) {
                ASGeometryInput dummyGeom{};
                dummyGeom.vertexBufferAddress = vertexBaseAddr;
                dummyGeom.indexBufferAddress = 0;
                dummyGeom.vertexCount = 2;
                dummyGeom.triangleCount = 1;
                dummyGeom.vertexStride = sizeof(glm::vec4);
                dummyGeom.indexType = VK_INDEX_TYPE_NONE_KHR;
                dummyGeom.isOpaque = true;
                geoms.push_back(dummyGeom);
            }
            allBlasGeoms.push_back(std::move(geoms));
        }
        m_blases = m_asManager->buildBLASBatch(allBlasGeoms);

        m_asInstances.clear();
        m_asInstances.reserve(sceneData.instances.size());
        for (const auto& inst : sceneData.instances) {
            ASInstanceInput asInst{};
            uint32_t bIdx = std::min(inst.blasIndex, static_cast<uint32_t>(m_blases.size() - 1));
            asInst.blasAddress = m_blases[bIdx]->getDeviceAddress();
            asInst.transform = inst.transform;
            asInst.customIndex = inst.customIndex;
            if (bIdx < sceneData.blasRanges.size()) {
                const auto& range = sceneData.blasRanges[bIdx];
                if (range.triangleCount == range.numOpaqueTriangles) {
                    asInst.mask = 0x01; // Pure opaque
                } else if (range.numOpaqueTriangles == 0) {
                    asInst.mask = 0x02; // Pure non-opaque / dielectric
                } else {
                    asInst.mask = 0x03; // Mixed
                }
            } else {
                asInst.mask = 0xFF;
            }
            asInst.hitGroupId = 0;
            asInst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            m_asInstances.push_back(asInst);
        }
        m_tlas = m_asManager->buildTLAS(m_asInstances);
        if (!m_tlas) {
            throw std::runtime_error("Hardware Ray Tracing Multi-BLAS TLAS build failed.");
        }
        Logger::info("Hardware Ray Tracing Multi-BLAS Acceleration Structures initialized successfully ({} BLASes, {} TLAS Instances).",
                     m_blases.size(), m_asInstances.size());
    } else {
        // Monolithic scene path (up to 3 BLAS buckets: Opaque, Solid Non-Opaque, Alpha Non-Opaque)
        m_asManager->resetStats();
        m_asInstances.clear();

        std::vector<std::vector<ASGeometryInput>> allMonoGeoms;
        allMonoGeoms.reserve(monoBuckets.size());
        for (const auto& b : monoBuckets) {
            std::vector<ASGeometryInput> geoms;
            ASGeometryInput geom{};
            if (b.triCount > 0) {
                geom.vertexBufferAddress = vertexBaseAddr + static_cast<VkDeviceSize>(b.firstTri) * kTriVertexBytes;
                geom.indexBufferAddress = 0;
                geom.vertexCount = 3 * b.triCount - 1;
                geom.triangleCount = b.triCount;
                geom.vertexStride = sizeof(glm::vec4);
                geom.indexType = VK_INDEX_TYPE_NONE_KHR;
                geom.isOpaque = b.isGeomOpaque;
            } else {
                geom.vertexBufferAddress = vertexBaseAddr;
                geom.indexBufferAddress = 0;
                geom.vertexCount = 2;
                geom.triangleCount = 1;
                geom.vertexStride = sizeof(glm::vec4);
                geom.indexType = VK_INDEX_TYPE_NONE_KHR;
                geom.isOpaque = true;
            }
            geoms.push_back(geom);
            allMonoGeoms.push_back(std::move(geoms));
        }

        if (allMonoGeoms.size() == 1) {
            m_blases.push_back(m_asManager->buildBLAS(allMonoGeoms[0]));
        } else {
            m_blases = m_asManager->buildBLASBatch(allMonoGeoms);
        }

        m_asInstances.reserve(monoBuckets.size());
        for (size_t idx = 0; idx < monoBuckets.size(); ++idx) {
            ASInstanceInput inst{};
            inst.blasAddress = m_blases[idx]->getDeviceAddress();
            inst.transform = glm::mat4(1.0f);
            inst.customIndex = static_cast<uint32_t>(idx);
            inst.mask = monoBuckets[idx].mask;
            inst.hitGroupId = 0;
            inst.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            m_asInstances.push_back(inst);
        }

        m_tlas = m_asManager->buildTLAS(m_asInstances);
        if (!m_tlas) {
            throw std::runtime_error("Hardware Ray Tracing TLAS build failed.");
        }
        Logger::info("Hardware Ray Tracing Acceleration Structures initialized successfully (Monolithic {} BLASes & {} TLAS Instances).",
                     m_blases.size(), m_asInstances.size());
    }
}

} // namespace pathways
