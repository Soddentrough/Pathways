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
    m_asIndexBuffer.reset();
    m_instanceBuffer.reset();
    m_asManager.reset();
    m_asInstances.clear();
}

void AccelerationStructurePipeline::build(
    const SceneData& sceneData,
    uint32_t numTriangles,
    uint32_t numOpaqueTriangles,
    Buffer* positionBuffer,
    SceneGeometryPipeline* geomPipeline
) {
    reset();

    VkDeviceSize indexBufferSize = std::max(static_cast<VkDeviceSize>(sizeof(uint32_t) * 3 * numTriangles), static_cast<VkDeviceSize>(sizeof(uint32_t) * 3));
    m_asIndexBuffer = std::make_unique<Buffer>(
        m_allocator, indexBufferSize,
        VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
        VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE,
        0
    );
    if (geomPipeline) {
        geomPipeline->uploadIndexBuffer(*m_asIndexBuffer, numTriangles);
    }

    m_asManager = std::make_unique<AccelerationStructureManager>(
        m_device, m_allocator, m_queue, m_queueFamily
    );

    // 1. Instance Buffer (std430, binding 30)
    std::vector<InstanceGPU> instanceUpload;
    if (!sceneData.instanceData.empty()) {
        instanceUpload = sceneData.instanceData;
    } else {
        uint32_t numNonOpaque = numTriangles - numOpaqueTriangles;
        if (numOpaqueTriangles > 0 && numNonOpaque > 0) {
            InstanceGPU instOpaque{};
            instOpaque.firstTriangle = 0;
            instOpaque.numOpaqueTriangles = numOpaqueTriangles;
            instOpaque.materialOffset = 0;
            instOpaque.flags = 0;
            instanceUpload.push_back(instOpaque);

            InstanceGPU instNonOpaque{};
            instNonOpaque.firstTriangle = numOpaqueTriangles;
            instNonOpaque.numOpaqueTriangles = 0;
            instNonOpaque.materialOffset = 0;
            instNonOpaque.flags = 0;
            instanceUpload.push_back(instNonOpaque);
        } else {
            InstanceGPU defaultInst{};
            defaultInst.firstTriangle = 0;
            defaultInst.numOpaqueTriangles = numOpaqueTriangles;
            defaultInst.materialOffset = 0;
            defaultInst.flags = 0;
            instanceUpload.push_back(defaultInst);
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

    // 2. Acceleration Structures
    VkDeviceAddress vertexBaseAddr = positionBuffer ? positionBuffer->getDeviceAddress(m_device) : 0;
    VkDeviceAddress indexBaseAddr = m_asIndexBuffer->getDeviceAddress(m_device);

    if (!sceneData.blasRanges.empty()) {
        // Multi-BLAS path: Batch all BLAS builds into a single GPU dispatch & compaction pass
        m_asManager->resetStats();
        std::vector<std::vector<ASGeometryInput>> allBlasGeoms;
        allBlasGeoms.reserve(sceneData.blasRanges.size());
        for (const auto& range : sceneData.blasRanges) {
            std::vector<ASGeometryInput> geoms;
            if (range.numOpaqueTriangles > 0) {
                ASGeometryInput geomOpaque{};
                geomOpaque.vertexBufferAddress = vertexBaseAddr;
                geomOpaque.indexBufferAddress = indexBaseAddr + static_cast<VkDeviceSize>(range.firstTriangle) * 3 * sizeof(uint32_t);
                geomOpaque.vertexCount = 3 * (range.firstTriangle + range.numOpaqueTriangles);
                geomOpaque.triangleCount = range.numOpaqueTriangles;
                geomOpaque.vertexStride = sizeof(glm::vec4);
                geomOpaque.indexType = VK_INDEX_TYPE_UINT32;
                geomOpaque.isOpaque = true;
                geoms.push_back(geomOpaque);
            }
            uint32_t numNonOpaque = (range.triangleCount > range.numOpaqueTriangles) ? (range.triangleCount - range.numOpaqueTriangles) : 0;
            if (numNonOpaque > 0) {
                ASGeometryInput geomNonOpaque{};
                geomNonOpaque.vertexBufferAddress = vertexBaseAddr;
                geomNonOpaque.indexBufferAddress = indexBaseAddr + static_cast<VkDeviceSize>(range.firstTriangle + range.numOpaqueTriangles) * 3 * sizeof(uint32_t);
                geomNonOpaque.vertexCount = 3 * (range.firstTriangle + range.triangleCount);
                geomNonOpaque.triangleCount = numNonOpaque;
                geomNonOpaque.vertexStride = sizeof(glm::vec4);
                geomNonOpaque.indexType = VK_INDEX_TYPE_UINT32;
                geomNonOpaque.isOpaque = false;
                geoms.push_back(geomNonOpaque);
            }
            if (geoms.empty()) {
                ASGeometryInput dummyGeom{};
                dummyGeom.vertexBufferAddress = vertexBaseAddr;
                dummyGeom.indexBufferAddress = indexBaseAddr;
                dummyGeom.vertexCount = 3;
                dummyGeom.triangleCount = 1;
                dummyGeom.vertexStride = sizeof(glm::vec4);
                dummyGeom.indexType = VK_INDEX_TYPE_UINT32;
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
        // Monolithic scene path
        uint32_t numNonOpaque = numTriangles - numOpaqueTriangles;
        m_asInstances.clear();

        if (numOpaqueTriangles > 0 && numNonOpaque > 0) {
            std::vector<ASGeometryInput> geomsOpaque;
            ASGeometryInput geomOpaque{};
            geomOpaque.vertexBufferAddress = vertexBaseAddr;
            geomOpaque.indexBufferAddress = indexBaseAddr;
            geomOpaque.vertexCount = 3 * numOpaqueTriangles;
            geomOpaque.triangleCount = numOpaqueTriangles;
            geomOpaque.vertexStride = sizeof(glm::vec4);
            geomOpaque.indexType = VK_INDEX_TYPE_UINT32;
            geomOpaque.isOpaque = true;
            geomsOpaque.push_back(geomOpaque);

            std::vector<ASGeometryInput> geomsNonOpaque;
            ASGeometryInput geomNonOpaque{};
            geomNonOpaque.vertexBufferAddress = vertexBaseAddr;
            geomNonOpaque.indexBufferAddress = indexBaseAddr + static_cast<VkDeviceSize>(numOpaqueTriangles) * 3 * sizeof(uint32_t);
            geomNonOpaque.vertexCount = 3 * numTriangles;
            geomNonOpaque.triangleCount = numNonOpaque;
            geomNonOpaque.vertexStride = sizeof(glm::vec4);
            geomNonOpaque.indexType = VK_INDEX_TYPE_UINT32;
            geomNonOpaque.isOpaque = false;
            geomsNonOpaque.push_back(geomNonOpaque);

            m_blases = m_asManager->buildBLASBatch({ geomsOpaque, geomsNonOpaque });

            ASInstanceInput inst0{};
            inst0.blasAddress = m_blases[0]->getDeviceAddress();
            inst0.transform = glm::mat4(1.0f);
            inst0.customIndex = 0;
            inst0.mask = 0x01; // RAY_MASK_OPAQUE
            inst0.hitGroupId = 0;
            inst0.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            m_asInstances.push_back(inst0);

            ASInstanceInput inst1{};
            inst1.blasAddress = m_blases[1]->getDeviceAddress();
            inst1.transform = glm::mat4(1.0f);
            inst1.customIndex = 1;
            inst1.mask = 0x02; // RAY_MASK_NON_OPAQUE
            inst1.hitGroupId = 0;
            inst1.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
            m_asInstances.push_back(inst1);
        } else {
            std::vector<ASGeometryInput> geoms;
            bool isPureOpaque = (numOpaqueTriangles > 0);
            if (isPureOpaque) {
                ASGeometryInput geomOpaque{};
                geomOpaque.vertexBufferAddress = vertexBaseAddr;
                geomOpaque.indexBufferAddress = indexBaseAddr;
                geomOpaque.vertexCount = 3 * numOpaqueTriangles;
                geomOpaque.triangleCount = numOpaqueTriangles;
                geomOpaque.vertexStride = sizeof(glm::vec4);
                geomOpaque.indexType = VK_INDEX_TYPE_UINT32;
                geomOpaque.isOpaque = true;
                geoms.push_back(geomOpaque);
            } else if (numNonOpaque > 0) {
                ASGeometryInput geomNonOpaque{};
                geomNonOpaque.vertexBufferAddress = vertexBaseAddr;
                geomNonOpaque.indexBufferAddress = indexBaseAddr;
                geomNonOpaque.vertexCount = 3 * numTriangles;
                geomNonOpaque.triangleCount = numNonOpaque;
                geomNonOpaque.vertexStride = sizeof(glm::vec4);
                geomNonOpaque.indexType = VK_INDEX_TYPE_UINT32;
                geomNonOpaque.isOpaque = false;
                geoms.push_back(geomNonOpaque);
            } else {
                ASGeometryInput dummyGeom{};
                dummyGeom.vertexBufferAddress = vertexBaseAddr;
                dummyGeom.indexBufferAddress = indexBaseAddr;
                dummyGeom.vertexCount = 3;
                dummyGeom.triangleCount = 1;
                dummyGeom.vertexStride = sizeof(glm::vec4);
                dummyGeom.indexType = VK_INDEX_TYPE_UINT32;
                dummyGeom.isOpaque = true;
                geoms.push_back(dummyGeom);
            }

            m_blases.push_back(m_asManager->buildBLAS(geoms));

            ASInstanceInput inst{};
            inst.blasAddress = m_blases[0]->getDeviceAddress();
            inst.transform = glm::mat4(1.0f);
            inst.customIndex = 0;
            inst.mask = isPureOpaque ? 0x01 : 0x02;
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
