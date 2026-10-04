#include "scene/SceneGeometryPipeline.hpp"
#include "core/Logger.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace pathways {

SceneGeometryPipeline::SceneGeometryPipeline(
    VkDevice device,
    VkQueue queue,
    VmaAllocator allocator,
    VkCommandPool commandPool
) : m_device(device),
    m_queue(queue),
    m_allocator(allocator),
    m_commandPool(commandPool)
{
}

void SceneGeometryPipeline::updateSceneTransparency(
    const std::vector<MaterialGPU>& materials,
    bool& outHasNonOpaque,
    bool& outHasAlphaMask
) {
    outHasNonOpaque = false;
    outHasAlphaMask = false;
    for (const auto& mat : materials) {
        if (mat.alphaMode != 0) {
            outHasAlphaMask = true;
            outHasNonOpaque = true;
        } else if (mat.transmission > 0.05f || mat.type == 2) {
            outHasNonOpaque = true;
        }
    }
}

void SceneGeometryPipeline::partitionSceneGeometry(SceneData& sceneData, uint32_t& outNumOpaqueTriangles) {
    if (sceneData.triangles.empty()) {
        outNumOpaqueTriangles = 0;
        sceneData.numOpaqueTriangles = 0;
        return;
    }

    auto isOpaqueTriangle = [&sceneData](const TriangleGPU& tri) {
        if (tri.materialId >= sceneData.materials.size()) {
            return true;
        }
        const auto& mat = sceneData.materials[tri.materialId];
        return (mat.alphaMode == ALPHA_MODE_OPAQUE && mat.transmission <= 0.05f && mat.type != MATERIAL_DIELECTRIC);
    };

    if (!sceneData.blasRanges.empty()) {
        uint32_t totalOpaque = 0;
        for (auto& range : sceneData.blasRanges) {
            if (range.firstTriangle + range.triangleCount <= sceneData.triangles.size()) {
                auto rangeBegin = sceneData.triangles.begin() + range.firstTriangle;
                auto rangeEnd = rangeBegin + range.triangleCount;
                auto it = std::stable_partition(rangeBegin, rangeEnd, isOpaqueTriangle);
                range.numOpaqueTriangles = static_cast<uint32_t>(std::distance(rangeBegin, it));
                totalOpaque += range.numOpaqueTriangles;
            }
        }
        for (size_t i = 0; i < sceneData.instances.size(); ++i) {
            uint32_t bIdx = sceneData.instances[i].blasIndex;
            if (bIdx < sceneData.blasRanges.size() && i < sceneData.instanceData.size()) {
                sceneData.instanceData[i].firstTriangle = sceneData.blasRanges[bIdx].firstTriangle;
                sceneData.instanceData[i].numOpaqueTriangles = sceneData.blasRanges[bIdx].numOpaqueTriangles;
            }
        }
        outNumOpaqueTriangles = totalOpaque;
        sceneData.numOpaqueTriangles = totalOpaque;
    } else {
        auto it = std::stable_partition(sceneData.triangles.begin(), sceneData.triangles.end(), isOpaqueTriangle);
        outNumOpaqueTriangles = static_cast<uint32_t>(std::distance(sceneData.triangles.begin(), it));
        sceneData.numOpaqueTriangles = outNumOpaqueTriangles;
    }

    Logger::info("Geometry Partitioning: {} Opaque triangles, {} Non-Opaque triangles (Total: {})",
                 outNumOpaqueTriangles, sceneData.triangles.size() - outNumOpaqueTriangles, sceneData.triangles.size());
}

void SceneGeometryPipeline::clusterInstancesToMacroBlas(SceneData& scene, const Config& config) {
    if (!config.enable_macro_blas || scene.instances.size() <= 16 || scene.blasRanges.empty()) {
        return;
    }

    const uint32_t origInstanceCount = static_cast<uint32_t>(scene.instances.size());
    const uint32_t origBlasCount = static_cast<uint32_t>(scene.blasRanges.size());

    // 1. Calculate local bounding boxes for all prototype BLASes
    struct ProtoBounds {
        glm::vec3 minBound{ 1e30f };
        glm::vec3 maxBound{ -1e30f };
    };
    std::vector<ProtoBounds> protoBounds(scene.blasRanges.size());
    for (size_t b = 0; b < scene.blasRanges.size(); ++b) {
        const auto& range = scene.blasRanges[b];
        for (uint32_t t = 0; t < range.triangleCount; ++t) {
            uint32_t triIdx = range.firstTriangle + t;
            if (triIdx < scene.triangles.size()) {
                const auto& tri = scene.triangles[triIdx];
                protoBounds[b].minBound = glm::min(protoBounds[b].minBound, glm::vec3(tri.v0.position));
                protoBounds[b].minBound = glm::min(protoBounds[b].minBound, glm::vec3(tri.v1.position));
                protoBounds[b].minBound = glm::min(protoBounds[b].minBound, glm::vec3(tri.v2.position));
                protoBounds[b].maxBound = glm::max(protoBounds[b].maxBound, glm::vec3(tri.v0.position));
                protoBounds[b].maxBound = glm::max(protoBounds[b].maxBound, glm::vec3(tri.v1.position));
                protoBounds[b].maxBound = glm::max(protoBounds[b].maxBound, glm::vec3(tri.v2.position));
            }
        }
    }

    // 2. Compute world-space bounds and centers for all instances
    struct InstInfo {
        uint32_t origIdx;
        uint32_t bIdx;
        glm::vec3 worldCenter;
        uint32_t triCount;
    };
    std::vector<InstInfo> instInfos;
    instInfos.reserve(scene.instances.size());
    glm::vec3 sceneMin{ 1e30f };
    glm::vec3 sceneMax{ -1e30f };

    for (uint32_t i = 0; i < scene.instances.size(); ++i) {
        const auto& inst = scene.instances[i];
        if (inst.blasIndex >= scene.blasRanges.size()) continue;
        const auto& pb = protoBounds[inst.blasIndex];
        glm::vec3 localCenter = (pb.minBound + pb.maxBound) * 0.5f;
        glm::vec3 worldCenter = glm::vec3(inst.transform * glm::vec4(localCenter, 1.0f));

        uint32_t triCount = scene.blasRanges[inst.blasIndex].triangleCount;
        instInfos.push_back({ i, inst.blasIndex, worldCenter, triCount });
        sceneMin = glm::min(sceneMin, worldCenter);
        sceneMax = glm::max(sceneMax, worldCenter);
    }

    if (instInfos.empty()) return;

    // 3. Partition into spatial 2D grid along (X, Z)
    glm::vec3 extent = sceneMax - sceneMin;
    float extentX = std::max(extent.x, 1.0f);
    float extentZ = std::max(extent.z, 1.0f);

    uint32_t targetClusterCount = std::max(1u, static_cast<uint32_t>(instInfos.size() / std::max(1u, config.macro_blas_target_cluster)));
    float aspect = extentX / extentZ;
    uint32_t gridX = std::clamp(static_cast<uint32_t>(std::round(std::sqrt(static_cast<float>(targetClusterCount) * aspect))), 2u, 32u);
    uint32_t gridZ = std::clamp(static_cast<uint32_t>(std::round(std::sqrt(static_cast<float>(targetClusterCount) / std::max(aspect, 1e-4f)))), 2u, 32u);

    std::vector<std::vector<uint32_t>> cells(gridX * gridZ);
    for (uint32_t k = 0; k < instInfos.size(); ++k) {
        float u = std::clamp((instInfos[k].worldCenter.x - sceneMin.x) / extentX, 0.0f, 0.99999f);
        float v = std::clamp((instInfos[k].worldCenter.z - sceneMin.z) / extentZ, 0.0f, 0.99999f);
        uint32_t gx = static_cast<uint32_t>(u * gridX);
        uint32_t gz = static_cast<uint32_t>(v * gridZ);
        cells[gz * gridX + gx].push_back(k);
    }

    // 4. Decide which cells to merge into Macro-BLASes vs keep unmerged
    std::vector<std::vector<uint32_t>> macroClusters;
    std::vector<uint32_t> unmergedInstIndices;
    uint64_t accumulatedMacroTris = 0;
    const uint64_t maxMacroTrisBudget = static_cast<uint64_t>(config.macro_blas_max_tris);

    const uint32_t maxTrisPerMacroBlas = 65536;

    for (const auto& cell : cells) {
        if (cell.empty()) continue;
        if (cell.size() == 1) {
            // Single instance in cell: keeping prototype reference is optimal
            unmergedInstIndices.push_back(cell[0]);
            continue;
        }

        std::vector<uint32_t> currentSubCluster;
        uint32_t currentSubTris = 0;

        for (uint32_t idx : cell) {
            uint32_t instTris = instInfos[idx].triCount;
            // Keep large structural instances (e.g. multi-thousand tri skyscrapers) as clean prototype references
            if (instTris > config.macro_blas_max_prop_tris) {
                unmergedInstIndices.push_back(idx);
                continue;
            }
            if (currentSubTris + instTris > maxTrisPerMacroBlas && !currentSubCluster.empty()) {
                if (currentSubCluster.size() > 1 && (accumulatedMacroTris + currentSubTris <= maxMacroTrisBudget)) {
                    macroClusters.push_back(currentSubCluster);
                    accumulatedMacroTris += currentSubTris;
                } else {
                    for (uint32_t id : currentSubCluster) {
                        unmergedInstIndices.push_back(id);
                    }
                }
                currentSubCluster.clear();
                currentSubTris = 0;
            }
            currentSubCluster.push_back(idx);
            currentSubTris += instTris;
        }

        if (!currentSubCluster.empty()) {
            if (currentSubCluster.size() > 1 && (accumulatedMacroTris + currentSubTris <= maxMacroTrisBudget)) {
                macroClusters.push_back(currentSubCluster);
                accumulatedMacroTris += currentSubTris;
            } else {
                for (uint32_t id : currentSubCluster) {
                    unmergedInstIndices.push_back(id);
                }
            }
        }
    }

    if (macroClusters.empty()) {
        Logger::info("Macro-BLAS: Geometry budget ({} tris) precluded clustering; retained fine-grained instancing.", maxMacroTrisBudget);
        return;
    }

    // 5. Construct new SceneData structures
    std::vector<TriangleGPU> newTriangles;
    std::vector<BlasGeometryRange> newBlasRanges;
    std::vector<SceneInstance> newInstances;
    std::vector<InstanceGPU> newInstanceData;

    // Track which original prototype BLASes are still referenced by unmerged instances
    std::vector<int32_t> oldProtoToNew(origBlasCount, -1);
    for (uint32_t idx : unmergedInstIndices) {
        uint32_t oldB = instInfos[idx].bIdx;
        if (oldProtoToNew[oldB] == -1) {
            const auto& srcRange = scene.blasRanges[oldB];
            uint32_t newStart = static_cast<uint32_t>(newTriangles.size());
            for (uint32_t t = 0; t < srcRange.triangleCount; ++t) {
                newTriangles.push_back(scene.triangles[srcRange.firstTriangle + t]);
            }
            BlasGeometryRange dstRange{};
            dstRange.firstTriangle = newStart;
            dstRange.triangleCount = srcRange.triangleCount;
            dstRange.numOpaqueTriangles = srcRange.numOpaqueTriangles;
            oldProtoToNew[oldB] = static_cast<int32_t>(newBlasRanges.size());
            newBlasRanges.push_back(dstRange);
        }
    }

    // Bake Macro-BLAS clusters
    for (size_t c = 0; c < macroClusters.size(); ++c) {
        const auto& cluster = macroClusters[c];
        uint32_t macroStartTri = static_cast<uint32_t>(newTriangles.size());

        for (uint32_t k : cluster) {
            const auto& info = instInfos[k];
            const auto& inst = scene.instances[info.origIdx];
            const auto& protoRange = scene.blasRanges[info.bIdx];
            glm::mat4 M = inst.transform;
            float det = glm::determinant(glm::mat3(M));
            glm::mat3 normMat = (std::abs(det) > 1e-6f) ? glm::transpose(glm::inverse(glm::mat3(M))) : glm::mat3(1.0f);
            glm::mat3 tanMat = glm::mat3(M);
            uint32_t matOffset = (info.origIdx < scene.instanceData.size()) ? scene.instanceData[info.origIdx].materialOffset : 0;

            auto xformPos = [&](glm::vec4 p) {
                glm::vec4 worldPos = M * glm::vec4(glm::vec3(p), 1.0f);
                return glm::vec4(glm::vec3(worldPos), p.w);
            };

            auto xformNorm = [&](glm::vec4 n) {
                glm::vec3 v = normMat * glm::vec3(n);
                float l = glm::length(v);
                return glm::vec4(l > 1e-6f ? (v / l) : glm::vec3(0, 1, 0), n.w);
            };

            auto xformTan = [&](glm::vec4 tan) {
                glm::vec3 v = tanMat * glm::vec3(tan);
                float l = glm::length(v);
                return glm::vec4(l > 1e-6f ? (v / l) : glm::vec3(1, 0, 0), tan.w);
            };

            for (uint32_t t = 0; t < protoRange.triangleCount; ++t) {
                const auto& src = scene.triangles[protoRange.firstTriangle + t];
                TriangleGPU dst{};
                // Position (transform 3D pos with homogeneous w=1.0, preserving texture coord u in .w)
                dst.v0.position = xformPos(src.v0.position);
                dst.v1.position = xformPos(src.v1.position);
                dst.v2.position = xformPos(src.v2.position);

                // Normal
                dst.v0.normal = xformNorm(src.v0.normal);
                dst.v1.normal = xformNorm(src.v1.normal);
                dst.v2.normal = xformNorm(src.v2.normal);

                // Tangent
                dst.v0.tangent = xformTan(src.v0.tangent);
                dst.v1.tangent = xformTan(src.v1.tangent);
                dst.v2.tangent = xformTan(src.v2.tangent);

                dst.materialId = src.materialId + matOffset;
                newTriangles.push_back(dst);
            }
        }

        uint32_t macroTriCount = static_cast<uint32_t>(newTriangles.size() - macroStartTri);
        uint32_t macroBlasIdx = static_cast<uint32_t>(newBlasRanges.size());

        BlasGeometryRange macroRange{};
        macroRange.firstTriangle = macroStartTri;
        macroRange.triangleCount = macroTriCount;
        macroRange.numOpaqueTriangles = macroTriCount; // Updated during partitionSceneGeometry
        newBlasRanges.push_back(macroRange);

        SceneInstance macroInst{};
        macroInst.blasIndex = macroBlasIdx;
        macroInst.transform = glm::mat4(1.0f); // Identity
        macroInst.customIndex = static_cast<uint32_t>(newInstances.size());
        newInstances.push_back(macroInst);

        InstanceGPU macroGpu{};
        macroGpu.firstTriangle = macroStartTri;
        macroGpu.numOpaqueTriangles = macroTriCount;
        macroGpu.materialOffset = 0;
        macroGpu.flags = 0;
        newInstanceData.push_back(macroGpu);
    }

    // Append unmerged instances
    for (uint32_t idx : unmergedInstIndices) {
        const auto& info = instInfos[idx];
        const auto& origInst = scene.instances[info.origIdx];
        uint32_t newBIdx = static_cast<uint32_t>(oldProtoToNew[info.bIdx]);

        SceneInstance remInst{};
        remInst.blasIndex = newBIdx;
        remInst.transform = origInst.transform;
        remInst.customIndex = static_cast<uint32_t>(newInstances.size());
        newInstances.push_back(remInst);

        InstanceGPU remGpu{};
        remGpu.firstTriangle = newBlasRanges[newBIdx].firstTriangle;
        remGpu.numOpaqueTriangles = newBlasRanges[newBIdx].triangleCount;
        remGpu.materialOffset = (info.origIdx < scene.instanceData.size()) ? scene.instanceData[info.origIdx].materialOffset : 0;
        remGpu.flags = (info.origIdx < scene.instanceData.size()) ? scene.instanceData[info.origIdx].flags : 0;
        newInstanceData.push_back(remGpu);
    }

    uint32_t finalInstanceCount = static_cast<uint32_t>(newInstances.size());
    uint32_t finalBlasCount = static_cast<uint32_t>(newBlasRanges.size());

    scene.triangles = std::move(newTriangles);
    scene.blasRanges = std::move(newBlasRanges);
    scene.instances = std::move(newInstances);
    scene.instanceData = std::move(newInstanceData);

    float compressionPct = (1.0f - static_cast<float>(finalInstanceCount) / static_cast<float>(origInstanceCount)) * 100.0f;
    Logger::info("Macro-BLAS Merging: Clustered {} instances -> {} TLAS instances across {} BLASes ({:.1f}% TLAS reduction, +{:.2f}M baked tris, budget: {:.2f}M)",
                 origInstanceCount, finalInstanceCount, finalBlasCount, compressionPct,
                 static_cast<double>(accumulatedMacroTris) / 1e6, static_cast<double>(maxMacroTrisBudget) / 1e6);
}

void SceneGeometryPipeline::uploadToDeviceBuffer(Buffer& dstBuffer, const void* srcData, VkDeviceSize dataSize) {
    if (dataSize == 0 || !srcData) return;

    const VkDeviceSize maxChunkSize = 64 * 1024 * 1024; // 64 MB bounded staging buffer
    VkDeviceSize stagingSize = std::min(dataSize, maxChunkSize);

    Buffer stagingBuffer(
        m_allocator, stagingSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(m_device, &allocInfo, &cmd);

    VkDeviceSize offset = 0;
    while (offset < dataSize) {
        VkDeviceSize currentChunk = std::min(maxChunkSize, dataSize - offset);
        std::memcpy(stagingBuffer.map(), static_cast<const uint8_t*>(srcData) + offset, currentChunk);
        stagingBuffer.flush(0, currentChunk);

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = offset;
        copyRegion.size = currentChunk;
        vkCmdCopyBuffer(cmd, stagingBuffer.getBuffer(), dstBuffer.getBuffer(), 1, &copyRegion);

        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = cmd;
        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_queue);

        offset += currentChunk;
    }

    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
}

void SceneGeometryPipeline::uploadIndexBuffer(Buffer& dstBuffer, uint32_t triangleCount) {
    if (triangleCount == 0) {
        uint32_t dummy[3] = { 0, 1, 2 };
        uploadToDeviceBuffer(dstBuffer, dummy, sizeof(dummy));
        return;
    }

    const uint32_t chunkTriangles = 1048576; // 1M triangles = 12 MB chunk
    VkDeviceSize stagingSize = std::min(static_cast<VkDeviceSize>(triangleCount), static_cast<VkDeviceSize>(chunkTriangles)) * 3 * sizeof(uint32_t);

    Buffer stagingBuffer(
        m_allocator, stagingSize,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
        VMA_MEMORY_USAGE_AUTO_PREFER_HOST,
        VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT
    );

    VkCommandBufferAllocateInfo allocInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
    allocInfo.commandPool = m_commandPool;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;

    VkCommandBuffer cmd = VK_NULL_HANDLE;
    vkAllocateCommandBuffers(m_device, &allocInfo, &cmd);

    uint32_t* mappedStaging = static_cast<uint32_t*>(stagingBuffer.map());

    uint32_t triOffset = 0;
    while (triOffset < triangleCount) {
        uint32_t currentChunkTriangles = std::min(chunkTriangles, triangleCount - triOffset);
        for (uint32_t i = 0; i < currentChunkTriangles; ++i) {
            uint32_t k = triOffset + i;
            mappedStaging[i * 3 + 0] = 3 * k + 0;
            mappedStaging[i * 3 + 1] = 3 * k + 1;
            mappedStaging[i * 3 + 2] = 3 * k + 2;
        }
        VkDeviceSize currentBytes = currentChunkTriangles * 3 * sizeof(uint32_t);
        stagingBuffer.flush(0, currentBytes);

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        VkBufferCopy copyRegion{};
        copyRegion.srcOffset = 0;
        copyRegion.dstOffset = static_cast<VkDeviceSize>(triOffset) * 3 * sizeof(uint32_t);
        copyRegion.size = currentBytes;
        vkCmdCopyBuffer(cmd, stagingBuffer.getBuffer(), dstBuffer.getBuffer(), 1, &copyRegion);

        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = cmd;
        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(m_queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(m_queue);

        triOffset += currentChunkTriangles;
    }

    vkFreeCommandBuffers(m_device, m_commandPool, 1, &cmd);
}

} // namespace pathways
