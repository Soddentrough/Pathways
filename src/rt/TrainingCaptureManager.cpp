#include "rt/TrainingCaptureManager.hpp"
#include "core/Engine.hpp"
#include "core/Logger.hpp"
#include "utils/ImageDumper.hpp"
#include "vulkan/VulkanContext.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "scene/Camera.hpp"
#include "scene/ProceduralScene.hpp"
#include "scene/SceneManager.hpp"
#include "rt/WavefrontPipeline.hpp"
#include "rt/ReSTIRManager.hpp"
#include "rt/UpwaysPipeline.hpp"

#include <filesystem>
#include <format>
#include <cmath>
#include <algorithm>
#include <vector>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/packing.hpp>
#include <glm/detail/type_half.hpp>

namespace pathways {

TrainingCaptureManager::TrainingCaptureManager(Engine* engine)
    : m_engine(engine)
{
}

void TrainingCaptureManager::resetChoreography() {
    m_choreoInitialized = false;
    m_choreoInitialPos = glm::vec3(0.0f);
    m_choreoInitialYaw = 0.0f;
    m_choreoInitialPitch = 0.0f;
    m_choreoInitialFov = 45.0f;
}

void TrainingCaptureManager::captureTrainingFrame(uint32_t frameIdx, bool isReference, uint32_t spp) {
    if (!m_engine) return;

    VkDevice device = m_engine->m_context->getDevice();
    VkQueue queue = m_engine->m_context->getGraphicsQueue();
    VmaAllocator allocator = m_engine->m_context->getAllocator();
    uint32_t width = m_engine->m_config.width;
    uint32_t height = m_engine->m_config.height;
    if (isReference) {
        if (m_engine->m_config.capture_ref_width > 0 && m_engine->m_config.capture_ref_height > 0) {
            width = std::min(m_engine->m_config.capture_ref_width, m_engine->m_config.width);
            height = std::min(m_engine->m_config.capture_ref_height, m_engine->m_config.height);
        }
    } else {
        if (m_engine->m_config.capture_input_width > 0 && m_engine->m_config.capture_input_height > 0) {
            width = std::min(m_engine->m_config.capture_input_width, m_engine->m_config.width);
            height = std::min(m_engine->m_config.capture_input_height, m_engine->m_config.height);
        }
    }

    if (!m_engine->m_wavefrontPipeline || !m_engine->m_accumImage || !m_engine->m_camera) {
        Logger::error("Cannot capture training frame: Wavefront pipeline, accum image, or camera is null!");
        return;
    }

    uint32_t requiredCapacity = width * height;
    if (m_engine->m_wavefrontPipeline->getMaxCapacity() < requiredCapacity) {
        m_engine->m_wavefrontPipeline->resize(width, height, requiredCapacity);
        m_engine->updateWavefrontSceneDescriptors();
    }
    if (!isReference && m_engine->m_config.capture_channels >= 44u &&
        (!m_engine->m_renderTargets || !m_engine->m_renderTargets->hasSplitSignalResources())) {
        m_engine->ensureSplitSignalResources();
    }

    uint32_t flags = 0;
    if (m_engine->m_config.enable_direct_light)    flags |= (1 << 0);
    if (m_engine->m_config.enable_indirect_light)  flags |= (1 << 1);
    flags |= (1 << 2); // Specular
    if (m_engine->m_config.enable_refraction)      flags |= (1 << 3);
    if (m_engine->m_config.enable_shadows)         flags |= (1 << 4);
    if (m_engine->m_sceneHasNonOpaque)             flags |= (1 << 5);
    if (m_engine->m_sceneHasAlphaMask)             flags |= (1 << 10);
    if (m_engine->m_config.inline_primary_shadows) flags |= (1 << 6);
    const bool useCaptureRestir = (!isReference && m_engine->isRestirActive() && m_engine->m_config.capture_channels >= 23);
    if (m_engine->m_config.enable_light_tree || (!m_engine->m_sceneData.lightTreeNodes.empty() && useCaptureRestir)) flags |= (1 << 7);
    if (useCaptureRestir) flags |= (1 << 9);
    if (m_engine->m_config.enable_delta_unroll) flags |= (1 << 11);
    if (m_engine->m_config.transparent_shadows && !m_engine->m_config.reference_mode) flags |= (1 << 26);

    VkClearColorValue clearZero{};
    clearZero.float32[0] = 0.0f; clearZero.float32[1] = 0.0f; clearZero.float32[2] = 0.0f; clearZero.float32[3] = 0.0f;
    VkImageSubresourceRange colorRange{ VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

    auto clearImage = [&](VkCommandBuffer cmd, Image* img) {
        if (!img) return;
        img->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, VK_ACCESS_2_NONE,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        vkCmdClearColorImage(cmd, img->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &clearZero, 1, &colorRange);
        img->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
    };

    const uint32_t outChannels = m_engine->m_config.capture_channels;
    Image* targetAccumImg = m_engine->m_frameImages[0];
    const VkFormat targetFmt = targetAccumImg ? targetAccumImg->getFormat() : VK_FORMAT_R32G32B32A32_SFLOAT;
    const bool isTargetFp16 = (targetFmt == VK_FORMAT_R16G16B16A16_SFLOAT);
    const bool isTargetRg11 = (targetFmt == VK_FORMAT_B10G11R11_UFLOAT_PACK32);

    auto sanitizeHalf = [](uint16_t h) -> uint16_t {
        if ((h & 0x7C00u) == 0x7C00u) {
            return ((h & 0x03FFu) != 0u) ? 0x0000u : ((h & 0x8000u) ? 0xFBFFu : 0x7BFFu);
        }
        return h;
    };
    auto safeToFp16 = [](float val) -> uint16_t {
        if (!std::isfinite(val)) {
            return (val > 0.0f) ? glm::detail::toFloat16(65500.0f) : glm::detail::toFloat16(0.0f);
        }
        return glm::detail::toFloat16(std::clamp(val, -65500.0f, 65500.0f));
    };

    if (isReference) {
        // --- GROUND TRUTH REFERENCE CAPTURE PASS ---
        VkDeviceSize accumPixelBytes = isTargetRg11 ? sizeof(uint32_t) : (isTargetFp16 ? (4 * sizeof(uint16_t)) : (4 * sizeof(float)));
        VkDeviceSize stagingSize = static_cast<VkDeviceSize>(width) * height * accumPixelBytes;
        Buffer stagingAccum(allocator, stagingSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        auto renderReferenceSeed = [&](uint32_t seedBase, uint32_t seedSpp, std::vector<float>& outRgb) {
            outRgb.assign(static_cast<size_t>(width) * height * 3, 0.0f);
            {
                VkCommandBuffer cmd = m_engine->m_commandBuffers[0];
                vkResetCommandBuffer(cmd, 0);
                VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(cmd, &beginInfo);
                clearImage(cmd, targetAccumImg);
                vkEndCommandBuffer(cmd);

                VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
                cmdSubmitInfo.commandBuffer = cmd;
                VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
                submitInfo.commandBufferInfoCount = 1;
                submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
                vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
                vkQueueWaitIdle(queue);
            }

            uint32_t remainingSpp = seedSpp;
            uint32_t currentSppOffset = 0;
            constexpr uint32_t BATCH_SIZE = 16;

            WavefrontSceneData wfSceneData{};
            wfSceneData.numTriangles = m_engine->m_numTriangles;
            wfSceneData.numMaterials = m_engine->m_numMaterials;
            wfSceneData.numLights = m_engine->m_numLights;
            wfSceneData.hasEnvMap = m_engine->m_environmentMap ? 1u : 0u;
            wfSceneData.envMapIntensity = 1.0f;
            wfSceneData.useMorton = m_engine->m_config.use_morton ? 1u : 0u;
            wfSceneData.accumulateHistory = 1u;
            wfSceneData.sortMode = static_cast<uint32_t>(m_engine->getEffectiveWavefrontSortMode());
            wfSceneData.numOpaqueTriangles = m_engine->m_numOpaqueTriangles;
            wfSceneData.secondarySortMode = static_cast<uint32_t>(m_engine->m_config.secondary_sort_mode);
            wfSceneData.cameraFlags = flags;
            wfSceneData.boundsMin = m_engine->m_sceneData.boundsMin;
            wfSceneData.boundsMax = m_engine->m_sceneData.boundsMax;
            wfSceneData.streamlineSecondaryShading = m_engine->m_config.streamline_secondary_shading;
            wfSceneData.enableDistanceClamping = m_engine->m_config.distance_clamping;
            wfSceneData.maxSecondaryRayDistance = m_engine->m_config.max_secondary_distance;
            wfSceneData.indirectClamp = m_engine->m_config.indirect_clamp;
            wfSceneData.enableTailMegakernel = m_engine->m_config.enable_tail_megakernel;
            wfSceneData.tailMegakernelBounce = m_engine->m_config.tail_megakernel_bounce;
            wfSceneData.inlineShadows = m_engine->m_config.inline_primary_shadows;
            wfSceneData.captureMlData = 0;

            uint32_t activeOfflineBounces = m_engine->m_config.max_bounces;
            while (remainingSpp > 0) {
                uint32_t batchSpp = std::min(remainingSpp, BATCH_SIZE);
                wfSceneData.frameIndex = seedBase + frameIdx * 10000u + currentSppOffset;

                CameraUniform ubo = m_engine->m_camera->getUniformData(wfSceneData.frameIndex, 1, activeOfflineBounces, flags,
                                                             false, width, height, 0, /*updatePrev=*/false,
                                                             m_engine->m_config.capture_halton_length);
                m_engine->m_cameraUBOs[0]->copyFrom(&ubo, sizeof(CameraUniform));

                VkCommandBuffer cmd = m_engine->m_commandBuffers[0];
                vkResetCommandBuffer(cmd, 0);
                VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(cmd, &beginInfo);
                m_engine->m_wavefrontPipeline->recordFrame(cmd, 0, width, height, batchSpp, activeOfflineBounces, wfSceneData);
                vkEndCommandBuffer(cmd);

                VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
                cmdSubmitInfo.commandBuffer = cmd;
                VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
                submitInfo.commandBufferInfoCount = 1;
                submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
                vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
                vkQueueWaitIdle(queue);

                auto profData = m_engine->m_wavefrontPipeline->getProfilingData(0, m_engine->m_timestampPeriod, activeOfflineBounces);
                if (profData.valid && !profData.bounces.empty()) {
                    uint32_t usedBounces = static_cast<uint32_t>(profData.bounces.size());
                    if (profData.bounces.back().nextCount == 0) {
                        activeOfflineBounces = usedBounces;
                    } else {
                        activeOfflineBounces = std::min(m_engine->m_config.max_bounces, usedBounces + 4);
                    }
                }

                remainingSpp -= batchSpp;
                currentSppOffset += batchSpp;
            }

            {
                VkCommandBuffer cmd = m_engine->m_commandBuffers[0];
                vkResetCommandBuffer(cmd, 0);
                VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(cmd, &beginInfo);

                targetAccumImg->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

                VkBufferImageCopy copyRegion{};
                copyRegion.bufferOffset = 0;
                copyRegion.bufferRowLength = width;
                copyRegion.bufferImageHeight = height;
                copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                copyRegion.imageSubresource.layerCount = 1;
                copyRegion.imageExtent = { width, height, 1 };
                vkCmdCopyImageToBuffer(cmd, targetAccumImg->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, stagingAccum.getBuffer(), 1, &copyRegion);

                targetAccumImg->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
                vkEndCommandBuffer(cmd);

                VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
                cmdSubmitInfo.commandBuffer = cmd;
                VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
                submitInfo.commandBufferInfoCount = 1;
                submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
                vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
                vkQueueWaitIdle(queue);
            }

            stagingAccum.invalidate();
            const float invSeedSpp = (seedSpp > 0u) ? (1.0f / static_cast<float>(seedSpp)) : 1.0f;
            const size_t numPixels = static_cast<size_t>(width) * height;
            if (isTargetRg11) {
                const uint32_t* src = static_cast<const uint32_t*>(stagingAccum.map());
                for (size_t p = 0; p < numPixels; ++p) {
                    glm::vec3 rgb = glm::unpackF2x11_1x10(src[p]);
                    outRgb[p * 3 + 0] = rgb.r * invSeedSpp;
                    outRgb[p * 3 + 1] = rgb.g * invSeedSpp;
                    outRgb[p * 3 + 2] = rgb.b * invSeedSpp;
                }
                stagingAccum.unmap();
            } else if (isTargetFp16) {
                const uint16_t* src = static_cast<const uint16_t*>(stagingAccum.map());
                for (size_t p = 0; p < numPixels; ++p) {
                    outRgb[p * 3 + 0] = glm::detail::toFloat32(src[p * 4 + 0]) * invSeedSpp;
                    outRgb[p * 3 + 1] = glm::detail::toFloat32(src[p * 4 + 1]) * invSeedSpp;
                    outRgb[p * 3 + 2] = glm::detail::toFloat32(src[p * 4 + 2]) * invSeedSpp;
                }
                stagingAccum.unmap();
            } else {
                const float* src = static_cast<const float*>(stagingAccum.map());
                for (size_t p = 0; p < numPixels; ++p) {
                    outRgb[p * 3 + 0] = src[p * 4 + 0] * invSeedSpp;
                    outRgb[p * 3 + 1] = src[p * 4 + 1] * invSeedSpp;
                    outRgb[p * 3 + 2] = src[p * 4 + 2] * invSeedSpp;
                }
                stagingAccum.unmap();
            }
        };

        glm::vec3 camPos = m_engine->m_camera ? m_engine->m_camera->getPosition() : glm::vec3(0.0f);
        float fov = m_engine->m_camera ? m_engine->m_camera->getFov() : 45.0f;
        float aspect = m_engine->m_camera ? m_engine->m_camera->getAspect() : (static_cast<float>(width) / height);
        float camPosArr[3] = { camPos.x, camPos.y, camPos.z };
        float zeroJitter[2] = { 0.0f, 0.0f };
        const size_t numPixels = static_cast<size_t>(width) * height;
        std::string refPath = std::format("{}/frame_{:05d}_reference.bin", m_engine->m_config.capture_training_data_dir, frameIdx);

        if (outChannels >= 32u) {
            const uint32_t sppPerSeed = std::max(8u, spp / 2u);
            std::vector<float> refA, refB;
            renderReferenceSeed(200000u, sppPerSeed, refA);
            renderReferenceSeed(700000u, sppPerSeed, refB);

            std::vector<uint16_t> refPayload(numPixels * 8);
            const uint16_t sppSeedFp16 = safeToFp16(static_cast<float>(sppPerSeed));
            for (size_t p = 0; p < numPixels; ++p) {
                refPayload[p * 8 + 0] = safeToFp16(refA[p * 3 + 0]);
                refPayload[p * 8 + 1] = safeToFp16(refA[p * 3 + 1]);
                refPayload[p * 8 + 2] = safeToFp16(refA[p * 3 + 2]);
                refPayload[p * 8 + 3] = sppSeedFp16;
                refPayload[p * 8 + 4] = safeToFp16(refB[p * 3 + 0]);
                refPayload[p * 8 + 5] = safeToFp16(refB[p * 3 + 1]);
                refPayload[p * 8 + 6] = safeToFp16(refB[p * 3 + 2]);
                refPayload[p * 8 + 7] = sppSeedFp16;
            }
            ImageDumper::savePTTD(refPath, width, height, 8, 0 /* Float16 */,
                                  frameIdx, sppPerSeed * 2u, refPayload.data(), refPayload.size() * sizeof(uint16_t),
                                  fov, aspect, camPosArr, zeroJitter);
            Logger::info("  -> Saved reference: {} (8 channels dual-seed, {}x2 SPP)", refPath, sppPerSeed);
        } else {
            std::vector<float> refA;
            renderReferenceSeed(0u, spp, refA);
            std::vector<uint16_t> refPayload(numPixels * 4);
            const uint16_t sppFp16 = safeToFp16(static_cast<float>(spp));
            for (size_t p = 0; p < numPixels; ++p) {
                refPayload[p * 4 + 0] = safeToFp16(refA[p * 3 + 0]);
                refPayload[p * 4 + 1] = safeToFp16(refA[p * 3 + 1]);
                refPayload[p * 4 + 2] = safeToFp16(refA[p * 3 + 2]);
                refPayload[p * 4 + 3] = sppFp16;
            }
            ImageDumper::savePTTD(refPath, width, height, 4, 0 /* Float16 */,
                                  frameIdx, spp, refPayload.data(), refPayload.size() * sizeof(uint16_t),
                                  fov, aspect, camPosArr,
                                  (outChannels >= 23u) ? zeroJitter : nullptr);
            Logger::info("  -> Saved reference: {} (4 channels, {} SPP)", refPath, spp);
        }

    } else {
        // --- NOISY INPUT + ML FEATURE EXTRACTION PASS (PTTD v3 / v4) ---
        // 1. Clear targetAccumImg and ML images to zero
        VkCommandBuffer cmd = m_engine->m_commandBuffers[0];
        vkResetCommandBuffer(cmd, 0);

        VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
        beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        vkBeginCommandBuffer(cmd, &beginInfo);

        clearImage(cmd, targetAccumImg);
        clearImage(cmd, m_engine->m_mlDiffuseImage);
        clearImage(cmd, m_engine->m_mlSpecularImage);
        clearImage(cmd, m_engine->m_mlAlbedoRoughnessImage);
        clearImage(cmd, m_engine->m_mlSpecularMotionImage);
        if (outChannels >= 32u) {
            clearImage(cmd, m_engine->m_mlSkipImage);
            clearImage(cmd, m_engine->m_mlSpecAlbedoImage);
        }
        if (outChannels >= 44u) {
            clearImage(cmd, m_engine->m_mlDirectDiffuseImage);
            clearImage(cmd, m_engine->m_mlIndirectDiffuseImage);
            clearImage(cmd, m_engine->m_mlDirectSpecularImage);
            clearImage(cmd, m_engine->m_mlIndirectSpecularImage);
        }
        clearImage(cmd, m_engine->m_motionVectorImage);
        clearImage(cmd, m_engine->m_normalDepthImage);
        if (frameIdx == 0 && m_engine->m_prevNormalDepthImage) {
            clearImage(cmd, m_engine->m_prevNormalDepthImage);
        }

        // 2. Set camera UBO (advances m_prevViewProj to track true velocity; enables Halton subpixel jitter for SR / V4)
        const uint32_t inputSpp = std::max(1u, spp);
        const bool enableInputJitter = (m_engine->m_config.capture_halton_length > 1u) &&
                                       (outChannels >= 32u || width < m_engine->m_config.width || height < m_engine->m_config.height);
        CameraUniform ubo = m_engine->m_camera->getUniformData(frameIdx, inputSpp, m_engine->m_config.max_bounces, flags,
                                                     enableInputJitter, width, height, 0, /*updatePrev=*/true,
                                                     m_engine->m_config.capture_halton_length);
        m_engine->m_cameraUBOs[0]->copyFrom(&ubo, sizeof(CameraUniform));

        // 3. Dispatch inputSpp with captureMlData
        WavefrontSceneData wfSceneData{};
        wfSceneData.numTriangles = m_engine->m_numTriangles;
        wfSceneData.numMaterials = m_engine->m_numMaterials;
        wfSceneData.numLights = m_engine->m_numLights;
        wfSceneData.hasEnvMap = m_engine->m_environmentMap ? 1u : 0u;
        wfSceneData.envMapIntensity = 1.0f;
        wfSceneData.frameIndex = frameIdx;
        wfSceneData.useMorton = m_engine->m_config.use_morton ? 1u : 0u;
        wfSceneData.accumulateHistory = 0u;
        wfSceneData.sortMode = static_cast<uint32_t>(m_engine->getEffectiveWavefrontSortMode());
        wfSceneData.numOpaqueTriangles = m_engine->m_numOpaqueTriangles;
        wfSceneData.secondarySortMode = static_cast<uint32_t>(m_engine->m_config.secondary_sort_mode);
        wfSceneData.cameraFlags = flags;
        wfSceneData.boundsMin = m_engine->m_sceneData.boundsMin;
        wfSceneData.boundsMax = m_engine->m_sceneData.boundsMax;
        wfSceneData.streamlineSecondaryShading = m_engine->m_config.streamline_secondary_shading;
        wfSceneData.enableDistanceClamping = m_engine->m_config.distance_clamping;
        wfSceneData.maxSecondaryRayDistance = m_engine->m_config.max_secondary_distance;
        wfSceneData.indirectClamp = m_engine->m_config.indirect_clamp;
        wfSceneData.enableTailMegakernel = m_engine->m_config.enable_tail_megakernel;
        wfSceneData.tailMegakernelBounce = m_engine->m_config.tail_megakernel_bounce;
        wfSceneData.inlineShadows = m_engine->m_config.inline_primary_shadows;
        wfSceneData.captureMlData = (outChannels >= 44u) ? (1u | 4u | 8u)
                                  : ((outChannels >= 32u) ? (1u | 4u) : 1u);

        m_engine->m_wavefrontPipeline->recordFrame(cmd, 0, width, height, inputSpp, m_engine->m_config.max_bounces, wfSceneData);

        // 4. Staging copy for 6 (or 8 for V4, or 12 for V5) ML images
        VkDeviceSize numPixels = static_cast<VkDeviceSize>(width) * height;
        VkDeviceSize rgba16Size = numPixels * 4 * sizeof(uint16_t);

        VkDeviceSize offsetDiff = 0;
        VkDeviceSize offsetSpec = offsetDiff + rgba16Size;
        VkDeviceSize offsetAR   = offsetSpec + rgba16Size;
        VkDeviceSize offsetND   = offsetAR   + rgba16Size;
        VkDeviceSize offsetSM   = offsetND   + rgba16Size;
        VkDeviceSize offsetMV   = offsetSM   + rgba16Size;
        VkDeviceSize totalInputStagingSize = offsetMV + rgba16Size;

        VkDeviceSize offsetSkip = 0;
        VkDeviceSize offsetSA   = 0;
        if (outChannels >= 32u) {
            offsetSkip = totalInputStagingSize;
            offsetSA   = offsetSkip + rgba16Size;
            totalInputStagingSize = offsetSA + rgba16Size;
        }

        VkDeviceSize offsetDD = 0;
        VkDeviceSize offsetID = 0;
        VkDeviceSize offsetDS = 0;
        VkDeviceSize offsetIS = 0;
        if (outChannels >= 44u) {
            offsetDD = totalInputStagingSize;
            offsetID = offsetDD + rgba16Size;
            offsetDS = offsetID + rgba16Size;
            offsetIS = offsetDS + rgba16Size;
            totalInputStagingSize = offsetIS + rgba16Size;
        }

        VkDeviceSize offsetRes = 0;
        VkDeviceSize resSize = 0;
        Buffer* resBuffer = nullptr;
        if (outChannels >= 23u && outChannels < 32u && m_engine->m_restirManager && m_engine->isRestirActive()) {
            resBuffer = m_engine->m_restirManager->getReservoirBuffer(0);
            if (resBuffer) {
                offsetRes = totalInputStagingSize;
                resSize = numPixels * sizeof(DiReservoirGPU);
                totalInputStagingSize += resSize;
            }
        }

        Buffer stagingInput(allocator, totalInputStagingSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                            VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        auto copyImgToBuffer = [&](Image* img, VkDeviceSize bufOffset) {
            if (!img) return;
            img->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);

            VkBufferImageCopy region{};
            region.bufferOffset = bufOffset;
            region.bufferRowLength = width;
            region.bufferImageHeight = height;
            region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
            region.imageSubresource.layerCount = 1;
            region.imageExtent = { width, height, 1 };
            vkCmdCopyImageToBuffer(cmd, img->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   stagingInput.getBuffer(), 1, &region);

            img->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
        };

        copyImgToBuffer(m_engine->m_mlDiffuseImage, offsetDiff);
        copyImgToBuffer(m_engine->m_mlSpecularImage, offsetSpec);
        copyImgToBuffer(m_engine->m_mlAlbedoRoughnessImage, offsetAR);
        copyImgToBuffer(m_engine->m_normalDepthImage, offsetND);
        copyImgToBuffer(m_engine->m_mlSpecularMotionImage, offsetSM);
        copyImgToBuffer(m_engine->m_motionVectorImage, offsetMV);
        if (outChannels >= 32u) {
            copyImgToBuffer(m_engine->m_mlSkipImage, offsetSkip);
            copyImgToBuffer(m_engine->m_mlSpecAlbedoImage, offsetSA);
        }
        if (outChannels >= 44u) {
            copyImgToBuffer(m_engine->m_mlDirectDiffuseImage, offsetDD);
            copyImgToBuffer(m_engine->m_mlIndirectDiffuseImage, offsetID);
            copyImgToBuffer(m_engine->m_mlDirectSpecularImage, offsetDS);
            copyImgToBuffer(m_engine->m_mlIndirectSpecularImage, offsetIS);
        }

        if (useCaptureRestir && m_engine->m_normalDepthImage && m_engine->m_prevNormalDepthImage) {
            m_engine->m_normalDepthImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            m_engine->m_prevNormalDepthImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);

            VkImageCopy ndCopy{};
            ndCopy.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            ndCopy.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            ndCopy.extent = { width, height, 1 };
            vkCmdCopyImage(cmd,
                           m_engine->m_normalDepthImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                           m_engine->m_prevNormalDepthImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           1, &ndCopy);

            m_engine->m_normalDepthImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
            m_engine->m_prevNormalDepthImage->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
        }

        if (resBuffer) {
            VkBufferMemoryBarrier2 bufBarrier{ VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2 };
            bufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bufBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            bufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            bufBarrier.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            bufBarrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            bufBarrier.buffer = resBuffer->getBuffer();
            bufBarrier.offset = 0;
            bufBarrier.size = resSize;

            VkDependencyInfo depInfo{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            depInfo.bufferMemoryBarrierCount = 1;
            depInfo.pBufferMemoryBarriers = &bufBarrier;
            vkCmdPipelineBarrier2(cmd, &depInfo);

            VkBufferCopy copyRegion{};
            copyRegion.srcOffset = 0;
            copyRegion.dstOffset = offsetRes;
            copyRegion.size = resSize;
            vkCmdCopyBuffer(cmd, resBuffer->getBuffer(), stagingInput.getBuffer(), 1, &copyRegion);

            bufBarrier.srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT;
            bufBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            bufBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            bufBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            vkCmdPipelineBarrier2(cmd, &depInfo);
        }

        vkEndCommandBuffer(cmd);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = cmd;
        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        stagingInput.invalidate();
        const uint8_t* basePtr = static_cast<const uint8_t*>(stagingInput.map());
        const uint16_t* diffPixels = reinterpret_cast<const uint16_t*>(basePtr + offsetDiff);
        const uint16_t* specPixels = reinterpret_cast<const uint16_t*>(basePtr + offsetSpec);
        const uint16_t* arPixels   = reinterpret_cast<const uint16_t*>(basePtr + offsetAR);
        const uint16_t* ndPixels   = reinterpret_cast<const uint16_t*>(basePtr + offsetND);
        const uint16_t* smPixels   = reinterpret_cast<const uint16_t*>(basePtr + offsetSM);
        const uint16_t* mvPixels   = reinterpret_cast<const uint16_t*>(basePtr + offsetMV);
        const uint16_t* skipPixels = (outChannels >= 32u) ? reinterpret_cast<const uint16_t*>(basePtr + offsetSkip) : nullptr;
        const uint16_t* saPixels   = (outChannels >= 32u) ? reinterpret_cast<const uint16_t*>(basePtr + offsetSA)   : nullptr;
        const uint16_t* ddPixels   = (outChannels >= 44u) ? reinterpret_cast<const uint16_t*>(basePtr + offsetDD)   : nullptr;
        const uint16_t* idPixels   = (outChannels >= 44u) ? reinterpret_cast<const uint16_t*>(basePtr + offsetID)   : nullptr;
        const uint16_t* dsPixels   = (outChannels >= 44u) ? reinterpret_cast<const uint16_t*>(basePtr + offsetDS)   : nullptr;
        const uint16_t* isPixels   = (outChannels >= 44u) ? reinterpret_cast<const uint16_t*>(basePtr + offsetIS)   : nullptr;
        const DiReservoirGPU* resPixels = resBuffer ? reinterpret_cast<const DiReservoirGPU*>(basePtr + offsetRes) : nullptr;

        std::vector<uint16_t> inputPayload(static_cast<size_t>(numPixels) * outChannels);
        const uint16_t zeroHalf = glm::packHalf1x16(0.0f);

        for (size_t p = 0; p < static_cast<size_t>(numPixels); ++p) {
            if (outChannels >= 32u) {
                // Upways 4.0 (PTTD v4, 32 channels) and 4-signal un-demodulated extension (PTTD v5, 44 channels)
                // Group 0 (0..3): Demodulated Diffuse Irradiance RGB + DiffHitDist
                inputPayload[p * outChannels + 0]  = diffPixels[p * 4 + 0];
                inputPayload[p * outChannels + 1]  = diffPixels[p * 4 + 1];
                inputPayload[p * outChannels + 2]  = diffPixels[p * 4 + 2];
                inputPayload[p * outChannels + 3]  = diffPixels[p * 4 + 3];
                // Group 1 (4..7): Demodulated Specular Radiance RGB + ReactiveMask
                inputPayload[p * outChannels + 4]  = specPixels[p * 4 + 0];
                inputPayload[p * outChannels + 5]  = specPixels[p * 4 + 1];
                inputPayload[p * outChannels + 6]  = specPixels[p * 4 + 2];
                inputPayload[p * outChannels + 7]  = specPixels[p * 4 + 3];
                // Group 2 (8..11): Diffuse Demod Albedo RGB + Roughness
                inputPayload[p * outChannels + 8]  = arPixels[p * 4 + 0];
                inputPayload[p * outChannels + 9]  = arPixels[p * 4 + 1];
                inputPayload[p * outChannels + 10] = arPixels[p * 4 + 2];
                inputPayload[p * outChannels + 11] = arPixels[p * 4 + 3];
                // Group 3 (12..15): EnvBRDF Specular Demod Albedo RGB + ReSTIR_M_Norm
                inputPayload[p * outChannels + 12] = saPixels[p * 4 + 0];
                inputPayload[p * outChannels + 13] = saPixels[p * 4 + 1];
                inputPayload[p * outChannels + 14] = saPixels[p * 4 + 2];
                inputPayload[p * outChannels + 15] = saPixels[p * 4 + 3];
                // Group 4 (16..19): Skip (Emissive + Sky) RGB + MatClassNorm
                inputPayload[p * outChannels + 16] = skipPixels[p * 4 + 0];
                inputPayload[p * outChannels + 17] = skipPixels[p * 4 + 1];
                inputPayload[p * outChannels + 18] = skipPixels[p * 4 + 2];
                inputPayload[p * outChannels + 19] = skipPixels[p * 4 + 3];
                // Group 5 (20..23): View-Space Shading Normal XYZ + Linear Depth
                inputPayload[p * outChannels + 20] = ndPixels[p * 4 + 0];
                inputPayload[p * outChannels + 21] = ndPixels[p * 4 + 1];
                inputPayload[p * outChannels + 22] = ndPixels[p * 4 + 2];
                inputPayload[p * outChannels + 23] = ndPixels[p * 4 + 3];
                // Group 6 (24..27): Surface MV XY + Specular MV XY
                inputPayload[p * outChannels + 24] = mvPixels[p * 4 + 0];
                inputPayload[p * outChannels + 25] = mvPixels[p * 4 + 1];
                inputPayload[p * outChannels + 26] = smPixels[p * 4 + 0];
                inputPayload[p * outChannels + 27] = smPixels[p * 4 + 1];
                // Group 7 (28..31): LinDepthDelta + SpecInvDepthSigned + SpecHitDist + Reserved(0)
                inputPayload[p * outChannels + 28] = mvPixels[p * 4 + 2];
                inputPayload[p * outChannels + 29] = smPixels[p * 4 + 2];
                inputPayload[p * outChannels + 30] = smPixels[p * 4 + 3];
                inputPayload[p * outChannels + 31] = zeroHalf;

                if (outChannels >= 44u) {
                    // Canonical 4-signal un-demodulated radiance decomposition (PTTD v5, ch 32..43)
                    inputPayload[p * outChannels + 32] = ddPixels ? ddPixels[p * 4 + 0] : zeroHalf;
                    inputPayload[p * outChannels + 33] = ddPixels ? ddPixels[p * 4 + 1] : zeroHalf;
                    inputPayload[p * outChannels + 34] = ddPixels ? ddPixels[p * 4 + 2] : zeroHalf;
                    inputPayload[p * outChannels + 35] = idPixels ? idPixels[p * 4 + 0] : zeroHalf;
                    inputPayload[p * outChannels + 36] = idPixels ? idPixels[p * 4 + 1] : zeroHalf;
                    inputPayload[p * outChannels + 37] = idPixels ? idPixels[p * 4 + 2] : zeroHalf;
                    inputPayload[p * outChannels + 38] = dsPixels ? dsPixels[p * 4 + 0] : zeroHalf;
                    inputPayload[p * outChannels + 39] = dsPixels ? dsPixels[p * 4 + 1] : zeroHalf;
                    inputPayload[p * outChannels + 40] = dsPixels ? dsPixels[p * 4 + 2] : zeroHalf;
                    inputPayload[p * outChannels + 41] = isPixels ? isPixels[p * 4 + 0] : zeroHalf;
                    inputPayload[p * outChannels + 42] = isPixels ? isPixels[p * 4 + 1] : zeroHalf;
                    inputPayload[p * outChannels + 43] = isPixels ? isPixels[p * 4 + 2] : zeroHalf;
                    for (uint32_t c = 44u; c < outChannels; ++c) {
                        inputPayload[p * outChannels + c] = zeroHalf;
                    }
                }
            } else if (outChannels >= 20) {
                // Vector 0 (Ch 00..03): Diffuse Radiance RGB + Roughness
                inputPayload[p * outChannels + 0]  = diffPixels[p * 4 + 0]; // diffuse R
                inputPayload[p * outChannels + 1]  = diffPixels[p * 4 + 1]; // diffuse G
                inputPayload[p * outChannels + 2]  = diffPixels[p * 4 + 2]; // diffuse B
                inputPayload[p * outChannels + 3]  = arPixels[p * 4 + 3];   // roughness
                // Vector 1 (Ch 04..07): Specular Radiance RGB + Metallic (or 2.0 when primary emissive)
                inputPayload[p * outChannels + 4]  = specPixels[p * 4 + 0]; // specular R
                inputPayload[p * outChannels + 5]  = specPixels[p * 4 + 1]; // specular G
                inputPayload[p * outChannels + 6]  = specPixels[p * 4 + 2]; // specular B
                {
                    float isEmissive = glm::unpackHalf1x16(diffPixels[p * 4 + 3]);
                    inputPayload[p * outChannels + 7] = (isEmissive > 0.5f)
                        ? glm::packHalf1x16(2.0f)
                        : specPixels[p * 4 + 3]; // metallic in [0, 1]
                }
                // Vector 2 (Ch 08..11): Base Color Albedo RGB + Linear Depth
                inputPayload[p * outChannels + 8]  = arPixels[p * 4 + 0];   // albedo R
                inputPayload[p * outChannels + 9]  = arPixels[p * 4 + 1];   // albedo G
                inputPayload[p * outChannels + 10] = arPixels[p * 4 + 2];   // albedo B
                inputPayload[p * outChannels + 11] = ndPixels[p * 4 + 3];   // linear depth
                // Vector 3 (Ch 12..15): Surface Motion XY + Specular Motion XY
                inputPayload[p * outChannels + 12] = mvPixels[p * 4 + 0];   // surface motion X
                inputPayload[p * outChannels + 13] = mvPixels[p * 4 + 1];   // surface motion Y
                inputPayload[p * outChannels + 14] = smPixels[p * 4 + 0];   // specular motion X
                inputPayload[p * outChannels + 15] = smPixels[p * 4 + 1];   // specular motion Y
                // Vector 4 (Ch 16..19): Surface Normal XYZ + Specular Hit Distance
                inputPayload[p * outChannels + 16] = ndPixels[p * 4 + 0];   // normal X
                inputPayload[p * outChannels + 17] = ndPixels[p * 4 + 1];   // normal Y
                inputPayload[p * outChannels + 18] = ndPixels[p * 4 + 2];   // normal Z
                inputPayload[p * outChannels + 19] = smPixels[p * 4 + 3];   // specular hit distance (reflDist)

                if (outChannels >= 23) {
                    if (resPixels) {
                        const auto& res = resPixels[p];
                        uint32_t M = (res.idxM >> 16) & 0xFFFFu;
                        float res_m = std::clamp(static_cast<float>(M) / 64.0f, 0.0f, 1.0f);
                        float pHat = std::max(res.targetPdf, 1e-4f);
                        float res_w = std::clamp(res.wSum / (std::max(static_cast<float>(M), 1.0f) * pHat), 0.0f, 4.0f) * 0.25f;

                        inputPayload[p * outChannels + 20] = glm::packHalf1x16(res_m);
                        inputPayload[p * outChannels + 21] = glm::packHalf1x16(res_w);
                        inputPayload[p * outChannels + 22] = mvPixels[p * 4 + 2]; // vLinearDepthDelta (mvExpectDz)
                    } else {
                        inputPayload[p * outChannels + 20] = glm::packHalf1x16(0.04f);
                        inputPayload[p * outChannels + 21] = glm::packHalf1x16(0.04f);
                        inputPayload[p * outChannels + 22] = mvPixels[p * 4 + 2]; // vLinearDepthDelta (mvExpectDz)
                    }
                    if (outChannels >= 24) {
                        float linDepth = glm::unpackHalf1x16(ndPixels[p * 4 + 3]);
                        float mvW      = glm::unpackHalf1x16(mvPixels[p * 4 + 3]);
                        float isEmis   = glm::unpackHalf1x16(diffPixels[p * 4 + 3]);
                        bool isSurf    = (linDepth > 1e-4f && linDepth < 9999.0f);
                        bool isPsr     = isSurf && (std::abs(mvW) < 1e-4f);
                        float reactive = isPsr ? 0.85f : ((isEmis > 0.5f) ? 0.5f : 0.0f);
                        inputPayload[p * outChannels + 23] = glm::packHalf1x16(reactive);
                    }
                }
            } else {
                inputPayload[p * outChannels + 0]  = diffPixels[p * 4 + 0]; // diffuse R
                inputPayload[p * outChannels + 1]  = diffPixels[p * 4 + 1]; // diffuse G
                inputPayload[p * outChannels + 2]  = diffPixels[p * 4 + 2]; // diffuse B
                inputPayload[p * outChannels + 3]  = specPixels[p * 4 + 0]; // specular R
                inputPayload[p * outChannels + 4]  = specPixels[p * 4 + 1]; // specular G
                inputPayload[p * outChannels + 5]  = specPixels[p * 4 + 2]; // specular B
                inputPayload[p * outChannels + 6]  = arPixels[p * 4 + 0];   // albedo R
                inputPayload[p * outChannels + 7]  = arPixels[p * 4 + 1];   // albedo G
                inputPayload[p * outChannels + 8]  = arPixels[p * 4 + 2];   // albedo B
                inputPayload[p * outChannels + 9]  = arPixels[p * 4 + 3];   // roughness
                inputPayload[p * outChannels + 10] = ndPixels[p * 4 + 3];   // linear depth
                inputPayload[p * outChannels + 11] = smPixels[p * 4 + 2];   // specular hit distance
                inputPayload[p * outChannels + 12] = mvPixels[p * 4 + 0];   // surface motion X
                inputPayload[p * outChannels + 13] = mvPixels[p * 4 + 1];   // surface motion Y
                inputPayload[p * outChannels + 14] = smPixels[p * 4 + 0];   // specular motion X
                inputPayload[p * outChannels + 15] = smPixels[p * 4 + 1];   // specular motion Y
                if (outChannels >= 19) {
                    inputPayload[p * outChannels + 16] = ndPixels[p * 4 + 0]; // normal X
                    inputPayload[p * outChannels + 17] = ndPixels[p * 4 + 1]; // normal Y
                    inputPayload[p * outChannels + 18] = ndPixels[p * 4 + 2]; // normal Z
                }
            }
        }
        stagingInput.unmap();

        for (uint16_t& h : inputPayload) {
            h = sanitizeHalf(h);
        }

        glm::vec3 camPos = m_engine->m_camera ? m_engine->m_camera->getPosition() : glm::vec3(0.0f);
        float fov = m_engine->m_camera ? m_engine->m_camera->getFov() : 45.0f;
        float aspect = m_engine->m_camera ? m_engine->m_camera->getAspect() : (static_cast<float>(width) / height);
        float camPosArr[3] = { camPos.x, camPos.y, camPos.z };
        float jitterArr[2] = { ubo.jitterOffset.x, ubo.jitterOffset.y };

        std::string inpPath = std::format("{}/frame_{:05d}_input.bin", m_engine->m_config.capture_training_data_dir, frameIdx);
        ImageDumper::savePTTD(inpPath, width, height, outChannels, 0 /* Float16 */,
                              frameIdx, inputSpp, inputPayload.data(), inputPayload.size() * sizeof(uint16_t),
                              fov, aspect, camPosArr,
                              (outChannels >= 23 || enableInputJitter) ? jitterArr : nullptr);
        Logger::info("  -> Saved input    : {} ({} channels, {} SPP, jitter=({:.4f}, {:.4f}))",
                     inpPath, outChannels, inputSpp, jitterArr[0], jitterArr[1]);

        if (outChannels >= 32u) {
            // Unbiased dual-seed LR demodulated target (ReSTIR OFF, exact same Halton jitter & resolution)
            const uint32_t lrFlags = flags & ~((1u << 9) | (1u << 24));
            const uint32_t lrSppPerSeed = std::clamp(m_engine->m_config.capture_reference_spp / 4u, 8u, 64u);
            Buffer stagingLr(allocator, rgba16Size * 2, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

            auto renderLrRefSeed = [&](uint32_t seedBase, uint32_t targetSpp,
                                       std::vector<float>& outDiffRgb, std::vector<float>& outSpecRgb) {
                outDiffRgb.assign(static_cast<size_t>(numPixels) * 3, 0.0f);
                outSpecRgb.assign(static_cast<size_t>(numPixels) * 3, 0.0f);
                uint32_t remaining = targetSpp;
                uint32_t offsetSpp = 0;
                constexpr uint32_t BATCH_SIZE = 16;

                WavefrontSceneData lrScene = wfSceneData;
                lrScene.cameraFlags = lrFlags;
                lrScene.accumulateHistory = 0u;
                lrScene.captureMlData = (1u | 4u);

                while (remaining > 0u) {
                    const uint32_t batchSpp = std::min(remaining, BATCH_SIZE);
                    CameraUniform lrUbo = m_engine->m_camera->getUniformData(
                        frameIdx, batchSpp, m_engine->m_config.max_bounces, lrFlags,
                        enableInputJitter, width, height, 0, /*updatePrev=*/false,
                        m_engine->m_config.capture_halton_length);
                    lrUbo.frameIndex = seedBase + frameIdx * 1000u + offsetSpp;
                    m_engine->m_cameraUBOs[0]->copyFrom(&lrUbo, sizeof(CameraUniform));
                    lrScene.frameIndex = lrUbo.frameIndex;

                    VkCommandBuffer lrCmd = m_engine->m_commandBuffers[0];
                    vkResetCommandBuffer(lrCmd, 0);
                    VkCommandBufferBeginInfo lrBegin{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                    lrBegin.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    vkBeginCommandBuffer(lrCmd, &lrBegin);

                    clearImage(lrCmd, targetAccumImg);
                    clearImage(lrCmd, m_engine->m_mlDiffuseImage);
                    clearImage(lrCmd, m_engine->m_mlSpecularImage);
                    clearImage(lrCmd, m_engine->m_mlAlbedoRoughnessImage);
                    clearImage(lrCmd, m_engine->m_mlSpecularMotionImage);
                    clearImage(lrCmd, m_engine->m_mlSkipImage);
                    clearImage(lrCmd, m_engine->m_mlSpecAlbedoImage);

                    m_engine->m_wavefrontPipeline->recordFrame(lrCmd, 0, width, height, batchSpp,
                                                               m_engine->m_config.max_bounces, lrScene);

                    auto copyLrBuf = [&](Image* img, VkDeviceSize bufOffset) {
                        if (!img) return;
                        img->transitionLayout(lrCmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
                        VkBufferImageCopy region{};
                        region.bufferOffset = bufOffset;
                        region.bufferRowLength = width;
                        region.bufferImageHeight = height;
                        region.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
                        region.imageSubresource.layerCount = 1;
                        region.imageExtent = { width, height, 1 };
                        vkCmdCopyImageToBuffer(lrCmd, img->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                               stagingLr.getBuffer(), 1, &region);
                        img->transitionLayout(lrCmd, VK_IMAGE_LAYOUT_GENERAL,
                            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
                            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_READ_BIT);
                    };
                    copyLrBuf(m_engine->m_mlDiffuseImage, 0);
                    copyLrBuf(m_engine->m_mlSpecularImage, rgba16Size);
                    vkEndCommandBuffer(lrCmd);

                    VkCommandBufferSubmitInfo lrSubCmd{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
                    lrSubCmd.commandBuffer = lrCmd;
                    VkSubmitInfo2 lrSub{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
                    lrSub.commandBufferInfoCount = 1;
                    lrSub.pCommandBufferInfos = &lrSubCmd;
                    vkQueueSubmit2(queue, 1, &lrSub, VK_NULL_HANDLE);
                    vkQueueWaitIdle(queue);

                    stagingLr.invalidate();
                    const uint8_t* lrPtr = static_cast<const uint8_t*>(stagingLr.map());
                    const uint16_t* dPtr = reinterpret_cast<const uint16_t*>(lrPtr);
                    const uint16_t* sPtr = reinterpret_cast<const uint16_t*>(lrPtr + rgba16Size);
                    const float wBatch = static_cast<float>(batchSpp);
                    for (size_t p = 0; p < static_cast<size_t>(numPixels); ++p) {
                        outDiffRgb[p * 3 + 0] += wBatch * glm::detail::toFloat32(sanitizeHalf(dPtr[p * 4 + 0]));
                        outDiffRgb[p * 3 + 1] += wBatch * glm::detail::toFloat32(sanitizeHalf(dPtr[p * 4 + 1]));
                        outDiffRgb[p * 3 + 2] += wBatch * glm::detail::toFloat32(sanitizeHalf(dPtr[p * 4 + 2]));
                        outSpecRgb[p * 3 + 0] += wBatch * glm::detail::toFloat32(sanitizeHalf(sPtr[p * 4 + 0]));
                        outSpecRgb[p * 3 + 1] += wBatch * glm::detail::toFloat32(sanitizeHalf(sPtr[p * 4 + 1]));
                        outSpecRgb[p * 3 + 2] += wBatch * glm::detail::toFloat32(sanitizeHalf(sPtr[p * 4 + 2]));
                    }
                    stagingLr.unmap();

                    remaining -= batchSpp;
                    offsetSpp += batchSpp;
                }
                const float invTotal = (targetSpp > 0u) ? (1.0f / static_cast<float>(targetSpp)) : 1.0f;
                for (float& v : outDiffRgb) v *= invTotal;
                for (float& v : outSpecRgb) v *= invTotal;
            };

            std::vector<float> diffA, specA, diffB, specB;
            renderLrRefSeed(100000u, lrSppPerSeed, diffA, specA);
            renderLrRefSeed(500000u, lrSppPerSeed, diffB, specB);

            std::vector<uint16_t> lrPayload(static_cast<size_t>(numPixels) * 12);
            for (size_t p = 0; p < static_cast<size_t>(numPixels); ++p) {
                lrPayload[p * 12 + 0]  = safeToFp16(diffA[p * 3 + 0]);
                lrPayload[p * 12 + 1]  = safeToFp16(diffA[p * 3 + 1]);
                lrPayload[p * 12 + 2]  = safeToFp16(diffA[p * 3 + 2]);
                lrPayload[p * 12 + 3]  = safeToFp16(specA[p * 3 + 0]);
                lrPayload[p * 12 + 4]  = safeToFp16(specA[p * 3 + 1]);
                lrPayload[p * 12 + 5]  = safeToFp16(specA[p * 3 + 2]);
                lrPayload[p * 12 + 6]  = safeToFp16(diffB[p * 3 + 0]);
                lrPayload[p * 12 + 7]  = safeToFp16(diffB[p * 3 + 1]);
                lrPayload[p * 12 + 8]  = safeToFp16(diffB[p * 3 + 2]);
                lrPayload[p * 12 + 9]  = safeToFp16(specB[p * 3 + 0]);
                lrPayload[p * 12 + 10] = safeToFp16(specB[p * 3 + 1]);
                lrPayload[p * 12 + 11] = safeToFp16(specB[p * 3 + 2]);
            }
            std::string lrRefPath = std::format("{}/frame_{:05d}_lr_ref.bin", m_engine->m_config.capture_training_data_dir, frameIdx);
            ImageDumper::savePTTD(lrRefPath, width, height, 12, 0 /* Float16 */,
                                  frameIdx, lrSppPerSeed * 2u, lrPayload.data(), lrPayload.size() * sizeof(uint16_t),
                                  fov, aspect, camPosArr, jitterArr);
            Logger::info("  -> Saved lr_ref   : {} (12 channels dual-seed, {}x2 SPP)", lrRefPath, lrSppPerSeed);
        }
    }
}

void TrainingCaptureManager::updateGamingChoreography(Camera* camera, uint32_t frameIdx, uint32_t totalFrames, const std::string& sceneName) {
    if (!camera) return;

    if (!m_choreoInitialized) {
        m_choreoInitialPos = camera->getPosition();
        m_choreoInitialYaw = camera->getYaw();
        m_choreoInitialPitch = camera->getPitch();
        m_choreoInitialFov = camera->getFov();
        m_choreoInitialized = true;
        Logger::info("[Gaming Choreography] Initialized baseline camera at pos=({:.2f}, {:.2f}, {:.2f}), yaw={:.1f}°, pitch={:.1f}°, fov={:.1f}° for {}",
                     m_choreoInitialPos.x, m_choreoInitialPos.y, m_choreoInitialPos.z,
                     m_choreoInitialYaw, m_choreoInitialPitch, m_choreoInitialFov, sceneName);
    }

    if (totalFrames <= 1) return;

    // Basis vectors in local coordinate system
    float radYaw = glm::radians(m_choreoInitialYaw);
    float radPitch = glm::radians(m_choreoInitialPitch);
    glm::vec3 forward0(
        std::cos(radYaw) * std::cos(radPitch),
        std::sin(radPitch),
        std::sin(radYaw) * std::cos(radPitch)
    );
    forward0 = glm::normalize(forward0);

    glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
    glm::vec3 right0 = glm::normalize(glm::cross(forward0, worldUp));
    glm::vec3 groundForward0 = glm::normalize(glm::cross(worldUp, right0));

    glm::vec3 pos = m_choreoInitialPos;
    float yaw = m_choreoInitialYaw;
    float pitch = m_choreoInitialPitch;
    float fov = m_choreoInitialFov;

    constexpr float PI = 3.14159265358979323846f;

    // Normalize frame progress across 240 frames (or advance at bounded 60Hz rate from active walk/strafe phase for short sequences)
    float normalizedFrame = (totalFrames >= 120u)
        ? ((static_cast<float>(frameIdx) / static_cast<float>(totalFrames)) * 240.0f)
        : (15.0f + static_cast<float>(frameIdx) * 2.5f);

    // Landmark positions for phase continuity
    const float walkDist = 0.22f;
    const float sprintDist = 0.38f;
    const float jumpForwardDist = 0.15f;
    const float crawlDist = 0.08f;

    glm::vec3 pEndPhase1 = m_choreoInitialPos + groundForward0 * walkDist;
    glm::vec3 pEndPhase2 = pEndPhase1 + groundForward0 * sprintDist;
    glm::vec3 pEndPhase3 = pEndPhase2 + groundForward0 * jumpForwardDist;
    glm::vec3 pEndPhase5 = pEndPhase3 + groundForward0 * crawlDist;

    if (normalizedFrame < 40.0f) {
        // --- PHASE 1 (Frames 0..39): Stationary warmup (0..9) -> Walking acceleration (10..39) ---
        if (normalizedFrame >= 10.0f) {
            float u = (normalizedFrame - 10.0f) / 30.0f; // 0..1
            float easeU = u * u; // gentle acceleration
            float d = walkDist * easeU;
            float bob = 0.015f * std::sin(2.0f * PI * (normalizedFrame - 10.0f) / 12.0f) * u;
            float glance = 1.0f * std::sin(2.0f * PI * (normalizedFrame - 10.0f) / 24.0f) * u;

            pos = m_choreoInitialPos + groundForward0 * d + worldUp * bob;
            yaw = m_choreoInitialYaw + glance;
        }
    } else if (normalizedFrame < 90.0f) {
        // --- PHASE 2 (Frames 40..89): Sprint & Lateral Strafe + Sprint FOV (+5 deg) ---
        float u = (normalizedFrame - 40.0f) / 50.0f; // 0..1
        float d = sprintDist * u;
        float strafe = 0.16f * std::sin(2.0f * PI * u * 1.5f);
        float bob = 0.025f * std::sin(2.0f * PI * (normalizedFrame - 40.0f) / 8.0f);
        float dynamicFov = 5.0f * std::sin(PI * u);
        float counterYaw = -1.8f * std::sin(2.0f * PI * u * 1.5f);

        pos = pEndPhase1 + groundForward0 * d + right0 * strafe + worldUp * bob;
        yaw = m_choreoInitialYaw + counterYaw;
        fov = m_choreoInitialFov + dynamicFov;
    } else if (normalizedFrame < 130.0f) {
        // --- PHASE 3 (Frames 90..129): Vertical Jump & Landing Shock ---
        float fInPhase = normalizedFrame - 90.0f;
        float forwardProgress = (fInPhase / 40.0f) * jumpForwardDist;
        glm::vec3 baseP = pEndPhase2 + groundForward0 * forwardProgress;

        if (fInPhase < 24.0f) {
            // Parabolic Jump Ascent & Descent
            float tau = fInPhase / 24.0f; // 0..1
            float jumpHeight = 4.0f * 0.28f * tau * (1.0f - tau);
            float apexFloorLook = -4.0f * std::sin(PI * tau);

            pos = baseP + worldUp * jumpHeight;
            pitch = m_choreoInitialPitch + apexFloorLook;
        } else {
            // Landing Shock (Damped Spring Squish Oscillation)
            float tL = fInPhase - 24.0f;
            float decay = std::exp(-0.25f * tL);
            float squishY = -0.045f * decay * std::cos(2.0f * PI * tL / 5.0f);
            float chinNod = 3.5f * decay * std::sin(2.0f * PI * tL / 4.5f);

            pos = baseP + worldUp * squishY;
            pitch = m_choreoInitialPitch + chinNod;
        }
    } else if (normalizedFrame < 170.0f) {
        // --- PHASE 4 (Frames 130..169): Rapid Twitch Mouse Flick (+55 deg in 4 frames) & Snap Return ---
        pos = pEndPhase3;
        float fInPhase = normalizedFrame - 130.0f;

        float flickYaw = 0.0f;
        if (fInPhase < 4.0f) {
            // Pre-flick stationary pause
            flickYaw = 0.0f;
        } else if (fInPhase < 8.0f) {
            // 4-frame high-velocity flick (+13.75 deg / frame)
            float t = (fInPhase - 4.0f) / 4.0f;
            flickYaw = 55.0f * (t * t * (3.0f - 2.0f * t)); // smooth cubic flick
        } else if (fInPhase < 18.0f) {
            // Snap pause with micro-tremor
            float snapT = fInPhase - 8.0f;
            float microSway = 0.6f * std::sin(2.0f * PI * snapT / 5.0f);
            flickYaw = 55.0f + microSway;
        } else if (fInPhase < 22.0f) {
            // 4-frame rapid return flick back to 0 deg
            float t = (fInPhase - 18.0f) / 4.0f;
            float s = t * t * (3.0f - 2.0f * t);
            flickYaw = 55.0f * (1.0f - s);
        } else {
            // Stabilization
            flickYaw = 0.0f;
        }

        yaw = m_choreoInitialYaw + flickYaw;
    } else if (normalizedFrame < 210.0f) {
        // --- PHASE 5 (Frames 170..209): Aim Down Sights (ADS) Dynamic Zoom (45 -> 24 deg) & Tactical Crawl ---
        float fInPhase = normalizedFrame - 170.0f;
        float targetFov = 24.0f;

        if (fInPhase < 10.0f) {
            // Smooth ADS Zoom In
            float s = fInPhase / 10.0f;
            s = s * s * (3.0f - 2.0f * s);
            fov = m_choreoInitialFov - (m_choreoInitialFov - targetFov) * s;
            pos = pEndPhase3;
        } else if (fInPhase < 28.0f) {
            // Full ADS hold + tactical crawl + breathing sway
            fov = targetFov;
            float crawlT = (fInPhase - 10.0f) / 18.0f;
            float crawlProgress = crawlDist * crawlT;
            float breathYaw = 0.25f * std::sin(2.0f * PI * (fInPhase - 10.0f) / 14.0f);
            float breathPitch = 0.20f * std::cos(2.0f * PI * (fInPhase - 10.0f) / 14.0f);

            pos = pEndPhase3 + groundForward0 * crawlProgress;
            yaw = m_choreoInitialYaw + breathYaw;
            pitch = m_choreoInitialPitch + breathPitch;
        } else if (fInPhase < 38.0f) {
            // Smooth ADS Zoom Out
            float s = (fInPhase - 28.0f) / 10.0f;
            s = s * s * (3.0f - 2.0f * s);
            fov = targetFov + (m_choreoInitialFov - targetFov) * s;
            pos = pEndPhase5;
        } else {
            fov = m_choreoInitialFov;
            pos = pEndPhase5;
        }
    } else {
        // --- PHASE 6 (Frames 210..239): Arc-Strafe / Turntable Orbit (Non-uniform parallax flow) ---
        float fInPhase = normalizedFrame - 210.0f;
        float u = fInPhase / 29.0f; // 0..1
        float orbitR = 1.6f;
        glm::vec3 focalPoint = pEndPhase5 + groundForward0 * orbitR + worldUp * 0.05f;

        float maxAngleDeg = 16.0f;
        float curAngleDeg = maxAngleDeg * std::sin((PI * 0.5f) * u); // smooth ease-out arc
        float curAngleRad = glm::radians(curAngleDeg);

        // Orbit around focal point in horizontal plane
        glm::vec3 rel = -groundForward0 * std::cos(curAngleRad) + right0 * std::sin(curAngleRad);
        pos = focalPoint + rel * orbitR;

        // Camera gaze tracks focal point
        glm::vec3 lookDir = glm::normalize(focalPoint - pos);
        yaw = glm::degrees(std::atan2(lookDir.z, lookDir.x));
        pitch = glm::degrees(std::asin(std::clamp(lookDir.y, -0.999f, 0.999f)));
        fov = m_choreoInitialFov;
    }

    camera->setPose(pos, yaw, pitch);
    camera->setFov(fov);
}

void TrainingCaptureManager::updateCaptureCamera(Camera* camera, uint32_t frameIdx, uint32_t totalFrames, const std::string& sceneName) {
    if (!camera || !m_engine) return;

    if (!m_choreoInitialized) {
        m_choreoInitialPos = camera->getPosition();
        m_choreoInitialYaw = camera->getYaw();
        m_choreoInitialPitch = camera->getPitch();
        m_choreoInitialFov = camera->getFov();
        m_choreoInitialized = true;
        Logger::info("[Capture Camera] Initialized baseline camera at pos=({:.2f}, {:.2f}, {:.2f}), yaw={:.1f}°, pitch={:.1f}°, fov={:.1f}° for {} (Mode: {})",
                     m_choreoInitialPos.x, m_choreoInitialPos.y, m_choreoInitialPos.z,
                     m_choreoInitialYaw, m_choreoInitialPitch, m_choreoInitialFov, sceneName,
                     static_cast<int>(m_engine->m_config.capture_camera_mode));
    }

    if (totalFrames <= 1) return;

    constexpr float PI = 3.14159265358979323846f;
    float t = static_cast<float>(frameIdx) / static_cast<float>(std::max(totalFrames, 120u));

    switch (m_engine->m_config.capture_camera_mode) {
        case CaptureCameraMode::Static: {
            // Completely stationary camera: zero position change, zero rotation change, zero FOV change
            camera->setPose(m_choreoInitialPos, m_choreoInitialYaw, m_choreoInitialPitch);
            camera->setFov(m_choreoInitialFov);
            break;
        }
        case CaptureCameraMode::Rotate: {
            // Pure rotation / angular view shift: fixed position, bounded smooth pan yaw + tilt pitch
            float phase = static_cast<float>(frameIdx) / static_cast<float>(std::max(totalFrames, 16u));
            float sweepYaw = 14.0f * std::sin(2.0f * PI * phase);
            float nodPitch = 4.0f * std::sin(4.0f * PI * phase);
            camera->setPose(m_choreoInitialPos, m_choreoInitialYaw + sweepYaw, m_choreoInitialPitch + nodPitch);
            camera->setFov(m_choreoInitialFov);
            break;
        }
        case CaptureCameraMode::Translate: {
            // Pure spatial translation: fixed orientation, forward dolly + lateral strafe
            float radYaw = glm::radians(m_choreoInitialYaw);
            float radPitch = glm::radians(m_choreoInitialPitch);
            glm::vec3 forward0(
                std::cos(radYaw) * std::cos(radPitch),
                std::sin(radPitch),
                std::sin(radYaw) * std::cos(radPitch)
            );
            forward0 = glm::normalize(forward0);
            glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
            glm::vec3 right0 = glm::normalize(glm::cross(forward0, worldUp));
            glm::vec3 groundForward0 = glm::normalize(glm::cross(worldUp, right0));

            float forwardDist = 1.8f * t;
            float strafeDist = 0.45f * std::sin(2.0f * PI * t);
            glm::vec3 newPos = m_choreoInitialPos + groundForward0 * forwardDist + right0 * strafeDist;
            camera->setPose(newPos, m_choreoInitialYaw, m_choreoInitialPitch);
            camera->setFov(m_choreoInitialFov);
            break;
        }
        case CaptureCameraMode::Orbit: {
            // Smooth bounded turntable arc around forward focal point (keeps enclosed rooms in frame)
            float radYaw = glm::radians(m_choreoInitialYaw);
            float radPitch = glm::radians(m_choreoInitialPitch);
            glm::vec3 forward0(
                std::cos(radYaw) * std::cos(radPitch),
                std::sin(radPitch),
                std::sin(radYaw) * std::cos(radPitch)
            );
            forward0 = glm::normalize(forward0);
            glm::vec3 worldUp(0.0f, 1.0f, 0.0f);
            glm::vec3 right0 = glm::normalize(glm::cross(forward0, worldUp));
            glm::vec3 groundForward0 = glm::normalize(glm::cross(worldUp, right0));

            glm::vec3 boundsCenter = m_engine->m_sceneData.boundsMin + (m_engine->m_sceneData.boundsMax - m_engine->m_sceneData.boundsMin) * 0.5f;
            float distToCenter = glm::dot(boundsCenter - m_choreoInitialPos, groundForward0);
            float orbitR = std::clamp(distToCenter, 1.5f, 3.5f);
            glm::vec3 focalPoint = m_choreoInitialPos + groundForward0 * orbitR;

            float phase = static_cast<float>(frameIdx) / static_cast<float>(std::max(totalFrames, 16u));
            float angleRad = glm::radians(16.0f * std::sin(2.0f * PI * phase));
            glm::vec3 rel = -groundForward0 * std::cos(angleRad) + right0 * std::sin(angleRad);
            glm::vec3 orbitPos = focalPoint + rel * orbitR + worldUp * (0.06f * std::sin(2.0f * PI * phase));

            glm::vec3 lookDir = glm::normalize((m_choreoInitialPos + forward0 * orbitR) - orbitPos);
            float yaw = glm::degrees(std::atan2(lookDir.z, lookDir.x));
            float pitch = glm::degrees(std::asin(std::clamp(lookDir.y, -0.999f, 0.999f)));
            camera->setPose(orbitPos, yaw, pitch);
            camera->setFov(m_choreoInitialFov);
            break;
        }
        case CaptureCameraMode::Gaming:
        default: {
            updateGamingChoreography(camera, frameIdx, totalFrames, sceneName);
            break;
        }
    }
}

void TrainingCaptureManager::run() {
    if (!m_engine) return;

    Logger::info("========================================================================================");
    Logger::info("  Pathways -> Upways Training Data Capture Pipeline (DGC Wavefront / Wave32)");
    Logger::info("  Target Directory : {}", m_engine->m_config.capture_training_data_dir);
    Logger::info("  Frames to Capture: {}", m_engine->m_config.capture_frames);
    Logger::info("  Reference SPP    : {}", m_engine->m_config.capture_reference_spp);
    Logger::info("  Capture Channels : {} (PTTD v{})", m_engine->m_config.capture_channels,
                 m_engine->m_config.capture_channels >= 44 ? 5 : (m_engine->m_config.capture_channels >= 32 ? 4 : (m_engine->m_config.capture_channels >= 23 ? 3 : (m_engine->m_config.capture_channels == 20 ? 2 : 1))));
    Logger::info("  Capture Normals  : {}", m_engine->m_config.capture_normals ? "YES" : "NO");
    Logger::info("  Camera Mode      : {}",
                 m_engine->m_config.capture_camera_mode == CaptureCameraMode::Static ? "Static (Stationary)" :
                 (m_engine->m_config.capture_camera_mode == CaptureCameraMode::Rotate ? "Rotate (Pure Angular)" :
                 (m_engine->m_config.capture_camera_mode == CaptureCameraMode::Translate ? "Translate (Pure Spatial)" :
                 (m_engine->m_config.capture_camera_mode == CaptureCameraMode::Orbit ? "Orbit (Spherical)" : "Gaming (6-DOF)"))));
    Logger::info("  Animated Objects : {}", m_engine->m_config.animate_objects ? "YES" : "NO");
    Logger::info("  Halton Phasing   : {} phases", m_engine->m_config.capture_halton_length);
    Logger::info("========================================================================================");

    std::filesystem::create_directories(m_engine->m_config.capture_training_data_dir);

    // Warm up camera & reset choreography state
    m_choreoInitialized = false;
    if (m_engine->m_camera) {
        m_engine->m_camera->resetMoved();
    }

    for (uint32_t f = 0; f < m_engine->m_config.capture_frames; ++f) {
        Logger::info("--- Capturing Training Frame [{}/{}] ---", f + 1, m_engine->m_config.capture_frames);

        if (m_engine->m_sceneData.isMineshaftCoaster) {
            bool advanceCoaster = m_engine->m_config.animate_objects &&
                                  (m_engine->m_config.capture_camera_mode != CaptureCameraMode::Static &&
                                   m_engine->m_config.capture_camera_mode != CaptureCameraMode::Rotate);
            float animSpeed = (m_engine->m_config.animation_speed > 0.0001f) ? m_engine->m_config.animation_speed : 1.0f;
            float animDt = (1.0f / 60.0f) * animSpeed;
            float evalTime = advanceCoaster ? (static_cast<float>(f) * animDt) : 0.0f;

            glm::vec3 cCamPos, cCamTarget, cCamUp;
            float cCamFov = 58.0f;
            glm::mat4 riderMat(1.0f), lead1Mat(1.0f), lead2Mat(1.0f);
            ProceduralScene::updateMineshaftCoaster(
                m_engine->m_sceneData, evalTime,
                cCamPos, cCamTarget, cCamUp, cCamFov,
                riderMat, lead1Mat, lead2Mat
            );
            if (m_engine->m_sceneData.coasterRiderCartInst != UINT32_MAX) {
                m_engine->updateInstanceTransform(m_engine->m_sceneData.coasterRiderCartInst, riderMat);
            }
            if (m_engine->m_sceneData.coasterLeadCart1Inst != UINT32_MAX) {
                m_engine->updateInstanceTransform(m_engine->m_sceneData.coasterLeadCart1Inst, lead1Mat);
            }
            if (m_engine->m_sceneData.coasterLeadCart2Inst != UINT32_MAX) {
                m_engine->updateInstanceTransform(m_engine->m_sceneData.coasterLeadCart2Inst, lead2Mat);
            }
            if (m_engine->m_sceneManager) {
                m_engine->m_sceneManager->updateDynamicLights(m_engine->m_sceneData);
                m_engine->m_numLights = m_engine->m_sceneManager->getNumLights();
            }
            if (m_engine->m_tlasUpdatePipeline && m_engine->m_tlas) {
                VkCommandBuffer cmd = m_engine->m_commandBuffers[0];
                vkResetCommandBuffer(cmd, 0);
                VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                vkBeginCommandBuffer(cmd, &beginInfo);
                m_engine->recordGpuTlasUpdate(cmd, false);
                vkEndCommandBuffer(cmd);

                VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
                cmdSubmitInfo.commandBuffer = cmd;
                VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
                submitInfo.commandBufferInfoCount = 1;
                submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
                vkQueueSubmit2(m_engine->m_context->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
                vkQueueWaitIdle(m_engine->m_context->getGraphicsQueue());
            }
            if (m_engine->m_camera) {
                m_engine->m_camera->setAnimatedPose(cCamPos, cCamTarget, cCamUp, cCamFov);
                constexpr float PI = 3.14159265358979323846f;
                uint32_t totalFrames = m_engine->m_config.capture_frames;
                if (m_engine->m_config.capture_camera_mode == CaptureCameraMode::Static || f == 0u) {
                    if (m_engine->m_config.capture_camera_mode == CaptureCameraMode::Static) {
                        m_engine->m_camera->resetMoved();
                    }
                }
                if (m_engine->m_config.capture_camera_mode == CaptureCameraMode::Rotate && totalFrames > 1u) {
                    float degPerFrame = 360.0f / std::max(totalFrames, 120u);
                    float sweepYaw = degPerFrame * static_cast<float>(f);
                    float nodPitch = 5.0f * std::sin(2.0f * PI * (static_cast<float>(f) / std::max(totalFrames, 60u)));
                    m_engine->m_camera->setPose(cCamPos, m_engine->m_camera->getYaw() + sweepYaw, m_engine->m_camera->getPitch() + nodPitch);
                } else if (m_engine->m_config.capture_camera_mode == CaptureCameraMode::Gaming && totalFrames > 1u) {
                    float glanceYaw = 6.0f * std::sin(2.0f * PI * (static_cast<float>(f) / 16.0f));
                    float glancePitch = 2.5f * std::cos(2.0f * PI * (static_cast<float>(f) / 12.0f));
                    m_engine->m_camera->setPose(cCamPos, m_engine->m_camera->getYaw() + glanceYaw, m_engine->m_camera->getPitch() + glancePitch);
                }
            }
        } else {
            // Update camera for frame f according to capture_camera_mode
            if (m_engine->m_camera) {
                updateCaptureCamera(m_engine->m_camera.get(), f, m_engine->m_config.capture_frames, m_engine->m_config.scene_path);
            }

            // Step dynamic object animations for frame f if enabled
            if (m_engine->m_config.animate_objects && !m_engine->m_sceneData.animatedInstances.empty()) {
                float animDt = (1.0f / 60.0f) * m_engine->m_config.animation_speed;
                m_engine->updateAnimatedInstances(animDt);
                // Record GPU TLAS update/refit
                if (m_engine->m_tlasUpdatePipeline && m_engine->m_tlas) {
                    VkCommandBuffer cmd = m_engine->m_commandBuffers[0];
                    vkResetCommandBuffer(cmd, 0);
                    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
                    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    vkBeginCommandBuffer(cmd, &beginInfo);
                    m_engine->recordGpuTlasUpdate(cmd, true);
                    vkEndCommandBuffer(cmd);

                    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
                    cmdSubmitInfo.commandBuffer = cmd;
                    VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
                    submitInfo.commandBufferInfoCount = 1;
                    submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
                    vkQueueSubmit2(m_engine->m_context->getGraphicsQueue(), 1, &submitInfo, VK_NULL_HANDLE);
                    vkQueueWaitIdle(m_engine->m_context->getGraphicsQueue());
                }
            }
        }

        // 1. Render Noisy Input Frame FIRST (evaluates true inter-frame motion vectors from f-1 to f)
        captureTrainingFrame(f, /*isReference=*/false, std::max(1u, m_engine->m_config.capture_input_spp));

        // 2. Render Ground Truth Reference Frame SECOND (same instantaneous animation and camera pose)
        captureTrainingFrame(f, /*isReference=*/true, m_engine->m_config.capture_reference_spp);
    }

    Logger::info("========================================================================================");
    Logger::info("  Training Data Capture COMPLETE! {} frames written to {}",
                 m_engine->m_config.capture_frames, m_engine->m_config.capture_training_data_dir);
    Logger::info("========================================================================================");
}

} // namespace pathways
