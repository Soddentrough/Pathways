#include "mgpu/MultiGpuCoordinator.hpp"
#include "core/QualityGovernor.hpp"
#include "rt/CausticsPipeline.hpp"

#include <algorithm>

namespace pathways {

MultiGpuCoordinator::~MultiGpuCoordinator() {
    if (m_device != VK_NULL_HANDLE) {
        destroy(m_device);
    }
}

void MultiGpuCoordinator::init(VkDevice device, uint32_t asyncComputeQueueFamily, bool hasDedicatedAsyncCompute, float timestampPeriod) {
    m_device = device;
    m_hasDedicatedAsyncCompute = hasDedicatedAsyncCompute;
    m_timestampPeriod = timestampPeriod;
    if (hasDedicatedAsyncCompute) {
        VkCommandPoolCreateInfo asyncPoolInfo{};
        asyncPoolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
        asyncPoolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        asyncPoolInfo.queueFamilyIndex = asyncComputeQueueFamily;
        vkCreateCommandPool(device, &asyncPoolInfo, nullptr, &m_asyncComputeCommandPool);

        VkCommandBufferAllocateInfo asyncAllocInfo{};
        asyncAllocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
        asyncAllocInfo.commandPool = m_asyncComputeCommandPool;
        asyncAllocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        asyncAllocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
        vkAllocateCommandBuffers(device, &asyncAllocInfo, m_mergeCommandBuffers.data());

        VkQueryPoolCreateInfo queryInfo{ VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO };
        queryInfo.queryType = VK_QUERY_TYPE_TIMESTAMP;
        queryInfo.queryCount = MAX_FRAMES_IN_FLIGHT * 2;
        vkCreateQueryPool(device, &queryInfo, nullptr, &m_queryPool);
    }

    VkSemaphoreCreateInfo semInfo{};
    semInfo.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        vkCreateSemaphore(device, &semInfo, nullptr, &m_mergeCompleteSemaphores[i]);
    }
}

void MultiGpuCoordinator::destroy(VkDevice device) {
    m_hasDedicatedAsyncCompute = false;
    for (uint32_t i = 0; i < MAX_FRAMES_IN_FLIGHT; ++i) {
        if (m_mergeCompleteSemaphores[i] != VK_NULL_HANDLE) {
            vkDestroySemaphore(device, m_mergeCompleteSemaphores[i], nullptr);
            m_mergeCompleteSemaphores[i] = VK_NULL_HANDLE;
        }
    }
    if (m_queryPool != VK_NULL_HANDLE) {
        vkDestroyQueryPool(device, m_queryPool, nullptr);
        m_queryPool = VK_NULL_HANDLE;
    }
    m_slotHasExecuted.fill(false);
    if (m_asyncComputeCommandPool != VK_NULL_HANDLE) {
        vkDestroyCommandPool(device, m_asyncComputeCommandPool, nullptr);
        m_asyncComputeCommandPool = VK_NULL_HANDLE;
    }
    m_mergeCommandBuffers.fill(VK_NULL_HANDLE);
}

MultiGpuFramePlan MultiGpuCoordinator::planAndLaunchSecondary(
    const MultiGpuPlanParams& params,
    uint32_t& inOutAccumulatedSamples,
    bool& inOutAccumReset
) {
    MultiGpuFramePlan plan{};
    plan.formatMode = (params.config.accum_format == AccumFormat::RGBA16_SFLOAT) ? 0u :
                      (params.config.accum_format == AccumFormat::RGBA32_SFLOAT) ? 1u : 2u;
    uint32_t bytesPerPixel = (plan.formatMode == 0u) ? 8 : (plan.formatMode == 2u) ? 4 : 16;
    plan.frameBytes = 0;
    plan.dstHost = nullptr;

    plan.activeMode = params.config.mgpu_mode;
    if (plan.activeMode == MultiGpuMode::Auto) {
        plan.activeMode = (params.activeSpp > 1) ? MultiGpuMode::SampleParallel : MultiGpuMode::CheckerboardTile;
    }
    if (plan.activeMode != m_lastActiveMgpuMode) {
        inOutAccumReset = true;
        inOutAccumulatedSamples = 0;
        m_lastActiveMgpuMode = plan.activeMode;
    }

    plan.tileOffsetX_sec = 2u;
    plan.tileOffsetY_sec = 0u;
    plan.tileOffsetX_prim = 1u;
    plan.tileOffsetY_prim = 0u;
    plan.mgpuBaseW = params.renderW;
    plan.mgpuBaseH = params.renderH;
    plan.dispatchWidth = plan.mgpuBaseW;
    plan.dispatchHeight = (plan.mgpuBaseH + 1) / 2;
    plan.isSampleBlendFsr3 = (params.config.upscaler_mode == UpscalerMode::FSR &&
                              params.config.mgpu_upscale_mode == MgpuUpscaleMode::SampleBlend &&
                              params.config.denoiser_mode != DenoiserMode::Svgf &&
                              params.config.denoiser_mode != DenoiserMode::FfxRayRegeneration);
    plan.totalCompositeSpp = 0u;
    plan.secAccumHistory = 0u;
    plan.mergeMode = 0u; // 0 = InterleavedScanline, 1 = CheckerboardTile, 2 = SampleParallel
    plan.primSpp = params.activeSpp;
    plan.secSpp = 0u;

    plan.uboSec = params.ubo;

    plan.secDispatchWidth = plan.dispatchWidth;
    plan.primDispatchWidth = plan.dispatchWidth;

    if (plan.activeMode == MultiGpuMode::CheckerboardTile) {
        plan.mergeMode = 1u;
        plan.tileOffsetX_sec = 2u;
        plan.tileOffsetY_sec = params.config.tile_size;
        plan.tileOffsetX_prim = 1u;
        plan.tileOffsetY_prim = params.config.tile_size;
        uint32_t tileSize = (params.config.tile_size == 0u) ? 64u : params.config.tile_size;
        uint32_t numTilesX = (plan.mgpuBaseW + tileSize - 1u) / tileSize;
        uint32_t maxTilesPerGpuX = (numTilesX + 1u) / 2u;
        plan.secDispatchWidth = maxTilesPerGpuX * tileSize;
        plan.primDispatchWidth = plan.secDispatchWidth;
        plan.dispatchWidth = plan.primDispatchWidth;
        plan.dispatchHeight = plan.mgpuBaseH;
        plan.secAccumHistory = 0u;
    } else if (plan.activeMode == MultiGpuMode::SampleParallel) {
        plan.mergeMode = 2u;
        plan.tileOffsetX_sec = 0u;
        plan.tileOffsetY_sec = 0u;
        plan.tileOffsetX_prim = 0u;
        plan.tileOffsetY_prim = 0u;
        plan.secDispatchWidth = plan.mgpuBaseW;
        plan.primDispatchWidth = plan.mgpuBaseW;
        plan.dispatchWidth = plan.mgpuBaseW;
        plan.dispatchHeight = plan.mgpuBaseH;
        plan.secAccumHistory = plan.isSampleBlendFsr3
            ? ((params.config.progressive_accumulation && !inOutAccumReset) ? 1u : 0u)
            : 0u;

        uint32_t currentTotalSpp = params.activeSpp;
        if (plan.isSampleBlendFsr3) {
            plan.primSpp = std::max(1u, currentTotalSpp / 2u);
            plan.secSpp = std::max(1u, currentTotalSpp / 2u);
        } else {
            plan.primSpp = std::max(1u, (currentTotalSpp + 1) / 2);
            plan.secSpp = std::max(1u, currentTotalSpp / 2);
        }
        if (params.governor && params.config.adaptive_spp && params.governor->getState().active) {
            plan.primSpp = params.governor->getState().primSpp;
            plan.secSpp = params.governor->getState().secSpp;
        }
        CameraUniform uboModified = params.ubo;
        uboModified.spp = plan.primSpp;
        plan.uboSec.spp = plan.secSpp;
        plan.uboSec.frameIndex = params.frameIndex + 1000003u;

        bool isStationaryAccum = params.config.progressive_accumulation && !params.cameraMovedLastFrame && (inOutAccumulatedSamples > 1);
        bool isDenoisingRenderRes = (params.config.denoiser_mode == DenoiserMode::Svgf ||
                                     params.config.denoiser_mode == DenoiserMode::FfxRayRegeneration);
        bool enableJitter = !isStationaryAccum && !isDenoisingRenderRes &&
                            (params.config.upscaler_mode == UpscalerMode::FSR ||
                             params.config.upscaler_mode == UpscalerMode::Upways ||
                             params.config.upscaler_mode == UpscalerMode::Gateway ||
                             params.config.denoiser_mode == DenoiserMode::Upways ||
                             params.config.denoiser_mode == DenoiserMode::Gateway);
        if (enableJitter && params.camera) {
            uint32_t phaseOffset = (params.config.mgpu_mode == MultiGpuMode::SampleParallel) ? 4 : 0;
            uint32_t jitterSeqLen = (params.config.upscaler_mode == UpscalerMode::FSR)
                ? getFsrJitterPhaseCount(plan.mgpuBaseW, params.config.width)
                : 32u;
            plan.uboSec = params.camera->getUniformData(params.frameIndex, plan.secSpp, params.activeBounces, params.flags,
                                                       true, plan.mgpuBaseW, plan.mgpuBaseH, phaseOffset, false, jitterSeqLen);
            plan.uboSec.frameIndex = params.frameIndex + 1000003u;
        }

        plan.totalCompositeSpp = (plan.activeMode == MultiGpuMode::SampleParallel && !plan.isSampleBlendFsr3) ? (plan.primSpp + plan.secSpp) : 0u;
        plan.uboPrim = uboModified;
        if (plan.activeMode == MultiGpuMode::SampleParallel && params.config.pipeline_type == PipelineType::Wavefront && plan.totalCompositeSpp > 0u) {
            plan.uboPrim.spp = plan.totalCompositeSpp;
        } else {
            plan.uboPrim.spp = plan.primSpp;
        }
        if (params.cameraUbo) {
            params.cameraUbo->copyFrom(&plan.uboPrim, sizeof(CameraUniform));
        }
    }

    if (params.mgpu) {
        params.mgpu->setConfig(params.config);
    }

    if (plan.isSampleBlendFsr3) {
        plan.frameBytes = static_cast<size_t>(params.config.width) * params.config.height * 8;
    } else {
        plan.frameBytes = static_cast<size_t>(plan.secDispatchWidth) * plan.dispatchHeight * bytesPerPixel;
        bool needNormals = (params.config.upscaler_mode == UpscalerMode::Upways ||
                            params.config.upscaler_mode == UpscalerMode::Gateway ||
                            params.config.denoiser_mode == DenoiserMode::Upways ||
                            params.config.denoiser_mode == DenoiserMode::Gateway ||
                            params.config.denoiser_mode == DenoiserMode::Svgf ||
                            params.config.denoiser_mode == DenoiserMode::FfxRayRegeneration ||
                            params.config.enable_restir ||
                            params.config.enable_caustics);
        bool needMvDepth = needNormals || (params.config.upscaler_mode == UpscalerMode::FSR);
        bool copyMvImage = needNormals || (needMvDepth && params.config.accum_format == AccumFormat::R11G11B10_UFLOAT);
        if (plan.tileOffsetX_sec == 2u && copyMvImage) {
            plan.frameBytes += static_cast<size_t>(plan.secDispatchWidth) * plan.dispatchHeight * (needNormals ? 16 : 8);
        }
    }
    m_lastSecToPrimBytes = plan.frameBytes;

    if (params.mgpu && !params.mgpu->isZeroCopyActive() && !params.mgpu->isP2PDirectBarActive()) {
        plan.dstHost = params.secTransferBuffer ? params.secTransferBuffer->map() : nullptr;
    }

    plan.slot = params.config.double_buffered_shared_mem ? (params.currentFrame % 2) : 0;

    // Launch secondary GPU concurrently for current frame
    if (!params.skipRayTracing && params.mgpu) {
        glm::vec2 jitterSec(0.0f);
        bool isDenoisingRenderRes = (params.config.denoiser_mode == DenoiserMode::Svgf ||
                                     params.config.denoiser_mode == DenoiserMode::FfxRayRegeneration);
        if (!isDenoisingRenderRes &&
            (params.config.upscaler_mode == UpscalerMode::FSR ||
             params.config.upscaler_mode == UpscalerMode::Upways ||
             params.config.upscaler_mode == UpscalerMode::Gateway)) {
            if (!params.config.progressive_accumulation || params.cameraMovedLastFrame || inOutAccumulatedSamples <= 1) {
                uint32_t jitterSeqLen = (params.config.upscaler_mode == UpscalerMode::FSR)
                    ? getFsrJitterPhaseCount(plan.mgpuBaseW, params.config.width)
                    : 32u;
                jitterSec = getHaltonJitter(params.frameIndex + 4, jitterSeqLen);
            }
        }
        bool isSecMotion = params.cameraMovedThisFrame || params.cameraMovedLastFrame;
        params.mgpu->launchSecondaryWork(plan.uboSec, plan.slot, plan.tileOffsetX_sec, plan.tileOffsetY_sec, plan.mgpuBaseW, plan.mgpuBaseH,
                                         params.numTriangles, params.numSpheres, params.numMaterials, params.numLights, params.useHwRT,
                                         params.hasEnvMap, params.envIntensity, plan.secAccumHistory, params.activeFractionalSpp, plan.dstHost, plan.frameBytes,
                                         plan.totalCompositeSpp, params.numOpaqueTriangles,
                                         plan.isSampleBlendFsr3, params.renderW, params.renderH, params.config.width, params.config.height,
                                         jitterSec, (params.hardReset || params.temporalResetRequested), isSecMotion, params.frameIndex,
                                         params.config.upscaler_sharpening, params.config.upscaler_sharpening ? params.config.upscaler_sharpness : 0.0f,
                                         (params.config.progressive_accumulation && inOutAccumulatedSamples > 0) ? inOutAccumulatedSamples : 1u,
                                         static_cast<uint32_t>(params.effectiveWavefrontSortMode));
    }

    return plan;
}

void MultiGpuCoordinator::recordPrimaryRayTracing(
    VkCommandBuffer cmd,
    VkQueue queue,
    VkDevice device,
    const MultiGpuFramePlan& plan,
    const MultiGpuPlanParams& params,
    uint32_t qBase,
    VkQueryPool queryPool,
    VkSemaphore rtCompleteSemaphore,
    RayTracingOrchestrator* rtOrchestrator,
    Image* frameImage,
    Image* accumImage,
    Image* mlDiffuseImage,
    Image* mlSpecularImage,
    Image* normalDepthImage,
    Image* prevNormalDepthImage,
    VkDescriptorSet rtDescSet,
    CausticsPipeline* causticsPipeline,
    const SceneData& sceneData,
    const std::function<void(VkCommandBuffer)>& onPreRecord
) {
    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo rtBeginInfo{};
    rtBeginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    vkBeginCommandBuffer(cmd, &rtBeginInfo);

    if (queryPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, queryPool, qBase, 6);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queryPool, qBase + 0);
    }

    if (onPreRecord) {
        onPreRecord(cmd);
    }

    if (!params.skipRayTracing && rtOrchestrator) {
        uint32_t primDispatchSpp = (plan.activeMode == MultiGpuMode::SampleParallel) ? plan.primSpp : params.activeSpp;
        RayTracingDispatchParams rtParams{};
        rtParams.frameSlot = params.currentFrame;
        rtParams.dispatchWidth = plan.dispatchWidth;
        rtParams.dispatchHeight = plan.dispatchHeight;
        rtParams.fullWidth = plan.mgpuBaseW;
        rtParams.fullHeight = plan.mgpuBaseH;
        rtParams.tileOffsetX = plan.tileOffsetX_prim;
        rtParams.tileOffsetY = plan.tileOffsetY_prim;
        rtParams.spp = primDispatchSpp;
        rtParams.bounces = params.activeBounces;
        rtParams.fractionalSpp = params.activeFractionalSpp;
        rtParams.totalCompositeSpp = plan.totalCompositeSpp;
        rtParams.cameraFlags = params.flags;
        rtParams.accumReset = params.accumReset;
        rtParams.isMultiGpu = true;
        rtParams.mgpuMode = plan.activeMode;
        rtParams.skipRayTracing = false;
        rtParams.frameIndex = params.frameIndex;
        rtParams.cameraMovedLastFrame = params.cameraMovedThisFrame || params.cameraMovedLastFrame;
        rtParams.diagnosticHalfTiles = false;
        rtParams.isRestirActive = params.config.enable_restir;

        rtOrchestrator->recordRayTracing(
            cmd, rtParams,
            frameImage,
            accumImage,
            mlDiffuseImage,
            mlSpecularImage,
            normalDepthImage,
            prevNormalDepthImage,
            nullptr,
            rtDescSet,
            causticsPipeline,
            params.governor,
            params.camera,
            sceneData,
            params.numTriangles, params.numSpheres, params.numMaterials, params.numLights, params.numOpaqueTriangles,
            params.hasEnvMap, params.envIntensity, params.useHwRT,
            params.config
        );
    }

    if (queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, queryPool, qBase + 1);
    }
    vkEndCommandBuffer(cmd);

    VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdSubmitInfo.commandBuffer = cmd;

    VkSemaphoreSubmitInfo signalInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    signalInfo.semaphore = rtCompleteSemaphore;
    signalInfo.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;

    VkSubmitInfo2 rtSubmit{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    rtSubmit.commandBufferInfoCount = 1;
    rtSubmit.pCommandBufferInfos = &cmdSubmitInfo;
    rtSubmit.signalSemaphoreInfoCount = 1;
    rtSubmit.pSignalSemaphoreInfos = &signalInfo;
    if (!params.skipRayTracing && params.mgpu && params.mgpu->isP2PDirectBarActive()) {
        // Ensure the secondary GPU (DMA-BUF importer) submits before the primary GPU (DMA-BUF exporter).
        // In Linux amdgpu, imported DMA-BUF BOs use implicit AMDGPU_SYNC_NE_OWNER synchronization while
        // exported native BOs use AMDGPU_SYNC_EXPLICIT; submitting the importer first avoids a false
        // dma_resv dependency on the primary GPU's ray tracing submission.
        params.mgpu->waitWorkerIdle();
    }
    vkQueueSubmit2(queue, 1, &rtSubmit, VK_NULL_HANDLE);
}

void MultiGpuCoordinator::recordMergePass(
    VkCommandBuffer activeCmd,
    const MultiGpuFramePlan& plan,
    const MultiGpuPlanParams& params,
    PostProcessPipeline* postProcess,
    Image* frameImage,
    Image* mlDiffuseImage
) {
    // Barrier: Ensure primary RT writes and secondary DMA host writes are visible before merge compute reads/writes
    VkMemoryBarrier2 rtToMergeBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    rtToMergeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_HOST_BIT;
    rtToMergeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT | VK_ACCESS_2_HOST_WRITE_BIT;
    rtToMergeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    rtToMergeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;

    VkDependencyInfo rtToMergeDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    rtToMergeDep.memoryBarrierCount = 1;
    rtToMergeDep.pMemoryBarriers = &rtToMergeBarrier;
    vkCmdPipelineBarrier2(activeCmd, &rtToMergeDep);

    // Merge Pass (Skipped for SampleBlend + FSR3 since both GPUs upscale to 4K independently before resolve)
    if (!params.skipRayTracing && !plan.isSampleBlendFsr3 && postProcess) {
        bool needNormals = (params.config.upscaler_mode == UpscalerMode::Upways ||
                            params.config.upscaler_mode == UpscalerMode::Gateway ||
                            params.config.denoiser_mode == DenoiserMode::Upways ||
                            params.config.denoiser_mode == DenoiserMode::Gateway ||
                            params.config.denoiser_mode == DenoiserMode::Svgf ||
                            params.config.denoiser_mode == DenoiserMode::FfxRayRegeneration ||
                            params.config.enable_restir ||
                            params.config.enable_caustics);
        bool needMvDepth = needNormals || (params.config.upscaler_mode == UpscalerMode::FSR);
        uint32_t mergeGbuffersMode = needNormals ? 1u : (needMvDepth ? ((plan.formatMode != 2u) ? 3u : 2u) : 0u);
        glm::mat4 viewToWorldScaled = params.ubo.viewInverse;
        viewToWorldScaled[0] *= params.ubo.projInverse[0][0];
        viewToWorldScaled[1] *= params.ubo.projInverse[1][1];
        viewToWorldScaled[2] *= -1.0f;
        glm::mat4 reprojMat = params.ubo.prevViewProj * viewToWorldScaled;
        PostProcessPipeline::MergePushConstants mergePC{
            plan.mgpuBaseW, plan.mgpuBaseH, plan.secSpp, params.config.tile_size, plan.formatMode, plan.mergeMode, plan.primSpp,
            plan.secDispatchWidth, mergeGbuffersMode,
            params.cameraMovedThisFrame ? 1u : 0u,
            params.ubo.jitterOffset.z, params.ubo.jitterOffset.w,
            reprojMat
        };

        uint32_t mergeGroupsX = (plan.mergeMode == 1u && plan.secDispatchWidth > 0u)
            ? ((plan.secDispatchWidth + 15) / 16)
            : ((plan.mgpuBaseW + 15) / 16);
        uint32_t mergeGroupsY = (plan.mergeMode == 0u) ? (((plan.mgpuBaseH + 1) / 2 + 15) / 16) : ((plan.mgpuBaseH + 15) / 16);
        postProcess->recordMerge(activeCmd, plan.slot, mergePC, mergeGroupsX, mergeGroupsY);
    }

    VkMemoryBarrier2 mergeBarrier{ VK_STRUCTURE_TYPE_MEMORY_BARRIER_2 };
    mergeBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mergeBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
    mergeBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    mergeBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

    VkDependencyInfo mergeDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
    mergeDep.memoryBarrierCount = 1;
    mergeDep.pMemoryBarriers = &mergeBarrier;
    vkCmdPipelineBarrier2(activeCmd, &mergeDep);

    // Copy merged FP16 radiance into mlDiffuseImage for Upways/Gateway neural reconstruction / ReLAX SVGF / FFX denoiser
    if (!params.skipRayTracing && mlDiffuseImage && frameImage && 
        (params.config.upscaler_mode == UpscalerMode::Upways ||
         params.config.upscaler_mode == UpscalerMode::Gateway ||
         params.config.denoiser_mode == DenoiserMode::Upways ||
         params.config.denoiser_mode == DenoiserMode::Gateway ||
         params.config.denoiser_mode == DenoiserMode::Svgf ||
         params.config.denoiser_mode == DenoiserMode::FfxRayRegeneration)) {
        VkImageCopy copyRegion{};
        copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copyRegion.extent = { plan.mgpuBaseW, plan.mgpuBaseH, 1 };
        vkCmdCopyImage(activeCmd, frameImage->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                       mlDiffuseImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, 1, &copyRegion);

        VkImageMemoryBarrier2 diffCopyBarrier{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
        diffCopyBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        diffCopyBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
        diffCopyBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        diffCopyBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
        diffCopyBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        diffCopyBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        diffCopyBarrier.image = mlDiffuseImage->getImage();
        diffCopyBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo diffCopyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        diffCopyDep.imageMemoryBarrierCount = 1;
        diffCopyDep.pImageMemoryBarriers = &diffCopyBarrier;
        vkCmdPipelineBarrier2(activeCmd, &diffCopyDep);
    }

    const bool needDecoupledPost = (params.config.denoiser_mode == DenoiserMode::Svgf ||
                                    params.config.denoiser_mode == DenoiserMode::FfxRayRegeneration ||
                                    params.config.denoiser_mode == DenoiserMode::Upways ||
                                    params.config.denoiser_mode == DenoiserMode::Gateway ||
                                    params.config.upscaler_mode == UpscalerMode::Upways ||
                                    params.config.upscaler_mode == UpscalerMode::Gateway ||
                                    params.config.upscaler_mode == UpscalerMode::FSR);
    const bool isDenoisedWithoutRunningAvg =
        (params.config.denoiser_mode == DenoiserMode::Upways || params.config.upscaler_mode == UpscalerMode::Upways) ||
        (params.config.denoiser_mode == DenoiserMode::Gateway || params.config.upscaler_mode == UpscalerMode::Gateway) ||
        ((params.config.denoiser_mode == DenoiserMode::Svgf || params.config.denoiser_mode == DenoiserMode::FfxRayRegeneration) &&
         params.config.capture_training_data_dir.empty());
    const bool useFusedMgpuTonemap = !m_hasDedicatedAsyncCompute && !needDecoupledPost && !params.skipRayTracing && !plan.isSampleBlendFsr3 &&
                                     !params.config.visualize_mgpu_split && (plan.slot == params.currentFrame) &&
                                     postProcess && postProcess->hasFusedPipeline();

    // Running Average Accumulation Pass for Multi-GPU (FP16 Merged Frame -> FP32 Persistent History)
    if (!params.skipRayTracing && !useFusedMgpuTonemap && !isDenoisedWithoutRunningAvg &&
        postProcess && postProcess->hasRunningAvgPipeline()) {
        PostProcessPipeline::RunningAvgPushConstants avgPC{};
        avgPC.width = plan.mgpuBaseW;
        avgPC.height = plan.mgpuBaseH;
        avgPC.sampleCount = (params.config.progressive_accumulation && !params.accumReset) ? params.accumulatedSamples : 1u;
        avgPC.invSpp = 1.0f;

        postProcess->recordRunningAvg(activeCmd, params.currentFrame, avgPC);
    }
}

void MultiGpuCoordinator::syncSecondaryTransfer(MultiGpuManager* mgpu, const MultiGpuFramePlan& plan, bool skipRayTracing) {
    if (mgpu && !skipRayTracing) {
        mgpu->syncAndTransfer(plan.slot, plan.dstHost, plan.frameBytes);
    }
}

bool MultiGpuCoordinator::executeAsyncMerge(
    VkQueue asyncComputeQueue,
    const MultiGpuFramePlan& plan,
    const MultiGpuPlanParams& params,
    PostProcessPipeline* postProcess,
    Image* frameImage,
    Image* mlDiffuseImage,
    VkSemaphore rtCompleteSemaphore
) {
    if (!m_hasDedicatedAsyncCompute || params.skipRayTracing || plan.isSampleBlendFsr3 || !postProcess) {
        return false;
    }

    uint32_t slot = params.currentFrame % MAX_FRAMES_IN_FLIGHT;
    VkCommandBuffer cmd = m_mergeCommandBuffers[slot];
    if (cmd == VK_NULL_HANDLE) return false;

    // Read timestamp query results from previous execution of this slot
    if (m_slotHasExecuted[slot] && m_queryPool != VK_NULL_HANDLE) {
        uint64_t ts[2] = {0, 0};
        if (vkGetQueryPoolResults(m_device, m_queryPool, slot * 2, 2, sizeof(ts), ts, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            if (ts[1] > ts[0]) {
                m_lastAsyncMergeTimeMs = (ts[1] - ts[0]) * m_timestampPeriod * 1e-6;
            }
        }
    }

    vkResetCommandBuffer(cmd, 0);
    VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
    beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cmd, &beginInfo);

    if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdResetQueryPool(cmd, m_queryPool, slot * 2, 2);
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_queryPool, slot * 2 + 0);
    }

    recordMergePass(cmd, plan, params, postProcess, frameImage, mlDiffuseImage);

    if (m_queryPool != VK_NULL_HANDLE) {
        vkCmdWriteTimestamp2(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, m_queryPool, slot * 2 + 1);
    }

    vkEndCommandBuffer(cmd);
    m_slotHasExecuted[slot] = true;

    std::vector<VkSemaphoreSubmitInfo> waitInfos;
    VkSemaphoreSubmitInfo waitRt{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    waitRt.semaphore = rtCompleteSemaphore;
    waitRt.stageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    waitInfos.push_back(waitRt);

    if (params.mgpu && params.mgpu->isCrossGpuSyncActive()) {
        VkSemaphore secSem = params.mgpu->getPrimaryImportedTimelineSemaphore();
        if (secSem != VK_NULL_HANDLE) {
            VkSemaphoreSubmitInfo waitSec{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            waitSec.semaphore = secSem;
            waitSec.value = params.mgpu->getCurrentTimelineValue();
            waitSec.stageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            waitInfos.push_back(waitSec);
        }
    }

    VkSemaphoreSubmitInfo signalInfo{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    signalInfo.semaphore = m_mergeCompleteSemaphores[slot];
    signalInfo.stageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;

    VkCommandBufferSubmitInfo cmdInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
    cmdInfo.commandBuffer = cmd;

    VkSubmitInfo2 submit{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
    submit.commandBufferInfoCount = 1;
    submit.pCommandBufferInfos = &cmdInfo;
    submit.waitSemaphoreInfoCount = static_cast<uint32_t>(waitInfos.size());
    submit.pWaitSemaphoreInfos = waitInfos.data();
    submit.signalSemaphoreInfoCount = 1;
    submit.pSignalSemaphoreInfos = &signalInfo;

    vkQueueSubmit2(asyncComputeQueue, 1, &submit, VK_NULL_HANDLE);
    return true;
}

void MultiGpuCoordinator::populatePostWaitSemaphores(
    std::vector<VkSemaphoreSubmitInfo>& waitSemaphoreInfos,
    MultiGpuManager* mgpu,
    uint32_t currentFrame,
    VkSemaphore rtCompleteSemaphore,
    bool skipRayTracing,
    bool asyncMergeExecuted
) {
    if (asyncMergeExecuted) {
        uint32_t slot = currentFrame % MAX_FRAMES_IN_FLIGHT;
        VkSemaphoreSubmitInfo waitMerge{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
        waitMerge.semaphore = m_mergeCompleteSemaphores[slot];
        waitMerge.stageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        waitSemaphoreInfos.push_back(waitMerge);
        return;
    }

    VkSemaphoreSubmitInfo waitRt{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
    waitRt.semaphore = rtCompleteSemaphore;
    waitRt.stageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
    waitSemaphoreInfos.push_back(waitRt);

    if (mgpu && mgpu->isCrossGpuSyncActive() && !skipRayTracing) {
        VkSemaphore secSem = mgpu->getPrimaryImportedTimelineSemaphore();
        if (secSem != VK_NULL_HANDLE) {
            VkSemaphoreSubmitInfo waitSec{ VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO };
            waitSec.semaphore = secSem;
            waitSec.value = mgpu->getCurrentTimelineValue();
            waitSec.stageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT | VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            waitSemaphoreInfos.push_back(waitSec);
        }
    }
}

} // namespace pathways
