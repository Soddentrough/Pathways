#include "rt/RayTracingOrchestrator.hpp"
#include "core/Logger.hpp"
#include "scene/Material.hpp"
#include "scene/Camera.hpp"

#include <bit>
#include <cmath>
#include <algorithm>

namespace pathways {

RayTracingOrchestrator::RayTracingOrchestrator(
    VkDevice device,
    VmaAllocator allocator,
    VkPhysicalDevice physicalDevice,
    VkPhysicalDeviceType deviceType
)
    : m_device(device),
      m_allocator(allocator),
      m_physicalDevice(physicalDevice),
      m_deviceType(deviceType) {
}

RayTracingOrchestrator::~RayTracingOrchestrator() {
    m_wavefrontPipeline.reset();
    m_rtpKhrPipeline.reset();
    m_restirManager.reset();
    m_nrcManager.reset();

    if (m_rtpPipelineLayout != VK_NULL_HANDLE && m_device != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_rtpPipelineLayout, nullptr);
        m_rtpPipelineLayout = VK_NULL_HANDLE;
    }
}

void RayTracingOrchestrator::initRTPipeline(
    VkPhysicalDeviceRayTracingPipelinePropertiesKHR rtProps,
    VkDescriptorSetLayout rtDescLayout,
    const std::vector<char>& rgenCode,
    const std::vector<char>& rmissCode,
    const std::vector<char>& shadowMissCode,
    const std::vector<char>& rchitCode,
    bool hasRtSubgroupSizeControl
) {
    VkShaderStageFlags rtStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;

    VkPushConstantRange rtpPushConstant{};
    rtpPushConstant.stageFlags = rtStages;
    rtpPushConstant.offset = 0;
    rtpPushConstant.size = sizeof(uint32_t) * 16;

    VkPipelineLayoutCreateInfo rtpPipeLayoutInfo{};
    rtpPipeLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    rtpPipeLayoutInfo.setLayoutCount = 1;
    rtpPipeLayoutInfo.pSetLayouts = &rtDescLayout;
    rtpPipeLayoutInfo.pushConstantRangeCount = 1;
    rtpPipeLayoutInfo.pPushConstantRanges = &rtpPushConstant;

    if (m_rtpPipelineLayout != VK_NULL_HANDLE) {
        vkDestroyPipelineLayout(m_device, m_rtpPipelineLayout, nullptr);
        m_rtpPipelineLayout = VK_NULL_HANDLE;
    }
    vkCreatePipelineLayout(m_device, &rtpPipeLayoutInfo, nullptr, &m_rtpPipelineLayout);

    m_rtpKhrPipeline = std::make_unique<RTPipeline>(
        m_device, m_allocator,
        rtProps,
        m_rtpPipelineLayout,
        rgenCode, rmissCode, shadowMissCode, rchitCode,
        hasRtSubgroupSizeControl
    );
    Logger::info("Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline) created successfully.");
}

void RayTracingOrchestrator::initWavefrontPipeline(
    uint32_t width,
    uint32_t height,
    const std::vector<char>& classifyCode,
    const std::vector<char>& intersectCode,
    const std::vector<char>& shadeCode,
    const std::vector<char>& shadowCode,
    const std::vector<char>& shadeDiffuseCode,
    const std::vector<char>& shadeDielectricCode,
    const std::vector<char>& shadeConductorCode,
    const std::vector<char>& shadeComplexCode,
    const std::vector<char>& shadeEmissiveCode,
    const std::vector<char>& shadePassthroughCode,
    bool hasDgcExecutionSet,
    const std::vector<char>& shadeDiffuseSecCode,
    const std::vector<char>& shadeComplexSecCode,
    bool enableDgcPreprocess,
    bool hasSubgroupSizeControl,
    uint32_t initialBatchPixels,
    const std::vector<char>& tailMegakernelCode
) {
    m_wavefrontPipeline = std::make_unique<WavefrontPipeline>(
        m_device, m_allocator,
        width, height,
        classifyCode, intersectCode, shadeCode, shadowCode,
        shadeDiffuseCode, shadeDielectricCode, shadeConductorCode, shadeComplexCode,
        shadeEmissiveCode, shadePassthroughCode,
        hasDgcExecutionSet,
        shadeDiffuseSecCode, shadeComplexSecCode,
        enableDgcPreprocess,
        hasSubgroupSizeControl,
        initialBatchPixels,
        tailMegakernelCode
    );
    Logger::info("Wavefront Path Tracing Pipeline (Ray Queues & DGC) initialized successfully.");
}

void RayTracingOrchestrator::initNRC(
    uint32_t width,
    uint32_t height,
    const std::vector<char>& inferCode,
    const std::vector<char>& trainCode,
    const std::vector<char>& resolveCode
) {
    m_nrcManager = std::make_unique<NRCManager>(
        m_device, m_allocator,
        width, height,
        inferCode, trainCode, resolveCode
    );
    Logger::info("Neural Radiance Caching Subsystem (Wave32 WMMA & Atomic Buffer) initialized successfully.");
}

void RayTracingOrchestrator::createReSTIRResources(
    uint32_t width,
    uint32_t height,
    const std::vector<char>& temporalCode,
    const std::vector<char>& spatialCode
) {
    m_restirManager = std::make_unique<ReSTIRManager>(
        m_device, m_allocator,
        width, height,
        temporalCode, spatialCode
    );
    Logger::info("Ultra-Lean ReSTIR PT Subsystem (32B Reservoirs, Fused Temporal & LDS Spatial Reuse) initialized successfully.");
}

void RayTracingOrchestrator::destroyReSTIRResources() {
    m_restirManager.reset();
}

uint32_t RayTracingOrchestrator::getTargetBatchPixels(const Config& config) const {
    if (config.batch_pixels > 0) {
        return config.batch_pixels;
    }
    if (config.batch_count > 0) {
        uint32_t totalPixels = config.width * config.height;
        return (totalPixels + config.batch_count - 1) / config.batch_count;
    }

    if (m_deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        return 207360u;
    }

    VkDeviceSize totalDeviceVram = 0;
    VkPhysicalDeviceMemoryProperties memProps;
    vkGetPhysicalDeviceMemoryProperties(m_physicalDevice, &memProps);
    for (uint32_t i = 0; i < memProps.memoryHeapCount; ++i) {
        if (memProps.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            totalDeviceVram += memProps.memoryHeaps[i].size;
        }
    }

    if (totalDeviceVram <= 8ULL * 1024 * 1024 * 1024) {
        return 1500000u;
    }

    if (totalDeviceVram < 16ULL * 1024 * 1024 * 1024) {
        return 2000000u;
    }

    return 8294400u;
}

uint32_t RayTracingOrchestrator::getEffectiveBatchCount(uint32_t renderW, uint32_t renderH, const Config& config) const {
    if (config.batch_count > 0) {
        return config.batch_count;
    }
    uint32_t totalPixels = renderW * renderH;
    uint32_t targetBatch = getTargetBatchPixels(config);
    if (targetBatch == 0 || totalPixels <= targetBatch) {
        return 1u;
    }
    uint32_t count = (totalPixels + targetBatch - 1) / targetBatch;
    uint32_t maxAutoBatches = (m_deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) ? 8u : 4u;
    if (config.max_bounces > 2 && count > maxAutoBatches) {
        count = maxAutoBatches;
    }
    return count;
}

uint32_t RayTracingOrchestrator::getEffectiveBatchPixels(uint32_t renderW, uint32_t renderH, uint32_t batchCount) const {
    if (batchCount <= 1) return renderW * renderH;
    float screenAspect = static_cast<float>(renderW) / static_cast<float>(renderH);
    float bestMetric = 1e9f;
    uint32_t bestNx = 1, bestNy = 1;
    for (uint32_t nx = 1; nx <= batchCount; ++nx) {
        if (batchCount % nx == 0) {
            uint32_t ny = batchCount / nx;
            float tileAspect = screenAspect * (static_cast<float>(ny) / static_cast<float>(nx));
            float metric = std::abs(std::log(tileAspect));
            if (metric < bestMetric) {
                bestMetric = metric;
                bestNx = nx;
                bestNy = ny;
            }
        }
    }
    uint32_t maxW = (renderW + bestNx - 1) / bestNx;
    uint32_t maxH = (renderH + bestNy - 1) / bestNy;
    return maxW * maxH;
}

float RayTracingOrchestrator::getDivergentAreaRatio(const SceneData& sceneData) const {
    if (m_cachedDivergentAreaRatio >= 0.0f) {
        return m_cachedDivergentAreaRatio;
    }
    if (sceneData.triangles.empty() || sceneData.materials.empty()) {
        m_cachedDivergentAreaRatio = 0.0f;
        return 0.0f;
    }
    double totalArea = 0.0;
    double divergentArea = 0.0;
    const size_t numMats = sceneData.materials.size();
    std::vector<uint8_t> isDivergent(numMats, 0);
    for (size_t i = 0; i < numMats; ++i) {
        uint32_t arch = computeMaterialArchetype(sceneData.materials[i]);
        if (arch == MATERIAL_ARCHETYPE_COMPLEX || arch == MATERIAL_ARCHETYPE_DIELECTRIC) {
            isDivergent[i] = 1;
        }
    }

    for (const auto& tri : sceneData.triangles) {
        glm::vec3 e1 = glm::vec3(tri.v1.position - tri.v0.position);
        glm::vec3 e2 = glm::vec3(tri.v2.position - tri.v0.position);
        double area = 0.5 * glm::length(glm::cross(e1, e2));
        totalArea += area;
        if (tri.materialId < numMats && isDivergent[tri.materialId]) {
            divergentArea += area;
        }
    }
    m_cachedDivergentAreaRatio = (totalArea > 1e-6) ? static_cast<float>(divergentArea / totalArea) : 0.0f;
    return m_cachedDivergentAreaRatio;
}

WavefrontSortMode RayTracingOrchestrator::getEffectiveWavefrontSortMode(const Config& config, const SceneData& sceneData) const {
    if (config.wavefront_sort_mode != WavefrontSortMode::Auto) {
        return config.wavefront_sort_mode;
    }

    if (sceneData.materials.empty()) {
        return WavefrontSortMode::None;
    }

    float divergentRatio = getDivergentAreaRatio(sceneData);
    if (divergentRatio < 0.001f) {
        return WavefrontSortMode::None;
    }

    if (m_deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
        if (divergentRatio < 0.12f) {
            return WavefrontSortMode::None;
        }
    }

    return WavefrontSortMode::Dual;
}

void RayTracingOrchestrator::resize(uint32_t width, uint32_t height, const Config& config) {
    if (m_nrcManager) {
        m_nrcManager->resize(width, height);
    }
    if (m_restirManager) {
        m_restirManager->resize(width, height);
    }
    if (m_wavefrontPipeline) {
        uint32_t batchCount = getEffectiveBatchCount(width, height, config);
        uint32_t batchPixels = getEffectiveBatchPixels(width, height, batchCount);
        m_currentBatchCount = batchCount;
        m_currentBatchPixels = batchPixels;
        m_wavefrontPipeline->resize(width, height, batchPixels);
    }
}

void RayTracingOrchestrator::recordRayTracing(
    VkCommandBuffer cmd,
    const RayTracingDispatchParams& params,
    Image* frameImage,
    Image* accumImage,
    Image* mlDiffuseImage,
    Image* mlSpecularImage,
    Image* normalDepthImage,
    Image* prevNormalDepthImage,
    Buffer* centerDepthBuffer,
    VkDescriptorSet rtDescSet,
    CausticsPipeline* causticsPipeline,
    QualityGovernor* governor,
    Camera* camera,
    const SceneData& sceneData,
    uint32_t numTriangles,
    uint32_t numSpheres,
    uint32_t numMaterials,
    uint32_t numLights,
    uint32_t numOpaqueTriangles,
    uint32_t hasEnvMap,
    float envIntensity,
    uint32_t useHwRT,
    const Config& config
) {
    if (params.skipRayTracing) return;

    bool useWavefront = (config.pipeline_type == PipelineType::Wavefront && m_wavefrontPipeline);
    uint32_t activeDispatchSpp = params.spp;

    if (useWavefront) {
        if (config.enable_nrc && m_nrcManager) {
            m_nrcManager->resetCounters(cmd);
        }

        // Clear per-frame ray tracing target (FP16)
        VkClearColorValue clearZero = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };
        if (frameImage) {
            vkCmdClearColorImage(cmd, frameImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
        }

        bool needAccumReset = params.accumReset || !config.progressive_accumulation;
        bool isUpwaysSuperResActive = (config.upscaler_mode == UpscalerMode::Upways || config.upways_superres);
        if (needAccumReset || isUpwaysSuperResActive) {
            if (needAccumReset && accumImage) {
                vkCmdClearColorImage(cmd, accumImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
            }
            if (mlDiffuseImage) {
                vkCmdClearColorImage(cmd, mlDiffuseImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
            }
            if (mlSpecularImage) {
                vkCmdClearColorImage(cmd, mlSpecularImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
            }
        }

        std::vector<VkImageMemoryBarrier2> clearBarriers;
        auto addClearBarrier = [&](Image* img) {
            if (!img) return;
            VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            b.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
            b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.image = img->getImage();
            b.subresourceRange = clearRange;
            clearBarriers.push_back(b);
        };
        addClearBarrier(frameImage);
        if (needAccumReset) {
            addClearBarrier(accumImage);
            addClearBarrier(mlDiffuseImage);
            addClearBarrier(mlSpecularImage);
        }

        if (!clearBarriers.empty()) {
            VkDependencyInfo clearDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            clearDep.imageMemoryBarrierCount = static_cast<uint32_t>(clearBarriers.size());
            clearDep.pImageMemoryBarriers = clearBarriers.data();
            vkCmdPipelineBarrier2(cmd, &clearDep);
        }

        WavefrontSceneData wfSceneData{};
        wfSceneData.numTriangles = numTriangles;
        wfSceneData.numSpheres = numSpheres;
        wfSceneData.numMaterials = numMaterials;
        wfSceneData.numLights = numLights;
        wfSceneData.hasEnvMap = hasEnvMap;
        wfSceneData.envMapIntensity = envIntensity;
        wfSceneData.useHardwareRT = useHwRT;
        wfSceneData.frameIndex = params.frameIndex;
        wfSceneData.useMorton = config.use_morton ? 1u : 0u;
        wfSceneData.accumulateHistory = (config.progressive_accumulation && !params.accumReset) ? 1u : 0u;
        wfSceneData.sortMode = static_cast<uint32_t>(getEffectiveWavefrontSortMode(config, sceneData));
        wfSceneData.numOpaqueTriangles = numOpaqueTriangles;
        wfSceneData.secondarySortMode = static_cast<uint32_t>(config.secondary_sort_mode);
        wfSceneData.cameraFlags = params.cameraFlags;

        bool enableNrcThisFrame = config.enable_nrc;
        if (params.isMultiGpu && params.mgpuMode == MultiGpuMode::CheckerboardTile) {
            enableNrcThisFrame = false; // Secondary GPU has no NRC inference network; disable to maintain tile parity
        }
        wfSceneData.enableNrc = enableNrcThisFrame;
        wfSceneData.nrcBounce = config.nrc_bounce;
        wfSceneData.nrcTrainRatio = config.nrc_train_ratio;
        wfSceneData.boundsMin = sceneData.boundsMin;
        wfSceneData.boundsMax = sceneData.boundsMax;
        wfSceneData.streamlineSecondaryShading = config.streamline_secondary_shading;
        wfSceneData.enableDistanceClamping = config.distance_clamping;
        wfSceneData.maxSecondaryRayDistance = config.max_secondary_distance;
        wfSceneData.indirectClamp = config.indirect_clamp;
        wfSceneData.enableTailMegakernel = config.enable_tail_megakernel;
        wfSceneData.tailMegakernelBounce = config.tail_megakernel_bounce;
        wfSceneData.deltaUnroll = config.delta_unroll;
        wfSceneData.inlineShadows = config.inline_primary_shadows;

        if (params.isMultiGpu) {
            wfSceneData.tileOffsetX = params.tileOffsetX;
            wfSceneData.tileOffsetY = params.tileOffsetY;
            wfSceneData.fullWidth = params.fullWidth;
            wfSceneData.fullHeight = params.fullHeight;
            uint32_t mgpuRequiredCapacity = params.dispatchWidth * params.dispatchHeight;
            if (m_wavefrontPipeline->getMaxCapacity() < mgpuRequiredCapacity) {
                m_currentBatchPixels = mgpuRequiredCapacity;
                m_currentBatchCount = 1;
                m_wavefrontPipeline->resize(params.dispatchWidth, params.dispatchHeight, mgpuRequiredCapacity);
            }
        } else {
            uint32_t activeBatchCount = getEffectiveBatchCount(params.dispatchWidth, params.dispatchHeight, config);
            uint32_t activeBatchPixels = getEffectiveBatchPixels(params.dispatchWidth, params.dispatchHeight, activeBatchCount);
            if (activeBatchCount != m_currentBatchCount || activeBatchPixels != m_currentBatchPixels) {
                m_currentBatchCount = activeBatchCount;
                m_currentBatchPixels = activeBatchPixels;
                m_wavefrontPipeline->resize(params.dispatchWidth, params.dispatchHeight, activeBatchPixels);
            }
            wfSceneData.macroTileSize = config.macro_tile_size;
            wfSceneData.batchCount = activeBatchCount;
            wfSceneData.batchPixels = activeBatchPixels;
            wfSceneData.fullWidth = params.dispatchWidth;
        }

        bool needGbuffers = (config.upscaler_mode == UpscalerMode::FSR3 ||
                             config.upscaler_mode == UpscalerMode::Upways ||
                             config.denoiser_mode == DenoiserMode::Upways ||
                             params.isRestirActive ||
                             config.enable_caustics);
        bool captureMl = (config.denoiser_mode == DenoiserMode::Upways ||
                          config.upscaler_mode == UpscalerMode::Upways ||
                          config.upways_superres ||
                          !config.capture_training_data_dir.empty());
        wfSceneData.captureMlData = (captureMl ? 1u : 0u) | (needGbuffers ? 2u : 0u);

        if (causticsPipeline && config.enable_caustics && sceneData.hasDielectrics && numLights > 0) {
            causticsPipeline->recordTrace(
                cmd, params.frameSlot,
                numTriangles, numSpheres, numMaterials, numLights,
                numOpaqueTriangles, params.frameIndex,
                sceneData.hasDielectrics,
                sceneData.dielectricBoundsMin,
                sceneData.dielectricBoundsMax
            );
        }

        if (params.diagnosticHalfTiles && !params.isMultiGpu) {
            uint32_t tileSize = (config.tile_size == 0u) ? 64u : config.tile_size;
            uint32_t numTilesX = (params.dispatchWidth + tileSize - 1u) / tileSize;
            uint32_t maxTilesPerGpuX = (numTilesX + 1u) / 2u;
            uint32_t halfDispatchWidth = maxTilesPerGpuX * tileSize;
            wfSceneData.tileOffsetX = 1u;
            wfSceneData.tileOffsetY = tileSize;
            wfSceneData.fullWidth = params.dispatchWidth;
            wfSceneData.fullHeight = params.dispatchHeight;
            m_wavefrontPipeline->recordFrame(cmd, params.frameSlot, halfDispatchWidth, params.dispatchHeight,
                                             activeDispatchSpp, params.bounces, wfSceneData);
        } else {
            m_wavefrontPipeline->recordFrame(cmd, params.frameSlot, params.dispatchWidth, params.dispatchHeight,
                                             activeDispatchSpp, params.bounces, wfSceneData);
        }

        if (enableNrcThisFrame && m_nrcManager) {
            uint32_t nrcW = params.isMultiGpu ? params.fullWidth : params.dispatchWidth;
            uint32_t nrcH = params.isMultiGpu ? params.fullHeight : params.dispatchHeight;
            m_nrcManager->recordInference(cmd, nrcW, nrcH, sceneData.boundsMin, sceneData.boundsMax);
            m_nrcManager->recordTraining(cmd, params.frameIndex, sceneData.boundsMin, sceneData.boundsMax, 1e-3f, 1024);
        }

        if (!params.isMultiGpu && params.isRestirActive && normalDepthImage && prevNormalDepthImage) {
            VkImageCopy copyRegion{};
            copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.extent = { params.dispatchWidth, params.dispatchHeight, 1 };

            VkImageMemoryBarrier2 imgBarriers[2]{};
            imgBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            imgBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            imgBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            imgBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            imgBarriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            imgBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            imgBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            imgBarriers[0].image = normalDepthImage->getImage();
            imgBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            imgBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            imgBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            imgBarriers[1].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            imgBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            imgBarriers[1].dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            imgBarriers[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            imgBarriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            imgBarriers[1].image = prevNormalDepthImage->getImage();
            imgBarriers[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo copyDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            copyDep.imageMemoryBarrierCount = 2;
            copyDep.pImageMemoryBarriers = imgBarriers;
            vkCmdPipelineBarrier2(cmd, &copyDep);

            vkCmdCopyImage(cmd,
                           normalDepthImage->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                           prevNormalDepthImage->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                           1, &copyRegion);

            VkImageMemoryBarrier2 postBarrier{};
            postBarrier.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            postBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            postBarrier.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            postBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            postBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            postBarrier.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            postBarrier.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            postBarrier.image = prevNormalDepthImage->getImage();
            postBarrier.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            postDep.imageMemoryBarrierCount = 1;
            postDep.pImageMemoryBarriers = &postBarrier;
            vkCmdPipelineBarrier2(cmd, &postDep);
        }
    } else {
        if (causticsPipeline && config.enable_caustics && sceneData.hasDielectrics && numLights > 0) {
            causticsPipeline->recordFullPass(
                cmd, params.frameSlot,
                numTriangles, numSpheres, numMaterials, numLights,
                numOpaqueTriangles, params.frameIndex,
                normalDepthImage,
                camera ? camera->getFov() : 45.0f,
                config.progressive_accumulation,
                params.cameraMovedLastFrame,
                sceneData.hasDielectrics,
                sceneData.dielectricBoundsMin,
                sceneData.dielectricBoundsMax
            );
        }
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtpKhrPipeline->getPipeline());
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_RAY_TRACING_KHR, m_rtpPipelineLayout, 0, 1, &rtDescSet, 0, nullptr);

        uint32_t fracSppBits = std::bit_cast<uint32_t>(params.fractionalSpp);
        uint32_t rtPushConstants[16] = {
            numTriangles, numSpheres, numMaterials, numLights,
            params.tileOffsetX, params.tileOffsetY,
            params.isMultiGpu ? params.fullWidth : params.dispatchWidth,
            params.isMultiGpu ? params.fullHeight : params.dispatchHeight,
            useHwRT,
            hasEnvMap,
            std::bit_cast<uint32_t>(envIntensity),
            (config.progressive_accumulation && !params.accumReset) ? 1u : 0u,
            fracSppBits,
            params.totalCompositeSpp,
            numOpaqueTriangles,
            std::bit_cast<uint32_t>(config.indirect_clamp)
        };
        VkShaderStageFlags rtpStages = VK_SHADER_STAGE_RAYGEN_BIT_KHR | VK_SHADER_STAGE_CLOSEST_HIT_BIT_KHR | VK_SHADER_STAGE_MISS_BIT_KHR;
        vkCmdPushConstants(cmd, m_rtpPipelineLayout, rtpStages, 0, sizeof(rtPushConstants), rtPushConstants);
        m_rtpKhrPipeline->traceRays(cmd, params.dispatchWidth, params.dispatchHeight, 1);
    }

    if (governor) {
        governor->recordDispatch(params.frameSlot, activeDispatchSpp, params.bounces);
    }

    if (!useWavefront) {
        VkMemoryBarrier2 memBarrier{};
        memBarrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2;
        memBarrier.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR | VK_PIPELINE_STAGE_2_COPY_BIT;
        memBarrier.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT;
        memBarrier.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
        memBarrier.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;

        VkDependencyInfo depInfo{};
        depInfo.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO;
        depInfo.memoryBarrierCount = 1;
        depInfo.pMemoryBarriers = &memBarrier;
        vkCmdPipelineBarrier2(cmd, &depInfo);
    }

    if (!params.isMultiGpu && normalDepthImage && centerDepthBuffer) {
        VkBufferImageCopy copyRegion{};
        copyRegion.bufferOffset = 0;
        copyRegion.bufferRowLength = 0;
        copyRegion.bufferImageHeight = 0;
        copyRegion.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
        copyRegion.imageOffset = { static_cast<int32_t>(params.dispatchWidth / 2), static_cast<int32_t>(params.dispatchHeight / 2), 0 };
        copyRegion.imageExtent = { 1, 1, 1 };

        VkImageMemoryBarrier2 preCopy{};
        preCopy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        preCopy.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        preCopy.srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        preCopy.dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        preCopy.dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        preCopy.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        preCopy.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        preCopy.image = normalDepthImage->getImage();
        preCopy.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo preDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        preDep.imageMemoryBarrierCount = 1;
        preDep.pImageMemoryBarriers = &preCopy;
        vkCmdPipelineBarrier2(cmd, &preDep);

        vkCmdCopyImageToBuffer(cmd, normalDepthImage->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                               centerDepthBuffer->getBuffer(), 1, &copyRegion);

        VkImageMemoryBarrier2 postCopy{};
        postCopy.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
        postCopy.srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
        postCopy.srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
        postCopy.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_2_RAY_TRACING_SHADER_BIT_KHR;
        postCopy.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
        postCopy.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
        postCopy.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        postCopy.image = normalDepthImage->getImage();
        postCopy.subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
        postDep.imageMemoryBarrierCount = 1;
        postDep.pImageMemoryBarriers = &postCopy;
        vkCmdPipelineBarrier2(cmd, &postDep);
    }
}

} // namespace pathways
