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
    VkPhysicalDeviceType deviceType,
    VkPipelineCache pipelineCache
)
    : m_device(device),
      m_allocator(allocator),
      m_physicalDevice(physicalDevice),
      m_deviceType(deviceType),
      m_pipelineCache(pipelineCache) {
}

RayTracingOrchestrator::~RayTracingOrchestrator() {
    m_wavefrontPipeline.reset();
    m_restirManager.reset();
    m_nrcManager.reset();
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
    const std::vector<char>& shadeComplexCode,
    const std::vector<char>& shadeEmissiveCode,
    bool hasDgcExecutionSet,
    const std::vector<char>& shadeDiffuseSecCode,
    const std::vector<char>& shadeComplexSecCode,
    bool enableDgcPreprocess,
    bool hasSubgroupSizeControl,
    uint32_t initialBatchPixels,
    const std::vector<char>& tailMegakernelCode,
    const std::vector<char>& shadeDielectricSecCode
) {
    m_wavefrontPipeline = std::make_unique<WavefrontPipeline>(
        m_device, m_allocator,
        width, height,
        classifyCode, intersectCode, shadeCode, shadowCode,
        shadeDiffuseCode, shadeDielectricCode, shadeComplexCode,
        shadeEmissiveCode,
        hasDgcExecutionSet,
        shadeDiffuseSecCode, shadeComplexSecCode,
        enableDgcPreprocess,
        hasSubgroupSizeControl,
        initialBatchPixels,
        tailMegakernelCode,
        m_pipelineCache,
        shadeDielectricSecCode
    );
    Logger::info("Wavefront Path Tracing Pipeline (Ray Queues & DGC) initialized successfully.");
}

void RayTracingOrchestrator::initWavefrontSpecializedVariants(
    const std::vector<char>& intersectAlphaCode,
    const std::vector<char>& shadeDiffuseRestirCode,
    const std::vector<char>& shadeDiffuseSecRestirCode,
    const std::vector<char>& shadeComplexRestirCode,
    const std::vector<char>& shadeComplexSecRestirCode,
    const std::vector<char>& shadeDielectricRestirCode,
    const std::vector<char>& shadeEmissiveRestirCode,
    const std::vector<char>& restirGiReuseCode,
    const std::vector<char>& restirDiReuseCode,
    const std::vector<char>& shadeDielectricSecRestirCode
) {
    if (m_wavefrontPipeline) {
        m_wavefrontPipeline->initSpecializedVariants(
            intersectAlphaCode,
            shadeDiffuseRestirCode,
            shadeDiffuseSecRestirCode,
            shadeComplexRestirCode,
            shadeComplexSecRestirCode,
            shadeDielectricRestirCode,
            shadeEmissiveRestirCode,
            restirGiReuseCode,
            restirDiReuseCode,
            shadeDielectricSecRestirCode
        );
    }
}

void RayTracingOrchestrator::initWavefrontAlphaPipeline(const std::vector<char>& intersectAlphaCode) {
    if (m_wavefrontPipeline) {
        m_wavefrontPipeline->initAlphaIntersectPipeline(intersectAlphaCode);
    }
}

void RayTracingOrchestrator::initWavefrontTailMegakernelPipeline(const std::vector<char>& tailMegakernelCode) {
    if (m_wavefrontPipeline) {
        m_wavefrontPipeline->initTailMegakernelPipeline(tailMegakernelCode);
    }
}

void RayTracingOrchestrator::initWavefrontRestirVariants(
    const std::vector<char>& shadeDiffuseRestirCode,
    const std::vector<char>& shadeDiffuseSecRestirCode,
    const std::vector<char>& shadeComplexRestirCode,
    const std::vector<char>& shadeComplexSecRestirCode,
    const std::vector<char>& shadeDielectricRestirCode,
    const std::vector<char>& shadeEmissiveRestirCode,
    const std::vector<char>& restirGiReuseCode,
    const std::vector<char>& restirDiReuseCode,
    const std::vector<char>& shadeDielectricSecRestirCode
) {
    if (m_wavefrontPipeline) {
        m_wavefrontPipeline->initRestirVariants(
            shadeDiffuseRestirCode,
            shadeDiffuseSecRestirCode,
            shadeComplexRestirCode,
            shadeComplexSecRestirCode,
            shadeDielectricRestirCode,
            shadeEmissiveRestirCode,
            restirGiReuseCode,
            restirDiReuseCode,
            shadeDielectricSecRestirCode
        );
    }
}

void RayTracingOrchestrator::initWavefrontNrcVariants(
    const std::vector<char>& shadeDiffuseNrcCode,
    const std::vector<char>& shadeDiffuseSecNrcCode,
    const std::vector<char>& shadeComplexNrcCode,
    const std::vector<char>& shadeComplexSecNrcCode,
    const std::vector<char>& shadeDielectricNrcCode,
    const std::vector<char>& shadeDielectricSecNrcCode
) {
    if (m_wavefrontPipeline) {
        m_wavefrontPipeline->initNrcVariants(
            shadeDiffuseNrcCode,
            shadeDiffuseSecNrcCode,
            shadeComplexNrcCode,
            shadeComplexSecNrcCode,
            shadeDielectricNrcCode,
            shadeDielectricSecNrcCode
        );
    }
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
        inferCode, trainCode, resolveCode,
        m_physicalDevice,
        m_pipelineCache
    );
    Logger::info("Neural Direct-Light Caching Subsystem (Wave32 WMMA & Atomic Buffer) initialized. Note: caches unshadowed direct light only — see docs/NRC.md.");
}

void RayTracingOrchestrator::createReSTIRResources() {
    // Buffer allocation is deferred: the grid is only (re)allocated when
    // ReSTIR is actually enabled (Engine::renderFrame lazy-init path) to
    // avoid ~144 B/pixel of idle VRAM when the feature is off.
    m_restirManager = std::make_unique<ReSTIRManager>(m_device, m_allocator);
    Logger::info("ReSTIR PT subsystem initialized (path-space reservoirs; inline temporal + spatial-tap reuse in wavefront shading).");
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

    if (totalDeviceVram < 15ULL * 1024 * 1024 * 1024) {
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
    if (config.max_bounces >= 2 && count > maxAutoBatches) {
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
    if (sceneData.materials.empty()) {
        m_cachedDivergentAreaRatio = 0.0f;
        return 0.0f;
    }
    if (sceneData.triangles.empty()) {
        if (sceneData.hasDielectrics) {
            m_cachedDivergentAreaRatio = 0.5f;
            return 0.5f;
        }
        for (const auto& mat : sceneData.materials) {
            uint32_t arch = computeMaterialArchetype(mat);
            if (arch == MATERIAL_ARCHETYPE_COMPLEX || arch == MATERIAL_ARCHETYPE_DIELECTRIC) {
                m_cachedDivergentAreaRatio = 0.5f;
                return 0.5f;
            }
        }
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

    if (config.enable_restir) {
        return WavefrontSortMode::Dual;
    }

    if (sceneData.materials.empty()) {
        return WavefrontSortMode::None;
    }

    // Pure Archetype sorting consistently outperforms 12-bit Dual Morton sorting across
    // non-ReSTIR scenes without spatial shuffle scatter penalties (10/11 sorted scenes faster).
    (void)getDivergentAreaRatio(sceneData);
    return WavefrontSortMode::Archetype;
}

void RayTracingOrchestrator::resize(uint32_t width, uint32_t height, const Config& config) {
    if (m_nrcManager) {
        m_nrcManager->resize(width, height);
    }
    if (m_restirManager && m_restirManager->isInitialized()) {
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
    CausticsPipeline* causticsPipeline,
    QualityGovernor* governor,
    Camera* camera,
    const SceneData& sceneData,
    uint32_t numTriangles,
    uint32_t numMaterials,
    uint32_t numLights,
    uint32_t numOpaqueTriangles,
    uint32_t hasEnvMap,
    float envIntensity,
    const Config& config,
    Image* mlSkipImage,
    Image* mlSpecAlbedoImage,
    Image* mlDirectDiffuseImage,
    Image* mlIndirectDiffuseImage,
    Image* mlDirectSpecularImage,
    Image* mlIndirectSpecularImage,
    Image* mlIrradianceImage,
    Image* mlPrimaryGuideImage,
    Image* mlSurfaceMvImage,
    Image* mlDiffHitDistImage,
    Image* mlSpecVmvImage,
    Image* mlSkipCompactImage,
    Image* motionVectorImage
) {
    if (params.skipRayTracing || !m_wavefrontPipeline) return;

    uint32_t activeDispatchSpp = params.spp;
    {
        if (config.enable_nrc && m_nrcManager) {
            m_nrcManager->resetCounters(cmd);
        }

        // Zero-fill freshly allocated ReSTIR reservoir grids before any shade
        // dispatch reads them (uninitialized memory must never alias a valid
        // reservoir; review 2026-10-10 §4.6).
        if (params.isRestirActive && m_restirManager) {
            m_restirManager->recordClearIfNeeded(cmd);
        }

        // Clear per-frame ray tracing target (FP16)
        VkClearColorValue clearZero = { { 0.0f, 0.0f, 0.0f, 0.0f } };
        VkImageSubresourceRange clearRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

        bool needAccumReset = params.accumReset || !config.progressive_accumulation;
        const bool isUpwaysActive = (config.denoiser_mode == DenoiserMode::Upways ||
                                     config.upscaler_mode == UpscalerMode::Upways ||
                                     config.upways_superres);
        const bool isFfxRrActive = (config.denoiser_mode == DenoiserMode::FfxRayRegeneration);
        const bool isTrainingCapture = !config.capture_training_data_dir.empty();
        bool isDenoisingActive = (isFfxRrActive || isUpwaysActive);
        const bool needSplit4SigBuffers = (isFfxRrActive || config.capture_channels >= 44u);
        const bool useCompactMlLive = isUpwaysActive && !isTrainingCapture && !isFfxRrActive &&
                                      (mlIrradianceImage != nullptr) &&
                                      (mlPrimaryGuideImage != nullptr) &&
                                      (mlSurfaceMvImage != nullptr) &&
                                      (mlDiffHitDistImage != nullptr) &&
                                      (mlSpecVmvImage != nullptr) &&
                                      (mlSkipCompactImage != nullptr);
        bool keepAccumWithMl = (isTrainingCapture || (params.isMultiGpu && params.mgpuMode == MultiGpuMode::SampleParallel && !useCompactMlLive));
        bool needFrameImage = !isDenoisingActive || keepAccumWithMl;
        if (frameImage && needFrameImage) {
            vkCmdClearColorImage(cmd, frameImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
        }

        // Compact live mode only needs to zero-clear sparse/additive compact targets
        // (classify overwrites every primary guide & surface MV pixel);
        // legacy training / FFX Ray Regeneration keeps the 64 B/px cleared float targets.
        bool needMlBuffers = isDenoisingActive || isTrainingCapture || needSplit4SigBuffers;
        if (needMlBuffers) {
            if (useCompactMlLive) {
                VkClearColorValue clearUintZero{};
                vkCmdClearColorImage(cmd, mlIrradianceImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearUintZero, 1, &clearRange);
                vkCmdClearColorImage(cmd, mlDiffHitDistImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearUintZero, 1, &clearRange);
                vkCmdClearColorImage(cmd, mlSpecVmvImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearUintZero, 1, &clearRange);
                vkCmdClearColorImage(cmd, mlSkipCompactImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearUintZero, 1, &clearRange);
            } else {
                if (mlDiffuseImage) {
                    vkCmdClearColorImage(cmd, mlDiffuseImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
                }
                if (mlSpecularImage) {
                    vkCmdClearColorImage(cmd, mlSpecularImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
                }
                if (mlSkipImage) {
                    vkCmdClearColorImage(cmd, mlSkipImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
                }
                if (mlSpecAlbedoImage) {
                    vkCmdClearColorImage(cmd, mlSpecAlbedoImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
                }
            }
        }
        if (needSplit4SigBuffers) {
            if (mlDirectDiffuseImage) {
                vkCmdClearColorImage(cmd, mlDirectDiffuseImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
            }
            if (mlIndirectDiffuseImage) {
                vkCmdClearColorImage(cmd, mlIndirectDiffuseImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
            }
            if (mlDirectSpecularImage) {
                vkCmdClearColorImage(cmd, mlDirectSpecularImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
            }
            if (mlIndirectSpecularImage) {
                vkCmdClearColorImage(cmd, mlIndirectSpecularImage->getImage(), VK_IMAGE_LAYOUT_GENERAL, &clearZero, 1, &clearRange);
            }
        }

        std::vector<VkImageMemoryBarrier2> clearBarriers;
        auto addClearBarrier = [&](Image* img) {
            if (!img) return;
            VkImageMemoryBarrier2 b{ VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2 };
            b.srcStageMask = VK_PIPELINE_STAGE_2_CLEAR_BIT;
            b.srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            b.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            b.dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            b.oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.image = img->getImage();
            b.subresourceRange = clearRange;
            clearBarriers.push_back(b);
        };
        if (needFrameImage) {
            addClearBarrier(frameImage);
        }
        if (needMlBuffers) {
            if (useCompactMlLive) {
                addClearBarrier(mlIrradianceImage);
                addClearBarrier(mlDiffHitDistImage);
            } else {
                addClearBarrier(mlDiffuseImage);
                addClearBarrier(mlSpecularImage);
                addClearBarrier(mlSkipImage);
                addClearBarrier(mlSpecAlbedoImage);
            }
        }
        if (needSplit4SigBuffers) {
            addClearBarrier(mlDirectDiffuseImage);
            addClearBarrier(mlIndirectDiffuseImage);
            addClearBarrier(mlDirectSpecularImage);
            addClearBarrier(mlIndirectSpecularImage);
        }

        if (!clearBarriers.empty()) {
            VkDependencyInfo clearDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            clearDep.imageMemoryBarrierCount = static_cast<uint32_t>(clearBarriers.size());
            clearDep.pImageMemoryBarriers = clearBarriers.data();
            vkCmdPipelineBarrier2(cmd, &clearDep);
        }

        WavefrontSceneData wfSceneData{};
        wfSceneData.numTriangles = numTriangles;
        wfSceneData.numMaterials = numMaterials;
        wfSceneData.numLights = numLights;
        wfSceneData.hasEnvMap = hasEnvMap;
        wfSceneData.envMapIntensity = envIntensity;
        wfSceneData.frameIndex = params.frameIndex;
        wfSceneData.useMorton = config.use_morton ? 1u : 0u;
        wfSceneData.accumulateHistory = (config.progressive_accumulation && !params.accumReset) ? 1u : 0u;
        wfSceneData.sortMode = static_cast<uint32_t>(getEffectiveWavefrontSortMode(config, sceneData));
        wfSceneData.numOpaqueTriangles = numOpaqueTriangles;
        wfSceneData.secondarySortMode = static_cast<uint32_t>(config.secondary_sort_mode);
        wfSceneData.cameraFlags = params.cameraFlags;

        bool enableNrcThisFrame = config.enable_nrc;
        wfSceneData.enableNrc = enableNrcThisFrame;
        wfSceneData.nrcBounce = config.nrc_bounce;
        wfSceneData.nrcTrainRatio = config.nrc_train_ratio;
        const bool hasFocalBounds =
            (sceneData.focalBoundsMin.x < sceneData.focalBoundsMax.x) &&
            (sceneData.focalBoundsMin.y < sceneData.focalBoundsMax.y) &&
            (sceneData.focalBoundsMin.z < sceneData.focalBoundsMax.z);
        const glm::vec3 nrcBMin = hasFocalBounds ? sceneData.focalBoundsMin : sceneData.boundsMin;
        const glm::vec3 nrcBMax = hasFocalBounds ? sceneData.focalBoundsMax : sceneData.boundsMax;
        wfSceneData.boundsMin = nrcBMin;
        wfSceneData.boundsMax = nrcBMax;
        wfSceneData.sceneBoundsMin = sceneData.boundsMin;
        wfSceneData.sceneBoundsMax = sceneData.boundsMax;
        wfSceneData.envNeeEnabled = params.envNeeEnabled;
        wfSceneData.streamlineSecondaryShading = config.streamline_secondary_shading;
        wfSceneData.enableDistanceClamping = config.distance_clamping;
        wfSceneData.maxSecondaryRayDistance = config.max_secondary_distance;
        wfSceneData.indirectClamp = config.indirect_clamp;
        wfSceneData.enableTailMegakernel = config.enable_tail_megakernel;
        wfSceneData.tailMegakernelBounce = config.tail_megakernel_bounce;
        wfSceneData.deltaUnroll = config.delta_unroll;
        wfSceneData.inlineShadows = config.inline_primary_shadows;
        wfSceneData.restirMCap = config.restir_m_cap;
        wfSceneData.earlyBounceCutoff = config.early_bounce_cutoff;
        wfSceneData.earlyBounceCutoffThreshold = config.early_bounce_cutoff_threshold;

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

        bool needNormals = (config.upscaler_mode == UpscalerMode::Upways ||
                            config.upscaler_mode == UpscalerMode::FSR ||
                            config.denoiser_mode == DenoiserMode::Upways ||
                            config.denoiser_mode == DenoiserMode::FfxRayRegeneration ||
                            config.enable_caustics ||
                            isTrainingCapture);
        bool needNdStore = (!useCompactMlLive && needNormals) ||
                           (config.upscaler_mode == UpscalerMode::FSR) ||
                           isFfxRrActive ||
                           config.enable_caustics ||
                           isTrainingCapture;
        bool needMvStore = (!useCompactMlLive && needNormals) ||
                           (config.upscaler_mode == UpscalerMode::FSR) ||
                           isFfxRrActive ||
                           params.isRestirActive ||
                           config.enable_caustics ||
                           isTrainingCapture;
        bool captureMl = (isUpwaysActive || isFfxRrActive || isTrainingCapture);
        bool captureMlV4 = (isUpwaysActive || config.capture_channels >= 32u);
        const bool captureSplit4Sig = (isFfxRrActive || config.capture_channels >= 44u);
        wfSceneData.captureMlData = (captureMl ? 1u : 0u)
                                  | (((captureMl || captureSplit4Sig) && keepAccumWithMl) ? 2u : 0u)
                                  | (captureMlV4 ? 4u : 0u)
                                  | (captureSplit4Sig ? 8u : 0u)
                                  | (useCompactMlLive ? 16u : 0u)
                                  | ((useCompactMlLive && needNdStore) ? 32u : 0u);
        wfSceneData.gbufferFlags = (1u << 31) | (needNormals ? 1u : 0u) | (needNdStore ? 2u : 0u) | (needMvStore ? 4u : 0u);

        if (causticsPipeline && config.enable_caustics && sceneData.hasDielectrics && numLights > 0) {
            causticsPipeline->recordTrace(
                cmd, params.frameSlot,
                numTriangles, numMaterials, numLights,
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

        const bool isBandParallel = (params.isMultiGpu && params.tileOffsetX == 3u);
        if (enableNrcThisFrame && m_nrcManager) {
            m_nrcManager->setBackend(config.nrc_backend);
            uint32_t nrcW = (params.isMultiGpu && !isBandParallel) ? params.fullWidth : params.dispatchWidth;
            uint32_t nrcH = (params.isMultiGpu && !isBandParallel) ? params.fullHeight : params.dispatchHeight;
            const uint32_t adamStep = m_nrcManager->getAdamStep();
            const bool nrcWarmBatch = (adamStep < 20u);
            const bool nrcWarmLr = (adamStep < 20u);
            float nrcLr = nrcWarmLr
                ? 5.8e-3f
                : ((!params.cameraMovedLastFrame && config.progressive_accumulation) ? 2.3e-3f : 2.8e-3f);
            uint32_t nrcBatch = nrcWarmBatch ? 64u : 32u;
            m_nrcManager->recordInferenceAndTraining(cmd, nrcW, nrcH, nrcBMin, nrcBMax,
                                                     0, params.frameSlot, wfSceneData.captureMlData,
                                                     std::min(config.indirect_clamp, 35.0f),
                                                     params.frameIndex, nrcLr, nrcBatch,
                                                     activeDispatchSpp == 1u);
        }

        const bool deferNdHistoryToMgpuMerge = params.isMultiGpu && (params.mgpuMode == MultiGpuMode::CheckerboardTile) && !isBandParallel;
        Image* depthHistorySrc = (needNdStore || !motionVectorImage) ? normalDepthImage : motionVectorImage;
        if ((params.isRestirActive || config.enable_caustics) && !deferNdHistoryToMgpuMerge && depthHistorySrc && prevNormalDepthImage) {
            VkImageCopy copyRegion{};
            copyRegion.srcSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.dstSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            copyRegion.extent = {
                (params.isMultiGpu && !isBandParallel) ? params.fullWidth : params.dispatchWidth,
                (params.isMultiGpu && !isBandParallel) ? params.fullHeight : params.dispatchHeight,
                1
            };

            VkImageMemoryBarrier2 imgBarriers[2]{};
            imgBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            imgBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            imgBarriers[0].srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            imgBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            imgBarriers[0].dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            imgBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            imgBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            imgBarriers[0].image = depthHistorySrc->getImage();
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
                           depthHistorySrc->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                           prevNormalDepthImage->getImage(), VK_IMAGE_LAYOUT_GENERAL,
                           1, &copyRegion);

            VkImageMemoryBarrier2 postBarriers[2]{};
            postBarriers[0].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            postBarriers[0].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            postBarriers[0].srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT;
            postBarriers[0].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            postBarriers[0].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT;
            postBarriers[0].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            postBarriers[0].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            postBarriers[0].image = prevNormalDepthImage->getImage();
            postBarriers[0].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            postBarriers[1].sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2;
            postBarriers[1].srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT;
            postBarriers[1].srcAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT;
            postBarriers[1].dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
            postBarriers[1].dstAccessMask = VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT;
            postBarriers[1].oldLayout = VK_IMAGE_LAYOUT_GENERAL;
            postBarriers[1].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            postBarriers[1].image = depthHistorySrc->getImage();
            postBarriers[1].subresourceRange = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1 };

            VkDependencyInfo postDep{ VK_STRUCTURE_TYPE_DEPENDENCY_INFO };
            postDep.imageMemoryBarrierCount = 2;
            postDep.pImageMemoryBarriers = postBarriers;
            vkCmdPipelineBarrier2(cmd, &postDep);
        }
    }

    if (governor) {
        governor->recordDispatch(params.frameSlot, activeDispatchSpp, params.bounces);
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
        preCopy.srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
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
        postCopy.dstStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT;
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
