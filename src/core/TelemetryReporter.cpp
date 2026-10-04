#include "core/TelemetryReporter.hpp"
#include "core/Engine.hpp"
#include "core/Logger.hpp"
#include "core/HwMonitor.hpp"
#include "utils/ImageDumper.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"
#include "vulkan/Swapchain.hpp"
#include "scene/Camera.hpp"
#include "rt/AccelerationStructure.hpp"
#include "rt/GpuTlasUpdatePipeline.hpp"
#include "rt/WavefrontPipeline.hpp"
#include "mgpu/MultiGpuManager.hpp"

#include <glm/glm.hpp>
#include <glm/detail/type_half.hpp>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <format>
#ifdef __linux__
#include <sys/utsname.h>
#endif

namespace pathways {

TelemetryReporter::TelemetryReporter(Engine* engine)
    : m_engine(engine) {
}

void TelemetryReporter::dumpOutputFiles() {
    if (!m_engine || !m_engine->m_context) return;
    VkDevice device = m_engine->m_context->getDevice();
    VmaAllocator allocator = m_engine->m_context->getAllocator();
    VkQueue queue = m_engine->m_context->getGraphicsQueue();

    vkDeviceWaitIdle(device);

    // Drain remaining in-flight queries after vkDeviceWaitIdle
    uint32_t drainCount = std::min(m_engine->m_totalFramesRendered, Engine::MAX_FRAMES_IN_FLIGHT);
    for (uint32_t d = 0; d < drainCount; ++d) {
        uint32_t slot = (m_engine->m_totalFramesRendered - drainCount + d) % Engine::MAX_FRAMES_IN_FLIGHT;
        uint32_t qBase = slot * Engine::QUERIES_PER_FRAME;
        uint64_t timestamps[Engine::QUERIES_PER_FRAME] = {0};
        vkGetQueryPoolResults(device, m_engine->m_queryPool, qBase, Engine::QUERIES_PER_FRAME, sizeof(timestamps), timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        double gpuRtMs = 0.0;
        if (timestamps[1] > timestamps[0]) {
            gpuRtMs = (timestamps[1] - timestamps[0]) * m_engine->m_timestampPeriod * 1e-6;
        }
        double gpuTonemapMs = 0.0;
        if (timestamps[3] > timestamps[2]) {
            gpuTonemapMs = (timestamps[3] - timestamps[2]) * m_engine->m_timestampPeriod * 1e-6;
        }
        double gpuUpwaysMs = 0.0;
        if (timestamps[5] > timestamps[4]) {
            gpuUpwaysMs = (timestamps[5] - timestamps[4]) * m_engine->m_timestampPeriod * 1e-6;
        }
        if (gpuRtMs > 10000.0) gpuRtMs = 0.0;
        if (gpuTonemapMs > 10000.0) gpuTonemapMs = 0.0;
        if (gpuUpwaysMs > 10000.0) gpuUpwaysMs = 0.0;

        double secGpuMs = 0.0;
        if (m_engine->m_mgpu && m_engine->m_mgpu->isMultiGpuActive()) {
            secGpuMs = m_engine->m_mgpu->getSecondaryGpuTimeMs();
        }
        double totalGpuMs = (m_engine->m_mgpu && m_engine->m_mgpu->isMultiGpuActive())
            ? (std::max(gpuRtMs, secGpuMs) + gpuTonemapMs + gpuUpwaysMs)
            : (gpuRtMs + gpuTonemapMs + gpuUpwaysMs);

        uint32_t lastCompletedSlot = (m_engine->m_totalFramesRendered - 1) % Engine::MAX_FRAMES_IN_FLIGHT;
        bool lastSlotSkipped = m_engine->m_slotSkippedRayTracing[lastCompletedSlot];

        if (m_engine->m_config.pipeline_type == PipelineType::Wavefront && m_engine->m_wavefrontPipeline && m_engine->m_totalFramesRendered > 0) {
            m_engine->m_lastWavefrontProfile = m_engine->m_wavefrontPipeline->getProfilingData(lastCompletedSlot, m_engine->m_timestampPeriod, m_engine->m_config.max_bounces);
        }

        if (!lastSlotSkipped && totalGpuMs > 0.01) {
            m_engine->m_lastGpuRtMs = gpuRtMs;
            m_engine->m_lastSecGpuMs = secGpuMs;
            m_engine->m_lastTonemapMs = gpuTonemapMs;
            m_engine->m_lastUpwaysMs = gpuUpwaysMs;
            m_engine->m_lastFrameTimeMs = totalGpuMs;
            m_engine->m_frameTimesMs.push_back(m_engine->m_lastFrameTimeMs);

            WavefrontStageSample wfSample;
            if (m_engine->m_lastWavefrontProfile.valid && m_engine->m_config.pipeline_type == PipelineType::Wavefront) {
                wfSample.classifyMs = m_engine->m_lastWavefrontProfile.classifyMs;
                wfSample.tailMegakernelMs = m_engine->m_lastWavefrontProfile.tailMegakernelMs;
                wfSample.tailMegakernelBounce = m_engine->m_lastWavefrontProfile.tailMegakernelBounce;
                uint32_t traceW = (m_engine->m_config.render_scale < 1.0f && m_engine->m_config.upscaler_mode != UpscalerMode::None) ?
                    static_cast<uint32_t>(m_engine->m_config.width * m_engine->m_config.render_scale) : m_engine->m_config.width;
                uint32_t traceH = (m_engine->m_config.render_scale < 1.0f && m_engine->m_config.upscaler_mode != UpscalerMode::None) ?
                    static_cast<uint32_t>(m_engine->m_config.height * m_engine->m_config.render_scale) : m_engine->m_config.height;
                wfSample.primaryRays = static_cast<uint64_t>(traceW) * traceH * m_engine->m_config.spp;
                for (const auto& bp : m_engine->m_lastWavefrontProfile.bounces) {
                    wfSample.bounces.push_back({bp.shadeMs, bp.shadowMs, bp.intersectMs, bp.gapBeforeShadeMs, bp.gapBeforeShadowMs, bp.gapBeforeIntersectMs, bp.activeCount, bp.nextCount, bp.shadowCount});
                }
            }
            recordFrameTally(totalGpuMs, gpuRtMs, secGpuMs, gpuTonemapMs, (wfSample.bounces.empty() && wfSample.tailMegakernelMs <= 0.0) ? nullptr : &wfSample);
        }
    }

    // 1. Dump LDR PNG
    if (!m_engine->m_config.dump_frame_path.empty() && m_engine->m_outputImage) {
        m_engine->saveScreenshot(m_engine->m_config.dump_frame_path);
        if (m_engine->m_pendingScreenshotFuture.valid()) {
            m_engine->m_pendingScreenshotFuture.wait();
        }
    }

    // 2. Dump HDR OpenEXR
    if (!m_engine->m_config.dump_hdr_path.empty() && m_engine->m_accumImage) {
        VkFormat accumFormat = m_engine->m_accumImage->getFormat();
        bool isFp16 = (accumFormat == VK_FORMAT_R16G16B16A16_SFLOAT);
        VkDeviceSize bytesPerPixel = isFp16 ? (4 * sizeof(uint16_t)) : (4 * sizeof(float));
        VkDeviceSize bufferSize = m_engine->m_config.width * m_engine->m_config.height * bytesPerPixel;
        Buffer staging(allocator, bufferSize, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VMA_MEMORY_USAGE_AUTO_PREFER_HOST, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

        vkResetCommandBuffer(m_engine->m_commandBuffers[0], 0);
        VkCommandBufferBeginInfo beginInfo{};
        beginInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
        vkBeginCommandBuffer(m_engine->m_commandBuffers[0], &beginInfo);

        m_engine->m_accumImage->transitionLayout(
            m_engine->m_commandBuffers[0], VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT
        );

        VkBufferImageCopy copyRegion{};
        copyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
        copyRegion.imageSubresource.layerCount = 1;
        copyRegion.imageExtent = { m_engine->m_config.width, m_engine->m_config.height, 1 };

        vkCmdCopyImageToBuffer(m_engine->m_commandBuffers[0], m_engine->m_accumImage->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, staging.getBuffer(), 1, &copyRegion);

        m_engine->m_accumImage->transitionLayout(
            m_engine->m_commandBuffers[0], VK_IMAGE_LAYOUT_GENERAL,
            VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT,
            VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT
        );

        vkEndCommandBuffer(m_engine->m_commandBuffers[0]);

        VkCommandBufferSubmitInfo cmdSubmitInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO };
        cmdSubmitInfo.commandBuffer = m_engine->m_commandBuffers[0];

        VkSubmitInfo2 submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO_2 };
        submitInfo.commandBufferInfoCount = 1;
        submitInfo.pCommandBufferInfos = &cmdSubmitInfo;
        vkQueueSubmit2(queue, 1, &submitInfo, VK_NULL_HANDLE);
        vkQueueWaitIdle(queue);

        staging.invalidate();
        if (isFp16) {
            const uint16_t* halfPixels = static_cast<const uint16_t*>(staging.map());
            std::vector<float> floatPixels(static_cast<size_t>(m_engine->m_config.width) * m_engine->m_config.height * 4);
            for (size_t p = 0; p < floatPixels.size(); ++p) {
                floatPixels[p] = glm::detail::toFloat32(halfPixels[p]);
            }
            ImageDumper::saveEXR(m_engine->m_config.dump_hdr_path, m_engine->m_config.width, m_engine->m_config.height, floatPixels.data());
            staging.unmap();
        } else {
            const float* floatPixels = static_cast<const float*>(staging.map());
            ImageDumper::saveEXR(m_engine->m_config.dump_hdr_path, m_engine->m_config.width, m_engine->m_config.height, floatPixels);
            staging.unmap();
        }
    }

    // 3. Dump UI Viewport Backbuffer
    if (!m_engine->m_config.dump_ui_path.empty() && m_engine->m_uiDumpBuffer && m_engine->m_swapchain) {
        uint32_t w = m_engine->m_swapchain->getExtent().width;
        uint32_t h = m_engine->m_swapchain->getExtent().height;
        VkFormat fmt = m_engine->m_swapchain->getFormat();
        std::vector<uint8_t> rgba(static_cast<size_t>(w) * h * 4);

        m_engine->m_uiDumpBuffer->invalidate();
        if (fmt == VK_FORMAT_A2R10G10B10_UNORM_PACK32 || fmt == VK_FORMAT_A2B10G10R10_UNORM_PACK32) {
            const uint32_t* raw32 = static_cast<const uint32_t*>(m_engine->m_uiDumpBuffer->map());
            bool isRgb = (fmt == VK_FORMAT_A2R10G10B10_UNORM_PACK32);
            float invPaperWhite = 1.0f / (m_engine->m_config.hdr_paper_white_nits > 0.0f ? m_engine->m_config.hdr_paper_white_nits : 200.0f);
            const float m1 = 0.1593017578125f;
            const float m2 = 78.84375f;
            const float c1 = 0.8359375f;
            const float c2 = 18.8515625f;
            const float c3 = 18.6875f;

            auto pqToSrgb = [&](float v) -> uint8_t {
                float v_pow = std::pow(std::max(v, 0.0f), 1.0f / m2);
                float num = std::max(v_pow - c1, 0.0f);
                float den = std::max(c2 - c3 * v_pow, 1e-6f);
                float linearNits = std::pow(num / den, 1.0f / m1) * 10000.0f;
                float srgb = std::pow(std::clamp(linearNits * invPaperWhite, 0.0f, 1.0f), 1.0f / 2.2f);
                return static_cast<uint8_t>(std::clamp(srgb * 255.0f + 0.5f, 0.0f, 255.0f));
            };

            for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
                uint32_t val = raw32[i];
                float c0 = static_cast<float>((val >> 20) & 0x3FF) / 1023.0f;
                float c1_val = static_cast<float>((val >> 10) & 0x3FF) / 1023.0f;
                float c2_val = static_cast<float>(val & 0x3FF) / 1023.0f;

                float r_norm = isRgb ? c0 : c2_val;
                float g_norm = c1_val;
                float b_norm = isRgb ? c2_val : c0;

                rgba[i * 4 + 0] = pqToSrgb(r_norm);
                rgba[i * 4 + 1] = pqToSrgb(g_norm);
                rgba[i * 4 + 2] = pqToSrgb(b_norm);
                rgba[i * 4 + 3] = 255;
            }
            m_engine->m_uiDumpBuffer->unmap();
        } else if (fmt == VK_FORMAT_R16G16B16A16_SFLOAT) {
            const uint16_t* raw16 = static_cast<const uint16_t*>(m_engine->m_uiDumpBuffer->map());
            float invPaperWhite = 80.0f / (m_engine->m_config.hdr_paper_white_nits > 0.0f ? m_engine->m_config.hdr_paper_white_nits : 200.0f);
            for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
                for (int c = 0; c < 3; ++c) {
                    float val = glm::detail::toFloat32(raw16[i * 4 + c]);
                    float srgb = std::pow(std::clamp(val * invPaperWhite, 0.0f, 1.0f), 1.0f / 2.2f);
                    rgba[i * 4 + c] = static_cast<uint8_t>(std::clamp(srgb * 255.0f + 0.5f, 0.0f, 255.0f));
                }
                rgba[i * 4 + 3] = 255;
            }
            m_engine->m_uiDumpBuffer->unmap();
        } else {
            const uint8_t* raw = static_cast<const uint8_t*>(m_engine->m_uiDumpBuffer->map());
            bool isBgra = (fmt == VK_FORMAT_B8G8R8A8_UNORM || fmt == VK_FORMAT_B8G8R8A8_SRGB);
            for (size_t i = 0; i < static_cast<size_t>(w) * h; ++i) {
                if (isBgra) {
                    rgba[i * 4 + 0] = raw[i * 4 + 2];
                    rgba[i * 4 + 1] = raw[i * 4 + 1];
                    rgba[i * 4 + 2] = raw[i * 4 + 0];
                    rgba[i * 4 + 3] = raw[i * 4 + 3];
                } else {
                    rgba[i * 4 + 0] = raw[i * 4 + 0];
                    rgba[i * 4 + 1] = raw[i * 4 + 1];
                    rgba[i * 4 + 2] = raw[i * 4 + 2];
                    rgba[i * 4 + 3] = raw[i * 4 + 3];
                }
            }
            m_engine->m_uiDumpBuffer->unmap();
        }
        ImageDumper::savePNG(m_engine->m_config.dump_ui_path, w, h, rgba.data());
    }

    // 4. Dump Stats JSON
    if (!m_engine->m_config.dump_stats_path.empty()) {
        FrameStats stats = getStats();
        ImageDumper::saveStatsJSON(m_engine->m_config.dump_stats_path, stats);
    }

    if (m_engine->m_camera) {
        m_engine->m_camera->advanceFrame();
    }
}

FrameStats TelemetryReporter::getStats() const {
    FrameStats stats;
    if (!m_engine || !m_engine->m_context) return stats;

    stats.gpu_name = m_engine->m_context->getDeviceName();
    if (m_engine->m_mgpu && m_engine->m_mgpu->isMultiGpuActive()) {
        stats.topology_name = stats.gpu_name + " + " + m_engine->m_mgpu->getSecondaryDeviceName();
    } else {
        stats.topology_name = stats.gpu_name;
    }
    stats.width = m_engine->m_config.width;
    stats.height = m_engine->m_config.height;
    stats.spp = m_engine->m_config.spp;
    stats.total_frames = m_engine->m_totalFramesRendered;
    stats.total_samples = m_engine->m_accumulatedSamples;
    stats.max_accum_frames = m_engine->m_config.max_accum_frames;
    stats.accumulation_complete = m_engine->m_accumulationComplete;
    stats.is_animating = m_engine->m_config.animate_objects && (m_engine->m_config.animation_speed > 0.0001f) && !m_engine->m_sceneData.animatedInstances.empty();
    stats.validation_errors = m_engine->m_context->getValidationErrors();
    stats.last_screenshot_path = m_engine->m_lastScreenshotPath;
    stats.screenshot_notification_timer = m_engine->m_screenshotNotificationTimer;

    stats.is_scene_loading = m_engine->isSceneLoading() || m_engine->m_pendingSceneChange;
    stats.loading_scene_name = m_engine->getLoadingSceneName();
    if (stats.is_scene_loading) {
        stats.loading_elapsed_sec = m_engine->getSceneLoadingElapsedSec();
    }

    stats.current_frame_time_ms = m_engine->m_lastFrameTimeMs;
    stats.current_fps = m_engine->m_lastFrameTimeMs > 0.0001 ? (1000.0 / m_engine->m_lastFrameTimeMs) : 0.0;

    stats.presentation_time_ms = m_engine->m_lastPresentationTimeMs;
    stats.presentation_fps = m_engine->m_lastPresentationTimeMs > 0.0001 ? (1000.0 / m_engine->m_lastPresentationTimeMs) : stats.current_fps;

    if (!m_engine->m_presentationTimesMs.empty()) {
        double pSum = std::accumulate(m_engine->m_presentationTimesMs.begin(), m_engine->m_presentationTimesMs.end(), 0.0);
        double avgPresTime = pSum / m_engine->m_presentationTimesMs.size();
        stats.avg_presentation_fps = avgPresTime > 0.0001 ? (1000.0 / avgPresTime) : stats.presentation_fps;
    } else {
        stats.avg_presentation_fps = stats.presentation_fps;
    }

    stats.target_fps = m_engine->m_config.target_fps;
    stats.adaptive_spp = m_engine->m_config.adaptive_spp;
    if (m_engine->m_governor && m_engine->m_config.adaptive_spp && m_engine->m_governor->getState().active) {
        stats.dynamic_spp = m_engine->m_governor->getState().currentSpp;
        stats.dynamic_bounces = m_engine->m_governor->getState().currentBounces;
    } else {
        stats.dynamic_spp = m_engine->m_config.spp;
        stats.dynamic_bounces = (m_engine->m_dynamicWavefrontBounces > 0) ? m_engine->m_dynamicWavefrontBounces : m_engine->m_config.max_bounces;
    }

    if (!m_engine->m_frameTimesMs.empty()) {
        double sum = std::accumulate(m_engine->m_frameTimesMs.begin(), m_engine->m_frameTimesMs.end(), 0.0);
        stats.avg_frame_time_ms = sum / m_engine->m_frameTimesMs.size();
        stats.min_frame_time_ms = *std::min_element(m_engine->m_frameTimesMs.begin(), m_engine->m_frameTimesMs.end());
        stats.max_frame_time_ms = *std::max_element(m_engine->m_frameTimesMs.begin(), m_engine->m_frameTimesMs.end());
        stats.avg_fps = stats.avg_frame_time_ms > 0.0 ? 1000.0 / stats.avg_frame_time_ms : 0.0;
        stats.target_frame_time_ms = m_engine->m_config.target_frame_time_ms;
        stats.target_achieved = (stats.avg_frame_time_ms < m_engine->m_config.target_frame_time_ms);

        double frameTimeForThroughput = stats.current_frame_time_ms > 0.001 ? stats.current_frame_time_ms : stats.avg_frame_time_ms;
        uint32_t traceW = (m_engine->m_config.render_scale < 1.0f && m_engine->m_config.upscaler_mode != UpscalerMode::None) ?
            static_cast<uint32_t>(m_engine->m_config.width * m_engine->m_config.render_scale) : m_engine->m_config.width;
        uint32_t traceH = (m_engine->m_config.render_scale < 1.0f && m_engine->m_config.upscaler_mode != UpscalerMode::None) ?
            static_cast<uint32_t>(m_engine->m_config.height * m_engine->m_config.render_scale) : m_engine->m_config.height;
        double raysPerFrame = static_cast<double>(traceW) * traceH * stats.dynamic_spp * stats.dynamic_bounces;
        stats.rays_per_second = (frameTimeForThroughput > 0.0) ? (raysPerFrame / (frameTimeForThroughput / 1000.0)) : 0.0;
    }

    switch (m_engine->m_config.mgpu_mode) {
        case MultiGpuMode::CheckerboardTile: stats.mgpu_mode_str = "checkerboard_tile"; break;
        case MultiGpuMode::SampleParallel: stats.mgpu_mode_str = "sample_parallel"; break;
        case MultiGpuMode::Auto: stats.mgpu_mode_str = (m_engine->m_config.spp > 1) ? "auto (sample_parallel)" : "auto (checkerboard_tile)"; break;
        default: stats.mgpu_mode_str = "single_gpu"; break;
    }

    if (m_engine->m_mgpu && m_engine->m_mgpu->isMultiGpuActive()) {
        stats.mgpu_transfer_mode_str = m_engine->m_mgpu->getTransferModeString();
    }

    stats.pipeline_type_str = (m_engine->m_config.pipeline_type == PipelineType::Wavefront) ? "wavefront" : "rtp";

    if (m_engine->m_config.pipeline_type == PipelineType::Wavefront) {
        WavefrontSortMode effectiveSort = m_engine->getEffectiveWavefrontSortMode();
        switch (m_engine->m_config.wavefront_sort_mode) {
            case WavefrontSortMode::Archetype: stats.wavefront_stats.sort_mode_str = "archetype"; break;
            case WavefrontSortMode::Dual: stats.wavefront_stats.sort_mode_str = "dual"; break;
            case WavefrontSortMode::None: stats.wavefront_stats.sort_mode_str = "none"; break;
            case WavefrontSortMode::Auto:
                stats.wavefront_stats.sort_mode_str = (effectiveSort == WavefrontSortMode::Dual) ? "auto (dual)" :
                                                      (effectiveSort == WavefrontSortMode::Archetype) ? "auto (archetype)" : "auto (none)";
                break;
        }
        switch (m_engine->m_config.secondary_sort_mode) {
            case SecondarySortMode::DirectionalDGC: stats.wavefront_stats.secondary_sort_mode_str = "directional"; break;
            case SecondarySortMode::DirectCoherent: stats.wavefront_stats.secondary_sort_mode_str = "direct-coherent"; break;
            case SecondarySortMode::DirectCoherentK8: stats.wavefront_stats.secondary_sort_mode_str = "direct-coherent-k8"; break;
            default: stats.wavefront_stats.secondary_sort_mode_str = "none"; break;
        }
        if (m_engine->m_lastWavefrontProfile.valid) {
            stats.wavefront_stats.valid = true;
            stats.wavefront_stats.total_ms = m_engine->m_lastWavefrontProfile.totalMs;
            stats.wavefront_stats.classify_ms = m_engine->m_lastWavefrontProfile.classifyMs;
            stats.wavefront_stats.resolve_ms = m_engine->m_lastWavefrontProfile.resolveMs;
            stats.wavefront_stats.tail_megakernel_ms = m_engine->m_lastWavefrontProfile.tailMegakernelMs;
            stats.wavefront_stats.tail_megakernel_bounce = m_engine->m_lastWavefrontProfile.tailMegakernelBounce;
            stats.wavefront_stats.queue_memory_footprint_mb = m_engine->m_lastWavefrontProfile.queueMemoryFootprintMb;
            stats.wavefront_stats.estimated_vram_traffic_mb = m_engine->m_lastWavefrontProfile.estimatedVramTrafficMb;
            for (const auto& bp : m_engine->m_lastWavefrontProfile.bounces) {
                FrameStats::BounceProfile bProf{};
                bProf.bounce = bp.bounce;
                bProf.shade_ms = bp.shadeMs;
                bProf.shadow_ms = bp.shadowMs;
                bProf.intersect_ms = bp.intersectMs;
                bProf.active_rays = bp.activeCount;
                bProf.shadow_rays = bp.shadowCount;
                bProf.next_rays = bp.nextCount;
                bProf.diff_rays = bp.diffCount;
                bProf.diel_rays = bp.dielCount;
                bProf.cond_rays = bp.condCount;
                bProf.comp_rays = bp.compCount;
                bProf.emis_rays = bp.emisCount;
                bProf.pass_rays = bp.passCount;
                stats.wavefront_stats.bounces.push_back(bProf);
            }
        }
    }

    stats.primary_gpu_time_ms = m_engine->m_lastGpuRtMs;
    stats.secondary_gpu_time_ms = m_engine->m_lastSecGpuMs;
    stats.tonemap_time_ms = m_engine->m_lastTonemapMs;
    stats.upways_time_ms = m_engine->m_lastUpwaysMs;
    stats.num_triangles = m_engine->m_numTriangles;
    stats.num_instanced_triangles = m_engine->m_numInstancedTriangles;
    stats.num_instances = m_engine->m_numInstances;
    stats.num_spheres = m_engine->m_numSpheres;
    stats.num_materials = m_engine->m_numMaterials;
    stats.num_lights = m_engine->m_numLights;
    stats.num_textures = static_cast<uint32_t>(m_engine->getSceneTextures().size());
    stats.width = m_engine->m_config.width;
    stats.height = m_engine->m_config.height;
    stats.spp = m_engine->m_config.spp;
    stats.max_bounces = m_engine->m_config.max_bounces;
    stats.render_scale = m_engine->m_config.render_scale;
    stats.total_frames = m_engine->m_totalFramesRendered;
    stats.validation_errors = m_engine->m_context->getValidationErrors();

    // 1. Session & Host Platform Metadata
    stats.os_name = HwMonitor::getOsName();
#ifdef __linux__
    struct utsname uts{};
    if (uname(&uts) == 0) {
        stats.kernel_version = std::string(uts.sysname) + " " + uts.release;
    }
#endif
    stats.cpu_model = HwMonitor::getCpuModel();
    stats.ram_total_gb = static_cast<double>(HwMonitor::getTotalRamMb()) / 1024.0;

    // 2. Physical & Driver Device Information
    stats.gpu_name = m_engine->m_context->getDeviceName();
    stats.vendor_id = m_engine->m_context->getVendorID();
    stats.device_id = m_engine->m_context->getDeviceID();
    stats.arch_name = m_engine->m_context->getArchitectureName();
    stats.short_arch = m_engine->m_context->getShortArchName();
    stats.ray_accelerator_name = m_engine->m_context->getRayAcceleratorName();
    stats.is_rdna3 = m_engine->m_context->isRDNA3();
    stats.is_rdna4 = m_engine->m_context->isRDNA4();
    uint32_t drvVer = m_engine->m_context->getDriverVersion();
    stats.driver_version_str = std::format("{}.{}.{}", (drvVer >> 22) & 0x3FF, (drvVer >> 12) & 0x3FF, drvVer & 0xFFF);
    uint32_t apiVer = m_engine->m_context->getApiVersion();
    stats.vulkan_api_str = std::format("{}.{}.{}", VK_API_VERSION_MAJOR(apiVer), VK_API_VERSION_MINOR(apiVer), VK_API_VERSION_PATCH(apiVer));
    stats.device_type_str = (m_engine->m_context->getDeviceType() == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) ? "Discrete GPU" : "Integrated GPU";
    stats.total_vram_mb = static_cast<double>(m_engine->m_context->getTotalVramBytes()) / (1024.0 * 1024.0);
    stats.vram_used_mb = static_cast<double>(m_engine->m_context->getAllocatedVramBytes()) / (1024.0 * 1024.0);
    stats.vram_budget_mb = stats.total_vram_mb;

    // Primary GPU PCIe and Sensors
    const auto& primPci = m_engine->m_context->getPciLinkInfo();
    stats.primary_pci_link = primPci.formattedLink;
    stats.primary_pci_speed = primPci.currentSpeed;
    stats.primary_pci_width = primPci.currentWidth;
    stats.primary_pci_max_speed = primPci.maxSpeed;
    stats.primary_pci_max_width = primPci.maxWidth;
    stats.primary_pci_degraded = primPci.isDegraded;
    stats.primary_pci_degraded_reason = primPci.degradationReason;
    stats.primary_gpu_clock_mhz = m_engine->m_hwMonitor ? m_engine->m_hwMonitor->getGpu0ClockMhz() : 0;
    stats.primary_gpu_temp_c = m_engine->m_hwMonitor ? m_engine->m_hwMonitor->getGpu0TempC() : 0;

    // 3. Secondary GPU Hardware & Driver
    if (m_engine->m_mgpu && m_engine->m_mgpu->isMultiGpuActive()) {
        stats.is_mgpu_active = true;
        stats.secondary_gpu_name = m_engine->m_mgpu->getSecondaryDeviceName();
        if (m_engine->m_mgpu->getSecondaryContext()) {
            const auto& secPci = m_engine->m_mgpu->getSecondaryContext()->getPciLinkInfo();
            stats.secondary_arch_name = m_engine->m_mgpu->getSecondaryContext()->getArchitectureName();
            stats.secondary_pci_link = secPci.formattedLink;
            if (!secPci.formattedLink.empty() && secPci.formattedLink != "PCIe N/A") {
                stats.mgpu_interconnect_str = secPci.formattedLink;
            }
            stats.secondary_pci_speed = secPci.currentSpeed;
            stats.secondary_pci_width = secPci.currentWidth;
            stats.secondary_pci_max_speed = secPci.maxSpeed;
            stats.secondary_pci_max_width = secPci.maxWidth;
            stats.secondary_pci_degraded = secPci.isDegraded;
            stats.secondary_pci_degraded_reason = secPci.degradationReason;
            stats.secondary_gpu_clock_mhz = m_engine->m_hwMonitor ? m_engine->m_hwMonitor->getGpu1ClockMhz() : 0;
            stats.secondary_gpu_temp_c = m_engine->m_hwMonitor ? m_engine->m_hwMonitor->getGpu1TempC() : 0;
        }
    } else if (m_engine->m_mgpu && m_engine->m_mgpu->isSecondaryInitialized()) {
        stats.is_mgpu_active = false;
        stats.secondary_gpu_name = m_engine->m_mgpu->getSecondaryDeviceName();
        if (m_engine->m_mgpu->getSecondaryContext()) {
            const auto& secPci = m_engine->m_mgpu->getSecondaryContext()->getPciLinkInfo();
            stats.secondary_arch_name = m_engine->m_mgpu->getSecondaryContext()->getArchitectureName();
            stats.secondary_pci_link = secPci.formattedLink;
            stats.secondary_pci_speed = secPci.currentSpeed;
            stats.secondary_pci_width = secPci.currentWidth;
            stats.secondary_pci_max_speed = secPci.maxSpeed;
            stats.secondary_pci_max_width = secPci.maxWidth;
            stats.secondary_pci_degraded = secPci.isDegraded;
            stats.secondary_pci_degraded_reason = secPci.degradationReason;
        }
        stats.secondary_gpu_clock_mhz = m_engine->m_hwMonitor ? m_engine->m_hwMonitor->getGpu1ClockMhz() : 0;
        stats.secondary_gpu_temp_c = m_engine->m_hwMonitor ? m_engine->m_hwMonitor->getGpu1TempC() : 0;
    } else {
        stats.is_mgpu_active = false;
        stats.secondary_gpu_name = "";
        stats.secondary_arch_name = "";
        stats.secondary_pci_link = "";
        stats.secondary_pci_speed = "";
        stats.secondary_pci_width = 0;
        stats.secondary_pci_max_speed = "";
        stats.secondary_pci_max_width = 0;
        stats.secondary_pci_degraded = false;
        stats.secondary_pci_degraded_reason = "";
        stats.secondary_gpu_clock_mhz = 0;
        stats.secondary_gpu_temp_c = 0;
    }

    // 4. Hardware Support Levels
    stats.has_hw_rt = (m_engine->m_tlas != nullptr);
    stats.has_ray_query = true;
    stats.has_as = (m_engine->m_tlas != nullptr);
    stats.has_bda = true;
    stats.has_dho = true;
    stats.has_rt_pipeline = (m_engine->m_rtpKhrPipeline != nullptr);
    stats.has_dgc = (m_engine->m_rtpKhrPipeline && m_engine->m_rtpKhrPipeline->isIndirectSupported());
    stats.dgc_preprocess = true;
    stats.has_subgroup_control = m_engine->m_context->hasSubgroupSizeControl();
    stats.subgroup_size = 32;
    stats.has_dynamic_rendering = true;
    stats.has_timeline_semaphores = true;
    stats.has_sync2 = true;

    // 5. Engine Settings & State
    stats.visualize_mgpu_split = m_engine->m_config.visualize_mgpu_split;
    switch (m_engine->m_config.upscaler_mode) {
        case UpscalerMode::FSR3: stats.upscaler_mode_str = "FSR 3.1"; break;
        case UpscalerMode::Upways: stats.upscaler_mode_str = "Upways"; break;
        case UpscalerMode::None: default: stats.upscaler_mode_str = "None"; break;
    }
    stats.render_scale = m_engine->m_config.render_scale;
    stats.max_bounces = m_engine->m_config.max_bounces;
    stats.checkerboard_tile_size = m_engine->m_config.tile_size;
    stats.enable_direct_light = m_engine->m_config.enable_direct_light;
    stats.enable_indirect_light = m_engine->m_config.enable_indirect_light;
    stats.enable_refraction = m_engine->m_config.enable_refraction;
    stats.enable_shadows = m_engine->m_config.enable_shadows;
    stats.aces_tonemap = m_engine->m_config.aces_tonemap;
    if (m_engine->m_swapchain && !m_engine->m_config.headless) {
        stats.swapchain_format_str = m_engine->m_swapchain->getFormatName();
        stats.swapchain_color_space_str = m_engine->m_swapchain->getColorSpaceName();
        stats.is_hdr_display = m_engine->m_swapchain->isHdr();
        stats.hdr_mode_str = (m_engine->m_swapchain->getHdrMode() == HdrDisplayMode::scRGB) ? "scRGB Linear (16-bit Float)" :
                             ((m_engine->m_swapchain->getHdrMode() == HdrDisplayMode::HDR10) ? "HDR10 PQ (10-bit Rec.2020)" : "SDR sRGB (8-bit)");
    } else {
        stats.swapchain_format_str = "R8G8B8A8_UNORM (Headless Offscreen)";
        stats.swapchain_color_space_str = "SRGB_NONLINEAR";
        stats.is_hdr_display = false;
        stats.hdr_mode_str = "Headless SDR";
    }
    stats.hdr_peak_nits = m_engine->m_config.hdr_peak_nits;
    stats.hdr_paper_white_nits = m_engine->m_config.hdr_paper_white_nits;
    stats.scene_path = m_engine->m_config.scene_path.empty() ? "Cornell Box + Specular/Refraction Spheres" : m_engine->m_config.scene_path;
    stats.hdri_path = m_engine->m_config.hdri_path;

    // 6. Active Camera Framing
    if (m_engine->m_camera) {
        glm::vec3 pos = m_engine->m_camera->getPosition();
        stats.cam_pos[0] = pos.x;
        stats.cam_pos[1] = pos.y;
        stats.cam_pos[2] = pos.z;
        stats.cam_yaw = m_engine->m_camera->getYaw();
        stats.cam_pitch = m_engine->m_camera->getPitch();
        stats.cam_fov = m_engine->m_camera->getFov();
    }

    for (const auto& tally : m_configTallies) {
        if (tally.frameCount <= 2 && m_configTallies.size() > 1) {
            bool hasSubstantial = false;
            for (const auto& other : m_configTallies) {
                if (other.frameCount >= 5) {
                    hasSubstantial = true;
                    break;
                }
            }
            if (hasSubstantial) {
                continue;
            }
        }
        FrameStats::ConfigTallySummary s;
        s.label = tally.label;
        s.frame_count = tally.frameCount;
        s.avg_frame_time_ms = tally.getAvgFrameTimeMs();
        s.min_frame_time_ms = tally.minFrameTimeMs;
        s.max_frame_time_ms = tally.maxFrameTimeMs;
        s.avg_fps = tally.getAvgFps();
        s.primary_gpu_time_ms = tally.getAvgPrimaryRtMs();
        s.secondary_gpu_time_ms = tally.getAvgSecondaryRtMs();
        s.tonemap_time_ms = tally.getAvgTonemapMs();
        s.gigarays_per_second = tally.getRayThroughput() * 1e-9;
        s.target_frame_time_ms = m_engine->m_config.target_frame_time_ms;
        s.target_achieved = tally.isTargetAchieved(m_engine->m_config.target_frame_time_ms);

        if (tally.hasWavefrontStages && tally.wavefrontSampleCount > 0) {
            s.pipeline_stages.is_wavefront = true;
            s.pipeline_stages.classify_ms = tally.getAvgClassifyMs();
            s.pipeline_stages.primary_rays = tally.getAvgPrimaryRays();
            s.pipeline_stages.tail_megakernel_ms = tally.getAvgTailMegakernelMs();
            s.pipeline_stages.tail_megakernel_bounce = tally.tailMegakernelBounce;
            s.pipeline_stages.tonemap_ms = tally.getAvgTonemapMs();
            auto bounces = tally.getAvgBounces();
            for (const auto& b : bounces) {
                FrameStats::StageBounceSummary sb;
                sb.bounce = b.bounce;
                sb.shade_ms = b.shadeMs;
                sb.shadow_ms = b.shadowMs;
                sb.intersect_ms = b.intersectMs;
                sb.total_bounce_ms = b.totalMs;
                sb.active_rays = b.activeCount;
                sb.rays_left = b.nextCount;
                sb.shadow_rays = b.shadowCount;
                s.pipeline_stages.bounces.push_back(sb);
            }
        } else {
            s.pipeline_stages.is_wavefront = false;
            s.pipeline_stages.ray_tracing_pass_ms = tally.getAvgPrimaryRtMs();
            s.pipeline_stages.tonemap_ms = tally.getAvgTonemapMs();
        }

        s.blas_build_time_ms = tally.blasBuildTimeMs;
        s.blas_size_kb = tally.blasSizeKb;
        s.blas_triangles = tally.blasTriangles;
        s.tlas_build_time_ms = tally.tlasBuildTimeMs;
        s.tlas_size_kb = tally.tlasSizeKb;
        s.tlas_instances = tally.tlasInstances;
        s.sec_blas_build_time_ms = tally.secBlasBuildTimeMs;
        s.sec_blas_size_kb = tally.secBlasSizeKb;
        s.sec_tlas_build_time_ms = tally.secTlasBuildTimeMs;
        s.sec_tlas_size_kb = tally.secTlasSizeKb;
        s.tlas_gpu_updates = tally.tlasGpuUpdateCount;

        stats.configurations_breakdown.push_back(std::move(s));
    }

    if (m_engine->m_asManager) {
        stats.blas_build_time_ms = m_engine->m_asManager->getLastBlasBuildTimeMs();
        stats.blas_size_kb = m_engine->m_asManager->getBlasSizeKb();
        stats.blas_triangles = m_engine->m_asManager->getBlasTriangles();
        stats.tlas_build_time_ms = m_engine->m_asManager->getLastTlasBuildTimeMs();
        stats.tlas_size_kb = m_engine->m_asManager->getTlasSizeKb();
        stats.tlas_instances = m_engine->m_asManager->getTlasInstances();
    }
    if (m_engine->m_mgpu && m_engine->m_mgpu->getSecondaryAsManager()) {
        auto* secAs = m_engine->m_mgpu->getSecondaryAsManager();
        stats.sec_blas_build_time_ms = secAs->getLastBlasBuildTimeMs();
        stats.sec_blas_size_kb = secAs->getBlasSizeKb();
        stats.sec_tlas_build_time_ms = secAs->getLastTlasBuildTimeMs();
        stats.sec_tlas_size_kb = secAs->getTlasSizeKb();
    }
    stats.tlas_gpu_updates = m_engine->m_tlasUpdatePipeline ? m_engine->m_tlasUpdatePipeline->getGpuUpdateCount() : 0;

    return stats;
}

void TelemetryReporter::recordFrameTally(double frameTimeMs, double primRtMs, double secRtMs, double tonemapMs,
                                         const WavefrontStageSample* wfSample) {
    if (!m_engine) return;
    ConfigKey key;
    key.scene_name = m_engine->getActiveSceneName();
    key.pipeline_type = m_engine->m_config.pipeline_type;
    key.mgpu_mode = (m_engine->m_mgpu && m_engine->m_mgpu->isMultiGpuActive() && m_engine->m_config.mgpu_mode != MultiGpuMode::Off) ? m_engine->m_config.mgpu_mode : MultiGpuMode::Off;
    key.denoiser = (m_engine->m_config.denoiser_mode == DenoiserMode::Upways) ? DenoiserMode::Upways : DenoiserMode::None;
    key.enable_nrc = m_engine->m_config.enable_nrc;
    key.width = m_engine->m_config.width;
    key.height = m_engine->m_config.height;
    key.spp = (m_engine->m_governor && m_engine->m_config.adaptive_spp && m_engine->m_governor->getState().active) ? m_engine->m_governor->getState().currentSpp : m_engine->m_config.spp;
    key.max_bounces = (m_engine->m_governor && m_engine->m_config.adaptive_spp && m_engine->m_governor->getState().active) ? m_engine->m_governor->getState().currentBounces : m_engine->m_config.max_bounces;
    key.accum_format = m_engine->m_config.accum_format;
    key.tile_size = m_engine->m_config.tile_size;
    key.upscaler = m_engine->m_config.upscaler_mode;
    key.mgpu_upscale_mode = m_engine->m_config.mgpu_upscale_mode;
    key.render_scale = m_engine->m_config.render_scale;
    key.enable_tail_megakernel = m_engine->m_config.enable_tail_megakernel;
    key.tail_bounce = m_engine->m_config.tail_megakernel_bounce;

    auto updateAsMetrics = [this](ConfigStatsTally& t) {
        if (m_engine->m_asManager) {
            t.blasBuildTimeMs = m_engine->m_asManager->getLastBlasBuildTimeMs();
            t.blasSizeKb = m_engine->m_asManager->getBlasSizeKb();
            t.uncompactedBlasSizeKb = m_engine->m_asManager->getUncompactedBlasSizeKb();
            t.blasCompacted = m_engine->m_asManager->isBlasCompacted();
            t.blasTriangles = m_engine->m_asManager->getBlasTriangles();
            t.tlasBuildTimeMs = m_engine->m_asManager->getLastTlasBuildTimeMs();
            t.tlasSizeKb = m_engine->m_asManager->getTlasSizeKb();
            t.tlasInstances = m_engine->m_asManager->getTlasInstances();
        }
        if (m_engine->m_mgpu && m_engine->m_mgpu->getSecondaryAsManager()) {
            auto* secAs = m_engine->m_mgpu->getSecondaryAsManager();
            t.secBlasBuildTimeMs = secAs->getLastBlasBuildTimeMs();
            t.secBlasSizeKb = secAs->getBlasSizeKb();
            t.secTlasBuildTimeMs = secAs->getLastTlasBuildTimeMs();
            t.secTlasSizeKb = secAs->getTlasSizeKb();
        }
        t.tlasGpuUpdateCount = m_engine->m_tlasUpdatePipeline ? m_engine->m_tlasUpdatePipeline->getGpuUpdateCount() : 0;
    };

    for (auto& tally : m_configTallies) {
        if (tally.key == key) {
            tally.addSample(frameTimeMs, primRtMs, secRtMs, tonemapMs, wfSample);
            updateAsMetrics(tally);
            return;
        }
    }
    ConfigStatsTally newTally(key);
    newTally.addSample(frameTimeMs, primRtMs, secRtMs, tonemapMs, wfSample);
    updateAsMetrics(newTally);
    m_configTallies.push_back(std::move(newTally));
}

void TelemetryReporter::printExecutionSummary() const {
    if (!m_engine) return;
    Logger::info("========================================================================================");
    if (m_configTallies.empty()) {
        Logger::info("  Execution Summary: No frames rendered.");
        Logger::info("========================================================================================");
        return;
    }

    // Prune transient startup tallies (e.g. <= 2 frames sampled before window settled)
    // when a primary configuration exists
    std::vector<const ConfigStatsTally*> activeTallies;
    for (const auto& tally : m_configTallies) {
        if (tally.frameCount <= 2 && m_configTallies.size() > 1) {
            bool hasSubstantial = false;
            for (const auto& other : m_configTallies) {
                if (other.frameCount >= 5) {
                    hasSubstantial = true;
                    break;
                }
            }
            if (hasSubstantial) {
                continue; // Skip startup transient tally
            }
        }
        activeTallies.push_back(&tally);
    }

    if (activeTallies.empty()) {
        for (const auto& tally : m_configTallies) {
            activeTallies.push_back(&tally);
        }
    }

    Logger::info("  Pathways Hybrid Path Tracing Engine - Execution Summary ({} Configuration{})",
                 activeTallies.size(), activeTallies.size() == 1 ? "" : "s");
    Logger::info("========================================================================================");

    for (size_t i = 0; i < activeTallies.size(); ++i) {
        const auto& tally = *activeTallies[i];
        Logger::info("  [Config {}/{}] {}", i + 1, activeTallies.size(), tally.label);
        Logger::info("    Frames Sampled:      {}", tally.frameCount);
        Logger::info("    Average Frame Time:  {:.3f} ms ({:.1f} FPS)", tally.getAvgFrameTimeMs(), tally.getAvgFps());
        Logger::info("    Frame Time Range:    min: {:.3f} ms | max: {:.3f} ms", tally.minFrameTimeMs, tally.maxFrameTimeMs);
        Logger::info("    Acceleration Structures:");
        if (tally.blasCompacted) {
            double ratio = (1.0 - (tally.blasSizeKb / tally.uncompactedBlasSizeKb)) * 100.0;
            Logger::info("      - BLAS Build:        {:.3f} ms ({:.2f} KB compacted from {:.2f} KB, -{:.1f}%, {} Triangles)",
                         tally.blasBuildTimeMs, tally.blasSizeKb, tally.uncompactedBlasSizeKb, ratio, tally.blasTriangles);
        } else {
            Logger::info("      - BLAS Build:        {:.3f} ms ({:.2f} KB, {} Triangles)",
                         tally.blasBuildTimeMs, tally.blasSizeKb, tally.blasTriangles);
        }
        Logger::info("      - TLAS Build:        {:.3f} ms ({:.2f} KB, {} Instance{})",
                         tally.tlasBuildTimeMs, tally.tlasSizeKb, tally.tlasInstances, tally.tlasInstances == 1 ? "" : "s");
        if (tally.key.mgpu_mode != MultiGpuMode::Off && tally.secBlasBuildTimeMs > 0.0) {
            Logger::info("      - Secondary BLAS:    {:.3f} ms ({:.2f} KB)",
                         tally.secBlasBuildTimeMs, tally.secBlasSizeKb);
            Logger::info("      - Secondary TLAS:    {:.3f} ms ({:.2f} KB)",
                         tally.secTlasBuildTimeMs, tally.secTlasSizeKb);
        }
        Logger::info("      - GPU TLAS Updates:  {} update{}",
                     tally.tlasGpuUpdateCount, tally.tlasGpuUpdateCount == 1 ? "" : "s");

        if (tally.key.mgpu_mode != MultiGpuMode::Off) {
            double secXferMs = m_engine->m_mgpu ? m_engine->m_mgpu->getSecondaryTransferTimeMs() : 0.0;
            Logger::info("    GPU Breakdown:       GPU 0 RT: {:.3f} ms | GPU 1 RT: {:.3f} ms (Xfer: {:.3f} ms) | Tonemap: {:.3f} ms",
                         tally.getAvgPrimaryRtMs(), tally.getAvgSecondaryRtMs(), secXferMs, tally.getAvgTonemapMs());
            // Find single GPU baseline for speedup calculation
            double baselineMs = 0.0;
            for (const auto* other : activeTallies) {
                if (other->key.scene_name == tally.key.scene_name &&
                    other->key.pipeline_type == tally.key.pipeline_type &&
                    other->key.mgpu_mode == MultiGpuMode::Off &&
                    other->key.width == tally.key.width &&
                    other->key.height == tally.key.height &&
                    other->key.spp == tally.key.spp &&
                    other->key.max_bounces == tally.key.max_bounces &&
                    other->key.accum_format == tally.key.accum_format &&
                    other->key.denoiser == tally.key.denoiser &&
                    other->key.enable_nrc == tally.key.enable_nrc) {
                    baselineMs = other->getAvgFrameTimeMs();
                    break;
                }
            }
            if (baselineMs > 0.001) {
                double speedup = baselineMs / tally.getAvgFrameTimeMs();
                double efficiency = (speedup / 2.0) * 100.0;
                Logger::info("    Multi-GPU Scaling:   {:.2f}x speedup vs Single GPU ({:.1f}% efficiency)", speedup, efficiency);
            }
        } else {
            Logger::info("    GPU Breakdown:       GPU 0 RT: {:.3f} ms | Tonemap: {:.3f} ms (Single GPU)",
                         tally.getAvgPrimaryRtMs(), tally.getAvgTonemapMs());
        }

        if (tally.hasWavefrontStages && tally.wavefrontSampleCount > 0) {
            bool inlineShadowsActive = m_engine->m_config.inline_primary_shadows;
            Logger::info("    Pipeline Stages{}:", inlineShadowsActive ? " (Inline Hardware Shadows Active)" : "");
            Logger::info("      - Classify (Primary RayGen): {:.3f} ms | {:L} rays left (100.0%)",
                         tally.getAvgClassifyMs(), tally.getAvgPrimaryRays());
            auto avgBounces = tally.getAvgBounces();
            for (const auto& b : avgBounces) {
                double bounceTotalMs = b.shadeMs + b.shadowMs + b.intersectMs + b.gapBeforeShadeMs + b.gapBeforeShadowMs + b.gapBeforeIntersectMs;
                double rayPercent = tally.getAvgPrimaryRays() > 0 ? (static_cast<double>(b.nextCount) / tally.getAvgPrimaryRays()) * 100.0 : 0.0;
                if (b.intersectMs >= 0.0) {
                    Logger::info("      - Bounce {}: Gap1: {:.3f}ms | Shade: {:.3f} ms | Gap2: {:.3f}ms | Shadow: {:.3f} ms | Gap3: {:.3f}ms | Intersect: {:.3f} ms (Total: {:.3f} ms) | {:12L} rays left ({:5.1f}%)",
                                 b.bounce, b.gapBeforeShadeMs, b.shadeMs, b.gapBeforeShadowMs, b.shadowMs, b.gapBeforeIntersectMs, b.intersectMs,
                                 bounceTotalMs, b.nextCount, rayPercent);
                } else {
                    Logger::info("      - Bounce {}: Gap1: {:.3f}ms | Shade: {:.3f} ms | Gap2: {:.3f}ms | Shadow: {:.3f} ms | Gap3: {:.3f}ms | Intersect:   None   (Total: {:.3f} ms) | {:12L} rays left ({:5.1f}%)",
                                 b.bounce, b.gapBeforeShadeMs, b.shadeMs, b.gapBeforeShadowMs, b.shadowMs, b.gapBeforeIntersectMs,
                                 bounceTotalMs, b.nextCount, rayPercent);
                }
            }
            if (tally.getAvgTailMegakernelMs() > 0.0) {
                Logger::info("      - Tail Megakernel (Bounce {}..{}): {:.3f} ms",
                             tally.tailMegakernelBounce, tally.key.max_bounces, tally.getAvgTailMegakernelMs());
            }
            Logger::info("      - Tonemap / Resolve:         {:.3f} ms", tally.getAvgTonemapMs());
        }

        Logger::info("    Ray Throughput:      {:.2f} GigaRays/sec", tally.getRayThroughput() * 1e-9);
        if (tally.isTargetAchieved(m_engine->m_config.target_frame_time_ms)) {
            Logger::info("    Sub-{:.1f}ms Budget:    \033[32mACHIEVED\033[0m", m_engine->m_config.target_frame_time_ms);
        } else {
            Logger::info("    Sub-{:.1f}ms Budget:    \033[33mEXCEEDED\033[0m (+{:.2f} ms)",
                         m_engine->m_config.target_frame_time_ms, tally.getAvgFrameTimeMs() - m_engine->m_config.target_frame_time_ms);
        }
    }

    Logger::info("----------------------------------------------------------------------------------------");
    Logger::info("  Total Frames Rendered: {} | Validation Errors: {}", m_engine->m_totalFramesRendered, m_engine->m_context->getValidationErrors());
    Logger::info("========================================================================================");
}

} // namespace pathways
