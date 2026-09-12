#pragma once

#include "core/Config.hpp"
#include <string>
#include <vector>
#include <cstdint>
#include <format>
#include <algorithm>
#include <numeric>

namespace pathways {

struct ConfigKey {
    std::string scene_name = "Cornell Box";
    PipelineType pipeline_type = PipelineType::Wavefront;
    MultiGpuMode mgpu_mode = MultiGpuMode::Off;
    uint32_t width = 3840;
    uint32_t height = 2160;
    uint32_t spp = 1;
    uint32_t max_bounces = 4;
    DenoiserMode denoiser = DenoiserMode::None;
    AccumFormat accum_format = AccumFormat::RGBA16_SFLOAT;
    uint32_t tile_size = 64;
    bool enable_nrc = false;

    bool operator==(const ConfigKey& o) const {
        if (scene_name != o.scene_name) return false;
        if (pipeline_type != o.pipeline_type) return false;
        if (mgpu_mode != o.mgpu_mode) return false;
        if (denoiser != o.denoiser) return false;
        if (enable_nrc != o.enable_nrc) return false;
        if (width != o.width || height != o.height) return false;
        if (spp != o.spp || max_bounces != o.max_bounces) return false;
        if (accum_format != o.accum_format) return false;
        if (mgpu_mode == MultiGpuMode::CheckerboardTile && tile_size != o.tile_size) return false;
        return true;
    }

    std::string getLabel() const {
        std::string pipeStr = (pipeline_type == PipelineType::Wavefront) ? "Wavefront" : "RTP";
        std::string modeStr;
        switch (mgpu_mode) {
            case MultiGpuMode::Off:
                modeStr = "Single GPU";
                break;
            case MultiGpuMode::CheckerboardTile:
                modeStr = std::format("Dual GPU (Checkerboard {}x{})", tile_size, tile_size);
                break;
            case MultiGpuMode::SampleParallel:
                modeStr = "Dual GPU (Sample Parallelism)";
                break;
            case MultiGpuMode::Auto:
                modeStr = (spp > 1) ? "Dual GPU (Auto: Sample Parallel)" : std::format("Dual GPU (Auto: Checkerboard {}x{})", tile_size, tile_size);
                break;
        }
        std::string fmtStr = (accum_format == AccumFormat::RGBA16_SFLOAT) ? "FP16" : "FP32";
        std::string denoiserStr = "";
        if (denoiser == DenoiserMode::BMFR) {
            denoiserStr = " [BMFR]";
        }
        std::string nrcStr = enable_nrc ? " [NRC]" : "";
        return std::format("[{}]{}{} [{}] [{}] {}x{} | {} SPP | {} Bounces | {}", scene_name, denoiserStr, nrcStr, pipeStr, modeStr, width, height, spp, max_bounces, fmtStr);
    }
};

struct WavefrontStageSample {
    double classifyMs = 0.0;
    uint64_t primaryRays = 0;
    struct Bounce {
        double shadeMs = 0.0;
        double shadowMs = 0.0;
        double intersectMs = 0.0;
        uint64_t activeCount = 0;
        uint64_t nextCount = 0;
        uint64_t shadowCount = 0;
    };
    std::vector<Bounce> bounces;
};

struct BounceStageAvg {
    uint32_t bounce = 0;
    double shadeMs = 0.0;
    double shadowMs = 0.0;
    double intersectMs = 0.0;
    double totalMs = 0.0;
    uint64_t activeCount = 0;
    uint64_t nextCount = 0;
    uint64_t shadowCount = 0;
};

inline std::string formatRayCount(uint64_t count) {
    std::string s = std::to_string(count);
    int n = static_cast<int>(s.length()) - 3;
    while (n > 0) {
        s.insert(n, ",");
        n -= 3;
    }
    return s;
}

struct ConfigStatsTally {
    ConfigKey key;
    std::string label;
    uint32_t frameCount = 0;
    double sumFrameTimeMs = 0.0;
    double minFrameTimeMs = 1e9;
    double maxFrameTimeMs = 0.0;
    double sumPrimaryRtMs = 0.0;
    double sumSecondaryRtMs = 0.0;
    double sumTonemapMs = 0.0;

    // Acceleration structures
    double blasBuildTimeMs = 0.0;
    double blasSizeKb = 0.0;
    uint32_t blasTriangles = 0;
    double tlasBuildTimeMs = 0.0;
    double tlasSizeKb = 0.0;
    uint32_t tlasInstances = 0;
    double secBlasBuildTimeMs = 0.0;
    double secBlasSizeKb = 0.0;
    double secTlasBuildTimeMs = 0.0;
    double secTlasSizeKb = 0.0;
    uint32_t tlasGpuUpdateCount = 0;

    // Detailed pipeline stages
    bool hasWavefrontStages = false;
    uint32_t wavefrontSampleCount = 0;
    double sumClassifyMs = 0.0;
    uint64_t sumPrimaryRays = 0;

    struct BounceTally {
        double sumShadeMs = 0.0;
        double sumShadowMs = 0.0;
        double sumIntersectMs = 0.0;
        uint64_t sumActiveCount = 0;
        uint64_t sumNextCount = 0;
        uint64_t sumShadowCount = 0;
        uint32_t count = 0;
    };
    std::vector<BounceTally> bounceTallies;

    explicit ConfigStatsTally(const ConfigKey& k)
        : key(k), label(k.getLabel()) {}

    void addSample(double frameTimeMs, double primRtMs, double secRtMs, double tonemapMs,
                   const WavefrontStageSample* wfSample = nullptr) {
        if (frameTimeMs <= 0.001) return;
        frameCount++;
        sumFrameTimeMs += frameTimeMs;
        minFrameTimeMs = std::min(minFrameTimeMs, frameTimeMs);
        maxFrameTimeMs = std::max(maxFrameTimeMs, frameTimeMs);
        sumPrimaryRtMs += primRtMs;
        sumSecondaryRtMs += secRtMs;
        sumTonemapMs += tonemapMs;

        if (wfSample && !wfSample->bounces.empty()) {
            hasWavefrontStages = true;
            wavefrontSampleCount++;
            sumClassifyMs += wfSample->classifyMs;
            sumPrimaryRays += wfSample->primaryRays;

            if (bounceTallies.size() < wfSample->bounces.size()) {
                bounceTallies.resize(wfSample->bounces.size());
            }
            for (size_t b = 0; b < wfSample->bounces.size(); ++b) {
                bounceTallies[b].sumShadeMs += wfSample->bounces[b].shadeMs;
                bounceTallies[b].sumShadowMs += wfSample->bounces[b].shadowMs;
                bounceTallies[b].sumIntersectMs += wfSample->bounces[b].intersectMs;
                bounceTallies[b].sumActiveCount += wfSample->bounces[b].activeCount;
                bounceTallies[b].sumNextCount += wfSample->bounces[b].nextCount;
                bounceTallies[b].sumShadowCount += wfSample->bounces[b].shadowCount;
                bounceTallies[b].count++;
            }
        }
    }

    double getAvgFrameTimeMs() const {
        return frameCount > 0 ? (sumFrameTimeMs / frameCount) : 0.0;
    }

    double getAvgFps() const {
        double avg = getAvgFrameTimeMs();
        return avg > 0.0001 ? (1000.0 / avg) : 0.0;
    }

    double getAvgPrimaryRtMs() const {
        return frameCount > 0 ? (sumPrimaryRtMs / frameCount) : 0.0;
    }

    double getAvgSecondaryRtMs() const {
        return frameCount > 0 ? (sumSecondaryRtMs / frameCount) : 0.0;
    }

    double getAvgTonemapMs() const {
        return frameCount > 0 ? (sumTonemapMs / frameCount) : 0.0;
    }

    double getAvgClassifyMs() const {
        return wavefrontSampleCount > 0 ? (sumClassifyMs / wavefrontSampleCount) : 0.0;
    }

    uint64_t getAvgPrimaryRays() const {
        if (wavefrontSampleCount > 0 && sumPrimaryRays > 0) {
            return sumPrimaryRays / wavefrontSampleCount;
        }
        return static_cast<uint64_t>(key.width) * key.height * key.spp;
    }

    std::vector<BounceStageAvg> getAvgBounces() const {
        std::vector<BounceStageAvg> result;
        for (size_t b = 0; b < bounceTallies.size(); ++b) {
            const auto& bt = bounceTallies[b];
            if (bt.count == 0) continue;
            BounceStageAvg bAvg;
            bAvg.bounce = static_cast<uint32_t>(b);
            bAvg.shadeMs = bt.sumShadeMs / bt.count;
            bAvg.shadowMs = bt.sumShadowMs / bt.count;
            bAvg.intersectMs = bt.sumIntersectMs / bt.count;
            bAvg.totalMs = bAvg.shadeMs + bAvg.shadowMs + bAvg.intersectMs;
            bAvg.activeCount = bt.sumActiveCount / bt.count;
            bAvg.nextCount = bt.sumNextCount / bt.count;
            bAvg.shadowCount = bt.sumShadowCount / bt.count;
            result.push_back(bAvg);
        }
        return result;
    }

    double getRayThroughput() const {
        double avgMs = getAvgFrameTimeMs();
        if (avgMs <= 0.0001) return 0.0;
        double raysPerFrame = static_cast<double>(key.width) * key.height * key.spp * key.max_bounces;
        return raysPerFrame / (avgMs * 1e-3);
    }

    bool isTargetAchieved() const {
        return getAvgFrameTimeMs() < 8.0;
    }
};

} // namespace pathways
