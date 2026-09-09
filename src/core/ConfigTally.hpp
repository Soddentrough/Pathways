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
    MultiGpuMode mgpu_mode = MultiGpuMode::Off;
    uint32_t width = 3840;
    uint32_t height = 2160;
    uint32_t spp = 1;
    uint32_t max_bounces = 4;
    AccumFormat accum_format = AccumFormat::RGBA16_SFLOAT;
    uint32_t tile_size = 64;

    bool operator==(const ConfigKey& o) const {
        if (scene_name != o.scene_name) return false;
        if (mgpu_mode != o.mgpu_mode) return false;
        if (width != o.width || height != o.height) return false;
        if (spp != o.spp || max_bounces != o.max_bounces) return false;
        if (accum_format != o.accum_format) return false;
        if (mgpu_mode == MultiGpuMode::CheckerboardTile && tile_size != o.tile_size) return false;
        return true;
    }

    std::string getLabel() const {
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
        return std::format("[{}] [{}] {}x{} | {} SPP | {} Bounces | {}", scene_name, modeStr, width, height, spp, max_bounces, fmtStr);
    }
};

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

    explicit ConfigStatsTally(const ConfigKey& k)
        : key(k), label(k.getLabel()) {}

    void addSample(double frameTimeMs, double primRtMs, double secRtMs, double tonemapMs) {
        if (frameTimeMs <= 0.001) return;
        frameCount++;
        sumFrameTimeMs += frameTimeMs;
        minFrameTimeMs = std::min(minFrameTimeMs, frameTimeMs);
        maxFrameTimeMs = std::max(maxFrameTimeMs, frameTimeMs);
        sumPrimaryRtMs += primRtMs;
        sumSecondaryRtMs += secRtMs;
        sumTonemapMs += tonemapMs;
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
