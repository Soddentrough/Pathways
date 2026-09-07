#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace pathways {

enum class MultiGpuMode {
    Off,
    SampleParallel,
    CheckerboardTile,
    DynamicWorkQueue
};

enum class PipelineType {
    Wavefront,
    Megakernel
};

struct Config {
    uint32_t width = 3840;
    uint32_t height = 2160;
    bool custom_resolution = false; // Set to true when --width or --height is passed explicitly on CLI
    uint32_t spp = 1;
    uint32_t max_bounces = 4;
    uint32_t frame_limit = 0; // 0 = continuous (until window closed or interactive exit)
    float render_scale = 1.0f;

    bool headless = false;
    bool benchmark = false;
    bool validation_layers = true;
    bool aces_tonemap = true;
    bool enable_refraction = true;
    bool enable_shadows = true;
    bool enable_direct_light = true;
    bool enable_indirect_light = true;
    bool enable_hardware_rt = true; // Auto-enabled when VK_KHR_ray_query is supported

    uint32_t gpu_index = 0;
    MultiGpuMode mgpu_mode = MultiGpuMode::Off;
    PipelineType pipeline_type = PipelineType::Wavefront;
    bool enable_morton_order = true;
    uint32_t tile_size = 64;
    float log_interval_sec = 0.0f; // 0.0 = disabled by default (no console spam); >0.0 logs every N seconds

    std::string scene_path = "";
    std::string hdri_path = "";
    std::string dump_frame_path = "";
    std::string dump_ui_path = "";
    std::string dump_hdr_path = "";
    std::string dump_stats_path = "";

    static Config parse(int argc, char* argv[]);
    static void printUsage(const char* progName);
};

} // namespace pathways
