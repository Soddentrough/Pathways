#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace pathways {

struct FrameStats {
    std::string gpu_name;
    uint32_t width = 0;
    uint32_t height = 0;
    uint32_t spp = 0;
    uint32_t total_frames = 0;
    double avg_frame_time_ms = 0.0;
    double min_frame_time_ms = 0.0;
    double max_frame_time_ms = 0.0;
    double avg_fps = 0.0;
    double rays_per_second = 0.0;
    double vram_used_mb = 0.0;
    uint32_t validation_errors = 0;
    bool target_achieved = false; // true if avg_frame_time_ms < 8.0
    std::string mgpu_mode_str = "off";

    // Profiler HUD & Diagnostic Metrics
    double primary_gpu_time_ms = 0.0;
    double secondary_gpu_time_ms = 0.0;
    double tonemap_time_ms = 0.0;
    double pcie_transfer_time_ms = 0.0;
    uint32_t num_triangles = 0;
    uint32_t num_spheres = 0;
    uint32_t num_materials = 0;
    uint32_t num_lights = 0;
    uint32_t num_textures = 0;
    bool has_hw_rt = true;
    bool has_dgc = true;
    bool is_rdna3 = true;
    bool is_rdna4 = false;
    std::string arch_name = "AMD RDNA3 (Navi 3x)";
    std::string short_arch = "RDNA3";
    std::string ray_accelerator_name = "AMD RDNA3 2nd Gen Ray Accelerators";
};

class ImageDumper {
public:
    static bool savePNG(const std::string& filepath, uint32_t width, uint32_t height, const uint8_t* rgbaPixels);
    static bool saveEXR(const std::string& filepath, uint32_t width, uint32_t height, const float* rgbaFloatPixels);
    static bool saveStatsJSON(const std::string& filepath, const FrameStats& stats);
};

} // namespace pathways
