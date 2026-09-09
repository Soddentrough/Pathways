#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace pathways {

enum class MultiGpuMode {
    Off,
    InterleavedScanline, // Linearly scalable 50/50 scanline work distribution (any SPP) [Default when mGPU active]
    CheckerboardTile,    // Checkerboard 2D Tiling (16x16, 32x32, 64x64)
    SampleParallel,      // Temporal Sample Parallelism
    Auto                 // Adaptive: SampleParallel if SPP > 1, else InterleavedScanline
};

enum class AccumFormat {
    RGBA16_SFLOAT, // 64-bit Half Float HDR (Industry standard for real-time graphics, 50% VRAM/PCIe footprint) [Default]
    RGBA32_SFLOAT  // 128-bit Full Float HDR
};

struct Config {
    uint32_t width = 3840;
    uint32_t height = 2160;
    bool custom_resolution = false; // Set to true when --width or --height is passed explicitly on CLI
    bool fullscreen = true;         // Default: true (fullscreen by default, disable with --windowed)
    uint32_t spp = 1;
    uint32_t max_bounces = 4;
    uint32_t frame_limit = 0; // 0 = continuous (until window closed or interactive exit)
    float render_scale = 1.0f;

    // Dynamic Quality Governor & Target Frame Rate Limiter
    uint32_t target_fps = 0;          // 0 = uncapped [Default]
    bool adaptive_spp = false;        // Enable 3-axis dynamic sample rate governor [Default: false]
    uint32_t min_spp = 1;             // Minimum SPP floor [Default: 1]
    uint32_t max_spp = 16;            // Maximum SPP ceiling [Default: 16]
    uint32_t min_bounces = 2;         // Minimum bounce floor [Default: 2]
    uint32_t max_dynamic_bounces = 8; // Maximum bounce ceiling [Default: 8]

    bool headless = false;
    bool benchmark = false;
    bool validation_layers = true;
    bool aces_tonemap = true;
    bool enable_refraction = true;
    bool enable_shadows = true;
    bool enable_direct_light = true;
    bool enable_restir_di = false;
    bool enable_restir_spatial = true;
    uint32_t restir_spatial_samples = 3;
    float restir_spatial_radius = 8.0f;
    bool enable_indirect_light = true;

    uint32_t gpu_index = 0;
    MultiGpuMode mgpu_mode = MultiGpuMode::Off; // Default: Primary GPU (Multi-GPU only when passed via CLI or selected in menu)
    AccumFormat accum_format = AccumFormat::RGBA16_SFLOAT; // Default: RGBA16_SFLOAT (Industry standard for real-time HDR)
    bool double_buffered_shared_mem = true; // Double-buffered inter-GPU host memory for pipelined DMA transfers
    bool visualize_mgpu_split = false; // Visualize real-time load distribution across Dual GPUs
    uint32_t tile_size = 64;
    float log_interval_sec = 0.0f; // 0.0 = disabled by default (no console spam); >0.0 logs every N seconds
    bool camera_motion = false;    // Simulate continuous camera motion (e.g. for testing interactive motion artifacts)
    bool test_scene_switching = false; // Run headless dynamic scene switching verification test

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
