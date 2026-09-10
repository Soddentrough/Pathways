#pragma once

#include <string>
#include <vector>
#include <cstdint>
#include <optional>
#include <glm/glm.hpp>

namespace pathways {

enum class MultiGpuMode {
    Off,
    CheckerboardTile,    // Checkerboard 2D Tiling (16x16, 32x32, 64x64) [Default when mGPU active]
    SampleParallel,      // Temporal Sample Parallelism
    Auto                 // Adaptive: SampleParallel if SPP > 1, else CheckerboardTile
};

enum class AccumFormat {
    RGBA16_SFLOAT, // 64-bit Half Float HDR (Industry standard for real-time graphics, 50% VRAM/PCIe footprint) [Default]
    RGBA32_SFLOAT  // 128-bit Full Float HDR
};

enum class PipelineType {
    RTP,       // Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline)
    Wavefront  // Wavefront Path Tracing with Work Lists & DGC
};

enum class WavefrontSortMode {
    None,      // Monolithic shade kernel, no material partitioning
    Archetype, // Technique A & B: Multi-queue wave-ballot partitioning with DGC Execution Sets
    BDA,       // Technique C: Buffer Device Address queue pointers
    Dual       // Technique D: 2D Spatial-Morton + Material Dual-Binning
};

struct Config {
    PipelineType pipeline_type = PipelineType::Wavefront; // Default: Wavefront Path Tracing
    WavefrontSortMode wavefront_sort_mode = WavefrontSortMode::None; // Default: Monolithic
    uint32_t width = 3840;
    uint32_t height = 2160;
    bool custom_resolution = false; // Set to true when --width or --height is passed explicitly on CLI
    bool fullscreen = true;         // Default: true (fullscreen by default, disable with --windowed)
    uint32_t spp = 1;
    uint32_t max_bounces = 4;
    uint32_t frame_limit = 0; // 0 = continuous (until window closed or interactive exit)
    uint32_t warmup_frames = 0; // Number of initial frames to discard from benchmark statistics
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
    bool enable_shadow_denoiser = false;
    float shadow_denoiser_depth_sigma = 0.02f;
    float shadow_denoiser_normal_power = 16.0f;
    bool enable_taa = false;              // Temporal Anti-Aliasing [Deprecated, default: disabled]
    float taa_blend_alpha = 0.10f;        // TAA temporal blend factor (0.10 current, 0.90 history)
    float taa_clipping_gamma = 2.25f;     // TAA variance clipping bounding box multiplier (optimized for stochastic 1-SPP)
    bool enable_atrous = false;           // Hierarchical Edge-Avoiding A-Trous Wavelet Diffuse Denoiser [Default: disabled]
    uint32_t atrous_passes = 3;           // Number of A-Trous filter iterations (1-5, default: 3 passes: s=1,2,4)
    float atrous_normal_power = 32.0f;    // Normal edge-stopping sensitivity
    float atrous_depth_sigma = 0.03f;     // Depth edge-stopping sensitivity
    bool enable_indirect_light = true;
    bool progressive_accumulation = true; // Accumulate samples over static frames (uncheck to evaluate real-time noise)

    uint32_t gpu_index = 0;
    MultiGpuMode mgpu_mode = MultiGpuMode::Off; // Default: Primary GPU (Multi-GPU only when passed via CLI or selected in menu)
    AccumFormat accum_format = AccumFormat::RGBA16_SFLOAT; // Default: RGBA16_SFLOAT (Industry standard for real-time HDR)
    bool double_buffered_shared_mem = true; // Double-buffered inter-GPU host memory for pipelined DMA transfers
    bool visualize_mgpu_split = false; // Visualize real-time load distribution across Dual GPUs
    uint32_t tile_size = 64;
    uint32_t wavefront_tile_size = 0; // Wavefront cache-resident tile size (0 = full frame monolithic, 256 = 256x256, 512 = 512x256, default: 0)
    float log_interval_sec = 0.0f; // 0.0 = disabled by default (no console spam); >0.0 logs every N seconds
    bool camera_motion = false;    // Simulate continuous camera motion (e.g. for testing interactive motion artifacts)
    bool test_scene_switching = false; // Run headless dynamic scene switching verification test

    // Camera view overrides (useful for headless testing & reproducible framing)
    std::optional<glm::vec3> camera_pos;
    std::optional<glm::vec3> camera_target;
    std::optional<glm::vec3> camera_up;
    std::optional<float> camera_fov;

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
