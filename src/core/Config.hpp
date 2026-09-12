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

enum class OutputFormat {
    A2B10G10R10_UNORM, // 10-bit Deep Color / HDR output backbuffer (1024 levels) [Default]
    RGBA8_UNORM        // 8-bit SDR fallback (256 levels)
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

enum class SecondarySortMode {
    None,           // Standard unsorted secondary rays
    DirectionalDGC, // Option 1: On-Chip Directional Multi-Queue Binning via DGC & subgroup ballots
    SpatialIndex    // Option 2: 4-Byte Index-Only Spatial-Morton Reordering
};

enum class DenoiserMode {
    None,     // Raw stochastic path traced output (unfiltered progressive)
    Temporal, // Motion-vector guided Temporal Radiance Accumulation [Default]
    BMFR      // Blockwise Multi-Order Feature Regression [Experimental]
};

struct Config {
    PipelineType pipeline_type = PipelineType::Wavefront; // Default: Wavefront Path Tracing
    WavefrontSortMode wavefront_sort_mode = WavefrontSortMode::Dual; // Default: Technique D (3D Spatial-Morton + Material Dual-Binning)
    SecondarySortMode secondary_sort_mode = SecondarySortMode::None; // Secondary ray BVH traversal coherency mode
    uint32_t width = 3840;
    uint32_t height = 2160;
    bool custom_resolution = false; // Set to true when --width or --height is passed explicitly on CLI
    bool fullscreen = true;         // Default: true (fullscreen by default, disable with --windowed)
    uint32_t spp = 1;
    uint32_t max_bounces = 4;
    uint32_t frame_limit = 0; // 0 = continuous (until window closed or interactive exit)
    uint32_t warmup_frames = 0; // Number of initial frames to discard from benchmark statistics
    float render_scale = 1.0f;
    float exposure = 1.0f;

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
    bool enable_shadow_denoiser = false;
    float shadow_denoiser_depth_sigma = 0.02f;
    float shadow_denoiser_normal_power = 16.0f;
    bool enable_taa = false;              // Temporal Anti-Aliasing [Deprecated, default: disabled]
    float taa_blend_alpha = 0.10f;        // TAA temporal blend factor (0.10 current, 0.90 history)
    float taa_clipping_gamma = 2.25f;     // TAA variance clipping bounding box multiplier (optimized for stochastic 1-SPP)
    DenoiserMode denoiser_mode = DenoiserMode::None;     // Default: Pure Monte Carlo
    bool enable_temporal_accum = false;   // Motion-vector guided temporal accumulation [Default: disabled, opt-in via --temporal-accum / --denoiser temporal]
    bool enable_bmfr = false;             // Blockwise Multi-Order Feature Regression [Default: disabled, opt-in via --bmfr]
    float temporal_clamping_gamma = 1.25f;// Neighborhood variance clamp box multiplier
    float temporal_outlier_h = 0.75f;     // wRLS outlier rejection bandwidth
    float temporal_max_history = 32.0f;   // Maximum temporal history sample accumulation limit
    bool enable_indirect_light = true;
    bool progressive_accumulation = true; // Accumulate samples over static frames (uncheck to evaluate real-time noise)
    uint32_t max_accum_frames = 2048;     // Max accumulation frames before freezing stationary render (0 = Unlimited, default: 2048)

    // Neural Radiance Caching (NRC) with Wave32 WMMA (gfx1201 / Vulkan 1.4)
    bool enable_nrc = false;              // Neural Radiance Cache indirect query termination [Default: disabled]
    uint32_t nrc_bounce = 2;              // Path bounce depth where NRC terminates tracing and queries cache (default: 2)
    float nrc_train_ratio = 0.03f;        // Ratio of paths (2%-5%, default 0.03 = 3%) continuing tracing to ground truth depth for training

    uint32_t gpu_index = 0;
    MultiGpuMode mgpu_mode = MultiGpuMode::Off; // Default: Primary GPU (Multi-GPU only when passed via CLI or selected in menu)
    enum class MgpuTransferMode {
        Host,     // VK_EXT_external_memory_host (Zero-Copy Pinned Host Memory, high performance default)
        P2P,      // Linux DMA-BUF Direct PCIe P2P (Device-Local BAR)
        Staging   // CPU memcpy staging (Fallback)
    };
    MgpuTransferMode mgpu_transfer_mode = MgpuTransferMode::Host;
    AccumFormat accum_format = AccumFormat::RGBA16_SFLOAT; // Default: RGBA16_SFLOAT (Industry standard for real-time HDR)
    OutputFormat output_format = OutputFormat::A2B10G10R10_UNORM; // Default: 10-bit Deep Color / HDR output backbuffer
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
    bool dump_8bit_png = false; // Save dumped PNG frames as 8-bit instead of default 10/16-bit (conforms to single-negation rule)

    static Config parse(int argc, char* argv[]);
    static void printUsage(const char* progName);
};

} // namespace pathways
