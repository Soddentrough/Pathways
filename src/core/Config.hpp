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

enum class MgpuUpscaleMode {
    PostMerge,   // Final Frame Upscaling: Checkerboard 64x64 tiles merged on GPU0, then upscaled to 4K (Max FPS, 0 seams) [Default for Tile mode]
    SampleBlend  // Merge Upscaled Frames: Dual full passes independently upscaled to 4K, then averaged on GPU0 (Max Quality, 2x SPP denoising) [Default for Sample mode]
};

enum class AccumFormat {
    RGBA16_SFLOAT, // 64-bit Half Float HDR (Industry standard for real-time graphics, 50% VRAM/PCIe footprint) [Default]
    RGBA32_SFLOAT  // 128-bit Full Float HDR
};

enum class PipelineType {
    RTP,       // Dedicated Hardware Ray Tracing Pipeline (VK_KHR_ray_tracing_pipeline)
    Wavefront  // Wavefront Path Tracing with Ray Queues & DGC
};

enum class WavefrontSortMode {
    None,      // Monolithic shade kernel, no material partitioning
    Archetype, // Multi-queue wave-ballot partitioning with DGC Execution Sets
    Dual       // 3D Spatial-Morton intra-wave sort + Archetype dual-binning (Default)
};

enum class SecondarySortMode {
    None = 0,           // Standard unsorted secondary rays (Default)
    DirectionalDGC = 1, // Option 1: On-Chip Directional Multi-Queue Binning via DGC & subgroup ballots
    DirectCoherent = 2, // Direct Coherent Ray Generation via Tangent Space Reuse (Xiang et al. 2023, K=4)
    DirectCoherentK8 = 3 // Direct Coherent Ray Generation via Tangent Space Reuse (Xiang et al. 2023, K=8)
};

enum class DenoiserMode {
    None,     // Raw stochastic path traced output (unfiltered progressive)
    Upways    // Upways Neural Reconstruction & Super-Resolution (Wave32 WMMA)
};

enum class UpscalerMode {
    None,   // Native resolution or direct linear blit
    FSR3,   // AMD FidelityFX Super Resolution 3.1 (Temporal Accumulation)
    Upways, // Pathways Native Neural Combined Denoiser & Super-Resolution (Wave32 WMMA)
    FSR1    // AMD FidelityFX Super Resolution 1.0 (Spatial EASU + RCAS)
};

struct Config {
    PipelineType pipeline_type = PipelineType::Wavefront; // Default: Wavefront Path Tracing
    WavefrontSortMode wavefront_sort_mode = WavefrontSortMode::Dual; // Default: Technique D (3D Spatial-Morton + Material Dual-Binning)
    bool use_morton = false; // 2D Morton Z-curve mapping for wavefront classification (default: false / linear raster)
    uint32_t macro_tile_size = 0; // Legacy macro-tile cache panning (deprecated in favor of coarse batches)
    uint32_t batch_count = 0;    // Coarse batch count (0 = auto-detect based on GPU profile, 1 = monolithic, 2, 4, 8, etc.)
    uint32_t batch_pixels = 0;   // Coarse batch ray budget in pixels (0 = auto-detect based on GPU profile)
    SecondarySortMode secondary_sort_mode = SecondarySortMode::DirectCoherent; // Direct Coherent Ray Generation via Tangent Space Reuse (Xiang et al. 2023, K=4) [Default]
    bool streamline_secondary_shading = true; // Streamline secondary bounce shading (1-sample NEE, pure Lambertian BRDF) [Default: true]
    bool distance_clamping = true;            // Scene-scale invariant secondary ray distance clamping [Default: true]
    float max_secondary_distance = 0.0f;      // Override maximum secondary ray distance in world units (0 = automatic scene diameter * 1.25)
    float indirect_clamp = 35.0f;             // Maximum indirect / secondary bounce radiance luminance (0.0 = unlimited / unclamped, default: 35.0)
    uint32_t width = 3840;
    uint32_t height = 2160;
    bool custom_resolution = false; // Set to true when --width or --height is passed explicitly on CLI
    bool fullscreen = true;         // Default: true (fullscreen by default, disable with --windowed)
    uint32_t spp = 1;
    uint32_t max_bounces = 4;
    uint32_t frame_limit = 0; // 0 = continuous (until window closed or interactive exit)
    uint32_t warmup_frames = 0; // Number of initial frames to discard from benchmark statistics
    float render_scale = 1.0f;
    uint32_t render_width = 0;   // Explicit internal ray tracing render width (0 = derived from render_scale * width)
    uint32_t render_height = 0;  // Explicit internal ray tracing render height (0 = derived from render_scale * height)
    float exposure = 1.0f;

    uint32_t getRenderWidth() const {
        if (render_width > 0) return render_width;
        if (render_scale < 1.0f && (upscaler_mode != UpscalerMode::None || upways_superres)) {
            return std::max(1u, static_cast<uint32_t>(std::round(width * render_scale)));
        }
        return width;
    }
    uint32_t getRenderHeight() const {
        if (render_height > 0) return render_height;
        if (render_scale < 1.0f && (upscaler_mode != UpscalerMode::None || upways_superres)) {
            return std::max(1u, static_cast<uint32_t>(std::round(height * render_scale)));
        }
        return height;
    }

    // Dynamic Quality Governor & Target Frame Rate Limiter
    uint32_t target_fps = 0;          // 0 = uncapped [Default]
    float target_frame_time_ms = 8.3f; // Target frame time budget in milliseconds [Default: 8.3 ms]
    bool adaptive_spp = false;        // Enable 3-axis dynamic sample rate governor [Default: false]
    uint32_t min_spp = 1;             // Minimum SPP floor [Default: 1]
    uint32_t max_spp = 16;            // Maximum SPP ceiling [Default: 16]
    uint32_t min_bounces = 2;         // Minimum bounce floor [Default: 2]
    uint32_t max_dynamic_bounces = 8; // Maximum bounce ceiling [Default: 8]

    bool headless = false;
    bool benchmark = false;
    bool validation_layers = true;
    bool aces_tonemap = true;
    // High Dynamic Range (HDR) Display
    bool enable_hdr = true;              // Auto-negotiate HDR display formats (scRGB Linear / HDR10 PQ) [Default: true]
    float hdr_peak_nits = 1000.0f;       // Display peak luminance in cd/m^2 (nits) [Default: 1000.0]
    float hdr_paper_white_nits = 200.0f; // Reference paper white luminance in cd/m^2 (nits) [Default: 200.0]
    bool custom_hdr_peak = false;        // Set to true when --hdr-peak is passed explicitly on CLI
    bool enable_refraction = true;
    bool enable_shadows = true;
    bool inline_primary_shadows = false; // Detached shadow queue evaluation (Default: false for max occupancy & 0 LDS)
    bool enable_direct_light = true;
    bool enable_light_tree = false;      // Hierarchical Light Tree importance sampling for many-light scenes [Default: false]
    bool dgc_preprocess = true;          // DGC explicit preprocessing enabled by default (disable via --no-dgc-preprocess)
    bool dgc_batch_preprocess = true;    // Batched DGC preprocessing enabled by default (disable via --no-dgc-batch-preprocess)
    // Denoiser Defaults & Rationale:
    // IMPORTANT: Default is Pure Monte Carlo (DenoiserMode::None).
    // Keeping Pure Monte Carlo as default ensures unbiased, perfectly isotropic 1-SPP noise
    // with zero motion trails during interactive navigation, and seamless progressive convergence (up to 2048 spp)
    // when stationary. Real-time reconstruction is provided via Upways or FSR 3.1.
    DenoiserMode denoiser_mode = DenoiserMode::None;     // Default: Pure Monte Carlo
    bool upways_superres = false;         // Upways 2x Continuous Super-Resolution (e.g. 1080p -> 4K)
    std::string upways_weights_path = ""; // Custom path to upways_weights.bin
    UpscalerMode upscaler_mode = UpscalerMode::None; // Super-resolution upscaler [Default: None, opt-in via --upscaler]
    bool upscaler_sharpening = false;                // Enable RCAS sharpening pass [default: false]
    float upscaler_sharpness = 0.0f;                 // RCAS contrast-adaptive sharpness [0.0 - 1.0] (default: 0.0)
    bool enable_indirect_light = true;
    bool progressive_accumulation = true; // Accumulate samples over static frames (uncheck to evaluate real-time noise)
    uint32_t max_accum_frames = 2048;     // Max accumulation frames before freezing stationary render (0 = Unlimited, default: 2048)

    // Neural Radiance Caching (NRC) with Wave32 WMMA (gfx1201 / Vulkan 1.4)
    bool enable_nrc = false;              // Neural Radiance Cache indirect query termination [Default: disabled]
    uint32_t nrc_bounce = 2;              // Path bounce depth where NRC terminates tracing and queries cache (default: 2)
    float nrc_train_ratio = 0.03f;        // Ratio of paths (2%-5%, default 0.03 = 3%) continuing tracing to ground truth depth for training

    // Caustics & Forward Photon Injection (Vulkan 1.4 hardware rayQueryEXT)
    bool enable_caustics = false;         // Enable real-time forward ray-traced caustics [Default: disabled, opt-in via --caustics]
    uint32_t caustic_photons = 1048576;   // Number of caustic photons traced per frame (default: 1048576 = 1024x1024)

    // ReSTIR Spatio-Temporal Reservoir Resampling
    bool enable_restir_di = false;        // Spatio-temporal reservoir resampling [Default: false, opt-in via --restir]
    uint32_t restir_di_m_cap = 30;        // Temporal history M-cap for ReSTIR (default: 30)

    uint32_t gpu_index = 0;
    MultiGpuMode mgpu_mode = MultiGpuMode::Off; // Default: Primary GPU (Multi-GPU only when passed via CLI or selected in menu)
    MgpuUpscaleMode mgpu_upscale_mode = MgpuUpscaleMode::PostMerge; // Multi-GPU upscaling topology: PostMerge (Final Frame) or SampleBlend (Merged Frames)
    enum class MgpuTransferMode {
        Host,     // VK_EXT_external_memory_host (Zero-Copy Pinned Host Memory, high performance default)
        P2P       // Linux DMA-BUF Direct PCIe P2P (Device-Local BAR)
    };
    MgpuTransferMode mgpu_transfer_mode = MgpuTransferMode::Host;
    AccumFormat accum_format = AccumFormat::RGBA16_SFLOAT; // Default: RGBA16_SFLOAT (Preserve FP16 bandwidth and performance)
    bool double_buffered_shared_mem = true; // Double-buffered inter-GPU host memory for pipelined DMA transfers
    bool visualize_mgpu_split = false; // Visualize real-time load distribution across Dual GPUs
    uint32_t tile_size = 64;
    float log_interval_sec = 0.0f; // 0.0 = disabled by default (no console spam); >0.0 logs every N seconds
    bool camera_motion = false;    // Simulate continuous camera motion (e.g. for testing interactive motion artifacts)
    float gamepad_deadzone = 0.15f; // Analog stick deadzone threshold [0.01 - 0.50] (default: 0.15)
    bool adaptive_speed = true;    // Distance-adaptive camera movement speed (smooth approach to focus) [Default: true]
    bool test_scene_switching = false; // Run headless dynamic scene switching verification test

    // Camera view overrides (useful for headless testing & reproducible framing)
    std::optional<glm::vec3> camera_pos;
    std::optional<glm::vec3> camera_target;
    std::optional<glm::vec3> camera_up;
    std::optional<float> camera_fov;

    std::string scene_path = "";
    std::string hdri_path = "";
    bool custom_hdri = false; // User explicitly provided --hdri on CLI (do not auto-override during scene changes)
    std::string dump_frame_path = "";
    std::string dump_ui_path = "";
    std::string dump_hdr_path = "";
    std::string dump_stats_path = "";
    bool dump_8bit_png = false; // Save dumped PNG frames as 8-bit instead of default 10/16-bit (conforms to single-negation rule)

    // ML Neural Reconstruction Dataset Capture (Upways PTTD)
    std::string capture_training_data_dir = "";
    uint32_t capture_frames = 0;
    uint32_t capture_reference_spp = 256;
    bool capture_normals = true;
    uint32_t capture_channels = 20; // 16, 19, 20, or 23 (PTTD v3 default)

    // Asset Ingestion & Point Instancing
    float instance_density = 1.0f; // Scale factor for point instancing (0.0 to 1.0, default: 1.0)
    float cull_distance = 0.0f;    // Max distance in meters from camera to cull instances (0 = disabled)

    static Config parse(int argc, char* argv[]);
    static void printUsage(const char* progName);
};

} // namespace pathways
