#include "core/Config.hpp"
#include "core/Logger.hpp"
#include <iostream>
#include <cstring>

namespace pathways {

void Config::printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --headless              Run in headless offscreen mode (no window)\n"
              << "  --fullscreen            Run in fullscreen mode [default]\n"
              << "  --windowed              Run in windowed / non-fullscreen mode\n"
              << "  --width <int>           Viewport width (default: 3840)\n"
              << "  --height <int>          Viewport height (default: 2160)\n"
              << "  --spp <int>             Samples per pixel to accumulate (default: 1)\n"
              << "  --max-bounces <int>     Maximum ray bounces (default: 4)\n"
              << "  --frames <int>          Number of frames to execute (default: 0 = infinite)\n"
              << "  --render-scale <float>  Internal rendering scale (default: 1.0)\n"
              << "  --scene <path>          Path to glTF 2.0 scene (default: procedural Cornell box)\n"
              << "  --hdri <path>           Path to HDR/EXR environment map\n"
              << "  --dump-frame <path.png> Save tonemapped LDR frame to PNG\n"
              << "  --dump-ui <path.png>    Save full window framebuffer with ImGui UI overlay to PNG\n"
              << "  --dump-hdr <path.exr>   Save linear HDR radiance buffer to OpenEXR\n"
              << "  --mgpu                  Enable Multi-GPU mode (default: CheckerboardTile [50/50 balanced load])\n"
              << "  --mgpu-mode <mode>      Multi-GPU mode: 'tile' (Checkerboard [default]), 'interleave' (Interleaved Scanlines), 'sample' (Sample Parallel), 'auto', or 'off'\n"
              << "  --tile-size <int>       Tile size for tile mode: 16, 32, 64, or 128 (default: 64)\n"
              << "  --accum-format <fmt>    HDR Accumulation Format: 'rgba16' (16-bit Half HDR [default]) or 'rgba32' (32-bit Float HDR)\n"
              << "  --double-buffer-shared  Enable double-buffering for inter-GPU shared host memory [default]\n"
              << "  --single-buffer-shared  Disable double-buffering for inter-GPU shared host memory\n"
              << "  --visualize-split       Visualize real-time workload split between Dual GPUs (overlay)\n"
              << "  --no-visualize-split    Disable GPU load split visualization overlay [default]\n"
              << "  --single-gpu            Force single GPU mode (alias for --mgpu-mode off)\n"
              << "  --morton                Enable 2D Morton Z-curve ray indexing for cache locality [default]\n"
              << "  --no-morton             Disable 2D Morton ordering (linear scanline order)\n"
              << "  --benchmark             Enable per-frame latency logging and verification\n"
              << "  --log-interval <float>  Console frame stats log interval in seconds (default: 0 = disabled)\n"
              << "  --hw-rt                 Hardware Ray Tracing is mandatory [default]\n"
              << "  --no-hw-rt              (Deprecated) Software fallback has been removed; ignored\n"
              << "  --no-validation         Disable Vulkan validation layers\n"
              << "  --debug                 Enable verbose debug logging\n"
              << "  -h, --help              Show this help message\n";
}

Config Config::parse(int argc, char* argv[]) {
    Config cfg;

    bool fullscreen_explicit = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--headless") {
            cfg.headless = true;
        } else if (arg == "--fullscreen") {
            cfg.fullscreen = true;
            fullscreen_explicit = true;
        } else if (arg == "--windowed" || arg == "--no-fullscreen") {
            cfg.fullscreen = false;
            fullscreen_explicit = true;
        } else if (arg == "--width" && i + 1 < argc) {
            cfg.width = static_cast<uint32_t>(std::stoul(argv[++i]));
            cfg.custom_resolution = true;
        } else if (arg == "--height" && i + 1 < argc) {
            cfg.height = static_cast<uint32_t>(std::stoul(argv[++i]));
            cfg.custom_resolution = true;
        } else if (arg == "--spp" && i + 1 < argc) {
            cfg.spp = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--max-bounces" && i + 1 < argc) {
            cfg.max_bounces = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if ((arg == "--frames" || arg == "--frame-limit") && i + 1 < argc) {
            cfg.frame_limit = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--render-scale" && i + 1 < argc) {
            cfg.render_scale = std::stof(argv[++i]);
        } else if (arg == "--scene" && i + 1 < argc) {
            cfg.scene_path = argv[++i];
        } else if (arg == "--hdri" && i + 1 < argc) {
            cfg.hdri_path = argv[++i];
        } else if (arg == "--dump-frame" && i + 1 < argc) {
            cfg.dump_frame_path = argv[++i];
        } else if (arg == "--dump-ui" && i + 1 < argc) {
            cfg.dump_ui_path = argv[++i];
        } else if (arg == "--dump-hdr" && i + 1 < argc) {
            cfg.dump_hdr_path = argv[++i];
        } else if (arg == "--dump-stats" && i + 1 < argc) {
            cfg.dump_stats_path = argv[++i];
        } else if (arg == "--gpu" && i + 1 < argc) {
            cfg.gpu_index = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--morton") {
            cfg.enable_morton_order = true;
        } else if (arg == "--no-morton") {
            cfg.enable_morton_order = false;
        } else if (arg == "--mgpu") {
            cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
        } else if (arg == "--single-gpu") {
            cfg.mgpu_mode = MultiGpuMode::Off;
        } else if (arg == "--mgpu-mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "off" || mode == "none") cfg.mgpu_mode = MultiGpuMode::Off;
            else if (mode == "interleave" || mode == "interleaved" || mode == "scanline" || mode == "line") cfg.mgpu_mode = MultiGpuMode::InterleavedScanline;
            else if (mode == "tile" || mode == "split" || mode == "checkerboard") cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
            else if (mode == "sample" || mode == "sample_parallel") cfg.mgpu_mode = MultiGpuMode::SampleParallel;
            else if (mode == "auto") cfg.mgpu_mode = MultiGpuMode::Auto;
            else cfg.mgpu_mode = MultiGpuMode::Off;
        } else if ((arg == "--tile-size" || arg == "--checker-tile-size") && i + 1 < argc) {
            uint32_t sz = static_cast<uint32_t>(std::stoul(argv[++i]));
            if (sz == 16 || sz == 32 || sz == 64 || sz == 128) {
                cfg.tile_size = sz;
            } else {
                Logger::warn("Invalid tile size {} specified. Must be 16, 32, 64, or 128. Defaulting to 64.", sz);
                cfg.tile_size = 64;
            }
        } else if ((arg == "--accum-format" || arg == "--format") && i + 1 < argc) {
            std::string fmt = argv[++i];
            if (fmt == "rgba32" || fmt == "fp32" || fmt == "r32g32b32a32_sfloat" || fmt == "32") {
                cfg.accum_format = AccumFormat::RGBA32_SFLOAT;
            } else {
                cfg.accum_format = AccumFormat::RGBA16_SFLOAT;
            }
        } else if (arg == "--double-buffer-shared") {
            cfg.double_buffered_shared_mem = true;
        } else if (arg == "--single-buffer-shared") {
            cfg.double_buffered_shared_mem = false;
        } else if (arg == "--camera-motion") {
            cfg.camera_motion = true;
        } else if (arg == "--visualize-split" || arg == "--show-split") {
            cfg.visualize_mgpu_split = true;
        } else if (arg == "--no-visualize-split") {
            cfg.visualize_mgpu_split = false;
        } else if (arg == "--log-interval") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.log_interval_sec = std::stof(argv[++i]);
            } else {
                cfg.log_interval_sec = 10.0f;
            }
        } else if (arg == "--benchmark") {
            cfg.benchmark = true;
        } else if (arg == "--test-scene-switching") {
            cfg.test_scene_switching = true;
            cfg.headless = true;
        } else if (arg == "--hw-rt") {
            Logger::info("Command line: Hardware RT is permanently active.");
        } else if (arg == "--no-hw-rt" || arg == "--software-rt") {
            Logger::warn("Command line: Software RT fallback has been removed. Hardware ray tracing is mandatory.");
        } else if (arg == "--no-validation") {
            cfg.validation_layers = false;
        } else if (arg == "--debug") {
            Logger::setLogLevel(LogLevel::Debug);
        } else if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
        } else {
            Logger::warn("Unknown command-line argument: {}", arg);
        }
    }

    if (cfg.custom_resolution && !fullscreen_explicit) {
        cfg.fullscreen = false;
    }

    if (cfg.headless && cfg.frame_limit == 0) {
        // In headless mode, default to 1 frame unless explicitly told to run more
        cfg.frame_limit = 1;
    }

    return cfg;
}

} // namespace pathways
