#include "core/Config.hpp"
#include "core/Logger.hpp"
#include <iostream>
#include <cstring>
#include <cctype>
#include <string_view>

namespace pathways {

namespace {
bool parseResolutionString(std::string_view str, uint32_t& outW, uint32_t& outH) {
    while (!str.empty() && (str.front() == ' ' || str.front() == '\t')) str.remove_prefix(1);
    while (!str.empty() && (str.back() == ' ' || str.back() == '\t')) str.remove_suffix(1);
    if (str.empty()) return false;

    std::string s;
    s.reserve(str.size());
    for (char c : str) s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    if (s == "720" || s == "720p" || s == "hd") {
        outW = 1280; outH = 720;
        return true;
    }
    if (s == "1080" || s == "1080p" || s == "fhd") {
        outW = 1920; outH = 1080;
        return true;
    }
    if (s == "1440" || s == "1440p" || s == "qhd" || s == "2k") {
        outW = 2560; outH = 1440;
        return true;
    }
    if (s == "4k" || s == "4kp" || s == "uhd" || s == "2160" || s == "2160p") {
        outW = 3840; outH = 2160;
        return true;
    }
    if (s == "5k" || s == "5kp" || s == "5k2k" || s == "uw5k") {
        outW = 5120; outH = 2160;
        return true;
    }
    if (s == "8k" || s == "8kp" || s == "4320" || s == "4320p") {
        outW = 7680; outH = 4320;
        return true;
    }
    if (s == "dualup") {
        outW = 1280; outH = 2048;
        return true;
    }
    if (s == "square" || s == "1:1") {
        outW = 1440; outH = 1440;
        return true;
    }
    if (s == "native") {
        outW = 0; outH = 0;
        return true;
    }

    size_t sep = s.find_first_of("x*,");
    if (sep != std::string::npos && sep > 0 && sep + 1 < s.size()) {
        try {
            unsigned long w = std::stoul(s.substr(0, sep));
            unsigned long h = std::stoul(s.substr(sep + 1));
            if (w > 0 && h > 0) {
                outW = static_cast<uint32_t>(w);
                outH = static_cast<uint32_t>(h);
                return true;
            }
        } catch (...) {
            return false;
        }
    }

    return false;
}
} // namespace

void Config::printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --headless              Run in headless offscreen mode (no window)\n"
              << "  --fullscreen            Run in fullscreen mode [default]\n"
              << "  --windowed              Run in windowed / non-fullscreen mode\n"
              << "  -r, --res <preset>      Resolution preset: 1080, 1440, 4k, 5k, 8k, dualup, square, or <W>x<H>\n"
              << "  --width <int>           Viewport width in pixels (default: native display, or 3840 in headless)\n"
              << "  --height <int>          Viewport height in pixels (default: native display, or 2160 in headless)\n"
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
              << "  --no-double-buffer      Disable double-buffering for inter-GPU shared host memory\n"
              << "  --visualize-split       Visualize real-time workload split between Dual GPUs (overlay)\n"
              << "  --benchmark             Enable per-frame latency logging and verification\n"
              << "  --log-interval <float>  Console frame stats log interval in seconds (default: 0 = disabled)\n"
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
        } else if ((arg == "-r" || arg == "--res" || arg == "--resolution") && i + 1 < argc) {
            std::string val = argv[++i];
            uint32_t rw = 0, rh = 0;
            if (parseResolutionString(val, rw, rh)) {
                if (rw == 0 && rh == 0) {
                    cfg.custom_resolution = false;
                } else {
                    cfg.width = rw;
                    cfg.height = rh;
                    cfg.custom_resolution = true;
                }
            } else {
                Logger::warn("Unknown resolution preset '{}'. Expected 1080, 1440, 4k, 5k, 8k, dualup, square, or <W>x<H>.", val);
            }
        } else if (arg.starts_with("--res=") || arg.starts_with("--resolution=") || arg.starts_with("-r=")) {
            size_t eq = arg.find('=');
            std::string val = arg.substr(eq + 1);
            uint32_t rw = 0, rh = 0;
            if (parseResolutionString(val, rw, rh)) {
                if (rw == 0 && rh == 0) {
                    cfg.custom_resolution = false;
                } else {
                    cfg.width = rw;
                    cfg.height = rh;
                    cfg.custom_resolution = true;
                }
            } else {
                Logger::warn("Unknown resolution preset '{}'. Expected 1080, 1440, 4k, 5k, 8k, dualup, square, or <W>x<H>.", val);
            }
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
        } else if (arg == "--mgpu") {
            cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
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
        } else if (arg == "--no-double-buffer" || arg == "--no-double-buffer-shared" || arg == "--single-buffer-shared") {
            cfg.double_buffered_shared_mem = false;
        } else if (arg == "--double-buffer-shared" || arg == "--double-buffer") {
            cfg.double_buffered_shared_mem = true;
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
