#include "core/Config.hpp"
#include "core/Logger.hpp"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cctype>
#include <string_view>
#include <cstdlib>

namespace pathways {

namespace {

static inline void setEnvVar(const char* name, const char* value) {
#if defined(_WIN32)
    _putenv_s(name, value);
#else
    setenv(name, value, 1);
#endif
}
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

std::vector<float> parseNumbers(std::string_view str) {
    std::vector<float> numbers;
    while (!str.empty() && (str.front() == ' ' || str.front() == '\t' || str.front() == '[' || str.front() == '(')) {
        str.remove_prefix(1);
    }
    while (!str.empty() && (str.back() == ' ' || str.back() == '\t' || str.back() == ']' || str.back() == ')')) {
        str.remove_suffix(1);
    }
    if (str.empty()) return numbers;

    std::string s(str);
    for (char& c : s) {
        if (c == ',' || c == ';' || c == '/' || c == '[' || c == ']' || c == '(' || c == ')') {
            c = ' ';
        }
    }

    size_t start = 0;
    while (start < s.size()) {
        while (start < s.size() && (s[start] == ' ' || s[start] == '\t')) ++start;
        if (start >= s.size()) break;
        size_t end = start;
        while (end < s.size() && s[end] != ' ' && s[end] != '\t') ++end;
        try {
            numbers.push_back(std::stof(s.substr(start, end - start)));
        } catch (...) {
            return {};
        }
        start = end;
    }
    return numbers;
}

bool parseVec3(std::string_view str, glm::vec3& outVec) {
    auto nums = parseNumbers(str);
    if (nums.size() == 3) {
        outVec = glm::vec3(nums[0], nums[1], nums[2]);
        return true;
    }
    return false;
}
} // namespace

void Config::printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n\n"
              << "General & Display:\n"
              << "  --headless              Run in headless offscreen mode (no window)\n"
              << "  --fullscreen            Run in fullscreen mode [default]\n"
              << "  --windowed              Run in windowed / non-fullscreen mode\n"
              << "  -r, --res <preset>      Resolution preset: 1080, 1440, 4k, 5k, 8k, dualup, square, or <W>x<H>\n"
              << "  --width <int>           Viewport width in pixels (default: native display, or 3840 in headless)\n"
              << "  --height <int>          Viewport height in pixels (default: native display, or 2160 in headless)\n"
              << "  --render-scale <float>  Internal rendering scale (default: 1.0)\n"
              << "  --no-hdr                Disable HDR display auto-negotiation (force SDR sRGB)\n"
              << "  --hdr-peak <float>      Display peak luminance in nits (default: 1000.0)\n"
              << "  --hdr-white <float>     Reference paper white luminance in nits (default: 200.0)\n\n"
              << "Rendering & Path Tracing:\n"
              << "  --pipeline <type>       Path tracing pipeline: 'wavefront' (Wavefront Work Lists & DGC [default]) or 'rtp' (KHR RTP)\n"
              << "  --spp <int>             Samples per pixel to accumulate (default: 1)\n"
              << "  --max-bounces <int>     Maximum ray bounces / depth (or --bounces, default: 4)\n"
              << "  --scene <path>          Path to glTF 2.0 scene (default: procedural Cornell box)\n"
              << "  --hdri <path>           Path to HDR/EXR environment map\n"
              << "  --accum-format <fmt>    HDR Accumulation Format: 'rgba16' (16-bit Half HDR [default]) or 'rgba32' (32-bit Float HDR)\n"
              << "  --output-format <fmt>   Output backbuffer format: '10bit' (10-bit A2B10G10R10 [default]) or '8bit' (8-bit RGBA8)\n"
              << "  --no-accumulation, --realtime  Disable progressive static frame accumulation (evaluate real-time noise)\n"
              << "  --temporal-accum, --tra Enable motion-vector guided temporal accumulation [default: disabled]\n"
              << "  --bmfr                  Enable experimental Blockwise Multi-Order Feature Regression [default: disabled]\n"
              << "  --denoiser <mode>       Denoising mode: 'none' (Pure MC [default]), 'temporal' (Temporal Accumulation), or 'bmfr'\n"
              << "  --nrc                   Enable Neural Radiance Caching with Wave32 WMMA [default: disabled]\n"
              << "  --nrc-bounce <int>      Path bounce depth where NRC terminates tracing (default: 2)\n"
              << "  --nrc-train-ratio <float> Ratio of paths continuing to ground truth for training (default: 0.03)\n\n"
              << "Frame Pacing & Dynamic Governor:\n"
              << "  --target-fps <int>      Target frame rate limit (e.g. 30, 60, 90, 120, 240; 0 = uncapped [default])\n"
              << "  --adaptive-spp          Enable dynamic 3-axis sample rate governor to track target FPS\n"
              << "  --min-spp <int>         Minimum dynamic SPP floor (default: 1)\n"
              << "  --max-spp <int>         Maximum dynamic SPP ceiling (default: 16)\n"
              << "  --min-bounces <int>     Minimum dynamic bounce floor (default: 2)\n"
              << "  --max-dynamic-bounces <int> Maximum dynamic bounce ceiling (default: 8)\n\n"
              << "Multi-GPU Subsystem:\n"
              << "  --mgpu                  Enable Multi-GPU mode (default: CheckerboardTile [50/50 balanced load])\n"
              << "  --mgpu-mode <mode>      Multi-GPU mode: 'tile' (Checkerboard [default]), 'sample' (Sample Parallel), 'auto', or 'off'\n"
              << "  --mgpu-transfer <mode>  Multi-GPU transfer mode: 'host' (Zero-Copy Host Memory [default]), 'p2p' (Direct BAR), 'staging'\n"
              << "  --tile-size <int>       Tile size for tile mode: 16, 32, 64, or 128 (default: 64)\n"
              << "  --no-double-buffer      Disable double-buffering for inter-GPU shared host memory\n"
              << "  --visualize-split       Visualize real-time workload split between Dual GPUs (overlay)\n\n"
              << "Wavefront Architecture:\n"
              << "  --wavefront-tile <int>  Wavefront cache-resident tile size (0 = full frame, 256 = 256x256, default: 0)\n"
              << "  --wavefront-sort <mode> Wavefront material sorting mode: 'dual' (D) [default], 'none', 'archetype' (A & B), or 'bda' (C)\n"
              << "  --sec-sort <mode>       Secondary ray coherency sort mode: 'none' [default], 'directional' (Option 1 DGC), or 'spatial' (Option 2 Morton)\n"
              << "  --no-streamlined-secondary Disable streamlined secondary bounce shading (keep primary shading math on all bounces)\n"
              << "  --no-distance-clamping  Disable scene-scale intelligent secondary ray distance clamping\n"
              << "  --sec-max-dist <float>  Override maximum secondary ray distance in world units (default: 0 = auto)\n"
              << "  --no-dgc-preprocess     Disable explicit DGC preprocessing and unordered flags (fallback to baseline implicit DGC)\n"
              << "  --no-dgc-batch-preprocess Disable batched DGC preprocessing (fallback to sequential stop-and-wait preprocessing)\n"
              << "  --no-async-preprocess   Disable dedicated async compute queue DGC preprocessing\n"
              << "  --dgc-execset           Enable experimental DGC Execution Sets for material archetypes\n\n"
              << "Camera & Navigation:\n"
              << "  --camera-motion         Simulate continuous camera motion\n"
              << "  --camera <px,py,pz,tx,ty,tz[,fov]> Set camera position, target look-at, and optional FOV\n"
              << "  --camera-pos <x,y,z>    Set camera position (or --cam-pos, space or comma separated)\n"
              << "  --camera-target <x,y,z> Set camera target look-at point (or --cam-target)\n"
              << "  --camera-up <x,y,z>     Set camera world up vector (default: 0,1,0)\n"
              << "  --camera-fov <degrees>  Set camera vertical field of view in degrees (or --fov)\n\n"
              << "Benchmarking & Diagnostics:\n"
              << "  --benchmark             Enable per-frame latency logging and verification\n"
              << "  --frames <int>          Number of frames to execute (default: 0 = infinite)\n"
              << "  --warmup-frames <int>   Initial frames to exclude from benchmark stats (default: 0)\n"
              << "  --log-interval <float>  Console frame stats log interval in seconds (default: 0 = disabled)\n"
              << "  --dump-frame <path.png> Save tonemapped frame to PNG (10/16-bit by default)\n"
              << "  --dump-8bit             Force 8-bit PNG dump instead of default 10/16-bit\n"
              << "  --no-inline-shadows     Disable hybrid inline primary shadows\n"
              << "  --dump-ui <path.png>    Save full window framebuffer with ImGui UI overlay to PNG\n"
              << "  --dump-hdr <path.exr>   Save linear HDR radiance buffer to OpenEXR\n"
              << "  --capture-training-data <dir> Output directory for high-speed raw binary training tensors\n"
              << "  --capture-frames <int>        Number of training sequence frames to capture (default: 60)\n"
              << "  --capture-reference-spp <int> Sample count for stationary ground truth reference (default: 512)\n"
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
        } else if (arg.starts_with("--spp=")) {
            cfg.spp = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
        } else if ((arg == "--max-bounces" || arg == "--bounces") && i + 1 < argc) {
            cfg.max_bounces = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg.starts_with("--max-bounces=")) {
            cfg.max_bounces = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
        } else if (arg.starts_with("--bounces=")) {
            cfg.max_bounces = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
        } else if ((arg == "--frames" || arg == "--frame-limit") && i + 1 < argc) {
            cfg.frame_limit = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--warmup-frames" && i + 1 < argc) {
            cfg.warmup_frames = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--render-scale" && i + 1 < argc) {
            cfg.render_scale = std::stof(argv[++i]);
        } else if (arg.starts_with("--render-scale=")) {
            cfg.render_scale = std::stof(arg.substr(arg.find('=') + 1));
        } else if (arg == "--exposure" && i + 1 < argc) {
            cfg.exposure = std::stof(argv[++i]);
        } else if (arg == "--scene" && i + 1 < argc) {
            cfg.scene_path = argv[++i];
        } else if (arg == "--hdri" && i + 1 < argc) {
            cfg.hdri_path = argv[++i];
        } else if (arg == "--dump-frame" && i + 1 < argc) {
            cfg.dump_frame_path = argv[++i];
        } else if (arg == "--dump-8bit" || arg == "--png-8bit") {
            cfg.dump_8bit_png = true;
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
            else if (mode == "interleave" || mode == "interleaved" || mode == "scanline" || mode == "line") {
                Logger::info("Interleaved scanline mode deprecated; defaulting to CheckerboardTile.");
                cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
            }
            else if (mode == "tile" || mode == "split" || mode == "checkerboard") cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
            else if (mode == "sample" || mode == "sample_parallel") cfg.mgpu_mode = MultiGpuMode::SampleParallel;
            else if (mode == "auto") cfg.mgpu_mode = MultiGpuMode::Auto;
            else cfg.mgpu_mode = MultiGpuMode::Off;
        } else if (arg == "--mgpu-transfer" && i + 1 < argc) {
            std::string tmode = argv[++i];
            if (tmode == "p2p" || tmode == "bar" || tmode == "dma-buf") {
                cfg.mgpu_transfer_mode = Config::MgpuTransferMode::P2P;
            } else if (tmode == "staging" || tmode == "cpu") {
                cfg.mgpu_transfer_mode = Config::MgpuTransferMode::Staging;
            } else {
                cfg.mgpu_transfer_mode = Config::MgpuTransferMode::Host;
            }
        } else if (arg == "--shadow-denoiser" || arg == "--denoise-shadows") {
            Logger::info("FidelityFX Shadow Denoiser option is deprecated and has been removed from the active pipeline.");
        } else if (arg == "--taa" || arg == "--no-taa") {
            Logger::info("TAA option is deprecated and has been removed from the active pipeline.");
        } else if (arg == "--taa-alpha" && i + 1 < argc) {
            ++i;
        } else if (arg == "--taa-gamma" && i + 1 < argc) {
            ++i;
        } else if (arg == "--denoiser" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "bmfr") {
                cfg.enable_bmfr = true;
                cfg.enable_temporal_accum = true;
                cfg.denoiser_mode = DenoiserMode::BMFR;
            } else if (mode == "temporal" || mode == "tra") {
                cfg.enable_bmfr = false;
                cfg.enable_temporal_accum = true;
                cfg.denoiser_mode = DenoiserMode::Temporal;
            } else if (mode == "none" || mode == "off") {
                cfg.enable_bmfr = false;
                cfg.enable_temporal_accum = false;
                cfg.denoiser_mode = DenoiserMode::None;
            }
        } else if (arg.starts_with("--denoiser=")) {
            std::string mode = arg.substr(arg.find('=') + 1);
            if (mode == "bmfr") {
                cfg.enable_bmfr = true;
                cfg.enable_temporal_accum = true;
                cfg.denoiser_mode = DenoiserMode::BMFR;
            } else if (mode == "temporal" || mode == "tra") {
                cfg.enable_bmfr = false;
                cfg.enable_temporal_accum = true;
                cfg.denoiser_mode = DenoiserMode::Temporal;
            } else if (mode == "none" || mode == "off") {
                cfg.enable_bmfr = false;
                cfg.enable_temporal_accum = false;
                cfg.denoiser_mode = DenoiserMode::None;
            }
        } else if (arg == "--bmfr") {
            cfg.enable_bmfr = true;
            cfg.enable_temporal_accum = true;
            cfg.denoiser_mode = DenoiserMode::BMFR;
        } else if (arg == "--temporal-accum" || arg == "--tra") {
            cfg.enable_temporal_accum = true;
            if (cfg.denoiser_mode == DenoiserMode::None) {
                cfg.denoiser_mode = DenoiserMode::Temporal;
            }
        } else if (arg == "--no-temporal-accum") {
            cfg.enable_temporal_accum = false;
            if (cfg.denoiser_mode == DenoiserMode::Temporal) {
                cfg.denoiser_mode = DenoiserMode::None;
            }
        } else if (arg == "--atrous" || arg.starts_with("--atrous")) {
            Logger::warn("A-Trous Wavelet denoiser has been removed. Use --temporal-accum or --bmfr.");
        } else if (arg == "--nrc") {
            cfg.enable_nrc = true;
        } else if (arg == "--nrc-bounce" && i + 1 < argc) {
            cfg.nrc_bounce = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg.starts_with("--nrc-bounce=")) {
            cfg.nrc_bounce = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
        } else if (arg == "--nrc-train-ratio" && i + 1 < argc) {
            cfg.nrc_train_ratio = std::stof(argv[++i]);
        } else if (arg.starts_with("--nrc-train-ratio=")) {
            cfg.nrc_train_ratio = std::stof(arg.substr(arg.find('=') + 1));
        } else if ((arg == "--tile-size" || arg == "--checker-tile-size") && i + 1 < argc) {
            uint32_t sz = static_cast<uint32_t>(std::stoul(argv[++i]));
            if (sz == 16 || sz == 32 || sz == 64 || sz == 128) {
                cfg.tile_size = sz;
            } else {
                Logger::warn("Invalid tile size {} specified. Must be 16, 32, 64, or 128. Defaulting to 64.", sz);
                cfg.tile_size = 64;
            }
        } else if ((arg == "--wavefront-tile" || arg == "--wf-tile" || arg == "--wavefront-tile-size") && i + 1 < argc) {
            cfg.wavefront_tile_size = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg.starts_with("--wavefront-tile=") || arg.starts_with("--wf-tile=") || arg.starts_with("--wavefront-tile-size=")) {
            cfg.wavefront_tile_size = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
        } else if ((arg == "--wavefront-sort" || arg == "--wf-sort" || arg == "--material-sort") && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "archetype" || s == "a" || s == "b" || s == "ab") cfg.wavefront_sort_mode = WavefrontSortMode::Archetype;
            else if (s == "bda" || s == "c") cfg.wavefront_sort_mode = WavefrontSortMode::BDA;
            else if (s == "dual" || s == "d") cfg.wavefront_sort_mode = WavefrontSortMode::Dual;
            else cfg.wavefront_sort_mode = WavefrontSortMode::None;
        } else if (arg.starts_with("--wavefront-sort=") || arg.starts_with("--wf-sort=") || arg.starts_with("--material-sort=")) {
            std::string s = arg.substr(arg.find('=') + 1);
            if (s == "archetype" || s == "a" || s == "b" || s == "ab") cfg.wavefront_sort_mode = WavefrontSortMode::Archetype;
            else if (s == "bda" || s == "c") cfg.wavefront_sort_mode = WavefrontSortMode::BDA;
            else if (s == "dual" || s == "d") cfg.wavefront_sort_mode = WavefrontSortMode::Dual;
            else cfg.wavefront_sort_mode = WavefrontSortMode::None;
        } else if ((arg == "--sec-sort" || arg == "--secondary-sort" || arg == "-ss") && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "directional" || s == "dir" || s == "dgc" || s == "octant" || s == "1") cfg.secondary_sort_mode = SecondarySortMode::DirectionalDGC;
            else if (s == "spatial" || s == "morton" || s == "index" || s == "2") cfg.secondary_sort_mode = SecondarySortMode::SpatialIndex;
            else cfg.secondary_sort_mode = SecondarySortMode::None;
        } else if (arg.starts_with("--sec-sort=") || arg.starts_with("--secondary-sort=") || arg.starts_with("-ss=")) {
            std::string s = arg.substr(arg.find('=') + 1);
            if (s == "directional" || s == "dir" || s == "dgc" || s == "octant" || s == "1") cfg.secondary_sort_mode = SecondarySortMode::DirectionalDGC;
            else if (s == "spatial" || s == "morton" || s == "index" || s == "2") cfg.secondary_sort_mode = SecondarySortMode::SpatialIndex;
            else cfg.secondary_sort_mode = SecondarySortMode::None;
        } else if (arg == "--no-streamlined-secondary" || arg == "--no-secondary-shading-opt") {
            cfg.streamline_secondary_shading = false;
        } else if (arg == "--no-distance-clamping" || arg == "--no-ray-clamping") {
            cfg.distance_clamping = false;
        } else if ((arg == "--sec-max-dist" || arg == "--secondary-max-distance") && i + 1 < argc) {
            cfg.max_secondary_distance = std::stof(argv[++i]);
        } else if (arg.starts_with("--sec-max-dist=") || arg.starts_with("--secondary-max-distance=")) {
            cfg.max_secondary_distance = std::stof(arg.substr(arg.find('=') + 1));
        } else if ((arg == "--accum-format" || arg == "--format") && i + 1 < argc) {
            std::string fmt = argv[++i];
            if (fmt == "rgba32" || fmt == "fp32" || fmt == "r32g32b32a32_sfloat" || fmt == "32") {
                cfg.accum_format = AccumFormat::RGBA32_SFLOAT;
            } else {
                cfg.accum_format = AccumFormat::RGBA16_SFLOAT;
            }
        } else if ((arg == "--output-format" || arg == "--out-format") && i + 1 < argc) {
            std::string fmt = argv[++i];
            if (fmt == "8bit" || fmt == "rgba8" || fmt == "8" || fmt == "rgba8_unorm") {
                cfg.output_format = OutputFormat::RGBA8_UNORM;
            } else {
                cfg.output_format = OutputFormat::A2B10G10R10_UNORM;
            }
        } else if (arg == "--no-dgc-preprocess" || arg == "--no-dgc-tier1") {
            setEnvVar("PATHWAYS_DISABLE_DGC_PREPROCESS", "1");
        } else if (arg == "--no-async-preprocess") {
            cfg.async_dgc_preprocess = false;
            setEnvVar("PATHWAYS_DISABLE_ASYNC_PREPROCESS", "1");
        } else if (arg == "--no-inline-shadows") {
            cfg.inline_primary_shadows = false;
        } else if (arg == "--dgc-execset" || arg == "--dgc-tier2-execset") {
            setEnvVar("PATHWAYS_ENABLE_DGC_EXECSET", "1");
        } else if (arg == "--no-double-buffer" || arg == "--no-double-buffer-shared" || arg == "--single-buffer-shared") {
            cfg.double_buffered_shared_mem = false;
        } else if (arg == "--double-buffer-shared" || arg == "--double-buffer") {
            cfg.double_buffered_shared_mem = true;
        } else if (arg == "--camera-motion") {
            cfg.camera_motion = true;
        } else if ((arg == "--camera" || arg == "-c") && i + 1 < argc) {
            std::string combinedStr = argv[++i];
            auto nums = parseNumbers(combinedStr);
            while (nums.size() < 6 && i + 1 < argc) {
                std::string nextArg = argv[i + 1];
                if (nextArg.starts_with("--") || (nextArg.starts_with("-") && nextArg.size() > 1 && !std::isdigit(static_cast<unsigned char>(nextArg[1])) && nextArg[1] != '.')) {
                    break;
                }
                combinedStr += " " + nextArg;
                nums = parseNumbers(combinedStr);
                ++i;
            }
            if (nums.size() == 6 && i + 1 < argc) {
                std::string nextArg = argv[i + 1];
                if (!nextArg.starts_with("--") && (!nextArg.starts_with("-") || (nextArg.size() > 1 && (std::isdigit(static_cast<unsigned char>(nextArg[1])) || nextArg[1] == '.')))) {
                    auto testNums = parseNumbers(combinedStr + " " + nextArg);
                    if (testNums.size() == 7) {
                        nums = testNums;
                        combinedStr += " " + nextArg;
                        ++i;
                    }
                }
            }
            if (nums.size() >= 6) {
                cfg.camera_pos = glm::vec3(nums[0], nums[1], nums[2]);
                cfg.camera_target = glm::vec3(nums[3], nums[4], nums[5]);
                if (nums.size() == 7) {
                    cfg.camera_fov = nums[6];
                } else if (nums.size() == 9) {
                    cfg.camera_up = glm::vec3(nums[6], nums[7], nums[8]);
                } else if (nums.size() >= 10) {
                    cfg.camera_up = glm::vec3(nums[6], nums[7], nums[8]);
                    cfg.camera_fov = nums[9];
                }
            } else {
                Logger::warn("Invalid --camera argument '{}'. Expected at least 6 values: px,py,pz,tx,ty,tz[,fov]", combinedStr);
            }
        } else if (arg.starts_with("--camera=") || arg.starts_with("-c=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            auto nums = parseNumbers(val);
            if (nums.size() >= 6) {
                cfg.camera_pos = glm::vec3(nums[0], nums[1], nums[2]);
                cfg.camera_target = glm::vec3(nums[3], nums[4], nums[5]);
                if (nums.size() == 7) {
                    cfg.camera_fov = nums[6];
                } else if (nums.size() == 9) {
                    cfg.camera_up = glm::vec3(nums[6], nums[7], nums[8]);
                } else if (nums.size() >= 10) {
                    cfg.camera_up = glm::vec3(nums[6], nums[7], nums[8]);
                    cfg.camera_fov = nums[9];
                }
            } else {
                Logger::warn("Invalid --camera argument '{}'. Expected at least 6 values: px,py,pz,tx,ty,tz[,fov]", val);
            }
        } else if ((arg == "--camera-pos" || arg == "--cam-pos" || arg == "--camera-position" || arg == "--cam-position") && i + 1 < argc) {
            std::string combinedStr = argv[++i];
            glm::vec3 pos;
            if (parseVec3(combinedStr, pos)) {
                cfg.camera_pos = pos;
            } else if (i + 2 < argc) {
                std::string s3 = combinedStr + " " + argv[i + 1] + " " + argv[i + 2];
                if (parseVec3(s3, pos)) {
                    cfg.camera_pos = pos;
                    i += 2;
                } else {
                    Logger::warn("Invalid camera position '{}'. Expected x,y,z", combinedStr);
                }
            } else {
                Logger::warn("Invalid camera position '{}'. Expected x,y,z", combinedStr);
            }
        } else if (arg.starts_with("--camera-pos=") || arg.starts_with("--cam-pos=") ||
                   arg.starts_with("--camera-position=") || arg.starts_with("--cam-position=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            glm::vec3 pos;
            if (parseVec3(val, pos)) {
                cfg.camera_pos = pos;
            } else {
                Logger::warn("Invalid camera position '{}'. Expected x,y,z", val);
            }
        } else if ((arg == "--camera-target" || arg == "--cam-target" || arg == "--camera-lookat" || arg == "--cam-lookat") && i + 1 < argc) {
            std::string combinedStr = argv[++i];
            glm::vec3 target;
            if (parseVec3(combinedStr, target)) {
                cfg.camera_target = target;
            } else if (i + 2 < argc) {
                std::string s3 = combinedStr + " " + argv[i + 1] + " " + argv[i + 2];
                if (parseVec3(s3, target)) {
                    cfg.camera_target = target;
                    i += 2;
                } else {
                    Logger::warn("Invalid camera target '{}'. Expected x,y,z", combinedStr);
                }
            } else {
                Logger::warn("Invalid camera target '{}'. Expected x,y,z", combinedStr);
            }
        } else if (arg.starts_with("--camera-target=") || arg.starts_with("--cam-target=") ||
                   arg.starts_with("--camera-lookat=") || arg.starts_with("--cam-lookat=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            glm::vec3 target;
            if (parseVec3(val, target)) {
                cfg.camera_target = target;
            } else {
                Logger::warn("Invalid camera target '{}'. Expected x,y,z", val);
            }
        } else if ((arg == "--camera-up" || arg == "--cam-up") && i + 1 < argc) {
            std::string combinedStr = argv[++i];
            glm::vec3 up;
            if (parseVec3(combinedStr, up)) {
                cfg.camera_up = up;
            } else if (i + 2 < argc) {
                std::string s3 = combinedStr + " " + argv[i + 1] + " " + argv[i + 2];
                if (parseVec3(s3, up)) {
                    cfg.camera_up = up;
                    i += 2;
                } else {
                    Logger::warn("Invalid camera up vector '{}'. Expected x,y,z", combinedStr);
                }
            } else {
                Logger::warn("Invalid camera up vector '{}'. Expected x,y,z", combinedStr);
            }
        } else if (arg.starts_with("--camera-up=") || arg.starts_with("--cam-up=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            glm::vec3 up;
            if (parseVec3(val, up)) {
                cfg.camera_up = up;
            } else {
                Logger::warn("Invalid camera up vector '{}'. Expected x,y,z", val);
            }
        } else if ((arg == "--camera-fov" || arg == "--cam-fov" || arg == "--fov") && i + 1 < argc) {
            try {
                cfg.camera_fov = std::stof(argv[++i]);
            } catch (...) {
                Logger::warn("Invalid camera fov '{}'. Expected degrees float", argv[i]);
            }
        } else if (arg.starts_with("--camera-fov=") || arg.starts_with("--cam-fov=") || arg.starts_with("--fov=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            try {
                cfg.camera_fov = std::stof(val);
            } catch (...) {
                Logger::warn("Invalid camera fov '{}'. Expected degrees float", val);
            }
        } else if ((arg == "--pipeline" || arg == "-p") && i + 1 < argc) {
            std::string pipeStr = argv[++i];
            std::transform(pipeStr.begin(), pipeStr.end(), pipeStr.begin(), ::tolower);
            if (pipeStr == "rtp" || pipeStr == "khr" || pipeStr == "rtpipeline") {
                cfg.pipeline_type = PipelineType::RTP;
            } else {
                cfg.pipeline_type = PipelineType::Wavefront;
            }
        } else if (arg.starts_with("--pipeline=")) {
            std::string pipeStr = arg.substr(arg.find('=') + 1);
            std::transform(pipeStr.begin(), pipeStr.end(), pipeStr.begin(), ::tolower);
            if (pipeStr == "rtp" || pipeStr == "khr" || pipeStr == "rtpipeline") {
                cfg.pipeline_type = PipelineType::RTP;
            } else {
                cfg.pipeline_type = PipelineType::Wavefront;
            }
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
        } else if (arg == "--target-fps" && i + 1 < argc) {
            cfg.target_fps = static_cast<uint32_t>(std::stoul(argv[++i]));
            cfg.adaptive_spp = (cfg.target_fps > 0);
        } else if (arg.starts_with("--target-fps=")) {
            cfg.target_fps = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
            cfg.adaptive_spp = (cfg.target_fps > 0);
        } else if (arg == "--adaptive-spp") {
            cfg.adaptive_spp = true;
        } else if (arg == "--no-adaptive-spp") {
            cfg.adaptive_spp = false;
        } else if (arg == "--min-spp" && i + 1 < argc) {
            cfg.min_spp = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--max-spp" && i + 1 < argc) {
            cfg.max_spp = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--min-bounces" && i + 1 < argc) {
            cfg.min_bounces = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--max-dynamic-bounces" && i + 1 < argc) {
            cfg.max_dynamic_bounces = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg == "--no-accumulation" || arg == "--realtime") {
            cfg.progressive_accumulation = false;
        } else if (arg == "--accumulation") {
            cfg.progressive_accumulation = true;
        } else if ((arg == "--accum-cutoff" || arg == "--max-accum-frames" || arg == "--accum-limit") && i + 1 < argc) {
            cfg.max_accum_frames = static_cast<uint32_t>(std::stoul(argv[++i]));
        } else if (arg.starts_with("--accum-cutoff=") || arg.starts_with("--max-accum-frames=") || arg.starts_with("--accum-limit=")) {
            size_t eq = arg.find('=');
            cfg.max_accum_frames = static_cast<uint32_t>(std::stoul(arg.substr(eq + 1)));
        } else if (arg == "--no-indirect" || arg == "--direct-only") {
            cfg.enable_indirect_light = false;
        } else if (arg == "--indirect") {
            cfg.enable_indirect_light = true;
        } else if (arg == "--no-hdr") {
            cfg.enable_hdr = false;
        } else if (arg == "--hdr-peak" && i + 1 < argc) {
            cfg.hdr_peak_nits = std::stof(argv[++i]);
        } else if (arg.starts_with("--hdr-peak=")) {
            cfg.hdr_peak_nits = std::stof(arg.substr(arg.find('=') + 1));
        } else if ((arg == "--hdr-white" || arg == "--hdr-paper-white") && i + 1 < argc) {
            cfg.hdr_paper_white_nits = std::stof(argv[++i]);
        } else if (arg.starts_with("--hdr-white=") || arg.starts_with("--hdr-paper-white=")) {
            cfg.hdr_paper_white_nits = std::stof(arg.substr(arg.find('=') + 1));
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
