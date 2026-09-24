#include "core/Config.hpp"
#include "core/Logger.hpp"
#include <algorithm>
#include <iostream>
#include <cstring>
#include <cctype>
#include <string_view>
#include <cstdlib>
#include <filesystem>

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

bool parseUpscaleRatioOrResolution(std::string_view val, float& outScale, uint32_t& outW, uint32_t& outH) {
    while (!val.empty() && (val.front() == ' ' || val.front() == '\t')) val.remove_prefix(1);
    while (!val.empty() && (val.back() == ' ' || val.back() == '\t')) val.remove_suffix(1);
    if (val.empty()) return false;

    std::string s;
    s.reserve(val.size());
    for (char c : val) s.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    // Standard presets / ratios
    if (s == "native" || s == "1x" || s == "1:1") {
        outScale = 1.0f;
        outW = 0; outH = 0;
        return true;
    }
    if (s == "quality" || s == "q") {
        outScale = 1.0f / 1.5f; // 1.5x upscaling (e.g. 1440p -> 4K, 0.666667f)
        outW = 0; outH = 0;
        return true;
    }
    if (s == "balanced" || s == "bal") {
        outScale = 1.0f / 1.7f; // ~1.7x upscaling (~0.588235f)
        outW = 0; outH = 0;
        return true;
    }
    if (s == "performance" || s == "perf" || s == "p" || s == "2x") {
        outScale = 1.0f / 2.0f; // 2.0x upscaling (e.g. 1080p -> 4K, 0.5f)
        outW = 0; outH = 0;
        return true;
    }
    if (s == "ultra-performance" || s == "ultra_performance" || s == "ultra" || s == "up" || s == "3x" || s == "ultra performance") {
        outScale = 1.0f / 3.0f; // 3.0x upscaling (e.g. 720p -> 4K, 0.333333f)
        outW = 0; outH = 0;
        return true;
    }

    // Check for arbitrary resolution string (e.g. 1080p, 1440p, 720p, 1920x1080, 1280x720, etc.)
    uint32_t rw = 0, rh = 0;
    if (parseResolutionString(val, rw, rh) && rw > 0 && rh > 0) {
        outScale = 0.0f;
        outW = rw;
        outH = rh;
        return true;
    }

    // Check for arbitrary numeric float / ratio (e.g. 0.75, 0.5, 0.6667, 1.5, 2.0)
    try {
        size_t idx = 0;
        float scale = std::stof(std::string(val), &idx);
        if (idx > 0 && scale > 0.05f && scale <= 4.0f) {
            if (scale > 1.0f) {
                scale = 1.0f / scale;
            }
            outScale = scale;
            outW = 0; outH = 0;
            return true;
        }
    } catch (...) {}

    return false;
}

bool applyScaler(Config& cfg, std::string_view algo, std::string_view param = {}) {
    std::string a;
    a.reserve(algo.size());
    for (char c : algo) a.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(c))));

    if (a == "upways" || a == "upways-sr" || a == "upways_sr" || a == "upways4" || a == "kpn") {
        cfg.upscaler_mode = UpscalerMode::Upways;
        cfg.upways_superres = true;
        cfg.denoiser_mode = DenoiserMode::Upways;
        if (cfg.render_scale >= 1.0f && cfg.render_width == 0) cfg.render_scale = 0.6667f; // Default Quality (1.5x)
    } else if (a == "fsr" || a == "fsr3" || a == "fsr3.1") {
        cfg.upscaler_mode = UpscalerMode::FSR3;
        if (cfg.render_scale >= 1.0f && cfg.render_width == 0) cfg.render_scale = 0.6667f; // Default Quality (1.5x)
    } else if (a == "fsr1" || a == "cas" || a == "spatial") {
        cfg.upscaler_mode = UpscalerMode::FSR1;
        if (cfg.render_scale >= 1.0f && cfg.render_width == 0) cfg.render_scale = 0.6667f; // Default Quality (1.5x)
    } else if (a == "none" || a == "off" || a == "disable") {
        cfg.upscaler_mode = UpscalerMode::None;
        cfg.upways_superres = false;
        cfg.render_scale = 1.0f;
        cfg.render_width = 0;
        cfg.render_height = 0;
        return true;
    } else {
        Logger::warn("Unknown scaler mode '{}'. Available: upways, fsr, fsr1, none", algo);
        return false;
    }

    if (!param.empty()) {
        float scale = 0.0f;
        uint32_t rw = 0, rh = 0;
        if (parseUpscaleRatioOrResolution(param, scale, rw, rh)) {
            if (rw > 0 && rh > 0) {
                cfg.render_width = rw;
                cfg.render_height = rh;
                cfg.render_scale = 0.0f;
            } else {
                cfg.render_scale = scale;
                cfg.render_width = 0;
                cfg.render_height = 0;
                if (scale >= 0.999f) {
                    cfg.upways_superres = false;
                    if (cfg.upscaler_mode == UpscalerMode::Upways) {
                        cfg.upscaler_mode = UpscalerMode::None;
                    }
                } else if (cfg.upscaler_mode == UpscalerMode::Upways) {
                    cfg.upways_superres = true;
                }
            }
        } else {
            Logger::warn("Unknown scaler preset / resolution '{}'. Expected quality, balanced, performance, ultra, native, or resolution (e.g. 1080, 1440, 1920x1080)", param);
        }
    }
    return true;
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
              << "  --no-hdr                Disable HDR display auto-negotiation (force SDR sRGB)\n"
              << "  --hdr-peak <float>      Display peak luminance in nits (default: 1000.0)\n"
              << "  --hdr-white <float>     Reference paper white luminance in nits (default: 200.0)\n\n"
              << "Rendering & Path Tracing:\n"
              << "  --pipeline <type>       Path tracing pipeline: 'wavefront' (Wavefront Ray Queues & DGC [default]) or 'rtp' (KHR RTP)\n"
              << "  --spp <int>             Samples per pixel to accumulate (default: 1)\n"
              << "  --max-bounces <int>     Maximum ray bounces / depth (or --bounces, default: 4)\n"
              << "  --scene <path>          Path to glTF 2.0 scene (default: procedural Cornell box)\n"
              << "  --hdri <path>           Path to HDR/EXR environment map\n"
              << "  --no-accumulation, --realtime  Disable progressive static frame accumulation (evaluate real-time noise)\n"
              << "  --denoiser <mode>       Denoising mode: 'none' (Pure MC [default]), 'upways' (Wave32 WMMA)\n"
              << "  --scaler <mode> [ratio|WxH] Consolidated upscaler configuration: 'upways', 'fsr'/'fsr3', 'fsr1', 'none'.\n"
              << "                          Presets: 'native', 'quality', 'balanced', 'performance', 'ultra', or internal resolution (e.g. 1080, 1440, 1920x1080, 0.75)\n"
              << "                          Examples: --res 4k --scaler upways quality | --res 4k --scaler fsr 1080\n"
              << "  --upscaler-sharpening   Enable additional RCAS sharpening pass [default: disabled]\n"
              << "  --upscaler-sharpness <f> RCAS contrast-adaptive sharpness factor [0.0 - 1.0] (default: 0.0)\n"
              << "  --render-scale <float>  Continuous internal render scale factor (e.g. 0.6667 for 1.5x, 0.5 for 2.0x)\n"
              << "  --upways                Enable Upways Neural Denoising with Wave32 WMMA (native resolution)\n"
              << "  --preset <preset|WxH>   Scaling ratio preset: native, quality, balanced, performance, ultra, or <W>x<H>\n"
              << "  --upways-weights <path> Path to Upways weights binary (default: data/models/upways_weights.bin, or embedded fallback)\n"
              << "  --light-tree            Enable Hierarchical Light Tree importance sampling for many-light scenes [default: disabled]\n"
              << "  --nrc                   Enable Neural Radiance Caching with Wave32 WMMA [default: disabled]\n"
              << "  --nrc-bounce <int>      Path bounce depth where NRC terminates tracing (default: 2)\n"
              << "  --nrc-train-ratio <float> Ratio of paths continuing to ground truth for training (default: 0.03)\n"
              << "  --caustics              Enable real-time forward ray-traced caustics [default: disabled]\n"
              << "  --caustic-photons <int> Number of caustic photons traced per frame (default: 1048576)\n"
              << "  --restir                Enable ReSTIR spatio-temporal reservoir resampling [default: disabled]\n"
              << "  --restir-m-cap <int>    Temporal history M-cap for ReSTIR (default: 30)\n\n"
              << "Frame Pacing & Dynamic Governor:\n"
              << "  --target-fps <int>      Target frame rate limit (e.g. 30, 60, 90, 120, 240; 0 = uncapped [default])\n"
              << "  --target-frame-time <float> Target frame time budget in ms (default: 8.3)\n"
              << "  --adaptive-spp          Enable dynamic 3-axis sample rate governor to track target FPS\n"
              << "  --min-spp <int>         Minimum dynamic SPP floor (default: 1)\n"
              << "  --max-spp <int>         Maximum dynamic SPP ceiling (default: 16)\n"
              << "  --min-bounces <int>     Minimum dynamic bounce floor (default: 2)\n"
              << "  --max-dynamic-bounces <int> Maximum dynamic bounce ceiling (default: 8)\n\n"
              << "Multi-GPU Subsystem:\n"
              << "  --mgpu                  Enable Multi-GPU mode (default: CheckerboardTile [50/50 balanced load])\n"
              << "  --mgpu-mode <mode>      Multi-GPU mode: 'tile' (Checkerboard [default]), 'sample' (Sample Parallel), 'auto', or 'off'\n"
              << "  --mgpu-transfer <mode>  Multi-GPU transfer mode: 'host' (Zero-Copy Host Memory [default]), 'p2p' (Direct BAR)\n"
              << "  --tile-size <int>       Tile size for tile mode: 16, 32, 64, or 128 (default: 64)\n"
              << "  --no-double-buffer      Disable double-buffering for inter-GPU shared host memory\n"
              << "  --visualize-split       Visualize real-time workload split between Dual GPUs (overlay)\n\n"
              << "Wavefront Architecture:\n"
              << "  --wavefront-sort <mode> Wavefront material sorting mode: 'dual' (D) [default], 'none', or 'archetype' (A & B)\n"
              << "  --use-morton            Enable 2D Morton Z-curve mapping for wavefront classification (default: disabled / linear raster)\n"
              << "  --sec-sort <mode>       Secondary ray coherency mode: 'direct'/'coherent' (Xiang 2023, K=4) [default], 'coherent-k8' (K=8), 'none', or 'directional' (Option 1 DGC)\n"
              << "  --no-streamlined-secondary Disable streamlined secondary bounce shading (keep primary shading math on all bounces)\n"
              << "  --no-distance-clamping  Disable scene-scale intelligent secondary ray distance clamping\n"
              << "  --sec-max-dist <float>  Override maximum secondary ray distance in world units (default: 0 = auto)\n"
              << "  --indirect-clamp <float> Maximum indirect / secondary bounce radiance luminance (default: 35.0, 0 = disabled)\n"
              << "  --no-dgc-preprocess     Disable explicit DGC preprocessing and unordered flags (fallback to baseline implicit DGC)\n"
              << "  --no-dgc-batch-preprocess Disable batched DGC preprocessing (fallback to sequential stop-and-wait preprocessing)\n"
              << "  --batches <int|auto>    Number of coarse 2D batches / tiles (default: auto, 1 = monolithic, alias: --macro-tiles)\n"
              << "  --macro-tiles <int|auto> Alias for --batches\n"
              << "  --batch-size <int|auto> Coarse batch pixel budget (e.g. 1000000, 2000000; default: auto)\n"
              << "  --dgc-execset           Enable experimental DGC Execution Sets for material archetypes.\n"
              << "                            What it is: Uses VK_EXT_device_generated_commands Indirect Execution\n"
              << "                            Sets (VkIndirectExecutionSetEXT) to dynamically bind specialized\n"
              << "                            compute material pipelines on the GPU via indirect token streams\n"
              << "                            in a single vkCmdExecuteGeneratedCommandsEXT call.\n"
              << "                            Needs: Driver support for compute indirect execution set pipeline switching.\n"
              << "                            Why disabled: Current Vulkan drivers (including Mesa RADV 26.x) fail to switch\n"
              << "                            compute pipelines dynamically via indirect execution set tokens, running\n"
              << "                            only the initial pipeline and dropping secondary rays/reflections.\n"
              << "                            The default multi-dispatch indirect path is already 100% GPU-driven,\n"
              << "                            skips empty material queues with zero wave launches, and is fully correct.\n"
              << "  --no-dgc-execset        Explicitly disable DGC Execution Sets (enforce default multi-dispatch indirect)\n\n"
              << "Camera & Navigation:\n"
              << "  --adaptive-speed        Enable distance-adaptive camera speed (smooth approach) [default: enabled]\n"
              << "  --no-adaptive-speed     Disable distance-adaptive camera speed (constant velocity)\n"
              << "  --camera-motion         Simulate continuous camera motion\n"
              << "  --gamepad-deadzone <float> Analog stick deadzone threshold [0.01 - 0.50] (default: 0.15)\n"
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
              << "  --no-inline-shadows     Disable hybrid inline hardware shadow queries\n"
              << "  --capture-training-data <dir> Save Upways neural reconstruction dataset to directory\n"
              << "  --capture-frames <int>  Number of continuous sequence frames to capture for ML dataset\n"
              << "  --capture-reference-spp <int> Accumulated SPP for ground truth reference (default: 1 for noisy input)\n"
              << "  --capture-channels <int> Number of channels: 16, 19, or 20 (default: 20 PTTD v2)\n"
              << "  --no-capture-normals    Legacy 16-channel export without surface normals\n"
              << "  --no-validation         Disable Vulkan validation layers\n"
              << "  --debug                 Enable verbose debug logging\n"
              << "  -h, --help              Show this help message\n";
}

Config Config::parse(int argc, char* argv[]) {
    Config cfg;

    bool fullscreen_explicit = false;
    bool frame_time_explicit = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--headless") {
            cfg.headless = true;
            continue;
        }
        if (arg == "--fullscreen") {
            cfg.fullscreen = true;
            fullscreen_explicit = true;
            continue;
        }
        if (arg == "--windowed" || arg == "--no-fullscreen") {
            cfg.fullscreen = false;
            fullscreen_explicit = true;
            continue;
        }
        if ((arg == "-r" || arg == "--res" || arg == "--resolution") && i + 1 < argc) {
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
            continue;
        }
        if (arg.starts_with("--res=") || arg.starts_with("--resolution=") || arg.starts_with("-r=")) {
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
            continue;
        }
        if (arg == "--width" && i + 1 < argc) {
            cfg.width = static_cast<uint32_t>(std::stoul(argv[++i]));
            cfg.custom_resolution = true;
            continue;
        }
        if (arg == "--height" && i + 1 < argc) {
            cfg.height = static_cast<uint32_t>(std::stoul(argv[++i]));
            cfg.custom_resolution = true;
            continue;
        }
        if (arg == "--spp" && i + 1 < argc) {
            cfg.spp = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg.starts_with("--spp=")) {
            cfg.spp = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
            continue;
        }
        if ((arg == "--max-bounces" || arg == "--bounces") && i + 1 < argc) {
            cfg.max_bounces = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg.starts_with("--max-bounces=")) {
            cfg.max_bounces = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
            continue;
        }
        if (arg.starts_with("--bounces=")) {
            cfg.max_bounces = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
            continue;
        }
        if ((arg == "--frames" || arg == "--frame-limit") && i + 1 < argc) {
            cfg.frame_limit = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--warmup-frames" && i + 1 < argc) {
            cfg.warmup_frames = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--render-scale" && i + 1 < argc) {
            cfg.render_scale = std::stof(argv[++i]);
            continue;
        }
        if (arg.starts_with("--render-scale=")) {
            cfg.render_scale = std::stof(arg.substr(arg.find('=') + 1));
            continue;
        }
        if (arg == "--exposure" && i + 1 < argc) {
            cfg.exposure = std::stof(argv[++i]);
            continue;
        }
        if (arg == "--scene" && i + 1 < argc) {
            cfg.scene_path = argv[++i];
            continue;
        }
        if (arg == "--hdri" && i + 1 < argc) {
            cfg.hdri_path = argv[++i];
            cfg.custom_hdri = true;
            continue;
        }
        if (arg == "--dump-frame" && i + 1 < argc) {
            cfg.dump_frame_path = argv[++i];
            continue;
        }
        if (arg == "--dump-8bit" || arg == "--png-8bit") {
            cfg.dump_8bit_png = true;
            continue;
        }
        if (arg == "--dump-ui" && i + 1 < argc) {
            cfg.dump_ui_path = argv[++i];
            continue;
        }
        if (arg == "--dump-hdr" && i + 1 < argc) {
            cfg.dump_hdr_path = argv[++i];
            continue;
        }
        if (arg == "--dump-stats" && i + 1 < argc) {
            cfg.dump_stats_path = argv[++i];
            continue;
        }
        if ((arg == "--capture-training-data" || arg == "--capture-data") && i + 1 < argc) {
            cfg.capture_training_data_dir = argv[++i];
            continue;
        }
        if (arg == "--capture-frames" && i + 1 < argc) {
            cfg.capture_frames = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--capture-reference-spp" && i + 1 < argc) {
            cfg.capture_reference_spp = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--capture-channels" && i + 1 < argc) {
            cfg.capture_channels = static_cast<uint32_t>(std::stoul(argv[++i]));
            cfg.capture_normals = (cfg.capture_channels >= 19);
            continue;
        }
        if (arg == "--no-capture-normals") {
            cfg.capture_normals = false;
            cfg.capture_channels = 16;
            continue;
        }
        if (arg == "--gpu" && i + 1 < argc) {
            cfg.gpu_index = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--mgpu") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string mode = argv[++i];
                if (mode == "off" || mode == "none") cfg.mgpu_mode = MultiGpuMode::Off;
                else if (mode == "interleave" || mode == "interleaved" || mode == "scanline" || mode == "line") {
                    Logger::info("Interleaved scanline mode deprecated; defaulting to CheckerboardTile.");
                    cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
                    cfg.mgpu_upscale_mode = MgpuUpscaleMode::PostMerge;
                }
                else if (mode == "tile" || mode == "split" || mode == "checkerboard") {
                    cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
                    cfg.mgpu_upscale_mode = MgpuUpscaleMode::PostMerge;
                }
                else if (mode == "sample" || mode == "sample_parallel") {
                    cfg.mgpu_mode = MultiGpuMode::SampleParallel;
                    cfg.mgpu_upscale_mode = MgpuUpscaleMode::SampleBlend;
                }
                else if (mode == "auto") cfg.mgpu_mode = MultiGpuMode::Auto;
                else cfg.mgpu_mode = MultiGpuMode::Off;
            } else {
                cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
                cfg.mgpu_upscale_mode = MgpuUpscaleMode::PostMerge;
            }
            continue;
        }
        if (arg.starts_with("--mgpu=")) {
            std::string mode = arg.substr(7);
            if (mode == "off" || mode == "none") cfg.mgpu_mode = MultiGpuMode::Off;
            else if (mode == "interleave" || mode == "interleaved" || mode == "scanline" || mode == "line") {
                Logger::info("Interleaved scanline mode deprecated; defaulting to CheckerboardTile.");
                cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
            }
            else if (mode == "tile" || mode == "split" || mode == "checkerboard") {
                cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
                cfg.mgpu_upscale_mode = MgpuUpscaleMode::PostMerge;
            }
            else if (mode == "sample" || mode == "sample_parallel") {
                cfg.mgpu_mode = MultiGpuMode::SampleParallel;
                cfg.mgpu_upscale_mode = MgpuUpscaleMode::SampleBlend;
            }
            else if (mode == "auto") cfg.mgpu_mode = MultiGpuMode::Auto;
            else cfg.mgpu_mode = MultiGpuMode::Off;
            continue;
        }
        if (arg == "--mgpu-mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "off" || mode == "none") cfg.mgpu_mode = MultiGpuMode::Off;
            else if (mode == "interleave" || mode == "interleaved" || mode == "scanline" || mode == "line") {
                Logger::info("Interleaved scanline mode deprecated; defaulting to CheckerboardTile.");
                cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
            }
            else if (mode == "tile" || mode == "split" || mode == "checkerboard") {
                cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
                cfg.mgpu_upscale_mode = MgpuUpscaleMode::PostMerge;
            }
            else if (mode == "sample" || mode == "sample_parallel") {
                cfg.mgpu_mode = MultiGpuMode::SampleParallel;
                cfg.mgpu_upscale_mode = MgpuUpscaleMode::SampleBlend;
            }
            else if (mode == "auto") cfg.mgpu_mode = MultiGpuMode::Auto;
            else cfg.mgpu_mode = MultiGpuMode::Off;
            continue;
        }
        if (arg.starts_with("--mgpu-mode=")) {
            std::string mode = arg.substr(12);
            if (mode == "off" || mode == "none") cfg.mgpu_mode = MultiGpuMode::Off;
            else if (mode == "interleave" || mode == "interleaved" || mode == "scanline" || mode == "line") {
                Logger::info("Interleaved scanline mode deprecated; defaulting to CheckerboardTile.");
                cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
            }
            else if (mode == "tile" || mode == "split" || mode == "checkerboard") {
                cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
                cfg.mgpu_upscale_mode = MgpuUpscaleMode::PostMerge;
            }
            else if (mode == "sample" || mode == "sample_parallel") {
                cfg.mgpu_mode = MultiGpuMode::SampleParallel;
                cfg.mgpu_upscale_mode = MgpuUpscaleMode::SampleBlend;
            }
            else if (mode == "auto") cfg.mgpu_mode = MultiGpuMode::Auto;
            else cfg.mgpu_mode = MultiGpuMode::Off;
            continue;
        }
        if (arg == "--mgpu-transfer" && i + 1 < argc) {
            std::string tmode = argv[++i];
            if (tmode == "p2p" || tmode == "bar" || tmode == "dma-buf") {
                cfg.mgpu_transfer_mode = Config::MgpuTransferMode::P2P;
            } else if (tmode == "staging" || tmode == "cpu") {
                throw std::runtime_error("CPU staging transfer mode has been eliminated under Vulkan 1.4 baseline. Slower workarounds are not supported.");
            } else {
                cfg.mgpu_transfer_mode = Config::MgpuTransferMode::Host;
            }
            continue;
        }
        if (arg.starts_with("--mgpu-transfer=")) {
            std::string tmode = arg.substr(16);
            if (tmode == "p2p" || tmode == "bar" || tmode == "dma-buf") {
                cfg.mgpu_transfer_mode = Config::MgpuTransferMode::P2P;
            } else if (tmode == "staging" || tmode == "cpu") {
                throw std::runtime_error("CPU staging transfer mode has been eliminated under Vulkan 1.4 baseline. Slower workarounds are not supported.");
            } else {
                cfg.mgpu_transfer_mode = Config::MgpuTransferMode::Host;
            }
            continue;
        }
        if ((arg == "--macro-tile" || arg == "--macro-tile-size") && i + 1 < argc) {
            cfg.macro_tile_size = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg.starts_with("--macro-tile=") || arg.starts_with("--macro-tile-size=")) {
            cfg.macro_tile_size = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
            continue;
        }
        if ((arg == "--batches" || arg == "--macro-tiles") && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "auto") {
                cfg.batch_count = 0;
            } else {
                cfg.batch_count = static_cast<uint32_t>(std::stoul(val));
            }
            continue;
        }
        if (arg.starts_with("--batches=") || arg.starts_with("--macro-tiles=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            if (val == "auto") {
                cfg.batch_count = 0;
            } else {
                cfg.batch_count = static_cast<uint32_t>(std::stoul(val));
            }
            continue;
        }
        if ((arg == "--batch-size" || arg == "--batch-pixels") && i + 1 < argc) {
            std::string val = argv[++i];
            if (val == "auto") {
                cfg.batch_pixels = 0;
            } else {
                cfg.batch_pixels = static_cast<uint32_t>(std::stoul(val));
            }
            continue;
        }
        if (arg.starts_with("--batch-size=") || arg.starts_with("--batch-pixels=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            if (val == "auto") {
                cfg.batch_pixels = 0;
            } else {
                cfg.batch_pixels = static_cast<uint32_t>(std::stoul(val));
            }
            continue;
        }
        if (arg == "--denoiser" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "upways") {
                cfg.upways_superres = false;
                cfg.denoiser_mode = DenoiserMode::Upways;
            } else if (mode == "upways_sr" || mode == "upways-sr" || mode == "upways_2x") {
                cfg.upways_superres = true;
                cfg.denoiser_mode = DenoiserMode::Upways;
            } else if (mode == "none" || mode == "off") {
                cfg.denoiser_mode = DenoiserMode::None;
            }
            continue;
        }
        if (arg.starts_with("--denoiser=")) {
            std::string mode = arg.substr(arg.find('=') + 1);
            if (mode == "upways") {
                cfg.upways_superres = false;
                cfg.denoiser_mode = DenoiserMode::Upways;
            } else if (mode == "upways_sr" || mode == "upways-sr" || mode == "upways_2x") {
                cfg.upways_superres = true;
                cfg.denoiser_mode = DenoiserMode::Upways;
            } else if (mode == "none" || mode == "off") {
                cfg.denoiser_mode = DenoiserMode::None;
            }
            continue;
        }
        // Consolidated Scaler / Upscaler CLI flag:
        //   --scaler <upways|fsr|fsr1|none> [preset|WxH]
        // Examples:
        //   --res 4k --scaler upways quality
        //   --res 4k --scaler fsr quality
        //   --res 4k --scaler upways 1080
        //   --res 4k --scaler fsr 1080
        if ((arg == "--scaler" || arg == "--upscaler") && i + 1 < argc) {
            std::string algo = argv[++i];
            std::string param;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                std::string testParam = argv[i + 1];
                if (testParam == "ultra" && i + 2 < argc && std::string_view(argv[i + 2]) == "performance") {
                    testParam = "ultra-performance";
                    i++;
                }
                float dummyScale = 0.0f;
                uint32_t dummyW = 0, dummyH = 0;
                if (parseUpscaleRatioOrResolution(testParam, dummyScale, dummyW, dummyH)) {
                    param = testParam;
                    ++i;
                }
            }
            applyScaler(cfg, algo, param);
            continue;
        }
        if (arg.starts_with("--scaler=") || arg.starts_with("--upscaler=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            std::string algo, param;
            size_t sep = val.find_first_of(",: =");
            if (sep != std::string::npos) {
                algo = val.substr(0, sep);
                param = val.substr(sep + 1);
            } else {
                algo = val;
                if (i + 1 < argc && argv[i + 1][0] != '-') {
                    std::string testParam = argv[i + 1];
                    float dummyScale = 0.0f;
                    uint32_t dummyW = 0, dummyH = 0;
                    if (parseUpscaleRatioOrResolution(testParam, dummyScale, dummyW, dummyH)) {
                        param = testParam;
                        ++i;
                    }
                }
            }
            applyScaler(cfg, algo, param);
            continue;
        }

        // Preset ratio / internal resolution override:
        if ((arg == "--preset" || arg == "--upscaler-preset" || arg == "--scale-preset" || arg == "--upscaler-quality") && i + 1 < argc) {
            std::string param = argv[++i];
            if (param == "ultra" && i + 1 < argc && std::string_view(argv[i + 1]) == "performance") {
                param = "ultra-performance";
                i++;
            }
            float scale = 0.0f;
            uint32_t rw = 0, rh = 0;
            if (parseUpscaleRatioOrResolution(param, scale, rw, rh)) {
                if (rw > 0 && rh > 0) {
                    cfg.render_width = rw;
                    cfg.render_height = rh;
                    cfg.render_scale = 0.0f;
                } else {
                    cfg.render_scale = scale;
                    cfg.render_width = 0;
                    cfg.render_height = 0;
                }
            }
            continue;
        }

        // Standalone shortcuts:
        if (arg == "--upways") {
            cfg.denoiser_mode = DenoiserMode::Upways;
            continue;
        }
        if (arg == "--upways-sr" || arg == "--upways-superres") {
            applyScaler(cfg, "upways");
            continue;
        }
        if (arg == "--fsr" || arg == "--fsr3" || arg == "--fsr3.1") {
            applyScaler(cfg, "fsr");
            continue;
        }
        if (arg == "--upscaler-sharpening") {
            cfg.upscaler_sharpening = true;
            if (cfg.upscaler_sharpness <= 0.0f) {
                cfg.upscaler_sharpness = 0.5f;
            }
            continue;
        }
        if (arg == "--upscaler-sharpness" && i + 1 < argc) {
            cfg.upscaler_sharpness = std::clamp(std::stof(argv[++i]), 0.0f, 1.0f);
            cfg.upscaler_sharpening = (cfg.upscaler_sharpness > 0.0f);
            continue;
        }
        if (arg.starts_with("--upscaler-sharpness=")) {
            cfg.upscaler_sharpness = std::clamp(std::stof(arg.substr(arg.find('=') + 1)), 0.0f, 1.0f);
            cfg.upscaler_sharpening = (cfg.upscaler_sharpness > 0.0f);
            continue;
        }
        if (arg == "--upways-weights" && i + 1 < argc) {
            cfg.upways_weights_path = argv[++i];
            continue;
        }
        if (arg.starts_with("--upways-weights=")) {
            cfg.upways_weights_path = arg.substr(arg.find('=') + 1);
            continue;
        }
        if (arg == "--light-tree") {
            cfg.enable_light_tree = true;
            continue;
        }
        if (arg == "--no-light-tree") {
            cfg.enable_light_tree = false;
            continue;
        }
        if (arg == "--nrc") {
            cfg.enable_nrc = true;
            continue;
        }
        if (arg == "--nrc-bounce" && i + 1 < argc) {
            cfg.nrc_bounce = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg.starts_with("--nrc-bounce=")) {
            cfg.nrc_bounce = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
            continue;
        }
        if (arg == "--nrc-train-ratio" && i + 1 < argc) {
            cfg.nrc_train_ratio = std::stof(argv[++i]);
            continue;
        }
        if (arg.starts_with("--nrc-train-ratio=")) {
            cfg.nrc_train_ratio = std::stof(arg.substr(arg.find('=') + 1));
            continue;
        }
        if (arg == "--caustics") {
            cfg.enable_caustics = true;
            continue;
        }
        if (arg == "--caustic-photons" && i + 1 < argc) {
            cfg.caustic_photons = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg.starts_with("--caustic-photons=")) {
            cfg.caustic_photons = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
            continue;
        }
        if (arg == "--restir-pt" || arg == "--restir-di" || arg == "--restir") {
            cfg.enable_restir_di = true;
            continue;
        }
        if (arg == "--restir-m-cap" && i + 1 < argc) {
            cfg.restir_di_m_cap = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg.starts_with("--restir-m-cap=")) {
            cfg.restir_di_m_cap = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
            continue;
        }
        if ((arg == "--tile-size" || arg == "--checker-tile-size") && i + 1 < argc) {
            uint32_t sz = static_cast<uint32_t>(std::stoul(argv[++i]));
            if (sz == 16 || sz == 32 || sz == 64 || sz == 128) {
                cfg.tile_size = sz;
            } else {
                Logger::warn("Invalid tile size {} specified. Must be 16, 32, 64, or 128. Defaulting to 64.", sz);
                cfg.tile_size = 64;
            }
            continue;
        }
        if ((arg == "--wavefront-sort" || arg == "--wf-sort" || arg == "--material-sort") && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "archetype" || s == "a" || s == "b" || s == "ab") cfg.wavefront_sort_mode = WavefrontSortMode::Archetype;
            else if (s == "dual" || s == "d") cfg.wavefront_sort_mode = WavefrontSortMode::Dual;
            else cfg.wavefront_sort_mode = WavefrontSortMode::None;
            continue;
        }
        if (arg.starts_with("--wavefront-sort=") || arg.starts_with("--wf-sort=") || arg.starts_with("--material-sort=")) {
            std::string s = arg.substr(arg.find('=') + 1);
            if (s == "archetype" || s == "a" || s == "b" || s == "ab") cfg.wavefront_sort_mode = WavefrontSortMode::Archetype;
            else if (s == "dual" || s == "d") cfg.wavefront_sort_mode = WavefrontSortMode::Dual;
            else cfg.wavefront_sort_mode = WavefrontSortMode::None;
            continue;
        }
        if (arg == "--use-morton" || arg == "--morton") {
            cfg.use_morton = true;
            continue;
        }
        if ((arg == "--sec-sort" || arg == "--secondary-sort" || arg == "-ss") && i + 1 < argc) {
            std::string s = argv[++i];
            if (s == "directional" || s == "dir" || s == "dgc" || s == "octant" || s == "1") cfg.secondary_sort_mode = SecondarySortMode::DirectionalDGC;
            else if (s == "coherent" || s == "direct" || s == "direct-coherent" || s == "xiang" || s == "2") cfg.secondary_sort_mode = SecondarySortMode::DirectCoherent;
            else if (s == "coherent-k8" || s == "direct-k8" || s == "3") cfg.secondary_sort_mode = SecondarySortMode::DirectCoherentK8;
            else cfg.secondary_sort_mode = SecondarySortMode::None;
            continue;
        }
        if (arg.starts_with("--sec-sort=") || arg.starts_with("--secondary-sort=") || arg.starts_with("-ss=")) {
            std::string s = arg.substr(arg.find('=') + 1);
            if (s == "directional" || s == "dir" || s == "dgc" || s == "octant" || s == "1") cfg.secondary_sort_mode = SecondarySortMode::DirectionalDGC;
            else if (s == "coherent" || s == "direct" || s == "direct-coherent" || s == "xiang" || s == "2") cfg.secondary_sort_mode = SecondarySortMode::DirectCoherent;
            else if (s == "coherent-k8" || s == "direct-k8" || s == "3") cfg.secondary_sort_mode = SecondarySortMode::DirectCoherentK8;
            else cfg.secondary_sort_mode = SecondarySortMode::None;
            continue;
        }
        if (arg == "--no-streamlined-secondary" || arg == "--no-secondary-shading-opt") {
            cfg.streamline_secondary_shading = false;
            continue;
        }
        if (arg == "--no-distance-clamping" || arg == "--no-ray-clamping") {
            cfg.distance_clamping = false;
            continue;
        }
        if ((arg == "--sec-max-dist" || arg == "--secondary-max-distance") && i + 1 < argc) {
            cfg.max_secondary_distance = std::stof(argv[++i]);
            continue;
        }
        if (arg.starts_with("--sec-max-dist=") || arg.starts_with("--secondary-max-distance=")) {
            cfg.max_secondary_distance = std::stof(arg.substr(arg.find('=') + 1));
            continue;
        }
        if ((arg == "--indirect-clamp" || arg == "--sec-clamp") && i + 1 < argc) {
            cfg.indirect_clamp = std::max(0.0f, std::stof(argv[++i]));
            continue;
        }
        if (arg.starts_with("--indirect-clamp=") || arg.starts_with("--sec-clamp=")) {
            cfg.indirect_clamp = std::max(0.0f, std::stof(arg.substr(arg.find('=') + 1)));
            continue;
        }
        if ((arg == "--accum-format" || arg == "--format") && i + 1 < argc) {
            std::string fmt = argv[++i];
            if (fmt == "rgba32" || fmt == "fp32" || fmt == "r32g32b32a32_sfloat" || fmt == "32") {
                cfg.accum_format = AccumFormat::RGBA32_SFLOAT;
            } else {
                cfg.accum_format = AccumFormat::RGBA16_SFLOAT;
            }
            continue;
        }
        if (arg == "--no-dgc-preprocess" || arg == "--no-dgc-tier1") {
            cfg.dgc_preprocess = false;
            setEnvVar("PATHWAYS_DISABLE_DGC_PREPROCESS", "1");
            continue;
        }
        if (arg == "--no-dgc-batch-preprocess" || arg == "--no-dgc-tier2-batch") {
            cfg.dgc_batch_preprocess = false;
            setEnvVar("PATHWAYS_DISABLE_DGC_BATCH_PREPROCESS", "1");
            continue;
        }
        if (arg == "--no-inline-shadows") {
            cfg.inline_primary_shadows = false;
            continue;
        }
        if (arg == "--dgc-execset" || arg == "--dgc-tier2-execset") {
            setEnvVar("PATHWAYS_ENABLE_DGC_EXECSET", "1");
            setEnvVar("PATHWAYS_ENABLE_MATERIAL_DGC", "1");
            Logger::warn("--dgc-execset enabled: Experimental compute execution sets active. Note: drivers such as Mesa RADV may fail to switch compute pipelines dynamically via execution set tokens.");
            continue;
        }
        if (arg == "--no-dgc-execset" || arg == "--no-dgc-tier2-execset") {
            setEnvVar("PATHWAYS_DISABLE_DGC_EXECSET", "1");
            setEnvVar("PATHWAYS_DISABLE_MATERIAL_DGC", "1");
            continue;
        }
        if (arg == "--no-double-buffer" || arg == "--no-double-buffer-shared" || arg == "--single-buffer-shared") {
            cfg.double_buffered_shared_mem = false;
            continue;
        }
        if (arg == "--camera-motion") {
            cfg.camera_motion = true;
            continue;
        }
        if (arg == "--adaptive-speed" || arg == "--distance-adaptive-speed") {
            cfg.adaptive_speed = true;
            continue;
        }
        if (arg == "--no-adaptive-speed" || arg == "--no-distance-adaptive-speed") {
            cfg.adaptive_speed = false;
            continue;
        }
        if (arg == "--gamepad-deadzone" && i + 1 < argc) {
            cfg.gamepad_deadzone = std::clamp(std::stof(argv[++i]), 0.01f, 0.50f);
            continue;
        }
        if (arg.starts_with("--gamepad-deadzone=")) {
            cfg.gamepad_deadzone = std::clamp(std::stof(arg.substr(arg.find('=') + 1)), 0.01f, 0.50f);
            continue;
        }
        if ((arg == "--camera" || arg == "-c") && i + 1 < argc) {
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
            continue;
        }
        if (arg.starts_with("--camera=") || arg.starts_with("-c=")) {
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
            continue;
        }
        if ((arg == "--camera-pos" || arg == "--cam-pos" || arg == "--camera-position" || arg == "--cam-position") && i + 1 < argc) {
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
            continue;
        }
        if (arg.starts_with("--camera-pos=") || arg.starts_with("--cam-pos=") ||
                   arg.starts_with("--camera-position=") || arg.starts_with("--cam-position=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            glm::vec3 pos;
            if (parseVec3(val, pos)) {
                cfg.camera_pos = pos;
            } else {
                Logger::warn("Invalid camera position '{}'. Expected x,y,z", val);
            }
            continue;
        }
        if ((arg == "--camera-target" || arg == "--cam-target" || arg == "--camera-lookat" || arg == "--cam-lookat") && i + 1 < argc) {
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
            continue;
        }
        if (arg.starts_with("--camera-target=") || arg.starts_with("--cam-target=") ||
                   arg.starts_with("--camera-lookat=") || arg.starts_with("--cam-lookat=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            glm::vec3 target;
            if (parseVec3(val, target)) {
                cfg.camera_target = target;
            } else {
                Logger::warn("Invalid camera target '{}'. Expected x,y,z", val);
            }
            continue;
        }
        if ((arg == "--camera-up" || arg == "--cam-up") && i + 1 < argc) {
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
            continue;
        }
        if (arg.starts_with("--camera-up=") || arg.starts_with("--cam-up=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            glm::vec3 up;
            if (parseVec3(val, up)) {
                cfg.camera_up = up;
            } else {
                Logger::warn("Invalid camera up vector '{}'. Expected x,y,z", val);
            }
            continue;
        }
        if ((arg == "--camera-fov" || arg == "--cam-fov" || arg == "--fov") && i + 1 < argc) {
            try {
                cfg.camera_fov = std::stof(argv[++i]);
            } catch (...) {
                Logger::warn("Invalid camera fov '{}'. Expected degrees float", argv[i]);
            }
            continue;
        }
        if (arg.starts_with("--camera-fov=") || arg.starts_with("--cam-fov=") || arg.starts_with("--fov=")) {
            std::string val = arg.substr(arg.find('=') + 1);
            try {
                cfg.camera_fov = std::stof(val);
            } catch (...) {
                Logger::warn("Invalid camera fov '{}'. Expected degrees float", val);
            }
            continue;
        }
        if ((arg == "--pipeline" || arg == "-p") && i + 1 < argc) {
            std::string pipeStr = argv[++i];
            std::transform(pipeStr.begin(), pipeStr.end(), pipeStr.begin(), ::tolower);
            if (pipeStr == "rtp" || pipeStr == "khr" || pipeStr == "rtpipeline") {
                cfg.pipeline_type = PipelineType::RTP;
            } else {
                cfg.pipeline_type = PipelineType::Wavefront;
            }
            continue;
        }
        if (arg.starts_with("--pipeline=")) {
            std::string pipeStr = arg.substr(arg.find('=') + 1);
            std::transform(pipeStr.begin(), pipeStr.end(), pipeStr.begin(), ::tolower);
            if (pipeStr == "rtp" || pipeStr == "khr" || pipeStr == "rtpipeline") {
                cfg.pipeline_type = PipelineType::RTP;
            } else {
                cfg.pipeline_type = PipelineType::Wavefront;
            }
            continue;
        }
        if (arg == "--visualize-split" || arg == "--show-split") {
            cfg.visualize_mgpu_split = true;
            continue;
        }
        if (arg == "--log-interval") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.log_interval_sec = std::stof(argv[++i]);
            } else {
                cfg.log_interval_sec = 10.0f;
            }
            continue;
        }
        if (arg == "--benchmark") {
            cfg.benchmark = true;
            continue;
        }
        if (arg == "--test-scene-switching") {
            cfg.test_scene_switching = true;
            cfg.headless = true;
            continue;
        }
        if (arg == "--target-fps" && i + 1 < argc) {
            cfg.target_fps = static_cast<uint32_t>(std::stoul(argv[++i]));
            cfg.adaptive_spp = (cfg.target_fps > 0);
            if (!frame_time_explicit && cfg.target_fps > 0) {
                cfg.target_frame_time_ms = 1000.0f / static_cast<float>(cfg.target_fps);
            }
            continue;
        }
        if (arg.starts_with("--target-fps=")) {
            cfg.target_fps = static_cast<uint32_t>(std::stoul(arg.substr(arg.find('=') + 1)));
            cfg.adaptive_spp = (cfg.target_fps > 0);
            if (!frame_time_explicit && cfg.target_fps > 0) {
                cfg.target_frame_time_ms = 1000.0f / static_cast<float>(cfg.target_fps);
            }
            continue;
        }
        if ((arg == "--target-frame-time" || arg == "--frame-budget") && i + 1 < argc) {
            cfg.target_frame_time_ms = std::stof(argv[++i]);
            frame_time_explicit = true;
            continue;
        }
        if (arg.starts_with("--target-frame-time=") || arg.starts_with("--frame-budget=")) {
            cfg.target_frame_time_ms = std::stof(arg.substr(arg.find('=') + 1));
            frame_time_explicit = true;
            continue;
        }
        if (arg == "--adaptive-spp") {
            cfg.adaptive_spp = true;
            continue;
        }
        if (arg == "--min-spp" && i + 1 < argc) {
            cfg.min_spp = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--max-spp" && i + 1 < argc) {
            cfg.max_spp = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--min-bounces" && i + 1 < argc) {
            cfg.min_bounces = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--max-dynamic-bounces" && i + 1 < argc) {
            cfg.max_dynamic_bounces = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--no-accumulation" || arg == "--realtime") {
            cfg.progressive_accumulation = false;
            continue;
        }
        if ((arg == "--accum-cutoff" || arg == "--max-accum-frames" || arg == "--accum-limit") && i + 1 < argc) {
            cfg.max_accum_frames = static_cast<uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg.starts_with("--accum-cutoff=") || arg.starts_with("--max-accum-frames=") || arg.starts_with("--accum-limit=")) {
            size_t eq = arg.find('=');
            cfg.max_accum_frames = static_cast<uint32_t>(std::stoul(arg.substr(eq + 1)));
            continue;
        }
        if (arg == "--no-indirect" || arg == "--direct-only") {
            cfg.enable_indirect_light = false;
            continue;
        }
        if (arg == "--no-hdr") {
            cfg.enable_hdr = false;
            continue;
        }
        if (arg == "--hdr-peak" && i + 1 < argc) {
            cfg.hdr_peak_nits = std::stof(argv[++i]);
            cfg.custom_hdr_peak = true;
            continue;
        }
        if (arg.starts_with("--hdr-peak=")) {
            cfg.hdr_peak_nits = std::stof(arg.substr(arg.find('=') + 1));
            cfg.custom_hdr_peak = true;
            continue;
        }
        if ((arg == "--hdr-white" || arg == "--hdr-paper-white") && i + 1 < argc) {
            cfg.hdr_paper_white_nits = std::stof(argv[++i]);
            continue;
        }
        if (arg.starts_with("--hdr-white=") || arg.starts_with("--hdr-paper-white=")) {
            cfg.hdr_paper_white_nits = std::stof(arg.substr(arg.find('=') + 1));
            continue;
        }
        if (arg == "--no-validation") {
            cfg.validation_layers = false;
            continue;
        }
        if (arg == "--debug") {
            Logger::setLogLevel(LogLevel::Debug);
            continue;
        }
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            std::exit(0);
            continue;
        }

        Logger::warn("Unknown command-line argument: {}", arg);
    }

    if (cfg.custom_resolution && !fullscreen_explicit) {
        cfg.fullscreen = false;
    }

    if (cfg.render_width > 0 && cfg.render_height > 0) {
        if (cfg.render_width < cfg.width || cfg.render_height < cfg.height) {
            cfg.render_scale = static_cast<float>(cfg.render_width) / static_cast<float>(cfg.width);
            if (cfg.denoiser_mode == DenoiserMode::Upways && cfg.upscaler_mode == UpscalerMode::None) {
                cfg.upways_superres = true;
                cfg.upscaler_mode = UpscalerMode::Upways;
            }
        } else {
            cfg.render_scale = 1.0f;
            if (cfg.upscaler_mode == UpscalerMode::Upways) {
                cfg.upways_superres = false;
                cfg.upscaler_mode = UpscalerMode::None;
            }
        }
    }

    if (!cfg.capture_training_data_dir.empty()) {
        cfg.headless = true;
        if (cfg.capture_frames == 0) {
            cfg.capture_frames = 20;
        }
        cfg.frame_limit = cfg.capture_frames * (cfg.capture_reference_spp > 0 ? cfg.capture_reference_spp : 1);
    } else if (cfg.headless && cfg.frame_limit == 0) {
        // In headless mode, default to 1 frame unless explicitly told to run more
        cfg.frame_limit = 1;
    }

    // Auto-detect co-located scene HDRI environment map if not explicitly specified
    if (cfg.hdri_path.empty() && !cfg.scene_path.empty()) {
        std::filesystem::path sp(cfg.scene_path);
        std::filesystem::path sceneDir = sp.has_parent_path() ? sp.parent_path() : std::filesystem::current_path();

        std::vector<std::filesystem::path> hdriCandidates = {
            sceneDir / "textures and hdri" / "golden_gate_hills_2k.exr",
            sceneDir / "textures" / "studio_small_08_4k.exr",
            sceneDir / "textures" / "studio_small_08_4k.hdr",
            sceneDir / ".." / "textures" / "studio_small_08_4k.exr",
            sceneDir / "textures and hdri" / "golden_gate_hills_2k.hdr",
        };
        for (const auto& cand : hdriCandidates) {
            std::error_code ec;
            if (std::filesystem::exists(cand, ec)) {
                cfg.hdri_path = cand.string();
                Logger::info("Config: Auto-detected scene HDRI environment map: '{}'", cfg.hdri_path);
                break;
            }
        }
    }

    return cfg;
}

} // namespace pathways
