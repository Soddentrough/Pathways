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
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --headless              Run in headless offscreen mode (no window)\n"
              << "  --pipeline <type>       Path tracing pipeline: 'wavefront' (Wavefront Work Lists & DGC [default]) or 'rtp' (KHR Ray Tracing Pipeline)\n"
              << "  --fullscreen            Run in fullscreen mode [default]\n"
              << "  --windowed              Run in windowed / non-fullscreen mode\n"
              << "  -r, --res <preset>      Resolution preset: 1080, 1440, 4k, 5k, 8k, dualup, square, or <W>x<H>\n"
              << "  --width <int>           Viewport width in pixels (default: native display, or 3840 in headless)\n"
              << "  --height <int>          Viewport height in pixels (default: native display, or 2160 in headless)\n"
              << "  --spp <int>             Samples per pixel to accumulate (default: 1)\n"
              << "  --frames <int>          Number of frames to execute (default: 0 = infinite)\n"
              << "  --warmup-frames <int>   Initial frames to exclude from benchmark stats (default: 0)\n"
              << "  --render-scale <float>  Internal rendering scale (default: 1.0)\n"
              << "  --scene <path>          Path to glTF 2.0 scene (default: procedural Cornell box)\n"
              << "  --hdri <path>           Path to HDR/EXR environment map\n"
              << "  --dump-frame <path.png> Save tonemapped LDR frame to PNG\n"
              << "  --dump-ui <path.png>    Save full window framebuffer with ImGui UI overlay to PNG\n"
              << "  --dump-hdr <path.exr>   Save linear HDR radiance buffer to OpenEXR\n"
              << "  --mgpu                  Enable Multi-GPU mode (default: CheckerboardTile [50/50 balanced load])\n"
              << "  --mgpu-mode <mode>      Multi-GPU mode: 'tile' (Checkerboard [default]), 'sample' (Sample Parallel), 'auto', or 'off'\n"
              << "  --tile-size <int>       Tile size for tile mode: 16, 32, 64, or 128 (default: 64)\n"
              << "  --accum-format <fmt>    HDR Accumulation Format: 'rgba16' (16-bit Half HDR [default]) or 'rgba32' (32-bit Float HDR)\n"
              << "  --no-double-buffer      Disable double-buffering for inter-GPU shared host memory\n"
              << "  --visualize-split       Visualize real-time workload split between Dual GPUs (overlay)\n"
              << "  --wavefront-tile <int>  Wavefront cache-resident tile size (0 = full frame, 256 = 256x256, default: 0)\n"
              << "  --wavefront-sort <mode> Wavefront material sorting mode: 'none' [default], 'archetype' (A & B), 'bda' (C), or 'dual' (D)\n"
              << "  --benchmark             Enable per-frame latency logging and verification\n"
              << "  --atrous                Enable A-Trous Wavelet Diffuse Denoiser\n"
              << "  --no-atrous             Disable A-Trous Wavelet Diffuse Denoiser [default: disabled]\n"
              << "  --atrous-passes <int>   Number of A-Trous filter iterations (1..5, default: 3)\n"
               << "  --target-fps <int>      Target frame rate limit (e.g. 30, 60, 90, 120, 240; 0 = uncapped [default])\n"
              << "  --adaptive-spp          Enable dynamic 3-axis sample rate governor\n"
              << "  --no-adaptive-spp       Disable dynamic sample rate governor\n"
              << "  --min-spp <int>         Minimum dynamic SPP floor (default: 1)\n"
              << "  --max-spp <int>         Maximum dynamic SPP ceiling (default: 16)\n"
              << "  --min-bounces <int>     Minimum dynamic bounce floor (default: 2)\n"
              << "  --max-dynamic-bounces <int> Maximum dynamic bounce ceiling (default: 8)\n"
              << "  --log-interval <float>  Console frame stats log interval in seconds (default: 0 = disabled)\n"
              << "  --no-accumulation, --realtime  Disable progressive static frame accumulation (evaluate real-time noise)\n"
              << "  --camera-motion         Simulate continuous camera motion\n"
              << "  --camera <px,py,pz,tx,ty,tz[,fov]> Set camera position, target look-at, and optional FOV\n"
              << "  --camera-pos <x,y,z>    Set camera position (or --cam-pos, space or comma separated)\n"
              << "  --camera-target <x,y,z> Set camera target look-at point (or --cam-target)\n"
              << "  --camera-up <x,y,z>     Set camera world up vector (default: 0,1,0)\n"
              << "  --camera-fov <degrees>  Set camera vertical field of view in degrees (or --fov)\n"
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
        } else if (arg == "--warmup-frames" && i + 1 < argc) {
            cfg.warmup_frames = static_cast<uint32_t>(std::stoul(argv[++i]));
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
            else if (mode == "interleave" || mode == "interleaved" || mode == "scanline" || mode == "line") {
                Logger::info("Interleaved scanline mode deprecated; defaulting to CheckerboardTile.");
                cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
            }
            else if (mode == "tile" || mode == "split" || mode == "checkerboard") cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
            else if (mode == "sample" || mode == "sample_parallel") cfg.mgpu_mode = MultiGpuMode::SampleParallel;
            else if (mode == "auto") cfg.mgpu_mode = MultiGpuMode::Auto;
            else cfg.mgpu_mode = MultiGpuMode::Off;
        } else if (arg == "--shadow-denoiser" || arg == "--denoise-shadows") {
            Logger::info("FidelityFX Shadow Denoiser option is deprecated and has been removed from the active pipeline.");
        } else if (arg == "--taa" || arg == "--no-taa") {
            Logger::info("TAA option is deprecated and has been removed from the active pipeline.");
        } else if (arg == "--taa-alpha" && i + 1 < argc) {
            ++i;
        } else if (arg == "--taa-gamma" && i + 1 < argc) {
            ++i;
        } else if (arg == "--atrous") {
            cfg.enable_atrous = true;
        } else if (arg == "--no-atrous") {
            cfg.enable_atrous = false;
        } else if (arg == "--atrous-passes" && i + 1 < argc) {
            cfg.atrous_passes = static_cast<uint32_t>(std::clamp(std::stoi(argv[++i]), 1, 5));
        } else if (arg.starts_with("--atrous-passes=")) {
            cfg.atrous_passes = static_cast<uint32_t>(std::clamp(std::stoi(arg.substr(arg.find('=') + 1)), 1, 5));
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
        } else if (arg == "--restir-di" || arg == "--restir" || arg == "--no-restir-di" || arg == "--no-restir" ||
                   arg == "--restir-spatial" || arg == "--no-restir-spatial") {
            Logger::info("ReSTIR DI option is deprecated and has been removed from the active pipeline.");
        } else if ((arg == "--restir-spatial-samples" || arg == "--restir-spatial-radius") && i + 1 < argc) {
            ++i;
        } else if (arg.starts_with("--restir-spatial-samples=") || arg.starts_with("--restir-spatial-radius=")) {
            // Ignored deprecated flag
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
        } else if (arg == "--no-indirect" || arg == "--direct-only") {
            cfg.enable_indirect_light = false;
        } else if (arg == "--indirect") {
            cfg.enable_indirect_light = true;
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

    // Scale default spatial search radius for 4K (3840x2160) to maintain wide angular neighbor coverage
    if (cfg.width >= 3840 || cfg.height >= 2160) {
        if (cfg.restir_spatial_radius == 8.0f) {
            cfg.restir_spatial_radius = 16.0f;
        }
    }

    return cfg;
}

} // namespace pathways
