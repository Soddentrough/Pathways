#include "core/Config.hpp"
#include "core/Logger.hpp"
#include <iostream>
#include <cstring>

namespace pathways {

void Config::printUsage(const char* progName) {
    std::cout << "Usage: " << progName << " [options]\n"
              << "Options:\n"
              << "  --headless              Run in headless offscreen mode (no window)\n"
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
              << "  --dump-stats <path.json>Save benchmark & profiling statistics to JSON\n"
              << "  --gpu <int>             Physical GPU device index (default: 0)\n"
              << "  --mgpu-mode <mode>      Multi-GPU mode: 'sample' (Sample Parallelism across Dual GPUs),\n"
              << "                          'tile' (Split-Frame Tiling), 'dynamic' (Work Queue), or 'off'\n"
              << "  --single-gpu            Force single GPU mode (alias for --mgpu-mode off)\n"
              << "  --pipeline <mode>       Pipeline: 'wavefront' (decomposed DGC compaction, default), 'persistent' (Persistent Wavefront Work Queue), or 'megakernel'\n"
              << "  --morton                Enable 2D Morton Z-curve ray indexing for cache locality [default]\n"
              << "  --no-morton             Disable 2D Morton ordering (linear scanline order)\n"
              << "  --benchmark             Enable per-frame latency logging and verification\n"
              << "  --log-interval <float>  Console frame stats log interval in seconds (default: 0 = disabled)\n"
              << "  --hw-rt                 Enable Hardware Ray Tracing (VK_KHR_ray_query, VK_KHR_acceleration_structure) [default]\n"
              << "  --no-hw-rt              Disable Hardware RT; fallback to compute ALU software loop (LDS/SSBO)\n"
              << "  --no-validation         Disable Vulkan validation layers\n"
              << "  --debug                 Enable verbose debug logging\n"
              << "  -h, --help              Show this help message\n";
}

Config Config::parse(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--headless") {
            cfg.headless = true;
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
        } else if (arg == "--pipeline" && i + 1 < argc) {
            std::string p = argv[++i];
            if (p == "megakernel" || p == "mega") {
                cfg.pipeline_type = PipelineType::Megakernel;
            } else if (p == "persistent" || p == "pwf") {
                cfg.pipeline_type = PipelineType::Persistent;
            } else if (p == "rtp" || p == "raygen" || p == "khr") {
                cfg.pipeline_type = PipelineType::RTP;
            } else {
                cfg.pipeline_type = PipelineType::Wavefront;
            }
        } else if (arg == "--morton") {
            cfg.enable_morton_order = true;
        } else if (arg == "--no-morton") {
            cfg.enable_morton_order = false;
        } else if (arg == "--single-gpu") {
            cfg.mgpu_mode = MultiGpuMode::Off;
        } else if (arg == "--mgpu-mode" && i + 1 < argc) {
            std::string mode = argv[++i];
            if (mode == "off") cfg.mgpu_mode = MultiGpuMode::Off;
            else if (mode == "sample") cfg.mgpu_mode = MultiGpuMode::SampleParallel;
            else if (mode == "tile") cfg.mgpu_mode = MultiGpuMode::CheckerboardTile;
            else if (mode == "dynamic") cfg.mgpu_mode = MultiGpuMode::DynamicWorkQueue;
            else cfg.mgpu_mode = MultiGpuMode::Off;
        } else if (arg == "--log-interval") {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                cfg.log_interval_sec = std::stof(argv[++i]);
            } else {
                cfg.log_interval_sec = 10.0f;
            }
        } else if (arg == "--benchmark") {
            cfg.benchmark = true;
        } else if (arg == "--hw-rt") {
            cfg.enable_hardware_rt = true;
            Logger::info("Command line: Hardware RT requested (VK_KHR_ray_query, VK_KHR_acceleration_structure, VK_KHR_buffer_device_address, VK_KHR_deferred_host_operations)");
        } else if (arg == "--no-hw-rt" || arg == "--software-rt") {
            cfg.enable_hardware_rt = false;
            Logger::info("Command line: Software RT requested. Hardware ray tracing extensions will be bypassed.");
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

    if (cfg.headless && cfg.frame_limit == 0) {
        // In headless mode, default to 1 frame unless explicitly told to run more
        cfg.frame_limit = 1;
    }

    return cfg;
}

} // namespace pathways
