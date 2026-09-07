#include "core/Config.hpp"
#include "core/Engine.hpp"
#include "core/Logger.hpp"

#include <iostream>
#include <exception>

int main(int argc, char* argv[]) {
    try {
        pathways::Config config = pathways::Config::parse(argc, argv);

        pathways::Logger::info("==========================================================");
        pathways::Logger::info("  Pathways: Pure Vulkan 1.4 Real-Time Path Tracer Engine");
        pathways::Logger::info("  Target Hardware: AMD RDNA4 (Dual Radeon AI PRO R9700)");
        pathways::Logger::info("  Resolution: {}x{} | SPP: {} | Max Bounces: {}",
                               config.width, config.height, config.spp, config.max_bounces);
        pathways::Logger::info("  Mode: {} | Target Frame Budget: <8.0 ms",
                               config.headless ? "Headless Testing & Verification" : "Interactive Real-Time Viewport");
        pathways::Logger::info("==========================================================");

        pathways::Engine engine(config);
        engine.run();

        auto stats = engine.getStats();
        pathways::Logger::info("----------------------------------------------------------");
        pathways::Logger::info("  Execution Summary:");
        pathways::Logger::info("  Rendered Frames: {}", stats.total_frames);
        pathways::Logger::info("  Average Frame Time: {:.3f} ms ({:.1f} FPS)", stats.avg_frame_time_ms, stats.avg_fps);
        pathways::Logger::info("  Ray Throughput: {:.2e} rays/sec", stats.rays_per_second);
        pathways::Logger::info("  Validation Errors: {}", stats.validation_errors);
        pathways::Logger::info("  Sub-8ms Target Status: {}", stats.target_achieved ? "\033[32mACHIEVED\033[0m" : "\033[33mEXCEEDED\033[0m");
        pathways::Logger::info("==========================================================");

        return stats.validation_errors == 0 ? 0 : 1;

    } catch (const std::exception& e) {
        pathways::Logger::error("Fatal Engine Exception: {}", e.what());
        return 1;
    } catch (...) {
        pathways::Logger::error("Fatal Unknown Exception!");
        return 1;
    }
}
