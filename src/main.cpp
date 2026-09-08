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
        pathways::Logger::info("  Hardware-Accelerated Ray Tracing & Multi-GPU Viewport");
        if (config.custom_resolution || config.headless) {
            pathways::Logger::info("  Target Resolution: {}x{} | SPP: {} | Max Bounces: {}",
                                   config.width, config.height, config.spp, config.max_bounces);
        } else {
            pathways::Logger::info("  Target Resolution: Auto (Native Display) | SPP: {} | Max Bounces: {}",
                                   config.spp, config.max_bounces);
        }
        pathways::Logger::info("  Mode: {} | Target Frame Budget: <8.0 ms",
                               config.headless ? "Headless Testing & Verification" : "Interactive Real-Time Viewport");
        pathways::Logger::info("==========================================================");

        if (config.test_scene_switching) {
            config.headless = true;
            if (config.scene_path.empty()) {
                config.scene_path = "scenes/cornell-box/cornell_box_extended.glb";
            }
            pathways::Engine engine(config);
            const auto& scenes = engine.getAvailableScenes();
            if (scenes.empty()) {
                pathways::Logger::error("Test failed: No scenes discovered.");
                return 1;
            }
            pathways::Logger::info("Testing Dynamic Scene Switching across {} discovered scenes...", scenes.size());

            // Render 3 frames on initial scene
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Initial scene rendered 3 frames cleanly.");

            // Switch to scene 1: Coffee Maker
            if (!engine.loadScene("scenes/coffee-maker/coffee_maker_extended.glb")) {
                pathways::Logger::error("Test failed: loadScene coffee maker failed.");
                return 1;
            }
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched to Coffee Maker and rendered 3 frames cleanly.");

            // Switch to scene 2: Damaged Helmet
            if (!engine.loadScene("scenes/DamagedHelmet.glb")) {
                pathways::Logger::error("Test failed: loadScene DamagedHelmet failed.");
                return 1;
            }
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched to Damaged Helmet and rendered 3 frames cleanly.");

            // Switch to scene 3: Living Room
            if (!engine.loadScene("scenes/living-room/living_room_extended.glb")) {
                pathways::Logger::error("Test failed: loadScene Living Room failed.");
                return 1;
            }
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched to Living Room and rendered 3 frames cleanly.");

            engine.printExecutionSummary();
            auto stats = engine.getStats();
            if (stats.validation_errors > 0) {
                pathways::Logger::error("Test failed: Validation errors encountered: {}", stats.validation_errors);
                return 1;
            }
            pathways::Logger::info("[SUCCESS] Dynamic scene switching test PASSED cleanly with 0 validation errors!");
            return 0;
        }

        pathways::Engine engine(config);
        engine.run();

        engine.printExecutionSummary();
        auto stats = engine.getStats();
        return stats.validation_errors == 0 ? 0 : 1;

    } catch (const std::exception& e) {
        pathways::Logger::error("Fatal Engine Exception: {}", e.what());
        return 1;
    } catch (...) {
        pathways::Logger::error("Fatal Unknown Exception!");
        return 1;
    }
}
