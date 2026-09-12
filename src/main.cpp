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

            // Switch to scene 4: Breakfast Room
            if (!engine.loadScene("scenes/breakfast-room/breakfast_room_extended.glb")) {
                pathways::Logger::error("Test failed: loadScene Breakfast Room failed.");
                return 1;
            }
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched to Breakfast Room and rendered 3 frames cleanly.");

            // Switch to scene 5: Dragon Dispersion
            if (!engine.loadScene("scenes/DragonDispersion.glb")) {
                pathways::Logger::error("Test failed: loadScene Dragon Dispersion failed.");
                return 1;
            }
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched to Dragon Dispersion and rendered 3 frames cleanly.");

            // Switch to scene 6: Car Concept
            if (!engine.loadScene("scenes/CarConcept.glb")) {
                pathways::Logger::error("Test failed: loadScene Car Concept failed.");
                return 1;
            }
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched to Car Concept and rendered 3 frames cleanly.");

            // Switch to scene 7: Cornell Caustic Extended (Tier 1 Research Scene)
            if (!engine.loadScene("scenes/cornell-caustic/cornell_caustic_extended.glb")) {
                pathways::Logger::error("Test failed: loadScene Cornell Caustic failed.");
                return 1;
            }
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched to Cornell Caustic Extended (Single-GPU) and rendered 3 frames cleanly.");

            // Dynamically switch to Dual-GPU Checkerboard Tiling mode
            pathways::Logger::info("Dynamically switching to Dual-GPU Checkerboard Tiling mode on Cornell Caustic scene...");
            engine.setMgpuMode(pathways::MultiGpuMode::CheckerboardTile);
            for (int i = 0; i < 5; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Dual-GPU Checkerboard Tiling rendered 5 frames cleanly on Cornell Caustic scene.");

            // Dynamically switch to Dual-GPU Sample Parallel mode
            pathways::Logger::info("Dynamically switching to Dual-GPU Sample Parallel mode on Cornell Caustic scene...");
            engine.setMgpuMode(pathways::MultiGpuMode::SampleParallel);
            for (int i = 0; i < 5; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Dual-GPU Sample Parallel rendered 5 frames cleanly on Cornell Caustic scene.");

            // Dynamically switch back to Single-GPU mode
            pathways::Logger::info("Dynamically switching back to Single-GPU mode...");
            engine.setMgpuMode(pathways::MultiGpuMode::Off);
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched back to Single-GPU and rendered 3 frames cleanly.");

            engine.printExecutionSummary();
            auto stats = engine.getStats();
            if (stats.validation_errors > 0) {
                pathways::Logger::error("Test failed: Validation errors encountered: {}", stats.validation_errors);
                return 1;
            }
            pathways::Logger::info("[SUCCESS] Dynamic scene & Multi-GPU mode switching test PASSED cleanly with 0 validation errors!");
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
