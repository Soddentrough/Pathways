#include "core/Config.hpp"
#include "core/Engine.hpp"
#include "core/Logger.hpp"

#include <iostream>
#include <exception>
#include <filesystem>

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
        pathways::Logger::info("  Mode: {} | Target Frame Budget: <{:.1f} ms",
                               config.headless ? "Headless Testing & Verification" : "Interactive Real-Time Viewport",
                               config.target_frame_time_ms);
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

            // Switch to scene 1: Coffee Maker (OpenUSD)
            if (!engine.loadScene("scenes/coffee-maker/coffee_maker.usda")) {
                pathways::Logger::error("Test failed: loadScene coffee maker failed.");
                return 1;
            }
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched to Coffee Maker (OpenUSD) and rendered 3 frames cleanly.");

            // Switch to scene 2: Damaged Helmet
            if (!engine.loadScene("scenes/DamagedHelmet.glb")) {
                pathways::Logger::error("Test failed: loadScene DamagedHelmet failed.");
                return 1;
            }
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched to Damaged Helmet and rendered 3 frames cleanly.");

            // Switch to scene 2b: Procedural Cyber City
            if (!engine.loadScene("procedural:cyber-city")) {
                pathways::Logger::error("Test failed: loadScene procedural:cyber-city failed.");
                return 1;
            }
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched to Procedural Cyber City and rendered 3 frames cleanly.");

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

            // Switch to scene 6: BMW M6
            if (!engine.loadScene("scenes/bmw-m6/bmw_m6_extended.glb")) {
                pathways::Logger::error("Test failed: loadScene BMW M6 failed.");
                return 1;
            }
            for (int i = 0; i < 3; ++i) engine.renderFrame();
            pathways::Logger::info("[PASS] Switched to BMW M6 and rendered 3 frames cleanly.");

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

            // Switch to scene 8: Point Instanced Med City (OpenUSD Scene)
            if (std::filesystem::exists("scenes/PointInstancedMedCity/PointInstancedMedCity.usd")) {
                if (!engine.loadScene("scenes/PointInstancedMedCity/PointInstancedMedCity.usd")) {
                    pathways::Logger::error("Test failed: loadScene PointInstancedMedCity USD failed.");
                    return 1;
                }
                for (int i = 0; i < 3; ++i) engine.renderFrame();
                pathways::Logger::info("[PASS] Switched to Point Instanced Med City (OpenUSD) and rendered 3 frames cleanly.");
            }

            // Switch to scene 9: Buick Riviera (OpenUSD Scene)
            std::string buickPath = "scenes/BuickRiviera/BuickRiviera.usdc";
            if (std::filesystem::exists(buickPath)) {
                if (!engine.loadScene(buickPath)) {
                    pathways::Logger::error("Test failed: loadScene Buick Riviera USD failed.");
                    return 1;
                }
                for (int i = 0; i < 3; ++i) engine.renderFrame();
                pathways::Logger::info("[PASS] Switched to Buick Riviera (OpenUSD) and rendered 3 frames cleanly.");

                // Dynamically switch to Dual-GPU Checkerboard on Buick Riviera
                pathways::Logger::info("Switching to Dual-GPU Checkerboard on Buick Riviera...");
                engine.setMgpuMode(pathways::MultiGpuMode::CheckerboardTile);
                for (int i = 0; i < 5; ++i) engine.renderFrame();
                pathways::Logger::info("[PASS] Dual-GPU Checkerboard rendered 5 frames on Buick Riviera cleanly.");

                // Now switch to another scene (e.g. Damaged Helmet) WHILE Multi-GPU is active!
                pathways::Logger::info("Switching scene to Damaged Helmet WHILE Dual-GPU is active...");
                if (!engine.loadScene("scenes/DamagedHelmet.glb")) {
                    pathways::Logger::error("Test failed: loadScene DamagedHelmet while MGPU active failed.");
                    return 1;
                }
                for (int i = 0; i < 3; ++i) engine.renderFrame();
                pathways::Logger::info("[PASS] Switched to Damaged Helmet while Dual-GPU active.");

                // Now switch BACK to Buick Riviera WHILE Multi-GPU is active!
                pathways::Logger::info("Switching scene back to Buick Riviera WHILE Dual-GPU is active...");
                if (!engine.loadScene(buickPath)) {
                    pathways::Logger::error("Test failed: loadScene Buick Riviera while MGPU active failed.");
                    return 1;
                }
                for (int i = 0; i < 3; ++i) engine.renderFrame();
                pathways::Logger::info("[PASS] Switched back to Buick Riviera while Dual-GPU active.");
                engine.setMgpuMode(pathways::MultiGpuMode::Off);
            }

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
