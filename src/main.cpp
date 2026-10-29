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

            // Switch to scene 6b: Veach Ajar
            if (std::filesystem::exists("scenes/veach-ajar/veach_ajar_extended.glb")) {
                if (!engine.loadScene("scenes/veach-ajar/veach_ajar_extended.glb")) {
                    pathways::Logger::error("Test failed: loadScene Veach Ajar failed.");
                    return 1;
                }
                for (int i = 0; i < 3; ++i) engine.renderFrame();
                pathways::Logger::info("[PASS] Switched to Veach Ajar and rendered 3 frames cleanly.");
            }

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

            // Dynamically switch wavefront batch partitioning modes across all configurations
            pathways::Logger::info("Dynamically testing wavefront batching mode changes (1, 2, 4, 8, 16, Auto)...");
            const uint32_t testBatchCounts[] = { 1, 2, 4, 8, 16, 0 };
            for (uint32_t batchCount : testBatchCounts) {
                pathways::Logger::info("  Switching to batch_count = {}...", batchCount);
                engine.setBatchCount(batchCount);
                for (int i = 0; i < 4; ++i) engine.renderFrame();
            }
            pathways::Logger::info("[PASS] Dynamic wavefront batching mode switching rendered all configurations cleanly.");

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

        if (config.test_dynamic_options) {
            config.headless = true;
            pathways::Engine engine(config);
            auto& cfg = engine.getMutableConfig();

            auto stepAndVerify = [&](const char* stepName, int frames,
                                     const std::string& expectedUpscaler,
                                     const std::string& expectedDenoiser) -> bool {
                engine.requestAccumulationReset();
                for (int i = 0; i < frames; ++i) {
                    engine.renderFrame();
                }
                auto st = engine.getStats();
                if (st.validation_errors > 0) {
                    pathways::Logger::error("[FAIL] {} produced {} Vulkan validation error(s)!", stepName, st.validation_errors);
                    return false;
                }
                if (!expectedUpscaler.empty() && st.upscaler_mode_str != expectedUpscaler) {
                    pathways::Logger::error("[FAIL] {} expected upscaler_mode='{}', got '{}'",
                                            stepName, expectedUpscaler, st.upscaler_mode_str);
                    return false;
                }
                if (!expectedDenoiser.empty() && st.denoiser_mode_str != expectedDenoiser) {
                    pathways::Logger::error("[FAIL] {} expected denoiser_mode='{}', got '{}'",
                                            stepName, expectedDenoiser, st.denoiser_mode_str);
                    return false;
                }
                pathways::Logger::info("[PASS] {} -> Upscaler='{}' ({}x{} -> {}x{}), Denoiser='{}', NRC={}, ReSTIR={}, Caustics={}, ValErrors=0",
                                       stepName, st.upscaler_mode_str, cfg.getRenderWidth(), cfg.getRenderHeight(),
                                       cfg.width, cfg.height, st.denoiser_mode_str,
                                       st.enable_nrc ? "ON" : "OFF",
                                       cfg.enable_restir ? "ON" : "OFF",
                                       st.enable_caustics ? "ON" : "OFF");
                return true;
            };

            const auto initStats = engine.getStats();
            const std::string expectedFsr3 = initStats.supports_fsr3 ? "FSR 3.1" : "None";
            const std::string expectedFsr4 = initStats.supports_fsr4 ? "FSR 4.1 ML" : "None";
            const std::string expectedRayRegen = initStats.supports_fsr_rr ? "AMD FSR Ray Regeneration 1.2.1" : "None";

            // 1. Baseline Native 1.0x + Off (MC)
            cfg.upscaler_mode = pathways::UpscalerMode::None;
            cfg.render_scale = 1.0f;
            cfg.denoiser_mode = pathways::DenoiserMode::None;
            if (!stepAndVerify("1. Native 1.0x + Off (MC)", 4, "None", "None")) return 1;

            // 2. Switch to FSR 3.1 across all 4 Quality Presets (0.67x, 0.59x, 0.50x, 0.33x)
            cfg.upscaler_mode = pathways::UpscalerMode::FSR;
            cfg.fsr_version = pathways::FsrVersion::Fsr3;
            cfg.render_scale = 0.6666667f;
            if (!stepAndVerify("2a. FSR 3.1 Quality (0.67x)", 4, expectedFsr3, "None")) return 1;

            if (initStats.supports_fsr3) {
                cfg.render_scale = 0.5882353f;
                if (!stepAndVerify("2b. FSR 3.1 Balanced (0.59x)", 4, expectedFsr3, "None")) return 1;

                cfg.render_scale = 0.5f;
                if (!stepAndVerify("2c. FSR 3.1 Performance (0.50x)", 4, expectedFsr3, "None")) return 1;

                cfg.render_scale = 0.3333333f;
                if (!stepAndVerify("2d. FSR 3.1 UltraPerf (0.33x)", 4, expectedFsr3, "None")) return 1;
            }

            // 3. Request FSR 4.1 ML at runtime (verifies strict capability gating: never falls back to FSR 3.1)
            cfg.upscaler_mode = pathways::UpscalerMode::FSR;
            cfg.fsr_version = pathways::FsrVersion::Fsr4;
            cfg.render_scale = 0.6666667f;
            if (!stepAndVerify("3. FSR 4.1 ML Request (Runtime Gating)", 4, expectedFsr4, "None")) return 1;

            // 4. Switch to Upways SR and Gateway SR, then back to FSR 3.1 and Native 1.0x
            if (initStats.has_cooperative_matrix) {
                cfg.upscaler_mode = pathways::UpscalerMode::Upways;
                cfg.upways_superres = true;
                cfg.gateway_superres = false;
                cfg.render_scale = 0.6666667f;
                if (!stepAndVerify("4a. Upways SR (0.67x)", 4, "Upways", "None")) return 1;

                cfg.upscaler_mode = pathways::UpscalerMode::Gateway;
                cfg.upways_superres = false;
                cfg.gateway_superres = true;
                cfg.render_scale = 0.6666667f;
                if (!stepAndVerify("4b. Gateway SR (0.67x)", 4, "Gateway", "None")) return 1;
            }

            cfg.upscaler_mode = pathways::UpscalerMode::FSR;
            cfg.fsr_version = pathways::FsrVersion::Fsr3;
            cfg.upways_superres = false;
            cfg.gateway_superres = false;
            cfg.render_scale = 0.6666667f;
            if (!stepAndVerify("4c. Switch Gateway SR -> FSR 3.1 (0.67x)", 4, expectedFsr3, "None")) return 1;

            cfg.upscaler_mode = pathways::UpscalerMode::None;
            cfg.render_scale = 1.0f;
            if (!stepAndVerify("4d. Switch FSR 3.1 -> Native 1.0x", 4, "None", "None")) return 1;

            // 5. Cycle through Denoisers at Native 1.0x
            cfg.denoiser_mode = pathways::DenoiserMode::Svgf;
            if (!stepAndVerify("5a. Native 1.0x + E-SVGF Denoiser", 4, "None", "SVGF")) return 1;

            cfg.denoiser_mode = pathways::DenoiserMode::FfxRayRegeneration;
            if (!stepAndVerify("5b. Native 1.0x + Ray Regen Request (Runtime Gating)", 4, "None", expectedRayRegen)) return 1;

            if (initStats.has_cooperative_matrix) {
                cfg.denoiser_mode = pathways::DenoiserMode::Upways;
                if (!stepAndVerify("5c. Native 1.0x + Upways Denoiser", 4, "None", "Upways")) return 1;

                cfg.denoiser_mode = pathways::DenoiserMode::Gateway;
                if (!stepAndVerify("5d. Native 1.0x + Gateway Denoiser", 4, "None", "Gateway")) return 1;
            }

            cfg.denoiser_mode = pathways::DenoiserMode::None;
            if (!stepAndVerify("5e. Native 1.0x + Off (MC)", 4, "None", "None")) return 1;

            // 6. Combined FSR 3.1 (0.67x) + Denoisers + GI & Caching (NRC, ReSTIR PT, Caustics)
            cfg.upscaler_mode = pathways::UpscalerMode::FSR;
            cfg.fsr_version = pathways::FsrVersion::Fsr3;
            cfg.render_scale = 0.6666667f;
            cfg.denoiser_mode = pathways::DenoiserMode::Svgf;
            if (!stepAndVerify("6a. FSR 3.1 (0.67x) + E-SVGF", 4, expectedFsr3, "SVGF")) return 1;

            if (initStats.has_cooperative_matrix) {
                cfg.denoiser_mode = pathways::DenoiserMode::Upways;
                if (!stepAndVerify("6b. FSR 3.1 (0.67x) + Upways Denoiser", 4, expectedFsr3, "Upways")) return 1;

                cfg.denoiser_mode = pathways::DenoiserMode::Gateway;
                if (!stepAndVerify("6c. FSR 3.1 (0.67x) + Gateway Denoiser", 4, expectedFsr3, "Gateway")) return 1;
            }

            cfg.denoiser_mode = pathways::DenoiserMode::Svgf;
            cfg.enable_nrc = true;
            cfg.nrc_backend = pathways::NrcBackend::FfxRadianceCache;
            if (!stepAndVerify("7a. Request FfxRadianceCache live (Runtime Gating)", 4, expectedFsr3, "SVGF")) return 1;
            if (!initStats.supports_fsr_rc && cfg.enable_nrc) {
                pathways::Logger::error("[FAIL] Unsupported FfxRadianceCache did not disable enable_nrc!");
                return 1;
            }
            if (initStats.has_cooperative_matrix) {
                cfg.enable_nrc = true;
                cfg.nrc_backend = pathways::NrcBackend::NativeWave32;
                if (!stepAndVerify("7a-ii. Enable NativeWave32 NRC live", 4, expectedFsr3, "SVGF")) return 1;
                if (!cfg.enable_nrc) {
                    pathways::Logger::error("[FAIL] Supported NativeWave32 NRC failed to enable!");
                    return 1;
                }
            }

            cfg.enable_restir = true;
            if (!stepAndVerify("7b. Enable ReSTIR PT live", 4, expectedFsr3, "SVGF")) return 1;

            cfg.enable_caustics = true;
            if (!stepAndVerify("7c. Enable Real-Time Caustics live", 4, expectedFsr3, "SVGF")) return 1;

            // 8. Disable GI options & return to clean FSR 3.1 + E-SVGF for final frame dump
            cfg.enable_caustics = false;
            cfg.enable_restir = false;
            cfg.enable_nrc = false;
            if (!stepAndVerify("8. Return to FSR 3.1 (0.67x) + E-SVGF", 6, expectedFsr3, "SVGF")) return 1;

            engine.dumpOutputFiles();
            engine.printExecutionSummary();
            auto finalStats = engine.getStats();
            if (finalStats.validation_errors > 0) {
                pathways::Logger::error("Test failed: Validation errors encountered: {}", finalStats.validation_errors);
                return 1;
            }
            pathways::Logger::info("[SUCCESS] Dynamic runtime GUI options test PASSED all state transitions with 0 validation errors!");
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
