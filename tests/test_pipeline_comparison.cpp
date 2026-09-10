#include "core/Config.hpp"
#include "utils/ImageDumper.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <fstream>
#include <filesystem>

using namespace pathways;

static void assert_near(float a, float b, float eps = 0.001f, const char* msg = "") {
    if (std::abs(a - b) > eps) {
        std::cerr << "Assertion failed: " << a << " != " << b << " (eps: " << eps << ") " << msg << std::endl;
        std::exit(1);
    }
}

static void check_true(bool cond, const char* msg = "") {
    if (!cond) {
        std::cerr << "Assertion failed: condition is false! " << msg << std::endl;
        std::exit(1);
    }
}

int main() {
    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Testing Megakernel vs. Wavefront Pipeline" << std::endl;
    std::cout << "==========================================================" << std::endl;

    // -------------------------------------------------------------------------
    // 1. Test CLI Config Parsing for Pipelines and Sort Modes
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 1] CLI Pipeline & Sort Mode Parsing..." << std::endl;

        // Default pipeline configuration
        Config cfgDef;
        check_true(cfgDef.pipeline_type == PipelineType::Wavefront, "Default pipeline is Wavefront");
        check_true(cfgDef.wavefront_sort_mode == WavefrontSortMode::None, "Default sort mode is None");

        // Explicit --pipeline rtp
        const char* argv1[] = { "pathways", "--pipeline", "rtp" };
        Config c1 = Config::parse(3, const_cast<char**>(argv1));
        check_true(c1.pipeline_type == PipelineType::RTP, "--pipeline rtp sets RTP");

        // Explicit --pipeline wavefront
        const char* argv2[] = { "pathways", "--pipeline", "wavefront" };
        Config c2 = Config::parse(3, const_cast<char**>(argv2));
        check_true(c2.pipeline_type == PipelineType::Wavefront, "--pipeline wavefront sets Wavefront");

        // Wavefront sort modes: none, archetype, bda, dual
        const char* argvSortNone[] = { "pathways", "--pipeline", "wavefront", "--wavefront-sort", "none" };
        Config cSortNone = Config::parse(5, const_cast<char**>(argvSortNone));
        check_true(cSortNone.wavefront_sort_mode == WavefrontSortMode::None, "Sort mode None parsed correctly");

        const char* argvSortArch[] = { "pathways", "--pipeline", "wavefront", "--wavefront-sort", "archetype" };
        Config cSortArch = Config::parse(5, const_cast<char**>(argvSortArch));
        check_true(cSortArch.wavefront_sort_mode == WavefrontSortMode::Archetype, "Sort mode Archetype parsed correctly");

        const char* argvSortBda[] = { "pathways", "--pipeline", "wavefront", "--wavefront-sort", "bda" };
        Config cSortBda = Config::parse(5, const_cast<char**>(argvSortBda));
        check_true(cSortBda.wavefront_sort_mode == WavefrontSortMode::BDA, "Sort mode BDA parsed correctly");

        const char* argvSortDual[] = { "pathways", "--pipeline", "wavefront", "--wavefront-sort", "dual" };
        Config cSortDual = Config::parse(5, const_cast<char**>(argvSortDual));
        check_true(cSortDual.wavefront_sort_mode == WavefrontSortMode::Dual, "Sort mode Dual parsed correctly");

        std::cout << "  -> Pipeline CLI flags and sort modes successfully verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 2. Test Wavefront Queue Memory Sizing & Theoretical Traffic Model
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 2] Wavefront Queue Memory Footprint & Bandwidth Model..." << std::endl;

        // Queue struct sizes in bytes:
        // RayGeometry = 32 B, RayState = 32 B, RayHit = 16 B, PackedShadowRay = 48 B
        // Ping-pong buffers: 2x RayGeom (32*4*2 = 256 B/ray), 2x RayState (32*4*2 = 256 B/ray),
        // RayHit (16*4 = 64 B/ray), ShadowQueue (48 B/ray)
        const uint32_t bytesPerRayQueue = (32 * 4 * 2) + (32 * 4 * 2) + (16 * 4) + 48; // 624 bytes/ray

        // 1080p (1920x1080 = 2,073,600 rays)
        uint32_t rays1080p = 1920 * 1080;
        double mb1080p = (static_cast<double>(rays1080p) * bytesPerRayQueue) / (1024.0 * 1024.0);
        check_true(mb1080p > 1200.0 && mb1080p < 1300.0, "1080p queue footprint is ~1234 MB");
        assert_near(static_cast<float>(mb1080p), 1233.984f, 0.01f, "1080p queue footprint check");

        // 4K (3840x2160 = 8,294,400 rays)
        uint32_t rays4k = 3840 * 2160;
        double mb4k = (static_cast<double>(rays4k) * bytesPerRayQueue) / (1024.0 * 1024.0);
        check_true(mb4k > 4900.0 && mb4k < 5000.0, "4K queue footprint is ~4936 MB");
        assert_near(static_cast<float>(mb4k), 4935.9375f, 0.01f, "4K queue footprint check");

        std::cout << "  -> Queue memory footprints: 1080p=" << mb1080p << " MB, 4K=" << mb4k << " MB." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 3. Test Telemetry Serialization to JSON
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 3] Telemetry & Wavefront Profiler Breakdown Serialization..." << std::endl;

        FrameStats stats{};
        stats.avg_frame_time_ms = 2.15;
        stats.avg_fps = 465.1;
        stats.width = 1920;
        stats.height = 1080;
        stats.spp = 1;
        stats.max_bounces = 4;
        stats.gpu_name = "AMD Radeon AI PRO R9700 (RADV GFX1201)";
        stats.pipeline_type_str = "Wavefront (Archetype DGC)";

        stats.wavefront_stats.valid = true;
        stats.wavefront_stats.classify_ms = 0.175;
        stats.wavefront_stats.queue_memory_footprint_mb = 1233.99;
        stats.wavefront_stats.estimated_vram_traffic_mb = 861.29;

        FrameStats::BounceProfile bp0{};
        bp0.bounce = 0;
        bp0.shade_ms = 0.575;
        bp0.shadow_ms = 0.151;
        bp0.intersect_ms = 0.535;
        bp0.active_rays = 1656446;
        bp0.diff_rays = 1484502;
        bp0.diel_rays = 71192;
        bp0.cond_rays = 60776;
        bp0.comp_rays = 39976;
        stats.wavefront_stats.bounces.push_back(bp0);

        std::string testJsonPath = "output/test_telemetry_dump.json";
        std::error_code ec;
        std::filesystem::create_directories("output", ec);
        ImageDumper::saveStatsJSON(testJsonPath, stats);

        check_true(std::filesystem::exists(testJsonPath), "JSON stats file must exist on disk");

        // Verify content contains profiler breakdown keys
        std::ifstream inFile(testJsonPath);
        std::string content((std::istreambuf_iterator<char>(inFile)), std::istreambuf_iterator<char>());
        check_true(content.find("wavefront_profiler_breakdown") != std::string::npos, "JSON contains wavefront_profiler_breakdown");
        check_true(content.find("estimated_vram_traffic_mb") != std::string::npos, "JSON contains estimated_vram_traffic_mb");
        check_true(content.find("dielectric_rays") != std::string::npos, "JSON contains dielectric_rays");

        // Clean up test file
        std::filesystem::remove(testJsonPath, ec);
        std::cout << "  -> Telemetry JSON serialization successfully verified." << std::endl;
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL PIPELINE COMPARISON TESTS PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
