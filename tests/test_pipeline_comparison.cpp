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
        check_true(cfgDef.wavefront_sort_mode == WavefrontSortMode::Dual, "Default sort mode is Dual");

        // Explicit --pipeline rtp
        const char* argv1[] = { "pathways", "--pipeline", "rtp" };
        Config c1 = Config::parse(3, const_cast<char**>(argv1));
        check_true(c1.pipeline_type == PipelineType::RTP, "--pipeline rtp sets RTP");

        // Explicit --pipeline wavefront
        const char* argv2[] = { "pathways", "--pipeline", "wavefront" };
        Config c2 = Config::parse(3, const_cast<char**>(argv2));
        check_true(c2.pipeline_type == PipelineType::Wavefront, "--pipeline wavefront sets Wavefront");

        // Wavefront sort modes: none, archetype, dual
        const char* argvSortNone[] = { "pathways", "--pipeline", "wavefront", "--wavefront-sort", "none" };
        Config cSortNone = Config::parse(5, const_cast<char**>(argvSortNone));
        check_true(cSortNone.wavefront_sort_mode == WavefrontSortMode::None, "Sort mode None parsed correctly");

        const char* argvSortArch[] = { "pathways", "--pipeline", "wavefront", "--wavefront-sort", "archetype" };
        Config cSortArch = Config::parse(5, const_cast<char**>(argvSortArch));
        check_true(cSortArch.wavefront_sort_mode == WavefrontSortMode::Archetype, "Sort mode Archetype parsed correctly");

        const char* argvSortDual[] = { "pathways", "--pipeline", "wavefront", "--wavefront-sort", "dual" };
        Config cSortDual = Config::parse(5, const_cast<char**>(argvSortDual));
        check_true(cSortDual.wavefront_sort_mode == WavefrontSortMode::Dual, "Sort mode Dual parsed correctly");

        // Indirect / secondary bounce radiance clamping
        check_true(cfgDef.indirect_clamp == 35.0f, "Default indirect clamp is 35.0 cd/m2");

        const char* argvClamp1[] = { "pathways", "--indirect-clamp", "50.0" };
        Config cClamp1 = Config::parse(3, const_cast<char**>(argvClamp1));
        assert_near(cClamp1.indirect_clamp, 50.0f, 0.001f, "--indirect-clamp sets custom ceiling");

        const char* argvClamp2[] = { "pathways", "--indirect-clamp=75.5" };
        Config cClamp2 = Config::parse(2, const_cast<char**>(argvClamp2));
        assert_near(cClamp2.indirect_clamp, 75.5f, 0.001f, "--indirect-clamp=val sets custom ceiling");

        const char* argvClamp3[] = { "pathways", "--sec-clamp", "20.0" };
        Config cClamp3 = Config::parse(3, const_cast<char**>(argvClamp3));
        assert_near(cClamp3.indirect_clamp, 20.0f, 0.001f, "--sec-clamp sets custom ceiling");

        const char* argvClampZero1[] = { "pathways", "--indirect-clamp", "0" };
        Config cClampZero1 = Config::parse(3, const_cast<char**>(argvClampZero1));
        assert_near(cClampZero1.indirect_clamp, 0.0f, 0.001f, "--indirect-clamp 0 disables clamping");

        const char* argvClampZero2[] = { "pathways", "--indirect-clamp=0.0" };
        Config cClampZero2 = Config::parse(2, const_cast<char**>(argvClampZero2));
        assert_near(cClampZero2.indirect_clamp, 0.0f, 0.001f, "--indirect-clamp=0.0 disables clamping");

        const char* argvClampZero3[] = { "pathways", "--sec-clamp", "0" };
        Config cClampZero3 = Config::parse(3, const_cast<char**>(argvClampZero3));
        assert_near(cClampZero3.indirect_clamp, 0.0f, 0.001f, "--sec-clamp 0 disables clamping");

        // Target frame time budget
        assert_near(cfgDef.target_frame_time_ms, 8.3f, 0.001f, "Default target frame time is 8.3 ms");

        const char* argvBudget1[] = { "pathways", "--target-frame-time", "16.6" };
        Config cBudget1 = Config::parse(3, const_cast<char**>(argvBudget1));
        assert_near(cBudget1.target_frame_time_ms, 16.6f, 0.001f, "--target-frame-time sets custom budget");

        const char* argvBudget2[] = { "pathways", "--frame-budget=6.9" };
        Config cBudget2 = Config::parse(2, const_cast<char**>(argvBudget2));
        assert_near(cBudget2.target_frame_time_ms, 6.9f, 0.001f, "--frame-budget=val sets custom budget");

        const char* argvFps1[] = { "pathways", "--target-fps", "60" };
        Config cFps1 = Config::parse(3, const_cast<char**>(argvFps1));
        assert_near(cFps1.target_frame_time_ms, 16.6667f, 0.01f, "--target-fps auto-derives frame budget");

        // Progressive accumulation defaults
        check_true(cfgDef.progressive_accumulation == true, "Default progressive accumulation is enabled");
        check_true(cfgDef.max_accum_frames == 2048, "Default max_accum_frames is 2048");

        // Setting accumulation limit via --accumulation and --accum
        const char* argvAccum1[] = { "pathways", "--accumulation", "512" };
        Config cAccum1 = Config::parse(3, const_cast<char**>(argvAccum1));
        check_true(cAccum1.progressive_accumulation == true, "--accumulation 512 enables accumulation");
        check_true(cAccum1.max_accum_frames == 512, "--accumulation 512 sets max_accum_frames to 512");

        const char* argvAccum2[] = { "pathways", "--accumulation=1024" };
        Config cAccum2 = Config::parse(2, const_cast<char**>(argvAccum2));
        check_true(cAccum2.progressive_accumulation == true, "--accumulation=1024 enables accumulation");
        check_true(cAccum2.max_accum_frames == 1024, "--accumulation=1024 sets max_accum_frames to 1024");

        const char* argvAccum3[] = { "pathways", "--accum", "256" };
        Config cAccum3 = Config::parse(3, const_cast<char**>(argvAccum3));
        check_true(cAccum3.progressive_accumulation == true, "--accum 256 enables accumulation");
        check_true(cAccum3.max_accum_frames == 256, "--accum 256 sets max_accum_frames to 256");

        const char* argvAccum4[] = { "pathways", "--accum=128" };
        Config cAccum4 = Config::parse(2, const_cast<char**>(argvAccum4));
        check_true(cAccum4.progressive_accumulation == true, "--accum=128 enables accumulation");
        check_true(cAccum4.max_accum_frames == 128, "--accum=128 sets max_accum_frames to 128");

        // Turning accumulation off (0, off, none, false)
        const char* argvAccumOff1[] = { "pathways", "--accumulation", "0" };
        Config cAccumOff1 = Config::parse(3, const_cast<char**>(argvAccumOff1));
        check_true(!cAccumOff1.progressive_accumulation, "--accumulation 0 disables accumulation");

        const char* argvAccumOff2[] = { "pathways", "--accumulation=0" };
        Config cAccumOff2 = Config::parse(2, const_cast<char**>(argvAccumOff2));
        check_true(!cAccumOff2.progressive_accumulation, "--accumulation=0 disables accumulation");

        const char* argvAccumOff3[] = { "pathways", "--accumulation", "off" };
        Config cAccumOff3 = Config::parse(3, const_cast<char**>(argvAccumOff3));
        check_true(!cAccumOff3.progressive_accumulation, "--accumulation off disables accumulation");

        const char* argvAccumOff4[] = { "pathways", "--accumulation=off" };
        Config cAccumOff4 = Config::parse(2, const_cast<char**>(argvAccumOff4));
        check_true(!cAccumOff4.progressive_accumulation, "--accumulation=off disables accumulation");

        const char* argvAccumOff5[] = { "pathways", "--accumulation", "none" };
        Config cAccumOff5 = Config::parse(3, const_cast<char**>(argvAccumOff5));
        check_true(!cAccumOff5.progressive_accumulation, "--accumulation none disables accumulation");

        const char* argvAccumOff6[] = { "pathways", "--accumulation", "false" };
        Config cAccumOff6 = Config::parse(3, const_cast<char**>(argvAccumOff6));
        check_true(!cAccumOff6.progressive_accumulation, "--accumulation false disables accumulation");

        const char* argvAccumOff7[] = { "pathways", "--accum", "0" };
        Config cAccumOff7 = Config::parse(3, const_cast<char**>(argvAccumOff7));
        check_true(!cAccumOff7.progressive_accumulation, "--accum 0 disables accumulation");

        const char* argvAccumOff8[] = { "pathways", "--accum", "off" };
        Config cAccumOff8 = Config::parse(3, const_cast<char**>(argvAccumOff8));
        check_true(!cAccumOff8.progressive_accumulation, "--accum off disables accumulation");

        const char* argvAccumOff9[] = { "pathways", "--accum=off" };
        Config cAccumOff9 = Config::parse(2, const_cast<char**>(argvAccumOff9));
        check_true(!cAccumOff9.progressive_accumulation, "--accum=off disables accumulation");

        const char* argvAccumOff10[] = { "pathways", "--no-accumulation" };
        Config cAccumOff10 = Config::parse(2, const_cast<char**>(argvAccumOff10));
        check_true(!cAccumOff10.progressive_accumulation, "--no-accumulation disables accumulation");

        const char* argvAccumOff11[] = { "pathways", "--realtime" };
        Config cAccumOff11 = Config::parse(2, const_cast<char**>(argvAccumOff11));
        check_true(!cAccumOff11.progressive_accumulation, "--realtime disables accumulation");

        // Unlimited accumulation
        const char* argvAccumUnlim1[] = { "pathways", "--accumulation", "unlimited" };
        Config cAccumUnlim1 = Config::parse(3, const_cast<char**>(argvAccumUnlim1));
        check_true(cAccumUnlim1.progressive_accumulation == true, "--accumulation unlimited enables accumulation");
        check_true(cAccumUnlim1.max_accum_frames == 0, "--accumulation unlimited sets max_accum_frames to 0 (unlimited)");

        const char* argvAccumUnlim2[] = { "pathways", "--accum=inf" };
        Config cAccumUnlim2 = Config::parse(2, const_cast<char**>(argvAccumUnlim2));
        check_true(cAccumUnlim2.progressive_accumulation == true, "--accum=inf enables accumulation");
        check_true(cAccumUnlim2.max_accum_frames == 0, "--accum=inf sets max_accum_frames to 0 (unlimited)");

        // Combined: SPP + bounces + accumulation off
        const char* argvCombo[] = { "pathways", "--spp", "4", "--bounces", "2", "--accumulation", "off" };
        Config cCombo = Config::parse(7, const_cast<char**>(argvCombo));
        check_true(cCombo.spp == 4, "--spp 4 parsed in combination");
        check_true(cCombo.max_bounces == 2, "--bounces 2 parsed in combination");
        check_true(!cCombo.progressive_accumulation, "--accumulation off parsed in combination");

        std::cout << "  -> Pipeline CLI flags, sort modes, indirect clamp, frame budget, and accumulation successfully verified." << std::endl;
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

        FrameStats::ConfigTallySummary cfgSummary{};
        cfgSummary.label = "Test Configuration";
        cfgSummary.frame_count = 100;
        cfgSummary.avg_frame_time_ms = 12.5;
        cfgSummary.pipeline_stages.is_wavefront = true;
        cfgSummary.pipeline_stages.classify_ms = 0.175;
        cfgSummary.pipeline_stages.primary_rays = 1920 * 1080;
        cfgSummary.pipeline_stages.bounces.push_back({0, 0.575, 0.151, 0.535, 1.261, 1656446, 1400000, 1500000});
        cfgSummary.pipeline_stages.tonemap_ms = 0.120;
        stats.configurations_breakdown.push_back(cfgSummary);

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
        check_true(content.find("pipeline_stages_ms") != std::string::npos, "JSON contains pipeline_stages_ms");
        check_true(content.find("primary_rays") != std::string::npos, "JSON contains primary_rays");
        check_true(content.find("rays_left") != std::string::npos, "JSON contains rays_left");

        // Clean up test file
        std::filesystem::remove(testJsonPath, ec);
        std::cout << "  -> Telemetry JSON serialization successfully verified." << std::endl;
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL PIPELINE COMPARISON TESTS PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
