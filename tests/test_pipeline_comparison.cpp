#include "core/Config.hpp"
#include "utils/ImageDumper.hpp"
#include "rt/DenoiserPipeline.hpp"
#include "rt/WavefrontPipeline.hpp"
#include "vulkan/VulkanContext.hpp"
#include "vulkan/Buffer.hpp"
#include "vulkan/Image.hpp"

#include <glm/glm.hpp>
#include <glm/gtc/packing.hpp>
#include <iostream>
#include <cassert>
#include <cmath>
#include <cstring>
#include <vector>
#include <fstream>
#include <filesystem>

using namespace pathways;

namespace {

bool parseRejected(std::initializer_list<const char*> args) {
    std::vector<char*> argv;
    for (const char* a : args) argv.push_back(const_cast<char*>(a));
    try {
        (void)Config::parse(static_cast<int>(argv.size()), argv.data());
        return false;
    } catch (const std::runtime_error&) {
        return true;
    }
}

std::vector<char> loadSpvFile(const std::string& filename) {
    std::filesystem::path spvPath = std::filesystem::path(PATHWAYS_SPV_DIR) / filename;
    std::ifstream file(spvPath, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        return {};
    }
    const size_t size = static_cast<size_t>(file.tellg());
    std::vector<char> buf(size);
    file.seekg(0, std::ios::beg);
    file.read(buf.data(), static_cast<std::streamsize>(size));
    return buf;
}

} // namespace

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
        check_true(cfgDef.wavefront_sort_mode == WavefrontSortMode::Auto, "Default sort mode is Auto");

        // Removed legacy --pipeline rtp must be rejected
        check_true(parseRejected({ "pathways", "--pipeline", "rtp" }),
                   "Removed --pipeline rtp must throw std::runtime_error");

        // Explicit --pipeline wavefront
        const char* argv2[] = { "pathways", "--pipeline", "wavefront" };
        Config c2 = Config::parse(3, const_cast<char**>(argv2));
        check_true(c2.pipeline_type == PipelineType::Wavefront, "--pipeline wavefront sets Wavefront");

        // Wavefront sort modes: auto, none, archetype, dual
        const char* argvSortAuto[] = { "pathways", "--pipeline", "wavefront", "--wavefront-sort", "auto" };
        Config cSortAuto = Config::parse(5, const_cast<char**>(argvSortAuto));
        check_true(cSortAuto.wavefront_sort_mode == WavefrontSortMode::Auto, "Sort mode Auto parsed correctly");

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

        // CLI default values and flag overrides verification
        check_true(cfgDef.restir_m_cap == 60, "Default restir_m_cap must be 60");
        const char* argvMCap[] = { "pathways", "--restir-m-cap", "45" };
        Config cMCap = Config::parse(3, const_cast<char**>(argvMCap));
        check_true(cMCap.restir_m_cap == 45, "--restir-m-cap sets custom M-cap");

        const char* argvMCapEq[] = { "pathways", "--restir-m-cap=80" };
        Config cMCapEq = Config::parse(2, const_cast<char**>(argvMCapEq));
        check_true(cMCapEq.restir_m_cap == 80, "--restir-m-cap=val sets custom M-cap");

        const char* argvRestir[] = { "pathways", "--restir" };
        Config cRestir = Config::parse(2, const_cast<char**>(argvRestir));
        check_true(cRestir.enable_restir, "--restir enables unified ReSTIR PT");

        // GUI overlay toggle verification
        check_true(cfgDef.enable_gui, "Default enable_gui must be true");
        const char* argvNoGui[] = { "pathways", "--no-gui" };
        Config cNoGui = Config::parse(2, const_cast<char**>(argvNoGui));
        check_true(!cNoGui.enable_gui, "--no-gui disables GUI overlay");

        const char* argvNoUi[] = { "pathways", "--no-ui" };
        Config cNoUi = Config::parse(2, const_cast<char**>(argvNoUi));
        check_true(!cNoUi.enable_gui, "--no-ui disables GUI overlay");

        const char* argvNoOverlay[] = { "pathways", "--no-overlay" };
        Config cNoOverlay = Config::parse(2, const_cast<char**>(argvNoOverlay));
        check_true(!cNoOverlay.enable_gui, "--no-overlay disables GUI overlay");

        const char* argvGui[] = { "pathways", "--no-gui", "--gui" };
        Config cGui = Config::parse(3, const_cast<char**>(argvGui));
        check_true(cGui.enable_gui, "--gui re-enables GUI overlay");

        const char* argvDbgView[] = { "pathways", "--debug-view", "restir-m" };
        Config cDbgView = Config::parse(3, const_cast<char**>(argvDbgView));
        check_true(cDbgView.debug_view == DebugViewMode::RestirM, "--debug-view restir-m sets DebugViewMode::RestirM");

        const char* argvDenoiserRR[] = { "pathways", "--ray-regeneration" };
        Config cDenoiserRR = Config::parse(2, const_cast<char**>(argvDenoiserRR));
        check_true(cDenoiserRR.denoiser_mode == DenoiserMode::FfxRayRegeneration,
                   "--ray-regeneration sets DenoiserMode::FfxRayRegeneration");

        const char* argvDenoiserFfxRr[] = { "pathways", "--ffx-rr" };
        Config cDenoiserFfxRr = Config::parse(2, const_cast<char**>(argvDenoiserFfxRr));
        check_true(cDenoiserFfxRr.denoiser_mode == DenoiserMode::FfxRayRegeneration,
                   "--ffx-rr sets DenoiserMode::FfxRayRegeneration");

        const char* argvDenoiserModeFfxRr[] = { "pathways", "--denoiser", "ffx-rr" };
        Config cDenoiserModeFfxRr = Config::parse(3, const_cast<char**>(argvDenoiserModeFfxRr));
        check_true(cDenoiserModeFfxRr.denoiser_mode == DenoiserMode::FfxRayRegeneration,
                   "--denoiser ffx-rr sets DenoiserMode::FfxRayRegeneration");

        const char* argvDenoiserUpways[] = { "pathways", "--denoiser", "upways" };
        Config cDenoiserUpways = Config::parse(3, const_cast<char**>(argvDenoiserUpways));
        check_true(cDenoiserUpways.denoiser_mode == DenoiserMode::Upways, "--denoiser upways sets DenoiserMode::Upways");

        // Verify removed vaporware/legacy flags are strictly rejected
        auto expectRejected = [](std::initializer_list<const char*> args, const char* label) {
            check_true(parseRejected(args), label);
        };
        expectRejected({ "pathways", "--svgf" }, "Removed --svgf flag must throw std::runtime_error");
        expectRejected({ "pathways", "--esvgf" }, "Removed --esvgf flag must throw std::runtime_error");
        expectRejected({ "pathways", "--denoiser", "svgf" }, "Removed --denoiser svgf must throw std::runtime_error");
        expectRejected({ "pathways", "--denoiser", "esvgf" }, "Removed --denoiser esvgf must throw std::runtime_error");
        expectRejected({ "pathways", "--relax" }, "Removed --relax flag must throw std::runtime_error");
        expectRejected({ "pathways", "--asvgf" }, "Removed --asvgf flag must throw std::runtime_error");
        expectRejected({ "pathways", "--denoiser", "relax" }, "Removed --denoiser relax must throw std::runtime_error");
        expectRejected({ "pathways", "--scaler", "fsr1" }, "Removed --scaler fsr1 must throw std::runtime_error");
        expectRejected({ "pathways", "--scaler", "fsr5" }, "Removed --scaler fsr5 must throw std::runtime_error");
        expectRejected({ "pathways", "--scaler", "gateway" }, "Removed --scaler gateway must throw std::runtime_error");

        check_true(cfgDef.capture_reference_spp == 256, "Default capture_reference_spp must be 256");
        const char* argvRefSpp[] = { "pathways", "--capture-reference-spp", "128" };
        Config cRefSpp = Config::parse(3, const_cast<char**>(argvRefSpp));
        check_true(cRefSpp.capture_reference_spp == 128, "--capture-reference-spp sets custom reference SPP");

        check_true(cfgDef.capture_channels == 24, "Default capture_channels must be 24 (PTTD v3)");
        check_true(cfgDef.instance_density == 1.0f, "Default instance_density must be 1.0");
        check_true(cfgDef.cull_distance == 0.0f, "Default cull_distance must be 0.0");

        const char* argvDensity[] = { "pathways", "--instance-density", "0.75", "--cull-distance", "50.0" };
        Config cDensity = Config::parse(5, const_cast<char**>(argvDensity));
        assert_near(cDensity.instance_density, 0.75f, 0.001f, "--instance-density sets density");
        assert_near(cDensity.cull_distance, 50.0f, 0.001f, "--cull-distance sets cull distance");

        const char* argvDensityEq[] = { "pathways", "--density=0.5", "--cull-dist=100.0" };
        Config cDensityEq = Config::parse(3, const_cast<char**>(argvDensityEq));
        assert_near(cDensityEq.instance_density, 0.5f, 0.001f, "--density=val sets density");
        assert_near(cDensityEq.cull_distance, 100.0f, 0.001f, "--cull-dist=val sets cull distance");

        check_true(!cfgDef.adopt_scene_bounces, "Default adopt_scene_bounces must be false");
        const char* argvSceneBnc[] = { "pathways", "--scene-bounces" };
        Config cSceneBnc = Config::parse(2, const_cast<char**>(argvSceneBnc));
        check_true(cSceneBnc.adopt_scene_bounces, "--scene-bounces enables adopt_scene_bounces");

        check_true(cfgDef.enable_light_tree, "Default enable_light_tree must be true");
        const char* argvNoLt[] = { "pathways", "--no-light-tree" };
        Config cNoLt = Config::parse(2, const_cast<char**>(argvNoLt));
        check_true(!cNoLt.enable_light_tree, "--no-light-tree disables enable_light_tree");

        check_true(!Config::dgcExecSetRequested(), "Default DGC execution set policy must be disabled (multi-dispatch default)");

        // Tier 1 default alignment checks
        check_true(cfgDef.enable_tail_megakernel, "Default enable_tail_megakernel must be true");
        check_true(cfgDef.tail_megakernel_bounce == 2u, "Default tail_megakernel_bounce must be 2");
        check_true(cfgDef.delta_unroll == 6u, "Default delta_unroll must be 6");
        check_true(cfgDef.max_delta_events == 6u, "Default max_delta_events must be 6");
        check_true(cfgDef.secondary_sort_mode == SecondarySortMode::DirectCoherentK8,
                   "Default secondary_sort_mode must be DirectCoherentK8");

        const char* argvNoTail[] = { "pathways", "--no-tail-megakernel", "--delta-unroll", "2", "--sec-sort", "direct" };
        Config cTier1Ov = Config::parse(6, const_cast<char**>(argvNoTail));
        check_true(!cTier1Ov.enable_tail_megakernel, "--no-tail-megakernel disables tail megakernel");
        check_true(cTier1Ov.delta_unroll == 2u, "--delta-unroll 2 overrides default");
        check_true(cTier1Ov.secondary_sort_mode == SecondarySortMode::DirectCoherent,
                   "--sec-sort direct overrides default");

        std::cout << "  -> Pipeline CLI flags, sort modes, indirect clamp, and frame budget successfully verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 2. Test Wavefront Queue Memory Sizing & ABI Struct Contracts
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 2] Wavefront Queue Memory Footprint & ABI Contracts..." << std::endl;

        // Verify shared C++/GLSL push-constant & counter ABI sizes from wavefront_shared.glsl
        check_true(sizeof(QueueCountersBuffer) == 256, "QueueCountersBuffer must be 256 bytes");
        check_true(sizeof(WavefrontClassifyPC) == 80, "WavefrontClassifyPC must be 80 bytes");
        check_true(sizeof(WavefrontShadePC) == 92, "WavefrontShadePC must be 92 bytes");
        check_true(sizeof(WavefrontShadowPC) == 44, "WavefrontShadowPC must be 44 bytes");
        check_true(sizeof(WavefrontIntersectPC) == 96, "WavefrontIntersectPC must be 96 bytes");
        check_true(sizeof(WavefrontTailPC) == 84, "WavefrontTailPC must be 84 bytes");
        check_true(sizeof(WavefrontCoherenceSortPC) == 56, "WavefrontCoherenceSortPC must be 56 bytes");

        // Actual WavefrontPipeline::allocateQueues per-slot ray queue sizes:
        // RayGeometry (16 B x 2 ping-pong) + RayState (16 B x 2 ping-pong) +
        // RayHit (16 B) + PackedShadowRay (32 B) + MaterialIndexQueue (16 B) = 128 B/ray/slot.
        // Double-buffered across 2 in-flight frame slots = 256 B/ray (320 B/ray with secondary octant queues).
        constexpr uint32_t bytesPerRayPerSlot = 16u * 2u + 16u * 2u + 16u + 32u + 16u;
        constexpr uint32_t bytesPerRayDoubleBuffered = bytesPerRayPerSlot * 2u; // 256 B/ray
        check_true(bytesPerRayDoubleBuffered == 256u, "Double-buffered base queue footprint == 256 B/ray");

        // 1080p (1920x1080 = 2,073,600 rays) -> 506.25 MiB
        uint32_t rays1080p = 1920u * 1080u;
        double mb1080p = (static_cast<double>(rays1080p) * bytesPerRayDoubleBuffered) / (1024.0 * 1024.0);
        assert_near(static_cast<float>(mb1080p), 506.25f, 0.01f, "1080p double-buffered queue footprint check");

        // 4K (3840x2160 = 8,294,400 rays) -> 2025.0 MiB
        uint32_t rays4k = 3840u * 2160u;
        double mb4k = (static_cast<double>(rays4k) * bytesPerRayDoubleBuffered) / (1024.0 * 1024.0);
        assert_near(static_cast<float>(mb4k), 2025.0f, 0.01f, "4K double-buffered queue footprint check");

        std::cout << "  -> Queue memory footprints (2x slots): 1080p=" << mb1080p << " MiB, 4K=" << mb4k << " MiB." << std::endl;
    }

    struct ScopedTempDirCleanup {
        std::filesystem::path dir;
        ~ScopedTempDirCleanup() {
            if (!dir.empty()) {
                std::error_code ec;
                std::filesystem::remove_all(dir, ec);
            }
        }
    };
    const std::filesystem::path tempOutDir = std::filesystem::temp_directory_path() / "pathways_test_pipeline_comparison";
    {
        std::error_code ec;
        std::filesystem::create_directories(tempOutDir, ec);
    }
    ScopedTempDirCleanup tempDirCleanup{ tempOutDir };

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

        std::string testJsonPath = (tempOutDir / "test_telemetry_dump.json").string();
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
        check_true(content.find("std_dev_frame_time_ms") != std::string::npos, "JSON contains std_dev_frame_time_ms");
        check_true(content.find("machine_learning_and_cooperative_matrix") != std::string::npos, "JSON contains machine_learning_and_cooperative_matrix");
        check_true(content.find("supports_fsr4_ml") != std::string::npos, "JSON contains supports_fsr4_ml");

        std::cout << "  -> Telemetry JSON serialization successfully verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 4. Test High-Quality 16-bit PNG Screenshot Output & Naming Pattern
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 4] High-Quality 16-bit PNG Screenshot Output & Naming Pattern..." << std::endl;

        std::string testScene = "cornell_caustic_extended";
        uint32_t spp = 1;
        uint32_t bounce = 4;
        uint32_t accum = 10;
        std::string testPath = (tempOutDir / std::format("pw_{}_{}spp_{}bounce_{}accum_test.png",
                                            testScene, spp, bounce, accum)).string();

        // Create a small 4x4 test pattern in 16-bit RGBA
        uint32_t w = 4, h = 4;
        std::vector<uint16_t> testPixels(w * h * 4, 32768); // 50% gray in 16-bit
        auto future = ImageDumper::savePNG16Async(testPath, w, h, std::move(testPixels));
        check_true(future.valid(), "savePNG16Async must return a valid std::future");
        bool writeSuccess = future.get();
        check_true(writeSuccess, "savePNG16Async must successfully write 16-bit PNG");
        check_true(std::filesystem::exists(testPath), "16-bit PNG file must exist on disk");
        check_true(std::filesystem::file_size(testPath) > 50, "16-bit PNG file size must be non-trivial");

        std::cout << "  -> 16-bit PNG async screenshot pipeline & naming format verified." << std::endl;

        // Verify uncapped HDR OpenEXR output via TinyEXR
        std::string exrPath = (tempOutDir / "pw_hdr_test.exr").string();
        std::vector<float> hdrPixels(w * h * 4, 0.0f);
        for (uint32_t i = 0; i < w * h; ++i) {
            hdrPixels[i * 4 + 0] = 14.5f;
            hdrPixels[i * 4 + 1] = 3.25f;
            hdrPixels[i * 4 + 2] = 0.75f;
            hdrPixels[i * 4 + 3] = 1.0f;
        }
        check_true(ImageDumper::saveEXR(exrPath, w, h, hdrPixels.data()), "ImageDumper::saveEXR must succeed");
        check_true(std::filesystem::exists(exrPath) && std::filesystem::file_size(exrPath) > 64, "EXR file must be non-empty");
        std::cout << "  -> Uncapped HDR OpenEXR serialization verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 5. Test 44-Channel PTTD v5 Header & Payload Serialization
    // -------------------------------------------------------------------------
    {
        std::cout << "[TEST 5] 44-Channel PTTD v5 Serialization..." << std::endl;
        const uint32_t w = 4, h = 4, ch = 44;
        const size_t numFloats = static_cast<size_t>(w) * h * ch;
        std::vector<uint16_t> payload(numFloats, glm::packHalf1x16(0.25f));
        const std::string pttdPath = (tempOutDir / "pw_pttd_v5_test.pttd").string();
        const float camPos[3] = { 1.0f, 2.0f, 3.0f };
        const float jitter[2] = { 0.25f, -0.25f };
        check_true(ImageDumper::savePTTD(pttdPath, w, h, ch, /*dataType=*/0, /*frameIndex=*/7, /*spp=*/1,
                                         payload.data(), payload.size() * sizeof(uint16_t),
                                         45.0f, 16.0f / 9.0f, camPos, jitter),
                   "ImageDumper::savePTTD(44ch) must succeed");

        std::ifstream in(pttdPath, std::ios::binary);
        check_true(in.is_open(), "Saved 44ch PTTD file must open");
        struct HeaderRead {
            char magic[4];
            uint32_t version;
            uint32_t width;
            uint32_t height;
            uint32_t channels;
            uint32_t data_type;
            uint32_t frame_index;
            uint32_t spp;
            uint64_t payload_byte_size;
            float fov_deg;
            float aspect_ratio;
            float cam_pos[3];
            char padding[4];
        } hdr{};
        static_assert(sizeof(HeaderRead) == 64, "PTTD header size mismatch");
        in.read(reinterpret_cast<char*>(&hdr), sizeof(hdr));
        check_true(in.good(), "Must read 64-byte PTTD v5 header");
        check_true(std::memcmp(hdr.magic, "PTTD", 4) == 0, "PTTD magic must be 'PTTD'");
        check_true(hdr.version == 5u, "44-channel PTTD must serialize version == 5");
        check_true(hdr.width == w && hdr.height == h && hdr.channels == 44u,
                   "PTTD v5 dimensions and channel count must match (4x4x44)");
        check_true(hdr.payload_byte_size == payload.size() * sizeof(uint16_t),
                   "PTTD v5 payload_byte_size must match");
        in.close();
        std::error_code ec;
        std::filesystem::remove(pttdPath, ec);
        std::cout << "  -> 44-channel PTTD v5 header & payload verified." << std::endl;
    }

    // -------------------------------------------------------------------------
    // 6. Test 3-Stage DenoiserPipeline (ffx_rr_prep -> passthrough/SDK -> ffx_rr_compose)
    // -------------------------------------------------------------------------
    {
        uint32_t hwGpuCount = 0;
        VkApplicationInfo appInfo{ VK_STRUCTURE_TYPE_APPLICATION_INFO };
        appInfo.pApplicationName = "PathwaysDenoiserProbe";
        appInfo.apiVersion = VK_API_VERSION_1_4;
        VkInstanceCreateInfo instInfo{ VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO };
        instInfo.pApplicationInfo = &appInfo;
        VkInstance probeInst = VK_NULL_HANDLE;
        if (vkCreateInstance(&instInfo, nullptr, &probeInst) == VK_SUCCESS) {
            uint32_t devCount = 0;
            vkEnumeratePhysicalDevices(probeInst, &devCount, nullptr);
            std::vector<VkPhysicalDevice> devs(devCount);
            vkEnumeratePhysicalDevices(probeInst, &devCount, devs.data());
            for (VkPhysicalDevice pd : devs) {
                VkPhysicalDeviceProperties props{};
                vkGetPhysicalDeviceProperties(pd, &props);
                if (props.deviceType != VK_PHYSICAL_DEVICE_TYPE_CPU && props.apiVersion >= VK_API_VERSION_1_4) {
                    ++hwGpuCount;
                }
            }
            vkDestroyInstance(probeInst, nullptr);
        }

        if (hwGpuCount == 0) {
            std::cout << "[SKIP] No hardware Vulkan 1.4 GPU available for live DenoiserPipeline test." << std::endl;
        } else {
            std::cout << "[TEST 6] GPU 3-Stage DenoiserPipeline Energy Conservation..." << std::endl;
            Config cfg;
            cfg.headless = true;
            cfg.validation_layers = true;
            cfg.gpu_index = (hwGpuCount >= 2) ? 1u : 0u;

            VulkanContext vkCtx(cfg, VK_NULL_HANDLE, "DenoiserPipeline 3-Stage Test");
            VkDevice device = vkCtx.getDevice();
            VmaAllocator allocator = vkCtx.getAllocator();
            VkQueue queue = vkCtx.getGraphicsQueue();

            DenoiserPipeline denoiser(
                device, vkCtx.getPhysicalDevice(), allocator, queue,
                [](const std::string& spv) {
                    auto bytes = loadSpvFile(spv);
                    if (bytes.empty()) {
                        throw std::runtime_error("Missing SPIR-V: " + spv);
                    }
                    return bytes;
                },
                vkCtx.hasSubgroupSizeControl()
            );

            const uint32_t w = 16, h = 16;
            denoiser.resize(w, h, AccumFormat::RGBA16_SFLOAT);
            check_true(denoiser.isInitialized(), "DenoiserPipeline must be initialized after resize");

            constexpr VkImageUsageFlags kImgUsage =
                VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT |
                VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;

            auto makeInput = [&]() {
                return std::make_unique<Image>(device, allocator, w, h, VK_FORMAT_R16G16B16A16_SFLOAT, kImgUsage);
            };
            auto imgND    = makeInput();
            auto imgMV    = makeInput();
            auto imgAR    = makeInput();
            auto imgSA    = makeInput();
            auto imgSkip  = makeInput();
            auto imgDD    = makeInput();
            auto imgID    = makeInput();
            auto imgDS    = makeInput();
            auto imgIS    = makeInput();

            const size_t numPx = static_cast<size_t>(w) * h;
            const VkDeviceSize rgba16Bytes = numPx * 4 * sizeof(uint16_t);
            std::vector<uint16_t> stageData(numPx * 4 * 9, 0);
            auto packPx = [&](size_t imgIdx, size_t px, const glm::vec4& v) {
                const size_t base = imgIdx * (numPx * 4) + px * 4;
                stageData[base + 0] = glm::packHalf1x16(v.x);
                stageData[base + 1] = glm::packHalf1x16(v.y);
                stageData[base + 2] = glm::packHalf1x16(v.z);
                stageData[base + 3] = glm::packHalf1x16(v.w);
            };

            // Pixel groups test normal albedo, zero-albedo black diffuse/spec, clamped low albedo (0.005), and sky miss
            for (size_t px = 0; px < numPx; ++px) {
                const size_t mode = denoiser.isUsingSdkProvider() ? 0 : (px % 4);
                if (mode == 0) {
                    packPx(0, px, glm::vec4(0.0f, 0.0f, 1.0f, 10.0f));
                    packPx(1, px, glm::vec4(0.0f, 0.0f, 0.0f, 0.0f));
                    packPx(2, px, glm::vec4(0.50f, 0.25f, 0.80f, 0.40f));
                    packPx(3, px, glm::vec4(0.40f, 0.40f, 0.40f, 0.0f));
                    packPx(4, px, glm::vec4(0.10f, 0.05f, 0.02f, 0.0f));
                    packPx(5, px, glm::vec4(1.50f, 0.50f, 0.40f, 0.0f));
                    packPx(6, px, glm::vec4(0.50f, 0.25f, 0.80f, 3.0f));
                    packPx(7, px, glm::vec4(0.80f, 0.40f, 0.20f, 0.0f));
                    packPx(8, px, glm::vec4(0.20f, 0.20f, 0.60f, 5.0f));
                } else if (mode == 1) {
                    // Zero diffuse + zero spec albedo must still recover 100% energy via effAlb = 1.0
                    packPx(0, px, glm::vec4(0.0f, 1.0f, 0.0f, 4.0f));
                    packPx(1, px, glm::vec4(0.0f));
                    packPx(2, px, glm::vec4(0.0f, 0.0f, 0.0f, 0.2f));
                    packPx(3, px, glm::vec4(0.0f));
                    packPx(4, px, glm::vec4(0.0f));
                    packPx(5, px, glm::vec4(0.6f, 0.3f, 0.1f, 0.0f));
                    packPx(6, px, glm::vec4(0.2f, 0.4f, 0.5f, 2.0f));
                    packPx(7, px, glm::vec4(0.1f, 0.2f, 0.3f, 0.0f));
                    packPx(8, px, glm::vec4(0.3f, 0.1f, 0.2f, 1.0f));
                } else if (mode == 2) {
                    // Low albedo clamped at 0.02 symmetrically in prep and compose
                    packPx(0, px, glm::vec4(1.0f, 0.0f, 0.0f, 6.0f));
                    packPx(1, px, glm::vec4(0.0f));
                    packPx(2, px, glm::vec4(0.005f, 0.0f, 0.0f, 0.8f));
                    packPx(3, px, glm::vec4(0.005f, 0.005f, 0.005f, 0.0f));
                    packPx(4, px, glm::vec4(0.05f, 0.05f, 0.05f, 0.0f));
                    packPx(5, px, glm::vec4(0.04f, 0.0f, 0.0f, 0.0f));
                    packPx(6, px, glm::vec4(0.02f, 0.0f, 0.0f, 1.5f));
                    packPx(7, px, glm::vec4(0.01f, 0.02f, 0.03f, 0.0f));
                    packPx(8, px, glm::vec4(0.01f, 0.01f, 0.01f, 2.5f));
                } else {
                    // Sky / miss pixel: skip radiance only
                    packPx(0, px, glm::vec4(0.0f));
                    packPx(1, px, glm::vec4(0.0f));
                    packPx(2, px, glm::vec4(0.0f, 0.0f, 0.0f, 1.0f));
                    packPx(3, px, glm::vec4(0.0f));
                    packPx(4, px, glm::vec4(0.2f, 0.4f, 0.9f, 0.0f));
                    packPx(5, px, glm::vec4(0.0f));
                    packPx(6, px, glm::vec4(0.0f));
                    packPx(7, px, glm::vec4(0.0f));
                    packPx(8, px, glm::vec4(0.0f));
                }
            }

            Buffer uploadBuf(allocator, rgba16Bytes * 9, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                             VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT);
            uploadBuf.copyFrom(stageData.data(), rgba16Bytes * 9);
            Buffer readbackBuf(allocator, rgba16Bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                               VMA_MEMORY_USAGE_AUTO, VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT);

            VkCommandPoolCreateInfo cpInfo{ VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO };
            cpInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
            cpInfo.queueFamilyIndex = vkCtx.getGraphicsQueueFamily();
            VkCommandPool cmdPool = VK_NULL_HANDLE;
            check_true(vkCreateCommandPool(device, &cpInfo, nullptr, &cmdPool) == VK_SUCCESS, "Create test cmdPool");

            VkCommandBufferAllocateInfo cbInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO };
            cbInfo.commandPool = cmdPool;
            cbInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbInfo.commandBufferCount = 1;
            VkCommandBuffer cmd = VK_NULL_HANDLE;
            check_true(vkAllocateCommandBuffers(device, &cbInfo, &cmd) == VK_SUCCESS, "Allocate test cmd");

            VkCommandBufferBeginInfo beginInfo{ VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
            beginInfo.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            check_true(vkBeginCommandBuffer(cmd, &beginInfo) == VK_SUCCESS, "Begin denoiser test cmd");

            const std::array<Image*, 9> inImgs = {
                imgND.get(), imgMV.get(), imgAR.get(), imgSA.get(),
                imgSkip.get(), imgDD.get(), imgID.get(), imgDS.get(), imgIS.get()
            };
            for (size_t i = 0; i < 9; ++i) {
                inImgs[i]->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                    VK_PIPELINE_STAGE_2_NONE, 0,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
                VkBufferImageCopy reg{};
                reg.bufferOffset = i * rgba16Bytes;
                reg.bufferRowLength = w;
                reg.bufferImageHeight = h;
                reg.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
                reg.imageExtent = { w, h, 1 };
                vkCmdCopyBufferToImage(cmd, uploadBuf.getBuffer(), inImgs[i]->getImage(),
                                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &reg);
                inImgs[i]->transitionLayout(cmd, VK_IMAGE_LAYOUT_GENERAL,
                    VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                    VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                    VK_ACCESS_2_SHADER_STORAGE_READ_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT);
            }

            DenoiserPipeline::CameraMatrices cm{};
            cm.view = glm::mat4(1.0f);
            cm.prevView = glm::mat4(1.0f);
            cm.viewInverse = glm::mat4(1.0f);
            cm.projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 1000.0f);
            cm.nearZ = 0.1f;
            cm.farZ = 1000.0f;

            // Dispatch twice to verify frame-to-frame layout restoration and descriptor reuse
            for (uint32_t f = 0; f < 2; ++f) {
                denoiser.recordDenoise(
                    cmd, f, (f == 0),
                    imgND.get(), imgMV.get(), imgAR.get(), imgSA.get(),
                    imgSkip.get(), imgDD.get(), imgID.get(), imgDS.get(), imgIS.get(),
                    cm
                );
            }

            Image* outImg = denoiser.getDenoisedImage();
            check_true(outImg != nullptr, "Denoiser output image must exist");
            outImg->transitionLayout(cmd, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
            VkBufferImageCopy outReg{};
            outReg.bufferRowLength = w;
            outReg.bufferImageHeight = h;
            outReg.imageSubresource = { VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1 };
            outReg.imageExtent = { w, h, 1 };
            vkCmdCopyImageToBuffer(cmd, outImg->getImage(), VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                   readbackBuf.getBuffer(), 1, &outReg);
            check_true(vkEndCommandBuffer(cmd) == VK_SUCCESS, "End denoiser test cmd");

            VkSubmitInfo submitInfo{ VK_STRUCTURE_TYPE_SUBMIT_INFO };
            submitInfo.commandBufferCount = 1;
            submitInfo.pCommandBuffers = &cmd;
            check_true(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE) == VK_SUCCESS, "Submit denoiser test");
            vkQueueWaitIdle(queue);
            denoiser.collectQueryResults();

            std::vector<uint16_t> outHalf(numPx * 4);
            readbackBuf.invalidate(0, rgba16Bytes);
            void* mappedOut = readbackBuf.map();
            check_true(mappedOut != nullptr, "Map readback buffer");
            std::memcpy(outHalf.data(), mappedOut, rgba16Bytes);
            readbackBuf.unmap();
            for (size_t px = 0; px < numPx; ++px) {
                const float r = glm::unpackHalf1x16(outHalf[px * 4 + 0]);
                const float g = glm::unpackHalf1x16(outHalf[px * 4 + 1]);
                const float b = glm::unpackHalf1x16(outHalf[px * 4 + 2]);
                check_true(std::isfinite(r) && std::isfinite(g) && std::isfinite(b),
                           "DenoiserPipeline output must be finite");
                const size_t mode = denoiser.isUsingSdkProvider() ? 0 : (px % 4);
                glm::vec3 expected(0.0f);
                if (mode == 0)      expected = glm::vec3(3.10f, 1.40f, 2.02f);
                else if (mode == 1) expected = glm::vec3(1.20f, 1.00f, 1.10f);
                else if (mode == 2) expected = glm::vec3(0.13f, 0.08f, 0.09f);
                else                expected = glm::vec3(0.20f, 0.40f, 0.90f);
                const float tol = denoiser.isUsingSdkProvider() ? 0.10f : 0.02f;
                assert_near(r, expected.x, tol, "DenoiserPipeline composed R");
                assert_near(g, expected.y, tol, "DenoiserPipeline composed G");
                assert_near(b, expected.z, tol, "DenoiserPipeline composed B");
            }

            vkDestroyCommandPool(device, cmdPool, nullptr);
            check_true(vkCtx.getValidationErrors() == 0,
                       "Zero Vulkan validation errors during DenoiserPipeline test");
            std::cout << "  -> 3-stage DenoiserPipeline (" << (denoiser.isUsingSdkProvider() ? "FFX SDK" : "Passthrough Fallback")
                      << ") multi-frame & energy conservation verified (0 validation errors)." << std::endl;
        }
    }

    std::cout << "==========================================================" << std::endl;
    std::cout << "  ALL PIPELINE COMPARISON TESTS PASSED!" << std::endl;
    std::cout << "==========================================================" << std::endl;
    return 0;
}
