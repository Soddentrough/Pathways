#include "core/QualityGovernor.hpp"
#include "core/ConfigTally.hpp"
#include "core/InputController.hpp"
#include "core/Config.hpp"
#include "core/Logger.hpp"
#include "scene/Camera.hpp"
#include "scene/SceneRegistry.hpp"

#include <SDL3/SDL.h>
#include <glm/glm.hpp>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace pathways;

namespace {

void check_true(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "[FAIL] " << msg << std::endl;
        std::exit(1);
    }
}

void assert_near(float actual, float expected, float eps, const char* msg) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > eps) {
        std::cerr << "[FAIL] " << msg << ": actual=" << actual << " expected=" << expected
                  << " (eps=" << eps << ")" << std::endl;
        std::exit(1);
    }
}

struct ScopedDirCleanup {
    std::filesystem::path dir;
    ~ScopedDirCleanup() {
        if (!dir.empty()) {
            std::error_code ec;
            std::filesystem::remove_all(dir, ec);
        }
    }
};

struct ScopedEnvVarRestore {
    std::string name;
    bool hadOld = false;
    std::string oldVal;
    explicit ScopedEnvVarRestore(const char* envName) : name(envName) {
        if (const char* v = std::getenv(envName)) {
            hadOld = true;
            oldVal = v;
        }
    }
    ~ScopedEnvVarRestore() {
#if defined(_WIN32)
        _putenv_s(name.c_str(), hadOld ? oldVal.c_str() : "");
#else
        if (hadOld) setenv(name.c_str(), oldVal.c_str(), 1);
        else unsetenv(name.c_str());
#endif
    }
};

} // namespace

int main() {
    Logger::setLogLevel(LogLevel::Warn);

    std::cout << "==========================================================" << std::endl;
    std::cout << "  Pathways: Core Subsystems Pure-CPU Unit Tests           " << std::endl;
    std::cout << "==========================================================" << std::endl;

    // -------------------------------------------------------------------------
    // TEST 1: QualityGovernor (SPP/Bounce Ramping, Emergency Downscale, Lockout, mGPU Split)
    // -------------------------------------------------------------------------
    std::cout << "[TEST 1] QualityGovernor dynamics, cooldown & lockout..." << std::endl;
    {
        GovernorConfig gcfg;
        gcfg.enabled = false;
        gcfg.targetFps = 60;
        QualityGovernor gov(gcfg);
        gov.recordDispatch(0, 1, 4);
        gov.update(0, 2.0f, 0.5f, false, false);
        check_true(!gov.getState().active && gov.getState().currentSpp == 1u,
                   "Disabled governor remains inactive");

        gcfg.enabled = true;
        gcfg.targetFps = 0;
        gov.init(gcfg);
        gov.update(0, 2.0f, 0.5f, false, false);
        check_true(!gov.getState().active, "Uncapped (targetFps == 0) governor remains inactive");

        // Enabled at 60 FPS (budget = 16.667 ms, rtBudget ~= 16.17 ms)
        gcfg.targetFps = 60;
        gcfg.minSpp = 1;
        gcfg.maxSpp = 8;
        gcfg.minBounces = 2;
        gcfg.maxBounces = 6;
        gov.init(gcfg);
        check_true(gov.getState().active, "Governor active at 60 FPS");

        // Feed fast 0.8 ms per-SPP frames -> should ramp SPP up to 8, then bounces to 6
        uint32_t maxSeenSpp = 1;
        uint32_t maxSeenBounces = 4;
        for (uint32_t f = 0; f < 60; ++f) {
            uint32_t slot = f % 2;
            uint32_t spp = gov.getState().currentSpp;
            uint32_t bnc = gov.getState().currentBounces;
            gov.recordDispatch(slot, spp, bnc);
            float simRt = 0.8f * static_cast<float>(spp) * (bnc <= 4 ? (0.35f + 0.1625f * bnc) : (1.0f + 0.25f * (bnc - 4)));
            gov.update(slot, simRt, 0.5f, false, false);
            maxSeenSpp = std::max(maxSeenSpp, gov.getState().currentSpp);
            maxSeenBounces = std::max(maxSeenBounces, gov.getState().currentBounces);
        }
        check_true(maxSeenSpp == 8u, "Governor ramped SPP to maxSpp=8 under ample headroom");
        check_true(maxSeenBounces > 4u && maxSeenBounces <= 6u,
                   "Governor increased bounces above 4 once at maxSpp");

        // Feed emergency overshoot (> 1.01 * rtBudget) -> steps down bounces above 4 first, enters 2-frame cooldown
        const uint32_t sppBeforeSpike = gov.getState().currentSpp;
        const uint32_t bncBeforeSpike = gov.getState().currentBounces;
        gov.recordDispatch(0, sppBeforeSpike, bncBeforeSpike);
        gov.update(0, 25.0f, 0.5f, false, false);
        check_true(gov.getState().currentBounces == bncBeforeSpike - 1u && gov.getState().currentSpp == sppBeforeSpike,
                   "Overshoot steps down elevated bounces before dropping SPP");

        // Immediate follow-up frames during 2-frame cooldown must NOT cascade downscale further
        const uint32_t bncAfterDrop = gov.getState().currentBounces;
        const uint32_t sppAfterDrop = gov.getState().currentSpp;
        gov.recordDispatch(1, sppBeforeSpike, bncBeforeSpike);
        gov.update(1, 25.0f, 0.5f, false, false);
        check_true(gov.getState().currentBounces == bncAfterDrop && gov.getState().currentSpp == sppAfterDrop,
                   "Cooldown prevents back-to-back cascading downscales");

        // Fail-lockout prevents immediate re-upgrade to failed (sppBeforeSpike, bncBeforeSpike)
        for (int k = 0; k < 6; ++k) {
            gov.recordDispatch(k % 2, gov.getState().currentSpp, gov.getState().currentBounces);
            gov.update(k % 2, 1.0f, 0.5f, false, false);
        }
        check_true(!(gov.getState().currentSpp >= sppBeforeSpike && gov.getState().currentBounces >= bncBeforeSpike),
                   "90-frame fail-lockout blocks immediate re-entry to failed config");

        // Multi-GPU Sample Parallel SPP split & setBounds clamping
        gov.setBounds(2, 5, 2, 4);
        check_true(gov.getState().currentSpp <= 5u && gov.getState().currentBounces <= 4u,
                   "setBounds clamps active SPP and bounces");
        gov.recordDispatch(0, 5, 4);
        gov.update(0, 2.0f, 0.5f, false, true);
        if (gov.getState().currentSpp > 1u) {
            check_true(gov.getState().primSpp + gov.getState().secSpp == gov.getState().currentSpp,
                       "mGPU sample-parallel primSpp + secSpp == currentSpp");
            check_true(gov.getState().primSpp >= gov.getState().secSpp,
                       "Primary GPU takes >= half of odd SPP");
        }
    }
    std::cout << "[PASS] QualityGovernor tests passed." << std::endl;

    // -------------------------------------------------------------------------
    // TEST 2: ConfigTally (ConfigKey Equality/Labels, formatRayCount, ConfigStatsTally)
    // -------------------------------------------------------------------------
    std::cout << "[TEST 2] ConfigTally keys, labels & statistical aggregation..." << std::endl;
    {
        ConfigKey a;
        ConfigKey b;
        check_true(a == b, "Identical default ConfigKeys compare equal");

        // Inactive differences ignored
        b.fsr_version = FsrVersion::Fsr4;
        b.render_scale = 0.5f;
        b.tile_size = 32;
        b.tail_bounce = 3;
        check_true(a == b, "Inactive fsr_version/render_scale/tile_size/tail_bounce ignored in ConfigKey::operator==");

        // Active differences distinguished
        b.upscaler = UpscalerMode::FSR;
        check_true(!(a == b), "Upscaler mismatch detected");
        a.upscaler = UpscalerMode::FSR;
        a.fsr_version = FsrVersion::Fsr3;
        check_true(!(a == b), "Active fsr_version mismatch detected");

        ConfigKey lblKey;
        lblKey.mgpu_mode = MultiGpuMode::CheckerboardTile;
        lblKey.tile_size = 64;
        lblKey.denoiser = DenoiserMode::Upways;
        lblKey.upscaler = UpscalerMode::FSR;
        lblKey.fsr_version = FsrVersion::Fsr4;
        lblKey.mgpu_upscale_mode = MgpuUpscaleMode::PostMerge;
        lblKey.enable_nrc = true;
        lblKey.enable_tail_megakernel = true;
        lblKey.tail_bounce = 2;
        lblKey.accum_format = AccumFormat::R11G11B10_UFLOAT;
        std::string lbl = lblKey.getLabel();
        check_true(lbl.find("[Upways]") != std::string::npos &&
                   lbl.find("[FSR 4.1 ML]") != std::string::npos &&
                   lbl.find("[Post-Merge SR]") != std::string::npos &&
                   lbl.find("[NRC]") != std::string::npos &&
                   lbl.find("[Tail-MK b>=2]") != std::string::npos &&
                   lbl.find("Checkerboard 64x64") != std::string::npos &&
                   lbl.find("R11G11B10F") != std::string::npos,
                   "ConfigKey::getLabel includes all active mode tags");

        check_true(formatRayCount(0) == "0", "formatRayCount(0)");
        check_true(formatRayCount(999) == "999", "formatRayCount(999)");
        check_true(formatRayCount(1000) == "1,000", "formatRayCount(1000)");
        check_true(formatRayCount(12345678) == "12,345,678", "formatRayCount(12345678)");

        ConfigKey tk;
        tk.width = 1920;
        tk.height = 1080;
        tk.spp = 2;
        tk.max_bounces = 4;
        ConfigStatsTally tally(tk);
        tally.addSample(0.0005, 1.0, 1.0, 0.1); // ignored (<= 0.001)
        check_true(tally.frameCount == 0u, "Sub-microsecond sentinel frameTimeMs ignored");

        WavefrontStageSample wf0{};
        wf0.classifyMs = 0.2;
        wf0.primaryRays = 1920ull * 1080ull * 2ull;
        wf0.bounces.push_back({ 0.5, 0.2, 0.4, 0.01, 0.01, 0.01, 1000, 500, 800 });
        wf0.bounces.push_back({ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0, 0 });

        const double denP[5] = { 0.1, 0.2, 0.3, 0.4, 0.5 };
        tally.addSample(4.0, 1.0, 2.0, 0.3, &wf0, 0.6, denP, 5);
        tally.addSample(6.0, 2.0, 3.0, 0.5, &wf0, 0.8, denP, 5);
        tally.addSample(8.0, 3.0, 4.0, 0.7, &wf0, 1.0, denP, 5);

        check_true(tally.frameCount == 3u, "Valid sample count == 3");
        assert_near(static_cast<float>(tally.getAvgFrameTimeMs()), 6.0f, 1e-5f, "Mean frame time == 6.0 ms");
        assert_near(static_cast<float>(tally.minFrameTimeMs), 4.0f, 1e-5f, "Min frame time == 4.0 ms");
        assert_near(static_cast<float>(tally.maxFrameTimeMs), 8.0f, 1e-5f, "Max frame time == 8.0 ms");
        assert_near(static_cast<float>(tally.getStdDevFrameTimeMs()), 2.0f, 1e-5f, "Sample std-dev (ddof=1) == 2.0 ms");
        assert_near(static_cast<float>(tally.getAvgFps()), 1000.0f / 6.0f, 1e-3f, "Average FPS");
        assert_near(static_cast<float>(tally.getAvgPrimaryRtMs()), 2.0f, 1e-5f, "Average primary RT ms");
        assert_near(static_cast<float>(tally.getAvgSecondaryRtMs()), 3.0f, 1e-5f, "Average secondary RT ms");
        assert_near(static_cast<float>(tally.getAvgTonemapMs()), 0.5f, 1e-5f, "Average tonemap ms");
        assert_near(static_cast<float>(tally.getAvgDenoiserUpscaleMs()), 0.8f, 1e-5f, "Average denoiser/upscale ms");
        assert_near(static_cast<float>(tally.getAvgDenoisePassMs(2)), 0.3f, 1e-5f, "Average denoise pass 2");
        check_true(tally.getAvgPrimaryRays() == 1920ull * 1080ull * 2ull, "Average primary rays");
        check_true(tally.isTargetAchieved(8.3) && !tally.isTargetAchieved(5.0), "isTargetAchieved threshold");
        check_true(tally.getRayThroughput() > 0.0, "Positive ray throughput");
        auto avgB = tally.getAvgBounces();
        check_true(avgB.size() == 1u, "Trailing zero-ray bounce pruned by getAvgBounces");
    }
    std::cout << "[PASS] ConfigTally tests passed." << std::endl;

    // -------------------------------------------------------------------------
    // TEST 3: InputController (SDL3 Offscreen Events, Hotkeys, Mouse, Gamepad)
    // -------------------------------------------------------------------------
    std::cout << "[TEST 3] InputController SDL3 event routing & gamepad control..." << std::endl;
    {
        SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "offscreen");
        check_true(SDL_Init(SDL_INIT_VIDEO), "SDL_Init(SDL_INIT_VIDEO) with offscreen driver");

        Config icCfg;
        icCfg.headless = false;
        icCfg.enable_gui = false;
        icCfg.gamepad_deadzone = 0.20f;
        icCfg.gamepad_sensitivity = 1.5f;
        icCfg.gamepad_invert_y = false;

        Camera cam(glm::vec3(0.0f, 1.0f, 4.0f), glm::vec3(0.0f, 1.0f, 0.0f), 45.0f, 16.0f / 9.0f);
        InputController ic(nullptr, nullptr, icCfg);
        ic.setCamera(&cam);
        ic.setSceneTarget(glm::vec3(0.0f, 1.0f, 0.0f), 2.0f);

        int guiToggles = 0, fsToggles = 0, shots = 0, nextSc = 0, prevSc = 0;
        ic.setGuiToggleCallback([&]() { guiToggles++; });
        ic.setFullscreenToggleCallback([&]() { fsToggles++; });
        ic.setScreenshotCallback([&]() { shots++; });
        ic.setNextSceneCallback([&]() { nextSc++; });
        ic.setPrevSceneCallback([&]() { prevSc++; });

        auto sendKey = [&](SDL_Keycode key, bool down, bool repeat = false) {
            SDL_Event ev{};
            ev.type = down ? SDL_EVENT_KEY_DOWN : SDL_EVENT_KEY_UP;
            ev.key.key = key;
            ev.key.repeat = repeat;
            return ic.handleEvent(ev);
        };

        // Hotkeys & key-repeat suppression
        check_true(sendKey(SDLK_F1, true, false) && guiToggles == 1, "F1 fires GUI toggle");
        check_true(!sendKey(SDLK_F1, true, true) && guiToggles == 1, "Repeated F1 suppressed");
        check_true(sendKey(SDLK_F11, true, false) && fsToggles == 1, "F11 fires fullscreen toggle");
        check_true(sendKey(SDLK_F12, true, false) && shots == 1, "F12 fires screenshot");
        check_true(sendKey(SDLK_PRINTSCREEN, true, false) && shots == 2, "PrintScreen fires screenshot");
        check_true(sendKey(SDLK_N, true, false) && nextSc == 1, "N fires next scene");
        check_true(sendKey(SDLK_P, true, false) && prevSc == 1, "P fires prev scene");
        check_true(sendKey(SDLK_G, true, false) && guiToggles == 2, "G fires GUI toggle");

        // Mode toggling via TAB, ESC, and WINDOW_FOCUS_LOST
        check_true(!ic.isCameraMode(), "Starts in UI mode");
        check_true(sendKey(SDLK_TAB, true) && ic.isCameraMode(), "TAB enters camera mode");
        check_true(sendKey(SDLK_ESCAPE, true) && !ic.isCameraMode(), "ESC exits camera mode");
        check_true(sendKey(SDLK_TAB, true) && ic.isCameraMode(), "TAB re-enters camera mode");
        {
            SDL_Event focusEv{};
            focusEv.type = SDL_EVENT_WINDOW_FOCUS_LOST;
            ic.handleEvent(focusEv);
            check_true(!ic.isCameraMode(), "WINDOW_FOCUS_LOST exits camera mode");
        }

        // Mouse motion & wheel inside camera mode
        check_true(sendKey(SDLK_TAB, true) && ic.isCameraMode(), "Re-entered camera mode");
        float yawBefore = cam.getYaw();
        {
            SDL_Event mEv{};
            mEv.type = SDL_EVENT_MOUSE_MOTION;
            mEv.motion.xrel = 50.0f;
            mEv.motion.yrel = -20.0f;
            check_true(ic.handleEvent(mEv), "Mouse motion consumed in camera mode");
            check_true(cam.getYaw() > yawBefore, "Mouse motion rotates camera yaw");
        }
        float speedBefore = cam.getSpeed();
        {
            SDL_Event wEv{};
            wEv.type = SDL_EVENT_MOUSE_WHEEL;
            wEv.wheel.y = 1.0f;
            wEv.wheel.direction = SDL_MOUSEWHEEL_NORMAL;
            ic.handleEvent(wEv);
            check_true(cam.getSpeed() > speedBefore, "Normal wheel up increases speed");

            wEv.wheel.y = 1.0f;
            wEv.wheel.direction = SDL_MOUSEWHEEL_FLIPPED;
            ic.handleEvent(wEv);
            assert_near(cam.getSpeed(), speedBefore, 1e-3f, "Flipped wheel down restores speed");

            wEv.wheel.y = std::numeric_limits<float>::quiet_NaN();
            ic.handleEvent(wEv);
            check_true(std::isfinite(cam.getSpeed()), "NaN mouse wheel safely ignored");
        }

        // Gamepad axis deadzone + deflection + update()
        auto sendAxis = [&](SDL_GamepadAxis axis, Sint16 value) {
            SDL_Event ev{};
            ev.type = SDL_EVENT_GAMEPAD_AXIS_MOTION;
            ev.gaxis.axis = axis;
            ev.gaxis.value = value;
            return ic.handleEvent(ev);
        };
        auto sendBtn = [&](SDL_GamepadButton btn, bool down) {
            SDL_Event ev{};
            ev.type = down ? SDL_EVENT_GAMEPAD_BUTTON_DOWN : SDL_EVENT_GAMEPAD_BUTTON_UP;
            ev.gbutton.button = static_cast<Uint8>(btn);
            return ic.handleEvent(ev);
        };

        // Below 0.20 deadzone (~3000 / 32767 = 0.091) -> zero movement
        glm::vec3 posPreDead = cam.getPosition();
        cam.resetMoved();
        sendAxis(SDL_GAMEPAD_AXIS_LEFTY, -3000);
        ic.update(0.05f, 0.0f, false);
        assert_near(glm::length(cam.getPosition() - posPreDead), 0.0f, 1e-5f,
                    "Gamepad stick inside deadzone produces zero movement");

        // Above deadzone (left stick pushed forward -> value = -32767 -> m_gamepadLeftY = +1.0)
        sendAxis(SDL_GAMEPAD_AXIS_LEFTY, -32767);
        ic.update(0.1f, 0.0f, false);
        float baseMoveDist = glm::length(cam.getPosition() - posPreDead);
        check_true(baseMoveDist > 0.05f, "Full forward stick moves camera");

        // Right trigger sprint boost increases displacement vs baseline
        glm::vec3 posPreRt = cam.getPosition();
        sendAxis(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, 32767);
        ic.update(0.1f, 0.0f, false);
        float rtMoveDist = glm::length(cam.getPosition() - posPreRt);
        check_true(rtMoveDist > baseMoveDist * 1.8f, "Right trigger progressively boosts gamepad speed");

        // Reset stick & trigger; test right stick look & NORTH pitch reset
        sendAxis(SDL_GAMEPAD_AXIS_LEFTY, 0);
        sendAxis(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER, 0);
        float pitchPre = cam.getPitch();
        sendAxis(SDL_GAMEPAD_AXIS_RIGHTY, +32767);
        ic.update(0.05f, 0.0f, false);
        check_true(cam.getPitch() < pitchPre, "Right stick down-Y lowers pitch when invert_y == false");
        sendAxis(SDL_GAMEPAD_AXIS_RIGHTY, 0);
        sendBtn(SDL_GAMEPAD_BUTTON_NORTH, true);
        assert_near(cam.getPitch(), 0.0f, 1e-4f, "Gamepad NORTH levels camera pitch to 0");

        // Bumpers elevate/descend; WEST enters/exits orbit; EAST exits camera mode; START toggles
        float yPreBump = cam.getPosition().y;
        sendBtn(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, true);
        ic.update(0.1f, 0.0f, false);
        check_true(cam.getPosition().y > yPreBump, "Right shoulder elevates camera");
        sendBtn(SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER, false);

        sendBtn(SDL_GAMEPAD_BUTTON_WEST, true);
        ic.update(0.016f, 4.0f, true);
        check_true(cam.isOrbiting(), "Gamepad WEST enters orbit around hit point");
        sendBtn(SDL_GAMEPAD_BUTTON_WEST, false);
        ic.update(0.016f, 4.0f, true);
        check_true(!cam.isOrbiting(), "Releasing WEST exits orbit");

        sendBtn(SDL_GAMEPAD_BUTTON_EAST, true);
        check_true(!ic.isCameraMode(), "Gamepad EAST exits camera mode");
        sendBtn(SDL_GAMEPAD_BUTTON_START, true);
        check_true(ic.isCameraMode(), "Gamepad START toggles camera mode back on");

        SDL_Quit();
    }
    std::cout << "[PASS] InputController tests passed." << std::endl;

    // -------------------------------------------------------------------------
    // TEST 4: SceneRegistry Formatting & Directory Scanning
    // -------------------------------------------------------------------------
    std::cout << "[TEST 4] SceneRegistry formatting & directory scan filtering..." << std::endl;
    {
        SceneEntry e0{};
        check_true(e0.formatTriangles() == "0 tris", "formatTriangles(0)");
        e0.triangleCount = 512;
        check_true(e0.formatTriangles() == "512 tris", "formatTriangles(512)");
        e0.triangleCount = 2500;
        check_true(e0.formatTriangles() == "2.5K tris", "formatTriangles(2500)");
        e0.triangleCount = 2500000;
        check_true(e0.formatTriangles() == "2.50M tris", "formatTriangles(2.5M)");
        e0.triangleCount = 5000;
        e0.instancedTriangleCount = 20000;
        check_true(e0.formatTriangles() == "5000 Base / 20000 Inst", "formatTriangles instanced < 1M");
        e0.instancedTriangleCount = 2500000;
        check_true(e0.formatTriangles() == "5.0K Base / 2.50M Inst", "formatTriangles instanced >= 1M");

        e0.fileSizeBytes = 0;
        check_true(e0.formatFileSize().empty(), "formatFileSize(0) empty");
        e0.fileSizeBytes = 4096;
        check_true(e0.formatFileSize() == "4 KB", "formatFileSize(4096)");
        e0.fileSizeBytes = 2u * 1024u * 1024u;
        check_true(e0.formatFileSize() == "2.0 MB", "formatFileSize(2MB)");

        e0.label = "Test";
        e0.triangleCount = 100;
        e0.instancedTriangleCount = 0;
        e0.materialCount = 1;
        check_true(e0.formatComboPreview() == "Test  [100 tris, 1 mat]", "formatComboPreview singular mat");
        e0.materialCount = 3;
        check_true(e0.formatComboPreview() == "Test  [100 tris, 3 mats]", "formatComboPreview plural mats");

        check_true(SceneRegistry::formatSceneName("DamagedHelmet") == "Damaged Helmet", "camelCase split");
        check_true(SceneRegistry::formatSceneName("cornell_caustic") == "Cornell Caustic", "underscore split");
        check_true(SceneRegistry::formatSceneName("living-room") == "Living Room", "hyphen split");
        check_true(SceneRegistry::formatSceneName("BuickRiviera") == "Buick Riviera", "BuickRiviera special case");
        check_true(SceneRegistry::formatSceneName("classic-car") == "Buick Riviera", "classic-car special case");

        // Non-existent dir still includes the 5 built-in procedural scenes
        auto procOnly = SceneRegistry::scan("__nonexistent_scenes_dir__");
        check_true(procOnly.size() == 5u &&
                   procOnly[0].filepath == "cornell-box" &&
                   procOnly[1].filepath == "many-lights" &&
                   procOnly[2].filepath == "cyber-city" &&
                   procOnly[3].filepath == "infinity-mirror" &&
                   procOnly[4].filepath == "mineshaft",
                   "Non-existent scenesDir returns 5 built-in procedural scenes");

        // Scan temp dir with valid, stub (<100B), Scanlands, and missing-buffer glTF files
        const std::filesystem::path tmpScenes = std::filesystem::temp_directory_path() / "pathways_test_scene_registry";
        ScopedDirCleanup tmpScenesCleanup{ tmpScenes };
        {
            std::error_code ec;
            std::filesystem::remove_all(tmpScenes, ec);
            std::filesystem::create_directories(tmpScenes, ec);
        }

        // Embedded 1-triangle base64 glTF (> 100 bytes)
        // 3 vec3 floats * 3 = 36 bytes of zeroed positions -> base64 = 48 'A's
        const std::string validGltf = R"({
  "asset": { "version": "2.0" },
  "buffers": [ { "byteLength": 36, "uri": "data:application/octet-stream;base64,AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA" } ],
  "bufferViews": [ { "buffer": 0, "byteOffset": 0, "byteLength": 36 } ],
  "accessors": [ { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3" } ],
  "materials": [ { "name": "M" } ],
  "meshes": [ { "primitives": [ { "attributes": { "POSITION": 0 }, "material": 0 } ] } ]
})";
        {
            std::ofstream f(tmpScenes / "ValidAsset.gltf");
            f << validGltf;
        }
        {
            std::ofstream f(tmpScenes / "Scanlands.gltf");
            f << validGltf;
        }
        {
            std::ofstream f(tmpScenes / "TinyStub.gltf");
            f << "{}";
        }
        {
            std::ofstream f(tmpScenes / "MissingBin.gltf");
            f << R"({"asset":{"version":"2.0"},"buffers":[{"byteLength":36,"uri":"does_not_exist.bin"}],"bufferViews":[{"buffer":0,"byteOffset":0,"byteLength":36}],"accessors":[{"bufferView":0,"componentType":5126,"count":3,"type":"VEC3"}],"meshes":[{"primitives":[{"attributes":{"POSITION":0}}]}]})";
        }

        auto scanned = SceneRegistry::scan(tmpScenes.string());
        check_true(scanned.size() == 6u, "Only ValidAsset.gltf added alongside 5 procedural scenes");
        check_true(scanned[5].label == "Valid Asset" && scanned[5].triangleCount == 1u && scanned[5].materialCount == 1u,
                   "ValidAsset metadata populated accurately");
    }
    std::cout << "[PASS] SceneRegistry tests passed." << std::endl;

    // -------------------------------------------------------------------------
    // TEST 5: Config::parse Edge Cases & Rejection Paths
    // -------------------------------------------------------------------------
    std::cout << "[TEST 5] Config::parse edge flags & rejection checks..." << std::endl;
    {
        auto parseArgs = [](std::initializer_list<const char*> args) {
            std::vector<char*> argv;
            for (const char* a : args) argv.push_back(const_cast<char*>(a));
            return Config::parse(static_cast<int>(argv.size()), argv.data());
        };
        auto expectThrows = [&](std::initializer_list<const char*> args, const char* msg) {
            bool threw = false;
            try {
                (void)parseArgs(args);
            } catch (const std::runtime_error&) {
                threw = true;
            }
            check_true(threw, msg);
        };

        // --reference mode disables all biased shortcuts and selects RGBA32_SFLOAT
        Config ref = parseArgs({ "pathways", "--reference" });
        check_true(ref.reference_mode && ref.indirect_clamp == 0.0f &&
                   !ref.enable_restir && !ref.enable_tail_megakernel &&
                   !ref.enable_nrc && !ref.enable_caustics &&
                   !ref.enable_delta_unroll && !ref.streamline_secondary_shading &&
                   !ref.distance_clamping && ref.accum_format == AccumFormat::RGBA32_SFLOAT,
                   "--reference enforces unbiased ground-truth config");

        Config flags = parseArgs({
            "pathways", "--transparent-shadows", "--seed", "5000",
            "--camera-path", "tour.json", "--camera-path-loop", "--camera-path-speed", "2.5", "--camera-path-time", "1.25",
            "--early-bounce-threshold", "0.5%"
        });
        check_true(flags.transparent_shadows, "--transparent-shadows parsed");
        check_true(flags.rng_seed == (5000u & 0xFFFu), "--seed masked to 12 bits");
        check_true(flags.camera_path == "tour.json" && flags.camera_path_loop, "--camera-path & loop parsed");
        assert_near(flags.camera_path_speed, 2.5f, 1e-4f, "--camera-path-speed parsed");
        assert_near(flags.camera_path_freeze_time, 1.25f, 1e-4f, "--camera-path-time parsed");
        assert_near(flags.early_bounce_cutoff_threshold, 0.005f, 1e-6f, "--early-bounce-threshold 0.5% parsed");

        // Resolution presets & fullscreen override rules
        Config r1080 = parseArgs({ "pathways", "--res", "1080p" });
        check_true(r1080.width == 1920u && r1080.height == 1080u && r1080.custom_resolution && !r1080.fullscreen,
                   "--res 1080p sets 1920x1080 windowed");
        Config rDualUp = parseArgs({ "pathways", "--res=dualup", "--fullscreen" });
        check_true(rDualUp.width == 1280u && rDualUp.height == 2048u && rDualUp.fullscreen,
                   "--res=dualup --fullscreen preserves fullscreen");
        Config rCustom = parseArgs({ "pathways", "-r", "1600x900" });
        check_true(rCustom.width == 1600u && rCustom.height == 900u, "-r 1600x900 parsed");
        Config rNative = parseArgs({ "pathways", "--res", "1080", "--res", "native" });
        check_true(!rNative.custom_resolution && rNative.fullscreen, "--res native resets custom_resolution");

        // --dgc-execset / --no-dgc-execset env policy
        {
            ScopedEnvVarRestore envGuard("PATHWAYS_DGC_EXECSET");
            (void)parseArgs({ "pathways", "--dgc-execset" });
            check_true(Config::dgcExecSetRequested(), "--dgc-execset activates dgcExecSetRequested()");
            (void)parseArgs({ "pathways", "--no-dgc-execset" });
            check_true(!Config::dgcExecSetRequested(), "--no-dgc-execset clears dgcExecSetRequested()");
        }

        // Rejection tests
        expectThrows({ "pathways", "--pipeline", "foo" }, "--pipeline foo rejected");
        expectThrows({ "pathways", "--scaler", "unknown" }, "--scaler unknown rejected");
        expectThrows({ "pathways", "--mgpu-transfer", "staging" }, "--mgpu-transfer staging rejected");
        expectThrows({ "pathways", "--taa" }, "--taa rejected");
        expectThrows({ "pathways", "--restir-di" }, "--restir-di rejected");
        expectThrows({ "pathways", "--restir-gi" }, "--restir-gi rejected");
        expectThrows({ "pathways", "--gateway" }, "--gateway rejected");

        // Malformed camera args warn safely without crashing
        Config badCam = parseArgs({ "pathways", "--camera", "bad", "--camera-fov", "notanumber" });
        check_true(!badCam.camera_pos.has_value() && !badCam.camera_fov.has_value(),
                   "Malformed camera CLI inputs leave optionals unset");
    }
    std::cout << "[PASS] Config::parse edge & rejection tests passed." << std::endl;

    std::cout << "\nALL CORE SUBSYSTEMS UNIT TESTS PASSED!" << std::endl;
    return 0;
}
