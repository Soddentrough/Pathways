#pragma once

#include <cstdint>
#include <chrono>
#include <algorithm>

namespace pathways {

struct GovernorConfig {
    uint32_t targetFps = 0;          // 0 = uncapped
    bool enabled = false;            // Whether dynamic regulation is active
    uint32_t minSpp = 1;             // Minimum allowable SPP
    uint32_t maxSpp = 16;            // Maximum allowable SPP
    uint32_t minBounces = 2;         // Minimum allowable bounce depth
    uint32_t maxBounces = 8;         // Maximum allowable bounce depth
};

struct GovernorState {
    uint32_t currentSpp = 1;
    uint32_t currentBounces = 4;
    float effectiveSpp = 1.0f;       // Continuous dynamic sample rate (e.g. 1.45 SPP)
    float fractionalSpp = 0.0f;      // Fractional part for halftone blue-noise sampling [0.0, 1.0)
    uint32_t primSpp = 1;
    uint32_t secSpp = 0;
    float targetBudgetMs = 0.0f;
    float targetRtBudgetMs = 0.0f;
    float filteredRtMs = 0.0f;
    float lastFrameTimeMs = 0.0f;
    float headroomMs = 0.0f;
    float headroomPercent = 0.0f;
    bool active = false;
    uint32_t warmUpFrames = 0;
};

class QualityGovernor {
public:
    QualityGovernor() = default;
    explicit QualityGovernor(const GovernorConfig& config);

    void init(const GovernorConfig& config);
    void update(float measuredRtMs, float fixedOverheadMs, bool cameraMoving, bool isMgpuSampleParallel);
    void paceFrame(std::chrono::high_resolution_clock::time_point frameStart);

    void setTargetFps(uint32_t fps);
    void setEnabled(bool enabled);
    void setBounds(uint32_t minSpp, uint32_t maxSpp, uint32_t minBounces, uint32_t maxBounces);

    const GovernorState& getState() const { return m_state; }
    const GovernorConfig& getConfig() const { return m_config; }
    float getEffectiveSpp() const { return m_state.effectiveSpp; }
    float getFractionalSpp() const { return m_state.fractionalSpp; }

private:
    GovernorConfig m_config;
    GovernorState m_state;

    float m_emaRtMs = 0.0f;
    uint32_t m_cooldown = 0;
    bool m_initialized = false;
};

} // namespace pathways
