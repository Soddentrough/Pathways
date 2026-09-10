#pragma once

#include <cstdint>
#include <chrono>
#include <algorithm>
#include <array>

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
    float effectiveSpp = 1.0f;       // Backwards-compatible SPP representation
    float fractionalSpp = 0.0f;      // Kept for interface compatibility (always 0.0 to prevent SIMD divergence)
    uint32_t primSpp = 1;            // GPU 0 sample count
    uint32_t secSpp = 0;             // GPU 1 sample count
    float targetBudgetMs = 0.0f;     // 1000 / targetFps
    float targetRtBudgetMs = 0.0f;   // Available RT budget (minus fixed tonemap/UI overhead)
    float filteredRtMs = 0.0f;       // Filtered ray tracing time (ms)
    float lastFrameTimeMs = 0.0f;    // Last measured total frame time (ms)
    float headroomMs = 0.0f;         // Available RT headroom (ms)
    float headroomPercent = 0.0f;    // Headroom percentage
    float costPerSpp = 0.0f;         // Filtered normalized cost per sample (ms)
    float predictedRtMs = 0.0f;      // Model-predicted RT time for current workload
    bool active = false;             // Whether governor is currently regulating
    bool cameraMoving = false;       // Whether camera was moving during evaluation
    uint32_t warmUpFrames = 0;       // Warmup frame counter
};

class QualityGovernor {
public:
    QualityGovernor() = default;
    ~QualityGovernor();
    explicit QualityGovernor(const GovernorConfig& config);

    void init(const GovernorConfig& config);

    // Records the workload dispatched into a specific in-flight slot (0 or 1)
    void recordDispatch(uint32_t slot, uint32_t spp, uint32_t bounces);

    // Updates governor state using GPU measurements corresponding to the completed in-flight slot
    void update(uint32_t slot, float measuredRtMs, float fixedOverheadMs, bool cameraMoving, bool isMgpuSampleParallel);

    // Legacy overload without slot (assumes slot 0)
    void update(float measuredRtMs, float fixedOverheadMs, bool cameraMoving, bool isMgpuSampleParallel) {
        update(0, measuredRtMs, fixedOverheadMs, cameraMoving, isMgpuSampleParallel);
    }

    // High-resolution CPU frame pacer / sleep until deadline
    void paceFrame(std::chrono::high_resolution_clock::time_point frameStart);

    void setTargetFps(uint32_t fps);
    void setEnabled(bool enabled);
    void setBounds(uint32_t minSpp, uint32_t maxSpp, uint32_t minBounces, uint32_t maxBounces);

    const GovernorState& getState() const { return m_state; }
    const GovernorConfig& getConfig() const { return m_config; }
    float getEffectiveSpp() const { return static_cast<float>(m_state.currentSpp); }
    float getFractionalSpp() const { return 0.0f; }

private:
    float predictTime(uint32_t spp, uint32_t bounces, bool isMgpu) const;
    static float getBounceMultiplier(uint32_t bounces);

    GovernorConfig m_config;
    GovernorState m_state;

    struct SlotRecord {
        uint32_t spp = 1;
        uint32_t bounces = 4;
        bool valid = false;
    };
    std::array<SlotRecord, 2> m_slotRecords{};

    float m_emaCostPerSpp = 0.0f;
    uint32_t m_cooldown = 0;
    uint32_t m_failedSpp = 0;
    uint32_t m_failedBounces = 0;
    uint32_t m_failLockoutFrames = 0;
    bool m_initialized = false;
};

} // namespace pathways
