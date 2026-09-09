#include "core/QualityGovernor.hpp"
#include <thread>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

#if defined(_WIN32)
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>

    static void highPrecisionSleep(std::chrono::nanoseconds duration) {
        if (duration <= std::chrono::nanoseconds::zero()) return;
        static HANDLE hTimer = CreateWaitableTimerExW(NULL, NULL, 0x00000002 /*CREATE_WAITABLE_TIMER_HIGH_RESOLUTION*/, TIMER_ALL_ACCESS);
        if (hTimer) {
            LARGE_INTEGER dueTime;
            dueTime.QuadPart = -(duration.count() / 100);
            if (SetWaitableTimer(hTimer, &dueTime, 0, NULL, NULL, FALSE)) {
                WaitForSingleObject(hTimer, INFINITE);
                return;
            }
        }
        std::this_thread::sleep_for(duration);
    }
#endif

namespace pathways {

QualityGovernor::~QualityGovernor() = default;

QualityGovernor::QualityGovernor(const GovernorConfig& config) {
    init(config);
}

void QualityGovernor::init(const GovernorConfig& config) {
    m_config = config;
    m_state.currentSpp = std::clamp(1u, m_config.minSpp, m_config.maxSpp);
    m_state.currentBounces = std::clamp(4u, m_config.minBounces, m_config.maxBounces);
    m_state.effectiveSpp = static_cast<float>(m_state.currentSpp);
    m_state.fractionalSpp = 0.0f;
    m_state.primSpp = m_state.currentSpp;
    m_state.secSpp = 0;
    m_state.active = m_config.enabled && (m_config.targetFps > 0);
    m_state.warmUpFrames = 0;
    m_emaRtMs = 0.0f;
    m_cooldown = 0;
    m_initialized = true;

    if (m_config.targetFps > 0) {
        m_state.targetBudgetMs = 1000.0f / static_cast<float>(m_config.targetFps);
        m_state.targetRtBudgetMs = std::max(0.5f, m_state.targetBudgetMs - 0.8f);
    } else {
        m_state.targetBudgetMs = 0.0f;
        m_state.targetRtBudgetMs = 0.0f;
    }
}

void QualityGovernor::setTargetFps(uint32_t fps) {
    m_config.targetFps = fps;
    m_state.active = m_config.enabled && (m_config.targetFps > 0);
    if (fps > 0) {
        m_state.targetBudgetMs = 1000.0f / static_cast<float>(fps);
        m_state.targetRtBudgetMs = std::max(0.5f, m_state.targetBudgetMs - 0.8f);
    } else {
        m_state.targetBudgetMs = 0.0f;
        m_state.targetRtBudgetMs = 0.0f;
    }
    m_cooldown = 0;
}

void QualityGovernor::setEnabled(bool enabled) {
    m_config.enabled = enabled;
    m_state.active = m_config.enabled && (m_config.targetFps > 0);
    m_cooldown = 0;
}

void QualityGovernor::setBounds(uint32_t minSpp, uint32_t maxSpp, uint32_t minBounces, uint32_t maxBounces) {
    m_config.minSpp = std::max(1u, minSpp);
    m_config.maxSpp = std::max(m_config.minSpp, maxSpp);
    m_config.minBounces = std::max(1u, minBounces);
    m_config.maxBounces = std::max(m_config.minBounces, maxBounces);

    m_state.effectiveSpp = std::clamp(m_state.effectiveSpp, static_cast<float>(m_config.minSpp), static_cast<float>(m_config.maxSpp));
    m_state.currentSpp = static_cast<uint32_t>(std::floor(m_state.effectiveSpp));
    m_state.fractionalSpp = m_state.effectiveSpp - static_cast<float>(m_state.currentSpp);
    m_state.currentBounces = std::clamp(m_state.currentBounces, m_config.minBounces, m_config.maxBounces);
}

void QualityGovernor::update(float measuredRtMs, float fixedOverheadMs, bool cameraMoving, bool isMgpuSampleParallel) {
    if (!m_initialized || !m_config.enabled || m_config.targetFps == 0) {
        m_state.active = false;
        m_state.effectiveSpp = static_cast<float>(m_state.currentSpp);
        m_state.fractionalSpp = 0.0f;
        return;
    }

    m_state.active = true;
    m_state.warmUpFrames++;

    // Total target budget and available ray tracing budget
    float targetTotalMs = 1000.0f / static_cast<float>(m_config.targetFps);
    float safeOverhead = std::clamp(fixedOverheadMs, 0.2f, 2.5f);
    float rtBudget = std::max(0.5f, targetTotalMs - safeOverhead);

    m_state.targetBudgetMs = targetTotalMs;
    m_state.targetRtBudgetMs = rtBudget;
    m_state.lastFrameTimeMs = measuredRtMs + safeOverhead;

    if (measuredRtMs <= 0.001f) {
        return;
    }

    // Filter measured ray tracing duration via Exponential Moving Average
    if (m_emaRtMs <= 0.001f) {
        m_emaRtMs = measuredRtMs;
    } else {
        // Responsive during warmup, stable during steady state
        float alpha = (m_state.warmUpFrames <= 10) ? 0.35f : 0.15f;
        m_emaRtMs = alpha * measuredRtMs + (1.0f - alpha) * m_emaRtMs;
    }
    m_state.filteredRtMs = m_emaRtMs;

    if (m_cooldown > 0) {
        m_cooldown--;
    }

    // Continuous Adaptive Sample Rate Governor
    // Target 94% of available RT budget to absorb OS compositing jitter
    float targetRtMs = 0.94f * rtBudget;
    float currentEff = std::max(1.0f, m_state.effectiveSpp);
    float timePerSpp = std::max(0.1f, m_emaRtMs / currentEff);
    float errorMs = targetRtMs - m_emaRtMs;

    if (m_emaRtMs > 1.02f * rtBudget) {
        // Fast emergency step-down when overshooting budget
        float delta = errorMs / timePerSpp;
        m_state.effectiveSpp = std::clamp(m_state.effectiveSpp + delta,
                                          static_cast<float>(m_config.minSpp),
                                          static_cast<float>(m_config.maxSpp));
        m_cooldown = 4;
    } else if (m_cooldown == 0) {
        // Smooth proportional adjustment to continuously track target budget
        float gain = (m_state.warmUpFrames <= 15) ? 0.35f : 0.12f;
        float delta = gain * (errorMs / timePerSpp);
        m_state.effectiveSpp = std::clamp(m_state.effectiveSpp + delta,
                                          static_cast<float>(m_config.minSpp),
                                          static_cast<float>(m_config.maxSpp));
    }

    m_state.currentSpp = static_cast<uint32_t>(std::floor(m_state.effectiveSpp));
    if (m_state.currentSpp >= m_config.maxSpp) {
        m_state.currentSpp = m_config.maxSpp;
        m_state.fractionalSpp = 0.0f;
    } else {
        m_state.fractionalSpp = m_state.effectiveSpp - static_cast<float>(m_state.currentSpp);
    }

    // Secondary fine vernier: modulate bounces only at SPP extremes
    if (m_state.effectiveSpp >= static_cast<float>(m_config.maxSpp) && m_emaRtMs < 0.85f * rtBudget && m_cooldown == 0) {
        if (m_state.currentBounces < m_config.maxBounces) {
            m_state.currentBounces++;
            m_cooldown = 8;
        }
    } else if (m_state.effectiveSpp <= static_cast<float>(m_config.minSpp) + 0.05f && m_emaRtMs > 0.98f * rtBudget) {
        if (m_state.currentBounces > m_config.minBounces) {
            m_state.currentBounces--;
            m_cooldown = 6;
        }
    }

    // Assign Multi-GPU Sample Parallel split
    if (isMgpuSampleParallel && m_state.currentSpp > 1) {
        m_state.primSpp = (m_state.currentSpp + 1) / 2;
        m_state.secSpp = m_state.currentSpp / 2;
    } else {
        m_state.primSpp = m_state.currentSpp;
        m_state.secSpp = 0;
    }

    // Compute headroom
    m_state.headroomMs = rtBudget - m_emaRtMs;
    m_state.headroomPercent = (rtBudget > 0.001f) ? (m_state.headroomMs / rtBudget) * 100.0f : 0.0f;
}

void QualityGovernor::paceFrame(std::chrono::high_resolution_clock::time_point frameStart) {
    if (m_config.targetFps == 0) {
        return;
    }

    auto targetInterval = std::chrono::duration<double, std::milli>(1000.0 / static_cast<double>(m_config.targetFps));
    auto deadline = frameStart + std::chrono::duration_cast<std::chrono::nanoseconds>(targetInterval);
    auto now = std::chrono::high_resolution_clock::now();

    if (now < deadline) {
        auto remaining = deadline - now;
        // Sleep for the coarse duration (leaving 250 microseconds for precision spinning)
        if (remaining > std::chrono::microseconds(350)) {
#if defined(_WIN32)
            highPrecisionSleep(remaining - std::chrono::microseconds(250));
#else
            std::this_thread::sleep_for(remaining - std::chrono::microseconds(250));
#endif
        }

        // High-precision spin-lock tail
        while (std::chrono::high_resolution_clock::now() < deadline) {
            #if defined(__x86_64__) || defined(_M_X64)
            _mm_pause();
            #endif
        }
    }
}

} // namespace pathways
