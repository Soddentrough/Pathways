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
    m_state.cameraMoving = false;
    m_state.warmUpFrames = 0;
    m_state.costPerSpp = 0.0f;
    m_state.predictedRtMs = 0.0f;
    m_emaCostPerSpp = 0.0f;
    m_cooldown = 0;
    m_initialized = true;

    for (auto& rec : m_slotRecords) {
        rec.spp = m_state.currentSpp;
        rec.bounces = m_state.currentBounces;
        rec.valid = false;
    }

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

    m_state.currentSpp = std::clamp(m_state.currentSpp, m_config.minSpp, m_config.maxSpp);
    m_state.currentBounces = std::clamp(m_state.currentBounces, m_config.minBounces, m_config.maxBounces);
    m_state.effectiveSpp = static_cast<float>(m_state.currentSpp);
}

void QualityGovernor::recordDispatch(uint32_t slot, uint32_t spp, uint32_t bounces) {
    uint32_t s = slot % 2;
    m_slotRecords[s].spp = spp;
    m_slotRecords[s].bounces = bounces;
    m_slotRecords[s].valid = true;
}

float QualityGovernor::getBounceMultiplier(uint32_t bounces) {
    // In real path-traced interior scenes, secondary bounces require significant recursive traversal.
    // Calibrated model relative to nominal 4 bounces (1.00x):
    // 1: 0.51x, 2: 0.68x, 3: 0.84x, 4: 1.00x, 5: 1.25x, 6: 1.50x, 7: 1.75x, 8: 2.00x
    bounces = std::clamp(bounces, 1u, 16u);
    if (bounces <= 4) {
        return 0.35f + 0.1625f * static_cast<float>(bounces);
    }
    return 1.00f + 0.25f * static_cast<float>(bounces - 4);
}

float QualityGovernor::predictTime(uint32_t spp, uint32_t bounces, bool isMgpu) const {
    if (m_emaCostPerSpp <= 0.001f) {
        return 0.0f;
    }
    uint32_t effectiveSppPerGpu = (isMgpu && spp > 1) ? ((spp + 1) / 2) : spp;
    float bounceMult = getBounceMultiplier(bounces);
    return static_cast<float>(effectiveSppPerGpu) * m_emaCostPerSpp * bounceMult;
}

void QualityGovernor::update(uint32_t slot, float measuredRtMs, float fixedOverheadMs, bool cameraMoving, bool isMgpuSampleParallel) {
    if (!m_initialized || !m_config.enabled || m_config.targetFps == 0) {
        m_state.active = false;
        m_state.effectiveSpp = static_cast<float>(m_state.currentSpp);
        m_state.fractionalSpp = 0.0f;
        return;
    }

    m_state.active = true;
    m_state.cameraMoving = cameraMoving;
    m_state.warmUpFrames++;

    float targetTotalMs = 1000.0f / static_cast<float>(m_config.targetFps);
    float safeOverhead = std::clamp(fixedOverheadMs, 0.2f, 2.5f);
    float rtBudget = std::max(0.5f, targetTotalMs - safeOverhead);

    m_state.targetBudgetMs = targetTotalMs;
    m_state.targetRtBudgetMs = rtBudget;
    m_state.lastFrameTimeMs = measuredRtMs + safeOverhead;

    if (measuredRtMs <= 0.001f) {
        return;
    }

    // 1. In-flight Latency Compensation: Retrieve the exact dispatch that produced measuredRtMs
    uint32_t s = slot % 2;
    uint32_t dispSpp = m_slotRecords[s].valid ? m_slotRecords[s].spp : m_state.currentSpp;
    uint32_t dispBounces = m_slotRecords[s].valid ? m_slotRecords[s].bounces : m_state.currentBounces;

    uint32_t dispGpuSpp = (isMgpuSampleParallel && dispSpp > 1) ? ((dispSpp + 1) / 2) : dispSpp;
    dispGpuSpp = std::max(1u, dispGpuSpp);
    float dispBounceMult = getBounceMultiplier(dispBounces);

    // Compute normalized cost per sample (at nominal 4 bounces)
    float sampleCost = measuredRtMs / (static_cast<float>(dispGpuSpp) * dispBounceMult);
    if (sampleCost > 0.01f && sampleCost < 200.0f) {
        if (m_emaCostPerSpp <= 0.001f) {
            m_emaCostPerSpp = sampleCost;
        } else {
            float alpha = (m_state.warmUpFrames <= 10) ? 0.35f : 0.15f;
            m_emaCostPerSpp = alpha * sampleCost + (1.0f - alpha) * m_emaCostPerSpp;
        }
    }
    m_state.costPerSpp = m_emaCostPerSpp;
    m_state.filteredRtMs = measuredRtMs;

    if (m_failLockoutFrames > 0) {
        m_failLockoutFrames--;
    }

    // Pipeline Drain Protection: If cooling down from a recent change, do not take action.
    // This completely prevents in-flight cascading downscales.
    if (m_cooldown > 0) {
        m_cooldown--;
        m_state.predictedRtMs = predictTime(m_state.currentSpp, m_state.currentBounces, isMgpuSampleParallel);
        m_state.effectiveSpp = static_cast<float>(m_state.currentSpp);
        m_state.fractionalSpp = 0.0f;
        if (isMgpuSampleParallel && m_state.currentSpp > 1) {
            m_state.primSpp = (m_state.currentSpp + 1) / 2;
            m_state.secSpp = m_state.currentSpp / 2;
        } else {
            m_state.primSpp = m_state.currentSpp;
            m_state.secSpp = 0;
        }
        m_state.headroomMs = rtBudget - m_state.predictedRtMs;
        m_state.headroomPercent = (rtBudget > 0.001f) ? (m_state.headroomMs / rtBudget) * 100.0f : 0.0f;
        return;
    }

    // 2. Safe Headroom Target Line: Require 15% margin for upgrades
    float targetRtMs = 0.85f * rtBudget;
    uint32_t nominalBounces = std::clamp(4u, m_config.minBounces, m_config.maxBounces);

    // 3. Emergency Downscale if overshooting budget
    if (measuredRtMs > 1.01f * rtBudget) {
        // Record this configuration as failed to prevent hunting
        m_failedSpp = dispSpp;
        m_failedBounces = dispBounces;
        m_failLockoutFrames = 90; // Lock out failed config for 90 frames (~1.5s)

        // If bounces were elevated above 4, step down bounces first
        if (m_state.currentBounces > nominalBounces) {
            m_state.currentBounces--;
        } else if (m_state.currentSpp > m_config.minSpp) {
            m_state.currentSpp--;
        } else if (m_state.currentBounces > m_config.minBounces) {
            m_state.currentBounces--;
        }

        m_cooldown = 2; // Drain in-flight frames before any further decision
    } else if (m_emaCostPerSpp > 0.001f) {
        auto isConfigAllowed = [&](uint32_t spp, uint32_t bounces) -> bool {
            if (m_failLockoutFrames > 0 && spp >= m_failedSpp && bounces >= m_failedBounces) {
                return false;
            }
            return true;
        };

        // 4. Model-Predictive Optimal Configuration Search
        // User Policy:
        // - Adaptive SPP changes primary rays (SPP) and leaves bounces at 4.
        // - If we are at 16 SPP (or maxSpp) and execution time remains lower than target FPS,
        //   we can then also increase bounces.
        uint32_t candidateSpp = m_config.minSpp;
        for (uint32_t spp = m_config.minSpp; spp <= m_config.maxSpp; ++spp) {
            if (!isConfigAllowed(spp, nominalBounces)) break;
            if (predictTime(spp, nominalBounces, isMgpuSampleParallel) <= targetRtMs) {
                candidateSpp = spp;
            } else {
                break;
            }
        }

        uint32_t candidateBounces = nominalBounces;
        // If at maximum SPP (>= 16 SPP or >= maxSpp) and execution time remains lower than target budget,
        // we can then also increase bounces beyond 4 up to maxBounces.
        if (candidateSpp >= 16u || candidateSpp >= m_config.maxSpp) {
            for (uint32_t b = nominalBounces + 1; b <= m_config.maxBounces; ++b) {
                if (!isConfigAllowed(candidateSpp, b)) break;
                if (predictTime(candidateSpp, b, isMgpuSampleParallel) <= targetRtMs) {
                    candidateBounces = b;
                } else {
                    break;
                }
            }
        } else if (candidateSpp == m_config.minSpp && predictTime(m_config.minSpp, nominalBounces, isMgpuSampleParallel) > targetRtMs) {
            // If even minSpp with nominal bounces exceeds budget, step down bounces toward minBounces
            for (uint32_t b = nominalBounces; b >= m_config.minBounces; --b) {
                if (predictTime(m_config.minSpp, b, isMgpuSampleParallel) <= targetRtMs || b == m_config.minBounces) {
                    candidateBounces = b;
                    break;
                }
            }
        }

        uint32_t bestSpp = candidateSpp;
        uint32_t bestBounces = candidateBounces;

        // Apply rate-limiting with fast warmup
        if (bestSpp > m_state.currentSpp) {
            float predictedNext = predictTime(m_state.currentSpp + 1, m_state.currentBounces, isMgpuSampleParallel);
            if (m_state.warmUpFrames <= 20 || predictedNext < 0.65f * targetRtMs) {
                uint32_t step = std::min(bestSpp - m_state.currentSpp, 2u);
                m_state.currentSpp += step;
                m_cooldown = 1;
            } else {
                m_state.currentSpp++;
                m_cooldown = 4;
            }
        } else if (bestBounces > m_state.currentBounces) {
            m_state.currentBounces++;
            m_cooldown = 4;
        } else if (bestSpp < m_state.currentSpp) {
            m_state.currentSpp--;
            m_cooldown = 2;
        } else if (bestBounces < m_state.currentBounces) {
            m_state.currentBounces--;
            m_cooldown = 2;
        }
    }

    m_state.currentSpp = std::clamp(m_state.currentSpp, m_config.minSpp, m_config.maxSpp);
    m_state.currentBounces = std::clamp(m_state.currentBounces, m_config.minBounces, m_config.maxBounces);
    m_state.effectiveSpp = static_cast<float>(m_state.currentSpp);
    m_state.fractionalSpp = 0.0f;
    m_state.predictedRtMs = predictTime(m_state.currentSpp, m_state.currentBounces, isMgpuSampleParallel);

    // Multi-GPU Sample Parallel balancing
    if (isMgpuSampleParallel && m_state.currentSpp > 1) {
        m_state.primSpp = (m_state.currentSpp + 1) / 2;
        m_state.secSpp = m_state.currentSpp / 2;
    } else {
        m_state.primSpp = m_state.currentSpp;
        m_state.secSpp = 0;
    }

    // Telemetry Headroom
    m_state.headroomMs = rtBudget - m_state.predictedRtMs;
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
        // Coarse sleep (leave 250 microseconds for precision spin-locking)
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
