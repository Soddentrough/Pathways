#include "core/QualityGovernor.hpp"
#include <thread>
#include <cmath>

#if defined(__x86_64__) || defined(_M_X64)
    #include <immintrin.h>
#endif

namespace pathways {

QualityGovernor::QualityGovernor(const GovernorConfig& config) {
    init(config);
}

void QualityGovernor::init(const GovernorConfig& config) {
    m_config = config;
    m_state.currentSpp = std::clamp(1u, m_config.minSpp, m_config.maxSpp);
    m_state.currentBounces = std::clamp(4u, m_config.minBounces, m_config.maxBounces);
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

    m_state.currentSpp = std::clamp(m_state.currentSpp, m_config.minSpp, m_config.maxSpp);
    m_state.currentBounces = std::clamp(m_state.currentBounces, m_config.minBounces, m_config.maxBounces);
}

void QualityGovernor::update(float measuredRtMs, float fixedOverheadMs, bool cameraMoving, bool isMgpuSampleParallel) {
    if (!m_initialized || !m_config.enabled || m_config.targetFps == 0) {
        m_state.active = false;
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
        // More responsive EMA during first 10 frames of warmup, then smoother
        float alpha = (m_state.warmUpFrames <= 10) ? 0.35f : 0.15f;
        m_emaRtMs = alpha * measuredRtMs + (1.0f - alpha) * m_emaRtMs;
    }
    m_state.filteredRtMs = m_emaRtMs;

    if (m_cooldown > 0) {
        m_cooldown--;
    }

    // 1. Emergency Downgrade Threshold: if RT exceeds 95% of budget, step down immediately
    if (m_emaRtMs > 0.95f * rtBudget || measuredRtMs > 1.05f * rtBudget) {
        if (m_state.currentBounces > m_config.minBounces + 1) {
            m_state.currentBounces--;
            m_cooldown = 4;
        } else if (m_state.currentSpp > m_config.minSpp) {
            m_state.currentSpp--;
            // Slightly lift bounces on SPP drop to soften visual step
            m_state.currentBounces = std::min(m_config.maxBounces, m_state.currentBounces + 1);
            m_cooldown = 8;
        } else if (m_state.currentBounces > m_config.minBounces) {
            m_state.currentBounces--;
            m_cooldown = 4;
        }
    }
    // 2. Conservative Upgrade: only when cooldown expired and projected time is safely under budget
    else if (m_cooldown == 0) {
        // Project RT time for next SPP step
        float projectedNextSppRt = m_emaRtMs * (static_cast<float>(m_state.currentSpp + 1) / static_cast<float>(m_state.currentSpp));

        // In continuous camera motion, prioritize SPP over bounce depth for noise reduction
        if (m_state.currentSpp < m_config.maxSpp && projectedNextSppRt < 0.85f * rtBudget) {
            m_state.currentSpp++;
            m_cooldown = (m_state.warmUpFrames <= 15) ? 5 : 12;
        }
        // If SPP cannot be increased without overshooting, modulate bounce depth (fine vernier)
        else if (m_state.currentBounces < m_config.maxBounces && m_emaRtMs < 0.88f * rtBudget) {
            m_state.currentBounces++;
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
    if (!m_config.enabled || m_config.targetFps == 0) {
        return;
    }

    auto targetInterval = std::chrono::duration<double, std::milli>(1000.0 / static_cast<double>(m_config.targetFps));
    auto deadline = frameStart + std::chrono::duration_cast<std::chrono::nanoseconds>(targetInterval);
    auto now = std::chrono::high_resolution_clock::now();

    if (now < deadline) {
        auto remaining = deadline - now;
        // Sleep for the coarse duration (leaving 250 microseconds for precision spinning)
        if (remaining > std::chrono::microseconds(350)) {
            std::this_thread::sleep_for(remaining - std::chrono::microseconds(250));
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
