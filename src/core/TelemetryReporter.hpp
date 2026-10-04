#pragma once

#include "core/ConfigTally.hpp"
#include "utils/ImageDumper.hpp"

#include <vector>
#include <string>
#include <memory>

namespace pathways {

class Engine;

class TelemetryReporter {
public:
    explicit TelemetryReporter(Engine* engine);
    ~TelemetryReporter() = default;

    TelemetryReporter(const TelemetryReporter&) = delete;
    TelemetryReporter& operator=(const TelemetryReporter&) = delete;

    void dumpOutputFiles();
    [[nodiscard]] FrameStats getStats() const;
    void recordFrameTally(double frameTimeMs, double primRtMs, double secRtMs, double tonemapMs,
                          const WavefrontStageSample* wfSample = nullptr);
    void printExecutionSummary() const;

    [[nodiscard]] const std::vector<ConfigStatsTally>& getConfigTallies() const noexcept { return m_configTallies; }
    [[nodiscard]] std::vector<ConfigStatsTally>& getConfigTallies() noexcept { return m_configTallies; }

private:
    Engine* m_engine = nullptr;
    std::vector<ConfigStatsTally> m_configTallies;
};

} // namespace pathways
