#pragma once

#include <cstdint>
#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>

namespace pathways {

class VulkanContext;
struct FrameStats;

/**
 * @brief Subsystem managing host platform telemetry, background GPU sensor polling,
 *        PCIe link status, and asynchronous telemetry dataset exports.
 */
class HwMonitor {
public:
    using SecondaryContextGetter = std::function<VulkanContext*()>;

    explicit HwMonitor(VulkanContext* primaryContext, SecondaryContextGetter secContextGetter = nullptr);
    ~HwMonitor();

    HwMonitor(const HwMonitor&) = delete;
    HwMonitor& operator=(const HwMonitor&) = delete;

    void start();
    void stop();

    void sampleSensors();
    void refreshPciStatus();

    [[nodiscard]] uint32_t getGpu0ClockMhz() const noexcept { return m_gpu0ClockMhz.load(std::memory_order_relaxed); }
    [[nodiscard]] uint32_t getGpu0TempC() const noexcept { return m_gpu0TempC.load(std::memory_order_relaxed); }
    [[nodiscard]] uint32_t getGpu1ClockMhz() const noexcept { return m_gpu1ClockMhz.load(std::memory_order_relaxed); }
    [[nodiscard]] uint32_t getGpu1TempC() const noexcept { return m_gpu1TempC.load(std::memory_order_relaxed); }

    static const std::string& getOsName();
    static const std::string& getCpuModel();
    static uint64_t getTotalRamMb();

    std::string exportTelemetry(const std::string& customPath, const FrameStats& stats);

private:
    VulkanContext* m_primaryContext = nullptr;
    SecondaryContextGetter m_getSecondaryContext;

    std::atomic<bool> m_running{false};
    std::thread m_thread;
    std::mutex m_mutex;
    std::condition_variable m_cv;

    std::atomic<uint32_t> m_gpu0ClockMhz{0};
    std::atomic<uint32_t> m_gpu0TempC{0};
    std::atomic<uint32_t> m_gpu1ClockMhz{0};
    std::atomic<uint32_t> m_gpu1TempC{0};

    std::thread m_telemetryWorker;
};

} // namespace pathways
