#include "HwMonitor.hpp"
#include "vulkan/VulkanContext.hpp"
#include "core/Logger.hpp"
#include "utils/ImageDumper.hpp"

#include <fstream>
#include <chrono>

#ifdef _WIN32
    #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
    #endif
    #include <windows.h>
#endif
#ifdef __linux__
    #include <sys/utsname.h>
#endif

namespace pathways {

namespace {

uint64_t readSysfsUint64(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) return 0;
    uint64_t val = 0;
    if (file >> val) return val;
    return 0;
}

std::string queryOperatingSystem() {
    std::ifstream osRelease("/etc/os-release");
    if (osRelease.is_open()) {
        std::string line;
        while (std::getline(osRelease, line)) {
            if (line.starts_with("PRETTY_NAME=")) {
                std::string val = line.substr(12);
                if (val.size() >= 2 && val.front() == '"' && val.back() == '"') {
                    val = val.substr(1, val.size() - 2);
                }
                return val;
            }
        }
    }
#ifdef __linux__
    struct utsname uts{};
    if (uname(&uts) == 0) {
        return std::string(uts.sysname) + " " + uts.release + " (" + uts.machine + ")";
    }
#endif
    return "Linux";
}

std::string queryCpuModel() {
    std::ifstream cpuInfo("/proc/cpuinfo");
    if (cpuInfo.is_open()) {
        std::string line;
        while (std::getline(cpuInfo, line)) {
            if (line.starts_with("model name")) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    size_t first = line.find_first_not_of(" \t", colon + 1);
                    if (first != std::string::npos) {
                        return line.substr(first);
                    }
                }
            }
        }
    }
    return "AMD Threadripper Processor";
}

uint64_t queryTotalRamMB() {
    std::ifstream memInfo("/proc/meminfo");
    if (memInfo.is_open()) {
        std::string line;
        while (std::getline(memInfo, line)) {
            if (line.starts_with("MemTotal:")) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    uint64_t kb = std::strtoull(line.c_str() + colon + 1, nullptr, 10);
                    return kb / 1024;
                }
            }
        }
    }
    return 65536;
}

} // namespace

HwMonitor::HwMonitor(VulkanContext* primaryContext, SecondaryContextGetter secContextGetter)
    : m_primaryContext(primaryContext), m_getSecondaryContext(std::move(secContextGetter))
{
}

HwMonitor::~HwMonitor() {
    stop();
    if (m_telemetryWorker.joinable()) {
        m_telemetryWorker.join();
    }
}

void HwMonitor::start() {
    if (m_running.load()) return;
    m_running = true;
    m_thread = std::thread([this]() {
        sampleSensors();
        while (m_running) {
            std::unique_lock<std::mutex> lock(m_mutex);
            if (m_cv.wait_for(lock, std::chrono::milliseconds(2000), [this] { return !m_running.load(); })) {
                break;
            }
            sampleSensors();
        }
    });
}

void HwMonitor::stop() {
    if (m_running.load()) {
        m_running = false;
        m_cv.notify_all();
        if (m_thread.joinable()) {
            m_thread.join();
        }
    }
}

void HwMonitor::sampleSensors() {
    // GPU 0
    if (m_primaryContext) {
        const auto& pciInfo = m_primaryContext->getPciLinkInfo();
        if (!pciInfo.hwmonPath.empty()) {
            uint64_t rawFreq = readSysfsUint64(pciInfo.hwmonPath + "/freq1_input");
            uint64_t rawTemp = readSysfsUint64(pciInfo.hwmonPath + "/temp1_input");
            if (rawFreq > 0) {
                m_gpu0ClockMhz.store(static_cast<uint32_t>(rawFreq / 1000000ULL), std::memory_order_relaxed);
            }
            if (rawTemp > 0) {
                m_gpu0TempC.store(static_cast<uint32_t>(rawTemp / 1000ULL), std::memory_order_relaxed);
            }
        }
    }

    // GPU 1
    VulkanContext* secCtx = m_getSecondaryContext ? m_getSecondaryContext() : nullptr;
    if (secCtx) {
        const auto& secPciInfo = secCtx->getPciLinkInfo();
        if (!secPciInfo.hwmonPath.empty()) {
            uint64_t rawFreq = readSysfsUint64(secPciInfo.hwmonPath + "/freq1_input");
            uint64_t rawTemp = readSysfsUint64(secPciInfo.hwmonPath + "/temp1_input");
            if (rawFreq > 0) {
                m_gpu1ClockMhz.store(static_cast<uint32_t>(rawFreq / 1000000ULL), std::memory_order_relaxed);
            }
            if (rawTemp > 0) {
                m_gpu1TempC.store(static_cast<uint32_t>(rawTemp / 1000ULL), std::memory_order_relaxed);
            }
        }
    }
}

void HwMonitor::refreshPciStatus() {
    if (m_primaryContext) {
        m_primaryContext->refreshPciLinkInfo();
    }
    VulkanContext* secCtx = m_getSecondaryContext ? m_getSecondaryContext() : nullptr;
    if (secCtx) {
        secCtx->refreshPciLinkInfo();
    }
    sampleSensors();
    Logger::info("Manually refreshed PCIe status.");
}

const std::string& HwMonitor::getOsName() {
    static const std::string s_os = queryOperatingSystem();
    return s_os;
}

const std::string& HwMonitor::getCpuModel() {
    static const std::string s_cpu = queryCpuModel();
    return s_cpu;
}

uint64_t HwMonitor::getTotalRamMb() {
    static const uint64_t s_ram = queryTotalRamMB();
    return s_ram;
}

std::string HwMonitor::exportTelemetry(const std::string& customPath, const FrameStats& stats) {
    std::string path = customPath.empty() ? ImageDumper::generateDefaultTelemetryPath() : customPath;
    if (m_telemetryWorker.joinable()) {
        m_telemetryWorker.join();
    }
    m_telemetryWorker = std::thread([path, stats]() {
        if (ImageDumper::saveStatsJSON(path, stats)) {
            Logger::info("Exported comprehensive telemetry dataset to: {}", path);
        } else {
            Logger::error("Failed to export telemetry dataset to: {}", path);
        }
    });
    return path;
}

} // namespace pathways
