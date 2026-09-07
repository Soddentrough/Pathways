#pragma once

#include <iostream>
#include <string>
#include <string_view>
#include <format>
#include <chrono>

namespace pathways {

enum class LogLevel {
    Debug,
    Info,
    Warn,
    Error
};

class Logger {
public:
    static void setLogLevel(LogLevel level);
    static LogLevel getLogLevel();

    template <typename... Args>
    static void debug(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Debug, "\033[36m[DEBUG]\033[0m", fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void info(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Info, "\033[32m[INFO]\033[0m ", fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void warn(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Warn, "\033[33m[WARN]\033[0m ", fmt, std::forward<Args>(args)...);
    }

    template <typename... Args>
    static void error(std::format_string<Args...> fmt, Args&&... args) {
        log(LogLevel::Error, "\033[31m[ERROR]\033[0m", fmt, std::forward<Args>(args)...);
    }

private:
    static LogLevel s_level;

    template <typename... Args>
    static void log(LogLevel level, std::string_view prefix, std::format_string<Args...> fmt, Args&&... args) {
        if (level < s_level) return;
        auto now = std::chrono::system_clock::now();
        auto formatted_time = std::format("{:%H:%M:%S}", now);
        std::string msg = std::format(fmt, std::forward<Args>(args)...);
        std::cout << std::format("[{}] {} {}\n", formatted_time, prefix, msg);
    }
};

} // namespace pathways
