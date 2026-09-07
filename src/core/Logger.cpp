#include "core/Logger.hpp"

namespace pathways {

LogLevel Logger::s_level = LogLevel::Info;

void Logger::setLogLevel(LogLevel level) {
    s_level = level;
}

LogLevel Logger::getLogLevel() {
    return s_level;
}

} // namespace pathways
