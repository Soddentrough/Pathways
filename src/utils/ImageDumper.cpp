#include "utils/ImageDumper.hpp"
#include "core/Logger.hpp"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define TINYEXR_USE_MINIZ 0
#include <zlib.h>
#define TINYEXR_IMPLEMENTATION
#include "tinyexr.h"

#include <fstream>
#include <filesystem>
#include <format>

namespace pathways {

bool ImageDumper::savePNG(const std::string& filepath, uint32_t width, uint32_t height, const uint8_t* rgbaPixels) {
    if (!rgbaPixels || width == 0 || height == 0) {
        Logger::error("Invalid image buffer passed to savePNG");
        return false;
    }

    std::filesystem::path p(filepath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    int stride = static_cast<int>(width * 4);
    int res = stbi_write_png(filepath.c_str(), static_cast<int>(width), static_cast<int>(height), 4, rgbaPixels, stride);
    if (res != 0) {
        Logger::info("Successfully dumped LDR PNG frame to: {}", filepath);
        return true;
    } else {
        Logger::error("Failed to write PNG to: {}", filepath);
        return false;
    }
}

bool ImageDumper::saveEXR(const std::string& filepath, uint32_t width, uint32_t height, const float* rgbaFloatPixels) {
    if (!rgbaFloatPixels || width == 0 || height == 0) {
        Logger::error("Invalid float image buffer passed to saveEXR");
        return false;
    }

    std::filesystem::path p(filepath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    // tinyexr SaveEXR:
    // data is float RGBA buffer. save_as_fp16 = 1 (16-bit half) or 0 (32-bit float).
    const char* err = nullptr;
    int res = SaveEXR(rgbaFloatPixels, static_cast<int>(width), static_cast<int>(height), 4, 0 /* 32-bit float */, filepath.c_str(), &err);
    if (res == TINYEXR_SUCCESS) {
        Logger::info("Successfully dumped HDR OpenEXR frame to: {}", filepath);
        return true;
    } else {
        Logger::error("Failed to write EXR to {}: {}", filepath, err ? err : "unknown error");
        if (err) FreeEXRErrorMessage(err);
        return false;
    }
}

bool ImageDumper::saveStatsJSON(const std::string& filepath, const FrameStats& stats) {
    std::filesystem::path p(filepath);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }

    std::ofstream out(filepath);
    if (!out.is_open()) {
        Logger::error("Failed to open JSON output file: {}", filepath);
        return false;
    }

    out << "{\n"
        << std::format("  \"gpu_name\": \"{}\",\n", stats.gpu_name)
        << std::format("  \"architecture\": \"{}\",\n", stats.arch_name)
        << std::format("  \"short_arch\": \"{}\",\n", stats.short_arch)
        << std::format("  \"ray_accelerator\": \"{}\",\n", stats.ray_accelerator_name)
        << std::format("  \"is_rdna3\": {},\n", stats.is_rdna3 ? "true" : "false")
        << std::format("  \"is_rdna4\": {},\n", stats.is_rdna4 ? "true" : "false")
        << std::format("  \"resolution\": [{}, {}],\n", stats.width, stats.height)
        << std::format("  \"spp\": {},\n", stats.spp)
        << std::format("  \"total_frames\": {},\n", stats.total_frames)
        << std::format("  \"current_frame_time_ms\": {:.3f},\n", stats.current_frame_time_ms)
        << std::format("  \"current_fps\": {:.1f},\n", stats.current_fps)
        << std::format("  \"avg_frame_time_ms\": {:.3f},\n", stats.avg_frame_time_ms)
        << std::format("  \"min_frame_time_ms\": {:.3f},\n", stats.min_frame_time_ms)
        << std::format("  \"max_frame_time_ms\": {:.3f},\n", stats.max_frame_time_ms)
        << std::format("  \"avg_fps\": {:.1f},\n", stats.avg_fps)
        << std::format("  \"rays_per_second\": {:.2e},\n", stats.rays_per_second)
        << std::format("  \"vram_used_mb\": {:.2f},\n", stats.vram_used_mb)
        << std::format("  \"validation_errors\": {},\n", stats.validation_errors)
        << std::format("  \"mgpu_mode\": \"{}\",\n", stats.mgpu_mode_str)
        << std::format("  \"primary_gpu_time_ms\": {:.3f},\n", stats.primary_gpu_time_ms)
        << std::format("  \"secondary_gpu_time_ms\": {:.3f},\n", stats.secondary_gpu_time_ms)
        << std::format("  \"hardware_rt\": {},\n", stats.has_hw_rt ? "true" : "false")
        << std::format("  \"dgc_enabled\": {},\n", stats.has_dgc ? "true" : "false")
        << std::format("  \"target_achieved_sub_8ms\": {}\n", stats.target_achieved ? "true" : "false")
        << "}\n";

    Logger::info("Saved verification statistics JSON to: {}", filepath);
    return true;
}

} // namespace pathways
