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
#include <chrono>
#include <ctime>
#include <iomanip>
#include <sstream>

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

std::string ImageDumper::generateDefaultTelemetryPath() {
    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_struct{};
#if defined(_WIN32)
    localtime_s(&tm_struct, &now_c);
#else
    localtime_r(&now_c, &tm_struct);
#endif
    char buf[64];
    std::strftime(buf, sizeof(buf), "pathways_telemetry_%Y%m%d_%H%M%S.json", &tm_struct);
    return std::string(buf);
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

    auto now = std::chrono::system_clock::now();
    std::time_t now_c = std::chrono::system_clock::to_time_t(now);
    std::tm tm_struct{};
#if defined(_WIN32)
    localtime_s(&tm_struct, &now_c);
#else
    localtime_r(&now_c, &tm_struct);
#endif
    char timeBuf[64];
    std::strftime(timeBuf, sizeof(timeBuf), "%Y-%m-%dT%H:%M:%S%z", &tm_struct);

    out << "{\n"
        << "  \"metadata\": {\n"
        << std::format("    \"timestamp_iso8601\": \"{}\",\n", timeBuf)
        << std::format("    \"timestamp_unix\": {},\n", static_cast<uint64_t>(now_c))
        << "    \"application\": \"Pathways Pure Vulkan 1.4 Path Tracer\",\n"
        << "    \"engine_version\": \"1.4.0\"\n"
        << "  },\n"
        << "  \"platform\": {\n"
        << std::format("    \"os\": \"{}\",\n", stats.os_name)
        << std::format("    \"kernel\": \"{}\",\n", stats.kernel_version)
        << std::format("    \"cpu\": \"{}\",\n", stats.cpu_model)
        << std::format("    \"system_ram_gb\": {:.1f}\n", stats.ram_total_gb)
        << "  },\n"
        << "  \"primary_gpu\": {\n"
        << std::format("    \"device_name\": \"{}\",\n", stats.gpu_name)
        << std::format("    \"vendor_id\": \"0x{:04x}\",\n", stats.vendor_id)
        << std::format("    \"device_id\": \"0x{:04x}\",\n", stats.device_id)
        << std::format("    \"device_type\": \"{}\",\n", stats.device_type_str)
        << std::format("    \"driver_version\": \"{}\",\n", stats.driver_version_str)
        << std::format("    \"vulkan_api_version\": \"{}\",\n", stats.vulkan_api_str)
        << std::format("    \"architecture\": \"{}\",\n", stats.arch_name)
        << std::format("    \"short_architecture\": \"{}\",\n", stats.short_arch)
        << std::format("    \"ray_accelerator\": \"{}\",\n", stats.ray_accelerator_name)
        << std::format("    \"is_rdna3\": {},\n", stats.is_rdna3 ? "true" : "false")
        << std::format("    \"is_rdna4\": {},\n", stats.is_rdna4 ? "true" : "false")
        << "    \"pcie\": {\n"
        << std::format("      \"link_status\": \"{}\",\n", stats.primary_pci_link)
        << std::format("      \"active_speed\": \"{}\",\n", stats.primary_pci_speed)
        << std::format("      \"active_width\": {},\n", stats.primary_pci_width)
        << std::format("      \"max_speed\": \"{}\",\n", stats.primary_pci_max_speed)
        << std::format("      \"max_width\": {},\n", stats.primary_pci_max_width)
        << std::format("      \"is_degraded\": {},\n", stats.primary_pci_degraded ? "true" : "false")
        << std::format("      \"degradation_reason\": \"{}\"\n", stats.primary_pci_degraded_reason)
        << "    },\n"
        << "    \"telemetry\": {\n"
        << std::format("      \"clock_mhz\": {},\n", stats.primary_gpu_clock_mhz)
        << std::format("      \"temperature_c\": {}\n", stats.primary_gpu_temp_c)
        << "    },\n"
        << "    \"memory\": {\n"
        << std::format("      \"total_vram_mb\": {:.2f},\n", stats.total_vram_mb)
        << std::format("      \"allocated_vram_mb\": {:.2f},\n", stats.vram_used_mb)
        << std::format("      \"budget_vram_mb\": {:.2f}\n", stats.vram_budget_mb)
        << "    }\n"
        << "  },\n"
        << "  \"secondary_gpu\": {\n"
        << std::format("    \"active\": {},\n", stats.is_mgpu_active ? "true" : "false")
        << std::format("    \"device_name\": \"{}\",\n", stats.secondary_gpu_name)
        << std::format("    \"architecture\": \"{}\",\n", stats.secondary_arch_name)
        << std::format("    \"interconnect\": \"{}\",\n", stats.mgpu_interconnect_str)
        << std::format("    \"mgpu_mode\": \"{}\",\n", stats.mgpu_mode_str)
        << std::format("    \"visualize_load_split\": {},\n", stats.visualize_mgpu_split ? "true" : "false")
        << "    \"pcie\": {\n"
        << std::format("      \"link_status\": \"{}\",\n", stats.secondary_pci_link)
        << std::format("      \"active_speed\": \"{}\",\n", stats.secondary_pci_speed)
        << std::format("      \"active_width\": {},\n", stats.secondary_pci_width)
        << std::format("      \"max_speed\": \"{}\",\n", stats.secondary_pci_max_speed)
        << std::format("      \"max_width\": {},\n", stats.secondary_pci_max_width)
        << std::format("      \"is_degraded\": {},\n", stats.secondary_pci_degraded ? "true" : "false")
        << std::format("      \"degradation_reason\": \"{}\"\n", stats.secondary_pci_degraded_reason)
        << "    },\n"
        << "    \"telemetry\": {\n"
        << std::format("      \"clock_mhz\": {},\n", stats.secondary_gpu_clock_mhz)
        << std::format("      \"temperature_c\": {}\n", stats.secondary_gpu_temp_c)
        << "    }\n"
        << "  },\n"
        << "  \"support_levels\": {\n"
        << "    \"hardware_ray_tracing\": {\n"
        << "      \"mandatory\": true,\n"
        << std::format("      \"vk_khr_ray_query\": {},\n", stats.has_ray_query ? "true" : "false")
        << std::format("      \"vk_khr_acceleration_structure\": {},\n", stats.has_as ? "true" : "false")
        << std::format("      \"vk_khr_buffer_device_address\": {},\n", stats.has_bda ? "true" : "false")
        << std::format("      \"vk_khr_deferred_host_operations\": {},\n", stats.has_dho ? "true" : "false")
        << std::format("      \"vk_khr_ray_tracing_pipeline\": {},\n", stats.has_rt_pipeline ? "true" : "false")
        << "      \"limits\": {\n"
        << std::format("        \"shader_group_handle_size_bytes\": {},\n", stats.rt_handle_size)
        << std::format("        \"shader_group_base_alignment_bytes\": {},\n", stats.rt_base_align)
        << std::format("        \"shader_group_handle_alignment_bytes\": {},\n", stats.rt_handle_align)
        << std::format("        \"max_ray_recursion_depth\": {}\n", stats.rt_max_recursion)
        << "      }\n"
        << "    },\n"
        << "    \"device_generated_commands\": {\n"
        << std::format("      \"supported\": {},\n", stats.has_dgc ? "true" : "false")
        << std::format("      \"max_indirect_commands_token_count\": {},\n", stats.dgc_max_tokens)
        << std::format("      \"max_indirect_sequence_count\": {},\n", stats.dgc_max_sequences)
        << std::format("      \"max_indirect_commands_total_stride\": {}\n", stats.dgc_max_stride)
        << "    },\n"
        << "    \"subgroups\": {\n"
        << std::format("      \"subgroup_size_control\": {},\n", stats.has_subgroup_control ? "true" : "false")
        << std::format("      \"native_subgroup_size\": {}\n", stats.subgroup_size)
        << "    },\n"
        << "    \"core_vulkan_1_4\": {\n"
        << std::format("      \"dynamic_rendering\": {},\n", stats.has_dynamic_rendering ? "true" : "false")
        << std::format("      \"timeline_semaphores\": {},\n", stats.has_timeline_semaphores ? "true" : "false")
        << std::format("      \"synchronization2\": {}\n", stats.has_sync2 ? "true" : "false")
        << "    }\n"
        << "  },\n"
        << "  \"engine_settings\": {\n"
        << std::format("    \"pipeline_type\": \"{}\",\n", stats.pipeline_type_str)
        << std::format("    \"resolution\": [{}, {}],\n", stats.width, stats.height)
        << std::format("    \"render_scale\": {:.2f},\n", stats.render_scale)
        << std::format("    \"spp\": {},\n", stats.spp)
        << std::format("    \"max_bounces\": {},\n", stats.max_bounces)
        << std::format("    \"morton_order\": {},\n", stats.enable_morton ? "true" : "false")
        << std::format("    \"checkerboard_tile_size\": {},\n", stats.checkerboard_tile_size)
        << "    \"shading\": {\n"
        << std::format("      \"direct_lighting\": {},\n", stats.enable_direct_light ? "true" : "false")
        << std::format("      \"indirect_diffuse_gi\": {},\n", stats.enable_indirect_light ? "true" : "false")
        << std::format("      \"dielectric_refraction\": {},\n", stats.enable_refraction ? "true" : "false")
        << std::format("      \"soft_shadows\": {},\n", stats.enable_shadows ? "true" : "false")
        << std::format("      \"aces_tonemapping\": {}\n", stats.aces_tonemap ? "true" : "false")
        << "    },\n"
        << "    \"scene\": {\n"
        << std::format("      \"path\": \"{}\",\n", stats.scene_path)
        << std::format("      \"num_triangles\": {},\n", stats.num_triangles)
        << std::format("      \"num_spheres\": {},\n", stats.num_spheres)
        << std::format("      \"num_materials\": {},\n", stats.num_materials)
        << std::format("      \"num_lights\": {},\n", stats.num_lights)
        << std::format("      \"num_textures\": {},\n", stats.num_textures)
        << std::format("      \"hdri_path\": \"{}\"\n", stats.hdri_path)
        << "    },\n"
        << "    \"camera\": {\n"
        << std::format("      \"position\": [{:.3f}, {:.3f}, {:.3f}],\n", stats.cam_pos[0], stats.cam_pos[1], stats.cam_pos[2])
        << std::format("      \"yaw\": {:.2f},\n", stats.cam_yaw)
        << std::format("      \"pitch\": {:.2f},\n", stats.cam_pitch)
        << std::format("      \"fov\": {:.1f}\n", stats.cam_fov)
        << "    }\n"
        << "  },\n"
        << "  \"performance\": {\n"
        << std::format("    \"current_frame_time_ms\": {:.3f},\n", stats.current_frame_time_ms)
        << std::format("    \"current_fps\": {:.1f},\n", stats.current_fps)
        << std::format("    \"avg_frame_time_ms\": {:.3f},\n", stats.avg_frame_time_ms)
        << std::format("    \"min_frame_time_ms\": {:.3f},\n", stats.min_frame_time_ms)
        << std::format("    \"max_frame_time_ms\": {:.3f},\n", stats.max_frame_time_ms)
        << std::format("    \"avg_fps\": {:.1f},\n", stats.avg_fps)
        << std::format("    \"target_achieved_sub_8ms\": {},\n", stats.target_achieved ? "true" : "false")
        << std::format("    \"rays_per_second\": {:.2e},\n", stats.rays_per_second)
        << std::format("    \"gigarays_per_second\": {:.3f},\n", stats.rays_per_second * 1e-9)
        << "    \"gpu_profiler_breakdown_ms\": {\n"
        << std::format("      \"primary_gpu_time_ms\": {:.3f},\n", stats.primary_gpu_time_ms)
        << std::format("      \"secondary_gpu_time_ms\": {:.3f},\n", stats.secondary_gpu_time_ms)
        << std::format("      \"tonemap_and_merge_time_ms\": {:.3f},\n", stats.tonemap_time_ms)
        << std::format("      \"pcie_transfer_time_ms\": {:.3f}\n", stats.pcie_transfer_time_ms)
        << "    },\n"
        << std::format("    \"total_frames\": {},\n", stats.total_frames)
        << std::format("    \"total_accumulated_samples\": {},\n", stats.total_samples)
        << std::format("    \"validation_errors\": {}\n", stats.validation_errors)
        << "  }\n"
        << "}\n";

    Logger::info("Saved verification statistics JSON to: {}", filepath);
    return true;
}

} // namespace pathways
