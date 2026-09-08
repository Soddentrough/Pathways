#pragma once

#include <string>
#include <vector>
#include <cstdint>

namespace pathways {

struct FrameStats {
    // Platform & System Details
    std::string os_name = "Linux";
    std::string kernel_version = "";
    std::string cpu_model = "";
    double ram_total_gb = 0.0;

    // Primary GPU Hardware Specs
    std::string gpu_name;
    std::string topology_name;
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    std::string driver_version_str = "";
    std::string vulkan_api_str = "";
    std::string device_type_str = "Discrete GPU";
    std::string arch_name = "AMD RDNA4 (GFX1201)";
    std::string short_arch = "RDNA4";
    std::string ray_accelerator_name = "AMD RDNA4 3rd Gen Ray Accelerators";
    bool is_rdna3 = false;
    bool is_rdna4 = true;
    double total_vram_mb = 0.0;
    double vram_used_mb = 0.0;
    double vram_budget_mb = 0.0;

    // Secondary GPU & Multi-GPU Topology
    std::string secondary_gpu_name = "";
    std::string secondary_arch_name = "";
    std::string mgpu_interconnect_str = "PCIe 5.0 x16 (32 GT/s / ~64 GB/s Full-Duplex)";
    std::string mgpu_mode_str = "off";
    bool is_mgpu_active = false;
    bool visualize_mgpu_split = false;

    // Live Hardware Telemetry
    std::string primary_pci_link = "PCIe N/A";
    std::string primary_pci_speed = "";
    uint32_t primary_pci_width = 0;
    std::string primary_pci_max_speed = "";
    uint32_t primary_pci_max_width = 0;
    bool primary_pci_degraded = false;
    std::string primary_pci_degraded_reason = "";
    uint32_t primary_gpu_clock_mhz = 0;
    uint32_t primary_gpu_temp_c = 0;

    std::string secondary_pci_link = "";
    std::string secondary_pci_speed = "";
    uint32_t secondary_pci_width = 0;
    std::string secondary_pci_max_speed = "";
    uint32_t secondary_pci_max_width = 0;
    bool secondary_pci_degraded = false;
    std::string secondary_pci_degraded_reason = "";
    uint32_t secondary_gpu_clock_mhz = 0;
    uint32_t secondary_gpu_temp_c = 0;

    // Vulkan & Hardware RT Support Levels
    bool has_hw_rt = true;
    bool has_rt_pipeline = true;
    bool has_ray_query = true;
    bool has_as = true;
    bool has_bda = true;
    bool has_dho = true;
    uint32_t rt_handle_size = 32;
    uint32_t rt_base_align = 32;
    uint32_t rt_handle_align = 16;
    uint32_t rt_max_recursion = 31;
    bool has_dgc = true;
    uint32_t dgc_max_tokens = 128;
    uint32_t dgc_max_sequences = 1048576;
    uint32_t dgc_max_stride = 2048;
    bool has_subgroup_control = true;
    uint32_t subgroup_size = 32;
    bool has_dynamic_rendering = true;
    bool has_timeline_semaphores = true;
    bool has_sync2 = true;

    // Engine Settings & Configuration
    std::string pipeline_type_str = "rtp";
    uint32_t width = 0;
    uint32_t height = 0;
    float render_scale = 1.0f;
    uint32_t spp = 0;
    uint32_t max_bounces = 0;
    uint32_t checkerboard_tile_size = 64;
    bool enable_direct_light = true;
    bool enable_indirect_light = true;
    bool enable_refraction = true;
    bool enable_shadows = true;
    bool aces_tonemap = true;
    std::string scene_path = "";
    uint32_t num_triangles = 0;
    uint32_t num_spheres = 0;
    uint32_t num_materials = 0;
    uint32_t num_lights = 0;
    uint32_t num_textures = 0;
    std::string hdri_path = "";
    float cam_pos[3] = {0, 0, 0};
    float cam_yaw = 0.0f;
    float cam_pitch = 0.0f;
    float cam_fov = 45.0f;

    // Real-Time Performance & Benchmark Metrics
    uint32_t total_frames = 0;
    uint32_t total_samples = 0;
    double current_frame_time_ms = 0.0;
    double current_fps = 0.0;
    double avg_frame_time_ms = 0.0;
    double min_frame_time_ms = 0.0;
    double max_frame_time_ms = 0.0;
    double avg_fps = 0.0;
    double rays_per_second = 0.0;
    uint32_t validation_errors = 0;
    bool target_achieved = false; // true if avg_frame_time_ms < 8.0

    // GPU Timestamp Profiler Breakdown
    double primary_gpu_time_ms = 0.0;
    double secondary_gpu_time_ms = 0.0;
    double tonemap_time_ms = 0.0;
    double pcie_transfer_time_ms = 0.0;

    // Tallied unique configurations breakdown
    struct ConfigTallySummary {
        std::string label;
        uint32_t frame_count = 0;
        double avg_frame_time_ms = 0.0;
        double min_frame_time_ms = 0.0;
        double max_frame_time_ms = 0.0;
        double avg_fps = 0.0;
        double primary_gpu_time_ms = 0.0;
        double secondary_gpu_time_ms = 0.0;
        double tonemap_time_ms = 0.0;
        double gigarays_per_second = 0.0;
        bool target_achieved = false;
    };
    std::vector<ConfigTallySummary> configurations_breakdown;
};

class ImageDumper {
public:
    static bool savePNG(const std::string& filepath, uint32_t width, uint32_t height, const uint8_t* rgbaPixels);
    static bool saveEXR(const std::string& filepath, uint32_t width, uint32_t height, const float* rgbaFloatPixels);
    static bool saveStatsJSON(const std::string& filepath, const FrameStats& stats);
    static std::string generateDefaultTelemetryPath();
};

} // namespace pathways
