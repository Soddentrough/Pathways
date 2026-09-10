# Pathways

Pathways is a high-performance, real-time path tracing and renderer engine built from scratch on pure **Vulkan 1.4**. It embraces cutting-edge GPU architectures and paradigms, specifically **Device Generated Commands** (`VK_EXT_device_generated_commands`), **Wavefront Path Tracing**, and direct **Peer-to-Peer (P2P) Multi-GPU scaling**.

> **Design Philosophy**: No megakernel — only efficient, GPU-autonomous Device Generated Commands, decoupled wavefront microkernels, and modern real-time rendering principles. Pathways uses strictly standard Vulkan 1.4, KHR, and EXT specifications with **no proprietary extensions**.

- **API Baseline**: Vulkan 1.4 (1.4.341+)
- **Primary Hardware Targets**: AMD RDNA4 (`gfx1201`) and RDNA3 architectures (and compatible Vulkan 1.4 hardware)
- **Platforms**: Linux (Fedora 40+, Ubuntu 24.04+, Arch) & Windows 11 (MSVC x64 / Clang 20)

---

## Key Capabilities & Architecture

### 1. Wavefront Path Tracing & Autonomous DGC
- **Wavefront Architecture**: Decomposes ray tracing into decoupled compute stages (Ray Classification, Ray Intersection, Material Shading, Shadow Queries, Accumulation Resolve), eliminating execution divergence.
- **GPU-Autonomous Material Sorting via DGC**: Uses `vkCmdExecuteGeneratedCommandsEXT` to dynamically group rays by BSDF archetype (diffuse, dielectric, conductor, complex) and dispatch specialized compute kernels directly on the device with zero CPU intervention.
- **Buffer Device Address (BDA) Ray Queuing**: Lock-free, atomic queue allocation using 64-bit device addresses for high-throughput ray staging.
- **Hardware Ray Tracing**: Full support for dedicated hardware BVH traversal via `VK_KHR_ray_tracing_pipeline` (RTP) and inline `VK_KHR_ray_query`.

### 2. Real-Time Multi-GPU Scaling
- **P2P Direct BAR DMA-BUF Transfer**: Exports secondary GPU accumulation buffers via Linux DMA-BUF (`VK_EXT_external_memory_dma_buf`, `VK_KHR_external_memory_fd`) and imports them directly into primary GPU VRAM over PCIe BAR, slashing inter-GPU transfer/compositing latency to **<0.1 ms** (~80% reduction over host staging).
- **Fine-Grained 2D Checkerboard Tiling**: Dynamically distributes screen space across $64\times 64$ alternating tiles (2,040 tiles at 4K) for balanced spatial and shading workload division across dual GPUs.
- **Sample Parallelism**: Temporal sample division mode for multi-SPP scenarios.
- **Cross-Platform Fallback**: Automatic detection and transparent fallback to double-buffered shared host memory for platforms without DMA-BUF (such as Windows).

### 3. Advanced Lighting, Materials & Denoising
- **glTF 2.0 PBR & Extensions**: Physically-based materials with metallic-roughness, normal mapping, emissive meshes, and advanced extensions:
  - `KHR_materials_transmission` (specular & diffuse transmission)
  - `KHR_materials_clearcoat` (secondary reflective coats)
  - `KHR_materials_ior` (Fresnel index of refraction)
  - `KHR_materials_volume` (absorption, attenuation color, and distance)
  - Alpha MASK and BLEND transparency modes
- **ReSTIR DI**: Reservoir-based Spatiotemporal Importance Resampling for direct illumination with candidate reuse.
- **AMD FidelityFX Shadow Denoiser**: Two-pass tile classification and spatial filtering for direct shadow rays.
- **À-Trous Wavelet Filter**: Edge-avoiding À-trous wavelet spatial diffuse denoising with configurable iterations.
- **Temporal Anti-Aliasing (TAA)**: Halton(2,3) sub-pixel jitter with neighborhood clamping and velocity-guided history rejection.
- **HDR Environment Maps**: Pre-filtered high-dynamic-range EXR/HDR skyboxes.
- **ACES Tonemapping**: High-quality filmic tone curve mapping linear HDR radiance into sRGB display space.

### 4. Dynamic Quality Regulation & Telemetry
- **Dynamic Quality Governor**: Closed-loop frame-time budget regulation targeting user-defined FPS (e.g., 60, 90, 120 FPS), dynamically scaling SPP and bounce depth to guarantee smooth interactive framerates.
- **Dear ImGui HUD & Controls**: Interactive in-engine UI overlay featuring collapsible controls, live GPU frame time histograms, camera controls, and real-time pipeline toggles.
- **Telemetry & Benchmarking**: Headless automated benchmark suite (`--headless`) exporting comprehensive JSON telemetry breakdowns (MAE, RMSE, PSNR, bounce-by-bounce ray counts, and queue footprints).

---

## Deliverables & Interactive Controls

Load any glTF 2.0 or procedural scene and fly through it in real time with high-fidelity global illumination, reflections, refractions, and contact shadows.

| Control | Action |
| :--- | :--- |
| **W / A / S / D** | Fly forward / left / backward / right |
| **E / Q** or **Space / C** | Fly up / fly down |
| **Right Click + Drag** | Look / rotate camera |
| **Shift** (hold) | Sprint / speed multiplier |
| **F1** | Toggle Dear ImGui HUD & Controls overlay |
| **F2** | Cycle multi-GPU load visualization overlay |
| **Esc** | Exit Pathways |

---

## Requirements

- **Vulkan**: Version 1.4 or newer (Vulkan SDK >= 1.4.341)
- **Windowing & Input**: SDL3 (v3.1+)
- **Compression**: zlib / zlib-ng
- **Math**: GLM (header-only, bundled in `third_party/glm`)
- **GPU**: AMD Radeon RDNA4 / RDNA3 (or any modern GPU with Vulkan 1.4, ray queries, and DGC support)

---

## Building and Running

### Linux (Fedora / Ubuntu / Arch)

#### 1. Install Build Dependencies
**Fedora**:
```bash
sudo dnf install git gcc-c++ cmake ninja-build vulkan-loader-devel vulkan-headers glslc SDL3-devel zlib-ng-compat-devel
```

**Ubuntu / Debian (24.04+)**:
```bash
sudo apt-get install git g++ cmake ninja-build libvulkan-dev vulkan-tools libsdl3-dev zlib1g-dev
```

#### 2. Configure & Build
```bash
# Configure with Ninja in Release mode
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_NATIVE_OPTIMIZATION=ON

# Compile (capped to your core count, e.g. -j16)
ninja -C build -j16
```

#### 3. Run Unit Tests
```bash
ctest --test-dir build --output-on-failure
```

#### 4. Launch Pathways
```bash
# Launch interactive Cornell Box
./build/bin/pathways

# Launch custom glTF scene with Multi-GPU enabled
./build/bin/pathways --scene scenes/classroom/classroom_extended.glb --mgpu --spp 1

# Launch Wavefront path tracer with Autonomous DGC material sorting
./build/bin/pathways --pipeline wavefront --wavefront-sort archetype --res 1440p
```

---

### Windows 11

Pathways provides full Windows 11 support using MSVC 2022/2026 or Clang 20 + LLD (MSYS2 UCRT64) with Ninja and Vulkan SDK 1.4+.

```powershell
# Launch interactive GUI (auto-builds if needed)
.\run.ps1
# Or from cmd: run.bat

# Build in Release mode
.\build.ps1

# Run unit tests
.\build.ps1 -Test

# Run full automated headless test suite
.\scripts\run_headless_tests.ps1
```

For complete Windows toolchain configuration and presets, see [BUILD_WINDOWS.md](BUILD_WINDOWS.md).

---

## CLI Reference

| Flag | Argument | Description | Default |
| :--- | :--- | :--- | :--- |
| `--scene` | `<path>` | Path to glTF 2.0 file (`.glb` / `.gltf`) | Cornell Box |
| `--pipeline` | `wavefront` \| `rtp` | Pipeline architecture: Wavefront DGC or KHR RTP | `wavefront` |
| `--res`, `-r` | `1080` \| `1440` \| `4k` \| `WxH` | Viewport resolution preset or dimensions | Display native |
| `--spp` | `<int>` | Samples per pixel accumulated per frame | `1` |
| `--max-bounces` | `<int>` | Maximum path depth / bounce limit | `4` |
| `--mgpu` | *(flag)* | Enable Multi-GPU load balancing | Disabled |
| `--mgpu-mode` | `tile` \| `sample` \| `auto` | Multi-GPU distribution strategy | `tile` |
| `--tile-size` | `16` \| `32` \| `64` \| `128` | Checkerboard tile dimensions in pixels | `64` |
| `--visualize-split` | *(flag)* | Show colored overlay indicating GPU assignment | Off |
| `--wavefront-sort` | `none` \| `archetype` \| `bda` \| `dual` | Material sorting mode for DGC wavefront | `none` |
| `--atrous` | *(flag)* | Enable À-Trous wavelet diffuse denoiser | Disabled |
| `--atrous-passes` | `1..5` | Number of À-Trous filtering iterations | `3` |
| `--target-fps` | `<int>` | Quality Governor target FPS (`0` = uncapped) | `0` |
| `--adaptive-spp` | *(flag)* | Enable dynamic 3-axis quality regulation | Disabled |
| `--headless` | *(flag)* | Run offscreen without opening a window | Disabled |
| `--frames` | `<int>` | Total frame execution limit (`0` = run continuously) | `0` |
| `--dump-frame` | `<path.png>` | Save tonemapped LDR frame to PNG on exit | None |
| `--dump-hdr` | `<path.exr>` | Save linear radiance buffer to OpenEXR on exit | None |
| `--dump-stats` | `<path.json>` | Export per-frame telemetry breakdown to JSON | None |
| `--no-validation` | *(flag)* | Disable Vulkan validation layers | Validation on |

---

## Profiling & Developer Tools

- **AMD Developer Tools**: Compatible with Radeon Developer Tool Suite (`/opt/RadeonDeveloperToolSuite-2026-05-28-1806/`), Radeon GPU Profiler (RGP), and Radeon Raytracing Analyzer (RRA).
- **Automated Profiling Suite**:
  ```bash
  python3 scripts/benchmark_megakernel_vs_wavefront.py
  python3 scripts/deep_profile_scenes.py
  ```

---

## Installed Packages (Linux Reference)

```
glslc-2026.1-1.fc44.x86_64
vulkan-loader-1.4.341.0-1.fc44.x86_64
vulkan-headers-1.4.341.0-1.fc44.noarch
vulkan-validation-layers-1.4.341.0-2.fc44.x86_64
vulkan-loader-devel-1.4.341.0-1.fc44.x86_64
vulkan-tools-1.4.341.0-1.fc44.x86_64
vulkan-utility-libraries-devel-1.4.341.0-1.fc44.x86_64
mesa-vulkan-drivers-26.1.8-1.fc44.x86_64
```

---

## Thanks

Additional thanks to **MrMPFR**.

---

## References

- [Vulkan Device Generated Commands Specification](https://docs.vulkan.org/spec/latest/chapters/device_generated_commands/generatedcommands.html)
- [vkdoc Device Generated Commands Guide](https://vkdoc.net/chapters/device-generated-commands)
- [Vulkan Ray Tracing Overview](https://docs.vulkan.org/tutorial/latest/courses/18_Ray_tracing/00_Overview.html)
- [Supergoodcode: Device Generated Commands](https://www.supergoodcode.com/device-generated-commands/)
- [GPUOpen: RGP Work Graphs & DGC](https://gpuopen.com/learn/rgp-work-graphs/)
- [AMD RDNA4 Instruction Set Architecture (ISA)](https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture)
- [AMD RDNA Performance Guide](https://gpuopen.com/learn/rdna-performance-guide/)
- [Improving Ray Tracing Performance with RRA](https://gpuopen.com/learn/improving-rt-perf-with-rra/)

