# Pathways

Pathways is a high-performance, real-time path tracing and renderer engine built from scratch on pure **Vulkan 1.4**. Clean sheet design to use **Device Generated Commands** (`VK_EXT_device_generated_commands`) for Ray Compaction and Material Binning, **Wavefront Path Tracing**, and high-throughput **Zero-Copy Host Memory (`VK_EXT_external_memory_host`) Multi-GPU scaling**. 

> **Design Philosophy**: No megakernel — only efficient, GPU-autonomous Device Generated Commands, decoupled wavefront microkernels, and modern real-time rendering principles. Pathways uses strictly standard Vulkan 1.4, KHR, and EXT specifications with **no proprietary extensions**.

- **API Baseline**: Vulkan 1.4 (1.4.341+)
- **Primary Hardware Targets**: AMD RDNA4 (`gfx1201`) and RDNA3 architectures (and compatible Vulkan 1.4 hardware; functions on any compliant Vulkan 1.4 driver)
- **Platforms**: Linux (Fedora 40+, Ubuntu 24.04+, Arch) & Windows 11 (MSVC x64 / Clang 20)

---

## Key Capabilities & Architecture

### 1. Wavefront Path Tracing & Autonomous DGC
- **Wavefront Architecture**: Decomposes ray tracing into decoupled compute stages (Ray Classification, Ray Intersection, Material Shading, Shadow Queries, Accumulation Resolve), eliminating execution divergence.
- **GPU-Autonomous Material Sorting via DGC**: Uses `vkCmdExecuteGeneratedCommandsEXT` to dynamically group rays by BSDF archetype (diffuse, dielectric, conductor, complex) and dispatch specialized compute kernels directly on the device with zero CPU intervention.
- **Buffer Device Address (BDA) Ray Queuing**: Lock-free, atomic queue allocation using 64-bit device addresses for high-throughput ray staging.
- **Hardware Ray Tracing**: Full support for dedicated hardware BVH traversal via `VK_KHR_ray_tracing_pipeline` (RTP) and inline `VK_KHR_ray_query`.

### 2. Real-Time Multi-GPU Scaling
- **Zero-Copy Host Memory Streaming (`VK_EXT_external_memory_host`)**: Secondary GPU streams tiled render buffers into pinned host memory via high-speed CP DMA posted writes at PCIe bus line rate (~25 GB/s, <0.5 ms). Primary GPU merges and resolves tiles in ~0.12 ms without PCIe bus contention, ensuring jitter-free, consistent 250+ FPS camera motion.
- **Configurable Inter-GPU Transfer Modes (`--mgpu-transfer`)**:
  - `host` (Default): Pinned zero-copy host memory. Highly recommended for discrete PCIe topologies without dedicated inter-GPU fabric bridges.
  - `p2p`: Direct Linux DMA-BUF export/import (`VK_EXT_external_memory_dma_buf`, `VK_KHR_external_memory_fd`). Ideal on hardware architectures with coherent inter-GPU links (e.g., Infinity Fabric bridges); on discrete PCIe slots, direct shader reads across PCIe BAR encounter non-posted read latency.
  - `staging`: Dedicated asynchronous transfer queues with CPU staging buffers.
- **Fine-Grained 2D Checkerboard Tiling**: Dynamically distributes screen space across $64\times 64$ alternating tiles (2,040 tiles at 4K) for balanced spatial and shading workload division across dual GPUs.
- **Sample Parallelism**: Temporal sample division mode for multi-SPP scenarios.
- **Cross-Platform Fallback**: Automatic detection and transparent fallback to double-buffered shared host memory for platforms without DMA-BUF (such as Windows).


### 3. Pure Path-Traced Lighting & Physical Materials
- **glTF 2.0 PBR & Extensions**: Physically-based materials with metallic-roughness, normal mapping, emissive meshes, and advanced Khronos extensions:
  - `KHR_materials_transmission` (specular & diffuse transmission with Snell's law refraction)
  - `KHR_materials_clearcoat` (secondary reflective coats with independent roughness)
  - `KHR_materials_ior` (Fresnel index of refraction)
  - `KHR_materials_volume` (volumetric Beer-Lambert absorption, attenuation color, and distance)
  - Alpha MASK and BLEND transparency modes
- **Multiple Importance Sampling (MIS)**: Veach balance heuristic combining Next-Event Estimation (direct light sampling) with BSDF importance sampling across diffuse, dielectric, and conductor microkernels.
- **Pure Monte Carlo Convergence**: Unbiased physical ray tracing with progressive sample accumulation and Russian roulette path termination. (Screen-space filters such as ReSTIR DI, FidelityFX Shadow Denoiser, À-Trous wavelet, and TAA are disabled in favor of true physical Monte Carlo convergence and dynamic SPP regulation).
- **Physical Sky & HDR Environment Maps**: Procedural physical sky dome and pre-filtered high-dynamic-range EXR/HDR image-based lighting.
- **ACES Tonemapping**: High-quality filmic tone curve mapping linear HDR radiance into sRGB display space via compute shader.

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

### Hardware & Extension Support Note

Pathways is designed for modern desktop workstations and gaming PCs with hardware-accelerated ray tracing and autonomous GPU execution. Coverage statistics from the Vulkan Hardware Database ([vulkan.gpuinfo.org](https://vulkan.gpuinfo.org/)) illustrate adoption levels across desktop platforms (Windows and Linux PCs) compared to the mobile-skewed global database:

| Extension / Feature | Spec Date | Desktop Coverage (Win + Linux) | Mobile Coverage (Android) | Global Coverage (All Devices) | Role & Status in Pathways |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **`VK_EXT_device_generated_commands`** | Dec 14, 2023 | **~49.3%** | **~0.1%** | **~16.8%** | **Wavefront Acceleration**: Enables GPU-autonomous material binning and kernel dispatch directly on the device with zero CPU intervention. Standard on modern desktop drivers (AMD RDNA3/RDNA4 on Mesa RADV, NVIDIA Ada/Blackwell). Fallback via monolithic compute is supported (`--pipeline wavefront --wavefront-sort none`). |
| **`VK_KHR_acceleration_structure`** | Nov 20, 2020 | **~52.9%** | **~26.6%** | **~34.1%** | **BVH Management**: Required for building and querying hardware Top-Level (TLAS) and Bottom-Level (BLAS) ray tracing acceleration structures. |
| **`VK_KHR_ray_tracing_pipeline`** | Nov 20, 2020 | **~51.3%** | **~7.6%** | **~21.8%** | **Hardware RTP**: Powers the dedicated hardware ray tracing pipeline (`--pipeline rtp` megakernel mode). |
| **`VK_EXT_external_memory_host`** | Jan 17, 2018 | **~84.9%** | **~6.5%** | **~36.1%** | **Multi-GPU Zero-Copy**: Enables secondary GPU to stream rendered HDR tiles directly into pinned host RAM at PCIe line rate (~25 GB/s), eliminating peer PCIe BAR read stalls. |
| **Vulkan 1.4 Core** | Jan 15, 2025 | **~61.2%** | **~7.9%** | **~28.2%** | **Engine Baseline**: Core driver version requirement (72.6% on Linux Mesa, 53.2% on Windows). |

> [!NOTE]
> **Desktop vs. Global Metrics**: Over 61.7% of all devices recorded in the Vulkan Hardware Database are low-power Android mobile phones and embedded SoCs (1,467 out of 2,376 devices), which drastically pulls down global percentages for high-end rendering features. Among desktop PCs (976 reported Windows and Linux devices), hardware ray tracing and DGC achieve ~50–53% coverage across all recorded hardware generations, and approach ~100% on contemporary discrete gaming GPUs (AMD RDNA2+, NVIDIA RTX 20+). For a complete breakdown and call-site citations across the entire engine, see [VULKAN_API_AUDIT.md](VULKAN_API_AUDIT.md) or run `python3 scripts/audit_vulkan_api.py --compare-platforms`.

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
| `--mgpu-transfer` | `host` \| `p2p` \| `staging` | Inter-GPU transfer mechanism: zero-copy host pinned memory (default), direct DMA-BUF P2P BAR, or staging buffers | `host` |
| `--tile-size` | `16` \| `32` \| `64` \| `128` | Checkerboard tile dimensions in pixels | `64` |
| `--visualize-split` | *(flag)* | Show colored overlay indicating GPU assignment | Off |
| `--wavefront-sort` | `none` \| `archetype` \| `bda` \| `dual` | Material sorting mode for DGC wavefront | `none` |
| `--no-accumulation` | *(flag)* | Disable progressive accumulation (evaluate real-time per-frame noise) | Accumulation on |
| `--warmup-frames` | `<int>` | Number of initial frames to discard from benchmark stats | `0` |
| `--target-fps` | `<int>` | Quality Governor target FPS (`0` = uncapped) | `0` |
| `--adaptive-spp` | *(flag)* | Enable dynamic 3-axis quality regulation | Disabled |
| `--headless` | *(flag)* | Run offscreen without opening a window | Disabled |
| `--frames` | `<int>` | Total frame execution limit (`0` = run continuously) | `0` |
| `--dump-frame` | `<path.png>` | Save tonemapped LDR frame to PNG on exit | None |
| `--dump-hdr` | `<path.exr>` | Save linear radiance buffer to OpenEXR on exit | None |
| `--dump-stats` | `<path.json>` | Export per-frame telemetry breakdown to JSON | None |
| `--no-validation` | *(flag)* | Disable Vulkan validation layers | Validation on |

---

## Performance Benchmarks & Tested Hardware

Pathways is continuously tested and profiled on modern high-end multi-GPU AMD hardware. Below are representative performance metrics and scaling benchmarks.

### Reference Hardware Specifications
- **Host CPU**: AMD Ryzen Threadripper 3970X (32 cores / 64 threads, 128 MB L3 cache)
- **System Memory**: 64 GB DDR4 Quad-Channel
- **Primary GPU (GPU 0)**: AMD Radeon AI PRO R9700 (32 GB GDDR6, 256-bit, PCIe 4.0 x16, RDNA 4 `gfx1201`)
- **Secondary GPU (GPU 1)**: AMD Radeon AI PRO R9700 (32 GB GDDR6, 256-bit, PCIe 4.0 x8, RDNA 4 `gfx1201`)
- **Interconnect**: Zero-Copy Host Memory (`VK_EXT_external_memory_host`) and DMA-BUF P2P over PCIe 4.0, synchronized via `VK_KHR_external_semaphore_fd`
- **OS & Driver**: Fedora Linux 44 (Kernel 7.1), Mesa RADV 26.1.8, Vulkan 1.4.354

---

### Multi-GPU Scaling & Resolution Benchmarks

| Scene / Workload | Resolution & Settings | Single-GPU (ms / FPS) | Dual-GPU (ms / FPS) | Multi-GPU Mode | Speedup / Scaling | Throughput Gain |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Procedural Cornell Box** | **4K Native** (3840x2160), 1 SPP, 4 Bounces | 7.35 ms (136.0 FPS) | **3.87 ms** (258.1 FPS) | Checkerboard ($64\times 64$) | **1.90x** (Smooth Camera Motion) | 8.56 GigaRays/s |
| **Damaged Helmet (`.glb`)** | **1080p** (1920x1080), 16 SPP, 4 Bounces | 7.79 ms (128.5 FPS) | **3.41 ms** (293.0 FPS) | Sample Parallelism | **2.28x** (>100% Efficiency) | 1.88x ($3.86 \times 10^{10}$ rays/s) |
| **Pontiac GTO Extended** (1,063,260 Triangles) | **4K Native** (3840x2160), 1 SPP, 4 Bounces | 11.56 ms (86.5 FPS) | **5.82 ms** (171.8 FPS) | Checkerboard ($64\times 64$) | **1.99x** (99.3% Efficiency) | 5.70 GigaRays/s |
| **Pontiac GTO Extended** (1,063,260 Triangles) | **4K Native** (3840x2160), 1 SPP, 4 Bounces | 11.56 ms (86.5 FPS) | **6.97 ms** (143.4 FPS) | Sample Parallelism | **1.66x** (83.0% Efficiency) | 4.76 GigaRays/s |
| **Zero-Copy Host Compositing** | 4K HDR Tile Merge (63.3 MB) | — | **0.12 ms** (8.3 kHz) | Host DMA (`VK_EXT_external_memory_host`) | **Zero PCIe BAR Stalls** | Smooth 258 FPS Motion |

> **Discrete PCIe vs. Coherent Fabric Note**: On dual discrete GPUs connected across standard PCIe slots, reading directly across peer PCIe BAR via compute shaders issues uncached, non-posted PCIe reads which incur high per-transaction latency (stalling GPU compute queues for up to 28 ms per frame under heavy traffic). Pathways solves this by defaulting to `VK_EXT_external_memory_host`, where the secondary GPU writes to pinned host memory via high-speed DMA posted writes at full bus line rate (~25 GB/s, <0.5 ms), allowing the primary GPU to composite in 0.12 ms without bus stalls. Direct P2P BAR remains selectable via `--mgpu-transfer p2p` for systems equipped with hardware-coherent interconnects.

---

### 4K Architecture Comparison (Megakernel RTP vs. Wavefront DGC)

Measured at native **3840x2160 (4K)**, 1 SPP, 4 Bounces, FP16 on primary Radeon AI PRO R9700:

| Scene | Scene Characteristics | Megakernel RTP (ms / FPS) | Wavefront Monolithic (ms / FPS) | Wavefront Sorted (ms / FPS) | Relative Speedup (WF / RTP) | Visual Parity (PSNR / MAE) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Cornell Box** | Low triangle count, diffuse dominant | **5.05 ms** (173.7 FPS) | 7.72 ms (128.1 FPS) | 8.84 ms (111.8 FPS) | 0.65x | **24.25 dB** [PASS] |
| **Dragon Attenuation** | High geometry, pure dielectric Beer-Lambert | **6.41 ms** (152.3 FPS) | 6.54 ms (150.9 FPS) | 6.79 ms (144.4 FPS) | 0.98x | **33.76 dB** / 0.0083 [PASS] |
| **Living Room Extended** | Complex architectural interior, divergent rays | 11.49 ms (81.1 FPS) | **11.29 ms** (86.4 FPS) | 11.76 ms (83.0 FPS) | **1.02x** (Wavefront Wins) | **29.25 dB** [PASS] |
| **Coffee Maker Extended** | High specular, metallic, complex transmission | **6.87 ms** (141.6 FPS) | 8.57 ms (114.5 FPS) | 9.73 ms (100.9 FPS) | 0.80x | **23.29 dB** [PASS] |

> **Takeaway**: Megakernel RTP excels in scenes where rays can remain in fast registers (~120 VGPRs) without VRAM queue round-trips. Wavefront excels in complex scenes with heavy divergence (e.g., Living Room), where stream compaction strips terminated paths early and prevents SIMD thread idling.

---

## Driver Tuning & Mesa RADV Environment Variables

Pathways runs natively on the open-source **Mesa RADV** Vulkan driver. For performance testing, benchmarking, and developer profiling, Mesa exposes several environment variables that can optimize shader generation, wave scheduling, and memory allocation.

### `RADV_PERFTEST` Options

Set via `export RADV_PERFTEST=opt1,opt2` or prefixing your launch command:

```bash
RADV_PERFTEST=cswave32,nogttspill ./build/bin/pathways --res 4k
```

| Flag | Category | Description | Recommendation for Pathways |
| :--- | :--- | :--- | :--- |
| **`cswave32`** | Wave Scheduling | Forces Wave32 execution mode for all compute shaders on RDNA (GFX10+). | **Recommended**: Pathways defaults to Wave32 for ray sorting and compaction to reduce register pressure and SIMD divergence. |
| **`rtwave64`** | Ray Tracing | Forces Wave64 execution mode for ray tracing shaders instead of Wave32. | **Experimental / Benchmarking**: Test when evaluating ray tracing SIMD divergence vs. VGPR occupancy trade-offs. |
| **`nogttspill`** | Memory Allocation | Strictly prioritizes on-card VRAM allocations and disables GTT (system RAM) spilling. | **Recommended for Benchmarks**: Prevents memory thrashing across PCIe bus during high-resolution ray queue staging. |
| **`transfer_queue`** | Queues & DMA | Enables dedicated SDMA transfer queues for asynchronous DMA operations. | **Recommended for Multi-GPU**: Offloads cross-device image copies and DMA-BUF / host memory transfers from primary compute queues. |
| **`dmashaders`** | Memory Placement | Uploads compiled shaders to invisible/device VRAM using DMA. | Useful for systems where Resizable BAR (ReBAR) is constrained or disabled. |
| **`dccmsaa`** | Compression | Enables Delta Color Compression (DCC) for multi-sample images. | Useful if running MSAA passes or resolve blits. |
| **`localbos`** | Command Submission | Uses local buffer object lists per queue submission, minimizing driver mutex contention. | Beneficial during high-frequency DGC execution dispatches. |
| **`rtcps`** | Ray Tracing | Enables Ray Tracing Coarse Pixel Shading on supported hardware. | Experimental. |
| **`nosam`** | Memory Testing | Disables Smart Access Memory (Resizable BAR) emulation/support. | Diagnostic flag: Use to measure Resizable BAR performance uplift. |

---

### GPU Clock Profiles for Benchmarking (`RADV_PROFILE_PSTATE`)

Modern AMD GPU drivers utilize dynamic power management (DPM), which downclocks GPU cores during brief compute dispatches or light scenes. To eliminate clock flutter and ensure strictly deterministic frame timing:

```bash
# Force maximum performance clocks for repeatable benchmarking
RADV_PROFILE_PSTATE=peak ./build/bin/pathways --headless --benchmark --frames 200

# Or standard profile
RADV_PROFILE_PSTATE=standard ./build/bin/pathways --headless --benchmark
```

### Useful Debug Flags (`RADV_DEBUG`)

```bash
# Print detailed compiler statistics (VGPRs, SGPRs, LDS usage, and code size per microkernel)
RADV_DEBUG=shaderstats ./build/bin/pathways --pipeline wavefront

# Force synchronous shader compilation (prevents background hitching)
RADV_DEBUG=syncshaders ./build/bin/pathways
```

---

## Profiling & Developer Tools

- **AMD Developer Tools**: Fully compatible with the Radeon Developer Tool Suite (`/opt/RadeonDeveloperToolSuite-2026-05-28-1806/`), Radeon GPU Profiler (RGP), Radeon Raytracing Analyzer (RRA), and Radeon GPU Detective (RGD).
- **Automated Profiling Suite**:
  ```bash
  python3 scripts/benchmark_megakernel_vs_wavefront.py
  python3 scripts/deep_profile_scenes.py
  ```
- **Vulkan API & Platform Support Auditor**:
  ```bash
  # View multi-platform comparison (Desktop vs Mobile vs Global)
  python3 scripts/audit_vulkan_api.py -w raytracing --compare-platforms
  # Filter out mobile devices to view desktop PC metrics
  python3 scripts/audit_vulkan_api.py -w dgc --exclude-mobile
  # Generate interactive HTML dashboard
  python3 scripts/audit_vulkan_api.py --format html -o output/vulkan_api_dashboard.html
  ```

---

## Packages (Linux Reference)

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

