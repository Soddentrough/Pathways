# Building and Running Pathways on Linux

Pathways is designed for Linux workstations and gaming systems targeting modern AMD RDNA 3 / RDNA 4 architectures (and any conformant Vulkan 1.4 hardware). It features GPU-autonomous Device-Generated Commands (DGC), wavefront path tracing microkernels, zero-copy multi-GPU scaling, OpenUSD stage ingestion, and AMD FSR 3.1 super-resolution.

---

## 1. System Prerequisites & Hardware Baseline

- **Operating System**: Linux (Fedora 40+, Ubuntu 24.04+, Arch Linux, or modern distribution with Linux Kernel >= 6.8).
- **GPU Hardware**: 
  - Recommended: AMD Radeon AI PRO R9700 / RX 7000+ series (`gfx1201` / RDNA 4 or RDNA 3).
  - Compatible: Any discrete or integrated GPU supporting the Vulkan 1.4 specification (Vulkan Core >= 1.4.341).
- **Vulkan Driver**:
  - Mesa RADV 26.1+ (recommended for AMD RDNA series).
  - Proprietary or AMDVLK Vulkan 1.4-compliant driver.
- **CPU & RAM**: Threadripper or multi-core modern x86_64 CPU with >= 32 GB system memory. Note: Cap parallel builds (`-j16` or `-j32`) to prevent memory exhaustion during heavy template instantiation and shader baking.

---

## 2. Package Dependencies

Install the required C++23 compiler, build generator, Vulkan SDK headers, and runtime libraries using your distribution package manager:

### Fedora (Workstation / Server)
```bash
sudo dnf install git gcc-c++ clang lld cmake ninja-build \
                 vulkan-loader-devel vulkan-headers glslc \
                 SDL3-devel zlib-ng-compat-devel python3-pillow
```

### Ubuntu / Debian (24.04 LTS+)
```bash
sudo apt-get update
sudo apt-get install git g++ clang lld cmake ninja-build \
                     libvulkan-dev vulkan-tools glslc \
                     libsdl3-dev zlib1g-dev python3-pil
```

### Arch Linux
```bash
sudo pacman -S --needed git gcc clang lld cmake ninja \
                        vulkan-devel vulkan-headers shaderc \
                        sdl3 zlib python-pillow
```

---

## 3. Quickstart: Building Pathways

### Option A: Using CMake Presets (Recommended)

Pathways provides standardized configure, build, and test presets in [`CMakePresets.json`](../CMakePresets.json):

```bash
# 1. Configure in Release mode (-O3, native microarchitecture optimization)
cmake --preset linux-release

# 2. Compile binaries and shaders (capped at -j16)
cmake --build --preset linux-release -j16

# 3. Run unit test suite
ctest --preset linux-test
```

For a debug build with symbols (`-g -O0`):
```bash
cmake --preset linux-debug
cmake --build --preset linux-debug -j16
```

### Option B: Classic CMake & Ninja Workflow

```bash
# 1. Create build directory and configure
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DENABLE_NATIVE_OPTIMIZATION=ON

# 2. Build engine executable and compile SPIR-V shaders
ninja -C build -j16

# 3. Execute unit tests
ctest --test-dir build --output-on-failure
```

---

## 4. Running Pathways

The compiled binary is located at `build/linux-release/bin/pathways` (or `build/bin/pathways` when using the manual workflow).

### Interactive Real-Time Launch

```bash
# Launch interactive Cornell Box (default: fullscreen, native 4K, dual-binning wavefront)
./build/bin/pathways

# Launch OpenUSD production scene
./build/bin/pathways --scene scenes/PointInstancedMedCity/PointInstancedMedCity.usd

# Launch in windowed mode at 1440p capped at 120 FPS
./build/bin/pathways --scene scenes/coffee-maker/coffee_maker.usda --windowed --res 1440p --target-fps 120

# Launch glTF scene with AMD FSR 3.1 Super-Resolution
./build/bin/pathways --scene scenes/classroom/classroom_extended.glb --upscaler fsr3 --upscaler-preset quality
```

### Multi-GPU Scaling Configuration

Pathways supports multi-GPU rendering across dual AMD GPUs without requiring physical bridges:

```bash
# Zero-Copy Host Memory Streaming (Default & recommended for discrete PCIe)
./build/bin/pathways --scene scenes/PointInstancedMedCity/PointInstancedMedCity.usd --mgpu --mgpu-transfer host

# Linux DMA-BUF Direct P2P Streaming
./build/bin/pathways --scene scenes/PointInstancedMedCity/PointInstancedMedCity.usd --mgpu --mgpu-transfer p2p
```

### Headless Verification & 1-Frame Smoke Tests

```bash
# Dump a single 1080p verification frame
./build/bin/pathways --headless --frames 1 --width 1920 --height 1080 --dump-frame output/smoke_test.png

# High-precision 4K 16 SPP path traced frame dump
./build/bin/pathways --scene scenes/cornell-caustic/cornell_box_caustic.glb \
                    --headless --frames 16 --width 3840 --height 2160 \
                    --dump-frame output/cornell_4k.png
```

---

## 5. Interactive Navigation & Keybindings

When running in interactive GUI mode:

| Keybinding | Action |
|:---|:---|
| **`TAB`** | Toggle between **FPS Camera Look** (mouse captured) and **ImGui Control Panel** (free mouse cursor). |
| **`W / A / S / D`** | Dolly / Strafe camera forward, left, backward, and right. |
| **`Q / E`** | Crane camera down and up along world axis. |
| **`Shift`** | Hold to boost camera movement speed (fly speed multiplier). |
| **`Mouse Wheel`** | Incrementally adjust camera base fly speed. |
| **`F11`** | Toggle Fullscreen / Windowed display mode. |
| **`ESC`** | Release mouse cursor or exit application. |

---

## 6. Automated Testing & Verification Suites

Pathways includes an automated testing and regression suite:

```bash
# 1. Full headless test harness (12 regression checks & visual verifications)
./scripts/run_headless_tests.sh

# 2. CTest unit tests (covering cameras, descriptors, WMMA, and point instancing)
ctest --test-dir build --output-on-failure

# 3. Vulkan API specification auditor
python3 scripts/audit_vulkan_api.py -w all --exclude-mobile
```

---

## 7. Telemetry, Monitoring & Profiling

Comprehensive profiling guidance and the full diagnostic decision matrix are detailed in **[docs/PROFILING.md](PROFILING.md)**.

### Hardware Telemetry with AMD SMI
Use `/opt/rocm/core-10.0/bin/amd-smi` to monitor power, clock frequencies, memory controller saturation, and thermals on AMD GPUs:
```bash
# Live dashboard of clocks, power, thermals, engine %, and memory %:
/opt/rocm/core-10.0/bin/amd-smi monitor -u -m -p -t -v -w 1

# Full metric snapshot across all GPUs:
/opt/rocm/core-10.0/bin/amd-smi metric -g all

# Process resource consumption and engine occupancy:
/opt/rocm/core-10.0/bin/amd-smi process -G -e

# Multi-GPU PCIe & P2P DMA link topology:
/opt/rocm/core-10.0/bin/amd-smi topology -a -t -d -b
```

#### Key Telemetry Bottleneck Signals
* **`GFX_ACTIVITY` < 85%**: Host CPU bottleneck, swapchain presentation stall, or synchronization barrier wait (`vkWaitForFences`, `vkQueuePresentKHR`).
* **`UMC_ACTIVITY` > 75%**: Memory-bandwidth-bound (off-chip GDDR6 traffic saturation). Contrast with low UMC% (<30%) and 100% GFX%, which indicates an ALU/compute-bound workload.
* **`USED_VISIBLE_VRAM` at 256 MB**: Resizable BAR (Smart Access Memory) is disabled or staging buffers are spilling into dynamically paged system memory.
* **`USED_GTT` Non-Zero & Growing**: VRAM oversubscribed; allocations have spilled across PCIe into system RAM.
* **Clock Dips / Deep Sleep**: Sparse command submission or inter-frame idle bubbles causing the AMDGPU governor to drop P-states. Fix for benchmarks via `amd-smi set -l high -g all`.
* **Hotspot > 100°C / Power Limit**: Thermal or package power limit (TGP/PPT) throttling, verified via `amd-smi metric -v`.
* **PCIe Replay Count**: Non-zero `REPLAY_COUNT` in `amd-smi metric -P` indicates bus transmission errors or degraded riser cables.

### Driver Performance Hints & Compiler Diagnostics (Mesa RADV)
* **`RADV_PERFEST=1`**: Flags driver-level optimization gaps (uncompressed DCC fallback, slow blits, sub-optimal layouts).
* **`RADV_PERFTEST=rtcps`**: Toggles Continuation-Passing Style (CPS) lowering for ray tracing shaders.
* **`RADV_DEBUG=shaderstats,nocache`**: Dumps ACO compiler statistics (VGPRs, SGPRs, scratch memory spilling, and Wave32 occupancy).

### Profiling with Radeon Developer Tool Suite (RDTS)
Pathways is fully compatible with AMD RDTS tools located at `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/`:
- **Radeon GPU Profiler (RGP)**: Capture instruction-level wave execution, barrier stalls, and compute queue dispatches via `MESA_VK_TRACE=rgp`.
- **Radeon Raytracing Analyzer (RRA)**: Inspect hardware BLAS/TLAS acceleration structures, surface areas, and traversal costs via `MESA_VK_TRACE=rra`.
- **Radeon GPU Detective (RGD)**: Crash analysis and post-mortem page fault triage.
- **Radeon GPU Analyzer (RGA)**: Offline ISA shader compilation and VGPR register pressure auditing (`rga -s vk-spv-offline --asic gfx1201`).
