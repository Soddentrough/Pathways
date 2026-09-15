# Pathways Agent Guidelines & Core Tools Environment

## GPU & Hardware Specifications
- **Target GPUs**: 2x AMD Radeon AI PRO R9700 (32GB GDDR6 each, total 64GB VRAM).
- **Target Architecture**: `gfx1201` (RDNA 4).
- **CPU & RAM**: AMD Ryzen Threadripper 3970X (32 cores / 64 threads) with 64GB RAM.
- **Job Concurrency Limit**: Always cap parallel build jobs to `-j16` or `-j32` (e.g. `cmake --build build -j16` or `ninja -C build -j16`). Never launch unbounded parallel tasks.

## Core Tools Inventory (DO NOT Search Across Runs)
Never run `find /` or search commands to locate core profiling, monitoring, or compiler tools. Always use these exact paths:

### 1. GPU Monitoring & Metrics
- Metrics / Monitoring: `/opt/rocm/core-10.0/bin/amd-smi` (Always run outside sandbox / BypassSandbox: true to access /opt).
- **Instruction**: Check VRAM utilization using `/opt/rocm/core-10.0/bin/amd-smi` directly before and during intensive GPU rendering or benchmarking. Do not probe the filesystem for SMI tools.

### 2. Radeon Developer Tool Suite (RDTS)
Suite root: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/`
- **RGA (Radeon GPU Analyzer)**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/rga`
- **RRA (Radeon Raytracing Analyzer)**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonRaytracingAnalyzer`
- **RGP (Radeon GPU Profiler)**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonGPUProfiler`
- **RGD (Radeon GPU Detective / Crash Dumps)**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/rgd`
- **Developer Panel CLI**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonDeveloperPanelCLI`
- **Developer Service CLI**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonDeveloperServiceCLI`

### 3. Compilers & Vulkan Toolchain
- **Vulkan Shader Compiler**: `glslc` (Vulkan 1.4 target: `glslc --target-env=vulkan1.4`)
- **Vulkan Diagnostics**: `vulkaninfo`
- **Vulkan Headers**: `/usr/include/vulkan`
- **SDL3 Headers**: `/usr/include/SDL3`
- **C/C++ Compilers**: `g++`, `gcc` (C++23)
- **Debugger**: `gdb`
- **Build Generators**: `cmake`, `ninja`

## Project Build & Test Workflows
- **Build Directory**: `build`
- **Configure**: `cmake -B build -S . -G Ninja`
- **Build**: `cmake --build build -j16` (or `ninja -C build -j16`)
- **Main Binary**: `./build/bin/pathways`
- **Headless Tests**: `./scripts/run_headless_tests.sh`
- **Execution Policy**: Standard build tools (`cmake`, `ninja`, `g++`, `gdb`) and project binaries are pre-authorized.

## Proportional Verification & Test Execution
Always calibrate verification depth strictly to the nature and scope of the modifications. Never launch heavy end-to-end test suites for minor or non-functional edits.

1. **Tier 0: Non-Functional Edits (Comments, Documentation, Formatting, Config Strings)**:
   - **Action**: DO NOT run `./scripts/run_headless_tests.sh`, `python3 tests/test_image_quality.py`, or benchmark suites.
   - If C++/GLSL files were touched, at most run a fast incremental compile check (`ninja -C build -j16`) to confirm syntax. Report back immediately without running regression suites.

2. **Tier 1: Targeted Local Logic & Component Changes**:
   - **Action**: Run only the specific relevant test binary or CTest filter (e.g. `ctest -R CameraControls`).
   - Do not trigger full-engine headless or multi-GPU regression sweeps unless multiple subsystems are touched.

3. **Tier 2: Core Rendering Pipelines, Shaders, Synchronization & Multi-GPU**:
   - **Action**: Run the full headless regression suite (`./scripts/run_headless_tests.sh`) and image quality verification (`python3 tests/test_image_quality.py`).
   - Use only when shader algorithms, memory barriers, DGC dispatch queues, or multi-GPU transport paths are modified, or when explicitly requested by the user.

## Command-Line Option & CLI Guidelines
- **Binary / Mutually Exclusive Options Rule**: When an option represents a mutually exclusive binary condition, there must be only **ONE** option, which is to negate the default state.
  - If a feature is **disabled by default**, provide only the flag to enable it (e.g., `--bmfr`). Do NOT add a redundant `--no-<feature>` flag.
  - If a feature is **enabled by default**, provide only the flag to disable/negate it (e.g., `--no-double-buffer`, `--no-temporal-accum`). Do NOT add a redundant positive flag.
  - Never introduce redundant pairs of opposing flags (e.g., having both `--bmfr` and `--no-bmfr` when BMFR is disabled by default).

# GENERAL 

Do not make assumptions. Do not guess. Check code, check recent commits, use performance profiling tools. Verify everything and be sure of things before you say them. 
