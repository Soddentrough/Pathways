# Pathways - Agent Instructions & Workspace Rules

This repository contains **Pathways**, a pure Vulkan 1.4 hardware ray-tracing and wavefront path tracer targeting modern GPUs (AMD RDNA 3 / RDNA 4 and compatible Vulkan 1.4 hardware).

---

## 1. Approval Minimization & Tool Usage Policy

To prevent unnecessary interactive approval prompts for the user:

- **Always Use Native Read Tools for Codebase Exploration**:
  - Use `view_file` to read source code, shaders, configurations, and documentation.
  - Use `grep_search` to search for strings, classes, functions, and symbols across the codebase.
  - Use `find_by_name` and `list_dir` to find files and locate directories.
  - **NEVER** run terminal commands (e.g. PowerShell `Select-String`, `Get-ChildItem`, `cat`, `dir`, `ls`, or `grep`) for read-only inspection. Built-in read tools execute silently, whereas shell commands trigger interactive approval prompts.

- **Consolidate Build and Test Invocations**:
  - Do not execute piecemeal compilation or inspection commands in series.
  - Use the project's single-command orchestrators:
    - Build in Release mode: `powershell -ExecutionPolicy Bypass -File .\build.ps1`
    - Build and run unit tests: `powershell -ExecutionPolicy Bypass -File .\build.ps1 -Test`
    - Run full automated test suite & validation: `powershell -ExecutionPolicy Bypass -File .\scripts\run_headless_tests.ps1`
  - When building or running tasks in the background, wait for reactive notifications rather than polling task status in a loop.

- **Batch File Edits**:
  - Make cohesive, verified code edits rather than repeated single-line edit-and-test loops.

---

## 2. GPU & Hardware Specifications

- **Target GPUs**: 2x AMD Radeon AI PRO R9700 (32GB GDDR6 each, total 64GB VRAM).
- **Target Architecture**: `gfx1201` (RDNA 4).
- **CPU & RAM**: AMD Ryzen Threadripper 3970X (32 cores / 64 threads) with 64GB RAM.
- **Job Concurrency Limit**: Always cap parallel build jobs to `-j16` or `-j32` (e.g. `cmake --build build -j16` or `ninja -C build -j16`). Never launch unbounded parallel tasks.

---

## 3. Toolchain & Build System

### Windows Dev Environment
- **Compiler**: Clang 20 (`clang++.exe`) with LLD (`-fuse-ld=lld`) and Ninja via MSYS2 UCRT64 (`C:\msys64\ucrt64\bin`).
- **Standard**: C++23 (`-std=c++23`).
- **Vulkan SDK**: LunarG Vulkan SDK $\ge 1.4.341$ (`$env:VULKAN_SDK\Bin\glslc.exe`).
- **Presets**: Modern CMake Presets (`CMakePresets.json`):
  - Configure: `windows-clang-release`
  - Build: `windows-release`
  - Test: `windows-test`

### Linux Production & Profiling Environment
- **Vulkan Shader Compiler**: `glslc` (Vulkan 1.4 target: `glslc --target-env=vulkan1.4`)
- **Vulkan Diagnostics**: `vulkaninfo`
- **Vulkan Headers**: `/usr/include/vulkan`
- **SDL3 Headers**: `/usr/include/SDL3`
- **C/C++ Compilers**: `g++`, `gcc` (C++23)
- **Debugger**: `gdb`
- **Build Generators**: `cmake`, `ninja`

#### GPU Monitoring & Metrics (Linux)
Never run `find /` or search commands to locate core profiling, monitoring, or compiler tools. Use these exact paths:
- **`amd-smi`**:
  - Primary: `/opt/rocm/core-10.0/bin/amd-smi`
  - User bin: `/home/naoki/.local/bin/amd-smi`
- **`rocm-smi`**:
  - Primary: `/opt/rocm/core-10.0/bin/rocm-smi`
  - User bin: `/home/naoki/.local/bin/rocm-smi`
- **Instruction**: Check VRAM utilization using `/opt/rocm/core-10.0/bin/amd-smi` directly before and during intensive GPU rendering or benchmarking. Do not probe the filesystem for SMI tools.

#### Radeon Developer Tool Suite (RDTS)
Suite root: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/`
- **RGA (Radeon GPU Analyzer)**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/rga`
- **RRA (Radeon Raytracing Analyzer)**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonRaytracingAnalyzer`
- **RGP (Radeon GPU Profiler)**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonGPUProfiler`
- **RGD (Radeon GPU Detective / Crash Dumps)**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/rgd`
- **Developer Panel CLI**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonDeveloperPanelCLI`
- **Developer Service CLI**: `/opt/RadeonDeveloperToolSuite-2026-05-28-1806/RadeonDeveloperServiceCLI`

---

## 4. Architecture & Engine Guardrails

- **Vulkan 1.4 DGC Compatibility**:
  - Single-dispatch DGC tokens (`VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT`) are supported for compute pipelines.
  - Pipeline switching via `VkIndirectExecutionSetEXT` requires checking `VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT::supportedIndirectCommandsShaderStagesPipelineBinding & VK_SHADER_STAGE_COMPUTE_BIT`. On hardware/drivers where this flag is 0 (such as AMD Windows drivers), the engine must gracefully route through the multi-dispatch indirect fallback.
  - Always ensure all child Vulkan resources (`WavefrontPipeline`, `DGCManager`) are destroyed prior to `VulkanContext` and `VkDevice` teardown.
- **Zero Validation Errors**:
  - All test runs and headless frame dumps must exit with `Validation Errors: 0`.

---

## 5. Quick Verification Commands

### Windows (PowerShell)
```powershell
# Build Release and run all 5 unit tests
.\build.ps1 -Test

# Run complete 6-stage automated test harness (1080p, 4K, glTF models, and validation checks)
.\scripts\run_headless_tests.ps1

# Quick 1-frame headless render test
.\build\windows-clang-release\bin\pathways.exe --headless --frames 1 --width 1280 --height 720 --dump-frame output/test.png
```

### Linux (Bash)
```bash
# Configure & Build
cmake -B build -S . -G Ninja
cmake --build build -j16

# Run headless tests
./scripts/run_headless_tests.sh
```
