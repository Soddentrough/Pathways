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

## 2. Toolchain & Build System

- **Compiler**: Clang 20 (`clang++.exe`) with LLD (`-fuse-ld=lld`) and Ninja via MSYS2 UCRT64 (`C:\msys64\ucrt64\bin`).
- **Standard**: C++23 (`-std=c++23`).
- **Vulkan SDK**: LunarG Vulkan SDK $\ge 1.4.341$ (`$env:VULKAN_SDK\Bin\glslc.exe`).
- **Presets**: Modern CMake Presets (`CMakePresets.json`):
  - Configure: `windows-clang-release`
  - Build: `windows-release`
  - Test: `windows-test`

---

## 3. Architecture & Engine Guardrails

- **Vulkan 1.4 DGC Compatibility**:
  - Single-dispatch DGC tokens (`VK_INDIRECT_COMMANDS_TOKEN_TYPE_DISPATCH_EXT`) are supported for compute pipelines.
  - Pipeline switching via `VkIndirectExecutionSetEXT` requires checking `VkPhysicalDeviceDeviceGeneratedCommandsPropertiesEXT::supportedIndirectCommandsShaderStagesPipelineBinding & VK_SHADER_STAGE_COMPUTE_BIT`. On hardware/drivers where this flag is 0 (such as AMD Windows drivers), the engine must gracefully route through the multi-dispatch indirect fallback.
  - Always ensure all child Vulkan resources (`WavefrontPipeline`, `DGCManager`) are destroyed prior to `VulkanContext` and `VkDevice` teardown.
- **Zero Validation Errors**:
  - All test runs and headless frame dumps must exit with `Validation Errors: 0`.

---

## 4. Quick Verification Commands

```powershell
# Build Release and run all 5 unit tests
.\build.ps1 -Test

# Run complete 6-stage automated test harness (1080p, 4K, glTF models, and validation checks)
.\scripts\run_headless_tests.ps1

# Quick 1-frame headless render test
.\build\windows-clang-release\bin\pathways.exe --headless --frames 1 --width 1280 --height 720 --dump-frame output/test.png
```
