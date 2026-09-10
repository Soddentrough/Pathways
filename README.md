The goal is to create a pure Vulkan API based real-time path tracer and renderer engine using leading edge systems and methods such as asynchronous processing (vkCmdExecuteGeneratedCommandsEXT) and device generated commands (VK_EXT_device_generated_commands).

No megakernel, only efficient Device Generated Commands and moden ideas.

The baseline API version is 1.4.341.
Baseline hardware target is AMD's RDNA4 architecture (and RDNA3).

# Functions

- CLI options for various settings (resolution, samples per pixel, etc)
- A minimal GUI which allows changing these settings (such as sample rate) and interacting with the scene (camera location, view etc)
- Scene loading (glTF but ideally Blender and USD files)
- ACES tonemapping support
- EXR/HDRI texture support
- PBR material shader support
- Support for spot and area lights
- Ability to toggle on/off parts of the path tracer and renderer in real-time
- A stats panel showing frame time and profile breakdown for all parts of the pipeline

# Deliverables

Load a glTF scene and allow the user 'fly' through it in real time.

# Requiemts

Vulkan 1.4
SDL3

# Software and tools

/opt/RadeonDeveloperToolSuite-2026-05-28-1806/

# Windows 11 Build & Run

Pathways features full Windows 11 support using modern C++23 tooling, Clang 20 + LLD / GCC 15 (MSYS2 UCRT64), Ninja, and Vulkan SDK 1.4+.

To build and run:
```powershell
# Launch interactive GUI (auto-builds if needed)
.\run.ps1
# Or from cmd / double click: run.bat

# Build in Release mode using Clang 20 + LLD + Ninja
.\build.ps1

# Run unit tests
.\build.ps1 -Test

# Run full automated test suite
.\scripts\run_headless_tests.ps1
```
For complete details on presets, controls, and dependencies, see [BUILD_WINDOWS.md](BUILD_WINDOWS.md).

# Installed Packages (Linux Reference)

glslc-2026.1-1.fc44.x86_64
vulkan-loader-1.4.341.0-1.fc44.x86_64
vulkan-headers-1.4.341.0-1.fc44.noarch
vulkan-validation-layers-1.4.341.0-2.fc44.x86_64
vulkan-loader-devel-1.4.341.0-1.fc44.x86_64
vulkan-tools-1.4.341.0-1.fc44.x86_64
vulkan-utility-libraries-devel-1.4.341.0-1.fc44.x86_64
vulkan-loader-1.4.341.0-1.fc44.i686
mesa-vulkan-drivers-26.1.8-1.fc44.x86_64
mesa-vulkan-drivers-26.1.8-1.fc44.i686

# Thanks

Additional thanks to MrMPFR

# References

- https://docs.vulkan.org/spec/latest/chapters/device_generated_commands/generatedcommands.html
- https://vkdoc.net/chapters/device-generated-commands
- https://docs.vulkan.org/tutorial/latest/courses/18_Ray_tracing/00_Overview.html
- https://www.supergoodcode.com/device-generated-commands/
- https://gpuopen.com/learn/rgp-work-graphs/
- https://docs.amd.com/v/u/en-US/rdna4-instruction-set-architecture
- https://gpuopen.com/learn/rdna-performance-guide/
- https://gpuopen.com/learn/improving-rt-perf-with-rra/
