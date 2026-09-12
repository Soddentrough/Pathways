# Building and Running Pathways on Windows 11

Pathways includes a modern, high-performance build system designed for Windows 11 targeting AMD RDNA3/RDNA4 architectures (and compatible Vulkan 1.4 hardware) with asynchronous processing, Device-Generated Commands (DGC), and hardware ray tracing.

---

## 1. Prerequisites

1. **Vulkan SDK** (>= 1.4.341):
   - Download and install the [Vulkan SDK for Windows](https://vulkan.lunarg.com/sdk/home).
   - Ensure the SDK installer sets `VULKAN_SDK` (e.g. `C:\VulkanSDK\1.4.357.0`).
   - `glslc.exe` is provided in `$env:VULKAN_SDK\Bin`.

2. **MSYS2 UCRT64 Toolchain** (Native Universal C Runtime):
   - Pathways leverages cutting-edge C++23 compilers linking against Microsoft Universal C Runtime (`ucrtbase.dll`).
   - Ensure the following packages are installed via pacman (`C:\msys64\usr\bin\pacman.exe`):
     ```bash
     pacman -S --noconfirm mingw-w64-ucrt-x86_64-clang \
                           mingw-w64-ucrt-x86_64-lld \
                           mingw-w64-ucrt-x86_64-gcc \
                           mingw-w64-ucrt-x86_64-ninja \
                           mingw-w64-ucrt-x86_64-sdl3 \
                           mingw-w64-ucrt-x86_64-zlib \
                           mingw-w64-ucrt-x86_64-python-pillow
     ```

3. **CMake** (>= 3.25):
   - Available via [cmake.org](https://cmake.org/download/) or MSYS2.

---

## 2. Quickstart: One-Command Build & Run

A PowerShell orchestrator (`build.ps1`) and batch wrapper (`build.bat`) are provided for single-command workflows:

### Quick Launch (Interactive GUI)
Double click [`run.bat`](file:///c:/Users/naoki/Development/Pathways/run.bat) or run from any terminal:
```powershell
.\run.ps1
```
*or from Command Prompt:*
```cmd
run
```

### Build in Release Mode (Clang 20 + LLD + Ninja)
```powershell
.\build.ps1
```
*or from Command Prompt:*
```cmd
build
```

### Build & Run Unit Tests
```powershell
.\build.ps1 -Test
```

### Build & Run Headless Verification Frame
```powershell
.\build.ps1 -Run -EngineArgs "--headless --frames 1 --width 1920 --height 1080 --dump-frame output/test.png"
```

### Build with GCC 15
```powershell
.\build.ps1 -Toolchain gcc -Config Release
```

### Clean Rebuild
```powershell
.\build.ps1 -Clean
```

### Interactive Controls
- **`ESC`**: Exit application / release camera lock.
- **`F11`**: Toggle Fullscreen.
- **`TAB`**: Toggle between UI Control Panel (free mouse) and FPS Navigation (mouse look).
- **`W / A / S / D` + `Q / E`**: Move through scene.
- **`Mouse Wheel`**: Adjust fly speed.

---

## 3. CMake Presets (CLI & IDEs)

Pathways includes modern CMake Presets (`CMakePresets.json`) compatible with VS Code, Visual Studio, CLion, and the command line:

### Available Configure Presets
- `windows-clang-release`: LLVM/Clang 20 + LLD, `-O3 -march=native`, Ninja.
- `windows-clang-debug`: LLVM/Clang 20 + LLD, `-O0 -g`, Ninja.
- `windows-gcc-release`: GCC 15, `-O3 -march=native`, Ninja.
- `windows-gcc-debug`: GCC 15, `-O0 -g`, Ninja.

### Manual CMake Commands
```powershell
# Configure
cmake --preset windows-clang-release

# Build
cmake --build --preset windows-release

# Test
ctest --preset windows-test
```

---

## 4. Automatic Dependency & Shader Deployment

Upon build completion, CMake's post-build pipeline automatically stages:
- **`SDL3.dll`** (from UCRT64)
- **`zlib1.dll`** (from UCRT64)
- **`libwinpthread-1.dll`** (POSIX threads runtime)
- **Compiled SPIR-V compute shaders** (`*.spv` into `bin/shaders/`)

The generated executable `build/<preset>/bin/pathways.exe` is completely self-sufficient and can be launched directly from Explorer or any terminal without manual DLL path modifications.

---

## 5. Automated Test Suite

To run the complete automated test suite (matching the Linux test harness):
```powershell
.\scripts\run_headless_tests.ps1
```

This runs:
1. Compilation check
2. Camera & FPS navigation unit tests
3. 1080p 16 SPP path tracing quality verification
4. 4K native real-time benchmark (<8ms frame budget target)
5. glTF scene ingestion tests (`DamagedHelmet.glb`)
6. Frame integrity and radiance verification via PIL analysis
