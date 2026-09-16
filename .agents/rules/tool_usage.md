---
name: tool_usage
description: Enforce silent native read tools and consolidated build/test execution to minimize approval prompts.
trigger: always_on
---

# Tool Usage & Approval Minimization

1. **Native Read Tools Only**:
   - For all file reading, syntax inspection, and codebase exploration, use `view_file`.
   - For searching strings, patterns, symbols, and functions across the repo, use `grep_search`.
   - For finding files and listing directories, use `find_by_name` and `list_dir`.
   - Never run shell commands (`Select-String`, `Get-ChildItem`, `cat`, `dir`, `ls`, `grep`) for read-only inspection.

2. **Single-Command Orchestration**:
   - Always prefer `.\build.ps1 -Test` for compilation and unit test execution.
   - Always prefer `.\scripts\run_headless_tests.ps1` for end-to-end regression validation.
   - Do not invoke multiple shell commands when a single script handles the pipeline.

3. **Pre-Authorized & Auto-Approved Commands (Top 10 Safe Operations)**:
   The following safe, read-only inspection, diagnostic, and standard validation commands are pre-authorized and should be auto-approved without requiring manual user approval:
   1. `git status` (including `-s`, `--short`) — Working directory status check.
   2. `git diff` (including `--stat`, path-specific diffs) — Read-only code and stage diff inspection.
   3. `git log` (e.g. `git log -n <N> --oneline`, `git log HEAD..origin/main`) — History and branch divergence inspection.
   4. `git fetch` / `git branch` / `git remote` (e.g. `git fetch origin`, `git branch -a`, `git remote -v`) — Remote and branch queries.
   5. `.\build.ps1 -Test` (or `powershell -ExecutionPolicy Bypass -File .\build.ps1 -Test`) — Consolidated release build and unit test execution.
   6. `.\scripts\run_headless_tests.ps1` (or `powershell -ExecutionPolicy Bypass -File .\scripts\run_headless_tests.ps1`) — Full headless regression validation suite.
   7. `cmake --preset windows-clang-release` / `cmake --build --preset windows-release` — Standard CMake project configuration and build.
   8. `ninja -C build/windows-clang-release -j16` — Incremental compilation strictly capped at `-j16`.
   9. `ctest --test-dir build/windows-clang-release -R <regex> --output-on-failure` — Filtered unit test runner.
   10. `.\build\windows-clang-release\bin\pathways.exe --headless --frames 1 --width 1280 --height 720 --dump-frame <path>` (and `--help`) — Headless 1-frame smoke verification.

   *Native Tools (Always Auto-Approved & Preferred)*: `view_file`, `grep_search`, `find_by_name`, `list_dir`.

4. **Development Tools Allow List (Auto-Approved Tools & Binaries)**:
   All development operations involving compiling, linking, debugging, build generators, process inspection, and diagnostic binaries within the workspace are fully pre-authorized and auto-approved for autonomous execution:

   - **Compilers & Translators**:
     - `clang`, `clang++`, `clang-cl` (`C:\msys64\ucrt64\bin\clang++.exe`, etc.)
     - `gcc`, `g++` (`C:\msys64\ucrt64\bin\g++.exe`, etc.)
     - `glslc` (`$env:VULKAN_SDK\Bin\glslc.exe`, `C:\VulkanSDK\*\Bin\glslc.exe`)
     - `cl`, `cl.exe`
     - `rga` (Radeon GPU Analyzer)

   - **Linkers & Archivers**:
     - `lld`, `ld.lld`, `lld-link`
     - `ld`
     - `llvm-ar`, `ar`, `ranlib`
     - `link`, `link.exe`

   - **Debuggers, Profilers & Diagnostics**:
     - `gdb`, `gdb.exe` (`C:\msys64\ucrt64\bin\gdb.exe`)
     - `lldb`, `lldb.exe`
     - `cdb`, `cdb.exe`
     - `rgd`, `rra`, `rgp` (Radeon Developer Tool Suite)
     - `amd-smi`
     - `vulkaninfo`

   - **Build Systems, Make & Test Runners**:
     - `cmake`, `cmake.exe` (all configuration, preset, build, and target subcommands)
     - `ninja`, `ninja.exe` (`C:\msys64\ucrt64\bin\ninja.exe` — parallel jobs capped at `-j16`)
     - `make`, `mingw32-make`, `nmake`
     - `ctest`, `ctest.exe`

   - **Process Inspection & Management**:
     - `Get-Process`, `Stop-Process`, `Wait-Process` (PowerShell process viewing and lifecycle management)
     - `tasklist`, `taskkill` (Windows process management)

   - **Binary & Object Inspection**:
     - `dumpbin`
     - `objdump`, `llvm-objdump`
     - `nm`, `llvm-nm`
     - `strings`, `llvm-strings`
     - `readelf`, `llvm-readelf`

   - **Runtime Environments, Scripting & Package Management**:
     - `python`, `python3` (for test harnesses, regression verification, image quality checks, diagnostic scripts)
     - `pacman` (MSYS2 package inspection and environment sync)
     - `powershell`, `powershell.exe`, `pwsh` (running workspace `.ps1` automation scripts)
     - `cmd.exe`

   - **Project Artifacts & Binaries**:
     - `.\build.ps1`, `build.ps1` (with `-Test`, `-Clean`, etc.)
     - `.\scripts\run_headless_tests.ps1`, `.\scripts\run_headless_tests.sh`
     - All test and engine executables in `build\windows-clang-release\bin\*` (e.g. `pathways.exe`, `test_*.exe`)
     - `./build/bin/pathways`


