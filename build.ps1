<#
.SYNOPSIS
    High-Performance Build & Run Orchestrator for Pathways on Windows 11.

.DESCRIPTION
    Configures, compiles, tests, and runs the Pathways Vulkan 1.4 Path Tracer.
    Utilizes CMake Presets, Clang 20 (LLD) / GCC 15 via MSYS2 UCRT64, and Ninja.

.PARAMETER Config
    Build configuration: "Release" (default), "Debug".

.PARAMETER Toolchain
    Compiler toolchain: "clang" (default, LLVM + LLD), "gcc" (GCC 15).

.PARAMETER Clean
    Wipes the corresponding build directory before configuring.

.PARAMETER Test
    Runs unit tests (CameraControls) via CTest after compilation.

.PARAMETER Run
    Executes pathways.exe after building.

.PARAMETER EngineArgs
    Arguments to pass directly to pathways.exe when -Run is specified.

.EXAMPLE
    .\build.ps1
    Builds the project in Release mode using Clang 20 + LLD + Ninja.

.EXAMPLE
    .\build.ps1 -Test
    Builds in Release mode and runs the CTest unit test suite.

.EXAMPLE
    .\build.ps1 -Run -EngineArgs "--headless --frames 1 --dump-frame test.png"
    Builds and executes 1 frame in headless mode.

.EXAMPLE
    .\build.ps1 -Toolchain gcc -Config Debug
    Builds in Debug mode using GCC 15.
#>

[CmdletBinding()]
param (
    [ValidateSet("Release", "Debug")]
    [string]$Config = "Release",

    [ValidateSet("clang", "gcc")]
    [string]$Toolchain = "clang",

    [switch]$Clean,
    [switch]$Test,
    [switch]$Run,
    [string]$EngineArgs = ""
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host "`n==> [Pathways] $Message" -ForegroundColor Cyan
}

function Write-Success {
    param([string]$Message)
    Write-Host "[PASS] $Message" -ForegroundColor Green
}

function Write-Failure {
    param([string]$Message)
    Write-Host "[FAIL] $Message" -ForegroundColor Red
}

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ScriptDir

# 1. Ensure MSYS2 UCRT64 toolchain is in PATH
$UcrtBin = "C:\msys64\ucrt64\bin"
if (Test-Path $UcrtBin) {
    if ($env:PATH -notlike "*$UcrtBin*") {
        $env:PATH = "$UcrtBin;$env:PATH"
    }
} else {
    Write-Warning "MSYS2 UCRT64 not found at $UcrtBin. Ensure toolchain is in system PATH."
}

# 2. Ensure Vulkan SDK is located
if (-not $env:VULKAN_SDK) {
    if (Test-Path "C:\VulkanSDK") {
        $latestSdk = Get-ChildItem "C:\VulkanSDK" -Directory | Sort-Object Name -Descending | Select-Object -First 1
        if ($latestSdk) {
            $env:VULKAN_SDK = $latestSdk.FullName
        }
    }
}

# 3. Locate CMake
$CMake = Get-Command "cmake.exe" -ErrorAction SilentlyContinue
if (-not $CMake) {
    if (Test-Path "C:\Program Files\CMake\bin\cmake.exe") {
        $CMake = "C:\Program Files\CMake\bin\cmake.exe"
    } else {
        throw "CMake not found! Please install CMake or ensure it is in PATH."
    }
} else {
    $CMake = $CMake.Source
}

# Determine Preset names
$ConfigLower = $Config.ToLower()
$ConfigurePreset = "windows-$Toolchain-$ConfigLower"
$BuildPreset = if ($Toolchain -eq "clang") {
    "windows-$ConfigLower"
} else {
    "windows-gcc-$ConfigLower"
}
$TestPreset = if ($Toolchain -eq "clang") {
    "windows-test"
} else {
    "windows-gcc-test"
}

$BuildDir = Join-Path $ScriptDir "build\$ConfigurePreset"
$BinDir = Join-Path $BuildDir "bin"
$PathwaysExe = Join-Path $BinDir "pathways.exe"

Write-Host "==========================================================" -ForegroundColor Magenta
Write-Host "  Pathways: Modern Windows 11 Build Orchestrator" -ForegroundColor Magenta
Write-Host "  Toolchain: $Toolchain | Config: $Config | Generator: Ninja" -ForegroundColor Magenta
Write-Host "  Presets: Config='$ConfigurePreset', Build='$BuildPreset'" -ForegroundColor Magenta
Write-Host "==========================================================" -ForegroundColor Magenta

# Clean if requested
if ($Clean) {
    Write-Step "Cleaning build directory: $BuildDir"
    if (Test-Path $BuildDir) {
        Remove-Item -Recurse -Force $BuildDir
    }
}

# Configure
Write-Step "Configuring project with preset: $ConfigurePreset"
$swConfig = [System.Diagnostics.Stopwatch]::StartNew()
& $CMake --preset $ConfigurePreset
if ($LASTEXITCODE -ne 0) {
    Write-Failure "CMake configure failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
$swConfig.Stop()
Write-Success "Configuration complete in $($swConfig.Elapsed.TotalSeconds.ToString("F2"))s."

# Build
Write-Step "Building target pathways with preset: $BuildPreset"
$swBuild = [System.Diagnostics.Stopwatch]::StartNew()
& $CMake --build --preset $BuildPreset
if ($LASTEXITCODE -ne 0) {
    Write-Failure "Build failed with exit code $LASTEXITCODE."
    exit $LASTEXITCODE
}
$swBuild.Stop()
Write-Success "Build complete in $($swBuild.Elapsed.TotalSeconds.ToString("F2"))s."

# Test if requested
if ($Test) {
    Write-Step "Running test suite with preset: $TestPreset"
    $CTest = Get-Command "ctest.exe" -ErrorAction SilentlyContinue
    if (-not $CTest) {
        $CTest = Join-Path (Split-Path -Parent $CMake) "ctest.exe"
    } else {
        $CTest = $CTest.Source
    }

    & $CTest --preset $TestPreset
    if ($LASTEXITCODE -ne 0) {
        Write-Failure "Unit tests failed!"
        exit $LASTEXITCODE
    }
    Write-Success "All unit tests passed cleanly!"
}

# Run if requested
if ($Run) {
    Write-Step "Launching Pathways ($PathwaysExe)..."
    if (-not (Test-Path $PathwaysExe)) {
        throw "Binary not found at $PathwaysExe!"
    }

    if ($EngineArgs) {
        $argList = @()
        $tokens = [System.Management.Automation.Language.Parser]::Tokenize("fake_cmd $EngineArgs", [ref]$null, [ref]$null) |
                  Where-Object { $_.Kind -ne 'EndOfInput' }
        for ($i = 1; $i -lt $tokens.Count; $i++) {
            $argList += $tokens[$i].Text.Trim('"', "'")
        }
        & $PathwaysExe @argList
    } else {
        & $PathwaysExe
    }

    if ($LASTEXITCODE -ne 0) {
        Write-Failure "Pathways exited with code $LASTEXITCODE."
        exit $LASTEXITCODE
    }
}

Write-Host "`nAll operations completed successfully.`n" -ForegroundColor Green
