<#
.SYNOPSIS
    Pathways Automated Headless Test Suite & Frame Capture for Windows 11.

.DESCRIPTION
    Compiles with Release preset, runs unit tests, executes headless 1080p and 4K benchmarks,
    ingests glTF scenes, dumps output frames/stats, and validates output with Python.
#>

[CmdletBinding()]
param (
    [ValidateSet("clang", "gcc")]
    [string]$Toolchain = "clang"
)

$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RootDir = Split-Path -Parent $ScriptDir
Set-Location $RootDir

# Ensure MSYS2 UCRT64 toolchain and Python are in PATH
$UcrtBin = "C:\msys64\ucrt64\bin"
if (Test-Path $UcrtBin) {
    if ($env:PATH -notlike "*$UcrtBin*") {
        $env:PATH = "$UcrtBin;$env:PATH"
    }
}

$Python = "C:\msys64\ucrt64\bin\python.exe"
if (-not (Test-Path $Python)) {
    $Python = (Get-Command "python.exe" -ErrorAction SilentlyContinue).Source
}

Write-Host "==========================================================" -ForegroundColor Magenta
Write-Host "  Pathways: Automated Headless Test Suite & Frame Capture" -ForegroundColor Magenta
Write-Host "  Windows 11 Native Test Runner (Toolchain: $Toolchain)" -ForegroundColor Magenta
Write-Host "==========================================================" -ForegroundColor Magenta

# 1. Build project
Write-Host "`n[1/7] Building project via build.ps1..." -ForegroundColor Cyan
.\build.ps1 -Toolchain $Toolchain -Config Release
if ($LASTEXITCODE -ne 0) {
    Write-Error "Build failed!"
    exit $LASTEXITCODE
}

$BinDir = if ($Toolchain -eq "clang") {
    Join-Path $RootDir "build\windows-clang-release\bin"
} else {
    Join-Path $RootDir "build\windows-gcc-release\bin"
}

$PathwaysExe = Join-Path $BinDir "pathways.exe"
$TestExe = Join-Path $BinDir "test_camera_controls.exe"
$OutputDir = Join-Path $RootDir "output"
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

# 2. Run camera controls unit test suite
Write-Host "`n[2/7] Running Camera & FPS Navigation Unit Tests..." -ForegroundColor Cyan
& $TestExe
if ($LASTEXITCODE -ne 0) {
    Write-Error "Camera unit tests failed!"
    exit $LASTEXITCODE
}

# 3. Test Suite 1: 1080p 16 SPP Full Quality Verification
Write-Host "`n[3/7] Running Test Suite 1: 1080p @ 16 SPP (PNG + OpenEXR + Stats)..." -ForegroundColor Cyan
& $PathwaysExe `
    --headless `
    --width 1920 `
    --height 1080 `
    --spp 16 `
    --max-bounces 4 `
    --dump-frame (Join-Path $OutputDir "test_cornell_1080p.png") `
    --dump-hdr (Join-Path $OutputDir "test_cornell_1080p.exr") `
    --dump-stats (Join-Path $OutputDir "stats_1080p.json")

& $Python scripts/verify_frame.py (Join-Path $OutputDir "test_cornell_1080p.png") (Join-Path $OutputDir "stats_1080p.json") 1920 1080 40.0
if ($LASTEXITCODE -ne 0) {
    Write-Error "Test Suite 1 verification failed!"
    exit $LASTEXITCODE
}

# 4. Test Suite 2: 4K Native Real-Time Benchmark (<8ms Target)
Write-Host "`n[4/7] Running Test Suite 2: 4K Native (3840x2160) @ 1 SPP (Benchmark Mode)..." -ForegroundColor Cyan
& $PathwaysExe `
    --headless `
    --width 3840 `
    --height 2160 `
    --spp 1 `
    --frames 200 `
    --warmup-frames 30 `
    --no-accumulation `
    --benchmark `
    --dump-frame (Join-Path $OutputDir "test_cornell_4k.png") `
    --dump-stats (Join-Path $OutputDir "stats_4k.json")

& $Python scripts/verify_frame.py (Join-Path $OutputDir "test_cornell_4k.png") (Join-Path $OutputDir "stats_4k.json") 3840 2160 10.0
if ($LASTEXITCODE -ne 0) {
    Write-Error "Test Suite 2 verification failed!"
    exit $LASTEXITCODE
}

# 5. Test Suite 3: glTF 2.0 Ingestion Pipeline (Damaged Helmet)
Write-Host "`n[5/7] Running glTF Damaged Helmet scene test..." -ForegroundColor Cyan
& $PathwaysExe `
    --headless `
    --width 1920 `
    --height 1080 `
    --spp 16 `
    --max-bounces 4 `
    --scene scenes/DamagedHelmet.glb `
    --dump-frame (Join-Path $OutputDir "test_gltf_helmet.png") `
    --dump-stats (Join-Path $OutputDir "stats_gltf_helmet.json")

& $Python scripts/verify_frame.py (Join-Path $OutputDir "test_gltf_helmet.png") (Join-Path $OutputDir "stats_gltf_helmet.json") 1920 1080 45.0
if ($LASTEXITCODE -ne 0) {
    Write-Error "glTF Damaged Helmet verification failed!"
    exit $LASTEXITCODE
}

# 6. Test Suite 4: Many-Lights Scene (64 Lights) Procedural Cornell Box
Write-Host "`n[6/7] Running Many-Lights (64 Lights) Scene Test..." -ForegroundColor Cyan
& $PathwaysExe `
    --headless `
    --width 1920 `
    --height 1080 `
    --spp 4 `
    --max-bounces 4 `
    --scene many-lights `
    --dump-frame (Join-Path $OutputDir "test_many_lights.png") `
    --dump-stats (Join-Path $OutputDir "stats_many_lights.json")

& $Python scripts/verify_frame.py (Join-Path $OutputDir "test_many_lights.png") (Join-Path $OutputDir "stats_many_lights.json") 1920 1080 40.0
if ($LASTEXITCODE -ne 0) {
    Write-Error "Many-Lights scene verification failed!"
    exit $LASTEXITCODE
}

# 7. Test Suite 5: glTF Research Scene (Veach Ajar)
$VeachAjarPath = Join-Path $RootDir "scenes\veach-ajar\veach_ajar_extended.glb"
if (Test-Path $VeachAjarPath) {
    Write-Host "`n[7/7] Running glTF Veach Ajar research scene test..." -ForegroundColor Cyan
    & $PathwaysExe `
        --headless `
        --width 1920 `
        --height 1080 `
        --spp 4 `
        --max-bounces 4 `
        --frames 10 `
        --scene "scenes/veach-ajar/veach_ajar_extended.glb" `
        --dump-frame (Join-Path $OutputDir "test_veach_ajar.png") `
        --dump-stats (Join-Path $OutputDir "stats_veach_ajar.json")

    & $Python scripts/verify_frame.py (Join-Path $OutputDir "test_veach_ajar.png") (Join-Path $OutputDir "stats_veach_ajar.json") 1920 1080 30.0 --max-mean-lum 0.85 --max-blown-pct 25.0
    if ($LASTEXITCODE -ne 0) {
        Write-Error "glTF Veach Ajar verification failed!"
        exit $LASTEXITCODE
    }
}

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host "  Pathways: All Automated Windows 11 Tests Passed Cleanly!" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
