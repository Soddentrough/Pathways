<#
.SYNOPSIS
    Convenience launcher for the Pathways Vulkan 1.4 Path Tracer.
.PARAMETER Toolchain
    Compiler toolchain: "clang", "gcc", or "auto" (default, detects existing build).
.EXAMPLE
    .\run.ps1
    .\run.ps1 -Scene scenes/DamagedHelmet.glb
    .\run.ps1 -Toolchain gcc
#>
param(
    [ValidateSet("clang", "gcc", "auto")]
    [string]$Toolchain = "auto",

    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$EngineArgs
)

Set-Location $PSScriptRoot

$clangExe = Join-Path $PSScriptRoot "build\windows-clang-release\bin\pathways.exe"
$gccExe = Join-Path $PSScriptRoot "build\windows-gcc-release\bin\pathways.exe"

$exe = $null
if ($Toolchain -eq "clang") {
    $exe = $clangExe
} elseif ($Toolchain -eq "gcc") {
    $exe = $gccExe
} else {
    if (Test-Path $clangExe) {
        $exe = $clangExe
    } elseif (Test-Path $gccExe) {
        $exe = $gccExe
    } else {
        $exe = $clangExe
    }
}

if (-not (Test-Path $exe)) {
    $targetToolchain = if ($Toolchain -eq "gcc") { "gcc" } else { "clang" }
    Write-Host "[INFO] Executable not found. Building project with $targetToolchain..." -ForegroundColor Cyan
    & "$PSScriptRoot\build.ps1" -Toolchain $targetToolchain -Config Release
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed!"
        exit $LASTEXITCODE
    }
}

Write-Host "[INFO] Launching Pathways ($exe)..." -ForegroundColor Green
if ($EngineArgs -and $EngineArgs.Count -gt 0) {
    Start-Process -FilePath $exe -ArgumentList $EngineArgs -WorkingDirectory $PSScriptRoot
} else {
    Start-Process -FilePath $exe -WorkingDirectory $PSScriptRoot
}
