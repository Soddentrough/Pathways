<#
.SYNOPSIS
    Convenience launcher for the Pathways Vulkan 1.4 Path Tracer.
.EXAMPLE
    .\run.ps1
    .\run.ps1 -Scene assets/models/cornell_box.gltf
#>
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$EngineArgs
)

Set-Location $PSScriptRoot
$exe = Join-Path $PSScriptRoot "build\windows-clang-release\bin\pathways.exe"

if (-not (Test-Path $exe)) {
    Write-Host "[INFO] Executable not found. Building project..." -ForegroundColor Cyan
    & "$PSScriptRoot\build.ps1" -Config Release
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Build failed!"
        exit $LASTEXITCODE
    }
}

Write-Host "[INFO] Launching Pathways..." -ForegroundColor Green
Start-Process -FilePath $exe -ArgumentList $EngineArgs -WorkingDirectory $PSScriptRoot
