@echo off
setlocal
cd /d "%~dp0"

set "TARGET_EXE=%~dp0build\windows-clang-release\bin\pathways.exe"
if not exist "%TARGET_EXE%" (
    if exist "%~dp0build\windows-gcc-release\bin\pathways.exe" (
        set "TARGET_EXE=%~dp0build\windows-gcc-release\bin\pathways.exe"
    )
)

if not exist "%TARGET_EXE%" (
    echo [INFO] Binary not found. Running Release build first...
    call "%~dp0build.bat" -Config Release
    if errorlevel 1 (
        echo [ERROR] Build failed! Cannot launch Pathways.
        exit /b %errorlevel%
    )
)

if not exist "%TARGET_EXE%" (
    echo [ERROR] Expected executable was not found at %TARGET_EXE%.
    exit /b 1
)

start "" "%TARGET_EXE%" %*
