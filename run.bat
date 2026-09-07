@echo off
setlocal
cd /d "%~dp0"
if not exist "build\windows-clang-release\bin\pathways.exe" (
    echo [ERROR] Binary not found. Running build first...
    call "%~dp0build.bat" -Config Release
)
start "" "%~dp0build\windows-clang-release\bin\pathways.exe" %*
