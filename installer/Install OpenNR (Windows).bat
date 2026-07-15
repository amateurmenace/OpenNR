@echo off
rem OpenNR installer — double-click to install OpenNR.ofx.bundle (next to this
rem script) into the system-wide OpenFX plugin folder. Self-elevates to admin.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Requesting administrator access...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

set "SRC=%~dp0OpenNR.ofx.bundle"
set "DST=C:\Program Files\Common Files\OFX\Plugins\OpenNR.ofx.bundle"

if not exist "%SRC%" (
    echo OpenNR.ofx.bundle not found next to this script.
    pause
    exit /b 1
)

if not exist "C:\Program Files\Common Files\OFX\Plugins" mkdir "C:\Program Files\Common Files\OFX\Plugins"
if exist "%DST%" rmdir /s /q "%DST%"
xcopy /E /I /Y "%SRC%" "%DST%" >nul

if exist "%DST%" (
    echo.
    echo Installed. Restart DaVinci Resolve, then find it under:
    echo   Color page - Effects - OpenFX - Filters - OpenNR - OpenNR Denoise
) else (
    echo Installation failed.
)
pause
