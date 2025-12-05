@echo off
cd /d "%~dp0"
echo ========================================
echo  Soundboard Pro - Auto Setup
echo ========================================
echo.

REM Step 1: Check Python
echo [Step 1/3] Checking Python installation...
echo.

REM Check if Python exists first
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python is already installed
    python --version
    echo.
    goto install_deps
)

REM Python not found - install it
echo [!] Python not found - installing automatically...
echo.
call check_python.bat

REM After Python install, need to restart
echo.
echo ========================================
echo [!] Python Installation Complete
echo ========================================
echo.
echo IMPORTANT: Python was just installed.
echo You need to:
echo   1. Close this window
echo   2. Open a NEW command prompt
echo   3. Run setup.bat again
echo.
echo Press any key to exit...
pause >nul
exit /b 0

:install_deps
echo ========================================
echo [Step 2/3] Installing Dependencies
echo ========================================
echo.
echo This will install:
echo   - Python packages (pygame, pyaudio, numpy)
echo   - VB-Audio Virtual Cable (for Discord/Game routing)
echo.
echo [*] Starting installation...
echo.

REM Step 2: Run installer
cd ..
python scripts\setup.py

if %errorlevel% neq 0 (
    echo.
    echo [X] Installation failed!
    echo [*] Try running as Administrator
    echo.
    pause
    exit /b 1
)

echo.
echo ========================================
echo [Step 3/3] Setup Complete!
echo ========================================
echo.
echo Next steps:
echo   1. Restart your computer (REQUIRED for VB-Cable)
echo   2. Run: run.bat
echo   3. Click "Audio Setup" to configure
echo.
echo Press any key to exit...
pause >nul
