@echo off
REM Auto-install VB-Cable with admin rights
echo ========================================
echo VB-Cable Auto Installer
echo ========================================
echo.

REM Check for admin rights
net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Running with Administrator privileges
    goto :install
) else (
    echo [!] Requesting Administrator privileges...
    echo.
    powershell -Command "Start-Process '%~f0' -Verb RunAs"
    exit /b
)

:install
cd /d "%~dp0\.."

REM Run Python installer
echo [1/2] Running installer...
python scripts\installer.py
if %errorLevel% neq 0 (
    echo.
    echo [ERROR] Installation failed
    echo.
    pause
    exit /b 1
)

echo.
echo [2/2] Installation complete!
echo.
echo ========================================
echo IMPORTANT: Restart your computer now
echo ========================================
echo.
echo After restart, run: run.bat
echo.
pause
