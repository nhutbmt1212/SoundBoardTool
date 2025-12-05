@echo off
cd /d "%~dp0\.."
title Soundboard Pro

echo ========================================
echo  Soundboard Pro
echo ========================================
echo.

REM Quick Python check
echo Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [X] Python Not Found!
    echo.
    echo Python is not installed or not in PATH.
    echo.
    echo Please run: setup.bat first
    echo Or install Python from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Python found
python --version
echo.
echo Starting Soundboard Pro...
echo.

python src\main.py

if errorlevel 1 (
    echo.
    echo ========================================
    echo  Error: Failed to start
    echo ========================================
    echo.
    echo Possible solutions:
    echo   1. Run setup.bat first to install dependencies
    echo   2. Check error message above
    echo   3. Try: pip install -r requirements.txt
    echo.
    pause
)
