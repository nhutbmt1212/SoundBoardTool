@echo off
echo ========================================
echo  Python Auto-Installer
echo ========================================
echo.

REM Check if Python is already installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] Python is already installed!
    python --version
    echo.
    pause
    exit /b 0
)

echo [*] Python not found - installing automatically...
echo.

echo [1/3] Downloading Python 3.11.7...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe' -OutFile '%TEMP%\python_installer.exe'}"

if %errorlevel% neq 0 (
    echo [X] Download failed!
    echo [*] Please download manually from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [OK] Download complete
echo.

echo [2/3] Installing Python...
echo.
echo IMPORTANT: Python installer will show a progress bar
echo Please wait for it to complete (1-2 minutes)
echo.

REM Use /passive to show progress bar - more reliable
"%TEMP%\python_installer.exe" /passive PrependPath=1 Include_launcher=1

if %errorlevel% neq 0 (
    echo.
    echo [!] Installation may have failed
    echo [*] Trying alternative method...
    echo.
    "%TEMP%\python_installer.exe" /quiet InstallAllUsers=0 PrependPath=1
)

echo.
echo [OK] Installation complete
echo.

echo [3/3] Cleaning up...
del "%TEMP%\python_installer.exe" >nul 2>&1
echo [OK] Cleanup complete

echo.
echo ========================================
echo  Installation Complete!
echo ========================================
echo.
echo [!] IMPORTANT:
echo     1. Close this window
echo     2. Open a NEW command prompt
echo     3. Run: setup.bat
echo.
timeout /t 5 /nobreak >nul
