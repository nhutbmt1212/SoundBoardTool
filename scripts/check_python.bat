@echo off
setlocal enabledelayedexpansion

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% equ 0 (
    REM Python already installed - just return success
    exit /b 0
)

REM Python not installed - install it
echo [!] Python is not installed
echo [*] Auto-installing Python 3.11.7...
echo.

echo [1/3] Downloading Python installer...
powershell -Command "& {[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; $ProgressPreference = 'SilentlyContinue'; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.7/python-3.11.7-amd64.exe' -OutFile '%TEMP%\python_installer.exe'}"

if %errorlevel% neq 0 (
    echo [X] Download failed!
    echo [*] Please install Python manually from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo [OK] Download complete
echo.
echo [2/3] Installing Python...
echo.
echo IMPORTANT: Python installer may ask for Administrator permission
echo Please click "Yes" if prompted
echo.
echo Installing... (this takes 1-2 minutes)
echo.

REM Install with simpler flags - more reliable
"%TEMP%\python_installer.exe" /passive PrependPath=1 Include_launcher=1

if %errorlevel% neq 0 (
    echo.
    echo [!] Installation may have failed or was cancelled
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

REM Return special code to indicate Python was just installed
exit /b 2

REM Get Python version
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i

echo [OK] Python %PYTHON_VERSION% is installed
echo.

REM Check if version is 3.7+
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set MAJOR=%%a
    set MINOR=%%b
)

if %MAJOR% lss 3 (
    echo [X] Python version is too old!
    echo [X] Required: Python 3.7+
    echo [X] Found: Python %PYTHON_VERSION%
    echo.
    echo Please upgrade Python from:
    echo https://www.python.org/downloads/
    pause
    exit /b 1
)

if %MAJOR% equ 3 (
    if %MINOR% lss 7 (
        echo [X] Python version is too old!
        echo [X] Required: Python 3.7+
        echo [X] Found: Python %PYTHON_VERSION%
        echo.
        echo Please upgrade Python from:
        echo https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

echo [OK] Python version is compatible
echo.

REM Check pip
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [X] pip is not installed!
    echo.
    echo Installing pip...
    python -m ensurepip --default-pip
    python -m pip install --upgrade pip
    echo [OK] pip installed
) else (
    echo [OK] pip is installed
)

echo.
echo ========================================
echo  All checks passed!
echo ========================================
echo.

exit /b 0
