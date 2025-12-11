@echo off
REM Activate virtual environment and install dependencies

echo ========================================
echo  SoundBoardTool - Setup Environment
echo ========================================
echo.

REM Check if venv exists
if not exist "venv\" (
    echo Creating virtual environment...
    python -m venv venv
    echo Virtual environment created!
    echo.
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Upgrade pip
echo.
echo Upgrading pip...
python -m pip install --upgrade pip

REM Install main dependencies
echo.
echo Installing main dependencies...
if exist "requirements.txt" (
    pip install -r requirements.txt
) else (
    echo requirements.txt not found, skipping...
)

REM Install backup dependencies
echo.
echo Installing Google Drive backup dependencies...
if exist "requirements_backup.txt" (
    pip install -r requirements_backup.txt
) else (
    echo requirements_backup.txt not found, skipping...
)

echo.
echo ========================================
echo  Setup Complete!
echo ========================================
echo.
echo Virtual environment is now active.
echo To run the app: python src/app.py
echo To deactivate: deactivate
echo.

REM Keep the window open
cmd /k
