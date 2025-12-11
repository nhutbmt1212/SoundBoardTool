@echo off
REM Run SoundBoardTool with virtual environment

REM Check if venv exists
if not exist "venv\" (
    echo Virtual environment not found!
    echo Please run setup_env.bat first.
    pause
    exit /b 1
)

REM Activate virtual environment
call venv\Scripts\activate.bat

REM Run the application
echo Starting SoundBoardTool...
python src/app.py

REM Deactivate when done
deactivate
