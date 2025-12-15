@echo off
echo ==========================================
echo      ViralizeIt - Easy Launch Script
echo ==========================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python from https://www.python.org/
    pause
    exit /b
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
call venv\Scripts\activate

REM Install dependencies
echo [INFO] Checking and installing dependencies...
pip install -r requirements.txt >nul

REM Start the application
echo.
echo [INFO] Starting ViralizeIt...
echo [INFO] Web browser should open automatically.
echo.

REM Open browser in a separate process
start "" "http://localhost:5000"

REM Run the Flask app
python app.py

pause
