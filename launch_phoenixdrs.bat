@echo off
title PhoenixDRS Professional - Digital Recovery Suite
echo.
echo ========================================================
echo   PhoenixDRS Professional v2.0.0
echo   Advanced Digital Recovery and Forensics Suite
echo ========================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or later and try again.
    pause
    exit /b 1
)

REM Check if required packages are installed
echo Checking dependencies...
python -c "import PySide6; print('✓ PySide6 GUI framework')" 2>nul || (
    echo Installing required packages...
    pip install -r requirements.txt
)

REM Launch the application
echo.
echo Starting PhoenixDRS Professional...
echo.
python main.py

if %errorlevel% neq 0 (
    echo.
    echo Application encountered an error. Check the logs above.
    pause
)