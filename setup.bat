@echo off
echo ====================================================
echo PhoenixDRS Professional - Windows Setup Script
echo ====================================================
echo.

:: Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] This script requires administrator privileges.
    echo Please run as administrator.
    pause
    exit /b 1
)

:: Set script directory
cd /d "%~dp0"

echo [INFO] Starting PhoenixDRS Professional setup...
echo.

:: Check Python installation
echo [STEP 1/7] Checking Python installation...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

:: Check Node.js installation
echo [STEP 2/7] Checking Node.js installation...
node --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Node.js not found. Please install Node.js 16+ from https://nodejs.org
    pause
    exit /b 1
)

:: Check CMake installation
echo [STEP 3/7] Checking CMake installation...
cmake --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] CMake not found. C++ components will not be built.
    echo You can install CMake from https://cmake.org
)

:: Install Python dependencies
echo [STEP 4/7] Installing Python dependencies...
python -m pip install --upgrade pip
pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo [ERROR] Failed to install Python dependencies.
    pause
    exit /b 1
)

:: Install Node.js dependencies
echo [STEP 5/7] Installing Node.js dependencies...
npm install
if %errorLevel% neq 0 (
    echo [ERROR] Failed to install Node.js dependencies.
    pause
    exit /b 1
)

:: Build C++ components (if CMake is available)
echo [STEP 6/7] Building C++ components...
cmake --version >nul 2>&1
if %errorLevel% equ 0 (
    cd src\cpp
    if not exist build mkdir build
    cd build
    cmake .. -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release
    if %errorLevel% equ 0 (
        cmake --build . --config Release --parallel
        if %errorLevel% equ 0 (
            echo [SUCCESS] C++ components built successfully.
        ) else (
            echo [WARNING] C++ build failed, but Python/JS components are ready.
        )
    ) else (
        echo [WARNING] CMake configuration failed.
    )
    cd ..\..\..
) else (
    echo [SKIP] CMake not available, skipping C++ build.
)

:: Build desktop application
echo [STEP 7/7] Building desktop application...
npm run build
if %errorLevel% neq 0 (
    echo [ERROR] Failed to build desktop application.
    pause
    exit /b 1
)

echo.
echo ====================================================
echo          PhoenixDRS Professional Setup Complete!
echo ====================================================
echo.
echo You can now start the application using:
echo   npm start                 # Desktop GUI
echo   python main.py --help     # Command line interface
echo.
echo For development mode:
echo   npm run dev               # Development GUI with hot reload
echo.
echo Documentation available at: docs/README.md
echo.
pause