@echo off
chcp 1255
REM PhoenixDRS Professional C++ GUI Build Script
REM מסגרת בנייה לממשק C++ של PhoenixDRS

echo =====================================
echo PhoenixDRS Professional C++ GUI Build
echo מסגרת בנייה לממשק C++ של PhoenixDRS  
echo =====================================
echo.

REM Check if CMake is available
cmake --version >nul 2>&1
if errorlevel 1 (
    echo Error: CMake is not installed or not in PATH
    echo שגיאה: CMake לא מותקן או לא נמצא בPATH
    echo Please install CMake from https://cmake.org/
    pause
    exit /b 1
)

REM Check if Qt6 is available
if "%Qt6_DIR%"=="" (
    echo Warning: Qt6_DIR environment variable not set
    echo אזהרה: משתנה הסביבה Qt6_DIR לא מוגדר
    echo Trying common Qt6 installation paths...
    
    set "Qt6_DIR=C:\Qt\6.5.0\msvc2022_64\lib\cmake\Qt6"
    if not exist "%Qt6_DIR%" (
        set "Qt6_DIR=C:\Qt\6.4.0\msvc2022_64\lib\cmake\Qt6"
    )
    if not exist "%Qt6_DIR%" (
        echo Error: Qt6 not found. Please install Qt6 and set Qt6_DIR
        echo שגיאה: Qt6 לא נמצא. אנא התקן Qt6 והגדר את Qt6_DIR
        pause
        exit /b 1
    )
    
    echo Found Qt6 at: %Qt6_DIR%
    echo נמצא Qt6 ב: %Qt6_DIR%
)

REM Create build directory
if not exist "build" mkdir build
cd build

echo Configuring build with CMake...
echo מגדיר בנייה עם CMake...

REM Configure with CMake
cmake .. -G "Visual Studio 17 2022" -A x64 ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DQt6_DIR="%Qt6_DIR%" ^
    -DCMAKE_PREFIX_PATH="%Qt6_DIR%\..\..\.." ^
    -DCMAKE_INSTALL_PREFIX="../install"

if errorlevel 1 (
    echo CMake configuration failed
    echo תצורת CMake נכשלה
    cd ..
    pause
    exit /b 1
)

echo.
echo Building project...
echo בונה פרויקט...

REM Build the project
cmake --build . --config Release --parallel

if errorlevel 1 (
    echo Build failed
    echo הבנייה נכשלה
    cd ..
    pause
    exit /b 1
)

echo.
echo Installing...
echo מתקין...

REM Install the project
cmake --install . --config Release

if errorlevel 1 (
    echo Installation failed
    echo ההתקנה נכשלה
    cd ..
    pause
    exit /b 1
)

cd ..

echo.
echo =====================================
echo Build completed successfully!
echo הבנייה הושלמה בהצלחה!
echo =====================================
echo.
echo Executable location: install\bin\PhoenixDRS_GUI.exe
echo מיקום הקובץ ההפעלה: install\bin\PhoenixDRS_GUI.exe
echo.

REM Ask if user wants to run the application
set /p "choice=Run PhoenixDRS GUI now? (y/n) הפעל ממשק גרפי כעת? "
if /i "%choice%"=="y" (
    if exist "install\bin\PhoenixDRS_GUI.exe" (
        echo Starting PhoenixDRS GUI...
        echo מפעיל ממשק גרפי...
        start "" "install\bin\PhoenixDRS_GUI.exe"
    ) else (
        echo Error: Executable not found
        echo שגיאה: קובץ הפעלה לא נמצא
    )
)

pause