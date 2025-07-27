#!/bin/bash
# PhoenixDRS Professional C++ GUI Build Script for Linux/macOS
# מסגרת בנייה לממשק C++ של PhoenixDRS עבור Linux/macOS

set -e  # Exit on any error

echo "====================================="
echo "PhoenixDRS Professional C++ GUI Build"
echo "מסגרת בנייה לממשק C++ של PhoenixDRS"
echo "====================================="
echo

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if CMake is available
if ! command -v cmake &> /dev/null; then
    print_error "CMake is not installed or not in PATH"
    print_error "CMake לא מותקן או לא נמצא בPATH"
    echo "Please install CMake:"
    echo "  Ubuntu/Debian: sudo apt-get install cmake"
    echo "  CentOS/RHEL: sudo yum install cmake"
    echo "  macOS: brew install cmake"
    exit 1
fi

print_status "CMake version: $(cmake --version | head -n1)"

# Check if Qt6 is available
if [ -z "$Qt6_DIR" ]; then
    print_warning "Qt6_DIR environment variable not set"
    print_warning "משתנה הסביבה Qt6_DIR לא מוגדר"
    print_status "Trying to find Qt6..."
    
    # Common Qt6 installation paths
    QT6_PATHS=(
        "/usr/lib/cmake/Qt6"
        "/usr/local/lib/cmake/Qt6"
        "/opt/Qt/6.5.0/gcc_64/lib/cmake/Qt6"
        "/opt/Qt/6.4.0/gcc_64/lib/cmake/Qt6"
        "~/Qt/6.5.0/gcc_64/lib/cmake/Qt6"
        "~/Qt/6.4.0/gcc_64/lib/cmake/Qt6"
    )
    
    for path in "${QT6_PATHS[@]}"; do
        expanded_path=$(eval echo "$path")
        if [ -d "$expanded_path" ]; then
            export Qt6_DIR="$expanded_path"
            break
        fi
    done
    
    if [ -z "$Qt6_DIR" ]; then
        print_error "Qt6 not found. Please install Qt6 and set Qt6_DIR"
        print_error "Qt6 לא נמצא. אנא התקן Qt6 והגדר את Qt6_DIR"
        echo "Install Qt6:"
        echo "  Ubuntu/Debian: sudo apt-get install qt6-base-dev qt6-tools-dev"
        echo "  CentOS/RHEL: sudo yum install qt6-qtbase-devel qt6-qttools-devel"
        echo "  macOS: brew install qt6"
        exit 1
    fi
fi

print_status "Using Qt6 at: $Qt6_DIR"
print_status "משתמש ב-Qt6 ב: $Qt6_DIR"

# Detect number of CPU cores for parallel build
if command -v nproc &> /dev/null; then
    CORES=$(nproc)
elif command -v sysctl &> /dev/null; then
    CORES=$(sysctl -n hw.ncpu)
else
    CORES=4
fi

print_status "Building with $CORES parallel jobs"
print_status "בונה עם $CORES משימות מקבילות"

# Create build directory
mkdir -p build
cd build

print_status "Configuring build with CMake..."
print_status "מגדיר בנייה עם CMake..."

# Detect build system
if command -v ninja &> /dev/null; then
    GENERATOR="Ninja"
    BUILD_TOOL="ninja"
else
    GENERATOR="Unix Makefiles"
    BUILD_TOOL="make -j$CORES"
fi

print_status "Using build system: $GENERATOR"

# Configure with CMake
cmake .. \
    -G "$GENERATOR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DQt6_DIR="$Qt6_DIR" \
    -DCMAKE_PREFIX_PATH="$(dirname $(dirname $(dirname $Qt6_DIR)))" \
    -DCMAKE_INSTALL_PREFIX="../install" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

print_success "CMake configuration completed"
print_success "תצורת CMake הושלמה"

echo
print_status "Building project..."
print_status "בונה פרויקט..."

# Build the project
if [ "$GENERATOR" = "Ninja" ]; then
    ninja
else
    make -j$CORES
fi

print_success "Build completed successfully"
print_success "הבנייה הושלמה בהצלחה"

echo
print_status "Installing..."
print_status "מתקין..."

# Install the project
cmake --install . --config Release

print_success "Installation completed"
print_success "ההתקנה הושלמה"

cd ..

echo
echo "====================================="
print_success "Build completed successfully!"
print_success "הבנייה הושלמה בהצלחה!"
echo "====================================="
echo
print_status "Executable location: install/bin/PhoenixDRS_GUI"
print_status "מיקום הקובץ ההפעלה: install/bin/PhoenixDRS_GUI"
echo

# Check if executable exists and is runnable
if [ -x "install/bin/PhoenixDRS_GUI" ]; then
    echo -n "Run PhoenixDRS GUI now? (y/n) הפעל ממשק גרפי כעת? "
    read -r choice
    case "$choice" in
        y|Y|yes|YES)
            print_status "Starting PhoenixDRS GUI..."
            print_status "מפעיל ממשק גרפי..."
            ./install/bin/PhoenixDRS_GUI &
            ;;
        *)
            print_status "You can run the GUI later with: ./install/bin/PhoenixDRS_GUI"
            print_status "ניתן להפעיל את הממשק הגרפי מאוחר יותר עם: ./install/bin/PhoenixDRS_GUI"
            ;;
    esac
else
    print_error "Executable not found or not executable"
    print_error "קובץ הפעלה לא נמצא או לא ניתן להפעלה"
fi

echo
print_status "Build script completed"
print_status "סקריפט הבנייה הושלם"