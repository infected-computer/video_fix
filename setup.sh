#!/bin/bash

# PhoenixDRS Professional - Linux/macOS Setup Script
# This script sets up the complete PhoenixDRS development and runtime environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
        if command -v apt &> /dev/null; then
            DISTRO="debian"
        elif command -v yum &> /dev/null; then
            DISTRO="rhel"
        elif command -v pacman &> /dev/null; then
            DISTRO="arch"
        else
            DISTRO="unknown"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        DISTRO="macos"
    else
        OS="unknown"
        DISTRO="unknown"
    fi
}

# Check if command exists
command_exists() {
    command -v "$1" &> /dev/null
}

# Install system dependencies
install_system_deps() {
    log_info "Installing system dependencies for $OS ($DISTRO)..."
    
    case $DISTRO in
        "debian")
            sudo apt update
            sudo apt install -y build-essential cmake python3 python3-pip nodejs npm ffmpeg git
            ;;
        "rhel")
            sudo yum groupinstall -y "Development Tools"
            sudo yum install -y cmake python3 python3-pip nodejs npm ffmpeg git
            ;;
        "arch")
            sudo pacman -S --noconfirm base-devel cmake python python-pip nodejs npm ffmpeg git
            ;;
        "macos")
            if ! command_exists brew; then
                log_info "Installing Homebrew..."
                /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
            fi
            brew install cmake python@3.11 node ffmpeg git
            ;;
        *)
            log_warning "Unknown distribution. Please install dependencies manually:"
            log_warning "  - build-essential/Development Tools"
            log_warning "  - cmake (3.20+)"
            log_warning "  - python3 (3.9+) with pip"
            log_warning "  - nodejs (16+) with npm"
            log_warning "  - ffmpeg"
            log_warning "  - git"
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    local missing_deps=()
    
    if ! command_exists python3; then
        missing_deps+=("python3")
    fi
    
    if ! command_exists pip3; then
        missing_deps+=("pip3")
    fi
    
    if ! command_exists node; then
        missing_deps+=("nodejs")
    fi
    
    if ! command_exists npm; then
        missing_deps+=("npm")
    fi
    
    if ! command_exists cmake; then
        log_warning "CMake not found. C++ components will not be built."
    fi
    
    if ! command_exists ffmpeg; then
        log_warning "FFmpeg not found. Some video processing features may not work."
    fi
    
    if [ ${#missing_deps[@]} -ne 0 ]; then
        log_error "Missing required dependencies: ${missing_deps[*]}"
        log_info "Would you like to install them automatically? (y/N)"
        read -r response
        if [[ "$response" == "y" || "$response" == "Y" ]]; then
            install_system_deps
        else
            log_error "Please install missing dependencies and run this script again."
            exit 1
        fi
    fi
}

# Setup Python virtual environment
setup_python_env() {
    log_info "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        log_success "Virtual environment created."
    fi
    
    source venv/bin/activate
    log_info "Virtual environment activated."
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install requirements
    log_info "Installing Python dependencies..."
    pip install -r requirements.txt
    
    log_success "Python dependencies installed."
}

# Setup Node.js environment
setup_nodejs_env() {
    log_info "Setting up Node.js environment..."
    
    # Check Node.js version
    NODE_VERSION=$(node --version | cut -d 'v' -f 2 | cut -d '.' -f 1)
    if [ "$NODE_VERSION" -lt 16 ]; then
        log_warning "Node.js version $NODE_VERSION detected. Version 16+ recommended."
    fi
    
    # Install dependencies
    log_info "Installing Node.js dependencies..."
    npm install
    
    log_success "Node.js dependencies installed."
}

# Build C++ components
build_cpp_components() {
    if ! command_exists cmake; then
        log_warning "CMake not found. Skipping C++ build."
        return
    fi
    
    log_info "Building C++ components..."
    
    cd src/cpp
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure
    cmake .. -DCMAKE_BUILD_TYPE=Release
    
    # Build
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    
    # Run tests if available
    if [ -f "CTestTestfile.cmake" ]; then
        log_info "Running C++ tests..."
        ctest --output-on-failure
    fi
    
    cd ../../..
    
    log_success "C++ components built successfully."
}

# Build desktop application
build_desktop_app() {
    log_info "Building desktop application..."
    
    npm run build
    
    log_success "Desktop application built successfully."
}

# Create launcher scripts
create_launchers() {
    log_info "Creating launcher scripts..."
    
    # Desktop GUI launcher
    cat > start_gui.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
npm start
EOF
    chmod +x start_gui.sh
    
    # CLI launcher
    cat > phoenixdrs << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
python main.py "$@"
EOF
    chmod +x phoenixdrs
    
    # Development GUI launcher
    cat > start_dev.sh << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
source venv/bin/activate 2>/dev/null || true
npm run dev
EOF
    chmod +x start_dev.sh
    
    log_success "Launcher scripts created."
}

# Main setup function
main() {
    echo "===================================================="
    echo "     PhoenixDRS Professional - Setup Script"
    echo "===================================================="
    echo
    
    # Detect OS
    detect_os
    log_info "Detected OS: $OS ($DISTRO)"
    
    # Change to script directory
    cd "$(dirname "$0")"
    
    # Check prerequisites
    check_prerequisites
    
    # Setup Python environment
    setup_python_env
    
    # Setup Node.js environment
    setup_nodejs_env
    
    # Build C++ components
    build_cpp_components
    
    # Build desktop application
    build_desktop_app
    
    # Create launcher scripts
    create_launchers
    
    echo
    echo "===================================================="
    echo "        PhoenixDRS Professional Setup Complete!"
    echo "===================================================="
    echo
    echo "You can now start the application using:"
    echo "  ./start_gui.sh            # Desktop GUI"
    echo "  ./phoenixdrs --help       # Command line interface"
    echo "  ./start_dev.sh            # Development GUI with hot reload"
    echo
    echo "To activate the Python environment manually:"
    echo "  source venv/bin/activate"
    echo
    echo "Documentation available at: docs/README.md"
    echo
}

# Run main function
main "$@"