# PhoenixDRS Video Repair Engine
## מנוע תיקון וידאו מתקדם

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![C++17](https://img.shields.io/badge/C++-17-red.svg)](https://en.cppreference.com/w/cpp/17)
[![CMake](https://img.shields.io/badge/CMake-3.20+-blue.svg)](https://cmake.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)

PhoenixDRS Video Repair Engine is a high-performance C++ library for video file corruption analysis and repair. It features RAII memory management, thread-safe frame processing, and comprehensive container format support.

**🚀 Current Features:**
- **Video Container Analysis** for MP4, AVI, and MKV formats
- **Corruption Detection** with detailed analysis reports
- **Frame Reconstruction** using optical flow and feature-based interpolation
- **Thread-Safe Processing** with concurrent frame buffer management
- **RAII Memory Management** for FFmpeg and CUDA resources
- **Comprehensive Testing** with 70+ unit and integration tests

---

## 📋 Table of Contents

- [System Architecture](#-system-architecture)
- [Quick Start](#-quick-start)
- [Installation](#-installation)
- [Usage](#-usage)
- [Supported Formats](#-supported-formats)
- [AI Features](#-ai-features)
- [Development](#-development)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🏗️ System Architecture

The PhoenixDRS Video Repair Engine uses a modular C++ architecture:

```
┌─────────────────────────────────────────────────────────┐
│             Application Layer (User Code)               │
├─────────────────────────────────────────────────────────┤
│          AdvancedVideoRepairEngine (Main API)           │
├─────────────────────────────────────────────────────────┤
│  ContainerAnalyzer  │  FrameReconstructor  │  Utilities │
├─────────────────────────────────────────────────────────┤
│   FFmpeg RAII   │  OpenCV Processing  │   CUDA RAII    │
├─────────────────────────────────────────────────────────┤
│            Hardware Layer (CPU, GPU, Memory)            │
└─────────────────────────────────────────────────────────┘
```

### Core Components:
- **🔧 AdvancedVideoRepairEngine**: Main API for video analysis and repair
- **📦 ContainerAnalyzer**: MP4/AVI/MKV container structure analysis
- **🔄 FrameReconstructor**: Frame interpolation and corruption repair
- **🧵 ThreadSafeFrameBuffer**: Concurrent frame processing with shared_mutex
- **💾 RAII Wrappers**: Memory-safe FFmpeg and CUDA resource management

---

## 🚀 Quick Start

### Prerequisites
- **CMake 3.20+** for building
- **C++17 compatible compiler** (GCC 9+, Clang 10+, MSVC 2019+)
- **OpenCV 4.x** for image processing
- **FFmpeg libraries** (libavformat, libavcodec, libavutil, libswscale)
- **Google Test** (automatically downloaded)
- **CUDA Toolkit** (optional, for GPU acceleration)

### Build Instructions

#### Linux/macOS
```bash
# Install dependencies (Ubuntu/Debian)
sudo apt update
sudo apt install build-essential cmake libopencv-dev ffmpeg libavformat-dev libavcodec-dev

# Clone repository
git clone <repository-url>
cd "videoFix software"

# Build C++ engine
cd src/cpp
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)

# Run tests
cd tests
./VideoRepair_Tests
```

#### Windows
```cmd
# Install vcpkg (package manager)
git clone https://github.com/Microsoft/vcpkg.git
cd vcpkg
.\bootstrap-vcpkg.bat

# Install dependencies
.\vcpkg install opencv4 ffmpeg

# Build project
cd "videoFix software\src\cpp"
mkdir build && cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=C:\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build . --config Release

# Run tests
cd tests\Release
VideoRepair_Tests.exe
```

---

## 💿 Installation

### System Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **OS** | Windows 10, Ubuntu 20.04, macOS 11 | Windows 11, Ubuntu 22.04, macOS 13+ |
| **CPU** | 4 cores, 2.5GHz | 8+ cores, 3.0GHz+ |
| **RAM** | 8 GB | 32 GB+ |
| **GPU** | DirectX 11 compatible | NVIDIA RTX 3070+ with 8GB VRAM |
| **Storage** | 10 GB free space | 100 GB+ SSD |

### Windows Installation
```cmd
# Install Visual Studio Build Tools
winget install Microsoft.VisualStudio.2022.BuildTools

# Install Python
winget install Python.Python.3.11

# Install Node.js
winget install OpenJS.NodeJS

# Install FFmpeg
winget install Gyan.FFmpeg

# Clone and setup
git clone <repository-url>
cd "videoFix software"
pip install -r requirements.txt
npm install
npm run build:windows
```

### Linux Installation
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install build-essential cmake python3 python3-pip nodejs npm ffmpeg

# CentOS/RHEL/Fedora
sudo yum groupinstall "Development Tools"
sudo yum install cmake python3 python3-pip nodejs npm ffmpeg

# Arch Linux
sudo pacman -S base-devel cmake python python-pip nodejs npm ffmpeg

# Setup project
git clone <repository-url>
cd "videoFix software"
pip3 install -r requirements.txt
npm install
npm run build:linux
```

### macOS Installation
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install cmake python@3.11 node ffmpeg

# Setup project
git clone <repository-url>
cd "videoFix software"
pip3 install -r requirements.txt
npm install
npm run build:mac
```

---

## 🎯 Usage

### Basic C++ API Example
```cpp
#include "AdvancedVideoRepair/AdvancedVideoRepairEngine.h"

using namespace AdvancedVideoRepair;

int main() {
    // Initialize the video repair engine
    auto engine = std::make_unique<AdvancedVideoRepairEngine>();
    if (!engine->initialize()) {
        std::cerr << "Failed to initialize engine" << std::endl;
        return -1;
    }
    
    // Analyze video corruption
    CorruptionAnalysis analysis = engine->analyze_corruption("input.mp4");
    std::cout << "Corruption level: " << analysis.overall_corruption_percentage << "%" << std::endl;
    std::cout << "Issues found: " << analysis.detected_issues.size() << std::endl;
    
    // Configure repair strategy
    RepairStrategy strategy;
    strategy.enable_motion_compensation = true;
    strategy.enable_post_processing = true;
    strategy.preserve_original_quality = true;
    
    // Repair the video
    AdvancedRepairResult result = engine->repair_video_file(
        "input.mp4", "output.mp4", strategy
    );
    
    if (result.success) {
        std::cout << "Repair completed successfully!" << std::endl;
        std::cout << "Processing time: " << result.processing_time.count() << "ms" << std::endl;
    }
    
    engine->shutdown();
    return 0;
}
```

### Thread-Safe Frame Processing
```cpp
#include "AdvancedVideoRepair/ThreadSafeFrameBuffer.h"
#include "AdvancedVideoRepair/FrameReconstructor.h"

using namespace VideoRepair;

// Create thread-safe frame buffer
ThreadSafeFrameBuffer frame_buffer(100);  // capacity: 100 frames

// Producer thread: add frames
std::thread producer([&frame_buffer]() {
    cv::Mat frame = cv::imread("frame.jpg");
    if (frame_buffer.push_frame(frame)) {
        std::cout << "Frame added successfully" << std::endl;
    }
});

// Consumer thread: process frames
std::thread consumer([&frame_buffer]() {
    if (frame_buffer.size() > 0) {
        cv::Mat frame = frame_buffer.get_frame(0);
        // Process frame...
    }
});

producer.join();
consumer.join();
```

### RAII Memory Management Example
```cpp
#include "AdvancedVideoRepair/FFmpegUtils.h"

using namespace VideoRepair;

void process_video(const std::string& filename) {
    // RAII wrappers automatically manage memory
    AVFormatContextPtr format_ctx;
    AVCodecContextPtr codec_ctx;
    AVFramePtr frame;
    AVPacketPtr packet;
    
    // Open input file
    AVFormatContext* ctx = avformat_alloc_context();
    format_ctx.reset(ctx);
    
    if (avformat_open_input(format_ctx.get_ptr(), filename.c_str(), nullptr, nullptr) < 0) {
        throw std::runtime_error("Could not open file");
    }
    
    // Find stream info
    if (avformat_find_stream_info(format_ctx.get(), nullptr) < 0) {
        throw std::runtime_error("Could not find stream info");
    }
    
    // All resources automatically cleaned up when leaving scope
}
```

---

## 📹 Supported Formats

### Currently Implemented Formats
| Container Format | Analysis Support | Repair Support | Notes |
|------------------|------------------|----------------|-------|
| **MP4** | ✅ Full | ✅ Full | ftyp, moov, mdat box analysis |
| **AVI** | ✅ Full | ✅ Full | RIFF/AVI structure analysis |
| **MKV** | ✅ Full | ✅ Full | EBML/Matroska structure analysis |

### Codec Support (via FFmpeg)
| Video Codec | Detection | Frame Processing | Notes |
|-------------|-----------|------------------|-------|
| **H.264 (AVC)** | ✅ Yes | ✅ Yes | Most common format |
| **H.265 (HEVC)** | ✅ Yes | ✅ Yes | High efficiency |
| **MPEG-4** | ✅ Yes | ✅ Yes | Legacy support |
| **VP8/VP9** | ✅ Yes | ✅ Yes | WebM containers |
| **AV1** | ✅ Yes | ✅ Yes | Modern codec |

### Analysis Capabilities
- **Container Structure Analysis**: Parse and validate container headers and metadata
- **Stream Information Extraction**: Detect video/audio streams and properties
- **Corruption Detection**: Identify corrupted regions and estimate damage level
- **File Integrity Validation**: Check structural integrity of video containers
- **Format Detection**: Automatically identify container and codec formats

### Repair Capabilities
- **Header Reconstruction**: Repair corrupted container headers (ftyp, moov, etc.)
- **Frame Interpolation**: Reconstruct missing frames using optical flow
- **Corrupted Region Repair**: Fix damaged areas using reference frames
- **Thread-Safe Processing**: Concurrent frame processing for performance

---

## 🛠️ Development

### Building from Source

#### C++ Video Repair Engine
```bash
cd src/cpp
mkdir build && cd build

# Debug build with all tests
cmake .. -DCMAKE_BUILD_TYPE=Debug
make -j$(nproc)

# Release build with optimizations
cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_CUDA=ON
make -j$(nproc)

# Run comprehensive test suite
cd tests
./VideoRepair_Tests --gtest_verbose

# Run specific test categories
./VideoRepair_Tests --gtest_filter="*MemoryStress*"
./VideoRepair_Tests --gtest_filter="*Integration*"
```

### Project Structure
```
PhoenixDRS Video Repair Engine/
├── src/cpp/
│   ├── include/AdvancedVideoRepair/
│   │   ├── AdvancedVideoRepairEngine.h      # Main API
│   │   ├── ContainerAnalyzer.h              # MP4/AVI/MKV analysis
│   │   ├── FrameReconstructor.h             # Frame interpolation
│   │   ├── ThreadSafeFrameBuffer.h          # Thread-safe processing
│   │   ├── FFmpegUtils.h                    # FFmpeg RAII wrappers
│   │   └── CudaUtils.h                      # CUDA RAII wrappers
│   ├── src/
│   │   ├── AdvancedVideoRepairEngine.cpp
│   │   ├── ContainerAnalyzer.cpp
│   │   ├── FrameReconstructor.cpp
│   │   └── ThreadSafeFrameBuffer.cpp
│   ├── tests/                               # 70+ comprehensive tests
│   │   ├── test_video_repair_engine.cpp     # Main engine tests
│   │   ├── test_container_analyzer.cpp      # Container analysis tests
│   │   ├── test_frame_reconstructor.cpp     # Frame reconstruction tests
│   │   ├── test_ffmpeg_integration.cpp      # FFmpeg RAII tests
│   │   ├── test_cuda_kernels.cu             # CUDA functionality tests
│   │   └── test_integration.cpp             # Integration/performance tests
│   └── CMakeLists.txt                       # Build configuration
├── todo.md                                  # Implementation requirements
└── README.md                               # This file
```

### Testing Framework
```bash
# Run all 70+ tests
./VideoRepair_Tests

# Test categories:
# - Unit tests (50+ tests): Core functionality validation
# - Integration tests (15+ tests): Large file processing, parallel operations
# - Memory tests (5+ tests): RAII wrapper validation, memory stress testing
# - CUDA tests (10+ tests): GPU memory management and kernel execution

# View test coverage
./VideoRepair_Tests --gtest_list_tests

# Performance benchmarks
./VideoRepair_Tests --gtest_filter="*Performance*"
./VideoRepair_Tests --gtest_filter="*Stress*"
```

### Memory Safety & Thread Safety
The engine features comprehensive RAII memory management:
- **FFmpeg RAII Wrappers**: Automatic cleanup of AVFormatContext, AVCodecContext, AVFrame, AVPacket
- **CUDA RAII Wrappers**: Safe GPU memory management with CudaDeviceBuffer, CudaStreamPtr, CudaEventPtr  
- **Thread-Safe Frame Processing**: SharedMutex-based concurrent access with atomic operations
- **Exception Safety**: All operations are exception-safe with proper cleanup

### Code Quality Standards
- **C++17**: Modern C++ features with move semantics and smart pointers
- **Memory Safety**: Zero raw pointer usage, comprehensive RAII patterns
- **Thread Safety**: Proper synchronization with shared_mutex and atomic operations
- **Testing**: 60%+ code coverage with unit, integration, and stress tests
- **Documentation**: Inline documentation with usage examples

---

## 🛠️ Troubleshooting

### Common Build Issues

#### FFmpeg Not Found
```bash
# Ubuntu/Debian
sudo apt install libavformat-dev libavcodec-dev libavutil-dev libswscale-dev

# macOS
brew install ffmpeg

# Windows (vcpkg)
vcpkg install ffmpeg
```

#### OpenCV Not Found
```bash
# Ubuntu/Debian  
sudo apt install libopencv-dev

# macOS
brew install opencv

# Windows (vcpkg)
vcpkg install opencv4
```

#### CUDA Build Errors
```bash
# Ensure CUDA Toolkit is installed
nvcc --version

# Set CUDA_ROOT if needed
export CUDA_ROOT=/usr/local/cuda

# Build without CUDA if not needed
cmake .. -DENABLE_CUDA=OFF
```

### Runtime Issues

#### "Cannot open video file"
- Check file path and permissions
- Verify FFmpeg can decode the format: `ffprobe input.mp4`
- Ensure video file is not corrupted beyond repair

#### Memory Allocation Errors
- Reduce buffer capacity in ThreadSafeFrameBuffer constructor
- Check available system memory
- Use smaller test files for development

#### Test Failures
```bash
# Run individual test suites
./VideoRepair_Tests --gtest_filter="FFmpegIntegrationTest.*"
./VideoRepair_Tests --gtest_filter="CudaKernelsTest.*"

# Skip CUDA tests if GPU not available
./VideoRepair_Tests --gtest_filter="-*Cuda*"
```

---

## 🤝 Contributing

Contributions are welcome! Please ensure:
1. All tests pass: `./VideoRepair_Tests`
2. Code follows C++17 standards with RAII patterns
3. New features include corresponding tests
4. Documentation is updated for API changes

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- **FFmpeg Team** for multimedia processing framework
- **OpenCV Community** for computer vision algorithms  
- **Google Test** for comprehensive testing framework
- **CUDA Team** for GPU computing platform

---

**PhoenixDRS Video Repair Engine** - Professional C++ video corruption analysis and repair 🔧📹