# PhoenixDRS Professional C++ GUI
# High-Performance Native Implementation with Qt6
# מימוש C++ מקורי בביצועים גבוהים עם Qt6

cmake_minimum_required(VERSION 3.20)

project(PhoenixDRS_GUI 
    VERSION 2.0.0
    DESCRIPTION "PhoenixDRS Professional Data Recovery Suite - Native C++ GUI"
    LANGUAGES CXX
)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required Qt6 components
find_package(Qt6 REQUIRED COMPONENTS 
    Core 
    Widgets 
    Concurrent 
    Network
    Sql
    Test
)

# Enable automatic Qt6 features
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

# Compiler-specific options for performance
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # MSVC optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /GL /DNDEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    # GCC/Clang optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -Wall -Wextra")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Source files
set(SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/WorkerThread.cpp
    src/LogViewer.cpp
    src/SettingsDialog.cpp
    src/DiskImager.cpp
    src/FileCarver.cpp
    src/RaidReconstructor.cpp
    src/VideoRebuilder.cpp
    src/ForensicLogger.cpp
    src/PerformanceMonitor.cpp
    src/CaseManager.cpp
    src/NewCaseDialog.cpp

)

# Header files
set(HEADERS
    include/MainWindow.h
    include/WorkerThread.h
    include/LogViewer.h
    include/SettingsDialog.h
    include/DiskImager.h
    include/FileCarver.h
    include/RaidReconstructor.h
    include/VideoRebuilder.h
    include/ForensicLogger.h
    include/PerformanceMonitor.h
    include/Common.h
)

# UI files
set(UI_FILES
    ui/MainWindow.ui
    ui/SettingsDialog.ui
)

# Resource files
set(RESOURCES
    resources/resources.qrc
)

# Create executable
add_executable(PhoenixDRS_GUI ${SOURCES} ${HEADERS} ${UI_FILES} ${RESOURCES})

# Link Qt6 libraries
target_link_libraries(PhoenixDRS_GUI
    Qt6::Core
    Qt6::Widgets
    Qt6::Concurrent
    Qt6::Network
    Qt6::Sql
)

# Platform-specific libraries
if(WIN32)
    # Windows-specific libraries
    target_link_libraries(PhoenixDRS_GUI
        setupapi
        advapi32
        user32
        kernel32
    )
elseif(UNIX AND NOT APPLE)
    # Linux-specific libraries
    target_link_libraries(PhoenixDRS_GUI
        pthread
        dl
    )
endif()

# Install configuration
install(TARGETS PhoenixDRS_GUI
    RUNTIME DESTINATION bin
    COMPONENT Runtime
)

# Install Qt6 dependencies (Windows)
if(WIN32)
    find_program(WINDEPLOYQT_EXECUTABLE windeployqt HINTS ${Qt6_DIR}/../../../bin)
    if(WINDEPLOYQT_EXECUTABLE)
        install(CODE "execute_process(COMMAND ${WINDEPLOYQT_EXECUTABLE} \${CMAKE_INSTALL_PREFIX}/bin/PhoenixDRS_GUI.exe)")
    endif()
endif()

# Testing
enable_testing()
add_subdirectory(tests)

# Package configuration
set(CPACK_PACKAGE_NAME "PhoenixDRS_Professional")
set(CPACK_PACKAGE_VERSION_MAJOR ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH ${PROJECT_VERSION_PATCH})
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "Professional Data Recovery Suite")
set(CPACK_PACKAGE_VENDOR "PhoenixDRS Project")

if(WIN32)
    set(CPACK_GENERATOR "NSIS")
    set(CPACK_NSIS_DISPLAY_NAME "PhoenixDRS Professional")
    set(CPACK_NSIS_PACKAGE_NAME "PhoenixDRS Professional")
    set(CPACK_NSIS_HELP_LINK "https://github.com/phoenixdrs")
elseif(UNIX AND NOT APPLE)
    set(CPACK_GENERATOR "DEB;RPM")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "PhoenixDRS Team")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libqt6core6, libqt6widgets6, libqt6concurrent6")
    set(CPACK_RPM_PACKAGE_LICENSE "Commercial")
endif()

include(CPack)