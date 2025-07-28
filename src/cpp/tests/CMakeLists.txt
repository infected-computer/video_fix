# Fetch GoogleTest automatically if not found
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
FetchContent_MakeAvailable(googletest)

# Find required packages
find_package(OpenCV REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFMPEG REQUIRED libavformat libavcodec libavutil libswscale)

# Optional CUDA support
find_package(CUDA QUIET)

# Define the test executable with all test files
set(TEST_SOURCES
    main.cpp
    test_video_repair_engine.cpp
    test_container_analyzer.cpp
    test_frame_reconstructor.cpp
    test_ffmpeg_integration.cpp
    test_integration.cpp
)

# Add CUDA test if available
if(CUDA_FOUND)
    enable_language(CUDA)
    list(APPEND TEST_SOURCES test_cuda_kernels.cu)
    add_definitions(-DHAVE_CUDA)
endif()

add_executable(VideoRepair_Tests ${TEST_SOURCES})

# Set C++ standard
set_property(TARGET VideoRepair_Tests PROPERTY CXX_STANDARD 17)

# Include directories
target_include_directories(VideoRepair_Tests PRIVATE 
    ${CMAKE_SOURCE_DIR}/src/cpp/include
    ${OpenCV_INCLUDE_DIRS}
    ${FFMPEG_INCLUDE_DIRS}
)

# Link libraries to the test executable
target_link_libraries(VideoRepair_Tests
    PRIVATE
    gtest_main
    ${OpenCV_LIBS}
    ${FFMPEG_LIBRARIES}
)

# Add CUDA libraries if available
if(CUDA_FOUND)
    target_link_libraries(VideoRepair_Tests PRIVATE ${CUDA_LIBRARIES})
endif()

# Compiler flags
target_compile_options(VideoRepair_Tests PRIVATE ${FFMPEG_CFLAGS_OTHER})

# Add test to CTest
include(GoogleTest)
gtest_discover_tests(VideoRepair_Tests)
