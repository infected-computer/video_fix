# Fetch GoogleTest automatically if not found
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)
FetchContent_MakeAvailable(googletest)

# Find required Qt6 components for testing
find_package(Qt6 REQUIRED COMPONENTS Test)

# Define the test executable
add_executable(PhoenixDRS_Tests
    main.cpp
    test_example.cpp
    # Add other test files here
    # test_filecarver.cpp
)

# Link libraries to the test executable
target_link_libraries(PhoenixDRS_Tests
    PRIVATE
    gtest_main
    Qt6::Test
    ${PROJECT_NAME} # Link against the main application library to test its code
)

# Add test to CTest
include(GoogleTest)
gtest_discover_tests(PhoenixDRS_Tests)
