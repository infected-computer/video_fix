cmake_minimum_required(VERSION 3.16)
project(FileFormatRepair VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /O2 /Ox /Oi /Ot /GL")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /LTCG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native -flto")
endif()

# General file format repair
set(GENERAL_SOURCES
    main.cpp
    file_format_detector.cpp
    format_repairers.cpp
)

set(GENERAL_HEADERS
    file_format_detector.h
    format_repairers.h
)

# Professional video format repair
set(PROFESSIONAL_SOURCES
    professional_main.cpp
    professional_video_formats.cpp
    red_camera_parser.cpp
    performance_optimizer.cpp
)

set(PROFESSIONAL_HEADERS
    professional_video_formats.h
    red_camera_parser.h
    arri_alexa_parser.h
    blackmagic_parser.h
    performance_optimizer.h
)

# Create general file repair executable
add_executable(file_repair ${GENERAL_SOURCES} ${GENERAL_HEADERS})

# Create professional video repair executable
add_executable(professional_video_repair ${PROFESSIONAL_SOURCES} ${PROFESSIONAL_HEADERS})

target_compile_features(file_repair PRIVATE cxx_std_17)
target_compile_features(professional_video_repair PRIVATE cxx_std_17)

if(WIN32)
    target_link_libraries(file_repair)
    target_link_libraries(professional_video_repair)
else()
    target_link_libraries(file_repair pthread)
    target_link_libraries(professional_video_repair pthread)
endif()

install(TARGETS file_repair professional_video_repair
    RUNTIME DESTINATION bin
)