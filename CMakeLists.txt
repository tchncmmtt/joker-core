cmake_minimum_required(VERSION 3.20)
project(joker_core VERSION 2.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

include(CheckIPOSupported)
check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)

find_package(OpenMP QUIET)

set(JOKER_CORE_SOURCES
        src/core/Card.cpp
        src/core/Deck.cpp
        src/core/GameState.cpp
        src/engine/JokerEngine.cpp
        src/engine/JokerVectorEnv.cpp
        src/joker_c_api.cpp
)

add_library(joker_core SHARED ${JOKER_CORE_SOURCES})

target_include_directories(joker_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found: enabling multi-threaded JokerVectorEnv")
    target_link_libraries(joker_core PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(joker_core PRIVATE
            $<$<CONFIG:Release>:/O2 /Ob3 /arch:AVX2 /fp:fast /DNDEBUG>
    )
else()
    target_compile_options(joker_core PRIVATE
            -Wall -Wextra -Wpedantic
            $<$<CONFIG:Release>:-O3 -march=native -mtune=native -ffast-math -funroll-loops -DNDEBUG>
    )
endif()

if(IPO_SUPPORTED)
    set_target_properties(joker_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
endif()

# Опциональная сборка CLI-исполняемого файла, если присутствует src/main_cli.cpp
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main_cli.cpp")
    add_executable(joker_cli src/main_cli.cpp)
    target_link_libraries(joker_cli PRIVATE joker_core)
endif()