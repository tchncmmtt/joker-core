cmake_minimum_required(VERSION 3.20)
project(joker_core VERSION 2.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ============================================================================
# Кроссплатформенные опции сборки (Desktop / Android / iOS / WebAssembly)
# ============================================================================
set(JOKER_IS_CROSSCOMPILING OFF)
if(EMSCRIPTEN OR ANDROID OR IOS OR CMAKE_CROSSCOMPILING)
    set(JOKER_IS_CROSSCOMPILING ON)
endif()

set(JOKER_DEFAULT_SHARED ON)
set(JOKER_DEFAULT_OPENMP ON)
set(JOKER_DEFAULT_NATIVE ON)
set(JOKER_DEFAULT_CLI ON)

if(EMSCRIPTEN OR IOS)
    set(JOKER_DEFAULT_SHARED OFF)
    set(JOKER_DEFAULT_OPENMP OFF)
    set(JOKER_DEFAULT_CLI OFF)
endif()
if(ANDROID)
    set(JOKER_DEFAULT_CLI OFF)
endif()
if(JOKER_IS_CROSSCOMPILING)
    set(JOKER_DEFAULT_NATIVE OFF)
endif()

option(JOKER_ENABLE_OPENMP "Enable OpenMP parallelization" ${JOKER_DEFAULT_OPENMP})
option(JOKER_NATIVE_ARCH   "Enable -march=native / AVX2 optimizations" ${JOKER_DEFAULT_NATIVE})
option(JOKER_BUILD_SHARED  "Build shared library (OFF = static library)" ${JOKER_DEFAULT_SHARED})
option(JOKER_BUILD_CLI     "Build CLI executable" ${JOKER_DEFAULT_CLI})

if(JOKER_IS_CROSSCOMPILING AND JOKER_NATIVE_ARCH)
    message(STATUS "Cross-compilation detected: forcing JOKER_NATIVE_ARCH=OFF")
    set(JOKER_NATIVE_ARCH OFF)
endif()

set(JOKER_CORE_SOURCES
        src/core/Card.cpp
        src/core/Deck.cpp
        src/core/GameState.cpp
        src/engine/HeuristicBot.cpp
        src/engine/JokerEngine.cpp
        src/engine/JokerVectorEnv.cpp
        src/joker_c_api.cpp
)

if(JOKER_BUILD_SHARED)
    add_library(joker_core SHARED ${JOKER_CORE_SOURCES})
    target_compile_definitions(joker_core PRIVATE JOKER_CORE_EXPORTS)
else()
    add_library(joker_core STATIC ${JOKER_CORE_SOURCES})
    target_compile_definitions(joker_core PUBLIC JOKER_CORE_STATIC)
endif()

target_include_directories(joker_core PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

if(JOKER_ENABLE_OPENMP AND NOT EMSCRIPTEN)
    find_package(OpenMP QUIET)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: enabling multi-threaded JokerVectorEnv")
        target_link_libraries(joker_core PUBLIC OpenMP::OpenMP_CXX)
    else()
        message(STATUS "OpenMP not found: building single-threaded JokerVectorEnv")
    endif()
else()
    message(STATUS "OpenMP disabled for this target")
endif()

if(MSVC)
    target_compile_options(joker_core PRIVATE
            /W4
            $<$<CONFIG:Release>:/O2 /Ob3 /DNDEBUG>
    )
    if(JOKER_NATIVE_ARCH)
        target_compile_options(joker_core PRIVATE $<$<CONFIG:Release>:/arch:AVX2>)
    endif()
elseif(EMSCRIPTEN)
    target_compile_options(joker_core PRIVATE
            -Wall -Wextra -Wpedantic
            $<$<CONFIG:Release>:-O3 -DNDEBUG>
    )
else()
    target_compile_options(joker_core PRIVATE
            -Wall -Wextra -Wpedantic
            $<$<CONFIG:Release>:-O3 -funroll-loops -fno-math-errno -DNDEBUG>
    )
    if(JOKER_NATIVE_ARCH)
        target_compile_options(joker_core PRIVATE
                $<$<CONFIG:Release>:-march=native -mtune=native>
        )
    endif()
endif()

if(NOT EMSCRIPTEN)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT IPO_SUPPORTED OUTPUT IPO_ERROR)
    if(IPO_SUPPORTED)
        set_target_properties(joker_core PROPERTIES INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE)
    endif()
endif()

if(JOKER_BUILD_CLI AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/main_cli.cpp")
    add_executable(joker_cli src/main_cli.cpp)
    target_link_libraries(joker_cli PRIVATE joker_core)
    if(MSVC)
        target_compile_options(joker_cli PRIVATE /W4)
    else()
        target_compile_options(joker_cli PRIVATE -Wall -Wextra -Wpedantic)
    endif()
endif()