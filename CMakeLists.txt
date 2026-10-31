cmake_minimum_required(VERSION 3.20)
project(llm_engine LANGUAGES CXX)   # later: LANGUAGES CXX CUDA

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Lets the compiler use AVX2/AVX-512/FMA in the micro-kernel. Turn off when
# building binaries that must run on other machines.
option(ENGINE_NATIVE "Tune for the host CPU (-march=native)" ON)

# Treat warnings as errors. Off by default so a newer compiler's new warnings
# don't break someone else's build; turn on in CI and while developing.
option(ENGINE_WERROR "Treat compiler warnings as errors" OFF)

find_package(Threads REQUIRED)

# Warning flags shared by every target, so tests and benchmarks are held to
# the same bar as the library.
add_library(engine_warnings INTERFACE)
if(MSVC)
  target_compile_options(engine_warnings INTERFACE /W4 $<$<BOOL:${ENGINE_WERROR}>:/WX>)
else()
  target_compile_options(engine_warnings INTERFACE
    -Wall -Wextra -Wpedantic -Wshadow -Wconversion
    $<$<BOOL:${ENGINE_WERROR}>:-Werror>)
endif()

add_library(engine_core src/matmul.cpp)
target_include_directories(engine_core PUBLIC src)
target_link_libraries(engine_core PUBLIC Threads::Threads PRIVATE engine_warnings)
if(ENGINE_NATIVE AND NOT MSVC)
  target_compile_options(engine_core PRIVATE -march=native)
endif()

add_executable(bench_matmul bench/bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE engine_core engine_warnings)

enable_testing()
add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE engine_core engine_warnings)
add_test(NAME matmul COMMAND test_matmul)
