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

find_package(Threads REQUIRED)

add_library(engine_core src/matmul.cpp)
target_include_directories(engine_core PUBLIC src)
target_link_libraries(engine_core PUBLIC Threads::Threads)
if(MSVC)
  target_compile_options(engine_core PRIVATE /W4)
else()
  target_compile_options(engine_core PRIVATE -Wall -Wextra -Wpedantic)
  if(ENGINE_NATIVE)
    target_compile_options(engine_core PRIVATE -march=native)
  endif()
endif()

add_executable(bench_matmul bench/bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE engine_core)

enable_testing()
add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE engine_core)
add_test(NAME matmul COMMAND test_matmul)
