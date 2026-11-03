cmake_minimum_required(VERSION 3.20)

# Default to Release. Set before project(): MSVC toolchains otherwise
# initialize the build type to Debug there, and the old after-the-fact check
# silently built /Od binaries.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type")
endif()

project(llm_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Lets the compiler use AVX2/AVX-512/FMA in the micro-kernel. Turn off when
# building binaries that must run on other machines.
option(ENGINE_NATIVE "Tune for the host CPU (-march=native)" ON)

# Treat warnings as errors. Off by default so a newer compiler's new warnings
# don't break someone else's build; turn on in CI and while developing.
option(ENGINE_WERROR "Treat compiler warnings as errors" OFF)

# GPU kernels (src/cuda_matmul.cu). Needs the CUDA toolkit; on Windows nvcc
# only works with MSVC, so configure this build with cl as the C++ compiler.
option(ENGINE_CUDA "Build the CUDA kernels, their tests and benchmark" OFF)

find_package(Threads REQUIRED)

# Warning flags shared by every target, so tests and benchmarks are held to
# the same bar as the library.
add_library(engine_warnings INTERFACE)
if(MSVC)
  target_compile_options(engine_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:/W4 $<$<BOOL:${ENGINE_WERROR}>:/WX>>)
else()
  target_compile_options(engine_warnings INTERFACE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -Wpedantic -Wshadow -Wconversion
                              $<$<BOOL:${ENGINE_WERROR}>:-Werror>>)
endif()

add_library(engine_core
  src/matmul.cpp src/model.cpp src/ops.cpp src/quant.cpp src/sampler.cpp
  src/thread_pool.cpp src/tokenizer.cpp)
target_include_directories(engine_core PUBLIC src)
target_link_libraries(engine_core PUBLIC Threads::Threads PRIVATE engine_warnings)
if(ENGINE_NATIVE)
  # MSVC has no -march=native; AVX2 is the closest flag every recent x86 has.
  target_compile_options(engine_core PRIVATE $<IF:$<BOOL:${MSVC}>,/arch:AVX2,-march=native>)
endif()

add_executable(bench_matmul bench/bench_matmul.cpp)
target_link_libraries(bench_matmul PRIVATE engine_core engine_warnings)

add_executable(generate tools/generate.cpp)
target_link_libraries(generate PRIVATE engine_core engine_warnings)

enable_testing()
add_executable(test_matmul tests/test_matmul.cpp)
target_link_libraries(test_matmul PRIVATE engine_core engine_warnings)
add_test(NAME matmul COMMAND test_matmul)

add_executable(test_ops tests/test_ops.cpp)
target_link_libraries(test_ops PRIVATE engine_core engine_warnings)
add_test(NAME ops COMMAND test_ops)

add_executable(test_thread_pool tests/test_thread_pool.cpp)
target_link_libraries(test_thread_pool PRIVATE engine_core engine_warnings)
add_test(NAME thread_pool COMMAND test_thread_pool)

add_executable(test_model tests/test_model.cpp)
target_link_libraries(test_model PRIVATE engine_core engine_warnings)
add_test(NAME model COMMAND test_model)

add_executable(test_quant tests/test_quant.cpp)
target_link_libraries(test_quant PRIVATE engine_core engine_warnings)
add_test(NAME quant COMMAND test_quant)

add_executable(test_tokenizer tests/test_tokenizer.cpp)
target_link_libraries(test_tokenizer PRIVATE engine_core engine_warnings)
add_test(NAME tokenizer COMMAND test_tokenizer)

if(ENGINE_CUDA)
  # Compile real code for the GPU in this machine unless told otherwise.
  # nvcc's default target is older, and running it on a newer GPU means
  # JIT-compiling PTX, which fails when the driver predates the toolkit.
  # Must be set before enable_language(), which fills in that default.
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
  endif()
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  set(CMAKE_CUDA_STANDARD 17)

  add_library(engine_cuda src/cuda_matmul.cu)
  target_include_directories(engine_cuda PUBLIC src)
  target_link_libraries(engine_cuda PUBLIC CUDA::cudart)
  # C4211 fires on the stub code nvcc generates for kernels in an anonymous
  # namespace, not on anything in the source.
  target_compile_options(engine_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=$<IF:$<BOOL:${MSVC}>,/W4$<COMMA>/wd4211,-Wall>
                               $<$<BOOL:${ENGINE_WERROR}>:-Werror=all-warnings>>)

  add_executable(test_cuda tests/test_cuda.cpp)
  target_link_libraries(test_cuda PRIVATE engine_cuda engine_core engine_warnings)
  add_test(NAME cuda COMMAND test_cuda)
  set_tests_properties(cuda PROPERTIES SKIP_RETURN_CODE 77)

  add_executable(bench_cuda bench/bench_cuda.cpp)
  target_link_libraries(bench_cuda PRIVATE engine_cuda CUDA::cublas engine_warnings)
endif()
