cmake_minimum_required(VERSION 3.20)
project(llm_engine LANGUAGES CXX)   # later: LANGUAGES CXX CUDA

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

add_executable(engine src/main.cpp)