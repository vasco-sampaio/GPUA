# -- Can be removed if not on OpenStack --
find_program(GCC_PATH gcc)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --allow-unsupported-compiler")
add_compile_options("-Wno-error" "-Wno-pedantic" "-Wno-old-style-cast")
set(CMAKE_CUDA_HOST_COMPILER ${GCC_PATH})
# -- /Can be removed if not on OpenStack --
project(template)
cmake_minimum_required(VERSION 3.18)

include(FindCUDAToolkit)
include(FetchContent)

enable_language(CUDA)

FetchContent_Declare(googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.12.x)

FetchContent_Declare(googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG main) # need master for benchmark::benchmark

FetchContent_MakeAvailable(
        googletest
        googlebenchmark)
# -- Can be removed if not on OpenStack --
target_compile_options(benchmark PRIVATE "-Wno-pedantic")
target_compile_options(benchmark PRIVATE "-Wno-error")
target_compile_options(benchmark PRIVATE "-Wno-old-style-cast")
target_compile_options(gtest PRIVATE "-Wno-error" "-Wno-pedantic" "-Wno-old-style-cast")
target_compile_options(gtest_main PRIVATE "-Wno-error" "-Wno-pedantic" "-Wno-old-style-cast")
# -- /Can be removed if not on OpenStack --
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED TRUE)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -extended-lambda --generate-line-info --expt-relaxed-constexpr")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_FLAGS "--ptxas-options=-v")

add_library(to_bench
  src/to_bench.cu
  src/cuda_tools/host_shared_ptr.cu)

include_directories(src)

add_executable(bench bench/main.cc)
set_property(TARGET bench PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")

target_link_libraries(bench to_bench benchmark::benchmark)

# BUILD_TYPE
if (NOT CMAKE_BUILD_TYPE)
    string(REGEX REPLACE ".*/build_" "" BUILD_NAME ${CMAKE_BINARY_DIR})
    # Default BUILD_TYPE is Release
    set(CMAKE_BUILD_TYPE Release)
    # If folder is named "build_debug" BUILD_TYPE is set to: Debug
    if(${BUILD_NAME} STREQUAL "Debug" OR ${BUILD_NAME} STREQUAL "debug")
        set(CMAKE_BUILD_TYPE Debug)
    endif()
    message(STATUS "Build type : ${CMAKE_BUILD_TYPE}")
endif()
