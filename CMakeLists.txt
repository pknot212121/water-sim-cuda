cmake_minimum_required(VERSION 4.1)
project(water LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
find_package(CUDAToolkit REQUIRED)
add_executable(water
        engine.h
        engine.cu
        common.cuh
        main.cpp)

target_link_libraries(water
    PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
)