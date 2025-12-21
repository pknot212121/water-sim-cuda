cmake_minimum_required(VERSION 4.1)
project(water LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)

add_executable(water
        engine.h
        engine.cpp
        kernels.cu
        common.cuh
        kernels.h)
