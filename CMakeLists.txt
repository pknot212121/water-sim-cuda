cmake_minimum_required(VERSION 4.1)
project(water LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
find_package(CUDAToolkit REQUIRED)
add_executable(water
        engine.h
        engine.cu
        common.cuh
        ObjLoader.h
        ObjLoader.cpp
        VoxelEngine.h
        VoxelEngine.cpp
        VoxelPipeline.h
        VoxelPipeline.cpp
        Simulation.h
        Simulation.cpp
        main.cpp
        kernels.cu
        renderer.cpp
        renderer.h
        renderer.h
        shader.cpp
        shader.h
        resource_manager.h
        resource_manager.cpp)

# --- Windows Compatibility Fixes ---
if(MSVC)
    target_compile_definitions(water PRIVATE NOMINMAX)

    target_compile_options(water PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()
# ----------------------------------

find_package(tinyobjloader CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(glfw3 CONFIG REQUIRED)
find_package(glad CONFIG REQUIRED)
find_package(glm CONFIG REQUIRED)


target_link_libraries(water
    PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    tinyobjloader::tinyobjloader
    nlohmann_json::nlohmann_json
    glfw
    glad::glad
    glm::glm
)

