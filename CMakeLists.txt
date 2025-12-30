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
        kernels.cu)

# --- Windows Compatibility Fixes ---
if(MSVC)
    target_compile_definitions(water PRIVATE NOMINMAX)

    # Apply the standard preprocessor to both C++ and CUDA files
    target_compile_options(water PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
    )
endif()
# ----------------------------------

find_package(tinyobjloader CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)

target_link_libraries(water
    PRIVATE
    CUDA::cuda_driver
    CUDA::cudart
    tinyobjloader::tinyobjloader
    nlohmann_json::nlohmann_json
)

# Copy test.obj to build directory
add_custom_command(TARGET water POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${CMAKE_SOURCE_DIR}/test.obj"
        "$<TARGET_FILE_DIR:water>/test.obj"
    COMMENT "Copying test.obj to build directory"
)

