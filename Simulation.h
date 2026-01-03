#pragma once

#include "VoxelEngine.h"
#include "VoxelPipeline.h"
#include "ObjLoader.h"
#include "engine.h"
#include "renderer.h"
#include <vector>
#include <string>

class Simulation {
    public:
    Simulation();
    ~Simulation();
    void run();
    std::vector<VoxelData> Prepare_object(const std::string& objPath, float scale = 1.0f, float3 displacement = {0.0f, 0.0f, 0.0f});
    std::vector<Triangle> Prepare_triangles(const std::string& objPath, float scale = 1.0f, float3 displacement = {0.0f, 0.0f, 0.0f});
    private:
    ObjLoader objLoader;
    VoxelEngine voxelEngine;
    VoxelPipeline voxelPipeline;
    Engine engine;
    Renderer renderer;
    Engine createEngine();
};
