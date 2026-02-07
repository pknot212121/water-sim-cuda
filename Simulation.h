#pragma once

#include "VoxelEngine.h"
#include "VoxelPipeline.h"
#include "ObjLoader.h"
#include "engine.h"
#include "renderer.h"
#include <vector>
#include <string>
#include "game_configdata.h"

class Simulation {
    public:
    Simulation();
    ~Simulation();
    void run();
    void initDeviceParams();
    VoxelData Prepare_object(const std::string& objPath, float scale = 1.0f, float3 displacement = {0.0f, 0.0f, 0.0f});
    std::vector<Triangle> Prepare_triangles(const std::string& objPath, float scale = 1.0f, float3 displacement = {0.0f, 0.0f, 0.0f});
    VoxelData MergeVoxelData(const std::vector<VoxelData>& voxelDataArray);
    std::vector<Triangle> MergeTriangles(const std::vector<std::vector<Triangle>>& triangleArrays);

    private:
    ObjLoader objLoader;
    VoxelEngine voxelEngine;
    VoxelPipeline voxelPipeline;
    Engine engine;
    Renderer renderer;
};
