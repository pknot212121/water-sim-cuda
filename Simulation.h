#pragma once

#include "VoxelEngine.h"
#include "VoxelPipeline.h"
#include "ObjLoader.h"
#include "engine.h"

class Simulation {
    public:
    Simulation();
    ~Simulation();
    void run();
    private:
    ObjLoader objLoader;
    VoxelEngine voxelEngine;
    VoxelPipeline voxelPipeline;
    Engine engine;
    Engine createEngine();
};
