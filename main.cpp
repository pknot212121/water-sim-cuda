#include "engine.h"
#include "common.cuh"
#include "ObjLoader.h"
#include "VoxelEngine.h"

#include <iostream>

#include "VoxelPipeline.h"
#include "Simulation.h"


int main()
{
    Simulation simulation;
    simulation.run();
    return 0;
}
