#include "Simulation.h"
#include "common.cuh"

Simulation::Simulation() : engine(createEngine()), renderer(engine.getNumber()) // Initialize engine with 1 million particles
{
}

Simulation::~Simulation() {}

/* --- HAS TO BE DONE IN THIS OREDER - OTHERWISE WILL NOT WORK! ---- */
void Simulation::run() {
    while (!renderer.isWindowClosed()) {
        this->engine.step();
        this->renderer.draw(engine.getNumber(),engine.getPositions());
    }
}

Engine Simulation::createEngine() {
    this->objLoader = ObjLoader();
    this->voxelEngine = VoxelEngine();
    this->voxelPipeline = VoxelPipeline();
    ObjData objData = objLoader.loadObj("test.obj");
    if (!objData.success)
        throw std::runtime_error("Failed to load obj data");

    VoxelData voxelData = voxelEngine.voxelize(objData, 0.1f);
    if (voxelData.count<1)
        throw std::runtime_error("Failed to load voxel data");

    float3 displacement = {0.0f, 0.0f, 0.0f};
    voxelEngine.normalize(voxelData, SIZE_X, 1.0f, displacement);

    std::vector<float> result = voxelPipeline.process(voxelData);

    size_t bufferSize = result.size();
    float* h_buffer = new float[bufferSize];
    std::copy(result.begin(), result.end(), h_buffer);

    return Engine(bufferSize / 26, h_buffer);
}



