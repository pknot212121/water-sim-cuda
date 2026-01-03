#include "Simulation.h"
#include "common.cuh"

Simulation::Simulation() : engine(createEngine()), renderer(engine.getNumber()) // Initialize engine with 1 million particles
{
}

Simulation::~Simulation() {}

/* --- HAS TO BE DONE IN THIS ORDER - OTHERWISE WILL NOT WORK! ---- */
void Simulation::run() {
    while (!renderer.isWindowClosed()) {
        this->engine.step();
        this->renderer.draw(engine.getNumber(),engine.getPositions());
        getchar();
    }
}

Engine Simulation::createEngine() {
    this->objLoader = ObjLoader();
    this->voxelEngine = VoxelEngine();
    this->voxelPipeline = VoxelPipeline();

    // Wczytanie kilka obiektow voxelowych
    std::vector<VoxelData> voxelObjects;
    voxelObjects.push_back(Prepare_object("test.obj")[0]);
    voxelObjects.push_back(Prepare_object("test.obj")[0]);
    voxelObjects.push_back(Prepare_object("test.obj")[0]);
    voxelObjects.push_back(Prepare_object("test.obj")[0]);

    std::vector<float> combinedResult;
    for (const auto& voxelData : voxelObjects) {
        std::vector<float> result = voxelPipeline.process(voxelData, RESOLUTION);
        combinedResult.insert(combinedResult.end(), result.begin(), result.end());
    }

    size_t bufferSize = combinedResult.size();
    float* h_buffer = new float[bufferSize]();
    std::copy(combinedResult.begin(), combinedResult.end(), h_buffer);

    // Wczytanie kilka obiektow kolizyjnych
    std::vector<Triangle> allTriangles;
    std::vector<Triangle> triangles1 = Prepare_triangles("pipes.obj");
    std::vector<Triangle> triangles2 = Prepare_triangles("pipes.obj");
    std::vector<Triangle> triangles3 = Prepare_triangles("pipes.obj");
    std::vector<Triangle> triangles4 = Prepare_triangles("pipes.obj");

    allTriangles.insert(allTriangles.end(), triangles1.begin(), triangles1.end());
    allTriangles.insert(allTriangles.end(), triangles2.begin(), triangles2.end());
    allTriangles.insert(allTriangles.end(), triangles3.begin(), triangles3.end());
    allTriangles.insert(allTriangles.end(), triangles4.begin(), triangles4.end());

    return Engine(bufferSize / 26, h_buffer);
}

std::vector<VoxelData> Simulation::Prepare_object(const std::string& objPath, float scale, float3 displacement) {
    ObjData objData = objLoader.loadObj(objPath);
    if (!objData.success)
        throw std::runtime_error("Failed to load obj data");

    VoxelData voxelData = voxelEngine.voxelize(objData, RESOLUTION);
    if (voxelData.count < 1)
        throw std::runtime_error("Failed to load voxel data");

    voxelEngine.normalize(voxelData, SIZE_X, scale, displacement);

    return {voxelData};
}

std::vector<Triangle> Simulation::Prepare_triangles(const std::string& objPath, float scale, float3 displacement) {
    ObjData coliderData = objLoader.loadObj(objPath);
    if (!coliderData.success)
        throw std::runtime_error("Failed to load colider data");

    std::vector<Triangle> coliderTriangles = voxelEngine.extractTriangles(coliderData);
    voxelEngine.normalize(coliderTriangles, SIZE_X, scale, displacement);

    return coliderTriangles;
}
