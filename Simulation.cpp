#include "Simulation.h"
#include "common.cuh"

Simulation::Simulation()
{
    GameConfigData::setConfigDataFromFile("config.txt");
    initDeviceParams();
    this->objLoader = ObjLoader();
    this->voxelEngine = VoxelEngine();
    this->voxelPipeline = VoxelPipeline();
    std::vector<VoxelData> voxelObjects = {
        Prepare_object("models/sphere.obj",35.0f, {0.0f,20.0f,25.0f}),
    };
    VoxelData combinedVoxelData = MergeVoxelData(voxelObjects);
    std::vector<float> combinedResult = voxelPipeline.process(combinedVoxelData);
    size_t bufferSize = combinedResult.size();
    float* h_buffer = new float[bufferSize]();
    std::copy(combinedResult.begin(), combinedResult.end(), h_buffer);

    initDeviceParams();
    engine.init(bufferSize / 26, h_buffer);
    renderer.init(engine.getNumber());

    std::vector<std::vector<Triangle>> triangleObjects =
    {
        Prepare_triangles("models/blender/u.obj",110.0f,{0.0f,-10.0f,0.0f})
    };

    std::vector<Triangle> allTriangles = MergeTriangles(triangleObjects);
    renderer.setTriangles(allTriangles);
    engine.initSDF(allTriangles);
}

Simulation::~Simulation() {}

/* --- HAS TO BE DONE IN THIS ORDER - OTHERWISE WILL NOT WORK! ---- */
void Simulation::run() {
     while (!renderer.isWindowClosed()) {
         if (!renderer.isPaused())
         {
             for (int i=0;i<GameConfigData::getInt("SUBSTEPS");i++)
             {
                 this->engine.step();
             }
         }
         this->renderer.draw(engine.getNumber(),engine.getPositions());
         //getchar();
    }
}

void Simulation::initDeviceParams()
{
    SimConfig h_cfg;
    h_cfg.SIZE_X = GameConfigData::getInt("SIZE_X");
    h_cfg.SIZE_Y = GameConfigData::getInt("SIZE_Y");
    h_cfg.SIZE_Z = GameConfigData::getInt("SIZE_Z");
    h_cfg.PADDING = GameConfigData::getInt("PADDING");
    h_cfg.GRAVITY = GameConfigData::getFloat("GRAVITY");
    h_cfg.DT = GameConfigData::getFloat("DT");
    h_cfg.GAMMA = GameConfigData::getFloat("GAMMA");
    h_cfg.COMPRESSION = GameConfigData::getFloat("COMPRESSION");
    h_cfg.RESOLUTION = GameConfigData::getFloat("RESOLUTION");
    h_cfg.SUBSTEPS = GameConfigData::getInt("SUBSTEPS");
    h_cfg.SDF_RESOLUTION = GameConfigData::getInt("SDF_RESOLUTION");
    h_cfg.GRID_SIZE = h_cfg.SIZE_X*h_cfg.SIZE_Y*h_cfg.SIZE_Z*CELL_SIZE;
    h_cfg.GRID_NUMBER = h_cfg.SIZE_X*h_cfg.SIZE_Y*h_cfg.SIZE_Z;
    h_cfg.GRID_BLOCKS = (h_cfg.GRID_NUMBER + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    GameConfigData::setNewInt("GRID_SIZE",std::to_string(h_cfg.GRID_SIZE));
    GameConfigData::setNewInt("GRID_NUMBER",std::to_string(h_cfg.GRID_NUMBER));
    GameConfigData::setNewInt("GRID_BLOCKS",std::to_string(h_cfg.GRID_BLOCKS));
    uploadConfigToGPU(h_cfg);
}


VoxelData Simulation::Prepare_object(const std::string& objPath, float scale, float3 displacement) {
    ObjData objData = objLoader.loadObj(objPath);
    if (!objData.success)
        throw std::runtime_error("Failed to load obj data");

    VoxelData voxelData = voxelEngine.voxelize(objData, 100);
    if (voxelData.count < 1)
        throw std::runtime_error("Failed to load voxel data");

    voxelEngine.normalize(voxelData, GameConfigData::getInt("SIZE_X"), scale, displacement);

    return voxelData;
}

std::vector<Triangle> Simulation::Prepare_triangles(const std::string& objPath, float scale, float3 displacement) {
    try {
        ObjData coliderData = objLoader.loadObj(objPath);
        if (!coliderData.success)
        {
            std::cerr << "Failed to load colider data from: " << objPath << std::endl;
            return std::vector<Triangle>();
        }

        std::vector<Triangle> coliderTriangles = voxelEngine.extractTriangles(coliderData);
        std::cout << "Extracted " << coliderTriangles.size() << " triangles from " << objPath << std::endl;

        if (coliderTriangles.empty())
        {
            std::cerr << "Warning: No triangles extracted from " << objPath << std::endl;
            return std::vector<Triangle>();
        }

        voxelEngine.normalize(coliderTriangles, GameConfigData::getInt("SIZE_X"), scale, displacement);

        return coliderTriangles;
    }
    catch (const std::exception& e) {
        std::cerr << "Error podczas Prepare_Triangles: " << e.what() << std::endl;
        return std::vector<Triangle>();
    }
}

VoxelData Simulation::MergeVoxelData(const std::vector<VoxelData>& voxelDataArray) {
    if (voxelDataArray.empty()) {
        return VoxelData();
    }

    size_t totalCount = 0;
    for (const auto& voxelData : voxelDataArray) {
        totalCount += voxelData.count;
    }

    if (totalCount == 0) {
        std::cerr << "Warning: MergeVoxelData - total count is 0" << std::endl;
        return VoxelData();
    }


    VoxelData combinedVoxelData;
    combinedVoxelData.count = totalCount;
    combinedVoxelData.resolution = voxelDataArray[0].resolution;
    combinedVoxelData.pos[0] = new float[totalCount];
    combinedVoxelData.pos[1] = new float[totalCount];
    combinedVoxelData.pos[2] = new float[totalCount];

    std::cout << "MergeVoxelData: Merging " << voxelDataArray.size() << " VoxelData objects, total voxels: " << totalCount << std::endl;


    size_t offset = 0;
    for (size_t i = 0; i < voxelDataArray.size(); i++) {
        const auto& voxelData = voxelDataArray[i];

        if (voxelData.count == 0) {
            std::cout << "  VoxelData[" << i << "] is empty, skipping" << std::endl;
            continue;
        }


        if (!voxelData.pos[0] || !voxelData.pos[1] || !voxelData.pos[2]) {
            std::cerr << "Error: VoxelData[" << i << "] has null pointers! count=" << voxelData.count << std::endl;
            std::cerr << "  pos[0]=" << voxelData.pos[0] << " pos[1]=" << voxelData.pos[1] << " pos[2]=" << voxelData.pos[2] << std::endl;
            continue;
        }

        std::cout << "  Copying VoxelData[" << i << "] with " << voxelData.count << " voxels at offset " << offset << std::endl;

        std::copy(voxelData.pos[0], voxelData.pos[0] + voxelData.count, combinedVoxelData.pos[0] + offset);
        std::copy(voxelData.pos[1], voxelData.pos[1] + voxelData.count, combinedVoxelData.pos[1] + offset);
        std::copy(voxelData.pos[2], voxelData.pos[2] + voxelData.count, combinedVoxelData.pos[2] + offset);
        offset += voxelData.count;
    }

    std::cout << "MergeVoxelData: Successfully merged, final offset: " << offset << std::endl;

    return combinedVoxelData;
}

std::vector<Triangle> Simulation::MergeTriangles(const std::vector<std::vector<Triangle>>& triangleArrays) {
    if (triangleArrays.empty()) {
        return std::vector<Triangle>();
    }


    size_t totalCount = 0;
    for (const auto& triangles : triangleArrays) {
        totalCount += triangles.size();
    }


    std::vector<Triangle> combinedTriangles;
    combinedTriangles.reserve(totalCount);


    for (const auto& triangles : triangleArrays) {
        combinedTriangles.insert(combinedTriangles.end(), triangles.begin(), triangles.end());
    }

    return combinedTriangles;
}

