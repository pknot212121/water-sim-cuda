#include "Simulation.h"
#include "common.cuh"

Simulation::Simulation() : engine(createEngine()), renderer(engine.getNumber()) // Initialize engine with 1 million particles
{
    std::vector<std::vector<Triangle>> triangleObjects = {
        // Prepare_triangles("models/Glass_Cup.obj",76.8f,{0.0f,0.0f,0.0f}),
        //Prepare_triangles("models/box.obj",100.0f,{0.0f,0.0f,0.0f})
        //Prepare_triangles("models/connected_containers.obj",128.0f,{0.0f,0.0f,0.0f})
        //Prepare_triangles("models/blender/shape_1_1.obj",110.0f,{0.0f,-10.0f,0.0f})
        //Prepare_triangles("models/blender/shape_2.obj",110.0f,{0.0f,-10.0f,0.0f})
        //Prepare_triangles("models/blender/pipe_system.obj",110.0f,{0.0f,-10.0f,10.0f})
        //Prepare_triangles("models/blender/box_open.obj",110.0f,{0.0f,-10.0f,0.0f})
        Prepare_triangles("models/blender/u.obj",110.0f,{0.0f,-10.0f,0.0f})
        //Prepare_triangles("models/blender/bottle.obj",110.0f,{0.0f,-10.0f,0.0f})

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
             for (int i=0;i<SUBSTEPS;i++)
             {
                 this->engine.step();
             }
         }
         this->renderer.draw(engine.getNumber(),engine.getPositions());
         //getchar();
    }
}

Engine Simulation::createEngine() {
    this->objLoader = ObjLoader();
    this->voxelEngine = VoxelEngine();
    this->voxelPipeline = VoxelPipeline();


    std::vector<VoxelData> voxelObjects = {
        //Prepare_object("models/sphere.obj",9.0f, {0.0f,9.0f,0.0f}),  // displacement can be any value - VoxelEngine will clamp to [0,SIZE_X]
        Prepare_object("models/sphere.obj",35.0f, {0.0f,0.0f,30.0f}),  // displacement can be any value - VoxelEngine will clamp to [0,SIZE_X]
        //Prepare_object("models/sphere.obj",15.0f, {0.0f,45.0f,25.0f}),  // displacement can be any value - VoxelEngine will clamp to [0,SIZE_X]
        //Prepare_object("models/sphere.obj",40.0f, {0.0f,40.0f,0.0f}),  // displacement can be any value - VoxelEngine will clamp to [0,SIZE_X]
        // Prepare_object("models/sphere.obj",48.0f, {50.0f,50.0f,0.0f}),
        //Prepare_object("models/u.obj",100.0f,{0.0f,.0f,0.0f})
    };

    VoxelData combinedVoxelData = MergeVoxelData(voxelObjects);

    // Wywołanie process tylko raz na scalonych danych
    std::vector<float> combinedResult = voxelPipeline.process(combinedVoxelData);

    size_t bufferSize = combinedResult.size();
    float* h_buffer = new float[bufferSize]();
    std::copy(combinedResult.begin(), combinedResult.end(), h_buffer);
    return Engine(bufferSize / 26, h_buffer);
}

VoxelData Simulation::Prepare_object(const std::string& objPath, float scale, float3 displacement) {
    ObjData objData = objLoader.loadObj(objPath);
    if (!objData.success)
        throw std::runtime_error("Failed to load obj data");

    VoxelData voxelData = voxelEngine.voxelize(objData, 100);
    if (voxelData.count < 1)
        throw std::runtime_error("Failed to load voxel data");

    voxelEngine.normalize(voxelData, SIZE_X, scale, displacement);

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

        voxelEngine.normalize(coliderTriangles, SIZE_X, scale, displacement);

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

    // Oblicz całkowitą liczbę voxeli
    size_t totalCount = 0;
    for (const auto& voxelData : voxelDataArray) {
        totalCount += voxelData.count;
    }

    if (totalCount == 0) {
        std::cerr << "Warning: MergeVoxelData - total count is 0" << std::endl;
        return VoxelData();
    }

    // Utwórz nowy VoxelData
    VoxelData combinedVoxelData;
    combinedVoxelData.count = totalCount;
    combinedVoxelData.resolution = voxelDataArray[0].resolution;
    combinedVoxelData.pos[0] = new float[totalCount];
    combinedVoxelData.pos[1] = new float[totalCount];
    combinedVoxelData.pos[2] = new float[totalCount];

    std::cout << "MergeVoxelData: Merging " << voxelDataArray.size() << " VoxelData objects, total voxels: " << totalCount << std::endl;

    // Skopiuj dane z każdego VoxelData
    size_t offset = 0;
    for (size_t i = 0; i < voxelDataArray.size(); i++) {
        const auto& voxelData = voxelDataArray[i];

        if (voxelData.count == 0) {
            std::cout << "  VoxelData[" << i << "] is empty, skipping" << std::endl;
            continue;
        }

        // Walidacja wskaźników
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

    // Oblicz całkowitą liczbę trójkątów
    size_t totalCount = 0;
    for (const auto& triangles : triangleArrays) {
        totalCount += triangles.size();
    }

    // Utwórz nowy wektor i zarezerwuj pamięć
    std::vector<Triangle> combinedTriangles;
    combinedTriangles.reserve(totalCount);

    // Skopiuj trójkąty z każdego wektora
    for (const auto& triangles : triangleArrays) {
        combinedTriangles.insert(combinedTriangles.end(), triangles.begin(), triangles.end());
    }

    return combinedTriangles;
}

