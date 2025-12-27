#include "engine.h"

#include <iostream>

void HandleCUError(CUresult result)
{
    if (result != CUDA_SUCCESS)
    {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        std::cerr << "DEVICE API ERROR: " << errStr << std::endl;
        exit(1);
    }
}

void HandleCUDAError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}


Engine::Engine(int n)
{
    number = n;
    gen = std::mt19937(std::random_device{}());
    InitCuda();
    InitParticles();
    Particles p = getParticles();
    summonTestKernel(p,number);
    cudaDeviceSynchronize();
    std::cout << "GRANULARITY: " << granularity << std::endl;
    InitGrid();
    getchar();
}

Engine::~Engine()
{
    delete[] h_buffer;
    cudaFree(d_buffer);
    cuCtxPopCurrent(&context);
}

void Engine::Step()
{

}

void Engine::InitParticles()
{
    std::uniform_real_distribution<float> distX(0, SIZE_X);
    std::uniform_real_distribution<float> distY(0, SIZE_Y);
    std::uniform_real_distribution<float> distZ(0, SIZE_Z);
    h_buffer = new float[number * PARTICLE_SIZE];
    for (int i = 0; i < number; i++)
    {
        h_buffer[i] = distX(gen);
        h_buffer[i + number] = distY(gen);
        h_buffer[i + number * 2] = distZ(gen);
        h_buffer[i + number * (PARTICLE_SIZE - 2)] = 10;
        h_buffer[i + number * (PARTICLE_SIZE - 1)] = 10;
    }
    HandleCUDAError(cudaMalloc((void**)&d_buffer, number * PARTICLE_SIZE * sizeof(float)));
    HandleCUDAError(cudaMemcpy(d_buffer, h_buffer, number * PARTICLE_SIZE * sizeof(float), cudaMemcpyHostToDevice));

}

void Engine::InitCuda()
{
    cuInit(0);
    cuDeviceGet(&device,0);
    cuDevicePrimaryCtxRetain(&context,device);
    cuCtxPushCurrent(context);
    CUmemAllocationProp prop = {};
    prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id=device;
    prop.requestedHandleTypes=CU_MEM_HANDLE_TYPE_NONE;
    cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    cellsPerPage = granularity / CELL_SIZE;

    HandleCUError(cuMemAddressReserve(&virtPtr,GRID_SIZE,0,0,0));

}

void Engine::InitGrid()
{
    bool* occupancy_d;
    HandleCUDAError(cudaMalloc((void**)&occupancy_d,GRID_SIZE/granularity));
    HandleCUDAError(cudaMemset(occupancy_d,0,GRID_SIZE/granularity));
    summonOccupancyCheckInit(getParticles(),number,cellsPerPage,occupancy_d);
    std::vector<unsigned char> occupancy_h(GRID_SIZE/granularity);
    HandleCUDAError(cudaMemcpy(occupancy_h.data(),occupancy_d,occupancy_h.size()*sizeof(occupancy_h[0]),cudaMemcpyDeviceToHost));
    CUmemAllocationProp prop = {};
    prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id=device;
    prop.requestedHandleTypes=CU_MEM_HANDLE_TYPE_NONE;
    for (int i=0;i<GRID_SIZE/granularity;i++)
    {
        std::cout << "IS OCCUPIED: "<< (int)occupancy_h[i] << std::endl;
        if (occupancy_h[i] !=0)
        {
            CUmemGenericAllocationHandle h;
            HandleCUError(cuMemCreate(&h,granularity,&prop,0));
            HandleCUError(cuMemMap(virtPtr+(i*granularity),granularity,0,h,0));
            CUmemAccessDesc dest = {};
            dest.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            dest.location.id = device;
            dest.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            HandleCUError(cuMemSetAccess(virtPtr+(i*granularity),granularity,&dest,1));
            handles.push_back(h);
        }
    }
    HandleCUDAError(cudaFree(occupancy_d));

}
