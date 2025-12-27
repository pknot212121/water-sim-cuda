#include "engine.h"

#include <iostream>

void HandleError(CUresult result)
{
    if (result != CUDA_SUCCESS)
    {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        std::cerr << "ERROR: " << errStr << std::endl;
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
    cudaError_t err = cudaMalloc((void**)&d_buffer, number * PARTICLE_SIZE * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Błąd alokacji: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    err = cudaMemcpy(d_buffer, h_buffer, number * PARTICLE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        std::cerr << "Błąd kopiowania: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
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

    HandleError(cuMemAddressReserve(&virtPtr,SIZE_X*SIZE_Y*SIZE_Z*CELL_SIZE,0,0,0));

}

void Engine::InitGrid()
{

}
