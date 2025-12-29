#include "engine.h"




__global__ void testKernel(Particles p)
{
    const int threadIndex = threadIdx.x;
    printf("Hello Particle: %f\n",p.pos[0][threadIndex]);
}

__global__ void occupancyCheckInit(Particles p, size_t cellsPerPage,bool* occupancy)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int posX = (int)(p.pos[0][threadIndex]-0.5f);
    const int posY = (int)(p.pos[1][threadIndex]-0.5f);
    const int posZ = (int)(p.pos[2][threadIndex]-0.5f);

    for (int i=0;i<3;i++)
    {
        for (int j=0;j<3;j++)
        {
            for (int k=0;k<3;k++)
            {
                size_t cellIdx = (posZ+k) * SIZE_X * SIZE_Y + (posY+j) * SIZE_X + (posX+i);
                size_t pageIdx = cellIdx / cellsPerPage;
                occupancy[pageIdx] = true;
            }
        }
    }

}

__global__ void p2GTransfer(Particles p,Grid g,int number)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex>=number) return;
    const int posX = (int)(p.pos[0][threadIndex]-0.5f);
    const int posY = (int)(p.pos[1][threadIndex]-0.5f);
    const int posZ = (int)(p.pos[2][threadIndex]-0.5f);

    float c00 = p.c[0][threadIndex]; float c01 = p.c[1][threadIndex]; float c02 = p.c[2][threadIndex];
    float c10 = p.c[3][threadIndex]; float c11 = p.c[4][threadIndex]; float c12 = p.c[5][threadIndex];
    float c20 = p.c[6][threadIndex]; float c21 = p.c[7][threadIndex]; float c22 = p.c[8][threadIndex];

    for (int i=0;i<3;i++)
    {
        for (int j=0;j<3;j++)
        {
            for (int k=0;k<3;k++)
            {
                if (g.isInBounds(posX+i,posY+j,posZ+k))
                {
                    size_t cellIdx = g.getGridIdx(posX+i,posY+j,posZ+k);
                    float3 d = {p.pos[0][threadIndex] - (posX+i),p.pos[1][threadIndex] - (posY+j),p.pos[2][threadIndex] - (posZ+k)};
                    float weight = g.spline(d.x)*g.spline(d.y)*g.spline(d.z);
                    atomicAdd(&g.mass[cellIdx],p.m[threadIndex]*weight);
                    float3 Cxd = p.multiplyCxd(c00,c01,c02,c10,c11,c12,c20,c21,c22,d);
                    atomicAdd(&g.momentum[0][cellIdx],weight*p.m[threadIndex]*p.vel[0][threadIndex]+Cxd.x);
                    atomicAdd(&g.momentum[1][cellIdx],weight*p.m[threadIndex]*p.vel[1][threadIndex]+Cxd.y);
                    atomicAdd(&g.momentum[2][cellIdx],weight*p.m[threadIndex]*p.vel[2][threadIndex]+Cxd.z);
                }
            }
        }
    }
}

__global__ void gridTest(Grid g,int targetX, int targetY, int targetZ)
{
    size_t idx = g.getGridIdx(targetX,targetY,targetZ);
    g.mass[idx] = 2137.0f;
}

void handleCUError(CUresult result)
{
    if (result != CUDA_SUCCESS)
    {
        const char* errStr;
        cuGetErrorString(result, &errStr);
        std::cerr << "DEVICE API ERROR: " << errStr << std::endl;
        exit(1);
    }
}

void handleCUDAError(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA ERROR: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
}

size_t roundUp(size_t value,size_t rounder)
{
    if (value % rounder ==0) return value;
    else return (value/rounder) * (rounder+1);
}

CUmemAllocationProp Engine::getProp()
{
    CUmemAllocationProp prop = {};
    prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type=CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id=device;
    prop.requestedHandleTypes=CU_MEM_HANDLE_TYPE_NONE;
    return prop;
}

CUmemAccessDesc Engine::getDesc()
{
    CUmemAccessDesc desc = {};
    desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc.location.id = device;
    desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    return desc;
}


Engine::Engine(int n)
{
    number = n;
    gen = std::mt19937(std::random_device{}());
    blocksPerGrid = (number+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    initCuda();
    initParticles();
    initGrid();
    step();
    getchar();
}

Engine::~Engine()
{
    delete[] h_buffer;
    cudaFree(d_buffer);
    cuCtxPopCurrent(&context);
}

void Engine::step()
{
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(blocksPerGrid);
    p2GTransfer<<<gridDim,blockDim>>>(getParticles(),getGrid(),number);
    handleCUDAError(cudaDeviceSynchronize());
}

Grid Engine::getGrid()
{
    Grid g;
    size_t attributeSize = roundUp(SIZE_X * SIZE_Y * SIZE_Z * sizeof(float),granularity);
    g.mass = (float*)virtPtr;
    for (int i=0;i<3;i++) g.momentum[i] = (float*)(virtPtr + i*attributeSize);
    return g;
}

void Engine::initParticles()
{
    std::uniform_real_distribution<float> distX(0, SIZE_X);
    std::uniform_real_distribution<float> distY(0, SIZE_Y);
    std::uniform_real_distribution<float> distZ(0, 10);
    h_buffer = new float[number * PARTICLE_SIZE];
    for (int i = 0; i < number; i++)
    {
        h_buffer[i] = distX(gen);
        h_buffer[i + number] = distY(gen);
        h_buffer[i + number * 2] = distZ(gen);
        h_buffer[i + number * (PARTICLE_SIZE - 2)] = 10;
        h_buffer[i + number * (PARTICLE_SIZE - 1)] = 10;
    }
    handleCUDAError(cudaMalloc((void**)&d_buffer, number * PARTICLE_SIZE * sizeof(float)));
    handleCUDAError(cudaMemcpy(d_buffer, h_buffer, number * PARTICLE_SIZE * sizeof(float), cudaMemcpyHostToDevice));

}

void Engine::initCuda()
{
    cuInit(0);
    cuDeviceGet(&device,0);
    cuDevicePrimaryCtxRetain(&context,device);
    cuCtxPushCurrent(context);
    CUmemAllocationProp prop = getProp();
    cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    cellsPerPage = granularity / CELL_SIZE;
    gridAlignedSize = roundUp(GRID_SIZE,granularity);
    pageCount = gridAlignedSize/granularity;
    handleCUError(cuMemAddressReserve(&virtPtr,gridAlignedSize,0,0,0));
}



void Engine::initGrid()
{
    bool* occupancy_d;
    handleCUDAError(cudaMalloc((void**)&occupancy_d,pageCount));
    handleCUDAError(cudaMemset(occupancy_d,0,pageCount));
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(blocksPerGrid);
    occupancyCheckInit<<<gridDim,blockDim>>>(getParticles(),cellsPerPage,occupancy_d);
    handleCUDAError(cudaDeviceSynchronize());
    std::vector<unsigned char> occupancy_h(pageCount);
    handleCUDAError(cudaMemcpy(occupancy_h.data(),occupancy_d,occupancy_h.size()*sizeof(occupancy_h[0]),cudaMemcpyDeviceToHost));
    CUmemAllocationProp prop = getProp();
    int occupiedCount = 0;
    for (int i=0;i<pageCount;i++)
    {
        size_t attributeSize = roundUp(SIZE_X * SIZE_Y * SIZE_Z * sizeof(float),granularity);
        if (occupancy_h[i] !=0)
        {
            occupiedCount++;
            for (int j=0;j<4;j++)
            {
                CUmemGenericAllocationHandle h;
                handleCUError(cuMemCreate(&h,granularity,&prop,0));
                handleCUError(cuMemMap(virtPtr+(i*granularity)+(j*attributeSize),granularity,0,h,0));
                CUmemAccessDesc desc = getDesc();
                handleCUError(cuMemSetAccess(virtPtr+(i*granularity)+(j*attributeSize),granularity,&desc,1));
                handles.push_back(h);
            }

        }
    }
    std::cout << "OCCUPIED: " << occupiedCount << std::endl;
    std::cout << "PAGE COUNT: " << pageCount << std::endl;
    handleCUDAError(cudaFree(occupancy_d));

}


