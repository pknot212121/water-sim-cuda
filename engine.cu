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

__global__ void p2GTransfer(Particles p,Grid g,int number,unsigned int* sortedIndices)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int particleIdx = __ldg(&sortedIndices[threadIndex]);
    const int firstIdx = __ldg(&sortedIndices[blockIdx.x * blockDim.x]);
    if (threadIndex>=number) return;
    const int posX = (int)(p.pos[0][threadIndex]-0.5f);
    const int posY = (int)(p.pos[1][threadIndex]-0.5f);
    const int posZ = (int)(p.pos[2][threadIndex]-0.5f);
    const int minX = (int)(p.pos[0][firstIdx]-0.5f);
    const int minY = (int)(p.pos[1][firstIdx]-0.5f);
    const int minZ = (int)(p.pos[2][firstIdx]-0.5f);

    float c00 = p.c[0][threadIndex]; float c01 = p.c[1][threadIndex]; float c02 = p.c[2][threadIndex];
    float c10 = p.c[3][threadIndex]; float c11 = p.c[4][threadIndex]; float c12 = p.c[5][threadIndex];
    float c20 = p.c[6][threadIndex]; float c21 = p.c[7][threadIndex]; float c22 = p.c[8][threadIndex];

    const float3 pPos = {p.pos[0][threadIndex],p.pos[1][threadIndex],p.pos[2][threadIndex]};
    const float3 pVel = {p.vel[0][threadIndex],p.vel[1][threadIndex],p.vel[2][threadIndex]};
    const float pM = p.m[threadIndex];

    __shared__ float shMass[SHARED_GRID_SIZE];
    __shared__ float shMomX[SHARED_GRID_SIZE];
    __shared__ float shMomY[SHARED_GRID_SIZE];
    __shared__ float shMomZ[SHARED_GRID_SIZE];
    for (int i=threadIdx.x;i<SHARED_GRID_SIZE;i+=blockDim.x)
    {
        shMass[i] = 0;
        shMomX[i] = 0;
        shMomY[i] = 0;
        shMomZ[i] = 0;
    }

    __syncthreads();

    for (int i=0;i<3;i++)
        for (int j=0;j<3;j++)
            for (int k=0;k<3;k++)
                if (g.isInBounds(posX+i,posY + j, posZ + k))
                {
                    size_t cellIdx = g.getGridIdx(posX + i, posY + j, posZ + k);

                    int dx = posX + i - minX + 1;
                    int dy = posY + j - minY + 1;
                    int dz = posZ + k - minZ + 1;
                    int localIdx = dz * (SHARED_GRID_HEIGHT * SHARED_GRID_HEIGHT) + dy * SHARED_GRID_HEIGHT + dx;


                    float3 d = {pPos.x - (posX + i), pPos.y - (posY + j), pPos.z - (posZ + k)};
                    float weight = g.spline(d.x) * g.spline(d.y) * g.spline(d.z);
                    float weightedMass = pM * weight;
                    float3 Cxd = p.multiplyCxd(c00, c01, c02, c10, c11, c12, c20, c21, c22, d);
                    float velX = weightedMass * pVel.x + Cxd.x;
                    float velY = weightedMass * pVel.y + Cxd.y;
                    float velZ = weightedMass * pVel.z + Cxd.z;

                    if (dx >= 0 && dx < SHARED_GRID_HEIGHT && dy >= 0 && dy < SHARED_GRID_HEIGHT && dz >= 0 && dz <
                        SHARED_GRID_HEIGHT)
                    {
                        atomicAdd(&shMass[localIdx], weightedMass);
                        atomicAdd(&shMomX[localIdx], velX);
                        atomicAdd(&shMomY[localIdx], velY);
                        atomicAdd(&shMomZ[localIdx], velZ);
                    }
                }

    __syncthreads();
    for (int i=threadIdx.x;i<SHARED_GRID_SIZE;i+=blockDim.x)
    {
        int gx = minX + (i % SHARED_GRID_HEIGHT);
        int gy = minY + (i / SHARED_GRID_HEIGHT) % SHARED_GRID_HEIGHT;
        int gz = minZ + i / (SHARED_GRID_HEIGHT * SHARED_GRID_HEIGHT);
        size_t gIdx = g.getGridIdx(gx, gy, gz);
        if (shMass[i]>1e-9)
        {
            atomicAdd(&g.mass[gIdx],shMass[i]);
            atomicAdd(&g.momentum[0][gIdx],shMomX[i]);
            atomicAdd(&g.momentum[1][gIdx],shMomY[i]);
            atomicAdd(&g.momentum[2][gIdx],shMomZ[i]);
        }

    }


}

__global__ void gridUpdate(Grid g,size_t cellsPerPage,int* active, size_t numOfActive)
{
    const int globalThreadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int pageIdx = globalThreadIndex / cellsPerPage;
    if (pageIdx >= numOfActive) return;
    const int localIdx = globalThreadIndex % cellsPerPage;
    const int threadIndex = active[pageIdx] * cellsPerPage + localIdx;
    float mass = g.mass[threadIndex];
    float3 momentum = {g.momentum[0][threadIndex],g.momentum[1][threadIndex],g.momentum[2][threadIndex]};
    float3 velocity = {0.0f,0.0f,0.0f};
    if (mass> 1e-9f)
    {
        velocity.x += momentum.x/mass;
        velocity.y += momentum.y/mass - GRAVITY*DT;
        velocity.z += momentum.z/mass;
        g.momentum[0][threadIndex] = velocity.x;
        g.momentum[1][threadIndex] = velocity.y;
        g.momentum[2][threadIndex] = velocity.z;
    }
}

__global__ void setKeys(Particles p,unsigned int* keys,int number)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex>=number) return;
    const int posX = (int)(p.pos[0][threadIndex]);
    const int posY = (int)(p.pos[1][threadIndex]);
    const int posZ = (int)(p.pos[2][threadIndex]);
    unsigned int key = (p.expandBits(posX) << 0) | (p.expandBits(posY) << 1) | (p.expandBits(posZ) << 2);
    if (posX <0 || posX >= SIZE_X-1 || posY < 0 || posY >= SIZE_Y-1 || posZ < 0 || posZ >= SIZE_Z-1) key = 0xFFFFFFFF;
    keys[threadIndex] = key;
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
    else return ((value + rounder - 1) / rounder) * rounder;
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
    sortParticles();
    initGrid();
    step();
    getchar();
}

Engine::Engine(int n, float *h_buffer)
{
    number = n;
    this->h_buffer = h_buffer;
    gen = std::mt19937(std::random_device{}());
    blocksPerGrid = (number+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    initCuda();
    initParticles();
    sortParticles();
    initGrid();
    step();
    getchar();
}

Engine::~Engine()
{
    delete[] h_buffer;
    cudaFree(d_buffer);
    cudaFree(d_buffer_B);
    cuCtxPopCurrent(&context);
}

void Engine::step()
{
    unsigned int *d_keys;
    unsigned int *d_values;
    handleCUDAError(cudaMalloc((void**)&d_keys,number*sizeof(unsigned int)));
    handleCUDAError(cudaMalloc((void**)&d_values,number*sizeof(unsigned int)));
    setKeys<<<blocksPerGrid,THREADS_PER_BLOCK>>>(getParticles(),d_keys,number);
    thrust::sequence(thrust::device,d_values,d_values+number);
    thrust::sort_by_key(thrust::device,d_keys,d_keys+number,d_values);
    Particles p1 = getParticles();
    Particles p2 = getParticles_B();


    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(blocksPerGrid);
    p2GTransfer<<<gridDim,blockDim>>>(getParticles(),getGrid(),number,d_values);
    handleCUDAError(cudaDeviceSynchronize());
    size_t activePageThreadCount = activeIndices.size() * cellsPerPage;
    size_t blocksPerActive = (activePageThreadCount+THREADS_PER_BLOCK-1)/THREADS_PER_BLOCK;
    std::cout << "SIZE: " << activeIndices.size() << std::endl;
    int* d_active;
    handleCUDAError(cudaMalloc((void**)&d_active,activeIndices.size()*sizeof(float)));
    handleCUDAError(cudaMemcpy(d_active,activeIndices.data(),activeIndices.size()*sizeof(float),cudaMemcpyHostToDevice));
    gridUpdate<<<blocksPerActive,THREADS_PER_BLOCK>>>(getGrid(),cellsPerPage,d_active,activeIndices.size());
    handleCUDAError(cudaDeviceSynchronize());
    handleCUDAError(cudaFree(d_keys));
    handleCUDAError(cudaFree(d_values));
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
    // std::uniform_real_distribution<float> distX(0, SIZE_X);
    // std::uniform_real_distribution<float> distY(0, SIZE_Y);
    // std::uniform_real_distribution<float> distZ(0, 10);
    // h_buffer = new float[number * PARTICLE_SIZE];
    // for (int i = 0; i < number; i++)
    // {
    //     h_buffer[i] = distX(gen);
    //     h_buffer[i + number] = distY(gen);
    //     h_buffer[i + number * 2] = distZ(gen);
    //     h_buffer[i + number * (PARTICLE_SIZE - 2)] = 10;
    //     h_buffer[i + number * (PARTICLE_SIZE - 1)] = 10;
    // }
    handleCUDAError(cudaMalloc((void**)&d_buffer, number * PARTICLE_SIZE * sizeof(float)));
    handleCUDAError(cudaMalloc((void**)&d_buffer_B,number * PARTICLE_SIZE * sizeof(float)));
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
            activeIndices.push_back(i);
        }
    }
    std::cout << "OCCUPIED: " << occupiedCount << std::endl;
    std::cout << "PAGE COUNT: " << pageCount << std::endl;
    handleCUDAError(cudaFree(occupancy_d));

}

void Engine::sortParticles()
{

    //reorderParticles<<<blocksPerGrid,THREADS_PER_BLOCK>>>(p1,p2,number,d_values);


}


