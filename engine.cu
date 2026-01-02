#include "engine.h"

__global__ void testKernel(Particles p);
__global__ void p2GTransferGather(Particles p,Grid g,int number,int* sortedIndices,int* cellOffsets);
__global__ void g2PTransfer(Particles p, Grid g,int number,int *sortedIndices);
__global__ void p2GTransferScatter(Particles p,Grid g,int number,int* sortedIndices);
__global__ void gridUpdate(Grid g);
__global__ void gridTest(Grid g,int targetX, int targetY, int targetZ);
__global__ void setKeys(Particles p,int* keys,int number);
__global__ void sortedTest(int *sorted);
__global__ void changeFormat(Particles p,float3 *buf,int number);

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


Engine::Engine(int n, float *h_buffer)
{
    number = n;
    this->h_buffer = h_buffer;
    blocksPerGrid = (number+THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
    initParticles();
    initGrid();
}

Engine::~Engine()
{
    delete[] h_buffer;
    cudaFree(d_buffer);
    cudaFree(d_grid_buffer);
    cudaFree(d_values);
    cudaFree(d_cell_offsets);
}

void Engine::step()
{
    sortParticles();
    p2GTransferScatter<<<blocksPerGrid,THREADS_PER_BLOCK>>>(getParticles(),getGrid(),number,d_values);
    handleCUDAError(cudaDeviceSynchronize());
    gridUpdate<<<GRID_BLOCKS,THREADS_PER_BLOCK>>>(getGrid());
    handleCUDAError(cudaDeviceSynchronize());
    g2PTransfer<<<blocksPerGrid,THREADS_PER_BLOCK>>>(getParticles(),getGrid(),number,d_values);
    handleCUDAError(cudaDeviceSynchronize());
    changeFormat<<<blocksPerGrid,THREADS_PER_BLOCK>>>(getParticles(),positionsToOpenGL,number);
    handleCUDAError(cudaDeviceSynchronize());
}

void Engine::initParticles()
{
    handleCUDAError(cudaMalloc((void**)&d_buffer, number * PARTICLE_SIZE));
    handleCUDAError(cudaMalloc((void**)&d_values,number*sizeof(int)));
    handleCUDAError(cudaMalloc((void**)&d_cell_offsets,GRID_NUMBER*sizeof(int)));
    handleCUDAError(cudaMalloc((void**)&positionsToOpenGL,number*sizeof(float3)));
    handleCUDAError(cudaMemcpy(d_buffer, h_buffer, number * PARTICLE_SIZE, cudaMemcpyHostToDevice));
}


void Engine::initGrid()
{
    handleCUDAError(cudaMalloc((void**)&d_grid_buffer,GRID_SIZE));
    handleCUDAError(cudaMemset(d_grid_buffer,0.0f,GRID_SIZE));
}

void Engine::sortParticles()
{
    int *d_keys;
    handleCUDAError(cudaMalloc((void**)&d_keys,number*sizeof(int)));
    setKeys<<<blocksPerGrid,THREADS_PER_BLOCK>>>(getParticles(), d_keys, number);
    handleCUDAError(cudaDeviceSynchronize());
    thrust::sequence(thrust::device, d_values, d_values + number);
    thrust::sort_by_key(thrust::device, d_keys, d_keys + number, d_values);

    cudaFree(d_keys);
}

