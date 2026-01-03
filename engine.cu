#include "engine.h"

#include<fstream>
#include<vector>

__global__ void testKernel(Particles p,int number);
__global__ void g2PTransfer(Particles p, Grid g,int number,int *sortedIndices);
__global__ void p2GTransferScatter(Particles p,Grid g,int number,int* sortedIndices);
__global__ void gridUpdate(Grid g);
__global__ void gridTest(Grid g,int targetX, int targetY, int targetZ);
__global__ void setKeys(Particles p,int* keys,int number);
__global__ void sortedTest(int *sorted);
__global__ void changeFormat(Particles p,float3 *buf,int number);
__global__ void emptyGrid(Grid g);
__global__ void initFMatrices(Particles p,int number);
__global__ void checkForNANs(Particles p,int number);

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
    initSDF();
    initGrid();
}

Engine::~Engine()
{
    delete[] h_buffer;
    cudaFree(d_buffer);
    cudaFree(d_grid_buffer);
    cudaFree(d_values);
    cudaFree(d_cell_offsets);
    cudaDestroyTextureObject(sdfTex);
    cudaFreeArray(d_sdfArray);
}

void Engine::step()
{

    sortParticles();
    // testKernel<<<blocksPerGrid,THREADS_PER_BLOCK>>>(getParticles(),number);
    // handleCUDAError(cudaDeviceSynchronize());
    emptyGrid<<<GRID_BLOCKS,THREADS_PER_BLOCK>>>(getGrid());
    handleCUDAError(cudaDeviceSynchronize());
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

    initFMatrices<<<blocksPerGrid,THREADS_PER_BLOCK>>>(getParticles(),number);
    handleCUDAError(cudaDeviceSynchronize());

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

std::vector<float> loadSDF(const std::string& filename, int res) {
    size_t size = res * res * res;
    std::vector<float> data(size);

    std::ifstream is(filename, std::ios::binary);
    if (is) {
        is.read(reinterpret_cast<char*>(data.data()), size * sizeof(float));
    }
    else
    {
        std::cout << "Nie znaleziono!" << std::endl;
        exit(1);
    }
    return data;
}


void Engine::initSDF()
{
    float sdfSize = 10.0f;
    float3 sdfOffset = {0.0f,0.0f,0.0f};

    std::vector<float> h_sdfData = loadSDF("sdf_creator/model_pcu.sdf",SDF_RESOLUTION);
    if (h_sdfData.size() < (size_t)SDF_RESOLUTION * SDF_RESOLUTION * SDF_RESOLUTION) {
        std::cerr << "ERROR: SDF RESOLUTION IS NOT EQUAL TO THE PARAMETER IN COMMON.CUH" << std::endl;
        exit(1);
    }
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(SDF_RESOLUTION,SDF_RESOLUTION,SDF_RESOLUTION);
    handleCUDAError(cudaMalloc3DArray(&d_sdfArray,&channelDesc,extent));
    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr(h_sdfData.data(),SDF_RESOLUTION*sizeof(float),SDF_RESOLUTION,SDF_RESOLUTION);
    copyParams.dstArray = d_sdfArray;
    copyParams.extent = extent;
    copyParams.kind = cudaMemcpyHostToDevice;
    handleCUDAError(cudaMemcpy3D(&copyParams));
    cudaResourceDesc resDesc = {};
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = d_sdfArray;

    cudaTextureDesc texDesc = {};
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = true;
    handleCUDAError(cudaCreateTextureObject(&sdfTex,&resDesc,&texDesc,nullptr));
    float3 gridCenter = {SIZE_X/2,SIZE_Y/2,SIZE_Z/2};
    float halfSize = sdfSize/2;

    this->sdfBoxMin = {
        gridCenter.x + sdfOffset.x - halfSize,
        gridCenter.y + sdfOffset.y - halfSize,
        gridCenter.z +sdfOffset.z - halfSize
    };

    this->sdfBoxMax = {
        gridCenter.x + sdfOffset.x + halfSize,
        gridCenter.y + sdfOffset.y + halfSize,
        gridCenter.z +sdfOffset.z + halfSize
    };

}

