#include "common.cuh"

__constant__ SimConfig d_config;

void uploadConfigToGPU(const SimConfig& h_config)
{
    cudaMemcpyToSymbol(d_config,&h_config,sizeof(SimConfig));
}