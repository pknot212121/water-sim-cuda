#include <stdio.h>
#include "kernels.h"


__global__ void testKernel(Particles p)
{
    const int threadIndex = threadIdx.x;
    printf("Hello Particle: %f\n",p.pos[0][threadIndex]);
}

__global__ void occupancyCheckInit(Particles p, size_t cellsPerPage,bool* occupancy)
{
    const int threadIndex = threadIdx.x;
    const int posX = (int)p.pos[0][threadIndex];
    const int posY = (int)p.pos[1][threadIndex];
    const int posZ = (int)p.pos[2][threadIndex];

    const int cellIdx = posZ * SIZE_X * SIZE_Y + posY * SIZE_X + posX;
    const int pageIdx = cellIdx / cellsPerPage;
    occupancy[pageIdx] = true;
}

void summonTestKernel(Particles p,int number)
{
    testKernel<<<1,number>>>(p);
}

void summonOccupancyCheckInit(Particles p,int number,size_t cellsPerPage,bool* occupancy)
{
    occupancyCheckInit<<<1,number>>>(p,cellsPerPage,occupancy);
}


