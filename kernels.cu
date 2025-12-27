#include <stdio.h>
#include "kernels.h"


__global__ void testKernel(Particles p)
{
    const int threadIndex = threadIdx.x;
    printf("Hello Particle: %f\n",p.pos[0][threadIndex]);
}

__global__ void occupancyCheck(Particles p)
{
    const int threadIndex = threadIdx.x;
    const int posX = p.pos[0][threadIndex];
    const int posY = p.pos[1][threadIndex];
    const int posZ = p.pos[2][threadIndex];
    
    const int page = posZ * SIZE_X * SIZE_Y + posY * SIZE_X + posX;
    occupancy[page] = true;
}

void summonTestKernel(Particles p,int number)
{
    testKernel<<<1,number>>>(p);
}


