#include <stdio.h>
#include "kernels.h"


__global__ void testKernel(Particles p)
{
    const int threadIndex = threadIdx.x;
    printf("Hello Particle: %f\n",p.pos[0][threadIndex]);
}

void summonTestKernel(Particles p,int number)
{
    testKernel<<<1,number>>>(p);
}


