#include <stdio.h>
#include "kernels.h"

__global__ void testKernel()
{
    const int threadIndex = threadIdx.x;
    printf("Hello Thread: %d\n",threadIndex);
}

void summonTestKernel(int number)
{
    testKernel<<<1,number>>>();
}


