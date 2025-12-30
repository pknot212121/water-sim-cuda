#include <device_launch_parameters.h>
#include <c++/13/cstdio>
#include "common.cuh"

__global__ void testKernel(Particles p)
{
    const int threadIndex = threadIdx.x;
    printf("Hello Particle: %f\n",p.pos[0][threadIndex]);
}

__global__ void p2GTransfer(Particles p,Grid g,int number,int* sortedIndices,int* cellOffsets)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex>= GRID_NUMBER) return;

    int x = threadIndex % SIZE_X;
    int y = (threadIndex / SIZE_X) & SIZE_Y;
    int z = threadIndex / (SIZE_X*SIZE_Y);

    float totalMass = 0.0f;
    float3 totalMomentum = {0.0f,0.0f,0.0f};
    #pragma unroll
    for (int i=-1;i<=1;i++)
    {
        #pragma unroll
        for (int j=-1;j<=1;j++)
        {
            #pragma unroll
            for (int k=-1;k<=1;k++)
            {
                int gx = x+i; int gy = y+j; int gz = z+k;
                if (gx>=0 && gx< SIZE_X && gy>=0 && gy < SIZE_Y && gz>=0 && gz<SIZE_Z)
                {
                    int neighborIdx = g.getGridIdx(gx,gy,gz);
                    int startIdx = cellOffsets[neighborIdx];
                    int endIdx = cellOffsets[neighborIdx+1];

                    for (int p_map_idx = startIdx; p_map_idx < endIdx; p_map_idx++)
                    {
                        int particleIdx = sortedIndices[p_map_idx];
                        float3 pPos = {p.pos[0][particleIdx],p.pos[1][particleIdx],p.pos[2][particleIdx]};
                        float3 pVel = {p.vel[0][particleIdx],p.vel[1][particleIdx],p.vel[2][particleIdx]};
                        float c[9] = {
                            p.c[0][particleIdx],p.c[1][particleIdx],p.c[2][particleIdx],
                            p.c[3][particleIdx],p.c[4][particleIdx],p.c[5][particleIdx],
                            p.c[6][particleIdx],p.c[7][particleIdx],p.c[8][particleIdx]
                        };

                        float pM = p.m[particleIdx];
                        float3 d = {pPos.x-gx+0.5f,pPos.y - gy+0.5f,pPos.z - gz+0.5f};
                        float weight = g.spline(d.x)*g.spline(d.y)*g.spline(d.z);
                        float weightedMass = pM * weight;
                        float3 Cxd = p.multiplyCxd(c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7],c[8],d);
                        float velX = weightedMass*pVel.x+Cxd.x;
                        float velY = weightedMass*pVel.y+Cxd.y;
                        float velZ = weightedMass*pVel.z+Cxd.z;
                        totalMass += weightedMass;
                        totalMomentum.x += velX;
                        totalMomentum.y += velY;
                        totalMomentum.z += velZ;
                    }
                }
            }
        }
    }
    g.mass[threadIndex] = totalMass;
    g.momentum[0][threadIndex] = totalMomentum.x;
    g.momentum[1][threadIndex] = totalMomentum.y;
    g.momentum[2][threadIndex] = totalMomentum.z;
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

__global__ void gridTest(Grid g,int targetX, int targetY, int targetZ)
{
    size_t idx = g.getGridIdx(targetX,targetY,targetZ);
    g.mass[idx] = 2137.0f;
}

__global__ void sortedTest(int *sorted)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (sorted[threadIndex]!=0) printf("Hello Sorted: %u\n",sorted[threadIndex]);
}

__global__ void setKeys(Particles p,int* keys,int number)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex>=number) return;
    const int posX = (int)(p.pos[0][threadIndex]);
    const int posY = (int)(p.pos[1][threadIndex]);
    const int posZ = (int)(p.pos[2][threadIndex]);
    unsigned int key = posZ * SIZE_X * SIZE_Y + posY * SIZE_X + posX;
    keys[threadIndex] = key;
}
