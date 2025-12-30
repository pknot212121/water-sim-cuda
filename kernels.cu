#include <device_launch_parameters.h>
#include <c++/13/cstdio>
#include "common.cuh"

__global__ void testKernel(Particles p)
{
    const int threadIndex = threadIdx.x;
    printf("Hello Particle: %f\n",p.pos[0][threadIndex]);
}

__global__ void p2GTransferScatter(Particles p,Grid g,int number,int* sortedIndices)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    const int particleIdx = __ldg(&sortedIndices[threadIndex]);
    const int firstIdx = __ldg(&sortedIndices[blockIdx.x * blockDim.x]);
    const int posX = (int)(p.pos[0][particleIdx]-0.5f);
    const int posY = (int)(p.pos[1][particleIdx]-0.5f);
    const int posZ = (int)(p.pos[2][particleIdx]-0.5f);
    const int minX = (int)(p.pos[0][firstIdx]-0.5f);
    const int minY = (int)(p.pos[1][firstIdx]-0.5f);
    const int minZ = (int)(p.pos[2][firstIdx]-0.5f);

    float c00 = p.c[0][particleIdx]; float c01 = p.c[1][particleIdx]; float c02 = p.c[2][particleIdx];
    float c10 = p.c[3][particleIdx]; float c11 = p.c[4][particleIdx]; float c12 = p.c[5][particleIdx];
    float c20 = p.c[6][particleIdx]; float c21 = p.c[7][particleIdx]; float c22 = p.c[8][particleIdx];

    const float3 pPos = {p.pos[0][particleIdx],p.pos[1][particleIdx],p.pos[2][particleIdx]};
    const float3 pVel = {p.vel[0][particleIdx],p.vel[1][particleIdx],p.vel[2][particleIdx]};
    const float pM = p.m[particleIdx];

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

    if (threadIndex<number)
    {
        #pragma unroll
        for (int i=0;i<3;i++)
        {
            #pragma unroll
            for (int j=0;j<3;j++)
            {
                #pragma unroll
                for (int k=0;k<3;k++)
                {
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

                        if (dx >= 0 && dx < SHARED_GRID_HEIGHT && dy >= 0 && dy < SHARED_GRID_HEIGHT && dz >= 0 && dz <SHARED_GRID_HEIGHT)
                        {
                            atomicAdd(&shMass[localIdx], weightedMass);
                            atomicAdd(&shMomX[localIdx], velX);
                            atomicAdd(&shMomY[localIdx], velY);
                            atomicAdd(&shMomZ[localIdx], velZ);
                        }
                        else
                        {
                            atomicAdd(&g.mass[cellIdx], weightedMass);
                            atomicAdd(&g.momentum[0][cellIdx], velX);
                            atomicAdd(&g.momentum[1][cellIdx], velY);
                            atomicAdd(&g.momentum[2][cellIdx], velZ);
                        }
                    }
                }
            }
        }
    }


    __syncthreads();
    for (int i=threadIdx.x;i<SHARED_GRID_SIZE;i+=blockDim.x)
    {
        int gx = minX + (i % SHARED_GRID_HEIGHT)-1;
        int gy = minY + (i / SHARED_GRID_HEIGHT) % SHARED_GRID_HEIGHT-1;
        int gz = minZ + i / (SHARED_GRID_HEIGHT * SHARED_GRID_HEIGHT)-1;
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

__global__ void p2GTransferGather(Particles p,Grid g,int number,int* sortedIndices,int* cellOffsets)
{
    int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex>= GRID_NUMBER) return;

    int x = threadIndex % SIZE_X;
    int y = (threadIndex / SIZE_X) % SIZE_Y;
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

__device__ unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}


__device__ unsigned int calculateMorton(unsigned int x, unsigned int y, unsigned int z)
{
    return (expandBits(z) << 2) | (expandBits(y) << 1) | expandBits(x);
}


__global__ void setKeys(Particles p, int* keys, int number)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex >= number) return;

    const unsigned int posX = (unsigned int)max(0.0f, p.pos[0][threadIndex]);
    const unsigned int posY = (unsigned int)max(0.0f, p.pos[1][threadIndex]);
    const unsigned int posZ = (unsigned int)max(0.0f, p.pos[2][threadIndex]);

    keys[threadIndex] = calculateMorton(posX, posY, posZ);
}
