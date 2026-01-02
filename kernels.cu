#include <device_launch_parameters.h>
#include <c++/13/cstdio>
#include "common.cuh"

__global__ void testKernel(Particles p,int number)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex>=number) return;
    if (p.m[threadIndex]<0.00001f) printf("Hello Mass: %f\n",p.m[threadIndex]);
    if (p.v[threadIndex]<0.00001f) printf("Hello volume: %f\n",p.v[threadIndex]);
}

__global__ void p2GTransferScatter(Particles p,Grid g,int number,int* sortedIndices)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int particleIdx = 0;
    if (threadIndex < number) particleIdx = __ldg(&sortedIndices[threadIndex]);
    int blockStartIdx = blockIdx.x * blockDim.x;
    if (blockStartIdx >= number) return;
    const int firstIdx = __ldg(&sortedIndices[blockStartIdx]);
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
        float wx[3], wy[3], wz[3];
        #pragma unroll
        for(int i=0; i<3; i++) wx[i] = g.spline(pPos.x - (posX + i));
        #pragma unroll
        for(int i=0; i<3; i++) wy[i] = g.spline(pPos.y - (posY + i));
        #pragma unroll
        for(int i=0; i<3; i++) wz[i] = g.spline(pPos.z - (posZ + i));

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
                        float3 dist = {-d.x,-d.y,-d.z};
                        float weight = wx[i] * wy[j] * wz[k];
                        float weightedMass = pM * weight;
                        float3 Cxd = p.multiplyCxd(c00, c01, c02, c10, c11, c12, c20, c21, c22, dist);
                        float velX = weightedMass * (pVel.x + Cxd.x);
                        float velY = weightedMass * (pVel.y + Cxd.y);
                        float velZ = weightedMass * (pVel.z + Cxd.z);

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
        if (g.isInBounds(gx,gy,gz))
        {
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

}


__device__ void multiply3x3(float *A,float *B,float *C)
{
        C[0]=A[0]*B[0] + A[1]*B[3] + A[2]*B[6];
        C[1]=A[0]*B[1] + A[1]*B[4] + A[2]*B[7];
        C[2]=A[0]*B[2] + A[1]*B[5] + A[2]*B[8];

        C[3]=A[3]*B[0] + A[4]*B[3] + A[5]*B[6];
        C[4]=A[3]*B[1] + A[4]*B[4] + A[5]*B[7];
        C[5]=A[3]*B[2] + A[4]*B[5] + A[5]*B[8];

        C[6]=A[6]*B[0] + A[7]*B[3] + A[8]*B[6];
        C[7]=A[6]*B[1] + A[7]*B[4] + A[8]*B[7];
        C[8]=A[6]*B[2] + A[7]*B[5] + A[8]*B[8];
}

__device__ void add3x3(float* A,float* B,float *C)
{
    for (int i=0;i<9;i++) C[i]=A[i]+B[i];
}

__device__ void multiply3x3ByConst(float a,float* B,float* C)
{
    for (int i=0;i<9;i++) C[i]=a*B[i];
}

__device__ __forceinline__ float det3x3(float *A)
{
    return A[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(A[3]*A[8]-A[5]*A[6]) + A[2]*(A[3]*A[7] - A[4]*A[6]);
}

__device__ void calculateNewF(float *C,float *oldF,float *newF)
{
    float I[9] = {1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f};
    multiply3x3ByConst(DT,C,C);
    add3x3(I,C,C);
    multiply3x3(C,oldF,newF);
}

__device__ void forceC(float vP,float P,float mass,float* fC)
{
    float I[9] = {1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f};
    if (mass>0.0001)
    {
        multiply3x3ByConst(4.0f*DT*vP*P/mass,I,fC);
    }
    else
    {
        multiply3x3ByConst(4.0f*DT*vP*P,I,fC);
    }

}

__global__ void g2PTransfer(Particles p, Grid g,int number,int *sortedIndices)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex>=number) return;
    const int particleIdx = __ldg(&sortedIndices[threadIndex]);
    const int posX = (int)(p.pos[0][particleIdx]-0.5f);
    const int posY = (int)(p.pos[1][particleIdx]-0.5f);
    const int posZ = (int)(p.pos[2][particleIdx]-0.5f);
    const float3 pPos = {p.pos[0][particleIdx],p.pos[1][particleIdx],p.pos[2][particleIdx]};

    float3 totalVel = {0.0f,0.0f,0.0f};
    float totalC[9] = {0,0,0,0,0,0,0,0,0};
    float wx[3], wy[3], wz[3];
    #pragma unroll
    for(int i=0; i<3; i++) wx[i] = g.spline(pPos.x - (posX + i));
    #pragma unroll
    for(int i=0; i<3; i++) wy[i] = g.spline(pPos.y - (posY + i));
    #pragma unroll
    for(int i=0; i<3; i++) wz[i] = g.spline(pPos.z - (posZ + i));

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
                    float3 d = {pPos.x - (float)(posX + i), pPos.y - (float)(posY + j), pPos.z - (float)(posZ + k)};
                    float weight = wx[i]*wy[j]*wz[k];



                    totalVel.x += weight * g.momentum[0][cellIdx];
                    totalVel.y += weight * g.momentum[1][cellIdx];
                    totalVel.z += weight * g.momentum[2][cellIdx];

                    float3 dist = {-d.x,-d.y,-d.z};
                    float term = 4.0f * weight;
                    totalC[0] += term * g.momentum[0][cellIdx] * dist.x;
                    totalC[1] += term * g.momentum[0][cellIdx] * dist.y;
                    totalC[2] += term * g.momentum[0][cellIdx] * dist.z;
                    totalC[3] += term * g.momentum[1][cellIdx] * dist.x;
                    totalC[4] += term * g.momentum[1][cellIdx] * dist.y;
                    totalC[5] += term * g.momentum[1][cellIdx] * dist.z;
                    totalC[6] += term * g.momentum[2][cellIdx] * dist.x;
                    totalC[7] += term * g.momentum[2][cellIdx] * dist.y;
                    totalC[8] += term * g.momentum[2][cellIdx] * dist.z;
                }
            }
        }
    }

    float oldF[9];
    #pragma unroll
    for (int i=0;i<9;i++) oldF[i]=p.f[i][particleIdx];


    float newF[9]; float fC[9]; float tempC[9];
    for (int i=0;i<9;i++) tempC[i]=totalC[i];
    calculateNewF(tempC,oldF,newF);
    float J = det3x3(newF);

    if (J < 0.1f) J = 0.1f;
    if (J > 10.0f) J = 10.0f;

    float pressure = COMPRESSION * (J-1);
    //pressure = fmaxf(-10.0f, fminf(pressure, 10.0f));
    float Vp = p.v[particleIdx] * J;
    float I[9] = {1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f};
    forceC(Vp,pressure,p.m[particleIdx],fC);
    //p.v[particleIdx] *=J;

    // if (totalVel.x>10.0f) totalVel.x=10.0f; if (totalVel.x<-10.0f) totalVel.x=-10.0f;
    // if (totalVel.y>10.0f) totalVel.y=10.0f; if (totalVel.y<-10.0f) totalVel.y=-10.0f;
    // if (totalVel.z>10.0f) totalVel.z=10.0f; if (totalVel.z<-10.0f) totalVel.z=-10.0f;
    p.vel[0][particleIdx] = totalVel.x;
    p.vel[1][particleIdx] = totalVel.y;
    p.vel[2][particleIdx] = totalVel.z;

    p.pos[0][particleIdx] += totalVel.x * DT;
    p.pos[1][particleIdx] += totalVel.y * DT;
    p.pos[2][particleIdx] += totalVel.z * DT;
    for (int i=0;i<9;i++) p.c[i][particleIdx] = totalC[i] + fC[i];
    float J_root = powf(J, 1.0f/3.0f);
    for (int i=0; i<9; i++) p.f[i][particleIdx] = 0.0f;
    p.f[0][particleIdx] = J;
    p.f[4][particleIdx] = J;
    p.f[8][particleIdx] = J;
}

__global__ void emptyGrid(Grid g)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    g.mass[threadIndex] = 0.0f;
    g.momentum[0][threadIndex] = 0.0f;
    g.momentum[1][threadIndex] = 0.0f;
    g.momentum[2][threadIndex] = 0.0f;
}

__global__ void gridUpdate(Grid g)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int x = threadIndex % SIZE_X;
    int y = (threadIndex / SIZE_X) % SIZE_Y;
    int z = threadIndex / (SIZE_X*SIZE_Y);
    float mass = g.mass[threadIndex];
    float3 momentum = {g.momentum[0][threadIndex],g.momentum[1][threadIndex],g.momentum[2][threadIndex]};
    float3 velocity = {0.0f,0.0f,0.0f};
    if (mass> 1e-9f)
    {
        velocity.x = momentum.x/mass;
        velocity.y = momentum.y/mass - GRAVITY*DT;
        velocity.z = momentum.z/mass;

        if (x<PADDING && velocity.x<0) velocity.x=0.0f;
        if (y<PADDING && velocity.y<0) velocity.y=0.0f;
        if (z<PADDING && velocity.z<0) velocity.z=0.0f;

        if (x>SIZE_X-1-PADDING && velocity.x>0) velocity.x=0.0f;
        if (y>SIZE_Y-1-PADDING && velocity.y>0) velocity.y=0.0f;
        if (z>SIZE_Z-1-PADDING && velocity.z>0) velocity.z=0.0f;

        g.momentum[0][threadIndex] = velocity.x;
        g.momentum[1][threadIndex] = velocity.y;
        g.momentum[2][threadIndex] = velocity.z;
    }
    else
    {
        g.momentum[0][threadIndex] = 0.0f;
        g.momentum[1][threadIndex] = 0.0f;
        g.momentum[2][threadIndex] = 0.0f;
    }
}

__global__ void gridTest(Grid g,int targetX, int targetY, int targetZ)
{
    size_t idx = g.getGridIdx(targetX,targetY,targetZ);
    g.mass[idx] = 2137.0f;
}

__global__ void checkForNANs(Particles p,int number)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (isnan(p.m[threadIndex])) printf("MASA TO NAN!!!\n");
    if (isnan(p.v[threadIndex])) printf("OBJĘTOŚĆ TO NAN!!!\n");
    if (p.m[threadIndex]<0.00001f) printf("MASA TO ZERO!!!\n");
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

__global__ void changeFormat(Particles p,float3 *buf,int number)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex >= number) return;
    buf[threadIndex] = {p.pos[0][threadIndex],p.pos[1][threadIndex],p.pos[2][threadIndex]};
    //printf("PARTICLE: (%f,%f,%f)",buf[threadIndex].x,buf[threadIndex].y,buf[threadIndex].z);
}

__global__ void initFMatrices(Particles p,int number)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex >= number) return;
    p.f[0][threadIndex] = 1.0f;
    p.f[4][threadIndex] = 1.0f;
    p.f[8][threadIndex] = 1.0f;
    p.v[threadIndex] = 0.1f*0.1f*0.1f;
    p.m[threadIndex] = 1.0f;
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
