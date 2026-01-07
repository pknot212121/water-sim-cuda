#include <device_launch_parameters.h>
// #include <c++/13/cstdio>
#include <cstdio>
#include "common.cuh"

__global__ void p2GTransferScatter(Particles p,Grid g,int number,int* sortedIndices)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int particleIdx = 0;
    if (threadIndex < number) particleIdx = __ldg(&sortedIndices[threadIndex]);
    int blockStartIdx = blockIdx.x * blockDim.x;
    if (blockStartIdx >= number) return;
    const int firstIdx = __ldg(&sortedIndices[blockStartIdx]);
    const int posX = (int)(p.pos[0][particleIdx]-0.5f),posY = (int)(p.pos[1][particleIdx]-0.5f),posZ = (int)(p.pos[2][particleIdx]-0.5f);
    const int minX = (int)(p.pos[0][firstIdx]-0.5f), minY = (int)(p.pos[1][firstIdx]-0.5f), minZ = (int)(p.pos[2][firstIdx]-0.5f);

    float oldC[9]; float oldF[9];
    for (int i=0;i<9;i++) oldC[i]=p.c[i][particleIdx];

    const float3 pPos = {p.pos[0][particleIdx],p.pos[1][particleIdx],p.pos[2][particleIdx]};
    const float3 pVel = {p.vel[0][particleIdx],p.vel[1][particleIdx],p.vel[2][particleIdx]};
    const float pM = p.m[particleIdx];

    #pragma unroll
    for(int i=0; i<9; i++) oldF[i] = p.f[i][particleIdx];
    float J = det3x3(oldF);
    J = fmaxf(0.1f, fminf(J, 1.1f));
    float pressure = COMPRESSION * (powf(J,GAMMA) - 1.0f);
    if (pressure < 0.0f) pressure = 0.0f;
    float volume = p.v[particleIdx] * J;
    float stressTerm[9] = {1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f};

    float multi = 4.0f * DT * volume * pressure / pM;
    multiply3x3ByConst(multi,stressTerm,stressTerm);
    add3x3(oldC,stressTerm,oldC);

    __shared__ float shMass[SHARED_GRID_SIZE], shMomX[SHARED_GRID_SIZE],shMomY[SHARED_GRID_SIZE],shMomZ[SHARED_GRID_SIZE];
    for (int i=threadIdx.x;i<SHARED_GRID_SIZE;i+=blockDim.x)
    {
        shMass[i] = 0;shMomX[i] = 0;shMomY[i] = 0;shMomZ[i] = 0;
    }
    float distToWall = getSDF(pPos, g);
    float normNumber = 1.0f;
    // bool useOneSided = (distToWall < 1.0f) && (distToWall > 0.0f);
    bool useOneSided = false;
    float3 normal = {0,0,0};
    if (useOneSided) normal = calculateNormal(pPos, g);
    __syncthreads();

    if (threadIndex<number)
    {
        float wx[3], wy[3], wz[3];
        #pragma unroll
        for(int i=0; i<3; i++) wx[i] = spline(pPos.x - (posX + i));
        #pragma unroll
        for(int i=0; i<3; i++) wy[i] = spline(pPos.y - (posY + i));
        #pragma unroll
        for(int i=0; i<3; i++) wz[i] = spline(pPos.z - (posZ + i));

        if (useOneSided)
        {
            float validWeightSum = 0.0f;
            #pragma unroll
            for (int i=0;i<3;i++)
                #pragma unroll
                for (int j=0;j<3;j++)
                    #pragma unroll
                    for (int k=0;k<3;k++)
                        if (isInBounds(posX+i,posY+j,posZ+k))
                        {
                            float3 dist = {(posX + i) - pPos.x, (posY + j) - pPos.y, (posZ + k) - pPos.z};
                            if ((dist.x*normal.x + dist.y*normal.y + dist.z * normal.z) >= 0) validWeightSum += wx[i] * wy[j] * wz[k];
                        }

            if (validWeightSum > 0.5f)
            {
                normNumber = 1.0f / validWeightSum;
                if (normNumber>3.0f) normNumber=3.0f;
            }
            else {normNumber = 1.0f; useOneSided = false;}
        }

        #pragma unroll
        for (int i=0;i<3;i++)
            #pragma unroll
            for (int j=0;j<3;j++)
                #pragma unroll
                for (int k=0;k<3;k++)
                    if (isInBounds(posX+i,posY + j, posZ + k))
                    {
                        size_t cellIdx = getGridIdx(posX + i, posY + j, posZ + k);
                        int dx = posX + i - minX + 1; int dy = posY + j - minY + 1; int dz = posZ + k - minZ + 1;
                        int localIdx = dz * (SHARED_GRID_HEIGHT * SHARED_GRID_HEIGHT) + dy * SHARED_GRID_HEIGHT + dx;

                        float3 d = {pPos.x - (posX + i), pPos.y - (posY + j), pPos.z - (posZ + k)};
                        float3 dist = {-d.x,-d.y,-d.z};

                        float weight = wx[i] * wy[j] * wz[k];
                        float weightedMass = pM * weight * normNumber;

                        float3 Cxd = multiplyCxd(oldC, dist);
                        float velX = weightedMass * (pVel.x + Cxd.x);float velY = weightedMass * (pVel.y + Cxd.y);float velZ = weightedMass * (pVel.z + Cxd.z);

                        if (useOneSided && (dist.x*normal.x + dist.y*normal.y + dist.z*normal.z < 0) ) continue;
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


    __syncthreads();
    for (int i=threadIdx.x;i<SHARED_GRID_SIZE;i+=blockDim.x)
    {
        int gx = minX + (i % SHARED_GRID_HEIGHT)-1;
        int gy = minY + (i / SHARED_GRID_HEIGHT) % SHARED_GRID_HEIGHT-1;
        int gz = minZ + i / (SHARED_GRID_HEIGHT * SHARED_GRID_HEIGHT)-1;
        if (isInBounds(gx,gy,gz))
        {
            size_t gIdx = getGridIdx(gx, gy, gz);
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

__global__ void gridUpdate(Grid g)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    int x = threadIndex % SIZE_X; int y = (threadIndex / SIZE_X) % SIZE_Y; int z = threadIndex / (SIZE_X*SIZE_Y);
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

        g.momentum[0][threadIndex] = velocity.x; g.momentum[1][threadIndex] = velocity.y; g.momentum[2][threadIndex] = velocity.z;
    }
    else {g.momentum[0][threadIndex] = 0.0f; g.momentum[1][threadIndex] = 0.0f; g.momentum[2][threadIndex] = 0.0f;}
}


__global__ void g2PTransfer(Particles p, Grid g,int number,int *sortedIndices)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex>=number) return;
    const int particleIdx = __ldg(&sortedIndices[threadIndex]);
    const int posX = (int)(p.pos[0][particleIdx]-0.5f),posY = (int)(p.pos[1][particleIdx]-0.5f), posZ = (int)(p.pos[2][particleIdx]-0.5f);
    const float3 pPos = {p.pos[0][particleIdx],p.pos[1][particleIdx],p.pos[2][particleIdx]};

    float3 totalVel = {0.0f,0.0f,0.0f};
    float totalC[9] = {0,0,0,0,0,0,0,0,0};
    float wx[3], wy[3], wz[3];
    #pragma unroll
    for(int i=0; i<3; i++) wx[i] = spline(pPos.x - (posX + i));
    #pragma unroll
    for(int i=0; i<3; i++) wy[i] = spline(pPos.y - (posY + i));
    #pragma unroll
    for(int i=0; i<3; i++) wz[i] = spline(pPos.z - (posZ + i));

    #pragma unroll
    for (int i=0;i<3;i++)
        #pragma unroll
        for (int j=0;j<3;j++)
            #pragma unroll
            for (int k=0;k<3;k++)
                if (isInBounds(posX+i,posY + j, posZ + k))
                {
                    size_t cellIdx = getGridIdx(posX + i, posY + j, posZ + k);
                    float3 d = {pPos.x - (float)(posX + i), pPos.y - (float)(posY + j), pPos.z - (float)(posZ + k)};
                    float weight = wx[i]*wy[j]*wz[k];
                    float3 dist = {-d.x,-d.y,-d.z};

                    totalVel.x += weight * g.momentum[0][cellIdx];totalVel.y += weight * g.momentum[1][cellIdx];totalVel.z += weight * g.momentum[2][cellIdx];
                    float term = 4.0f * weight;
                    totalC[0] += term * g.momentum[0][cellIdx] * dist.x;totalC[1] += term * g.momentum[0][cellIdx] * dist.y;totalC[2] += term * g.momentum[0][cellIdx] * dist.z;
                    totalC[3] += term * g.momentum[1][cellIdx] * dist.x;totalC[4] += term * g.momentum[1][cellIdx] * dist.y;totalC[5] += term * g.momentum[1][cellIdx] * dist.z;
                    totalC[6] += term * g.momentum[2][cellIdx] * dist.x;totalC[7] += term * g.momentum[2][cellIdx] * dist.y;totalC[8] += term * g.momentum[2][cellIdx] * dist.z;
                }

    float oldF[9];float newF[9];float tempC[9];
    #pragma unroll
    for (int i = 0; i < 9; i++) oldF[i] = p.f[i][particleIdx];
    for (int i = 0; i < 9; i++) tempC[i] = totalC[i];
    calculateNewF(tempC, oldF, newF);
    float J = det3x3(newF);
    if (J < 0.3f)
    {
        float scale = powf(0.3f / J, 1.0f / 3.0f);
        for (int i = 0; i < 9; i++) newF[i] *= scale;
    }

    float3 nextPos = {p.pos[0][particleIdx] + totalVel.x * DT,p.pos[1][particleIdx] + totalVel.y * DT,p.pos[2][particleIdx] + totalVel.z * DT};

    float dist = getSDF(nextPos, g);
    if (dist < 0.0f)
    {
        float3 normal = calculateNormal(nextPos, g);
        nextPos.x -= dist * normal.x;
        nextPos.y -= dist * normal.y;
        nextPos.z -= dist * normal.z;
        float dot = totalVel.x * normal.x + totalVel.y * normal.y + totalVel.z * normal.z;
        if (dot < 0.0f)
        {
            totalVel.x -= dot * normal.x;
            totalVel.y -= dot * normal.y;
            totalVel.z -= dot * normal.z;
        }
    }

    p.vel[0][particleIdx] = totalVel.x;
    p.vel[1][particleIdx] = totalVel.y;
    p.vel[2][particleIdx] = totalVel.z;
    p.pos[0][particleIdx] = nextPos.x;
    p.pos[1][particleIdx] = nextPos.y;
    p.pos[2][particleIdx] = nextPos.z;
    for (int i = 0; i < 9; i++) p.c[i][particleIdx] = totalC[i];
    for (int i = 0; i < 9; i++) p.f[i][particleIdx] = newF[i];
}

__global__ void makeSDF(float* sdfGrid,const Triangle* __restrict__ triangles,int triangleNumber)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex>=SIZE_X*SIZE_Y*SIZE_Z) return;
    int x = threadIndex % SIZE_X; int y = (threadIndex / SIZE_X) % SIZE_Y; int z = threadIndex / (SIZE_X*SIZE_Y);
    float3 P = {x*1.0f,y*1.0f,z*1.0f};
    float totalWindingNumber = 0.0f;
    float minDistSq = 1e30f;

    for (int i=0;i<triangleNumber;i++)
    {
        Triangle t = triangles[i];
        totalWindingNumber+=calculateSolidAngle(P,t);
        float dSq = pointTriangleDistanceSq(P,t);
        if (dSq < minDistSq) minDistSq = dSq;
    }
    float wn = totalWindingNumber / (4.0f * 3.1415f);
    sdfGrid[threadIndex] = (wn > 0.5f ? -1.0f : 1.0f) * sqrtf(minDistSq);
}


__global__ void testKernel(Particles p,int number)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIndex>=number) return;
    if (p.m[threadIndex]<0.00001f) printf("Hello Mass: %f\n",p.m[threadIndex]);
    if (p.v[threadIndex]<0.00001f) printf("Hello volume: %f\n",p.v[threadIndex]);
}

__global__ void emptyGrid(Grid g)
{
    const int threadIndex = blockIdx.x * blockDim.x + threadIdx.x;
    g.mass[threadIndex] = 0.0f;
    g.momentum[0][threadIndex] = 0.0f;
    g.momentum[1][threadIndex] = 0.0f;
    g.momentum[2][threadIndex] = 0.0f;
}


__global__ void gridTest(Grid g,int targetX, int targetY, int targetZ)
{
    size_t idx = getGridIdx(targetX,targetY,targetZ);
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
    p.v[threadIndex] = RESOLUTION*RESOLUTION*RESOLUTION;
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
