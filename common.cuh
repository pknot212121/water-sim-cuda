#pragma once

#include <vector_types.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
// #include <c++/13/cstdint>
#include <cstdint>
#include <cmath>

/* ---- CONSTS FOR PARTICLES ---- */
enum ParticleAttr
{
    POS_X=0, POS_Y, POS_Z,

    VEL_X, VEL_Y, VEL_Z,

    C00, C01, C02,
    C10, C11, C12,
    C20, C21, C22,

    F00, F01, F02,
    F10, F11, F12,
    F20, F21, F22,

    VOL,
    MASS,
    PARTICLE_ATTRIBUTE_COUNT
};
constexpr size_t PARTICLE_SIZE = PARTICLE_ATTRIBUTE_COUNT * sizeof(float);


/* ---- CONSTS FOR GRID ---- */
constexpr size_t SIZE_X = 128;
constexpr size_t SIZE_Y = 128;
constexpr size_t SIZE_Z = 128;
constexpr size_t PADDING = 2;

constexpr size_t CELL_ATTRIBUTE_COUNT = 4;
constexpr size_t CELL_SIZE = CELL_ATTRIBUTE_COUNT*sizeof(float);

constexpr size_t GRID_SIZE = SIZE_X*SIZE_Y*SIZE_Z*CELL_SIZE;
constexpr size_t GRID_NUMBER = SIZE_X*SIZE_Y*SIZE_Z;





/* ---- OTHER CONSTS ---- */
constexpr size_t THREADS_PER_BLOCK = 256;

constexpr float GRAVITY = 9.81f;
constexpr float DT = 0.0005f;
//constexpr float DT = 0.0001f;
//constexpr float DT = 0.00005f;
constexpr float GAMMA = -3.0f;
//constexpr float COMPRESSION = 500.0f;
constexpr float COMPRESSION = 10.0f;
constexpr float RESOLUTION = 1.0f;
constexpr int SUBSTEPS = 50;
constexpr int SHARED_GRID_HEIGHT = 11;
constexpr int SDF_RESOLUTION = 256;
constexpr int SHARED_GRID_SIZE = SHARED_GRID_HEIGHT*SHARED_GRID_HEIGHT*SHARED_GRID_HEIGHT;
constexpr size_t GRID_BLOCKS = (GRID_NUMBER + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;
constexpr bool PHASING = false;


/* ---- GLOBAL STRUCTURES ---- */
struct __align__(16) Particles
{
    float* pos[3];
    float* vel[3];
    //TODO: change to halfs if possible
    float* c[9];
    float* f[9];
    float* v;
    float* m;


    __host__ __device__ Particles(float* buffer,size_t n)
    {
        for (int i=0;i<3;i++) pos[i] = buffer + n * (POS_X+i);
        for (int i=0;i<3;i++) vel[i] = buffer + n * (VEL_X+i);
        for (int i=0;i<9;i++) c[i] = buffer + n * (C00+i);
        for (int i=0;i<9;i++) f[i] = buffer + n * (F00+i);
        v = buffer + n * VOL;
        m = buffer + n * MASS;
    }

};

struct __align__(16) Grid
{
    float* mass;
    float* momentum[3];
    float* sdf;
    __host__ __device__ Grid(float* buffer,float* sdfBuffer)
    {
        mass = buffer + GRID_NUMBER * 0;
        for (int i=0;i<3;i++) momentum[i] = buffer + GRID_NUMBER * (i+1);
        sdf = sdfBuffer;
    }
};

struct Triangle
{
    float3 v0, v1, v2;
};



/* ---- GLOBAL FUNCTIONS ---- */
#ifdef __CUDACC__
__host__ __device__ inline int getGridIdx(int x,int y,int z)
{
    return z * SIZE_X*SIZE_Y + y * SIZE_X + x;
}

__device__ inline float getSDF(float3 p,Grid g)
{
    int i = floorf(p.x), j = floorf(p.y), k = floorf(p.z);
    float tx = p.x - (float)i, ty = p.y - (float)j, tz = p.z - (float)k;

    auto sample = [&](int x,int y,int z)
    {
        x = max(0,min(x,(int)SIZE_X -1));
        y = max(0,min(y,(int)SIZE_Y -1));
        z = max(0,min(z,(int)SIZE_Z -1));
        return g.sdf[getGridIdx(x,y,z)];
    };

    float N[8] = {
        sample(i,j,k),sample(i+1,j,k),sample(i,j+1,k),sample(i+1,j+1,k),
        sample(i,j,k+1),sample(i+1,j,k+1),sample(i,j+1,k+1),sample(i+1,j+1,k+1),
    };

    float inX[4] = {
        N[0] * (1.0f - tx) + N[1] * tx,
        N[2] * (1.0f - tx) + N[3] * tx,
        N[4] * (1.0f - tx) + N[5] * tx,
        N[6] * (1.0f - tx) + N[7] * tx,
    };

    float inY[2] = {
        inX[0] * (1.0f - ty) + inX[1] * ty,
        inX[2] * (1.0f - ty) + inX[3] * ty
    };

    return inY[0] * (1.0f - tz) + inY[1] * tz;
}

__device__ inline float3 calculateNormal(float3 pos,Grid g)
{
    float eps = 0.1f;
    float3 normal;
    normal.x = getSDF(make_float3(pos.x + eps, pos.y, pos.z), g) - 
               getSDF(make_float3(pos.x - eps, pos.y, pos.z), g);
    normal.y = getSDF(make_float3(pos.x, pos.y + eps, pos.z), g) - 
               getSDF(make_float3(pos.x, pos.y - eps, pos.z), g);
    normal.z = getSDF(make_float3(pos.x, pos.y, pos.z + eps), g) - 
               getSDF(make_float3(pos.x, pos.y, pos.z - eps), g);
    float len = sqrtf(normal.x*normal.x + normal.y*normal.y + normal.z*normal.z) + 1e-6f;
    normal.x /= len; normal.y /= len; normal.z /= len;
    return normal;
}



__device__ __forceinline__ float3 multiplyCxd(
        float* C, float3 d)
{
    return float3(
        C[0]*d.x+C[1]*d.y+C[2]*d.z,
        C[3]*d.x+C[4]*d.y+C[5]*d.z,
        C[6]*d.x+C[7]*d.y+C[8]*d.z);
}




__host__ __device__ inline float spline(float x)
{
    if (fabsf(x)<0.5) return 0.75-x*x;
    if (fabsf(x)>=0.5 && fabsf(x)<1.5) return (1.5-fabsf(x))*(1.5-fabsf(x))*0.5;
    return 0.0f;
}

__host__ __device__ inline bool isInBounds(int x,int y,int z)
{
    if (x<0 || x>=SIZE_X) return false;
    if (y<0 || y>=SIZE_Y) return false;
    if (z<0 || z>=SIZE_Z) return false;
    return true;
}

__device__ inline void multiply3x3(float *A,float *B,float *C)
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

__device__ inline void add3x3(float* A,float* B,float *C)
{
    for (int i=0;i<9;i++) C[i]=A[i]+B[i];
}

__device__ inline float dotVec3(float3 A,float3 B)
{
    return A.x*B.x+A.y*B.y+A.z*B.z;
}

__device__ inline void multiply3x3ByConst(float a,float* B,float* C)
{
    for (int i=0;i<9;i++) C[i]=a*B[i];
}

__device__ __forceinline__ float det3x3(float *A)
{
    return A[0]*(A[4]*A[8]-A[5]*A[7]) - A[1]*(A[3]*A[8]-A[5]*A[6]) + A[2]*(A[3]*A[7] - A[4]*A[6]);
}

__device__ inline void calculateNewF(float *C,float *oldF,float *newF)
{
    float I[9] = {1.0f,0.0f,0.0f,0.0f,1.0f,0.0f,0.0f,0.0f,1.0f};
    multiply3x3ByConst(DT,C,C);
    add3x3(I,C,C);
    multiply3x3(C,oldF,newF);
}

__device__ inline void forceC(float vP,float P,float mass,float* fC)
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

__device__ inline unsigned int expandBits(unsigned int v)
{
    v = (v * 0x00010001u) & 0xFF0000FFu;
    v = (v * 0x00000101u) & 0x0F00F00Fu;
    v = (v * 0x00000011u) & 0xC30C30C3u;
    v = (v * 0x00000005u) & 0x49249249u;
    return v;
}

__host__ __device__ inline float3 operator-(float3 a,float3 b)
{
    return make_float3(a.x-b.x,a.y-b.y,a.z-b.z);
}
__host__ __device__ inline float3 operator+(float3 a,float3 b)
{
    return make_float3(a.x+b.x,a.y+b.y,a.z+b.z);
}

__host__ __device__ inline float3 operator*(float a,float3 b)
{
    return make_float3(a*b.x,a*b.y,a*b.z);
}
__host__ __device__ inline float3 operator*(float3 b,float a)
{
    return make_float3(a*b.x,a*b.y,a*b.z);
}

__device__ inline unsigned int calculateMorton(unsigned int x, unsigned int y, unsigned int z)
{
    return (expandBits(z) << 2) | (expandBits(y) << 1) | expandBits(x);
}


__device__ inline float calculateSolidAngle(float3 P,Triangle t)
{
    float3 a = t.v0 - P;
    float3 b = t.v1 - P;
    float3 c = t.v2 - P;
    float lenA = norm3df(a.x,a.y,a.z);
    float lenB = norm3df(b.x,b.y,b.z);
    float lenC = norm3df(c.x,c.y,c.z);

    float det = a.x*(b.y*c.z - b.z*c.y) + a.y*(b.z*c.x - b.x*c.z) + a.z*(b.x*c.y - b.y*c.x);
    float div = (lenA*lenB*lenC) + dotVec3(a,b)*lenC + dotVec3(a,c)*lenB + dotVec3(b,c)*lenA;
    float omega = 2.0f * atan2f(det,div);
    return omega;
}

__device__ inline float pointTriangleDistanceSq(float3 p,Triangle t)
{
    float3 ab = t.v1 - t.v0;
    float3 ac = t.v2 - t.v0;
    float3 ap = p - t.v0;

    float d1 = dotVec3(ab,ap);
    float d2 = dotVec3(ac,ap);
    if (d1 <= 0.0f && d2 <= 0.0f) return dotVec3(ap,ap);

    float3 bp = p - t.v1;
    float d3 = dotVec3(ab,bp);
    float d4 = dotVec3(ac,bp);
    if (d3 >= 0.0f && d4 <= d3) return dotVec3(bp,bp);

    float vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f)
    {
        float v = d1 / (d1 - d3);
        float3 closest = t.v0 + v * ab;
        float3 diff = p - closest;
        return dotVec3(diff,diff);
    }

    float3 cp = p - t.v2;
    float d5 = dotVec3(ab,cp);
    float d6 = dotVec3(ac,cp);
    if (d6 >= 0.0f && d5 <= d6) return dotVec3(cp,cp);

    float vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f)
    {
        float w = d2 / (d2 - d6);
        float3 closest = t.v0 + w * ac;
        float3 diff = p - closest;
        return dotVec3(diff,diff);
    }

    float va = d3 * d6 - d5 * d4;
    if (va <= 0.0f && (d4-d3) >= 0.0f && (d5 - d6) >= 0.0f)
    {
        float w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        float3 closest = t.v1 + w * (t.v2-t.v1);
        float3 diff = p - closest;
        return dotVec3(diff,diff);
    }

    float denom = 1.0f / (va + vb + vc);
    float v = vb * denom;
    float w = vc * denom;
    float3 closest = t.v0 + ab * v + ac * w;
    float3 diff = p - closest;
    return dotVec3(diff,diff);
}

#endif




