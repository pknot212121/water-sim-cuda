#pragma once

#include <vector_types.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
// #include <c++/13/cstdint>
#include <cstdint>

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
constexpr size_t SIZE_X = 64;
constexpr size_t SIZE_Y = 64;
constexpr size_t SIZE_Z = 128;
constexpr size_t PADDING = 2;

constexpr size_t CELL_ATTRIBUTE_COUNT = 4;
constexpr size_t CELL_SIZE = CELL_ATTRIBUTE_COUNT*sizeof(float);

constexpr size_t GRID_SIZE = SIZE_X*SIZE_Y*SIZE_Z*CELL_SIZE;
constexpr size_t GRID_NUMBER = SIZE_X*SIZE_Y*SIZE_Z;





/* ---- OTHER CONSTS ---- */
constexpr size_t THREADS_PER_BLOCK = 256;

constexpr float GRAVITY = 9.81f;
constexpr float DT = 0.05f;
constexpr float GAMMA = 7.0f;
constexpr float COMPRESSION = 100.0f;
constexpr int SHARED_GRID_HEIGHT = 11;
constexpr int SHARED_GRID_SIZE = SHARED_GRID_HEIGHT*SHARED_GRID_HEIGHT*SHARED_GRID_HEIGHT;
constexpr size_t GRID_BLOCKS = (GRID_NUMBER + THREADS_PER_BLOCK-1) / THREADS_PER_BLOCK;


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

    __device__ __forceinline__ float3 multiplyCxd(
        float* C, float3 d)
    {
        return float3(
            C[0]*d.x+C[1]*d.y+C[2]*d.z,
            C[3]*d.x+C[4]*d.y+C[5]*d.z,
            C[6]*d.x+C[7]*d.y+C[8]*d.z);
    }

    __device__ inline unsigned int expandBits(unsigned int v)
    {
        v = v & 0x000003FF;
        v = (v | (v << 16)) & 0x030000FF;
        v = (v | (v << 8)) & 0x0300F00F;
        v = (v | (v << 4)) & 0x030C30C3;
        v = (v | (v << 2)) & 0x09249249;
        return v;
    }
};

struct __align__(16) Grid
{
    float* mass;
    float* momentum[3];

    __host__ __device__ Grid(float* buffer)
    {
        mass = buffer + GRID_NUMBER * 0;
        for (int i=0;i<3;i++) momentum[i] = buffer + GRID_NUMBER * (i+1);
    }

    __host__ __device__ int getGridIdx(int x,int y,int z)
    {
        return z * SIZE_X*SIZE_Y + y * SIZE_X + x;
    }

    __host__ __device__ float spline(float x)
    {
        if (fabsf(x)<0.5) return 0.75-x*x;
        if (fabsf(x)>=0.5 && fabsf(x)<1.5) return (1.5-fabsf(x))*(1.5-fabsf(x))*0.5;
        return 0.0f;
    }

    __host__ __device__ bool isInBounds(int x,int y,int z)
    {
        if (x<0 || x>=SIZE_X) return false;
        if (y<0 || y>=SIZE_Y) return false;
        if (z<0 || z>=SIZE_Z) return false;
        return true;
    }



};



