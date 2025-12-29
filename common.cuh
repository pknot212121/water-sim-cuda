#include <vector_types.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>
#include <c++/13/cstdint>

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
    PARTICLE_SIZE
};

constexpr size_t SIZE_X = 1024;
constexpr size_t SIZE_Y = 1024;
constexpr size_t SIZE_Z = 1024;

constexpr size_t CELL_SIZE = 16;

constexpr size_t GRID_SIZE = SIZE_X*SIZE_Y*SIZE_Z*CELL_SIZE;

constexpr size_t THREADS_PER_BLOCK = 256;

constexpr float GRAVITY = 9.81f;
constexpr float DT = 1.0f;

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
        float c00, float c01, float c02,
        float c10, float c11, float c12,
        float c20, float c21, float c22, float3 d)
    {
        return float3(
            c00*d.x+c01*d.y+c02*d.z,
            c10*d.x+c11*d.y+c12*d.z,
            c20*d.x+c21*d.y+c22*d.z);
    }
};

struct __align__(16) Grid
{
    float* mass;
    float* momentum[3];

    __host__ __device__ size_t getGridIdx(int x,int y,int z)
    {
        return (size_t)z * SIZE_X*SIZE_Y + (size_t)y * SIZE_X + (size_t)x;
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



