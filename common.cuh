#include <vector_types.h>
#include <crt/host_defines.h>
#include <cuda_bf16.h>


// This is ridiculous
struct __align__(16) Particles
{
    float* posX;float* posY;float* posZ;
    float* velX;float* velY;float* velZ;
    //TODO: change to halfs if possible
    float* c00; float* c01; float* c02;
    float* c10; float* c11; float* c12;
    float* c20; float* c21; float* c22;

    float* f00; float* f01; float* f02;
    float* f10; float* f11; float* f12;
    float* f20; float* f21; float* f22;

    float* V;
    float* m;


    __host__ __device__ Particles(void* bigBuffer)
    {

    }
};
