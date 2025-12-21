#include <vector_types.h>
#include <crt/host_defines.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

constexpr size_t PARTICLE_SIZE = 26;

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


    Particles(){}

    __host__ __device__ void Init(float* buffer,size_t n)
    {
        posX = buffer; posY = buffer + n; posZ = buffer + 2*n;
        velX = buffer +3*n; velY = buffer+4*n; velZ = buffer + 5*n;

        c00 = buffer+4*n; c01=buffer+5*n; c02 = buffer+6*n;
        c10 = buffer+7*n; c11=buffer+8*n; c12=buffer+9*n;
        c20 = buffer+10*n; c21=buffer+11*n; c22=buffer+12*n;

        f00=buffer+13*n; f01=buffer+14*n; f02=buffer+15*n;
        f10=buffer+16*n; f11=buffer+17*n; f12=buffer+18*n;
        f20=buffer+19*n; f21=buffer+20*n; f22=buffer+21*n;

        V = buffer+22*n;
        m = buffer+23*n;
    }
};
