#pragma once
#include <random>
#include "kernels.h"
#include "cuda.h"


class Engine
{
    public:
        Engine(int n);
        ~Engine();
        void Step();
        inline Particles getParticles(){ return Particles(d_buffer,number); }
        void InitParticles();
        void InitCuda();
        void InitGrid();

    private:
        float *h_buffer;
        float* d_buffer;
        std::mt19937 gen;
        size_t number;
        CUdevice device;
        CUcontext context;
        size_t granularity;
        size_t cellsPerPage;
        CUdeviceptr virtPtr;
        std::vector<CUmemGenericAllocationHandle> handles;

};