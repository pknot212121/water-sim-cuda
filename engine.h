#pragma once
#include <random>
#include "cuda.h"
#include <stdio.h>
#include "common.cuh"
#include <iostream>

#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <thrust/device_ptr.h>



class Engine
{
    public:
        Engine(int n);
        ~Engine();
        void step();
        inline Particles getParticles(){ return Particles(d_buffer,number); }
        inline Particles getParticles_B(){ return Particles(d_buffer_B,number);}
        Grid getGrid();
        void initParticles();
        CUmemAllocationProp getProp();
        CUmemAccessDesc getDesc();
        void initCuda();
        void initGrid();
        void sortParticles();
    private:
        float *h_buffer;
        float* d_buffer;
        float* d_buffer_B;
        std::mt19937 gen;
        size_t number;
        CUdevice device;
        CUcontext context;
        size_t granularity;
        size_t cellsPerPage;
        CUdeviceptr virtPtr;
        std::vector<CUmemGenericAllocationHandle> handles;
        size_t gridAlignedSize = GRID_SIZE;
        size_t pageCount;
        size_t blocksPerGrid;
        std::vector<int> activeIndices;

};