#pragma once
#include <random>
#include "cuda.h"
#include <stdio.h>
#include "common.cuh"
#include <iostream>

class Engine
{
    public:
        Engine(int n);
        ~Engine();
        void step();
        inline Particles getParticles(){ return Particles(d_buffer,number); }
        Grid getGrid();
        void initParticles();
        CUmemAllocationProp getProp();
        CUmemAccessDesc getDesc();
        void initCuda();
        void initGrid();
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
        size_t gridAlignedSize = GRID_SIZE;
        size_t pageCount;
        size_t blocksPerGrid;
        std::vector<int> activeIndices;

};