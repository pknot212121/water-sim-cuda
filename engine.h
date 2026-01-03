#pragma once
#include <random>
#include <stdio.h>
#include "common.cuh"
#include <iostream>

#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/counting_iterator.h>



class Engine
{
    public:
        Engine(int n, float *h_buffer);
        ~Engine();
        void step();
        inline Particles getParticles(){ return Particles(d_buffer,number); }
        inline Grid getGrid(){return Grid(d_grid_buffer,sdfTex,sdfBoxMin,sdfBoxMax);};
        inline float3* getPositions(){return positionsToOpenGL;}
        void initParticles();
        void sortParticles();
        void initSDF();
        void initGrid();
        inline int getNumber(){return number;}
        inline float* getBuffer(){return d_buffer;}
    private:
        float *h_buffer;
        float *d_buffer;
        float *d_grid_buffer;
        int *d_values;
        int *d_cell_offsets;
        float3 *positionsToOpenGL;
        cudaArray_t d_sdfArray;
        cudaTextureObject_t sdfTex;
        float3 sdfBoxMin;
        float3 sdfBoxMax;
        size_t number;
        size_t granularity;
        size_t cellsPerPage;
        size_t blocksPerGrid;
        std::vector<int> activeIndices;
};
