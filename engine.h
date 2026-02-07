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
        Engine() = default;
        ~Engine();
        void init(int n, float *h_buffer);
        void step();
        inline Particles getParticles(){ return Particles(d_buffer,number); }
        inline Grid getGrid(){return Grid(d_grid_buffer,d_sdf_buffer,GameConfigData::getInt("GRID_NUMBER"));};
        inline float3* getPositions(){return positionsToOpenGL;}
        void initParticles();
        void sortParticles();
        void initSDF(std::vector<Triangle> triangles);
        void initGrid();
        inline int getNumber(){return number;}
        inline float* getBuffer(){return d_buffer;}
    private:
        float *h_buffer;
        float *d_buffer;
        float *d_grid_buffer;
        float *d_sdf_buffer;
        int *d_values;
        int *d_cell_offsets;
        float3 *positionsToOpenGL;
        size_t number;
        size_t granularity;
        size_t cellsPerPage;
        size_t blocksPerGrid;
        std::vector<int> activeIndices;
};
