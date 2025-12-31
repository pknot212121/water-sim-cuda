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
        inline Grid getGrid(){return Grid(d_grid_buffer);};
        void initParticles();
        void sortParticles();
        void initGrid();
        inline int getNumber(){return number;}
        inline float* getBuffer(){return d_buffer;}
    private:
        float *h_buffer;
        float *d_buffer;
        float *d_grid_buffer;
        int *d_values;
        int *d_cell_offsets;
        size_t number;
        size_t granularity;
        size_t cellsPerPage;
        size_t blocksPerGrid;
        std::vector<int> activeIndices;
};
