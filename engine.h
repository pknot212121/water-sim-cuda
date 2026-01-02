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
        Engine(const Engine&) = delete;
        Engine& operator=(const Engine&) = delete;

        Engine(Engine&& other) noexcept {
            this->number = other.number;
            this->h_buffer = other.h_buffer;
            this->d_buffer = other.d_buffer;
            this->d_grid_buffer = other.d_grid_buffer;
            this->d_values = other.d_values;
            this->d_cell_offsets = other.d_cell_offsets;
            this->positionsToOpenGL = other.positionsToOpenGL;
            this->blocksPerGrid = other.blocksPerGrid;

            other.h_buffer = nullptr;
            other.d_buffer = nullptr;
            other.d_grid_buffer = nullptr;
            other.d_values = nullptr;
            other.d_cell_offsets = nullptr;
            other.positionsToOpenGL = nullptr;
        }
        ~Engine();
        void step();
        inline Particles getParticles(){ return Particles(d_buffer,number); }
        inline Grid getGrid(){return Grid(d_grid_buffer);};
        inline float3* getPositions(){return positionsToOpenGL;}
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
        float3 *positionsToOpenGL;
        size_t number;
        size_t granularity;
        size_t cellsPerPage;
        size_t blocksPerGrid;
        std::vector<int> activeIndices;
};
