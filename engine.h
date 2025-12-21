#pragma once
#include <random>
#include "kernels.h"
#include "common.cuh"



static std::mt19937 seed;

class Engine
{
    public:
    Engine(int n,int X,int Y,int Z);
    void Step();

    private:
    float *h_buffer;
    float* d_buffer;
    std::mt19937 gen;
    Particles particles;
    size_t number;
};