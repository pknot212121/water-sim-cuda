#pragma once
#include <random>
#include "kernels.h"


class Engine
{
    public:
    Engine(int n,int X,int Y,int Z);
    ~Engine();
    void Step();
    inline Particles getParticles(){ return Particles(d_buffer,number); }

    private:
    float *h_buffer;
    float* d_buffer;
    std::mt19937 gen;
    size_t number;
};