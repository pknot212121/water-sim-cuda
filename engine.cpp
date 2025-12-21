#include "engine.h"

#include <iostream>

Engine::Engine(int n,int X,int Y,int Z)
{
    number = n;
    gen = std::mt19937(seed);
    std::uniform_real_distribution<float> distX(0, X);
    std::uniform_real_distribution<float> distY(0, Y);
    std::uniform_real_distribution<float> distZ(0, Z);
    h_buffer = new float[n * PARTICLE_SIZE];
    for (int i = 0; i < n; i++)
    {
        h_buffer[i] = distX(gen);
        h_buffer[i + PARTICLE_SIZE] = distY(gen);
        h_buffer[i + PARTICLE_SIZE * 2] = distZ(gen);
        h_buffer[i + PARTICLE_SIZE * (PARTICLE_SIZE - 2)] = 10;
        h_buffer[i + PARTICLE_SIZE * (PARTICLE_SIZE - 1)] = 10;
    }
    cudaMalloc((void**)&d_buffer, n * PARTICLE_SIZE);
    cudaMemcpy(d_buffer, h_buffer, n * PARTICLE_SIZE, cudaMemcpyHostToDevice);
    particles.Init(d_buffer, number);
}

void Engine::Step()
{

}
