#include "engine.h"

#include <iostream>

Engine::Engine(int n,int X,int Y,int Z)
{
    number = n;
    gen = std::mt19937(std::random_device{}());
    std::uniform_real_distribution<float> distX(0, X);
    std::uniform_real_distribution<float> distY(0, Y);
    std::uniform_real_distribution<float> distZ(0, Z);
    h_buffer = new float[n * PARTICLE_SIZE];
    for (int i = 0; i < n; i++)
    {
        h_buffer[i] = distX(gen);
        h_buffer[i + n] = distY(gen);
        h_buffer[i + n * 2] = distZ(gen);
        h_buffer[i + n * (PARTICLE_SIZE - 2)] = 10;
        h_buffer[i + n * (PARTICLE_SIZE - 1)] = 10;
    }
    cudaError_t err = cudaMalloc((void**)&d_buffer, n * PARTICLE_SIZE * sizeof(float));
    if (err != cudaSuccess)
    {
        std::cerr << "Błąd alokacji: " << cudaGetErrorString(err) << std::endl;
        exit(1);
    }
    cudaMemcpy(d_buffer, h_buffer, n * PARTICLE_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    Particles p = getParticles();
    summonTestKernel(number);
    cudaDeviceSynchronize();
}

Engine::~Engine()
{
    delete[] h_buffer;
    cudaFree(d_buffer);
}

void Engine::Step()
{

}
