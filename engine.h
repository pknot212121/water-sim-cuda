#pragma once
#include <random>

static std::random_device seed = std::random_device();

class Engine
{
    public:
    void Init(int n,int X,int Y,int Z);
    void Step();

    private:
    void *buffer;
    std::mt19937 gen;
};