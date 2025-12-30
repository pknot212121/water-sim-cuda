#pragma once

#include "VoxelEngine.h"

class VoxelPipeline
{
public:
    VoxelPipeline() = default;
    ~VoxelPipeline() = default;

    // Process voxel data and return float buffer
    // You can modify the signature if needed (e.g., add size_t* outSize parameter)
    std::vector<float> process(const VoxelData& voxelData);

private:
    // Add your private helper functions here if needed
};
