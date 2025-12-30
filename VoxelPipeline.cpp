#include "VoxelPipeline.h"
#include <iostream>

std::vector<float> VoxelPipeline::process(const VoxelData& voxelData)
{
    std::cout << "VoxelPipeline::process called with " << voxelData.count << " voxels" << std::endl;
    std::vector<float> result = {};
    for (int x = 0; x < voxelData.count * 26; x++) {
        if (x < voxelData.count)//pos_x
        {
            result.push_back(voxelData.pos[0][x]);
        }
        else if (x < voxelData.count * 2)//pos_y
        {
            result.push_back(voxelData.pos[1][x - voxelData.count]);
        }
        else if (x < voxelData.count * 3)//pos_z
        {
            result.push_back(voxelData.pos[2][x - voxelData.count * 2]);
        }
        else if (x < voxelData.count * 24)//rest fill with 0
        {
            result.push_back(0.0f);
        }
        else //velocity and mass set to 10
        {
            result.push_back(10.0f);
        }
    }
    std::cout << "VoxelPipeline::process returning buffer of size " << result.size() << std::endl;
    return result;
}

