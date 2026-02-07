#include "VoxelEngine.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <set>
#include <map>
#include <tuple>
#include <glm/vec3.hpp>


#include "common.cuh"
#include "game_configdata.h"

VoxelData::VoxelData() : count(0), resolution(0.0f)
{
    pos[0] = nullptr;
    pos[1] = nullptr;
    pos[2] = nullptr;
    boundingBoxMin = {0.0f, 0.0f, 0.0f};
    boundingBoxMax = {0.0f, 0.0f, 0.0f};
}

VoxelData::VoxelData(const VoxelData& other) : count(other.count), resolution(other.resolution),
                                    boundingBoxMin(other.boundingBoxMin), boundingBoxMax(other.boundingBoxMax)
{
    if (count > 0)
    {
        pos[0] = new float[count];
        pos[1] = new float[count];
        pos[2] = new float[count];
        std::copy(other.pos[0], other.pos[0] + count, pos[0]);
        std::copy(other.pos[1], other.pos[1] + count, pos[1]);
        std::copy(other.pos[2], other.pos[2] + count, pos[2]);
    }
    else
    {
        pos[0] = nullptr;
        pos[1] = nullptr;
        pos[2] = nullptr;
    }
}

VoxelData::VoxelData(VoxelData&& other) noexcept : count(other.count), resolution(other.resolution),
                                         boundingBoxMin(other.boundingBoxMin), boundingBoxMax(other.boundingBoxMax)
{
    pos[0] = other.pos[0];
    pos[1] = other.pos[1];
    pos[2] = other.pos[2];

    other.pos[0] = nullptr;
    other.pos[1] = nullptr;
    other.pos[2] = nullptr;
    other.count = 0;
}

VoxelData& VoxelData::operator=(const VoxelData& other)
{
    if (this != &other)
    {
        if (pos[0]) delete[] pos[0];
        if (pos[1]) delete[] pos[1];
        if (pos[2]) delete[] pos[2];

        count = other.count;
        resolution = other.resolution;
        boundingBoxMin = other.boundingBoxMin;
        boundingBoxMax = other.boundingBoxMax;

        if (count > 0)
        {
            pos[0] = new float[count];
            pos[1] = new float[count];
            pos[2] = new float[count];
            std::copy(other.pos[0], other.pos[0] + count, pos[0]);
            std::copy(other.pos[1], other.pos[1] + count, pos[1]);
            std::copy(other.pos[2], other.pos[2] + count, pos[2]);
        }
        else
        {
            pos[0] = nullptr;
            pos[1] = nullptr;
            pos[2] = nullptr;
        }
    }
    return *this;
}

VoxelData& VoxelData::operator=(VoxelData&& other) noexcept
{
    if (this != &other)
    {
        if (pos[0]) delete[] pos[0];
        if (pos[1]) delete[] pos[1];
        if (pos[2]) delete[] pos[2];

        count = other.count;
        resolution = other.resolution;
        boundingBoxMin = other.boundingBoxMin;
        boundingBoxMax = other.boundingBoxMax;
        pos[0] = other.pos[0];
        pos[1] = other.pos[1];
        pos[2] = other.pos[2];

        other.pos[0] = nullptr;
        other.pos[1] = nullptr;
        other.pos[2] = nullptr;
        other.count = 0;
    }
    return *this;
}

VoxelData::~VoxelData()
{
    if (pos[0]) delete[] pos[0];
    if (pos[1]) delete[] pos[1];
    if (pos[2]) delete[] pos[2];
}


VoxelData VoxelEngine::voxelize(const ObjData& objData, float resolution)
{
    VoxelData voxelData;

    if (!objData.success || objData.attrib.vertices.empty())
    {
        std::cerr << "VoxelEngine: Invalid ObjData provided" << std::endl;
        return voxelData;
    }

    std::cout << "Starting voxelization with resolution: " << resolution << std::endl;

    BoundingBox bbox = calculateBoundingBox(objData);
    voxelData.boundingBoxMin = bbox.min;
    voxelData.boundingBoxMax = bbox.max;
    voxelData.resolution = resolution;

    std::cout << "Bounding box: (" << bbox.min.x << ", " << bbox.min.y << ", " << bbox.min.z << ") to ("
              << bbox.max.x << ", " << bbox.max.y << ", " << bbox.max.z << ")" << std::endl;

    float sizeX = bbox.max.x - bbox.min.x;
    float sizeY = bbox.max.y - bbox.min.y;
    float sizeZ = bbox.max.z - bbox.min.z;
    float maxsize = std::max(sizeX, std::max(sizeY, sizeZ));
    resolution = maxsize / resolution;

    int gridX = (int)std::ceil((bbox.max.x - bbox.min.x) / resolution);
    int gridY = (int)std::ceil((bbox.max.y - bbox.min.y) / resolution);
    int gridZ = (int)std::ceil((bbox.max.z - bbox.min.z) / resolution);

    std::cout << "Grid dimensions: " << gridX << " x " << gridY << " x " << gridZ << std::endl;

    std::vector<Triangle> triangles = extractTriangles(objData);
    std::cout << "Number of triangles: " << triangles.size() << std::endl;

    size_t totalVoxels = (size_t)gridX * gridY * gridZ;
    std::vector<bool> filled(totalVoxels, false);
    std::cout << "Grid total voxels: " << totalVoxels << std::endl;

    for (size_t triIdx = 0; triIdx < triangles.size(); triIdx++)
    {
        const Triangle& tri = triangles[triIdx];

        float triMinX = min3(tri.v0.x, tri.v1.x, tri.v2.x);
        float triMinY = min3(tri.v0.y, tri.v1.y, tri.v2.y);
        float triMinZ = min3(tri.v0.z, tri.v1.z, tri.v2.z);
        float triMaxX = max3(tri.v0.x, tri.v1.x, tri.v2.x);
        float triMaxY = max3(tri.v0.y, tri.v1.y, tri.v2.y);
        float triMaxZ = max3(tri.v0.z, tri.v1.z, tri.v2.z);

        int minGridX = std::max(0, (int)std::floor((triMinX - bbox.min.x) / resolution));
        int minGridY = std::max(0, (int)std::floor((triMinY - bbox.min.y) / resolution));
        int minGridZ = std::max(0, (int)std::floor((triMinZ - bbox.min.z) / resolution));
        int maxGridX = std::min(gridX - 1, (int)std::ceil((triMaxX - bbox.min.x) / resolution));
        int maxGridY = std::min(gridY - 1, (int)std::ceil((triMaxY - bbox.min.y) / resolution));
        int maxGridZ = std::min(gridZ - 1, (int)std::ceil((triMaxZ - bbox.min.z) / resolution));

        for (int z = minGridZ; z <= maxGridZ; z++)
        {
            for (int y = minGridY; y <= maxGridY; y++)
            {
                for (int x = minGridX; x <= maxGridX; x++)
                {
                    float3 voxelMin, voxelMax;
                    voxelMin.x = bbox.min.x + x * resolution;
                    voxelMin.y = bbox.min.y + y * resolution;
                    voxelMin.z = bbox.min.z + z * resolution;
                    voxelMax.x = voxelMin.x + resolution;
                    voxelMax.y = voxelMin.y + resolution;
                    voxelMax.z = voxelMin.z + resolution;

                    if (triangleAABBIntersection(tri, voxelMin, voxelMax))
                    {
                        size_t voxelIndex = (size_t)z * gridY * gridX + (size_t)y * gridX + (size_t)x;
                        filled[voxelIndex] = true;
                    }
                }
            }
        }

        if ((triIdx + 1) % 10000 == 0 || triIdx == triangles.size() - 1)
        {
            std::cout << "Processed " << (triIdx + 1) << "/" << triangles.size() << " triangles" << std::endl;
        }
    }

    std::vector<float> voxelX, voxelY, voxelZ;
    for (int z = 0; z < gridZ; z++)
    {
        for (int y = 0; y < gridY; y++)
        {
            for (int x = 0; x < gridX; x++)
            {
                size_t voxelIndex = (size_t)z * gridY * gridX + (size_t)y * gridX + (size_t)x;
                if (filled[voxelIndex])
                {
                    float voxelCenterX = bbox.min.x + (x + 0.5f) * resolution;
                    float voxelCenterY = bbox.min.y + (y + 0.5f) * resolution;
                    float voxelCenterZ = bbox.min.z + (z + 0.5f) * resolution;

                    voxelX.push_back(voxelCenterX);
                    voxelY.push_back(voxelCenterY);
                    voxelZ.push_back(voxelCenterZ);
                }
            }
        }
    }

    voxelData.count = voxelX.size();
    std::cout << "Generated " << voxelData.count << " voxels" << std::endl;

    if (voxelData.count > 0)
    {
        voxelData.pos[0] = new float[voxelData.count];
        voxelData.pos[1] = new float[voxelData.count];
        voxelData.pos[2] = new float[voxelData.count];

        std::copy(voxelX.begin(), voxelX.end(), voxelData.pos[0]);
        std::copy(voxelY.begin(), voxelY.end(), voxelData.pos[1]);
        std::copy(voxelZ.begin(), voxelZ.end(), voxelData.pos[2]);
    }

    return voxelData;
}

void VoxelEngine::normalize(VoxelData& data, float normalizeSize, float scale, const float3& displacement)
{
    if (data.count == 0)
    {
        std::cerr << "VoxelEngine::normalize: No voxels to normalize" << std::endl;
        return;
    }

    std::cout << "Normalizing " << data.count << " voxels..." << std::endl;
    std::cout << "  Normalize size: " << normalizeSize << std::endl;
    std::cout << "  Scale factor: " << scale << std::endl;
    std::cout << "  Displacement: (" << displacement.x << ", " << displacement.y << ", " << displacement.z << ")" << std::endl;

    float minX = data.pos[0][0], maxX = data.pos[0][0];
    float minY = data.pos[1][0], maxY = data.pos[1][0];
    float minZ = data.pos[2][0], maxZ = data.pos[2][0];

    for (size_t i = 0; i < data.count; i++)
    {
        minX = std::min(minX, data.pos[0][i]);
        maxX = std::max(maxX, data.pos[0][i]);
        minY = std::min(minY, data.pos[1][i]);
        maxY = std::max(maxY, data.pos[1][i]);
        minZ = std::min(minZ, data.pos[2][i]);
        maxZ = std::max(maxZ, data.pos[2][i]);
    }

    std::cout << "  Current bounds: (" << minX << ", " << minY << ", " << minZ << ") to ("
              << maxX << ", " << maxY << ", " << maxZ << ")" << std::endl;

    float sizeX = maxX - minX;
    float sizeY = maxY - minY;
    float sizeZ = maxZ - minZ;
    float maxDimension = max3(sizeX, sizeY, sizeZ);

    if (maxDimension < 1e-6f)
    {
        std::cerr << "VoxelEngine::normalize: Voxel data has zero size" << std::endl;
        return;
    }

    std::cout << "  Current size: (" << sizeX << ", " << sizeY << ", " << sizeZ << ")" << std::endl;
    std::cout << "  Max dimension: " << maxDimension << std::endl;

    float normalizationScale = normalizeSize / maxDimension;
    float3 currentCenter = {
        (minX + maxX) * 0.5f,
        (minY + maxY) * 0.5f,
        (minZ + maxZ) * 0.5f
    };

    for (size_t i = 0; i < data.count; i++)
    {
        data.pos[0][i] -= currentCenter.x;
        data.pos[1][i] -= currentCenter.y;
        data.pos[2][i] -= currentCenter.z;

        data.pos[0][i] *= normalizationScale;
        data.pos[1][i] *= normalizationScale;
        data.pos[2][i] *= normalizationScale;

        data.pos[0][i] += normalizeSize * 0.5f;
        data.pos[1][i] += normalizeSize * 0.5f;
        data.pos[2][i] += normalizeSize * 0.5f;
    }

    std::cout << "  After normalization, voxels are centered at ("
              << normalizeSize * 0.5f << ", "
              << normalizeSize * 0.5f << ", "
              << normalizeSize * 0.5f << ")" << std::endl;

    float currentMinX = data.pos[0][0], currentMaxX = data.pos[0][0];
    float currentMinY = data.pos[1][0], currentMaxY = data.pos[1][0];
    float currentMinZ = data.pos[2][0], currentMaxZ = data.pos[2][0];

    for (size_t i = 0; i < data.count; i++)
    {
        currentMinX = std::min(currentMinX, data.pos[0][i]);
        currentMaxX = std::max(currentMaxX, data.pos[0][i]);
        currentMinY = std::min(currentMinY, data.pos[1][i]);
        currentMaxY = std::max(currentMaxY, data.pos[1][i]);
        currentMinZ = std::min(currentMinZ, data.pos[2][i]);
        currentMaxZ = std::max(currentMaxZ, data.pos[2][i]);
    }

    float currentSizeX = currentMaxX - currentMinX;
    float currentSizeY = currentMaxY - currentMinY;
    float currentSizeZ = currentMaxZ - currentMinZ;
    float currentMaxDim = max3(currentSizeX, currentSizeY, currentSizeZ);

    float targetMaxDimension = scale;
    float scaleFactorToTarget = (currentMaxDim > 1e-6f) ? (targetMaxDimension / currentMaxDim) : 1.0f;

    float3 scaleCenter = {normalizeSize * 0.5f, normalizeSize * 0.5f, normalizeSize * 0.5f};

    for (size_t i = 0; i < data.count; i++)
    {
        data.pos[0][i] -= scaleCenter.x;
        data.pos[1][i] -= scaleCenter.y;
        data.pos[2][i] -= scaleCenter.z;

        data.pos[0][i] *= scaleFactorToTarget;
        data.pos[1][i] *= scaleFactorToTarget;
        data.pos[2][i] *= scaleFactorToTarget;

        data.pos[0][i] += scaleCenter.x;
        data.pos[1][i] += scaleCenter.y;
        data.pos[2][i] += scaleCenter.z;
    }

    std::cout << "  After scaling: longest dimension = " << targetMaxDimension
              << " (scale factor: " << scaleFactorToTarget << "x)" << std::endl;

    for (size_t i = 0; i < data.count; i++)
    {
        data.pos[0][i] += displacement.x;
        data.pos[1][i] += displacement.y;
        data.pos[2][i] += displacement.z;
    }

    std::cout << "  After displacement" << std::endl;

    std::cout << "  Snapping voxels to discrete grid (RESOLUTION) with 5x5x5 expansion and clamping to [0, " << normalizeSize << "]..." << std::endl;

    const float gridResolution = GameConfigData::getInt("RESOLUTION");
    int maxGridIndex = (int)(normalizeSize / gridResolution);
    const int expansionRadius = 2;

    std::set<std::tuple<int, int, int>> uniqueGridPositions;
    size_t voxelsClamped = 0;
    size_t voxelsOutOfBounds = 0;

    for (size_t i = 0; i < data.count; i++)
    {
        int gridX = (int)std::round(data.pos[0][i] / gridResolution);
        int gridY = (int)std::round(data.pos[1][i] / gridResolution);
        int gridZ = (int)std::round(data.pos[2][i] / gridResolution);

        bool wasClamped = false;
        if (gridX < 0 || gridX > maxGridIndex ||
            gridY < 0 || gridY > maxGridIndex ||
            gridZ < 0 || gridZ > maxGridIndex)
        {
            wasClamped = true;
            voxelsOutOfBounds++;
        }

        gridX = std::max(0, std::min(maxGridIndex, gridX));
        gridY = std::max(0, std::min(maxGridIndex, gridY));
        gridZ = std::max(0, std::min(maxGridIndex, gridZ));

        if (wasClamped) voxelsClamped++;

        for (int dx = -expansionRadius; dx <= expansionRadius; dx++)
        {
            for (int dy = -expansionRadius; dy <= expansionRadius; dy++)
            {
                for (int dz = -expansionRadius; dz <= expansionRadius; dz++)
                {
                    int newX = gridX + dx;
                    int newY = gridY + dy;
                    int newZ = gridZ + dz;

                    if (newX >= 0 && newX <= maxGridIndex &&
                        newY >= 0 && newY <= maxGridIndex &&
                        newZ >= 0 && newZ <= maxGridIndex)
                    {
                        uniqueGridPositions.insert(std::make_tuple(newX, newY, newZ));
                    }
                }
            }
        }
    }

    std::cout << "  Original voxel count: " << data.count << std::endl;
    if (voxelsOutOfBounds > 0)
    {
        std::cout << "  Voxels outside grid bounds [0, " << normalizeSize << "]: " << voxelsOutOfBounds
                  << " (" << (100.0f * voxelsOutOfBounds / data.count) << "%)" << std::endl;
        std::cout << "  These voxels were clamped to grid boundaries" << std::endl;
    }
    std::cout << "  After 5x5x5 expansion: " << uniqueGridPositions.size() << " unique grid positions" << std::endl;
    std::cout << "  Expansion factor: " << ((float)uniqueGridPositions.size() / data.count) << "x" << std::endl;

    std::vector<float> snappedX, snappedY, snappedZ;
    for (const auto& gridPos : uniqueGridPositions)
    {
        snappedX.push_back(std::get<0>(gridPos) * gridResolution);
        snappedY.push_back(std::get<1>(gridPos) * gridResolution);
        snappedZ.push_back(std::get<2>(gridPos) * gridResolution);
    }

    delete[] data.pos[0];
    delete[] data.pos[1];
    delete[] data.pos[2];

    data.count = snappedX.size();
    if (data.count > 0)
    {
        data.pos[0] = new float[data.count];
        data.pos[1] = new float[data.count];
        data.pos[2] = new float[data.count];

        std::copy(snappedX.begin(), snappedX.end(), data.pos[0]);
        std::copy(snappedY.begin(), snappedY.end(), data.pos[1]);
        std::copy(snappedZ.begin(), snappedZ.end(), data.pos[2]);

        minX = maxX = data.pos[0][0];
        minY = maxY = data.pos[1][0];
        minZ = maxZ = data.pos[2][0];

        for (size_t i = 0; i < data.count; i++)
        {
            minX = std::min(minX, data.pos[0][i]);
            maxX = std::max(maxX, data.pos[0][i]);
            minY = std::min(minY, data.pos[1][i]);
            maxY = std::max(maxY, data.pos[1][i]);
            minZ = std::min(minZ, data.pos[2][i]);
            maxZ = std::max(maxZ, data.pos[2][i]);
        }

        data.boundingBoxMin = {minX, minY, minZ};
        data.boundingBoxMax = {maxX, maxY, maxZ};

        std::cout << "  Final snapped bounds: (" << minX << ", " << minY << ", " << minZ << ") to ("
                  << maxX << ", " << maxY << ", " << maxZ << ")" << std::endl;

        std::cout << "\n  Sample voxels after grid snapping:" << std::endl;
        size_t samplesToShow = std::min((size_t)10, data.count);
        for (size_t i = 0; i < samplesToShow; i++)
        {
            std::cout << "    Voxel " << i << ": ("
                      << data.pos[0][i] << ", "
                      << data.pos[1][i] << ", "
                      << data.pos[2][i] << ")" << std::endl;
        }
        if (data.count > samplesToShow)
        {
            std::cout << "    ... and " << (data.count - samplesToShow) << " more voxels" << std::endl;
        }
    }
    else
    {
        data.pos[0] = nullptr;
        data.pos[1] = nullptr;
        data.pos[2] = nullptr;
    }

    std::cout << "Normalization complete!" << std::endl;
}

void VoxelEngine::expandVoxels(VoxelData& data, int expansionRadius)
{
    if (data.count == 0)
    {
        std::cerr << "VoxelEngine::expandVoxels: No voxels to expand" << std::endl;
        return;
    }

    std::cout << "Expanding voxels with radius " << expansionRadius
              << " (" << (2*expansionRadius+1) << "x" << (2*expansionRadius+1) << "x" << (2*expansionRadius+1)
              << " per voxel)..." << std::endl;

    const float gridResolution = 0.1f;
    int maxGridIndex = (GameConfigData::getInt("SIZE_X") * 10);

    std::set<std::tuple<int, int, int>> uniqueGridPositions;

    for (size_t i = 0; i < data.count; i++)
    {
        int gridX = (int)std::round(data.pos[0][i] / gridResolution);
        int gridY = (int)std::round(data.pos[1][i] / gridResolution);
        int gridZ = (int)std::round(data.pos[2][i] / gridResolution);

        for (int dx = -expansionRadius; dx <= expansionRadius; dx++)
        {
            for (int dy = -expansionRadius; dy <= expansionRadius; dy++)
            {
                for (int dz = -expansionRadius; dz <= expansionRadius; dz++)
                {
                    int newX = gridX + dx;
                    int newY = gridY + dy;
                    int newZ = gridZ + dz;

                    if (newX >= 0 && newX <= maxGridIndex &&
                        newY >= 0 && newY <= maxGridIndex &&
                        newZ >= 0 && newZ <= maxGridIndex)
                    {
                        uniqueGridPositions.insert(std::make_tuple(newX, newY, newZ));
                    }
                }
            }
        }
    }

    std::cout << "  Original voxel count: " << data.count << std::endl;
    std::cout << "  Expanded voxel count: " << uniqueGridPositions.size() << std::endl;
    std::cout << "  Expansion factor: " << (float)uniqueGridPositions.size() / data.count << "x" << std::endl;

    std::vector<float> expandedX, expandedY, expandedZ;
    expandedX.reserve(uniqueGridPositions.size());
    expandedY.reserve(uniqueGridPositions.size());
    expandedZ.reserve(uniqueGridPositions.size());

    for (const auto& gridPos : uniqueGridPositions)
    {
        expandedX.push_back(std::get<0>(gridPos) * gridResolution);
        expandedY.push_back(std::get<1>(gridPos) * gridResolution);
        expandedZ.push_back(std::get<2>(gridPos) * gridResolution);
    }

    delete[] data.pos[0];
    delete[] data.pos[1];
    delete[] data.pos[2];

    data.count = expandedX.size();
    if (data.count > 0)
    {
        data.pos[0] = new float[data.count];
        data.pos[1] = new float[data.count];
        data.pos[2] = new float[data.count];

        std::copy(expandedX.begin(), expandedX.end(), data.pos[0]);
        std::copy(expandedY.begin(), expandedY.end(), data.pos[1]);
        std::copy(expandedZ.begin(), expandedZ.end(), data.pos[2]);

        float minX = data.pos[0][0], maxX = data.pos[0][0];
        float minY = data.pos[1][0], maxY = data.pos[1][0];
        float minZ = data.pos[2][0], maxZ = data.pos[2][0];

        for (size_t i = 1; i < data.count; i++)
        {
            minX = std::min(minX, data.pos[0][i]);
            maxX = std::max(maxX, data.pos[0][i]);
            minY = std::min(minY, data.pos[1][i]);
            maxY = std::max(maxY, data.pos[1][i]);
            minZ = std::min(minZ, data.pos[2][i]);
            maxZ = std::max(maxZ, data.pos[2][i]);
        }

        data.boundingBoxMin = {minX, minY, minZ};
        data.boundingBoxMax = {maxX, maxY, maxZ};

        std::cout << "  Expanded bounds: (" << minX << ", " << minY << ", " << minZ << ") to ("
                  << maxX << ", " << maxY << ", " << maxZ << ")" << std::endl;
    }
    else
    {
        data.pos[0] = nullptr;
        data.pos[1] = nullptr;
        data.pos[2] = nullptr;
    }

    std::cout << "Expansion complete!" << std::endl;
}

void VoxelEngine::normalize(std::vector<Triangle>& triangles, float normalizeSize, float scale, const float3& displacement)
{
    if (triangles.empty())
    {
        std::cerr << "VoxelEngine::normalize: No triangles to normalize" << std::endl;
        return;
    }

    std::cout << "Normalizing " << triangles.size() << " triangles..." << std::endl;
    std::cout << "  Normalize size: " << normalizeSize << std::endl;
    std::cout << "  Scale factor: " << scale << std::endl;
    std::cout << "  Displacement: (" << displacement.x << ", " << displacement.y << ", " << displacement.z << ")" << std::endl;

    float minX = triangles[0].v0.x, maxX = triangles[0].v0.x;
    float minY = triangles[0].v0.y, maxY = triangles[0].v0.y;
    float minZ = triangles[0].v0.z, maxZ = triangles[0].v0.z;

    for (const auto& tri : triangles)
    {
        minX = min3(minX, tri.v0.x, tri.v1.x);
        minX = std::min(minX, tri.v2.x);
        maxX = max3(maxX, tri.v0.x, tri.v1.x);
        maxX = std::max(maxX, tri.v2.x);

        minY = min3(minY, tri.v0.y, tri.v1.y);
        minY = std::min(minY, tri.v2.y);
        maxY = max3(maxY, tri.v0.y, tri.v1.y);
        maxY = std::max(maxY, tri.v2.y);

        minZ = min3(minZ, tri.v0.z, tri.v1.z);
        minZ = std::min(minZ, tri.v2.z);
        maxZ = max3(maxZ, tri.v0.z, tri.v1.z);
        maxZ = std::max(maxZ, tri.v2.z);
    }

    std::cout << "  Current bounds: (" << minX << ", " << minY << ", " << minZ << ") to ("
              << maxX << ", " << maxY << ", " << maxZ << ")" << std::endl;

    float sizeX = maxX - minX;
    float sizeY = maxY - minY;
    float sizeZ = maxZ - minZ;
    float maxDimension = max3(sizeX, sizeY, sizeZ);

    if (maxDimension < 1e-6f)
    {
        std::cerr << "VoxelEngine::normalize: Triangle data has zero size" << std::endl;
        return;
    }

    std::cout << "  Current size: (" << sizeX << ", " << sizeY << ", " << sizeZ << ")" << std::endl;
    std::cout << "  Max dimension: " << maxDimension << std::endl;

    float normalizationScale = normalizeSize / maxDimension;
    float3 currentCenter = {
        (minX + maxX) * 0.5f,
        (minY + maxY) * 0.5f,
        (minZ + maxZ) * 0.5f
    };

    auto transformVertex = [&](float3& vertex) {
        vertex.x -= currentCenter.x;
        vertex.y -= currentCenter.y;
        vertex.z -= currentCenter.z;

        vertex.x *= normalizationScale;
        vertex.y *= normalizationScale;
        vertex.z *= normalizationScale;

        vertex.x += normalizeSize * 0.5f;
        vertex.y += normalizeSize * 0.5f;
        vertex.z += normalizeSize * 0.5f;
    };

    for (auto& tri : triangles)
    {
        transformVertex(tri.v0);
        transformVertex(tri.v1);
        transformVertex(tri.v2);
    }

    std::cout << "  After normalization, triangles are centered at ("
              << normalizeSize * 0.5f << ", "
              << normalizeSize * 0.5f << ", "
              << normalizeSize * 0.5f << ")" << std::endl;

    float currentMinX = triangles[0].v0.x, currentMaxX = triangles[0].v0.x;
    float currentMinY = triangles[0].v0.y, currentMaxY = triangles[0].v0.y;
    float currentMinZ = triangles[0].v0.z, currentMaxZ = triangles[0].v0.z;

    for (const auto& tri : triangles)
    {
        currentMinX = min3(currentMinX, tri.v0.x, tri.v1.x);
        currentMinX = std::min(currentMinX, tri.v2.x);
        currentMaxX = max3(currentMaxX, tri.v0.x, tri.v1.x);
        currentMaxX = std::max(currentMaxX, tri.v2.x);

        currentMinY = min3(currentMinY, tri.v0.y, tri.v1.y);
        currentMinY = std::min(currentMinY, tri.v2.y);
        currentMaxY = max3(currentMaxY, tri.v0.y, tri.v1.y);
        currentMaxY = std::max(currentMaxY, tri.v2.y);

        currentMinZ = min3(currentMinZ, tri.v0.z, tri.v1.z);
        currentMinZ = std::min(currentMinZ, tri.v2.z);
        currentMaxZ = max3(currentMaxZ, tri.v0.z, tri.v1.z);
        currentMaxZ = std::max(currentMaxZ, tri.v2.z);
    }

    float currentSizeX = currentMaxX - currentMinX;
    float currentSizeY = currentMaxY - currentMinY;
    float currentSizeZ = currentMaxZ - currentMinZ;
    float currentMaxDim = max3(currentSizeX, currentSizeY, currentSizeZ);

    float targetMaxDimension = scale;
    float scaleFactorToTarget = (currentMaxDim > 1e-6f) ? (targetMaxDimension / currentMaxDim) : 1.0f;

    float3 scaleCenter = {normalizeSize * 0.5f, normalizeSize * 0.5f, normalizeSize * 0.5f};

    auto scaleVertex = [&](float3& vertex) {
        vertex.x -= scaleCenter.x;
        vertex.y -= scaleCenter.y;
        vertex.z -= scaleCenter.z;

        vertex.x *= scaleFactorToTarget;
        vertex.y *= scaleFactorToTarget;
        vertex.z *= scaleFactorToTarget;

        vertex.x += scaleCenter.x;
        vertex.y += scaleCenter.y;
        vertex.z += scaleCenter.z;
    };

    for (auto& tri : triangles)
    {
        scaleVertex(tri.v0);
        scaleVertex(tri.v1);
        scaleVertex(tri.v2);
    }

    std::cout << "  After scaling: longest dimension = " << targetMaxDimension
              << " (scale factor: " << scaleFactorToTarget << "x)" << std::endl;

    for (auto& tri : triangles)
    {
        tri.v0.x += displacement.x;
        tri.v0.y += displacement.y;
        tri.v0.z += displacement.z;

        tri.v1.x += displacement.x;
        tri.v1.y += displacement.y;
        tri.v1.z += displacement.z;

        tri.v2.x += displacement.x;
        tri.v2.y += displacement.y;
        tri.v2.z += displacement.z;
    }

    std::cout << "  After displacement" << std::endl;

    std::vector<Triangle> validTriangles;
    size_t removedCount = 0;

    for (const auto& tri : triangles)
    {
        bool v0Valid = (tri.v0.x >= 0.0f && tri.v0.x <= normalizeSize &&
                        tri.v0.y >= 0.0f && tri.v0.y <= normalizeSize &&
                        tri.v0.z >= 0.0f && tri.v0.z <= normalizeSize);

        bool v1Valid = (tri.v1.x >= 0.0f && tri.v1.x <= normalizeSize &&
                        tri.v1.y >= 0.0f && tri.v1.y <= normalizeSize &&
                        tri.v1.z >= 0.0f && tri.v1.z <= normalizeSize);

        bool v2Valid = (tri.v2.x >= 0.0f && tri.v2.x <= normalizeSize &&
                        tri.v2.y >= 0.0f && tri.v2.y <= normalizeSize &&
                        tri.v2.z >= 0.0f && tri.v2.z <= normalizeSize);

        if (v0Valid || v1Valid || v2Valid)
        {
            validTriangles.push_back(tri);
        }
        else
        {
            removedCount++;
        }
    }

    if (validTriangles.size() != triangles.size())
    {
        std::cout << "  Removed " << removedCount << " triangles completely outside bounds [0, " << normalizeSize << "]" << std::endl;
        std::cout << "  Remaining triangles: " << validTriangles.size() << std::endl;

        triangles = std::move(validTriangles);
    }

    if (!triangles.empty())
    {
        minX = maxX = triangles[0].v0.x;
        minY = maxY = triangles[0].v0.y;
        minZ = maxZ = triangles[0].v0.z;

        for (const auto& tri : triangles)
        {
            minX = min3(minX, tri.v0.x, tri.v1.x);
            minX = std::min(minX, tri.v2.x);
            maxX = max3(maxX, tri.v0.x, tri.v1.x);
            maxX = std::max(maxX, tri.v2.x);

            minY = min3(minY, tri.v0.y, tri.v1.y);
            minY = std::min(minY, tri.v2.y);
            maxY = max3(maxY, tri.v0.y, tri.v1.y);
            maxY = std::max(maxY, tri.v2.y);

            minZ = min3(minZ, tri.v0.z, tri.v1.z);
            minZ = std::min(minZ, tri.v2.z);
            maxZ = max3(maxZ, tri.v0.z, tri.v1.z);
            maxZ = std::max(maxZ, tri.v2.z);
        }

        std::cout << "  Final bounds: (" << minX << ", " << minY << ", " << minZ << ") to ("
                  << maxX << ", " << maxY << ", " << maxZ << ")" << std::endl;
    }
    else
    {
        std::cout << "  Warning: All triangles were removed!" << std::endl;
    }

    std::cout << "Triangle normalization complete!" << std::endl;
}

VoxelEngine::BoundingBox VoxelEngine::calculateBoundingBox(const ObjData& objData)
{
    BoundingBox bbox;

    if (objData.attrib.vertices.empty())
    {
        bbox.min = {0.0f, 0.0f, 0.0f};
        bbox.max = {0.0f, 0.0f, 0.0f};
        return bbox;
    }

    bbox.min.x = bbox.max.x = objData.attrib.vertices[0];
    bbox.min.y = bbox.max.y = objData.attrib.vertices[1];
    bbox.min.z = bbox.max.z = objData.attrib.vertices[2];

    for (size_t i = 0; i < objData.attrib.vertices.size(); i += 3)
    {
        bbox.min.x = std::min(bbox.min.x, objData.attrib.vertices[i + 0]);
        bbox.min.y = std::min(bbox.min.y, objData.attrib.vertices[i + 1]);
        bbox.min.z = std::min(bbox.min.z, objData.attrib.vertices[i + 2]);

        bbox.max.x = std::max(bbox.max.x, objData.attrib.vertices[i + 0]);
        bbox.max.y = std::max(bbox.max.y, objData.attrib.vertices[i + 1]);
        bbox.max.z = std::max(bbox.max.z, objData.attrib.vertices[i + 2]);
    }

    return bbox;
}

std::vector<Triangle> VoxelEngine::extractTriangles(const ObjData& objData)
{
    std::vector<Triangle> triangles;

    for (const auto& shape : objData.shapes)
    {
        size_t indexOffset = 0;

        for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++)
        {
            int fv = shape.mesh.num_face_vertices[f];

            if (fv == 3)
            {
                Triangle tri;

                tinyobj::index_t idx0 = shape.mesh.indices[indexOffset + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[indexOffset + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[indexOffset + 2];

                tri.v0.x = objData.attrib.vertices[3 * idx0.vertex_index + 0];
                tri.v0.y = objData.attrib.vertices[3 * idx0.vertex_index + 1];
                tri.v0.z = objData.attrib.vertices[3 * idx0.vertex_index + 2];

                tri.v1.x = objData.attrib.vertices[3 * idx1.vertex_index + 0];
                tri.v1.y = objData.attrib.vertices[3 * idx1.vertex_index + 1];
                tri.v1.z = objData.attrib.vertices[3 * idx1.vertex_index + 2];

                tri.v2.x = objData.attrib.vertices[3 * idx2.vertex_index + 0];
                tri.v2.y = objData.attrib.vertices[3 * idx2.vertex_index + 1];
                tri.v2.z = objData.attrib.vertices[3 * idx2.vertex_index + 2];

                triangles.push_back(tri);
            }

            indexOffset += fv;
        }
    }

    return triangles;
}



bool VoxelEngine::triangleAABBIntersection(const Triangle& tri, const float3& boxMin, const float3& boxMax)
{
    float3 boxCenter;
    boxCenter.x = (boxMin.x + boxMax.x) * 0.5f;
    boxCenter.y = (boxMin.y + boxMax.y) * 0.5f;
    boxCenter.z = (boxMin.z + boxMax.z) * 0.5f;

    float3 boxHalfSize;
    boxHalfSize.x = (boxMax.x - boxMin.x) * 0.5f;
    boxHalfSize.y = (boxMax.y - boxMin.y) * 0.5f;
    boxHalfSize.z = (boxMax.z - boxMin.z) * 0.5f;

    float3 v0 = subtract(tri.v0, boxCenter);
    float3 v1 = subtract(tri.v1, boxCenter);
    float3 v2 = subtract(tri.v2, boxCenter);

    float3 e0 = subtract(v1, v0);
    float3 e1 = subtract(v2, v1);
    float3 e2 = subtract(v0, v2);

    float3 axes[9];
    axes[0] = {0, -e0.z, e0.y};
    axes[1] = {e0.z, 0, -e0.x};
    axes[2] = {-e0.y, e0.x, 0};
    axes[3] = {0, -e1.z, e1.y};
    axes[4] = {e1.z, 0, -e1.x};
    axes[5] = {-e1.y, e1.x, 0};
    axes[6] = {0, -e2.z, e2.y};
    axes[7] = {e2.z, 0, -e2.x};
    axes[8] = {-e2.y, e2.x, 0};

    for (int i = 0; i < 9; i++)
    {
        float p0 = dotProduct(v0, axes[i]);
        float p1 = dotProduct(v1, axes[i]);
        float p2 = dotProduct(v2, axes[i]);

        float r = boxHalfSize.x * std::abs(axes[i].x) +
                  boxHalfSize.y * std::abs(axes[i].y) +
                  boxHalfSize.z * std::abs(axes[i].z);

        if (std::max(-max3(p0, p1, p2), min3(p0, p1, p2)) > r)
            return false;
    }

    if (max3(v0.x, v1.x, v2.x) < -boxHalfSize.x || min3(v0.x, v1.x, v2.x) > boxHalfSize.x) return false;
    if (max3(v0.y, v1.y, v2.y) < -boxHalfSize.y || min3(v0.y, v1.y, v2.y) > boxHalfSize.y) return false;
    if (max3(v0.z, v1.z, v2.z) < -boxHalfSize.z || min3(v0.z, v1.z, v2.z) > boxHalfSize.z) return false;

    float3 normal = crossProduct(e0, e1);
    float d = dotProduct(normal, v0);
    float r = boxHalfSize.x * std::abs(normal.x) +
              boxHalfSize.y * std::abs(normal.y) +
              boxHalfSize.z * std::abs(normal.z);

    if (std::abs(d) > r)
        return false;

    return true;
}

float3 VoxelEngine::crossProduct(const float3& a, const float3& b)
{
    float3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

float VoxelEngine::dotProduct(const float3& a, const float3& b)
{
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

float3 VoxelEngine::subtract(const float3& a, const float3& b)
{
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

float VoxelEngine::min3(float a, float b, float c)
{
    return std::min(std::min(a, b), c);
}

float VoxelEngine::max3(float a, float b, float c)
{
    return std::max(std::max(a, b), c);
}


