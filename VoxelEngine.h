#pragma once

#include <cstddef>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include "ObjLoader.h"

struct VoxelData
{
    float* pos[3];  // x, y, z positions of voxels
    size_t count;   // number of voxels
    float resolution; // voxel size
    float3 boundingBoxMin;
    float3 boundingBoxMax;

    VoxelData();

    // Copy constructor - deep copy
    VoxelData(const VoxelData& other);

    // Move constructor
    VoxelData(VoxelData&& other) noexcept;

    // Copy assignment operator
    VoxelData& operator=(const VoxelData& other);

    // Move assignment operator
    VoxelData& operator=(VoxelData&& other) noexcept;

    ~VoxelData();
};

struct Triangle
{
    float3 v0, v1, v2;
};

class VoxelEngine
{
public:
    VoxelEngine() = default;
    ~VoxelEngine() = default;

    // Voxelize mesh using triangle intersection
    VoxelData voxelize(const ObjData& objData, float resolution);

    // Normalize, scale and displace voxel data
    void normalize(VoxelData& data, float normalizeSize, float scale, const float3& displacement);

    // Expand voxels - create 5x5x5 grid around each voxel (for thickening)
    void expandVoxels(VoxelData& data, int expansionRadius = 2);

    // Normalize, scale and displace triangle data
    void normalize(std::vector<Triangle>& triangles, float normalizeSize, float scale, const float3& displacement);

    std::vector<Triangle> extractTriangles(const ObjData& objData);

private:
    struct BoundingBox
    {
        float3 min;
        float3 max;
    };

    // Helper functions
    BoundingBox calculateBoundingBox(const ObjData& objData);

    bool triangleBoxIntersection(const Triangle& tri, const float3& boxCenter, float halfSize);
    bool triangleAABBIntersection(const Triangle& tri, const float3& boxMin, const float3& boxMax);

    // Math helpers
    float3 crossProduct(const float3& a, const float3& b);
    float dotProduct(const float3& a, const float3& b);
    float3 subtract(const float3& a, const float3& b);
    float min3(float a, float b, float c);
    float max3(float a, float b, float c);
};
