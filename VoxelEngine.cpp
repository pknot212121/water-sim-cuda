#include "VoxelEngine.h"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <set>
#include <tuple>

#include "common.cuh"

// VoxelData implementation
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
        // Clean up existing data
        if (pos[0]) delete[] pos[0];
        if (pos[1]) delete[] pos[1];
        if (pos[2]) delete[] pos[2];

        // Copy data
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
        // Clean up existing data
        if (pos[0]) delete[] pos[0];
        if (pos[1]) delete[] pos[1];
        if (pos[2]) delete[] pos[2];

        // Move data
        count = other.count;
        resolution = other.resolution;
        boundingBoxMin = other.boundingBoxMin;
        boundingBoxMax = other.boundingBoxMax;
        pos[0] = other.pos[0];
        pos[1] = other.pos[1];
        pos[2] = other.pos[2];

        // Reset other
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


// VoxelEngine implementation
VoxelData VoxelEngine::voxelize(const ObjData& objData, float resolution)
{
    VoxelData voxelData;

    if (!objData.success || objData.attrib.vertices.empty())
    {
        std::cerr << "VoxelEngine: Invalid ObjData provided" << std::endl;
        return voxelData;
    }

    std::cout << "Starting voxelization with resolution: " << resolution << std::endl;

    // Calculate bounding box
    BoundingBox bbox = calculateBoundingBox(objData);
    voxelData.boundingBoxMin = bbox.min;
    voxelData.boundingBoxMax = bbox.max;
    voxelData.resolution = resolution;

    std::cout << "Bounding box: (" << bbox.min.x << ", " << bbox.min.y << ", " << bbox.min.z << ") to ("
              << bbox.max.x << ", " << bbox.max.y << ", " << bbox.max.z << ")" << std::endl;

    // Calculate grid dimensions
    int gridX = (int)std::ceil((bbox.max.x - bbox.min.x) / resolution);
    int gridY = (int)std::ceil((bbox.max.y - bbox.min.y) / resolution);
    int gridZ = (int)std::ceil((bbox.max.z - bbox.min.z) / resolution);

    std::cout << "Grid dimensions: " << gridX << " x " << gridY << " x " << gridZ << std::endl;

    // Extract triangles from mesh
    std::vector<Triangle> triangles = extractTriangles(objData);
    std::cout << "Number of triangles: " << triangles.size() << std::endl;

    // Initialize a grid of booleans (much faster than storing positions directly)
    size_t totalVoxels = (size_t)gridX * gridY * gridZ;
    std::vector<bool> filled(totalVoxels, false);
    std::cout << "Grid total voxels: " << totalVoxels << std::endl;

    // Loop over triangles and mark filled voxels
    for (size_t triIdx = 0; triIdx < triangles.size(); triIdx++)
    {
        const Triangle& tri = triangles[triIdx];

        // Find triangle's local bounding box in world space
        float triMinX = min3(tri.v0.x, tri.v1.x, tri.v2.x);
        float triMinY = min3(tri.v0.y, tri.v1.y, tri.v2.y);
        float triMinZ = min3(tri.v0.z, tri.v1.z, tri.v2.z);
        float triMaxX = max3(tri.v0.x, tri.v1.x, tri.v2.x);
        float triMaxY = max3(tri.v0.y, tri.v1.y, tri.v2.y);
        float triMaxZ = max3(tri.v0.z, tri.v1.z, tri.v2.z);

        // Convert triangle bounding box to grid indices
        int minGridX = std::max(0, (int)std::floor((triMinX - bbox.min.x) / resolution));
        int minGridY = std::max(0, (int)std::floor((triMinY - bbox.min.y) / resolution));
        int minGridZ = std::max(0, (int)std::floor((triMinZ - bbox.min.z) / resolution));
        int maxGridX = std::min(gridX - 1, (int)std::ceil((triMaxX - bbox.min.x) / resolution));
        int maxGridY = std::min(gridY - 1, (int)std::ceil((triMaxY - bbox.min.y) / resolution));
        int maxGridZ = std::min(gridZ - 1, (int)std::ceil((triMaxZ - bbox.min.z) / resolution));

        // Only test voxels in the triangle's local bounding box
        for (int z = minGridZ; z <= maxGridZ; z++)
        {
            for (int y = minGridY; y <= maxGridY; y++)
            {
                for (int x = minGridX; x <= maxGridX; x++)
                {
                    // Calculate voxel AABB
                    float3 voxelMin, voxelMax;
                    voxelMin.x = bbox.min.x + x * resolution;
                    voxelMin.y = bbox.min.y + y * resolution;
                    voxelMin.z = bbox.min.z + z * resolution;
                    voxelMax.x = voxelMin.x + resolution;
                    voxelMax.y = voxelMin.y + resolution;
                    voxelMax.z = voxelMin.z + resolution;

                    // Test intersection
                    if (triangleAABBIntersection(tri, voxelMin, voxelMax))
                    {
                        size_t voxelIndex = (size_t)z * gridY * gridX + (size_t)y * gridX + (size_t)x;
                        filled[voxelIndex] = true;
                    }
                }
            }
        }

        // Progress indicator for large meshes
        if ((triIdx + 1) % 10000 == 0 || triIdx == triangles.size() - 1)
        {
            std::cout << "Processed " << (triIdx + 1) << "/" << triangles.size() << " triangles" << std::endl;
        }
    }

    // Convert boolean grid to output positions
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
                    // Calculate voxel center
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

    // Allocate and copy data
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

    // Step 1: Find current bounding box of voxels
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

    // Step 2: Calculate the maximum dimension to maintain aspect ratio
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

    // Step 3: Normalize to [0, normalizeSize] maintaining aspect ratio
    float normalizationScale = normalizeSize / maxDimension;
    float3 currentCenter = {
        (minX + maxX) * 0.5f,
        (minY + maxY) * 0.5f,
        (minZ + maxZ) * 0.5f
    };

    // First, center at origin, then scale to normalized size
    for (size_t i = 0; i < data.count; i++)
    {
        // Translate to origin
        data.pos[0][i] -= currentCenter.x;
        data.pos[1][i] -= currentCenter.y;
        data.pos[2][i] -= currentCenter.z;

        // Scale to normalized size
        data.pos[0][i] *= normalizationScale;
        data.pos[1][i] *= normalizationScale;
        data.pos[2][i] *= normalizationScale;

        // Translate to center of normalized space
        data.pos[0][i] += normalizeSize * 0.5f;
        data.pos[1][i] += normalizeSize * 0.5f;
        data.pos[2][i] += normalizeSize * 0.5f;
    }

    std::cout << "  After normalization, voxels are centered at ("
              << normalizeSize * 0.5f << ", "
              << normalizeSize * 0.5f << ", "
              << normalizeSize * 0.5f << ")" << std::endl;

    // Step 4: Scale from center of normalized space
    float3 scaleCenter = {normalizeSize * 0.5f, normalizeSize * 0.5f, normalizeSize * 0.5f};

    for (size_t i = 0; i < data.count; i++)
    {
        // Translate to scale center
        data.pos[0][i] -= scaleCenter.x;
        data.pos[1][i] -= scaleCenter.y;
        data.pos[2][i] -= scaleCenter.z;

        // Apply scale
        data.pos[0][i] *= scale;
        data.pos[1][i] *= scale;
        data.pos[2][i] *= scale;

        // Translate back
        data.pos[0][i] += scaleCenter.x;
        data.pos[1][i] += scaleCenter.y;
        data.pos[2][i] += scaleCenter.z;
    }

    std::cout << "  After scaling by " << scale << "x" << std::endl;

    // Step 5: Apply displacement
    for (size_t i = 0; i < data.count; i++)
    {
        data.pos[0][i] += displacement.x;
        data.pos[1][i] += displacement.y;
        data.pos[2][i] += displacement.z;
    }

    std::cout << "  After displacement" << std::endl;

    // Step 6: Filter out voxels that are outside normalization bounds [0, normalizeSize]
    std::vector<float> validX, validY, validZ;
    size_t removedCount = 0;

    for (size_t i = 0; i < data.count; i++)
    {
        float x = data.pos[0][i];
        float y = data.pos[1][i];
        float z = data.pos[2][i];

        // Check if voxel is within bounds
        if (x >= 0.0f && x <= normalizeSize &&
            y >= 0.0f && y <= normalizeSize &&
            z >= 0.0f && z <= normalizeSize)
        {
            validX.push_back(x);
            validY.push_back(y);
            validZ.push_back(z);
        }
        else
        {
            removedCount++;
        }
    }

    // Replace data with filtered voxels
    if (validX.size() != data.count)
    {
        std::cout << "  Removed " << removedCount << " voxels outside bounds [0, " << normalizeSize << "]" << std::endl;
        std::cout << "  Remaining voxels: " << validX.size() << std::endl;

        // Free old arrays
        delete[] data.pos[0];
        delete[] data.pos[1];
        delete[] data.pos[2];

        // Allocate new arrays with filtered data
        data.count = validX.size();
        if (data.count > 0)
        {
            data.pos[0] = new float[data.count];
            data.pos[1] = new float[data.count];
            data.pos[2] = new float[data.count];

            std::copy(validX.begin(), validX.end(), data.pos[0]);
            std::copy(validY.begin(), validY.end(), data.pos[1]);
            std::copy(validZ.begin(), validZ.end(), data.pos[2]);
        }
        else
        {
            data.pos[0] = nullptr;
            data.pos[1] = nullptr;
            data.pos[2] = nullptr;
        }
    }

    // Update bounding box in VoxelData
    if (data.count > 0)
    {
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

        std::cout << "  Final bounds: (" << minX << ", " << minY << ", " << minZ << ") to ("
                  << maxX << ", " << maxY << ", " << maxZ << ")" << std::endl;
    }
    else
    {
        data.boundingBoxMin = {0.0f, 0.0f, 0.0f};
        data.boundingBoxMax = {0.0f, 0.0f, 0.0f};
        std::cout << "  Warning: All voxels were removed!" << std::endl;
    }

    // Step 7: Snap voxels to discrete grid with resolution 0.1 AND expand to neighbors
    std::cout << "  Snapping voxels to discrete grid (0.1 resolution) with 5x5x5 expansion..." << std::endl;

    const float gridResolution = 0.1f;
    int maxGridIndex = (int)(normalizeSize / gridResolution);
    const int expansionRadius = 2; // Will create 5 voxels per dimension (-2, -1, 0, +1, +2)

    // Use a set to store unique grid positions (to remove duplicates)
    std::set<std::tuple<int, int, int>> uniqueGridPositions;

    for (size_t i = 0; i < data.count; i++)
    {
        // Round each coordinate to nearest grid position
        int gridX = (int)std::round(data.pos[0][i] / gridResolution);
        int gridY = (int)std::round(data.pos[1][i] / gridResolution);
        int gridZ = (int)std::round(data.pos[2][i] / gridResolution);

        // Clamp to valid range [0, maxGridIndex]
        gridX = std::max(0, std::min(maxGridIndex, gridX));
        gridY = std::max(0, std::min(maxGridIndex, gridY));
        gridZ = std::max(0, std::min(maxGridIndex, gridZ));

        // Generate 5x5x5 voxels around the snapped position
        for (int dx = -expansionRadius; dx <= expansionRadius; dx++)
        {
            for (int dy = -expansionRadius; dy <= expansionRadius; dy++)
            {
                for (int dz = -expansionRadius; dz <= expansionRadius; dz++)
                {
                    int newX = gridX + dx;
                    int newY = gridY + dy;
                    int newZ = gridZ + dz;

                    // Clamp to valid range
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
    std::cout << "  After 5x5x5 expansion: " << uniqueGridPositions.size() << " unique grid positions" << std::endl;
    std::cout << "  Expansion factor: " << ((float)uniqueGridPositions.size() / data.count) << "x" << std::endl;

    // Convert back to arrays
    std::vector<float> snappedX, snappedY, snappedZ;
    for (const auto& gridPos : uniqueGridPositions)
    {
        snappedX.push_back(std::get<0>(gridPos) * gridResolution);
        snappedY.push_back(std::get<1>(gridPos) * gridResolution);
        snappedZ.push_back(std::get<2>(gridPos) * gridResolution);
    }

    // Replace data with snapped voxels
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

        // Update bounding box after snapping
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

        // Print sample voxels to verify grid snapping
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
    int maxGridIndex = (int)(SIZE_X * 10); // Maximum grid index for SIZE_X range

    // Use a set to store unique grid positions
    std::set<std::tuple<int, int, int>> uniqueGridPositions;

    for (size_t i = 0; i < data.count; i++)
    {
        // Convert current position to grid indices
        int gridX = (int)std::round(data.pos[0][i] / gridResolution);
        int gridY = (int)std::round(data.pos[1][i] / gridResolution);
        int gridZ = (int)std::round(data.pos[2][i] / gridResolution);

        // Generate (2*radius+1)^3 voxels around this position
        for (int dx = -expansionRadius; dx <= expansionRadius; dx++)
        {
            for (int dy = -expansionRadius; dy <= expansionRadius; dy++)
            {
                for (int dz = -expansionRadius; dz <= expansionRadius; dz++)
                {
                    int newX = gridX + dx;
                    int newY = gridY + dy;
                    int newZ = gridZ + dz;

                    // Clamp to valid range [0, maxGridIndex]
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

    // Convert back to arrays
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

    // Replace data with expanded voxels
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

        // Update bounding box
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

    // Step 1: Find current bounding box of all triangle vertices
    float minX = triangles[0].v0.x, maxX = triangles[0].v0.x;
    float minY = triangles[0].v0.y, maxY = triangles[0].v0.y;
    float minZ = triangles[0].v0.z, maxZ = triangles[0].v0.z;

    for (const auto& tri : triangles)
    {
        // Check all three vertices
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

    // Step 2: Calculate the maximum dimension to maintain aspect ratio
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

    // Step 3: Normalize to [0, normalizeSize] maintaining aspect ratio
    float normalizationScale = normalizeSize / maxDimension;
    float3 currentCenter = {
        (minX + maxX) * 0.5f,
        (minY + maxY) * 0.5f,
        (minZ + maxZ) * 0.5f
    };

    // Lambda to transform a single vertex
    auto transformVertex = [&](float3& vertex) {
        // Translate to origin
        vertex.x -= currentCenter.x;
        vertex.y -= currentCenter.y;
        vertex.z -= currentCenter.z;

        // Scale to normalized size
        vertex.x *= normalizationScale;
        vertex.y *= normalizationScale;
        vertex.z *= normalizationScale;

        // Translate to center of normalized space
        vertex.x += normalizeSize * 0.5f;
        vertex.y += normalizeSize * 0.5f;
        vertex.z += normalizeSize * 0.5f;
    };

    // Apply normalization to all vertices
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

    // Step 4: Scale from center of normalized space
    float3 scaleCenter = {normalizeSize * 0.5f, normalizeSize * 0.5f, normalizeSize * 0.5f};

    auto scaleVertex = [&](float3& vertex) {
        // Translate to scale center
        vertex.x -= scaleCenter.x;
        vertex.y -= scaleCenter.y;
        vertex.z -= scaleCenter.z;

        // Apply scale
        vertex.x *= scale;
        vertex.y *= scale;
        vertex.z *= scale;

        // Translate back
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

    std::cout << "  After scaling by " << scale << "x" << std::endl;

    // Step 5: Apply displacement
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

    // Step 6: Filter out triangles that are completely outside normalization bounds [0, normalizeSize]
    std::vector<Triangle> validTriangles;
    size_t removedCount = 0;

    for (const auto& tri : triangles)
    {
        // Check if at least one vertex is within bounds
        bool v0Valid = (tri.v0.x >= 0.0f && tri.v0.x <= normalizeSize &&
                        tri.v0.y >= 0.0f && tri.v0.y <= normalizeSize &&
                        tri.v0.z >= 0.0f && tri.v0.z <= normalizeSize);

        bool v1Valid = (tri.v1.x >= 0.0f && tri.v1.x <= normalizeSize &&
                        tri.v1.y >= 0.0f && tri.v1.y <= normalizeSize &&
                        tri.v1.z >= 0.0f && tri.v1.z <= normalizeSize);

        bool v2Valid = (tri.v2.x >= 0.0f && tri.v2.x <= normalizeSize &&
                        tri.v2.y >= 0.0f && tri.v2.y <= normalizeSize &&
                        tri.v2.z >= 0.0f && tri.v2.z <= normalizeSize);

        // Keep triangle if at least one vertex is within bounds
        if (v0Valid || v1Valid || v2Valid)
        {
            validTriangles.push_back(tri);
        }
        else
        {
            removedCount++;
        }
    }

    // Replace triangles with filtered ones
    if (validTriangles.size() != triangles.size())
    {
        std::cout << "  Removed " << removedCount << " triangles completely outside bounds [0, " << normalizeSize << "]" << std::endl;
        std::cout << "  Remaining triangles: " << validTriangles.size() << std::endl;

        triangles = std::move(validTriangles);
    }

    // Calculate and display final bounds
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

            if (fv == 3) // Only triangles
            {
                Triangle tri;

                // Get vertex indices
                tinyobj::index_t idx0 = shape.mesh.indices[indexOffset + 0];
                tinyobj::index_t idx1 = shape.mesh.indices[indexOffset + 1];
                tinyobj::index_t idx2 = shape.mesh.indices[indexOffset + 2];

                // Get vertex positions
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

// Separating Axis Theorem (SAT) based triangle-AABB intersection
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

    // Translate triangle to box center
    float3 v0 = subtract(tri.v0, boxCenter);
    float3 v1 = subtract(tri.v1, boxCenter);
    float3 v2 = subtract(tri.v2, boxCenter);

    // Compute edge vectors
    float3 e0 = subtract(v1, v0);
    float3 e1 = subtract(v2, v1);
    float3 e2 = subtract(v0, v2);

    // Test the 9 edge cross-product axes
    float3 axes[9];
    // e0 cross product with box axes
    axes[0] = {0, -e0.z, e0.y};
    axes[1] = {e0.z, 0, -e0.x};
    axes[2] = {-e0.y, e0.x, 0};
    // e1 cross product with box axes
    axes[3] = {0, -e1.z, e1.y};
    axes[4] = {e1.z, 0, -e1.x};
    axes[5] = {-e1.y, e1.x, 0};
    // e2 cross product with box axes
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

    // Test the 3 face normals from the AABB
    if (max3(v0.x, v1.x, v2.x) < -boxHalfSize.x || min3(v0.x, v1.x, v2.x) > boxHalfSize.x) return false;
    if (max3(v0.y, v1.y, v2.y) < -boxHalfSize.y || min3(v0.y, v1.y, v2.y) > boxHalfSize.y) return false;
    if (max3(v0.z, v1.z, v2.z) < -boxHalfSize.z || min3(v0.z, v1.z, v2.z) > boxHalfSize.z) return false;

    // Test the triangle normal
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


