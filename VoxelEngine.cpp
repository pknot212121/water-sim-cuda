#include "VoxelEngine.h"
#include <iostream>
#include <cmath>
#include <algorithm>

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

    std::cout << "Normalization complete!" << std::endl;
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

std::vector<VoxelEngine::Triangle> VoxelEngine::extractTriangles(const ObjData& objData)
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

