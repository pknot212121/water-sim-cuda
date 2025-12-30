#pragma once

#include <string>
#include <tiny_obj_loader.h>

struct ObjData
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;
    bool success;

    ObjData() : success(false) {}
};

class ObjLoader
{
public:
    ObjLoader() = default;
    ~ObjLoader() = default;

    ObjData loadObj(const std::string& filePath);
    ObjData loadObj(const std::string& filePath, const std::string& mtlBaseDir);
};
