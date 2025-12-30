#include "ObjLoader.h"
#include <iostream>

ObjData ObjLoader::loadObj(const std::string& filePath)
{
    return loadObj(filePath, "");
}

ObjData ObjLoader::loadObj(const std::string& filePath, const std::string& mtlBaseDir)
{
    ObjData data;
    std::string warn, err;

    const char* mtlBasePath = mtlBaseDir.empty() ? nullptr : mtlBaseDir.c_str();

    data.success = tinyobj::LoadObj(&data.attrib, &data.shapes, &data.materials,
                                     &warn, &err, filePath.c_str(), mtlBasePath);

    if (!warn.empty())
    {
        data.warn = warn;
        std::cout << "TinyObjLoader Warning: " << warn << std::endl;
    }

    if (!err.empty())
    {
        data.err = err;
        std::cerr << "TinyObjLoader Error: " << err << std::endl;
    }

    return data;
}

