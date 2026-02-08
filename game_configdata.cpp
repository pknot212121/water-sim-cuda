#include "game_configdata.h"
#include <iostream>
#include <algorithm>
#include <sstream>

std::unordered_map<std::string, std::string> GameConfigData::configMap;
std::vector<ConfigObject> GameConfigData::waters;
std::vector<ConfigObject> GameConfigData::objects;

std::vector<std::string> GameConfigData::split(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream,token,delimiter))
    {
        tokens.push_back(token);
    }
    return tokens;
}

void GameConfigData::setConfigDataFromFile(std::string filename)
{
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file,line))
    {
        line.erase(std::remove(line.begin(),line.end(),' '),line.end());
        if (line.empty()) continue;
        size_t pos = line.find('=');
        if (pos!=std::string::npos)
        {
            std::string key = line.substr(0,pos);
            std::string value = line.substr(pos+1);
            if (key=="WATER" || key == "OBJECT")
            {
                std::vector<std::string> parts = split(value,';');
                if (parts.size() >= 5)
                {
                    ConfigObject obj;
                    obj.modelPath = parts[0];
                    obj.scale = std::stof(parts[1]);
                    obj.x = std::stof(parts[2]);
                    obj.y = std::stof(parts[3]);
                    obj.z = std::stof(parts[4]);
                    if (key=="WATER") waters.push_back(obj);
                    else objects.push_back(obj);
                }
                else{std::cerr << "ŹLE ZAKODOWANE " << key << " O WARTOŚCI " << value << std::endl; exit(1);}
            }
            else configMap[key]=value;
        }
    }
}

int GameConfigData::getInt(const std::string& key)
{
    if (configMap.contains(key)) return std::stoi(configMap[key]);
    std::cerr << "Nie ma klucza " << key << " w configu!!!" << std::endl;
    exit(1);
}

void GameConfigData::setInt(const std::string& key, const std::string& value)
{
    if (configMap.contains(key)) configMap[key]=value;
    else
    {
        std::cerr << "Nie ma klucza " << key << " w configu!!!" << std::endl;
        exit(1);
    }
}

void GameConfigData::setNewInt(const std::string& key, const std::string& value)
{
    configMap[key]=value;
}

float GameConfigData::getFloat(const std::string& key)
{
    if (configMap.contains(key)) return std::stof(configMap[key]);
    std::cerr << "Nie ma klucza " << key << " w configu!!!" << std::endl;
    exit(1);
}

std::string GameConfigData::getString(const std::string& key)
{
    if (configMap.contains(key)) return configMap[key];
    std::cerr << "Nie ma takiego klucza w configu!!!" << std::endl;
    exit(1);
}

