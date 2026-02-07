#ifndef LIFE_SIM_GAME_CONFIG_DATA_H
#define LIFE_SIM_GAME_CONFIG_DATA_H
#include <string>
#include <unordered_map>
#include <fstream>

class GameConfigData
{
public:
    static void setConfigDataFromFile(std::string filename);
    static int getInt(const std::string& key);
    static void setInt(const std::string& key, const std::string& value);
    static void setNewInt(const std::string& key, const std::string& value);
    static float getFloat(const std::string& key);
    static std::string getString(const std::string& key);
private:
    GameConfigData() {};
    static std::unordered_map<std::string,std::string> configMap;
};


#endif