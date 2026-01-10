#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include <cuda_gl_interop.h>
#include <vector>
#include "VoxelEngine.h"

#include "common.cuh"
#include "resource_manager.h"

constexpr int SCREEN_WIDTH = 800;
constexpr int SCREEN_HEIGHT = 600;

class Renderer
{
    public:
        Renderer(int number);
        ~Renderer();
        void draw(int number,float3* positionsFromCUDA);
        void setupShaders();
        void setTriangles(std::vector<Triangle> triangles);
        inline bool isWindowClosed(){return closed;}
    private:
        GLFWwindow* window;
        float rotationAngle = 0.0f;
        float rotationAngleVertical = 0.0f;
        bool closed = false;
        GLuint vbo,vao,shaderProgram;
        GLuint collVbo,collVao;
        int triCount = 0;
        cudaGraphicsResource* cudaResource;
};
