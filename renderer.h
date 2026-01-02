#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include <cuda_gl_interop.h>

constexpr int SCREEN_WIDTH = 800;
constexpr int SCREEN_HEIGHT = 600;





class Renderer
{
    public:
        Renderer(int number);
        ~Renderer();
        void draw(int number,float3* positionsFromCUDA);
        void setupShaders();
        inline bool isWindowClosed(){return closed;}
    private:
        GLFWwindow* window;
        bool closed = false;
        GLuint vbo,vao,shaderProgram;
        cudaGraphicsResource* cudaResource;
};
