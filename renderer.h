#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "cuda_runtime.h"
#include <cuda_gl_interop.h>
#include <vector>
#include "VoxelEngine.h"
#include "game_configdata.h"
#include "common.cuh"
#include "resource_manager.h"

constexpr int SCREEN_WIDTH = 1920;
constexpr int SCREEN_HEIGHT = 1600;

class Renderer
{
    public:
        Renderer();
        ~Renderer();
        void init(int number);
        void draw(int number,float3* positionsFromCUDA);
        void setupShaders();
        void setupQuad();
        void setupFramebuffer();
        void setTriangles(std::vector<Triangle> triangles);
        inline bool isWindowClosed(){return closed;}
        inline bool isPaused(){return paused;}
        void handleScroll(double yoffset);
        void handleResize(int width, int height);
        void toggleFullscreen();
        void rebuildFramebuffers();
    private:
        GLFWwindow* window;
        float rotationAngle = 0.0f;
        float rotationAngleVertical = 0.0f;
        bool closed = false;
        bool paused = true;
        bool pKeyWasPressed = false;
        bool wireframeMode = false; 
        bool lKeyWasPressed = false;
        bool glassMode = false;
        bool gKeyWasPressed = false;
        bool f11KeyWasPressed = false;
        float zoomDistance;


        bool isFullscreen = false;
        int currentWidth = SCREEN_WIDTH;
        int currentHeight = SCREEN_HEIGHT;
        int windowedWidth = SCREEN_WIDTH;
        int windowedHeight = SCREEN_HEIGHT;
        int windowedPosX = 100;
        int windowedPosY = 100;

        GLuint vbo,vao,fbo,textureColorBuffer,rbo;
        GLuint blurFbo, blurTextureBuffer;
        GLuint backgroundTexture;
        GLuint collVbo,collVao;
        GLuint quadVbo,quadVao; /* TEMP FOR VIEWING FBO */
        int triCount = 0;
        cudaGraphicsResource* cudaResource;
        glm::mat4 projection;
        glm::mat4 view;
        glm::mat4 model;
};
