#include <glad/glad.h>
#include <GLFW/glfw3.h>

constexpr int SCREEN_WIDTH = 800;
constexpr int SCREEN_HEIGHT = 600;

class Renderer
{
    public:
        Renderer();
        ~Renderer();
        void draw();
        inline bool isWindowClosed(){return closed;}
    private:
        GLFWwindow* window;
        bool closed = false;
};
