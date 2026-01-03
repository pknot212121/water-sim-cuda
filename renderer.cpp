#include "renderer.h"
#include <c++/12/iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

/* ---- WILL MAKE A SHADER FILE WRAPPER LATER ---- */
const char* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 aPos;\n"
"uniform mat4 mvp;\n"
"void main() { gl_Position = mvp * vec4(aPos,1.0); }\0";

const char* fragmentShaderSource = "#version 330 core\n"
"out vec4 FragColor;\n"
"void main() { FragColor = vec4(0.0,0.5,1.0,1.0); }\0";

glm::mat4 projection = glm::perspective(glm::radians(45.0f),(float)SCREEN_WIDTH/SCREEN_HEIGHT,0.1f,5000.0f);
glm::mat4 view = glm::lookAt(glm::vec3(SIZE_X/2,SIZE_Y/2,SIZE_Z*2),glm::vec3(SIZE_X/2,SIZE_Y/2,SIZE_Z/2),glm::vec3(0.0f,1.0f,0.0f));
glm::mat4 model = glm::mat4(1.0f);



Renderer::Renderer(int number)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
    glfwWindowHint(GLFW_RESIZABLE, false);
    window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Water", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
    }
    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);

    glBufferData(GL_ARRAY_BUFFER,number*sizeof(float3),NULL,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE, sizeof(float3), (void*)0);
    glEnableVertexAttribArray(0);

    cudaGraphicsGLRegisterBuffer(&cudaResource,vbo,cudaGraphicsMapFlagsWriteDiscard);
    setupShaders();
}

Renderer::~Renderer()
{
    cudaGraphicsUnregisterResource(cudaResource);
    glDeleteBuffers(1,&vbo);
    glDeleteVertexArrays(1,&vao);
    glfwTerminate();
}

void Renderer::draw(int number,float3* positionsFromCUDA)
{
    if (glfwWindowShouldClose(window)) {closed=true; return;}
    float3* positionsVBO;
    size_t numBytes;



    cudaGraphicsMapResources(1,&cudaResource,0);
    cudaGraphicsResourceGetMappedPointer((void**)&positionsVBO, &numBytes,cudaResource);

    cudaMemcpy(positionsVBO,positionsFromCUDA,number*sizeof(float3),cudaMemcpyDeviceToDevice);

    cudaGraphicsUnmapResources(1,&cudaResource,0);

    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);

    glm::mat4 mvp = projection * view;
    GLuint mvpLoc = glGetUniformLocation(shaderProgram,"mvp");
    glUniformMatrix4fv(mvpLoc,1,GL_FALSE,glm::value_ptr(mvp));

    glPointSize(3.0f);
    glBindVertexArray(vao);

    glDrawArrays(GL_POINTS,0,number);

    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Renderer::setupShaders()
{
    GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertexShader,1,&vertexShaderSource,NULL);
    glCompileShader(vertexShader);
    GLint success;
    glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(vertexShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragmentShader,1,&fragmentShaderSource,NULL);
    glCompileShader(fragmentShader);
    glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char infoLog[512];
        glGetShaderInfoLog(fragmentShader, 512, NULL, infoLog);
        std::cout << "ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" << infoLog << std::endl;
    }

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram,vertexShader);
    glAttachShader(shaderProgram,fragmentShader);
    glLinkProgram(shaderProgram);

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
}
