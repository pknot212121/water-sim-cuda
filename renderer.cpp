#include "renderer.h"
//#include <c++/12/iostream>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

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
    ResourceManager::Clear();
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
    glEnable(GL_DEPTH_TEST);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    Shader &s = ResourceManager::GetShader("dot");
    s.Use();
    glm::mat4 mvp = projection * view;
    s.SetMatrix4("mvp",mvp);

    if (triCount > 0 )
    {
        s.SetVector4f("uColor",glm::vec4(0.5f,0.5f,0.5f,1.0f));
        glBindVertexArray(collVao);
        //glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
        glDrawArrays(GL_TRIANGLES,0,triCount);
        glBindVertexArray(0);
    }

    s.SetVector4f("uColor",glm::vec4(0.0f,0.5f,1.0f,1.0f));

    glPointSize(3.0f);
    glBindVertexArray(vao);

    glDrawArrays(GL_POINTS,0,number);

    glBindVertexArray(0);
    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Renderer::setupShaders()
{
    ResourceManager::LoadShader("shaders/dot.vert","shaders/dot.frag",nullptr,"dot");
}

void Renderer::setTriangles(std::vector<Triangle> triangles)
{
    triCount = triangles.size() * 3;

    glGenVertexArrays(1,&collVao);
    glGenBuffers(1,&collVbo);

    glBindVertexArray(collVao);
    glBindBuffer(GL_ARRAY_BUFFER,collVbo);
    glBufferData(GL_ARRAY_BUFFER,triangles.size() * sizeof(Triangle), triangles.data(),GL_STATIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE,sizeof(float3),(void*)0);
    glEnableVertexAttribArray(0);

    glBindBuffer(GL_ARRAY_BUFFER,0);
    glBindVertexArray(0);
}
