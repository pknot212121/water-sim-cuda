#include "renderer.h"
//#include <c++/12/iostream>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

glm::mat4 projection = glm::perspective(glm::radians(45.0f),(float)SCREEN_WIDTH/SCREEN_HEIGHT,0.1f,5000.0f);
glm::mat4 view = glm::lookAt(glm::vec3(SIZE_X/2,SIZE_Y/2,SIZE_Z*2),glm::vec3(SIZE_X/2,SIZE_Y/2,SIZE_Z/2),glm::vec3(0.0f,1.0f,0.0f));
glm::mat4 model = glm::mat4(1.0f);

void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        std::cout << "AAAA" << std::endl;
    }
}

Renderer::Renderer(int number)
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, false);
    window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Water", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
    }
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
    //glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);

    glBufferData(GL_ARRAY_BUFFER,number*sizeof(float3),NULL,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE, sizeof(float3), (void*)0);
    glEnableVertexAttribArray(0);

    setupFramebuffer();
    setupQuad();
    cudaGraphicsGLRegisterBuffer(&cudaResource,vbo,cudaGraphicsMapFlagsWriteDiscard);
    setupShaders();
}

Renderer::~Renderer()
{
    cudaGraphicsUnregisterResource(cudaResource);
    glDeleteBuffers(1,&vbo);
    glDeleteVertexArrays(1,&vao);
    glDeleteFramebuffers(1,&fbo);
    glDeleteRenderbuffers(1,&rbo);
    glDeleteTextures(1,&textureColorBuffer);
    glDeleteVertexArrays(1,&quadVao);
    glDeleteBuffers(1,&quadVbo);
    glDeleteTextures(1,&blurTextureBuffer);
    ResourceManager::Clear();
    glfwTerminate();
}

void Renderer::draw(int number,float3* positionsFromCUDA)
{
    if (glfwWindowShouldClose(window)) {closed=true; return;}
    float rotationSpeed = 0.05f;

    if (glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS) rotationAngle += rotationSpeed;
    if (glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS) rotationAngle -= rotationSpeed;
    if (glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS) rotationAngleVertical += rotationSpeed;
    if (glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS) rotationAngleVertical -= rotationSpeed;

    float3* positionsVBO;
    size_t numBytes;

    cudaGraphicsMapResources(1,&cudaResource,0);
    cudaGraphicsResourceGetMappedPointer((void**)&positionsVBO, &numBytes,cudaResource);
    cudaMemcpy(positionsVBO,positionsFromCUDA,number*sizeof(float3),cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1,&cudaResource,0);



    glBindFramebuffer(GL_FRAMEBUFFER,fbo);
    glClearColor(99999.0f, 99999.0f, 99999.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Shader &s = ResourceManager::GetShader("spheres");
    s.Use();
    s.SetMatrix4("view",view);
    s.SetMatrix4("projection",projection);
    s.SetFloat("radius",2.0f);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS,0,number);
    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glBindVertexArray(0);



    glDisable(GL_DEPTH_TEST);
    glBindFramebuffer(GL_FRAMEBUFFER,blurFbo);
    glClearColor(99999.0f, 99999.0f, 99999.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    Shader &blurShader = ResourceManager::GetShader("blur");
    blurShader.Use();
    blurShader.SetFloat("filterSize",10.0f);
    blurShader.SetFloat("texelSize",1.0f / (float)SCREEN_WIDTH);
    blurShader.SetFloat("depthFalloff",0.01f);
    blurShader.SetVector2f("blurDir",glm::vec2(1.0f,0.0f));
    glBindVertexArray(quadVao);
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
    glDrawArrays(GL_TRIANGLES, 0,6);


    glBindFramebuffer(GL_FRAMEBUFFER,fbo);
    blurShader.SetFloat("texelSize",1.0f / (float)SCREEN_HEIGHT);
    blurShader.SetVector2f("blurDir",glm::vec2(0.0f,1.0f));
    glBindTexture(GL_TEXTURE_2D,blurTextureBuffer);
    glDrawArrays(GL_TRIANGLES,0,6);


    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glClearColor(0.2f, 0.2f, 0.2f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    Shader &screenShader = ResourceManager::GetShader("screen");
    screenShader.Use();
    screenShader.SetVector2f("texelSize",glm::vec2(1.0f / (float)SCREEN_WIDTH, 1.0 / (float)SCREEN_HEIGHT));
    screenShader.SetMatrix4("projection",projection);
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
    glDrawArrays(GL_TRIANGLES,0,6);
    glBindVertexArray(0);


    glfwSwapBuffers(window);
    glfwPollEvents();
}

void Renderer::setupShaders()
{
    ResourceManager::LoadShader("shaders/dot.vert","shaders/dot.frag",nullptr,"dot");
    ResourceManager::LoadShader("shaders/ssf_depth.vert","shaders/ssf_depth.frag",nullptr,"spheres");
    ResourceManager::LoadShader("shaders/screen.vert","shaders/screen.frag",nullptr,"screen");
    ResourceManager::LoadShader("shaders/screen.vert","shaders/ssf_blur.frag",nullptr,"blur");
}

void Renderer::setupQuad()
{
    float quadVertices[] = {
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };
    glGenVertexArrays(1,&quadVao);
    glGenBuffers(1,&quadVbo);
    glBindVertexArray(quadVao);
    glBindBuffer(GL_ARRAY_BUFFER,quadVbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices),&quadVertices,GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0,2,GL_FLOAT,GL_FALSE,4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1,2,GL_FLOAT,GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
}

void Renderer::setupFramebuffer()
{
    glGenFramebuffers(1,&fbo);
    glBindFramebuffer(GL_FRAMEBUFFER,fbo);
    glGenRenderbuffers(1,&rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glGenTextures(1,&textureColorBuffer);
    glBindTexture(GL_TEXTURE_2D,textureColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,textureColorBuffer,0);
    glRenderbufferStorage(GL_RENDERBUFFER,GL_DEPTH24_STENCIL8, SCREEN_WIDTH,SCREEN_HEIGHT);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER,GL_DEPTH_STENCIL_ATTACHMENT,GL_RENDERBUFFER,rbo);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "ERROR::FRAMEBUFFER:: Framebuffer is not complete" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER,0);



    glGenFramebuffers(1,&blurFbo);
    glBindFramebuffer(GL_FRAMEBUFFER,blurFbo);
    glGenTextures(1,&blurTextureBuffer);
    glBindTexture(GL_TEXTURE_2D,blurTextureBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RED, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,blurTextureBuffer,0);
    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "ERROR::FRAMEBUFFER:: Framebuffer is not complete" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER,0);
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
