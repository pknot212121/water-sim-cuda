#include "renderer.h"
//#include <c++/12/iostream>
#include <iostream>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>





void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT)
    {
        std::cout << "AAAA" << std::endl;
    }
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{

    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (renderer)
    {
        renderer->handleScroll(yoffset);
    }
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    Renderer* renderer = static_cast<Renderer*>(glfwGetWindowUserPointer(window));
    if (renderer)
    {
        renderer->handleResize(width, height);
    }
}

Renderer::Renderer()
{
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_RESIZABLE, true);
    window = glfwCreateWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Water Simulation", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
    }


    glfwSetWindowUserPointer(window, this);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    glViewport(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT);
    //glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


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
    glDeleteTextures(1,&backgroundTexture);
    ResourceManager::Clear();
    glfwTerminate();
}

void Renderer::init(int number)
{
    glGenVertexArrays(1,&vao);
    glGenBuffers(1,&vbo);

    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER,vbo);

    glBufferData(GL_ARRAY_BUFFER,number*sizeof(float3),NULL,GL_DYNAMIC_DRAW);
    glVertexAttribPointer(0,3,GL_FLOAT,GL_FALSE, sizeof(float3), (void*)0);
    glEnableVertexAttribArray(0);

    zoomDistance = GameConfigData::getInt("SIZE_Z") * 2.0f;
    projection = glm::perspective(glm::radians(45.0f),(float)SCREEN_WIDTH/SCREEN_HEIGHT,0.1f,5000.0f);
    view = glm::lookAt(glm::vec3(GameConfigData::getInt("SIZE_X")/2,GameConfigData::getInt("SIZE_Y")/2,GameConfigData::getInt("SIZE_Z")*2),
            glm::vec3(GameConfigData::getInt("SIZE_X")/2,GameConfigData::getInt("SIZE_Y")/2,GameConfigData::getInt("SIZE_Z")/2),glm::vec3(0.0f,1.0f,0.0f));
    model = glm::mat4(1.0f);
    setupFramebuffer();
    setupQuad();
    cudaGraphicsGLRegisterBuffer(&cudaResource,vbo,cudaGraphicsMapFlagsWriteDiscard);
    setupShaders();

}

/* I NEED A WRAPPER FOR TEXTURES HOLY SHIT */
void Renderer::draw(int number,float3* positionsFromCUDA)
{
    if (glfwWindowShouldClose(window)) {closed=true; return;}
    float rotationSpeed = 0.05f;

    if (glfwGetKey(window,GLFW_KEY_A) == GLFW_PRESS) rotationAngle += rotationSpeed;
    if (glfwGetKey(window,GLFW_KEY_D) == GLFW_PRESS) rotationAngle -= rotationSpeed;
    if (glfwGetKey(window,GLFW_KEY_W) == GLFW_PRESS) rotationAngleVertical += rotationSpeed;
    if (glfwGetKey(window,GLFW_KEY_S) == GLFW_PRESS) rotationAngleVertical -= rotationSpeed;

    if (glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS)
    {
        if (!pKeyWasPressed)
        {
            paused = !paused;
            pKeyWasPressed = true;
            std::cout << "Simulation " << (paused ? "PAUSED" : "RESUMED") << std::endl;
        }
    }
    else
    {
        pKeyWasPressed = false;
    }


    if (glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS)
    {
        if (!lKeyWasPressed)
        {
            wireframeMode = !wireframeMode;
            lKeyWasPressed = true;
            std::cout << "Wireframe mode " << (wireframeMode ? "ON (contours only)" : "OFF (filled)") << std::endl;
        }
    }
    else
    {
        lKeyWasPressed = false;
    }


    if (glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS)
    {
        if (!gKeyWasPressed)
        {
            glassMode = !glassMode;
            gKeyWasPressed = true;
            std::cout << "Glass mode " << (glassMode ? "ON (transparent, water visible)" : "OFF (opaque)") << std::endl;
        }
    }
    else
    {
        gKeyWasPressed = false;
    }


    if (glfwGetKey(window, GLFW_KEY_F11) == GLFW_PRESS)
    {
        if (!f11KeyWasPressed)
        {
            toggleFullscreen();
            f11KeyWasPressed = true;
        }
    }
    else
    {
        f11KeyWasPressed = false;
    }

    float3* positionsVBO;
    size_t numBytes;

    cudaGraphicsMapResources(1,&cudaResource,0);
    cudaGraphicsResourceGetMappedPointer((void**)&positionsVBO, &numBytes,cudaResource);
    cudaMemcpy(positionsVBO,positionsFromCUDA,number*sizeof(float3),cudaMemcpyDeviceToDevice);
    cudaGraphicsUnmapResources(1,&cudaResource,0);

    glm::vec3 center(GameConfigData::getInt("SIZE_X")/2.0f,GameConfigData::getInt("SIZE_Y")/2.0f,GameConfigData::getInt("SIZE_Z")/2.0f);


    glm::vec3 cameraPos = glm::vec3(GameConfigData::getInt("SIZE_X")/2.0f,GameConfigData::getInt("SIZE_Y")/2.0f, zoomDistance);
    auto view = glm::lookAt(cameraPos, center, glm::vec3(0.0f, 1.0f, 0.0f));

    auto model = glm::mat4(1.0f);
    model = glm::translate(model,center);
    model = glm::rotate(model,rotationAngleVertical, glm::vec3(1,0,0));
    model = glm::rotate(model,rotationAngle,glm::vec3(0,1,0));
    model = glm::translate(model, -center);

    auto worldLightDir = glm::normalize(glm::vec3(0.2f, 10.0f, 0.5f));
    auto lightDir = glm::vec3(view * glm::vec4(worldLightDir,0.0f));

    glDisable(GL_BLEND);
    glActiveTexture(GL_TEXTURE0);
    glBindFramebuffer(GL_FRAMEBUFFER,fbo);
    glClearColor(99999.0f, 99999.0f, 99999.0f, 1.0f);
    glEnable(GL_DEPTH_TEST);
    glEnable(GL_PROGRAM_POINT_SIZE);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    Shader &s = ResourceManager::GetShader("spheres");
    s.Use();
    s.SetMatrix4("view",view);
    s.SetMatrix4("projection",projection);
    s.SetMatrix4("model",model);
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
    blurShader.SetFloat("texelSize",1.0f / (float)currentWidth);
    blurShader.SetFloat("depthFalloff",0.01f);
    blurShader.SetVector2f("blurDir",glm::vec2(1.0f,0.0f));
    glBindVertexArray(quadVao);
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
    glDrawArrays(GL_TRIANGLES, 0,6);


    glBindFramebuffer(GL_FRAMEBUFFER,fbo);
    glClearColor(99999.0f, 99999.0f, 99999.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    blurShader.SetFloat("texelSize",1.0f / (float)currentHeight);
    blurShader.SetVector2f("blurDir",glm::vec2(0.0f,1.0f));
    glBindTexture(GL_TEXTURE_2D,blurTextureBuffer);
    glDrawArrays(GL_TRIANGLES,0,6);
    glBindVertexArray(0);

    glBindFramebuffer(GL_FRAMEBUFFER,0);
    glClearColor(0.85f, 0.9f, 0.95f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glEnable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);

    if (triCount > 0)
    {

        if (wireframeMode)
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
            glLineWidth(2.0f);
        }
        else
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }

        Shader &triShader = ResourceManager::GetShader("dot");
        triShader.Use();
        auto mvp = projection * view * model;
        triShader.SetMatrix4("mvp",mvp);
        triShader.SetMatrix4("model",model);
        triShader.SetVector3f("worldLightDir",worldLightDir);


        if (glassMode)
        {

            glEnable(GL_BLEND);
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
            glEnable(GL_CULL_FACE);


            glCullFace(GL_FRONT);
            glDepthMask(GL_FALSE);
            triShader.SetVector4f("uColor",glm::vec4(0.4f, 0.5f, 0.6f, 0.2f));
            glBindVertexArray(collVao);
            glDrawArrays(GL_TRIANGLES,0,triCount);


            glCullFace(GL_BACK);
            glDepthMask(GL_FALSE);
            triShader.SetVector4f("uColor",glm::vec4(0.6f, 0.7f, 0.8f, 0.35f));
            glDrawArrays(GL_TRIANGLES,0,triCount);


            glDisable(GL_CULL_FACE);
            glDepthMask(GL_TRUE);
            glDisable(GL_BLEND);
        }
        else
        {

            triShader.SetVector4f("uColor",glm::vec4(0.5f, 0.5f, 0.5f, 1.0f));
            glBindVertexArray(collVao);
            glDrawArrays(GL_TRIANGLES,0,triCount);
        }

        glBindVertexArray(0);


        if (wireframeMode)
        {
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
        }
    }

    glBindTexture(GL_TEXTURE_2D,backgroundTexture);
    glCopyTexSubImage2D(GL_TEXTURE_2D,0,0,0,0,0,currentWidth,currentHeight);

    if (!PHASING) glEnable(GL_DEPTH_TEST);
    else glDisable(GL_DEPTH_TEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    Shader &screenShader = ResourceManager::GetShader("screen");
    screenShader.Use();
    screenShader.SetVector2f("texelSize",glm::vec2(1.0f / (float)currentWidth, 1.0 / (float)currentHeight));
    screenShader.SetMatrix4("projection",projection);
    screenShader.SetVector3f("lightDir",lightDir);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D,backgroundTexture);
    screenShader.SetInteger("screenTexture",0);
    screenShader.SetInteger("backgroundTexture",1);

    glBindVertexArray(quadVao);
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


    glGenTextures(1, &backgroundTexture);
    glBindTexture(GL_TEXTURE_2D, backgroundTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, SCREEN_WIDTH, SCREEN_HEIGHT, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
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

void Renderer::handleScroll(double yoffset)
{

    float zoomSpeed = 5.0f;
    zoomDistance -= static_cast<float>(yoffset) * zoomSpeed;


    float minZoom = 20.0f;
    float maxZoom = 500.0f;
    if (zoomDistance < minZoom) zoomDistance = minZoom;
    if (zoomDistance > maxZoom) zoomDistance = maxZoom;

    std::cout << "Zoom distance: " << zoomDistance << std::endl;
}

void Renderer::handleResize(int width, int height)
{
    currentWidth = width;
    currentHeight = height;

    glViewport(0, 0, width, height);


    projection = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, 5000.0f);


    rebuildFramebuffers();

    std::cout << "Window resized to: " << width << "x" << height << std::endl;
}

void Renderer::toggleFullscreen()
{
    isFullscreen = !isFullscreen;

    if (isFullscreen)
    {

        glfwGetWindowPos(window, &windowedPosX, &windowedPosY);
        glfwGetWindowSize(window, &windowedWidth, &windowedHeight);


        GLFWmonitor* monitor = glfwGetPrimaryMonitor();
        const GLFWvidmode* mode = glfwGetVideoMode(monitor);
        glfwSetWindowMonitor(window, monitor, 0, 0, mode->width, mode->height, mode->refreshRate);

        std::cout << "Fullscreen mode ON (" << mode->width << "x" << mode->height << ")" << std::endl;
    }
    else
    {

        glfwSetWindowMonitor(window, nullptr, windowedPosX, windowedPosY, windowedWidth, windowedHeight, 0);

        std::cout << "Fullscreen mode OFF (windowed " << windowedWidth << "x" << windowedHeight << ")" << std::endl;
    }
}

void Renderer::rebuildFramebuffers()
{

    glDeleteFramebuffers(1, &fbo);
    glDeleteRenderbuffers(1, &rbo);
    glDeleteTextures(1, &textureColorBuffer);
    glDeleteFramebuffers(1, &blurFbo);
    glDeleteTextures(1, &blurTextureBuffer);
    glDeleteTextures(1, &backgroundTexture);


    glGenFramebuffers(1, &fbo);
    glBindFramebuffer(GL_FRAMEBUFFER, fbo);
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glGenTextures(1, &textureColorBuffer);
    glBindTexture(GL_TEXTURE_2D, textureColorBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, currentWidth, currentHeight, 0, GL_RED, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorBuffer, 0);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, currentWidth, currentHeight);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "ERROR::FRAMEBUFFER:: Framebuffer is not complete after resize" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenFramebuffers(1, &blurFbo);
    glBindFramebuffer(GL_FRAMEBUFFER, blurFbo);
    glGenTextures(1, &blurTextureBuffer);
    glBindTexture(GL_TEXTURE_2D, blurTextureBuffer);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, currentWidth, currentHeight, 0, GL_RED, GL_FLOAT, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, blurTextureBuffer, 0);

    if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "ERROR::FRAMEBUFFER:: Blur framebuffer is not complete after resize" << std::endl;
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    glGenTextures(1, &backgroundTexture);
    glBindTexture(GL_TEXTURE_2D, backgroundTexture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, currentWidth, currentHeight, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
}

