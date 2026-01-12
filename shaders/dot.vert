#version 330 core

layout (location = 0) in vec3 aPos;
uniform mat4 mvp;
uniform mat4 model;

out vec3 FragPos;
out vec3 Normal;

void main() {
    // Pozycja w world space
    FragPos = vec3(model * vec4(aPos, 1.0));

    // Normalna - obliczana automatycznie w fragment shader per-trójkąt
    // Tutaj przekazujemy tylko pozycję, normalna będzie z flat shading
    Normal = vec3(0.0, 0.0, 1.0); // Placeholder, użyjemy flat shading

    gl_Position = mvp * vec4(aPos, 1.0);
}