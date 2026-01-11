#version 330 core

layout (location = 0) in vec3 aPos;
uniform mat4 view;
uniform mat4 projection;
uniform mat4 model;
uniform float radius;

out vec3 posViewSpace;

void main() {
    vec4 viewPos = view * model * vec4(aPos,1.0);
    posViewSpace = viewPos.xyz;

    gl_Position = projection * viewPos;
    gl_PointSize = radius * (1000.0 / length(viewPos.xyz));
}