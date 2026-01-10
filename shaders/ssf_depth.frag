#version 330 core

in vec3 posViewSpace;
uniform mat4 projection;
uniform float radius;

void main() {
    vec3 normal;
    normal.xy = gl_PointCoord.xy * 2.0 - 1.0;

    float r2 = dot(normal.xy, normal.xy);
    if (r2 > 1.0) discard;

    normal.z = sqrt(1.0 - r2);
    vec3 pixelPosViewSpace = posViewSpace + normal * radius;
    vec4 clipPos = projection * vec4(pixelPosViewSpace,1.0);
    gl_FragDepth = (clipPos.z / clipPos.w) * 0.5 + 0.5;
    gl_FragColor = vec4(normal * 0.5 + 0.5,1.0);
}