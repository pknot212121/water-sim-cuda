#version 330 core
out vec4 FragColor;

in vec2 TexCoords;
uniform sampler2D screenTexture;
uniform sampler2D backgroundTexture;
uniform mat4 projection;
uniform vec2 texelSize;


vec3 getPos(vec2 uv) {
    float depth = texture(screenTexture,uv).r;
    float x = (uv.x * 2.0 - 1.0) / projection[0][0];
    float y = (uv.y * 2.0 - 1.0) / projection[1][1];
    return vec3(x * -depth, y * -depth, depth);
}

void main() {
    float depth = texture(screenTexture, TexCoords).r;
    if(depth > 5000.0) discard;

    vec3 posCenter = getPos(TexCoords);
    vec3 posRight = getPos(TexCoords + vec2(texelSize.x,0.0));
    vec3 posTop = getPos(TexCoords + vec2(0.0,texelSize.y));

    vec3 dx = posRight - posCenter;
    vec3 dy = posTop - posCenter;
    vec3 normal = normalize(cross(dx,dy));

    vec2 refractUV = TexCoords + normal.xy * 0.05;
    vec3 bgCol = texture(backgroundTexture, refractUV).rgb;

    vec3 lightDir = normalize(vec3(0.5,0.5,1.0));
    float diff = max(dot(normal,lightDir),0.0);
    float spec = pow(max(dot(reflect(-lightDir, normal), vec3(0.0, 0.0, 1.0)), 0.0), 128.0);


    vec3 waterColor = vec3(0.1,0.5,0.8);

    float frensel = pow(1.0 - max(dot(normal, vec3(0.0,0.0,1.0)), 0.0), 4.0);

    vec3 finalColor = mix(bgCol, waterColor, 0.3 + frensel * 0.5);
    finalColor += vec3(spec);

    FragColor = vec4(finalColor, 1.0);
}