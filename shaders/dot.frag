#version 330 core

in vec3 FragPos;
in vec3 Normal;

out vec4 FragColor;
uniform vec4 uColor;
uniform vec3 worldLightDir; // Kierunek światła w world space

void main() {
    // Oblicz normalną per-trójkąt (flat shading) używając pochodnych
    vec3 dFdxPos = dFdx(FragPos);
    vec3 dFdyPos = dFdy(FragPos);
    vec3 normal = normalize(cross(dFdxPos, dFdyPos));

    // Oblicz oświetlenie - prosty Lambertian diffuse
    float ambientStrength = 0.3;
    vec3 ambient = ambientStrength * uColor.rgb;

    // Światło kierunkowe (normalizuj kierunek światła)
    vec3 lightDirection = normalize(worldLightDir);
    float diff = max(dot(normal, lightDirection), 0.0);
    vec3 diffuse = diff * uColor.rgb;

    // Połącz ambient i diffuse
    vec3 result = ambient + diffuse;

    FragColor = vec4(result, uColor.a);
}