#version 330 core

out vec4 FragColor;

in vec2 TexCoords;
uniform sampler2D depthMap;
uniform vec2 blurDir;
uniform float filterSize;
uniform float texelSize;
uniform float depthFalloff;

void main() {
    float centerDepth = texture(depthMap,TexCoords).r;
    if(centerDepth > 5000.0){
        FragColor = vec4(centerDepth);
        return;
    }

    float sum = 0.0;
    float wSum = 0.0;

    for(float i=-filterSize; i<=filterSize; i++){
        float sampleDepth = texture(depthMap,TexCoords + i * blurDir * texelSize).r;

        float r = i * 1.0/filterSize;
        float w = exp(-r * r);

        float r2 = (sampleDepth - centerDepth) * depthFalloff;
        w *= exp(-r2 * r2);
        sum += sampleDepth * w;
        wSum += w;
    }
    FragColor = vec4(sum / wSum);
}