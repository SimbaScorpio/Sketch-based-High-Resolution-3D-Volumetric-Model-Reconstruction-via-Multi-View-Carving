#define PROCESSING_LIGHT_SHADER

uniform mat4 modelview;
uniform mat4 transform;
uniform mat3 normalMatrix;

uniform int lightCount;
uniform vec4 lightPosition[8];

attribute vec4 position;
attribute vec4 color;
attribute vec3 normal;

varying float count;
varying vec3 N;
varying vec3 P;
varying vec3 V;
varying vec3 L[8];

void main() {
    gl_Position = transform * position;
    P = position.xyz;
    V = -vec3(modelview * position);
    for(int i = 0; i < lightCount; ++i) {
        L[i] = vec3(modelview * (lightPosition[i] - position));
    }
    N = normalize(normalMatrix * normal);
    count = lightCount;
}
