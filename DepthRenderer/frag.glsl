#ifdef GL_ES
precision mediump float;
precision mediump int;
#endif

uniform vec3 AmbientColour;
uniform vec3 DiffuseColour;
uniform vec3 SpecularColour;
uniform float AmbientIntensity;
uniform float DiffuseIntensity;
uniform float SpecularIntensity;
uniform float Roughness;
uniform float Sharpness;

varying float count;
varying vec3 N;
varying vec3 P;
varying vec3 V;
varying vec3 L[8];

void main() {
    float w = 0.18*(1.0-Sharpness);
    
    vec3 n = normalize(N);
    vec3 v = normalize(V);
    
    vec3 color = vec3(0,0,0);
    for (int i = 0; i < int(count); ++i) {
        vec3 l = normalize(L[i]);
        vec3 h = normalize(l+v);
        
        float diffuse = dot(n,l);
        float specular = smoothstep(0.72-w, 0.72+w, pow(dot(n,h), 1.0/Roughness));
    
        color = color + vec3(DiffuseColour*DiffuseIntensity*diffuse +
                             SpecularColour*SpecularIntensity*specular);
    }
    gl_FragColor = vec4(AmbientColour*AmbientIntensity + color, 1);
}
