#ifdef GL_ES
precision mediump float;
precision mediump int;
#endif

void main() {
    float depth = 1.0 - gl_FragCoord.z;
    gl_FragColor = vec4(depth, depth, depth, 1);
}
