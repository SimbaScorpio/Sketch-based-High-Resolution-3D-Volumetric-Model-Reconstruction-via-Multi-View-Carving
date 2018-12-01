#ifdef GL_ES
precision mediump float;
precision mediump int;
#endif

void main() {
	// I want to make the center of the depth map brighter than the edges.
	// Perhaps it would help the cnn network focus more on the object rather than the blurry edges.
    float depth = 1 - gl_FragCoord.z;

    // Since I am using an 8-bit png image storing the depth values, it could only store 256 blocks for a single channel.
    // I use G channel represent the nearest 256 blocks.
    // I use R channel represent the furthest 256 blocks.
    // I use B channel as a flag to indicate whether G channel is activated.
    // It could typically store up to 512 blocks for one depth map pixel now.
    if (depth > 0.5 && depth <= 1) {
    	gl_FragColor = vec4(1, (depth-0.5)*2, 1, 1);
    } else if (depth >= 0 && depth <= 0.5) {
    	gl_FragColor = vec4(depth*2, 0, 0, 1);
    } else {
    	gl_FragColor = vec4(0, 0, 0, 0);
    }
    // To recover the z-depth: 
    // if (b == 0):
    // 	index: 511-r    distance: (index+0.5)/512
    // if (b == 1):
    //  index: 255-g	distance: (index+0.5)/512
}
