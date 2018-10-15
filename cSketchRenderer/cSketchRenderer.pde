String input = "../ShapeNet_Data/depths";
String output = "../ShapeNet_Data/sketches";

String[] classes = {"plane"};

DepthLoader loader;
int threshold = 100;
int index = 6;
Depth depth;

void setup() {
  size(256, 256, P3D);
  loader = new DepthLoader(input, classes);
}

void draw() {
  background(255);
  
  if (index >= 6) {
    index = 0;
    depth = loader.next();
  }
  
  if (depth != null) {
    String path = depth.path + "/" + index + ".png";
    PImage img = loadImage(path);
    image(img, 0, 0);
    loadPixels();
    
    // create a white image
    float[] temp = new float[pixels.length];
    for (int i = 0; i < pixels.length; ++i) {
      temp[i] = 255;
    }
    
    // sobel edge detection
    for (int i = 1; i < 255; ++i) {
      for (int j = 1; j < 255; ++j) {
        int p5 = i*256 + j;
        int p2 = p5-256;
        int p8 = p5+256;
        int p1 = p2-1;
        int p3 = p2+1;
        int p4 = p5-1;
        int p6 = p5+1;
        int p7 = p8-1;
        int p9 = p8+1;
        float v1 =  red(pixels[p1]);
        float v2 =  red(pixels[p2]);
        float v3 =  red(pixels[p3]);
        float v4 =  red(pixels[p4]);
        float v5 =  red(pixels[p5]);
        float v6 =  red(pixels[p6]);
        float v7 =  red(pixels[p7]);
        float v8 =  red(pixels[p8]);
        float v9 =  red(pixels[p9]);
        
        float vx = v1+2*v4+v7-v3-2*v6-v9;
        float vy = v1+2*v2+v3-v7-2*v8-v9;
        float v = sqrt(vx*vx+vy*vy);

        if (v > threshold) { temp[p5] = 0; }
      }
    }
    
    // restore pixel values
    for (int i = 0; i < pixels.length; ++i) {
      pixels[i] = color(temp[i]);
    }
    
    updatePixels();
    save(sketchPath() + "/" + output + "/" + depth.type + "/" + depth.name + "/" + index + ".png");
    index++;
  }
}
