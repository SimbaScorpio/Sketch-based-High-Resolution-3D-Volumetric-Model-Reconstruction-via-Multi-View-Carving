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
    
    // create a white image
    float[] temp = new float[img.pixels.length];
    for (int i = 0; i < img.pixels.length; ++i) {
      temp[i] = 255;
    }
    
    //// sobel edge detection
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
        float v1 =  depth(img.pixels[p1]);
        float v2 =  depth(img.pixels[p2]);
        float v3 =  depth(img.pixels[p3]);
        float v4 =  depth(img.pixels[p4]);
        float v5 =  depth(img.pixels[p5]);
        float v6 =  depth(img.pixels[p6]);
        float v7 =  depth(img.pixels[p7]);
        float v8 =  depth(img.pixels[p8]);
        float v9 =  depth(img.pixels[p9]);
        
        float vx = v1+2*v4+v7-v3-2*v6-v9;
        float vy = v1+2*v2+v3-v7-2*v8-v9;
        float v = sqrt(vx*vx+vy*vy);

        if (v > threshold) { temp[p5] = 0; }
      }
    }
    
    // restore pixel values
    for (int i = 0; i < img.pixels.length; ++i) {
      img.pixels[i] = color(temp[i]);
    }
    
    image(img, 0, 0);
    save(sketchPath() + "/" + output + "/" + depth.type + "/" + depth.name + "/" + index + ".png");
    index++;
  }
}

float depth(color pixel) {
  float r = red(pixel);
  float g = green(pixel);
  float b = blue(pixel);
  float a = alpha(pixel);
  float value = (a == 0) ? 51200 : ((b == 0) ? 511-r : 255-g);
  return value/2;
}
