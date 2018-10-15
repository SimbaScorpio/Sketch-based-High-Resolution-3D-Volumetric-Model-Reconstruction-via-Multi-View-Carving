String input = "../ShapeNet_Data/depths";
String output = "../ShapeNet_Data/silhouettes";

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
    
    for (int i = 0; i < pixels.length; ++i) {
      if (green(pixels[i]) > 0) { pixels[i] = color(255); }
    }
    
    updatePixels();
    save(sketchPath() + "/" + output + "/" + depth.type + "/" + depth.name + "/" + index + ".png");
    index++;
  }
}
