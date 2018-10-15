PGraphics pg;
PImage img;

boolean flag = false;

void setup() {
  size(256, 256, P3D);
  pg = createGraphics(width, height, P3D);
  img = loadImage("4.png");
  pg.image(img, 0, 0);
}

void draw() {
  background(0);
  image(img, 0, 0);
  if (!flag) {
    loadPixels();
    for (int i = 0; i < pixels.length; ++i) {
      float r = red(pixels[i]);
      if (r > 0) {
        pixels[i] = color(255);
      }
    }
    updatePixels();
    save("s4.png");
    flag = true;
  }
}
