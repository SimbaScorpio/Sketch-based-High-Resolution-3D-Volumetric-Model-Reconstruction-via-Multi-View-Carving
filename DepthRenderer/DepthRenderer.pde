import peasy.*;
import peasy.org.apache.commons.math.*;
import peasy.org.apache.commons.math.geometry.*;

PeasyCam peasyCam;
ModelLoader modelLoader;
Model currentModel;
PShader shader;
PGraphics pg;

String modelPath = "../Data/objects";
String depthPath = "../Data/depths";
String fragShader = "frag_depth.glsl";
String vertShader = "vert_depth.glsl";

int index = 0;

void setup() {
  size(256, 256, P3D);
  pg = createGraphics(width, height, P3D);
  //peasyCam = new PeasyCam(this, 1);
  modelLoader = new ModelLoader(modelPath);
  currentModel = modelLoader.next();
  shader = loadShader(fragShader, vertShader);
}

void draw() {
  // ensure no implicit edge smoothing operations!!!
  pg.noSmooth();
  // prepare depth rendering environment
  pg.beginDraw();
  pg.background(0);
  pg.shader(shader);
  pg.lights();
  
  //float cameraZ = ((height/2.0) / tan(PI*60.0/360.0));
  //perspective(PI/3.0, width/height, cameraZ/1000.0, cameraZ*10.0);
  pg.ortho(-0.5, 0.5, -0.5, 0.5, 0, 1);
  
  // 6 viewports
  if (index == 6) {
    index = 0;
    currentModel = modelLoader.next();
  }
  
  if (currentModel != null) {
    if (index == 0) {
      pg.camera(0.5, 0, 0, 0, 0, 0, 0, -1, 0);
    } else if (index == 1) {
      pg.camera(-0.5, 0, 0, 0, 0, 0, 0, -1, 0);
    } else if (index == 2) {
      pg.camera(0, 0, 0.5, 0, 0, 0, 0, -1, 0);
    } else if (index == 3) {
      pg.camera(0, 0, -0.5, 0, 0, 0, 0, -1, 0);
    } else if (index == 4) {
      pg.camera(0, 0.5, 0, 0, 0, 0, -1, 0, 0);
    } else if (index == 5) {
      pg.camera(0, -0.5, 0, 0, 0, 0, -1, 0, 0);
    }
    pg.shape(currentModel.shape, 0, 0);
    pg.endDraw();
    // draw the image to the screen at coordinate (0,0)
    image(pg, 0, 0);
    // save the image to the disk
    pg.save(depthPath + "/" + currentModel.type + "/" + currentModel.name + "/" + index + ".png");
    index += 1;
  }
}

void keyPressed() {
  if (key == '1') {
    currentModel = modelLoader.next();
  }
}
