String modelPath = "../ShapeNet_Data/objects";
String depthPath = "../ShapeNet_Data/depths";
String fragShader = "frag_depth.glsl";
String vertShader = "vert_depth.glsl";

String[] classes = {"plane"};

ModelLoader modelLoader;
Model currentModel;
PShader shader;
PGraphics pg;
int index = 0;

void setup() {
  size(256, 256, P3D);
  pg = createGraphics(width, height, P3D);
  shader = loadShader(fragShader, vertShader);
  modelLoader = new ModelLoader(modelPath, classes);
  currentModel = modelLoader.next();
}

void draw() {
  // ensure no implicit edge smoothing operations!!!
  pg.noSmooth();
  // prepare depth rendering environment
  pg.beginDraw();
  pg.background(0);
  pg.shader(shader);
  pg.lights();
  
  pg.ortho(-0.5, 0.5, -0.5, 0.5, 0, 1);
  
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
    // save the image
    pg.save(depthPath + "/" + currentModel.type + "/" + currentModel.name + "/" + index + ".png");
    index += 1;
    
    // load next model after 6 viewpoints are rendered
    if (index == 6) {
      index = 0;
      currentModel = modelLoader.next();
    }
  }
}
