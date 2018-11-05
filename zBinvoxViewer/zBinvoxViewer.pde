import peasy.*;
import java.io.*;
PeasyCam cam;

//String voxPath = "../ShapeNet_Data/binvox256/";
String voxPath = "../ShapeNet_Data/test/output/";
String objPath = "../ShapeNet_Data/objects/";
String[] classes = {"chair", "plane"};

ModelLoader loader;
Binvox binvox;
PShape shape;

PShape model;

void setup() {
  size(1024, 1024, P3D);
  cam = new PeasyCam(this, 100);
  loader = new ModelLoader(voxPath, classes);
  loadNext();
}

void draw() {
  background(0);
  directionalLight(80, 80, 80, 1, 1, 1);
  directionalLight(80, 80, 80, 1, 1, -1);
  directionalLight(80, 80, 80, -1, -1, 0);
  ambientLight(102, 102, 102);
  
  float fov = PI/3.0;
  float cameraZ = (height/2.0) / tan(fov/2.0);
  perspective(fov, float(width)/float(height), cameraZ/1000.0, cameraZ*1000.0);
  
  rotateX(PI);
  rotateY(PI/2);
  scale(100);
  
  drawCoord();
  if (shape != null)
    shape(shape);
  
  shape(model);
}

void keyPressed() {
  loadNext();
}

void loadNext() {
  binvox = loader.next();
  shape = getVoxelShape(binvox);
  if (binvox != null) {
    model = loadShape(objPath + binvox.type + "/" + binvox.name + ".obj");
  }
}
