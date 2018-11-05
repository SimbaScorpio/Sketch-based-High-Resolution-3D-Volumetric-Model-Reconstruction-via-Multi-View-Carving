import peasy.*;
import peasy.org.apache.commons.math.*;
import peasy.org.apache.commons.math.geometry.*;

String depthPath = "../ShapeNet_Data/depths";
String cloudPath = "../ShapeNet_Data/clouds";
Boolean saveObj = false;

String[] classes = {"plane"};

PeasyCam peasyCam;
PointCloudGenerator generator;
PointCloud pointCloud;
PShape cloudShape;

void setup() {
  size(1024, 1024, P3D);
  peasyCam = new PeasyCam(this, 100);
  generator = new PointCloudGenerator(depthPath, classes);
  PointCloud pointCloud = generator.next();
  if (pointCloud != null) {
    if (saveObj)
      generator.save(cloudPath, pointCloud.type, pointCloud.name, pointCloud.points);
    cloudShape = generator.display(pointCloud.points);
  }
}

void draw() {
  background(0);

  float fov = PI/3.0;
  float cameraZ = (height/2.0) / tan(fov/2.0);
  perspective(fov, float(width)/float(height), cameraZ/1000.0, cameraZ*1000.0);

  scale(100);

  if (cloudShape != null) {
    shape(cloudShape);
  }
  
  // bounding box
  strokeWeight(1.0/32);
  stroke(255);
  noFill();
  box(1);
}

void keyPressed() {
  if (key == '1') {
    PointCloud pointCloud = generator.next();
    if (pointCloud != null) {
      if (saveObj)
        generator.save(cloudPath, pointCloud.type, pointCloud.name, pointCloud.points);
      cloudShape = generator.display(pointCloud.points);
    }
  }
}
