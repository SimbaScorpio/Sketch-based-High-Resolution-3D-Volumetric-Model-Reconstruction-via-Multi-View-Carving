import peasy.*;
import peasy.org.apache.commons.math.*;
import peasy.org.apache.commons.math.geometry.*;

String depthPath = "..\\ShapeNet_Data\\depths";
String cloudPath = "..\\ShapeNet_Data\\clouds";
Boolean saveObj = false;

String[] classes = {"chair"};

PeasyCam peasyCam;
PointCloudGenerator generator;
PointCloud pointCloud;
PShape cloudShape;

void setup() {
  size(1024, 1024, P3D);
  peasyCam = new PeasyCam(this, 2);
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
  float cameraZ = ((height/2.0) / tan(PI*60.0/360.0));
  perspective(PI/3.0, width/height, cameraZ/1000.0, cameraZ*10.0);
  if (cloudShape != null) {
    shape(cloudShape);
  }
  noFill();
  stroke(255);
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
