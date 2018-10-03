import peasy.*;
import peasy.org.apache.commons.math.*;
import peasy.org.apache.commons.math.geometry.*;

PeasyCam peasyCam;
PointCloudGenerator generator;
PointCloud pointCloud;
PShape cloudShape;

String depthPath = "../Data/depths";
String cloudPath = "../Data/clouds";
Boolean saveObj = false;

void setup() {
  size(512, 512, P3D);
  peasyCam = new PeasyCam(this, 1);
  generator = new PointCloudGenerator(depthPath);
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
