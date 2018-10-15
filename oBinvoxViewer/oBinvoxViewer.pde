import peasy.*;
import java.io.*;
PeasyCam cam;

Binvox binvox;
PShape model;

void setup() {
  size(1024, 1024, P3D);
  //cam = new PeasyCam(this, 100);
  model = loadShape("plane.obj");
  try {
    binvox = parseBinvox("plane32.binvox");
  } catch(Exception e) { print(e); }
}


void draw() {
  background(0);
  lights();
  
  float fov = PI/3.0;
  float cameraZ = (height/2.0) / tan(fov/2.0);
  perspective(fov, float(width)/float(height), cameraZ/1000.0, cameraZ*1000.0);
  
  camera(35.0, 35.0, 60.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0);
  
  scale(40);
  
  shape(model);
  
  //noStroke();
  //fill(255, 0, 0);
  //sphere(0.01);
  
  strokeWeight(1.0/32);
  stroke(255);
  noFill();
  box(1);
  
  //if (binvox != null) {
  //  int d = binvox.d;
  //  float size = 1.0 / d * binvox.scale;
    
  //  strokeWeight(size);
  //  stroke(0);
  //  fill(255);
    
  //  for (int x = 0; x < d; ++x) {
  //    for (int z = 0; z < d; ++z) {
  //      for (int y = 0; y < d; ++y) {
  //        int index = x*d*d + z*d + y;
  //        byte value = binvox.voxels[index];
  //        if (value > 0) {
  //          pushMatrix();
  //          float px = (float(x)+0.5) * size - d/2*size;
  //          float py = (float(y)+0.5) * size - d/2*size;
  //          float pz = (float(z)+0.5) * size - d/2*size;
  //          translate(px, py, pz);
  //          box(size);
  //          popMatrix();
  //        }
  //      }
  //    }
  //  }
  //}
}


// Binvox class contains grid information from .binvox file
class Binvox {
  byte[] voxels;     // voxel occupancy values
  int d, h, w;       // depth, width and height
  float tx, ty, tz;  // translation (not useful)
  float scale;       // scale from unit cube
}


Binvox parseBinvox(String path) throws Exception {
  path = sketchPath() + '\\' + path;
  FileInputStream binvox_file = new FileInputStream(path);
  DataInputStream binvox_data = new DataInputStream(binvox_file);
  
  byte[] voxels;
  int d, h, w;
  int size;
  float tx, ty, tz;
  float scale;
  
  // read header
  String line = binvox_data.readLine();  // deprecated function though
  if (!line.startsWith("#binvox")) {
    print("Error: first line reads [" + line + "] instead of [#binvox]\n");
    return null;
  }
  
  String version_string = line.substring(8);
  int version = Integer.parseInt(version_string);
  print("reading binvox version " + version, "\n");

  d = h = w = 0;
  tx = ty = tz = 0;
  scale = 1;
  boolean done = false;
  
  while(!done) {
    
    line = binvox_data.readLine();
    String[] dimensions = line.split(" ");
    
    if (line.startsWith("data")) {
      done = true;
    } else {
      if (line.startsWith("dim")) {
        d = Integer.parseInt(dimensions[1]);
        h = Integer.parseInt(dimensions[2]);
        w = Integer.parseInt(dimensions[3]);
        print(d, h, w, "\n");
      }
      else {
        if (line.startsWith("translate")) {
           tx = Float.parseFloat(dimensions[1]);
           ty = Float.parseFloat(dimensions[2]);
           tz = Float.parseFloat(dimensions[3]);
           print(tx, ty, tz, "\n");
        }
        else {
          if (line.startsWith("scale")) {
             scale = Float.parseFloat(dimensions[1]);
             print(scale, "\n");
          }
          else {
            print("unrecognized keyword [" + line + "], skipping\n");
          }
        }
      }
    }
  }  // while
  
  if (!done) {
    print("error reading header\n");
    return null;
  }
  if (d == 0) {
    print("missing dimensions in header\n");
    return null;
  }
  
  size = d * w * h;
  voxels = new byte[size];
  
  // read voxel data
  byte value;
  int count;
  int index = 0;
  int end_index = 0;
  int nr_voxels = 0;
  
  // *input >> value;  // read the linefeed char

  while(end_index < size) {

    value = binvox_data.readByte();
    // idiotic Java language doesn't have unsigned types, so we have to use an int for 'count'
    // and make sure that we don't interpret it as a negative number if bit 7 (the sign bit) is on
    count = binvox_data.readByte() & 0xff;

    end_index = index + count;
    if (end_index > size) return null;
    for(int i = index; i < end_index; i++) voxels[i] = value;

    if (value > 0) nr_voxels += count;
    index = end_index;
      
  }  // while
  
  print("read " + nr_voxels + " voxels\n");
  
  Binvox binvox = new Binvox();
  binvox.voxels = voxels;
  binvox.d = d;
  binvox.h = h;
  binvox.w = w;
  binvox.tx = tx;
  binvox.ty = ty;
  binvox.tz = tz;
  binvox.scale = scale;
  
  return binvox;
}
