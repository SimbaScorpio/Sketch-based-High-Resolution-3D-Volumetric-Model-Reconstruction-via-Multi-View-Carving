class PointCloud {
  String name;
  String type;
  ArrayList<PVector> points;
}

class PointCloudGenerator {
  PrintWriter output;
  java.io.FilenameFilter filter;
  
  File[] classes;
  int currentClassIndex;
  File[] currentClassFiles;
  int currentNameIndex;
  
  PointCloudGenerator(String directory) {
    directory = sketchPath() + '/' + directory;
    output = createWriter("cloud_generator_log.txt");
    output.print("dir: " + directory + "\n");
    
    filter = new java.io.FilenameFilter() {
      boolean accept(File dir, String name) {
        return name.toLowerCase().endsWith(".obj");
      }
    };
    
    File root = new File(directory);
    if (root.isDirectory()) {
      classes = root.listFiles();
      if (classes != null) {
        for (int i = 0; i < classes.length; ++i) {
          if (classes[i].isDirectory() && classes[i].listFiles(filter) != null) {
            currentClassIndex = i;
            currentClassFiles = classes[i].listFiles(filter);
            currentNameIndex = 0;
            break;
          }
        }
      } else {
        output.print("empty dir: " + directory);
      }
    } else {
      output.print("error dir: " + directory);
    }
    output.flush();
    output.close();
  }
  
  PointCloud next() {
    if (currentClassFiles == null) {
      return null;
    }
    File objdir = currentClassFiles[currentNameIndex];
    ArrayList<PVector> points = reconstruct(objdir.getPath());
    PointCloud cloud = new PointCloud();
    cloud.name = objdir.getName();
    cloud.type = classes[currentClassIndex].getName();
    cloud.points = points;
    
    currentNameIndex += 1;
    if (currentNameIndex == currentClassFiles.length) {
      currentClassFiles = null;
      currentNameIndex = -1;
      for (int i = currentClassIndex + 1; i < classes.length; ++i) {
        if (classes[i].isDirectory() && classes[i].listFiles(filter) != null) {
          currentClassIndex = i;
          currentClassFiles = classes[i].listFiles(filter);
          currentNameIndex = 0;
          break;
        }
      }
    }
    return cloud;
  }
  
  // reconstruct points from different depth maps
  ArrayList<PVector> reconstruct(String modelPath) {
    ArrayList<PVector> allPoints = new ArrayList<PVector>();
    for (int i = 0; i < 6; ++i) {
      String viewPath = modelPath + "/" + i + ".png";
      ArrayList<PVector> viewPoints = pointsFromView(viewPath);
      registration(viewPoints, i);
      allPoints.addAll(viewPoints);
    }
    return allPoints;
  }
  
  // reallocate points through depth values
  ArrayList<PVector> pointsFromView(String viewPath) {
    ArrayList<PVector> vectors = new ArrayList<PVector>();
    PImage img = loadImage(viewPath);
    for (int j = 0; j < img.height; ++j) {
      for (int i = 0; i < img.width; ++i) {
        int loc = i + j*img.width;
        // -0.5 ~ 0.5
        float z = red(img.pixels[loc]) / 255.0;
        float x = 1.0 / img.width * i - 0.5;
        float y = 1.0 / img.height * j - 0.5;
        if (z != 0.0) {
          PVector vec = new PVector(x, y, z - 0.5);
          vectors.add(vec);
        }
      }
    }
    return vectors;
  }
  
  // we assume each depth map has z in depth, but render in different viewpoints,
  // points need to be transformed to corresponding world coordinates
  void registration(ArrayList<PVector> viewPoints, int index) {
    if (index == 0) {
      for (int i = 0; i < viewPoints.size(); ++i) {
        PVector v = viewPoints.get(i);
        v.x = v.x;
        v.y = v.y;
        v.z = v.z;
      }
    } else if (index == 1) {
      for (int i = 0; i < viewPoints.size(); ++i) {
        PVector v = viewPoints.get(i);
        v.x = -v.x;
        v.y = v.y;
        v.z = -v.z;
      }
    } else if (index == 2) {
      for (int i = 0; i < viewPoints.size(); ++i) {
        PVector v = viewPoints.get(i);
        float temp = v.x;
        v.x = v.z;
        v.y = v.y;
        v.z = -temp;
      }
    } else if (index == 3) {
      for (int i = 0; i < viewPoints.size(); ++i) {
        PVector v = viewPoints.get(i);
        float temp = v.x;
        v.x = -v.z;
        v.y = v.y;
        v.z = temp;
      }
    } else if (index == 4) {
      for (int i = 0; i < viewPoints.size(); ++i) {
        PVector v = viewPoints.get(i);
        float temp = v.y;
        v.x = -v.x;
        v.y = -v.z;
        v.z = -temp;
      }
    } else if (index == 5) {
      for (int i = 0; i < viewPoints.size(); ++i) {
        PVector v = viewPoints.get(i);
        float temp = v.y;
        v.x = v.x;
        v.y = v.z;
        v.z = -temp;
      }
    }
  }
  
  // define how point cloud is shown in Processing
  PShape display(ArrayList<PVector> vectors) {
    if (vectors == null) {
      return null;
    }
    PShape cloud = createShape();
    cloud.beginShape(POINTS);
    cloud.strokeWeight(1);
    //cloud.stroke(255, 255, 255);
    for (int i = 0; i < vectors.size(); ++i) {
      PVector v = vectors.get(i);
      //float cx = 0.45 + 0.35*sin( (0.05,0.08,0.10)*(v.y-1.0) );
      float cx = 0.45 + 0.35*sin( 0.05*v.x * 180 / PI );
      float cy = 0.45 + 0.35*sin( 0.08*v.y * 180 / PI );
      float cz = 0.45 + 0.35*sin( 0.10*v.z * 180 / PI );
      cx = cx / (cx+cy+cz) * 2;
      cy = cy / (cx+cy+cz) * 2;
      cz = cz / (cx+cy+cz) * 2;
      cloud.stroke(cx*255, cy*255, cz*255);
      cloud.vertex(v.x, v.y, v.z);
    }
    cloud.endShape();
    return cloud;
  }
  
  // save points as .obj format
  void save(String directory, String type, String name, ArrayList<PVector> vectors) {
    if (vectors == null) { return; }
    
    directory = sketchPath() + '/' + directory;
    File obj = new File(directory + "/" + type + "/" + name);
    obj.getParentFile().mkdirs();
    try { 
      obj.createNewFile();
      java.io.FileOutputStream out = new java.io.FileOutputStream(obj);
      
      String data = "o " + name + "\n";
      out.write(data.getBytes("gbk"));
      
      for (int i = 0; i < vectors.size(); ++i) {
        PVector v = vectors.get(i);
        data = "v " + v.x + " " + v.y + " " + v.z + "\n";
        out.write(data.getBytes("gbk"));
      }
      
      out.flush();
      out.close();
    } catch (IOException ex) { 
       System.err.format("IOException: %s%n", ex);
    }
    print("save: " + directory + "/" + type + "/" + name + "\n");
  }
}
