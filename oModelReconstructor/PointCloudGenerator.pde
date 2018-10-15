class PointCloud {
  String name;
  String type;
  ArrayList<PVector> points;
}

class PointCloudGenerator {
  PrintWriter output;
  java.io.FilenameFilter filter;
  
  HashMap<String, File[]> dir = new HashMap<String, File[]>();
  String[] classes;
  int classIndex = 0;
  int modelIndex = 0;
  
  PointCloudGenerator(String dataPath, String[] classes) {
    dataPath = sketchPath() + "/" + dataPath;
    output = createWriter("log.txt");
    output.print("dir: " + dataPath + "\r\n");
    
    this.classes = classes;
    
    filter = new java.io.FilenameFilter() {
      boolean accept(File dir, String name) {
        return name.toLowerCase().endsWith(".obj");
      }
    };
    
   File root = new File(dataPath);
   if (root.isDirectory()) {
     File[] tempClasses = root.listFiles();
     if (tempClasses != null) {
       for (int i = 0; i < classes.length; ++i) {
         dir.put(classes[i], null);
         for (int j = 0; j < tempClasses.length; ++j) {
           if (classes[i].equals(tempClasses[j].getName())) {
             dir.put(classes[i], tempClasses[j].listFiles(filter));
           }
         }
       }
     } else {
       output.print("empty dir: " + dataPath);
     }
    } else {
      output.print("error dir: " + dataPath);
    }
    output.flush();
  }
  
  PointCloud next() {
    if (classIndex == classes.length) return null;
    File[] modelFiles = dir.get(classes[classIndex]);
    if (modelFiles == null) { classIndex++; return null; }
    
    File objdir = modelFiles[modelIndex];
    
    String log = "class: " + (classIndex+1) + " / " + classes.length + "\t" +
                 "object: " + (modelIndex+1) + " / " + modelFiles.length + "\t" +
                 objdir.getName() + "\n";
    print(log);
    output.print(log);
    output.flush();
    
    ArrayList<PVector> points = reconstruct(objdir.getPath());
    PointCloud cloud = new PointCloud();
    cloud.name = objdir.getName();
    cloud.type = classes[classIndex];
    cloud.points = points;
    
    modelIndex++;
    if (modelIndex == modelFiles.length) {
      modelIndex = 0;
      classIndex++;
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
          //x = x + random(-0.02, 0.02);
          //y = y + random(-0.02, 0.02);
          //z = z + random(-0.02, 0.02);
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
    String log = "save: " + directory + "/" + type + "/" + name + "\n";
    print(log);
    output.print(log);
    output.flush();
  }
}
