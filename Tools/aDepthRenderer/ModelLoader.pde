// Model class holding 3d informations
class Model {
  PShape shape;                   // processing 3d shape
  String name = "Default.obj";    // object name
  String type = "Default";        // object class
  int vertexes;
  float minX, minY, minZ;
  float maxX, maxY, maxZ;
}

// Model Loader for reading .obj files
// Usage:
// * loader = new ModelLoader(dataPath, classes)
// 1. model = loader.next() enumerate all files in 'dataPath' directory specified in classes
// 2. model = loader.loadModel(modelPath) load specific model under 'modelPath'
class ModelLoader {
  PrintWriter output;
  java.io.FilenameFilter objFilter;
  
  HashMap<String, File[]> dir = new HashMap<String, File[]>();
  String[] classes;
  int classIndex = 0;
  int modelIndex = 0;

  ModelLoader(String dataPath, String[] classes) {
    dataPath = sketchPath() + '/' + dataPath;
    output = createWriter("log.txt");
    output.print("dir: " + dataPath + "\r\n");
    
    this.classes = classes;

    objFilter = new java.io.FilenameFilter() {
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
              dir.put(classes[i], tempClasses[j].listFiles(objFilter));
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

  Model next() {
    if (classIndex == classes.length) return null;
    File[] modelFiles = dir.get(classes[classIndex]);
    if (modelFiles == null) { classIndex++; return null; }
    
    File obj = modelFiles[modelIndex];
    
    print("class: ", classIndex+1, " / ", classes.length, "\t");
    print("object: ", modelIndex+1, " / ", modelFiles.length, "\t");
    print(obj.getName(), "\n");
    
    Model m = loadModel(obj.getAbsolutePath());
    m.type = classes[classIndex];
    m.name = obj.getName();
    
    output.print("load model: " + obj.getName() + "\t");
    output.print(m.vertexes + "\t");
    output.print("(" + m.minX + "," + m.minY + "," + m.minZ + ")\t-\t");
    output.print("(" + m.maxX + "," + m.maxY + "," + m.maxZ + ")\r\n");
    output.flush();
    
    modelIndex++;
    if (modelIndex == modelFiles.length) {
      modelIndex = 0;
      classIndex++;
    }
    return m;
  }

  Model loadModel(String path) {
    PShape shape = loadShape(path);
    
    int vertexCount = 0;
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    minX = minY = minZ = Float.MAX_VALUE;
    maxX = maxY = maxZ = Float.MIN_VALUE;
    for(PShape child : shape.getChildren()) {
      int n = child.getVertexCount();
      vertexCount += n;
      for(int j = 0; j < n; ++j) {
        float x = child.getVertexX(j);
        float y = child.getVertexY(j);
        float z = child.getVertexZ(j);
        if (x < minX) minX = x;
        if (x > maxX) maxX = x;
        if (y < minY) minY = y;
        if (y > maxY) maxY = y;
        if (z < minZ) minZ = z;
        if (z > maxZ) maxZ = z;
      }
    }
    //float offsetX = (minX + maxX) / 2;
    //float offsetY = (minY + maxY) / 2;
    //float offsetZ = (minZ + maxZ) / 2;
    //float sizeX = maxX - minX;
    //float sizeY = maxY - minY;
    //float sizeZ = maxZ - minZ;
    //float ratio = (sizeX >= sizeY ? sizeX : sizeY);
    //ratio = (ratio >= sizeZ ? ratio : sizeZ);
      
    // translate to center
    //for(PShape child : shape.getChildren()) {
    //  int n = child.getVertexCount();
    //  for(int j = 0; j < n; ++j) {
    //    float x = (child.getVertexX(j) - offsetX) / ratio;
    //    float y = (child.getVertexY(j) - offsetY) / ratio;
    //    float z = (child.getVertexZ(j) - offsetZ) / ratio;
    //    child.setVertex(j, x, y, z);
    //  }
    //}

    //print("Child Count: " + shape.getChildCount() + "\n");
    //print("Vertex Count: " + vertexCount + "\n");
    
    Model model = new Model();
    model.shape = shape;
    model.vertexes = vertexCount;
    model.minX = minX;
    model.minY = minY;
    model.minZ = minZ;
    model.maxX = maxX;
    model.maxY = maxY;
    model.maxZ = maxZ;
    
    //model.minX = (minX - offsetX) / ratio;
    //model.minY = (minY - offsetY) / ratio;
    //model.minZ = (minZ - offsetZ) / ratio;
    //model.maxX = (maxX - offsetX) / ratio;
    //model.maxY = (maxY - offsetY) / ratio;
    //model.maxZ = (maxZ - offsetZ) / ratio;
    return model;
  }
}
