// Model class holding 3d informations
class Model {
  PShape shape;   // processing 3d shape
  String name;    // object name
  String type;    // object class
  float minX, minY, minZ;
  float maxX, maxY, maxZ;
}

// Model Loader for reading .obj files
// Usage:
// * loader = new ModelLoader(dataPath)
// 1. model = loader.next() enumerate all files in 'dataPath' directory
// 2. model = loader.loadMdel(modelPath) load specific model under 'modelPath'
class ModelLoader {
  PrintWriter output;
  java.io.FilenameFilter objFilter;
  
  File[] classes;
  int currentClassIndex;
  File[] currentClassFiles;
  int currentNameIndex;

  ModelLoader(String directory) {
    directory = sketchPath() + '/' + directory;
    output = createWriter("model_loader_log.txt");
    output.print("dir: " + directory + "\n");

    objFilter = new java.io.FilenameFilter() {
      boolean accept(File dir, String name) {
        return name.toLowerCase().endsWith(".obj");
      }
    };

    File root = new File(directory);
    if (root.isDirectory()) {
      classes = root.listFiles();
      if (classes != null) {
        for (int i = 0; i < classes.length; ++i) {
          if (classes[i].isDirectory() && classes[i].listFiles(objFilter) != null) {
            currentClassIndex = i;
            currentClassFiles = classes[i].listFiles(objFilter);
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

  Model next() {
    if (currentClassFiles == null) {
      return null;
    }
    File obj = currentClassFiles[currentNameIndex];
    Model m = loadModel(obj.getAbsolutePath());
    m.type = classes[currentClassIndex].getName();
    m.name = obj.getName();
    
    currentNameIndex += 1;
    if (currentNameIndex == currentClassFiles.length) {
      currentClassFiles = null;
      currentNameIndex = -1;
      for (int i = currentClassIndex + 1; i < classes.length; ++i) {
        if (classes[i].isDirectory() && classes[i].listFiles(objFilter) != null) {
          currentClassIndex = i;
          currentClassFiles = classes[i].listFiles(objFilter);
          currentNameIndex = 0;
          break;
        }
      }
    }
    return m;
  }

  Model loadModel(String path) {
    print("load model: " + path + "\n");
    PShape shape = loadShape(path);
    
    // calculate center point
    //int vertexCount = 0;
    float minX, minY, minZ;
    float maxX, maxY, maxZ;
    minX = minY = minZ = Float.MAX_VALUE;
    maxX = maxY = maxZ = Float.MIN_VALUE;
    for(PShape child : shape.getChildren()) {
      int n = child.getVertexCount();
      //vertexCount += n;
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
    model.minX = minX;
    model.minY = minY;
    model.minZ = minZ;
    model.maxX = maxX;
    model.maxY = maxY;
    model.maxZ = maxZ;
    //print(minX + "," + maxX + "\n");
    //print(minY + "," + maxY + "\n");
    //print(minZ + "," + maxZ + "\n");
    //model.minX = (minX - offsetX) / ratio;
    //model.minY = (minY - offsetY) / ratio;
    //model.minZ = (minZ - offsetZ) / ratio;
    //model.maxX = (maxX - offsetX) / ratio;
    //model.maxY = (maxY - offsetY) / ratio;
    //model.maxZ = (maxZ - offsetZ) / ratio;
    return model;
  }
}
