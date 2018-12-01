// Model Loader for reading .binvox and .obj files
// Usage:
// * loader = new ModelLoader(dataPath, classes)
// 1. model = loader.next() enumerate all files in 'dataPath' directory specified in classes
// 2. model = loader.loadModel(modelPath) load specific model under 'modelPath'
class ModelLoader {
  PrintWriter output;
  java.io.FilenameFilter binvoxFilter;
  
  HashMap<String, File[]> dir = new HashMap<String, File[]>();
  String[] classes;
  int classIndex = -1;
  int modelIndex = -1;

  ModelLoader(String dataPath, String[] classes) {
    dataPath = sketchPath() + '/' + dataPath;
    output = createWriter("log.txt");
    output.print("dir: " + dataPath + "\r\n");
    
    this.classes = classes;

    binvoxFilter = new java.io.FilenameFilter() {
      boolean accept(File dir, String name) {
        return name.toLowerCase().endsWith(".binvox");
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
              dir.put(classes[i], tempClasses[j].listFiles(binvoxFilter));
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

  Binvox next() {
    if (classIndex == -1) {
      classIndex = 0;
    }
    while(classIndex < classes.length) {
      File[] modelFiles = dir.get(classes[classIndex]);
      // class directory is empty or doesn't exist
      if (modelFiles == null || modelFiles.length == 0) {
        modelIndex = -1;
        classIndex++;
        continue;
      }
      modelIndex++;
      if (modelIndex == modelFiles.length) {
        modelIndex = -1;
        classIndex++;
        continue;
      }
      break;
    }
    if (classIndex == classes.length) return null;
    
    File[] modelFiles = dir.get(classes[classIndex]);
    File vox = modelFiles[modelIndex];
    
    print("class: ", classIndex+1, " / ", classes.length, "\t");
    print("object: ", modelIndex+1, " / ", modelFiles.length, "\t");
    print(vox.getName(), "\n");
    
    Binvox binvox = null;
    try {
      binvox = parseBinvox(vox.getPath());
    } catch(Exception e) { 
      print(e);
      output.print(e);
      return null;
    }
    
    output.print("load model: " + vox.getName() + "\t");
    output.print("(" + binvox.w + "," + binvox.h + "," + binvox.d + ")\t");
    output.print("(" + binvox.tx + "," + binvox.ty + "," + binvox.tz + ")\t");
    output.print(binvox.scale + "\r\n");
    output.flush();
    
    return binvox;
  }
  
  Binvox prev() {
    if (classIndex == -1) return null;
    boolean jump = false;
    if (classIndex == classes.length) {
      jump = true;
      classIndex = classes.length - 1;
    }
    while(classIndex >= 0) {
      File[] modelFiles = dir.get(classes[classIndex]);
      // class directory is empty or doesn't exist
      if (modelFiles == null || modelFiles.length == 0) {
        jump = true;
        classIndex--;
        continue;
      }
      if (jump == true) {
        jump = false;
        modelIndex = modelFiles.length;
      }
      modelIndex--;
      if (modelIndex == -1) {
        jump = true;
        classIndex--;
        continue;
      }
      break;
    }
    if (classIndex == -1) return null;
    
    File[] modelFiles = dir.get(classes[classIndex]);
    File vox = modelFiles[modelIndex];
    
    print("class: ", classIndex+1, " / ", classes.length, "\t");
    print("object: ", modelIndex+1, " / ", modelFiles.length, "\t");
    print(vox.getName(), "\n");
    
    Binvox binvox = null;
    try {
      binvox = parseBinvox(vox.getPath());
    } catch(Exception e) { 
      print(e);
      output.print(e);
      return null;
    }
    
    output.print("load model: " + vox.getName() + "\t");
    output.print("(" + binvox.w + "," + binvox.h + "," + binvox.d + ")\t");
    output.print("(" + binvox.tx + "," + binvox.ty + "," + binvox.tz + ")\t");
    output.print(binvox.scale + "\r\n");
    output.flush();
    
    return binvox;
  }
}
