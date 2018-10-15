class Depth {
  String name;
  String type;
  String path;
}

class DepthLoader {
  PrintWriter output;
  java.io.FilenameFilter filter;
  
  HashMap<String, File[]> dir = new HashMap<String, File[]>();
  String[] classes;
  int classIndex = 0;
  int modelIndex = 0;
  
  DepthLoader(String dataPath, String[] classes) {
    dataPath = sketchPath() + "/" + dataPath;
    output = createWriter("log.txt");
    output.print("dir:" + dataPath + "\r\n");
    
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
  
  Depth next() {
    if (classIndex == classes.length) return null;
    File[] modelFiles = dir.get(classes[classIndex]);
    if (modelFiles == null || modelFiles.length == 0) { classIndex++; return null; }
    
    File objdir = modelFiles[modelIndex];
    Depth depth = new Depth();
    depth.type = classes[classIndex];
    depth.name = objdir.getName();
    depth.path = objdir.getPath();
    
    String log = "class: " + (classIndex+1) + " / " + classes.length + "\t" +
                 "object: " + (modelIndex+1) + " / " + modelFiles.length + "\t" +
                 objdir.getName() + "\n";
    print(log);
    output.print(log);
    output.flush();
    
    modelIndex++;
    if (modelIndex == modelFiles.length) {
      modelIndex = 0;
      classIndex++;
    }
    return depth;
  }
}
