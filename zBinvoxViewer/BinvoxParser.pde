// Binvox class contains grid information from .binvox file
class Binvox {
  String name;
  String type;
  byte[] voxels;     // voxel occupancy values
  int d, h, w;       // depth, width and height
  float tx, ty, tz;  // translation (not useful)
  float scale;       // scale to unit cube
}


Binvox parseBinvox(String path) throws Exception {
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
  
  String[] blocks = path.split("\\\\");
  binvox.name = blocks[blocks.length-1].split(".binvox")[0];
  binvox.type = blocks[blocks.length-2];
  
  return binvox;
}
