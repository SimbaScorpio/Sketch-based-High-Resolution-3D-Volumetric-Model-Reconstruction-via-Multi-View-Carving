PShape getVoxelShape(Binvox vox) {
  if (vox == null) return null;
  int d = binvox.d;
  float size = 1.0 / d * binvox.scale;
  
  noStroke();
  fill(255, 255, 255, 128);
  
  byte[][][] grid = new byte[d][d][d];
  for (int x = 0; x < d; ++x) {
    for (int z = 0; z < d; ++z) {
      for (int y = 0; y < d; ++y) {
        int index = x*d*d + z*d + y;
        grid[x][y][z] = binvox.voxels[index];
      }
    }
  }
  
  PShape shape = grid2mesh(grid, d);
  
  for (int i = 0; i < shape.getVertexCount(); ++i) {
    PVector v = shape.getVertex(i);
    v.x = v.x*size - d/2*size;
    v.y = v.y*size - d/2*size;
    v.z = v.z*size - d/2*size;
    shape.setVertex(i, v);
  }
  
  return shape;
}

PShape grid2mesh(byte[][][] grid, int d) {
  
  PShape shape = createShape();
  
  float[][] v = {{0, 0, 0}, {1, 0, 0}, {1, 0, 1}, {0, 0, 1},
                 {0, 1, 0}, {1, 1, 0}, {1, 1, 1}, {0, 1, 1}};
  
  // extend 1 voxel for easy boundry neighbour checking
  byte[][][] g = new byte[d+2][d+2][d+2];
  for (int x = 0; x < d; ++x) {
    for (int y = 0; y < d; ++y) {
      for (int z = 0; z < d; ++z) {
        g[x+1][y+1][z+1] = grid[x][y][z];
      }
    }
  }
  
  shape.beginShape(TRIANGLES);
  for (int x = 1; x < d+1; ++x) {
    for (int y = 1; y < d+1; ++y) {
      for (int z = 1; z < d+1; ++z) {
        if (g[x][y][z] == 0) continue;
        // front
        if (g[x+1][y][z] == 0) {
          // 1-5-6
          shape.vertex(x-1+v[1][0], y-1+v[1][1], z-1+v[1][2]);
          shape.vertex(x-1+v[5][0], y-1+v[5][1], z-1+v[5][2]);
          shape.vertex(x-1+v[6][0], y-1+v[6][1], z-1+v[6][2]);
          // 2-1-6
          shape.vertex(x-1+v[2][0], y-1+v[2][1], z-1+v[2][2]);
          shape.vertex(x-1+v[1][0], y-1+v[1][1], z-1+v[1][2]);
          shape.vertex(x-1+v[6][0], y-1+v[6][1], z-1+v[6][2]);
        }
        
        // back
        if (g[x-1][y][z] == 0) {
          // 0-7-4
          shape.vertex(x-1+v[0][0], y-1+v[0][1], z-1+v[0][2]);
          shape.vertex(x-1+v[7][0], y-1+v[7][1], z-1+v[7][2]);
          shape.vertex(x-1+v[4][0], y-1+v[4][1], z-1+v[4][2]);
          // 0-3-7
          shape.vertex(x-1+v[0][0], y-1+v[0][1], z-1+v[0][2]);
          shape.vertex(x-1+v[3][0], y-1+v[3][1], z-1+v[3][2]);
          shape.vertex(x-1+v[7][0], y-1+v[7][1], z-1+v[7][2]);
        }
        
        // left
        if (g[x][y][z-1] == 0) {
          // 0-4-5
          shape.vertex(x-1+v[0][0], y-1+v[0][1], z-1+v[0][2]);
          shape.vertex(x-1+v[4][0], y-1+v[4][1], z-1+v[4][2]);
          shape.vertex(x-1+v[5][0], y-1+v[5][1], z-1+v[5][2]);
          // 0-5-1
          shape.vertex(x-1+v[0][0], y-1+v[0][1], z-1+v[0][2]);
          shape.vertex(x-1+v[5][0], y-1+v[5][1], z-1+v[5][2]);
          shape.vertex(x-1+v[1][0], y-1+v[1][1], z-1+v[1][2]);
        }
        
        // right
        if (g[x][y][z+1] == 0) {
          // 2-6-7
          shape.vertex(x-1+v[2][0], y-1+v[2][1], z-1+v[2][2]);
          shape.vertex(x-1+v[6][0], y-1+v[6][1], z-1+v[6][2]);
          shape.vertex(x-1+v[7][0], y-1+v[7][1], z-1+v[7][2]);
          // 2-7-3
          shape.vertex(x-1+v[2][0], y-1+v[2][1], z-1+v[2][2]);
          shape.vertex(x-1+v[7][0], y-1+v[7][1], z-1+v[7][2]);
          shape.vertex(x-1+v[3][0], y-1+v[3][1], z-1+v[3][2]);
        }
        
        // top
        if (g[x][y+1][z] == 0) {
          // 5-4-7
          shape.vertex(x-1+v[5][0], y-1+v[5][1], z-1+v[5][2]);
          shape.vertex(x-1+v[4][0], y-1+v[4][1], z-1+v[4][2]);
          shape.vertex(x-1+v[7][0], y-1+v[7][1], z-1+v[7][2]);
          // 5-7-6
          shape.vertex(x-1+v[5][0], y-1+v[5][1], z-1+v[5][2]);
          shape.vertex(x-1+v[7][0], y-1+v[7][1], z-1+v[7][2]);
          shape.vertex(x-1+v[6][0], y-1+v[6][1], z-1+v[6][2]);
        }
        
        // bottom
        if (g[x][y-1][z] == 0) {
          // 0-1-3
          shape.vertex(x-1+v[0][0], y-1+v[0][1], z-1+v[0][2]);
          shape.vertex(x-1+v[1][0], y-1+v[1][1], z-1+v[1][2]);
          shape.vertex(x-1+v[3][0], y-1+v[3][1], z-1+v[3][2]);
          // 1-2-3
          shape.vertex(x-1+v[1][0], y-1+v[1][1], z-1+v[1][2]);
          shape.vertex(x-1+v[2][0], y-1+v[2][1], z-1+v[2][2]);
          shape.vertex(x-1+v[3][0], y-1+v[3][1], z-1+v[3][2]);
        }
      }
    }
  }
  shape.endShape();
  
  return shape;
}

void drawCoord() {
  // center point
  noStroke();
  fill(255, 0, 0);
  sphere(0.01);
  
  // bounding box
  strokeWeight(1.0/32);
  stroke(255);
  noFill();
  box(1);
  
  beginShape(LINES);
  
  // x-axis
  stroke(255, 0, 0);
  vertex(0, 0, 0);
  vertex(1, 0, 0);
  
  // y-axis
  stroke(0, 255, 0);
  vertex(0, 0, 0);
  vertex(0, 1, 0);
  
  // z-axis
  stroke(0, 0, 255);
  vertex(0, 0, 0);
  vertex(0, 0, 1);
  
  endShape();
}
