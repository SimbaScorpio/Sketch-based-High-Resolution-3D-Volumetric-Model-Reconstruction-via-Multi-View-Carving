import os
import math
import numpy as np
import binvox_rw
from PIL import Image
from scipy import ndimage


def img2grid(x, y, z, dim, axis):
	if axis == 0: return [dim-1-z, dim-1-y, x]
	if axis == 1: return [z, dim-1-y, dim-1-x]
	if axis == 2: return [dim-1-x, dim-1-y, dim-1-z]
	if axis == 3: return [x, dim-1-y, z]
	if axis == 4: return [dim-1-y, dim-1-z, dim-1-x]
	if axis == 5: return [dim-1-y, z, x]


def grid2img(x, y, z, dim, axis):
	if axis == 0: return [z, dim-1-y, dim-1-x]
	if axis == 1: return [dim-1-z, dim-1-y, x]
	if axis == 2: return [dim-1-x, dim-1-y, dim-1-z]
	if axis == 3: return [x, dim-1-y, z]
	if axis == 4: return [dim-1-z, dim-1-x, dim-1-y]
	if axis == 5: return [z, dim-1-x, y]


def bounddim(grid, dim, axis):
	minimum = dim
	maximum = -1
	for i in range(dim):
		if axis == 0:
			a, b = np.where(grid[i,:,:] == True)
		if axis == 1:
			a, b = np.where(grid[:,i,:] == True)
		if axis == 2:
			a, b = np.where(grid[:,:,i] == True)
		if len(a) != 0:
			if i < minimum: minimum = i
			if i > maximum: maximum = i
	return minimum, maximum


def bound(grid, dim):
	mx1, mx2 = bounddim(grid, dim, 0)
	my1, my2 = bounddim(grid, dim, 1)
	mz1, mz2 = bounddim(grid, dim, 2)
	print('x: ', mx1, mx2)
	print('y: ', my1, my2)
	print('z: ', mz1, mz2)
	return mx1, mx2, my1, my2, mz1, mz2


# def transCoord(v, dim, scale):
# 	v = v - dim/2
# 	if v < 0:
# 		v = -(v + 1)
# 		v = -math.floor((v + 0.5)*scale) + dim/2 - 1
# 	else:
# 		v = math.floor((v + 0.5)*scale) + dim/2
# 	return int(v)


# def normalize_grid(grid, dim, scale):
# 	new_grid = np.zeros((dim, dim, dim), dtype=bool)
# 	a, b, c = np.where(grid == True)
# 	for x, y, z in zip(a, b, c):
# 		x = transCoord(x, dim, scale)
# 		y = transCoord(y, dim, scale)
# 		z = transCoord(z, dim, scale)
# 		new_grid[x, y, z] = True
# 	return new_grid



# extracts odms from an object 
def odm(data, dim): 
	a,b,c = np.where(data == 1)
	large = int(dim *1.5)
	big_list = [[[[-1,large]for j in range(dim)] for i in range(dim)] for k in range(3)]
	# over the whole object extract for each face the first and last occurance of a voxel at each pixel
	# we take highest for convinience
	for i,j,k in zip(a,b,c):
		big_list[0][k][dim-1-j][0] = max(i, big_list[0][k][dim-1-j][0])
		big_list[0][k][dim-1-j][1] = min(i, big_list[0][k][dim-1-j][1])
		big_list[1][dim-1-i][dim-1-j][0] = max(k, big_list[1][dim-1-i][dim-1-j][0])
		big_list[1][dim-1-i][dim-1-j][1] = min(k, big_list[1][dim-1-i][dim-1-j][1])
		big_list[2][dim-1-k][dim-1-i][0] = max(j, big_list[2][dim-1-k][dim-1-i][0])
		big_list[2][dim-1-k][dim-1-i][1] = min(j, big_list[2][dim-1-k][dim-1-i][1])

	faces = np.zeros((6,dim,dim)) # will hold odms 
	for i in range(dim): 
		for j in range(dim): 
			faces[0,i,j] =   dim - 1 - big_list[0][i][j][0]     if    big_list[0][i][j][0]   		> -1 	else dim
			faces[1,i,j] =   big_list[0][dim-1-i][j][1]        	if    big_list[0][dim-1-i][j][1]   	< large else dim 
			faces[2,i,j] =   dim - 1 - big_list[1][i][j][0]     if    big_list[1][i][j][0]   		> -1 	else dim
			faces[3,i,j] =   big_list[1][dim-1-i][j][1]        	if    big_list[1][dim-1-i][j][1]   	< large else dim
			faces[4,i,j] =   dim -  1 - big_list[2][i][j][0]    if    big_list[2][i][j][0]   		> -1 	else dim
			faces[5,i,j] =   big_list[2][dim-1-i][j][1]        	if    big_list[2][dim-1-i][j][1]  	< large else dim
	return faces


# def calculate_depth(grid, dim, scale):
# 	# grid = normalize_grid(grid, dim, scale)
# 	depths = np.zeros((6, dim, dim), dtype='uint16')
# 	for axis in range(6):
# 		depths[axis] = depths[axis] + dim

# 	a, b, c = np.where(grid == True)
# 	for x, y, z in zip(a, b, c):
# 		for axis in range(6):
# 			m, n, d = grid2img(x, y, z, dim, axis)
# 			if d < depths[axis, m, n]:
# 				depths[axis, m, n] = d
# 	return depths


def get_silhouettes(depths, dim):
	silhouettes = np.zeros((6, dim, dim), dtype=bool)
	for axis in range(6):
		a, b = np.where(depths[axis] < dim)
		for x, y in zip(a, b):
			silhouettes[axis][x, y] = True
	return silhouettes
	

def render_depth(depth, dim):
	img = np.zeros((dim, dim, 4), dtype='uint8')
	a, b = np.where( (depth < dim) & (depth <= 255) )
	for x, y in zip(a, b):
		img[x, y] = np.array([255, 255 - depth[x, y], 255, 255])
	a, b = np.where( (depth < dim) & (depth > 255) )
	for x, y in zip(a, b):
		img[x, y] = np.array([511 - depth[x, y], 0, 0, 255])
	return img


def render_sketch(sketch, dim):
	img = np.full((dim, dim, 4), 255, dtype='uint8')
	a, b = np.where(sketch == True)
	for x, y in zip(a, b):
		img[x, y] = np.array([0, 0, 0, 255])
	return img


def save_binvox(voxel, path):
	assert( len(voxel.shape) == 3 )
	vox = binvox_rw.Voxels(voxel, dims=[voxel.shape[0],voxel.shape[1],voxel.shape[2]], translate=[0,0,0], scale=1.0, axis_order='xyz')
	with open(path, 'wb') as f:
		vox.write(f)


# this method creates a mesh copy of a voxel model
# it is efficinet as only exposed faces are drawn
#`could be written a tad better, but I think this way is the easiest to read 
def voxel2mesh(voxels):
    # these faces and verticies define the various sides of a cube
    top_verts = [[0,0,1], [1,0,1], [1,1,1], [0,1,1]]    
    top_faces = [[1,2,4], [2,3,4]] 

    bottom_verts = [[0,0,0], [1,0,0], [1,1,0], [0,1,0]]    
    bottom_faces = [[2,1,4], [3,2,4]] 

    left_verts = [[0,0,0], [0,0,1], [0,1,0], [0,1,1]]
    left_faces = [[1,2,4], [3,1,4]]

    right_verts = [[1,0,0], [1,0,1], [1,1,0], [1,1,1]]
    right_faces = [[2,1,4], [1,3,4]]

    front_verts = [[0,1,0], [1,1,0], [0,1,1], [1,1,1]]
    front_faces = [[2,1,4], [1,3,4]]

    back_verts = [[0,0,0], [1,0,0], [0,0,1], [1,0,1]]
    back_faces = [[1,2,4], [3,1,4]]

    top_verts = np.array(top_verts)
    top_faces = np.array(top_faces)
    bottom_verts = np.array(bottom_verts)
    bottom_faces = np.array(bottom_faces)
    left_verts = np.array(left_verts)
    left_faces = np.array(left_faces)
    right_verts = np.array(right_verts)
    right_faces = np.array(right_faces)
    front_verts = np.array(front_verts)
    front_faces = np.array(front_faces)
    back_verts = np.array(back_verts)
    back_faces = np.array(back_faces)

    dim = voxels.shape[0]
    new_voxels = np.zeros((dim+2, dim+2, dim+2))
    new_voxels[1:dim+1,1:dim+1,1:dim+1 ] = voxels
    voxels= new_voxels

    scale = 0.01
    cube_dist_scale = 1.
    verts = []
    faces = []
    curr_vert = 0
    a,b,c= np.where(voxels==True)
    for i,j,k in zip(a,b,c):
        #top
        if voxels[i,j,k+1]==False: 
            verts.extend(scale * (top_verts + cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(top_faces + curr_vert)
            curr_vert += len(top_verts)
    
        #bottom
        if voxels[i,j,k-1]==False: 
            verts.extend(scale * (bottom_verts + cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(bottom_faces + curr_vert)
            curr_vert += len(bottom_verts)
            
        #left
        if voxels[i-1,j,k]==False: 
            verts.extend(scale * (left_verts+ cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(left_faces + curr_vert)
            curr_vert += len(left_verts)
            
        #right
        if voxels[i+1,j,k]==False: 
            verts.extend(scale * (right_verts + cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(right_faces + curr_vert)
            curr_vert += len(right_verts)
            
        #front
        if voxels[i,j+1,k]==False: 
            verts.extend(scale * (front_verts + cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(front_faces + curr_vert)
            curr_vert += len(front_verts)
            
        #back
        if voxels[i,j-1,k]==False: 
            verts.extend(scale * (back_verts + cube_dist_scale * np.array([[i-1, j-1, k-1]])))
            faces.extend(back_faces + curr_vert)
            curr_vert += len(back_verts)
            
    return np.array(verts), np.array(faces)


def write_obj(filename, verts, faces):
    """ write the verts and faces on file."""
    with open(filename, 'w') as f:
        # write vertices
        f.write('g\n# %d vertex\n' % len(verts))
        for vert in verts:
            f.write('v %f %f %f\n' % tuple(vert))

        # write faces
        f.write('# %d faces\n' % len(faces))
        for face in faces:
            f.write('f %d %d %d\n' % tuple(face))


def save_obj(voxel, path):
    verts, faces = voxel2mesh(voxel)
    write_obj(path, verts, faces)