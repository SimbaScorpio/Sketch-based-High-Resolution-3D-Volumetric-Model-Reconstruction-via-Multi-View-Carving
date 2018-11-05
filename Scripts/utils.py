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