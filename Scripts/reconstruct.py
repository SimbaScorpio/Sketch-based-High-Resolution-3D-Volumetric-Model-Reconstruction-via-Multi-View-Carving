import os
import math
import numpy as np
import binvox_rw
from PIL import Image
from scipy import ndimage
from utils import *

voxelPath = '../ShapeNet_Data/test/binvox/plane/'
depthPath = '../ShapeNet_Data/test/depths/plane/'
outputPath = '../ShapeNet_Data/test/output/plane/'

name = 'fff513f407e00e85a9ced22d91ad7027'

d1 = 32
d2 = 256


def t(v, t1, t2):
	d = int(t2 // t1)
	return v * d, v * d + d


def get_silhouette(png, dim):
	silhouette = np.zeros((dim, dim), dtype=bool)
	img = png.resize((dim, dim)).load()
	for x in range(dim):
		for y in range(dim):
			if img[x, y][3] > 0: silhouette[x, y] = True
	return silhouette


def get_depth(png, dim):
	depth = np.zeros((dim, dim), dtype='uint16')
	img = png.resize((dim, dim)).load()
	for x in range(dim):
		for y in range(dim):
			if img[x, y][3] == 0: continue
			elif img[x, y][2] == 0:   depth[x, y] = int(511 - img[x, y][0])
			elif img[x, y][2] == 255: depth[x, y] = int(255 - img[x, y][1])
	return depth


def carve_edge(grid, silhouette, dim, axis):
	a, b, c = np.where(grid == True)
	for x, y, z in zip(a, b, c):
		m, n, p = grid2img(x, y, z, dim, axis)
		if not silhouette[m, n]:
			grid[x, y, z] = False


def carve_mass(grid, depth, dim, axis):
	a, b, c = np.where(grid == True)
	for x, y, z in zip(a, b, c):
		m, n, p = grid2img(x, y, z, dim, axis)
		if p < depth[m, n]:
			grid[x, y, z] = False


if __name__ == '__main__':

	if not os.path.exists(outputPath):
		os.makedirs(outputPath)

	with open(voxelPath + name + '.binvox', 'rb') as f:
		vox = binvox_rw.read_as_3d_array(f)

	print('\n----before----')
	a, b, c = np.where(vox.data == True)
	print('Voxels: ', len(a))
	print('Dims: ', vox.dims)
	print('Translate: ', vox.translate)
	print('Scale: ', vox.scale)
	bound(vox.data, d1)

	# upsample dimension from d1(low) to d2(high)
	# binvox bounds the model and leave no margin to one side
	# we render depth maps in unit cube with margin on four sides
	# so we need to scale down the upsampled grid to match depth maps
	grid = np.zeros((d2, d2, d2), dtype=bool)
	for x, y, z in zip(a, b ,c):
		xs, xe = t(x, d1, d2)
		ys, ye = t(y, d1, d2)
		zs, ze = t(z, d1, d2)
		grid[xs:xe, ys:ye, zs:ze] = True

	grid[ndimage.binary_fill_holes(grid)] = True

	for i in range(6):
		img = Image.open(depthPath + name + '/' + str(i) + '.png', 'r')
		silhouette = get_silhouette(img, d2)
		carve_edge(grid, silhouette, d2, i)

	for i in range(6):
		img = Image.open(depthPath + name + '/' + str(i) + '.png', 'r')
		depth = get_depth(img, d2)
		carve_mass(grid, depth, d2, i)

	# for axis in range(6):
	# 	png = Image.open(depthPath + name + '/' + str(axis) + '.png', 'r')
	# 	img = png.resize((d2, d2)).load()
	# 	for x in range(d2):
	# 		for y in range(d2):
	# 			if img[x, y][3] == 0: continue
	# 			elif img[x, y][2] == 0:
	# 				v = int(511 - img[x, y][0])
	# 			elif img[x, y][2] == 255:
	# 				v = int(255 - img[x, y][1])
	# 			i, j, k = img2grid(x, y, v, d2, axis)
	# 			grid[i, j, k] = True

	vox.data = np.array(grid)
	vox.dims = [d2, d2, d2]
	# vox.scale = 1

	print('\n----after----')
	a, b, c = np.where(grid == True)
	print('Voxels: ', len(a))
	print('Dims: ', vox.dims)
	print('Translate: ', vox.translate)
	print('Scale: ', vox.scale)
	bound(vox.data, d2)

	with open(outputPath + name + '.binvox', 'wb') as f:
		vox.write(f)