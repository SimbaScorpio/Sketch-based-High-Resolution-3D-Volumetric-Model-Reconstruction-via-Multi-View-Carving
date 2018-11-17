import os
import math
import numpy as np
from PIL import Image
from scipy import ndimage
import argparse

import sys
sys.path.append('../../Scripts')
from utils import *
import binvox_rw

depths_path = './validviews/chair256/epoch'
output_path = './binvox_valid/chair/'
dim = 256

parser = argparse.ArgumentParser(description='Depth predictor for 3D model reconstruction')
parser.add_argument('-e', '--epoch', default=0, type=int)
args = parser.parse_args()


def get_depth(png, dim):
	depth = np.zeros((dim, dim), dtype='uint16')
	img = png.resize((dim, dim)).load()
	for x in range(dim):
		for y in range(dim):
			if img[x, y][0] == 0: depth[x, y] = dim
			elif img[x, y][2] == 0:   depth[x, y] = int(511 - img[x, y][0])
			elif img[x, y][2] == 255: depth[x, y] = int(255 - img[x, y][1])
	return depth


def fill_mass(grid, depth, dim, axis):
	a, b = np.where(depth != dim)
	for x, y in zip(a, b):
		m, n, p = img2grid(x, y, depth[x, y], dim, axis)
		grid[m, n, p] = True


def carve_mass(grid, depth, dim, axis):
	a, b, c = np.where(grid == True)
	for x, y, z in zip(a, b, c):
		m, n, p = grid2img(x, y, z, dim, axis)
		if depth[m, n] == dim:
			grid[x, y, z] = False
		elif depth[m, n] > p:
			grid[m, n, p] = False


if __name__ == '__main__':

	if not os.path.exists(output_path):
		os.makedirs(output_path)

	grid = np.zeros((dim, dim, dim), dtype=bool)
	for i in range(6):
		img = Image.open(depths_path + str(args.epoch) + '_' + str(i) + '.png', 'r')
		depth = get_depth(img, dim)
		fill_mass(grid, depth, dim, i)

	data = np.array(grid)
	dims = [dim, dim, dim]
	scale = 1
	translate = [0, 0, 0]
	axis_order = 'xyz'

	vox = binvox_rw.Voxels(data, dims, translate, scale, axis_order)

	print('\n----after----')
	a, b, c = np.where(grid == True)
	print('Voxels: ', len(a))
	print('Dims: ', vox.dims)
	print('Translate: ', vox.translate)
	print('Scale: ', vox.scale)
	bound(vox.data, dim)

	with open(output_path + 'test0.binvox', 'wb') as f:
		vox.write(f)


	grid = np.ones((dim, dim, dim), dtype=bool)
	for i in range(6):
		img = Image.open(depths_path + str(args.epoch) + '_' + str(i) + '.png', 'r')
		depth = get_depth(img, dim)
		carve_mass(grid, depth, dim, i)

	data = np.array(grid)
	dims = [dim, dim, dim]
	scale = 1
	translate = [0, 0, 0]
	axis_order = 'xyz'

	vox = binvox_rw.Voxels(data, dims, translate, scale, axis_order)

	print('\n----after----')
	a, b, c = np.where(grid == True)
	print('Voxels: ', len(a))
	print('Dims: ', vox.dims)
	print('Translate: ', vox.translate)
	print('Scale: ', vox.scale)
	bound(vox.data, dim)

	with open(output_path + 'test1.binvox', 'wb') as f:
		vox.write(f)