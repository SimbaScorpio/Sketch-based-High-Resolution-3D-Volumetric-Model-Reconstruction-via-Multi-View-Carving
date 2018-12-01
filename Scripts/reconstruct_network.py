import os
import sys
import numpy as np
import binvox_rw
from utils import *
from tqdm import tqdm
import scipy.io as sio

voxelPath = 	 '../Networks/sketch_to_voxel/eval/chair256/threshold_0.12/'
silhouettePath = '../Networks/sketch_to_silhouette/eval/chair256/threshold_0.22/'
outputPath =     '../ShapeNet_Data/test/output/chair/'
# voxelPath = 	 '../Networks/sketch_to_voxel/eval/plane256/threshold_0.21/'
# silhouettePath = '../Networks/sketch_to_silhouette/eval/plane256/threshold_0.29/'
# outputPath =     '../ShapeNet_Data/test/output/plane/'

if not os.path.exists(outputPath):
	os.makedirs(outputPath)

dim_from, dim_to = 32, 256
# formats = ['obj', 'binvox']
formats = ['obj']
# formats = ['binvox']

def carve(voxel, silhouettes):
	# upsample voxel to higher resolution
	scale_factor = dim_to // dim_from
	grid = np.zeros((dim_to, dim_to, dim_to), dtype=bool)
	a, b, c = np.where(voxel==True)
	for x, y, z in zip(a, b, c):
		xs, xe = x*scale_factor, x*scale_factor + scale_factor
		ys, ye = y*scale_factor, y*scale_factor + scale_factor
		zs, ze = z*scale_factor, z*scale_factor + scale_factor
		grid[xs:xe, ys:ye, zs:ze] = True

	# carve silhouettes
	a, b, c = np.where(grid==True)
	for x, y, z in zip(a, b, c):
		for axis in range(6):
			m, n, p = grid2img(x, y, z, dim_to, axis)
			if silhouettes[axis][m, n] == False:
				grid[x, y, z] = False

	return grid


if __name__ == '__main__':
	# paths should exist
	if not os.path.exists(voxelPath):
		print('voxel path: [', voxelPath, '] does not exist.')
		sys.exit(0)
	if not os.path.exists(silhouettePath):
		print('silhouette path: [', silhouettePath, '] does not exist.')
		sys.exit(0)

	# load voxel, silhouette and depth mat data from network evaluation folder
	for idx in tqdm(range(0, 10)):
		voxel = sio.loadmat(voxelPath + str(idx) + '.mat')['voxel']
		silhouettes = sio.loadmat(silhouettePath + str(idx) + '.mat')['silhouettes']
		output = carve(voxel, silhouettes)
		if 'binvox' in formats:
			save_binvox(output, outputPath + str(idx) + '.binvox')
		if 'obj' in formats:
			save_obj(output, outputPath + str(idx) + '.obj')

	# pool = Pool()
	# with tqdm(total = len(infos)) as pbar:
	# 	for i, _ in tqdm(enumerate(pool.imap_unordered(call_png, infos))):
	# 		pbar.update()

