import os
import math
import numpy as np
import scipy.io as sio
import binvox_rw
from PIL import Image
from tqdm import tqdm
from utils import *
from scipy import ndimage
from multiprocessing import Pool

dim = 256
depths_path = '../ShapeNet_Data/depths' + str(dim) + '/'
output_path = '../ShapeNet_Data/sketches' + str(dim) + '/'
data_type = ['mat', 'png']
# data_type = ['mat']
# data_type = ['png']
wanted_classes = ['table']

threshold = 100


def call_mat(info):
	save_path = info['output_dir'] + info['filename'] + '/'
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	depths = sio.loadmat(info['depth_mat'])['depths']
	depths[np.where(depths == dim)] = 99999
	sketches = np.zeros((6, dim, dim), dtype=bool)
	for axis in range(6):
		padded = np.pad(depths[axis], 1, 'constant', constant_values=99999)
		sx = ndimage.sobel(padded, axis=0, mode='constant')
		sy = ndimage.sobel(padded, axis=1, mode='constant')
		sob = np.hypot(sx, sy)
		sob = sob[1:-1, 1:-1]

		a, b = np.where(sob > threshold)
		for x, y in zip(a, b):
			sketches[axis][x, y] = True

	sio.savemat(save_path + 'sketches.mat', {'sketches': sketches})


def call_png(info):
	save_path = info['output_dir'] + info['filename'] + '/'
	sketches = sio.loadmat(info['sketch_mat'])['sketches']
	for axis in range(6):
		img = render_sketch(sketches[axis], dim)
		png = Image.fromarray(img.transpose(1, 0, 2))
		png.save(save_path + str(axis) + '.png', 'png')


def create_new_sketches():
	for classname in wanted_classes:
		depths_dir = depths_path + classname + '/'
		output_dir = output_path + classname + '/'
		if not os.path.exists(depths_dir): continue

		depths_files = os.listdir(depths_dir)
		infos = []
		for filename in depths_files:
			depth_mat = depths_dir + filename + '/depths.mat'
			if not os.path.exists(depth_mat): continue
			infos.append({'output_dir': output_dir, 'depth_mat': depth_mat, 'filename': filename})
			
		pool = Pool()
		with tqdm(total = len(infos)) as pbar:
			for i, _ in tqdm(enumerate(pool.imap_unordered(call_mat, infos))):
				pbar.update()


def create_png_from_sketches():
	for classname in wanted_classes:
		output_dir = output_path + classname + '/'
		if not os.path.exists(output_dir): continue
		
		sketch_files = os.listdir(output_dir)
		infos = []
		for filename in sketch_files:
			sketch_mat = output_dir + filename + '/sketches.mat'
			if not os.path.exists(sketch_mat): continue
			infos.append({'output_dir': output_dir, 'sketch_mat': sketch_mat, 'filename': filename})

		pool = Pool()
		with tqdm(total = len(infos)) as pbar:
			for i, _ in tqdm(enumerate(pool.imap_unordered(call_png, infos))):
				pbar.update()


if __name__ == '__main__':
	if 'mat' in data_type:
		create_new_sketches()
	if 'png' in data_type:
		create_png_from_sketches()