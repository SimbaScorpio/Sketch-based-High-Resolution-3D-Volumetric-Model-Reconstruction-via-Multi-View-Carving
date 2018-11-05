import os
import math
import numpy as np
import scipy.io as sio
import binvox_rw
from PIL import Image
from tqdm import tqdm
from utils import *
from multiprocessing import Pool

dim = 256
binvox_path = '../ShapeNet_Data/binvox' + str(dim) + '/'
output_path = '../ShapeNet_Data/depths' + str(dim) + '/'
data_type = ['mat', 'png']
# data_type = ['mat']
# data_type = ['png']
wanted_classes = ['plane']


def call_mat(info):
	save_path = info['output_dir'] + info['filename'] + '/'
	if not os.path.exists(save_path):
		os.makedirs(save_path)

	with open(info['voxel_mat'], 'rb') as f:
		vox = binvox_rw.read_as_3d_array(f)

	# depths = calculate_depth(vox.data, dim, vox.scale)
	depths = odm(vox.data, dim)
	sio.savemat(save_path + 'depths.mat', {'depths': depths})


def call_png(info):
	save_path = info['output_dir'] + info['filename'] + '/'
	depths = sio.loadmat(info['depth_mat'])['depths']
	for axis in range(6):
		img = render_depth(depths[axis], dim)
		png = Image.fromarray(img.transpose(1, 0, 2))
		png.save(save_path + str(axis) + '.png', 'png')


def create_new_depths():
	for classname in wanted_classes:
		voxels_dir = binvox_path + classname + '/'
		output_dir = output_path + classname + '/'
		if not os.path.exists(voxels_dir): continue

		voxel_files = os.listdir(voxels_dir)
		infos = []
		for filename in voxel_files:
			filename = filename.split('.')[0]
			voxel_mat = voxels_dir + filename + '.binvox'
			if not os.path.exists(voxel_mat): continue
			infos.append({'output_dir': output_dir, 'voxel_mat': voxel_mat, 'filename': filename})

		pool = Pool()
		with tqdm(total = len(infos)) as pbar:
			for i, _ in tqdm(enumerate(pool.imap_unordered(call_mat, infos))):
				pbar.update()


def create_png_from_depths():
	for classname in wanted_classes:
		output_dir = output_path + classname + '/'
		if not os.path.exists(output_dir): continue
		
		depth_files = os.listdir(output_dir)
		infos = []
		for filename in depth_files:
			depth_mat = output_dir + filename + '/depths.mat'
			if not os.path.exists(depth_mat): continue
			infos.append({'output_dir': output_dir, 'depth_mat': depth_mat, 'filename': filename})

		pool = Pool()
		with tqdm(total = len(infos)) as pbar:
			for i, _ in tqdm(enumerate(pool.imap_unordered(call_png, infos))):
				pbar.update()


if __name__ == '__main__':
	if 'mat' in data_type:
		create_new_depths()
	if 'png' in data_type:
		create_png_from_depths()