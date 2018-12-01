import math
import numpy as np
import scipy.io as sio
import binvox_rw
from PIL import Image
from scipy import ndimage
from utils import *
from tqdm import tqdm
from multiprocessing import Pool


dim = 256
binvox_path = '../ShapeNet_Data/binvox32/'
depths_path = '../ShapeNet_Data/depths' + str(dim) + '/'
wanted_classes = ['table']


def call(info):
	depths = sio.loadmat(info['depths_file'])['depths']
	silhouettes = get_silhouettes(depths, dim)

	with open(info['binvox_file'], 'rb') as f:
		vox = binvox_rw.read_as_3d_array(f)

	binvox = vox.data.astype(bool)
	binvox[ndimage.binary_fill_holes(binvox)] = True

	sio.savemat(info['output_silhouettes_file'], {'silhouettes': silhouettes})
	sio.savemat(info['output_binvox_file'], {'binvox': binvox})


def convert():
	for classname in wanted_classes:
		binvox_dir = binvox_path + classname + '/'
		depths_dir = depths_path + classname + '/'
		if not os.path.exists(binvox_dir): continue
		if not os.path.exists(depths_dir): continue

		files = os.listdir(depths_dir)
		infos = []
		for filename in files:
			binvox_file = binvox_dir + filename + '.binvox'
			depths_file = depths_dir + filename + '/depths.mat'
			if not os.path.exists(binvox_file): continue
			if not os.path.exists(depths_file): continue

			output_silhouettes_file = depths_dir + filename + '/silhouettes.mat'
			output_binvox_file = binvox_dir + filename + '.mat'
			infos.append({'binvox_file': binvox_file,
						  'depths_file': depths_file, 
						  'output_silhouettes_file': output_silhouettes_file,
						  'output_binvox_file': output_binvox_file})

		pool = Pool()
		with tqdm(total = len(infos)) as pbar:
			for i, _ in tqdm(enumerate(pool.imap_unordered(call, infos))):
				pbar.update()


"""
I used to convert every object's binvox, depths, silhouettes and sketches into a single mat file.
It could waste a lot of memory when only several data were needed during training.:(
"""

# dim = 256
# binvox_path = '../ShapeNet_Data/binvox32/'
# depths_path = '../ShapeNet_Data/depths' + str(dim) + '/'
# sketch_path = '../ShapeNet_Data/sketches' + str(dim) + '/'
# output_path = '../ShapeNet_Data/mats' + str(dim) + '/'
# wanted_classes = ['chair', 'plane']


# def call(info):
# 	depths = sio.loadmat(info['depths_file'])['depths']
# 	sketches = sio.loadmat(info['sketch_file'])['sketches']
# 	silhouettes = get_silhouettes(depths, dim)

# 	with open(info['binvox_file'], 'rb') as f:
# 		vox = binvox_rw.read_as_3d_array(f)

# 	binvox = vox.data.astype(bool)
# 	binvox[ndimage.binary_fill_holes(binvox)] = True
# 	sio.savemat(info['output_file'], {'depths': depths, 'silhouettes': silhouettes, 'sketches': sketches, 'binvox': binvox})


# def convert():
# 	for classname in wanted_classes:
# 		binvox_dir = binvox_path + classname + '/'
# 		depths_dir = depths_path + classname + '/'
# 		sketch_dir = sketch_path + classname + '/'
# 		if not os.path.exists(binvox_dir): continue
# 		if not os.path.exists(depths_dir): continue
# 		if not os.path.exists(sketch_dir): continue

# 		output_dir = output_path + classname + '/'
# 		if not os.path.exists(output_dir):
# 			os.makedirs(output_dir)

# 		files = os.listdir(depths_dir)
# 		infos = []
# 		for filename in files:
# 			binvox_file = binvox_dir + filename + '.binvox'
# 			depths_file = depths_dir + filename + '/depths.mat'
# 			sketch_file = sketch_dir + filename + '/sketches.mat'
# 			if not os.path.exists(binvox_file): continue
# 			if not os.path.exists(depths_file): continue
# 			if not os.path.exists(sketch_file): continue

# 			output_file = output_dir + filename + '.mat'
# 			infos.append({'binvox_file': binvox_file, 'depths_file': depths_file, 'sketch_file': sketch_file, 'output_file': output_file})

# 		pool = Pool()
# 		with tqdm(total = len(infos)) as pbar:
# 			for i, _ in tqdm(enumerate(pool.imap_unordered(call, infos))):
# 				pbar.update()


if __name__ == '__main__':
	convert()