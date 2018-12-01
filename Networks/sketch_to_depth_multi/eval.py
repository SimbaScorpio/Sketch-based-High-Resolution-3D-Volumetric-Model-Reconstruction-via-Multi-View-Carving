from __future__ import print_function, division

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from multiprocessing import Pool

from sketch_to_depth_dataset import *
from sketch_to_depth_model import *

import sys
sys.path.append('../../Scripts')
from utils import *
import binvox_rw

# import warnings
# warnings.filterwarnings("ignore")

load_checkpoint = './checkpoint_single_tanh_0.0001_l1/chair256/best_checkpoint.tar'

dim = 256
classname = 'chair'

list_file = '../../ShapeNet_Data/sets/valid/' + classname + '.txt'
sketch_path = '../../ShapeNet_Data/sketches' + str(dim) + '/' + classname + '/'
depth_path = '../../ShapeNet_Data/depths' + str(dim) + '/' + classname + '/'
silhouette_path = '../../ShapeNet_Data/depths' + str(dim) + '/' + classname + '/'

eval_path = './eval/' + classname + str(dim) + '/'

if not os.path.exists(eval_path):
	os.makedirs(eval_path)

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_gt = False


def load_model(model, path):
	info = torch.load(path)
	model.load_state_dict(info['model_state_dict'])
	train_losses = info['train_losses']
	valid_losses = info['valid_losses']
	print(np.array(train_losses).min(), np.array(valid_losses).min())
	return train_losses, valid_losses


def save_loss(train_losses, valid_losses):
	_, ax1 = plt.subplots()
	ax1.plot(np.arange(len(train_losses)), train_losses, 'blue')
	ax1.plot(np.arange(len(valid_losses)), valid_losses, 'green')
	ax1.set_xlabel('epoch')
	ax1.set_ylabel('loss')
	# ax1.yaxis.set_major_locator(MultipleLocator(0.00005))
	plt.tight_layout()
	plt.grid()
	plt.savefig(eval_path + 'graph.png')
	plt.close()


def save_voxel(depths, silhouettes, path):
	data = np.zeros((dim, dim, dim), dtype=bool)
	for axis in range(6):
		a, b = np.where(silhouettes[axis] == True)
		for x, y in zip(a, b):
			m, n, p = img2grid(x, y, depths[axis, x, y], dim, axis)
			data[m, n, p] = True
	# sio.savemat(path, {'voxels': data})
	vox = binvox_rw.Voxels(data, dims=[dim,dim,dim], translate=[0,0,0], scale=1.0, axis_order='xyz')
	with open(path, 'wb') as f:
		vox.write(f)


def call(info):
	output = info['output']
	output_path = info['output_path']
	filename = info['filename']
	for axis in range(6):
		depth = render_depth(output[axis], dim)
		png = Image.fromarray(depth.transpose(1, 0, 2))
		png.save(output_path + str(info['index']) + '_depth_' + str(axis) + '.png', 'png')

	silhouettes = sio.loadmat(depth_path + filename + '/silhouettes.mat')['silhouettes']
	save_voxel(output, silhouettes, output_path + str(info['index']) + '.binvox')
	# path = output_path + str(i) + '.mat'
	# data = np.zeros((dim, dim, dim), dtype=bool)
	# for axis in range(6):
	# 	a, b = np.where(silhouettes[axis] == True)
	# 	for x, y in zip(a, b):
	# 		m, n, p = img2grid(x, y, output[axis, x, y], dim, axis)
	# 		data[m, n, p] = True
	# vox = binvox_rw.Voxels(data, dims=[dim,dim,dim], translate=[0,0,0], scale=1.0, axis_order='xyz')
	# with open(path, 'wb') as f:
	# 	vox.write(f)


def save_output(outputs):
	with open(list_file, 'r') as f:
		filenames = np.array(f.readlines())

	output_path = eval_path + 'prediction/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)

	infos = []
	for i in tqdm(range(len(filenames))):
		filename = filenames[i].rstrip('\n')
		if _gt:
			# save sketch ground truth
			output_path = eval_path + 'groundtruth/'
			if not os.path.exists(output_path):
				os.makedirs(output_path)

			sketches = sio.loadmat(sketch_path + filename + '/sketches.mat')['sketches']
			for axis in range(6):
				sideview = np.full((dim, dim, 4), 255, dtype='uint8')
				sideview[np.where(sketches[axis]==True)] = np.array([255, 255, 255, 255])
				sideview[np.where(sketches[axis]==False)] = np.array([0, 0, 0, 255])
				png = Image.fromarray(sideview.transpose(1, 0, 2))
				png.save(output_path + str(i) + '_sketch_' + str(axis) + '.png', 'png')

			# save voxel ground truth
			target = sio.loadmat(depth_path + filename + '/depths.mat')['depths']
			for axis in range(6):
				depth = render_depth(target[axis], dim)
				png = Image.fromarray(depth.transpose(1, 0, 2))
				png.save(output_path + str(i) + '_depth_' + str(axis) + '.png', 'png')
		infos.append({'filename': filename, 'index': i, 'output': outputs[i], 'output_path': output_path})

	pool = Pool()
	with tqdm(total = len(infos)) as pbar:
		for i, _ in tqdm(enumerate(pool.imap_unordered(call, infos))):
			pbar.update()



if __name__ == '__main__':
	dataset = SketchToDepthDataset(list_file=list_file, sketch_path=sketch_path, depth_path=depth_path, silhouette_path=silhouette_path)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
	model = SketchToDepthModel().to(device)

	train_losses, valid_losses = load_model(model, load_checkpoint)
	save_loss(train_losses, valid_losses)

	outputs = []
	with torch.no_grad():
		model.eval()
		for batch_idx, (data, target, mask) in enumerate(dataloader):
			data, target = data.to(device), target.to(device)
			output = model(data).cpu().numpy()
			for i in range(len(output)):
				outputs.append(((output[i]+1.0)*((dim-1.0)/2.0)).astype('uint16'))
	
	save_output(outputs)
