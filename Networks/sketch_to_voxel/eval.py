from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
from torch.utils.data import Dataset, DataLoader
from sketch_to_voxel_dataset import *
from sketch_to_voxel_model import *
from PIL import Image

import sys
sys.path.append('../../Scripts')
import binvox_rw
from utils import *

# import warnings
# warnings.filterwarnings("ignore")
random.seed(941112)

load_checkpoints = [#'./checkpoint_51288fc_adam_0.001_l1/chair256/best_checkpoint.tar',
					'./checkpoint_51288fc_adam_0.001_l2/chair256/best_checkpoint.tar',
					'./checkpoint_51288fc_adam_0.00002_bce/chair256/best_checkpoint.tar']
					#'./checkpoint_51288fc_adam_0.00002_bce/plane256/best_checkpoint.tar']

dim = 256
classname = 'chair'

list_file = '../../ShapeNet_Data/sets/valid/' + classname + '.txt'
sketch_path = '../../ShapeNet_Data/sketches' + str(dim) + '/' + classname + '/'
voxel_path = '../../ShapeNet_Data/binvox32/' + classname + '/'

eval_path = './eval/' + classname + str(dim) + '/'
if not os.path.exists(eval_path):
	os.makedirs(eval_path)

batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_gt = False
_threshold = False			# using fixed threshold
fix_threshold = 0.50		# fixed threshold
step = 0.01					# if not fixed, then use estimated threshold

w_tp, w_iou = 1.0, 1.0


def save_loss(train_losses, valid_losses, valid_tp_acc, valid_iou):
	_, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(np.arange(len(train_losses)), train_losses, 'blue')
	ax1.plot(np.arange(len(valid_losses)), valid_losses, 'green')
	ax2.plot(np.arange(len(valid_tp_acc)), valid_tp_acc, 'red')
	ax2.plot(np.arange(len(valid_iou)), valid_iou, 'violet')
	ax1.set_xlabel('epoch')
	ax1.set_ylabel('loss')
	ax2.set_ylabel('valid accuracy')
	# ax1.yaxis.set_major_locator(MultipleLocator(0.00005))
	ax2.yaxis.set_major_locator(MultipleLocator(0.05))
	plt.tight_layout()
	plt.grid()
	plt.savefig(eval_path + 'graph.png')
	plt.close()


def save_voxel(grid, threshold, path):
	data = np.zeros((32, 32, 32), dtype=bool)
	data[np.where(grid >= threshold)] = True
	vox = binvox_rw.Voxels(data, dims=[32,32,32], translate=[0,0,0], scale=1.0, axis_order='xyz')
	with open(path + '.binvox', 'wb') as f:
		vox.write(f)
	sio.savemat(path + '.mat', {'voxel': data})


def calculate_loss(outputs, targets, threshold):
	total_iou, total_tp = 0, 0
	for i in range(len(targets)):
		predict = outputs[i]
		target = targets[i]
		s, x, y = np.where( (predict>=threshold)&(target==True) )
		u, x, y = np.where( (predict>=threshold)|(target==True) )
		a, x, y = np.where(target==True)
		iou = len(s) / len(u)
		tp = len(s) / len(a)
		total_iou += iou
		total_tp += tp
	avg_iou = total_iou / len(targets)
	avg_tp = total_tp / len(targets)
	return avg_iou, avg_tp


def save_output(outputs, info):
	with open(list_file, 'r') as f:
		filenames = np.array(f.readlines())

	# save gt files
	targets = []
	print('saving groundtruths...')
	for i in tqdm(range(len(filenames))):
		filename = filenames[i].rstrip('\n')
		target = sio.loadmat(voxel_path + filename + '.mat')['binvox']
		targets.append(target)
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
			save_voxel(target, 0.5, output_path + str(i))
	
	best_threshold = 0
	best_avg_iou = 0
	best_avg_tp = 0
	if _threshold:
		# use pre-defined threshold
		avg_iou, avg_tp = calculate_loss(outputs, targets, fix_threshold)
		best_avg_iou = avg_iou
		best_avg_tp = avg_tp
		best_threshold = fix_threshold
	else:
		# enumerate threshold to find the best one
		print('estimating threshold...')
		for t in tqdm(range(1, int(1.0/step))):
		# for t in tqdm(range(1, 100)):
			threshold = step*t
			avg_iou, avg_tp = calculate_loss(outputs, targets, threshold)
			if (avg_iou*w_iou + avg_tp*w_tp > best_avg_iou*w_iou + best_avg_tp*w_tp):
				best_avg_iou = avg_iou
				best_avg_tp = avg_tp
				best_threshold = threshold

	# save predictions base on best estimated threshold
	output_path = eval_path + 'threshold_' + str(best_threshold) + '/'
	if not os.path.exists(output_path):
		os.makedirs(output_path)
	print('saving voxels...')
	for i in tqdm(range(len(targets))):
		predict = outputs[i]
		save_voxel(predict, best_threshold, output_path + str(i))

	train_losses = info['train_losses']
	valid_losses = info['valid_losses']
	valid_tp_acc = info['valid_tp_acc']
	# valid_fp_acc = info['valid_fp_acc']
	valid_iou = info['valid_iou']
	save_loss(train_losses, valid_losses, valid_tp_acc, valid_iou)

	min_train_loss = np.array(train_losses).min()
	min_valid_loss = np.array(valid_losses).min()
	max_valid_tp_acc = np.array(valid_tp_acc).max()
	# min_valid_fp_acc = np.array(valid_fp_acc).min()
	max_valid_iou = np.array(valid_iou).max()
	with open(eval_path + 'error.txt', 'w+') as f:
		f.write('--Train--\r\n' + 'min_train_loss: ' + str(min_train_loss) + '    min_valid_loss: ' + str(min_valid_loss) + '    max_valid_tp_acc: ' + str(max_valid_tp_acc) + '    max_valid_iou: ' + str(max_valid_iou) + '\r\n')
		f.write('--Valid--\r\n' + 'threshold: ' + str(best_threshold) + '    iou: ' + str(best_avg_iou) + '    tp: ' + str(best_avg_tp) + '\r\n')



if __name__ == '__main__':
	dataset = SketchToVoxelDataset( list_file=list_file, sketch_path=sketch_path, voxel_path=voxel_path)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
	model = SketchToVoxelModel().to(device)

	# ensemble different models together making voxel grid probability much more stable
	final_outputs = []
	for i in range(len(load_checkpoints)):
		info = torch.load(load_checkpoints[i])
		model.load_state_dict(info['model_state_dict'])
		outputs = []
		with torch.no_grad():
			model.eval()
			for batch_idx, (data, target) in enumerate(dataloader):
				data, target = data.to(device), target.to(device)
				output = model(data).cpu().numpy()
				outputs.extend(output)
		if len(final_outputs) == 0:
			final_outputs = np.array(outputs)
		else:
			final_outputs += np.array(outputs)
			
	final_outputs /= float(len(load_checkpoints))
	save_output(final_outputs, info)
