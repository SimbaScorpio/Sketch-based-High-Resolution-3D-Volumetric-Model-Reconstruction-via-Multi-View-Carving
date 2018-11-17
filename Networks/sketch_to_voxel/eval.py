from __future__ import print_function, division

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.ticker import MultipleLocator
from torch.utils.data import Dataset, DataLoader
from sketch_to_voxel_dataset import *
from sketch_to_voxel_model import *
from PIL import Image

import sys
sys.path.append('../../Scripts')
from utils import *
import binvox_rw

# import warnings
# warnings.filterwarnings("ignore")

load_checkpoint = './checkpoint_51288fc_adam_0.002/chair256/best_checkpoint.tar'

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

_gt = True
thresholds = [0.5]


def load_model(model, path):
	info = torch.load(path)
	model.load_state_dict(info['model_state_dict'])
	train_losses = info['train_losses']
	valid_losses = info['valid_losses']
	valid_tp_acc = info['valid_tp_acc']
	valid_fp_acc = info['valid_fp_acc']

	print(np.array(train_losses).min())
	print(np.array(valid_losses).min())
	print(np.array(valid_tp_acc).max())
	print(np.array(valid_fp_acc).min())

	return train_losses, valid_losses, valid_tp_acc, valid_fp_acc


def save_loss(train_losses, valid_losses, valid_tp_acc, valid_fp_acc):
	_, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(np.arange(len(train_losses)), train_losses, 'blue')
	ax1.plot(np.arange(len(valid_losses)), valid_losses, 'green')
	ax2.plot(np.arange(len(valid_tp_acc)), valid_tp_acc, 'red')
	ax2.plot(np.arange(len(valid_fp_acc)), valid_fp_acc, 'violet')
	ax1.set_xlabel('epoch')
	ax1.set_ylabel('loss')
	ax2.set_ylabel('valid accuracy')
	ax1.yaxis.set_major_locator(MultipleLocator(0.00005))
	ax2.yaxis.set_major_locator(MultipleLocator(0.05))
	plt.tight_layout()
	plt.grid()
	plt.savefig(eval_path + 'graph.png')
	plt.close()


def save_voxel(grid, index, threshold, path):
	data = np.zeros((32, 32, 32), dtype=bool)
	data[np.where(grid >= threshold)] = True
	vox = binvox_rw.Voxels(data, dims=[32,32,32], translate=[0,0,0], scale=1.0, axis_order='xyz')
	with open(path, 'wb') as f:
		vox.write(f)


def save_output(outputs):
	with open(list_file, 'r') as f:
		filenames = np.array(f.readlines())

	ious = [0. for i in range(len(thresholds))]
	tps = [0. for i in range(len(thresholds))]

	for i in tqdm(range(len(filenames))):
		filename = filenames[i].rstrip('\n')
		
		target = sio.loadmat(voxel_path + filename + '.mat')['binvox']
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
			save_voxel(target, i, 0.5, output_path + str(i) + '.binvox')

		# save predictions according to different thresholds
		predict = outputs[i]
		for j in range(len(thresholds)):
			t = thresholds[j]
			output_path = eval_path + 'threshold_' + str(t) + '/'
			if not os.path.exists(output_path):
				os.makedirs(output_path)
			save_voxel(predict, i, t, output_path + str(i) + '.binvox')

			# calculate error
			s, x, y = np.where( (predict>=t)&(target==True) )
			a, x, y = np.where(target==True)
			b, x, y = np.where(predict>=t)
			iou = len(s) / (len(a) + len(b) - len(s))
			tp = len(s) / len(a)
			ious[j] = ious[j] + iou / len(filenames)
			tps[j] = tps[j] + tp / len(filenames)

	with open(eval_path + 'error.txt', 'w+') as f:
		for i in range(len(thresholds)):
			f.write(str(thresholds[i]) + ' iou: ' + str(ious[i]) + '    tp: ' + str(tps[i]) + '\r\n')


if __name__ == '__main__':
	dataset = SketchToVoxelDataset( list_file=list_file, sketch_path=sketch_path, voxel_path=voxel_path)
	dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
	model = SketchToVoxelModel().to(device)

	train_losses, valid_losses, valid_tp_acc, valid_fp_acc = load_model(model, load_checkpoint)
	# save_loss(train_losses, valid_losses, valid_tp_acc, valid_fp_acc)

	# outputs = []
	# with torch.no_grad():
	# 	model.eval()
	# 	for batch_idx, (data, target) in enumerate(dataloader):
	# 		data, target = data.to(device), target.to(device)
	# 		output = model(data).cpu().numpy()
	# 		for i in range(len(output)):
	# 			outputs.append(output[i])
	
	# save_output(outputs)
