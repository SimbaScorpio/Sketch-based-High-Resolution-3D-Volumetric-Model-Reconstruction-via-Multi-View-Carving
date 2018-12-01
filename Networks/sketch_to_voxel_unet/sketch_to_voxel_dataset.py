import torch
import random
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SketchToVoxelDataset(Dataset):
	def __init__(self, list_file, sketch_path, voxel_path):
		with open(list_file, 'r') as f:
			self.filenames = np.array(f.readlines())
			for i in range(len(self.filenames)):
				self.filenames[i] = self.filenames[i].rstrip('\n')

		self.sketch_path = sketch_path
		self.voxel_path = voxel_path

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		sketches = sio.loadmat(self.sketch_path + self.filenames[idx] + '/sketches.mat')['sketches']
		voxels = sio.loadmat(self.voxel_path + self.filenames[idx] + '.mat')['binvox']

		# pick front, right, top view as input
		sketches = sketches[::2]

		#### transforms ####
		# bool to float
		sketches = sketches.astype('float64')
		voxels = voxels.astype('float64')

		# random center crop
		views = sketches.shape[0]
		dim = sketches.shape[1]
		pad = int(dim*0.1)
		for i in range(views):
			padded = np.pad(sketches[i], pad, 'constant', constant_values=0)
			leftcornerX = random.randint(0, 2*pad)
			leftcornerY = random.randint(0, 2*pad)
			crop = padded[leftcornerX:leftcornerX+dim, leftcornerY:leftcornerY+dim]
			sketches[i] = crop

		# resize to 128x128
		# r = dim/128.0
		# new_sketches = np.zeros((3, 128, 128), dtype='float64')
		# a, b, c = np.where(sketches == 1)
		# for i, x, y in zip(a, b, c):
		# 	new_sketches[i][int(x/r)][int(y/r)] = 1.0
		# sketches = new_sketches

		# totensor
		sketches = torch.from_numpy(sketches).type(torch.FloatTensor)
		voxels = torch.from_numpy(voxels).type(torch.FloatTensor)

		return sketches, voxels

