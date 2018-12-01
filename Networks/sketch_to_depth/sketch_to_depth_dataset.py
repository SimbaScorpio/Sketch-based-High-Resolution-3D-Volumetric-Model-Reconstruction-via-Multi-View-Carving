import torch
import random
import numpy as np
import scipy.io as sio
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class SketchToDepthDataset(Dataset):
	def __init__(self, list_file, sketch_path, depth_path, silhouette_path):
		with open(list_file, 'r') as f:
			self.filenames = np.array(f.readlines())
			for i in range(len(self.filenames)):
				self.filenames[i] = self.filenames[i].rstrip('\n')

		self.sketch_path = sketch_path
		self.depth_path = depth_path
		self.silhouette_path = silhouette_path

	def __len__(self):
		return len(self.filenames)

	def __getitem__(self, idx):
		sketches = sio.loadmat(self.sketch_path + self.filenames[idx] + '/sketches.mat')['sketches']
		depths = sio.loadmat(self.depth_path + self.filenames[idx] + '/depths.mat')['depths']
		silhouettes = sio.loadmat(self.silhouette_path + self.filenames[idx] + '/silhouettes.mat')['silhouettes']

		# pick front, right, top view as input
		sketches = sketches[::2]

		#### transforms ####
		# bool to float
		sketches = sketches.astype('float64')
		depths = depths.astype('float64')
		silhouettes = silhouettes.astype('float64')

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

		# depth value normalize to [0, 1]
		depths = depths / (dim-1.0)
		# depth value normalize to [-1, 1]
		# depths = depths / ((dim-1.0)/2.0) - 1.0

		# totensor
		sketches = torch.from_numpy(sketches).type(torch.FloatTensor)
		depths = torch.from_numpy(depths).type(torch.FloatTensor)
		silhouettes = torch.from_numpy(silhouettes).type(torch.FloatTensor)

		return sketches, depths, silhouettes

