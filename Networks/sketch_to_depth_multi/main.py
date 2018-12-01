from __future__ import print_function, division

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from sketch_to_depth_dataset import *
from sketch_to_depth_model import *

import sys
sys.path.append('../../Scripts')
from utils import *

from PIL import Image

# import warnings
# warnings.filterwarnings("ignore")

load = False
load_checkpoint = './checkpoint_single_tanh_0.0001/chair256/best_checkpoint.tar'

dim = 256
classname = 'chair'

list_file = {'train': '../../ShapeNet_Data/sets/train/' + classname + '.txt',
			 'valid': '../../ShapeNet_Data/sets/valid/' + classname + '.txt'}
sketch_path = '../../ShapeNet_Data/sketches' + str(dim) + '/' + classname + '/'
depth_path = '../../ShapeNet_Data/depths' + str(dim) + '/' + classname + '/'
silhouette_path = '../../ShapeNet_Data/depths' + str(dim) + '/' + classname + '/'

checkpoint_path = './checkpoint/' + classname + str(dim) + '/'
validviews_path = './validviews/' + classname + str(dim) + '/'
errorgraph_path = './errorgraph/' + classname + str(dim) + '/'

if not os.path.exists(checkpoint_path):
	os.makedirs(checkpoint_path)
if not os.path.exists(validviews_path):
	os.makedirs(validviews_path)
if not os.path.exists(errorgraph_path):
	os.makedirs(errorgraph_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

view = 5
num_epochs = 200
batch_size = 32
lr = 0.001
step_size = 10000
step_gamma = 0.9
log_rate = 10	# every {log_rate} batches logs training information
save_rate = 10	# every {save_rate} epochs saves model

random.seed(941112)


def display_images(images):
	for i in range(len(images)):
		plt.subplot(2, 3, i+1)
		plt.axis('off')
		plt.title(str(i))
		plt.imshow(images[i].T, cmap='gray')
	plt.show()


def save_depth(image, mask, epoch):
	# image *= (dim-1.0)
	image = (image+1.0)*((dim-1.0)/2.0)
	img = render_depth(image, dim)
	a, b = np.where(mask == 0)
	for x, y in zip(a, b):
		img[x, y] = np.array([0,0,0,255])
	png = Image.fromarray(img.transpose(1, 0, 2))
	png.save(validviews_path + str(view) + '_' + 'epoch' + str(epoch) + '.png', 'png')


def save_error(train_losses, valid_losses, base_losses, epoch):
	train_losses = [train_losses[i] for i in range(len(train_losses))]
	valid_losses = [valid_losses[i] for i in range(len(valid_losses))]
	base_losses = [base_losses[i] for i in range(len(base_losses))]
	_, ax1 = plt.subplots()
	ax1.plot(np.arange(len(train_losses)), train_losses, 'b')
	ax1.plot(np.arange(len(valid_losses)), valid_losses, 'g')
	ax1.plot(np.arange(len(base_losses)), base_losses, 'r')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('train loss')
	plt.tight_layout()
	plt.grid()
	plt.savefig(errorgraph_path + 'epoch' + str(epoch) + '.png')
	plt.close()


def save_model(model, optimizer, scheduler, epoch, best_loss, train_losses, valid_losses, base_losses, train_time, valid_time, path):
	info = {
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'scheduler_state_dict': scheduler.state_dict(),
			'epoch': epoch,
			'best_loss': best_loss,
			'train_losses': train_losses,
			'valid_losses': valid_losses,
			'base_losses': base_losses,
			'train_time': train_time,
			'valid_time': valid_time
			}
	torch.save(info, path)


def load_model(model, optimizer, scheduler, path):
	info = torch.load(path)
	model.load_state_dict(info['model_state_dict'])
	optimizer.load_state_dict(info['optimizer_state_dict'])
	scheduler.load_state_dict(info['scheduler_state_dict'])
	epoch = info['epoch']
	best_loss = info['best_loss']
	train_losses = info['train_losses']
	valid_losses = info['valid_losses']
	base_losses = info['base_losses']
	train_time = info['train_time']
	valid_time = info['valid_time']
	return epoch, best_loss, train_losses, valid_losses, base_losses, train_time, valid_time


def train_model(model, dataloader, criterion, optimizer, epoch):
	since = time.time()
	total_loss = 0
	total_len = len(dataloader.dataset)
	print('Training...')

	model.train()
	for batch_idx, (data, target, mask) in enumerate(dataloader):
		data, target, mask = data.to(device), target.to(device), mask.to(device)
		optimizer.zero_grad()
		output = model(data)
		output = torch.mul(output, mask).to(device)
		target = torch.mul(target, mask).to(device)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		total_loss = total_loss + loss.item()
		time_elapsed = time.time() - since

		if (batch_idx+1) % log_rate == 0:
			process = batch_idx * batch_size + len(data)
			print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f} \tTime: {:.0f}m {:.0f}s'.format(
				epoch, process, total_len, process / total_len * 100., loss, time_elapsed//60, time_elapsed%60))

	avg_loss = total_loss / len(dataloader)
	time_elapsed = time.time() - since
	return avg_loss, time_elapsed


def valid_model(model, dataloader, criterion, epoch):
	since = time.time()
	total_loss, total_l1loss = 0, 0
	l1criterion = nn.L1Loss()
	print('Validing...')

	with torch.no_grad():
		model.eval()
		for batch_idx, (data, target, mask) in enumerate(dataloader):
			data, target, mask = data.to(device), target.to(device), mask.to(device)
			output = model(data)
			output = torch.mul(output, mask).to(device)
			target = torch.mul(target, mask).to(device)
			loss = criterion(output, target)
			total_loss += loss.item()
			l1loss = l1criterion(output, target) # used for uniform comparison
			total_l1loss += l1loss.item()

	avg_loss = total_loss / len(dataloader)
	avg_l1loss = total_l1loss / len(dataloader)
	save_depth(output.cpu().numpy()[0], mask.cpu().numpy()[0], epoch)

	time_elapsed = time.time() - since
	print('Valid Loss: {:.6f} \tL1 Loss: {:.6f} \tTime: {:.0f}m {:.0f}s'.format(
		avg_loss, avg_l1loss, time_elapsed//60, time_elapsed%60))
	return avg_loss, avg_l1loss, time_elapsed


if __name__ == '__main__':
	datasets = {
		phase: SketchToDepthDataset( list_file=list_file[phase],
									 sketch_path=sketch_path,
									 depth_path=depth_path,
									 silhouette_path=silhouette_path,
									 view = view)
		for phase in ['train', 'valid']
	}
	dataloaders = {
		phase: DataLoader(datasets[phase], batch_size=batch_size, shuffle=True, num_workers=4)
		for phase in ['train', 'valid']
	}

	model = SketchToDepthModel().to(device)
	model.apply(weights_init)
	criterion = nn.MSELoss()
	# criterion = nn.L1Loss()
	optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_gamma)

	best_loss = 10000.0
	train_losses, valid_losses, base_losses = [], [], []
	total_train_time = 0
	total_valid_time = 0
	start_epoch = 0

	if load is True:
		start_epoch, best_loss, train_losses, valid_losses, base_losses, total_train_time, total_valid_time \
		= load_model(model, optimizer, scheduler, load_checkpoint)
		start_epoch += 1

	for epoch in range(start_epoch, num_epochs):
		# scheduler.step()
		train_loss, train_time = train_model(model, dataloaders['train'], criterion, optimizer, epoch)
		valid_loss, base_loss, valid_time = valid_model(model, dataloaders['valid'], criterion, epoch)

		train_losses.append(train_loss)
		valid_losses.append(valid_loss)
		base_losses.append(base_loss)
		total_train_time += train_time
		total_valid_time += valid_time

		if epoch % save_rate == 0:
			save_model(	model, optimizer, scheduler, epoch, best_loss, train_losses, valid_losses, base_losses,
						total_train_time, total_valid_time, checkpoint_path + 'checkpoint_' + str(epoch) + '.tar')
			save_error(train_losses, valid_losses, base_losses, epoch)
		if best_loss > valid_loss:
			best_loss = valid_loss
			save_model(	model, optimizer, scheduler, epoch, best_loss, train_losses, valid_losses, base_losses,
						total_train_time, total_valid_time, checkpoint_path + 'best_checkpoint.tar')
			save_error(train_losses, valid_losses, base_losses, epoch)

		time_elapsed = total_train_time + total_valid_time
		print('Time elapsed: {:.0f}m {:.0f}s \t[ Train: {:.0f}m {:.0f}s, Valid: {:.0f}m {:.0f}s ]'.format(
			time_elapsed//60, time_elapsed%60, total_train_time//60, total_train_time%60, total_valid_time//60, total_valid_time%60))

