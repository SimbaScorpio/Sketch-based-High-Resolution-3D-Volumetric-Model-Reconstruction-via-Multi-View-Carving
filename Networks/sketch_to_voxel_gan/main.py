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

from sketch_to_voxel_dataset import *
from sketch_to_voxel_model import *

from PIL import Image

import sys
sys.path.append('../../Scripts')
from utils import *
import binvox_rw

# import warnings
# warnings.filterwarnings("ignore")

load = False
load_checkpoint = './checkpoint/chair256/checkpoint_120.tar'

dim = 256
classname = 'chair'

list_file = {'train': '../../ShapeNet_Data/sets/train/' + classname + '.txt',
			 'valid': '../../ShapeNet_Data/sets/valid/' + classname + '.txt'}
sketch_path = '../../ShapeNet_Data/sketches' + str(dim) + '/' + classname + '/'
voxel_path = '../../ShapeNet_Data/binvox32/' + classname + '/'

checkpoint_path = './checkpoint/' + classname + str(dim) + '/'
validviews_path = './validviews/' + classname + str(dim) + '/'
errorgraph_path = './errorgraph/' + classname + str(dim) + '/'

if not os.path.exists(checkpoint_path):
	os.makedirs(checkpoint_path)
if not os.path.exists(validviews_path):
	os.makedirs(validviews_path)
if not os.path.exists(errorgraph_path):
	os.makedirs(errorgraph_path)

# gpu device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# for i in range(torch.cuda.device_count()):
# 	print(i, torch.cuda.get_device_name(i))


num_epochs = 500
batch_size = 64
lr_G = 0.001
lr_D = 0.001
lambdaGAN = 0.001
log_rate = 10	# every {log_rate} batches logs training information
save_rate = 10	# every {save_rate} epochs saves model
threshold = 0.5

manualSeed = 941112
random.seed(manualSeed)
torch.manual_seed(manualSeed)


def display_images(images):
	for i in range(len(images)):
		plt.subplot(2, 3, i+1)
		plt.axis('off')
		plt.title(str(i))
		plt.imshow(images[i].T, cmap='gray')
	plt.show()


def save_voxel(grid, epoch):
	data = np.zeros((32, 32, 32), dtype=bool)
	a,b,c = np.where(grid >= threshold)
	for x, y, z in zip(a, b, c):
		data[x, y, z] = True
	vox = binvox_rw.Voxels(data, dims=[32,32,32], translate=[0,0,0], scale=1.0, axis_order='xyz')
	with open(validviews_path + 'epoch' + str(epoch) + '.binvox', 'wb') as f:
		vox.write(f)


def save_error(G_losses, D_losses, train_losses, valid_losses, valid_tp_acc, valid_fp_acc, epoch):
	# voxel loss
	_, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(np.arange(len(train_losses)), train_losses, 'b')
	ax1.plot(np.arange(len(valid_losses)), valid_losses, 'g')
	ax2.plot(np.arange(len(valid_tp_acc)), valid_tp_acc, 'r')
	ax2.plot(np.arange(len(valid_fp_acc)), valid_fp_acc, 'violet')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('train loss')
	ax2.set_ylabel('valid accuracy')
	ax2.yaxis.set_major_locator(MultipleLocator(0.05))
	plt.tight_layout()
	plt.grid()
	plt.savefig(errorgraph_path + 'voxel_epoch' + str(epoch) + '.png')
	plt.close()
	# GAN loss
	_, ax1 = plt.subplots()
	ax1.plot(np.arange(len(G_losses)), G_losses, 'b')
	ax1.plot(np.arange(len(D_losses)), D_losses, 'g')
	ax1.set_xlabel('batches')
	ax1.set_ylabel('GAN loss')
	ax1.yaxis.set_major_locator(MultipleLocator(0.5))
	plt.tight_layout()
	plt.grid()
	plt.savefig(errorgraph_path + 'gan_epoch' + str(epoch) + '.png')
	plt.close()


def save_model(epoch, netG, netD, optimizerG, optimizerD, G_losses, D_losses, best_loss, train_losses, valid_losses, valid_tp_acc, valid_fp_acc, train_time, valid_time, path):
	info = {
			'netG_state_dict': netG.state_dict(),
			'netD_state_dict': netD.state_dict(),
			'optimizerG_state_dict': optimizerG.state_dict(),
			'optimizerD_state_dict': optimizerD.state_dict(),
			'epoch': epoch,
			'G_losses': G_losses,
			'D_losses': D_losses,
			'best_loss': best_loss,
			'train_losses': train_losses,
			'valid_losses': valid_losses,
			'valid_tp_acc': valid_tp_acc,
			'valid_fp_acc': valid_fp_acc,
			'train_time': train_time,
			'valid_time': valid_time
			}
	torch.save(info, path)


def load_model(netG, netD, optimizerG, optimizerD, path):
	info = torch.load(path)
	netG.load_state_dict(info['netG_state_dict'])
	netD.load_state_dict(info['netD_state_dict'])
	optimizerG.load_state_dict(info['optimizerG_state_dict'])
	optimizerD.load_state_dict(info['optimizerD_state_dict'])
	epoch = info['epoch']
	G_losses = info['G_losses']
	D_losses = info['D_losses']
	best_loss = info['best_loss']
	train_losses = info['train_losses']
	valid_losses = info['valid_losses']
	valid_tp_acc = info['valid_tp_acc']
	valid_fp_acc = info['valid_fp_acc']
	train_time = info['train_time']
	valid_time = info['valid_time']
	return epoch, G_losses, D_losses, best_loss, train_losses, valid_losses, valid_tp_acc, valid_fp_acc, train_time, valid_time


def train_model(netG, netD, dataloader, criterionG, criterionD, optimizerG, optimizerD, epoch):
	since = time.time()
	total_len = len(dataloader.dataset)
	G_losses = []
	D_Losses = []
	total_loss = 0
	print('Training...')

	netG.train()
	netD.train()
	for batch_idx, (data, target) in enumerate(dataloader):
		data, target = data.to(device), target.to(device)

		####################################
		# Update D network: maximize log(D(x)) + log(1-D(G(z)))
		####################################
		errD, errG, per_real, per_fake_1, per_fake_2 = -1, -1, -1, -1, -1
		# 1. calculate loss log(D(x))
		# format real labels
		real_label = torch.full((data.size(0),), 1, device=device)
		# calculate loss
		optimizerD.zero_grad()
		output_label = netD(target).view(-1)
		errD_real = criterionD(output_label, real_label)
		errD_real.backward()
		per_real = output_label.mean().item()

		# 2. calculate loss log(1-D(G(z)))
		# format fake labels
		fake_label = torch.full((data.size(0),), 0, device=device)
		# calculate loss
		voxel = netG(data)
		# fake_thres_voxel = torch.where(voxel > threshold, torch.full_like(voxel, 1), voxel).to(device)
		output_label = netD(voxel.detach()).view(-1)
		errD_fake = criterionD(output_label, fake_label)
		errD_fake.backward()
		optimizerD.step()
		per_fake_1 = output_label.mean().item()
		errD = errD_real + errD_fake

		####################################
		# Update G network: maximize log(D(G(z)))
		####################################
		# calculate loss
		optimizerG.zero_grad()
		output_label = netD(voxel).view(-1)
		errG = criterionD(output_label, real_label)
		errShape = criterionG(voxel, target)
		errRecon = lambdaGAN*errG + (1-lambdaGAN)*errShape
		errRecon.backward()
		optimizerG.step()
		per_fake_2 = output_label.mean().item()

		# save losses for plotting
		G_losses.append(errG.item())
		D_Losses.append(errD.item())
		total_loss = total_loss + errShape.item()
		time_elapsed = time.time() - since

		if (batch_idx+1) % log_rate == 0:
			process = batch_idx * batch_size + len(data)
			print('Train Epoch: {} [{}/{}] \tLoss: {:.6f} \tD_Loss: {:.4f} \tG_Loss: {:.4f} \tD(x): {:.4f} \tD(G(z)): {:.4f} / {:.4f} \tTime: {:.0f}m {:.0f}s'.format(
				epoch, process, total_len, errShape, errD, errG, per_real, per_fake_1, per_fake_2, time_elapsed//60, time_elapsed%60))

	avg_loss = total_loss / len(dataloader)
	time_elapsed = time.time() - since
	return G_losses, D_Losses, avg_loss, time_elapsed


def valid_model(model, dataloader, criterion, epoch):
	since = time.time()
	total_loss = 0
	tp_acc, fp_acc = [], []
	print('Validing...')

	with torch.no_grad():
		model.eval()
		for batch_idx, (data, target) in enumerate(dataloader):
			data, target = data.to(device), target.to(device)
			output = model(data)
			loss = criterion(output, target)
			total_loss += loss.item()

			output = output.cpu().numpy()
			target = target.cpu().numpy()
			tp,_,_,_ = np.where((output>threshold)&(target>threshold))
			fp,_,_,_ = np.where((output>threshold)&(target<threshold))
			p,_,_,_ = np.where((target>threshold))
			n,_,_,_ = np.where((target<threshold))
			tp_acc.append( len(tp) / len(p) )
			# fp_acc.append( len(fp) / len(n) )
			fp_acc.append( len(fp) / len(p) )

		avg_loss = total_loss / len(dataloader)
		avg_tp_acc = np.array(tp_acc).sum() / len(tp_acc)
		avg_fp_acc = np.array(fp_acc).sum() / len(fp_acc)

		save_voxel(output[0], epoch)

	time_elapsed = time.time() - since
	print('Valid Loss: {:.6f} \tTPacc: {:.6f} \tFPacc: {:.6f} \tTime: {:.0f}m {:.0f}s'.format(
		avg_loss, avg_tp_acc, avg_fp_acc, time_elapsed//60, time_elapsed%60))
	return avg_loss, avg_tp_acc, avg_fp_acc, time_elapsed


if __name__ == '__main__':
	datasets = {
		phase: SketchToVoxelDataset( list_file=list_file[phase],
									 sketch_path=sketch_path,
									 voxel_path=voxel_path)
		for phase in ['train', 'valid']
	}
	dataloaders = {
		phase: DataLoader(datasets[phase], batch_size=batch_size, shuffle=True, num_workers=4)
		for phase in ['train', 'valid']
	}

	netG = Generator().to(device)
	netG.apply(weights_init)
	netD = Discriminator().to(device)
	netD.apply(weights_init)
	criterionG = nn.MSELoss()
	criterionD = nn.BCELoss()
	optimizerG = optim.Adam(netG.parameters(), lr=lr_G, betas=(0.9, 0.999))
	optimizerD = optim.Adam(netD.parameters(), lr=lr_D, betas=(0.9, 0.999))

	best_loss = 100000.0
	G_losses, D_losses = [], []
	train_losses, valid_losses = [], []
	valid_tp_acc, valid_fp_acc = [], []
	total_train_time = 0
	total_valid_time = 0
	start_epoch = 0

	if load is True:
		start_epoch, G_losses, D_losses, best_loss, train_losses, valid_losses, valid_tp_acc, valid_fp_acc, total_train_time, total_valid_time \
		= load_model(netG, netD, optimizerG, optimizerD, load_checkpoint)
		start_epoch += 1

	for epoch in range(start_epoch, num_epochs):
		G_loss, D_loss, train_loss, train_time = train_model(netG, netD, dataloaders['train'], criterionG, criterionD, optimizerG, optimizerD, epoch)
		valid_loss, tp_acc, fp_acc, valid_time = valid_model(netG, dataloaders['valid'], criterionG, epoch)

		G_losses.extend(G_loss)
		D_losses.extend(D_loss)
		train_losses.append(train_loss)
		valid_losses.append(valid_loss)
		valid_tp_acc.append(tp_acc)
		valid_fp_acc.append(fp_acc)
		total_train_time += train_time
		total_valid_time += valid_time

		if epoch % save_rate == 0:
			save_model(	epoch, netG, netD, optimizerG, optimizerD, G_losses, D_losses, best_loss, train_losses, valid_losses, valid_tp_acc, valid_fp_acc,
						total_train_time, total_valid_time, checkpoint_path + 'checkpoint_' + str(epoch) + '.tar')
			save_error(G_losses, D_losses, train_losses, valid_losses, valid_tp_acc, valid_fp_acc, epoch)
		if best_loss > valid_loss:
			best_loss = valid_loss
			save_model(	epoch, netG, netD, optimizerG, optimizerD, G_losses, D_losses, best_loss, train_losses, valid_losses, valid_tp_acc, valid_fp_acc,
						total_train_time, total_valid_time, checkpoint_path + 'best_checkpoint.tar')
			save_error(G_losses, D_losses, train_losses, valid_losses, valid_tp_acc, valid_fp_acc, epoch)

		time_elapsed = total_train_time + total_valid_time
		print('Time elapsed: {:.0f}m {:.0f}s \t[ Train: {:.0f}m {:.0f}s, Valid: {:.0f}m {:.0f}s ]'.format(
			time_elapsed//60, time_elapsed%60, total_train_time//60, total_train_time%60, total_valid_time//60, total_valid_time%60))

