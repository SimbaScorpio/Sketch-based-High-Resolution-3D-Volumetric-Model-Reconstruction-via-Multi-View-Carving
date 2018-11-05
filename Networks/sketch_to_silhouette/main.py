from __future__ import print_function, division

import os
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from sketch_to_silhouette_dataset import *
from sketch_to_silhouette_model import *

from PIL import Image

# import warnings
# warnings.filterwarnings("ignore")

load = False
load_checkpoint = './checkpoint/chair256/checkpoint_120.tar'

dim = 256
classname = 'chair'

list_file = {'train': '../ShapeNet_Data/sets/train/' + classname + '.txt',
			 'valid': '../ShapeNet_Data/sets/valid/' + classname + '.txt'}
sketch_path = '../ShapeNet_Data/sketches' + str(dim) + '/' + classname + '/'
silhouette_path = '../ShapeNet_Data/depths' + str(dim) + '/' + classname + '/'

checkpoint_path = './checkpoint/' + classname + str(dim) + '/'
validviews_path = './validviews/' + classname + str(dim) + '/'
errorgraph_path = './errorgraph/' + classname + str(dim) + '/'

if not os.path.exists(checkpoint_path):
	os.makedirs(checkpoint_path)
if not os.path.exists(validviews_path):
	os.makedirs(validviews_path)
if not os.path.exists(errorgraph_path):
	os.makedirs(errorgraph_path)

dtype = torch.float
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 200
batch_size = 64
lr = 0.005
step_size = 10
step_gamma = 0.9
log_rate = 10	# every {log_rate} batches logs training information
save_rate = 10	# every {save_rate} epochs saves model
threshold = 0.5

random.seed(941112)


def display_images(images):
	for i in range(len(images)):
		plt.subplot(2, 3, i+1)
		plt.axis('off')
		plt.title(str(i))
		plt.imshow(images[i].T, cmap='gray')
	plt.show()


def save_silhouettes(images, epoch):
	# images = images.numpy().astype(float)
	for i in range(6):
		img = np.full((dim, dim, 4), 255, dtype='uint8')
		a, b = np.where(images[i] > threshold)
		for x, y in zip(a, b):
			img[x, y] = np.array([0, 0, 0, 255])
		png = Image.fromarray(img.transpose(1, 0, 2))
		png.save(validviews_path + 'epoch' + str(epoch) + '_' + str(i) + '.png', 'png')
		# print(i, images[i].min(), images[i].max())


def save_error(train_losses, valid_losses, valid_tp_acc, valid_fp_acc, epoch):
	_, ax1 = plt.subplots()
	ax2 = ax1.twinx()
	ax1.plot(np.arange(len(train_losses)), train_losses, 'b')
	ax1.plot(np.arange(len(valid_losses)), valid_losses, 'g')
	ax2.plot(np.arange(len(valid_tp_acc)), valid_tp_acc, 'r')
	ax2.plot(np.arange(len(valid_fp_acc)), valid_fp_acc, 'violet')
	ax1.set_xlabel('iteration')
	ax1.set_ylabel('train loss')
	ax2.set_ylabel('valid accuracy')
	plt.tight_layout()
	plt.savefig(errorgraph_path + 'epoch' + str(epoch) + '.png')
	plt.close()


def save_model(model, optimizer, scheduler, epoch, best_loss, train_losses, valid_losses, valid_tp_acc, valid_fp_acc, train_time, valid_time, path):
	info = {
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'scheduler_state_dict': scheduler.state_dict(),
			'epoch': epoch,
			'best_loss': best_loss,
			'train_losses': train_losses,
			'valid_losses': valid_losses,
			'valid_tp_acc': valid_tp_acc,
			'valid_fp_acc': valid_fp_acc,
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
	valid_tp_acc = info['valid_tp_acc']
	valid_fp_acc = info['valid_fp_acc']
	train_time = info['train_time']
	valid_time = info['valid_time']
	return epoch, best_loss, train_losses, valid_losses, valid_tp_acc, valid_fp_acc, train_time, valid_time


def train_model(model, dataloader, criterion, optimizer, epoch):
	since = time.time()
	total_loss = 0
	total_len = len(dataloader.dataset)
	print('Training...')

	model.train()
	for batch_idx, (data, target) in enumerate(dataloader):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()

		avg_batch_loss = loss.item() / len(data)
		total_loss = total_loss + loss.item()

		time_elapsed = time.time() - since

		if (batch_idx+1) % log_rate == 0:
			process = batch_idx * batch_size + len(data)
			print('Train Epoch: {} [{}/{} ({:.0f}%)] \tLoss: {:.6f} \tTime: {:.0f}m {:.0f}s'.format(
				epoch, process, total_len, process / total_len * 100., avg_batch_loss, time_elapsed//60, time_elapsed%60))

	avg_loss = total_loss / len(dataloader.dataset)
	time_elapsed = time.time() - since
	return avg_loss, time_elapsed


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
			fp_acc.append( len(fp) / len(n) )

		avg_loss = total_loss / len(dataloader.dataset)
		avg_tp_acc = np.array(tp_acc).sum() / len(tp_acc)
		avg_fp_acc = np.array(fp_acc).sum() / len(fp_acc)

		save_silhouettes(output[0], epoch)

	time_elapsed = time.time() - since
	print('Valid Loss: {:.6f} \tTPacc: {:.6f} \tFPacc: {:.6f} \tTime: {:.0f}m {:.0f}s'.format(
		avg_loss, avg_tp_acc, avg_fp_acc, time_elapsed//60, time_elapsed%60))
	return avg_loss, avg_tp_acc, avg_fp_acc, time_elapsed


if __name__ == '__main__':
	datasets = {
		phase: SketchToSilhouetteDataset( list_file=list_file[phase],
										  sketch_path=sketch_path,
										  silhouette_path=silhouette_path)
		for phase in ['train', 'valid']
	}
	dataloaders = {
		phase: DataLoader(datasets[phase], batch_size=batch_size, shuffle=True, num_workers=4)
		for phase in ['train', 'valid']
	}

	model = SketchToSilhouetteModel().to(device)
	model.apply(weights_init)
	criterion = nn.MSELoss()
	optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999))
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=step_gamma)

	best_loss = 1.0
	train_losses, valid_losses = [], []
	valid_tp_acc, valid_fp_acc = [], []
	total_train_time = 0
	total_valid_time = 0
	start_epoch = 0

	if load is True:
		start_epoch, best_loss, train_losses, valid_losses, valid_tp_acc, valid_fp_acc, total_train_time, total_valid_time \
		= load_model(model, optimizer, scheduler, load_checkpoint)
		start_epoch += 1

	for epoch in range(start_epoch, num_epochs):
		scheduler.step()
		train_loss, train_time = train_model(model, dataloaders['train'], criterion, optimizer, epoch)
		valid_loss, tp_acc, fp_acc, valid_time = valid_model(model, dataloaders['valid'], criterion, epoch)

		train_losses.append(train_loss)
		valid_losses.append(valid_loss)
		valid_tp_acc.append(tp_acc)
		valid_fp_acc.append(fp_acc)
		total_train_time += train_time
		total_valid_time += valid_time

		if epoch % save_rate == 0:
			save_model(	model, optimizer, scheduler, epoch, best_loss, train_losses, valid_losses, valid_tp_acc, valid_fp_acc,
						total_train_time, total_valid_time, checkpoint_path + 'checkpoint_' + str(epoch) + '.tar')
			save_error(train_losses, valid_losses, valid_tp_acc, valid_fp_acc, epoch)
		if best_loss > valid_loss:
			best_loss = valid_loss
			save_model(	model, optimizer, scheduler, epoch, best_loss, train_losses, valid_losses, valid_tp_acc, valid_fp_acc,
						total_train_time, total_valid_time, checkpoint_path + 'best_checkpoint.tar')
			save_error(train_losses, valid_losses, valid_tp_acc, valid_fp_acc, epoch)

		time_elapsed = total_train_time + total_valid_time
		print('Time elapsed: {:.0f}m {:.0f}s \t[ Train: {:.0f}m {:.0f}s, Valid: {:.0f}m {:.0f}s ]'.format(
			time_elapsed//60, time_elapsed%60, total_train_time//60, total_train_time%60, total_valid_time//60, total_valid_time%60))

