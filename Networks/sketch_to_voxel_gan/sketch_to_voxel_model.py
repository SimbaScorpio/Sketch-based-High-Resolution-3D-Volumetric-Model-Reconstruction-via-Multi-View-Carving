import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		kernal_dep 	= [3, 64, 128, 256, 512, 512]
		kernal_dim 	= [7, 7, 7, 7, 7]
		stride		= [2, 2, 2, 2, 1]
		padding 	= [3, 3, 3, 3, 3]

		# 3 128x128
		self.econv1 = nn.Conv2d(kernal_dep[0], kernal_dep[1], kernal_dim[0], stride=stride[0], padding=padding[0])
		self.ebn1 = nn.BatchNorm2d(kernal_dep[1])
		self.erelu1 = nn.LeakyReLU(0.2, inplace=True)
		# 64 64x64
		self.econv2 = nn.Conv2d(kernal_dep[1], kernal_dep[2], kernal_dim[1], stride=stride[1], padding=padding[1])
		self.ebn2 = nn.BatchNorm2d(kernal_dep[2])
		self.erelu2 = nn.LeakyReLU(0.2, inplace=True)
		# 128 32x32
		self.econv3 = nn.Conv2d(kernal_dep[2], kernal_dep[3], kernal_dim[2], stride=stride[2], padding=padding[2])
		self.ebn3 = nn.BatchNorm2d(kernal_dep[3])
		self.erelu3 = nn.LeakyReLU(0.2, inplace=True)
		# 256 16x16
		self.econv4 = nn.Conv2d(kernal_dep[3], kernal_dep[4], kernal_dim[3], stride=stride[3], padding=padding[3])
		self.ebn4 = nn.BatchNorm2d(kernal_dep[4])
		self.erelu4 = nn.LeakyReLU(0.2, inplace=True)
		# 512 8x8
		self.econv5 = nn.Conv2d(kernal_dep[4], kernal_dep[5], kernal_dim[4], stride=stride[4], padding=padding[4])
		self.ebn5 = nn.BatchNorm2d(kernal_dep[5])
		self.erelu5 = nn.LeakyReLU(0.2, inplace=True)
		# 512 8x8

		# flatten to 512*8*8
		self.fc1 = nn.Linear(512*8*8, 512*8)
		self.fc2 = nn.Linear(512*8, 512*2*2*2)
		# reshape to 512 2x2x2

		kernal_dep 	= [512, 256, 128, 64, 32, 1]
		kernal_dim 	= [4, 4, 4, 4, 3]
		stride 		= [2, 2, 2, 2, 1]
		padding 	= [1, 1, 1, 1, 1]

		# 512 2x2x2
		self.dconv1 = nn.ConvTranspose3d(kernal_dep[0], kernal_dep[1], kernal_dim[0], stride=stride[0], padding=padding[0])
		self.dbn1 = nn.BatchNorm3d(kernal_dep[1])
		self.drelu1 = nn.ReLU(inplace=True)
		# 256 4x4x4
		self.dconv2 = nn.ConvTranspose3d(kernal_dep[1], kernal_dep[2], kernal_dim[1], stride=stride[1], padding=padding[1])
		self.dbn2 = nn.BatchNorm3d(kernal_dep[2])
		self.drelu2 = nn.ReLU(inplace=True)
		# 128 8x8x8
		self.dconv3 = nn.ConvTranspose3d(kernal_dep[2], kernal_dep[3], kernal_dim[2], stride=stride[2], padding=padding[2])
		self.dbn3 = nn.BatchNorm3d(kernal_dep[3])
		self.drelu3 = nn.ReLU(inplace=True)
		# 64 16x16x16
		self.dconv4 = nn.ConvTranspose3d(kernal_dep[3], kernal_dep[4], kernal_dim[3], stride=stride[3], padding=padding[3])
		self.dbn4 = nn.BatchNorm3d(kernal_dep[4])
		self.drelu4 = nn.ReLU(inplace=True)
		# 32 32x32x32
		self.dconv5 = nn.Conv3d(kernal_dep[4], kernal_dep[5], kernal_dim[4], stride=stride[4], padding=padding[4])
		self.sigm = nn.Sigmoid()
		# 1 32x32x32


	def forward(self, x):
		x1 = self.erelu1( self.ebn1( self.econv1(x) ) )
		x2 = self.erelu2( self.ebn2( self.econv2(x1) ) )
		x3 = self.erelu3( self.ebn3( self.econv3(x2) ) )
		x4 = self.erelu4( self.ebn4( self.econv4(x3) ) )
		x5 = self.erelu5( self.ebn5( self.econv5(x4) ) )

		code = x5.view(x5.size(0), -1)
		code = self.fc1(code)
		code = self.fc2(code)
		code = code.reshape(code.size(0), 512, 2, 2, 2)

		y1 = self.drelu1( self.dbn1( self.dconv1(code) ) )
		y2 = self.drelu2( self.dbn2( self.dconv2(y1) ) )
		y3 = self.drelu3( self.dbn3( self.dconv3(y2) ) )
		y4 = self.drelu4( self.dbn4( self.dconv4(y3) ) )

		y = self.dconv5(y4)
		y = self.sigm(y)
		y = y.reshape(y.size(0), 32, 32, 32)

		return y


class Discriminator(nn.Module):
	def __init__(self):
		super(Discriminator, self).__init__()

		kernal_dep 	= [1, 32, 64, 128, 256, 512, 1]
		kernal_dim 	= [3, 4, 4, 4, 4, 2]
		stride 		= [1, 2, 2, 2, 2, 2]
		padding 	= [1, 1, 1, 1, 1, 0]

		# 1 32x32x32
		self.conv1 = nn.Conv3d(kernal_dep[0], kernal_dep[1], kernal_dim[0], stride=stride[0], padding=padding[0])
		self.bn1 = nn.BatchNorm3d(kernal_dep[1])
		self.relu1 = nn.LeakyReLU(0.2, inplace=True)
		# 32 32x32x32
		self.conv2 = nn.Conv3d(kernal_dep[1], kernal_dep[2], kernal_dim[1], stride=stride[1], padding=padding[1])
		self.bn2 = nn.BatchNorm3d(kernal_dep[2])
		self.relu2 = nn.LeakyReLU(0.2, inplace=True)
		# 64 16x16x16
		self.conv3 = nn.Conv3d(kernal_dep[2], kernal_dep[3], kernal_dim[2], stride=stride[2], padding=padding[2])
		self.bn3 = nn.BatchNorm3d(kernal_dep[3])
		self.relu3 = nn.LeakyReLU(0.2, inplace=True)
		# 128 8x8x8
		self.conv4 = nn.Conv3d(kernal_dep[3], kernal_dep[4], kernal_dim[3], stride=stride[3], padding=padding[3])
		self.bn4 = nn.BatchNorm3d(kernal_dep[4])
		self.relu4 = nn.LeakyReLU(0.2, inplace=True)
		# 256 4x4x4
		self.conv5 = nn.Conv3d(kernal_dep[4], kernal_dep[5], kernal_dim[4], stride=stride[4], padding=padding[4])
		self.bn5 = nn.BatchNorm3d(kernal_dep[5])
		self.relu5 = nn.LeakyReLU(0.2, inplace=True)
		# 512 2x2x2
		self.conv6 = nn.Conv3d(kernal_dep[5], kernal_dep[6], kernal_dim[5], stride=stride[5], padding=padding[5])
		self.sigm = nn.Sigmoid()
		# 1 1x1x1

	def forward(self, x):
		x = x.reshape(x.size(0), 1, 32, 32, 32)
		x = self.relu1( self.conv1(x) )
		x = self.relu2( self.bn2( self.conv2(x) ) )
		x = self.relu3( self.bn3( self.conv3(x) ) )
		x = self.relu4( self.bn4( self.conv4(x) ) )
		x = self.relu5( self.bn5( self.conv5(x) ) )
		x = self.sigm( self.conv6(x) )
		y = x.reshape(x.size(0), 1)

		return y


# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)


# test model
# GNet = Generator()
# GNet.apply(weights_init)
# DNet = Discriminator()
# DNet.apply(weights_init)

# g_inputs = torch.randn((64, 3, 128, 128))
# g_outputs = GNet(g_inputs)
# print(g_inputs.size())
# print(g_outputs.size())

# d_inputs = g_outputs
# d_outputs = DNet(d_inputs)
# print(d_inputs.size())
# print(d_outputs.size())