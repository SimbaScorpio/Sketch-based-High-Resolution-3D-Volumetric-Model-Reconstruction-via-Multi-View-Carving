import torch
import torch.nn as nn
import torch.nn.functional as F


class SketchToVoxelModel(nn.Module):
	def __init__(self):
		super(SketchToVoxelModel, self).__init__()

		kernal_dep 	= [3, 64, 128, 256, 512, 512, 512, 512, 512]
		kernal_dim 	= [7, 4, 4, 4, 4, 4, 4, 4]
		stride		= [2, 2, 2, 2, 2, 2, 2, 2]
		padding 	= [3, 1, 1, 1, 1, 1, 1, 1]

		# 3 256x256
		self.econv1 = nn.Conv2d(kernal_dep[0], kernal_dep[1], kernal_dim[0], stride=stride[0], padding=padding[0])
		self.ebn1 = nn.BatchNorm2d(kernal_dep[1])
		self.erelu1 = nn.LeakyReLU(0.2)
		# 64 128x128
		self.econv2 = nn.Conv2d(kernal_dep[1], kernal_dep[2], kernal_dim[1], stride=stride[1], padding=padding[1])
		self.ebn2 = nn.BatchNorm2d(kernal_dep[2])
		self.erelu2 = nn.LeakyReLU(0.2)
		# 128 64x64
		self.econv3 = nn.Conv2d(kernal_dep[2], kernal_dep[3], kernal_dim[2], stride=stride[2], padding=padding[2])
		self.ebn3 = nn.BatchNorm2d(kernal_dep[3])
		self.erelu3 = nn.LeakyReLU(0.2)
		# 256 32x32
		self.econv4 = nn.Conv2d(kernal_dep[3], kernal_dep[4], kernal_dim[3], stride=stride[3], padding=padding[3])
		self.ebn4 = nn.BatchNorm2d(kernal_dep[4])
		self.erelu4 = nn.LeakyReLU(0.2)
		# 512 16x16
		self.econv5 = nn.Conv2d(kernal_dep[4], kernal_dep[5], kernal_dim[4], stride=stride[4], padding=padding[4])
		self.ebn5 = nn.BatchNorm2d(kernal_dep[5])
		self.erelu5 = nn.LeakyReLU(0.2)
		# 512 8x8
		self.econv6 = nn.Conv2d(kernal_dep[5], kernal_dep[6], kernal_dim[5], stride=stride[5], padding=padding[5])
		self.ebn6 = nn.BatchNorm2d(kernal_dep[6])
		self.erelu6 = nn.LeakyReLU(0.2)
		# 512 4x4
		self.econv7 = nn.Conv2d(kernal_dep[6], kernal_dep[7], kernal_dim[6], stride=stride[6], padding=padding[6])
		self.ebn7 = nn.BatchNorm2d(kernal_dep[7])
		self.erelu7 = nn.LeakyReLU(0.2)
		# 512 2x2
		self.econv8 = nn.Conv2d(kernal_dep[7], kernal_dep[8], kernal_dim[7], stride=stride[7], padding=padding[7])
		self.ebn8 = nn.BatchNorm2d(kernal_dep[8])
		self.erelu8 = nn.LeakyReLU(0.2)
		# 512 1x1

		# 512 1x1
		self.dconv1 = nn.ConvTranspose2d(512, 512, 4, stride=2, padding=1)
		self.dbn1 = nn.BatchNorm2d(512)
		self.drelu1 = nn.ReLU()
		# 512 2x2
		self.dconv2 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
		self.dbn2 = nn.BatchNorm2d(512)
		self.drelu2 = nn.ReLU()
		# 512 4x4
		self.dconv3 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
		self.dbn3 = nn.BatchNorm2d(512)
		self.drelu3 = nn.ReLU()
		# 512 8x8
		self.dconv4 = nn.ConvTranspose2d(1024, 512, 4, stride=2, padding=1)
		self.dbn4 = nn.BatchNorm2d(512)
		self.drelu4 = nn.ReLU()
		# 512 16x16
		self.dconv5 = nn.ConvTranspose2d(1024, 256, 4, stride=2, padding=1)
		self.dbn5 = nn.BatchNorm2d(256)
		self.drelu5 = nn.ReLU()
		# 256 32x32
		self.dconv6 = nn.Conv2d(512, 32, 3, stride=1, padding=1)
		self.dbn6 = nn.BatchNorm2d(32)
		self.drelu6 = nn.ReLU()
		# 32 32x32
		self.sigm = nn.Sigmoid()


	def forward(self, x):
		x1 = self.erelu1( self.ebn1( self.econv1(x) ) )
		x2 = self.erelu2( self.ebn2( self.econv2(x1) ) )
		x3 = self.erelu3( self.ebn3( self.econv3(x2) ) )
		x4 = self.erelu4( self.ebn4( self.econv4(x3) ) )
		x5 = self.erelu5( self.ebn5( self.econv5(x4) ) )
		x6 = self.erelu6( self.ebn6( self.econv6(x5) ) )
		x7 = self.erelu7( self.ebn7( self.econv7(x6) ) )
		code = self.erelu8( self.ebn8( self.econv8(x7) ) )

		# code = x5.view(x5.size(0), -1)
		# code = self.fc1(code)
		# code = self.fc2(code)
		# code = code.reshape(code.size(0), 512, 2, 2, 2)

		y1 = F.dropout( self.drelu1( self.dbn1( self.dconv1(code) ) ), p=0.5)
		y2 = F.dropout( self.drelu2( self.dbn2( self.dconv2(torch.cat([x7, y1], dim=1)) ) ), p=0.5)
		y3 = F.dropout( self.drelu3( self.dbn3( self.dconv3(torch.cat([x6, y2], dim=1)) ) ), p=0.5)
		y4 = self.drelu4( self.dbn4( self.dconv4(torch.cat([x5, y3], dim=1)) ) )
		y5 = self.drelu5( self.dbn5( self.dconv5(torch.cat([x4, y4], dim=1)) ) )
		y6 = self.drelu6( self.dbn6( self.dconv6(torch.cat([x3, y5], dim=1)) ) )

		# y = self.dconv5(y4)
		y = self.sigm(y6)
		# y = y.reshape(y.size(0), 32, 32, 32)

		return y


# custom weights initialization called on netG and netD
def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)


# model = SketchToVoxelModel()
# model.apply(weights_init)
# print(model)

# inputs = torch.randn((64, 3, 256, 256))
# outputs = model(inputs)
# print(inputs.size())
# print(outputs.size())