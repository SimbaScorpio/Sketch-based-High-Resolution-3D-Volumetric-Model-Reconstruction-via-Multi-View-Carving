import torch
import torch.nn as nn
import torch.nn.functional as F


class SketchToDepthModel(nn.Module):
	def __init__(self):
		super(SketchToDepthModel, self).__init__()

		# 3 256x256
		self.econv1 = nn.Conv2d(3, 64, 7, 2, 3)
		self.ebn1 = nn.BatchNorm2d(64)
		self.erelu1 = nn.LeakyReLU(0.2)
		# 64 128x128
		self.econv2 = nn.Conv2d(64, 128, 5, 2, 2)
		self.ebn2 = nn.BatchNorm2d(128)
		self.erelu2 = nn.LeakyReLU(0.2)
		# 128 64x64
		self.econv3 = nn.Conv2d(128, 256, 5, 2, 2)
		self.ebn3 = nn.BatchNorm2d(256)
		self.erelu3 = nn.LeakyReLU(0.2)
		# 256 32x32
		self.econv4 = nn.Conv2d(256, 512, 3, 2, 1)
		self.ebn4 = nn.BatchNorm2d(512)
		self.erelu4 = nn.LeakyReLU(0.2)
		# 512 16x16
		self.econv5 = nn.Conv2d(512, 512, 3, 2, 1)
		self.ebn5 = nn.BatchNorm2d(512)
		self.erelu5 = nn.LeakyReLU(0.2)
		# 512 8x8
		self.econv6 = nn.Conv2d(512, 512, 3, 2, 1)
		self.ebn6 = nn.BatchNorm2d(512)
		self.erelu6 = nn.LeakyReLU(0.2)
		# 512 4x4
		self.econv7 = nn.Conv2d(512, 512, 3, 2, 1)
		self.ebn7 = nn.BatchNorm2d(512)
		self.erelu7 = nn.LeakyReLU(0.2)
		# 512 2x2

		self.fc1 = nn.Linear(512*2*2, 512*2*2)
		self.fc2 = nn.Linear(512*2*2, 512*2*2)

		# 512 2x2
		self.dconv1 = nn.ConvTranspose2d(512, 512, 4, 2, 1)
		self.dbn1 = nn.BatchNorm2d(512)
		self.drelu1 = nn.ReLU()
		# 512 4x4
		self.dconv2 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
		self.dbn2 = nn.BatchNorm2d(512)
		self.drelu2 = nn.ReLU()
		# 512 8x8
		self.dconv3 = nn.ConvTranspose2d(1024, 512, 4, 2, 1)
		self.dbn3 = nn.BatchNorm2d(512)
		self.drelu3 = nn.ReLU()
		# 512 16x16
		self.dconv4 = nn.ConvTranspose2d(1024, 256, 4, 2, 1)
		self.dbn4 = nn.BatchNorm2d(256)
		self.drelu4 = nn.ReLU()
		# 256 32x32
		self.dconv5 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
		self.dbn5 = nn.BatchNorm2d(128)
		self.drelu5 = nn.ReLU()
		# 128 64x64
		self.dconv6 = nn.ConvTranspose2d(256, 64, 4, 2, 1)
		self.dbn6 = nn.BatchNorm2d(64)
		self.drelu6 = nn.ReLU()
		# 64 128x128
		self.dconv7 = nn.ConvTranspose2d(128, 32, 4, 2, 1)
		self.dbn7 = nn.BatchNorm2d(32)
		self.drelu7 = nn.ReLU()
		# 32 256x256
		self.conv = nn.Conv2d(32, 1, 1)
		self.tanh = nn.Tanh()
		# 1 256x256

	def forward(self, x):
		x1 = self.erelu1(self.ebn1(self.econv1(x)))
		x2 = self.erelu2(self.ebn2(self.econv2(x1)))
		x3 = self.erelu3(self.ebn3(self.econv3(x2)))
		x4 = self.erelu4(self.ebn4(self.econv4(x3)))
		x5 = self.erelu5(self.ebn5(self.econv5(x4)))
		x6 = self.erelu6(self.ebn6(self.econv6(x5)))
		x7 = self.erelu7(self.ebn7(self.econv7(x6)))

		code = x7.view(x7.size(0), -1)
		code = self.fc1(code)
		code = self.fc2(code)
		code = code.reshape(code.size(0), 512, 2, 2)

		y1 = self.drelu1(self.dbn1(self.dconv1(code)))
		y2 = self.drelu2(self.dbn2(self.dconv2( torch.cat([x6, y1], dim=1) )))
		y3 = self.drelu3(self.dbn3(self.dconv3( torch.cat([x5, y2], dim=1) )))
		y4 = self.drelu4(self.dbn4(self.dconv4( torch.cat([x4, y3], dim=1) )))
		y5 = self.drelu5(self.dbn5(self.dconv5( torch.cat([x3, y4], dim=1) )))
		y6 = self.drelu6(self.dbn6(self.dconv6( torch.cat([x2, y5], dim=1) )))
		y7 = self.drelu7(self.dbn7(self.dconv7( torch.cat([x1, y6], dim=1) )))
		y = self.tanh(self.conv(y7))
		y = y.reshape(y.size(0), 256, 256)

		return y


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)