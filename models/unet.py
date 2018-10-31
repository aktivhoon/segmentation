import torch.nn as nn
from .unet_parts import *
import torch.nn.functional as F

class Unet2D(nn.Module):
	def __init__(self, feature_scale = 4, n_classes = 1,
		is_deconv = True, norm = nn.BatchNorm2d, act=nn.ReLU):

		super(Unet2D, self).__init__()
		filters = [64, 128, 256, 512, 1024]

		# downsampling
		self.conv1 = UnetConv2D(1, filters[0], norm, act = act)
		self.pool1 = nn.MaxPool2d(kernel_size = 2)

		self.conv2 = UnetConv2D(filters[0], filters[1], norm, act = act)
		self.pool2 = nn.MaxPool2d(kernel_size = 2)

		self.conv3 = UnetConv2D(filters[1], filters[2], norm, act = act)
		self.pool3 = nn.MaxPool2d(kernel_size = 2)

		self.conv4 = UnetConv2D(filters[2], filters[3], norm, act = act)
		self.pool4 = nn.MaxPool2d(kernel_size = 2)

		self.center = UnetConv2D(filters[3], filters[4], norm, act = act)

		# upsampling

		self.up_concat4 = UnetUpConv2D(filters[4], filters[3], norm, is_deconv, act = act)
		self.up_concat3 = UnetUpConv2D(filters[3], filters[2], norm, is_deconv, act = act)
		self.up_concat2 = UnetUpConv2D(filters[2], filters[1], norm, is_deconv, act = act)
		self.up_concat1 = UnetUpConv2D(filters[1], filters[0], norm, is_deconv, act = act)

		# final conv (without any concat)
		self.final = nn.Conv2d(filters[0], n_classes, 1)

		# initialize weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				m.apply(weights_init_kaiming)
			elif isinstance(m, nn.BatchNorm2d):
				m.apply(weights_init_kaiming)

	def forward(self, inputs):
		conv1 = self.conv1(inputs)
		pool1 = self.pool1(conv1)

		conv2 = self.conv2(pool1)
		pool2 = self.pool2(conv2)

		conv3 = self.conv3(pool2)
		pool3 = self.pool3(conv3)

		conv4 = self.conv4(pool3)
		pool4 = self.pool4(conv4)

		center = self.center(pool4)

		up4 = self.up_concat4(conv4, center)
		up3 = self.up_concat3(conv3, up4)
		up2 = self.up_concat2(conv2, up3)
		up1 = self.up_concat1(conv1, up2)

		final = self.final(up1)
		return final

if __name__ == "__main__":
	import torch
	input2D = torch.randn([1, 1, 512, 512])
	model = Unet2D()
	output2D = model(input2D)

	print("input shape : \t", input2D.shape)
	print("output shape: \t", output2D.shape)

	from torchsummary import summary
	summary(model.cuda(), (1, 512, 512))
