import torch
from torchvision import models
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.autograd import Variable
# from torch.legacy.nn import SpatialCrossMapLRN
# import module

# class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
# class torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
# class torch.nn.LocalResponseNorm(size, alpha=0.0001, beta=0.75, k=1)
# class torch.nn.PReLU(num_parameters=1, init=0.25)

class LRN(nn.Module):
    def __init__(self, local_size=1, alpha=1.0, beta=0.75, ACROSS_CHANNELS=True):
        super(LRN, self).__init__()
        self.ACROSS_CHANNELS = ACROSS_CHANNELS
        if ACROSS_CHANNELS:
            self.average=nn.AvgPool3d(kernel_size=(local_size, 1, 1),
                    stride=1,
                    padding=(int((local_size-1.0)/2), 0, 0))
        else:
            self.average=nn.AvgPool2d(kernel_size=local_size,
                    stride=1,
                    padding=int((local_size-1.0)/2))
        self.alpha = alpha
        self.beta = beta


    def forward(self, x):
        if self.ACROSS_CHANNELS:
            div = x.pow(2).unsqueeze(1)
            div = self.average(div).squeeze(1)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        else:
            div = x.pow(2)
            div = self.average(div)
            div = div.mul(self.alpha).add(1.0).pow(self.beta)
        x = x.div(div)
        return x



class Flatten(nn.Module):
	def forward(self, input):
		return input.view(input.size(0), -1)


class SkipBlock(nn.Module):
	"""docstring for SkipBlock"""
	def __init__(self, in_planes, out_planes, conv_kernel_size, conv_stride):
		super(SkipBlock, self).__init__()

		self.skip_block = nn.Sequential()
		self.skip_block.add_module("conv_reduce", nn.Conv2d(in_planes, out_planes, kernel_size=conv_kernel_size, stride=conv_stride))
		self.skip_block.add_module("relu_reduce", nn.PReLU())
		self.skip_block.add_module("conv_reduce_flat", Flatten())
		
		# self.conv_reduce = nn.Conv2d(in_planes, out_planes, kernel_size=par_kernel_size, stride=par_stride)
		# self.relu_reduce = torch.nn.PReLU()
		# self.conv_reduce_flat = Flatten()


	def forward(self,x):
		out = self.skip_block.forward(x)
		return out


class BasicBlock1(nn.Module):
	"""docstring for BasicBlock1"""
	def __init__(self, in_planes, out_planes, conv_kernel_size, conv_stride, pool_kernel_size, pool_stride):
		super(BasicBlock1, self).__init__()

		self.basic_block = nn.Sequential()
		self.basic_block.add_module("conv", nn.Conv2d(in_planes, out_planes, kernel_size=conv_kernel_size, stride=conv_stride))
		self.basic_block.add_module("relu", nn.ReLU())
		self.basic_block.add_module("pool", nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride))
		self.norm = LRN(local_size=5, alpha=0.0001, beta=0.75)

		## LRN is currently not present in pytorch. Have to ask someone how to do this
		# self.basic_block.add_module("norm", SpatialCrossMapLRN(size=5, alpha=0.0001, beta=0.75))

		# self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=conv_kernel_size, stride=conv_stride)
		# self.relu = torch.nn.ReLU()
		# self.pool = torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

	def forward(self,x):
		# print("here!")
		# out1 = self.conv(x)
		# return out1
		out = self.basic_block.forward(x)
		return out
	
class BasicBlock2(nn.Module):
	"""docstring for BasicBlock2"""
	def __init__(self, in_planes, out_planes, conv_kernel_size, conv_padding, conv_groups, pool_kernel_size, pool_stride):
		super(BasicBlock2, self).__init__()

		self.basic_block = nn.Sequential()
		self.basic_block.add_module("conv", nn.Conv2d(in_planes, out_planes, kernel_size=conv_kernel_size, padding=conv_padding, groups=conv_groups))
		self.basic_block.add_module("relu", nn.ReLU())
		self.basic_block.add_module("pool", nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride))
		## LRN is currently not present in pytorch. Have to ask someone how to do this
		# self.basic_block.add_module("norm", torch.nn.SpatialCrossMapLRN(size=5, alpha=0.0001, beta=0.75))

		# self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=par_kernel_size, stride=par_stride)
		# self.relu = torch.nn.ReLU()
		# self.pool = torch.nn.MaxPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
		# self.norm = torch.nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75)

	def forward(self,x):
		out = self.basic_block.forward(x)
		return out


class ConvReLUBlock(nn.Module):
	"""docstring for ConvReLUBlock"""
	def __init__(self, in_planes, out_planes, groups_present=False):
		super(ConvReLUBlock, self).__init__()
		
		self.conv_relu_block = nn.Sequential()
		if groups_present:
			self.conv_relu_block.add_module("conv", nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, groups=2))
		else:
			self.conv_relu_block.add_module("conv", nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1))

		self.conv_relu_block.add_module("relu", nn.ReLU())

		# self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, groups=2)
		# self.relu = torch.nn.ReLU()

	def forward(self,x):
		out = self.conv_relu_block.forward(x)
		return out

		
class FullBlock(nn.Module):
	"""docstring for FullBlock"""
	def __init__(self):
		super(FullBlock, self).__init__()
		self.block1 = BasicBlock1(in_planes=3, out_planes=96, conv_kernel_size=11, conv_stride=4, pool_kernel_size=3, pool_stride=2)
		self.skip1 = SkipBlock(in_planes=96, out_planes=16, conv_kernel_size=1, conv_stride=1)

		self.block2 = BasicBlock2(in_planes=96, out_planes=256,conv_kernel_size=5, conv_padding=2, conv_groups=2, pool_kernel_size=3, pool_stride=2)
		self.skip2 = SkipBlock(in_planes=256, out_planes=32, conv_kernel_size=1, conv_stride=1)

		self.conv_relu_block3 = ConvReLUBlock(in_planes=256, out_planes=384)
		self.conv_relu_block4 = ConvReLUBlock(in_planes=384, out_planes=384, groups_present=True)
		self.conv_relu_block5 = ConvReLUBlock(in_planes=384, out_planes=256, groups_present=True)

		self.skip5 = SkipBlock(in_planes=256, out_planes=64, conv_kernel_size=1, conv_stride=1)
		self.pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
		self.pool5_flat = Flatten()

		self.fc6 = nn.Sequential()
		self.fc6.add_module("fc6", nn.Linear(37104*2,2048))
		self.fc6.add_module("relu6", nn.ReLU())
		

	def forward(self,x, y):
	# def forward(self,x):
		x_out1 = self.block1(x)
		x_out_skip1 = self.skip1(x_out1)

		x_out2 = self.block2(x_out1)
		x_out_skip2 = self.skip2(x_out2)

		x_out3 = self.conv_relu_block3(x_out2)
		x_out4 = self.conv_relu_block4(x_out3)
		x_out5 = self.conv_relu_block5(x_out4)

		x_out_skip5 = self.skip5(x_out5)
		
		x_out_pool = self.pool5(x_out5)
		x_out_pool = self.pool5_flat(x_out_pool)
		x_out = torch.cat((x_out_skip1,x_out_skip2, x_out_skip5,x_out_pool), dim=1)
		# print(x_out.data.size(), x_out_pool.data.size(), x_out_skip1.data.size(), x_out_skip2.data.size(), x_out_skip5.data.size())

		y_out1 = self.block1(x)
		y_out_skip1 = self.skip1(y_out1)

		y_out2 = self.block2(y_out1)
		y_out_skip2 = self.skip2(y_out2)

		y_out3 = self.conv_relu_block3(y_out2)
		y_out4 = self.conv_relu_block4(y_out3)
		y_out5 = self.conv_relu_block5(y_out4)

		y_out_skip5 = self.skip5(y_out5)
		
		y_out_pool = self.pool5(y_out5)
		y_out_pool = self.pool5_flat(y_out_pool)
		y_out = torch.cat((y_out_skip1,y_out_skip2, y_out_skip5,y_out_pool), dim=1)
		# print(y_out.data.size(), y_out_pool.data.size(), y_out_skip1.data.size(), y_out_skip2.data.size(), y_out_skip5.data.size())

		final_out = torch.cat((x_out, y_out), dim=1)
		# print(final_out.data.size())

		in_lstm = self.fc6(final_out)
		# print(in_lstm.data.size())
		return in_lstm



class Re3Net(nn.Module):
	"""docstring for Re3Net"""
	def __init__(self):
		super(Re3Net, self).__init__()
		self.full_block = FullBlock()
		self.lstm1 = nn.LSTMCell(2048, 1024)
		self.lstm2 = nn.LSTMCell(2048 + 1024, 1024)
		self.fc_final = nn.Linear(1024,4)


	def forward(self,x,y,prev_LSTM_state=False):
		out = self.full_block(x,y)

		h0 = Variable(torch.rand(1,1024))
		c0 = Variable(torch.rand(1,1024))

		lstm_out, h0 = self.lstm1(out, (h0,c0))

		lstm2_in = torch.cat((out, lstm_out), dim=1)
		# print(out.data.size(), lstm_out.data.size(), h0.data.size(), lstm2_in.data.size())
		lstm2_out, h1 = self.lstm2(lstm2_in, (h0,c0))
		# print(lstm2_out.data.size())

		out = self.fc_final(lstm2_out)
		# print (out)
		return out


net = Re3Net()
# print(net)
bs = 1
shape = (3,227,227)
input1 = Variable(torch.rand(bs, *shape))
input2 = Variable(torch.rand(bs, *shape))

ot = net.forward(input1, input2)

# print(net)
# print(ot.data.size())