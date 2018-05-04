import sys, os
import time
from model import *
from process_data import *
from trainModel import *
import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from matplotlib.widgets  import RectangleSelector
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

min_x = 0
min_y = 0
max_x = 0
max_x = 0

use_gpu = torch.cuda.is_available()

model_weights = '/path/to/trained/model'
test_data_dir = '../alov/imagedata++/test/05-Shape_video00024/'
transform = transforms.Compose([Normalize(), ToTensor()])

def line_select_callback(eclick, erelease):
	'eclick and erelease are the press and release events'
	global min_x, min_y, max_x, max_y
	x1, y1 = eclick.xdata, eclick.ydata
	x2, y2 = erelease.xdata, erelease.ydata
	min_x = min(x1,x2)
	min_y = min(y1,y2)
	max_x = max(x1,x2)
	max_y = max(y1,y2)


def get_bbox(path_image):
	dpi = 80.0
	image = Image.open(path_image).convert('RGB')
	figsize = (image.size[0]/dpi, image.size[1]/dpi)

	fig = plt.figure(frameon=False, figsize=figsize, dpi=dpi)
	axis = plt.Axes(fig, [0., 0., 1., 1.])
	axis.set_axis_off()
	fig.add_axes(axis)
	img = axis.imshow(image, aspect='normal')
	rs = RectangleSelector(axis, line_select_callback, drawtype='box', useblit=False, button=[1],minspanx=5, minspany=5, spancoords='pixels',interactive=True)
	plt.show()
	return np.array([min_x, min_y, max_x, max_y])


def get_sample(idx, all_frames, previous_bbox):
	prev = io.imread(all_frames[idx][0])
	curr = io.imread(all_frames[idx][1])
	
	# Cropping the prev image with twice the size of  prev bounding box and scale the cropped image to (227,227,3)
	scale = Rescale((227,227))
	transform_prev = transforms.Compose([CropPrev(128), scale])
	
	# print("previous_bbox", previous_bbox)
	prev_img = transform_prev({'image':prev, 'bb':previous_bbox})['image']
	# print("size: ", prev.size, curr.size, previous_bbox)

	# Cropping the current image with twice the size of  prev bounding box and scale the cropped image to (227,227,3)
	curr_img = transform_prev({'image':curr, 'bb':previous_bbox})['image']
	sample = {'previmg': prev_img, 'currimg': curr_img}
	# print (sample)
	return transform(sample)


def get_bounding_box_rect(sample, model, previous_bbox):
	x1, x2 = sample['previmg'], sample['currimg']

	if not use_gpu:
		x1, x2 = Variable(x1), Variable(x2)
	else:
		x1, x2 = Variable(x1.cuda()), Variable(x2.cuda())
	x1 = x1[None,:,:,:]
	x2 = x2[None,:,:,:]
	y = model(x1, x2)
	bb = y.data.cpu().numpy().transpose((1,0))
	bb = bb[:,0]

	bb = list(bb*(227./10))

	# unscaling
	patch_width = (previous_bbox[2]-previous_bbox[0])*2
	patch_height = (previous_bbox[3]-previous_bbox[1])*2
	# input image size to network
	
	unscaled_bbox = [bb[0]*patch_width/227,
					bb[1]*patch_height/227,
					bb[2]*patch_width/227,
					bb[3]*patch_height/227]

	bb = previous_bbox
	h = bb[3]-bb[1]
	w = bb[2]-bb[0]
	top = bb[1]-h/2
	left = bb[0]-w/2
	orig_currbb = [left+unscaled_bbox[0], top+unscaled_bbox[1], left+unscaled_bbox[2], top+unscaled_bbox[3]]

	bb = orig_currbb
	return bb


def test():
	model = Re3Net()
	if use_gpu:
		model = model.cuda()
	# model.load_state_dict(torch.load(model_weights))

	# test_data = os.listdir(test_data_dir)
	# test_data = [test_data_dir + files for files in test_data]
	# num_test_data = len(test_data)-1
	# test_data = np.array(test_data)
	# test_data.sort()
	# for d in range(num_test_data):
	# 	frames = []
	# 	frames = os.listdir(test_data[d])
	# 	frames = [test_data[d] + '/'+frame for frame in frames]
	# 	num_frames = len(test_data)-1
	# 	frames = np.array(frames)
	# 	frames.sort()

	# 	all_frames = []
	# 	for i in range(num_frames):
	# 		all_frames.append([frames[i], frames[i+1]])
	# 	all_frames = np.array(all_frames)
	# 	initial_bbox = get_bbox(all_frames[0][0])
	# 	print(initial_bbox)

	# 	previous_bbox = initial_bbox

	# 	fig,ax = plt.subplots(1)
	# 	for i in range(num_frames):
	# 		sample = get_sample(i, all_frames, previous_bbox)
	# 		curr_bbox = get_bounding_box_rect(sample, model, previous_bbox)
	# 		img_show = io.imread(all_frames[i][1])

	# 		ax.clear()
	# 		ax.imshow(img_show)
	# 		rect = patches.Rectangle((curr_bbox[0], curr_bbox[1]),curr_bbox[2]-curr_bbox[0],curr_bbox[3]-curr_bbox[1],linewidth=1,edgecolor='r',facecolor='none')
	# 		ax.add_patch(rect)
	# 		previous_bbox = curr_bbox
	# 	plt.show()

	frames = os.listdir(test_data_dir)
	frames = [test_data_dir + files for files in frames]
	num_frames = len(frames)-1
	frames = np.array(frames)	
	frames.sort()

	all_frames = []
	for i in range(num_frames):
		all_frames.append([frames[i], frames[i+1]])
	all_frames = np.array(all_frames)
	initial_bbox = get_bbox(all_frames[0][0])
	
	previous_bbox = initial_bbox
	fig,ax = plt.subplots(1)
	for i in range(num_frames):
		print(i)
		sample = get_sample(i, all_frames, previous_bbox)
		curr_bbox = get_bounding_box_rect(sample, model, previous_bbox)
		img_show = io.imread(all_frames[i][1])

		ax.clear()
		ax.imshow(img_show)
		rect = patches.Rectangle((curr_bbox[0], curr_bbox[1]),curr_bbox[2]-curr_bbox[0],curr_bbox[3]-curr_bbox[1],linewidth=1,edgecolor='r',facecolor='none')
		ax.add_patch(rect)
		previous_bbox = curr_bbox

	plt.show()


if __name__ == "__main__":
	test()