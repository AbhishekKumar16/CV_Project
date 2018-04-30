import numpy as np
import math
from skimage import io, transform, img_as_ubyte
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

import warnings
warnings.filterwarnings("ignore")

class Rescale(object):
	"""Rescale image and bounding box.
	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.

	"""
	def __init__(self, output_size):
		assert isinstance(output_size, (int, tuple))
		self.output_size = output_size

	def __call__(self, sample):
		image, bb = sample['image'], sample['bb']
		h, w = image.shape[:2]
		if isinstance(self.output_size, int):
		   if h > w:
			   new_h, new_w = self.output_size*h/w, self.output_size
		   else:
			   new_h, new_w = self.output_size, self.output_size*w/h
		else:
			new_h, new_w = self.output_size

		new_h, new_w = int(new_h), int(new_w)
		img = transform.resize(image, (new_h, new_w))
		img = img_as_ubyte(img)
		bb = [bb[0]*new_w/w, bb[1]*new_h/h, bb[2]*new_w/w, bb[3]*new_h/h]
		return {'image': img, 'bb':bb}

class CropPrev(object):
	"""Crop the previous frame image using the bounding box specifications.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""
	def __call__(self, sample):
		image, bb = sample['image'], sample['bb']
		image = img_as_ubyte(image)
		if (len(image.shape) == 2):
			image = np.repeat(image[...,None],3,axis=2)
		im = Image.fromarray(image)
		w = bb[2]-bb[0]
		h = bb[3]-bb[1]
		left = bb[0]-w/2
		top = bb[1]-h/2
		right = left + 2*w
		bottom = top + 2*h
		box = (left, top, right, bottom)
		box = tuple([int(math.floor(x)) for x in box])
		res = np.array(im.crop(box))
		bb = [bb[0]-left, bb[1]-top, bb[2]-left, bb[3]-top]
		return {'image':res, 'bb':bb}

		
class CropCurr(object):
	"""Crop the current frame image using the bounding box specifications.

	Args:
		output_size (tuple or int): Desired output size. If int, square crop
			is made.
	"""
	def __call__(self, sample):
		image, prevbb, currbb = sample['image'], sample['prevbb'], sample['currbb']
		image = img_as_ubyte(image)
		if (len(image.shape) == 2):
			image = np.repeat(image[...,None],3,axis=2)
		im = Image.fromarray(image)
		w = prevbb[2]-prevbb[0]
		h = prevbb[3]-prevbb[1]
		left = prevbb[0]-w/2
		top = prevbb[1]-h/2
		right = left + 2*w
		bottom = top + 2*h
		box = (left, top, right, bottom)
		box = tuple([int(math.floor(x)) for x in box])
		res = np.array(im.crop(box))
		bb = [currbb[0]-left, currbb[1]-top, currbb[2]-left, currbb[3]-top]
		return {'image':res, 'bb':bb}


def random_crop(sample,
				lambda_scale_frac=15,
				lambda_shift_frac=5,
				min_scale=0.4,
				max_scale=0.4):
	image, bb = sample['image'], sample['bb']
	image = img_as_ubyte(image)
	if (len(image.shape) == 2):
		image = np.repeat(image[...,None],3,axis=2)
	im = Image.fromarray(image)
	cols = image.shape[1]
	rows = image.shape[0]
	width = bb[2]-bb[0]
	height = bb[3]-bb[1]
	center_x = bb[0] + width/2
	center_y = bb[1] + height/2

	# motion smoothness model
	# adapted from Held's implementation - https://github.com/davheld/GOTURN/
	kMaxNumTries = 10
	kContextFactor = 2.
	new_width = -1
	num_tries_width = 0
	# get new width
	while ((new_width < 0 or new_width > cols-1) and num_tries_width < kMaxNumTries):
		width_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sided(lambda_scale_frac)))
		new_width = width * (1 + width_scale_factor)
		new_width = max(1.0, min(cols - 1, new_width))
		num_tries_width = num_tries_width + 1;

	new_height = -1
	num_tries_height = 0
	# get new height
	while ((new_height < 0 or new_height > rows-1) and num_tries_height < kMaxNumTries):
		height_scale_factor = max(min_scale, min(max_scale, sample_exp_two_sided(lambda_scale_frac)))
		new_height = height * (1 + height_scale_factor)
		new_height = max(1.0, min(rows - 1, new_height))
		num_tries_height = num_tries_height + 1;

	first_time_x = True;
	new_center_x = -1
	num_tries_x = 0
	# get new center X
	while ((first_time_x or
			new_center_x < center_x - width * kContextFactor / 2 or
			new_center_x > center_x + width * kContextFactor / 2 or
			new_center_x - new_width / 2 < 0 or
			new_center_x + new_width / 2 > cols)
			and num_tries_x < kMaxNumTries):
		new_x_temp = center_x + width * sample_exp_two_sided(lambda_shift_frac)
		new_center_x = min(cols - new_width / 2, max(new_width / 2, new_x_temp))
		first_time_x = False
		num_tries_x = num_tries_x + 1

	first_time_y = True;
	new_center_y = -1
	num_tries_y = 0
	# get new center Y
	while ((first_time_y or
			new_center_y < center_y - height * kContextFactor / 2 or
			new_center_y > center_y + height * kContextFactor / 2 or
			new_center_y - new_height / 2 < 0 or
			new_center_y + new_height / 2 > rows)
			and num_tries_y < kMaxNumTries):
		new_y_temp = center_y + height * sample_exp_two_sided(lambda_shift_frac)
		new_center_y = min(rows - new_height / 2, max(new_height / 2, new_y_temp))
		first_time_y = False
		num_tries_y = num_tries_y + 1

	box = [new_center_x - new_width/2,
		   new_center_y - new_height/2,
		   new_center_x + new_width/2,
		   new_center_y + new_height/2]
	box = [int(math.floor(x)) for x in box]
	return box

def sample_exp_two_sided(lambda_):
	t = np.random.randint(2, size=1)[0]
	pos_or_neg = 1 if (t%2 == 0) else -1
	rand_uniform = np.random.rand(1)[0]
	return np.log(rand_uniform) / lambda_ * pos_or_neg
