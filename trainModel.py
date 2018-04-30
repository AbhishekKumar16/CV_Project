import os, sys
import torch
from torch.autograd import Variable
from torchvision import transforms

import math
# from torch.utils.data import Dataloader
import numpy as np
import torch.optim as optim

from accessories import *

import process_data
from process_data import ALOVDataset
import model

use_gpu = torch.cuda.is_available()

#globals
num_batches = 64
learning_rate = 0.00001 
save_directory = '../saved_models/'
save_model_step = 128

class ToTensor(object):
	"""Convert ndarrays in sample to Tensors."""

	def __call__(self, sample):
		prev_img, curr_img = sample['previmg'], sample['currimg']
		
		# swap color axis because
		# numpy image: H x W x C
		# torch image: C X H X W
		prev_img = prev_img.transpose((2, 0, 1))
		curr_img = curr_img.transpose((2, 0, 1))
		if 'currbb' in sample:
			currbb = sample['currbb']
			return {'previmg': torch.from_numpy(prev_img).float(),
					'currimg': torch.from_numpy(curr_img).float(),
					'currbb': torch.from_numpy(currbb).float()
					}
		else:
			return {'previmg': torch.from_numpy(prev_img).float(),
					'currimg': torch.from_numpy(curr_img).float()
					}


class Normalize(object):
	"""Returns image with zero mean and scales bounding box by factor of 10."""

	def __call__(self, sample):
		prev_img, curr_img = sample['previmg'], sample['currimg']
		self.mean = [104, 117, 123]
		prev_img = prev_img.astype(float)
		curr_img = curr_img.astype(float)
		prev_img -= np.array(self.mean).astype(float)
		curr_img -= np.array(self.mean).astype(float)

		if 'currbb' in sample:
			currbb = sample['currbb']
			currbb = currbb*(10./227);
			return {'previmg': prev_img,
					'currimg': curr_img,
					'currbb': currbb
					}
		else:
			return {'previmg': prev_img,
					'currimg': curr_img
					}



transform = transforms.Compose([Normalize(), ToTensor()])
num_synthetic_examples = 10


def make_training_sample(idx, dataset):
	orig_sample = dataset.get_orig_sample(idx)
	true_sample = dataset.get_sample(idx)
	true_tensor = transform(true_sample)

	orig_img = orig_sample['image']
	orig_bb = orig_sample['bb']

	x1_batch = torch.Tensor(num_synthetic_examples+1, 3, 227, 227)
	x2_batch = torch.Tensor(num_synthetic_examples+1, 3, 227, 227)
	y_batch = torch.Tensor(num_synthetic_examples+1, 4)

	x1_batch[0,:,:,:] = true_tensor['previmg']
	x1_batch[0,:,:,:] = true_tensor['currimg']
	y_batch[0,:] = true_tensor['currbb']

	for i in range(num_synthetic_examples):
		sample = {'image':orig_img, 'bb':orig_bb}
		prev_bb = random_crop(sample)
		crop_curr = transforms.Compose([CropCurr()])
		scale = Rescale((227,227))

		transform_prev = transforms.Compose([CropPrev(), scale])
		prev_img = transform_prev({'image':orig_img, 'bb':orig_bb})['image']
		# Crop current image with height and width twice the prev bounding box height and width
		# Scale the cropped image to (227,227,3)
		curr_obj = crop_curr({'image':orig_img, 'prevbb':prev_bb, 'currbb':orig_bb})
		curr_obj = scale(curr_obj)
		curr_img = curr_obj['image']
		curr_bb = curr_obj['bb']
		curr_bb = np.array(curr_bb)
		sample = {'previmg': prev_img,
				'currimg': curr_img,
				'currbb' : curr_bb
				}
		sample = transform(sample)
		x1_batch[i+1,:,:,:] = sample['previmg']
		x2_batch[i+1,:,:,:] = sample['currimg']
		y_batch[i+1,:] = sample['currbb']

	return x1_batch, x2_batch, y_batch


def train_model(net, datasets, optim, loss_function):

	curr_loss = 0

	for batch in range(num_batches):
		net.train()

		# currently training on just ALOV dataset
		for dataset in datasets:
			size_dataset = dataset.len
			rand_idx = np.random.randint(size_dataset, size=1)[0]
			print('size_dataset',size_dataset, rand_idx)

			x1, x2, y = make_training_sample(rand_idx, dataset)

			if use_gpu:
				x1, x2, y = Variable(x1.cuda()), Variable(x2.cuda()), Variable(y.cuda(), requires_grad=False)
			else:
				x1, x2, y = Variable(x1), Variable(x2), Variable(y, requires_grad=False)

			optim.zero_grad()

			output = net(x1,x2)
			loss = loss_function(output, y)

			loss.backward()
			optim.step()

			curr_loss = loss.data[0]
			print('[training] step = %d/%d, loss = %f, y=%d' % (batch, num_batches, curr_loss, len(y)))
			sys.stdout.flush()


		# val_loss = evaluate(model, dataloader, criterion, epoch)
		# print('Validation Loss: {:.4f}'.format(val_loss))

		if batch > 0 and batch%save_model_step==0:
			 path = save_directory + '_batch_' + str(batch) + '_loss_' + str(round(curr_loss, 3)) + '.pth'
			 torch.save(net.state_dict(), path)

	return net

def evaluate(model, dataloader, criterion, epoch):

	model.eval()
	dataset = dataloader.dataset
	toral_loss = 0

	
	for i in xrange(64):
		sample = dataset[i]
		sample['currimg'] = sample['currimg'][None,:,:,:]
		sample['previmg'] = sample['previmg'][None,:,:,:]
		x1, x2 = sample['previmg'], sample['currimg']
		y = sample['currbb']
		
		if use_gpu:
			x1 = Variable(x1.cuda())
			x2 = Variable(x2.cuda())
			y = Variable(y.cuda(), requires_grad=False)
		else:
			x1 = Variable(x1)
			x2 = Variable(x2)
			y = Variable(y, requires_grad=False)

		output = model(x1, x2)
		loss = criterion(output, y)
		total_loss += loss.data[0]
		print('[validation] epoch = %d, i = %d, loss = %f' % (epoch, i, loss.data[0]))

	seq_loss = running_loss/64
	return seq_loss


if __name__ == '__main__':

	alov = ALOVDataset('../alov/imagedata++/', '../alov/alov300++_rectangleAnnotation_full/')

	# intend to use Imganenet video dataset too
	# https://www.kaggle.com/c/imagenet-object-detection-from-video-challenge/data

	datasets = [alov]
	net = model.Re3Net()
	loss_function = torch.nn.L1Loss(size_average=False)

	if use_gpu:
		net = net.cuda()
		loss_function = loss_function.cuda()

	optim = optim.Adam(net.parameters(), lr=0.00001, weight_decay=0.0005)

	if os.path.exists(save_directory):
		print('Directory %s already exists', save_directory)
	else:
		os.makedirs(save_directory)

	net = train_model(net, datasets, optim, loss_function)
