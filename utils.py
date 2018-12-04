# utils.py

#################################################
# 2D version of surface DSC is defined based on # 
# N Stanislav et al. arXiv:1809.04430 (2018).   #
# Please check their paper for furhter details. #
#################################################

import os
import numpy as np
import scipy
import torch.nn.functional as F
import torch
import math

from scipy import ndimage, misc
from PIL import Image
from PIL import ImageFilter

def confusion_matrix_2d(x, y, th = 0.5, reduce = True):
	x_ = x.gt(th).to(torch.float)
	y_ = y.gt(th).to(torch.float)

	c = 2 * y_ - x_

	if reduce:
		dim = [0, 1, 2, 3]
	else:
		dim = [1, 2, 3]

	tp = (c == 1).sum(dim=dim, dtype=torch.float)
	tn = (c == 0).sum(dim=dim, dtype=torch.float)
	fp = (c == -1).sum(dim=dim, dtype=torch.float)
	fn = (c == 2).sum(dim=dim, dtype=torch.float)

	return tp, tn, fp, fn

def confusion_matrix_3d(x, y, th = 0.5, reduce = True):
	x_ = x.gt(th).to(torch.float)
	y_ = y.gt(th).to(torch.float)

	c = 2 * y_ - x_

	if reduce:
		dim = [0, 1, 2, 3, 4]
	else:
		dim = [1, 2, 3, 4]

	tp = (c == 1).sum(dim=dim, dtype=torch.float)
	tn = (c == 0).sum(dim=dim, dtype=torch.float)
	fp = (c == -1).sum(dim=dim, dtype=torch.float)
	fn = (c == 2).sum(dim=dim, dtype=torch.float)
	return tp, tn, fp, fn

def binary(mask, th):
	mask[mask >= th] = 1
	mask[mask <  th] = 0

	return mask

def small_contour(mask, contour_width):
	contour = Image.fromarray(mask)
	contour = contour.convert('L')
	contour = contour.filter(ImageFilter.FIND_EDGES)
	contour = np.array(contour)

	# make sure borders are not drawn
	contour[[0, -1], :] = 0
	contour[:, [0, -1]] = 0

	# use a gaussian to define the contour width
	radius = contour_width / 5
	contour = Image.fromarray(contour)
	contour = contour.filter(ImageFilter.GaussianBlur(radius=radius))
	contour = (np.array(contour) > 0).astype(float)
	return contour

def big_contour(npy, rate, filter_rate):
	# erosion
	ero = ndimage.binary_erosion(npy, structure=np.ones((rate,rate)))

	# dilation
	dil = ndimage.grey_dilation(npy, size=(rate,rate), structure=np.ones((rate,rate))) / 2.0
	dil = binary(dil, 0.9)

	contour = dil - ero
	contour[contour <= 0] = 0

	contour = ndimage.filters.median_filter(contour, size=(filter_rate,filter_rate))
	contour = np.array(contour).astype(float)
	return contour

def surface_DSC(output_, target_):
	prediction = binary(output_, 0.5)
	ground_truth = binary(target_, 0.5)

	b_gt = big_contour(ground_truth, 5, 1)
	s_p = small_contour(prediction, 3)
	bs_gtp = np.sum((b_gt + s_p) == 2)

	b_p = big_contour(prediction, 5, 1)
	s_gt = small_contour(ground_truth, 3)
	bs_pgt = np.sum((b_p + s_gt) == 2)

	s_DSC = (bs_gtp + bs_pgt) / (np.sum(s_p) + np.sum(s_gt))
	return s_DSC

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n = 1):
		self.val = val
		self.sum += val.mean() * n
		self.count += n
		self.avg = self.sum / self.count

class SurfaceDSC(object):
	def __init__(self):
		self.sDSC = AverageMeter()

	def update(self, sDSC, n):
		self.sDSC.update(sDSC, n=n)

class ConfusionMatrix(object):
	def __init__(self):
		self.tp = AverageMeter()
		self.tn = AverageMeter()
		self.fp = AverageMeter()
		self.fn = AverageMeter()
		self.dice = AverageMeter()
		self.jcc = AverageMeter()

		self.f1 = 0
		self.f05 = 0
		self.f2 = 0

		self.precision = 0
		self.recall = 0

	def update(self, cm, n):
		self.tp.update(cm[0], n = n)
		self.tn.update(cm[1], n = n)
		self.fp.update(cm[2], n = n)
		self.fn.update(cm[3], n = n)

		self.dice.update(2 * cm[0] / (2 * cm[0] + cm [2] + cm[3] + 0.00001), n = n)
		self.jcc.update( cm[0] / (cm [2] + cm[3] + cm[0] + 0.00001), n = n)

		self.total = self.tp.sum + self.tn.sum + self.fp.sum + self.fn.sum

		self.prec = self.tp.sum / (self.tp.sum + self.fp.sum + 0.00001)
		self.recall = self.tp.sum / (self.tp.sum + self.fn.sum + 0.00001)

		self.f1 = (2 * self.prec * self.recall / (self.prec + self.recall + 0.00001)).mean()
		self.f05 = ((1 + 0.25) * self.prec * self.recall / (0.25 * self.prec + self.recall + 0.00001)).mean()
		self.f2 = ((1 + 4) * self.prec * self.recall / (4 * self.prec + self.recall + 0.00001)).mean()

def pearson_correlation_coeff(x, y):
	std_x = np.std(x)
	std_y = np.std(y)

	mean_x = np.mean(x)
	mean_y = np.mean(y)

	vx = (x - mean_x) / (std_x + 0.0001)
	vy = (y - mean_y) / (std_y + 0.0001)

	return np.mean(vx * vy)

def psnr(x, y):
	mse = np.linalg.norm(y - x)

	if mse == 0:
		return 100

	PIXEL_MAX = 1.

	return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def get_save_dir(arg):
	save_path = "./outs/" + arg.save_dir
	if os.path.exists(save_path) is False:
		os.mkdir(save_path)
	return save_path

def RVD(output, target):
	output_sum = output.sum()
	target_sum = target.sum()
	if output_sum == target_sum:
		return 1

	score = (output_sum - target_sum) / target_sum
	# Higher the better
	return -score

def torch_downsample(img, scale):
	# Create grid
	out_size = img.shape[-1] // scale
	x = torch.linspace(-1, 1, out_size).view(-1, 1).repeat(1, out_size)
	y = torch.linspace(-1, 1, out_size).repeat(out_size, 1)
	grid = torch.cat((x.unsqueeze(2), y.unsqueeze(2)), 2).cuda()

	return F.grid_sample(img, grid)

def get_roc_pr(tn, fp, fn, tp):
	sensitivity = tp / (tp + fn) if (tp + fn) != 0 else 1
	specificity = tn / (tn + fp) if (tn + fp) != 0 else 1

	precision = tp / (tp + fp) if (tp + fp) != 0 else 1
	recall = tp / (tp + fn) if (tp + fn) != 0 else 1

	f1 = (2 * tp) / ((2 * tp) + fp + fn) if ((2 * tp) + fp + fn) != 0 else 1
	jaccard = tp / (tp + fp + fn) if (tp + fp + fn) != 0 else 1

	return sensitivity, 1 - specificity, precision, recall, f1, jaccard

def slice_threshold(_np, th):
	return (_np >= th).astype(int)

def image_save(save_path, *args):
	total = np.concatenate(args, axis = 1)
	np.save(save_path + '.npy', total)
	scipy.misc.imsave(save_path + '.jpg', total)

def voxel_save(save_path, *args):
	total = np.concatenate(args, axis = 2)
	np.save(save_path + '.npy', total)

def slack_alarm(send_id, send_msg="Train Done"):
	"""
    send_id : slack id. ex) zsef123
    """
	from slackclient import SlackClient
	slack_client = SlackClient(os.environ.get('SLACK_BOT_TOKEN'))
	if slack_client.rtm_connect(with_team_state=False):
		ret = slack_client.api_call("chat.postMessage", channel="@"+send_id, text=send_msg, as_user=True)
		resp = "Send Failed" if ret['ok'] == False else "To %s, send %s"%(send_id, send_msg)
		print(resp)
	else:
		print("Client connect Fail")

if __name__=="__main__":
	#x = torch.Tensor([[1,2,3,4,5],[6,7,8,9,10]])
	#y = torch.Tensor([[1,2,3,4,5],[6,7,8,9,10]])

	x = np.array([[1,2,3,4,5],[6,7,8,9,10]])
	y = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

	print(pearson_correlation_coeff(x, y))

