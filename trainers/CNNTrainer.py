import os
from multiprocessing import Pool, Queue, Process

import scipy
import utils
import numpy as np

import torch
import torch.nn as nn
from .BaseTrainer import BaseTrainer
from models.unet_parts import weights_init_kaiming
from utils import torch_downsample

from sklearn.metrics import f1_score, confusion_matrix, recall_score, jaccard_similarity_score, roc_curve, precision_recall_curve, roc_auc_score, auc

class CNNTrainer(BaseTrainer):
	def __init__(self, arg, G, torch_device, recon_loss):
		super(CNNTrainer, self).__init__(arg, torch_device)
		self.recon_loss = recon_loss

		self.G = G
		self.lrG = arg.lrG
		self.beta = arg.beta
		self.fold = arg.fold
		self.optim = torch.optim.Adam(self.G.parameters(), lr = arg.lrG, betas = arg.beta)

		self.best_metric = 0
		self.sigmoid = nn.Sigmoid().to(self.torch_device)
		self.name = arg.name

		self.load()
		self.prev_epoch_loss = 0

	def save(self, epoch):
		filename = self.name
		save_path = self.save_path + "/fold%s"%(self.fold)
		if os.path.exists(self.save_path) is False:
			os.mkdir(self.save_path)
		if os.path.exists(save_path) is False:
			os.mkdir(save_path)

		torch.save({"model_type": self.model_type,
			"start_epoch": epoch + 1,
			"network": self.G.state_dict(),
			"optimizer": self.optim.state_dict(),
			"best_metric": self.best_metric,
			}, save_path + "/%s.pth.tar" % (filename))
		print("Model saved %d epoch" % (epoch))

	def load(self):
		save_path = self.save_path + "/fold%s"%(self.fold)
		if os.path.exists(save_path + "/" + self.name + ".pth.tar") is True:
			print("Load %s File" % (save_path))
			ckpoint = torch.load(save_path + "/models.pth.tar")
			if ckpoint["model_type"] != self.model_type:
				raise ValueError("Ckpoint Model Type is %s" % (ckpoint["model_type"]))

			self.G.load_state_dict(ckpoint['network'])
			self.optim.load_state_dict(ckpoint['optimizer'])
			self.start_epoch = ckpoint['start_epoch']
			self.best_metric = ckpoint['best_metric']
			print("Load Model Type: %s, epoch: %d" % (ckpoint["model_type"], self.start_epoch))
		else:
			print("Load Failed, not exists file")

	def _init_model(self):
		for m in self.G.modules():
			m.apply(weights_init_kaiming)
		self.optim = torch.optim.Adam(self.G.parameters(), lr = self.lrG, betas = self.beta)

	def train(self, train_loader, val_loader = None):
		print("\nStart Train")

		for epoch in range(self.start_epoch, self.epoch):
			for i, (input_, target_, _) in enumerate(train_loader):
				self.G.train()
				input_, target_ = input_.to(self.torch_device), target_.to(self.torch_device)
				output_ = self.G(input_)

				recon_loss = self.recon_loss(output_, target_)
				
				self.optim.zero_grad()
				recon_loss.backward()
				self.optim.step()

				input_np = input_.type(torch.FloatTensor).numpy()
				target_np = target_.type(torch.FloatTensor).numpy()
				output_np = output_.data.cpu().numpy()

				if recon_loss < -100:
					target_name = "weird_target_" + str(epoch) + "_" + str(i)
					target_path = "%s/fold%s/%s" % (self.save_path, self.fold, target_name)
					np.save(target_path + '.npy', target_np)
					output_name = "weird_output_" + str(epoch) + "_" + str(i)
					output_path = "%s/fold%s/%s" % (self.save_path, self.fold, output_name)
					np.save(output_path + '.npy', output_np)

				if (i % 50) == 0:
					self.logger.will_write("[Train] epoch: %d loss: %f" % (epoch, recon_loss))

			if val_loader is not None:
				self.valid(epoch, val_loader)
			else:
				self.save(epoch)
		print("End Train\n")

	def forward_for_test(self, input_, target_):
		input_ = input_.to(self.torch_device)
		output_ = self.G(input_).sigmoid()
		target_ = target_.to(self.torch_device)
		return input_, output_, target_

	def valid(self, epoch, val_loader):
		self.G.eval()
		with torch.no_grad():
			# (tn, fp, fn, tp)
			cm = utils.ConfusionMatrix()

			for i, (input_, target_, _) in enumerate(val_loader):
				_, output_, target_ = self.forward_for_test(input_, target_)

				cm.update(utils.confusion_matrix_2d(output_, target_, 0.5, reduce = False), n = output_.shape[0])

			metric = cm.jcc.avg + cm.dice.avg
			if metric > self.best_metric:
				self.best_metric = metric
				self.save(epoch)

			self.logger.write("[Val] epoch: %d f05: %f f1: %f f2: %f jacard:%f dice: %f" % (epoch, cm.f05, cm.f1, cm.f2, cm.jcc.avg, cm.dice.avg))

	def get_best_th(self, loader):
		y_true = np.array([])
		y_pred = np.array([])
		for i, (input_, target_, _) in enumerate(loader):
			input_, output_, target_ = self.forward_for_test(input_, target_)
			target_np = utils.slice_threshold(target_, 0.5)

			y_true = np.concatenate([y_true, target_np.flatten()], axis = 0)
			y_pred = np.concatenate([y_pred, output_.flatten()], axis = 0)


		pr_values = np.array(precision_recall_curve(y_true, y_pred))

		# To Do : F0.5 score
		f_best, th_best = -1, 0
		for precision, recall, threshold in zip(*pr_value):
			f05 = (5 * precision * recall) / (precision + (4 * recall))
			if f05 > f_best:
				f_best = f05
				th_best = threshold

		return f_best, th_best

	def test(self, test_loader, val_loader):
		print("\nStart Test")
		self.G.eval()
		with torch.no_grad():
			cm = utils.ConfusionMatrix()

			y_true = np.array([])
			y_pred = np.array([])

			for i, (input_, target_, f_name) in enumerate(test_loader):
				input_, output_, target_ = self.forward_for_test(input_, target_)
				cm.update(utils.confusion_matrix_2d(output_, target_, 0.5, reduce=False), n=output_.shape[0])

				input_np = input_.type(torch.FloatTensor).numpy()
				target_np = target_.type(torch.FloatTensor).numpy()
				output_np = output_.type(torch.FloatTensor).numpy()

				y_true = np.concatenate([y_true, target_np.flatten()], axis = 0)
				y_pred = np.concatenate([y_pred, output_np.flatten()], axis = 0)

				for batch_idx in range(0, input_.shape[0]):
					input_b = input_np[batch_idx, 0, :, :]
					target_b = target_np[batch_idx, 0, :, :]
					output_b = output_np[batch_idx, 0, :, :]

					save_path = "%s/fold%s/%s" % (self.save_path, self.fold, f_name[batch_idx][:-4])
					utils.image_save(save_path, input_b, target_b, output_b)
					self.logger.will_write("[Save] fname:%s dice:%f jss:%f" % (f_name[batch_idx][:-4], cm.dice.val[batch_idx], cm.jcc.val[batch_idx]))

			pr_values = np.array(precision_recall_curve(y_true, y_pred))

			roc_auc = roc_auc_score(y_true, y_pred)
			pr_auc = auc(pr_values[0], pr_values[1], reorder=True)

		self.logger.write("Best dice:%f jcc:%f f05:%f f1:%f f2:%f roc:%f pr:%f" % (cm.dice.avg, cm.jcc.avg, cm.f05, cm.f1, cm.f2, roc_auc, pr_auc))
		print("End Test\n")
