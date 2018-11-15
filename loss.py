import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import filters


class FocalLoss(nn.Module):
	def __init__(self, gamma = 2):
		super().__init__()
		self.gamma = gamma

	def forward(self, input, target):
		if not (target.size() == input.size()):
			raise ValueError("Target size ({}) must be the same as input size({})".format(target.size(), input.size()))

		max_val = (-input).clamp(min=0)
		loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()

		invprobs = F.logsigmoid(-input * (target * 2 - 1))
		loss = (invprobs * self.gamma).exp() * loss

		return loss.mean()

class TverskyLoss:
	def __init__(self, alpha, torch_device):
		self.a = alpha
		self.b = 1 - alpha
		self.smooth = torch.tensor(1.0, device = torch_device)

	def __call__(self, predict, target_):
		predict = F.sigmoid(predict)
		target_f = target_.view(-1) 	# g
		predict_f = predict.view(-1) 	# p


		# PG + a * P_G + b * G_P
		PG = (predict_f * target_f).sum()
		P_G = (predict_f * (1 - target_f)).sum() * self.a
		G_P = ((1-predict_f) * target_f).sum() * self.b

		loss = PG / (PG + P_G + G_P + self.smooth)
		return loss * -1


if __name__ == "__main__":

	def get_grad(*args):
		print("Grad : \n", args)


	target = torch.tensor([[[0,1,0],[1,1,1],[0,1,0]]], dtype=torch.float, requires_grad=True)
	predicted = torch.tensor([[[1,1,0],[0,0,0],[1,0,0]]], dtype=torch.float, requires_grad=True)
	print("Prediction : \n", predicted); print("GroudTruth : \n", target)
	predicted.register_hook(get_grad)

	loss = TverskyLoss(0.3, torch.device("cpu"))
	l = loss(predicted, target)
	print("Loss : ", l)
	l.backward()

