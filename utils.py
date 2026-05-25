import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def init_module_weights(m, std: float = 0.02):
	"""
	Initialise weights for common layer types using truncated normal distribution.
	"""
	if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d, nn.Linear)):
		nn.init.trunc_normal_(m.weight, std=std)
		if m.bias is not None:
			nn.init.constant_(m.bias, 0)


class Projector(nn.Module):
	"""MLP projector built from a spec string like '256-512-128'."""

	def __init__(self, mlp_spec):
		super().__init__()
		layers = []
		f = list(map(int, mlp_spec.split("-")))
		for i in range(len(f) - 2):
			layers.append(nn.Linear(f[i], f[i + 1]))
			layers.append(nn.BatchNorm1d(f[i + 1]))
			layers.append(nn.ReLU(True))

		layers.append(nn.Linear(f[-2], f[-1], bias=False))
		self.net = nn.Sequential(*layers)
		self.out_dim = f[-1]

	def forward(self, x):
		return self.net(x)
