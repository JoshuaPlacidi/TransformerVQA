import torch
import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
	def __init__(self, h_dim):
		super(ResNet, self).__init__()
		self.resnet = models.resnet18(pretrained=False)
		self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
		self.to_hid = nn.Linear(512, h_dim)
	
	def forward(self, x):
		x = self.resnet(x).squeeze()
		x = self.to_hid(x)
		return x

def get_feature_extractor(model="resnet", h_dim=None):
	if model=="resnet":
		return ResNet(h_dim)
	else:
		raise Exception("Feature extractor model not recognized:", model)