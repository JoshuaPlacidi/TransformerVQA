import torch
import torch.nn as nn
from torchvision import models
from transformers import DeiTConfig, DeiTModel, DeiTFeatureExtractor
import numpy as np

class ResNet(nn.Module):
	def __init__(self, h_dim):
		super(ResNet, self).__init__()
		self.resnet = models.resnet18(pretrained=True)
		self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))
		#self.to_hid = nn.Linear(512, h_dim)
	
	def forward(self, x):
		x = self.resnet(x).squeeze()
		#x = self.to_hid(x)
		return x

# TODO Implement DeiT using huggingface
class DeiT(nn.Module):
	def __init__(self, h_dim):
		super(DeiT, self).__init__()
		# self.config = DeiTConfig()
		# self.visual_encoder = DeiTModel(self.config)
		self.feature_extractor = DeiTFeatureExtractor.from_pretrained("facebook/deit-base-distilled-patch16-224")
		self.deit_model = DeiTModel.from_pretrained("facebook/deit-base-distilled-patch16-224")
  

	def forward(self, x):
		ans = []
		for i in range(0, 72):
			inputs = self.feature_extractor(x[i, :, :, :], return_tensors="pt") #needs as a input or one image (single tensor) or a list of tensors 
			with torch.no_grad():
				outputs = self.deit_model(**inputs)[0]
				ans.append(outputs)
				# print(outputs.shape)
		# ans = self.feature_extractor(x[0, :, :, :], return_tensors="np")
		# print(torch.stack(ans).shape)
		return torch.stack(ans).squeeze()

	

def get_feature_extractor(model="resnet", h_dim=None):
	if model=="resnet":
		return ResNet(h_dim)
	elif model=="deit":
		return DeiT(h_dim)
	else:
		raise Exception("Feature extractor model not recognized:", model)