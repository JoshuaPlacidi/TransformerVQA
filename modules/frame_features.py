import torch
import torch.nn as nn
from torchvision import models

def get_feature_extractor(model="resnet"):
	if model=="resnet":
		return get_resnet_extractor()
	else:
		raise Exception("Feature extractor model not recognized:", model)

def get_resnet_extractor():
	resnet = models.resnet18(pretrained=False)
	resnet = torch.nn.Sequential(*(list(resnet.children())[:-1]))
	return resnet
