import torch
import torch.nn as nn

from modules.frame_features import get_feature_extractor

class PTVQA(nn.Module):
	def __init__(self):
		super(PTVQA, self).__init__()

		self.feature_extractor = get_feature_extractor("resnet")

	def forward(self, frames, question):
		frame_features = self.feature_extractor(frames)
		return frame_features

def get_PTVQA():
	return PTVQA()
