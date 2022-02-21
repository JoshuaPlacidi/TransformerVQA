import torch
import torch.nn as nn

from modules.frame_features import get_feature_extractor
from modules.video_encoder import get_transformer_video_encoder

class PTVQA(nn.Module):
	def __init__(self):
		super(PTVQA, self).__init__()

		self.feature_extractor = get_feature_extractor("resnet")
		self.video_encoder = get_transformer_video_encoder(h_dim=256, ff_dim=256)

	def forward(self, frames, question):
		frame_features = self.feature_extractor(frames)
		return frame_features

def get_PTVQA():
	return PTVQA()
