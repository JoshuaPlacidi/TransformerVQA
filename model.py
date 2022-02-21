import torch
import torch.nn as nn

from modules.frame_features import get_feature_extractor
from modules.transformer_modules import TransformerEncoder, TransformerDecoder

class PTVQA(nn.Module):
	def __init__(self):
		super(PTVQA, self).__init__()

		# Frame feature extractor
		self.feature_extractor = get_feature_extractor("resnet")
		
		# Video Encoder
		self.video_encoder = TransformerEncoder(h_dim=256, ff_dim=256, num_heads=8, num_layers=6, dropout=0.1)

		# Language Encoder
		self.language_encoder = TransformerEncoder(h_dim=256, ff_dim=256, num_heads=8, num_layers=6, dropout=0.1)

		# Multimodal Encoder
		self.multimodal_encoder = TransformerEncoder(h_dim=256, ff_dim=256, num_heads=8, num_layers=6, dropout=0.1)

		# Multimodal Decoder
		self.multimodal_decoder = TransformerDecoder(h_dim=256, vocab_size=100, num_heads=8, dropout=0.1)

	def forward(self, frames, questions, answers):
		frame_features = self.feature_extractor(frames)

		language_features = self.language_encoder(question)

		multimodal_features = self.multimodal_encoder(torch.cat([frame_features, language_features]))

		predictions = self.multimoda_decoder(answers, multimodal_features)

		return predictions

def get_PTVQA():
	return PTVQA()
