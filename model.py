import torch
import torch.nn as nn

from modules.frame_features import get_feature_extractor
from modules.language_encoder import get_language_encoder
from modules.transformer_modules import TransformerEncoder, TransformerDecoder

from transformers import DistilBertTokenizer

import config

class PTVQA(nn.Module):
	def __init__(self):
		super(PTVQA, self).__init__()

		# Frame feature extractor
		self.feature_extractor = get_feature_extractor("resnet", h_dim=config.h_dim)
		
		# Video Encoder
		self.video_encoder = TransformerEncoder(h_dim=256, ff_dim=256, num_heads=8, num_layers=6, dropout=0.1)

		# Language Encoder
		self.language_encoder = get_language_encoder()

		# Multimodal Encoder
		self.multimodal_encoder = TransformerEncoder(h_dim=config.h_dim, ff_dim=config.h_dim, num_heads=8, num_layers=6, dropout=0.1)

		# Multimodal Decoder
		# self.multimodal_decoder = TransformerDecoder(h_dim=config.h_dim, vocab_size=100, num_heads=8, dropout=0.1)

	def forward(self, frames, questions, answer_choices):
		question_tokens, question_masks = self.language_encoder.tokenize_text(questions) # shape: [batch_size, config.padded_language_length]
		answer_tokens, answer_masks = self.language_encoder.tokenize_text(answer_choices, is_answer_list=True) # shape: [batch_size, 5 (num answers), config.padded_language_length]
		
		#
		# Uni-Modal Feature Extraction
		#
		stacked_frames = torch.reshape(frames, shape=(config.batch_size*config.padded_frame_length, 3, config.image_size[0], config.image_size[1]))
		stacked_frame_features = self.feature_extractor(stacked_frames)
		frame_features = torch.reshape(stacked_frame_features, shape=(config.batch_size, config.padded_frame_length, -1))

		question_features = self.language_encoder(question_tokens, question_masks)

		stacked_answer_tokens = torch.reshape(answer_tokens, shape=(config.batch_size*config.padded_language_length, -1))
		stacked_answer_masks = torch.reshape(answer_masks, shape=(config.batch_size*config.padded_language_length, -1))
		stacked_answer_features = self.language_encoder(stacked_answer_tokens, stacked_answer_masks)
		answer_features = torch.reshape(stacked_answer_features, shape=(config.batch_size, 5, config.padded_language_length, -1))

		print(frame_features.shape)
		print(question_features.shape)
		print(answer_features.shape)

		#
		# Multi-Modal Combination
		#

		f_q = torch.cat((frame_features, question_features), dim=1)
		f_q = self.multimodal_encoder(f_q)

		print(f_q.shape)


		#multimodal_features = self.multimodal_encoder(torch.cat([frame_features, language_features]))

		#predictions = self.multimodal_decoder(answer_choices, multimodal_features)

		return 0

def get_PTVQA():
	return PTVQA()
