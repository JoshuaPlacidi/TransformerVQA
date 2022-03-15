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
		self.feature_extractor = get_feature_extractor('resnet', h_dim=config.h_dim)
		
		# Video Encoder
		self.video_encoder = TransformerEncoder(h_dim=config.h_dim, ff_dim=config.h_dim, num_heads=8, num_layers=6, dropout=0.1)

		# Language Encoder
		#self.language_encoder = get_language_encoder()

		# VQ Encoder
		self.vq_encoder = TransformerEncoder(h_dim=config.h_dim, ff_dim=config.h_dim, num_heads=8, num_layers=6, dropout=0.1)

		# VQA Encoder
		self.vqa_encoder = TransformerEncoder(h_dim=config.h_dim, ff_dim=config.h_dim, num_heads=8, num_layers=6, dropout=0.1)

		# Multimodal Decoder
		# self.multimodal_decoder = TransformerDecoder(h_dim=config.h_dim, vocab_size=100, num_heads=8, dropout=0.1)

		self.lan_to_hid = nn.Linear(768, config.h_dim)
		self.hid_to_one = nn.Linear(config.h_dim, 1)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, i, i_masks, q, q_masks, a, a_mask):
		#question_tokens, question_masks = self.language_encoder.tokenize_text(questions) # shape: [batch_size, config.padded_language_length]
		#answer_tokens, answer_masks = self.language_encoder.tokenize_text(answer_choices, is_answer_list=True) # shape: [batch_size, 5 (num answers), config.padded_language_length]
		
		#
		# Uni-Modal Feature Extraction
		#

		# Calculate v
		# stacked_frames = torch.reshape(frames, shape=(config.batch_size*config.padded_frame_length, 3, config.image_size[0], config.image_size[1]))
		# stacked_frame_features = self.feature_extractor(stacked_frames)
		# frame_features = torch.reshape(stacked_frame_features, shape=(config.batch_size, config.padded_frame_length, -1))
		v = self.video_encoder(i)		

		# Calculate q
		#q = self.language_encoder(question_tokens, question_masks)
		q = self.lan_to_hid(q)

		# Calculate a
		#stacked_answer_tokens = torch.reshape(answer_tokens, shape=(config.batch_size*config.padded_language_length, -1))
		#stacked_answer_masks = torch.reshape(answer_masks, shape=(config.batch_size*config.padded_language_length, -1))
		#stacked_answer_features = self.language_encoder(stacked_answer_tokens, stacked_answer_masks)
		#a = torch.reshape(stacked_answer_features, shape=(config.batch_size, 5, config.padded_language_length, -1))
		a = self.lan_to_hid(a)

		#
		# Multi-Modal Combination
		#
		vq = torch.cat((v, q), dim=1)
		vq = self.vq_encoder(vq)
		vq = vq.unsqueeze(1).repeat(1,5,1,1)

		vqa = torch.cat((vq,a), dim=2)
		stacked_vqa = torch.reshape(vqa, shape=(i.shape[0] * 5, -1, config.h_dim))
		stacked_vqa = self.vqa_encoder(stacked_vqa, [(0, 20), (20, 34), (34, 42)])

		vqa = torch.reshape(stacked_vqa, shape=(i.shape[0], 5, -1, config.h_dim))
		vqa = self.hid_to_one(vqa)
		vqa = vqa[:,:,0,:].squeeze()
		vqa = self.softmax(vqa)

		return vqa

class IQA(nn.Module):
	def __init__(self):
		super(IQA, self).__init__()

		# Frame feature extractor
		self.feature_extractor = get_feature_extractor('resnet', h_dim=config.h_dim)
		
		# Language Encoder
		self.language_encoder = get_language_encoder()

		self.iqa_encoder = TransformerEncoder(h_dim=config.h_dim, ff_dim=config.h_dim, num_heads=8, num_layers=6, dropout=0.1)

		self.lan_to_hid = nn.Linear(768, config.h_dim)
		self.hid_to_one = nn.Linear(config.h_dim, 1)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, i, q, q_mask, a, a_mask):
		i = self.feature_extractor(i) # [batch_size, 512] -> Output of resnet


		# Calculate q
		q = self.language_encoder(q, q_mask) # [batch_size, config.question_length, 768 (bert)]
		# TODO: We are destroying everything bert has done??
		q = self.lan_to_hid(q) # [batch_size, config.question_length, config.h_dim]

		# Calculate a
		stacked_a_tokens = torch.reshape(a, shape=(a.shape[0]*a.shape[1], -1)) # [batch_size * number_answers, padded_language_length_answer]
		stacked_a_masks = torch.reshape(a_mask, shape=(a_mask.shape[0]*a_mask.shape[1], -1)) # [batch_size * number_answers, padded_language_length_answer]

		#print(stacked_a_tokens.shape)
		stacked_answer_features = self.language_encoder(stacked_a_tokens, stacked_a_masks)
		a = torch.reshape(stacked_answer_features, shape=(i.shape[0], 5, 10, -1))
		#print(a.shape)
		a = self.lan_to_hid(a)

		#print(a[0])

		#
		# Multi-Modal Combination
		#
		i = i.unsqueeze(1).unsqueeze(1)
		i = i.repeat(1,5,1,1)
		q = q.unsqueeze(1)
		q = q.repeat(1,5,1,1)

	
		iqa = torch.cat((i,q,a), dim=2)

		stacked_iqa = torch.reshape(iqa, shape=(i.shape[0] * 5, -1, config.h_dim))
		stacked_iqa = self.iqa_encoder(stacked_iqa, [(0, 1), (1, 11), (11, 21)])
		iqa = torch.reshape(stacked_iqa, shape=(i.shape[0], 5, -1, config.h_dim))

		iqa = self.hid_to_one(iqa)
		iqa = iqa[:,:,0,:].squeeze()
		iqa = self.softmax(iqa)

		return iqa

def get_IQA():
	return IQA()

def get_PTVQA():
	return PTVQA()

class ImageFeatureExtractor(nn.Module):
	def __init__(self):
		super(ImageFeatureExtractor, self).__init__()
		
		self.feature_extractor = get_feature_extractor('resnet', h_dim=config.h_dim)
	
	def forward(self, x):
		stacked_x = torch.reshape(x, shape=(x.shape[0]*config.padded_frame_length, 3, config.image_size[0], config.image_size[1]))
		stacked_x = self.feature_extractor(stacked_x)
		x = torch.reshape(stacked_x, shape=(x.shape[0], config.padded_frame_length, -1))
		return x

class TextFeatureExtractor(nn.Module):
	def __init__(self):
		super(TextFeatureExtractor, self).__init__()
		self.feature_extractor = get_language_encoder('BERT')
	
	def forward(self, x):
		stacked_x = torch.reshape(x, shape=(x.shape[0]*config.padded_frame_length, 3, config.image_size[0], config.image_size[1]))
		stacked_x = self.feature_extractor(stacked_x)
		x = torch.reshape(stacked_x, shape=(x.shape[0], config.padded_frame_length, -1))
		return x

def get_ImageFeatureExtractor():
	return ImageFeatureExtractor()
