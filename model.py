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

		self.iqa_encoder = TransformerEncoder(h_dim=config.h_dim, ff_dim=config.h_dim, num_heads=8, num_layers=6, dropout=0)

		self.lan_to_hid = nn.Linear(768, config.h_dim)
		self.hid_to_one = nn.Linear(config.h_dim, 1)
		self.softmax = nn.Softmax(dim=1)

	def forward(self, i, q, q_mask, a, a_mask):
		i_new = self.feature_extractor(i) # [batch_size, 512] -> Output of resnet
		# Calculate q
		q_encoded = self.language_encoder(q, q_mask) # [batch_size, config.question_length, 768 (bert)]
		# TODO: We are destroying everything bert has done??
		q_encoded_hid = self.lan_to_hid(q_encoded) # [batch_size, config.question_length, config.h_dim]

		# Calculate a
		stacked_a_tokens = torch.reshape(a, shape=(a.shape[0]*a.shape[1], -1)) # [batch_size * number_answers, padded_language_length_answer]
		stacked_a_masks = torch.reshape(a_mask, shape=(a_mask.shape[0]*a_mask.shape[1], -1)) # [batch_size * number_answers, padded_language_length_answer]

		stacked_answer_features = self.language_encoder(stacked_a_tokens, stacked_a_masks) # [batch_size * number_answers, padded_language_length_answer, 768 (bert)]

		a = torch.reshape(stacked_answer_features, shape=(i.shape[0], 5, config.padded_language_length_answer, -1)) # [batch_size, number_answers, padded_language_length_answer, 768 (bert)]
		a = self.lan_to_hid(a) # [batch_size, number_answers, padded_language_length_answer, config.hdim]

		#
		# Multi-Modal Combination
		#
		i_new = i_new.unsqueeze(1).unsqueeze(1) # [batch_size, 1, 1, 512] -> Output of resnet
		i_new = i_new.repeat(1,5,1,1) # [batch_size, number_answers, 1, 512] -> Output of resnet
		q_encoded_hid_unsqueeze = q_encoded_hid.unsqueeze(1) # [batch_size, 1, config.question_length, config.h_dim]
		q_encoded_hid_repeat = q_encoded_hid_unsqueeze.repeat(1,5,1,1) # [batch_size, 5, config.question_length, config.h_dim]
	
		# iqa[0][0][15:] == iqa[0][1][15:] All false (makes sense, padded_language_length is the part that differs)
		iqa = torch.cat((i_new,q_encoded_hid_repeat,a), dim=2) # [batch_size, 5, 1 + config.question_length + padded_language_length_answer, config.h_dim]

		# iqa[0][1] == reshaped_iqa[1] All true. Makes sense
		reshaped_iqa = torch.reshape(iqa, shape=(i_new.shape[0] * 5, -1, config.h_dim))


		stacked_iqa = self.iqa_encoder(reshaped_iqa, [(0, 1), (1, 1+config.padded_language_length_question), (1+config.padded_language_length_question, 1+config.padded_language_length_question+config.padded_language_length_answer)])
		iqa_encoded = torch.reshape(stacked_iqa, shape=(i_new.shape[0], 5, -1, config.h_dim))

		# Is this really what we want?? I though we wanted to take the first token or something, not the whole thing
		# iqa_encoded[0][0][:15] == iqa_encoded[0][1][:15] all true
		iqa_encoded_linear = self.hid_to_one(iqa_encoded)

		# [batch_size, 5, 1 + config.question_length + padded_language_length_answer, 1]
		iqa_encoded_linear_squeeze = iqa_encoded_linear[:,:,0,:].squeeze() # For each sample, all the values are the same
		
		output = self.softmax(iqa_encoded_linear_squeeze)

		return output

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
