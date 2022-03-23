import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import config

class TransformerEncoder(nn.Module):
	'''
	Transformer encoder
	'''
	def __init__(self, h_dim, ff_dim, num_heads=8, num_layers=6, dropout=0.1):
		'''
		h_dim = hidden dimension size
		ff_dim = feedforard dimension size
		num_heads = number of attention heads
		num_layers = number of encoder layers
		dropout = dropout rate
		'''
		super(TransformerEncoder, self).__init__()
		self.pos_encoder = PositionalEncoding(h_dim)
		self.encoder_layer = TransformerEncoderLayer(h_dim=h_dim, ff_dim=ff_dim, num_heads=num_heads, dropout=dropout)
		self.layer_norm = nn.LayerNorm(h_dim)

		# TODO
		self.segment_embedding = nn.Embedding(3, h_dim)

		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=self.layer_norm)
 
	def forward(self, x, segment_mapping=None, key_padding_mask=None):
		# TODO is this correct?
		if segment_mapping:
			def generate_segment_embedding(label, num_repeat, current_device):
				return self.segment_embedding(torch.tensor(label).long().to(current_device)).unsqueeze(0).repeat(num_repeat, 1)

			pos_and_segment_embedding = torch.cat([self.pos_encoder(x[:, ini:fin]) + generate_segment_embedding(i, fin-ini, x.device) for i, (ini, fin) in enumerate(segment_mapping)], dim=1)
			new_x = x + pos_and_segment_embedding

			x = self.encoder(new_x, src_key_padding_mask=key_padding_mask)
			return x

			# emb_0 = self.segment_embedding(torch.tensor(0).long().to(config.device)).unsqueeze(0)
			# emb_1 = self.segment_embedding(torch.tensor(1).long().to(config.device)).unsqueeze(0)
			# emb_2 = self.segment_embedding(torch.tensor(2).long().to(config.device)).unsqueeze(0)


			# i_encodings = self.pos_encoder(x[:, 0:20]) + emb_0.repeat(20, 1)
			# q_encodings = self.pos_encoder(x[:, 20:34]) + emb_1.repeat(14, 1)
			# a_encodings = self.pos_encoder(x[:, 34:]) + emb_2.repeat(8, 1)

			# encodings = torch.cat([i_encodings, q_encodings, a_encodings], dim=1)

			# print(encodings.shape)

		#encodings = 
		#torch.cat([self.pos_encoder(x[:, ini:fin]) + embs[i].repeat(fin-ini, 1) for i, (ini, fin) in enumerate(self.pair_idx)])

		#print(encodings.shape)
		
		# x = self.pos_encoder(x)

		# x = self.pos_encoder(x)
		


		x = self.encoder(x, key_padding_mask=src_mask)
		return x

class TransformerEncoderLayer(nn.Module):

	def __init__(self, h_dim, ff_dim, num_heads, dropout=0.1, activation="relu"):
		super(TransformerEncoderLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(h_dim, num_heads, dropout=dropout, batch_first=True)
		# Implementation of Feedforward model
		self.linear1 = nn.Linear(h_dim, ff_dim)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(ff_dim, h_dim)

		self.norm1 = nn.LayerNorm(h_dim)
		self.norm2 = nn.LayerNorm(h_dim)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = torch.nn.functional.relu

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = torch.nn.functional.relu
		super(TransformerEncoderLayer, self).__setstate__(state)

	def forward(self, src, src_mask = None, src_key_padding_mask = None):
		# src.shape == [batch_size*5, 21, 512]
		# src[0] == src[1] True for the 15 first elements (makes sense, just image and question, and different for answers)

		src2 = self.self_attn(src, src, src, attn_mask=src_mask,
							  key_padding_mask=src_key_padding_mask)[0]
		# TODO
		# src2[0][:15] == src2[1][:15] ALL TRUE. Attention is the same??? makes no sense


		src = self.norm1(src)
		src = src + self.dropout1(src2)
		src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
		src = self.norm2(src)
		src = src + self.dropout2(src2)
		return src 

class TransformerDecoder(nn.Module):
	'''
	Transformer decoder class
	input: encoder output [batch, seq, hid dim], text (targets), overlap and scene [batch, num objs, embed dim], is_train bool
	return: sequence of probability distribution over num_classes [batch, seq, num_classes]
	'''
	def __init__(self, h_dim, vocab_size, num_heads=8, num_layers=6, dropout=0.1):
		super(TransformerDecoder, self).__init__()
		self.decoder_layer = TransformerDecoderLayer(h_dim=h_dim, ff_dim=h_dim, num_heads=8, dropout=0.1)
		self.layer_norm = nn.LayerNorm(h_dim)
		self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers, norm=self.layer_norm)
		self.pos_encoder = PositionalEncoding(h_dim)

		self.hid_to_emb = nn.Linear(h_dim, vocab_size)

	def _generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask


	def forward(self, multimodal_features, answers, is_train):
		if is_train: # Training

			# convert targets from [batch, seq, feats] -> [seq, batch, feats] and apply embedding and position encoding
			targets = answers[:memory.shape[1],:]
			targets = targets.permute(1,0)
			targets = self.pos_encoder(targets)

			# generate target mask and pass to decoder
			target_mask = self._generate_square_subsequent_mask(answers.shape[1])
			output = self.decoder(tgt=targets, memory=multimodal_features, tgt_mask=target_mask, is_train=is_train)

			output = self.emb_to_classes(output)

		else: # Inference

			# Declare targets and output as zero tensors of output shape
			targets = torch.zeros(multimodal_features.shape[1], answers.shape[1]).to(multimodal_features.device)
			targets = targets.permute(1,0)

			output = torch.zeros(config.MAX_TEXT_LENGTH, memory.shape[1], self.num_classes).to(encoder_output.device)

			for t in range(config.MAX_TEXT_LENGTH):

				target_mask = self._generate_square_subsequent_mask(t+1).to(encoder_output.device)
				
				# convert targets into embeddings and apply positional encoding
				emb_targets = self.pos_encoder(self.emb(answers.long()))
				
				# pass embed targets and encoder memory to decoder
				t_output = self.decoder(tgt=emb_targets[:t+1], memory=multimodal_features, tgt_mask=target_mask, is_train=is_train)

				# map embeding dim to number of classes
				t_output = self.emb_to_classes(t_output)

				# take index class with max probability and append to targets and output sequence
				_, char_index = t_output[-1].max(1)
				targets[t+1,:] = char_index
				output[t,:] = t_output[t]

		output = output.permute(1,0,2)

		return output

class TransformerDecoderLayer(nn.Module):
	'''
	Pytorch implementation of transformer decoder layer
	'''
	def __init__(self, h_dim, ff_dim=2048, num_heads=8, dropout=0.1):
		super(TransformerDecoderLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(h_dim, num_heads, dropout=dropout)
		self.multihead_attn = nn.MultiheadAttention(h_dim, num_heads, dropout=dropout)

		self.linear1 = nn.Linear(h_dim, ff_dim)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(ff_dim, h_dim)

		self.norm1 = nn.LayerNorm(h_dim)
		self.norm2 = nn.LayerNorm(h_dim)
		self.norm3 = nn.LayerNorm(h_dim)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)
		self.dropout3 = nn.Dropout(dropout)

		self.activation = F.relu


	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(TransformerDecoderLayer, self).__setstate__(state)

	def forward(self, tgt, memory, semantics, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, is_train=False):
		
		tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
							  key_padding_mask=tgt_key_padding_mask)[0]
		tgt = tgt + self.dropout1(tgt2)
		tgt = self.norm1(tgt)

		tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask,
								   key_padding_mask=memory_key_padding_mask)[0]
		tgt = tgt + self.dropout2(tgt2)
		tgt = self.norm2(tgt)

		tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
		tgt = tgt + self.dropout3(tgt2)
		tgt = self.norm3(tgt)
		return tgt

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# class PositionalEncoding(nn.Module):

# 	def __init__(self, d_model, dropout=0.1, max_len=26):
# 		super(PositionalEncoding, self).__init__()
# 		self.dropout = nn.Dropout(p=dropout)

# 		pe = torch.zeros(max_len, d_model)
# 		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
# 		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
# 		pe[:, 0::2] = torch.sin(position * div_term)
# 		pe[:, 1::2] = torch.cos(position * div_term)
# 		pe = pe.unsqueeze(0).transpose(0, 1)
# 		self.register_buffer('pe', pe)

# 	def forward(self, x):
# 		x = x + self.pe[:x.size(0), :]
# 		return self.dropout(x)
