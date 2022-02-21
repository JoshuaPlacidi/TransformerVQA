import torch
import torch.nn as nn
import math

class TransformerEncoder(nn.Module):
	'''
	Transformer encoder
	'''
	def __init__(self, h_dim, ff_dim, num_heads, num_layers, dropout):
		'''
		h_dim = hidden dimension size
		ff_dim = feedforard dimension size
		num_heads = number of attention heads
		num_layers = number of encoder layers
		dropout = dropout rate
		'''
		super(TransformerEncoder, self).__init__()
		self.pos_encoder = PositionalEncoding(h_dim)
		self.encoder_layer = TransformerEncoderLayer(d_model=h_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout)
		self.layer_norm = nn.LayerNorm(h_dim)
		self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers, norm=self.layer_norm)

	def forward(self, x):
		x = self.pos_encoder(x)
		x = self.encoder(x)
		return output

class TransformerEncoderLayer(nn.Module):

	def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
		super(TransformerEncoderLayer, self).__init__()
		self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
		# Implementation of Feedforward model
		self.linear1 = nn.Linear(d_model, dim_feedforward)
		self.dropout = nn.Dropout(dropout)
		self.linear2 = nn.Linear(dim_feedforward, d_model)

		self.norm1 = nn.LayerNorm(d_model)
		self.norm2 = nn.LayerNorm(d_model)
		self.dropout1 = nn.Dropout(dropout)
		self.dropout2 = nn.Dropout(dropout)

		self.activation = torch.nn.functional.relu

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = torch.nn.functional.relu
		super(TransformerEncoderLayer, self).__setstate__(state)

	def forward(self, src, src_mask = None, src_key_padding_mask = None):
		src2 = self.self_attn(src, src, src, attn_mask=src_mask,
							  key_padding_mask=src_key_padding_mask)[0]
		src = self.norm1(src)
		src = src + self.dropout1(src2)
		src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
		src = self.norm2(src)
		src = src + self.dropout2(src2)
		return src 

class PositionalEncoding(nn.Module):

	def __init__(self, d_model, dropout=0.1, max_len=26):
		super(PositionalEncoding, self).__init__()
		self.dropout = nn.Dropout(p=dropout)

		pe = torch.zeros(max_len, d_model)
		position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
		div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
		pe[:, 0::2] = torch.sin(position * div_term)
		pe[:, 1::2] = torch.cos(position * div_term)
		pe = pe.unsqueeze(0).transpose(0, 1)
		self.register_buffer('pe', pe)

	def forward(self, x):
		x = x + self.pe[:x.size(0), :]
		return self.dropout(x)

