import torch
import torch.nn as nn

from modules.transformer_modules import TransformerEncoder

#def get_video_encoder(model="transformer", h_dim, ff_dim, num_heads, num_layers, dropout=0.1):
#	if model=="transformer":
#		return get_transformer_video_encoder()
#	else:
#		raise Exception("Video encoder model not recognized:", model)

def get_transformer_video_encoder(h_dim, ff_dim, num_heads=8, num_layers=6, dropout=0.1):
	return TransformerEncoder(h_dim, ff_dim, num_heads=8, num_layers=6, dropout=0.1)
