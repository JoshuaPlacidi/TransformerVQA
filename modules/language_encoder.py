from tkinter import X
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

import config

class BertEncoder(nn.Module):
    def __init__(self, h_dim):
        super(BertEncoder, self).__init__()
        self.config = DistilBertConfig()
        self.encoder = DistilBertModel(self.config)#.from_pretrained("distilbert-base-uncased")
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        if h_dim:
            self.map_to_hid = True
            self.to_hid = nn.Linear(768, h_dim)
        else:
            self.map_to_hid = False

    def tokenize_text(self, x, max_length, is_multi_list=False, transpose_list=False):
        if is_multi_list: # Used to tokenize a list of lists of answer questions (multiple answers for each question)
            token_list = []
            mask_list = []
            
            if transpose_list:
                x = list(map(list, zip(*x)))

            for current_list in x:
                tokens, masks = self.get_tokens(current_list, max_length)
                token_list.append(tokens)
                mask_list.append(masks)
            return torch.stack(token_list), torch.stack(mask_list)
        else: # used to tokenize a list of question string
            return self.get_tokens(x, max_length)

    def get_tokens(self, x, max_length):
        tokenizer_output = self.tokenizer(x, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
        tokens, masks = tokenizer_output['input_ids'], tokenizer_output['attention_mask']
        return tokens, masks

    def forward(self, tokens, masks=None):
        x = self.encoder(input_ids=tokens, attention_mask=masks)[0]
        
        if self.map_to_hid:
            x = self.to_hid(x)

        return x

def get_language_encoder(encoder_source='BERT', h_dim=None):
    return BertEncoder(h_dim)
