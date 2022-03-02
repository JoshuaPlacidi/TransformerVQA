from tkinter import X
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer

import config

class BertEncoder(nn.Module):
    def __init__(self, hid_dim):
        super(BertEncoder, self).__init__()
        self.config = DistilBertConfig()
        self.encoder = DistilBertModel(self.config)#.from_pretrained("distilbert-base-uncased")
        self.to_hid = nn.Linear(768, hid_dim)
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    def tokenize_text(self, x, is_answer_list=False):
        if is_answer_list: # Used to tokenize a list of lists of answer questions (multiple answers for each question)
            token_list = []
            mask_list = []
            for current_list in list(map(list, zip(*x))):
                tokens, masks = self.get_tokens(current_list)
                token_list.append(tokens)
                mask_list.append(masks)
            return torch.stack(token_list), torch.stack(mask_list)
        else: # used to tokenize a list of question string
            return self.get_tokens(x)

    def get_tokens(self, x):
        tokenizer_output = self.tokenizer(x, max_length=config.padded_language_length, padding="max_length", truncation=True, return_tensors="pt")
        tokens, masks = tokenizer_output['input_ids'], tokenizer_output['attention_mask']
        return tokens, masks

    def forward(self, tokens, masks=None):
        x = self.encoder(input_ids=tokens, attention_mask=masks)[0]
        x = self.to_hid(x)
        return x

def get_language_encoder(encoder_source="BERT"):
    return BertEncoder(256)
