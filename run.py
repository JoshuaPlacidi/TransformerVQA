import torch

import config

#from training import train, evaluate
from model import get_PTVQA
from datasets import get_dataset

train_dataset, val_dataset, test_dataset = get_dataset("TGIF", "/data/tgif/images", "data/tgif/annotation_file.txt")

model = get_PTVQA()
print(model)
print('number of params:', sum(p.numel() for p in model.parameters() if p.requires_grad))
