import torch

import config

#from training import train, evaluate
from model import get_PTVQA
#from dataset import get_tgif_dataset

model = get_PTVQA()
print(model)
