import torch

import config

from training import train
from model import get_PTVQA
from datasets import get_dataset

train_set, val_set, test_set = get_dataset(
    data_source = "GIF_preproc", # TGIF
    image_folder = config.tgif_folder_location + "gifs/",
    annotation_file = config.tgif_folder_location + "train_action_question.csv"
    )

model = get_PTVQA()

train(model, train_set)


#print(model)
