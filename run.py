import torch

import config

from training import train_vqa, gif_preproc
from model import get_PTVQA, get_ImageFeatureExtractor
from datasets import get_dataset

#train_set, val_set, test_set = get_dataset(
#    data_source = "TGIF",
#    image_folder = config.tgif_folder_location + "gifs/",
#    annotation_file = config.tgif_folder_location + "train_action_question.csv"
#    )

gif_dataset = get_dataset(
	data_source = "GIF_preproc",
	image_folder = config.tgif_folder_location + "gifs/",
	annotation_file = None
	)

#model = get_PTVQA()

#train(model, train_set)

model = get_ImageFeatureExtractor()

gif_preproc(model=model, dataset=gif_dataset, save_folder="/Users/joshua/Documents/edinburgh/mlp/cw4/data")
