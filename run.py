import torch

import config

from training import train_vqa, gif_preproc, text_preproc
from model import get_PTVQA, get_ImageFeatureExtractor, get_language_encoder
from datasets import get_dataset

#
# Train VQA
#
# train_set, val_set, test_set = get_dataset(
#    data_source = "TGIF",
#    image_folder = config.tgif_folder_location + "gifs/",
#    annotation_file = config.tgif_folder_location + "train_action_question.csv"
#    )

# model = get_PTVQA()

# train_vqa(model, train_set)

#
# Gif preproc
#
gif_dataset = get_dataset(
	data_source = "GIF_preproc",
	image_folder = config.tgif_folder_location + "gifs/",
	annotation_file = None
	)
model = get_ImageFeatureExtractor()
gif_preproc(model=model, dataset=gif_dataset, save_folder="F:/Dev/datasets/tgif_image_features")


#
# Text preproc
#
# text_dataset = get_dataset(
# 	data_source = "text_preproc",
# 	annotation_file = config.tgif_folder_location + "action_question_annotations.csv"
# 	)
# model = get_language_encoder()
# text_preproc(lan_model=model, dataset=text_dataset, save_folder="F:/Dev/datasets/tgif_text_features")