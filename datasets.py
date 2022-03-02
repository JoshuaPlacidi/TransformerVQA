import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import pandas as pd
import config
import os
import numpy as np

class TGIF_dataset(Dataset):
	def __init__(self, image_folder, annotation_file, mode="train"):
		self.annotations = pd.read_csv(annotation_file, sep='\t', header=0)
		self.image_folder_path = image_folder
		self.to_tensor = transforms.ToTensor()
		self.resize = transforms.Resize(config.image_size)
		self.mode = mode

	def __len__(self):
		return self.annotations.shape[0]

	def get_image_frames(self, path):
		gif = Image.open(path)
		# We can access the number of frames using gif.n_frames 
		image_list = []

		# Calculation to account for different numbers of frames and how frames should be 'skipped'
		step = 1 if gif.n_frames < config.padded_frame_length else gif.n_frames/config.padded_frame_length		
		for f in np.arange(0, gif.n_frames, step):
			gif.seek(int(f))
			frame = self.resize(gif).convert('RGB')
			image_tensor = self.to_tensor(frame).squeeze()
			image_list.append(image_tensor)

		# Check if we need to pad
		needed_padding = config.padded_frame_length - len(image_list)

		# Add empty images if needed
		image_list.extend([torch.zeros_like(image_tensor) for _ in range(needed_padding)])

		# Create the mask
		mask = torch.cat([torch.ones(len(image_list)), torch.zeros(needed_padding)])

		return torch.stack(image_list)

	def __getitem__(self, idx):
		sample = self.annotations.iloc[[idx]]
		image_path = sample["gif_name"].item()
		question = sample['question'].item()
		ground_truth = sample['answer'].item()
		answer_choices = [sample[f'a{i}'].item() for i in range(1,6)]

		# image_path = self.image_folder_path + 'test.gif' # TODO: self.image_folder_path + sample['gif_name'].iloc[0] + '.gif'
		image_frames = self.get_image_frames(os.path.join(self.image_folder_path, image_path)+".gif") # TODO: avoid calculating the gif frames here, instead should be done for every gif in init()

		return image_frames, question, answer_choices, ground_truth

def get_dataset(data_source="TGIF", image_folder=None, annotation_file=None):
	if not (image_folder and annotation_file):
		raise Exception("Both image_folder and annotation_file location are required, 1 or both not passed")

	modes = ["train", "val", "test"]

	if data_source=="TGIF":
		dataset_class = TGIF_dataset
	else:
		raise Exception("data source not recognised:", data_source)

	return [DataLoader(
		dataset_class(image_folder, annotation_file, mode),
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=0) 
		for mode in modes]

