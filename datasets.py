import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
import pandas as pd
import config

class TGIF_dataset(Dataset):
	def __init__(self, image_folder, annotation_file, mode="train"):
		self.annotations = pd.read_csv(annotation_file, sep='\t', header=0)
		self.image_folder_path = image_folder
		self.to_tensor = transforms.ToTensor()
		self.resize = transforms.Resize(config.image_size)

	def __len__(self):
		return self.annotations.shape[0]

	def get_image_frames(self, path):
		gif = Image.open(path)
		image_list = []
		for f in range(0, gif.n_frames, 4):
			gif.seek(f)
			frame = self.resize(gif)
			image_tensor = self.to_tensor(frame).squeeze()
			image_list.append(image_tensor)
		return torch.stack(image_list)

	def __getitem__(self, idx):
		sample = self.annotations.iloc[[idx]]
		image_path = self.image_folder_path + 'test.gif' #self.image_folder_path + sample['gif_name'].iloc[0] + '.gif'
		image_frames = self.get_image_frames(image_path)
		question = sample['question'].iloc[0]
		answer_choices = [sample['a' + str(answer_key)].iloc[0] for answer_key in range(1,5)]
		ground_truth = sample['answer'].iloc[0]
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

