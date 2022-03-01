import torch
from torch.utils.data import Dataset

class TGIF_dataset():
	def __init__(self, image_folder, annotation_file, mode="train"):
		self.data = [1]

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		return self.data[idx]

def get_dataset(data_source="TGIF", image_folder=None, annotation_file=None):
	if not (image_folder and annotation_file):
		raise Exception("Both image_folder and annotation_file location are required, 1 or both not passed")

	modes = ["train", "val", "test"]

	if data_source=="TGIF":
		return [TGIF_dataset(image_folder, annotation_file, mode) for mode in modes]
	else:
		raise Exception("data source not recognised:", data_source)		
