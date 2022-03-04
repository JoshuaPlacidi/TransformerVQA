import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageFile
import pandas as pd
import config
import os
import numpy as np
import pickle

class vqa_preproc(Dataset):
	def __init__(self, image_folder, qa_folder, annotation_file, mode="train"):
		self.annotations = pd.read_csv(annotation_file)
		self.image_folder_path = image_folder
		self.qa_folder_path = qa_folder
		self.mode = mode

	def __len__(self):
		return self.annotations.shape[0]

	def __getitem__(self, idx):
		sample = self.annotations.iloc[[idx]]
		image_name = sample["gif_name"].item()

		image_path = os.path.join(self.image_folder_path, image_name)+".pkl"
		qa_path = os.path.join(self.qa_folder_path, image_name)+".pkl"

		with open(image_path, "rb") as handle:
			image_data = pickle.load(handle)

		with open(qa_path, "rb") as handle:
			qa_data = pickle.load(handle)


		def index_to_tensor(index, total_length):
			return torch.cat([torch.ones(index), torch.zeros(total_length-index)])

		ret = []

		image_tensor = image_data["tensor"]
		image_mask = index_to_tensor(image_data["mask_i"], len(image_tensor))

		ret.append(image_tensor)
		ret.append(image_mask)

		for c in ["question", "a1", "a2", "a3", "a4", "a5"]:
			tensor = qa_data[c]
			mask = index_to_tensor(qa_data[f"{c}_mask_idx"], len(tensor))
			ret.append(tensor)
			ret.append(mask)
	
		return ret

class text_preproc(Dataset):
	def __init__(self, annotation_file, mode="train"):
		self.annotation_file = annotation_file
		self.dataset = pd.read_csv(annotation_file)

	def __len__(self):
		return len(self.dataset)

	def __getitem__(self, idx):
		sample = self.dataset.iloc[idx]
		return sample["gif_name"], sample["question"], [sample["a1"], sample["a2"], sample["a3"], sample["a4"], sample["a5"]]

class GIF_preproc(Dataset):
	def __init__(self, image_folder, mode="train"):
		self.image_folder_path = image_folder
		self.filenames = [i.split(".gif")[0] for i in os.listdir(self.image_folder_path) if i.endswith(".gif")]
		self.to_tensor = transforms.ToTensor()
		self.resize = transforms.Resize(config.image_size)
		self.mode = mode
		ImageFile.LOAD_TRUNCATED_IMAGES = True

	def __len__(self):
		return len(self.filenames)

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

		# Create the mask (not used)
		mask = torch.cat([torch.ones(len(image_list)), torch.zeros(needed_padding)])

		# We return the images, and the index where the padding starts
		return torch.stack(image_list), len(image_list)-needed_padding

	def __getitem__(self, idx):
		image_name = self.filenames[idx]
		
		image_frames, masking_idx = self.get_image_frames(os.path.join(self.image_folder_path, image_name)+".gif")

		return image_name, image_frames, masking_idx


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

		# t = time.process_time()
		image_frames = self.get_image_frames(os.path.join(self.image_folder_path, image_path)+".gif") # TODO: avoid calculating the gif frames here, instead should be done for every gif in init()
		# self.time += time.process_time() - t
		# print(self.time)

		return image_frames, question, answer_choices, ground_truth

def get_dataset(data_source="TGIF", image_folder=None, annotation_file=None):
	if not (image_folder and annotation_file) and data_source=="TGIF":
		raise Exception("Both image_folder and annotation_file location are required, 1 or both not passed")

	modes = ["train", "val", "test"]

	if data_source=="TGIF":
		dataset_class = TGIF_dataset
	elif data_source=="GIF_preproc":
		return DataLoader(
			GIF_preproc(image_folder),
			batch_size=config.batch_size,
			shuffle=True,
			num_workers=0)
	
	elif data_source=="text_preproc":
		return DataLoader(
			text_preproc(annotation_file),
			batch_size=config.batch_size,
			shuffle=True,
			num_workers=0)

	else:
		raise Exception("data source not recognised:", data_source)

	return [DataLoader(
		dataset_class(image_folder, annotation_file, mode),
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=0) 
		for mode in modes]

