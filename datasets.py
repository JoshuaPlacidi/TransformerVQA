import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

from PIL import Image
from PIL import ImageFile
from modules.language_encoder import get_language_encoder, BertTokenizer
import pandas as pd
import config
import os
import numpy as np
import pickle

class IQA_Dataset(Dataset):
	def __init__(self, dataset_folder, annotation_file, mode="train"):
		self.annotations = pd.read_csv(annotation_file)
		self.image_folder_path = dataset_folder + 'train2014/'
		self.mode = mode
		self.resize = transforms.Resize(config.image_size)
		self.norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
		self.to_tensor = transforms.ToTensor()
		self.tokenize = get_language_encoder().tokenize_text


	def __len__(self):
		return self.annotations.shape[0]

	def __getitem__(self, idx):
		sample = self.annotations.iloc[[idx]]
		image_name = str(sample["image_id"].item())
		image_full_name = "COCO_train2014_"+"0"*(12-len(image_name)) + image_name + ".jpg"
		
		image_path = os.path.join(self.image_folder_path, image_full_name)

		image = Image.open(image_path).convert('RGB')

		image_tensor = self.norm(self.to_tensor(self.resize(image))) # [3, config.image_size, config.image_size]
		
		# question_tokens [1, config.padded_language_length_question]
		# question_mask [1, config.padded_language_length_question]
		question_tokens, question_mask = self.tokenize(sample['question'].item(), max_length=config.padded_language_length_question)

		answer_list = []
		answer_tokens_list = []
		answer_masks_list = []

		for i in range(5):
			current_answer = sample[f"a_{i}"].item()
			answer_list.append(current_answer)
			answer_tokens, answer_mask = self.tokenize(current_answer, max_length=config.padded_language_length_answer)
			answer_tokens_list.append(answer_tokens)
			answer_masks_list.append(answer_mask)

		answer_tokens = torch.stack(answer_tokens_list)
		answer_masks = torch.stack(answer_masks_list)

		return image_tensor, question_tokens, question_mask, answer_tokens, answer_masks, answer_list.index(sample["ground_truth"].item())

class TGIF_Dataset(Dataset):
	def __init__(self, dataset_folder, annotation_file, mode="train"):
		self.annotations = pd.read_csv(annotation_file)
		self.image_folder_path = dataset_folder + 'tgif_image_features/'
		self.qa_folder_path = dataset_folder + 'tgif_text_features/'
		self.mode = mode

	def __len__(self):
		return self.annotations.shape[0]

	def index_to_tensor(self, index, total_length):
		return torch.cat([torch.ones(index), torch.zeros(total_length-index)])

	def __getitem__(self, idx):
		sample = self.annotations.iloc[[idx]]
		image_name = sample["gif_name"].item()

		image_path = os.path.join(self.image_folder_path, image_name)+".pkl"
		qa_path = os.path.join(self.qa_folder_path, image_name)+".pkl"

		with open(image_path, "rb") as handle:
			image_data = pickle.load(handle)

		with open(qa_path, "rb") as handle:
			qa_data = pickle.load(handle)


		ret = []

		image_tensor = image_data["tensor"]
		image_mask = self.index_to_tensor(image_data["mask"], len(image_tensor))

		ret.append(image_tensor)
		ret.append(image_mask)


		question_tensor = qa_data['question']
		question_mask = self.index_to_tensor(qa_data['question_mask_idx'], question_tensor.shape[0])

		ret.append(question_tensor)
		ret.append(question_mask)

		answers = []
		answer_masks = []
		for c in ["a1", "a2", "a3", "a4", "a5"]:
			tensor = qa_data[c]
			a_mask = self.index_to_tensor(qa_data[f"{c}_mask_idx"], len(tensor))
			answers.append(tensor)
			answer_masks.append(a_mask)

		ret.append(torch.stack(answers))
		ret.append(torch.stack(answer_masks))

		ret.append(sample['answer'].item()) # ground truth

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


def get_dataset(data_source="TGIF", dataset_folder=None, annotation_file=None):
	if not (dataset_folder and annotation_file) and data_source=="TGIF":
		raise Exception("Both image_folder and annotation_file location are required, 1 or both not passed")

	modes = ["train", "val", "test"]

	if data_source=="TGIF":
		dataset_class = TGIF_Dataset

	elif data_source=="IQA":
		return DataLoader(
			IQA_Dataset(dataset_folder, annotation_file),
			batch_size=config.batch_size,
			shuffle=False,
			num_workers=0)


	elif data_source=="GIF_preproc":
		return DataLoader(
			GIF_preproc(dataset_folder),
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
		dataset_class(dataset_folder, annotation_file, mode),
		batch_size=config.batch_size,
		shuffle=True,
		num_workers=0) 
		for mode in modes]




#
# TGIF dataset that calculates encodings for each image and text at run time
#

# class old_TGIF_dataset(Dataset):
# 	def __init__(self, image_folder, annotation_file, mode="train"):
# 		self.annotations = pd.read_csv(annotation_file, sep='\t', header=0)
# 		self.image_folder_path = image_folder
# 		self.to_tensor = transforms.ToTensor()
# 		self.resize = transforms.Resize(config.image_size)
# 		self.mode = mode

# 	def __len__(self):
# 		return self.annotations.shape[0]

# 	def get_image_frames(self, path):
# 		gif = Image.open(path)
# 		# We can access the number of frames using gif.n_frames 
# 		image_list = []

# 		# Calculation to account for different numbers of frames and how frames should be 'skipped'
# 		step = 1 if gif.n_frames < config.padded_frame_length else gif.n_frames/config.padded_frame_length		
# 		for f in np.arange(0, gif.n_frames, step):
# 			gif.seek(int(f))
# 			frame = self.resize(gif).convert('RGB')
# 			image_tensor = self.to_tensor(frame).squeeze()
# 			image_list.append(image_tensor)

# 		# Check if we need to pad
# 		needed_padding = config.padded_frame_length - len(image_list)

# 		# Add empty images if needed
# 		image_list.extend([torch.zeros_like(image_tensor) for _ in range(needed_padding)])

# 		# Create the mask
# 		mask = torch.cat([torch.ones(len(image_list)), torch.zeros(needed_padding)])

# 		return torch.stack(image_list)

	# def __getitem__(self, idx):
	# 	sample = self.annotations.iloc[[idx]]
	# 	image_path = sample["gif_name"].item()
	# 	question = sample['question'].item()
	# 	ground_truth = sample['answer'].item()
	# 	answer_choices = [sample[f'a{i}'].item() for i in range(1,6)]

	# 	# image_path = self.image_folder_path + 'test.gif' # TODO: self.image_folder_path + sample['gif_name'].iloc[0] + '.gif'

	# 	# t = time.process_time()
	# 	image_frames = self.get_image_frames(os.path.join(self.image_folder_path, image_path)+".gif") # TODO: avoid calculating the gif frames here, instead should be done for every gif in init()
	# 	# self.time += time.process_time() - t
	# 	# print(self.time)

	# 	return image_frames, question, answer_choices, ground_truth