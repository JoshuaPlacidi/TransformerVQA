import torch
from tqdm import tqdm
import config
import time

# TODO Finish implementing training loop
def train_vqa(model, train_dataset, val_dataset=None, num_epochs=10):
	import torch.optim as optim
	optimizer = optim.AdamW(model.parameters(), lr=0.0001)
	criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

	
	for epoch in range(num_epochs):
		
		model.train()
		pbar = tqdm(train_dataset)
		pbar.set_description("Epoch %s" % epoch)
		for batch in pbar:
			frames, questions, answer_choices, ground_truth = batch
		
			predictions = model(frames, questions, answer_choices)

			loss = criterion(predictions, ground_truth)
			
			model.zero_grad()
			loss.backward()


def gif_preproc(model, dataset, save_folder):
	import pickle
	model.to(config.device)

	for batch in tqdm(dataset):
		filenames, gifs, masks = batch
		print(gifs.shape)
		break

		# Pass images through model
		with torch.no_grad():
			feature_tensor = model(gifs.to(config.device)).cpu()
		
		# For sample in batch
		for i in range(config.batch_size):
			filename_i = filenames[i]
			tensor_i = feature_tensor[i]
			mask_i = masks[i]

			# Store tensor and mask
			feature_dict = {'tensor':tensor_i,'mask':mask_i}

			# Save feature dict to pickle
			file_name = save_folder + '/' + filename_i + '.pkl' 
			
			#with open(file_name, "wb") as handle:
			#	pickle.dump(feature_dict, handle)

def text_preproc(lan_model, dataset, save_folder, q_max_length=14, a_max_length=8):
	import pickle
	lan_model.to(config.device)

	for batch in tqdm(dataset):
		filenames, questions, answers = batch
		
		# Create question and answer tokens
		q_tokens, q_masks = lan_model.tokenize_text(questions, max_length=q_max_length, is_multi_list=False)
		a_tokens, a_masks = lan_model.tokenize_text(answers, max_length=a_max_length, is_multi_list=True, transpose_list=True)
		
		# Pass models through language encoder
		with torch.no_grad():
			q_tensors = lan_model(q_tokens.to(config.device), q_masks.to(config.device)).cpu()

			# Stack answers: [batch, num_answers, num_tokens] -> [batch x num_answers, num_tokens]
			a_stacked_tokens = torch.reshape(a_tokens, shape=(config.batch_size*5, -1))
			a_stacked_masks = torch.reshape(a_masks, shape=(config.batch_size*config.padded_language_length, -1))
			a_stacked_tensors = lan_model(a_stacked_tokens.to(config.device), a_stacked_masks.to(config.device)).cpu()

			# Unstack answers: [batch x num_answers, num_tokens] -> [batch, num_answers, num_tokens]
			a_tensors = torch.reshape(a_stacked_tensors, shape=(config.batch_size, 5, a_max_length, -1))

		# Loop through each sample in batch and save it to pickle
		for i in range(config.batch_size):
			filename_i = filenames[i]
			q_tensor_i = q_tensors[i]
			a_tensor_i = a_tensors[i]

			# Find mask idx for questions
			q_mask_idx = torch.argmin(q_masks[i]).item()
			if q_mask_idx == 0: q_mask_idx = q_max_length

			# Store question tensor and mask idx
			feature_dict = {'question':q_tensor_i, 'question_mask_idx':q_mask_idx}

			for j in range(1,6):
				cur_a_tensor = a_tensor_i[j]
				
				# Find mask idx for answers
				cur_a_mask_idx = torch.argmin(a_masks[i,j]).item()
				if cur_a_mask_idx == 0: cur_a_mask_idx = a_max_length

				# Store answer tensor and mask idx
				feature_dict[f'a{j}'] = cur_a_tensor
				feature_dict[f'a{j}_mask_idx'] = cur_a_mask_idx

			# Save to sample to pickle object
			file_name = save_folder + '/' + filename_i + '.pkl'

			with open(file_name, "wb") as handle:
				pickle.dump(feature_dict, handle)

		# Just process the first batch for testing purposes, remove later
		break
