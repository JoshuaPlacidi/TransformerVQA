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

		# train
		model.train()
		start_time = time.time()
		for batch in tqdm(train_dataset):
			cur_time = time.time()
			print(cur_time-start_time)
			start_time = time.time()
	
			frames, questions, answer_choices, ground_truth = batch
		
#			predictions = model(frames, questions, answer_choices)

#			loss = criterion(predictions, ground_truth)
			
#			model.zero_grad()
#			loss.backward()


	# val

def gif_preproc(model, dataset, save_folder):
	import pickle
	model.to(config.device)

	for batch in tqdm(dataset):
		filenames, gifs, masks = batch
		feature_tensor = model(gifs.to(config.device))
		
		for i in range(config.batch_size):
			filename_i = filenames[i]
			tensor_i = feature_tensor[i]
			mask_i = masks[i]

			feature_dict = {'tensor':tensor_i,'mask_i':mask_i}

			file_name = save_folder + '/' + filename_i + '.pkl' 
			
			with open(file_name, "wb") as handle:
				pickle.dump(feature_dict, handle)

