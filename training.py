import torch
from tqdm import tqdm

def train(model, train_dataset, val_dataset=None, num_epochs=10):
	import torch.optim as optim
	optimizer = optim.AdamW(model.parameters(), lr=0.0001)
	criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

	for epoch in range(num_epochs):

		# train
		model.train()
		for sample in tqdm(train_dataset):
			frames, questions, answer_choices, ground_truth = sample
		
			predictions = model(frames, questions, answer_choices)

			loss = criterion(predictions, ground_truth)
			
			model.zero_grad()
			loss.backward()

			print(loss)

	# val
