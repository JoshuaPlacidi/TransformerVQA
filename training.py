import torch
from tqdm import tqdm

def train(model, dataset):
    for sample in tqdm(dataset):
        frames, questions, answer_choices, ground_truth = sample
        
        predictions = model(frames, questions, answer_choices)