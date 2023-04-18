import os
import sys
import torch
import numpy as np
from torch.utils.data import Dataset
from model import CNNAutoencoder
from dataset import MachineTrainDataset, MachineTestLoader
from train import train
from test import test
from config import BATCH_SIZE, LEARNING_RATE, RAW_PATH, RESULT_PATH
from metrics import metrics_data, overall_score

#start with empty cache
torch.cuda.empty_cache()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu\n')
print(f'Using device: {device}\n\n')

for machine_name in os.listdir(RAW_PATH):
    if machine_name == '.gitkeep':
        continue

    print(f"Machine: {machine_name}\n")

    train_set = MachineTrainDataset(machine_name)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    test_loader = MachineTestLoader(machine_name)

    model = CNNAutoencoder()

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Total number of parameters: {params}\n') #print number of parameters

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\nTraining\n")
    train(model, optimizer, train_loader, machine_name)
    print("\nTesting\n")
    test(model, test_loader, machine_name)
    print("\n\n")
    print(f"overall_score:{overall_score(RESULT_PATH)}")
    df = metrics_data(RESULT_PATH)
    df.to_csv('results/metrics.csv')

