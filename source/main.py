import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from model import CNNAutoencoder
from dataset import MachineDataset
from train import train
from config import EPOCHS, BATCH_SIZE, LEARNING_RATE

#start with empty cache
torch.cuda.empty_cache()

#default data location
spectrograms_path = os.path.join('data', 'processed', 'spectrograms')
waveform_path = os.path.join('data', 'processed', 'audio_segments')

#example: python source/main.py path/to/data
if len(sys.argv) > 1:
    spectrograms_path = sys.argv[1]

#initialize dataset
data_loader = MachineDataset(spectrogram_dir=spectrograms_path, segmented_audio_dir=waveform_path)

#initialize dataloader
trainloader = torch.utils.data.DataLoader(
    data_loader,
    batch_size=BATCH_SIZE,
    shuffle=True,
    )

#initialize model
model = CNNAutoencoder()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

#train model
train(model, optimizer,trainloader, EPOCHS, LEARNING_RATE)
