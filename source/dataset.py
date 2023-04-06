import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
from config import SAMPLE_RATE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MachineDataset(Dataset):

    def __init__(self, segmented_audio_dir, spectrogram_dir) -> None:
        self.segmented_audio_dir = segmented_audio_dir
        self.spectrogram_dir = spectrogram_dir
        count = 0
        # Iterate directory to find total number of samples in training data
        for path in os.listdir(self.segmented_audio_dir):
            #ignore gitkeep file
            if path == '.gitkeep':
                continue
            if os.path.isfile(os.path.join(self.segmented_audio_dir, path)):
                count += 1
        self.length = count


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        
        #load spectrogram from .npy numpy file
        #ignore .gitkeep
        if os.listdir(self.spectrogram_dir)[index] == '.gitkeep':
            index += 1
        spectrogram_file = os.listdir(self.spectrogram_dir)[index]
        spectrogram = torch.from_numpy(np.load(os.path.join(self.spectrogram_dir, spectrogram_file)))
        spectrogram = spectrogram[0:1,:, :] #get single channel spectrogram slicing [0:1] to preserve dimensions
        
        #load audio file waveform
        audio_file = os.listdir(self.segmented_audio_dir)[index]
        waveform, sample_rate = torchaudio.load(os.path.join(self.segmented_audio_dir, audio_file))

        #resample waveform if sample rate is higher than SAMPLE_RATE from config.py
        if sample_rate > SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
        waveform = waveform[0:1,:] #get single channel waveform from waveform with two channels; slicing [0:1] to preserve dimensions

        return spectrogram, waveform