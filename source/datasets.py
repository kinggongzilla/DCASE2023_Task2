import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import re
from config import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_mask(spectrogram, patch_size=PATCH_SIZE, num_patches_to_zero=NUM_PATCHES_TO_ZERO):

    _, height, width = spectrogram.shape

    # Ensure patch size divides the image without remainder
    if height % patch_size != 0:
        raise ValueError(f"Patch size {patch_size} does not divide height {height} evenly.")
    if width % patch_size != 0:
        raise ValueError(f"Patch size {patch_size} does not divide width {width} evenly.")

    # Divide the image into patches of the specified size
    patches = []
    for i in range(0, height, patch_size):
        for j in range(0, width, patch_size):
            patches.append((i, j))

    # Randomly choose patches to zero out
    np.random.shuffle(patches)
    selected_patches = patches[:num_patches_to_zero]

    # Set the values in the selected patches to zero
    mask = np.ones(spectrogram.shape)
    for (i, j) in selected_patches:
        mask[0, i:i+patch_size, j:j+patch_size] = 0

    return mask


def get_spectrogram(spectrogram_file_path):
    spectrogram = torch.from_numpy(np.load(spectrogram_file_path))
    return spectrogram

class MachineTrainDataset(Dataset):

    def __init__(self, machine_name) -> None:
        self.spectrograms_folder_path = os.path.join(SPECTROGRAMS_PATH, machine_name, "train")
        self.spectrograms_file_names = sorted(os.listdir(self.spectrograms_folder_path))
        self.length = len(self.spectrograms_file_names)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        spectrogram_file_name = self.spectrograms_file_names[index]
        spectrogram_file_path = os.path.join(self.spectrograms_folder_path, spectrogram_file_name)
        spectrogram = get_spectrogram(spectrogram_file_path)
        mask = random_mask(spectrogram)
        return mask, spectrogram


class MachineTestLoader:
    """
    Creates a batch for each original audio file
    """
    def __init__(self, machine_name):
        self.index = 0
        self.machine_name = machine_name
        self.spectrograms_folder_path = os.path.join(SPECTROGRAMS_PATH, machine_name, "test")
        self.spectrograms_file_names = sorted(os.listdir(self.spectrograms_folder_path))
        self.length = len(self.spectrograms_file_names)

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):

        if self.index > self.__len__()-1:
            raise StopIteration

        spectrogram_file_name = self.spectrograms_file_names[self.index]
        spectrogram_file_path = os.path.join(self.spectrograms_folder_path, spectrogram_file_name)

        spectrogram = get_spectrogram(spectrogram_file_path)
        masks = [random_mask(spectrogram) for _ in range(BATCH_SIZE)]

        #extract label from file name: index=0__segment_id=0__domain=source__label=anomaly.spec.npy
        if 'anomaly' in spectrogram_file_name:
            label = 1
        else:
            label = 0

        # Convert the list of masked_spectrograms to a torch tensor stacked along 0-dim
        masks = torch.stack([torch.tensor(mask) for mask in masks])

        # Duplicate "BATCH_SIZE" times the same spectrogram along 0-dim (using torch)
        spectrograms = torch.stack([torch.tensor(spectrogram) for _ in range(BATCH_SIZE)])

        #duplicate lables "bATCHES_SIZE" times as torch tensor
        labels = torch.stack([torch.tensor(label) for _ in range(BATCH_SIZE)])

        return masks, spectrograms, labels
