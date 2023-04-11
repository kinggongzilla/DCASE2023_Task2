import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import re
from config import SAMPLE_RATE, RAW_PATH, SPECTROGRAMS_PATH, AUDIO_SEGMENTS_PATH, IS_NORMAL, IS_ANOMALY, IS_SOURCE, IS_TARGET

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_filename(filename):
    """
    Parse a filename of the format 'index=0__segment_id=0__domain=source__label=anomaly.wav.spec'
    and return a tuple containing the values of 'index', 'segment_id', 'domain', and 'label'.
    """
    # Split the filename into its components using the '__' separator
    components = filename.split('__')

    # Extract the values of 'index', 'segment_id', 'domain', and 'label' from the components
    index = int(components[0].split('=')[1])
    segment_id = int(components[1].split('=')[1])
    domain = components[2].split('=')[1]
    label = components[3].split('=')[1].split('.')[0]

    # Return the values as a tuple
    return index, segment_id, domain, label


def get_data(audio_segment_file_path, spectrogram_file_path):

    audio_segment_file_name = os.path.basename(audio_segment_file_path)
    spectrogram_file_name = os.path.basename(spectrogram_file_path)

    index, segment_id, domain, label = parse_filename(spectrogram_file_name)

    if (index, segment_id, domain, label) != parse_filename(audio_segment_file_name):
        raise AttributeError("The dataset is inconsistent!")

    waveform, sample_rate = torchaudio.load(audio_segment_file_path)
    spectrogram = torch.from_numpy(np.load(spectrogram_file_path))

    spectrogram = spectrogram[0:1, :, :] #get single channel spectrogram slicing [0:1] to preserve dimensions
    #resample waveform if sample rate is higher than SAMPLE_RATE from config.py
    if sample_rate > SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=SAMPLE_RATE)
    waveform = waveform[0:1, :] #get single channel waveform from waveform with two channels; slicing [0:1] to preserve dimensions

    domain = IS_SOURCE if "source" in audio_segment_file_name else IS_TARGET
    label = IS_NORMAL if "normal" in audio_segment_file_name else IS_ANOMALY

    return spectrogram, waveform, index, domain, label


class MachineTrainDataset(Dataset):

    def __init__(self, machine_name) -> None:

        self.spectrograms_folder_path = os.path.join(SPECTROGRAMS_PATH, machine_name, "train")
        self.audio_segments_folder_path = os.path.join(AUDIO_SEGMENTS_PATH, machine_name, "train")
        self.spectrograms_file_names = sorted(os.listdir(self.spectrograms_folder_path))
        self.audio_segments_file_names = sorted(os.listdir(self.audio_segments_folder_path))
        self.length = len(self.spectrograms_file_names)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        audio_segment_file_name = self.audio_segments_file_names[index]
        spectrogram_file_name = self.spectrograms_file_names[index]
        audio_segment_file_path = os.path.join(self.audio_segments_folder_path, audio_segment_file_name)
        spectrogram_file_path = os.path.join(self.spectrograms_folder_path, spectrogram_file_name)
        return get_data(audio_segment_file_path, spectrogram_file_path)


class MachineTestLoader:
    """
    Creates a batch for each original audio file
    """
    def __init__(self, machine_name):
        self.index = 0
        self.machine_name = machine_name
        self.length = len(os.listdir(os.path.join(RAW_PATH, self.machine_name, "test")))
        self.spectrograms_folder_path = os.path.join(SPECTROGRAMS_PATH, machine_name, "test")
        self.audio_segments_folder_path = os.path.join(AUDIO_SEGMENTS_PATH, machine_name, "test")
        self.spectrograms_file_names = sorted(os.listdir(self.spectrograms_folder_path))
        self.audio_segments_file_names = sorted(os.listdir(self.audio_segments_folder_path))

    @property
    def index_file_names(self):
        """
        Get all filenames in the 'in_folder_path' that have the specified 'index'.
        """
        # Get a list of all filenames in the folder
        index_spectrograms_file_names = sorted([filename for filename in self.spectrograms_file_names if f"index={self.index}__" in filename])
        index_audio_segments_file_names = sorted([filename for filename in self.audio_segments_file_names if f"index={self.index}__" in filename])

        # Return the filtered filenames
        return index_spectrograms_file_names, index_audio_segments_file_names

    def __len__(self):
        return self.length

    def __iter__(self):
        return self

    def __next__(self):

        index_spectrograms_file_names, index_audio_segments_file_names = self.index_file_names

        if not index_spectrograms_file_names:
            raise StopIteration

        data = dict(
            spectrograms=[],
            waveforms=[],
            indices=[],
            domains=[],
            labels=[]
        )

        for spectrogram_file_name, audio_segment_file_name in zip(index_spectrograms_file_names, index_audio_segments_file_names):
            audio_segment_file_path = os.path.join(self.audio_segments_folder_path, audio_segment_file_name)
            spectrogram_file_path = os.path.join(self.spectrograms_folder_path, spectrogram_file_name)
            spectrogram, waveform, index, domain, label = get_data(audio_segment_file_path, spectrogram_file_path)
            data["spectrograms"].append(spectrogram)
            data["waveforms"].append(waveform)
            data["indices"].append(index)
            data["domains"].append(domain)
            data["labels"].append(label)

        # Convert the lists of numerical input features to tensors
        data["spectrograms"] = torch.unsqueeze(torch.cat(data["spectrograms"], dim=0), 1)
        data["waveforms"] = torch.unsqueeze(torch.cat(data["waveforms"], dim=0), 1)

        self.index += 1

        return data["spectrograms"], data["waveforms"], data["indices"], data["domains"], data["labels"]


