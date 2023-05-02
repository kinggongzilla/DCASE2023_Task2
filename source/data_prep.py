import os
import sys
from pydub import AudioSegment
from config import *
import torchaudio
import torchaudio.transforms as T
from torch.nn.functional import interpolate
import numpy as np
import torch
from tqdm import tqdm

#Note: ffmpeg package required for pydub

def transform_to_spectrogram(in_file_path, out_dir):
    if not os.path.exists(out_dir):
        raise f'out_dir {out_dir} does not exist'
    if not os.path.isfile(in_file_path):
        raise f'given in_path {in_file_path} is not a file'

    waveform, sample_rate = torchaudio.load(in_file_path)

    target_sample_rate = SAMPLE_RATE
    if sample_rate != target_sample_rate:
        resampler = T.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)



    # Create a Spectrogram object
    n_fft = N_FFT
    spectrogram_transform = T.Spectrogram(n_fft=n_fft, normalized=True)

    # Apply the Spectrogram transformation
    spectrogram = spectrogram_transform(waveform)

    temp_spectrogram = spectrogram[0].numpy()

    # Resize the resulting spectrogram
    target_size = (1, 64, 64)
    log_spectrogram = interpolate(spectrogram.unsqueeze(0), size=target_size[1:], mode="bilinear").squeeze(0)

    temp_log_spectrogram1 = np.copy(log_spectrogram[0])

    # Calculate the mean and standard deviation
    mean = log_spectrogram.mean(dim=2)
    std = log_spectrogram.std(dim=2)

    # Normalize the log_spectrogram
    log_spectrogram[0] = (log_spectrogram[0] - mean) / std

    #log_spectrogram = 20 * torch.log10(torch.clamp(log_spectrogram, min=1e-5)) - 20
    #log_spectrogram = torch.clamp((log_spectrogram + 100) / 100, 0.0, 1.0)

    temp_log_spectrogram2 = np.copy(log_spectrogram[0])

    file_name = os.path.basename(in_file_path)
    np.save(os.path.join(out_dir, f'{file_name[:-4]}.spec.npy'), log_spectrogram.cpu().numpy())


if __name__ == '__main__':

    # create spectograms
    for machine_name in tqdm(os.listdir(RAW_PATH)):
        #ignore .gitkeep file
        if machine_name == '.gitkeep':
            continue
        for set_name in ["train", "test"]:
            for file_name in os.listdir(os.path.join(RAW_PATH, machine_name, set_name)):
                file_path = os.path.join(RAW_PATH, machine_name, set_name, file_name)
                out_dir = os.path.join(SPECTROGRAMS_PATH, machine_name, set_name)
                os.makedirs(out_dir, exist_ok=True)
                transform_to_spectrogram(in_file_path=file_path, out_dir=out_dir)
