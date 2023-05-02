import os
import sys
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

    # Load the waveform and resample if needed
    waveform, sample_rate = torchaudio.load(in_file_path)
    target_sample_rate = 22500
    if sample_rate != target_sample_rate:
        resampler = T.Resample(sample_rate, target_sample_rate)
        waveform = resampler(waveform)

    # Create a MelSpectrogram object
    n_fft = 400
    n_mels = 128 # You can change this value if you want
    mel_spectrogram_transform = T.MelSpectrogram(sample_rate=target_sample_rate, n_fft=n_fft, n_mels=n_mels)

    # Apply the MelSpectrogram transformation
    mel_spectrogram = mel_spectrogram_transform(waveform)

    # Resize the resulting mel spectrogram
    target_size = (1, 64, 64)
    log_mel_spectrogram = interpolate(mel_spectrogram.unsqueeze(0), size=target_size[1:], mode="bilinear").squeeze(0)

    log_mel_spectrogram = 20 * torch.log10(torch.clamp(log_mel_spectrogram, min=1e-5)) - 20
    log_mel_spectrogram = torch.clamp((log_mel_spectrogram + 100) / 100, 0.0, 1.0)

    temp = log_mel_spectrogram[0].numpy()

    file_name = os.path.basename(in_file_path)
    np.save(os.path.join(out_dir, f'{file_name[:-4]}.spec.npy'), log_mel_spectrogram.cpu().numpy())


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
