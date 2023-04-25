import os
import sys
from pydub import AudioSegment
from config import *
import torchaudio
import numpy as np
import torch
from tqdm import tqdm
#Note: ffmpeg package required for pydub

def transform_to_spectrogram(in_file_path, out_dir):
    if not os.path.exists(out_dir):
        raise f'out_dir {out_dir} does not exist'
    if not os.path.isfile(in_file_path):
        raise f'given in_path {in_file_path} is not a file'

    audio = torchaudio.load(in_file_path)[0]
    log_spectrogram = torchaudio.transforms.Spectrogram(
        win_length=WINDOW_LENGTH,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT,
        power=POWER,
        normalized=NORMALIZED)(audio)

    log_spectrogram = 20 * torch.log10(torch.clamp(log_spectrogram, min=1e-5)) - 20
    log_spectrogram = torch.clamp((log_spectrogram + 100) / 100, 0.0, 1.0)
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
