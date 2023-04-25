import os
import sys
from pydub import AudioSegment
from config import SAMPLE_RATE, SAMPLE_LENGTH_SECONDS, WINDOW_LENGTH, HOP_LENGTH, N_FFT, N_MELS, FMIN, FMAX, POWER, NORMALIZED, RAW_PATH, AUDIO_SEGMENTS_PATH, SPECTROGRAMS_PATH
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


#load data of one wav and split it into chunks
def chop_wav(index, domain, label, sample_length, in_file_path: str, out_dir: str):

    #check if out_dir exists
    if not os.path.exists(out_dir):
        raise 'out_dir does not exist'
    if not os.path.isfile(in_file_path):
        raise 'given in_file_path is not a file'

    # load audio
    if in_file_path.endswith('.wav'):
        audio = AudioSegment.from_wav(in_file_path)
        file_ending = '.wav'
    elif in_file_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(in_file_path)
        file_ending = '.mp3'
    else:
        raise 'wav_path must be a .wav or .mp3 file'

    start = 0
    end = sample_length
    n_iters = int(len(audio)) // (SAMPLE_LENGTH_SECONDS * 1000)

    for segment_id in range(n_iters):
        newAudio = audio[start:end]
        out_file_name = f"index={index}__segment_id={segment_id}__domain={domain}__label={label}{file_ending}"
        newAudio.export(os.path.join(out_dir, out_file_name), format="wav")
        start += sample_length
        end += sample_length

if __name__ == '__main__':

    sample_length = SAMPLE_LENGTH_SECONDS * 1000 #milliseconds

    # create segments
    for machine_name in tqdm(os.listdir(RAW_PATH)):
        if machine_name == '.gitkeep':
            continue
        for set_name in ["train", "test"]:
            for index, file_name in enumerate(os.listdir(os.path.join(RAW_PATH, machine_name, set_name))):
                file_path = os.path.join(RAW_PATH, machine_name, set_name, file_name)
                domain = "source" if "source" in file_name else "target"
                label = "normal" if "normal" in file_name else "anomaly"
                out_dir = os.path.join(AUDIO_SEGMENTS_PATH, machine_name, set_name)
                os.makedirs(out_dir, exist_ok=True)
                chop_wav(index, domain, label, sample_length, in_file_path=file_path, out_dir=out_dir)

    # create spectograms
    for machine_name in tqdm(os.listdir(RAW_PATH)):
        #ignore .gitkeep file
        if machine_name == '.gitkeep':
            continue
        for set_name in ["train", "test"]:
            for file_name in os.listdir(os.path.join(AUDIO_SEGMENTS_PATH, machine_name, set_name)):
                file_path = os.path.join(AUDIO_SEGMENTS_PATH, machine_name, set_name, file_name)
                out_dir = os.path.join(SPECTROGRAMS_PATH, machine_name, set_name)
                os.makedirs(out_dir, exist_ok=True)
                transform_to_spectrogram(in_file_path=file_path, out_dir=out_dir)
