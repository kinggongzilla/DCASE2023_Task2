import os
import sys
from pydub import AudioSegment
from config import SAMPLE_RATE, SAMPLE_LENGTH_SECONDS, WINDOW_LENGTH, HOP_LENGTH, N_FFT, N_MELS, FMIN, FMAX, POWER, NORMALIZED, IS_TRAINING
import torchaudio
import numpy as np
import torch
from tqdm import tqdm
#Note: ffmpeg package required for pydub

def transform_to_spectrogram(
    audio_path: str, 
    out_path='data/processed/spectrograms', 
    sample_rate = SAMPLE_RATE, 
    win_length= WINDOW_LENGTH,
    hop_length= HOP_LENGTH,
    n_fft=N_FFT,
    f_min=FMIN,
    f_max = FMAX,
    n_mels = N_MELS,
    power = POWER,
    normalized = NORMALIZED,
):
    if not os.path.exists(out_path):
        raise 'out_dir does not exist'
    if not os.path.isfile(audio_path):
        raise 'given wav_path is not a file'

    filename = os.path.basename(audio_path)

    audio = torchaudio.load(audio_path)[0]
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        win_length=win_length,
        hop_length=hop_length,
        n_fft=n_fft,
        f_min=f_min,
        f_max=f_max,
        n_mels=n_mels,
        power=power,
        normalized=normalized)(audio)
    mel_spectrogram = 20 * torch.log10(torch.clamp(mel_spectrogram, min=1e-5)) - 20
    mel_spectrogram = torch.clamp((mel_spectrogram + 100) / 100, 0.0, 1.0)
    np.save(os.path.join(out_path, f'{filename}.spec.npy'), mel_spectrogram.cpu().numpy())



#load data of one wav and split it into chunks
def chop_wav(id: str, audio_path: str, out_dir: str, length: int):

    print('audio_path: ', audio_path)
    #check if out_dir exists
    if not os.path.exists(out_dir):
        raise 'out_dir does not exist'
    if not os.path.isfile(audio_path):
        raise 'given wav_path is not a file'

    #load audio
    if audio_path.endswith('.wav'):
        audio = AudioSegment.from_wav(audio_path)
        file_ending = '.wav'
    elif audio_path.endswith('.mp3'):
        audio = AudioSegment.from_mp3(audio_path)
        file_ending = '.mp3'
    else:
        raise 'wav_path must be a .wav or .mp3 file'

    start = 0
    end = length
    n_iters = int(len(audio)) // (SAMPLE_LENGTH_SECONDS * 1000)


    for i in range(n_iters):
        newAudio = audio[start:end]
        newAudio.export(os.path.join(out_dir, '{}_{}_{}{}'.format(i, id, start, file_ending)), format="wav")
        start += length
        end += length

if __name__ == '__main__':
    split_dir = 'train' if IS_TRAINING else 'test'
    raw_dir=os.path.join('data','raw')
    segmented_audio_dir=os.path.join('data','processed','audio_segments')
    sample_length = SAMPLE_LENGTH_SECONDS * 1000 #milliseconds

    if len(sys.argv) > 1:
        raw_dir = sys.argv[1]
    if len(sys.argv) > 2:
        segmented_audio_dir = sys.argv[2]
    if len(sys.argv) > 3:
        sample_length = int(sys.argv[3])

    #loop over files in raw
    #for every directory in raw_dir
    for dir in tqdm(os.listdir(raw_dir)):
        for i, subdir in enumerate(os.listdir(os.path.join(raw_dir, dir))):
            for j, file in enumerate(os.listdir(os.path.join(raw_dir, dir, subdir, split_dir))):
                id = str(i) + str(j)
                chop_wav(id, os.path.join(raw_dir, dir, subdir, split_dir, file), segmented_audio_dir, sample_length)

    #generate mel spectrograms from chopped audio
    for i, file in enumerate(os.listdir(segmented_audio_dir)):
        #ignore .gitkeep file
        if file == '.gitkeep':
            continue
        transform_to_spectrogram(os.path.join(segmented_audio_dir, file), out_path=os.path.join('data', 'processed', 'spectrograms'))