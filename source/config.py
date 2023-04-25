import torch
import numpy as np
import os

# GENERAL SETTINGS
IS_NORMAL = 0
IS_ANOMALY = 1
IS_SOURCE = 0
IS_TARGET = 1

# MODEL SETTINGS
BATCH_SIZE = 64
SAMPLE_RATE = 22500  # 22.5kHz
LEARNING_RATE = 1e-4
EPOCHS = 10
MODEL_PATH = 'models'
RESULTS_PATH = 'results'

DETECTION_TRESHOLD_DICT = dict(
    bearing=0.001,
    fan=0.001,
    gearbox=0.001,
    slider=0.001,
    ToyCar=0.001,
    ToyTrain=0.001,
    valve=0.001,
)

# if reconstruction loss is higher than this --> consider sample an anomaly

# DATA PREP SETTINGS
SAMPLE_LENGTH_SECONDS = 1
WINDOW_LENGTH = 124
HOP_LENGTH = 256
N_FFT = 124
N_MELS = 80
FMIN = 20.0
FMAX = SAMPLE_RATE / 2
POWER = 1.0
NORMALIZED = True
PARENT_PATH = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
DATA_PATH = os.path.join(PARENT_PATH, 'data')
RAW_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
SPECTROGRAMS_PATH = os.path.join(PROCESSED_PATH, 'spectrograms')
AUDIO_SEGMENTS_PATH = os.path.join(PROCESSED_PATH, 'audio_segments')
RESULT_PATH = 'results'