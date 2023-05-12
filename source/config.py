import torch
import numpy as np
import os

# GENERAL SETTINGS
IS_NORMAL = 0
IS_ANOMALY = 1
IS_SOURCE = 0
IS_TARGET = 1

# MODEL SETTINGS
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 256
SAMPLE_RATE = 22500  # 22.5kHz
LEARNING_RATE = 1e-4
EPOCHS = 1
N_MASKS_PER_SPECTROGRAM = 10
MODEL_PATH = 'models'
LOG_EVERY = 100

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

SAMPLE_LENGTH_SECONDS = 1
WINDOW_LENGTH = 124
HOP_LENGTH = 256
N_FFT = 400
N_MELS = 80
FMIN = 20.0
FMAX = SAMPLE_RATE / 2
POWER = 1.0
NORMALIZED = True
PATCH_SIZE = 8
NUM_PATCHES_TO_ZERO = 48


PARENT_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.environ.get("DATA_PATH", os.path.join(PARENT_PATH, 'data'))
RAW_PATH = os.path.join(DATA_PATH, 'raw')
PROCESSED_PATH = os.path.join(DATA_PATH, 'processed')
SPECTROGRAMS_PATH = os.path.join(PROCESSED_PATH, 'spectrograms')
RESULT_PATH = os.environ.get("RESULT_PATH", os.path.join(PARENT_PATH, 'result'))
