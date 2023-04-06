# DCASE Task 2
Repo for DCASE 2023 Task 2, anomaly detection.

# Set up environment with dependencies
1. Load conda env with "conda env create -f environment.yml"
2. Activate conda environment with "conda activate dcase2023"
3. If on Mac/Linux install sox with "conda install -c conda-forge sox" (a dependency of torchaudio)

# Prepare data
By default, when using source/data_prep.py, full length audio files are segmented into fixed length samples of 1 second (length is configurable). Mel spectrograms are computed for each segmented sample. The segmented audio samples and mel spectrograms are saved.


1. Set up a folder "data/raw" in root directory of this repository
2. Download the ZIP folders for each machine in the Task 2 train set: https://zenodo.org/record/7690157#.ZC7sl3tBzMY
3. Extract the ZIP folders into "data/raw" ("data" folder has to be on same level as "output" and "source" folders)
2. Make sure that "data/processed/audio_segments" and "data/processed/spectrograms" folders exist. If not, create them.
3. Run "python source/data_prep.py" to chop samples in "data/raw" into 1 second snippets and save to "data/processed/audio_segments" and spectrograms to "data/processed/spectrograms"

# How to train a model
1. Set desired config parameters in "source/config.py"
2. Run "python source/main.py" to start training. 