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
4. Remove one folder level for each machine such that the directory has the following structure:
   "raw_path/machine_name/train/...", for example /raw_path/gearbox/train/
5. Run "python source/data_prep.py" to chop samples in "data/raw" into 1 second snippets and save to "data/processed/audio_segments" and spectrograms to "data/processed/spectrograms"
   The resulting files will have following namings: f"index={index}__segment_id={segment_id}__domain={domain}__label={label}"
   index: a number indicating the original audio file index (row index in the sorted list of raw audio files) belonged to
   segment_id is a number from 0 to the number of audio segments produced from one audio file 
   label is "normal" or "anomaly" (only relevant for test data)
   domain is "source" or "target"

# How to train a model
1. Set desired config parameters in "source/config.py"
2. Run "python source/main.py" to start training and testing
   main.py will train a model for each machine type and produce evaluation results: Two files in each results/machine/ with the anomaly scores and the prediction
   The results follow the official submission format:
   "anomaly_score_<machine_type>_section_<section_index>.csv" with rows: "[filename (string)],[anomaly score (real value)]"
   "decision_result_<machine_type>_section_<section_index>.csv" with rows "[filename (string)],[decision result (0: normal, 1: anomaly)]"

# TBD in the future
1. From the generated files all scoring metrics can be computed (see Figure 3: Task overview). 
   Implement a function which computes basic performance metrics (Accuracy...) and most importantly AUC, pAUC (harmonic mean of them) to be called after the two eval files for a machine have been created
2. Find reasonable choices for the anomaly score detection threshold (for each machine). 
   For now this is set to a fixed value in the configs DETECTION_TRESHOLD_DICT 
3. If the full evaluation pipeline is done, we can start comparing to the baselines/previous years scores and improve the model

