import os
import csv
import torch
import numpy as np
import wandb
from tqdm import tqdm
from config import RESULTS_PATH, RAW_PATH, DETECTION_TRESHOLD_DICT, IS_ANOMALY, IS_NORMAL

def test(model, test_loader, machine_name):

    os.makedirs(os.path.join(RESULTS_PATH, machine_name), exist_ok=True)

    loss_func = torch.nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    # Get a sorted list of file names in the relevant directory
    file_names = sorted(os.listdir(os.path.join(RAW_PATH, machine_name, "test")))

    with open(os.path.join(RESULTS_PATH, machine_name, f'anomaly_score_{machine_name}_section_{0}.csv'), 'w', newline='\n') as _:
        pass
    with open(os.path.join(RESULTS_PATH, machine_name, f'decision_result_{machine_name}_section_{0}.csv'), 'w', newline='\n') as _:
        pass

    for (spectrograms, waveforms, indices, domains, labels) in tqdm(test_loader):

        # batch holds all segments of one test sample
        # indices, domains, labels hold the same values for the test batches.
        index = indices[0]
        domain = domains[0]
        label = labels[0]

        x = spectrograms.to(device)
        y = model.forward(x)

        anomaly_score = loss_func(y, x).view(-1).sum().item()/len(indices)

        if anomaly_score > DETECTION_TRESHOLD_DICT[machine_name]:
            prediction = IS_ANOMALY
        else:
            prediction = IS_NORMAL

        # Get the filename for the current iteration
        file_name = file_names[index]

        # Write the anomaly score and prediction to the CSV files
        with open(os.path.join(RESULTS_PATH, machine_name, f'anomaly_score_{machine_name}_section_{0}.csv'), 'a', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow([file_name, anomaly_score])

        with open(os.path.join(RESULTS_PATH, machine_name, f'decision_result_{machine_name}_section_{0}.csv'), 'a', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow([file_name, prediction])



