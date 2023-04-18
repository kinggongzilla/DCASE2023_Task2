import os
import csv
import torch
import numpy as np
import wandb
from tqdm import tqdm
from config import RESULTS_PATH, RAW_PATH, DETECTION_TRESHOLD_DICT, IS_ANOMALY, IS_NORMAL
from metrics import metrics

def test(model, test_loader, machine_name):

    os.makedirs(os.path.join(RESULTS_PATH, machine_name), exist_ok=True)

    loss_func = torch.nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    # Get a sorted list of file names in the relevant directory
    file_names = sorted(os.listdir(os.path.join(RAW_PATH, machine_name, "test")))
    anomaly_score_path = os.path.join(RESULTS_PATH, machine_name, f'anomaly_score_{machine_name}_section_{0}.csv')
    decision_result_path = os.path.join(RESULTS_PATH, machine_name, f'decision_result_{machine_name}_section_{0}.csv')

    with open(anomaly_score_path, 'w', newline='\n') as _:
        pass
    with open(decision_result_path, 'w', newline='\n') as _:
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
        with open(anomaly_score_path, 'a', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow([file_name, anomaly_score])

        with open(decision_result_path, 'a', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow([file_name, prediction])
    try:
        accurracy, auc, p_auc, prec, recall, f1 = metrics(anomaly_score_path, decision_result_path)

        wandb.log({f"{machine_name}_auc": auc})
        wandb.log({f"{machine_name}_accurracy": accurracy})
    except:
        print("logging was not possible")



