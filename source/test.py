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

    for index, (masked_spectrograms, spectrograms) in tqdm(enumerate(test_loader)):
        if index >= len(file_names):
            break

        inputs = masked_spectrograms.to(device)
        targets = spectrograms.to(device)

        outputs = model.forward(inputs)

        anomaly_score = loss_func(outputs, targets).view(-1).sum().item()/len(masked_spectrograms)

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



