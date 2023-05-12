import os
import csv
import torch
import numpy as np
import wandb
from tqdm import tqdm
from config import *
from metrics import metrics

def test(model, test_loader, machine_name):

    os.makedirs(os.path.join(RESULT_PATH, machine_name), exist_ok=True)

    loss_func = torch.nn.MSELoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    model.to(device)

    # Get a sorted list of file names in the relevant directory
    anomaly_score_path = os.path.join(RESULT_PATH, machine_name, f'anomaly_score_{machine_name}_section_{0}.csv')
    decision_result_path = os.path.join(RESULT_PATH, machine_name, f'decision_result_{machine_name}_section_{0}.csv')

    with open(anomaly_score_path, 'w', newline='\n') as _:
        pass
    with open(decision_result_path, 'w', newline='\n') as _:
        pass

    for index, (masks, spectrograms, labels, spectrogram_file_names) in tqdm(enumerate(test_loader)):

        masked_spectrograms = spectrograms.clone()
        masked_spectrograms[masks == 0] = 0

        inputs = masked_spectrograms.to(device)
        targets = spectrograms.to(device)
        outputs = model.forward(inputs)

        anomaly_score = loss_func(outputs[masks == 0], targets[masks == 0]).view(-1).sum().item()/len(spectrograms)

        if anomaly_score > DETECTION_TRESHOLD_DICT[machine_name]:
            prediction = IS_ANOMALY
        else:
            prediction = IS_NORMAL

        # Get the filename for the current iteration
        spectrogram_file_name = spectrogram_file_names[0]
        raw_file_name = f'{spectrogram_file_name[:-9]}.wav'

        # Write the anomaly score and prediction to the CSV files
        with open(anomaly_score_path, 'a', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow([raw_file_name, 1/anomaly_score])

        with open(decision_result_path, 'a', newline='\n') as f:
            writer = csv.writer(f)
            writer.writerow([raw_file_name, prediction])

        # log loss separately for normal and anomaly
        if labels[0] == IS_ANOMALY:
            print("anomaly")
            wandb.log({f"{machine_name}_reconstr_loss_anomaly": anomaly_score}, step=index)
        else:
            print("normal")
            wandb.log({f"{machine_name}_reconstr_loss_normal": anomaly_score}, step=index)

    try:
        accurracy, auc, p_auc, prec, recall, f1 = metrics(anomaly_score_path, decision_result_path)

        wandb.log({f"{machine_name}_auc": auc})
        wandb.log({f"{machine_name}_accurracy": accurracy})
    except:
        print("logging was not possible")



