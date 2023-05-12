import torch
import numpy as np
from tqdm import tqdm
from config import *
import os
import wandb
from test import test
from metrics import metrics


def train(model, optimizer, train_loader, test_loader, machine_name):

    save_path = os.path.join(MODEL_PATH, machine_name)
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, "model.pt")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.train()
    model.to(device)

    loss_func = torch.nn.MSELoss()

    step_count = 0

    for epoch in range(EPOCHS):

        for masks, spectrograms, labels, spectrogram_file_names in tqdm(train_loader):

            step_count += 1
            optimizer.zero_grad()

            masked_spectrograms = spectrograms.clone()
            masked_spectrograms[masks == 0] = 0

            inputs = masked_spectrograms.to(device)
            targets = spectrograms.to(device)
            outputs = model.forward(inputs)

            if step_count % LOG_EVERY == 0:
                # Split masks/spectrograms by labels (labels == IS_ANOMAL or labels == IS_NORMAL)
                normal_indices = (labels == IS_NORMAL)
                anomal_indices = (labels == IS_ANOMALY)

                normal_spectrograms = spectrograms[normal_indices].to(device)
                anomal_spectrograms = spectrograms[anomal_indices].to(device)

                normal_masks = masks[normal_indices].to(device)
                anomal_masks = masks[anomal_indices].to(device)

                # Perform forward passes with both normal and anomaly labels
                normal_outputs = model.forward(normal_spectrograms[normal_masks == 0])
                anomal_outputs = model.forward(anomal_spectrograms[anomal_masks == 0])

                # Compute and log both average losses
                train_loss_normal = torch.mean(loss_func(normal_outputs, normal_spectrograms[normal_masks == 0]))
                train_loss_anomaly = torch.mean(loss_func(anomal_outputs, anomal_spectrograms[anomal_masks == 0]))

                # Log loss separately for normal and anomaly with the step number
                wandb.log({f"{machine_name}_train_loss_anomaly": train_loss_anomaly}, step=step_count)
                wandb.log({f"{machine_name}_train_loss_normal": train_loss_normal}, step=step_count)

                # Calculate the average train loss based on the relative label frequency
                num_normal = normal_indices.sum().float()
                num_anomaly = anomal_indices.sum().float()
                train_loss = (num_normal * train_loss_normal + num_anomaly * train_loss_anomaly) / (num_normal + num_anomaly)

            else:
                train_loss = loss_func(outputs[masks == 0], targets[masks == 0])
                train_loss.backward()
                optimizer.step()

            wandb.log({f"{machine_name}_train_loss": train_loss}, step=step_count)




        #if step_count % LOG_EVERY == 0:
                            #model.load_state_dict(torch.load(save_path))
                            #model.eval()
                            #model.to(device)
                            #test(model, test_loader, machine_name)
                            #model.train()

    torch.save(model.state_dict(), save_path)

    return model




