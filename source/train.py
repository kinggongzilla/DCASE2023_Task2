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

    loss_func = torch.nn.MSELoss(reduction='none')

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

            # compute loss without reduction
            unmasked_loss = loss_func(outputs, targets)

            if step_count % LOG_EVERY == 0:

                # Split unreduced_loss by labels (labels == IS_ANOMAL or labels == IS_NORMAL)
                normal_indices = (labels == IS_NORMAL)
                anomal_indices = (labels == IS_ANOMALY)

                normal_masks = masks[normal_indices]
                anomal_masks = masks[anomal_indices]

                # Compute and log both average losses
                train_loss_normal = torch.mean(unmasked_loss[normal_indices][normal_masks == 0])
                train_loss_anomaly = torch.mean(unmasked_loss[anomal_indices][anomal_masks == 0])

                # Log loss separately for normal and anomaly with the step number
                wandb.log({f"{machine_name}_train_loss_anomaly": train_loss_anomaly}, step=step_count)
                wandb.log({f"{machine_name}_train_loss_normal": train_loss_normal}, step=step_count)

                # Calculate the average train loss based on the relative label frequency
                num_normal = normal_indices.sum().float()
                num_anomaly = anomal_indices.sum().float()
                train_loss = (num_normal * train_loss_normal + num_anomaly * train_loss_anomaly) / (num_normal + num_anomaly)
            else:
                train_loss = torch.mean(unmasked_loss[masks == 0].view(-1))

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




