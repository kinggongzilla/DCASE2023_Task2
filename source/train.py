import torch
import numpy as np
from tqdm import tqdm
from config import EPOCHS, MODEL_PATH, LOG_EVERY
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

        for masks, spectrograms, spectrogram_file_names in tqdm(train_loader):

            step_count += 1
            optimizer.zero_grad()

            masked_spectrograms = spectrograms.clone()
            masked_spectrograms[masks == 0] = 0

            inputs = masked_spectrograms.to(device)
            targets = spectrograms.to(device)
            outputs = model.forward(inputs)

            batch_loss = loss_func(outputs[masks == 0], targets[masks == 0])
            batch_loss.backward()
            optimizer.step()

            wandb.log({f"{machine_name}_step_loss": batch_loss}, step=step_count)

            if step_count % LOG_EVERY == 0:
                model.load_state_dict(torch.load(save_path))
                model.eval()
                model.to(device)
                test(model, test_loader, machine_name)
                model.train()

    torch.save(model.state_dict(), save_path)

    return model




