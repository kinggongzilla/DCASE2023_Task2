import torch
import numpy as np
from tqdm import tqdm
from config import EPOCHS, MODEL_PATH
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
    best_loss = 999999999999

    for epoch in range(EPOCHS):

        epoch_loss = 0

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
            epoch_loss += float(batch_loss.item())

        # normalize epoch_loss by total number of samples
        epoch_loss = epoch_loss/len(train_loader)
        wandb.log({f"{machine_name}_epoch_loss": epoch_loss})


        #save model if loss is new best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), save_path)
        print(f'epoch: {epoch} | loss: {epoch_loss}')

        #log area under the curve every 10 epochs
        if epoch % 2 == 0:
            model.load_state_dict(torch.load(save_path))
            model.eval()
            model.to(device)
            test(model, test_loader, machine_name)
            model.train()
            

    torch.save(model.state_dict(), save_path)

    return model




