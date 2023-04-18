import torch
import numpy as np
from tqdm import tqdm
from config import EPOCHS, MODEL_PATH
import os
import wandb


def train(model, optimizer, train_loader, machine_name):

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

        for spectrograms, waveforms, indices, domains, labels in tqdm(train_loader):

            step_count += 1
            optimizer.zero_grad()

            x = spectrograms.to(device)
            y = model.forward(x)

            #calculate loss, barward pass and optimizer step
            batch_loss = loss_func(y, x)
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

    torch.save(model.state_dict(), save_path)

    return model




