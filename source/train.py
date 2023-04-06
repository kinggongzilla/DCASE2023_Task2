import torch
import numpy as np
from tqdm import tqdm

def train(model, optimizer, trainloader, epochs, lr=1e-4):

    #check if cuda is availableand set as device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    #get and print number of parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f'Total number of parameters: {params}') #print number of parameters

    model.train()
    model.to(device)

    loss_func = torch.nn.MSELoss()

    step_count = 0
    best_loss = 999999999999
    for epoch in range(epochs):
        epoch_loss = 0
        for batch in tqdm(trainloader):
            #batch is a tuple (spectogram, waveform)
            step_count += 1
            optimizer.zero_grad()

            x = batch[0] .to(device)
            y = model.forward(x)

            #calculate loss, barward pass and optimizer step
            batch_loss = loss_func(y, x)
            batch_loss.backward()
            optimizer.step()
            epoch_loss += float(batch_loss.item())

        # normalize epoch_loss by total number of samples
        epoch_loss = epoch_loss/len(trainloader)

        #save model if loss is new best loss
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), 'output/models/best_model.pt')
        print(f'epoch: {epoch} | loss: {epoch_loss}')

    #save final model locally
    torch.save(model.state_dict(), 'output/models/last_model.pt')

    return model