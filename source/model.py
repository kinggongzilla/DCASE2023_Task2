import torch
from torch import nn

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 8, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(8)
        self.conv3 = nn.Conv2d(8, 8, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(8)
        
        #compressed latent space size: (8, 74, 57), 8 channels, 74x57 pixels

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(8, 8, kernel_size=3)
        self.t_bn1 = nn.BatchNorm2d(8)
        self.t_conv2 = nn.ConvTranspose2d(8, 16, kernel_size=3)
        self.t_bn2 = nn.BatchNorm2d(16)
        self.t_conv3 = nn.ConvTranspose2d(16, 1, kernel_size=3)

    def forward(self, x):
        # Encoder
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        
        # Decoder
        x = torch.relu(self.t_bn1(self.t_conv1(x)))
        x = torch.relu(self.t_bn2(self.t_conv2(x)))
        x = torch.sigmoid(self.t_conv3(x))
        
        return x