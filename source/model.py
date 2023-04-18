import torch
from torch import nn

class CNNAutoencoder(nn.Module):
    def __init__(self):
        super(CNNAutoencoder, self).__init__()

        #input size: (1, 80, 63), 1 channel, 80x63 pixels
        
        # Encoder
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=3)
        self.bn3 = nn.BatchNorm2d(8)
        self.conv4 = nn.Conv2d(8, 4, kernel_size=3)
        self.bn4 = nn.BatchNorm2d(4)
        self.conv5 = nn.Conv2d(4, 4, kernel_size=3)
        self.bn5 = nn.BatchNorm2d(4)
        
        #compressed latent space size: ???

        # Decoder
        self.t_conv1 = nn.ConvTranspose2d(4, 4, kernel_size=3)
        self.t_bn1 = nn.BatchNorm2d(4)
        self.t_conv2 = nn.ConvTranspose2d(4, 8, kernel_size=3)
        self.t_bn2 = nn.BatchNorm2d(8)
        self.t_conv3 = nn.ConvTranspose2d(8, 16, kernel_size=3)
        self.t_bn3 = nn.BatchNorm2d(16)
        self.t_conv4 = nn.ConvTranspose2d(16, 16, kernel_size=3)
        self.t_bn4 = nn.BatchNorm2d(16)
        self.t_conv5 = nn.ConvTranspose2d(16, 1, kernel_size=3)

        #output size: (1, 80, 63), 1 channel, 80x63 pixels


    def forward(self, x):
        # Encoder
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = torch.relu(self.bn5(self.conv5(x)))
        
        # Decoder
        x = torch.relu(self.t_bn1(self.t_conv1(x)))
        x = torch.relu(self.t_bn2(self.t_conv2(x)))
        x = torch.relu(self.t_bn3(self.t_conv3(x)))
        x = torch.relu(self.t_bn4(self.t_conv4(x)))
        x = torch.sigmoid(self.t_conv5(x))
        
        return x