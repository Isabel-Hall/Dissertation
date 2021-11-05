import torch
from torch import nn



class UNet(nn.Module):

    def __init__(self, input_channels=1000):
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(input_channels, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(256, 1, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        z1 = self.down1(x)
        z2 = self.up1(z1)
        return z1, z2
