import torch
from torch import nn

# Fully connected compression model to transform each fignerprint signal into 64 features that can be fed into the RCA U-Net

class Coarse_net(nn.Module):

    def __init__(self, input_channels=1000, output_channels=64):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels

        self.main = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.main(x)