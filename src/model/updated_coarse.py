import torch
from torch import nn

class Coarse_net(nn.Module):

    def __init__(self, input_channels=1, output_channels=64, rnninput_size=16, hidden_size=64):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.rnninput_size = rnninput_size
        self.hidden_size = hidden_size

        # Convultional layers to reduce length of fingerprint signals
        self.conv = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=(7,1), stride=(4,1), padding=(2,0)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 16, kernel_size=(7,1), stride=(4,1), padding=(2,0)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU()
        )

        # RNN layer
        self.rnn = nn.GRU(
            rnninput_size,
            hidden_size,
            1,
            batch_first=True
        )

    def forward(self, x):
        # Reshape into individual fingerprints and run convolutions to reduce length
        x_flat = x.flatten(2,3).unsqueeze(1)
        z1 = self.conv(x_flat)
        z1 = z1.squeeze().permute(2, 1, 0)
        # RNN layer
        z2, h = self.rnn(z1)
        # Reshape back into original spatial dimensions
        h = h.permute(0, 2, 1).contiguous().view(1, 64, 232, 232)
        # Return final hidden state from RNN
        return h
        