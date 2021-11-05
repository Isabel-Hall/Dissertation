import torch
from torch import nn


class CoarseModel(nn.Module):

    def __init__(self, input_channels=1000, output_channels=64, embed_dim=32):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.embed_dim = embed_dim

        # conv feature(channel) reduction
        self.conv_reduce = nn.Sequential(
            nn.Conv1d(1, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),

            nn.Conv1d(embed_dim, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),

            nn.Conv1d(embed_dim, 1, kernel_size=3, stride=2, padding=1)
        )
    
    def forward(self, x):
        batch_size, seq_len, h, w = x.size()

        # flatten from image format to sequence format
        x_flat = x.squeeze().view(self.input_channels, -1).permute(1, 0).unsqueeze(1)

        # temporal conv (summarise windows)
        z1 = self.conv_reduce(x_flat)

        # reshape back to an image
        z1 = z1.view(h, w, 1, -1)
        z1 = z1.permute(2, 3, 0, 1)

        return z1
    

        
