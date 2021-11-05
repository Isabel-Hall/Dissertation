import torch
from torch import nn


class Generator(nn.Module):

    def __init__(self, cuda, z_size=32):
        super().__init__()

        self.z_size = z_size
        self.dev = "cuda" if cuda else "cpu"

        self.main = nn.Sequential(
            nn.Linear(z_size, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1000),
            #nn.ReLU()
        )

        self.refine = nn.Sequential(
            nn.Conv1d(1, 32, 5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 1, 5, stride=1, padding=2)
        )

    def forward(self, batch_size):
        z = torch.randn(batch_size, 32, device=self.dev) 
        #print("z", z.shape)
        z1 = self.main(z).unsqueeze(2)
        #print("z1", z1.shape)
        #z2 = self.refine(z1).transpose(2, 1)
        #print("z2", z2.shape)

        return z1

