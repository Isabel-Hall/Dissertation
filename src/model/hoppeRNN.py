import torch
from torch import nn

class Hoppernn(nn.Module):
    
    def __init__(self, cuda):
        super().__init__()
        self.dev = "cuda" if cuda else "cpu"

        self.rnn = nn.LSTM(
            input_size=100,
            hidden_size=300,
            num_layers=1,
            batch_first=True
        )

        self.act = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(300)
        )

        self.linear = nn.Sequential(
            nn.Linear(3000, 1000),
            nn.ReLU(),
            nn.BatchNorm1d(1000),
            nn.Linear(1000, 400),
            nn.ReLU(),
            nn.BatchNorm1d(400),
            nn.Linear(400, 222),
            nn.ReLU(),
            nn.BatchNorm1d(222),
            nn.Linear(222, 2)
        )

    def forward(self, x):
        #print("x shape", x.shape)
        x = x.view(-1, 10, 100)
        #print("x shape", x.shape)
        z1, (h, c) = self.rnn(x)
        #print("z1 shape", z1.shape, "h shape", h.shape, "c shape", c.shape)
        z1 = z1.permute(0, 2, 1)
        #print("z1 shape", z1.shape)
        z2 = self.act(z1)
        batch_size = z2.shape[0]
        #print(batch_size)
        z2 = z2.view(batch_size, -1)
        #print("z2 shape", z2.shape)
        y = self.linear(z2)
        #print("y shape", y.shape)
        return y