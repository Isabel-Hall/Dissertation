import torch
from torch import nn
from torch.nn import functional as F
#from .minibatch import MinibatchDiscrimination
from torchgan.layers import MinibatchDiscrimination1d


class RNNGenerator(nn.Module):
    
    def __init__(self, cuda, seq_gen_length=125, norm=nn.BatchNorm1d):
        super().__init__()
        self.dev = "cuda" if cuda else "cpu"
        self.seq_gen_length = seq_gen_length
        # self.rnn tried as both an LSTM and a GRU

        # input_size = 32
        # hidden_size = 64
        # num_layers = 1
        # self.rnn = nn.LSTM(
        #     input_size,
        #     hidden_size,
        #     num_layers,
        #     batch_first=True
        # )
        self.rnn = nn.GRU(128, 128, 1, batch_first=True)

        # Convolutional layers
        rate = 0.25
        self.conv1 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
            norm(128),
            nn.ReLU(),
            nn.Dropout(rate),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
            norm(128),
            nn.ReLU(),
            nn.Dropout(rate),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=5, stride=1, padding=2),
            norm(128),
            nn.ReLU(),
            nn.Dropout(rate),
        )
        self.conv4 = nn.Conv1d(128, 1, kernel_size=5, stride=1, padding=2)



    def forward(self, batch_size):
        # Random noise as initial input for RNN
        z1 = torch.randn(batch_size, self.seq_gen_length, 128, device=self.dev)
        z2, h = self.rnn(z1)
        z2 = z2.transpose(2, 1)

        # Convolutional layers
        z3 = self.conv1(z2)
        z3 = F.interpolate(z3, scale_factor=2)

        z4 = self.conv2(z3)
        z4 = F.interpolate(z4, scale_factor=2)

        z5 = self.conv3(z4)
        z5 = F.interpolate(z5, scale_factor=2)

        z6 = self.conv4(z5)

        # Return generated fingerprint signal. Tanh used to match dataloader of real signals
        return torch.tanh(z6.squeeze())


class RNNDiscriminator(nn.Module):

    def __init__(self, conv_input_size=1, input_size=32, hidden_size=64, num_layers=1, norm=nn.BatchNorm1d):
        super().__init__()
        self.conv_input_size = conv_input_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        # self.rnn = nn.LSTM(
        #     input_size,
        #     hidden_size,
        #     num_layers,
        #     batch_first=False
        # )
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers
        )

        # final linear layer
        self.linear = nn.Sequential(
            nn.Linear(hidden_size + 3, 32),
            nn.LeakyReLU(),
            norm(32),
            nn.Linear(32, 1)
        )

        # Convolutional layers
        rate=0.5
        self.conv = nn.Sequential(
            nn.Conv1d(conv_input_size, 32, kernel_size=5, stride=1, padding=2),
            norm(32),
            nn.LeakyReLU(),
            nn.Dropout(rate),

            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            norm(32),
            nn.LeakyReLU(),
            nn.Dropout(rate),

            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            norm(32),
            nn.LeakyReLU(),
            nn.Dropout(rate),

            nn.Conv1d(32, 32, kernel_size=5, stride=2, padding=2),
            norm(32),
            nn.LeakyReLU(),
            nn.Dropout(rate),
        )

        # Minibatch discrimination from TorchGAN to reduce mode collapse
        self.minibatch_discrimination = MinibatchDiscrimination1d(64, 3)
    


    def forward(self, x):
        # Convolutional layers and reformatting
        x = x.unsqueeze(1)
        x_1 = self.conv(x)
        x_1 = x_1.permute(2,0,1)

        # RNN discriminator layer
        ##FOR LSTM y, (h, c) = self.rnn(x_1)
        y, h = self.rnn(x_1)
        h = h[-1]
        # Minibatch discrimination
        mbd = self.minibatch_discrimination(h)
        # Prediction of whether input is real or from the generator
        pred = self.linear(mbd.squeeze())
        return torch.sigmoid(pred)
