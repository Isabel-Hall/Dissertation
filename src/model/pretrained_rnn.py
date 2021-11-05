import torch
from torch import nn
from .rnn import RNNDiscriminator


class PretrainedRNN(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.pretrained = RNNDiscriminator()
        # Imports the state of the pre-trained RNN
        self.pretrained.load_state_dict(
            torch.load("/home/issie/code/py/MRI/src/artifacts/4/1c335acc79224d90b3f11daf14e29418/States/discriminator-6.state")
        )


    def forward(self, x):
        # Same architecture as the RNN from rnn.py but without the final linear layer
        x = x.unsqueeze(1)
        x_1 = self.pretrained.conv(x)
        x_1 = x_1.permute(2,0,1)
        y, h = self.pretrained.rnn(x_1)
        h = h[-1]
        # Return the final hidden state from the rnn 
        return h